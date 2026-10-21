#!/usr/bin/env python
"""Read-only diagnostic of periodic phase controls on saved particle states.

Run only through Slurm after coordinating real-model resources. This script
never fits coefficients or modifies sampler inputs. Saved raw scores, phase
controls and optional independently corrected scores remain separate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.galsbi_summary import SummaryBatch, SummaryLikelihood
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier
from sbsi.phase_control import (FrozenPhaseControl, digest, phase_stein_control,
                                rotate_summary_phase)
from sbsi.summary_gradients import SummaryGradients
from sbsi.disk_precision import (LOG_GAP_COLUMN, physical_log_prob_with_gap,
                                 transport_with_log_gap)
from sbsi.shear_map import apply_shear_to_ellipticity


NATIVE_SOURCES = ('galsbi_target_coordinates.py', 'galsbi_summary.py',
                  'summary_gradients.py', 'models_v3_9_like.json')
PRECISION_LIMITS = dict(sampling_replay_atol=1e-6, float64_replay_atol=1e-10,
                        saved_score_atol=1e-6, phase_gradient_scaled_atol=1e-4,
                        sampled_float32_float64_atol=1e-5)
DERIVATIVE_PROTOCOL = 'phase-fine-difference-v2'
PHASE_DIFFERENCE_STEPS = (2e-6, 1e-6, 5e-7)


@torch.no_grad()
def float64_log_kernel(flow, detector, values, observed, shear, batch_size):
    """Independent F64 forward replay; native dataframe wrappers require F32.

    Keep the original F32 SummaryLikelihood replay unchanged. Here NumPy
    applies intrinsic shear independently of the gradient path, then explicit
    F64 preprocessing avoids the legacy dataframe-to-F32 conversions. Shared
    physical transport/density primitives define the same continuous model;
    this function neither calls autograd nor SummaryGradients/tensor_log_kernel.
    """
    members = getattr(detector, 'members', (detector,))
    if any(next(model.parameters()).dtype != torch.float64 for model in
           [flow.model, *[member.model for member in members]]):
        raise ValueError('float64 forward replay requires float64 networks')
    values = np.asarray(values, float)
    raw = observed[list(flow.target_transform.target_names)].to_numpy(float)
    gaps = observed[LOG_GAP_COLUMN].to_numpy(float)
    if values.ndim != 2 or values.shape[1] != 9 or len(values) != len(raw):
        raise ValueError('aligned nine-column summaries and observations required')
    result = np.empty(len(values))
    for start in range(0, len(values), batch_size):
        stop = min(len(values), start+batch_size)
        rows = np.minimum(np.arange(start, start+batch_size), stop-1)
        context = values[rows, :8].copy()
        context[:, 0], context[:, 1] = apply_shear_to_ellipticity(
            context[:, 0], context[:, 1], *shear)
        tensor = lambda value: torch.as_tensor(value, device=flow.device, dtype=torch.float64)
        context = tensor(context)
        standard = flow.condition_preprocessor.transform_tensor(context)
        physical = tensor(raw[rows])
        shape, gap, ld = transport_with_log_gap(physical[:, :2], tensor(gaps[rows]),
            tensor(values[rows, 8, None]*np.asarray(shear)), inverse=True)
        density = physical_log_prob_with_gap(flow.model,
            torch.cat((shape, physical[:, 2:]), -1), gap, standard)+ld
        density += tensor(flow.target_transform.scales).log().sum()
        # Use probabilities and log(mean(p)), independently of the gradient
        # path's log-sigmoid/logsumexp expression, at the same F64 precision.
        probability = torch.stack([torch.sigmoid(member.logits_from_raw_tensor(context))
                                   for member in members]).mean(0)
        result[start:stop] = (density+probability.log()).cpu().numpy()[:stop-start]
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite independent float64 kernel replay')
    return result


def validate_saved_run(run, *, stride=1):
    """Reject changed inputs, missing/failed populations, and intermediate beta."""
    if type(stride) is not int or stride != 1:
        raise ValueError('phase estimator requires the complete population with stride=1; deterministic thinning is unsupported')
    protocol = json.loads((run/'protocol.json').read_text())
    receipt = json.loads((run/'result.json').read_text())
    if (receipt.get('status') != 'completed'
            or receipt.get('protocol_sha256') != digest(run/'protocol.json')
            or receipt.get('particles_sha256') != digest(run/'particles.npz')
            or receipt.get('observations') != protocol['observations']
            or receipt.get('particles') != protocol['particles']
            or receipt.get('final_stage', {}).get('stage') != protocol['stages']
            or receipt.get('final_stage', {}).get('beta') != 1.
            or np.asarray(receipt.get('log_normalizer'), float).shape != (len(protocol['observations']),)
            or not np.isfinite(np.asarray(receipt.get('log_normalizer'), float)).all()
            or any(receipt.get('extinct', []))):
        raise ValueError('complete unchanged beta=1 surviving population required; no dropping failures')
    with np.load(run/'particles.npz', allow_pickle=False) as saved:
        if 'extinct' in saved and np.any(saved['extinct']):
            raise ValueError('explicit extinct populations are not fitted or silently discarded')
        expected = (len(protocol['observations']), protocol['particles'])
        fields = {name: saved[name].copy() for name in ('summaries', 'scores', 'log_kernel')}
        for name, dimensions in [('summaries', (9,)), ('scores', (2,)), ('log_kernel', ())]:
            if fields[name].shape != expected+dimensions or not np.isfinite(fields[name]).all():
                raise ValueError('finite aligned saved summary, score, and kernel arrays required')
    return protocol, receipt, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=1, help='Must be1: complete populations are required')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--coefficients', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError('fresh diagnostic JSON and NPZ outputs required')
    protocol, receipt, saved = validate_saved_run(args.run, stride=args.stride)
    reference = Path(protocol['reference'])
    if (digest(reference/'protocol.json') != protocol['reference_protocol_sha256']
            or digest(reference/'observations.parquet') != protocol['observation_sha256']):
        raise ValueError('reference or observation identity changed')
    recorded = {Path(path).name: value for path, value in protocol['implementation_sha256'].items()}
    if len(recorded) != len(protocol['implementation_sha256']):
        raise ValueError('recorded source basenames must be unique')
    root = Path(__file__).resolve().parents[1]
    actual_sources = {name: digest(root/('configs' if name.endswith('.json') else 'sbsi')/name)
                      for name in NATIVE_SOURCES}
    if any(recorded.get(name) != value for name, value in actual_sources.items()):
        raise ValueError('native target or derivative source identity changed')
    identity = {key: protocol[key] for key in ('observations', 'fixed_shear',
        'reference_protocol_sha256', 'observation_sha256', 'density_batch_size', 'selection')}
    identity.update(native_source_sha256=actual_sources,
        sampling_precision='recorded float32 network target', derivative_precision='float64 network diagnostic',
        phase_derivative_protocol=DERIVATIVE_PROTOCOL,
        phase_control='intrinsic phase; uniform analytic prior; retained phase-independent response/crowding; beta=1')
    frozen = None
    if args.coefficients:
        frozen = FrozenPhaseControl.load(args.coefficients)
        frozen.validate_production(protocol['seed'], protocol['observations'], identity)
    observations = pd.read_parquet(reference/'observations.parquet').iloc[protocol['observations']].reset_index(drop=True)
    objects, particles, _ = saved['summaries'].shape
    observed = observations.iloc[np.repeat(np.arange(objects), particles)].reset_index(drop=True)
    values = saved['summaries'].reshape(-1, 9)
    shear = protocol['fixed_shear']
    batch_size = protocol['density_batch_size']
    torch.set_num_threads(1)
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device=args.device)
    detector = load_detection_classifier(V39_LIKE, device=args.device)
    members = getattr(detector, 'members', (detector,))
    if any(next(model.parameters()).dtype != torch.float32 for model in
           [flow.model, *[member.model for member in members]]):
        raise ValueError('recorded float32 sampling-network replay requires float32 loaded models')
    auxiliary = observed[list(flow.target_transform.target_names[2:])].to_numpy()
    likelihood = SummaryLikelihood(flow, detector, density_batch_size=batch_size)
    density32 = likelihood.log_kernel(SummaryBatch(values, auxiliary), observed, *shear)
    flow.model.double()
    for member in members:
        member.model.double()
    evaluator = SummaryGradients(flow, detector, batch_size=batch_size)
    density64, gradient = evaluator.evaluate(values, observed, shear, wrt='context')
    shear_density64, raw_scores = evaluator.evaluate(values, observed, shear, wrt='shear')
    replay64 = lambda x: float64_log_kernel(flow, detector, x, observed, shear, batch_size)
    independent64 = replay64(values)
    control = phase_stein_control(values, gradient)
    phase_checks = []
    for step in PHASE_DIFFERENCE_STEPS:
        upper = replay64(rotate_summary_phase(values, step))
        lower = replay64(rotate_summary_phase(values, -step))
        finite = (upper-lower)/(2*step)
        phase_checks.append(dict(step=step, max_absolute_error=float(np.max(np.abs(finite-control))),
            max_scaled_error=float(np.max(np.abs(finite-control)/np.maximum(1., np.abs(control))))))
    wrap = replay64(rotate_summary_phase(values, 2*np.pi))
    checks = dict(
        sampling_replay_max=float(np.max(np.abs(density32-saved['log_kernel'].ravel()))),
        float64_replay_max=float(max(np.max(np.abs(density64-independent64)),
                                    np.max(np.abs(density64-shear_density64)))),
        saved_score_replay_max=float(np.max(np.abs(raw_scores-saved['scores'].reshape(-1, 2)))),
        sampled_float32_float64_max=float(np.max(np.abs(density32-density64))),
        phase_wrap_max=float(np.max(np.abs(wrap-independent64))), phase_differences=phase_checks)
    passed = bool(checks['sampling_replay_max'] <= PRECISION_LIMITS['sampling_replay_atol']
        and checks['float64_replay_max'] <= PRECISION_LIMITS['float64_replay_atol']
        and checks['phase_wrap_max'] <= PRECISION_LIMITS['float64_replay_atol']
        and checks['saved_score_replay_max'] <= PRECISION_LIMITS['saved_score_atol']
        and all(v['max_scaled_error'] <= PRECISION_LIMITS['phase_gradient_scaled_atol'] for v in phase_checks))
    raw_scores, control = raw_scores.reshape(objects, particles, 2), control.reshape(objects, particles)
    arrays = dict(raw_scores=raw_scores, phase_control=control, saved_raw_scores=saved['scores'],
        density32=density32.reshape(objects, particles), density64=density64.reshape(objects, particles),
        delta_log_kernel=(density64-saved['log_kernel'].ravel()).reshape(objects, particles),
        observations=np.asarray(protocol['observations']), sampled_particle_indices=np.arange(protocol['particles'])[::args.stride])
    report = dict(schema='sbsi-phase-diagnostic-v1', run=str(args.run.resolve()), seed=protocol['seed'], observations=protocol['observations'],
        target_identity=identity, retained_particles=particles, stride=args.stride,
        derivative_protocol=DERIVATIVE_PROTOCOL,
        numerical_checks=checks, precision_limits=PRECISION_LIMITS, numerical_checks_pass=passed,
        sampled_float32_float64_agreement=checks['sampled_float32_float64_max'] <= PRECISION_LIMITS['sampled_float32_float64_atol'],
        exact_zero_mean_for_float32_target_established=False,
        device=args.device, device_name=torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu',
        raw_score_mean=raw_scores.mean(1).tolist(), raw_score_variance=raw_scores.var(1).tolist(),
        control_mean=control.mean(1).tolist(), control_variance=control.var(1).tolist(),
        log_normalizer=receipt['log_normalizer'], source_hashes={str(args.run/name): digest(args.run/name)
            for name in ('protocol.json', 'result.json', 'particles.npz')},
        implementation_sha256={str(Path(__file__).resolve()):digest(__file__),
            str(root/'sbsi/phase_control.py'):digest(root/'sbsi/phase_control.py')},
        scope='Read-only diagnostic; raw gates and normalizers unchanged. F64 controls on F32 states are qualified diagnostics, not exact F32 corrections.',
        limitations=['Finite-state precision replay is necessary, not a global target-equivalence bound.',
                     'Within-population variance is not independent-population MCSE.',
                     'A small corrected variance cannot establish mode coverage or rescue failed raw gates.'])
    if frozen is not None and passed:
        corrected = frozen.apply(raw_scores, control, seed=protocol['seed'],
            observations=protocol['observations'], target_identity=identity)['corrected_scores']
        arrays['corrected_scores'] = corrected
        report.update(coefficients=dict(path=str(args.coefficients.resolve()), sha256=digest(args.coefficients),
            metadata=frozen.metadata), corrected_score_mean=corrected.mean(1).tolist(),
            corrected_score_variance=corrected.var(1).tolist())
    elif frozen is not None:
        report['correction_blocked'] = 'numerical replay or precision check failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    report['arrays_sha256'] = digest(args.output.with_suffix('.npz'))
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('PHASE_CONTROL_DIAGNOSTIC', args.output, 'numerical_checks_pass', passed, flush=True)


if __name__ == '__main__':
    main()
