"""Compare model and image responses on identical signed-shear scene legs.

Development diagnostic only: one scene and two independent probes do not give
a case-based calibration interval. Actual per-leg usability is held identical
on model and data sides; each side applies the primary cut to its own outputs.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from plots.constgold_property_bias import load_lookup
from sbsi.disk_response_transport import disk_response_transport
from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS
from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import DiskResponseMoment
from scripts.analyze_random_sign_probe_pilot import digest, load_leg
from scripts.evaluate_constgold_truth_parent import load_candidate_pairs, output_cuts
from scripts.matched_model_probe import SourceShearSigns, signed_scene_draw_responses, signed_shear_frames
from scripts.matched_response_probe import contributions, summarize_probes
from scripts.train_fixed_g0_flow import write_json


def sufficient_rows(mass, flow, blend):
    """One row per truth parent: selected mass and two shape-numerator parts."""
    mass, flow, blend = (np.asarray(x, dtype=float) for x in (mass, flow, blend))
    if (mass.ndim != 1 or flow.shape != (len(mass), 2) or blend.shape != flow.shape
            or not all(np.isfinite(x).all() for x in (mass, flow, blend))
            or np.any((mass < 0) | (mass > 1))):
        raise ValueError('finite aligned probability masses and two-output numerators required')
    if np.any((mass == 0) & np.any((flow != 0) | (blend != 0), axis=1)):
        raise ValueError('zero selected mass must have zero numerator')
    return np.column_stack((mass, flow, blend))


def baseline(rows):
    total = rows.sum(0)
    if total[0] <= 0:
        raise ValueError('positive zero-leg selected mass required')
    return (total[1:3]+total[3:5])/total[0], float(total[0])


def response_parts(plus, minus, mean, mass, *, h=.02, signs=None):
    """Selection centering belongs to the flow part; transport preserves cuts."""
    if (plus.shape != minus.shape or plus.ndim != 2 or plus.shape[1] != 5
            or not np.isfinite(mass) or mass <= 0 or not 0 < h < 1):
        raise ValueError('aligned sufficient rows, positive baseline and step required')
    delta = (plus-minus)/(2*h*mass)
    flow = delta[:, 1:3]-delta[:, :1]*np.asarray(mean)
    blend = delta[:, 3:5]
    if signs is not None:
        signs = np.asarray(signs)
        if signs.shape != (len(plus),) or not np.isin(signs, [-1, 1]).all():
            raise ValueError('one aligned Rademacher sign per parent required')
        flow, blend = flow*signs[:, None], blend*signs[:, None]
    flow, blend = flow.sum(0), blend.sum(0)
    return {'flow_component': flow, 'blend_component': blend, 'total': flow+blend}


def summarize(rows, signs, reference=None):
    mean, mass = baseline(rows['zero']) if reference is None else reference
    axes = {}
    for axis in (1, 2):
        plus, minus = rows[f'g{axis}+'], rows[f'g{axis}-']
        coherent = response_parts(plus, minus, mean, mass)
        mu_plus, _ = baseline(plus)
        mu_minus, _ = baseline(minus)
        parts = [response_parts(rows[f'p{p}g{axis}+'], rows[f'p{p}g{axis}-'], mean, mass, signs=s)
                 for p, s in enumerate(signs)]
        probes = np.stack([part['total'] for part in parts])
        axes[f'g{axis}'] = {'coherent': coherent, 'endpoint_ratio': (mu_plus-mu_minus)/.04,
                            'probe_components': parts, 'probe_estimates': probes,
                            **summarize_probes(probes, coherent['total'])}
    return {'baseline_mean': mean, 'baseline_mass': mass, 'axes': axes}


def paired_difference(model, measured):
    result = {}
    for axis in ('g1', 'g2'):
        candidate, truth = model['axes'][axis], measured['axes'][axis]
        delta = candidate['probe_estimates']-truth['probe_estimates']
        coherent = candidate['coherent']['total']-truth['coherent']['total']
        result[axis] = {'coherent_difference': coherent,
                        'probe_differences': delta, **summarize_probes(delta, coherent)}
    return result


def observed_leg(root, manifest, measured, group, h, ids):
    shape, selected, report = load_leg(root, manifest, measured, group, h, ids)
    matching = [r for r in measured['legs'] if (r['group'], r['h']) == (group, h)]
    if len(matching) != 1:
        raise ValueError('unique measured leg required')
    path = Path(matching[0]['output'])
    cross = pd.read_feather(path.parent.parent/'CrossMatch/tile180.0_-0.5_rot0_matched.feather')
    table = pd.read_feather(path).set_index('NUMBER').reindex(cross.set_index('id_input').reindex(ids)['id_detec'])
    radius, mag = table['FLUX_RADIUS'].to_numpy(float), table['MAG_AUTO'].to_numpy(float)
    usable = (np.isfinite(shape).all(1) & (np.square(shape).sum(1) < 1)
              & np.isfinite(radius) & (radius > 0) & np.isfinite(mag))
    np.testing.assert_array_equal(selected, usable & (radius > 3) & (mag < 25.8))
    numerator, mass = contributions(shape, selected)
    return sufficient_rows(mass, numerator, np.zeros_like(numerator)), usable, report


def validate_scope(manifest, source, diagnostic, small):
    """Recognize the preserved pre-scope-field small pilot explicitly."""
    if small:
        if (manifest['format'] != 'random_sign_response_operational_v1'
                or not 0 < source['area_deg2'] <= .01
                or diagnostic.get('scope') not in (None, 'small_area_operational')):
            raise ValueError('small operational image provenance required')
    elif (manifest['format'] != 'random_sign_response_full_scene_development_v1'
          or diagnostic.get('scope') != 'full_scene_development'
          or source.get('scope') != 'full_scene_development' or source['area_deg2'] != 1.
          or source.get('reference_seed') != 123
          or source.get('scene_count_rule') != 'exact original generated count after prior replay'):
        raise ValueError('full-area original-prior image provenance required')


def prepare(args):
    root = args.probe_root
    manifest = json.loads((root/'manifest.json').read_text())
    measured = json.loads((root/'measurement_manifest.json').read_text())
    diagnostic = json.loads((root/'response_diagnostic.json').read_text())
    audit = json.loads((root/'render_validation.json').read_text())
    source_root = Path(manifest['source_root'])
    source = json.loads((source_root/'manifest.json').read_text())
    source_measured = json.loads((source_root/'measurement_manifest.json').read_text())
    case = source['case']
    if (measured['status'] != 'completed' or audit['status'] != 'passed'
            or diagnostic['probe_measurement_sha256'] != digest(root/'measurement_manifest.json')
            or not diagnostic['zero_replay_shapes_and_selection_exact']
            or not audit['zero_replay_image_and_psf_exact']
            or audit['manifest_sha256'] != digest(root/'manifest.json')
            or measured['render_audit_sha256'] != digest(root/'render_validation.json')
            or manifest['source_manifest_sha256'] != digest(source_root/'manifest.json')
            or manifest['source_measurement_sha256'] != digest(source_root/'measurement_manifest.json')
            or manifest['signs_sha256'] != digest(root/'probe_signs.npz')):
        raise ValueError('completed matching image-control and measurement receipts required')
    validate_scope(manifest, source, diagnostic, args.operational_small_scene)
    parent = args.prepared_root/'parent'/f'case{case:03d}.npz'
    parent_manifest = json.loads((parent.parent/'manifest.json').read_text())
    record = parent_manifest['case_reports'][str(case)]
    if (record['sha256'] != digest(parent)
            or record['scene_manifest_sha256'] != digest(source_root/'manifest.json')
            or record['measurement_manifest_sha256'] != digest(source_root/'measurement_manifest.json')):
        raise ValueError('parent population does not belong to the matched scene')
    with np.load(parent, allow_pickle=False) as data:
        ids, context = data['input_index'], data['context']
    if not len(ids) or not np.all(ids[1:] > ids[:-1]):
        raise ValueError('sorted unique nonempty truth-parent identities required')
    generated_path = Path(source['legs'][0]['catalogue'])
    if digest(generated_path) != source['legs'][0]['catalogue_sha256']:
        raise ValueError('generated scene changed')
    generated = pd.read_feather(generated_path, columns=['index', 'r'])
    expected = generated.loc[(generated['index'] >= len(generated)//2) & (generated['r'] < 26), 'index'].to_numpy()
    np.testing.assert_array_equal(ids, expected)
    if any(leg['unrendered_parent_ids'] for leg in audit['legs']):
        raise ValueError('unrendered parents require a revised explicit common population')
    with np.load(root/'probe_signs.npz', allow_pickle=False) as data:
        source_ids, sign_values = data['ids'], data['signs']
    np.testing.assert_array_equal(source_ids, generated['index'])
    if sign_values.shape != (2, len(source_ids)):
        raise ValueError('the frozen two-probe experiment is required')
    fields = [SourceShearSigns(source_ids, s) for s in sign_values]
    coherent = SourceShearSigns(source_ids, np.ones(len(source_ids), dtype=np.int8))
    descriptions = [('zero', source_root, source, source_measured, 'g0', 0., None, coherent)]
    for axis in (1, 2):
        for suffix, h in (('+', .02), ('-', -.02)):
            descriptions.append((f'g{axis}{suffix}', source_root, source, source_measured,
                                 f'g{axis}_h0.02', h, axis, coherent))
        for p, field in enumerate(fields):
            for suffix, h in (('+', .02), ('-', -.02)):
                descriptions.append((f'p{p}g{axis}{suffix}', root, manifest, measured,
                                     f'probe{p}_g{axis}', h, axis, field))
    rows, usability, frames, leg_fields, reports = {}, {}, {}, {}, {}
    for key, directory, scene, fitted, group, h, axis, field in descriptions:
        rows[key], usability[key], reports[key] = observed_leg(directory, scene, fitted, group, h, ids)
        frames[key] = (pd.DataFrame(context, columns=FLOW_FEATURES) if axis is None else
                       signed_shear_frames(context, ids, field, h=abs(h), axis=axis)[0 if h > 0 else 1])
        leg_fields[key] = (h, axis, field)
    replay, _, _ = observed_leg(root, manifest, measured, 'replay_g0', 0., ids)
    np.testing.assert_array_equal(replay, rows['zero'])
    signs = np.stack([field.at(ids) for field in fields])
    observed = summarize(rows, signs)
    for axis in ('g1', 'g2'):
        np.testing.assert_allclose(observed['axes'][axis]['probe_estimates'], diagnostic['results'][axis]['probe_estimates'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(observed['axes'][axis]['coherent']['total'], diagnostic['results'][axis]['coherent']['baseline_influence'], atol=1e-12, rtol=0)
    lookup = load_lookup((args.prepared_root/'pair_lookup.feather',), (case,))[case]
    np.testing.assert_array_equal(lookup['input_index'], ids)
    pairs, pair_metadata = load_candidate_pairs(args.prepared_root/'pairs', case, parent, ids, lookup['R_blend'])
    return case, ids, frames, leg_fields, usability, signs, observed, pairs, pair_metadata, reports


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-root', type=Path, required=True)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--flow', action='append', required=True, help='label=/absolute/checkpoint/path')
    parser.add_argument('--emulator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--draws', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=7301)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--operational-small-scene', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm and a new output path required')
    if args.draws < 2 or args.draws % 2 or args.batch_size < 1 or args.threads < 1:
        raise ValueError('positive batch/threads and even draws required')
    torch.set_num_threads(args.threads)
    case, ids, frames, fields, usability, signs, observed, pairs, pair_metadata, reports = prepare(args)
    print(f'MATCHED_MODEL_INPUTS_VERIFIED case={case} parents={len(ids)} legs={len(frames)}', flush=True)
    candidate = DiskResponseMoment(args.emulator, nthread=args.threads, device='cpu')
    models = {}
    for specification in args.flow:
        label, path = specification.split('=', 1)
        if not label or label in models:
            raise ValueError('distinct nonempty model labels required')
        bundle = load_measurement_model(Path(path), device=torch.device(args.device))
        bundle.model.eval()
        if (bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES)
                or bundle.target_transform.target_names != list(FLOW_TARGETS)):
            raise ValueError('flow feature/output contract mismatch')
        contexts = {key: bundle.context_tensor(frame) for key, frame in frames.items()}
        rows = {key: np.zeros((len(ids), 5)) for key in frames}
        pair_ids = pairs['index_input_p'].to_numpy()
        started = time.monotonic()
        for block, start in enumerate(range(0, len(ids), args.batch_size)):
            stop = min(start+args.batch_size, len(ids))
            lo = np.searchsorted(pair_ids, ids[start], side='left')
            hi = np.searchsorted(pair_ids, ids[stop-1], side='right')
            part = pairs.iloc[lo:hi]
            for key, context in contexts.items():
                # Reinitializing with this same seed is identical to shared
                # base draws, while retaining only one leg in device memory.
                values = physical_context_draws(bundle.model, [context[start:stop]], args.draws,
                                                 args.seed+10000019*case+1000003*block)[0]
                if not bool(torch.isfinite(values).all()):
                    raise ValueError('nonfinite model draw')
                h, axis, field = fields[key]
                if axis is None:
                    shift = torch.zeros_like(values[..., :2], dtype=torch.float64)
                else:
                    response, _ = signed_scene_draw_responses(candidate, part, ids[start:stop], values.cpu().numpy(), field)
                    direction = torch.zeros(2, dtype=torch.float64, device=values.device)
                    direction[axis-1] = h
                    velocity = torch.as_tensor(response, dtype=torch.float64, device=values.device)[..., None]*direction
                    moved, logdet = disk_response_transport(values[..., :2], velocity)
                    if not bool(torch.isfinite(moved).all() and torch.isfinite(logdet).all()):
                        raise ValueError('nonfinite disk transport')
                    shift = moved-values[..., :2]
                keep = output_cuts(values)[..., 1].double()
                mass = keep.mean(1).cpu().numpy()
                flow = (keep[..., None]*values[..., :2]).mean(1).cpu().numpy()
                blend = (keep[..., None]*shift).mean(1).cpu().numpy()
                use = usability[key][start:stop]
                rows[key][start:stop] = sufficient_rows(mass*use, flow*use[:, None], blend*use[:, None])
            print(f'MATCHED_MODEL_PROGRESS flow={label} rows={stop}/{len(ids)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        own = summarize(rows, signs)
        reference = summarize(rows, signs, (observed['baseline_mean'], observed['baseline_mass']))
        models[label] = {'checkpoint': path, 'checkpoint_sha256': digest(path), 'own_normalization': own,
                          'measured_reference_normalization': reference,
                          'own_minus_observed': paired_difference(own, observed),
                          'reference_minus_observed': paired_difference(reference, observed),
                          'selected_mass_by_leg': {key: float(value[:, 0].sum()) for key, value in rows.items()}}
    source_paths = [Path(__file__), Path('scripts/matched_model_probe.py'), Path('scripts/matched_response_probe.py'),
                    Path('scripts/analyze_random_sign_probe_pilot.py'), Path('scripts/evaluate_constgold_truth_parent.py'),
                    Path('sbsi/flow_guard_response.py'), Path('sbsi/output_conditioned_response.py'),
                    Path('sbsi/disk_response_transport.py')]
    result = {'status': 'completed', 'case': case, 'truth_parents': len(ids), 'h': .02,
              'primary_cut': 'measured radius>0.6arcsec (3px), MAG_AUTO<25.8', 'usability': 'actual per-leg U',
              'draws': args.draws, 'seed': args.seed, 'batch_size': args.batch_size, 'device': args.device,
              'scope': 'small_area_operational' if args.operational_small_scene else 'full_scene_development',
              'observed': observed, 'models': models, 'leg_reports': reports, 'pair_metadata': pair_metadata,
              'emulator_metadata_sha256': digest(args.emulator/'metadata.json'),
              'emulator_sha256': digest(args.emulator/'model.json'),
              'probe_measurement_sha256': digest(args.probe_root/'measurement_manifest.json'),
              'probe_diagnostic_sha256': digest(args.probe_root/'response_diagnostic.json'),
              'source_sha256': {str(path): digest(path) for path in source_paths},
              'calibration_claim': False,
              'limitations': ['one scene/two probes: conditional probe SE, no case-based interval',
                              'finite-shear and latent-integration error remain to be quantified',
                              'observed usability is an oracle; this is not a classifier validation',
                              'flow/blend algebraic components are not the physical self/neighbour split']}
    write_json(args.output, result)
    print('MATCHED_MODEL_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
