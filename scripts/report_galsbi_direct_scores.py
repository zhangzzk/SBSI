"""Direct shear scores and Louis curvature of saved persistent-scene chains.

Run under Slurm. This is an opt-in float64 diagnostic on the recorded sampler
target, not a new shear estimate or a certificate of posterior equilibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from sbsi.inference_moments import posterior_derivatives
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier
from sbsi.summary_gradients import SummaryGradients


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--validate-derivatives', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists() or args.stride < 1:
        raise ValueError('fresh output and positive stride required')
    protocol = json.loads((args.run/'protocol.json').read_text())
    centre = np.asarray(protocol['fixed_shear'], float)
    with np.load(args.run/'chain.npz') as chain:
        if chain['summaries'].shape[2] != 1:
            raise ValueError('persistent single-environment posterior histories required')
        values = chain['summaries'][protocol['warmup']::args.stride, :, 0].copy()
    n, rows, _ = values.shape
    ids = protocol['observations']
    if rows != 4*len(ids) or n < 4:
        raise ValueError('at least four retained time points and four chains per observation required')
    observed = pd.read_parquet(Path(protocol['reference'])/'observations.parquet').iloc[ids]
    repeated = observed.iloc[np.tile(np.repeat(np.arange(len(ids)), 4), n)].reset_index(drop=True)
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device=args.device)
    detector = load_detection_classifier(V39_LIKE, device=args.device)
    flow.model.double()
    for member in getattr(detector, 'members', (detector,)):
        member.model.double()
    torch.set_num_threads(2)
    evaluator = SummaryGradients(flow, detector, batch_size=4096)
    flat = values.reshape(-1, 9)
    density, score, hessian = evaluator.evaluate(flat, repeated, centre, wrt='shear', hessian=True)
    score = score.reshape(n, len(ids), 4, 2)
    hessian = hessian.reshape(n, len(ids), 4, 2, 2)
    report = dict(centre=centre.tolist(), run=str(args.run.resolve()), stride=args.stride,
        retained_times=n, observations=[], script_sha256=digest(__file__),
        source_hashes={str(args.run/p):digest(args.run/p) for p in ['protocol.json','chain.npz']},
        scope='Float64 direct derivatives on saved histories, with fixed intrinsic scenes and response. No selection term or shear estimate; posterior stationarity unestablished.')
    for i, oid in enumerate(ids):
        moment = posterior_derivatives(centre, score[:, i].reshape(-1, 2), hessian[:, i].reshape(-1, 2, 2))
        checks = [dict(rhat=float(az.rhat(score[:, i, :, a].T)),
                       ess=float(az.ess(score[:, i, :, a].T, method='bulk')),
                       mcse_stationarity_conditional=float(np.asarray(az.mcse(score[:, i, :, a].T, method='mean')).item()))
                  for a in range(2)]
        report['observations'].append(dict(observation=oid, score=moment.score.tolist(),
            log_numerator_hessian=moment.hessian.tolist(),
            mean_complete_data_hessian=hessian[:, i].mean((0, 1)).tolist(),
            observed_information_eigenvalues=np.linalg.eigvalsh(-moment.hessian).tolist(),
            diagnostics=checks, score_gate=all(c['rhat'] < 1.01 and c['ess'] > 400 for c in checks)))
    if args.validate_derivatives:
        # First retained time: every object and chain, no selection by outcome.
        q, data = flat[:rows], repeated.iloc[:rows]
        _, first, second = evaluator.evaluate(q, data, centre, wrt='shear', hessian=True)
        errors = []
        for axis in range(2):
            plus, minus = centre.copy(), centre.copy()
            plus[axis] += 1e-6; minus[axis] -= 1e-6
            numeric = (evaluator.evaluate(q, data, plus, wrt='shear')[1]
                       -evaluator.evaluate(q, data, minus, wrt='shear')[1])/2e-6
            relative = np.max(np.abs(numeric-second[:, :, axis])/np.maximum(1., np.abs(second[:, :, axis])))
            errors.append(float(relative))
        if max(errors) > 1e-4:
            raise ValueError(f'shear Hessian fails gradient-difference replay: {errors}')
        report['hessian_gradient_difference_max_scaled'] = errors
        # Same states/arithmetic for finite-ratio versus direct-derivative comparison.
        spacings = []
        for step in [.002, .0005, .000125, .00003125, .0000078125]:
            logprofiles = {}
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                moved = centre+step*np.array([dx, dy])
                kernel = evaluator.evaluate(flat, repeated, moved, wrt='shear')[0]
                ratio = (kernel-density).reshape(n, len(ids), 4)
                logprofiles[dx, dy] = logsumexp(ratio, axis=(0, 2))-np.log(n*4)
            finite_score = np.column_stack([(logprofiles[1, 0]-logprofiles[-1, 0])/(2*step),
                                           (logprofiles[0, 1]-logprofiles[0, -1])/(2*step)])
            finite_hessian = np.empty((len(ids), 2, 2))
            finite_hessian[:, 0, 0] = (logprofiles[1, 0]+logprofiles[-1, 0])/step**2
            finite_hessian[:, 1, 1] = (logprofiles[0, 1]+logprofiles[0, -1])/step**2
            finite_hessian[:, 0, 1] = finite_hessian[:, 1, 0] = (logprofiles[1, 1]-logprofiles[1, -1]
                -logprofiles[-1, 1]+logprofiles[-1, -1])/(4*step**2)
            spacings.append(dict(step=step, scores=finite_score.tolist(), hessians=finite_hessian.tolist()))
        report['matched_state_finite_mixture_stencils'] = spacings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), scores=score, hessians=hessian, centre=centre)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('direct-score report', args.output, 'gates', [x['score_gate'] for x in report['observations']], flush=True)


if __name__ == '__main__':
    main()
