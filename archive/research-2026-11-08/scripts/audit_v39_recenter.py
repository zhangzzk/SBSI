"""Audit saved recentering moments without evaluating or changing the likelihood."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from combine_hybrid_pass import _bfgs_information, _summarize
from v39_recenter_math import summarize, paired


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    finite = values[np.isfinite(values)]
    return dict(finite_count=len(finite), total_count=len(values),
                quantiles=np.quantile(finite, [.5, .9, .99]).tolist() if len(finite) else None)


def derivatives(path, center):
    data = json.loads(path.read_text())
    h = data['finite_difference_step']
    bank = {(round(p['g1'], 14), round(p['g2'], 14)):
            np.log(p['detected_and_selected_mass']) for p in data['points']}

    def value(x, y):
        return bank[round(center[0]+x*h, 14), round(center[1]+y*h, 14)]

    grad = np.array([value(1, 0)-value(-1, 0), value(0, 1)-value(0, -1)])/(2*h)
    hess = np.diag([value(1, 0)-2*value(0, 0)+value(-1, 0),
                    value(0, 1)-2*value(0, 0)+value(0, -1)])/h**2
    hess[0, 1] = hess[1, 0] = (value(1, 1)-value(1, -1)-value(-1, 1)+value(-1, -1))/(4*h**2)
    return grad, hess


def audit(root, output):
    source = Path(json.loads((root/'protocol.json').read_text())['source_root'])
    observed = pd.read_parquet(source/'mock/measurements.parquet')
    mag = (observed['measured_mag_auto'].to_numpy() if 'measured_mag_auto' in observed else
           30-2.5*np.log10(observed['measured_flux_from_mag_auto'].to_numpy()))
    bright = mag < 22
    centers = [np.array(json.loads((root/f'pass_{i:02d}/center.json').read_text())['center']) for i in (0, 1)]
    norms = [derivatives(path, center) for path, center in zip(
        [source/'assembled/normalization.json', root/'pass_01/normalization.json'], centers)]
    shapes = np.square(observed[['measured_ngmix_g1', 'measured_ngmix_g2']].to_numpy()).sum(1)
    masks = {'bright_mag_lt22': bright, 'mag22_to24': (mag >= 22) & (mag < 24),
             'mag_ge24': mag >= 24, 'shape_abs_gt099': shapes > .99**2}
    assert sum(masks[k].sum() for k in ('bright_mag_lt22', 'mag22_to24', 'mag_ge24')) == len(mag)
    record = dict(status='complete', source_root=str(root), n_observations=len(mag),
                  centers=[c.tolist() for c in centers], bright_rows=int(bright.sum()),
                  normalization=[dict(gradient=g.tolist(), hessian=h.tolist()) for g, h in norms],
                  arms={}, input_sha256={}, limitations=[
                      'Post-hoc diagnostic at two fixed centers, not a new converged chain.',
                      'BFGS only changes the solver; sampler and normalization convergence are unresolved.',
                      'Bright masks are measured properties and do not remove any rows from the inference target.'])
    for arm in ('gaussian', 'distance'):
        scores, infos, diagnostics = [], [], []
        for i in (0, 1):
            parts = []
            for j in range(20):
                folder = root/f'pass_{i:02d}/part_{j:02d}'
                receipt = json.loads((folder/'manifest.json').read_text())
                path = folder/f'{arm}.npz'
                assert digest(path) == receipt['output_sha256'][arm]
                record['input_sha256'][str(path)] = digest(path)
                with np.load(path) as saved:
                    parts.append({k: saved[k] for k in saved.files})
                np.testing.assert_array_equal(parts[-1]['object_rows'], np.arange(j*1000, (j+1)*1000))
            scores.append(np.concatenate([p['ladder_score'][-1] for p in parts]))
            infos.append(np.concatenate([p['ladder_information'][-1] for p in parts]))
            diagnostics.append({k: np.concatenate([p[k][-1] for p in parts]) for k in
                                ('weight_ess', 'weight_max_fraction', 'weight_relative_error', 'weight_pareto_k')})
            np.testing.assert_allclose(np.concatenate([p['score'] for p in parts]), scores[-1], rtol=0, atol=0)
            np.testing.assert_allclose(np.concatenate([p['information'] for p in parts]), infos[-1], rtol=0, atol=0)
        s0, s1 = scores
        i0, i1 = infos
        delta = centers[1]-centers[0]
        per_pass = []
        for i, (s, info, center) in enumerate(zip(scores, infos, centers)):
            result, _ = summarize(center, s, info)
            saved = json.loads((root/f'pass_{i:02d}/result.json').read_text())['rungs'][-1][arm]
            np.testing.assert_allclose(result['estimate'], saved['estimate'], rtol=0, atol=1.e-14)
            g, h = norms[i]
            no_norm, _ = summarize(center, s+g, info-h)
            per_pass.append(dict(result=result, score_sum=s.sum(0).tolist(),
                                 information_sum=info.sum(0).tolist(),
                                 numerator_only_estimate=no_norm.get('estimate'),
                                 normalization_effect=(np.array(result['estimate'])-no_norm['estimate']).tolist()))
        # Independent algebraic invariant: carrying every full moment unchanged
        # in a quadratic model returns the first-pass estimate at any centre.
        plain_carried = s0-np.einsum('nij,j->ni', i0, delta)
        carry_check, _ = summarize(centers[1], plain_carried, i0)
        np.testing.assert_allclose(carry_check['estimate'], per_pass[0]['result']['estimate'], atol=1.e-13, rtol=0)
        # Carry numerator only; replace common normalization on every row.
        carried_score = s0+norms[0][0]-np.einsum('nij,j->ni', i0-norms[0][1], delta)-norms[1][0]
        carried_info = i0-norms[0][1]+norms[1][1]
        hybrid_score = np.where(bright[:, None], s1, carried_score)
        hybrid_info = np.where(bright[:, None, None], i1, carried_info)
        full, full_influence = summarize(centers[1], s1, i1)
        hybrid, hybrid_influence = summarize(centers[1], hybrid_score, hybrid_info)
        nonlinear = s1-carried_score
        bfgs, history = _bfgs_information(centers[0], s0.sum(0), i0.sum(0), centers[1], s1.sum(0), None)
        bfgs_result = _summarize(centers[1], s1, i1, solver_information=bfgs,
                                 solver_method='historical_bfgs_from_saved_full_scores')
        hbfgs, _ = _bfgs_information(centers[0], s0.sum(0), i0.sum(0), centers[1], hybrid_score.sum(0), None)
        hybrid_bfgs = _summarize(centers[1], hybrid_score, hybrid_info, solver_information=hbfgs,
                                 solver_method='historical_bfgs_with_normalization_aware_bright_carry')
        group_results = {}
        direction = delta/np.linalg.norm(delta)
        scalar_error = nonlinear @ direction
        for name, mask in masks.items():
            group_results[name] = dict(count=int(mask.sum()),
                score_sum=[s[mask].sum(0).tolist() for s in scores],
                nonlinear_score_sum=nonlinear[mask].sum(0).tolist(),
                absolute_directional_nonlinearity_fraction=float(abs(scalar_error[mask]).sum()/abs(scalar_error).sum()),
                base_directional_information_fraction=float(direction@i0[mask].sum(0)@direction/(direction@i0.sum(0)@direction)),
                diagnostics=[{k: quantiles(v[mask]) for k, v in d.items()} for d in diagnostics])
        worst = np.argsort(abs(scalar_error))[-20:][::-1]
        record['arms'][arm] = dict(passes=per_pass, hybrid_observed_newton=hybrid,
            hybrid_minus_full=paired(full, hybrid, full_influence, hybrid_influence),
            bfgs_full=bfgs_result, bfgs_update=history, bfgs_hybrid=hybrid_bfgs,
            directional_secant_over_initial_curvature=float(delta@(s0.sum(0)-s1.sum(0))/(delta@i0.sum(0)@delta)),
            directional_secant_over_second_curvature=float(delta@(s0.sum(0)-s1.sum(0))/(delta@i1.sum(0)@delta)),
            groups=group_results,
            worst_nonlinearity_rows=[dict(row=int(j), magnitude=float(mag[j]),
                shape_abs=float(np.sqrt(shapes[j])), nonlinear_score=nonlinear[j].tolist(),
                relative_errors=[float(d['weight_relative_error'][j]) for d in diagnostics]) for j in worst])
    historical = source.parent/'inference_proxy_topk_size075_n100k_v1/hybrid/complete100k_proxytop_k1024_m8192_iter0_4_exactnorm_bfgs_v1/chain.json'
    if historical.exists():
        record['historical_complete100k'] = dict(path=str(historical), sha256=digest(historical),
                                               payload=json.loads(historical.read_text()))
    record['script_sha256'] = digest(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('input_sha256', 'historical_complete100k')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
