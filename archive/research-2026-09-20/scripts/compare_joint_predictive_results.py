"""Separate shear-even density offsets from shear-odd response discrepancies."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from plots.constgold_property_bias import make_case_weights
from scripts.compare_truth_parent_response import require_comparable
from scripts.train_fixed_g0_flow import sha256, write_json


def integrate_moments(raw, weights):
    mass = np.asarray(raw['mass'])
    defined = np.asarray(raw['moment_mass'])
    total = weights @ mass
    moment_total = weights @ defined
    if (total <= 0).any() or (moment_total <= 0).any():
        raise ValueError('empty scene-bootstrap diagnostic population')
    means = np.einsum('bc,ckd->bkd', weights, np.asarray(raw['first']))/moment_total[..., None]
    return {'mean': means, 'selection_fraction': total/total[:, :1]}


def residual_modes(per_case, weights, h):
    if not np.isfinite(h) or h <= 0:
        raise ValueError('positive finite shear amplitude required')
    residuals = {}
    for leg in ('plus', 'minus'):
        integrated = {source: integrate_moments(per_case[source][leg], weights)
                      for source in ('observed', 'generated')}
        residuals[leg] = {key: integrated['generated'][key]-integrated['observed'][key]
                          for key in ('mean', 'selection_fraction')}
    return {f'{field}/{mode}': (residuals['plus'][field]+sign*residuals['minus'][field])/scale
            for field in ('mean', 'selection_fraction')
            for mode, sign, scale in [('even', 1, 2), ('odd_derivative', -1, 2*h)]}


def uncertainty(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or len(values) < 3:
        raise ValueError('finite point and bootstrap estimates required')
    return {'estimate': values[0], 'case_bootstrap_se': values[1:].std(0, ddof=1),
            'case_bootstrap_ci95': np.moveaxis(np.quantile(values[1:], [.025, .975], axis=0), 0, -1)}


def marginal_residuals(per_case, weights, shape):
    result = {}
    ncases = weights.shape[1]
    for leg in ('plus', 'minus'):
        axes = {}
        for axis in range(len(shape)):
            probabilities = {}
            reduce = tuple(i+1 for i in range(len(shape)) if i != axis)
            for source in ('observed', 'generated'):
                stats = per_case[source][leg]
                hist = np.asarray(stats['histogram']).reshape(ncases, *shape)
                marginalized = hist.sum(axis=reduce)
                total = weights @ np.asarray(stats['mass'])[:, 0]
                probabilities[source] = (weights @ marginalized)/total[:, None]
            axes[str(axis)] = uncertainty(probabilities['generated']-probabilities['observed'])
        result[leg] = axes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('predictive case-bootstrap analysis requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    data = {name: json.loads(path.read_text()) for name, path in
            [('baseline', args.baseline), ('candidate', args.candidate)]}
    require_comparable(data['baseline'], data['candidate'])
    if data['baseline'].get('shear_axis', 1) != data['candidate'].get('shear_axis', 1):
        raise ValueError('shear axes differ')
    diagnostics = {name: result.get('predictive_diagnostics') for name, result in data.items()}
    if any(value is None for value in diagnostics.values()):
        raise ValueError('both joint evaluations must record predictive diagnostics')
    base, candidate = diagnostics['baseline'], diagnostics['candidate']
    if base['metadata'] != candidate['metadata'] or base['cuts'] != candidate['cuts']:
        raise ValueError('diagnostic definitions differ')
    for leg in ('plus', 'minus'):
        if base['per_case']['observed'][leg] != candidate['per_case']['observed'][leg]:
            raise ValueError('observed sufficient statistics differ')
    ncases = len(data['baseline']['cases'])
    weights = np.concatenate([np.ones((1, ncases)), make_case_weights(10000, ncases, 20260914)])
    modes = {name: residual_modes(value['per_case'], weights, data[name]['h'])
             for name, value in diagnostics.items()}
    changes = {key: uncertainty(modes['candidate'][key]-modes['baseline'][key])
               for key in modes['baseline']}
    results = {name: {'residual_modes': {key: uncertainty(value) for key, value in modes[name].items()},
                     'marginal_probability_residuals': marginal_residuals(
                         diagnostics[name]['per_case'], weights, base['metadata']['histogram_shape'])}
               for name in data}
    write_json(args.output, {
        'cases': data['baseline']['cases'], 'h': data['baseline']['h'],
        'shear_axis': data['baseline'].get('shear_axis', 1), 'cuts': base['cuts'],
        'moment_coordinates': base['metadata']['moment_coordinates'],
        'histogram_coordinates': base['metadata']['histogram_coordinates'],
        'histogram_boundaries': base['metadata']['histogram_boundaries'],
        'results': results, 'paired_candidate_minus_baseline': changes,
        'source_sha256': {name: sha256(path) for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]},
        'script_sha256': sha256(__file__),
        'definitions': {'even': '(generated-observed at +h + generated-observed at -h)/2',
                        'odd_derivative': '(generated-observed at +h - generated-observed at -h)/(2h)',
                        'mean': 'conditional on positive photometry, exactly as recorded in diagnostic metadata',
                        'histogram': 'unconditional actual-U mass including separately retained invalid photometry'},
        'limitations': ['descriptive discrepancies, not a density-equivalence acceptance gate',
                        'case bootstrap excludes training uncertainty and finite-draw bias',
                        'marginal bins do not establish full conditional joint density',
                        'no checkpoint or selection threshold is chosen from these results']})
    print(f'PREDICTIVE_COMPARISON_COMPLETE cases={ncases} output={args.output}', flush=True)


if __name__ == '__main__':
    main()
