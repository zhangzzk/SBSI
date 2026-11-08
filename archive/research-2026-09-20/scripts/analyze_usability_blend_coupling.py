"""Exact cached decomposition of modeled-U versus actual-U joint response.

All cross-normalized quantities are algebraic diagnostics, never corrections.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def response(numerator, denominator, h):
    if h <= 0 or np.any(denominator <= 0):
        raise ValueError('positive h and both leg masses required')
    return (numerator[:, 0]/denominator[:, 0, None]-numerator[:, 1]/denominator[:, 1, None])/(2*h)


def decompose(npred, dpred, nactual, dactual, h):
    pp = response(npred, dpred, h)
    aa = response(nactual, dactual, h)
    pa = response(npred, dactual, h)
    ap = response(nactual, dpred, h)
    numerator = .5*((pa-aa)+(pp-ap))
    denominator = .5*((ap-aa)+(pp-pa))
    np.testing.assert_allclose(numerator+denominator, pp-aa, atol=1e-12, rtol=1e-12)
    return {'actual_U_response': aa, 'modeled_U_response': pp, 'difference': pp-aa,
            'numerator_attribution': numerator, 'denominator_attribution': denominator}


def describe(x):
    return {'estimate': x[0], 'case_bootstrap_se': x[1:].std(0, ddof=1),
            'case_bootstrap_ci95': np.moveaxis(np.quantile(x[1:], [.025, .975], axis=0), 0, -1)}


def analyze(path):
    data = json.loads(path.read_text())
    n = len(data['cases'])
    if n < 2:
        raise ValueError('multiple cases required')
    weights = np.concatenate((np.ones((1, n)), np.random.default_rng(20260914).multinomial(n, np.full(n, 1/n), size=10000)))
    result = {}
    for cut in data['cuts']:
        key = 'truth_parent/'+cut+'/'
        branches = {}
        for name in ('actual_usable_flags', 'modeled_usable'):
            raw = data['per_case_sufficient'][key+name]['model']
            branches[name] = {field: np.stack([np.einsum('bc,c...->b...', weights,
                np.asarray(raw[leg][field])[:, 0]) for leg in ('plus', 'minus')], 1)
                for field in ('numerator', 'flow_numerator', 'blend_numerator', 'denominator')}
            for field, reference in [('numerator', 'model_response'), ('flow_numerator', 'model_flow_response'), ('blend_numerator', 'model_blend_response')]:
                replay = response(branches[name][field], branches[name]['denominator'], data['h'])[0]
                np.testing.assert_allclose(replay, data['summary'][key+name][reference], atol=1e-12, rtol=1e-12)
        a, p = branches['actual_usable_flags'], branches['modeled_usable']
        components = {name: decompose(p[field], p['denominator'], a[field], a['denominator'], data['h'])
                      for name, field in [('total', 'numerator'), ('flow', 'flow_numerator'), ('blend', 'blend_numerator')]}
        for field in components['total']:
            np.testing.assert_allclose(components['total'][field], components['flow'][field]+components['blend'][field], atol=1e-12, rtol=1e-12)
        result[cut] = {name: {field: describe(value) for field, value in part.items()} for name, part in components.items()}
        result[cut]['mean_leg_mass_relative_difference'] = describe(p['denominator'].mean(1)/a['denominator'].mean(1)-1)
    return {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'cases': data['cases'], 'h': data['h'], 'axis': data.get('shear_axis', 1), 'cuts': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('bootstrap requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {'results': [analyze(path) for path in args.result], 'bootstrap_seed': 20260914,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'definition': 'modeled-U minus actual-U model response; numerator/denominator attribution averages both substitution orders',
              'limitations': ['algebraic decomposition, not causal attribution or empirical correction',
                              'case bootstrap omits model-training uncertainty; no m or calibration claim',
                              'fixed cached predictions and all declared cuts; no model update']}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False,
                          default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())+'\n')
    print('USABILITY_BLEND_COUPLING_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
