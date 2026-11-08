"""Audit both response components from frozen, paired scene statistics."""
import argparse
import os
from pathlib import Path

import numpy as np

from plots.constgold_property_bias import bootstrap_response, make_case_weights
from scripts.summarize_classifier_joint_stability import read_verified
from scripts.train_full_parent_classifier import sha256, write_json


def response_vectors(sufficient, weights, h, axis):
    if axis not in (1, 2) or not np.isfinite(h) or not 0 < h < 1:
        raise ValueError('positive shear step and axis 1/2 required')
    values = {}
    for source in ('measured', 'model'):
        legs = {leg: {field: np.asarray(sufficient[source][leg][field], float)
                      for field in ('numerator', 'denominator')}
                for leg in ('plus', 'minus')}
        values[source] = bootstrap_response(legs['plus'], legs['minus'], weights, h)[:, 0, :]
    residual = values['measured']-values['model']
    diagonal = values['model'][:, axis-1]
    if not np.isfinite(diagonal).all() or np.any(diagonal == 0):
        raise ValueError('finite nonzero diagonal model response required')
    values['measured_minus_model'] = residual
    values['residual_over_model_diagonal_percent'] = 100*residual/diagonal[:, None]
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('nonfinite response vector')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm and new output required')
    endpoints = {}
    for endpoint in ('main', 'independent', 'fresh_g1', 'fresh_g2'):
        path = args.root/f'smooth_joint_{endpoint}.json'
        receipt_path = args.root/f'smooth_joint_{endpoint}_comparison.json'
        result, receipt = read_verified(path, receipt_path)
        if receipt['maximum_reference_sufficient_difference'] != 0:
            raise ValueError('exact reference replay required')
        n = len(result['cases'])
        weights = np.concatenate((np.ones((1, n)), make_case_weights(10000, n, 20260914)))
        summaries = {}
        for key, sufficient in result['per_case_sufficient'].items():
            vectors = response_vectors(sufficient, weights, result['h'], result['shear_axis'])
            original = result['summary'][key]
            for source in ('measured', 'model'):
                np.testing.assert_allclose(vectors[source][0], original[source+'_response'], atol=1e-12)
            normalized = vectors['residual_over_model_diagonal_percent']
            np.testing.assert_allclose(normalized[0, result['shear_axis']-1], original['m_percent'], atol=1e-10)
            np.testing.assert_allclose(normalized[1:, result['shear_axis']-1].std(ddof=1),
                                       original['m_standard_error_percentage_points'], atol=1e-10)
            summaries[key] = {name: {'point': v[0], 'case_bootstrap_se': v[1:].std(axis=0, ddof=1),
                                      'case_bootstrap_ci95': np.quantile(v[1:], [.025, .975], axis=0).T}
                              for name, v in vectors.items()}
        endpoints[endpoint] = {'cases': result['cases'], 'h': result['h'], 'axis': result['shear_axis'],
                               'result_sha256': sha256(path), 'receipt_sha256': sha256(receipt_path),
                               'summary': summaries}
        print('RESPONSE_VECTORS_COMPLETE', endpoint, flush=True)
    if (endpoints['fresh_g1']['cases'] != endpoints['fresh_g2']['cases']
            or endpoints['fresh_g1']['h'] != endpoints['fresh_g2']['h']):
        raise ValueError('two-axis endpoints must share scenes and amplitude')
    write_json(args.output, {'endpoints': endpoints, 'source_sha256': sha256(Path(__file__)),
        'vector_order': ['e1', 'e2'], 'bootstrap_replicates': 10000, 'bootstrap_seed': 20260914,
        'calibration_claim': False,
        'limitations': ['all existing cohorts, cuts and arms retained; no model or scene selection',
                       'off-diagonal residual normalized by the model diagonal is cross-component leakage, not multiplicative m',
                       'paired scene bootstrap omits training and finite-draw uncertainty',
                       'only the same four additional scenes supply both matrix columns; large samples supply g1 only',
                       'no property-resolved response statistics or untouched final acceptance added']})


if __name__ == '__main__':
    main()
