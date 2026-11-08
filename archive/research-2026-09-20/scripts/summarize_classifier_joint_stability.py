"""Report every fixed classifier/cut and leave-one-scene-out sensitivity.

No scene exclusion, model selection, calibration correction or acceptance gate.
Leave-one-out ranges are influence diagnostics, not confidence intervals.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from plots.constgold_property_bias import bootstrap_response
from scripts.train_full_parent_classifier import sha256, write_json
from scripts.transported_classifier_joint import ARMS


def bias_replicas(sufficient, weights, h, axis):
    if axis not in (1, 2) or not np.isfinite(h) or not 0 < h < 1:
        raise ValueError('positive shear and axis1/2 required')
    responses = {}
    for source in ('measured', 'model'):
        legs = {leg: {field: np.asarray(sufficient[source][leg][field], float)
                      for field in ('numerator', 'denominator')} for leg in ('plus', 'minus')}
        responses[source] = bootstrap_response(legs['plus'], legs['minus'], weights, h)[:, 0, axis-1]
    model = responses['model']
    if not np.isfinite(model).all() or np.any(model == 0):
        raise ValueError('finite nonzero model response required')
    result = 100*(responses['measured']/model-1)
    if not np.isfinite(result).all():
        raise ValueError('finite bias replicas required')
    return result


def influence_report(result):
    cases = result['cases']
    if len(cases) < 2 or len(set(cases)) != len(cases):
        raise ValueError('at least two distinct scenes required')
    n = len(cases)
    weights = np.concatenate((np.ones((1, n)), np.ones((n, n))-np.eye(n)))
    report = {}
    for cut in result['cuts']:
        replicas = {}
        for arm in ARMS:
            key = 'truth_parent/'+cut+'/'+arm
            values = bias_replicas(result['per_case_sufficient'][key], weights, result['h'], result['shear_axis'])
            np.testing.assert_allclose(values[0], result['summary'][key]['m_percent'], atol=1e-10, rtol=1e-12)
            replicas[arm] = values
        report[cut] = {}
        for arm, values in replicas.items():
            delta = values[1:]-values[0]
            reference = values-replicas['modeled_usable']
            key = 'truth_parent/'+cut+'/'+arm
            report[cut][arm] = {
                'm_percent': values[0],
                'case_bootstrap_se_pp': result['summary'][key]['m_standard_error_percentage_points'],
                'leave_one_out_m_percent': values[1:],
                'leave_one_out_min_max_percent': [values[1:].min(), values[1:].max()],
                'max_absolute_leave_one_out_change_pp': np.max(np.abs(delta)),
                'largest_influence_case': cases[int(np.argmax(np.abs(delta)))],
                'delta_from_historical_pp': reference[0],
                'leave_one_out_delta_from_historical_pp': reference[1:],
                'all_cases_in_point_estimate': True}
    return report


def read_verified(result_path, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    if receipt['result_sha256'] != sha256(result_path):
        raise ValueError('result differs from completed comparison receipt')
    result = json.loads(result_path.read_text())
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--receipt', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.result) != len(args.receipt) or args.output.exists():
        raise ValueError('paired results/receipts and new output required')
    endpoints = {}
    for path, receipt_path in zip(args.result, args.receipt):
        result, receipt = read_verified(path, receipt_path)
        name = receipt['endpoint']
        if name in endpoints:
            raise ValueError('duplicate endpoint')
        endpoints[name] = {'cases': result['cases'], 'h': result['h'], 'axis': result['shear_axis'],
            'result': str(path), 'result_sha256': sha256(path), 'receipt_sha256': sha256(receipt_path),
            'protocol_sha256': receipt['protocol_sha256'],
            'maximum_reference_sufficient_difference': receipt['maximum_reference_sufficient_difference'],
            'influence': influence_report(result)}
        print('CLASSIFIER_INFLUENCE_COMPLETE', name, len(result['cases']), flush=True)
    write_json(args.output, {'endpoints': endpoints, 'source_sha256': sha256(Path(__file__)),
        'calibration_claim': False,
        'limitations': ['all scenes/cuts/five classifier arms retained; no exclusions or model selection',
                       'leave-one-scene-out range measures influence, not a confidence interval',
                       'shared case-bootstrap errors omit finite-draw, training and simulation-family uncertainty',
                       'fresh four-scene influence is imprecise and not untouched final acceptance']})


if __name__ == '__main__':
    main()
