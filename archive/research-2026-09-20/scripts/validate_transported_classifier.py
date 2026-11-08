"""Coordinate-aware shared-draw classifier validation; all five arms retained."""
import argparse
import json
import hashlib
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from plots.constgold_property_bias import bootstrap_response, make_case_weights
from scripts.train_full_parent_classifier import sha256, write_json
from scripts.transported_classifier_joint import ARMS


def replay_reference(baseline, candidate):
    for key in ('cases', 'h', 'draws', 'seed', 'population', 'cohorts', 'cuts', 'case_reports'):
        if baseline[key] != candidate[key]:
            raise ValueError('reference contract mismatch: '+key)
    if baseline.get('shear_axis', 1) != candidate['shear_axis']:
        raise ValueError('reference axis mismatch')
    for key in ('parent_manifest_sha256', 'flow_sha256', 'classifiers', 'rblend'):
        if baseline['inputs'][key] != candidate['inputs'][key]:
            raise ValueError('reference input mismatch: '+key)
    maximum = 0.
    for key, sources in baseline['per_case_sufficient'].items():
        for source, legs in sources.items():
            for leg, fields in legs.items():
                for field, values in fields.items():
                    left = np.asarray(values)
                    right = np.asarray(candidate['per_case_sufficient'][key][source][leg][field])
                    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-10)
                    maximum = max(maximum, float(np.max(np.abs(left-right))))
        for field in ('measured_response', 'model_response', 'model_flow_response', 'model_blend_response'):
            np.testing.assert_allclose(baseline['summary'][key][field], candidate['summary'][key][field], rtol=1e-12, atol=1e-12)
    return maximum


def paired_changes(result):
    weights = make_case_weights(10000, len(result['cases']), 20260914)
    axis = result['shear_axis']-1
    report = {}
    for cut in result['cuts']:
        estimates, measured_reference = {}, None
        for arm in ARMS:
            key = 'truth_parent/'+cut+'/'+arm
            raw = result['per_case_sufficient'][key]
            stats = {source: {leg: {field: np.asarray(value) for field, value in fields.items()}
                     for leg, fields in legs.items()} for source, legs in raw.items()}
            if measured_reference is not None:
                for leg in ('plus', 'minus'):
                    for field in ('numerator', 'denominator'):
                        np.testing.assert_array_equal(measured_reference[leg][field], stats['measured'][leg][field])
            measured_reference = stats['measured']
            measured = bootstrap_response(stats['measured']['plus'], stats['measured']['minus'], weights, result['h'])[:, 0, axis]
            model = bootstrap_response(stats['model']['plus'], stats['model']['minus'], weights, result['h'])[:, 0, axis]
            estimates[arm] = 100*(measured/model-1)
            if not np.isfinite(estimates[arm]).all():
                raise ValueError('nonfinite classifier-arm bias replica')
        report[cut] = {}
        for left, right in combinations(ARMS, 2):
            delta = estimates[right]-estimates[left]
            point = result['summary']['truth_parent/'+cut+'/'+right]['m_percent']-result['summary']['truth_parent/'+cut+'/'+left]['m_percent']
            report[cut][right+'_minus_'+left] = {'delta_m_percentage_points': point,
                'paired_case_se_pp': delta.std(ddof=1), 'paired_case_ci95_pp': np.quantile(delta, [.025, .975])}
    return report



def execute_evaluator(protocol, script, arguments):
    """Run the exact pinned source bytes, independently of cached bytecode."""
    script = script.resolve()
    expected = protocol['files'][str(script)]
    if hashlib.sha256(script.read_bytes()).hexdigest() != expected:
        raise ValueError('evaluator source is not the pinned version')
    driver = """from pathlib import Path
import hashlib,sys
path,expected,*arguments=sys.argv[1:]
source=Path(path).read_bytes()
if hashlib.sha256(source).hexdigest()!=expected:
    raise RuntimeError('executed evaluator differs from frozen source')
sys.argv=[path,*arguments]
print('FROZEN_EVALUATOR_SOURCE',path,expected,flush=True)
exec(compile(source,path,'exec'),{'__name__':'__main__','__file__':path})
"""
    subprocess.run([sys.executable, '-u', '-c', driver, str(script), expected, *arguments], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('joint validation requires Slurm')
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen joint-validation input changed: '+path)
    endpoint = protocol['endpoints'][args.index]
    print('SCENE_CLASSIFIER_VALIDATION_START', endpoint['name'], flush=True)
    execute_evaluator(protocol, Path('scripts/evaluate_transported_classifier_joint.py'), endpoint['arguments'])
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('joint-validation input changed during execution: '+path)
    result_path, baseline_path = Path(endpoint['output']), Path(endpoint['baseline'])
    result = json.loads(result_path.read_text())
    maximum = replay_reference(json.loads(baseline_path.read_text()), result)
    comparisons = paired_changes(result)
    write_json(Path(endpoint['comparison']), {'endpoint': endpoint['name'], 'comparisons': comparisons,
        'maximum_reference_sufficient_difference': maximum, 'baseline_sha256': sha256(baseline_path),
        'result_sha256': sha256(result_path), 'protocol_sha256': sha256(args.protocol),
        'source_sha256': sha256(Path(__file__)), 'calibration_claim': False,
        'limitations': ['paired case errors exclude model training and finite-draw uncertainty',
                        'all fixed classifier arms/cuts retained; no checkpoint selection from joint bias']})
    for arm in ARMS:
        primary = result['summary']['truth_parent/mag25.8_radius_gt0.60/'+arm]
        print(f"SCENE_CLASSIFIER_JOINT_RESULT endpoint={endpoint['name']} arm={arm} m={primary['m_percent']:.8f} se={primary['m_standard_error_percentage_points']:.8f}", flush=True)
    print('SCENE_CLASSIFIER_VALIDATION_COMPLETE', endpoint['name'], flush=True)


if __name__ == '__main__':
    main()
