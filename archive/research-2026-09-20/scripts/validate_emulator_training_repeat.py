"""Validate a fixed-recipe emulator seed repeat with the original flow/classifier."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.compare_truth_parent_response import require_comparable
from scripts.train_full_parent_classifier import sha256, write_json
from scripts.validate_smooth_classifier import execute_evaluator


def require_training_repeat(original, repeat, seed, script_sha):
    allowed = {'params', 'model_sha256', 'script_sha256', 'train_history', 'development_pair_sums'}
    if ({k: v for k, v in original.items() if k not in allowed}
            != {k: v for k, v in repeat.items() if k not in allowed}
            or repeat['params'] != original['params'] | {'seed': seed}
            or repeat['script_sha256'] != script_sha
            or original['params']['seed'] == seed):
        raise ValueError('emulator repeat must change only its training seed')


def require_fixed_flow_classifier(reference, candidate):
    require_comparable(reference, candidate)
    if reference['inputs'] != candidate['inputs']:
        raise ValueError('emulator repeat changed flow, classifier, data or evaluator')
    for key in ('composition', 'output_conditions', 'prediction_device'):
        if reference['experimental_response'][key] != candidate['experimental_response'][key]:
            raise ValueError('emulator composition or execution changed')
    for key, stat in reference['per_case_sufficient'].items():
        for leg in ('plus', 'minus'):
            # Transport changes shapes only: radius/flux selection and U are fixed.
            np.testing.assert_array_equal(stat['model'][leg]['denominator'],
                candidate['per_case_sufficient'][key]['model'][leg]['denominator'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--freeze-output', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    protocol = json.loads(args.protocol.read_text())
    def verify():
        for path, expected in protocol['files'].items():
            if sha256(Path(path)) != expected:
                raise ValueError('frozen emulator-repeat input changed: '+path)
    verify()
    training_path = Path(protocol['emulator_training_protocol'])
    training = json.loads(training_path.read_text())
    for path, expected in training['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('emulator training source changed: '+path)
    original_root, repeat_root = Path(training['original_root']), Path(training['repeat_root'])
    original = json.loads((original_root/'metadata.json').read_text())
    repeat = json.loads((repeat_root/'metadata.json').read_text())
    require_training_repeat(original, repeat, training['repeat_seed'], sha256(Path(training['repeat_trainer'])))
    if repeat['model_sha256'] != sha256(repeat_root/'model.json'):
        raise ValueError('repeat model does not match completed training metadata')
    if args.freeze_output is not None:
        if args.index is not None or args.freeze_output.exists():
            raise ValueError('new frozen protocol required')
        for path in (repeat_root/'model.json', repeat_root/'metadata.json'):
            protocol['files'][str(path)] = sha256(path)
        protocol['template_sha256'] = sha256(args.protocol)
        protocol['repeated_emulator_sha256'] = repeat['model_sha256']
        verify()
        write_json(args.freeze_output, protocol)
        print('EMULATOR_REPEAT_FROZEN', repeat['model_sha256'], flush=True)
        return
    if args.index is None:
        raise ValueError('endpoint index required')
    endpoint = protocol['endpoints'][args.index]
    execute_evaluator(protocol, Path('scripts/evaluate_smooth_classifier_joint.py'), endpoint['arguments'])
    reference = json.loads(Path(endpoint['baseline']).read_text())
    candidate = json.loads(Path(endpoint['output']).read_text())
    require_fixed_flow_classifier(reference, candidate)
    if candidate['experimental_response']['metadata'] != repeat:
        raise ValueError('result used another emulator')
    verify()
    subprocess.run([sys.executable, '-u', '-m', 'scripts.compare_truth_parent_response',
        '--baseline', endpoint['baseline'], '--candidate', endpoint['output'], '--output', endpoint['comparison']], check=True)
    verify()
    write_json(Path(endpoint['receipt']), {'protocol_sha256': sha256(args.protocol),
        'result_sha256': sha256(Path(endpoint['output'])), 'comparison_sha256': sha256(Path(endpoint['comparison'])),
        'endpoint': endpoint['name'], 'emulator_sha256': repeat['model_sha256'],
        'original_candidate_retained': True, 'calibration_claim': False})
    s = candidate['summary']['truth_parent/mag25.8_radius_gt0.60/modeled_smooth']
    print('EMULATOR_REPEAT_JOINT_COMPLETE', endpoint['name'], s['m_percent'],
          s['m_standard_error_percentage_points'], flush=True)


if __name__ == '__main__':
    main()
