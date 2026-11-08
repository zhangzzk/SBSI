"""Freeze and evaluate one flow-refinement repeat with the original classifier."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.check_flow_refinement_repeat_protocol import main as check_training
from scripts.train_full_parent_classifier import sha256, write_json
from scripts.validate_smooth_classifier import execute_evaluator


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
                raise ValueError('frozen input changed: '+path)
    verify()
    if args.freeze_output is not None:
        if args.index is not None or args.freeze_output.exists():
            raise ValueError('new freeze output without endpoint index required')
        check_training()
        root = Path(protocol['repeat_root'])
        completed = json.loads((root/'completed.json').read_text())
        training = json.loads((root/'protocol.json').read_text())
        original = json.loads(Path(protocol['original_training_protocol']).read_text())
        if (completed['smoke'] or training['smoke']
                or completed['protocol_sha256'] != sha256(root/'protocol.json')
                or completed['selected_sha256'] != sha256(root/'selected.pt')
                or training['script_sha256'] != sha256(Path('scripts/refine_joint_guard_probability_flow_repeat.py'))):
            raise ValueError('matching completed full refinement required')
        for key in ('initial_flow_sha256', 'domain_manifest_sha256', 'steps', 'learning_rates',
                    'density_tolerance_nats_per_row', 'selection', 'reference_protocol_sha256',
                    'target_bank_manifest_sha256', 'probability_reference_sha256'):
            if training[key] != original[key]:
                raise ValueError('refinement recipe changed: '+key)
        for key in ('gradient_seed', 'line_search_seed', 'validation_seed', 'validation_mass_seed'):
            if training[key] != original[key]+101:
                raise ValueError('unexpected repeat seed')
        if completed['selected_full_validation_nll'] > completed['baseline_full_validation_nll']+.001:
            raise ValueError('repeat violates density gate')
        for path in (root/'completed.json', root/'protocol.json', root/'selected.pt', root/'history.json'):
            protocol['files'][str(path)] = sha256(path)
        protocol['template_sha256'] = sha256(args.protocol)
        protocol['selected_sha256'] = completed['selected_sha256']
        protocol['selected_step'] = completed['best_step']
        verify()
        write_json(args.freeze_output, protocol)
        print('FLOW_REPEAT_FROZEN', completed['best_step'], completed['selected_sha256'], flush=True)
        return
    if args.index is None:
        raise ValueError('endpoint index required')
    endpoint = protocol['endpoints'][args.index]
    execute_evaluator(protocol, Path('scripts/evaluate_smooth_classifier_joint.py'), endpoint['arguments'])
    verify()
    subprocess.run([sys.executable, '-u', '-m', 'scripts.compare_truth_parent_response',
                    '--baseline', endpoint['baseline'], '--candidate', endpoint['output'],
                    '--output', endpoint['comparison']], check=True)
    verify()
    result = json.loads(Path(endpoint['output']).read_text())
    record = result['summary']['truth_parent/mag25.8_radius_gt0.60/modeled_smooth']
    write_json(Path(endpoint['receipt']), {'protocol_sha256': sha256(args.protocol),
        'result_sha256': sha256(Path(endpoint['output'])), 'comparison_sha256': sha256(Path(endpoint['comparison'])),
        'selected_flow_sha256': protocol['selected_sha256'], 'endpoint': endpoint['name'],
        'calibration_claim': False, 'original_candidate_retained': True})
    print('FLOW_REPEAT_JOINT_COMPLETE', endpoint['name'], record['m_percent'], record['m_standard_error_percentage_points'], flush=True)


if __name__ == '__main__':
    main()
