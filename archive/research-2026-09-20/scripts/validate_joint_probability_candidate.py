"""Apply frozen downstream joint diagnostics to the common-descent candidate."""
import argparse
import json
import os
from pathlib import Path

from scripts import validate_per_leg_candidate as validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('endpoint', choices=('main', 'independent', 'fresh1', 'fresh2'))
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('diagnostics require Slurm')
    frozen = validation.read(args.protocol)
    for path, expected in frozen['file_sha256'].items():
        if validation.digest(path) != expected:
            raise ValueError(f'frozen validation input changed: {path}')
    # Each scheduled endpoint is a separate process. The shared evaluator
    # command construction is unchanged; only this declared candidate root varies.
    validation.CANDIDATE = validation.ROOT/'joint_probability_refinement_full'
    candidate = validation.CANDIDATE
    record, protocol = validation.read(candidate/'completed.json'), validation.read(candidate/'protocol.json')
    if (record['protocol_sha256'] != validation.digest(candidate/'protocol.json')
            or record['selected_sha256'] != validation.digest(candidate/'selected.pt')
            or protocol['initial_flow_sha256'] != validation.digest(validation.INITIAL)
            or protocol['script_sha256'] != validation.digest('scripts/refine_joint_guard_probability_flow.py')
            or protocol['probability_audit_result_sha256'] != validation.digest(validation.ROOT/'joint_gradient_full/result.json')):
        raise ValueError('common-descent refinement identity mismatch')
    eligible = validation.validate_selection(record, protocol)
    status = candidate/f'validation_{args.endpoint}_status.json'
    if status.exists():
        raise FileExistsError(status)
    report = {'endpoint': args.endpoint, 'selected_sha256': record['selected_sha256'],
              'refinement_completed_sha256': validation.digest(candidate/'completed.json'),
              'validation_protocol_sha256': validation.digest(args.protocol), 'calibration_claim': False}
    if eligible:
        print('JOINT_PROB_VALIDATION_START', args.endpoint, record['best_step'], flush=True)
        if args.endpoint.startswith('fresh'):
            validation.fresh(int(args.endpoint[-1]))
        else:
            validation.existing(args.endpoint)
        report['status'] = 'completed'
    else:
        report.update(status='skipped', reason='step zero retained; unchanged model already evaluated')
    status.write_text(json.dumps(report, indent=2)+'\n')
    print('JOINT_PROB_VALIDATION_END', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
