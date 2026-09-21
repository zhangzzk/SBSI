"""Same-process-host operational replay for the coordinate-aware evaluator."""
import argparse
import json
import os
from pathlib import Path

from scripts.train_full_parent_classifier import sha256, write_json
from scripts.validate_transported_classifier import execute_evaluator, replay_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('operational evaluation requires Slurm')
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('smoke input changed: '+path)
    for run in protocol['runs']:
        execute_evaluator(protocol, Path(run['script']), run['arguments'])
    baseline, candidate = (json.loads(Path(run['output']).read_text()) for run in protocol['runs'])
    maximum = replay_reference(baseline, candidate)
    for cut in candidate['cuts']:
        for arm in ('modeled_coordinate', 'modeled_transported'):
            if 'truth_parent/'+cut+'/'+arm not in candidate['summary']:
                raise ValueError('missing new classifier branch')
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('smoke input changed during evaluation: '+path)
    write_json(Path(protocol['receipt']), {'status': 'completed',
        'scope': 'operational small scene and two-epoch candidate only; no calibration interpretation',
        'maximum_reference_sufficient_difference': maximum, 'all_existing_branches_replayed': True,
        'classifier_arms': candidate['inputs']['classifier_arms'],
        'protocol_sha256': sha256(args.protocol),
        'outputs': {run['output']: sha256(Path(run['output'])) for run in protocol['runs']}})
    print('TRANSPORTED_CLASSIFIER_JOINT_SMOKE_COMPLETE', maximum, flush=True)


if __name__ == '__main__':
    main()
