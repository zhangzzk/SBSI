"""Repeat the frozen six-arm evaluator with only its latent seed changed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.train_full_parent_classifier import sha256
from scripts.validate_smooth_classifier import execute_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    protocol = json.loads(args.protocol.read_text())
    def verify():
        for path, expected in protocol['files'].items():
            if sha256(Path(path)) != expected:
                raise ValueError('frozen repeat input changed: '+path)
    verify()
    endpoint = protocol['endpoints'][args.index]
    execute_evaluator(protocol, Path('scripts/evaluate_smooth_classifier_joint.py'), endpoint['arguments'])
    verify()
    subprocess.run([sys.executable, '-u', '-m', 'scripts.compare_truth_parent_response',
                    '--baseline', endpoint['baseline'], '--candidate', endpoint['output'],
                    '--output', endpoint['comparison'], '--independent-latent-repeat'], check=True)
    verify()
    print('SMOOTH_LATENT_REPEAT_COMPLETE', endpoint['name'], flush=True)


if __name__ == '__main__':
    main()
