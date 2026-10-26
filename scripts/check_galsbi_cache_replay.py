"""Paired real-model check of invariant-work reuse; no convergence claim."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from check_galsbi_continuous_prior import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-c',
        'import runpy, sys; sys.path[:0]=[".", "scripts"]; '
        'runpy.run_path("scripts/check_galsbi_local_sampler.py", run_name="__main__")',
        '--reference', str(args.reference), '--device', args.device,
        '--seed', '2026100132', '--steps', '512', '--warmup', '256',
        '--initial-bank-seed', '2026100133',
        '--observation-indices', '0', '1', '2', '3', '--environment-update', 'persistent',
        '--environments', '1', '--marginalize-type', '--scale-adaptation', 'per-chain',
        '--sed-jump', '--joint-morphology']
    times = {}
    for name, extra in (('baseline', []), ('cached', ['--reuse-invariant-work'])):
        start = time.monotonic()
        with (args.output/f'{name}.log').open('w') as log:
            subprocess.run(command+['--output', str(args.output/name)]+extra, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        times[name] = time.monotonic()-start
        print('COMPLETED', name, times[name], flush=True)
    comparisons = {}
    with np.load(args.output/'baseline/chain.npz') as old, np.load(args.output/'cached/chain.npz') as new:
        for key in old.files:
            exact = np.array_equal(old[key], new[key])
            tolerance = 1e-6 if key == 'log_kernel' else 1e-10 if key == 'summaries' else 0.
            error = float(np.max(np.abs(old[key].astype(float)-new[key].astype(float))))
            comparisons[key] = dict(exact=exact, max_error=error, tolerance=tolerance,
                                    passed=exact or error <= tolerance)
    profile_error = 0.
    with np.load(args.output/'baseline/local_profiles.npz') as old, np.load(args.output/'cached/local_profiles.npz') as new:
        for key in old.files:
            profile_error = max(profile_error, float(np.max(np.abs(old[key]-new[key]))))
    audit = json.loads((args.output/'cached/invariant_replay.json').read_text())
    result = dict(device=args.device, initial_bank_seed=2026100133, total_seconds=times, arrays=comparisons,
        max_profile_error=profile_error, invariant_replay=audit,
        all_checks_pass=all(v['passed'] for v in comparisons.values()) and profile_error <= 1e-6,
        limitation='Short paired replay, not exploration or shear validation. CPU timing is not GPU throughput.')
    for name in times:
        # The driver's elapsed clock starts before model/bank initialization.
        # Differences of progress records isolate a sampling interval.
        progress = [ast.literal_eval(line.split('LOCAL_PROGRESS ', 1)[1])
                    for line in (args.output/f'{name}.log').read_text().splitlines()
                    if line.startswith('LOCAL_PROGRESS ')]
        first, last = progress[0], progress[-1]
        result.setdefault('loop_seconds_per_step', {})[name] = (
            (last['elapsed_seconds']-first['elapsed_seconds'])/(last['step']-first['step']))
    save(args.output/'result.json', result)
    print(json.dumps(result), flush=True)
    if not result['all_checks_pass']:
        raise ValueError('paired chain replay failed')


if __name__ == '__main__':
    main()
