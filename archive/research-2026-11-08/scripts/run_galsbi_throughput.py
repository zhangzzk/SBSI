"""Bounded full-process throughput and hybrid-ensemble experiments under Slurm."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from check_galsbi_continuous_prior import save
from report_galsbi_population import checked_run


def run(root, label, reference, indices, seed, bank_seed, *, steps=16384, warmup=4096, cache=True, ensemble=False):
    output = root/label
    command = [sys.executable, '-u', '-c',
        'import runpy,sys;sys.path[:0]=[".","scripts"];runpy.run_path("scripts/check_galsbi_local_sampler.py",run_name="__main__")',
        '--reference', str(reference), '--output', str(output), '--device', 'cuda',
        '--seed', str(seed), '--initial-bank-seed', str(bank_seed), '--initialization', 'measurement',
        '--steps', str(steps), '--warmup', str(warmup), '--observation-indices', *map(str, indices),
        '--environment-update', 'persistent', '--environments', '1', '--marginalize-type',
        '--scale-adaptation', 'per-chain', '--sed-jump', '--joint-morphology']
    if cache:
        command.append('--reuse-invariant-work')
    if ensemble:
        command.append('--ensemble-morphology')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic()-started
    _, result = checked_run(output)
    if cache:
        audit = json.loads((output/'invariant_replay.json').read_text())
        if not (audit['checked_proposals'] > 0 and audit['maximum_summary_change'] <= 1e-10
                and audit['maximum_log_kernel_change'] <= 1e-6):
            raise ValueError('invariant replay failed')
    receipt = dict(label=label, observations=len(indices), steps=steps, warmup=warmup,
        cache=cache, ensemble=ensemble, process_seconds=elapsed,
        gpu_seconds_per_galaxy_per_replica=elapsed/len(indices),
        timing_seconds=result['timing_seconds'], numerical_checks_pass=True,
        throughput_target_met=steps == 16384 and elapsed/len(indices) <= 10,
        accuracy_accepted=False)
    save(output/'timing.json', receipt)
    print('RUN_TIMING', json.dumps(receipt), flush=True)


def paired_replay(root):
    baseline, cached = root/'cache_validation/baseline', root/'cache_validation/cached'
    checks = {}
    for filename in ('chain.npz', 'local_profiles.npz'):
        with np.load(baseline/filename) as a, np.load(cached/filename) as b:
            if set(a.files) != set(b.files):
                raise ValueError('paired replay keys differ')
            for key in a.files:
                x, y = a[key], b[key]
                if x.shape != y.shape:
                    raise ValueError('paired replay shapes differ')
                error = float(np.max(np.abs(x.astype(float)-y.astype(float))))
                tolerance = 1e-10 if key == 'summaries' else 1e-6 if key == 'log_kernel' or filename == 'local_profiles.npz' else 0.
                checks[filename+'/'+key] = dict(error=error, tolerance=tolerance, passed=error <= tolerance)
    save(root/'cache_validation/replay.json', dict(checks=checks, all_checks_pass=all(v['passed'] for v in checks.values())))
    if not all(v['passed'] for v in checks.values()):
        raise ValueError('paired trajectories or local profiles changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['cache', 'throughput', 'smoke', 'ensemble'], required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    root = args.root
    reference = root/'reference'
    old = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_summary_mh_20260930_v2')
    if args.stage == 'cache':
        for label, cache in [('baseline', False), ('cached', True)]:
            run(root, 'cache_validation/'+label, reference, list(range(128)), 2026102501, 2026102601,
                steps=256, warmup=128, cache=cache)
        paired_replay(root)
    elif args.stage == 'throughput':
        replay = json.loads((root/'cache_validation/replay.json').read_text())
        if not replay['all_checks_pass']:
            raise ValueError('distinct-observation cache replay prerequisite failed')
        run(root, 'throughput1024', reference, list(range(1024)), 2026102301, 2026102401)
    else:
        indices = range(4) if args.stage == 'smoke' else [args.index]
        for i in indices:
            if i not in range(4):
                raise ValueError('ensemble task index must be0..3')
            # Counterbalance the two arms across the two seed pairs.
            ensemble = i in (1, 2)
            replica = i//2
            arm = 'stretch' if ensemble else 'control'
            smoke = args.stage == 'smoke'
            run(root, f'{args.stage}/{arm}/replica{replica}', old, [0, 1, 2, 3],
                2026102101+replica+(100 if smoke else 0), 2026102201+replica+(100 if smoke else 0),
                steps=256 if smoke else 16384, warmup=128 if smoke else 4096, ensemble=ensemble)


if __name__ == '__main__':
    main()
