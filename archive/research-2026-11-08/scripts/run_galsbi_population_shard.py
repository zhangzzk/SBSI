"""Run one bounded observation shard with two independently seeded replicas."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from check_galsbi_continuous_prior import save
from report_galsbi_population import checked_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = json.loads((args.root/'plan.json').read_text())
    size = plan['batch_observations']
    start, end = args.shard*size, min((args.shard+1)*size, plan['observations'])
    if not 0 <= start < end <= plan['observations']:
        raise ValueError('invalid shard')
    for replica in range(2):
        indices = list(range(start, end))
        if args.smoke and replica == 1:
            indices = list(range(plan['observations']-32, plan['observations']))
        offset = 10000 if args.smoke else 0
        output = args.root/('smoke' if args.smoke else 'runs')/f'shard{args.shard:04d}'/f'replica{replica}'
        argv = [sys.executable, '-u', '-c',
            'import runpy,sys;sys.path[:0]=[".","scripts"];runpy.run_path("scripts/check_galsbi_local_sampler.py",run_name="__main__")',
            '--reference', str(args.root/'reference'), '--output', str(output), '--device', 'cuda',
            '--seed', str(plan['mh_seed_base']+2*args.shard+replica+offset),
            '--initial-bank-seed', str(plan['bank_seed_base']+2*args.shard+replica+offset),
            '--initialization', 'measurement', '--steps', str(256 if args.smoke else plan['steps']),
            '--warmup', str(128 if args.smoke else plan['warmup']),
            '--observation-indices', *map(str, indices), '--environment-update', 'persistent',
            '--environments', '1', '--marginalize-type', '--scale-adaptation', 'per-chain',
            '--sed-jump', '--joint-morphology']
        subprocess.run(argv, check=True)
        checked_run(output)
    save(args.root/('smoke' if args.smoke else 'runs')/f'shard{args.shard:04d}'/'complete.json',
        dict(numerical_checks_pass=True, replicas=2, shard=args.shard))


if __name__ == '__main__':
    main()
