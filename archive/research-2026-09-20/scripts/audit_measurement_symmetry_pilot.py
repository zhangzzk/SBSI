"""Summarize the native stamp-symmetry pilot without common-fit conditioning."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.audit_common_optimizer_pilot import paired_marginal_summary
from scripts.train_fixed_g0_flow import sha256, write_json


ACTIONS = ('identity', 'flip_x', 'transpose', 'rot90')


def summarize_case(shapes, photometry):
    shapes, photometry = np.asarray(shapes), np.asarray(photometry)
    if shapes.ndim != 4 or shapes.shape[1:] != (4, 2, 2) or photometry.shape != (len(shapes), 2):
        raise ValueError('expected parent/action/stream/shape axes and aligned photometry')
    radius, magnitude = photometry.T
    valid = np.isfinite(photometry).all(1) & (radius > 0)
    cuts = {'uncut': valid, 'mag25.8_radius_gt0.60': valid & (radius > 3) & (magnitude < 25.8),
            'mag25.8_radius_ge0.75': valid & (radius >= 3.75) & (magnitude < 25.8)}
    result = {}
    for cut, selected in cuts.items():
        rows = {}
        for stream in range(2):
            for ai, action in enumerate(ACTIONS[1:], start=1):
                rows[f'{action}/stream{stream}'] = paired_marginal_summary(
                    shapes[:, 0, stream], shapes[:, ai, stream], selected)
        rows['identity/stream1_minus_stream0'] = paired_marginal_summary(
            shapes[:, 0, 0], shapes[:, 0, 1], selected)
        result[cut] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('measurement pilot summary requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.cases)) != len(args.cases):
        raise ValueError('distinct pilot scenes required')
    reports, sources = {}, {}
    for case in sorted(args.cases):
        path = args.pilot_root/f'case{case:03d}.npz'
        receipt = json.loads(path.with_suffix('.json').read_text())
        if (receipt['case'] != case or receipt['actions'] != list(ACTIONS)
                or receipt['output_sha256'] != sha256(path)
                or receipt['identity_reference_replay_passed'] is not True):
            raise ValueError('pilot action, data or identity replay mismatch')
        with np.load(path, allow_pickle=False) as stored:
            if len(np.unique(stored['ids'])) != len(stored['ids']):
                raise ValueError('nonunique sampled identities')
            reports[str(case)] = {'cuts': summarize_case(stored['shapes'], stored['photometry']),
                                  'flag_counts': {str(value): int((stored['flags'] == value).sum())
                                                  for value in np.unique(stored['flags'])}}
        sources[str(case)] = {'receipt_sha256': sha256(path.with_suffix('.json')), 'receipt': receipt}
    write_json(args.output, {'reports': reports, 'sources': sources, 'script_sha256': sha256(__file__),
                            'limitations': ['paired-object errors condition on two fixed scenes; not scene-generalization errors',
                                            'ngmix stamp symmetries only; detection, photometry and scene-prior symmetry untested',
                                            'same optimizer seeds are not exact rotated initializations; both streams retained',
                                            'no empirical correction, model fit or calibration claim']})
    print(f'MEASUREMENT_SYMMETRY_AUDIT_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
