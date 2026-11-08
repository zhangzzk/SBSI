#!/usr/bin/env python
"""Compare exact and paired-draw curvature probes with retained production."""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--probe-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite summary')
    root = args.run
    exact_root = root / f'curvature_probe_{args.probe_id}_exact'
    sample_root = root / f'curvature_probe_{args.probe_id}_sample'
    exact = json.loads((exact_root / 'report.json').read_text())
    sample = json.loads((sample_root / 'report.json').read_text())
    for key in ('preparation_sha256', 'audit_sha256', 'center', 'h', 'implementation'):
        if exact[key] != sample[key]:
            raise ValueError(f'probe identity differs: {key}')
    audit_path = root / 'curvature_audit_v1.json'
    if file_hash(audit_path) != exact['audit_sha256']:
        raise ValueError('audit changed')
    audit = json.loads(audit_path.read_text())
    direction = np.array(audit['negative_direction'])
    rows = np.array(sample['object_rows'])
    worst = np.isin(rows, [w['row'] for w in audit['worst'][:32]])
    baseline_score, baseline_info = np.empty((len(rows), 2)), np.empty((len(rows), 2, 2))
    for part in sorted((root / 'production_lru_v1').glob('part_*')):
        receipt = json.loads((part / 'result.json').read_text())
        start, stop = [receipt['observation_partition'][k] for k in ('start', 'stop')]
        mask = (rows >= start) & (rows < stop)
        if not mask.any():
            continue
        path = part / receipt['one_step_moments']['path']
        if file_hash(path) != receipt['one_step_moments']['sha256']:
            raise ValueError('production moments changed')
        with np.load(path) as data:
            baseline_score[mask] = data['score'][rows[mask]-start]
            baseline_info[mask] = data['information'][rows[mask]-start]
    arms = []
    for arm in sample['arms']:
        path = sample_root / arm['path']
        if file_hash(path) != arm['sha256']:
            raise ValueError('sample probe moments changed')
        with np.load(path) as data:
            np.testing.assert_array_equal(data['object_rows'], rows)
            for i, draws in enumerate((16384, 32768, 65536)):
                s, info = data['ladder_score'][i], data['ladder_information'][i]
                projection = np.einsum('i,nij,j->n', direction, info, direction)
                arms.append(dict(seed=arm['seed'], draws=draws,
                    worst32_projected_information=float(projection[worst].sum()),
                    controls_projected_information=float(projection[~worst].sum()),
                    baseline_max_score_change=float(np.abs(s-baseline_score).max()),
                    baseline_max_information_change=float(np.abs(info-baseline_info).max()),
                    worst_row_score=s[rows == exact['row']][0].tolist(),
                    worst_row_information=info[rows == exact['row']][0].tolist(),
                    worst_row_ess=float(data['weight_ess'][i][rows == exact['row']][0])))
    exact_score, exact_info = np.array(exact['score']), np.array(exact['information'])
    old = exact['production']
    report = dict(status='diagnostic_only', n_probed=len(rows),
        exact_probe_sha256=file_hash(exact_root / 'report.json'),
        sample_probe_sha256=file_hash(sample_root / 'report.json'),
        exact=dict(row=exact['row'], n_atoms=exact['n_atoms'], score=exact_score.tolist(),
                   information=exact_info.tolist(), projected_information=float(direction @ exact_info @ direction),
                   production_score_difference=(exact_score-np.array(old['score'])).tolist(),
                   production_information_difference=(exact_info-np.array(old['information'])).tolist(),
                   top_atom_ids=[v[:3] for v in exact['top_atoms']],
                   top_atom_weights=[v[:3] for v in exact['top_posterior_weights']]),
        arms=arms)
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
