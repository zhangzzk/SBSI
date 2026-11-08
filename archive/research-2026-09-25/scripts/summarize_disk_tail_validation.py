#!/usr/bin/env python
"""Summarize exact tail benchmarks; never promote a partial repair to inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    args = parser.parse_args()
    output = args.panel / 'summary.json'
    if output.exists():
        raise ValueError('refusing to overwrite tail summary')
    audit_path = args.run / 'curvature_audit_v1.json'
    audit = json.loads(audit_path.read_text())
    expected_rows = [entry['row'] for entry in audit['worst'][:32]]
    reports, provenance = {}, []
    reference = None
    for worker in (0, 1):
        root = args.panel / f'worker_{worker}'
        receipt = json.loads((root / 'complete.json').read_text())
        if (receipt['worker'] != worker or receipt['object_rows'] != expected_rows[worker::2]
                or receipt['audit_sha256'] != file_hash(audit_path)):
            raise ValueError('panel coverage or audit identity differs')
        if reference is None:
            reference = receipt
        for key in ('audit_sha256', 'preparation_sha256', 'source_sha256', 'protocol'):
            if receipt[key] != reference[key]:
                raise ValueError(f'worker identity differs: {key}')
        for row in receipt['object_rows']:
            path = root / f'row_{row}_comparison.json'
            report = json.loads(path.read_text())
            if row in reports or report['row'] != row or file_hash(root / f'row_{row}_exact.json') != report['exact_sha256']:
                raise ValueError('row identity or exact checksum mismatch')
            reports[row] = report
            provenance.append(dict(path=str(path), sha256=file_hash(path)))
    if set(reports) != set(expected_rows):
        raise ValueError('incomplete tail panel')
    ordered = [reports[row] for row in expected_rows]
    exact_score = np.array([r['exact']['score'] for r in ordered])
    exact_info = np.array([r['exact']['information'] for r in ordered])
    old_score = np.array([r['production']['score'] for r in ordered])
    old_info = np.array([r['production']['information'] for r in ordered])
    summaries = {}
    for k in (65536, 262144):
        arms = [[a for a in r['arms'] if a['k'] == k] for r in ordered]
        if any(len(a) != 1 for a in arms):
            raise ValueError('missing/duplicate candidate budget')
        score = np.array([a[0]['score'] for a in arms])
        info = np.array([a[0]['information'] for a in arms])
        ds, di = score-exact_score, info-exact_info
        summaries[str(k)] = dict(score_error_sum=ds.sum(0).tolist(), information_error_sum=di.sum(0).tolist(),
            max_absolute_score_error=float(np.abs(ds).max()), max_absolute_information_error=float(np.abs(di).max()),
            per_row_score_error_norm=np.linalg.norm(ds, axis=1).tolist(),
            per_row_information_error_norm=np.linalg.norm(di, axis=(1, 2)).tolist())
    hybrid_info = np.array(audit['information_sum']) + (exact_info-old_info).sum(0)
    report = dict(status='diagnostic_only', object_rows=expected_rows, provenance=provenance,
        exact_score_sum=exact_score.sum(0).tolist(), exact_information_sum=exact_info.sum(0).tolist(),
        production_score_error_sum=(old_score-exact_score).sum(0).tolist(),
        production_information_error_sum=(old_info-exact_info).sum(0).tolist(),
        candidate_budgets=summaries,
        production_with_only_worst32_replaced=dict(information_sum=hybrid_info.tolist(),
            information_eigenvalues=np.linalg.eigvalsh(hybrid_info).tolist()),
        limitations='Selected failure-tail diagnostic only. Remaining 499968 observations retain approximate production moments. No final shear estimate, covariance, or calibration claim.')
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
