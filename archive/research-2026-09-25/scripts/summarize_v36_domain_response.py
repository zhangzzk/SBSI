#!/usr/bin/env python
"""Pool diagnostic forward sums without treating a small pilot as acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json
from scripts.audit_v36_cut_transfer import pooled_response
from scripts.audit_v36_domain_response import GROUPS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    records, provenance, protocols = [], {}, []
    for root in args.roots:
        completed = json.loads((root/'completed.json').read_text())
        protocol = json.loads((root/'protocol.json').read_text())
        protocols.append(protocol)
        for case in completed['cases']:
            path = root/f'case{case}.json'
            records.append(json.loads(path.read_text()))
            provenance[str(path)] = file_hash(path)
    cases = [r['case'] for r in records]
    if len(set(cases)) != len(cases) or len(cases) < 2:
        raise ValueError('at least two distinct complete cases required')
    arms = tuple(records[0]['model'])
    for r in records:
        if tuple(r['model']) != arms:
            raise ValueError('inconsistent model arms')
    # Protocol identity may differ only by case list and output directory.
    comparable = []
    for item in protocols:
        item = json.loads(json.dumps(item))
        for name in ('cases', 'output'):
            item['arguments'].pop(name)
        comparable.append(item)
    if any(item != comparable[0] for item in comparable[1:]):
        raise ValueError('worker protocols differ')
    n = len(records)
    weights = np.random.default_rng(20260924).multinomial(n, np.full(n, 1/n), size=10000)
    all_results = {}
    boot_bias = {a: {} for a in arms}
    for group in GROUPS:
        measured = np.array([[r['measured'][leg][group] for r in records] for leg in ('plus', 'minus')])
        total = measured.sum(1)
        sampled = np.einsum('bc,lck->lbk', weights, measured)
        rb = pooled_response(sampled, .02)
        out = dict(parent_rows=sum(r['population_counts'][group] for r in records),
            selected_per_leg=total[:, 0].tolist(), measured_response=pooled_response(total, .02).tolist(),
            measured_response_case_se=np.std(rb, axis=0, ddof=1).tolist(), models={})
        for arm in arms:
            values = np.array([[r['model'][arm][leg][group] for r in records] for leg in ('plus', 'minus')])
            model_r = pooled_response(values.sum(1), .02)
            model_boot = pooled_response(np.einsum('bc,lck->lbk', weights, values), .02)
            mb = 100*(rb[:, 0]/model_boot[:, 0]-1)
            boot_bias[arm][group] = mb
            out['models'][arm] = dict(response=model_r.tolist(), selected_per_leg=values.sum(1)[:, 0].tolist(),
                m_percent=float(100*(out['measured_response'][0]/model_r[0]-1)),
                case_se_pp=float(np.std(mb, ddof=1)), ci95_pp=np.quantile(mb, [.025, .975]).tolist())
        all_results[group] = out
    shifts = {}
    for arm in arms:
        shifts[arm] = {}
        for group in ('all_bright', 'all'):
            delta = boot_bias[arm][group]-boot_bias[arm]['retained']
            shifts[arm][group] = dict(delta_m_pp=all_results[group]['models'][arm]['m_percent']-
                all_results['retained']['models'][arm]['m_percent'], case_se_pp=float(np.std(delta, ddof=1)),
                ci95_pp=np.quantile(delta, [.025, .975]).tolist())
    write_json(args.output, dict(cases=cases, h=.02, axis=1, results=all_results, paired_domain_shifts=shifts,
        source_sha256=provenance, protocols=protocols, bootstrap_seed=20260924, bootstrap_replicates=10000,
        invalid_by_case={str(r['case']): r['invalid'] for r in records},
        limitations=['Development diagnostic, not independent validation.',
            'A two-case bootstrap cannot assess 0.3% population calibration.',
            'Observed-only arms measure response change, not model bias.',
            'Model numerical underflow outside selected support is recorded; it is not density validation.',
            'Case errors exclude training and latent integration uncertainty.']))
    print(json.dumps(all_results, indent=2), flush=True)


if __name__ == '__main__':
    main()
