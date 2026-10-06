#!/usr/bin/env python
"""Report the frozen F8 development comparison; never promote a candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json
from scripts.compare_v36_saved_models import extract, estimate
from scripts.audit_v36_saved_rows import reduce_case, response_bin_contributions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or a.output.exists():
        raise ValueError('scheduler and new output required')
    root, cache = a.root, a.root.parent
    protocol = json.loads((root / 'experiment.json').read_text())
    candidate = json.loads((root / 'trained_candidate.json').read_text())
    row_baseline = cache / 'v36_domain_recovery_20260924_v1/rows_closecrowd.json'
    baseline = json.loads(row_baseline.read_text())
    baseline_records = sorted(baseline['per_case'], key=lambda r: r['case'])
    source_hashes = {str(row_baseline): file_hash(row_baseline),
                     str(root / 'experiment.json'): file_hash(root / 'experiment.json')}
    cases, collected = [], {k: [] for k in ('measured', 'modeled_smooth', 'actual_usable_flags')}
    key = 'truth_parent/mag25.8_radius_gt0.60/'
    row_files = []
    for chunk in range(8):
        path = root / f'evaluation/chunk{chunk}/results.json'
        d = json.loads(path.read_text())
        source_hashes[str(path)] = file_hash(path)
        expected_cases = list(range(40 + 10*chunk, 50 + 10*chunk))
        if d['cases'] != expected_cases or d['h'] != .02 or d['shear_axis'] != 1:
            raise ValueError('unexpected case/shear protocol')
        if d['draws'] != 64 or d['seed'] != 7301 or d['inputs']['flow_sha256'] != candidate['flow_sha256']:
            raise ValueError('unexpected model/integration protocol')
        cls = d['inputs']['classifier_arms']['modeled_smooth']
        if [r['sha256'] for r in cls] != [protocol['fixed_components']['classifier_sha256']]:
            raise ValueError('classifier changed')
        if d['experimental_response']['metadata']['model_sha256'] != protocol['fixed_components']['disk_emulator_sha256']:
            raise ValueError('emulator changed')
        cases.extend(d['cases'])
        for arm in collected:
            src, branch = ('measured', 'modeled_smooth') if arm == 'measured' else ('model', arm)
            collected[arm].append(extract(d['per_case_sufficient'][key + branch], src, len(d['cases'])))
        paths = sorted((path.parent / 'rows').glob('case*.npz'))
        if sorted(int(p.stem[4:]) for p in paths) != expected_cases:
            raise ValueError('row-dump case coverage differs')
        row_files.extend(paths)
    if cases != protocol['development_evaluation']['cases']:
        raise ValueError('case coverage differs from pre-fit protocol')
    arrays = {k: np.concatenate(v, axis=1) for k, v in collected.items()}
    old = {arm: np.array([r['sums'][arm] for r in baseline_records]).transpose(1, 0, 2, 3)
           for arm in collected}
    np.testing.assert_allclose(arrays['measured'], old['measured'].sum(2), rtol=0, atol=1e-8)
    weights = np.random.default_rng(20260924).multinomial(80, np.full(80, 1/80), size=10000)
    results = {}
    for arm in ('modeled_smooth', 'actual_usable_flags'):
        result, boot = estimate(arrays['measured'], arrays[arm], weights)
        control, control_boot = estimate(old['measured'].sum(2), old[arm].sum(2), weights)
        delta = boot - control_boot
        interval = np.quantile(delta, [.025, .975])
        results[arm] = dict(candidate=result, control=control,
            paired_delta_m_pp=result['m_percent']-control['m_percent'],
            paired_se_pp=float(delta.std(ddof=1)), paired_ci95_pp=interval.tolist(),
            response_equivalence_screen_pass=bool(np.all(interval > -.05) and np.all(interval < .05)))
    # Row replay also provides exact global-response contributions, so a small
    # total cannot conceal changed bright/faint cancellation in this report.
    with ProcessPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(reduce_case, row_files))
    records.sort(key=lambda r: r['case'])
    bins = {arm: np.array([r['sums'][arm] for r in records]).transpose(1, 0, 2, 3)
            for arm in collected}
    for arm in collected:
        np.testing.assert_allclose(bins[arm].sum(2), arrays[arm], rtol=2e-10, atol=2e-6)
    from scripts.compare_v36_saved_models import normalized_response
    binned = {}
    for arm in ('modeled_smooth', 'actual_usable_flags'):
        dr = response_bin_contributions(bins['measured'].sum(1))
        mr = response_bin_contributions(bins[arm].sum(1))
        denom = normalized_response(bins[arm].sum((1, 2)))[0]
        contribution = 100 * (dr - mr)[:, 0] / denom
        np.testing.assert_allclose(contribution.sum(), results[arm]['candidate']['m_percent'], atol=1e-8, rtol=0)
        binned[arm] = dict(bin_contributions_pp=contribution.tolist(),
            control_bin_contributions_pp=baseline['results'][arm]['bin_contributions_pp'],
            sum_absolute_contributions_pp=float(np.abs(contribution).sum()))
    completed_path = root / 'joint/joint_probability_refinement_full/completed.json'
    control_path = cache / 'flow_joint_closecrowd_20260924_v1/joint_probability_refinement_full/completed.json'
    completed, control = (json.loads(p.read_text()) for p in (completed_path, control_path))
    density_delta = completed['selected_full_validation_nll'] - control['selected_full_validation_nll']
    source_hashes.update({str(p): file_hash(p) for p in (completed_path, control_path)})
    write_json(a.output, dict(results=results, bins=['r<24', '24<=r<25.5', '25.5<=r<27.5', 'r>=27.5'],
        bin_diagnostics=binned, cases=cases, density_delta_nats_per_row=density_delta,
        density_screen_pass=density_delta <= .001, row_replay=records,
        source_hashes=source_hashes, script_sha256=file_hash(__file__),
        production_accepted=False, independent_acceptance=False,
        limitations=['Development screen only, with reused scenes and one training/latent stream.',
            'Passing the paired equivalence budget does not establish absolute 0.3% calibration.',
            'Regional contributions are diagnostic allocations, not conditional biases or causal effects.',
            'Both axes, numerical/training repeats, full-role transfer and untouched-scene acceptance remain required.']))
    print(json.dumps(dict(results=results, bin_diagnostics=binned,
                         density_delta_nats_per_row=density_delta, production_accepted=False), indent=2))


if __name__ == '__main__':
    main()
