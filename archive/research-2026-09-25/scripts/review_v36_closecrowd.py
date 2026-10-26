#!/usr/bin/env python
"""Independent algebra, scene influence and mixture-sensitivity review.

Consumes hashed, previously replayed row sums; imports no earlier reducer.
Mixture weights are diagnostic only: no cuts, model changes or new images.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read(path, provenance):
    raw = path.read_bytes()
    provenance[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def bias(data, model):
    # Axes: leg, [count, sum(e1), sum(e2)]. The common 2h cancels.
    def slope(s):
        if not np.isfinite(s).all() or np.any(s[:, 0] <= 0):
            raise ValueError('invalid sufficient sums')
        return s[0, 1] / s[0, 0] - s[1, 1] / s[1, 0]
    return float(100 * (slope(data) / slope(model) - 1))


def jackknife(data, model):
    # Input axes: leg, case, statistic. Pool first; never average case ratios.
    n = data.shape[1]
    ds, ms = data.sum(1), model.sum(1)
    point = bias(ds, ms)
    leave = np.array([bias(ds - data[:, i], ms - model[:, i]) for i in range(n)])
    se = np.sqrt((n - 1) / n * np.sum((leave - leave.mean())**2))
    return dict(m_percent=point, jackknife_se_pp=float(se),
                leave_one_out_range_percent=[float(leave.min()), float(leave.max())],
                maximum_single_deletion_shift_pp=float(np.max(np.abs(leave - point)))), leave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise ValueError('scheduler and new output required')
    provenance = {}
    saved = read(args.root / 'rows_closecrowd.json', provenance)
    control = read(args.root / 'rows_unbounded_rows.json', provenance)
    comparison = read(args.root / 'all_saved_models_v2.json', provenance)
    cache = args.root.parent
    classifier = read(cache / 'scene_classifier_unbounded_20260922_v1/smooth_classifier_full/protocol.json', provenance)
    domain = read(cache / 'harmonized_closecrowd_20260924_v1/domain/manifest.json', provenance)
    parent = read(cache / 'constgold_truth_parent_closecrowd_20260924_v1/parent/manifest.json', provenance)
    selection = read(cache / 'flow_joint_closecrowd_20260924_v1/joint_probability_refinement_full/completed.json', provenance)
    cases = np.array([r['case'] for r in saved['per_case']])
    np.testing.assert_array_equal(cases, np.arange(40, 120))
    np.testing.assert_array_equal(cases, [r['case'] for r in control['per_case']])
    population = dict(description=parent['population'], truth_analysis_cut=parent['truth_analysis_cut'],
        totals={k: sum(parent['case_reports'][str(c)][k] for c in cases)
                for k in ('input_parent_rows', 'truth_parent_rows', 'truth_cut_dropped',
                          'g0_keys_outside_truth_parent_dropped', 'g0_unusable_retained')})
    if population['truth_analysis_cut'] is not None or population['totals']['truth_cut_dropped'] != 0:
        raise ValueError('acceptance parent has a truth cut')
    arrays = {arm: np.array([r['sums'][arm] for r in saved['per_case']]).transpose(1, 0, 2, 3)
              for arm in ('measured', 'modeled_smooth', 'actual_usable_flags')}
    data, model = arrays['measured'].sum(2), arrays['modeled_smooth'].sum(2)
    reference = np.array([r['sums']['modeled_smooth'] for r in control['per_case']]).transpose(1, 0, 2, 3).sum(2)
    original_data = np.array([r['sums']['measured'] for r in control['per_case']]).transpose(1, 0, 2, 3).sum(2)
    np.testing.assert_allclose(data, original_data, rtol=0, atol=1e-8)
    full, leave = jackknife(data, model)
    expected = comparison['models']['closecrowd']['results']['mag25.8_radius_gt0.60/modeled_smooth']
    np.testing.assert_allclose(full['m_percent'], expected['m_percent'], rtol=0, atol=1e-9)
    full['previous_bootstrap_se_pp'] = expected['case_se_pp']
    full['influential_case'] = int(cases[np.argmax(np.abs(leave - full['m_percent']))])
    full['delete_case_estimates'] = {str(c): float(v) for c, v in zip(cases, leave)}
    full['measured_usability_m_percent'] = bias(data.sum(1), arrays['actual_usable_flags'].sum((1, 2)))
    splits = {'first40': cases < 80, 'last40': cases >= 80,
              'classifier_training_scenes': np.isin(cases, classifier['training_cases']),
              'classifier_tuning_scenes': np.isin(cases, classifier['tune_cases']),
              'flow_training_scenes': np.isin(cases, domain['split']['train_cases']),
              'flow_validation_scenes': np.isin(cases, domain['split']['validation_cases'])}
    split_results = {}
    for name, mask in splits.items():
        estimate, _ = jackknife(data[:, mask], model[:, mask])
        estimate['cases'] = cases[mask].tolist()
        estimate['n_cases'] = int(mask.sum())
        split_results[name] = estimate
    # Bootstrap seed fixed for this review, independent of the prior reducer.
    rng = np.random.default_rng(20260925)
    weights = rng.multinomial(len(cases), np.full(len(cases), 1 / len(cases)), size=20000)
    boot_data = np.einsum('bc,lcv->lbv', weights, data)
    boot_model = np.einsum('bc,lcv->lbv', weights, model)
    boot_reference = np.einsum('bc,lcv->lbv', weights, reference)
    boot = np.array([bias(d, m) for d, m in zip(boot_data.transpose(1, 0, 2), boot_model.transpose(1, 0, 2))])
    delta = boot - np.array([bias(d, m) for d, m in zip(boot_data.transpose(1, 0, 2), boot_reference.transpose(1, 0, 2))])
    full['review_bootstrap'] = dict(seed=20260925, replicates=len(boot),
        se_pp=float(boot.std(ddof=1)), ci95_percent=np.quantile(boot, [.025, .975]).tolist())
    full['paired_vs_unbounded'] = dict(delta_m_pp=full['m_percent'] - bias(data.sum(1), reference.sum(1)),
        se_pp=float(delta.std(ddof=1)), ci95_pp=np.quantile(delta, [.025, .975]).tolist())
    # Predetermined symmetric +/-10% positive weights, on the already declared
    # magnitude bins. Recompute *both* leg normalizations exactly, retaining all.
    totals = {arm: x.sum(1) for arm, x in arrays.items()}
    mixture = {}
    baseline_boot = boot.copy()
    for index, label in ((0, 'r_lt24'), (2, 'r_25p5_to27p5')):
        for factor in (.9, 1.1):
            w = np.ones(4)
            w[index] = factor
            d = np.einsum('k,lkv->lv', w, totals['measured'])
            m = np.einsum('k,lkv->lv', w, totals['modeled_smooth'])
            changed_data = np.einsum('k,lckv->lcv', w, arrays['measured'])
            changed_model = np.einsum('k,lckv->lcv', w, arrays['modeled_smooth'])
            db = np.einsum('bc,lcv->lbv', weights, changed_data)
            mb = np.einsum('bc,lcv->lbv', weights, changed_model)
            changed_boot = np.array([bias(x, y) for x, y in zip(db.transpose(1, 0, 2), mb.transpose(1, 0, 2))])
            shift = changed_boot - baseline_boot
            mixture[f'{label}_weight_{factor}'] = dict(weights=w.tolist(), m_percent=bias(d, m),
                delta_m_pp=bias(d, m) - full['m_percent'], paired_shift_se_pp=float(shift.std(ddof=1)),
                paired_shift_ci95_pp=np.quantile(shift, [.025, .975]).tolist())
    out = dict(full=full, splits=split_results, diagnostic_mixture_reweighting=mixture,
        population=population, flow_selection=dict(best_step=selection['best_step'],
            selected_sha256=selection['selected_sha256'],
            validation_nll_increase=selection['selected_full_validation_nll'] - selection['baseline_full_validation_nll'],
            production_accepted=selection['production_accepted']),
        source_hashes=provenance, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        conventions='m=100*(Rdata/Rmodel-1); each leg normalized separately; measured radius>0.6 arcsec, MAG_AUTO<25.8, h=0.02 g1; no truth cut.',
        limitations=['Shared development scenes; training/tuning labels are half-shear, not these coherent-shear observations.',
            'Splits and deletion checks diagnose sensitivity; none selects the headline or is independent acceptance.',
            'Mixture weights are diagnostic changes of target weights at fixed rendered scenes; all weights remain positive.',
            'Reweighting does not resimulate neighbours or establish calibration in a new physical domain.',
            'Case uncertainty conditions on the finite source catalogue, trained models and saved latent realization.',
            'This receipt rechecks replayed row sums, not new image measurements or forward draws.'])
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()
