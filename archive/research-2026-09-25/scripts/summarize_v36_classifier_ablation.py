"""Apply pre-fit non-degradation gates; never auto-promote classifier artifacts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.compare_v36_saved_models import normalized_response, estimate
from scripts.audit_v36_saved_rows import response_bin_contributions
from scripts.train_full_parent_classifier import sha256, write_json


def gate(point, draws, margin):
    point, draws = np.asarray(point), np.asarray(draws)
    lo, hi = np.quantile(draws, [.025, .975], axis=0)
    return dict(point_degradation=point.tolist(), ci95_lower=lo.tolist(), ci95_upper=hi.tolist(),
                margin=margin, passed=bool(np.all(point <= 1e-12) and np.all(hi <= margin)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or (a.root/'comparison.json').exists():
        raise ValueError('Slurm and new output required')
    p = json.loads((a.root/'protocol.json').read_text())
    records, hashes = [], None
    for i in range(8):
        d = json.loads((a.root/f'evaluation_{i}.json').read_text())
        if d['protocol_sha256'] != sha256(a.root/'protocol.json'):
            raise ValueError('protocol changed')
        if hashes is not None and hashes != d['classifier_hashes']:
            raise ValueError('models differ between chunks')
        hashes = d['classifier_hashes']
        records.extend(d['records'])
    records.sort(key=lambda r: r['case'])
    if [r['case'] for r in records] != p['evaluation_cases']:
        raise ValueError('exactly all declared cases required')
    sums = {name: np.array([r['sums'][name] for r in records]).transpose(1, 0, 2, 3, 4)
            for name in ['measured', *hashes]}
    rng = np.random.default_rng(p['bootstrap']['seed'])
    w = rng.multinomial(80, np.full(80, 1/80), size=p['bootstrap']['replicates'])
    boot = {k: np.einsum('bc,lctkv->lbtkv', w, v) for k, v in sums.items()}
    total = {k: v.sum(1) for k, v in sums.items()}
    responses = {k: normalized_response(v.sum(2)) for k, v in total.items()}
    rb = {k: normalized_response(v.sum(3)) for k, v in boot.items()}
    summaries, biasboot, cancellation, cb = {}, {}, {}, {}
    for name in hashes:
        summaries[name], biasboot[name] = estimate(sums['measured'][:, :, 1].sum(2), sums[name][:, :, 1].sum(2), w)
        parts = response_bin_contributions(total['measured'][:, 1]) - response_bin_contributions(total[name][:, 1])
        cancellation[name] = float(np.abs(100*parts[:, 0]/responses[name][1, 0]).sum())
        partsb = response_bin_contributions(boot['measured'][:, :, 1]) - response_bin_contributions(boot[name][:, :, 1])
        cb[name] = np.abs(100*partsb[..., 0]/rb[name][:, 1, 0, None]).sum(1)
    comparisons = {}
    wt = rng.multinomial(20, np.full(20, 1/20), size=p['bootstrap']['replicates']) / 20
    completed = {'retained': json.loads((Path(p['old_classifier']).parent/'completed.json').read_text())}
    completed.update({k: json.loads((a.root/k/'completed.json').read_text()) for k in hashes if k != 'retained'})
    limits = p['criteria']
    for seed in p['seeds']:
        name = f'C12_{seed}'
        for ref in (f'C17_{seed}', 'retained'):
            checks = {}
            checks['absolute_bias'] = gate(abs(summaries[name]['m_percent'])-abs(summaries[ref]['m_percent']),
                np.abs(biasboot[name])-np.abs(biasboot[ref]), limits['bias_absolute_degradation_margin_pp'])
            checks['cancellation'] = gate(cancellation[name]-cancellation[ref], cb[name]-cb[ref],
                limits['cancellation_degradation_margin_pp'])
            count_errors = {k: np.abs(total[k].sum(2)[..., 0]/total['measured'].sum(2)[..., 0]-1) for k in (name, ref)}
            count_boot = {k: np.abs(boot[k].sum(3)[..., 0]/boot['measured'].sum(3)[..., 0]-1) for k in (name, ref)}
            checks['selected_counts_all_three_cuts'] = gate(count_errors[name]-count_errors[ref],
                (count_boot[name]-count_boot[ref]).transpose(1, 0, 2), limits['selected_count_fraction_degradation_margin'])
            cross = {k: abs(responses['measured'][1, 1]-responses[k][1, 1]) for k in (name, ref)}
            crossboot = {k: np.abs(rb['measured'][:, 1, 1]-rb[k][:, 1, 1]) for k in (name, ref)}
            checks['cross_response'] = gate(cross[name]-cross[ref], crossboot[name]-crossboot[ref],
                limits['cross_response_degradation_margin'])
            for field in ('bce', 'brier', 'calibration'):
                scores = {k: np.array([[s[field] for s in r['scores'][k]] for r in records]) for k in (name, ref)}
                pts = {k: v.mean(0) for k, v in scores.items()}
                bs = {k: w@v/80 for k, v in scores.items()}
                if field == 'calibration':
                    pts, bs = ({k: np.abs(v) for k, v in t.items()} for t in (pts, bs))
                checks['constant_'+field] = gate(pts[name]-pts[ref], bs[name]-bs[ref], limits['bce_degradation_margin'])
            tune = {k: np.array([r['bce'] for r in completed[k]['tune_metrics']]) for k in (name, ref)}
            for k in (name, ref):
                if [r['case'] for r in completed[k]['tune_metrics']] != p['tune_cases']:
                    raise ValueError('tuning case order differs')
            delta = tune[name]-tune[ref]
            checks['half_shear_tune_bce'] = gate(delta.mean(), wt@delta, limits['bce_degradation_margin'])
            comparisons[f'{name}_versus_{ref}'] = dict(checks=checks, passed=all(v['passed'] for v in checks.values()))
    passed = all(v['passed'] for v in comparisons.values())
    result = dict(development_non_degradation_pass=passed, comparisons=comparisons,
        summaries=summaries, cancellation_pp=cancellation, classifier_hashes=hashes,
        protocol_sha256=sha256(a.root/'protocol.json'), production_accepted=False,
        independent_acceptance=False, selected_candidate=None,
        next_step='Review before C9; retain current classifier unless every gate passes.',
        limitations=['Reused development scenes and two training seeds; no independent acceptance.',
                     'Fixed64 F8 draws reused exactly; only axis1 measured.',
                     'All gates required; no favorable seed/model selection.'])
    write_json(a.root/'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
