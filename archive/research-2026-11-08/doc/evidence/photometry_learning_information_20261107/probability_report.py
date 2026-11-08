"""Paired case errors for conditional cut scores and selected shape moments."""
from pathlib import Path
import json
import numpy as np

H = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches') / H.name


def jackknife(values):
    n = len(values) - 1
    jk = values[1:]
    return {'value': values[0].tolist(),
            'jackknife_se': np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0)).tolist()}


def main():
    report = {'definition': 'Per-row conditional probability squared error; independent 32-draw banks remove integration variance. Shape errors here are single-leg conditional usable-population means, not coherent response or m.',
              'column_order': ['count', 'observed_pass', 'predicted_pass', 'unbiased_squared_error',
                               'integration_variance', 'observed_selected_e1', 'observed_selected_e2',
                               'predicted_selected_e1', 'predicted_selected_e2', 'plugin_squared_error'],
              'splits': {}}
    lines = ['# Conditional probability and shape diagnostics', '', report['definition'], '',
             '| Split | Hard-cut score: control | Hard-cut score: weighted | Paired change |',
             '|---|---:|---:|---:|']
    for split in ['train', 'validation']:
        with np.load(ROOT / f'own_{split}.npz') as z:
            raw, kinds = z['probability_sums'], list(z['probability_kinds'])
            cuts, cases, arms = list(z['cuts']), z['cases'], list(z['arms'])
            factor = z['factor_density']
        n = len(cases)
        weights = np.vstack([np.ones(n), np.ones((n, n)) - np.eye(n)])
        result = {'cases': cases.tolist(), 'scores': {}, 'factor_nll': {}}
        for kind, name in enumerate(kinds):
            result['scores'][name] = {}
            for leg, population in [(None, 'pooled'), (0, 'g0'), (1, 'g05')]:
                data = raw[:, kind].sum(2) if leg is None else raw[:, kind, :, leg]
                all_values = []
                for arm_index, arm in enumerate(arms):
                    total = np.einsum('wc,ckv->wkv', weights, data[arm_index])
                    assert (total[..., 0] > 0).all() and (total[..., 1:3] > 0).all()
                    values = {'probability_score': total[..., 3] / total[..., 0],
                              'integration_variance': total[..., 4] / total[..., 0],
                              'pass_error': (total[..., 1] - total[..., 2]) / total[..., 0],
                              'selected_shape_error': total[..., 5:7] / total[..., 1, None]
                                                      - total[..., 7:9] / total[..., 2, None],
                              'shape_numerator_error': (total[..., 5:7] - total[..., 7:9]) / total[..., 0, None]}
                    all_values.append(values)
                entries = {arm: {cut: {key: jackknife(value[:, ci]) for key, value in values.items()}
                                 for ci, cut in enumerate(cuts)} for arm, values in zip(arms, all_values)}
                entries['weighted_minus_control'] = {cut: {key: jackknife(all_values[1][key][:, ci] - all_values[0][key][:, ci])
                                                          for key in all_values[0]} for ci, cut in enumerate(cuts)}
                if len(arms)==4:
                    for label,coefs in [('input_only',[-1,0,1,0]),('loss_with_input',[0,0,-1,1]),('input_with_loss',[0,-1,0,1]),('both_vs_control',[-1,0,0,1]),('interaction',[1,-1,-1,1])]:
                        entries[label]={cut:{key:jackknife(sum(coefs[a]*all_values[a][key][:,ci] for a in range(4))) for key in all_values[0]} for ci,cut in enumerate(cuts)}
                rms = [value['probability_score'].mean(1) for value in all_values]
                entries['nine_cut_mean_score'] = {arm: jackknife(value) for arm, value in zip(arms, rms)}
                entries['nine_cut_mean_score']['weighted_minus_control'] = jackknife(rms[1] - rms[0])
                if len(arms)==4:
                    for label,coefs in [('input_only',[-1,0,1,0]),('loss_with_input',[0,0,-1,1]),('input_with_loss',[0,-1,0,1]),('both_vs_control',[-1,0,0,1]),('interaction',[1,-1,-1,1])]:
                        entries['nine_cut_mean_score'][label]=jackknife(sum(coefs[a]*rms[a] for a in range(4)))
                result['scores'][name][population] = entries
                if name == 'hard' and population == 'pooled':
                    stats = [jackknife(v) for v in [rms[0], rms[1], rms[1] - rms[0]]]
                    lines.append('| ' + split + ' | ' + ' | '.join(
                        f"{v['value']:+.7f} ± {v['jackknife_se']:.7f}" for v in stats) + ' |')
        densities = []
        for ai, arm in enumerate(arms):
            v = weights @ factor[ai].sum(1)
            densities.append(v[:, 1:] / v[:, :1])
            result['factor_nll'][arm] = {'photometry': jackknife(densities[-1][:, 0]),
                                        'conditional_shape': jackknife(densities[-1][:, 1])}
        result['factor_nll']['weighted_minus_control'] = {
            key: jackknife((densities[1] - densities[0])[:, i])
            for i, key in enumerate(['photometry', 'conditional_shape'])}
        if len(arms)==4:
            for label,coefs in [('input_only',[-1,0,1,0]),('loss_with_input',[0,0,-1,1]),('input_with_loss',[0,-1,0,1]),('both_vs_control',[-1,0,0,1]),('interaction',[1,-1,-1,1])]:
                result['factor_nll'][label]={key:jackknife(sum(coefs[a]*densities[a][:,i] for a in range(4))) for i,key in enumerate(['photometry','conditional_shape'])}
        report['splits'][split] = result
    lines += ['', 'Errors are paired delete-one-case standard errors. All nine cuts, both original legs, hard/soft targets, smoothing sensitivities and selected shape moments are retained in probability_report.json. Shape weights are fixed; selected-shape behavior need not be fixed.']
    for parent in [H, ROOT]:
        (parent / 'probability_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (H / 'probability_results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
