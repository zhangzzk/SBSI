"""Compare K1024/M16384 with the owner-requested K8192/M8192 split."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_bright_extension_20260925_v1')
MAPS = ROOT.parent/'inference_atom_maps_20260925_v1/results_16696310'
OUT = Path('/home/z/Zekang.Zhang/SBSI/doc/figures/inference_8k8k_20260925')


def main():
    before = json.loads((ROOT/'comparison/summary.json').read_text())
    after = json.loads((ROOT/'comparison_8k8k/summary.json').read_text())
    assert after['n_candidates'] == after['n_draws'] == 8192
    assert before['prior'] == after['prior']
    for name in ('frozen_floor', 'frozen_fractional_flux_floor', 'frozen_metric_center', 'frozen_metric_scale'):
        assert before[name] == after[name]
    OUT.mkdir(exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    metrics, checks = [], []
    lines = ['# Inference budget: 8k exact + 8k sampled', '',
        'Same models, observations, prior weights, proposal metric, floors and prefilter131072.',
        'Seeds8701–8716; three previously inspected diagnostic observations.',
        'Old budget: K1024 + M16384 =17408; new: K8192 + M8192 =16384.', '',
        '## Results', '',
        'RMSE is relative likelihood-numerator error against each prior’s exact centre sum.',
        'It is not shear bias. Exact mass is the fraction summed deterministically.', '',
        '| Prior | Observation | Old RMSE | 8k+8k RMSE | Old / new exact mass | Dominant atom in new exact set? |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for column, (oldrow, newrow) in enumerate(zip(before['rows'], after['rows'])):
        row = newrow['row']
        assert row == oldrow['row']
        for panel, name in enumerate(('baseline', 'expanded')):
            a = np.array([x['likelihood_ratio'] for x in oldrow[f'{name}_trials']])
            b = np.array([x['likelihood_ratio'] for x in newrow[f'{name}_trials']])
            assert len(a) == len(b) == 16 and np.isfinite(a).all() and np.isfinite(b).all()
            if name == 'baseline':
                terms = np.load(MAPS/f'row_{row}_log_terms.npy', mmap_mode='r')
                total = newrow['baseline_log_numerator']
                oldexact = np.load(MAPS/f'row_{row}_plot_data.npz')['exact_atoms']
                draws = np.load(ROOT/f'comparison_8k8k/row_{row}_baseline_draws.npz')
            else:
                terms = np.load(ROOT/f'comparison_8k8k/row_{row}_log_terms.npy', mmap_mode='r')
                total = newrow['expanded_log_numerator']
                oldexact = np.load(ROOT/f'comparison/row_{row}_draws.npz')['exact']
                draws = np.load(ROOT/f'comparison_8k8k/row_{row}_draws.npz')
                np.testing.assert_allclose(total, oldrow['expanded_log_numerator'], rtol=0, atol=1e-12)
            exact, ids, q = draws['exact'], draws['draws'], draws['probability']
            assert len(exact) == len(np.unique(exact)) == len(ids) == 8192
            # Same prefilter/ranking: increasing K should retain every old exact atom.
            assert np.isin(oldexact, exact).all()
            reduced = np.logaddexp.reduce(terms)
            np.testing.assert_allclose(total, reduced, rtol=0, atol=1e-8)
            mass_old = float(np.exp(terms[oldexact]-total).sum())
            mass_new = float(np.exp(terms[exact]-total).sum())
            outside = ~np.isin(ids, exact)
            replay = mass_new + np.sum(np.exp(terms[ids[outside]]-total)/q[outside])/8192
            np.testing.assert_allclose(replay, b[0], rtol=1e-10)
            dominant = int(np.argmax(terms))
            result = dict(prior=name, row=row, old_rmse=float(np.sqrt(np.mean((a-1)**2))),
                new_rmse=float(np.sqrt(np.mean((b-1)**2))), old_mean=float(a.mean()), new_mean=float(b.mean()),
                old_exact_mass=mass_old, new_exact_mass=mass_new,
                dominant_atom=dominant, dominant_mass=float(np.exp(terms[dominant]-total)),
                dominant_in_new_exact=bool(dominant in exact),
                new_seed8701_reached=newrow[f'{name}_trials'][0]['posterior_reached'],
                new_wasted_draws=int((~outside).sum()))
            metrics.append(result)
            checks.append(dict(prior=name, row=row, nested_exact_sets=True, log_sum_passed=True,
                               seed8701_replay=float(replay), passed=True))
            lines.append(f'| {name} | {row} | {result["old_rmse"]:.5f} | {result["new_rmse"]:.5f} | {mass_old:.2%} / {mass_new:.2%} | {result["dominant_in_new_exact"]} |')
            ax = axes[panel, column]
            jitter = np.linspace(-.1, .1, 16)
            for i in range(16):
                ax.plot([jitter[i], 1+jitter[i]], [a[i], b[i]], color='.8', lw=.7, zorder=1)
            ax.scatter(jitter, a, s=20, color='#0072B2', zorder=2)
            ax.scatter(1+jitter, b, s=20, color='#D55E00', zorder=2)
            ax.axhline(1, color='.25', ls='--', lw=1)
            ax.set(xticks=[0, 1], xticklabels=['1k + 16k', '8k + 8k'], xlim=(-.3, 1.3),
                   title=f'{name.capitalize()} prior · observation {row}', ylabel='Estimated / exact numerator')
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Exact / sampled atom budget: paired comparison across 16 seeds')
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.text(.5, .012, 'V3.6 models · fixed proposal and likelihood · centre-only diagnostic, not shear bias', ha='center', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'comparison.{ext}', dpi=180)
    plt.close(fig)
    lines += ['', '![Comparison](comparison.png)', '', '## Limits', '',
        'These are three previously inspected cases, not representative population validation.',
        'Only the numerator at the initial shear centre is evaluated; there is no new shear estimate.',
        'Sixteen seeds provide a diagnostic, not a precise characterization of rare importance-weight tails.',
        'Sampled draws landing in the exact set still use the full M denominator and contribute zero to the remainder.', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    (OUT/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (OUT/'validation.json').write_text(json.dumps(checks, indent=2)+'\n')
    shutil.copy2(ROOT/'comparison/summary.json', OUT/'reference_summary.json')
    shutil.copy2(ROOT/'comparison_8k8k/summary.json', OUT/'summary.json')
    shutil.copy2(Path(__file__), OUT/Path(__file__).name)
    shutil.copy2(ROOT/'bright_prior_extension.py', OUT/'comparison_source.py')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
