"""Publish the fixed-row numerical comparison after all evaluations finish."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_bright_extension_20260925_v1')
OUT = Path('/home/z/Zekang.Zhang/SBSI/doc/figures/bright_prior_20260925')


def main():
    summary = json.loads((ROOT/'comparison/summary.json').read_text())
    assert summary['status'] == 'complete'
    OUT.mkdir(exist_ok=False)
    shutil.copy2(ROOT/'comparison/summary.json', OUT/'summary.json')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    lines = ['# Bright-prior numerical comparison', '',
             'All 7,780,811 new catalogue rows are included in the experimental prior.',
             'Bright population mass is fixed; model artifacts and proposal settings are unchanged.', '',
             '## Fixed observations and 16 sampling seeds', '',
             'Each ratio compares the sampled likelihood numerator with its own prior’s exact sum.',
             'RMSE below is over seeds8701–8716, not uncertainty in shear bias.', '',
             '| Observation | Old RMSE | Expanded RMSE | Old mean ratio | Expanded mean ratio | Old / new posterior atom ESS |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    numeric = []
    validation = []
    new_domain = np.load(ROOT/'catalogue_audit_masks.npz')['within_generator_domain']
    for ax, row in zip(axes, summary['rows']):
        old = np.array([v['likelihood_ratio'] for v in row['baseline_trials']])
        new = np.array([v['likelihood_ratio'] for v in row['expanded_trials']])
        assert np.isfinite(old).all() and np.isfinite(new).all()
        oldterms = np.load(ROOT.parent/f'inference_atom_maps_20260925_v1/results_16696310/row_{row["row"]}_log_terms.npy', mmap_mode='r')
        oldmass = np.exp(oldterms-row['baseline_log_numerator'])
        np.testing.assert_allclose(oldmass.sum(), 1., rtol=1e-10)
        oldess = 1/np.square(oldmass).sum()
        olderr, newerr = np.sqrt(np.mean((old-1)**2)), np.sqrt(np.mean((new-1)**2))
        expanded_terms = np.load(ROOT/f'comparison/row_{row["row"]}_log_terms.npy', mmap_mode='r')
        # Independent reduction and saved-draw replay, rather than trusting
        # the comparison summary's normalized masses and ratio.
        independent_total = np.logaddexp.reduce(expanded_terms)
        np.testing.assert_allclose(independent_total, row['expanded_log_numerator'], rtol=0, atol=1e-8)
        draws = np.load(ROOT/f'comparison/row_{row["row"]}_draws.npz')
        exact, ids, probability = draws['exact'], draws['draws'], draws['probability']
        assert len(exact) == 1024 and len(np.unique(exact)) == 1024 and len(ids) == 16384
        outside = ~np.isin(ids, exact)
        exact_mass = np.exp(expanded_terms[exact]-independent_total).sum()
        ratio = exact_mass + np.sum(np.exp(expanded_terms[ids[outside]]-independent_total)/probability[outside])/len(ids)
        reached = np.exp(expanded_terms[np.union1d(exact, ids[outside])]-independent_total).sum()
        np.testing.assert_allclose(ratio, row['expanded_trials'][0]['likelihood_ratio'], rtol=1e-8)
        np.testing.assert_allclose(reached, row['expanded_trials'][0]['posterior_reached'], rtol=1e-8)
        validation.append(dict(row=row['row'], independent_total=float(independent_total),
                               replay_ratio=float(ratio), replay_reached=float(reached), passed=True))
        new_mass = np.exp(expanded_terms[summary['prior']['n_old']:]-row['expanded_log_numerator'])
        outside_mass = float(new_mass[~new_domain].sum())
        old_draws = np.load(ROOT.parent/f'inference_atom_maps_20260925_v1/results_16696310/row_{row["row"]}_plot_data.npz')
        old_exact = old_draws['exact_atoms']
        dominant = int(np.argmax(expanded_terms))
        lines.append(f'| {row["row"]} | {olderr:.4f} | {newerr:.4f} | {old.mean():.4f} | {new.mean():.4f} | {oldess:.1f} / {row["expanded_posterior_ess"]:.1f} |')
        numeric.append(dict(row=row['row'], old_rmse=float(olderr), expanded_rmse=float(newerr),
                            old_mean=float(old.mean()), expanded_mean=float(new.mean()),
                            old_posterior_ess=float(oldess), expanded_posterior_ess=row['expanded_posterior_ess'],
                            posterior_mass_outside_old_generator_domain=outside_mass,
                            dominant_atom=dominant, dominant_mass=row['expanded_max_atom_mass'],
                            dominant_in_old_exact=bool(dominant in old_exact),
                            dominant_in_new_exact=bool(dominant in exact),
                            dominant_sampled=bool(dominant in ids),
                            old_exact_mass_under_expanded_prior=float(np.exp(expanded_terms[old_exact]-independent_total).sum()),
                            new_exact_mass_under_expanded_prior=float(exact_mass),
                            new_atoms_in_exact=int(np.sum(exact >= summary['prior']['n_old'])),
                            old_new_exact_intersection=int(len(np.intersect1d(old_exact, exact)))))
        jitter = np.linspace(-.11, .11, len(old))
        for i in range(len(old)):
            ax.plot([jitter[i], 1+jitter[i]], [old[i], new[i]], color='.8', lw=.7, zorder=1)
        ax.scatter(jitter, old, color='#0072B2', s=25, label='Existing prior', zorder=2)
        ax.scatter(1+jitter, new, color='#D55E00', s=25, label='Expanded prior', zorder=2)
        ax.axhline(1, color='.2', ls='--', lw=1)
        ax.set(xticks=[0, 1], xticklabels=['Existing', 'Expanded'], title=f'Observation {row["row"]}',
               ylabel='Estimated / exact numerator', xlim=(-.3, 1.3))
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Bright-prior extension: fixed observations, 16 sampling seeds')
    fig.tight_layout(rect=(0, .075, 1, .94))
    fig.text(.5, .015, 'V3.6 models · K1024 + M16384 · centre-only numerical diagnostic; no shear-bias measurement', ha='center', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'comparison.{ext}', dpi=180)
    plt.close(fig)
    lines += ['', '## Limits', '',
              'These three observations were previously inspected failures, not a representative holdout.',
              'New morphology realizations need not match individual old values; a common parent distribution is intended.',
              'Retaining all rows also extends the old generator domain; its posterior contribution is in metrics.json.',
              'The comparison tests integration accuracy, not whether the changed prior is astrophysically better.',
              'No selection-normalized shear fit or 0.3% calibration result has been produced.', '',
              '![Seed comparison](comparison.png)', '',
              'Protocol: [bright prior extension](../../BRIGHT_PRIOR_EXTENSION.md#scope-and-frozen-comparison).', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    (OUT/'metrics.json').write_text(json.dumps(numeric, indent=2)+'\n')
    (OUT/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    shutil.copy2(ROOT/'distribution_audit.json', OUT/'distribution_audit.json')
    for name in ('bright_prior_extension.py', 'audit_bright_distribution.py', 'report_bright_prior_extension.py'):
        shutil.copy2(Path(__file__).parent/name, OUT/name)
    shutil.copy2(ROOT/'bright_prior_extension_before_metric_fix.py', OUT/'build_evaluation_source.py')
    print(json.dumps(numeric, indent=2), flush=True)


if __name__ == '__main__':
    main()
