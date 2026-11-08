"""Saved-chain and importance-overlap audit; figures are diagnostic, not calibration."""
import argparse
import importlib.util
import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from check_galsbi_continuous_prior import save


def q(values):
    return np.quantile(values, [.1, .5, .9]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    out = args.audit/'profiles'
    out.mkdir(exist_ok=False)
    cohorts = json.loads((args.source/'reference/cohorts.json').read_text())['cohorts']
    with np.load(args.audit/'chains/panel_draws.npz') as a:
        x, s, ids = a['coordinates'], a['summaries'], a['observations']
    audit = json.loads((args.audit/'chains/chain_audit.json').read_text())
    metrics = audit['observations']
    # First, median and worst bright examples from the predeclared forensic panel rule.
    chosen = [int(ids[0]), int(ids[2]), int(ids[3])]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for row, obs in enumerate(chosen):
        i = list(ids).index(obs)
        times = np.arange(len(x))*8+8193
        for c in range(8):
            color = plt.cm.tab10(c)
            axes[row, 0].plot(times, s[:, i, c, 2], lw=.6, color=color, alpha=.8)
            axes[row, 1].plot(times, s[:, i, c, 3], lw=.6, color=color, alpha=.8)
            axes[row, 2].scatter(s[::8, i, c, 2], s[::8, i, c, 4], s=3, color=color, alpha=.35)
        axes[row, 0].set_ylabel(f'Obs {obs}, mag {metrics[obs]["magnitude"]:.2f}\nSersic index')
        axes[row, 1].set_ylabel('Latent magnitude')
        axes[row, 2].set_ylabel('Circularized radius [arcsec]')
        axes[row, 0].set_xlabel('MH iteration')
        axes[row, 1].set_xlabel('MH iteration')
        axes[row, 2].set_xlabel('Sersic index')
    axes[0, 0].set_title('Eight chain traces; warmup excluded')
    axes[0, 1].set_title('Different colours = different chains')
    axes[0, 2].set_title('Explored morphology geometry')
    fig.savefig(out/'chain_traces.png', dpi=160)
    plt.close(fig)
    del x, s

    parts = []
    for r in range(2):
        with np.load(args.source/'runs'/f'replica{r}'/'shear_profiles.npz') as a:
            grid = a['grid'].copy()
            parts.append(a['log_ratios'].copy())
    means = [logsumexp(a, axis=(2, 3))-np.log(a.shape[2]*a.shape[3]) for a in parts]
    combined = np.logaddexp(*means)-np.log(2.)
    report = json.loads((args.source/'report.json').read_text())
    fields = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, (cut, cohort) in enumerate(cohorts.items()):
        selected = report['cohorts'][cut]['prefixes'][-1]['log_selected_profile']
        axes[row, 0].plot(grid, selected, marker='.', label='Pooled saved profile')
        axes[row, 0].axvline(.02, color='black', ls='--', label='Injected shear')
        axes[row, 0].set_xlim(-.01, .05)
        axes[row, 0].set_ylim(-10, 5)
        axes[row, 0].set_ylabel(f'Measured mag <{cut}\nRelative selected log likelihood')
        axes[row, 0].set_xlabel('g1 (g2 fixed at zero)')
        summary = []
        for j, g in enumerate(grid):
            raw = np.concatenate([a[cohort, j] for a in parts], axis=2).reshape(len(cohort), -1)
            logsum = logsumexp(raw, axis=1)
            ess = np.exp(2*logsum-logsumexp(2*raw, axis=1))
            largest = np.exp(raw.max(1)-logsum)
            topchain = []
            for pos, obs in enumerate(cohort):
                logweights = np.concatenate([logsumexp(a[obs, j], axis=0) for a in parts])
                topchain.append(float(np.exp(logweights.max()-logsumexp(logweights))))
            info = dict(g1=float(g), raw_weight_ess_q10_median_q90=q(ess),
                largest_draw_mass_q10_median_q90=q(largest), top_chain_mass_q10_median_q90=q(topchain),
                objects_ess_below400=int((ess < 400).sum()),
                replica_log_ratio_disagreement_q10_median_q90=q(np.abs(means[0][cohort, j]-means[1][cohort, j])))
            if any(np.isclose(g, node) for node in (.01, .015, .018, .022, .04)):
                _, khat = az.psislw(raw)
                info['pareto_k_q10_median_q90'] = q(khat)
                info['objects_k_above07'] = int((khat > .7).sum())
            summary.append(info)
        ess_median = [v['raw_weight_ess_q10_median_q90'][1] for v in summary]
        ess_low = [v['raw_weight_ess_q10_median_q90'][0] for v in summary]
        axes[row, 1].semilogy(grid, ess_median, label='Median raw weight ESS')
        axes[row, 1].semilogy(grid, ess_low, label='10th percentile')
        axes[row, 1].axhline(400, color='grey', ls=':')
        axes[row, 1].axvline(.02, color='black', ls='--')
        axes[row, 1].set_xlabel('g1 (g2 fixed at zero)')
        axes[row, 1].set_ylabel('Weight ESS (not MCMC ESS)')
        j = int(np.argmax(selected))
        contribution = combined[cohort, j]
        order = np.argsort(np.abs(contribution))[::-1][:8]
        fields[cut] = dict(nodes=summary,
            largest_absolute_numerator_contributions_at_grid_peak=[dict(observation=cohort[i],
                log_ratio=float(contribution[i])) for i in order])
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle('Unvalidated recovery profiles and importance-overlap diagnostics')
    fig.savefig(out/'shear_profiles.png', dpi=160)
    plt.close(fig)
    save(out/'overlap.json', dict(cohorts=fields,
        interpretation='Raw weight ESS and Pareto k do not account for autocorrelation or missing modes. No weights were smoothed for inference.'))
    # Correct only the known interval interpolation bug; preserve frozen original outputs.
    spec = importlib.util.spec_from_file_location('corrected_bright_math', args.audit/'scripts/galsbi_bright_math.py')
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    corrected = {}
    for cut, cohort in report['cohorts'].items():
        corrected[cut] = [dict(steps=e['steps'], fit=math.profile_fit(grid, e['log_selected_profile'])) for e in cohort['prefixes']]
    save(out/'corrected_intervals.json', dict(cohorts=corrected,
        limitation='Reporting correction only; chains and original likelihood estimates unchanged and recovery remains unvalidated.'))
    print('PROFILE_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
