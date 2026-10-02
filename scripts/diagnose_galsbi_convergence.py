"""Read-only chain forensics for the completed bright MH study; run under Slurm."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

from check_galsbi_continuous_prior import digest, save


def quantiles(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return np.quantile(x, [0, .1, .5, .9, 1]).tolist() if len(x) else None


def variance_split(values):
    """Population variance decomposition, keeping time and chain axes distinct."""
    x = np.asarray(values, float)
    within = x.var(axis=0).mean(axis=0)
    between = x.mean(axis=0).var(axis=0)
    return within, between, between/np.maximum(within+between, 1e-300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.source
    cohorts = json.loads((root/'reference/cohorts.json').read_text())
    latent = json.loads((root/'latent_diagnostics/report.json').read_text())['prefixes'][-1]['observations']
    nobs = len(latent)
    metrics = [dict(observation=i, magnitude=cohorts['measured_magnitude'][i], replicas=[]) for i in range(nobs)]
    traces, summaries, local, priors, kernels, initials = [], [], [], [], [], []
    for r in range(2):
        run = root/'runs'/f'replica{r}'
        p = json.loads((run/'protocol.json').read_text())
        result = json.loads((run/'result.json').read_text())
        assert p['observations'] == list(range(nobs))
        start, stride = p['warmup'], 8
        with np.load(run/'initial_state.npz') as a:
            initials.append(a['target_coordinates'].copy().reshape(nobs, 4, 10))
        with np.load(run/'chain.npz') as a:
            x = a['target_coordinates'][start::stride].copy().reshape(-1, nobs, 4, 10)
            s_full = a['summaries'][:, :, 0]
            s = s_full[start::stride].copy().reshape(-1, nobs, 4, 9)
            lp = a['log_prior'][start::stride].copy().reshape(-1, nobs, 4)
            lk = a['log_kernel'][start::stride, :, 0].copy().reshape(-1, nobs, 4)
            accepted = a['accepted'].copy()
            block = a['block'].copy()
            fraction = a['refresh_fraction'].copy()
            for obs in range(nobs):
                rows = slice(obs*4, obs*4+4)
                item = dict(acceptance={}, effective_summary_changes={}, frozen_scales=result['frozen_scale_multipliers'][rows])
                for b in (0, 1, 2, 4, 5):
                    steps = np.flatnonzero((block == b) & (np.arange(len(block)) >= start))
                    acc = accepted[steps, rows]
                    before, after = s_full[steps-1, rows], s_full[steps, rows]
                    delta = after-before
                    changed = np.any(np.abs(delta) > 1e-10, axis=2)
                    item['acceptance'][str(b)] = acc.mean(0).tolist()
                    item['effective_summary_changes'][str(b)] = changed.mean(0).tolist()
                    if b == 4:
                        item['environment_b_changed'] = (np.abs(delta[:, :, 8]) > 1e-10).mean(0).tolist()
                        item['environment_crowding_changed'] = np.any(np.abs(delta[:, :, 5:8]) > 1e-10, axis=2).mean(0).tolist()
                        full = fraction[steps] == 1.
                        item['full_refresh_acceptance'] = acc[full].mean(0).tolist()
                metrics[obs]['replicas'].append(item)
            del s_full, accepted, block, fraction
        traces.append(x)
        summaries.append(s)
        priors.append(lp)
        kernels.append(lk)
        with np.load(run/'local_profiles.npz') as a:
            local.append(np.stack([np.stack([a[f'observation_{i}_g1_{g:g}'][::stride]
                                            for g in (.018, .022)], axis=1) for i in range(nobs)], axis=1))
        print('READ_REPLICA', r, flush=True)
    x = np.concatenate(traces, axis=2)
    s = np.concatenate(summaries, axis=2)
    logpost = np.concatenate(priors, axis=2)+np.concatenate(kernels, axis=2)
    ratios = np.concatenate(local, axis=3)  # time, observation, shear, chain
    initial = np.concatenate(initials, axis=1)
    h = .002
    panel_order = []
    for obs in range(nobs):
        v = x[:, obs].copy()
        angle = np.angle(np.mean(np.exp(1j*v[:, :, 9])))
        v[:, :, 9] = angle+(v[:, :, 9]-angle+np.pi) % (2*np.pi)-np.pi
        within, between, fraction = variance_split(v)
        item = metrics[obs]
        item['between_chain_variance_fraction_coordinates'] = fraction.tolist()
        item['chain_means_coordinates'] = v.mean(0).tolist()
        item['chain_sd_coordinates'] = v.std(0).tolist()
        item['initial_coordinates'] = initial[obs].tolist()
        item['initial_final_sersic_rank_correlation'] = float(spearmanr(initial[obs, :, 7], v.mean(0)[:, 7]).statistic)
        halves = np.array_split(v, 2)
        item['late_drift_in_pooled_sd'] = (np.abs(halves[1].mean((0, 1))-halves[0].mean((0, 1)))
            /np.sqrt(np.maximum(within+between, 1e-300))).tolist()
        post = logpost[:, obs]
        item['posterior_chain_mean_range'] = float(np.ptp(post.mean(0)))
        item['posterior_chain_mean_last_minus_first_quarter'] = (post[-len(post)//4:].mean(0)-post[:len(post)//4].mean(0)).tolist()
        cols = [1, 6, 7, 8, 9]
        centered = v[:, :, cols]-v[:, :, cols].mean(0, keepdims=True)
        covariance = np.cov(centered.reshape(-1, 5).T)
        sd = np.sqrt(np.diag(covariance))
        corr = covariance/np.maximum(sd[:, None]*sd, 1e-300)
        item['within_chain_physical_correlation'] = corr.tolist()
        item['max_morphology_shape_correlation'] = float(np.max(np.abs(corr[:3, 3:])))
        # Finite-h diagnostic of Louis' identity, not an infinitesimal derivative claim.
        lr = ratios[:, obs]
        score = (lr[:, 1]-lr[:, 0])/(2*h)
        curvature = -(lr[:, 1]+lr[:, 0])/h**2
        ws, bs, fs = variance_split(score[:, :, None])
        means = logsumexp(lr, axis=(0, 2))-np.log(lr.shape[0]*lr.shape[2])
        item['shear_functionals'] = dict(mean_score=float(score.mean()),
            mean_conditional_information=float(curvature.mean()), score_variance=float(score.var()),
            score_between_chain_variance_fraction=float(fs[0]),
            information_identity_finite_h=float(curvature.mean()-score.var()),
            marginal_information_finite_h=float(-means.sum()/h**2),
            chain_mean_scores=score.mean(0).tolist())
        item['local_ratio_worst_rhat'] = max(v['diagnostics']['rhat'] for v in latent[obs]['profiles'].values())
        item['local_ratio_pass'] = all(v['diagnostics']['rhat'] < 1.01 and v['diagnostics']['ess_bulk'] > 400
                                      for v in latent[obs]['profiles'].values())
    # Include best/median/worst ratio examples in each actual magnitude band.
    for lo, hi in ((-np.inf, 20.), (20., 22.)):
        indices = [i for i, m in enumerate(metrics) if lo <= m['magnitude'] < hi]
        ordered = sorted(indices, key=lambda i: metrics[i]['local_ratio_worst_rhat'])
        for q in (0, .33, .67, 1.):
            panel_order.append(ordered[round(q*(len(ordered)-1))])
    for key in ('posterior_chain_mean_range', 'max_morphology_shape_correlation'):
        panel_order.append(max(range(nobs), key=lambda i: metrics[i][key]))
    panel_order = list(dict.fromkeys(panel_order))
    panel = dict(observations=panel_order,
        selection='Post-hoc diagnostic best/33rd/67th/worst ratio Rhat in each measured band, plus largest chain-density spread and cross-block correlation; not validation.',
        metrics=[metrics[i] for i in panel_order])
    save(args.output/'panel.json', panel)
    np.savez_compressed(args.output/'panel_draws.npz', coordinates=x[:, panel_order], summaries=s[:, panel_order],
        logposterior=logpost[:, panel_order], local_log_ratios=ratios[:, panel_order],
        observations=panel_order, initial_coordinates=initial[panel_order])
    groups = {}
    for cut, ids in cohorts['cohorts'].items():
        groups[cut] = dict(observations=len(ids), ratio_pass=sum(metrics[i]['local_ratio_pass'] for i in ids),
            acceptance={str(b): quantiles([v for i in ids for r in metrics[i]['replicas']
                for v in r['acceptance'][str(b)]]) for b in (0, 1, 2, 4, 5)},
            effective_changes={str(b): quantiles([v for i in ids for r in metrics[i]['replicas']
                for v in r['effective_summary_changes'][str(b)]]) for b in (0, 1, 2, 4, 5)},
            full_refresh_acceptance=quantiles([v for i in ids for r in metrics[i]['replicas'] for v in r['full_refresh_acceptance']]),
            between_chain_variance_fraction={str(c): quantiles([metrics[i]['between_chain_variance_fraction_coordinates'][c] for i in ids]) for c in (1, 6, 7, 8)},
            initial_final_sersic_correlation=quantiles([metrics[i]['initial_final_sersic_rank_correlation'] for i in ids]),
            maximum_cross_block_correlation=quantiles([metrics[i]['max_morphology_shape_correlation'] for i in ids]),
            posterior_chain_mean_range=quantiles([metrics[i]['posterior_chain_mean_range'] for i in ids]),
            posterior_late_gain=quantiles([v for i in ids for v in metrics[i]['posterior_chain_mean_last_minus_first_quarter']]),
            finite_h_information_sum=float(sum(metrics[i]['shear_functionals']['marginal_information_finite_h'] for i in ids)))
    save(args.output/'chain_audit.json', dict(source=str(root), source_plan_sha256=digest(root/'plan.json'),
        quantile_order=[0, .1, .5, .9, 1], cohorts=groups, observations=metrics))
    print('CHAIN_AUDIT_COMPLETE', json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
