"""Aggregate independent population replicas with within-shard MC covariance.

The population-selection normalizer is identical between replicas and cancels
from their differences. These are numerator profiles, not a shear estimate.
"""
import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
from scipy.special import logsumexp

from check_galsbi_continuous_prior import digest, save
from report_galsbi_sampler_runs import profile_summary


def aggregate_profile(log_ratios):
    """Return log mean and delta-method influence, preserving time and chain."""
    values = np.asarray(log_ratios, float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('finite time-by-chain profile draws required')
    mean = float(logsumexp(values)-np.log(values.size))
    return mean, np.exp(values-mean)-1.


def conditional_mcse(influence):
    """Within-shard covariance is retained by summing before computing MCSE."""
    if not np.isfinite(influence).all():
        raise ValueError('nonfinite aggregate influence')
    if np.ptp(influence) == 0:
        return 0.
    value = np.asarray(az.mcse(influence.T, method='mean')).item()
    return value if np.isfinite(value) else None


def checked_run(root):
    p = json.loads((root/'protocol.json').read_text())
    r = json.loads((root/'result.json').read_text())
    replay = json.loads((root/'initial_replay.json').read_text())
    sed = r.get('sed_replay') or {}
    if (r['protocol_sha256'] != digest(root/'protocol.json') or r['status'] != 'completed_diagnostic'
            or not np.isfinite(replay['max_abs_error']) or replay['max_abs_error'] > 1e-6
            or sed.get('valid_proposals', 0) <= 0
            or not 0 <= sed.get('maximum_summary_change', np.inf) <= 1e-10
            or not 0 <= sed.get('maximum_log_kernel_change', np.inf) <= 1e-6):
        raise ValueError('run failed numerical validation')
    return p, r


def report_shard(root, shard):
    plan = json.loads((root/'plan.json').read_text())
    paths = [root/'runs'/f'shard{shard:04d}'/f'replica{r}' for r in range(2)]
    runs = [checked_run(p) for p in paths]
    protocols = [p for p, _ in runs]
    first = protocols[0]
    different = {'seed', 'initial_bank_seed'}
    if any({k: v for k, v in p.items() if k not in different} !=
           {k: v for k, v in first.items() if k not in different} for p in protocols):
        raise ValueError('replica protocols differ beyond declared independent seeds')
    expected = list(range(shard*plan['batch_observations'],
                          min((shard+1)*plan['batch_observations'], plan['observations'])))
    if first['observations'] != expected:
        raise ValueError('shard observation coverage mismatch')
    if any(p['seed'] != plan['mh_seed_base']+2*shard+r or
           p['initial_bank_seed'] != plan['bank_seed_base']+2*shard+r for r, p in enumerate(protocols)):
        raise ValueError('independent shard/replica seed mismatch')
    report = dict(shard=shard, observations=expected, prefixes=[],
        protocol_sha256=[digest(p/'protocol.json') for p in paths],
        numerical_checks_pass=True,
        individual_latent_pass_counts=[sum(o['exploration_checks_pass'] for o in r['observations']) for _, r in runs],
        limitation='Conditional MCSE assumes adequate mixing. Two replicas and prefix agreement cannot prove absence of shared bias.')
    with np.load(paths[0]/'local_profiles.npz') as a, np.load(paths[1]/'local_profiles.npz') as b:
        archives = [a, b]
        required = {f'observation_{i}_g1_{g:g}' for i in expected for g in (.018, .022)}
        if any(set(z.files) != required for z in archives):
            raise ValueError('missing or extra observation profiles')
        for end in (8192, 16384):
            if end > first['steps']:
                continue
            n = end-first['warmup']
            sums = np.zeros((2, 2))
            influence = np.zeros((2, 2, n, 4))
            details = []
            for obs in expected:
                item = dict(observation=obs, profiles={})
                for j, g in enumerate((.018, .022)):
                    name = f'observation_{obs}_g1_{g:g}'
                    draws = [z[name] for z in archives]
                    if any(v.shape != (first['steps']-first['warmup'], 4) for v in draws):
                        raise ValueError('profile shape mismatch')
                    means = []
                    for r, values in enumerate(draws):
                        mean, phi = aggregate_profile(values[:n])
                        sums[r, j] += mean
                        influence[r, j] += phi
                        means.append(mean)
                    combined = profile_summary(np.concatenate([v[:n] for v in draws], axis=1))
                    item['profiles'][str(g)] = dict(replica_log_means=means, combined=combined)
                details.append(item)
            report['prefixes'].append(dict(steps=end, replica_log_profiles=sums.tolist(),
                replica_profile_mcse=[[conditional_mcse(influence[r, j]) for j in range(2)] for r in range(2)],
                replica_score_mcse=[conditional_mcse((influence[r, 1]-influence[r, 0])/.004) for r in range(2)],
                observations=details))
    output = root/'reports'/f'shard{shard:04d}.json'
    output.parent.mkdir(exist_ok=True)
    save(output, report)
    print('SHARD_REPORT', shard, len(expected), flush=True)


def combine(root):
    plan = json.loads((root/'plan.json').read_text())
    count = (plan['observations']+plan['batch_observations']-1)//plan['batch_observations']
    missing = [s for s in range(count) if not (root/'reports'/f'shard{s:04d}.json').exists()]
    if missing:
        save(root/'report.json', dict(status='incomplete', missing_shards=missing))
        raise ValueError(f'{len(missing)} missing shard reports')
    shards = [json.loads((root/'reports'/f'shard{s:04d}.json').read_text()) for s in range(count)]
    ids = [i for s in shards for i in s['observations']]
    if ids != list(range(plan['observations'])) or not all(s['numerical_checks_pass'] for s in shards):
        raise ValueError('population coverage or numerical checks failed')
    report = dict(status='completed_population_diagnostic', observations=len(ids), prefixes=[],
        no_selection_normalized_shear_estimate=True,
        limitation='Conditional MCSE assumes adequate mixing. Shared initialization/missed-mode/log-estimator bias is not ruled out by two replicas.')
    for end in (8192, 16384):
        parts = [next(p for p in s['prefixes'] if p['steps'] == end) for s in shards]
        sums = np.sum([p['replica_log_profiles'] for p in parts], axis=0)
        errors = np.array([p['replica_profile_mcse'] for p in parts], dtype=float)
        score_errors = np.array([p['replica_score_mcse'] for p in parts], dtype=float)
        if not np.isfinite(errors).all() or not np.isfinite(score_errors).all():
            raise ValueError('unavailable conditional aggregate MCSE')
        mcse = np.sqrt(np.square(errors).sum(0))
        score_mcse = np.sqrt(np.square(score_errors).sum(0))
        difference = sums[1]-sums[0]
        obs = [o for p in parts for o in p['observations']]
        good = []
        for o in obs:
            good.append(all(v['combined']['valid'] and v['combined']['diagnostics']['rhat'] is not None
                and v['combined']['diagnostics']['rhat'] < 1.01 and v['combined']['diagnostics']['ess_bulk'] > 400
                for v in o['profiles'].values()))
        report['prefixes'].append(dict(steps=end, replica_log_profiles=sums.tolist(),
            replica_profile_conditional_mcse=mcse.tolist(), replica_difference=difference.tolist(),
            difference_conditional_mcse=np.sqrt(np.square(mcse).sum(0)).tolist(),
            replica_central_score=((sums[:, 1]-sums[:, 0])/.004).tolist(),
            replica_score_conditional_mcse=score_mcse.tolist(),
            score_difference=float((difference[1]-difference[0])/.004),
            score_difference_conditional_mcse=float(np.linalg.norm(score_mcse)),
            observations_passing_both_ratio_diagnostics=sum(good)))
    save(root/'report.json', report)
    lines = ['# 20k continuous-prior sampling diagnostic', '',
        f'Completed {len(ids)} observations, two independent replicas, four chains each.', '',
        'The same selection normalizer cancels in replica differences. No selected shear estimate is reported.', '',
        '| Steps | Ratio diagnostics passing | Replica difference at .018 | Replica difference at .022 | Score difference ± conditional MCSE |',
        '| --- | --- | --- | --- | --- |']
    for p in report['prefixes']:
        lines.append(f"| {p['steps']} | {p['observations_passing_both_ratio_diagnostics']}/{len(ids)} | "
            f"{p['replica_difference'][0]:.6g} | {p['replica_difference'][1]:.6g} | "
            f"{p['score_difference']:.6g} ± {p['score_difference_conditional_mcse']:.6g} |")
    lines += ['', report['limitation'], '',
        'Score is the central finite difference over g1=.018,.022. Within-shard covariance and covariance between shear points are included. Independent-shard variances are added.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('POPULATION_REPORT', report['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    if args.shard is None:
        combine(args.root)
    else:
        report_shard(args.root, args.shard)


if __name__ == '__main__':
    main()
