"""Compare two independent ensembles, retaining dependence among their walkers."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from check_galsbi_continuous_prior import save
from report_galsbi_population import checked_run
from report_galsbi_sampler_runs import profile_summary


def ensemble_profile(log_ratios):
    """Collapse walkers on likelihood scale; independent ensembles are chains.

    Input shape is independent-ensemble, iteration, interacting-walker.
    The resulting diagnostics assess expectations, not the full latent law.
    """
    values = np.asarray(log_ratios, float)
    if values.ndim != 3 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError('finite draws from at least two independent ensembles required')
    log_mean_per_iteration = logsumexp(values, axis=2)-np.log(values.shape[2])
    result = profile_summary(log_mean_per_iteration.T)
    result['replica_log_mean_ratio'] = (logsumexp(values, axis=(1, 2))
                                       -np.log(values.shape[1]*values.shape[2])).tolist()
    result['interpretation'] = ('Walkers are averaged before independent-ensemble Rhat/ESS/MCSE. '
                                'Two ensemble means cannot rule out shared missed modes.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = dict(status='completed_bounded_comparison', arms={}, accuracy_accepted=False,
        limitation='Original four-observation stress panel, including difficult obs1. No population shear estimate.')
    lines = ['# Hybrid ensemble versus original MH', '', report['limitation'], '',
        'Each diagnostic uses two independent replica time series after averaging their four walkers.', '',
        '| Arm | Steps | Observation | Shear | Ensemble-mean Rhat | Bulk ESS | Relative MCSE |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    all_protocols = []
    for arm in ('control', 'stretch'):
        roots = [args.root/'ensemble'/arm/f'replica{i}' for i in range(2)]
        checked = [checked_run(p) for p in roots]
        protocols = [p for p, _ in checked]
        if (protocols[0]['seed'] == protocols[1]['seed'] or
                protocols[0]['initial_bank_seed'] == protocols[1]['initial_bank_seed']):
            raise ValueError('independent replica seeds required')
        invariant = {k: v for k, v in protocols[0].items() if k not in ('seed', 'initial_bank_seed')}
        if invariant != {k: v for k, v in protocols[1].items() if k not in ('seed', 'initial_bank_seed')}:
            raise ValueError('within-arm replica protocol mismatch')
        all_protocols += protocols
        with np.load(roots[0]/'local_profiles.npz') as a, np.load(roots[1]/'local_profiles.npz') as b:
            if set(a.files) != set(b.files):
                raise ValueError('profile coverage differs')
            arm_report = dict(timing=[json.loads((p/'timing.json').read_text()) for p in roots], prefixes=[])
            for end in (8192, 16384):
                n = end-protocols[0]['warmup']
                profiles = {}
                for obs in protocols[0]['observations']:
                    for g in (.018, .022):
                        name = f'observation_{obs}_g1_{g:g}'
                        values = np.stack([a[name][:n], b[name][:n]])
                        if values.shape != (2, n, 4):
                            raise ValueError('profile draws incomplete')
                        p = ensemble_profile(values)
                        profiles[name] = p
                        d = p['diagnostics']
                        lines.append(f"| {arm} | {end} | {obs} | {g} | {d['rhat']} | {d['ess_bulk']} | {p['relative_mcse']} |")
                arm_report['prefixes'].append(dict(steps=end, profiles=profiles))
            report['arms'][arm] = arm_report
    # Only proposal identities may differ across arms; scientific target and
    # measurement starts, budgets, source snapshot and caching must match.
    keys = ('reference_protocol_sha256', 'observations', 'fixed_shear', 'density_batch_size',
            'steps', 'warmup', 'initialization', 'reuse_invariant_work', 'implementation_sha256')
    if any(any(p.get(k) != all_protocols[0].get(k) for k in keys) for p in all_protocols):
        raise ValueError('cross-arm target, budget or implementation mismatch')
    save(args.root/'ensemble_report.json', report)
    (args.root/'ensemble_report.md').write_text('\n'.join(lines)+'\n')
    print('ENSEMBLE_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
