"""Read-only independent-chain/prefix report for the bounded overnight study.

Run under Slurm. Missing runs, incompatible protocols or failed numerical
checks cannot produce a passing report. This never promotes a sampler or
estimates selected-population shear.
"""
import argparse
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np


def diagnostics(values):
    """Same rank-Rhat/bulk-ESS convention as the pilots, without model imports."""
    if np.ptp(values) == 0:
        return dict(rhat=None, ess_bulk=None, constant=True)
    rhat, ess = float(az.rhat(values, method='rank')), float(az.ess(values, method='bulk'))
    return dict(rhat=rhat if np.isfinite(rhat) else None,
                ess_bulk=ess if np.isfinite(ess) else None, constant=False)


def passing(check):
    return (not check['constant'] and check['rhat'] is not None and check['ess_bulk'] is not None
            and check['rhat'] < 1.01 and check['ess_bulk'] > 400)


def profile_summary(log_ratios):
    """Likelihood-scale mean and MCSE; rank diagnostics alone do not give a mean error."""
    if np.isnan(log_ratios).any() or np.isposinf(log_ratios).any() or not np.isfinite(log_ratios).any():
        return dict(valid=False)
    maximum = float(np.max(log_ratios))
    scaled = np.exp(log_ratios-maximum)
    mean = float(scaled.mean())
    check = diagnostics(scaled.T)
    mcse = np.asarray(az.mcse(scaled.T, method='mean')).item()
    return dict(valid=True, log_mean_ratio=maximum+np.log(mean),
        relative_mcse=mcse/mean if np.isfinite(mcse) else None,
        largest_draw_fraction=float(scaled.max()/scaled.sum()), diagnostics=check,
        interpretation='MCSE is conditional on adequate mixing; this is a single-observation numerator ratio.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefixes', type=int, nargs='+')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status='incomplete', runs=[], all_checks_pass=False,
                  reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation='Small model-mock panel only; no selection normalization, shear estimate or 0.3% claim.')
    loaded = []
    for root in args.runs:
        receipt = dict(path=str(root.resolve()), complete=False)
        report['runs'].append(receipt)
        required = ('result.json', 'protocol.json', 'chain.npz', 'local_profiles.npz', 'initial_replay.json')
        missing = [name for name in required if not (root/name).exists()]
        if missing:
            receipt['missing'] = missing
            continue
        try:
            protocol_bytes = (root/'protocol.json').read_bytes()
            protocol = json.loads(protocol_bytes)
            result = json.loads((root/'result.json').read_text())
            replay = json.loads((root/'initial_replay.json').read_text())
            with np.load(root/'chain.npz') as archive:
                data = {key: archive[key] for key in ('target_coordinates', 'summaries',
                                                     'type_probability', 'neighbour_counts')}
            with np.load(root/'local_profiles.npz') as archive:
                profiles = {key: archive[key] for key in archive.files}
        except (OSError, ValueError, KeyError) as error:
            receipt['error'] = str(error)
            continue
        sed = result.get('sed_replay') or {}
        receipt.update(complete=True, seed=protocol['seed'],
            protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
            individual_run_pass=bool(result['all_exploration_checks_pass']))
        receipt['numerical_checks_pass'] = bool(
            result['status'] == 'completed_diagnostic'
            and receipt['protocol_sha256'] == result['protocol_sha256']
            and replay['max_abs_error'] <= replay['tolerance']
            and sed.get('valid_proposals', 0) > 0
            and sed.get('maximum_summary_change', np.inf) <= 1e-10
            and sed.get('maximum_log_kernel_change', np.inf) <= 1e-6)
        if protocol['environments'] != 1 or protocol['environment_update'] != 'persistent':
            receipt['error'] = 'Report requires one persistent scene per chain.'
            continue
        if protocol.get('ensemble_morphology'):
            receipt['error'] = 'Coupled walkers require an independent-ensemble report.'
            continue
        expected = (protocol['steps'], len(protocol['observations'])*protocol['chains_per_observation'])
        if any(value.shape[:2] != expected for value in data.values()):
            receipt['error'] = 'Chain dimensions do not match the completed protocol.'
            continue
        names = {f'observation_{obs}_g1_{shear:g}' for obs in protocol['observations'] for shear in (.018, .022)}
        profile_shape = (protocol['steps']-protocol['warmup'], protocol['chains_per_observation'])
        if set(profiles) != names or any(value.shape != profile_shape for value in profiles.values()):
            receipt['error'] = 'Profile draws do not match the completed protocol.'
            continue
        loaded.append((protocol, data, profiles))
    keys = ('reference_protocol_sha256', 'observations', 'fixed_shear', 'density_batch_size',
            'environments', 'environment_update', 'proposal_blocks', 'proposal_block_probabilities',
            'morphology_covariance', 'joint_environment', 'reuse_invariant_work', 'initialization',
            'steps', 'warmup', 'implementation_sha256')
    compatible = len(loaded) == len(args.runs) and len(loaded) >= 2
    if compatible:
        first = loaded[0][0]
        compatible = all(all(p.get(k) == first.get(k) for k in keys) for p, _, _ in loaded)
        compatible &= len({p['seed'] for p, _, _ in loaded}) == len(loaded)
    report['compatible_independent_runs'] = bool(compatible)
    if compatible:
        first = loaded[0][0]
        ends = sorted(set((args.prefixes or [])+[first['steps']]))
        report['prefixes'] = []
        for end in ends:
            start = first['warmup']
            if not start < end <= first['steps']:
                continue
            prefix = dict(total_steps=end, retained_draws_per_chain=end-start, observations=[])
            for i, obs in enumerate(first['observations']):
                views = {}
                for p, data, _ in loaded:
                    rows = slice(i*p['chains_per_observation'], (i+1)*p['chains_per_observation'])
                    x, s = data['target_coordinates'][start:end, rows], data['summaries'][start:end, rows, 0]
                    values = {f'summary_{j}': s[:, :, j] for j in range(9)}
                    values.update({f'coordinate_{j}': x[:, :, j] for j in (0, 2, 3, 4, 5, 7, 8)})
                    values['red_probability'] = data['type_probability'][start:end, rows, 0]
                    values['neighbour_count'] = data['neighbour_counts'][start:end, rows]
                    for name, value in values.items():
                        views.setdefault(name, []).append(value)
                checks = {name: diagnostics(np.concatenate(value, axis=1).T) for name, value in views.items()}
                profiles = {}
                for shear in (.018, .022):
                    name = f'observation_{obs}_g1_{shear:g}'
                    profiles[str(shear)] = profile_summary(np.concatenate(
                        [saved[name][:end-start] for _, _, saved in loaded], axis=1))
                active = [c for c in checks.values() if not c['constant']]
                profiles_pass = all(p['valid'] and p['relative_mcse'] is not None
                                    and passing(p['diagnostics']) for p in profiles.values())
                prefix['observations'].append(dict(reference_observation=obs, diagnostics=checks,
                    max_rhat=max(c['rhat'] for c in active) if active and all(c['rhat'] is not None for c in active) else None,
                    min_bulk_ess=min(c['ess_bulk'] for c in active) if active and all(c['ess_bulk'] is not None for c in active) else None,
                    profiles=profiles,
                    checks_pass=bool(active) and all(passing(c) for c in active) and profiles_pass))
            report['prefixes'].append(prefix)
        report['status'] = 'completed_diagnostic'
        report['all_checks_pass'] = (all(r.get('numerical_checks_pass') and r.get('individual_run_pass')
                                             for r in report['runs'])
                                    and all(o['checks_pass'] for o in report['prefixes'][-1]['observations']))
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Overnight sampler diagnostics', '', f'Status: {report["status"]}.', '',
             f'All individual and combined checks pass: **{report["all_checks_pass"]}**.', '',
             report['limitation'], '', '| Steps | Observation | Worst R-hat | Minimum bulk ESS | Pass |',
             '| --- | --- | --- | --- | --- |']
    for prefix in report.get('prefixes', []):
        for obs in prefix['observations']:
            rhat = f'{obs["max_rhat"]:.4f}' if obs['max_rhat'] is not None else 'unavailable'
            ess = f'{obs["min_bulk_ess"]:.1f}' if obs['min_bulk_ess'] is not None else 'unavailable'
            lines.append(f'| {prefix["total_steps"]} | {obs["reference_observation"]} | {rhat} | {ess} | {obs["checks_pass"]} |')
    lines += ['', 'Full diagnostics, numerical receipts and conditional numerator-ratio MCSEs are in report.json.',
              'Missing or incompatible runs cannot pass. No sampler is automatically promoted.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('OVERNIGHT_REPORT', report['status'], report['all_checks_pass'], flush=True)


if __name__ == '__main__':
    main()
