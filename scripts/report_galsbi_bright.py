"""Selected bright-cohort shear recovery with numerical uncertainty and failure gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import logsumexp

from check_galsbi_continuous_prior import digest, save
from galsbi_bright_math import profile_fit, selected_log_profile
from report_galsbi_population import checked_run, conditional_mcse
from report_galsbi_sampler_runs import passing, profile_summary
from run_galsbi_bright import selected_reference


def peak_gradient(grid, loglike):
    k = int(np.argmax(loglike))
    indices = np.arange(k-1, k+2)
    x = grid[indices]-grid[k]
    inverse = np.linalg.inv(np.column_stack((x*x, x, np.ones(3))))
    a, b, _ = inverse@loglike[indices]
    return indices, b/(2*a*a)*inverse[0]-inverse[1]/(2*a)


def report(root):
    plan = json.loads((root/'plan.json').read_text())
    reference, reference_result = selected_reference(root)
    cohorts = json.loads((reference/'cohorts.json').read_text())['cohorts']
    runs = [root/'runs'/f'replica{r}' for r in range(2)]
    protocols = [checked_run(p)[0] for p in runs]
    ignore = {'seed', 'initial_bank_seed'}
    if ({k: v for k, v in protocols[0].items() if k not in ignore}
            != {k: v for k, v in protocols[1].items() if k not in ignore}
            or protocols[0]['seed'] == protocols[1]['seed']
            or protocols[0]['initial_bank_seed'] == protocols[1]['initial_bank_seed']):
        raise ValueError('independent matched replica protocols required')
    expected = list(range(reference_result['observation_count']))
    if any(p['observations'] != expected for p in protocols):
        raise ValueError('missing or reordered bright observations')
    latent_output = root/'latent_diagnostics'
    command = [sys.executable, '-u', '-c',
        'import runpy,sys;sys.path[:0]=[".","scripts"];runpy.run_path("scripts/report_galsbi_sampler_runs.py",run_name="__main__")',
        *map(str, runs), '--output', str(latent_output), '--prefixes', *map(str, plan['prefixes'])]
    subprocess.run(command, check=True)
    latent = json.loads((latent_output/'report.json').read_text())
    if latent['status'] != 'completed_diagnostic' or not latent['compatible_independent_runs']:
        raise ValueError('latent diagnostic report incomplete')
    arrays, selections = [], []
    grid = np.asarray(plan['g1_grid'])
    steps = None
    for r, runroot in enumerate(runs):
        receipt = json.loads((runroot/'shear_profiles.json').read_text())
        if (receipt['data_sha256'] != digest(runroot/'shear_profiles.npz')
                or receipt['protocol_sha256'] != digest(runroot/'protocol.json')
                or receipt['observation_sha256'] != reference_result['observations_sha256']
                or receipt['plan_sha256'] != digest(root/'plan.json') or receipt['max_replay_error'] > 1e-6):
            raise ValueError('profile receipt mismatch')
        with np.load(runroot/'shear_profiles.npz') as z:
            if not np.array_equal(z['grid'], grid) or (steps is not None and not np.array_equal(z['steps'], steps)):
                raise ValueError('profile grid or retained steps mismatch')
            steps = z['steps'].copy()
            arrays.append(z['log_ratios'].copy())
        sr = root/f'selection{r}'
        receipt = json.loads((sr/'result.json').read_text())
        if (receipt['data_sha256'] != digest(sr/'chunks.npz')
                or receipt['plan_sha256'] != digest(root/'plan.json')
                or receipt['reference_protocol_sha256'] != digest(reference/'protocol.json')
                or receipt['seed_base'] != plan['selection_seed_base']+r*10000):
            raise ValueError('selection receipt mismatch')
        with np.load(sr/'chunks.npz') as z:
            if not np.array_equal(z['grid'], grid):
                raise ValueError('selection grid mismatch')
            selections.append(z['values'].copy())
    selection = np.concatenate(selections)
    if selection.shape != (2*plan['selection_chunks'], len(grid), len(cohorts), 2):
        raise ValueError('incomplete independent selection chunks')
    centre = int(np.flatnonzero(np.isclose(grid, .02, atol=1e-12, rtol=0))[0])
    result = dict(status='completed_bright_recovery_test', conditional_g2=0., injected_g1=.02,
        plan_sha256=digest(root/'plan.json'), cohorts={},
        limitation='Small GalSBI model closure; conditional g1 only. Numerical failures invalidate recovery. No sub-percent bias claim; generator/native-density approximation remains.')
    for c, limit in enumerate(plan['magnitude_limits']):
        ids = cohorts[str(float(limit))]
        n = len(ids)
        z_chunks = selection[:, :, c, 1]
        z_mean = z_chunks.mean(0)
        entries = []
        for end in plan['prefixes']:
            retain = steps <= end
            draws = [a[ids][:, :, retain] for a in arrays]
            joined = np.concatenate(draws, axis=3)
            logmeans = logsumexp(joined, axis=(2, 3))-np.log(joined.shape[2]*joined.shape[3])
            numerator = logmeans.sum(0)
            selected = selected_log_profile(numerator, z_mean, n, centre)
            fit = profile_fit(grid, selected)
            replica_fits = []
            for r, d in enumerate(draws):
                num = (logsumexp(d, axis=(2, 3))-np.log(d.shape[2]*d.shape[3])).sum(0)
                replica_fits.append(profile_fit(grid, selected_log_profile(num, selections[r][:, :, c, 1].mean(0), n, centre)))
            checks = next(p for p in latent['prefixes'] if p['total_steps'] == end)['observations']
            check_ids = {o['reference_observation']: o for o in checks}
            convergence_count = sum(check_ids[i]['checks_pass'] for i in ids)
            diagnostics = []
            relevant = set(np.flatnonzero(np.isclose(grid, .018) | np.isclose(grid, .022)).tolist())
            numerical = dict(mh_shear_mcse=None, selection_shear_mcse=None, combined_shear_mcse=None)
            ladders = {}
            if fit['valid']:
                indices, gradient = peak_gradient(grid, selected)
                relevant.update(indices.tolist())
                phi = np.zeros(joined.shape[2:])
                for j, derivative in zip(indices, gradient):
                    phi += derivative*(np.exp(joined[:, j]-logmeans[:, j, None, None])-1).sum(0)
                mh_mcse = conditional_mcse(phi)
                phi_z = -n*(z_chunks/z_mean-z_chunks[:, [centre]]/z_mean[centre])
                selection_mcse = float(np.std(phi_z[:, indices]@gradient, ddof=1)/np.sqrt(len(z_chunks)))
                numerical.update(mh_shear_mcse=mh_mcse, selection_shear_mcse=selection_mcse,
                    combined_shear_mcse=float(np.hypot(mh_mcse, selection_mcse)) if mh_mcse is not None else None)
                for name, alternative in (
                        ('half_flow_draws', selection[:, :, c, 0].mean(0)),
                        ('half_population', np.concatenate([s[:len(s)//2, :, c, 1] for s in selections]).mean(0))):
                    ladders[name] = profile_fit(grid, selected_log_profile(numerator, alternative, n, centre))
                for bound in fit['intervals']['95']:
                    if bound is not None:
                        j = int(np.searchsorted(grid, bound))
                        relevant.update([max(0, j-1), min(len(grid)-1, j)])
            relevant.discard(centre)
            for pos, obs in enumerate(ids):
                details = {}
                for j in sorted(relevant):
                    data = joined[pos, j]
                    d = profile_summary(data)
                    effective = float(np.exp(2*logsumexp(data)-logsumexp(2*data)))
                    details[str(float(grid[j]))] = dict(summary=d, raw_weight_ess=effective,
                        passing=bool(d['valid'] and passing(d['diagnostics']) and effective > 400))
                diagnostics.append(dict(observation=obs, nodes=details,
                    passing=all(d['passing'] for d in details.values())))
            overlap_pass = all(d['passing'] for d in diagnostics)
            ci = fit.get('intervals', {}).get('95', [None, None])
            bracket = all(v is not None for v in ci)
            compatible = bool(bracket and ci[0] <= .02 <= ci[1])
            mcse = numerical['combined_shear_mcse']
            numerical_small = bool(fit['valid'] and mcse is not None and mcse < .1*fit['curvature_se'])
            ladder_pass = bool(fit['valid'] and mcse is not None and all(
                v['valid'] and abs(v['estimate']-fit['estimate']) <= 2*max(mcse, 1e-8) for v in ladders.values()))
            entries.append(dict(steps=end, fit=fit, replica_fits=replica_fits,
                log_selected_profile=selected.tolist(), convergence_pass_count=convergence_count,
                observations=n, all_latent_and_local_ratio_checks_pass=convergence_count == n,
                profile_overlap_and_functional_checks_pass=overlap_pass,
                numerical=numerical, selection_ladders=ladders, selection_ladders_pass=ladder_pass,
                injected_shear_inside_exploratory_95_interval=compatible,
                recovery_accepted=bool(convergence_count == n and overlap_pass and numerical_small and ladder_pass and compatible),
                per_observation_profile_checks=diagnostics))
        prefix_shift = None
        if all(e['fit']['valid'] for e in entries):
            prefix_shift = entries[-1]['fit']['estimate']-entries[0]['fit']['estimate']
        prefix_pass = bool(prefix_shift is not None and entries[-1]['numerical']['combined_shear_mcse'] is not None
            and abs(prefix_shift) <= 2*max(entries[0]['numerical']['combined_shear_mcse'] or 0.,
                                         entries[-1]['numerical']['combined_shear_mcse'], 1e-8))
        entries[-1]['recovery_accepted'] &= prefix_pass
        result['cohorts'][str(float(limit))] = dict(count=n, prefixes=entries,
            prefix_shear_shift=prefix_shift, prefix_stability_pass=prefix_pass)
    save(root/'report.json', result)
    lines = ['# Bright-cohort MH and conditional shear recovery', '', result['limitation'], '',
        '| Measured cut | Steps | Convergence passing | Exploratory g1 | Conditional statistical SE | Numerical MCSE | Recovery accepted |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for cut, cohort in result['cohorts'].items():
        for e in cohort['prefixes']:
            f = e['fit']
            lines.append(f"| <{cut} | {e['steps']} | {e['convergence_pass_count']}/{cohort['count']} | "
                f"{f.get('estimate')} | {f.get('curvature_se')} | {e['numerical']['combined_shear_mcse']} | {e['recovery_accepted']} |")
    lines += ['', 'Selection normalization uses independent forward populations with the matching magnitude cut and radius>3 pixels.',
        'The two cut cohorts overlap; their fits are correlated. Failed objects remain included.',
        'Inspect report.json for independent replicas, 95% likelihood intervals, selection ladders, overlap and per-object diagnostics.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('BRIGHT_REPORT_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        report(args.root)
    except Exception as error:
        save(args.root/'report_failure.json', dict(status='incomplete_or_invalid', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
