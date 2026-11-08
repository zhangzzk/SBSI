"""Compare the predefined initializer arms, including cross-arm chain agreement."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from check_galsbi_continuous_prior import save
from report_galsbi_sampler_runs import profile_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root/'report'
    output.mkdir(parents=True, exist_ok=True)
    reports, profiles, initial, timings = {}, {}, {}, {}
    for arm in ('prior-bank', 'measurement'):
        runs = [args.root/'pilot'/arm/f'seed{seed}' for seed in (2026100141, 2026100142)]
        subprocess.run([sys.executable, 'scripts/report_galsbi_sampler_runs.py', *map(str, runs),
                        '--output', str(output/arm), '--prefixes', '8192', '16384'], check=True)
        reports[arm] = json.loads((output/arm/'report.json').read_text())
        if not reports[arm]['compatible_independent_runs'] or not all(
                r.get('numerical_checks_pass') for r in reports[arm]['runs']):
            save(output/'comparison.json', dict(status='incomplete_or_failed_replay', arms=reports))
            return
        profiles[arm], initial[arm], timings[arm] = [], [], []
        for run in runs:
            with np.load(run/'local_profiles.npz') as saved:
                profiles[arm].append({key: saved[key] for key in saved.files})
            with np.load(run/'initial_state.npz') as saved:
                initial[arm].append(dict(log_kernel=saved['log_kernel'].reshape(4, 4).tolist(),
                    log_prior=saved['log_prior'].reshape(4, 4).tolist(),
                    physical_summaries=saved['summaries'][:, 0].reshape(4, 4, 9).tolist()))
            timings[arm].append(dict(initialization=json.loads((run/'initialization.json').read_text())['elapsed_seconds'],
                through_sampling=json.loads((run/'progress.json').read_text())['elapsed_seconds']))
    comparison = dict(status='completed_diagnostic', initialization=initial, timings_seconds=timings,
        arm_all_checks_pass={arm: value['all_checks_pass'] for arm, value in reports.items()},
        observations=[], limitation='Four observations and two paired seeds only; no shear estimate or automatic promotion. '
        'The initializer package changes centring and candidate ranking; this does not isolate those effects. '
        'Matched seeds can correlate arm errors; no independent-error significance test is applied.')
    lines = ['# Measurement-informed initialization comparison', '', comparison['limitation'], '',
             '| Observation | Prior-bank Rhat / min ESS | Measurement Rhat / min ESS |',
             '| --- | --- | --- |']
    for i in range(4):
        item = dict(observation=i, profiles={})
        row = []
        for arm in reports:
            result = reports[arm]['prefixes'][-1]['observations'][i]
            item[arm] = dict(max_rhat=result['max_rhat'], min_bulk_ess=result['min_bulk_ess'],
                             checks_pass=result['checks_pass'])
            row.append(f"{result['max_rhat']:.4f} / {result['min_bulk_ess']:.1f}"
                       if result['max_rhat'] is not None and result['min_bulk_ess'] is not None else 'unavailable')
        for shear in (.018, .022):
            key = f'observation_{i}_g1_{shear:g}'
            arrays = {arm: np.concatenate([p[key] for p in profiles[arm]], axis=1) for arm in profiles}
            combined = np.concatenate(list(arrays.values()), axis=1)
            item['profiles'][str(shear)] = dict(
                arms={arm: profile_summary(value) for arm, value in arrays.items()},
                combined=profile_summary(combined))
        comparison['observations'].append(item)
        lines.append(f'| {i} | {row[0]} | {row[1]} |')
    save(output/'comparison.json', comparison)
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('INITIALIZATION_COMPARISON', comparison['status'], comparison['arm_all_checks_pass'], flush=True)


if __name__ == '__main__':
    main()
