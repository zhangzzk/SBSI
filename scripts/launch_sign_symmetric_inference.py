#!/usr/bin/env python
"""Submit the frozen comparison with at most two sampling and two selection GPUs."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'jobs.json'
    if output.exists():
        raise FileExistsError('job chain already recorded; inspect before resubmitting')
    jobs = dict(preflight=args.preflight)
    def submit(key, options, script, arguments):
        job = subprocess.check_output(['sbatch', '--parsable', *options, str(root / script),
                                       *arguments], text=True).strip().split(';')[0]
        jobs[key] = job
        output.write_text(json.dumps(jobs, indent=2) + '\n')
        print(key, job, flush=True)
        return job
    prerequisites = [f'--dependency=afterok:{args.preflight}', '--kill-on-invalid-dep=yes']
    gpu = ['--gres=gpu:1', '--exclude=met-cl-vis01,met-cl-vis02']
    for index in range(2):
        submit(f'c1_worker{index}', [*prerequisites, *gpu, '--job-name=signsym-c1',
            f'--output={root}/logs/c1_%j.out'], 'worker.sh',
            ['c1', '0', '0', '2028500001', '2033500001'])
    submit('selection', [*prerequisites, *gpu, '--array=0-7%2', '--job-name=signsym-selection',
        f'--output={root}/logs/selection_%A_%a.out'], 'pipeline.sh', ['selection'])
    cpu = ['--partition=cluster', '--gres=none', '--cpus-per-task=4', '--mem=64G', '--time=02:00:00']
    dependency = ':'.join(jobs[k] for k in ('c1_worker0', 'c1_worker1', 'selection'))
    submit('estimate_c1', [*cpu, f'--dependency=afterok:{dependency}', '--kill-on-invalid-dep=yes',
        '--job-name=signsym-estimate', f'--output={root}/logs/estimate_c1_%j.out'],
        'pipeline.sh', ['estimate-c1'])
    checks = []
    for options in ([f'--dependency=after:{jobs["c1_worker0"]}+2'],
                    [f'--dependency=after:{jobs["c1_worker1"]}+2'],
                    *([f'--begin=now+{hours}hours'] for hours in (2, 4, 6, 8, 12, 24))):
        job = subprocess.check_output(['sbatch', '--parsable', '--partition=cluster', '--gres=none',
            '--cpus-per-task=1', '--mem=2G', '--time=00:10:00', '--job-name=signsym-check',
            f'--output={root}/logs/check_%j.out', *options, str(root / 'pipeline.sh'), 'check'],
            text=True).strip().split(';')[0]
        checks.append(job)
    jobs['checks'] = checks
    output.write_text(json.dumps(jobs, indent=2) + '\n')
    print(json.dumps(jobs), flush=True)


if __name__ == '__main__':
    main()
