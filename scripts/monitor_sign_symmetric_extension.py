"""Save a bounded Slurm status receipt; keep only one future two-hour check."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('run this monitor through Slurm')
    root = args.root.resolve()
    jobs = json.loads((root/'jobs.json').read_text())
    report = dict(time=datetime.now(timezone.utc).isoformat(), jobs=jobs, stages={})
    failed = False
    for stage in ('c1', 'c2'):
        folder = root/stage
        labels = [line.split()[0] for line in (folder/'units.txt').read_text().splitlines()]
        failures = [label for label in labels if (folder/'claims'/label/'failed').exists()]
        failed |= bool(failures)
        report['stages'][stage] = dict(planned=len(labels),
            done=sum((folder/'claims'/label/'done').exists() for label in labels), failed=failures,
            estimate_written=(folder/'estimate.json').exists())
    accounting = subprocess.check_output(['sacct', '-X', '-P', '-n', '-j',
        ','.join(str(value) for value in jobs.values()), '-o', 'JobID,State,ExitCode,Elapsed'], text=True)
    report['accounting'] = accounting
    failed |= any(state in accounting for state in ('FAILED', 'CANCELLED', 'TIMEOUT',
                                                   'OUT_OF_MEMORY', 'NODE_FAIL'))
    complete = (root/'c2/estimate.json').exists()
    report['status'] = 'failed' if failed else 'result_written' if complete else 'in_progress'
    # These are on-disk receipts, not chat notifications. Completion/failure
    # terminates the monitor without leaving a series of future timer jobs.
    if not (failed or complete):
        next_job = subprocess.check_output(['sbatch', '--parsable', '--begin=now+2hours',
            '--partition=cluster', '--cpus-per-task=1', '--mem=2G', '--time=00:10:00',
            '--job-name=signsym1m-monitor', '--output='+str(root/'logs/monitor_%j.out'),
            str(root/'monitor.sh')], text=True).strip().split(';')[0]
        report['next_check'] = next_job
    target = root/'logs'/f'monitor_{os.environ["SLURM_JOB_ID"]}.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    (root/'monitor_latest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
