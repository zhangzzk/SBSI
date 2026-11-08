#!/usr/bin/env bash
#SBATCH --job-name=sbsi_independent_check
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/independent_joint_validation_20260919_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/independent_joint_validation_20260919_v1/logs/%x_%j.err
set -euo pipefail
squeue -u "$USER" -o '%.18i %.32j %.8T %.10M %.24R'
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python - <<'PY'
from pathlib import Path
import json
import subprocess
data = Path('/project/ls-gruen/users/zekang.zhang')
root = data/'sbsi_caches/independent_joint_validation_20260919_v1'
images = data/'blendemu_runs/joint_independent_validation_20260919_v1'
jobs = json.loads((root/'jobs.json').read_text())
names = ('generation', 'preparation', 'binding', 'joint_validation', 'report')
subprocess.run(['sacct', '-j', ','.join(str(jobs[n]) for n in names),
                '--format=JobID,State,Elapsed,ExitCode'], check=True)
for directory, prefix, key in ((images/'logs', 'blendemu_joint_independent', 'generation'),
        (root/'logs', 'sbsi_independent_parent', 'preparation'),
        (root/'logs', 'sbsi_independent_bind', 'binding'),
        (root/'logs', 'sbsi_independent_joint', 'joint_validation'),
        (root/'logs', 'sbsi_independent_report', 'report')):
    for path in sorted(directory.glob(f'{prefix}_{jobs[key]}*.out')):
        print('LOG', path, flush=True)
        print('\n'.join(path.read_text().splitlines()[-4:]), flush=True)
    for path in sorted(directory.glob(f'{prefix}_{jobs[key]}*.err')):
        text = path.read_text()
        if text:
            print('STDERR', path, '\n'.join(text.splitlines()[-5:]), flush=True)
print('MEASUREMENT_RECEIPTS', len(list(images.glob('case*/measurement_manifest.json'))), flush=True)
for path in sorted((root/'results').glob('receipt_*.json')):
    print('RESULT_RECEIPT', path.name, path.read_text(), flush=True)
for name in ('scene22000_cache_warm.out', 'scene22000_cache_warm.err',
             'scene22000_cache_hold.out', 'scene22000_cache_hold.err'):
    path = root/'logs'/name
    if path.exists():
        print('CACHE_OPERATION', name, '\n'.join(path.read_text().splitlines()[-5:]), flush=True)
PY
