#!/bin/bash
#SBATCH --job-name=sbsi-model-audit-check
#SBATCH --partition=cip
#SBATCH --nodelist=cip-cl-nv01
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=logs/v36_model_audit_check_%j.out
set -euo pipefail
python - <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

root = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1')
status = subprocess.run(['sacct', '-X', '-P', '-n', '-j', '16689736,16689747,16689756,16689757',
                         '--format=JobID,State,ExitCode,Elapsed'], capture_output=True, text=True, check=True)
report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'scheduler': status.stdout,
          'models': {}, 'scope': 'Scheduled read-only status and completed-summary check; two development scenes, not acceptance.'}
for name, job in [('unbounded',16689736),('closecrowd',16689736),('closecrowd_emulator265',16689747)]:
    path = root/f'new_{name}_{job}_summary.json'
    worker = root/f'new_{name}_{job}'
    entry = {'worker_complete': (worker/'completed.json').exists(), 'summary_complete': path.exists()}
    if path.exists():
        d = json.loads(path.read_text())
        entry['cases'] = d['cases']
        entry['full_parent'] = d['results']['all']['models']
    else:
        entry['case_files'] = [p.name for p in worker.glob('case*.json')]
    report['models'][name] = entry
with (root/'scheduled_status_2h.json').open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report, indent=2))
PY
