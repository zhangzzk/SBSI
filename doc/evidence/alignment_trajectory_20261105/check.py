"""One bounded status audit; never relaunches or modifies science settings."""
import datetime
import json
from pathlib import Path
import subprocess
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_trajectory_20261105')
JOBS=['17758686','17758694','17758743','17758753','17758754']
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':JOBS,
   'queue':command(['squeue','-j',','.join(JOBS),'-o','%.18i %.24j %.10T %.10M %R']),
   'accounting':command(['sacct','-j',','.join(JOBS),'--format=JobID,State,ExitCode,Elapsed','-n']),
   'logs':{},'completed':{},'reports':{phase:(ROOT/f'{phase}_report.json').exists() for phase in ['native','final']}}
for p in (ROOT/'logs').glob('*.log'):r['logs'][p.name]='\n'.join(p.read_text(errors='replace').splitlines()[-5:])
for name in ['fs2','galsbi','galsbi_fs2rate','fs2_eval','galsbi_eval','galsbi_fs2rate_eval']:
    p=ROOT/name/'completed.json';r['completed'][name]=json.loads(p.read_text()) if p.exists() else None
(ROOT/'scheduled_check.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
