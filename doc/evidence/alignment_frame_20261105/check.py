"""Bounded read-only scheduler/status receipt; does not relaunch anything."""
import datetime
import json
from pathlib import Path
import subprocess
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_frame_20261105')
JOBS=['17756866','17757916','17757938','17757925','17758019','17758135']
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':JOBS,
        'queue':command(['squeue','-j',','.join(JOBS),'-o','%.18i %.24j %.10T %.10M %R']),
        'accounting':command(['sacct','-j',','.join(JOBS),'--format=JobID,State,ExitCode,Elapsed','-n']),
        'logs':{},'completed':{},'comparison_exists':(ROOT/'comparison.json').exists()}
for p in (ROOT/'logs').glob('*.log'):
    result['logs'][p.name]='\n'.join(p.read_text(errors='replace').splitlines()[-5:])
for name in ['cartesian','frame','cartesian_eval','frame_eval']:
    p=ROOT/name/'completed.json'
    result['completed'][name]=json.loads(p.read_text()) if p.exists() else None
(ROOT/'scheduled_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
