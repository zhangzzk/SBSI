"""One bounded read-only status check, never release the paused broad array."""
import datetime,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_conditioned_flow_20261106')
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=45)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
j=json.loads((HERE/'jobs.json').read_text());ids=','.join(j['jobs'].values())
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'queue':command(['squeue','-j',ids,'-o','%.24i %.12T %.10M %R']),
     'accounting':command(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed','-n','-X']),
     'feature_cases':len(list((ROOT/'features').glob('case*.json'))),
     'constant_tasks':len(list((ROOT/'constant').glob('*/*/case*.json')))}
for arm in ['control','neighbour']:
    out[arm]={}
    for name in ['history','guard_history','completed']:
        p=ROOT/arm/(name+'.json')
        if p.exists():
            value=json.loads(p.read_text());out[arm][name]=value[-1] if isinstance(value,list) else value
out['report_exists']=(ROOT/'report.json').exists()
(ROOT/'scheduled_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
