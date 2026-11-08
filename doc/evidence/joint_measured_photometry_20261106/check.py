import datetime,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_measured_photometry_20261106')
j=json.loads((HERE/'jobs.json').read_text())['jobs'];ids=','.join(j.values())
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=45);return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'queue':command(['squeue','-j',ids,'-o','%.24i %.12T %.10M %R']),
    'accounting':command(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed','-n','-X']),
    'completed_tasks':len(list(ROOT.glob('axis*/case*.json')))}
for phase in ['four','full']:
    p=ROOT/f'{phase}_report.json'
    if p.exists():out[phase]=json.loads(p.read_text())
(ROOT/'scheduled_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
