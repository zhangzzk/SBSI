"""One bounded scheduler/output check; does not wake the chat."""
import datetime
import json
import subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_target9_20261106')
j=json.loads((HERE/'jobs.json').read_text())
ids=','.join(j[k] for k in ['pilot','remaining','pilot_report','report'])
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=45)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'queue':command(['squeue','-j',ids,'-o','%.20i %.12T %.10M %R']),
        'accounting':command(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed','-X','-n']),
        'cases_per_axis':{str(a):len(list((ROOT/f'axis{a}').glob('case*.npz'))) for a in [1,2]},
        'full_report_exists':(ROOT/'full_report.json').exists()}
(ROOT/'scheduled_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
