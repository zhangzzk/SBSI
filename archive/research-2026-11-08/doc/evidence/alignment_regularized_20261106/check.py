"""One bounded scheduler/progress check for the matched continuation study."""
import datetime
import json
import subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106')
j=json.loads((HERE/'jobs.json').read_text());ids=','.join(j[k] for k in ['prepare','train','training_report','constant_pilot','constant_remaining','constant_report'])
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=45)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
m=json.loads((HERE/'manifest.json').read_text())
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue':command(['squeue','-j',ids,'-o','%.20i %.12T %.10M %R']),
        'accounting':command(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed','-X','-n']),
        'training_complete':{a['name']:(ROOT/a['name']/'completed.json').exists() for a in m['arms']},
        'evaluation_files':len(list((ROOT/'constant').glob('*/*/case*.npz'))),'full_report_exists':(ROOT/'full_report.json').exists()}
(ROOT/'scheduled_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
