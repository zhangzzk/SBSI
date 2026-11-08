"""One bounded scheduled diagnostic-stage status check."""
import datetime
import json
import subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106')
j=json.loads((HERE/'jobs.json').read_text())
keys=['evaluate','standalone_report','fourstate_pilot','fourstate_remaining','fourstate_pilot_report','fourstate_report','release']
if 'fourstate_pilot_retry' in j:keys.append('fourstate_pilot_retry')
for key in ['fourstate_remaining_v100','fourstate_remaining_rtx',
            'measured_transport_pilot','measured_transport_pilot_report',
            'measured_transport_remaining','measured_transport_report',
            'conditional_transport_pilot','conditional_transport_remaining',
            'conditional_transport_baseline_report','conditional_transport_nine_report',
            'neighbour_input_synthesis']:
    if key in j:keys.append(key)
ids=','.join(j[k] for k in keys)
def command(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=45)
    return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'queue':command(['squeue','-j',ids,'-o','%.22i %.12T %.10M %R']),
     'accounting':command(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed','-n','-X']),
     'standalone_completed':[p.parent.name for p in ROOT.glob('*_eval/completed.json')],
     'fourstate_cases':{a:len(list((ROOT/'fourstate'/a).glob('case*.json'))) for a in ['baseline','nine_weak']},
     'reports':{f:(ROOT/f).exists() for f in ['constant_decomposition.json','standalone_report.json','fourstate_pilot_report.json','fourstate_report.json','measured_transport_pilot_report.json','measured_transport_report.json','conditional_transport_baseline_report.json','conditional_transport_nine_weak_report.json','neighbour_input_synthesis.json','release_receipt.json']}}
(ROOT/'scheduled_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
