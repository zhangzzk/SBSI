"""Bounded scheduled snapshot; never changes scientific settings or retries jobs."""
import os,json,subprocess,datetime
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
jobs=json.loads((O/'jobs.json').read_text())
ids=','.join(str(v) for v in jobs.values())
def command(args):
    p=subprocess.run(args,text=True,capture_output=True,timeout=30)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':jobs,
        'queue':command(['squeue','-j',ids,'-o','%.20i %.30j %.12T %.12M %R']),
        'accounting':command(['sacct','-j',ids,'--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS','--parsable2']),
        'outputs':{},'log_tails':{}}
for name in ['control_report.json','fit_report.json','fit_details_report.json','fit_parameters_report.json','report.json']:
    p=O/name
    report['outputs'][name]=json.loads(p.read_text()) if p.exists() else {'pending':True}
for p in sorted(O.glob('*.log')):
    if p.stat().st_size>0:
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-2500));report['log_tails'][p.name]=f.read().decode(errors='replace')
target=O/f"status_{os.environ['SLURM_JOB_ID']}.json"
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
