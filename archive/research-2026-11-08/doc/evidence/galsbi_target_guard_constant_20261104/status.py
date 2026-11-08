import json,subprocess,datetime,os
from pathlib import Path
R=Path(__file__).resolve().parent
jobs=json.loads((R/'jobs.json').read_text())
state={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monitor_job':os.environ.get('SLURM_JOB_ID'),'case_receipts':sorted(p.name for p in (R/'sums').glob('case*.json')),'case_outputs':sorted(p.name for p in (R/'sums').glob('case*.npz'))}
state['accounting']=subprocess.check_output(['sacct','-j',','.join(jobs[k] for k in ['cip','inter','summary']),'--format=JobID,State,ExitCode,Elapsed','-n'],text=True)
state['logs']={p.name:p.read_text()[-4000:] for p in (R/'logs').glob('target-constant*.log')}
if (R/'comparison.json').exists():
 r=json.loads((R/'comparison.json').read_text());state['primary']=r['cuts']['C8/mag25_radius_gt0.60'];state['verified']=r['simulation_exact_replay'] and r['baseline_report_replay']
(R/'two_hour_status.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
