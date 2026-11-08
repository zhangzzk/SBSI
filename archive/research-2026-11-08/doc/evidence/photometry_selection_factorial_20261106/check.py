"""One bounded status check; never release the held broad evaluation."""
import datetime
import json
import subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_selection_factorial_20261106')
j=json.loads((HERE/'jobs.json').read_text())
ids=','.join(j[k] for k in ['pilot','pilot_report','remaining_a40','remaining_v100','remaining_rtx','report','broad_array_held'])
bin_jobs=HERE.parent/'photometry_response_bins_20261106/jobs.json'
if bin_jobs.exists():
    bj=json.loads(bin_jobs.read_text())
    ids+=','+','.join(bj[k] for k in ['pilot','pilot_report','remaining_a40','remaining_v100','remaining_rtx','report'] if k in bj)
fixed_jobs=HERE.parent/'photometry_fixed_cell_20261106/jobs.json'
if fixed_jobs.exists():
    fj=json.loads(fixed_jobs.read_text())
    ids+=','+','.join(fj[k] for k in ['pilot','pilot_report','remaining_a40','remaining_v100','remaining_rtx','report','selection_pilot','selection_report'] if k in fj)
information_jobs=HERE.parent/'selection_information_20261106/jobs.json'
if information_jobs.exists():
    ij=json.loads(information_jobs.read_text())
    ids+=','+','.join(ij[k] for k in ['prepare','pilot','pilot_report','remaining','report'] if k in ij)
def command(args):
    result=subprocess.run(args,capture_output=True,text=True,timeout=45)
    return {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'queue':command(['squeue','-j',ids,'-o','%.24i %.12T %.10M %R']),
     'accounting':command(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed','-n','-X']),
     'completed_cases':[p.stem for p in ROOT.glob('case*.json')]}
for name in ['pilot_report','report']:
    path=ROOT/(name+'.json')
    if path.exists():
        d=json.loads(path.read_text());out[name]={'cases':d['cases'],'target':d['cuts']['mag25_radius_gt0.60']}
bins_root=ROOT.parent/'photometry_response_bins_20261106'
out['response_bins']={'completed_cases':[p.stem for p in bins_root.glob('case*.json')]}
for name in ['pilot_report','report']:
    path=bins_root/(name+'.json')
    if path.exists():out['response_bins'][name]=json.loads(path.read_text())
fixed_root=ROOT.parent/'photometry_fixed_cell_20261106'
out['fixed_cells']={'completed_cases':[p.stem for p in fixed_root.glob('case*.json')]}
for name in ['pilot_report','report','selection_pilot','selection']:
    path=fixed_root/(name+'.json')
    if path.exists():out['fixed_cells'][name]=json.loads(path.read_text())
information_root=ROOT.parent/'selection_information_20261106'
out['selection_information']={'completed_folds':[p.parent.name for p in information_root.glob('fold*/completed.json')]}
for name in ['pilot_report','report']:
    path=information_root/(name+'.json')
    if path.exists():out['selection_information'][name]=json.loads(path.read_text())
(ROOT/'scheduled_check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
