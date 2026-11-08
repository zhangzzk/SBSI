"""Audit actual Sersic grids and preserve timing/concurrency evidence."""
import os,json,re
from pathlib import Path
import numpy as np
import pandas as pd
assert os.environ.get('SLURM_JOB_ID')
here=Path(__file__).resolve().parent
base=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim')
root=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/here.name
m=json.loads((here/'manifest.json').read_text());grid=np.geomspace(.3,6.2,122)
rows=[]
for case in m['cases']:
 for leg in ['0.02','-0.02']:
  old=pd.read_feather(base/f'constant_g002/sims/gals{case}_{leg}.feather',columns=['sersic_n'])['sersic_n'].to_numpy()
  new=pd.read_feather(base/f'constant_axis2_20261105/sims/gals{case}_{leg}.feather',columns=['sersic_n'])['sersic_n'].to_numpy()
  np.testing.assert_array_equal(new,old)
  u=np.unique(new);distance=np.abs(u[:,None]-grid[None,:]).min(1)
  assert float(distance.max())<1e-14
  f=base/f'constant_axis2_20261105/sims/case{case}_{leg}/real0/catalogues/input/gals_info_tile180.0_-0.5.feather'
  rendered=None
  if f.exists():
   v=np.unique(pd.read_feather(f,columns=['sersic_n_input'])['sersic_n_input'])
   err=float(np.abs(v[:,None]-grid[None,:]).min(1).max());assert err<1e-14
   rendered={'unique_values':len(v),'grid_max_abs_error':err}
  rows.append(dict(case=case,leg=leg,rows=len(new),unique_values=len(u),grid_max_abs_error=float(distance.max()),unchanged_from_g1=True,rendered=rendered))
  print('AUDITED',case,leg,len(u),flush=True)
timings={}
for f in (root/'logs').glob('pilot_*.log'):
 s=f.read_text(errors='replace');timings[f.name]=[(int(a),float(b)*60) for a,b in re.findall(r'Task (\d): .*?finished in ([\d.]+) h',s)]
r={'grid':{'values':122,'min':.3,'max':6.2,'spacing':'logarithmic'},'catalogues':rows,'pilot_stage_minutes':timings,
   'historical_constant_job':{'job':'17643819_0','cases_per_job':10,'mpi_workers':10,'total_minutes':277.45,'render_detect_minutes':5792/60,'shape_minutes':10848/60},
   'historical_campaign':'28 constant jobs x10 MPI workers; up to500 workers including training family',
   'current_campaign':'initial8 single-worker pilot legs, then72 remaining legs; original throttle8 caused throughput bottleneck'}
(root/'runtime_audit.json').write_text(json.dumps(r,indent=2)+'\n');print('DONE',len(rows),'catalogues')
