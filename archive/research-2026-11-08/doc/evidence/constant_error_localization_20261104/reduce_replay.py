"""Paired case errors for the four-case matched-draw intervention."""
from pathlib import Path
import json
import numpy as np
O=Path(__file__).resolve().parent;cases=[1000,1040,1080,1120]
a=np.stack([np.load(O/f'replay_case{c}.npz')['sums'] for c in cases])
w=np.random.default_rng(20260925).multinomial(4,np.full(4,.25),size=10000)
out={'cases':cases,'draws':64,'bootstrap':'10000 whole-case paired samples; seed20260925; conditional on model and draw bank','rows':{}}
for co,cohort in enumerate(['own_selection','common_selection']):
 for bn,bs in [('all',[0,1,2,3]),('<24',[0]),('24-25.5',[1]),('25.5-27.5',[2])]:
  x=a[:,:,:,co][:,:,:,bs].sum(3) # case variant leg q
  def calc(agg):
   z=agg(x);r=(z[...,0,2]/z[...,0,0]-z[...,1,2]/z[...,1,0])/.04
   flow=(z[...,0,1]/z[...,0,0]-z[...,1,1]/z[...,1,0])/.04
   return np.concatenate([r,r[...,1:]-r[...,:-1],(r[...,-1]-r[...,0])[...,None],flow[...,:1],flow[...,-1:]],-1)
  p=calc(lambda x:x.sum(0));se=calc(lambda x:np.einsum('bc,c...->b...',w,x)).std(0,ddof=1)
  names=['emulator_original','emulator_measured_aux_same_draw_selection','emulator_measured_aux_and_selection','emulator_all_measured','change_aux_only','change_selection','change_shape','change_total','flow_original','simulation_response']
  out['rows'][f'{cohort}/{bn}']={q:[float(v),float(s)] for q,v,s in zip(names,p,se)}
(O/'replay_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
# Independent CPU measured-input construction must agree with GPU replay's all-measured branch.
with np.load(O/'measured_constant_sums.npz') as z:
 cpu=z['sums'];cc=list(z['cases'])
for j,c in enumerate(cases):
 np.testing.assert_allclose(a[j,3],cpu[cc.index(c),...,:3],rtol=2e-7,atol=1e-7)
print('Independent measured-input CPU/GPU comparison passed for all four cases.',flush=True)
