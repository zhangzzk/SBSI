"""Compare R11 and R22 rather than only the standalone half-trace."""
import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
with np.load(O.parent/'true_selection_flags_20261105/flow_sums.npz') as z:
 w=np.vstack([np.ones(40),np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)])
 a={k:np.einsum('bc,c...->b...',w,z[k]) for k in ['base','normal','count','cross_e','cross_w']}
mass=a['base'][...,0]/a['count'][:,None];mean=a['base'][...,1:]/a['base'][...,0,None]
cross=a['cross_e']-mean[:,:,None,:,None]*a['cross_w'][...,None,:]
R=np.einsum('bksoi,bij->bksoj',cross,np.linalg.inv(a['normal']))/mass[:,:,None,None,None]
def ent(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
out={}
for k,cut in enumerate(['soft','hard']):
 out[cut]={}
 for i,j in [(0,0),(0,1),(1,0),(1,1)]:
  v=R[:,k,: ,i,j]
  out[cut][f'R{i+1}{j+1}']={name:ent(q) for name,q in [('simulation',v[:,0]),('model_generated_selection',v[:,1]),('model_true_selection',v[:,2]),('error_generated_selection',v[:,0]-v[:,1]),('error_true_selection',v[:,0]-v[:,2])]}
(O/'flow_components.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
