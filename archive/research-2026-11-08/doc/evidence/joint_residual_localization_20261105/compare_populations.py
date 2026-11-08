"""Compare locations of errors; not an additive cross-family budget."""
import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
s=[];m=[]
for c in [1000,1040,1080,1120]:
 with np.load(O/f'joint_case{c}.npz') as z:s.append(z['sim'][:,3]);m.append(z['model'][:,3])
s=np.array(s);m=np.array(m)
w=np.vstack([np.ones(4),np.random.default_rng(20260925).multinomial(4,np.full(4,.25),size=10000)])
ts=np.einsum('bc,clq->blq',w,s.sum(2));tm=np.einsum('bc,clq->blq',w,m.sum(2))
idx=np.arange(100);sz=idx//20;cr=idx%5;core=np.isin(sz,[1,2])&(cr==0)
groups={'all':np.ones(100,bool),'size_015_04_crowd_lt01':core,'outside_core':~core,'size_015_04':np.isin(sz,[1,2]),'crowd_lt01':cr==0,'crowd_01_03':cr==1}
flow=json.loads((O/'flow_cell_report.json').read_text())['groups']
def ent(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
out={'definition':'Each group contributes to its own experiment global response error; do not add or subtract across families as an error budget','groups':{}}
for name,keep in groups.items():
 ss=np.einsum('bc,clq->blq',w,s[:,:,keep].sum(2));mm=np.einsum('bc,clq->blq',w,m[:,:,keep].sum(2))
 d=ss[...,1]/ts[...,0]-(mm[...,1]+mm[...,2])/tm[...,0];e=(d[:,0]-d[:,1])/.04
 out['groups'][name]={'joint_error_contribution':ent(e),'joint_simulation_mass_fraction':float((ss[0,:,0]/ts[0,:,0]).mean())}
 if name in flow:out['groups'][name]['flow_R11_error_contribution']=flow[name]['R11']['error_generated_selection']
# Flow complement with a paired case bootstrap, retaining covariance.
with np.load(O/'flow_sums.npz') as z:a={k:z[k] for k in ['base','normal','count']}
with np.load(O/'flow_cell_sums.npz') as z:c={k:z[k] for k in ['cross_e','cross_w']}
wf=np.vstack([np.ones(40),np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)])
b={k:np.einsum('bc,c...->b...',wf,v) for k,v in a.items()};mass=b['base'][:,1,0]/b['count'];mu=b['base'][:,1,1:]/b['base'][:,1,0,None]
ce=np.einsum('bc,c...->b...',wf,c['cross_e'][:,~core,1].sum(1));cw=np.einsum('bc,c...->b...',wf,c['cross_w'][:,~core,1].sum(1))
R=np.einsum('bsoi,bij->bsoj',ce-mu[:,None,:,None]*cw[:,:,None,:],np.linalg.inv(b['normal']))/mass[:,None,None,None]
out['groups']['outside_core']['flow_R11_error_contribution']=ent(R[:,0,0,0]-R[:,1,0,0])
(O/'population_comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
