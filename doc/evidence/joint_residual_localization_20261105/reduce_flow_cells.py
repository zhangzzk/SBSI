"""Globally normalized standalone error by fixed truth population."""
import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
with np.load(O/'flow_sums.npz') as z:a={k:z[k] for k in ['base','normal','count','cross_e','cross_w']}
with np.load(O/'flow_cell_sums.npz') as z:c={k:z[k] for k in ['base','cross_e','cross_w']}
w=np.vstack([np.ones(40),np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)])
b={k:np.einsum('bc,c...->b...',w,v) for k,v in a.items()}
mass=b['base'][:,1,0]/b['count'];mu=b['base'][:,1,1:]/b['base'][:,1,0,None];inv=np.linalg.inv(b['normal'])
def ent(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
idx=np.arange(100);sz=idx//20;cr=idx%5;mag=(idx//5)%4
groups={'all':np.ones(100,bool),'size_015_04_crowd_lt01':np.isin(sz,[1,2])&(cr==0),'size_015_04':np.isin(sz,[1,2])}
for i,label in enumerate(['lt015','015_025','025_04','04_07','ge07']):groups['size_'+label]=sz==i
for i,label in enumerate(['lt01','01_03','03_1','1_3','ge3']):groups['crowd_'+label]=cr==i
for i,label in enumerate(['lt24','24_255','255_275','ge275']):groups['mag_'+label]=mag==i
out={'definition':'Hard-cut response-error contribution with shared global baseline/normal; fixed truth cells; training-family 40 validation cases','groups':{}}
for name,keep in groups.items():
 ce=np.einsum('bc,c...->b...',w,c['cross_e'][:,keep,1].sum(1));cw=np.einsum('bc,c...->b...',w,c['cross_w'][:,keep,1].sum(1))
 base=np.einsum('bc,c...->b...',w,c['base'][:,keep,1].sum(1))
 R=np.einsum('bsoi,bij->bsoj',ce-mu[:,None,:,None]*cw[:,:,None,:],inv)/mass[:,None,None,None]
 out['groups'][name]={'baseline_mass_fraction':float(base[0,0]/b['base'][0,1,0])}
 for comp,v in [('R11',R[...,0,0]),('R22',R[...,1,1]),('halftrace',np.trace(R,axis1=-2,axis2=-1)/2)]:
  out['groups'][name][comp]={key:ent(x) for key,x in [('simulation',v[:,0]),('model_generated_selection',v[:,1]),('error_generated_selection',v[:,0]-v[:,1])]}
old=json.loads((O/'flow_components.json').read_text())['hard']
for comp in ['R11','R22']:
 for key in ['simulation','model_generated_selection','error_generated_selection']:
  np.testing.assert_allclose(out['groups']['all'][comp][key]['value'],old[comp][key]['value'],atol=2e-12,rtol=0)
(O/'flow_cell_report.json').write_text(json.dumps(out,indent=2)+'\n')
for name,g in out['groups'].items():print(name,g['baseline_mass_fraction'],g['R11']['error_generated_selection'],flush=True)
