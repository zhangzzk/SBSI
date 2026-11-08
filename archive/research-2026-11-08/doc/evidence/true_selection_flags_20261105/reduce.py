"""Like-for-like response errors under predicted versus observed selection."""
import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_dev_constant_grid_20261104')
cases=[1000,1040,1080,1120]
a=np.stack([np.load(O/f'replay_case{c}.npz')['sums'] for c in cases])
# case, variant, leg, q
x=a[:,:,:,0].sum(3)
models={};pred=[]
for c in cases:
 with np.load(D/f'sums/case{c}.npz') as z:
  k=list(z['cuts']).index('mag25_radius_gt0.60');p=z['model'][0,:,k].sum(1)
  pred.append(p[:,[0,1,3]])
models['predicted_U_generated_cut']=np.array(pred)
models['true_U_generated_cut']=x[:,0]
models['true_U_true_cut']=x[:,4]
models['true_U_true_cut_measured_emulator_aux']=x[:,2]
sim=x[:,3,:,:2]
w=np.random.default_rng(20260925).multinomial(4,np.full(4,.25),size=10000)
weights=np.vstack([np.ones(4),w])
def response(num,den):return (num[:,0]/den[:,0]-num[:,1]/den[:,1])/.04
ss=np.einsum('bc,c...->b...',weights,sim);rs=response(ss[...,1],ss[...,0])
def entry(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
out={'cases':cases,'cut':'per-leg measured FLUX_RADIUS>3 pixels and MAG_AUTO<25','sign':'simulation minus model','bootstrap':'10000 whole-case paired resamples, seed20260925; fixed model and latent draws','R_sim':entry(rs),'rows':{}}
errors={}
for name,m in models.items():
 mm=np.einsum('bc,c...->b...',weights,m);rf=response(mm[...,1],mm[...,0]);re=response(mm[...,2],mm[...,0]);rm=rf+re;errors[name]=rs-rm
 out['rows'][name]={'R_flow':entry(rf),'R_emulator':entry(re),'R_joint':entry(rm),'error':entry(rs-rm),'m_percent':entry(100*(rs/rm-1))}
 if name=='true_U_true_cut':np.testing.assert_allclose(mm[...,0],ss[...,0],atol=1e-7,rtol=1e-12)
out['change_in_error_true_cut_minus_generated_cut']=entry(errors['true_U_true_cut']-errors['true_U_generated_cut'])
original=json.loads((O.parent/'absolute_response_errors_20261104/report.json').read_text())
out['historical_140_case_joint_rows']={k:v for k,v in original['rows'].items() if k.startswith('joint_')}
out['limitations']='Four-case diagnostic only; sampling error is conditional. Actual object selection does not truncate generated outputs to their selected conditional distribution. R_flow and R_emulator are contributions, not independently measured component errors.'
(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
