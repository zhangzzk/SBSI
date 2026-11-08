import json,os
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')
P=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches');D=P/'galsbi_dev_constant_mag25_20261103';O=P/'constant_component_accounting_20261104'
summary=json.loads((D/'summary.json').read_text());model=[];sim=[];name='mag25_radius_gt0.60'
for c in range(1000,1140):
 with np.load(D/f'sums/case{c}.npz') as a:
  k=list(a['cuts']).index(name);model.append(a['model'][:,:,k,:,:5].sum(-2));sim.append(a['sim'][:,k,:,:3].sum(-2))
model,sim=np.array(model),np.array(sim)
W=np.random.default_rng(20260925).multinomial(140,np.full(140,1/140),size=10000)
def response(x,col):return (x[...,0,col]/x[...,0,0]-x[...,1,col]/x[...,1,0])/.04
rs=response(sim.sum(0),1);bs=response(np.einsum('bc,c...->b...',W,sim),1)
out={'model':'original GalSBI-dev, older16-cut base guard; later refinement selected unchanged weights','cases':[1000,1139],'cut':'own-leg measured radius>3 pixels AND MAG_AUTO<25','shear':'whole scene g1=+/-0.02, g2=0','bootstrap':{'replicates':10000,'seed':20260925,'unit':'case, paired across all component sums; own-leg selected mass recomputed'},'R_sim_total':{'value':float(rs),'se':float(bs.std(ddof=1))},'flags':{},'limitations':['Component contributions are not independent component calibration errors: constant-shear measurements supply only the total true response.','Emulator contribution is the actual Mobius-induced shift averaged on flow-generated outputs with the same selection/weights, not a standalone emulator forward model.','Case errors condition on the model and latent realizations; retraining, model-selection and independent finite-draw uncertainty are excluded.']}
for w,flag in enumerate(['C8','true flags']):
 x=model[:,w];point=x.sum(0);boot=np.einsum('bc,c...->b...',W,x)
 rf,re=response(point,1),response(point,3);bf,be=response(boot,1),response(boot,3)
 rj=rf+re;bj=bf+be
 row={}
 for key,p,b in [('R_flow',rf,bf),('R_emulator_contribution',re,be),('R_joint',rj,bj),('joint_m_percent',100*(rs/rj-1),100*(bs/bj-1)),('flow_without_emulator_vs_total_m_percent',100*(rs/rf-1),100*(bs/bf-1)),('emulator_contribution_without_flow_vs_total_m_percent',100*(rs/re-1),100*(bs/be-1)),('R_joint_minus_sim',rj-rs,bj-bs)]:
  row[key]={'value':float(p),'se':float(b.std(ddof=1))}
 row['component_bootstrap_covariance']=float(np.cov(bf,be,ddof=1)[0,1])
 for suffix,key in [('', 'joint_m_percent'),('/flow only (no emulator)','flow_without_emulator_vs_total_m_percent')]:
  old=summary['results'][flag+'/'+name+suffix]
  assert abs(row[key]['value']-old['m_percent'])<1e-10
  assert abs(row[key]['se']-old['case_se_pp'])<1e-10
 out['flags'][flag]=row
out['job_id']=os.environ['SLURM_JOB_ID'];out['saved_summary_replay']='both point values and case errors match within 1e-10'
(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
