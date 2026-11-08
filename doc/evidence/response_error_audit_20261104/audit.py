import json, os
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')
P=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
OUT=P/'response_error_audit_20261104'
A=P/'target_response_uncertainty_20261104'
report=json.loads((A/'report.json').read_text()); data=np.load(A/'case_sums.npz')
def response(s):
 n,mass=s[...,0],s[...,1];mu=s[...,2:4]/mass[...,None]
 gram=s[...,4:8].reshape(*s.shape[:-1],2,2);cross=s[...,8:12].reshape(*s.shape[:-1],2,2)
 r=(cross-mu[..., :,None]*s[...,None,12:14])/(mass/n)[...,None,None]@np.linalg.inv(gram)
 return np.trace(r,axis1=-2,axis2=-1)/2
train,val=data['train'],data['validation']; pool=np.concatenate([train,val]);total=pool.sum(0)
rng=np.random.default_rng(2026110407); differences=[]
for start in range(0,20000,1000):
 idx=np.argpartition(rng.random((1000,200)),40,axis=1)[:,:40]
 vs=pool[idx].sum(1);ts=total-vs
 differences.extend((response(vs)-response(ts)).tolist())
differences=np.asarray(differences);observed=float(response(val.sum(0))-response(train.sum(0)))
r={'job_id':os.environ['SLURM_JOB_ID'],'random_split_null':{'replicates':len(differences),'seed':2026110407,'observed_difference':observed,'null_sd':float(differences.std(ddof=1)),'two_sided_p':float((1+np.sum(abs(differences)>=abs(observed)))/(len(differences)+1))},'simulation':{}}
for split,x in [('train',train),('validation',val)]:
 theta=response(x.sum(0));jk=response(x.sum(0)-x);se=np.sqrt((len(x)-1)/len(x)*np.square(jk-jk.mean()).sum())
 infl=(len(x)-1)*(theta-jk);infl-=infl.mean()
 r['simulation'][split]={'bootstrap_se':report[split]['case_bootstrap_se'],'jackknife_se':float(se),'max_leave_one_out_shift':float(abs(jk-theta).max()),'largest_case_variance_share':float((infl**2).max()/(infl**2).sum())}
S=P/'galsbi_target_guard_repeat_20261103/seed1'
j=json.loads((S/'check_final.json').read_text());d=np.load(S/'check_final_sums.npz')
i=next(k for k,c in enumerate(j['cuts']) if c['radius_min']==3 and c['magnitude_max']==25)
sim=d['sim'][:,i,[0,3]].sum(1); model=d['model_lambda_0.01'][:,i,[0,3]].sum(1)
m=100*(sim.sum()/model.sum()-1);n=len(sim)
jk=100*((sim.sum()-sim)/(model.sum()-model)-1)
se=np.sqrt((n-1)/n*np.square(jk-jk.mean()).sum());infl=(n-1)*(m-jk);infl-=infl.mean()
w=np.random.default_rng(20260925).multinomial(n,np.full(n,1/n),size=10000)
boot=100*((w@sim)/(w@model)-1)
assert abs(m-j['flows']['lambda_0.01']['m_eq'][i])<1e-10
assert abs(boot.std(ddof=1)-j['flows']['lambda_0.01']['m_eq_se'][i])<1e-10
r['fixed_model_validation']={'m_percent':float(m),'bootstrap_se_pp':float(boot.std(ddof=1)),'jackknife_se_pp':float(se),'max_leave_one_out_shift_pp':float(abs(jk-m).max()),'largest_case_variance_share':float((infl**2).max()/(infl**2).sum()),'bootstrap_ci95_pp':np.quantile(boot,[.025,.975]).tolist(),'method':'resample saved per-case simulation and model sums together; model, baseline mean and shear design fixed as in original report'}
scene_root=P.parent/'sbsi/galsbi_resim/training/scenes'
seeds=[tuple(json.loads((scene_root/f'scene{c}.json').read_text())['seed']) for c in range(200)]
r['scene_seed_audit']={'cases':200,'unique_scene_seeds':len(set(seeds)),'seed_pattern_matches_case':all(seed==(20261101,c) for c,seed in enumerate(seeds))}
r['limitations']=['Conditional on the fixed simulation population prescription.','Model error calculation does not repeat training or model selection.','Random split test assumes exchangeable cases and tests raw measured responses, not model residuals.','Bootstrap and jackknife reuse the same observed cases; agreement does not prove coverage for unseen tails or unrecorded common errors.']
(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
