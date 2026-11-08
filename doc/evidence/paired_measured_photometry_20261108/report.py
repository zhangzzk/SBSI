"""Paired selected-calibration comparison with actual photometry/new membership."""
import hashlib,json,os,runpy
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
GEN=H.parent/'selected_seed_calibration_20261108'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def response(data,weights,part='total'):
 q=np.einsum('bc,clkd->blkd',weights,data)
 if q.shape[-1]==3:g=q[...,1:3]/q[...,:1]
 else:
  x=q[...,1:3] if part=='flow' else q[...,3:5] if part=='transport' else q[...,1:3]+q[...,3:5]
  g=x/q[...,:1]
 return np.stack([(g[:,0]-g[:,1])/.04,(g[:,2]-g[:,3])/.04],-1)
def diag(m):return np.stack([m[...,0,0],m[...,1,1]],-1)
def main():
 assert os.environ.get('SLURM_JOB_ID')
 cfg=json.loads((H/'manifest.json').read_text());pins=json.loads((H/'source_pins.json').read_text())
 for p,d in pins.items():assert sha(p)==d,p
 pilot=json.loads((ROOT/'pilot_passed.json').read_text());assert len(pilot)==2
 for rec in pilot:assert max(rec['full_replay'][k] for k in ['old_full_response_max_abs','new_full_vs_update_max_abs'])<=1e-6
 previous=json.loads((GEN/'report.json').read_text());path=ROOT.parent/GEN.name/'case_sums.npz';assert sha(path)==previous['case_sums_sha256']
 with np.load(path) as z:sim=z['new'];generated=z['model'];cases=list(z['cases']);cuts=list(z['cuts'])
 np.testing.assert_array_equal(cases,cfg['cases']);np.testing.assert_array_equal(cuts,[c[0] for c in cfg['cuts']])
 model=[];old=[];receipts=[]
 for ci,case in enumerate(cases):
  mm=[];oo=[]
  for axis in [1,2]:
   p=ROOT/f'axis{axis}/case{case}.npz';r=json.loads(p.with_suffix('.json').read_text());assert r['sha256']==sha(p) and r['source_sha256']==sha(H/'evaluate.py');assert r['case']==case and r['axis']==axis and r['checkpoints']==cfg['checkpoints'];assert r['emulator_sha256']==cfg['emulator_sha256']
   for k in ['selected_masses_exact','matched_new_simulation_sums','half_bank_identity_passed']:assert r[k]
   with np.load(p) as z:
    np.testing.assert_array_equal(z['cuts'],cuts);np.testing.assert_allclose(z['sim'],sim[ci,(axis-1)*2:axis*2],rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(z['model'],z['old_model']+z['delta'],rtol=0,atol=0);np.testing.assert_array_equal(z['model'][...,0],np.broadcast_to(z['sim'][...,0],z['model'][...,0].shape));np.testing.assert_allclose(z['model'][0],(z['model'][1]+z['model'][2])/2,rtol=1e-12,atol=2e-7)
    mm.append(z['model']);oo.append(z['old_model'])
   receipts.append(r)
  model.append(np.concatenate(mm,axis=1));old.append(np.concatenate(oo,axis=1))
 model=np.array(model);old=np.array(old)
 out={'definition':'New independent per-leg paired-seed measurements. Actual photometry/new usable membership replaces generated photometry/predicted usability jointly. Same frozen conditional shapes and emulator. m=100*(Rmeasured/Rmodel-1), reused development evidence, no unique causal component attribution.','checkpoints':cfg['checkpoints'],'receipts':receipts,'pilot':pilot,'cohorts':{}}
 for cohort,indices in [('all_forty',list(range(40))),('original_four',[i for i,c in enumerate(cases) if c in cfg['pilot_cases']]),('additional_thirty_six',[i for i,c in enumerate(cases) if c not in cfg['pilot_cases']])]:
  n=len(indices);w=np.vstack([np.ones(n),np.random.default_rng(202611061020).multinomial(n,np.full(n,1/n),10000),1-np.eye(n)])
  rs=response(sim[indices],w);rg=response(generated[indices],w);ro=response(model[indices,0],w);rf=response(model[indices,0],w,'flow');rt=response(model[indices,0],w,'transport')
  np.testing.assert_allclose(ro,rf+rt,rtol=1e-11,atol=2e-12)
  mg=100*(diag(rs)/diag(rg)-1);mo=100*(diag(rs)/diag(ro)-1)
  mb=[100*(diag(rs)/diag(response(model[indices,b],w))-1) for b in [1,2]]
  oldresponse=response(old[indices,0],w)
  def entry(x):
   jk=x[-n:];return {'value':x[0].tolist(),'jackknife_se':np.sqrt((n-1)/n*((jk-jk.mean(0))**2).sum(0)).tolist(),'bootstrap_ci95':np.quantile(x[1:10001],[.025,.975],axis=0).tolist()}
  rows={}
  for i,cut in enumerate(cuts):
   row={'measured_matrix':entry(rs[:,i]),'oracle_matrix':entry(ro[:,i]),'oracle_flow_matrix':entry(rf[:,i]),'oracle_transport_matrix':entry(rt[:,i]),'generated_matrix':entry(rg[:,i]),'generated_m_percent':entry(mg[:,i]),'oracle_m_percent':entry(mo[:,i]),'oracle_minus_generated_pp':entry(mo[:,i]-mg[:,i]),'oracle_mean_m_percent':entry(mo[:,i].mean(-1)),'oracle_m1_minus_m2_pp':entry(mo[:,i,0]-mo[:,i,1]),'oracle_half_bank_difference_pp':entry(mb[0][:,i]-mb[1][:,i]),'membership_update_R':entry(ro[:,i]-oldresponse[:,i])}
   if cohort=='all_forty':
    for key in ['value','jackknife_se']:np.testing.assert_allclose(row['generated_m_percent'][key],previous['cuts'][str(cut)]['new_m_percent'][key],rtol=1e-9,atol=1e-10)
   rows[str(cut)]=row
  rms_o=np.sqrt((mo**2).mean((1,2)));rms_g=np.sqrt((mg**2).mean((1,2)))
  contrast={}
  for a,b in [(3,6),(4,7),(5,8)]:
   delta_o=mo[:,b]-mo[:,a];delta_g=mg[:,b]-mg[:,a]
   contrast[str(cuts[b])+' minus '+str(cuts[a])]={'oracle_delta_m_pp':entry(delta_o),'generated_delta_m_pp':entry(delta_g),'oracle_minus_generated_delta_pp':entry(delta_o-delta_g)}
  out['cohorts'][cohort]={'cases':[int(cases[i]) for i in indices],'cuts':rows,'radius_contrasts':contrast,'joint_nine_cut_rms_percent':{'generated':entry(rms_g),'oracle':entry(rms_o),'oracle_minus_generated_pp':entry(rms_o-rms_g)}}
 np.savez(ROOT/'case_sums.npz',sim=sim,oracle=model,old_oracle=old,generated=generated,cases=cases,cuts=cuts);out['case_sums_sha256']=sha(ROOT/'case_sums.npz');(H/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print('CENTRAL',json.dumps(out['cohorts']['all_forty']['cuts']['mag25_radius_gt0.60']),flush=True);print('RADIUS',json.dumps(out['cohorts']['all_forty']['radius_contrasts']),flush=True);print('RMS',json.dumps(out['cohorts']['all_forty']['joint_nine_cut_rms_percent']),flush=True)
 runpy.run_path(str(H/'audit.py'),run_name='__main__')
 print('DONE_MATCHED_INPUT_REPORT',flush=True)
if __name__=='__main__':main()
