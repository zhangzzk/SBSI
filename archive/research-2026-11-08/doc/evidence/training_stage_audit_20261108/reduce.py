"""Re-reference saved training-stage predictions to paired-seed measurements."""
from pathlib import Path
import hashlib,json,os
import numpy as np
H=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def response(data,w):
 q=np.einsum('bc,clkd->blkd',w,data)
 v=(q[...,1:3]+q[...,3:5])/q[...,:1] if q.shape[-1]==5 else q[...,1:3]/q[...,:1]
 return np.stack([(v[:,0]-v[:,1])/.04,(v[:,2]-v[:,3])/.04],-1)
def main():
 assert os.environ.get('SLURM_JOB_ID')
 p=D/'selected_seed_calibration_20261108/case_sums.npz'
 previous=json.loads((H.parent/'selected_seed_calibration_20261108/report.json').read_text());assert sha(p)==previous['case_sums_sha256']
 with np.load(p) as z:cases=z['cases'].tolist();cuts=z['cuts'].tolist();sim=z['new'];old=z['old'];latest=z['model']
 cfg=json.loads((H.parent/'alignment_target9_20261106/manifest.json').read_text());pin=cfg['checkpoints']['cartesian'];assert sha(pin['path'])==pin['sha256']
 auxiliary=json.loads((H.parent/'joint_cartesian_ninecut_20261108/manifest.json').read_text())
 records=[]
 def read_arm(name,path_template,cohort,checkpoint=None,source=None):
  arrays=[];identity=None
  for case in cohort:
   legs=[];ci=cases.index(case)
   for axis in (1,2):
    p=Path(path_template.format(case=case,axis=axis));r=json.loads(p.with_suffix('.json').read_text())
    assert r['checkpoint_sha256']==(checkpoint or r['checkpoint_sha256'])
    assert r['emulator_sha256']==auxiliary['emulator_sha256'] and r['classifier_sha256']==auxiliary['classifier_sha256']
    assert r['latent_source_sha256']==sha(H.parent/'alignment_target9_20261106/portable_draws.py')
    if source:assert r['source_sha256']==sha(source)
    if identity is None:identity=r['checkpoint_sha256']
    assert identity==r['checkpoint_sha256']
    if 'data_sha256' in r:assert sha(p)==r['data_sha256']
    with np.load(p) as z:
     assert int(z['case'])==case and int(z['axis'])==axis
     ix=[z['cuts'].tolist().index(k) for k in cuts]
     # Normal predicted usability/generated selection is mode 0; sum true-size bins.
     legs.append(z['model'][0][:,ix].sum(-2))
     np.testing.assert_allclose(z['sim'][:,ix].sum(-2),old[ci,(axis-1)*2:axis*2],rtol=1e-12,atol=1e-8)
    records.append(dict(path=str(p),sha256=sha(p),receipt_sha256=sha(p.with_suffix('.json'))))
   arrays.append(np.concatenate(legs,axis=0))
  return np.array(arrays),identity
 base,basepin=read_arm('nll',str(D/'alignment_target9_20261106/axis{axis}/case{case}.npz'),cases,pin['sha256'],H.parent/'alignment_target9_20261106/evaluate.py')
 def comparison(cohort,arms,identities):
  n=len(cohort);ix=[cases.index(x) for x in cohort]
  weights=np.vstack([np.ones(n),np.random.default_rng(202611082101).multinomial(n,np.full(n,1/n),10000),1-np.eye(n)])
  rs=response(sim[ix],weights);allarms={**arms,'latest':latest[ix]}
  def entry(x):
   jk=x[-n:];return dict(value=x[0].tolist(),jackknife_se=np.sqrt((n-1)/n*((jk-jk.mean(0))**2).sum(0)).tolist(),bootstrap_ci95=np.quantile(x[1:10001],[.025,.975],axis=0).tolist())
  matrices={k:response(v,weights) for k,v in allarms.items()}
  bs={k:100*(np.diagonal(rs,axis1=-2,axis2=-1)/np.diagonal(v,axis1=-2,axis2=-1)-1) for k,v in matrices.items()}
  checks=0
  # Independent direct scalar reduction at full cohort and every leave-one-out.
  for name,data in allarms.items():
   for wi in [0]+list(range(len(weights)-n,len(weights))):
    w=weights[wi]
    for cut in range(9):
     for axis in range(2):
      means=[]
      for leg in [2*axis,2*axis+1]:
       den=sum(float(a)*float(row[leg,cut,0]) for a,row in zip(w,data))
       num=sum(float(a)*(float(row[leg,cut,1+axis])+float(row[leg,cut,3+axis])) for a,row in zip(w,data))
       means.append(num/den)
      np.testing.assert_allclose((means[0]-means[1])/.04,matrices[name][wi,cut,axis,axis],rtol=1e-10,atol=1e-11);checks+=1
  out={}
  rms0=np.sqrt((bs['latest']**2).mean((1,2)))
  for name,b in bs.items():
   out[name]=dict(matrix=entry(matrices[name]),m_percent=entry(b),delta_latest_pp=entry(b-bs['latest']),rms_percent=entry(np.sqrt((b*b).mean((1,2)))),rms_minus_latest_pp=entry(np.sqrt((b*b).mean((1,2)))-rms0))
  return dict(cases=cohort,arms=out,checkpoint_identities=identities,scalar_diagonal_checks=checks)
 full=comparison(cases,{'initial_nll':base},{'initial_nll':basepin})
 for k,cut in enumerate(cuts):
  for field in ['value','jackknife_se']:
   np.testing.assert_allclose(full['arms']['latest']['m_percent'][field][k],previous['cuts'][cut]['new_m_percent'][field],rtol=1e-9,atol=1e-10)
 pilotcases=[1000,1040,1080,1120];arms={'initial_nll':base[[cases.index(x) for x in pilotcases]]};pins={'initial_nll':basepin}
 for name in ['control_nll','neighbour_nll','control_nine','neighbour_nine']:
  arms[name],pins[name]=read_arm(name,str(D/f'neighbour_conditioned_flow_20261106/constant/{name}/axis{{axis}}/case{{case}}.npz'),pilotcases,source=H.parent/'neighbour_conditioned_flow_20261106/evaluate.py')
 pilot=comparison(pilotcases,arms,pins)
 result=dict(full40=full,pilot4=pilot,cuts=cuts,source_sha256=sha(__file__),measurement_sums_sha256=sha(D/'selected_seed_calibration_20261108/case_sums.npz'),cache_receipts=records,limitation='Reused development scenes; historical caches without original data hashes are hashed now, with original source/model receipts and measured-sum replay. Stage sequence necessity is not established by unrelated checkpoints.')
 write(H/'report.json',result)
 lines=['# Training-stage audit with paired-seed measurements','','## Full 40-scene comparison','','| Recipe | Central m1 (%) | Central m2 (%) | Nine-cut RMS (%) |','|---|---:|---:|---:|']
 def row(name,r):
  b=r['m_percent'];a=r['rms_percent'];return f'| {name} | {b["value"][4][0]:+.5f} ±{b["jackknife_se"][4][0]:.5f} | {b["value"][4][1]:+.5f} ±{b["jackknife_se"][4][1]:.5f} | {a["value"]:.5f} ±{a["jackknife_se"]:.5f} |'
 for name,r in full['arms'].items():lines.append(row(name,r))
 lines+=['','## Matched four-scene stage comparison','','The following rows use only the same four scenes; do not compare their absolute values to the full40 table.','','| Recipe | Central m1 (%) | Central m2 (%) | Nine-cut RMS (%) |','|---|---:|---:|---:|']
 for name,r in pilot['arms'].items():lines.append(row(name,r))
 lines+=['','Errors are whole-scene jackknife standard errors. Normal generated selection/predicted usability, independent per-leg normalization, m=100*(Rmeasured/Rmodel−1).',
 f'Independent scalar diagonal checks: {full["scalar_diagonal_checks"]+pilot["scalar_diagonal_checks"]}. Latest-model/new-measurement replay passes. No model evaluation or retraining.',
 '[Conventions](../../CONVENTIONS.md#6-response-estimators).',result['limitation']]
 (H/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
