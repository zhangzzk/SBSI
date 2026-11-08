"""Both axes/nine cuts against paired measurements and the frozen latest model."""
import hashlib,json,os,runpy
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name;D=ROOT.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def response(data,w,part='total'):
 q=np.einsum('bc,clkd->blkd',w,data)
 if q.shape[-1]==3:g=q[...,1:3]/q[...,:1]
 else:g=(q[...,1:3] if part=='flow' else q[...,3:5] if part=='transport' else q[...,1:3]+q[...,3:5])/q[...,:1]
 return np.stack([(g[:,0]-g[:,1])/.04,(g[:,2]-g[:,3])/.04],-1)
def diag(x):return np.stack([x[...,0,0],x[...,1,1]],-1)
def main():
 assert os.environ.get('SLURM_JOB_ID')
 cfg=json.loads((H/'manifest.json').read_text());gate=json.loads((ROOT/'pilot_passed.json').read_text());assert gate['manifest_sha256']==sha(H/'manifest.json')
 for p,d in json.loads((H/'source_pins.json').read_text()).items():assert sha(p)==d,p
 assert gate['records'][0]['legacy_replay']['legacy_response_max_abs']<=1e-6
 previous=json.loads((H.parent/'selected_seed_calibration_20261108/report.json').read_text())
 p=D/'selected_seed_calibration_20261108/case_sums.npz';assert sha(p)==cfg['measurement_sums_sha256']==previous['case_sums_sha256']
 with np.load(p) as z:sim=z['new'];latest=z['model'];cases=list(z['cases']);cuts=list(z['cuts'])
 np.testing.assert_array_equal(cases,cfg['cases']);np.testing.assert_array_equal(cuts,[c[0] for c in cfg['cuts']])
 model=[];receipts=[];legacy=[]
 for ci,case in enumerate(cases):
  legs=[]
  for axis in [1,2]:
   p=ROOT/f'axis{axis}/case{case}.npz';r=json.loads(p.with_suffix('.json').read_text());assert sha(p)==r['data_sha256'] and r['source_sha256']==sha(H/'evaluate.py');assert r['checkpoint_sha256']==cfg['checkpoint']['sha256'] and r['emulator_sha256']==cfg['emulator_sha256'] and r['classifier_sha256']==cfg['classifier_sha256'];assert r['case']==case and r['axis']==axis and r['half_bank_identity_passed'] and r['measurement_sums_sha256']==cfg['measurement_sums_sha256']
   with np.load(p) as z:
    np.testing.assert_array_equal(z['cuts'],cuts);np.testing.assert_array_equal(z['sim'],sim[ci,(axis-1)*2:axis*2]);np.testing.assert_allclose(z['model'][0],(z['model'][1]+z['model'][2])/2,rtol=1e-12,atol=2e-7);legs.append(z['model'])
   receipts.append(r)
  model.append(np.concatenate(legs,axis=1))
  p=D/f'alignment_matched_constant_20261105/joint_cartesian/case{case}.npz'
  with np.load(p) as z:k=list(z['cuts']).index('mag25_radius_gt0.60');legacy.append(z['model'][0,:,k].sum(-2)[...,:5])
 model=np.array(model);legacy=np.array(legacy);n=40
 w=np.vstack([np.ones(n),np.random.default_rng(cfg['bootstrap_seed']).multinomial(n,np.full(n,1/n),10000),1-np.eye(n)])
 def entry(x):
  jk=x[-n:];return {'value':x[0].tolist(),'jackknife_se':np.sqrt((n-1)/n*((jk-jk.mean(0))**2).sum(0)).tolist(),'bootstrap_ci95':np.quantile(x[1:10001],[.025,.975],axis=0).tolist()}
 rs=response(sim,w);rj=response(model[:,0],w);rl=response(latest,w);rf=response(model[:,0],w,'flow');rt=response(model[:,0],w,'transport');np.testing.assert_allclose(rj,rf+rt,rtol=1e-11,atol=2e-12)
 mj=100*(diag(rs)/diag(rj)-1);ml=100*(diag(rs)/diag(rl)-1)
 half=[100*(diag(rs)/diag(response(model[:,b],w))-1) for b in [1,2]]
 out={'definition':'Frozen older joint Cartesian vs latest weighted factorized flow, same40 scenes, portable64 antithetic draws, predicted usability/generated selection, new paired-seed measured reference. Independent per-leg normalization, m_i=100*(Rmeasured_ii/Rmodel_ii-1). Paired whole-scene errors. Development evidence, not independent acceptance.','checkpoint':cfg['checkpoint'],'source_sha256':sha(__file__),'measurement_matching_totals':{k:sum(m[k] for row in previous['case_details'] for m in row['matching']) for k in ['cross_without_shape','shape_without_cross']},'receipts':receipts,'pilot':gate,'cuts':{}}
 for ci,cut in enumerate(cuts):
  row={'measured_matrix':entry(rs[:,ci]),'joint_matrix':entry(rj[:,ci]),'joint_flow_matrix':entry(rf[:,ci]),'joint_transport_matrix':entry(rt[:,ci]),'latest_matrix':entry(rl[:,ci]),'joint_m_percent':entry(mj[:,ci]),'latest_m_percent':entry(ml[:,ci]),'joint_minus_latest_pp':entry(mj[:,ci]-ml[:,ci]),'joint_mean_m_percent':entry(mj[:,ci].mean(-1)),'joint_m1_minus_m2_pp':entry(mj[:,ci,0]-mj[:,ci,1]),'joint_half_bank_difference_pp':entry(half[0][:,ci]-half[1][:,ci])}
  for k in ['value','jackknife_se']:np.testing.assert_allclose(row['latest_m_percent'][k],previous['cuts'][str(cut)]['new_m_percent'][k],rtol=1e-9,atol=1e-10)
  out['cuts'][str(cut)]=row
 rmsj=np.sqrt((mj*mj).mean((1,2)));rmsl=np.sqrt((ml*ml).mean((1,2)));out['joint_nine_cut_rms_percent']={'joint':entry(rmsj),'latest':entry(rmsl),'joint_minus_latest_pp':entry(rmsj-rmsl)}
 out['per_axis_nine_cut_rms_percent']={'joint':entry(np.sqrt((mj*mj).mean(1))),'latest':entry(np.sqrt((ml*ml).mean(1)))}
 out['radius_contrasts']={}
 for a,b in [(3,6),(4,7),(5,8)]:
  out['radius_contrasts'][str(cuts[b])+' minus '+str(cuts[a])]={'joint_delta_m_pp':entry(mj[:,b]-mj[:,a]),'latest_delta_m_pp':entry(ml[:,b]-ml[:,a]),'joint_minus_latest_delta_pp':entry((mj[:,b]-mj[:,a])-(ml[:,b]-ml[:,a]))}
 # Quantify the integration-stream change in the previously reported g1 central result.
 q=np.einsum('bc,clq->blq',w,legacy);g=(q[...,1:3]+q[...,3:5])/q[...,:1];rlegacy=(g[:,0,0]-g[:,1,0])/.04;mlegacy=100*(rs[:,4,0,0]/rlegacy-1)
 ref=json.loads((H.parent/'joint_flow_recheck_20261108/report.json').read_text())['arms']['joint_cartesian']['new_m1_percent'];check=entry(mlegacy)
 for k in ['value','jackknife_se']:np.testing.assert_allclose(check[k],ref[k],rtol=1e-9,atol=1e-10)
 out['central_g1_stream_comparison']={'historical_cuda_m1_percent':check,'portable_m1_percent':entry(mj[:,4,0]),'portable_minus_historical_pp':entry(mj[:,4,0]-mlegacy)}
 np.savez(ROOT/'case_sums.npz',sim=sim,joint=model,latest=latest,cases=cases,cuts=cuts);out['case_sums_sha256']=sha(ROOT/'case_sums.npz');(H/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print('CENTRAL',json.dumps(out['cuts']['mag25_radius_gt0.60']),flush=True);print('RMS',json.dumps(out['joint_nine_cut_rms_percent']),flush=True)
 runpy.run_path(str(H/'audit.py'),run_name='__main__');runpy.run_path(str(H/'summarize.py'),run_name='__main__');print('DONE_JOINT9_REPORT',flush=True)
if __name__=='__main__':main()
