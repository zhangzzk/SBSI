"""Small saved-sum recheck: no predictions, catalogue scan or training."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches');OLD=H.parent/'alignment_matched_constant_20261105';NEW=H.parent/'selected_seed_calibration_20261108'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def response(s):return (s[:,0,1]/s[:,0,0]-s[:,1,1]/s[:,1,0])/.04
cfg=json.loads((OLD/'manifest.json').read_text());baseline=json.loads((OLD/'full_report.json').read_text());newrep=json.loads((NEW/'report.json').read_text());p=D/NEW.name/'case_sums.npz';assert sha(p)==newrep['case_sums_sha256']
with np.load(p) as z:
 np.testing.assert_array_equal(z['cases'],cfg['cases']);ci=list(z['cuts']).index('mag25_radius_gt0.60');oldsim=z['old'][:,:2,ci];newsim=z['new'][:,:2,ci];latest=z['model'][:,:2,ci]
w=np.vstack([np.ones(40),1-np.eye(40)])
def pool(s):return np.einsum('bc,clq->blq',w,s)
def total(m):return np.concatenate([m[...,:1],m[...,1:3]+m[...,3:5]],-1)
def entry(v):return {'value':float(v[0]),'jackknife_se':float(np.sqrt(39/40*((v[1:]-v[1:].mean())**2).sum()))}
rl=response(pool(total(latest)));rn=response(pool(newsim));ro=response(pool(oldsim));latest_m=100*(rn/rl-1);np.testing.assert_allclose(latest_m[0],newrep['cuts']['mag25_radius_gt0.60']['new_m_percent']['value'][0],atol=1e-10)
result={'definition':'Same40 scenes, central radius>3 pixels/magnitude<25, predicted usability/generated selection, first shear component only. m=100*(Rmeasured/Rmodel-1), paired whole-scene jackknife. Old cached predictions reused; no axis2/nine-target-cut acceptance claim.','latest_m1_percent':entry(latest_m),'arms':{}}
for arm in cfg['arms']:
 assert sha(cfg['checkpoints'][arm]['path'])==cfg['checkpoints'][arm]['sha256'];rows=[];receipts=[]
 for i,case in enumerate(cfg['cases']):
  p=D/OLD.name/arm/f'case{case}.npz';r=json.loads(p.with_suffix('.json').read_text());assert r['checkpoint_sha256']==cfg['checkpoints'][arm]['sha256'];assert r['source_sha256']==sha(OLD/'evaluate.py');assert r['arm']==arm and r['case']==case
  assert r['emulator_sha256']=='cd32781db7edd2cab576a2ef7478d722df2984788bc83a7fe12caa6adf19fb43' and r['classifier_sha256']=='ad5c05d2dcd9af9d711b18b532cd681ac5de6df6350bab288cdec94a3019a3c6'
  with np.load(p) as z:
   k=list(z['cuts']).index('mag25_radius_gt0.60');np.testing.assert_allclose(z['sim'][:,k].sum(-2),oldsim[i],rtol=1e-12,atol=1e-8);rows.append(z['model'][0,:,k].sum(-2)[...,:5])
  receipts.append({'case':case,'path':str(p),'current_sha256':sha(p)})
 rm=response(pool(total(np.array(rows))));oldm=100*(ro/rm-1);newm=100*(rn/rm-1);ref=baseline['arms'][arm]['predicted_usability_generated_selection']['mag25_radius_gt0.60'];np.testing.assert_allclose(oldm[0],ref['m_percent']['value'],atol=1e-10);np.testing.assert_allclose(rm[0],ref['model_response']['value'][0],atol=1e-12)
 # Independently replay every historical per-case result, not only the pooled value.
 ss=oldsim;mm=total(np.array(rows));percase=100*((ss[:,0,1]/ss[:,0,0]-ss[:,1,1]/ss[:,1,0])/(mm[:,0,1]/mm[:,0,0]-mm[:,1,1]/mm[:,1,0])-1);np.testing.assert_allclose(percase,ref['per_case_m_percent'],atol=1e-10)
 result['arms'][arm]={'checkpoint':cfg['checkpoints'][arm],'old_m1_percent':entry(oldm),'new_m1_percent':entry(newm),'new_minus_latest_pp':entry(newm-latest_m),'model_R11':entry(rm),'receipts':receipts}
(H/'report.json').write_text(json.dumps(result,indent=2)+'\n')
for a,r in result['arms'].items():print(a,r['new_m1_percent'],'differencefromlatest',r['new_minus_latest_pp'])
print('latest',result['latest_m1_percent'])
