"""Matched per-leg selected calibration against unchanged generated model outputs."""
import hashlib,json,os,runpy
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
MODEL_HOME=H.parent/'conditional_cut_replication_20261107'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sums(g,photo,cuts):
 valid=np.isfinite(g).all(-1)&(np.square(g).sum(-1)<1);out=np.zeros((4,9,3))
 for leg in range(4):
  for ci,c in enumerate(cuts):
   k=valid[:,leg]&(photo[:,leg,0]>c[1][1])&(photo[:,leg,1]>10**(-.4*(c[2]-30)))
   out[leg,ci]=[k.sum(),*g[k,leg].sum(0)]
 return out

def response(x):
 mean=x[...,1:]/x[...,:1]
 return np.stack([(mean[:,0]-mean[:,1])/.04,(mean[:,2]-mean[:,3])/.04],axis=-1)
def main():
 assert os.environ.get('SLURM_JOB_ID');cfg=json.loads((ROOT/'manifest.json').read_text());pins=json.loads((ROOT/'source_pins.json').read_text());modelcfg=json.loads((MODEL_HOME/'manifest.json').read_text());assert sha(MODEL_HOME/'manifest.json')==cfg['model_manifest_sha256']
 for p,v in pins.items():assert sha(p)==v
 assert cfg['cases']==modelcfg['cases'] and cfg['cuts']==modelcfg['cuts'][:9]
 assert cfg['model_checkpoints']==modelcfg['checkpoints']
 for rec in modelcfg['checkpoints'].values():assert sha(rec['path'])==rec['sha256']
 old_sums=[];new_sums=[];models=[];records=[];model_records=[]
 for rec in cfg['records']:
  case=rec['case'];p=ROOT/f'case{case}.npz';assert sha(p)==rec['data_sha256']
  with np.load(p) as z:ids=z['ids'];photo=z['photo'];old=z['old'];new=z['base'];reuse=z['reuse'];extra=z['extra'];candidate=z['candidate'];baseline=z['old_sums']
  np.testing.assert_allclose(sums(old,photo,cfg['cuts']),baseline,rtol=1e-12,atol=1e-8)
  coverage=reuse.astype(int);fit_errors=0
  for ci in range(rec['chunks']):
   p=ROOT/'chunks'/f'case{case}_chunk{ci}.npz';r=json.loads(p.with_suffix('.json').read_text());assert sha(p)==r['data_sha256'] and r['source_pins']==pins;assert r['case']==case and r['chunk']==ci and r['repeat_verified']
   assert r['start']==ci*cfg['chunk_size'] and r['stop']==min(len(extra),(ci+1)*cfg['chunk_size']);index=extra[r['start']:r['stop']];assert r['n']==len(index) and r['attempted']==int(candidate[index].sum())
   with np.load(p) as z:
    np.testing.assert_array_equal(z['indices'],index);np.testing.assert_array_equal(z['ids'],ids[index]);np.testing.assert_array_equal(z['attempted'],candidate[index]);assert not np.isfinite(z['g'][~candidate[index]]).any();new[index]=z['g']
   coverage[index]+=1;fit_errors+=len(r['errors'])
  assert (coverage==1).all();new_sum=sums(new,photo,cfg['cuts']);old_sums.append(baseline);new_sums.append(new_sum)
  valid_old=np.isfinite(old).all(-1)&(np.square(old).sum(-1)<1);valid_new=np.isfinite(new).all(-1)&(np.square(new).sum(-1)<1)
  records.append({'case':case,'union_candidates':len(ids),'reused_ids':int(reuse.sum()),'extra_fit_errors':fit_errors,'newly_valid_in_loose_domain':int((valid_new&~valid_old&candidate).sum()),'newly_invalid_in_loose_domain':int((~valid_new&valid_old&candidate).sum()),'old_selected_counts':baseline[...,0].tolist(),'new_selected_counts':new_sum[...,0].tolist(),'matching':rec['matching']})
  mr=[]
  for axis in [1,2]:
   original=case in modelcfg['pilot_cases'];root=Path(modelcfg['original_root']) if original else Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/conditional_cut_replication_20261107')
   p=root/'constant/weighted'/f'axis{axis}/case{case}.npz';r=json.loads(p.with_suffix('.json').read_text());assert sha(p)==r['data_sha256'];assert r['checkpoint_sha256']==modelcfg['checkpoints']['weighted']['sha256']
   expected_source=modelcfg['reference_evaluator_sha256'] if original else sha(MODEL_HOME/'evaluate.py');assert r['source_sha256']==expected_source
   assert r['arm']=='weighted' and r['case']==case and r['axis']==axis
   assert r['emulator_sha256']=='cd32781db7edd2cab576a2ef7478d722df2984788bc83a7fe12caa6adf19fb43' and r['classifier_sha256']=='ad5c05d2dcd9af9d711b18b532cd681ac5de6df6350bab288cdec94a3019a3c6'
   assert r['simulation_replay_passed'] and r['true_selection_mass_exact'] and r['pooled_emulator_roundoff_check_passed'] and r['reference_score_replay_passed']
   with np.load(p) as z:
    np.testing.assert_allclose(baseline[(axis-1)*2:axis*2],z['sim'].sum(-2)[:,:9],rtol=1e-12,atol=1e-8)
    np.testing.assert_array_equal(z['cuts'][:9],[c[0] for c in cfg['cuts']]);mr.append(z['model'].sum(-2)[0,:,:9])
   model_records.append({'case':case,'axis':axis,'path':str(p),'sha256':r['data_sha256']})
  models.append(np.concatenate(mr))
  print('SELECTED_REDUCED',case,flush=True)
 old_sums=np.array(old_sums);new_sums=np.array(new_sums);models=np.array(models);n=len(records)
 weights=np.vstack([np.ones(n),np.random.default_rng(202611061020).multinomial(n,np.full(n,1/n),10000),1-np.eye(n)])
 rs_old=response(np.einsum('bc,clkd->blkd',weights,old_sums));rs_new=response(np.einsum('bc,clkd->blkd',weights,new_sums));qm=np.einsum('bc,clkd->blkd',weights,models);rm=response(np.concatenate([qm[...,:1],qm[...,1:3]+qm[...,3:5]],axis=-1))
 def entry(v):
  jk=v[-n:];return {'value':v[0].tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0)).tolist(),'bootstrap_se':v[1:10001].std(0,ddof=1).tolist(),'bootstrap_ci95':np.quantile(v[1:10001],[.025,.975],axis=0).tolist()}
 def bias(r):return 100*(np.stack([r[...,0,0]/rm[...,0,0],r[...,1,1]/rm[...,1,1]],-1)-1)
 old_m=bias(rs_old);new_m=bias(rs_new);old_report=json.loads((MODEL_HOME/'report.json').read_text());out={'definition':'m_i=100*(Rmeasured_ii/Rmodel_ii-1). Independently normalized per-leg measured and generated selection, radius>2.8/3.0/3.2 pixels crossed with magnitude<24.9/25/25.1. Same40 parent scenes, fixed weighted model with predicted usability/generated photometry. Paired-seed default shape fits; no common-four conditioning. Reused development evidence, not independent acceptance.','checkpoints':cfg['model_checkpoints'],'case_details':records,'model_receipts':model_records,'cuts':{}}
 for ci,c in enumerate(cfg['cuts']):
  cut=c[0];row={'old_measured_matrix':entry(rs_old[:,ci]),'new_measured_matrix':entry(rs_new[:,ci]),'model_matrix':entry(rm[:,ci]),'old_m_percent':entry(old_m[:,ci]),'new_m_percent':entry(new_m[:,ci]),'change_m_pp':entry(new_m[:,ci]-old_m[:,ci]),'new_m1_minus_m2_pp':entry(new_m[:,ci,0]-new_m[:,ci,1]),'old_measured_diagonal_gap':entry(rs_old[:,ci,0,0]-rs_old[:,ci,1,1]),'change_measured_diagonal_gap':entry((rs_new[:,ci,0,0]-rs_new[:,ci,1,1])-(rs_old[:,ci,0,0]-rs_old[:,ci,1,1])),'new_measured_diagonal_gap':entry(rs_new[:,ci,0,0]-rs_new[:,ci,1,1]),'new_measured_offdiagonal_gap':entry(rs_new[:,ci,0,1]-rs_new[:,ci,1,0])}
  for axis in [1,2]:
   ref=old_report['cohorts']['all_forty']['axes'][str(axis)]['predicted_usability_generated_selection']['cuts'][cut]['weighted']['m_percent']
   np.testing.assert_allclose(row['old_m_percent']['value'][axis-1],ref['value'],rtol=1e-10,atol=1e-10)
   np.testing.assert_allclose(row['old_m_percent']['jackknife_se'][axis-1],ref['jackknife_se'],rtol=1e-9,atol=1e-10)
  out['cuts'][cut]=row
 out['joint_nine_cut_rms_m_percent']={'old':entry(np.sqrt(np.square(old_m).mean((1,2)))),'new':entry(np.sqrt(np.square(new_m).mean((1,2)))),'change':entry(np.sqrt(np.square(new_m).mean((1,2)))-np.sqrt(np.square(old_m).mean((1,2))))}
 np.savez(ROOT/'case_sums.npz',old=old_sums,new=new_sums,model=models,cases=cfg['cases'],cuts=[c[0] for c in cfg['cuts']]);out['case_sums_sha256']=sha(ROOT/'case_sums.npz');(H/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('CENTRAL',json.dumps(out['cuts']['mag25_radius_gt0.60']),flush=True);runpy.run_path(str(H/'audit.py'),run_name='__main__');print('DONE_SELECTED_CALIBRATION',flush=True)
if __name__=='__main__':main()
