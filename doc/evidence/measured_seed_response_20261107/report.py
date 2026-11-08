"""Verify prepared population, or report focused common-seed response."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def matrix(s):
 # s ends with (leg, mass/e1/e2).
 means=s[...,1:]/s[...,:1]
 return np.stack([(means[...,0,:]-means[...,1,:])/.04,(means[...,2,:]-means[...,3,:])/.04],axis=-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();m=json.loads((ROOT/'manifest.json').read_text());old_sums=[];paired=[];details=[]
 for rec in m['records']:
  path=ROOT/f"case{rec['case']}.npz";assert sha(path)==rec['data_sha256']
  with np.load(path) as z:ids=z['ids'];old=z['old']
  sums=np.column_stack([np.full(4,len(ids)),old.sum(0)]);old_sums.append(sums)
  if a.prepare:continue
  new=np.full_like(old,np.nan);flags=np.full(old.shape[:2],-999,dtype=int);coverage=np.zeros(len(ids),dtype=int);errors=0
  for ci in range(rec['chunks']):
   q=ROOT/'chunks'/f"case{rec['case']}_chunk{ci}.npz";r=json.loads(q.with_suffix('.json').read_text());assert sha(q)==r['data_sha256'];assert r['case']==rec['case'] and r['chunk']==ci and r['repeat_verified'];lo,hi=r['start'],r['stop']
   with np.load(q) as z:np.testing.assert_array_equal(z['ids'],ids[lo:hi]);new[lo:hi]=z['g'];flags[lo:hi]=z['flags']
   coverage[lo:hi]+=1;errors+=len(r['errors'])
  assert (coverage==1).all();valid=np.isfinite(new).all(-1)&(np.square(new).sum(-1)<1);keep=valid.all(1)
  paired.append(np.stack([np.column_stack([np.full(4,keep.sum()),v[keep].sum(0)]) for v in [old,new]]))
  details.append({'case':rec['case'],'original_n':len(ids),'common_valid_n':int(keep.sum()),'valid_per_leg':valid.sum(0).tolist(),'recorded_nonzero_flags':int(((flags!=0)&(flags!=-999)).sum()),'unrecorded_flags':int((flags==-999).sum()),'exceptions_or_missing_shapes':errors})
 original=matrix(np.sum(old_sums,axis=0));ref=json.loads((H.parent/'measured_response_symmetry_20261107/membership_report.json').read_text())['cuts']['mag25_radius_gt0.60']['common_four']['matrix']['value'];np.testing.assert_allclose(original,ref,atol=1e-12,rtol=1e-12)
 if a.prepare:
  out={'prepared_files':len(m['records']),'total_common_ids':sum(r['n'] for r in m['records']),'original_matrix_replayed':original.tolist(),'all_data_hashes_verified':True}
  (H/'prepare_verification.json').write_text(json.dumps(out,indent=2)+'\n');print('PREPARE_VERIFIED',out,flush=True);return
 paired=np.array(paired);n=len(paired);weights=np.vstack([np.ones(n),1-np.eye(n)]);s=np.einsum('bc,cmlk->bmlk',weights,paired);r=matrix(s)
 def stat(x):return {'value':x[0].tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(x[1:]-x[1:].mean(0)).sum(0)).tolist()}
 out={'definition':'Original and newly paired-seed default fits on exactly the same originally common-selected IDs that remain valid in all new legs. Original photometry fixed. This is measurement-noise localization on reused scenes, not a new model or independent acceptance.','cases':details,'original_full_common_matrix':original.tolist(),'results':{}}
 for name,q in [('original_on_new_valid_ids',r[:,0]),('paired_seed_default',r[:,1]),('new_minus_original',r[:,1]-r[:,0])]:out['results'][name]={'matrix':stat(q),'diagonal_gap':stat(q[:,0,0]-q[:,1,1]),'offdiagonal_gap':stat(q[:,0,1]-q[:,1,0])}
 np.savez(ROOT/'case_sums.npz',sums=paired,cases=[d['case'] for d in details]);(H/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out['results']),flush=True);print('DONE_SEED_RESPONSE_REPORT',flush=True)
if __name__=='__main__':main()
