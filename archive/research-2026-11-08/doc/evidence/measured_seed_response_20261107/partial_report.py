"""Interim response from whole completed scenes, never selected finished chunks."""
import hashlib,json
from pathlib import Path
import numpy as np
from report import matrix
H=Path(__file__).resolve().parent;ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 snapshot=json.loads((H/'partial_snapshot_20261108.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());pins=json.loads((ROOT/'source_pins.json').read_text());sums=[];details=[];originals=[]
 for rec in m['records']:
  case=rec['case']
  if case not in snapshot['complete_cases']:continue
  path=ROOT/f'case{case}.npz';assert sha(path)==rec['data_sha256']
  with np.load(path) as z:ids=z['ids'];old=z['old']
  new=np.full_like(old,np.nan);flags=np.full(old.shape[:2],-999,dtype=int);coverage=np.zeros(len(ids),int);err=0
  for ci in range(rec['chunks']):
   q=ROOT/'chunks'/f'case{case}_chunk{ci}.npz';r=json.loads(q.with_suffix('.json').read_text());assert sha(q)==r['data_sha256'];assert r['source_pins']==pins and r['repeat_verified'];assert r['case']==case and r['chunk']==ci;lo,hi=r['start'],r['stop']
   with np.load(q) as z:np.testing.assert_array_equal(z['ids'],ids[lo:hi]);new[lo:hi]=z['g'];flags[lo:hi]=z['flags']
   coverage[lo:hi]+=1;err+=len(r['errors'])
  assert (coverage==1).all();valid=np.isfinite(new).all(-1)&(np.square(new).sum(-1)<1);keep=valid.all(1)
  sums.append(np.stack([np.column_stack([np.full(4,keep.sum()),v[keep].sum(0)]) for v in [old,new]]));originals.append(np.column_stack([np.full(4,len(ids)),old.sum(0)]))
  details.append({'case':case,'original_n':len(ids),'paired_n':int(keep.sum()),'lost':int((~keep).sum()),'valid_per_leg':valid.sum(0).tolist(),'nonzero_recorded_flags':int(((flags!=0)&(flags!=-999)).sum()),'missing_flags':int((flags==-999).sum()),'exceptions_or_missing_shape':err})
  print('REDUCED',case,details[-1]['paired_n'],flush=True)
 sums=np.array(sums);originals=np.array(originals);cases=[d['case'] for d in details];out={'definition':'Interim whole-scene comparisons. Primary cohort is the20 fully completed group0/3 scenes, chosen by worker assignment, not measured response. All22 completed scenes are supplemental; completion can correlate with workload. Reused data, not independent acceptance. Each comparison recomputes old and new on identical newly valid IDs.','cases':details,'cohorts':{},'snapshot':snapshot}
 for name,chosen in [('complete_worker_groups_20',snapshot['primary_cases']),('all_completed_22',snapshot['complete_cases'])]:
  ix=[i for i,c in enumerate(cases) if c in chosen];n=len(ix);weights=np.vstack([np.ones(n),1-np.eye(n)]);r=matrix(np.einsum('bc,cmlk->bmlk',weights,sums[ix]));orig=matrix(np.einsum('bc,clk->blk',weights,originals[ix]))
  def stat(x):return {'value':x[0].tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(x[1:]-x[1:].mean(0)).sum(0)).tolist()}
  rows={}
  for label,q in [('original_all_common',orig),('original_paired',r[:,0]),('paired_seed',r[:,1]),('new_minus_original',r[:,1]-r[:,0])]:rows[label]={'matrix':stat(q),'diagonal_gap':stat(q[:,0,0]-q[:,1,1]),'offdiagonal_gap':stat(q[:,0,1]-q[:,1,0])}
  out['cohorts'][name]={'case_ids':[cases[i] for i in ix],'n':int(sums[ix,0,0,0].sum()),'lost':sum(details[i]['lost'] for i in ix),'results':rows};print(name,json.dumps(out['cohorts'][name]),flush=True)
 np.savez(ROOT/'partial_sums_20261108.npz',sums=sums,originals=originals,cases=cases);out['sum_data_sha256']=sha(ROOT/'partial_sums_20261108.npz');(H/'partial_report_20261108.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('DONE_PARTIAL_REPORT',flush=True)
if __name__=='__main__':main()
