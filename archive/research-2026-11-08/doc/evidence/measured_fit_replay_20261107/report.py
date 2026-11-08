"""Small saved-fit reduction, with row-conditional and four-scene uncertainties."""
import json,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
CASES=[1000,1001,1041,1081]
def response(g):return np.stack([(g[:,0]-g[:,1])/.04,(g[:,2]-g[:,3])/.04],-1)
def summary(v,case):
 mean=v.mean(0);n=len(v);means=np.array([v[case!=c].mean(0) for c in CASES]);jk=np.sqrt(3/4*np.square(means-means.mean(0)).sum(0))
 return {'value':mean.tolist(),'conditional_row_se':(v.std(0,ddof=1)/np.sqrt(n)).tolist(),'four_scene_jackknife_se':jk.tolist()}
def main():
 old=[];new=[];flags=[];cases=[];records=[]
 for case in CASES:
  p=ROOT/f'case{case}.npz';rec=json.loads(p.with_suffix('.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['data_sha256'];records.append(rec)
  with np.load(p) as z:old.append(z['old']);new.append(z['g']);flags.append(z['flags']);cases.extend([case]*len(z['ids']))
 old=np.concatenate(old);new=np.concatenate(new);flags=np.concatenate(flags);cases=np.array(cases)
 valid=np.isfinite(new).all(-1)&(np.square(new).sum(-1)<1)
 keep=valid.all((1,2));goodflags=(flags==0).all((1,2));out={'cases':CASES,'original_sample_n':len(old),'all_modes_legs_valid_n':int(keep.sum()),'all_modes_legs_flags_zero_n':int((keep&goodflags).sum()),'mode_recorded_nonzero_flags':((flags!=0)&(flags!=-999)).sum((0,1)).tolist(),'mode_missing_flag_or_shape':(flags==-999).sum((0,1)).tolist(),'mode_invalid_fits':(~valid).sum((0,1)).tolist(),'definition':'Fixed originally selected common IDs; every contrast uses exactly the same valid rows. Conditional row errors describe this subset; four-scene errors are unstable and not acceptance evidence.','populations':{},'records':records}
 for name,mask in [('all_valid',keep),('all_valid_zero_flags',keep&goodflags)]:
  c=cases[mask];o=old[mask];g=new[mask];rs={'historical':response(o)}
  for mi,label in enumerate(['seed_A','seed_B','tight_A']):rs[label]=response(g[:,:,mi])
  values={label:{'matrix':summary(r,c),'diagonal_gap':summary(r[:,0,0]-r[:,1,1],c)} for label,r in rs.items()}
  for a,b in [('seed_A','historical'),('seed_B','seed_A'),('tight_A','seed_A'),('tight_A','historical')]:
   d=rs[a]-rs[b];values[a+'_minus_'+b]={'matrix':summary(d,c),'diagonal_gap':summary(d[:,0,0]-d[:,1,1],c)}
  out['populations'][name]={'n':int(mask.sum()),'results':values}
 out['shape_change_counts']={}
 for label,a,b in [('seed_A_minus_historical',new[:,:,0],old),('seed_B_minus_A',new[:,:,1],new[:,:,0]),('tight_minus_A',new[:,:,2],new[:,:,0])]:
  delta=np.max(np.abs(a-b),axis=-1);valid_pair=np.isfinite(a).all(-1)&np.isfinite(b).all(-1)&(np.square(a).sum(-1)<1)&(np.square(b).sum(-1)<1)
  out['shape_change_counts'][label]={'valid_pair_n':int(valid_pair.sum()),'counts':{str(t):int(((delta>t)&valid_pair).sum()) for t in [.001,.01,.1]},'fraction_above_0.1':float(((delta>.1)&valid_pair).sum()/valid_pair.sum())}
 (H/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 for k,v in out['populations']['all_valid']['results'].items():print(k,v['diagonal_gap'])
 print('VALID',out['all_modes_legs_valid_n'],'FLAG0',out['all_modes_legs_flags_zero_n'],'COUNTS',out['shape_change_counts'])
if __name__=='__main__':main()
