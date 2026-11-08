"""Freeze matched changed-membership inputs and verify old/new measured sums."""
import hashlib,json,os
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
SRC=ROOT.parent/'selected_seed_calibration_20261108'
OLD=ROOT.parent/'current_measured_photometry_20261107'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('SLURM_JOB_ID')
 cfg=json.loads((H/'manifest.json').read_text());src=json.loads((SRC/'manifest.json').read_text());pins=json.loads((SRC/'source_pins.json').read_text())
 report=json.loads((H.parent/SRC.name/'report.json').read_text());assert sha(SRC/'case_sums.npz')==report['case_sums_sha256']
 with np.load(SRC/'case_sums.npz') as z:old_sums=z['old'];new_sums=z['new'];np.testing.assert_array_equal(z['cases'],cfg['cases'])
 records=[]
 for i,rec in enumerate(src['records']):
  case=rec['case'];p=SRC/f'case{case}.npz';assert sha(p)==rec['data_sha256']
  with np.load(p) as z:ids=z['ids'];old=z['old'];new=z['base'];extra=z['extra'];photo=z['photo'];candidate=z['candidate'];coverage=z['reuse'].astype(int)
  for ci in range(rec['chunks']):
   p=SRC/'chunks'/f'case{case}_chunk{ci}.npz';r=json.loads(p.with_suffix('.json').read_text());assert sha(p)==r['data_sha256'] and r['source_pins']==pins and r['repeat_verified']
   idx=extra[r['start']:r['stop']]
   with np.load(p) as z:np.testing.assert_array_equal(z['ids'],ids[idx]);np.testing.assert_array_equal(z['indices'],idx);new[idx]=z['g']
   coverage[idx]+=1
  assert (coverage==1).all()
  vo=np.isfinite(old).all(-1)&((old**2).sum(-1)<1)&candidate
  vn=np.isfinite(new).all(-1)&((new**2).sum(-1)<1)&candidate
  cuts=np.stack([(photo[...,0]>c[1][1])&(photo[...,1]>10**(-.4*(c[2]-30))) for c in cfg['cuts']],-1)
  for g,valid,expected in [(old,vo,old_sums[i]),(new,vn,new_sums[i])]:
   got=np.zeros((4,9,3))
   for leg in range(4):
    for cut in range(9):
     k=valid[:,leg]&cuts[:,leg,cut];got[leg,cut]=[k.sum(),*g[k,leg].sum(0)]
   np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-8)
  # Only validity changes alter model predictions; measured g does not enter the conditional oracle expectation.
  p=ROOT/f'case{case}.npz';np.savez(p,ids=ids,photo=photo,old_valid=vo,new_valid=vn,old_sums=old_sums[i],new_sums=new_sums[i])
  oracle_receipts=[]
  for axis in [1,2]:
   f=OLD/f'axis{axis}/case{case}.npz';r=json.loads(f.with_suffix('.json').read_text());assert sha(f)==r['data_sha256'] and r['source_sha256']==cfg['old_evaluator_sha256'];assert r['checkpoint_pair']==cfg['checkpoints']
   with np.load(f) as z:np.testing.assert_allclose(z['sim'].sum(-2)[:,:9],old_sums[i,(axis-1)*2:axis*2],rtol=1e-12,atol=1e-8)
   oracle_receipts.append({'axis':axis,'sha256':r['data_sha256']})
  records.append({'case':case,'sha256':sha(p),'changed_per_leg':(vo!=vn).sum(0).tolist(),'old_oracle_receipts':oracle_receipts,'matching':rec['matching']})
  print('PREPARED',case,records[-1]['changed_per_leg'],flush=True)
 (ROOT/'prepared.json').write_text(json.dumps(records,indent=2)+'\n');(H/'prepared.json').write_text(json.dumps(records,indent=2)+'\n');print('DONE_PREPARE',flush=True)
if __name__=='__main__':main()
