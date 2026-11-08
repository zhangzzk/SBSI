import hashlib,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/selection_information_20261106')
SOURCE=ROOT.parent/'photometry_fixed_cell_20261106'
def main():
    assert os.environ.get('SLURM_JOB_ID')
    ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'prepared.json').exists()
    receipts=[]
    for case in [1000,1040,1080,1120]:
        receipt=json.loads((SOURCE/f'case{case}.json').read_text());assert receipt['target_moment_identity_passed']
        with np.load(SOURCE/f'targets{case}.npz') as z:
            context=z['context'];score=z['reference_coefficient'].sum(-1);m=z['moments']
            flow=m[0,:,:,0].sum(0)/2;target=m[1,:,:,0].sum(0)/2;ids=z['input_index']
        x=np.column_stack([context,score]).astype(np.float32)
        assert np.isfinite(x).all() and np.all((target>=0)&(target<=1)) and np.all((flow>=0)&(flow<=1))
        assert len(np.unique(ids))==len(ids)
        np.savez(ROOT/f'case{case}.npz',x=x,target=target,flow=flow,score=score,input_index=ids,case=case)
        receipts.append({'case':case,'rows':len(target),'source_receipt_sha256':hashlib.sha256((SOURCE/f'case{case}.json').read_bytes()).hexdigest()})
    out={'cases':receipts,'features':['e1','e2','sersic_n','intrinsic_mag','circularized_Re','close050','close100','close200','reference_coefficient'],
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'target':'mean actual selection over four states, common U'}
    (ROOT/'prepared.json').write_text(json.dumps(out,indent=2)+'\n');(HERE/'prepared.json').write_bytes((ROOT/'prepared.json').read_bytes())
    print('PREPARED',receipts,flush=True)
if __name__=='__main__':main()
