"""Check evaluation preprocessing against independently assembled training rows."""
import importlib.util,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.train_fixed_g0_flow import load_case
from sbsi.fixed_g0_domain import FLOW_FEATURES
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neighbour_context_model',HERE/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.c
def main():
    assert os.environ.get('SLURM_JOB_ID')
    prep=c.ROOT/'prepared';norm=json.loads((prep/'standardization.json').read_text())
    _,base,_,_=c.fm.load(c.BASE,device='cpu');dm=json.loads((c.DOMAIN/'manifest.json').read_text());checked=[]
    for split in ['train','validation']:
        case=dm['split'][split+'_cases'][0];raw=load_case(c.DOMAIN/f'flow/g0/case{case:03d}.npz')
        with np.load(c.ROOT/f'features/case{case}.npz') as z:
            ix=pd.Index(z['input_index']).get_indexer(raw['input_index']);assert (ix>=0).all();score=z['score'][ix]
        if split=='train':
            ii=np.random.default_rng(202611069901).choice(len(raw['context']),1024,replace=False)
            expected=np.load(prep/'train/context.npy',mmap_mode='r')[ii]
        else:
            vi=np.load(c.ROOT.parent/'galsbi_f8c8_20261102/base/flow/validation_nll_indices.npy')
            ii=np.flatnonzero(vi<len(raw['context']))[:1024];expected=np.load(prep/'validation/nll_context.npy',mmap_mode='r')[ii];ii=vi[ii]
        frame=pd.DataFrame(raw['context'][ii],columns=FLOW_FEATURES);frame[c.FEATURE]=score[ii]
        for use_score in [False,True]:
            condition=m.Condition(base,norm['mean'],norm['scale'],use_score)
            actual=condition.transform_frame(frame);wanted=expected.copy()
            if not use_score:wanted[:,-1]=0
            np.testing.assert_array_equal(actual,wanted)
            restored=m.Condition.from_state(condition.to_state())
            np.testing.assert_array_equal(restored.transform_frame(frame),actual)
            checked.append({'split':split,'case':case,'rows':len(ii),'use_score':use_score,'exact':True})
    result={'passed':True,'checks':checked,'source_sha256':c.sha(__file__),'model_source_sha256':c.sha(HERE/'model.py')}
    c.write(c.ROOT/'context_verification.json',result);c.write(HERE/'context_verification.json',result);print(result,flush=True)
if __name__=='__main__':main()
