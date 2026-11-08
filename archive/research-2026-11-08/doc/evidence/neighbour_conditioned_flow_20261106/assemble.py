"""Align new truth features with the exact existing training and probe row order."""
import gc, importlib.util, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.train_fixed_g0_flow import load_case
from sbsi.fixed_g0_domain import FLOW_FEATURES
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neighbour_common_assemble',HERE/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def main():
    assert os.environ.get('SLURM_JOB_ID')
    dm=json.loads((c.DOMAIN/'manifest.json').read_text());out=c.ROOT/'prepared';out.mkdir(exist_ok=False)
    receipts=[json.loads((c.ROOT/f'features/case{case}.json').read_text()) for case in range(200)]
    assert all(r['unmatched_parents']==0 for r in receipts)
    assert all(r['source_sha256']==c.sha(HERE/'prepare_features.py') and r['common_sha256']==c.sha(HERE/'common.py') for r in receipts)
    _,condition,_,_=c.fm.load(c.BASE,device='cpu');details={};started=time.monotonic()
    for split in ['train','validation']:
        cases=dm['split'][split+'_cases'];dest=out/split;dest.mkdir()
        old=json.loads((c.OLD/'completed.json').read_text())['splits'][split]
        n=old['nll_rows'];scores=np.lib.format.open_memmap(dest/'raw_score.npy',mode='w+',dtype='float64',shape=(n,))
        offset=0;matching=[]
        for case in cases:
            feature=c.ROOT/f'features/case{case}.npz'
            receipt=json.loads(feature.with_suffix('.json').read_text());assert c.sha(feature)==receipt['data_sha256']
            with np.load(feature) as z:ids=z['input_index'];score=z['score']
            for leg in ['g0','g05']:
                d=load_case(c.DOMAIN/f'flow/{leg}/case{case:03d}.npz');ix=pd.Index(ids).get_indexer(d['input_index'])
                assert (ix>=0).all();nn=len(ix);scores[offset:offset+nn]=score[ix]
                if split=='train':
                    expected=np.load(c.OLD/'train/context.npy',mmap_mode='r')[offset:offset+nn]
                    actual=condition.transform_frame(pd.DataFrame(d['context'],columns=FLOW_FEATURES)).astype('float32')
                    np.testing.assert_array_equal(actual,expected)
                offset+=nn;matching.append({'case':case,'leg':leg,'rows':nn,'unmatched':0})
            if len(matching)%40==0:print('ALIGN',split,case,offset,flush=True)
        assert offset==n;scores.flush()
        if split=='train':
            mean=float(np.mean(scores));scale=float(np.std(scores));assert scale>0
            c.write(out/'standardization.json',{'mean':mean,'scale':scale,'source':'training rows only','rows':n,'feature':c.FEATURE})
            old_context=np.load(c.OLD/'train/context.npy',mmap_mode='r')
            context=np.lib.format.open_memmap(dest/'context.npy',mode='w+',dtype='float32',shape=(n,old_context.shape[1]+1))
            for lo in range(0,n,250000):
                hi=min(n,lo+250000);context[lo:hi,:-1]=old_context[lo:hi];context[lo:hi,-1]=(scores[lo:hi]-mean)/scale
            context.flush();del context,old_context
            pairs=np.load(c.OLD/'train/pairs.npy',mmap_mode='r')
            for lo in range(0,len(pairs),250000):np.testing.assert_array_equal(scores[pairs[lo:lo+250000,0]],scores[pairs[lo:lo+250000,1]])
        else:
            vi=np.load(c.ROOT.parent/'galsbi_f8c8_20261102/base/flow/validation_nll_indices.npy')
            old_context=np.load(c.OLD/'validation/nll_context.npy')
            np.save(dest/'nll_context.npy',np.column_stack([old_context,(scores[vi]-mean)/scale]).astype('float32'))
        # Probe row order is the original matched pair list, sampled by stored indices.
        from scripts.train_fixed_g0_flow import load_split
        if split=='train':pairs=np.load(c.OLD/'train/pairs.npy',mmap_mode='r')
        else:
            data=load_split(c.DOMAIN,cases);pairs=data['pairs'];del data;gc.collect()
        pi=np.load(c.OLD/split/'probe_indices.npy');probe_scores=scores[pairs[pi].reshape(-1)]
        pc=np.load(c.OLD/split/'probe_context.npy')
        np.save(dest/'probe_context.npy',np.column_stack([pc,(probe_scores-mean)/scale]).astype('float32'))
        details[split]={'rows':n,'cases':cases,'matching':matching,'score_min':float(scores.min()),'score_max':float(scores.max())}
        del scores,pairs;gc.collect()
    c.write(out/'completed.json',{'splits':details,'standardization':json.loads((out/'standardization.json').read_text()),
        'features':receipts,'source_sha256':c.sha(__file__),'domain_sha256':c.sha(c.DOMAIN/'manifest.json'),'seconds':time.monotonic()-started})
    print('DONE_ASSEMBLE',time.monotonic()-started,flush=True)
if __name__=='__main__':main()
