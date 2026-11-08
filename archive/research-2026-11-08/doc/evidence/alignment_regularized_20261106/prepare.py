"""Prepare full-training response targets and matched continuation arrays on CPU."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scripts.train_fixed_g0_flow import load_split
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import fit_guard_response

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(d,indent=2,default=convert,allow_nan=False)+'\n')

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    started=time.monotonic();m=json.loads((HERE/'manifest.json').read_text());cfg=m['training']
    spec=importlib.util.spec_from_file_location('factorized',HERE.parent/'alignment_factorized_20261105/frame_model.py')
    fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)
    pin=m['checkpoints']['cartesian'];assert sha(pin['path'])==pin['sha256']
    _,condition,target,_=fm.load(pin['path'],device='cpu')
    domain=Path(m['domain']);dm=json.loads((domain/'manifest.json').read_text())
    out=ROOT/'prepared';out.mkdir(exist_ok=False)
    protocol={'domain_sha256':sha(domain/'manifest.json'),'checkpoint':pin,'source_sha256':sha(__file__),
              'guard_source_sha256':sha(__import__('sbsi.flow_guard_response',fromlist=['x']).__file__),
              'job':os.environ['SLURM_JOB_ID'],'splits':{}}
    for split in ['train','validation']:
        data=load_split(domain,dm['split'][split+'_cases']);dest=out/split;dest.mkdir()
        print('LOADED',split,len(data['target_raw']),len(data['pairs']),flush=True)
        stats=fit_guard_response(data['pair_zero_target'],data['pair_shear_target'],data['gamma'],
            m['regularization_cuts'],radius_softness=cfg['radius_softness'],magnitude_softness=cfg['magnitude_softness'])
        write(dest/'statistics.json',stats)
        raw=data['context_raw'];n=len(raw)
        if split=='train':
            width=condition.transform_frame(pd.DataFrame(raw[:1],columns=FLOW_FEATURES)).shape[1]
            context=np.lib.format.open_memmap(dest/'context.npy',mode='w+',dtype='float32',shape=(n,width))
            values=np.lib.format.open_memmap(dest/'target.npy',mode='w+',dtype='float64',shape=(n,4))
            for lo in range(0,n,250000):
                hi=min(n,lo+250000)
                context[lo:hi]=condition.transform_frame(pd.DataFrame(raw[lo:hi],columns=FLOW_FEATURES))
                values[lo:hi]=target.transform_array(data['target_raw'][lo:hi])
            context.flush();values.flush();del context,values
            np.save(dest/'pairs.npy',data['pairs'])
            design=data['gamma']@np.linalg.inv(np.asarray(stats['shape_normal']))*len(data['pairs'])
            np.save(dest/'design.npy',design);del design
        else:
            # NLL uses the same fixed500k validation rows as the original checkpoint.
            vi=np.load(ROOT.parent/'galsbi_f8c8_20261102/base/flow/validation_nll_indices.npy')
            np.save(dest/'nll_context.npy',condition.transform_frame(pd.DataFrame(raw[vi],columns=FLOW_FEATURES)))
            np.save(dest/'nll_target.npy',target.transform_array(data['target_raw'][vi]))
        # Fixed pair probes have own compact contexts, full-domain measured targets/design.
        idx=np.random.default_rng(202611068101 if split=='train' else 202611068102).choice(len(data['pairs']),cfg['probe_pairs'],replace=False)
        pair=data['pairs'][idx]
        probe_context=condition.transform_frame(pd.DataFrame(raw[pair.reshape(-1)],columns=FLOW_FEATURES)).astype('float32')
        design=data['gamma'][idx]@np.linalg.inv(np.asarray(stats['shape_normal']))*len(data['pairs'])
        np.save(dest/'probe_context.npy',probe_context);np.save(dest/'probe_design.npy',design);np.save(dest/'probe_indices.npy',idx)
        protocol['splits'][split]={'cases':dm['split'][split+'_cases'],'nll_rows':n,'pairs':len(data['pairs']),
                                  'matching':data['per_case_matching'],'probe_pairs':len(idx)}
        print('PREPARED',split,'seconds',time.monotonic()-started,flush=True)
        del data,raw,pair,probe_context,design;gc.collect()
    write(out/'completed.json',protocol);print('DONE',time.monotonic()-started,flush=True)

if __name__=='__main__':main()
