"""Controlled NLL continuation: full versus intrinsic-small input population."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scripts.train_fixed_g0_flow import load_split, nll_pass
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model, save_measurement_model

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('alignment_probe',HERE/'probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
SOURCE=ROOT/'galsbi_f8c8_20261102'
OUT=ROOT/'alignment_learning_20261105'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    arm=['full','small'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    output=OUT/f'pilot_{arm}';output.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(202611052700)
    domain=SOURCE/'harmonized/domain';manifest=json.loads((domain/'manifest.json').read_text())
    traincases=manifest['split']['train_cases'][:40];valcases=manifest['split']['validation_cases']
    ckpt=SOURCE/'base/flow/nll/selected.pt'
    receipt=json.loads((SOURCE/'base/flow/nll_completed.json').read_text());assert sha(ckpt)==receipt['selected_sha256']
    bundle=load_measurement_model(ckpt,device='cuda');model=bundle.model
    saved=torch.load(SOURCE/'base/flow/nll/selected_optimizer.pt',map_location='cuda',weights_only=False)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,saved['model'][k],rtol=0,atol=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
    optimizer.load_state_dict(saved['optimizer'])
    for g in optimizer.param_groups:g['lr']=1e-4
    config=json.loads((SOURCE/'base/flow/protocol.json').read_text())['model_config']
    del saved
    started=time.monotonic();print('LOADING',arm,flush=True)
    train=load_split(domain,traincases)
    mask=np.ones(len(train['context_raw']),bool)
    if arm=='small':mask=(train['context_raw'][:,4]>=.15)&(train['context_raw'][:,4]<.25)
    contexts=bundle.context_tensor(pd.DataFrame(train['context_raw'][mask],columns=FLOW_FEATURES))
    targets=torch.as_tensor(bundle.target_transform.transform_array(train['target_raw'][mask]),device='cuda')
    trainrows=len(targets);assert trainrows>=4_000_000
    protocol={'arm':arm,'initial_checkpoint':str(ckpt),'initial_sha256':sha(ckpt),'train_cases':traincases,'validation_cases':valcases,
              'training_rows':trainrows,'epochs':20,'rows_per_epoch':4_000_000,'lr':1e-4,'batch':8192,'clip':5,
              'source_sha256':sha(__file__),'probe_sha256':sha(HERE/'probe.py'),'job':os.environ['SLURM_JOB_ID'],
              'matching':train['per_case_matching'],'training_uses_all_usable_rows':True}
    del train,mask;gc.collect()
    val=load_split(domain,valcases)
    ix=np.random.default_rng(202611052701).choice(len(val['target_raw']),500_000,replace=False)
    vc=bundle.context_tensor(pd.DataFrame(val['context_raw'][ix],columns=FLOW_FEATURES))
    vt=torch.as_tensor(bundle.target_transform.transform_array(val['target_raw'][ix]),device='cuda')
    del val;gc.collect()
    history=[{'epoch':0,'validation':nll_pass(model,vt,vc)}]
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('START',arm,trainrows,'baseline',history[0],flush=True)
    for epoch in range(1,21):
        seed=202611052800+1000003*epoch
        ix=torch.as_tensor(np.random.default_rng(seed).choice(trainrows,4_000_000,replace=False),device='cuda')
        result=nll_pass(model,targets[ix],contexts[ix],optimizer=optimizer,seed=seed)
        validation=nll_pass(model,vt,vc)
        history.append({'epoch':epoch,'train':result,'validation':validation})
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print('EPOCH',arm,epoch,'train',result['nll'],'val',validation['nll'],'seconds',round(time.monotonic()-started,1),flush=True)
    save_measurement_model(output/'endpoint.pt',model,bundle.condition_preprocessor,bundle.target_transform,config,
                           metadata={'diagnostic_only':True,'arm':arm,'epoch':20,'protocol':protocol})
    del targets,contexts,vt,vc,optimizer;gc.collect();torch.cuda.empty_cache();model.eval()
    for case in valcases:
        d=load_split(domain,[case]);raw=d['context_raw'][d['pairs'][:,0]]
        keep=(raw[:,4]>=.15)&(raw[:,4]<.25);raw=raw[keep]
        context=bundle.context_tensor(pd.DataFrame(raw,columns=FLOW_FEATURES))
        intr=torch.as_tensor(raw[:,:2],device='cuda',dtype=torch.float64)
        groups=torch.as_tensor(probe.group_indices(raw),device='cuda')
        profile=torch.zeros((2,2,len(probe.GROUPS)*32,7),device='cuda',dtype=torch.float64)
        with torch.no_grad():
            for block,lo in enumerate(range(0,len(raw),2048)):
                s=slice(lo,min(lo+2048,len(raw)))
                for bank in range(2):
                    seed=202611051900+10000019*case+1000003*bank+block
                    v=physical_context_draws(model,[context[s]],16,seed)[0]
                    assert bool(torch.isfinite(v).all())
                    probe.add_profile(profile[bank],v,intr[s],groups[s])
        arr=profile.cpu().numpy().reshape(2,2,len(probe.GROUPS),32,7)
        for start,width in [(1,4),(5,4),(9,4),(13,4),(17,3),(20,16)]:
            np.testing.assert_allclose(arr[:,:,start:start+width].sum(2),arr[:,:,0],rtol=1e-10,atol=1e-8)
        np.savez(output/f'case{case:03d}.npz',profile=arr)
        print('PROFILE',arm,case,round(time.monotonic()-started,1),flush=True)
    (output/'completed.json').write_text(json.dumps({'seconds':time.monotonic()-started,'endpoint_sha256':sha(output/'endpoint.pt'),'source_sha256':sha(__file__)},indent=2)+'\n')

if __name__=='__main__':main()
