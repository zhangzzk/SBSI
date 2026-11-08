"""Matched fresh world/intrinsic-frame density fits; Slurm only."""
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
from scripts.train_fixed_g0_flow import load_split,nll_pass
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.measurement_model import load_measurement_model

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('experimental_frame',HERE/'frame_model.py')
fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
OUT=ROOT/'alignment_frame_20261105'
SOURCE=ROOT/'galsbi_f8c8_20261102'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    arm=['cartesian','frame'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    output=OUT/arm;output.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(202611053111)
    domain=SOURCE/'harmonized/domain';manifest=json.loads((domain/'manifest.json').read_text())
    traincases=manifest['split']['train_cases'][:40];valcases=manifest['split']['validation_cases']
    old=load_measurement_model(SOURCE/'base/flow/nll/selected.pt',device='cpu')
    condition,target=old.condition_preprocessor,old.target_transform
    config=json.loads((SOURCE/'base/flow/protocol.json').read_text())['model_config']
    scale=float(np.sqrt(np.mean(np.square(config['disk_coordinate_scales']))))
    config['disk_coordinate_means']=[0.,0.];config['disk_coordinate_scales']=[scale,scale]
    del old
    torch.manual_seed(202611053111)
    model=fm.FrameFlow(config,condition,arm).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=8,threshold=1e-4,threshold_mode='abs',min_lr=1e-7)
    started=time.monotonic();print('LOADING',arm,flush=True)
    train=load_split(domain,traincases)
    contexts=torch.as_tensor(condition.transform_frame(pd.DataFrame(train['context_raw'],columns=FLOW_FEATURES)),device='cuda')
    targets=torch.as_tensor(target.transform_array(train['target_raw']),device='cuda')
    protocol={'arm':arm,'train_cases':traincases,'validation_cases':valcases,'training_rows':len(targets),'epochs':120,'rows_per_epoch':2_000_000,
       'initial_lr':3e-5,'seed':202611053111,'config':config,'matching':train['per_case_matching'],
       'near_zero_intrinsic_rows':int((np.linalg.norm(train['context_raw'][:,:2],axis=1)<1e-8).sum()),
       'source_sha256':sha(__file__),'frame_model_sha256':sha(HERE/'frame_model.py'),'job':os.environ['SLURM_JOB_ID']}
    del train;gc.collect()
    val=load_split(domain,valcases)
    ix=np.random.default_rng(202611052701).choice(len(val['target_raw']),500_000,replace=False)
    vc=torch.as_tensor(condition.transform_frame(pd.DataFrame(val['context_raw'][ix],columns=FLOW_FEATURES)),device='cuda')
    vt=torch.as_tensor(target.transform_array(val['target_raw'][ix]),device='cuda')
    del val;gc.collect()
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    history=[];best=float('inf');bestepoch=None
    initial=nll_pass(model,vt,vc);print('INITIAL',arm,initial,flush=True)
    for epoch in range(1,121):
        seed=202611053200+1000003*epoch
        ix=torch.as_tensor(np.random.default_rng(seed).choice(len(targets),2_000_000,replace=False),device='cuda')
        trained=nll_pass(model,targets[ix],contexts[ix],optimizer=optimizer,seed=seed)
        validation=nll_pass(model,vt,vc)
        scheduler.step(validation['nll'])
        history.append({'epoch':epoch,'train':trained,'validation':validation,'lr':optimizer.param_groups[0]['lr']})
        if validation['nll']<best:
            best=validation['nll'];bestepoch=epoch
            fm.save(output/'selected.pt',model,condition,target,config,{'protocol':protocol,'epoch':epoch,'validation_nll':best})
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print('EPOCH',arm,epoch,'train',trained['nll'],'val',validation['nll'],'best',bestepoch,'lr',optimizer.param_groups[0]['lr'],'sec',round(time.monotonic()-started,1),flush=True)
    fm.save(output/'endpoint.pt',model,condition,target,config,{'protocol':protocol,'epoch':120,'validation_nll':validation['nll']})
    loaded,_,_,_=fm.load(output/'endpoint.pt')
    with torch.no_grad():torch.testing.assert_close(model.log_prob(vt[:100],vc[:100]),loaded.log_prob(vt[:100],vc[:100]),rtol=0,atol=0)
    (output/'completed.json').write_text(json.dumps({'seconds':time.monotonic()-started,'best_epoch':bestepoch,'best_nll':best,'initial_nll':initial['nll'],'selected_sha256':sha(output/'selected.pt'),'reload_exact':True},indent=2)+'\n')

if __name__=='__main__':main()
