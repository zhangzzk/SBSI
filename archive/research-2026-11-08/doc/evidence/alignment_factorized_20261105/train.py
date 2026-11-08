"""Native F8 NLL replay with learning diagnostics and a matched schedule fork."""
import argparse
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
from scripts.train_fixed_g0_flow import load_split,nll_pass,cpu_optimizer_state
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.measurement_model import load_measurement_model

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('learning_trace',HERE.parent/'alignment_trajectory_20261105/trace.py')
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
spec=importlib.util.spec_from_file_location('experimental_frame',HERE/'frame_model.py')
fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
CONTROL=ROOT/'alignment_trajectory_20261105'
OUT=ROOT/'alignment_factorized_20261105'
FAMILIES={'fs2':'v36_f8_20260924_v1','galsbi':'galsbi_f8c8_20261102'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

def main():
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['cartesian','frame']);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    family='fs2' if a.arm=='fs2' else 'galsbi';source=ROOT/FAMILIES[family]
    output=OUT/a.arm;output.mkdir(exist_ok=False);(output/'states').mkdir()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    original=json.loads((source/'base/flow/protocol.json').read_text());domain=Path(original['domain_manifest']).parent
    manifest=json.loads((domain/'manifest.json').read_text());config=original['model_config']
    old=load_measurement_model(source/'base/flow/nll/selected.pt',device='cuda')
    condition,target=old.condition_preprocessor,old.target_transform
    started=time.monotonic();print('LOAD_TRAIN',a.arm,domain,flush=True)
    data=load_split(domain,manifest['split']['train_cases'])
    assert len(data['target_raw'])==original['counts']['train_nll_rows']
    td=trace.prepare(data,condition,target,202611055001,output,'train')
    contexts=torch.as_tensor(condition.transform_frame(pd.DataFrame(data['context_raw'],columns=FLOW_FEATURES)),dtype=torch.float32,device='cuda')
    targets=torch.as_tensor(target.transform_array(data['target_raw']),dtype=torch.float64,device='cuda')
    trainmatching=data['per_case_matching'];del data;gc.collect()
    print('LOAD_VALIDATION',a.arm,len(targets),flush=True)
    data=load_split(domain,manifest['split']['validation_cases'])
    vd=trace.prepare(data,condition,target,202611055002,output,'validation')
    vi=np.load(source/'base/flow/validation_nll_indices.npy')
    assert len(data['target_raw'])==original['counts']['validation_nll_rows'] and len(vi)==500000
    vc=torch.as_tensor(condition.transform_frame(pd.DataFrame(data['context_raw'][vi],columns=FLOW_FEATURES)),dtype=torch.float32,device='cuda')
    vt=torch.as_tensor(target.transform_array(data['target_raw'][vi]),dtype=torch.float64,device='cuda')
    valmatching=data['per_case_matching'];del data;gc.collect()
    protocol={'arm':a.arm,'family':family,'domain':str(domain),'domain_sha256':sha(domain/'manifest.json'),
        'original_protocol':str(source/'base/flow/protocol.json'),'original_protocol_sha256':sha(source/'base/flow/protocol.json'),
        'source_sha256':sha(__file__),'trace_sha256':sha(HERE.parent/'alignment_trajectory_20261105/trace.py'),'frame_model_sha256':sha(HERE/'frame_model.py'),'job':os.environ['SLURM_JOB_ID'],
        'train_rows':len(targets),'train_cases':manifest['split']['train_cases'],'validation_cases':manifest['split']['validation_cases'],
        'train_matching':trainmatching,'validation_matching':valmatching,'validation_indices_sha256':sha(source/'base/flow/validation_nll_indices.npy'),
        'architecture':'photometry5_shape5_affine','parameters':sum(v.numel() for v in fm.FrameFlow(config,condition,a.arm).parameters()),'epochs':140,'rows_per_epoch':4000000,'seed':611,'diagnostic_epochs':trace.EPOCHS,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'maximum_allocated_bytes_after_load':torch.cuda.max_memory_allocated()}
    reference={'validation':nll_pass(old.model,vt,vc)}
    for label,d in [('train',td),('validation',vd)]:reference[label+'_diagnostic']=trace.diagnose(old.model,d,output,-1,label)
    write(output/'reference.json',reference);del old;gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(611);torch.cuda.manual_seed_all(611)
    model=fm.FrameFlow(config,condition,a.arm).to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=4,threshold=1e-4,threshold_mode='abs',min_lr=1e-7)
    history=[];best=float('inf');best_epoch=None;start=1;fs_rates=None
    fs_history=json.loads((CONTROL/'fs2/history.json').read_text())
    fs_rates={x['epoch']:x['lr_used'] for x in fs_history if x['epoch']>0}
    protocol['schedule']='FS2 realized consumed rates, epochs1-140, fixed before frame training'
    protocol['schedule_sha256']=sha(CONTROL/'fs2/history.json')
    protocol['comparison']='Same GalSBI rows, initialization of base, AdamW, batches and rates as galsbi_fs2rate; factorized photometry-then-shape architecture; paired arms differ only in output frame'
    write(output/'protocol.json',protocol)
    def save_model(path,epoch,nll):
        fm.save(path,model,condition,target,config,metadata={'experiment':'alignment_factorized_20261105','arm':a.arm,'epoch':epoch,'validation_nll':nll,'protocol_sha256':sha(output/'protocol.json')})
    def save_state(path,epoch):
        temporary=path.with_suffix('.tmp')
        torch.save({'epoch':epoch,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':cpu_optimizer_state(optimizer),
                    'scheduler':scheduler.state_dict(),'history':history,'best_nll':best,'best_epoch':best_epoch},temporary)
        os.replace(temporary,path)
    baseline=nll_pass(model,vt,vc)
    if start==1:
        record={'epoch':0,'validation':baseline,'lr_used':1e-4,'lr_next':1e-4}
        for label,d in [('train',td),('validation',vd)]:record[label+'_diagnostic']=trace.diagnose(model,d,output,0,label)
        history.append(record);save_state(output/'states/epoch000.pt',0)
    else:
        for label,d in [('train',td),('validation',vd)]:trace.diagnose(model,d,output,20,label)
        write(output/'fork_validation.json',baseline)
        best=baseline['nll'];best_epoch=20
        save_model(output/'selected.pt',20,best);save_state(output/'selected_state.pt',20)
    print('TRAINING_READY',a.arm,'rows',len(targets),'reference_nll',reference['validation']['nll'],'seconds',time.monotonic()-started,flush=True)
    for epoch in range(start,141):
        if fs_rates is not None:
            for group in optimizer.param_groups:group['lr']=fs_rates[epoch]
        rate=optimizer.param_groups[0]['lr'];seed=611+1000003*epoch
        ix=torch.as_tensor(np.random.default_rng(seed).choice(len(targets),4000000,replace=False),device='cuda')
        trained=nll_pass(model,targets[ix],contexts[ix],optimizer=optimizer,seed=seed)
        validation=nll_pass(model,vt,vc)
        if fs_rates is None:scheduler.step(validation['nll'])
        record={'epoch':epoch,'train':trained,'validation':validation,'lr_used':rate,'lr_next':optimizer.param_groups[0]['lr']}
        if epoch in trace.EPOCHS:
            for label,d in [('train',td),('validation',vd)]:record[label+'_diagnostic']=trace.diagnose(model,d,output,epoch,label)
        history.append(record)
        if validation['nll']<best:
            best=validation['nll'];best_epoch=epoch;save_model(output/'selected.pt',epoch,best);save_state(output/'selected_state.pt',epoch)
        if epoch in trace.EPOCHS:save_state(output/f'states/epoch{epoch:03d}.pt',epoch)
        write(output/'history.json',history)
        print('EPOCH',a.arm,epoch,'nll',validation['nll'],'lr',rate,'next',optimizer.param_groups[0]['lr'],
              'alignment',record.get('validation_diagnostic',{}).get('alignment_error_bin8'),'seconds',round(time.monotonic()-started,1),flush=True)
    save_model(output/'endpoint.pt',140,validation['nll'])
    # Ensure a saved model predicts exactly as the final in-memory model.
    loaded,_,_,_=fm.load(output/'endpoint.pt',device='cuda')
    with torch.no_grad():torch.testing.assert_close(model.log_prob(vt[:100],vc[:100]),loaded.log_prob(vt[:100],vc[:100]),rtol=0,atol=0)
    write(output/'completed.json',{'seconds':time.monotonic()-started,'best_epoch':best_epoch,'best_nll':best,
        'reference_nll':reference['validation']['nll'],'selected_sha256':sha(output/'selected.pt'),'reload_exact':True})
    print('DONE',a.arm,flush=True)
if __name__=='__main__':main()
