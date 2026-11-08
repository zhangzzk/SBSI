"""Matched likelihood continuations, followed by the unchanged weak nine-cut guard."""
import gc, importlib.util, json, os, time
from pathlib import Path
import numpy as np
import torch
from scripts.train_fixed_g0_flow import nll_pass
from sbsi.flow_guard_response import sampled_guard_response_backward,evaluate_guard_response
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neighbour_training_model',HERE/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.c;g=c.module('neighbour_guard',HERE.parent/'alignment_regularized_20261106/train.py')
def tensor(path,dtype):return torch.as_tensor(np.load(path),dtype=dtype,device='cuda')
def main():
    assert os.environ.get('SLURM_JOB_ID')
    arm=['control','neighbour'][int(os.environ['SLURM_ARRAY_TASK_ID'])];out=c.ROOT/arm;out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(611);torch.cuda.manual_seed_all(611)
    check=json.loads((c.ROOT/'verification.json').read_text());assert check['passed'] and check['model_source_sha256']==c.sha(HERE/'model.py')
    prep=c.ROOT/'prepared';receipt=json.loads((prep/'completed.json').read_text());norm=receipt['standardization']
    model,condition,target,config=m.initialize(norm['mean'],norm['scale'],arm=='neighbour')
    print('LOADING',arm,torch.cuda.get_device_name(),flush=True)
    context=tensor(prep/'train/context.npy',torch.float32);values=tensor(c.OLD/'train/target.npy',torch.float64)
    vc=tensor(prep/'validation/nll_context.npy',torch.float32);vt=tensor(c.OLD/'validation/nll_target.npy',torch.float64)
    if arm=='control':context[:,-1]=0;vc[:,-1]=0
    started=time.monotonic();initial=nll_pass(model,vt,vc)
    # Same-device numerical gate after loading full training arrays.
    old,_,_,_=c.fm.load(c.BASE)
    with torch.no_grad():
        torch.testing.assert_close(old.log_prob(vt[:512],vc[:512,:-1].contiguous()),model.log_prob(vt[:512],vc[:512]),rtol=0,atol=0)
    del old;gc.collect();torch.cuda.empty_cache()
    cfg=json.loads((HERE.parent/'alignment_regularized_20261106/manifest.json').read_text())['training']
    stats=json.loads((c.OLD/'train/statistics.json').read_text())
    pc=torch.as_tensor(np.load(prep/'train/probe_context.npy',mmap_mode='r')[:1024].copy(),dtype=torch.float32,device='cuda')
    pd=torch.as_tensor(np.load(c.OLD/'train/probe_design.npy',mmap_mode='r')[:512].copy(),dtype=torch.float64,device='cuda')
    if arm=='control':pc[:,-1]=0
    initial_population=g.Population(pc,torch.arange(1024,device='cuda').reshape(-1,2),pd,stats,list(range(9)),cfg['relative_response_scale'])
    initial_gradient=g.gradient_check(model,initial_population)
    c.write(out/'initial_gradient_check.json',initial_gradient)
    del initial_population,pc,pd;gc.collect();torch.cuda.empty_cache()
    protocol={'arm':arm,'baseline_sha256':c.BASE_SHA,'model_source_sha256':c.sha(HERE/'model.py'),
        'source_sha256':c.sha(__file__),'prepared_sha256':c.sha(prep/'completed.json'),
        'initial_nll':initial,'epochs':40,'rows_per_epoch':4000000,'rates':[1e-5,3e-6],
        'normalization':norm,'gpu':torch.cuda.get_device_name(),'initial_same_device_exact':True,'initial_guard_gradient':initial_gradient}
    c.write(out/'protocol.json',protocol);print('TRAIN_READY',arm,initial,flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.)
    history=[]
    def save(label,metadata):
        path=out/f'{label}.pt';m.save(path,model,condition,target,config,{**protocol,**metadata})
        loaded,_,_,_=m.load(path)
        with torch.no_grad():torch.testing.assert_close(model.log_prob(vt[:512],vc[:512]),loaded.log_prob(vt[:512],vc[:512]),rtol=0,atol=0)
        del loaded;gc.collect();torch.cuda.empty_cache()
        c.write(out/f'{label}_completed.json',{'path':str(path),'sha256':c.sha(path),'reload_exact':True,**metadata})
    for epoch in range(1,41):
        rate=1e-5 if epoch<=20 else 3e-6
        for group in optimizer.param_groups:group['lr']=rate
        seed=611+1000003*epoch;ix=torch.as_tensor(np.random.default_rng(seed).choice(len(values),4000000,replace=False),device='cuda')
        train=nll_pass(model,values[ix],context[ix],optimizer=optimizer,seed=seed);validation=nll_pass(model,vt,vc)
        history.append({'epoch':epoch,'train':train,'validation':validation,'lr':rate})
        c.write(out/'history.json',history)
        print('EPOCH',arm,epoch,'nll',validation['nll'],'seconds',round(time.monotonic()-started,1),flush=True)
    save('nll',{'epoch':40,'validation_nll':validation['nll']})
    del optimizer,ix;gc.collect();torch.cuda.empty_cache()
    cfg=json.loads((HERE.parent/'alignment_regularized_20261106/manifest.json').read_text())['training']
    pairs=tensor(c.OLD/'train/pairs.npy',torch.long);design=tensor(c.OLD/'train/design.npy',torch.float64)
    stats=json.loads((c.OLD/'train/statistics.json').read_text());indices=list(range(9))
    population=g.Population(context,pairs,design,stats,indices,cfg['relative_response_scale'])
    probes={}
    for split in ['train','validation']:
        pc=tensor(prep/split/'probe_context.npy',torch.float32)
        if arm=='control':pc[:,-1]=0
        pd=tensor(c.OLD/split/'probe_design.npy',torch.float64);ps=json.loads((c.OLD/split/'statistics.json').read_text())
        pp=torch.arange(len(pc),device='cuda').reshape(-1,2)
        probes[split]=g.Population(pc,pp,pd,ps,indices,cfg['relative_response_scale'])
    def probe():
        result={'nll':nll_pass(model,vt,vc)}
        for split,pop in probes.items():
            result[split]=evaluate_guard_response(model,pop,np.arange(len(pop.pairs)),draws=cfg['probe_draws'],groups=cfg['probe_groups'],batch_size=1024,seed=202611068201)
        return result
    gradient=g.gradient_check(model,population);c.write(out/'gradient_check.json',gradient)
    c.write(out/'nll_probe.json',probe());print('GUARD_READY',arm,gradient,flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0.);guard_history=[]
    for step in range(cfg['steps']):
        rows=torch.as_tensor(np.random.default_rng(202611068401+step).integers(len(values),size=cfg['nll_batch']),device='cuda')
        optimizer.zero_grad(set_to_none=True);loss=-model.log_prob(values[rows],context[rows]).mean();assert bool(torch.isfinite(loss));loss.backward()
        guard=sampled_guard_response_backward(model,population,cfg['pairs_per_group'],cfg['draws'],chunk_size=cfg['chunk'],seed=202611068501+100003*step,weight=.001,independent_pairs=True)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True);optimizer.step()
        guard_history.append({'step':step+1,'nll':float(loss.detach()),'guard':guard['loss'],'gradient_norm':float(norm)})
        if (step+1)%8==0:
            c.write(out/'guard_history.json',guard_history);print('GUARD_STEP',arm,step+1,'seconds',round(time.monotonic()-started,1),flush=True)
    endpoint=probe();c.write(out/'nine_probe.json',endpoint)
    save('nine',{'guard_steps':64,'guard_weight':.001,'validation_nll':endpoint['nll']['nll']})
    c.write(out/'completed.json',{'arm':arm,'seconds':time.monotonic()-started,'reload_exact':True,'gradient_check':gradient})
    print('DONE',arm,time.monotonic()-started,flush=True)
if __name__=='__main__':main()
