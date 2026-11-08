"""Matched NLL-only, one-cut and nine-cut continuations; fixed endpoints."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scripts.train_fixed_g0_flow import nll_pass
from sbsi.flow_guard_response import GuardResponsePopulation, independent_guard_score, sampled_guard_response_backward, evaluate_guard_response

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2,default=convert,allow_nan=False)+'\n');temp.replace(p)
def tensor(path,dtype):return torch.as_tensor(np.load(path),dtype=dtype,device='cuda')

class Population(GuardResponsePopulation):
    def __init__(self,context,pairs,design,statistics,indices,relative_scale):
        self.context=context;self.pairs=pairs;self.design=design
        self.cuts=tuple(statistics['cuts'][i] for i in indices)
        self.radius_softness=statistics['radius_softness'];self.magnitude_softness=statistics['magnitude_softness'];self.zero_point=30.
        def convert(key):return torch.as_tensor(np.asarray(statistics[key])[indices],dtype=torch.float64,device='cuda')
        self.target=convert('target');self.baseline_mass=convert('baseline_mass');self.baseline_mean=convert('baseline_mean_shape')
        # One scale per cut; all four response components constrained separately.
        scale=relative_scale*.5*(self.target[:,0]+self.target[:,3]).abs()
        assert bool(torch.isfinite(scale).all() and (scale>1e-8).all())
        self.response_scale=scale[:,None];self.statistics=statistics

def gradient_check(model,population):
    """Replay backward must match direct autograd on identical independent banks."""
    count,draws,chunk,seed=128,4,64,202611068301
    model.zero_grad(set_to_none=True)
    result=sampled_guard_response_backward(model,population,count,draws,chunk_size=chunk,seed=seed,weight=1.,independent_pairs=True)
    replay=[p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True);rng=np.random.default_rng(seed)
    groups=[torch.as_tensor(rng.integers(len(population.pairs),size=count),device='cuda') for _ in range(2)]
    means=[]
    for stream,indices in enumerate(groups):
        parts=[population.contributions(model,indices[start:start+chunk],draws,seed+100000000*stream+block)
               for block,start in enumerate(range(0,count,chunk))]
        means.append(torch.cat(parts).mean(0))
    direct=independent_guard_score(means[0],means[1],population.target,population.response_scale);direct.backward()
    error=0.;norm=0.
    for expected,p in zip(replay,model.parameters()):
        error+=float(((expected-p.grad).double()**2).sum());norm+=float((p.grad.double()**2).sum())
    relative=(error/max(norm,1e-30))**.5
    assert relative<3e-5,(relative,error,norm)
    assert abs(float(direct.detach())-result['loss'])<1e-8*max(1.,abs(result['loss']))
    model.zero_grad(set_to_none=True)
    return {'relative_gradient_error':relative,'loss':result['loss'],'independent_pair_groups':True}

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    m=json.loads((HERE/'manifest.json').read_text());cfg=m['training'];arm=m['arms'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    out=ROOT/arm['name'];out.mkdir(exist_ok=False);torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('factorized',HERE.parent/'alignment_factorized_20261105/frame_model.py')
    fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)
    pin=m['checkpoints']['cartesian'];assert sha(pin['path'])==pin['sha256']
    prepared=ROOT/'prepared';receipt=json.loads((prepared/'completed.json').read_text());assert receipt['checkpoint']==pin
    model,condition,target,artifact=fm.load(pin['path']);model.eval()
    stats=json.loads((prepared/'train/statistics.json').read_text());center=[i for i,c in enumerate(stats['cuts']) if c['radius_min']==3. and c['magnitude_max']==25.]
    assert len(center)==1;indices=center if arm['cuts']=='single' else list(range(9))
    print('LOADING',arm['name'],'GPU',torch.cuda.get_device_name(),flush=True)
    context=tensor(prepared/'train/context.npy',torch.float32);values=tensor(prepared/'train/target.npy',torch.float64)
    pairs=tensor(prepared/'train/pairs.npy',torch.long);design=tensor(prepared/'train/design.npy',torch.float64)
    population=Population(context,pairs,design,stats,indices,cfg['relative_response_scale'])
    vc=tensor(prepared/'validation/nll_context.npy',torch.float32);vt=tensor(prepared/'validation/nll_target.npy',torch.float64)
    probes={}
    for split in ['train','validation']:
        pc=tensor(prepared/split/'probe_context.npy',torch.float32);pd=tensor(prepared/split/'probe_design.npy',torch.float64)
        ps=json.loads((prepared/split/'statistics.json').read_text());pp=torch.arange(len(pc),device='cuda').reshape(-1,2)
        probes[split]=Population(pc,pp,pd,ps,list(range(9)),cfg['relative_response_scale'])
    started=time.monotonic();print('DATA_READY',arm['name'],'rows',len(values),'pairs',len(pairs),flush=True)
    check=gradient_check(model,population);write(out/'gradient_check.json',check)
    print('GRADIENT_CHECK',arm['name'],check,flush=True)
    protocol={'arm':arm,'training':cfg,'source_sha256':sha(__file__),'manifest_sha256':sha(HERE/'manifest.json'),
              'prepared_sha256':sha(prepared/'completed.json'),'checkpoint':pin,'cuts':population.cuts,
              'target':population.target.cpu().numpy(),'response_scale':population.response_scale.cpu().numpy(),
              'job':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(),'gradient_check':check}
    write(out/'protocol.json',protocol)
    def evaluate():
        report={'nll':nll_pass(model,vt,vc)}
        for split,pop in probes.items():
            report[split]=evaluate_guard_response(model,pop,np.arange(len(pop.pairs)),draws=cfg['probe_draws'],groups=cfg['probe_groups'],batch_size=1024,seed=202611068201)
            print('PROBE',arm['name'],split,'seconds',time.monotonic()-started,flush=True)
        return report
    write(out/'baseline.json',evaluate());print('TRAIN_READY',arm['name'],check,'seconds',time.monotonic()-started,flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    history=[]
    for step in range(cfg['steps']):
        rows=torch.as_tensor(np.random.default_rng(202611068401+step).integers(len(values),size=cfg['nll_batch']),device='cuda')
        optimizer.zero_grad(set_to_none=True);loss=-model.log_prob(values[rows],context[rows]).mean();assert bool(torch.isfinite(loss));loss.backward()
        nll_norm=sum(float((p.grad.detach().double()**2).sum()) for p in model.parameters())**.5
        guard={'loss':0.}
        if arm['weight']:
            guard=sampled_guard_response_backward(model,population,cfg['pairs_per_group'],cfg['draws'],chunk_size=cfg['chunk'],seed=202611068501+100003*step,weight=arm['weight'],independent_pairs=True)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True);optimizer.step()
        history.append({'step':step+1,'nll':float(loss.detach()),'guard_score':guard['loss'],'nll_gradient_norm':nll_norm,'combined_gradient_norm':float(norm),'clipped':bool(norm>cfg['clip'])})
        if (step+1)%8==0:
            write(out/'history.json',history);print('STEP',arm['name'],step+1,'nll',history[-1]['nll'],'guard',guard['loss'],'seconds',time.monotonic()-started,flush=True)
    fm.save(out/'final.pt',model,condition,target,artifact['config'],metadata={**artifact.get('metadata',{}),'regularization_continuation':protocol,'step':cfg['steps']})
    loaded,_,_,_=fm.load(out/'final.pt')
    with torch.no_grad():torch.testing.assert_close(model.log_prob(vt[:100],vc[:100]),loaded.log_prob(vt[:100],vc[:100]),rtol=0,atol=0)
    del loaded;gc.collect();torch.cuda.empty_cache()
    write(out/'endpoint.json',evaluate())
    write(out/'completed.json',{'arm':arm,'steps':cfg['steps'],'sha256':sha(out/'final.pt'),'reload_exact':True,'seconds':time.monotonic()-started})
    print('DONE',arm['name'],time.monotonic()-started,flush=True)

if __name__=='__main__':main()
