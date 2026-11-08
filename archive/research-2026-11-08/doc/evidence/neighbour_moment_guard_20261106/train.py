"""One allocation prepares targets and trains the two frozen, matched endpoints."""
import gc,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from sbsi.flow_guard_response import numpy_guard_weights,sampled_guard_response_backward,evaluate_guard_response
from scripts.train_fixed_g0_flow import nll_pass
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('moment_training_common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
moment=c.module('moment_population',HERE/'moment.py')
def prepare(cfg):
    dest=c.ROOT/'targets.json';old=Path(cfg['original_prepared_root']);prep=Path(cfg['prepared_root'])
    protocol={'manifest_sha256':c.sha(HERE/'manifest.json'),'source_sha256':c.sha(__file__),'prepared_sha256':c.sha(prep/'completed.json')}
    if dest.exists():
        d=json.loads(dest.read_text());assert d['provenance']==protocol;return d
    pin=cfg['initial_checkpoint'];assert c.sha(pin['path'])==pin['sha256']
    model,condition,target,_=c.m.load(pin['path'],device='cpu');del model
    y=np.load(old/'train/target.npy',mmap_mode='r');x=np.load(prep/'train/context.npy',mmap_mode='r');b=np.load(prep/'train/raw_score.npy',mmap_mode='r')
    assert len(y)==len(x)==len(b);total=np.zeros((len(cfg['regularization_cuts']),2));started=time.monotonic()
    for lo in range(0,len(y),250000):
        hi=min(len(y),lo+250000);physical=target.inverse_transform_array(y[lo:hi])
        np.testing.assert_array_equal(x[lo:hi,-1],((b[lo:hi]-condition.mean)/condition.scale).astype(np.float32))
        w=numpy_guard_weights(physical,cfg['regularization_cuts'],radius_softness=cfg['training']['radius_softness'],magnitude_softness=cfg['training']['magnitude_softness'])
        total[:,0]+=w.sum(0);total[:,1]+=(w*b[lo:hi,None]).sum(0)
        if lo%5000000==0:print('PREPARE',hi,len(y),round(time.monotonic()-started,1),flush=True)
    d={'cuts':cfg['regularization_cuts'],'mass':total[:,0]/len(y),'selected_score':total[:,1]/total[:,0],'rows':len(y),'provenance':protocol,'input_score_alignment_exact':True,'target_population':'all original training NLL rows, both usable legs, no new population cuts'}
    c.write(dest,d);c.write(HERE/'targets.json',d);return json.loads(dest.read_text())
def tensor(path,dtype):return torch.as_tensor(np.load(path),dtype=dtype,device='cuda')
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4);torch.manual_seed(611);torch.cuda.manual_seed_all(611)
    cfg=c.manifest();stats=prepare(cfg);prep=Path(cfg['prepared_root']);old=Path(cfg['original_prepared_root']);tcfg=cfg['training']
    provenance={'manifest_sha256':c.sha(HERE/'manifest.json'),'train_source_sha256':c.sha(__file__),'moment_source_sha256':c.sha(HERE/'moment.py'),'initial_checkpoint':cfg['initial_checkpoint'],'targets_sha256':c.sha(c.ROOT/'targets.json')}
    pin=cfg['initial_checkpoint'];assert c.sha(pin['path'])==pin['sha256']
    print('LOADING',torch.cuda.get_device_name(),flush=True)
    context=tensor(prep/'train/context.npy',torch.float32);values=tensor(old/'train/target.npy',torch.float64);score=tensor(prep/'train/raw_score.npy',torch.float64)
    vc=tensor(prep/'validation/nll_context.npy',torch.float32);vt=tensor(old/'validation/nll_target.npy',torch.float64)
    pairs=tensor(old/'train/pairs.npy',torch.long);design=tensor(old/'train/design.npy',torch.float64)
    response_stats=json.loads((old/'train/statistics.json').read_text())
    response=c.g.Population(context,pairs,design,response_stats,list(range(9)),tcfg['relative_response_scale'])
    probes={}
    for split in ['train','validation']:
        pc=tensor(prep/split/'probe_context.npy',torch.float32);pd=tensor(old/split/'probe_design.npy',torch.float64)
        ps=json.loads((old/split/'statistics.json').read_text())
        probes[split]=c.g.Population(pc,torch.arange(len(pc),device='cuda').reshape(-1,2),pd,ps,list(range(9)),tcfg['relative_response_scale'])
    for arm in cfg['arms']:
        out=c.ROOT/arm;out.mkdir(exist_ok=True)
        if (out/'completed.json').exists():
            done=json.loads((out/'completed.json').read_text());assert done['provenance']==provenance and c.sha(out/'final.pt')==done['sha256'];print('SKIP_COMPLETE',arm,flush=True);continue
        model,condition,target,artifact=c.m.load(pin['path']);assert condition.use_score
        aux=moment.Population(context,values,score,target,stats,cfg)
        checks={'response':c.g.gradient_check(model,response),'moment':moment.gradient_check(model,aux)}
        with torch.no_grad():
            z=torch.from_numpy(c.portable.latent_bank(128,4,4,202611061008)).to('cuda').reshape(-1,4)
            ctx=context[:128,None,:].expand(-1,4,-1).reshape(-1,context.shape[1])
            joint,_=model.forward_physical(z,ctx)
            torch.testing.assert_close(c.photo_module.photo(model,z,ctx),joint[:,2:],rtol=0,atol=0)
        checks['photo_equals_joint_exactly']=True;c.write(out/'checks.json',checks)
        def probe():
            result={'nll':nll_pass(model,vt,vc)}
            for split,pop in probes.items():
                result[split]=evaluate_guard_response(model,pop,np.arange(len(pop.pairs)),draws=tcfg['probe_draws'],groups=tcfg['probe_groups'],batch_size=1024,seed=202611068201)
            return result
        if not (c.ROOT/'initial_probe.json').exists():c.write(c.ROOT/'initial_probe.json',probe())
        optimizer=torch.optim.AdamW(model.parameters(),lr=tcfg['learning_rate'],weight_decay=0.)
        history=[];start=0;resume=out/'resume.pt'
        if resume.exists():
            d=torch.load(resume,map_location='cuda',weights_only=False);assert d['provenance']==provenance and d['arm']==arm
            model.load_state_dict(d['model']);optimizer.load_state_dict(d['optimizer']);history=d['history'];start=d['step']
        c.write(out/'protocol.json',{'arm':arm,'provenance':provenance,'moment_weight':cfg['moment_weight'][arm],'response_weight':cfg['response_weight'],'gpu':torch.cuda.get_device_name(),'job':os.environ['SLURM_JOB_ID'],'checks':checks})
        started=time.monotonic();print('TRAIN_READY',arm,'start',start,'checks',checks,flush=True)
        for step in range(start,tcfg['steps']):
            rows=torch.as_tensor(np.random.default_rng(202611068401+step).integers(len(values),size=tcfg['nll_batch']),device='cuda')
            optimizer.zero_grad(set_to_none=True);loss=-model.log_prob(values[rows],context[rows]).mean();assert bool(torch.isfinite(loss));loss.backward()
            guard=sampled_guard_response_backward(model,response,tcfg['pairs_per_group'],tcfg['draws'],chunk_size=tcfg['chunk'],seed=202611068501+100003*step,weight=cfg['response_weight'],independent_pairs=True)
            extra=sampled_guard_response_backward(model,aux,tcfg['pairs_per_group'],tcfg['draws'],chunk_size=tcfg['chunk'],seed=cfg['moment_seed']+100003*step,weight=cfg['moment_weight'][arm],independent_pairs=True)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tcfg['clip'],error_if_nonfinite=True);optimizer.step()
            history.append({'step':step+1,'nll':float(loss.detach()),'response_score':guard['loss'],'moment_score':extra['loss'],'moment_replicas':extra['group_means'].tolist(),'gradient_norm':float(norm)})
            if (step+1)%8==0:
                temp=out/'resume.tmp';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'history':history,'provenance':provenance,'arm':arm},temp);temp.replace(resume)
                c.write(out/'history.json',history);print('STEP',arm,step+1,'seconds',round(time.monotonic()-started,1),'moment',extra['loss'],flush=True)
        endpoint=probe();c.write(out/'probe.json',endpoint)
        if arm=='control':
            reference=c.c.ROOT/'neighbour/nine.pt';old_model,_,_,_=c.m.load(reference)
            with torch.no_grad():
                difference=(model.log_prob(vt[:1024],vc[:1024])-old_model.log_prob(vt[:1024],vc[:1024])).abs()
                replay={'reference_sha256':c.sha(reference),'max_log_density_difference':float(difference.max()),'mean_log_density_difference':float(difference.mean())}
            c.write(out/'previous_control_replay.json',replay);del old_model
        c.m.save(out/'final.pt',model,condition,target,artifact['config'],{'provenance':provenance,'arm':arm,'step':tcfg['steps'],'moment_weight':cfg['moment_weight'][arm],'response_weight':cfg['response_weight']})
        loaded,_,_,_=c.m.load(out/'final.pt')
        with torch.no_grad():torch.testing.assert_close(model.log_prob(vt[:512],vc[:512]),loaded.log_prob(vt[:512],vc[:512]),rtol=0,atol=0)
        c.write(out/'completed.json',{'arm':arm,'sha256':c.sha(out/'final.pt'),'reload_exact':True,'provenance':provenance,'seconds':time.monotonic()-started,'step':tcfg['steps']})
        print('DONE_TRAIN',arm,time.monotonic()-started,flush=True)
        del model,loaded,optimizer,aux;gc.collect();torch.cuda.empty_cache()
    c.write(c.ROOT/'training_completed.json',{'provenance':provenance,'arms':{arm:json.loads((c.ROOT/arm/'completed.json').read_text()) for arm in cfg['arms']}})
if __name__=='__main__':main()
