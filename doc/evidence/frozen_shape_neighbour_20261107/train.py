"""Train only the photometry factor; the frozen shape remains differentiable in its inputs."""
import gc,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixedshape_setup',H/'common.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
c=s.c
prior=s.bound('train')
moment=prior.moment
swap=s.load('fixedshape_direct_density',H.parent/'neighbour_factor_swap_20261107/prepare.py')
PREFIX='base.base.photometry.'
def main():
    assert os.environ.get('SLURM_JOB_ID')
    s.verify();torch.set_num_threads(4);torch.manual_seed(611);torch.cuda.manual_seed_all(611)
    cfg=s.manifest();pin=cfg['initial_checkpoint'];assert c.sha(pin['path'])==pin['sha256']
    p=Path(cfg['target_source']);assert c.sha(p)==cfg['target_sha256'];stats=json.loads(p.read_text())
    assert stats['input_score_alignment_exact'] and stats['cuts']==cfg['regularization_cuts']
    prep=Path(cfg['prepared_root']);old=Path(cfg['original_prepared_root'])
    assert c.sha(prep/'completed.json')==stats['provenance']['prepared_sha256']
    tc=cfg['training'];tensor=prior.tensor
    provenance={'manifest':c.sha(H/'manifest.json'),'train':c.sha(__file__),'targets':c.sha(p),'initial':pin}
    print('LOAD',torch.cuda.get_device_name(),flush=True)
    ctx=tensor(prep/'train/context.npy',torch.float32);obs=tensor(old/'train/target.npy',torch.float64)
    score=tensor(prep/'train/raw_score.npy',torch.float64);assert len(score)==len(obs)==len(ctx)==stats['rows']
    vc=tensor(prep/'validation/nll_context.npy',torch.float32);vt=tensor(old/'validation/nll_target.npy',torch.float64)
    pops={}
    def pop(x,ids,design,split):
        return c.g.Population(x,ids,design,json.loads((old/split/'statistics.json').read_text()),list(range(9)),tc['relative_response_scale'])
    guard=pop(ctx,tensor(old/'train/pairs.npy',torch.long),tensor(old/'train/design.npy',torch.float64),'train')
    for name in ['train','validation']:
        x=tensor(prep/name/'probe_context.npy',torch.float32)
        pops[name]=pop(x,torch.arange(len(x),device='cuda').reshape(-1,2),tensor(old/name/'probe_design.npy',torch.float64),name)
    origin,_,_,_=c.m.load(pin['path'])
    with torch.no_grad():fixed_density=swap.conditional_shape_density(origin,vt[:1024],vc[:1024])[0]
    def probe(model):
        d={'nll':prior.nll_pass(model,vt,vc)}
        for name,population in pops.items():
            d[name]=prior.evaluate_guard_response(model,population,np.arange(len(population.pairs)),draws=tc['probe_draws'],groups=tc['probe_groups'],batch_size=1024,seed=202611068201)
        return d
    # Same starting model and probe recipe as the prior control endpoint: reuse its receipt.
    prev=Path(pin['path']).parent
    assert json.loads((prev/'completed.json').read_text())['sha256']==pin['sha256']
    c.write(s.ROOT/'initial_probe.json',json.loads((prev/'probe.json').read_text()))
    def identity(model,require_changed=True):
        sd=model.state_dict();ref=origin.state_dict();assert sd.keys()==ref.keys()
        for k in sd:
            if not k.startswith(PREFIX):assert torch.equal(sd[k],ref[k]),k
        with torch.no_grad():
            q=swap.conditional_shape_density(model,vt[:1024],vc[:1024])[0]
            torch.testing.assert_close(q,fixed_density,rtol=0,atol=0)
        changed=sum(not torch.equal(sd[k],ref[k]) for k in sd if k.startswith(PREFIX))
        assert not require_changed or changed>0
        return {'identical_nonphotometry_tensors':sum(not k.startswith(PREFIX) for k in sd),'changed_photometry_tensors':changed,'fixed_conditional_density_max_abs':float((q-fixed_density).abs().max()),'checked_rows':len(q)}
    for arm in cfg['arms']:
        out=s.ROOT/arm;out.mkdir(exist_ok=True);end=out/'completed.json'
        if end.is_file():
            d=json.loads(end.read_text());assert d['provenance']==provenance and c.sha(out/'final.pt')==d['sha256'];continue
        model,condition,target,art=c.m.load(pin['path'])
        aux=moment.Population(ctx,obs,score,target,stats,cfg)
        respcheck=c.g.gradient_check(model,guard)
        model.zero_grad(set_to_none=True)
        for k,par in model.named_parameters():par.requires_grad_(k.startswith(PREFIX))
        phot=list(model.base.base.photometry.parameters());fixed=[p for p in model.parameters() if not p.requires_grad]
        assert phot and fixed and set(map(id,phot))=={id(p) for p in model.parameters() if p.requires_grad}
        checks={'moment':moment.gradient_check(model,aux),'response_before_freeze':respcheck,'initial_identity':identity(model,False)}
        optimizer=torch.optim.AdamW(phot,lr=tc['learning_rate'],weight_decay=0.)
        history=[];resume=out/'resume.pt'
        if resume.exists():
            d=torch.load(resume,map_location='cuda',weights_only=False);assert d['provenance']==provenance and d['arm']==arm
            model.load_state_dict(d['model']);optimizer.load_state_dict(d['optimizer']);history=d['history'];identity(model)
        print('TRAIN_READY',arm,'step',len(history),'checks',checks,flush=True)
        for step in range(len(history),tc['steps']):
            rows=torch.as_tensor(np.random.default_rng(202611068401+step).integers(len(obs),size=tc['nll_batch']),device='cuda')
            optimizer.zero_grad(set_to_none=True);loss=-model.log_prob(obs[rows],ctx[rows]).mean();assert bool(torch.isfinite(loss));loss.backward()
            def backward(population,seed,weight):
                return prior.sampled_guard_response_backward(model,population,tc['pairs_per_group'],tc['draws'],chunk_size=tc['chunk'],seed=seed+100003*step,weight=weight,independent_pairs=True)
            a=backward(guard,202611068501,cfg['response_weight']);b=backward(aux,cfg['moment_seed'],cfg['moment_weight'][arm])
            assert all(p.grad is None for p in fixed)
            norm=torch.nn.utils.clip_grad_norm_(phot,tc['clip'],error_if_nonfinite=True);optimizer.step()
            history.append(dict(step=step+1,nll=float(loss.detach()),guard=a['loss'],moment=b['loss'],photo_gradient_norm=float(norm)))
            if not (step+1)%8:
                tmp=out/'resume.tmp';torch.save(dict(provenance=provenance,arm=arm,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history),tmp);tmp.replace(resume)
                print('STEP',arm,step+1,flush=True)
        checks['trained_identity']=identity(model)
        c.m.save(out/'final.pt',model,condition,target,art['config'],dict(provenance=provenance,step=tc['steps'],arm=arm))
        loaded,_,_,_=c.m.load(out/'final.pt');checks['reloaded_identity']=identity(loaded)
        with torch.no_grad():torch.testing.assert_close(loaded.log_prob(vt[:1024],vc[:1024]),model.log_prob(vt[:1024],vc[:1024]),rtol=0,atol=0)
        c.write(out/'history.json',history);c.write(out/'probe.json',probe(loaded));c.write(out/'checks.json',checks)
        c.write(end,dict(arm=arm,provenance=provenance,sha256=c.sha(out/'final.pt'),reload_exact=True,checks=checks))
        print('DONE_TRAIN',arm,checks['reloaded_identity'],flush=True)
        del model,optimizer,loaded,aux,phot,fixed
        gc.collect();torch.cuda.empty_cache()
    c.write(s.ROOT/'training_completed.json',{'provenance':provenance,'arms':{arm:json.loads((s.ROOT/arm/'completed.json').read_text()) for arm in cfg['arms']}})
