"""Train only the photometry factor; the frozen shape remains differentiable in its inputs."""
import gc,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cutprob_setup',H/'common.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
c=s.c
prior=s.bound('train')
probability=s.load('cutprob_loss',H/'probability.py')
swap=s.load('cutprob_direct_density',H.parent/'neighbour_factor_swap_20261107/prepare.py')
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
    provenance={'manifest':c.sha(H/'manifest.json'),'train':c.sha(__file__),'probability':c.sha(H/'probability.py'),'source_pins':c.sha(H/'source_pins.json'),'targets':c.sha(p),'initial':pin}
    print('LOAD',torch.cuda.get_device_name(),flush=True)
    ctx=tensor(prep/'train/context.npy',torch.float32);obs=tensor(old/'train/target.npy',torch.float64)
    assert len(obs)==len(ctx)==stats['rows']
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
    # Weight calibration sees only training rows at the untouched initial model.
    calibration_file=s.ROOT/'weight_calibration.json'
    if calibration_file.exists():
        calibration=json.loads(calibration_file.read_text());assert calibration['provenance']==provenance
    else:
        calibration_model,_,calibration_target,_=c.m.load(pin['path'])
        for name,par in calibration_model.named_parameters():par.requires_grad_(name.startswith(PREFIX))
        parameters=list(calibration_model.base.base.photometry.parameters())
        auxcal=probability.Population(ctx,obs,calibration_target,cfg,c)
        checks=probability.gradient_check(calibration_model,auxcal)
        samples=[];setting=cfg['probability_loss']
        def snapshot():return [p.grad.detach().double().clone() if p.grad is not None else torch.zeros_like(p,dtype=torch.float64) for p in parameters]
        for batch in range(setting['calibration_batches']):
            seed=setting['calibration_seed']+100003*batch
            rows=torch.as_tensor(np.random.default_rng(seed).integers(len(obs),size=tc['nll_batch']),device='cuda')
            calibration_model.zero_grad(set_to_none=True)
            nll=-calibration_model.log_prob(obs[rows],ctx[rows]).mean();nll.backward()
            guard_result=prior.sampled_guard_response_backward(calibration_model,guard,tc['pairs_per_group'],tc['draws'],chunk_size=tc['chunk'],seed=seed+10000000,weight=cfg['response_weight'],independent_pairs=True)
            base_gradient=snapshot();calibration_model.zero_grad(set_to_none=True)
            result=auxcal.backward(calibration_model,seed=seed+20000000,weight=1.)
            extra_gradient=snapshot()
            bn=sum(float(g.square().sum()) for g in base_gradient);an=sum(float(g.square().sum()) for g in extra_gradient)
            dot=sum(float((a*b).sum()) for a,b in zip(base_gradient,extra_gradient))
            assert bn>0 and an>0 and np.isfinite([bn,an,dot]).all()
            samples.append(dict(batch=batch,nll=float(nll.detach()),guard=guard_result['loss'],probability_score=result['loss'],base_norm=bn**.5,extra_norm=an**.5,cosine=dot/(bn*an)**.5))
        weight=setting['gradient_fraction']*(sum(v['base_norm']**2 for v in samples)/sum(v['extra_norm']**2 for v in samples))**.5
        assert np.isfinite(weight) and weight>0
        calibration=dict(provenance=provenance,weight=weight,samples=samples,gradient_check=checks,training_only=True,parameter_updates=0)
        c.write(calibration_file,calibration);c.write(H/'weight_calibration.json',calibration)
        print('WEIGHT_CALIBRATION',json.dumps(calibration),flush=True)
        del calibration_model,parameters,auxcal,base_gradient,extra_gradient
        gc.collect();torch.cuda.empty_cache()
    for arm in cfg['arms']:
        out=s.ROOT/arm;out.mkdir(exist_ok=True);end=out/'completed.json'
        if end.is_file():
            d=json.loads(end.read_text());assert d['provenance']==provenance and c.sha(out/'final.pt')==d['sha256'];continue
        model,condition,target,art=c.m.load(pin['path'])
        aux=probability.Population(ctx,obs,target,cfg,c)
        respcheck=c.g.gradient_check(model,guard)
        model.zero_grad(set_to_none=True)
        for k,par in model.named_parameters():par.requires_grad_(k.startswith(PREFIX))
        phot=list(model.base.base.photometry.parameters());fixed=[p for p in model.parameters() if not p.requires_grad]
        assert phot and fixed and set(map(id,phot))=={id(p) for p in model.parameters() if p.requires_grad}
        checks={'probability':probability.gradient_check(model,aux),'response_before_freeze':respcheck,'initial_identity':identity(model,False),'probability_weight':calibration['weight'] if arm=='weighted' else 0.}
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
            a=backward(guard,202611068501,cfg['response_weight'])
            base_norm=sum(float(p.grad.detach().double().square().sum()) for p in phot if p.grad is not None)**.5
            b=aux.backward(model,seed=cfg['probability_loss']['training_seed']+100003*step,weight=checks['probability_weight'])
            assert all(p.grad is None for p in fixed)
            norm=torch.nn.utils.clip_grad_norm_(phot,tc['clip'],error_if_nonfinite=True);optimizer.step()
            history.append(dict(step=step+1,nll=float(loss.detach()),guard=a['loss'],probability_score=b['loss'],probability_weight=checks['probability_weight'],base_photo_gradient_norm=base_norm,photo_gradient_norm=float(norm)))
            if not (step+1)%8:
                tmp=out/'resume.tmp';torch.save(dict(provenance=provenance,arm=arm,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history),tmp);tmp.replace(resume)
                print('STEP',arm,step+1,'probability_score',b['loss'],'photo_gradient_norm',float(norm),flush=True)
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
