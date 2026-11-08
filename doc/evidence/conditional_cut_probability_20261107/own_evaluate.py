"""Same fixed own-data rows and CRNs for the two matched endpoints."""
import importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from sbsi.flow_guard_response import torch_guard_weights
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cutprob_setup',HERE/'common.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);c=s.c
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    cfg=c.manifest();oldcfg=json.loads(Path(cfg['selection_diagnostic_manifest']).read_text());prep=Path(cfg['prepared_root']);old=Path(cfg['original_prepared_root'])
    receipt=json.loads((prep/'completed.json').read_text());names=cfg['arms'];models={};pins={};target=None;condition=None
    for name in names:
        d=json.loads((c.ROOT/name/'completed.json').read_text());p=c.ROOT/name/'final.pt';assert c.sha(p)==d['sha256']
        model,cc,tt,_=c.m.load(p);assert cc.use_score;models[name]=model;pins[name]=d['sha256'];target=tt;condition=cc
    for si,split in enumerate(['train','validation']):
        out=c.ROOT/f'own_{split}.npz';done=out.with_suffix('.json')
        if done.exists():
            r=json.loads(done.read_text());assert r['pins']==pins and r['source_sha256']==c.sha(__file__) and r['data_sha256']==c.sha(out);continue
        assert not out.exists();info=receipt['splits'][split];cases=info['cases'];ncase=len(cases);n=oldcfg['rows_per_split']
        indices=(np.sort(np.random.default_rng(oldcfg['train_sample_seed']).choice(info['rows'],n,replace=False)) if split=='train' else np.load(oldcfg['validation_indices']))
        ends=np.cumsum([r['rows'] for r in info['matching']]);assert ends[-1]==info['rows'];record=np.searchsorted(ends,indices,side='right')
        case=np.array([cases.index(info['matching'][i]['case']) for i in record]);leg=np.array([0 if info['matching'][i]['leg']=='g0' else 1 for i in record]);group=case*2+leg
        score=np.load(prep/split/'raw_score.npy',mmap_mode='r')[indices]
        if split=='train':
            x=np.load(prep/split/'context.npy',mmap_mode='r')[indices];y=np.load(old/split/'target.npy',mmap_mode='r')[indices]
        else:x=np.load(prep/split/'nll_context.npy');y=np.load(old/split/'nll_target.npy')
        np.testing.assert_array_equal(x[:,-1],((score-condition.mean)/condition.scale).astype(np.float32));physical=target.inverse_transform_array(y)
        nc=len(oldcfg['cuts']);probability_sums=np.zeros((len(names),4,ncase*2,nc,10));factor_density=np.zeros((len(names),ncase*2,3));sums=np.zeros((len(names),2,3,ncase*2,nc,5));density=np.zeros((len(names),ncase*2,2));checks={};started=time.monotonic()
        kwargs={k:cfg['training'][k] for k in ['radius_softness','magnitude_softness']}
        with torch.no_grad():
            for block,lo in enumerate(range(0,n,1024)):
                hi=min(n,lo+1024);nb=hi-lo;ctx=torch.as_tensor(x[lo:hi],device='cuda');obs=torch.as_tensor(y[lo:hi],device='cuda')
                phys=torch.as_tensor(physical[lo:hi],device='cuda');observed=[c.photo_module.cuts(phys[:,2:],oldcfg).double().cpu().numpy(),torch_guard_weights(phys,cfg['regularization_cuts'],**kwargs).cpu().numpy()]
                z=torch.from_numpy(c.portable.latent_bank(nb,64,4,oldcfg['latent_seed']+si*100000000+block)).to('cuda').reshape(-1,4)
                expanded=ctx[:,None,:].expand(-1,64,-1).reshape(len(z),-1);ids=group[lo:hi];q=score[lo:hi]
                for ai,name in enumerate(names):
                    model=models[name];joint,_=model.forward_physical(z,expanded);joint=joint.reshape(nb,64,4)
                    assert bool(torch.isfinite(joint).all())
                    p=joint[:,:,2:]
                    if name not in checks:
                        photo_only=c.photo_module.photo(model,z,expanded);torch.testing.assert_close(p.reshape(-1,2),photo_only,rtol=0,atol=0);checks[name]=True
                    pp=torch.cat([torch.zeros_like(p),p],-1)
                    masks=[c.photo_module.cuts(p,oldcfg),torch_guard_weights(pp,cfg['regularization_cuts'],**kwargs)]
                    lp=model.log_prob(obs,ctx);assert bool(torch.isfinite(lp).all())
                    photo_lp=c.photo_module.photo_logprob(model,obs,ctx)
                    for k,values in enumerate([np.ones(nb),-photo_lp.cpu().numpy(),-(lp-photo_lp).cpu().numpy()]):factor_density[ai,:,k]+=np.bincount(ids,weights=values,minlength=ncase*2)
                    density[ai,:,0]+=np.bincount(ids,minlength=ncase*2);density[ai,:,1]+=np.bincount(ids,weights=-lp.double().cpu().numpy(),minlength=ncase*2)
                    extra_masks=[torch_guard_weights(pp,cfg['regularization_cuts'],radius_softness=w,magnitude_softness=w) for w in [.025,.1]]
                    extra_observed=[torch_guard_weights(phys,cfg['regularization_cuts'],radius_softness=w,magnitude_softness=w).cpu().numpy() for w in [.025,.1]]
                    for kind,(mask,target_values) in enumerate(zip(masks+extra_masks,observed+extra_observed)):
                        qa=mask[:,c.photo_module.BANKS[1]].double().mean(1).cpu().numpy()
                        qb=mask[:,c.photo_module.BANKS[2]].double().mean(1).cpu().numpy()
                        qp=.5*(qa+qb)
                        pred_shape=torch.einsum('ndk,ndj->nkj',mask.double(),joint[:,:,:2].double()).cpu().numpy()/64
                        actual_shape=target_values[:,:,None]*physical[lo:hi,None,:2]
                        values=[np.ones_like(qa),target_values,qp,(qa-target_values)*(qb-target_values),.25*(qa-qb)**2,
                                actual_shape[:,:,0],actual_shape[:,:,1],pred_shape[:,:,0],pred_shape[:,:,1],(qp-target_values)**2]
                        for cut in range(nc):
                            for k,vv in enumerate(values):probability_sums[ai,kind,:,cut,k]+=np.bincount(ids,weights=vv[:,cut],minlength=ncase*2)
                    for kind,mask in enumerate(masks):
                        for bank,draws in enumerate(c.photo_module.BANKS):
                            probability=mask[:,draws].double().mean(1).cpu().numpy()
                            for cut in range(nc):
                                for k,weights in enumerate([np.ones(nb),observed[kind][:,cut],observed[kind][:,cut]*q,probability[:,cut],probability[:,cut]*q]):
                                    sums[ai,kind,bank,:,cut,k]+=np.bincount(ids,weights=weights,minlength=ncase*2)
                if block%100==0:print('OWN_PROGRESS',split,block,round(time.monotonic()-started,1),flush=True)
        np.testing.assert_allclose(sums[:,:,0],.5*(sums[:,:,1]+sums[:,:,2]),rtol=2e-12,atol=2e-8)
        np.testing.assert_array_equal(sums[...,:3],np.broadcast_to(sums[:1,:,:1,...,:3],sums[...,:3].shape))
        if split=='validation':
            for ai,name in enumerate(names):
                expected=json.loads((c.ROOT/name/'probe.json').read_text())['nll']['nll'];actual=density[ai,:,1].sum()/n;assert abs(actual-expected)<2e-5
        np.testing.assert_allclose(probability_sums[...,9],probability_sums[...,3]+probability_sums[...,4],rtol=1e-10,atol=1e-8)
        np.savez(out,probability_sums=probability_sums.reshape(len(names),4,ncase,2,nc,10),probability_kinds=['hard','soft_0.05','soft_0.025','soft_0.1'],factor_density=factor_density.reshape(len(names),ncase,2,3),sums=sums.reshape(len(names),2,3,ncase,2,nc,5),density=density.reshape(len(names),ncase,2,2),cases=cases,arms=names,cuts=[x[0] for x in oldcfg['cuts']],kinds=['hard','soft'])
        c.write(done,{'pins':pins,'source_sha256':c.sha(__file__),'data_sha256':c.sha(out),'rows':n,'cases':len(cases),'photo_equals_joint_exactly':checks,'score_alignment_exact':True,'half_bank_identity':True,'seconds':time.monotonic()-started})
        print('DONE_OWN',split,flush=True)
    c.module('moment_report_own',HERE/'report.py').own_report()
if __name__=='__main__':main()
