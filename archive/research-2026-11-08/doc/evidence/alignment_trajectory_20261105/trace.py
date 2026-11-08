"""Fixed-context profile and local joint-density diagnostics; no training loss."""
import numpy as np
import pandas as pd
import torch
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import physical_context_draws

EPOCHS=[0,1,2,5,10,15,20,25,30,35,40,50,60,70,80,90,100,110,120,130,140]

def prepare(data,condition,target,seed,output,label):
    raw=data['context_raw'][data['pairs'][:,0]]
    eligible=np.flatnonzero((raw[:,4]>=.15)&(raw[:,4]<.25))
    ix=np.sort(np.random.default_rng(seed).choice(eligible,min(160_000,len(eligible)),replace=False))
    raw=raw[ix];measured=data['pair_zero_target'][ix]
    cases,ci=np.unique(data['pair_case'][ix],return_inverse=True)
    strata=(np.digitize(raw[:,2],[1,2,4])*4+np.digitize(raw[:,3],[23,24,25]))*3+np.digitize(np.linalg.norm(raw[:,:2],axis=1),[.2,.5])
    np.savez(output/f'{label}_sample.npz',pair_indices=ix,cases=cases,case_index=ci,raw=raw,measured=measured,stratum=strata)
    return {'context':torch.as_tensor(condition.transform_frame(pd.DataFrame(raw,columns=FLOW_FEATURES)),device='cuda'),
            'target':torch.as_tensor(target.transform_array(measured),device='cuda'),
            'physical':torch.as_tensor(measured,device='cuda'),
            'intrinsic':torch.as_tensor(raw[:,:2],dtype=torch.float64,device='cuda'),
            'ci':torch.as_tensor(ci,device='cuda'),'strata':torch.as_tensor(strata,device='cuda'),
            'cases':cases,'input_counts':np.bincount(strata,minlength=48),'sample_seed':seed}

def moments(v,intrinsic):
    draws=v.shape[1]
    dot=(v[...,:2]*intrinsic[:,None,:]).sum(-1)
    power=intrinsic.square().sum(-1)[:,None].expand_as(dot)
    return torch.stack([torch.ones_like(dot),power,dot,(dot<0).double()],-1)/draws

def profile_add(out,v,intrinsic,group):
    rad=torch.bucketize(v[...,2].contiguous(),torch.arange(2.,5.001,.1,device=v.device,dtype=torch.float64),right=True)
    cell=(group[:,None]*32+rad).reshape(-1)
    m=moments(v,intrinsic)
    for j,flux in enumerate([100.,10**(-.4*(25.8-30))]):
        out[j].index_add_(0,cell,(m*(v[...,3]>flux)[...,None]).reshape(-1,4))

def diagnose(model,d,output,epoch,label):
    model.eval();nc=len(d['cases']);n=len(d['context'])
    profiles=torch.zeros((3,2,nc*32,4),dtype=torch.float64,device='cuda')
    strata=torch.zeros((3,2,48*32,4),dtype=torch.float64,device='cuda')
    density=torch.zeros((2,nc*32,3),dtype=torch.float64,device='cuda')
    uncut=torch.zeros((nc,3),dtype=torch.float64,device='cuda')
    with torch.no_grad():
        for block,lo in enumerate(range(0,n,2048)):
            s=slice(lo,min(n,lo+2048));c=d['context'][s];intr=d['intrinsic'][s];ci=d['ci'][s]
            measured=d['physical'][s,None,:]
            for b in range(3):
                v=measured if b==0 else physical_context_draws(model,[c],16,202611055700+d['sample_seed']+1000003*(b-1)+block)[0]
                assert bool(torch.isfinite(v).all())
                profile_add(profiles[b],v,intr,ci)
                profile_add(strata[b],v,intr,d['strata'][s])
            lp=model.log_prob(d['target'][s],c).double()
            assert bool(torch.isfinite(lp).all())
            cols=torch.stack([torch.ones_like(lp),lp,lp.square()],-1)
            uncut.index_add_(0,ci,cols)
            rad=torch.bucketize(measured[:,0,2].contiguous(),torch.arange(2.,5.001,.1,device='cuda',dtype=torch.float64),right=True)
            for j,flux in enumerate([100.,10**(-.4*(25.8-30))]):
                density[j].index_add_(0,ci*32+rad,cols*(measured[:,0,3]>flux)[:,None])
    p=profiles.cpu().numpy().reshape(3,2,nc,32,4)
    q=strata.cpu().numpy().reshape(3,2,48,32,4)
    np.testing.assert_allclose(p.sum(2),q.sum(2),rtol=1e-10,atol=1e-8)
    np.savez(output/f'{label}_epoch{epoch:03d}.npz',profile=p,strata=q,density=density.cpu().numpy().reshape(2,nc,32,3),
             uncut_density=uncut.cpu().numpy(),cases=d['cases'],input_counts=d['input_counts'])
    sums=p.sum(2);sim=sums[0,:,8];mod=sums[1:,:,8].mean(0)
    align_sim=np.divide(sim[:,2],sim[:,1],out=np.full(2,np.nan),where=sim[:,1]>0)
    align_mod=np.divide(mod[:,2],mod[:,1],out=np.full(2,np.nan),where=mod[:,1]>0)
    return {'alignment_error_bin8':[float(x) if np.isfinite(x) else None for x in (align_mod-align_sim)],'observed_count_bin8':sim[:,0].tolist(),
            'generated_count_bin8':mod[:,0].tolist(),'small_joint_nll':float(-uncut[:,1].sum()/uncut[:,0].sum())}
