"""Symmetric, exact decomposition under a fixed common-latent coupling."""
import json,hashlib
import numpy as np
import torch
from pathlib import Path
EVENTS=['both selected','radius-only switch','flux-only switch','joint switch','neither selected']

def per_draw(base,new,base_shift,new_shift,radius,flux):
    # Cut comparisons retain original float32 threshold behavior.
    rb,ra=base[...,2]>radius,new[...,2]>radius
    fb,fa=base[...,3]>flux,new[...,3]>flux
    ib,ia=rb&fb,ra&fa
    switch=ib!=ia
    event=torch.full_like(ib,4,dtype=torch.long)
    event[ib&ia]=0
    event[switch&(rb!=ra)&(fb==fa)]=1
    event[switch&(rb==ra)&(fb!=fa)]=2
    event[switch&(rb!=ra)&(fb!=fa)]=3
    assert not bool((switch&(event==4)).any())
    b=torch.cat([base[...,:2].double(),base_shift.double()],-1)
    a=torch.cat([new[...,:2].double(),new_shift.double()],-1)
    ib,ia=ib.double()[...,None],ia.double()[...,None]
    membership=(ia-ib)*(a+b)*.5
    value=(ia+ib)*(a-b)*.5
    return event,torch.cat([ib,ia,ib*b,ia*a,membership,value],-1)

def terms(raw):
    """raw: (..., leg, cut, bin, event, statistic); use global leg denominators."""
    n0=raw[...,0].sum((-2,-1),keepdims=True)
    n1=raw[...,1].sum((-2,-1),keepdims=True)
    assert np.all(n0>0) and np.all(n1>0)
    inv=.5*(1/n1+1/n0)[...,None]
    membership=raw[...,10:14]*inv
    value=raw[...,14:18]*inv
    norm=.5*(raw[...,2:6]+raw[...,6:10])*(1/n1-1/n0)[...,None]
    direct=raw[...,6:10]/n1[...,None]-raw[...,2:6]/n0[...,None]
    np.testing.assert_allclose(membership+value+norm,direct,atol=2e-14,rtol=1e-10)
    return np.stack([membership,value,norm],-2),direct

class Accumulator:
    def __init__(self,size,score,cfg):
        assert np.isfinite(size).all() and np.isfinite(score).all()
        self.edges=np.quantile(score,[.25,.5,.75])
        self.bins=np.digitize(size,cfg['intrinsic_size_edges_arcsec'])*4+np.digitize(score,self.edges)
        self.size_edges=cfg['intrinsic_size_edges_arcsec'];self.cfg=cfg
        self.cuts=[x for x in cfg['cuts'] if x[0] in cfg['target_cuts']]
        self.arms=cfg['arms'][1:]
        self.raw=torch.zeros((3,2,2,9,100,18),dtype=torch.float64,device='cuda')
        self.direct=torch.zeros((3,2,2,9,18),dtype=torch.float64,device='cuda')
    def add(self,draws,shifts,prob,usable,lo,hi):
        bins=torch.as_tensor(self.bins[lo:hi],device='cuda')[:,None]
        for ai,arm in enumerate(self.arms):
            for leg in [0,1]:
                weights=[torch.as_tensor(x[leg],device='cuda',dtype=torch.float64)[:,None,None] for x in [prob,usable]]
                for ki,(_,radius,mag) in enumerate(self.cuts):
                    event,stats=per_draw(draws['control'][leg],draws[arm][leg],shifts['control'][leg],shifts[arm][leg],radius[1],10**(-.4*(mag-30)))
                    index=(bins*5+event).reshape(-1)
                    for mi,weight in enumerate(weights):
                        values=stats*weight/64
                        self.raw[ai,mi,leg,ki].index_add_(0,index,values.reshape(-1,18))
                        self.direct[ai,mi,leg,ki]+=values.sum((0,1))
    def finish(self,path,sums,oldroot,case,axis,source,pins,decompose_sha):
        raw=self.raw.cpu().numpy().reshape(3,2,2,9,20,5,18);direct=self.direct.cpu().numpy()
        assert np.isfinite(raw).all()
        np.testing.assert_allclose(raw.sum((4,5)),direct,rtol=2e-10,atol=2e-7)
        replay=0.;double_response_error=0.
        for arm,new in sums.items():
            with np.load(oldroot/'constant'/arm/f'axis{axis}'/f'case{case}.npz') as z:old=z['model']
            np.testing.assert_allclose(new,old,rtol=2e-10,atol=2e-7)
            replay=max(replay,float(np.abs(new-old).max()))
        for ai,arm in enumerate(self.arms):
            for mi in [0,1]:
                for name,sl in [('control',[0,2,3,4,5]),(arm,[1,6,7,8,9])]:
                    got=np.take(direct[ai,mi],sl,axis=-1)
                    ref=sums[name][mi,:,:9].sum(-2)
                    np.testing.assert_allclose(got[...,0],ref[...,0],rtol=2e-10,atol=2e-7)
                    a=(got[0,:,1:]/got[0,:,:1]-got[1,:,1:]/got[1,:,:1])/.04
                    b=(ref[0,:,1:]/ref[0,:,:1]-ref[1,:,1:]/ref[1,:,:1])/.04
                    double_response_error=max(double_response_error,float(np.abs(a-b).max()))
                    np.testing.assert_allclose(a,b,rtol=0,atol=1e-7)
        terms(raw)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.stem+'.tmp.npz')
        np.savez(temp,raw=raw,direct=direct,score_quartiles=self.edges,size_edges=self.size_edges,parent_counts=np.bincount(self.bins,minlength=20),arms=self.arms,cuts=[x[0] for x in self.cuts],events=EVENTS,case=case,axis=axis)
        temp.replace(path)
        receipt={'case':case,'axis':axis,'source':source,'decompose_sha256':decompose_sha,'model_pins':pins,'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'replay_max_abs':replay,'double_accumulation_response_max_abs':double_response_error,'partition_and_decomposition_passed':True}
        path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('LOCALIZATION_PASSED',receipt,flush=True)
