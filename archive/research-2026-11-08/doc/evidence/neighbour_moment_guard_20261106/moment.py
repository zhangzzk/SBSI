"""Differentiable mass and selected-reference-coefficient supervision."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from sbsi.flow_guard_response import torch_guard_weights,independent_guard_score,sampled_guard_response_backward
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('moment_common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
class Population:
    def __init__(self,context,observed,score,standardizer,stats,cfg):
        self.context=context;self.observed=observed;self.score=score;self.pairs=range(len(context))
        self.cuts=stats['cuts'];self.cfg=cfg
        self.means=torch.as_tensor(standardizer.means,device=context.device,dtype=torch.float64)
        self.scales=torch.as_tensor(standardizer.scales,device=context.device,dtype=torch.float64)
        self.mass=torch.as_tensor(stats['mass'],device=context.device,dtype=torch.float64)
        self.center=torch.as_tensor(stats['selected_score'],device=context.device,dtype=torch.float64)
        self.target=torch.zeros((len(self.cuts),2),device=context.device,dtype=torch.float64)
        self.response_scale=torch.tensor([cfg['moment_scales']['relative_pass_mass'],cfg['moment_scales']['selected_coefficient']],device=context.device,dtype=torch.float64)
    def contributions(self,model,indices,draws,seed):
        ctx=self.context[indices];n=len(indices)
        z=torch.from_numpy(c.portable.latent_bank(n,draws,4,seed)).to(ctx.device).reshape(-1,4)
        expanded=ctx[:,None,:].expand(-1,draws,-1).reshape(n*draws,-1)
        photo=c.photo_module.photo(model,z,expanded).reshape(n,draws,2)
        physical=torch.cat([torch.zeros_like(photo),photo],-1)
        kwargs={k:self.cfg['training'][k] for k in ['radius_softness','magnitude_softness']}
        generated=torch_guard_weights(physical,self.cuts,**kwargs).mean(1)
        actual=self.observed[indices]*self.scales+self.means
        measured=torch_guard_weights(actual,self.cuts,**kwargs)
        difference=(generated-measured)/self.mass
        value=torch.stack([difference,difference*(self.score[indices,None]-self.center)],-1)
        if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite moment contribution')
        return value
def gradient_check(model,pop):
    count,draws,chunk,seed=128,4,64,202611061009
    def gradients():return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    replay=sampled_guard_response_backward(model,pop,count,draws,chunk_size=chunk,seed=seed,weight=1.,independent_pairs=True)
    expected=gradients();model.zero_grad(set_to_none=True);rng=np.random.default_rng(seed)
    groups=[torch.as_tensor(rng.integers(len(pop.pairs),size=count),device=pop.context.device) for _ in range(2)]
    means=[]
    for stream,indices in enumerate(groups):
        values=[pop.contributions(model,indices[lo:lo+chunk],draws,seed+100000000*stream+block) for block,lo in enumerate(range(0,count,chunk))]
        means.append(torch.cat(values).mean(0))
    direct=independent_guard_score(means[0],means[1],pop.target,pop.response_scale);direct.backward()
    actual=gradients();den=sum(float(x.double().square().sum()) for x in actual)
    error=sum(float((x-y).double().square().sum()) for x,y in zip(expected,actual));relative=(error/max(den,1e-30))**.5
    assert den>0 and relative<3e-5,(relative,den)
    assert abs(float(direct)-replay['loss'])<1e-8*max(1.,abs(replay['loss']))
    shape_gradient=sum(float(p.grad.double().square().sum()) for p in model.base.base.shape.parameters() if p.grad is not None)
    assert shape_gradient==0
    model.zero_grad(set_to_none=True)
    return {'relative_gradient_error':relative,'direct_loss':float(direct.detach()),'replayed_loss':replay['loss'],'shape_factor_gradient_squared_norm':shape_gradient,'independent_rows':True}
