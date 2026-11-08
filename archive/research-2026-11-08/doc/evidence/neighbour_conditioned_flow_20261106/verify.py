"""Verify added conditioning preserves initialization and the normalized density."""
import importlib.util, math, os, tempfile
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neighbour_verify_model',HERE/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.c
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(1);torch.manual_seed(123)
    old,_,_,_=c.fm.load(c.BASE,device='cpu')
    model,condition,target,config=m.initialize(.1,.2,True,device='cpu')
    x=torch.from_numpy(np.load(c.OLD/'validation/nll_context.npy',mmap_mode='r')[:64].copy())
    y=torch.from_numpy(np.load(c.OLD/'validation/nll_target.npy',mmap_mode='r')[:64].copy())
    score=torch.randn(len(x),1);context=torch.cat([x,score],-1);z=torch.randn(len(x),4)
    with torch.no_grad():
        torch.testing.assert_close(old.log_prob(y,x),model.log_prob(y,context),rtol=0,atol=0)
        torch.testing.assert_close(old.forward_physical(z,x)[0],model.forward_physical(z,context)[0],rtol=0,atol=0)
    loss=-model.log_prob(y,context).mean();loss.backward()
    extra_grad=sum(float(layer.net[0].extra_weight.grad.double().square().sum()) for flow in [model.base.base.photometry,model.base.base.shape] for layer in flow.layers)
    assert extra_grad>1e-10
    with torch.no_grad():
        for flow in [model.base.base.photometry,model.base.base.shape]:
            for layer in flow.layers:layer.net[0].extra_weight.normal_(0,.005)
    physical,ld=model.forward_physical(z,context)
    means=torch.as_tensor(target.means,dtype=torch.float64);scales=torch.as_tensor(target.scales,dtype=torch.float64)
    lp=model.log_prob((physical-means)/scales,context)
    expected=-.5*(z.double().square()+math.log(2*math.pi)).sum(1)-ld+scales.log().sum()
    torch.testing.assert_close(lp,expected,atol=5e-5,rtol=5e-6)
    assert bool((physical[:,:2].square().sum(1)<1).all() and (physical[:,2:]>0).all())
    changed=model.forward_physical(z,torch.cat([x,score+1],-1))[0]
    assert float((changed-physical).detach().abs().max())>1e-6
    with tempfile.TemporaryDirectory(dir=c.ROOT) as folder:
        path=Path(folder)/'test.pt';m.save(path,model,condition,target,config,{})
        loaded,cc,_,_=m.load(path,device='cpu')
        torch.testing.assert_close(loaded.log_prob(y,context),model.log_prob(y,context),rtol=0,atol=0)
        torch.testing.assert_close(loaded.forward_physical(z,context)[0],physical,rtol=0,atol=0)
        assert (cc.mean,cc.scale,cc.use_score)==(condition.mean,condition.scale,condition.use_score)
        for key,value in condition.base.to_state().items():np.testing.assert_array_equal(cc.base.to_state()[key],value)
    transport=model.base.base.double();ctx=context[:1].double();u=torch.tensor([[.2,-.3,.4,.1]],dtype=torch.float64)
    v,j=transport.forward(u,ctx);back,ij=transport.inverse(v,ctx)
    torch.testing.assert_close(back,u,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(j+ij,torch.zeros_like(j),rtol=0,atol=1e-10)
    jac=torch.autograd.functional.jacobian(lambda a:transport.forward(a[None],ctx)[0][0],u[0])
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],j[0],rtol=1e-9,atol=1e-9)
    result={'passed':True,'initial_log_density_exact':True,'initial_physical_draws_exact':True,
        'extra_weight_gradient_squared_norm':extra_grad,'nonzero_feature_effect':True,
        'physical_density_max_error':float((lp-expected).detach().abs().max()),
        'inverse_and_jacobian':True,'reload_exact':True,'model_source_sha256':c.sha(HERE/'model.py')}
    c.write(c.ROOT/'verification.json',result);c.write(HERE/'verification.json',result);print(result,flush=True)
if __name__=='__main__':main()
