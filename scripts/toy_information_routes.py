"""Which Eulerian information route is unreliable: analytic Louis, FD, or both?
Ground truth = brute-force -d2/dg2 log p(xhat|g) by direct quadrature."""
import numpy as np, sys
sys.path.insert(0,"/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b")
from scipy.special import logsumexp
from sbs_shear.posterior_shape import make_e_grid
from sbs_shear.score_inference import ShapeScoreNodes, SmoothRadialPrior, scores_from_loglike
from sbs_shear.shear_map import apply_shear_to_ellipticity
from sbs_shear.lagrangian_score import curve_derivatives, score_and_information
A,SIG,N=0.35,0.25,400
xh=np.random.default_rng(5).normal(0,0.30,(N,2))
grid,_=make_e_grid(n=61,emax=0.96,rmax=0.95)
def loglike_at(g):
    e1,e2=apply_shear_to_ellipticity(grid[:,0],grid[:,1],g[0],g[1])
    mu=A*np.stack([e1,e2],1); d=xh[:,None,:]-mu[None,:,:]
    return -0.5*np.sum(d**2,axis=2)/SIG**2
ll=loglike_at((0.,0.))
def run(nsamp,knots,bins,tag):
    rng=np.random.default_rng(3)
    r=np.abs(rng.normal(0,0.25,nsamp)); r=r[r<0.9]; th=rng.uniform(0,2*np.pi,r.size)
    prior=SmoothRadialPrior(r*np.cos(th),r*np.sin(th),n_bins=bins,n_knots=knots,knot_margin=0.02)
    nodes=ShapeScoreNodes(grid,prior,delta=0.01,info_delta=0.0025)
    _,i_an,_=scores_from_loglike(ll.astype(np.float32),nodes,device="cpu",analytic_info=True)
    _,i_fd,_=scores_from_loglike(ll.astype(np.float32),nodes,device="cpu",analytic_info=False)
    def f(t): return loglike_at([t,0.])
    p0,d1,d2=curve_derivatives(f,delta=0.01)
    _,i_lag=score_and_information(p0,d1,d2,log_prior=nodes.log_prior)
    lp=np.where(nodes.support,nodes.log_prior,-np.inf); dd=0.004
    ev=lambda t: logsumexp(loglike_at([t,0.])+lp[None,:],axis=1)
    truth=(-(ev(+dd)-2*ev(0.)+ev(-dd))/dd**2).mean()
    print(f"{tag:30s} TRUTH={truth:+8.4f} | Eul-analytic={i_an[:,0,0].mean():+9.4f} "
          f"Eul-FD={i_fd[:,0,0].mean():+9.4f} Lagrangian={i_lag.mean():+8.4f}")
run(400000,12,120,"healthy")
run(400000, 6,120,"fewer knots k=6")
run( 40000,12,120,"10x fewer samples")
run(400000,24,400,"overfit spline k=24 b=400")
run(400000,30,600,"very overfit k=30 b=600")
