"""Matched shape-fit repeat audit; distinguish sample error and fitter noise."""
import os,json
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/constant_split_20261105/fit_audit')
CASES=[1000,1040,1080,1120];arrays=[];sizes=[];case_reports=[]
for case in CASES:
    legs=[];ids=None;repstats=[]
    for leg in range(2):
        with np.load(D/f'case{case}_leg{leg}.npz') as z:
            if ids is None:ids=z['input_index'].copy()
            else:np.testing.assert_array_equal(ids,z['input_index'])
            g=np.concatenate([z['old'][None],z['new']],axis=0)
            selected=(z['radius']>3)&(z['mag']<25)&np.isfinite(z['radius'])&np.isfinite(z['mag'])
            u=np.isfinite(g).all(2)&((g*g).sum(2)<1)&selected
            legs.append(np.stack([u,np.where(u,g[...,0],0)],-1))
            repstats.append({'leg':leg,'selected_counts':u.sum(1).tolist(),'usability_changed_from_old':[(u[r]!=u[0]).sum().item() for r in [1,2]],'finite_shape_change_rms': [float(np.sqrt(np.mean((g[r][u[r]&u[0]]-g[0][u[r]&u[0]])**2))) for r in [1,2]]})
            pop=int(z['pool_size'])
    a=np.stack(legs,axis=1) # repeat,leg,object,q
    arrays.append(a);sizes.append((len(ids),pop))
    sums=a.sum(2);resp=np.diff((sums[...,1]/sums[...,0])[:,::-1],axis=1).ravel()/.04
    case_reports.append({'case':case,'sample':len(ids),'pool':pop,'response_old_repeat0_repeat1':resp.tolist(),'legs':repstats})
factors=np.array([pop/n for n,pop in sizes]);totals=np.stack([a.sum(2)*f for a,f in zip(arrays,factors)])
w=np.vstack([np.ones(4),np.random.default_rng(20260925).multinomial(4,np.full(4,.25),size=10000)])
all_sums=np.einsum('bc,crlq->brlq',w,totals);means=all_sums[...,1]/all_sums[...,0]
response=(means[:,:,0]-means[:,:,1])/.04
out={'cases':case_reports,'definition':'Response=(selected mean g1 plus - selected mean g1 minus)/.04. Horvitz-Thompson weighted sampled union. Differences are on the same sampled IDs and original pixels.','limitations':'Four cases; two seeded repetitions plus original entropy-seeded fit. Fitter-noise projection uses first-order ratio influence and independent per-object optimizer streams; it excludes image noise, flow integration, scene variance and systematics. Sample change SE conditions on the fitted catalogues; paired case bootstrap describes only these four scenes.','differences':{}}
for name,a,b in [('repeat0_minus_original',1,0),('repeat1_minus_original',2,0),('repeat1_minus_repeat0',2,1)]:
    delta=response[:,a]-response[:,b];sample_variance=0.;projected_variance=0.
    for i,(values,(n,pop),factor) in enumerate(zip(arrays,sizes,factors)):
        influence=np.zeros(n)
        for rep,sign in [(a,1),(b,-1)]:
            for leg,lsign in [(0,1),(1,-1)]:
                centered=values[rep,leg,:,1]-means[0,rep,leg]*values[rep,leg,:,0]
                influence+=sign*lsign*centered/all_sums[0,rep,leg,0]/.04
        sample_variance+=pop**2*(1-n/pop)*influence.var(ddof=1)/n
        projected_variance+=factor*np.square(influence).sum()/2
    out['differences'][name]={'value':float(delta[0]),'paired_four_case_bootstrap_se':float(delta[1:].std(ddof=1)),'conditional_subsampling_se':float(np.sqrt(sample_variance)),'estimated_single_fit_full_catalogue_se':float(np.sqrt(projected_variance))}
(O/'fit_report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
