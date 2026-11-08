"""Odd cross-response and even shape offset on the existing g1 scenes."""
import os,json
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')
here=Path(__file__).resolve().parent
root=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_factorized_constant_20261105')
m=json.loads((here/'manifest.json').read_text());cases=m['cases'];n=len(cases)
w=np.vstack([np.ones(n),np.random.default_rng(202611057001).multinomial(n,np.full(n,1/n),10000)])
r={'definition':'e2 even=(mean_e2_plus+mean_e2_minus)/2, finite-shear additive diagnostic; R21 odd=(mean_e2_plus-mean_e2_minus)/.04; error simulation-model. Not a fitted/inferred g2 or a measurement of R22.','cases':cases,'arms':{}}
def ent(x):return dict(value=float(x[0]),case_se=float(x[1:].std(ddof=1)))
for arm in ['retained','cartesian','frame']:
 models=[];sims=[]
 for case in cases:
  path=(Path(m['baseline_root'])/'sums' if arm=='retained' else root/arm)/f'case{case}.npz'
  with np.load(path) as z:
   ci=[list(z['cuts']).index(c[0]) for c in m['cuts']]
   models.append(z['model'][:,:,ci,:,:5].sum(-2));sims.append(z['sim'][:,ci,:,:3].sum(-2))
 model=np.asarray(models);sim=np.asarray(sims);r['arms'][arm]={}
 for mode,label in enumerate(['predicted','actual_usability']):
  out={}
  for k,cut in enumerate(m['cuts']):
   s=np.einsum('bc,clq->blq',w,sim[:,:,k]);x=np.einsum('bc,clq->blq',w,model[:,mode,:,k])
   es=s[...,2]/s[...,0];em=(x[...,2]+x[...,4])/x[...,0]
   out[cut[0]]={'simulation_even_e2':ent(es.mean(-1)),'model_even_e2':ent(em.mean(-1)),
               'delta_even_e2':ent((es-em).mean(-1)),
               'delta_R21':ent(((es[:,0]-es[:,1])-(em[:,0]-em[:,1]))/.04)}
  r['arms'][arm][label]=out
(root/'null_component.json').write_text(json.dumps(r,indent=2)+'\n')
for a,v in r['arms'].items():print(a,v['predicted']['mag25_radius_gt0.60'])
