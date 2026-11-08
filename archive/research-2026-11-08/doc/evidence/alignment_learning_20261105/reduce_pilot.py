"""Fixed-endpoint learning intervention: paired validation comparisons."""
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_learning_20261105')
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
protocol=json.loads((ROOT/'galsbi/protocol.json').read_text())
for arm in ['full','small']:assert (ROOT/f'pilot_{arm}/completed.json').exists()
cases=protocol['cases']['validation'];x=[]
for case in cases:
    with np.load(ROOT/'galsbi'/f'validation_case{case:03d}.npz') as z:
        v=z['profile'];row=[v[0],v[1:3].mean(0)]
    for arm in ['full','small']:
        with np.load(ROOT/f'pilot_{arm}'/f'case{case:03d}.npz') as z:row.append(z['profile'].mean(0))
    x.append(np.stack(row))
x=np.array(x)
w=np.vstack([np.ones(len(x)),np.random.default_rng(202611053000).multinomial(len(x),np.full(len(x),1/len(x)),10000)])
def ent(a):return {'value':a[0].tolist(),'case_se':np.std(a[1:],axis=0,ddof=1).tolist()}
out={'definition':'Fixed20-epoch NLL continuations, identical initial weights/moments and resetlr1e-4; full or intrinsic-small input distribution; common evaluation random draws. Paired40-case bootstrap conditional on these runs.',
     'branches':['simulation','initial_nll','full_continuation','small_continuation'],'groups':{},'validation_nll':{}}
for gi,group in enumerate(protocol['groups']):
    d=x[:,:,:,gi];tot=d.sum(0)
    with np.errstate(divide='ignore',invalid='ignore'):
        profile=tot[...,2]/tot[...,1]
    bands={}
    for name,ix in [('2.7-2.8',[8]),('3.0-3.1',[11]),('2.4-3.0',list(range(5,11))),('3.0-3.4',list(range(11,15)))]:
        t=np.einsum('nc,cbmk->nbmk',w,d[:,:,:,ix,:].sum(3))
        with np.errstate(divide='ignore',invalid='ignore'):
            a=t[...,2]/t[...,1]
        bands[name]={'alignment':ent(a),'error':ent(a[:,1:]-a[:,:1]),
                     'full_minus_initial':ent(a[:,2]-a[:,1]),'small_minus_initial':ent(a[:,3]-a[:,1]),
                     'small_minus_full':ent(a[:,3]-a[:,2])}
    out['groups'][group]={'alignment':profile.tolist(),'count':tot[...,0].tolist(),'bands':bands}
for arm in ['full','small']:
    h=json.loads((ROOT/f'pilot_{arm}/history.json').read_text())
    out['validation_nll'][arm]={'initial':h[0]['validation']['nll'],'final':h[-1]['validation']['nll'],
          'change':h[-1]['validation']['nll']-h[0]['validation']['nll'],'epochs':h[-1]['epoch'],
          'training_rows':json.loads((ROOT/f'pilot_{arm}/protocol.json').read_text())['training_rows']}
np.testing.assert_allclose(out['validation_nll']['full']['initial'],out['validation_nll']['small']['initial'],atol=1e-12,rtol=0)
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
(ROOT/'pilot_report.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n')
print(out['validation_nll'],flush=True)
print(clean(out['groups']['all']['bands']['2.7-2.8']),flush=True)
