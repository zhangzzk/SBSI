"""Paired FS2 refinement ablation and population-standardization changes."""
import json
import os
from pathlib import Path
import numpy as np
from reduce_flow import responses, ent

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/fs2_galsbi_transfer_20261105')


def read(family):
    p=json.loads((O/family/'protocol.json').read_text())
    d={k:[] for k in ['base','cross','normal','count']}
    for case in p['cases']:
        with np.load(O/family/f'case{case:03d}.npz') as z:
            for k in d:
                a=z[k]
                if k in ['base','cross']:a=np.stack([a[0],a[1:].mean(0)])
                d[k].append(a)
    return {k:np.array(v) for k,v in d.items()},p


def compute(d,w,rw):
    curves=[];cells=[]
    for lo in range(0,len(w),250):
        ww=w[lo:lo+250]
        b=np.einsum('nc,cb...->nb...',ww,d['base']*rw[None,None,:,None,None])
        c=np.einsum('nc,cb...->nb...',ww,d['cross']*rw[None,None,:,None,None,None])
        n=np.einsum('nc,c...->n...',ww,d['normal']*rw[None,:,None,None])
        count=ww@(d['count']*rw[None,:])
        cell=responses(b,c,n,count)
        curves.append(cell.sum(2));cells.append(cell)
    return np.concatenate(curves),np.concatenate(cells)


data={};prot={}
for f in ['fs2','galsbi','fs2_base']:data[f],prot[f]=read(f)
report={'definition':'R11 response, own baseline normalization; paired whole-case resamples within a family; two banks averaged; standardization weights treated as fixed.', 'refinement':{},'reweighting':{}}
w=np.vstack([np.ones(40),np.random.default_rng(20261105123).multinomial(40,np.full(40,1/40),10000)])
r_after,_=compute(data['fs2'],w,np.ones(25));r_before,_=compute(data['fs2_base'],w,np.ones(25))
np.testing.assert_allclose(r_after[:,0],r_before[:,0],atol=1e-14)
for k,cut in enumerate(prot['fs2']['cuts']):
    a,b=r_after[:,:,k,0,0],r_before[:,:,k,0,0]
    ma,mb=100*(a[:,0]/a[:,1]-1),100*(b[:,0]/b[:,1]-1)
    report['refinement'][cut]={'m_before':ent(mb),'m_after':ent(ma),'paired_change_pp':ent(ma-mb),
                              'model_response_change':ent(a[:,1]-b[:,1])}
for f,other in [('fs2','galsbi'),('galsbi','fs2')]:
    frac=data[f]['count'].sum(0);frac=frac/frac.sum()
    otherfrac=data[other]['count'].sum(0);otherfrac=otherfrac/otherfrac.sum()
    common=(frac>0)&(otherfrac>0)
    rw=np.divide(otherfrac,frac,out=np.zeros_like(frac),where=common)
    native,cells=compute(data[f],w,np.ones(25))
    shared,_=compute(data[f],w,common.astype(float));weighted,_=compute(data[f],w,rw)
    out={}
    for k,cut in enumerate(prot[f]['cuts']):
        a,b=native[:,:,k,0,0],weighted[:,:,k,0,0]
        ma,mb=100*(a[:,0]/a[:,1]-1),100*(b[:,0]/b[:,1]-1)
        delta=cells[:,0,:,k,0,0]-cells[:,1,:,k,0,0]
        small=delta.reshape(-1,5,5)[:,1:3].sum((1,2))
        sc=shared[:,:,k,0,0];ms=100*(sc[:,0]/sc[:,1]-1)
        out[cut]={'native_m':ent(ma),'common_support_native_m':ent(ms),
                  'weighted_m':ent(mb),'paired_change_pp':ent(mb-ms),
                  'small_015_04_delta_R11':ent(small)}
    report['reweighting'][f]={'native_mass_excluded':float(frac[~common].sum()),
        'target_mass_excluded':float(otherfrac[~common].sum()),'cuts':out}
(O/'refinement_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for name,v in report['refinement'].items():print(name,v,flush=True)
