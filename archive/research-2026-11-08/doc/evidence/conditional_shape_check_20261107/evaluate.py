"""Conditional shape moments on measured photometry, unsheared validation rows."""
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch

H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def features(e):
    a,b=e[...,0],e[...,1]
    return torch.stack([a,b,a*a,a*b,b*b,a*a-b*b],-1)
def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    else:np.testing.assert_array_equal(a,b)
def ent(v,n):
    jk=v[-n:]
    return {'value':v[0].tolist(),'bootstrap_se':v[1:10001].std(0,ddof=1).tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0)).tolist()}

def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    for p,d in read(H/'source_pins.json').items():assert sha(p)==d,p
    c=read(H/'manifest.json');prep=Path(c['prepared_root']);old=Path(c['original_prepared_root'])
    cp=load('shape_check_conditional',H.parent/'alignment_conditioned_photometry_20261105/evaluate.py')
    nm=load('shape_check_neighbour',H.parent/'neighbour_conditioned_flow_20261106/model.py')
    portable=load('shape_check_portable',H.parent/'alignment_target9_20261106/portable_draws.py')
    models={};conditions={};target=None
    for name,pin in c['arms'].items():
        assert sha(pin['path'])==pin['sha256']
        model,condition,t,_=(cp.fm.load if pin['loader']=='original' else nm.load)(pin['path'])
        models[name]=model;conditions[name]=condition
        if target is None:target=t;base=condition
        equal(t.to_state(),target.to_state())
        equal((condition.base if pin['loader']=='neighbour' else condition).to_state(),base.to_state())
    info=read(prep/'completed.json')['splits']['validation'];cases=info['cases'];ncase=len(cases)
    indices=np.load(c['validation_indices']);assert len(indices)==c['rows_per_split']
    ends=np.cumsum([r['rows'] for r in info['matching']]);assert ends[-1]==info['rows']
    record=np.searchsorted(ends,indices,side='right')
    group=np.array([cases.index(info['matching'][i]['case']) for i in record])
    zero=np.array([info['matching'][i]['leg']=='g0' for i in record])
    ctx=np.load(prep/'validation/nll_context.npy');y=np.load(old/'validation/nll_target.npy')
    physical=target.inverse_transform_array(y);score=np.load(prep/'validation/raw_score.npy',mmap_mode='r')[indices]
    np.testing.assert_array_equal(ctx[:,-1],((score-conditions['current'].mean)/conditions['current'].scale).astype(np.float32))
    assert len(ctx)==len(y)==len(indices) and np.isfinite(physical).all()
    names=list(models);nc=len(c['cuts']);nf=6
    # count; observed moments; bankA moments; bankB moments; independent-bank squared score.
    sums=np.zeros((len(names),ncase,nc,1+4*nf));checks={};started=time.monotonic()
    banks=[list(range(16))+list(range(32,48)),list(range(16,32))+list(range(48,64))]
    print('READY',int(zero.sum()),'g0 rows',len(models),'models',torch.cuda.get_device_name(),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(indices),c['block'])):
            hi=min(len(indices),lo+c['block']);keep=zero[lo:hi];nb=hi-lo
            if not keep.any():continue
            raw=physical[lo:hi][keep];ids=group[lo:hi][keep];observed=torch.as_tensor(raw,device='cuda')
            masks=np.stack([(raw[:,2]>cut[1][1])&(raw[:,3]>10**(.4*(30-cut[2]))) for cut in c['cuts']],-1)
            bank=portable.latent_bank(nb,c['draws'],4,c['latent_seed']+100000000+block)
            z=torch.from_numpy(bank[keep]).to('cuda').reshape(-1,4)
            photo=observed[:,None,2:].expand(-1,64,-1).reshape(-1,2)
            actual=features(observed[:,:2]).cpu().numpy()
            for ai,name in enumerate(names):
                context=ctx[lo:hi][keep]
                if c['arms'][name]['loader']=='original':context=context[:,:-1]
                x=torch.as_tensor(context,device='cuda')[:,None,:].expand(-1,64,-1).reshape(len(z),-1)
                v=cp.conditional_physical(models[name],z,x,photo).reshape(-1,64,4)
                torch.testing.assert_close(v[:,:,2:],observed[:,None,2:].expand(-1,64,-1),rtol=0,atol=0)
                assert bool(torch.isfinite(v).all())
                if name not in checks:
                    q,_=models[name].forward_physical(z[:256],x[:256]);valid=(q[:,2:]>0).all(-1)
                    reconstructed=cp.conditional_physical(models[name],z[:256][valid],x[:256][valid],q[valid,2:])
                    torch.testing.assert_close(reconstructed,q[valid],rtol=2e-5,atol=2e-6)
                    checks[name]={'reconstruction_max_abs':float((reconstructed-q[valid]).abs().max()),'exact_measured_photometry':True}
                f=features(v[...,:2]);a=f[:,banks[0]].mean(1).cpu().numpy();b=f[:,banks[1]].mean(1).cpu().numpy()
                values=np.column_stack([np.ones(len(raw)),actual,a,b,(a-actual)*(b-actual)])
                for ci in range(nc):
                    ix=masks[:,ci]
                    for k in range(values.shape[1]):sums[ai,:,ci,k]+=np.bincount(ids[ix],weights=values[ix,k],minlength=ncase)
            if block%100==0:print('PROGRESS',block,round(time.monotonic()-started,1),flush=True)
    np.testing.assert_array_equal(sums[...,:1+nf],np.broadcast_to(sums[:1,...,:1+nf],sums[...,:1+nf].shape))
    out=ROOT/'sums.npz';np.savez(out,sums=sums,cases=cases,arms=names,cuts=[v[0] for v in c['cuts']],features=['e1','e2','e1_squared','e1_e2','e2_squared','quadrupole'])
    w=np.vstack([np.ones(ncase),np.random.default_rng(c['bootstrap_seed']).multinomial(ncase,np.ones(ncase)/ncase,10000),1-np.eye(ncase)])
    report={'definition':'Zero-shear usable validation rows, conditional shapes at actual photometry and actual selection. Shape moments, not shear bias. Generated minus measured. Independent-bank conditional moment squared scores.', 'cases':cases,'g0_rows':int(zero.sum()),'checkpoints':c['arms'],'checks':checks,'source_sha256':sha(__file__),'data_sha256':sha(out),'cuts':{}}
    pooled=np.einsum('bc,actq->abtq',w,sums)
    for ci,(cut,_,_) in enumerate(c['cuts']):
        x=pooled[:,:,ci];count=x[...,:1];assert (count>0).all()
        truth=x[...,1:7]/count;pred=(x[...,7:13]+x[...,13:19])/(2*count);loss=x[...,19:25]/count
        rows={name:{'observed':ent(truth[ai],ncase),'predicted':ent(pred[ai],ncase),'error':ent(pred[ai]-truth[ai],ncase),'conditional_score':ent(loss[ai],ncase),'half_bank_difference':ent((x[ai,:,7:13]-x[ai,:,13:19])/count[ai],ncase)} for ai,name in enumerate(names)}
        report['cuts'][cut]={'selected_rows':int(sums[0,:,ci,0].sum()),'arms':rows,'current_minus_earlier':{name:{'error':ent(pred[-1]-pred[ai],ncase),'conditional_score':ent(loss[-1]-loss[ai],ncase)} for ai,name in enumerate(names[:-1])}}
    for dest in [H,ROOT]:(dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CENTRAL',json.dumps(report['cuts']['mag25_radius_gt0.60']),flush=True)
    print('DONE_CONDITIONAL_SHAPE_CHECK',time.monotonic()-started,flush=True)

if __name__=='__main__':main()
