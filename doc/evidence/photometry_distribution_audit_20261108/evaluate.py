"""Conditional photometry diagnostics, without training or new image measurements."""
import json,os,time,types
from pathlib import Path
import numpy as np
import torch
from sbsi.flow_physical_disk import positive_inverse
H=Path(__file__).resolve().parent
c=types.ModuleType('photo_distribution_common');c.__file__=str(H/'common.py')
exec(compile((H/'common.py').read_bytes(),c.__file__,'exec'),c.__dict__)
helper=c.load('photo_distribution_original',H.parent/'neighbour_flow_selection_20261106/evaluate.py')

def photo(model,z,x):
    b=model.base;w,_=b.base.photometry.forward(z[:,2:],x)
    u=w.double()*b.physical_coordinate_scales+b.physical_coordinate_means
    p=b.physical_units*torch.logaddexp(u,torch.zeros_like(u))
    # Below -36 the omitted correction is below float64 relative roundoff.
    logsoft=torch.where(u < -36,u,torch.log(torch.logaddexp(u,torch.zeros_like(u))))
    magnitude=30-2.5/np.log(10)*(b.physical_units[1].log()+logsoft[:,1])
    return p,magnitude

def masks(p):
    r=torch.stack([p[:,:,0]>v for v in (2.8,3.,3.2)],-1)
    f=torch.stack([p[:,:,1]>10**(.4*(30-v)) for v in (24.9,25.,25.1)],-1)
    joint=(r[:,:,:,None]&f[:,:,None,:]).reshape(len(p),p.shape[1],9)
    return torch.cat([r,f,joint],-1)

def inverse_latent(model,y,x):
    b=model.base;p=y[:,2:]*b.physical_scales+b.physical_means
    u=positive_inverse(p/b.physical_units)
    w=((u-b.physical_coordinate_means)/b.physical_coordinate_scales).to(x.dtype)
    z,_=b.base.photometry.inverse(w,x)
    return z

def main():
    assert os.environ.get('SLURM_JOB_ID');c.verify();cfg=c.config();torch.set_num_threads(4)
    oldcfg=json.loads(Path(cfg['sample_manifest']).read_text());prep=Path(oldcfg['prepared_root']);old=Path(oldcfg['original_prepared_root'])
    receipt=json.loads((prep/'completed.json').read_text());assert c.sha(prep/'completed.json')==cfg['prepared_sha256']
    models={};target=None;base=None
    for name,pin in cfg['arms'].items():
        assert c.sha(pin['path'])==pin['sha256']
        model,pre,t,_=(helper.c.fm.load(pin['path']) if pin['loader']=='original' else helper.m.load(pin['path']))
        models[name]=(model,pre);target=t;base=pre if pin['loader']=='original' else pre.base
    names=list(models);n=oldcfg['rows_per_split'];a=len(names)
    print('MODELS_READY',names,torch.cuda.get_device_name(),flush=True)
    for si,split in enumerate(['train','validation']):
        dest=c.ROOT/split;dest.mkdir(exist_ok=True)
        provenance={'source_sha256':c.sha(__file__),'manifest_sha256':c.sha(H/'manifest.json'),'pins':cfg['arms']}
        if (dest/'completed.json').exists():
            r=json.loads((dest/'completed.json').read_text());assert r['provenance']==provenance
            for k,v in r['outputs'].items():assert c.sha(dest/k)==v
            print('SPLIT_ALREADY_COMPLETE',split,flush=True);continue
        info=receipt['splits'][split];cases=info['cases']
        indices=np.sort(np.random.default_rng(oldcfg['train_sample_seed']).choice(info['rows'],n,replace=False)) if split=='train' else np.load(oldcfg['validation_indices'])
        ends=np.cumsum([r['rows'] for r in info['matching']]);assert ends[-1]==info['rows'];records=np.searchsorted(ends,indices,side='right')
        case=np.array([info['matching'][i]['case'] for i in records]);leg=np.array([0 if info['matching'][i]['leg']=='g0' else 1 for i in records])
        score=np.load(prep/split/'raw_score.npy',mmap_mode='r')[indices]
        if split=='train':x=np.load(prep/split/'context.npy',mmap_mode='r')[indices];y=np.load(old/split/'target.npy',mmap_mode='r')[indices]
        else:x=np.load(prep/split/'nll_context.npy');y=np.load(old/split/'nll_target.npy')
        assert len(x)==len(y)==len(indices)==n and len(np.unique(indices))==n
        np.testing.assert_array_equal(x[:,-1],((score-receipt['standardization']['mean'])/receipt['standardization']['scale']).astype(np.float32))
        obs=target.inverse_transform_array(y)[:,2:];assert np.isfinite(obs).all() and (obs>0).all()
        raw=x[:,:8].astype(float)*base.scales+base.means
        for j,feature in enumerate(base.feature_names):
            if feature in base.log_features:raw[:,j]=np.exp(raw[:,j])
        missing=x[:,8:16].astype(bool);assert not missing.any(),'Do not silently bin imputed contexts'
        np.savez(dest/'rows.npz',indices=indices,case=case,leg=leg,raw=np.column_stack([raw,score]),observed=obs,cases=cases)
        # Persistent per-row files enable detailed follow-ups without another GPU pass.
        shapes={'probabilities.npy':((a,n,3,15),'float32'),'rank.npy':((a,n,2),'uint8'),
                'latent.npy':((a,n,2),'float32'),'moments.npy':((a,n,8),'float64'),'photo_nll.npy':((a,n),'float64')}
        resume=dest/'progress.json';start=0;checks={};underflow={name:[0,0] for name in names}
        if resume.exists():
            d=json.loads(resume.read_text());assert d['provenance']==provenance;start=d['next_block'];checks=d['checks'];underflow=d['underflow']
        arrays={k:np.lib.format.open_memmap(dest/k,mode='r+' if resume.exists() else 'w+',dtype=dtype,shape=shape) for k,(shape,dtype) in shapes.items()}
        started=time.monotonic()
        with torch.no_grad():
            for block,lo in enumerate(range(0,n,1024)):
                if block<start:continue
                hi=min(n,lo+1024);nb=hi-lo;ctx=torch.as_tensor(x[lo:hi],device='cuda');yy=torch.as_tensor(y[lo:hi],device='cuda')
                z=torch.from_numpy(helper.portable.latent_bank(nb,64,4,oldcfg['latent_seed']+si*100000000+block)).to('cuda').reshape(-1,4)
                iid=torch.from_numpy(np.random.default_rng(cfg['rank_seed']+si*100000000+block).standard_normal((nb*64,4)).astype('float32')).to('cuda')
                for ai,name in enumerate(names):
                    model,pre=models[name];pin=cfg['arms'][name]
                    cc=ctx[:,:16].contiguous() if pin['loader']=='original' else ctx.clone()
                    if pin['loader']!='original' and not pre.use_score:cc[:,-1]=0
                    expanded=cc[:,None].expand(nb,64,cc.shape[1]).reshape(-1,cc.shape[1])
                    pp,mag=photo(model,z,expanded);pp=pp.reshape(nb,64,2);mag=mag.reshape(nb,64)
                    assert torch.isfinite(pp).all() and torch.isfinite(mag).all() and (pp>=0).all()
                    if name not in checks:
                        original=helper.photo(model,z,expanded).reshape(nb,64,2);torch.testing.assert_close(pp,original,rtol=0,atol=0)
                        joint,_=model.forward_physical(z,expanded);torch.testing.assert_close(pp.reshape(-1,2),joint[:,2:],rtol=0,atol=0)
                        valid=pp[:,:,1]>0
                        torch.testing.assert_close(mag[valid],30-2.5*torch.log10(pp[:,:,1][valid]),rtol=1e-12,atol=1e-10)
                        checks[name]={'photo_equals_joint':True,'stable_magnitude_replay':True}
                    mm=masks(pp)
                    for bank,ix in enumerate(helper.BANKS):arrays['probabilities.npy'][ai,lo:hi,bank]=mm[:,ix].float().mean(1).cpu().numpy()
                    # Marginal ranks need IID draws, not an antithetic rank-uniformity assumption.
                    ip,_=photo(model,iid,expanded);ip=ip.reshape(nb,64,2)
                    measured=torch.as_tensor(obs[lo:hi],device='cuda')
                    less=(ip<measured[:,None]).sum(1).cpu().numpy()
                    ties=(ip==measured[:,None]).sum(1).cpu().numpy()
                    tie_u=np.random.default_rng(cfg['rank_seed']+si*100000000+block+10000000000).random((nb,2))
                    arrays['rank.npy'][ai,lo:hi]=less+np.floor(tie_u*(ties+1)).astype(int)
                    checks[name]['rank_tied_rows']=checks[name].get('rank_tied_rows',0)+int((ties>0).any(1).sum())
                    latent=inverse_latent(model,yy,cc);assert torch.isfinite(latent).all()
                    arrays['latent.npy'][ai,lo:hi]=latent.cpu().numpy()
                    lp=helper.photo_logprob(model,yy,cc);assert torch.isfinite(lp).all();arrays['photo_nll.npy'][ai,lo:hi]=-lp.cpu().numpy()
                    rq=torch.quantile(pp[:,:,0],torch.tensor([.16,.5,.84],device='cuda',dtype=torch.float64),dim=1)
                    mq=torch.quantile(mag,torch.tensor([.16,.5,.84],device='cuda',dtype=torch.float64),dim=1)
                    arrays['moments.npy'][ai,lo:hi]=torch.stack([pp[:,:,0].mean(1),mag.mean(1),rq[0],rq[1],rq[2],mq[0],mq[1],mq[2]],-1).cpu().numpy()
                    underflow[name][0]+=int((pp[:,:,0]==0).sum());underflow[name][1]+=int((pp[:,:,1]==0).sum())
                if block%32==0:print('PROGRESS',split,block,'/',(n+1023)//1024,round(time.monotonic()-started,1),flush=True)
                if (block+1)%64==0 or hi==n:
                    for arr in arrays.values():arr.flush()
                    c.write(resume,dict(provenance=provenance,next_block=block+1,checks=checks,underflow=underflow))
        np.testing.assert_array_equal(arrays['probabilities.npy'][:,:,0],.5*(arrays['probabilities.npy'][:,:,1]+arrays['probabilities.npy'][:,:,2]))
        output={p.name:c.sha(p) for p in dest.glob('*.npy')};output['rows.npz']=c.sha(dest/'rows.npz')
        c.write(dest/'completed.json',dict(provenance=provenance,outputs=output,rows=n,cases=len(cases),checks=checks,underflow=underflow,seconds=time.monotonic()-started))
        print('SPLIT_COMPLETE',split,flush=True)
    print('PHOTOMETRY_DISTRIBUTION_EVALUATION_DONE',flush=True)
if __name__=='__main__':main()
