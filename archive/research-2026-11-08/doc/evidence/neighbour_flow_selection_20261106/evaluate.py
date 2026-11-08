"""Persistent evaluation of selected-neighbour association on original flow data."""
import importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sbsi.flow_physical_disk import positive_inverse
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_flow_selection_20261106')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('selection_neighbour_model',HERE.parent/'neighbour_conditioned_flow_20261106/model.py');c=m.c
portable=module('selection_portable',HERE.parent/'alignment_target9_20261106/portable_draws.py')
BANKS=[list(range(64)),list(range(16))+list(range(32,48)),list(range(16,32))+list(range(48,64))]
def photo(model,z,context):
    b=model.base;w,_=b.base.photometry.forward(z[:,2:],context)
    u=w.double()*b.physical_coordinate_scales+b.physical_coordinate_means
    return b.physical_units*torch.logaddexp(u,torch.zeros_like(u))
def photo_logprob(model,y,context,check=False):
    b=model.base;p=y[:,2:]*b.physical_scales+b.physical_means
    u=positive_inverse(p/b.physical_units)
    w=((u-b.physical_coordinate_means)/b.physical_coordinate_scales).to(context.dtype)
    z,ld=b.base.photometry.inverse(w,context)
    if check:
        replay,forward_ld=b.base.photometry.forward(z,context)
        torch.testing.assert_close(replay,w,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(forward_ld+ld,torch.zeros_like(ld),rtol=0,atol=2e-5)
    jac=(b.physical_units.log()+b.physical_coordinate_scales.log()+F.logsigmoid(u)-b.physical_scales.log()).sum(-1)
    return (-.5*z.double().square()-0.5*np.log(2*np.pi)).sum(-1)+ld.double()-jac
def cuts(p,manifest):
    return torch.stack([(p[...,0]>cut[1][1])&(p[...,1]>10**(-.4*(cut[2]-30))) for cut in manifest['cuts']],-1)
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    manifest=json.loads((HERE/'manifest.json').read_text());prep=Path(manifest['prepared_root']);old=Path(manifest['original_prepared_root'])
    receipt=json.loads((prep/'completed.json').read_text());norm=receipt['standardization'];names=list(manifest['arms'])
    source=c.sha(__file__);manifest_sha=c.sha(HERE/'manifest.json');loaded={};arms={};standardizer=None
    module('selection_completed_audit',HERE/'completed_audit.py').main()
    for name,pin in manifest['arms'].items():
        if pin['path'] not in loaded:
            assert c.sha(pin['path'])==pin['sha256']
            model,condition,target,_=(m.load(pin['path']) if pin['loader']=='neighbour' else c.fm.load(pin['path']))
            loaded[pin['path']]=(model,condition,target)
        model,condition,target=loaded[pin['path']];arms[name]=(model,condition)
        if standardizer is None:standardizer=target
        else:
            for k,v in standardizer.to_state().items():np.testing.assert_array_equal(target.to_state()[k],v)
    print('MODELS_READY',len(loaded),torch.cuda.get_device_name(),flush=True)
    for si,split in enumerate(manifest['split_order']):
        destination=ROOT/f'{split}.npz';done=destination.with_suffix('.json')
        if done.exists():
            saved=json.loads(done.read_text());assert saved['source_sha256']==source and saved['manifest_sha256']==manifest_sha and c.sha(destination)==saved['data_sha256']
            print('SKIP_COMPLETE',split,flush=True);continue
        assert not destination.exists()
        info=receipt['splits'][split];cases=info['cases'];ncase=len(cases);n=manifest['rows_per_split']
        indices=(np.sort(np.random.default_rng(manifest['train_sample_seed']).choice(info['rows'],n,replace=False)) if split=='train' else np.load(manifest['validation_indices']))
        assert len(indices)==n and len(np.unique(indices))==n and indices.min()>=0 and indices.max()<info['rows']
        ends=np.cumsum([row['rows'] for row in info['matching']]);assert ends[-1]==info['rows']
        record=np.searchsorted(ends,indices,side='right')
        case=np.array([cases.index(info['matching'][i]['case']) for i in record]);leg=np.array([0 if info['matching'][i]['leg']=='g0' else 1 for i in record])
        score=np.load(prep/split/'raw_score.npy',mmap_mode='r')[indices]
        if split=='train':
            context=np.load(prep/split/'context.npy',mmap_mode='r')[indices].copy();target=np.load(old/split/'target.npy',mmap_mode='r')[indices].copy()
            edges=np.quantile(score,[.2,.4,.6,.8]);c.write(ROOT/'score_edges.json',{'edges':edges,'source':'fixed training sample','sample_seed':manifest['train_sample_seed']})
        else:
            context=np.load(prep/split/'nll_context.npy');target=np.load(old/split/'nll_target.npy')
            edges=np.array(json.loads((ROOT/'score_edges.json').read_text())['edges'])
        np.testing.assert_array_equal(context[:,-1],((score-norm['mean'])/norm['scale']).astype(np.float32))
        physical=standardizer.inverse_transform_array(target);assert np.isfinite(physical).all() and (physical[:,2:]>0).all()
        bins=np.digitize(score,edges);group=(case*2+leg)*5+bins;ng=ncase*2*5;nc=len(manifest['cuts'])
        # arm, antithetic bank, case, simulation leg, training-score quintile, cut, moments
        sums=np.zeros((len(names),3,ng,nc,5));density=np.zeros((len(names),ncase*2,3));checks={};started=time.monotonic()
        with torch.no_grad():
            for block,lo in enumerate(range(0,n,manifest['block'])):
                hi=min(n,lo+manifest['block']);nb=hi-lo
                ctx=torch.as_tensor(context[lo:hi],device='cuda');y=torch.as_tensor(target[lo:hi],device='cuda')
                z=torch.from_numpy(portable.latent_bank(nb,64,4,manifest['latent_seed']+si*100000000+block)).to('cuda').reshape(-1,4)
                measured=cuts(torch.as_tensor(physical[lo:hi,2:],device='cuda'),manifest).cpu().numpy()
                ids=group[lo:hi];denids=case[lo:hi]*2+leg[lo:hi];q=score[lo:hi]
                for ai,name in enumerate(names):
                    model,condition=arms[name];pin=manifest['arms'][name]
                    if pin['loader']=='original':cc=ctx[:,:-1].contiguous()
                    else:
                        cc=ctx.clone()
                        if not condition.use_score or pin['mean_score']:cc[:,-1]=0
                    expanded=cc[:,None,:].expand(-1,64,-1).reshape(len(z),-1)
                    pp=photo(model,z,expanded).reshape(nb,64,2)
                    assert bool(torch.isfinite(pp).all() and (pp>=0).all())
                    if name not in checks:
                        joint,_=model.forward_physical(z,expanded)
                        torch.testing.assert_close(pp.reshape(-1,2),joint[:,2:],rtol=0,atol=0)
                        checks[name]={'photometry_equals_joint_exactly':True}
                    lp=photo_logprob(model,y,cc,check=(block==0));lj=model.log_prob(y,cc).double()
                    assert bool(torch.isfinite(lp).all() and torch.isfinite(lj).all())
                    for k,values in enumerate([np.ones(nb),-lj.cpu().numpy(),-lp.cpu().numpy()]):density[ai,:,k]+=np.bincount(denids,weights=values,minlength=ncase*2)
                    masks=cuts(pp,manifest)
                    for bi,draws in enumerate(BANKS):
                        probability=masks[:,draws].double().mean(1).cpu().numpy()
                        for cut in range(nc):
                            for moment,values in enumerate([np.ones(nb),measured[:,cut],measured[:,cut]*q,probability[:,cut],probability[:,cut]*q]):
                                sums[ai,bi,:,cut,moment]+=np.bincount(ids,weights=values,minlength=ng)
                if block%100==0:print('PROGRESS',split,block,'seconds',round(time.monotonic()-started,1),flush=True)
        np.testing.assert_allclose(sums[:,0],.5*(sums[:,1]+sums[:,2]),rtol=2e-13,atol=2e-8)
        np.testing.assert_array_equal(sums[...,:3],np.broadcast_to(sums[0,0,...,:3],sums[...,:3].shape))
        np.testing.assert_array_equal(density[:,:,0],np.broadcast_to(density[0,:,0],density[:,:,0].shape))
        nll=density[:,:,1].sum(1)/n
        if split=='validation':
            reports=json.loads((HERE.parent/'neighbour_conditioned_flow_20261106/report.json').read_text())
            for ai,name in enumerate(names):
                if name in ['control_nll','control_nine','neighbour_nll','neighbour_nine']:
                    arm,stage=name.split('_');expected=reports['training'][arm][stage+'_probe']['nll']['nll']
                    assert abs(nll[ai]-expected)<2e-5,(name,nll[ai],expected)
        np.savez(destination,sums=sums.reshape(len(names),3,ncase,2,5,nc,5),density=density.reshape(len(names),ncase,2,3),arms=names,cases=cases,cuts=[x[0] for x in manifest['cuts']],score_edges=edges,indices=indices)
        c.write(done,{'source_sha256':source,'manifest_sha256':manifest_sha,'data_sha256':c.sha(destination),'split':split,'rows':n,'case_count':ncase,'checks':checks,'score_alignment_exact':True,'validation_nll_replay':split=='validation','seconds':time.monotonic()-started})
        print('DONE_SPLIT',split,'NLL',dict(zip(names,nll)),flush=True)
    module('selection_probe_report',HERE/'report.py').main()
    print('DONE',flush=True)
if __name__=='__main__':main()
