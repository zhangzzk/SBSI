"""Cross photometry inputs and membership, preserving original flow streams."""
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scripts import joint_response_axes as axes

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'alignment_regularization_shift_20261106'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=module('factorial_anchor',OLD/'evaluate_measured_transport.py')
cp=module('factorial_conditional',HERE.parent/'alignment_conditioned_photometry_20261105/evaluate.py')
portable=module('factorial_portable',HERE.parent/'alignment_target9_20261106/portable_draws.py')
e=a.e
ROOT=a.DATA/'sbsi_caches/photometry_selection_factorial_20261106'
BANKS=[list(range(64)),list(range(16))+list(range(32,48)),list(range(16,32))+list(range(48,64))]
QUANTITIES=['mass','shape_difference','log_radius','magnitude','opposite_coefficient','positive_radius_mass','positive_flux_mass']

def properties(v,b):
    r,f=v[...,2],v[...,3]
    logr=torch.where(r>0,torch.log(r),torch.zeros_like(r))
    mag=torch.where(f>0,30-2.5*torch.log10(f),torch.zeros_like(f))
    return torch.stack([logr,mag,b.expand_as(r),(r>0).double(),(f>0).double()],-1)

def edge_change(shape,coeff,s,t):
    moved=[]
    for state in [s,t]:
        velocity=.02*(coeff*coeff.new_tensor(a.SIGNS[state])).sum(-1)
        v=torch.stack([velocity,velocity*0],-1)
        if v.ndim==2:v=v[:,None,:]
        moved.append(a.disk_response_transport(shape,v)[0])
    return moved[0][...,0]-moved[1][...,0]

def weighted_statistics(weight,change,properties):
    """Per target and cut: averaging paired draws before summing targets."""
    mass=weight.mean(1)
    numerator=(weight*change[...,None]).mean(1)
    props=(weight[...,None]*properties[:,:,None,:]).mean(1)
    return torch.cat([mass[...,None],numerator[...,None],props],-1)

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    index=int(os.environ['SLURM_ARRAY_TASK_ID']);case=a.CASES[index]
    device=torch.cuda.get_device_name();family=['V100','A40','V100','RTX 2080 Ti'][index]
    assert family in device,(family,device)
    ROOT.mkdir(parents=True,exist_ok=True);out=ROOT/f'case{case}.npz';assert not out.exists()
    torch.set_num_threads(4);e.CUTS=a.CUTS;e.CUT_NAMES=tuple(x[0] for x in a.CUTS)
    preset,paths,conditions=e.preset()
    pin=json.loads((OLD/'manifest.json').read_text())['arms']['nine_weak']
    assert e.sha256(pin['path'])==pin['sha256']
    model,condition,_,_=cp.fm.load(pin['path'])
    emulator=a.DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    with np.load(e.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context'].astype(float)
    with np.load(a.STUDY/f'groups_case{case}.npz') as z:lookup=pd.Series(z['group'],index=z['input_index'])
    group=lookup.loc[ids].to_numpy();values=[];usable=[];receipts=[]
    for r,h in [(e.SIMS,.02),(a.STUDY/'sims',.02),(a.STUDY/'sims',-.02),(e.SIMS,-.02)]:
        v,u,record=a.ev.measured_leg(r,case,h,ids);values.append(v);usable.append(u);receipts.append(record)
    common=np.logical_and.reduce(usable);values=np.array(values);cuts=e.output_cuts(values).astype(float)
    size=np.digitize(context[:,4],[.15,.25,.4,.7])
    contexts=[torch.as_tensor(condition.transform_frame(f),dtype=torch.float32,device='cuda')
              for f in axes.shear_frames(e.full_context(context),.02,1)]
    truth,_=e.truth(case,.02);np.testing.assert_array_equal(truth['index_input'],ids)
    pairs=e.pair_table(e.scene_frame(truth),np.flatnonzero(common),conditions)
    pairids=pairs['index_input_p'].to_numpy();pg=lookup.loc[pairs['index_input_s']].to_numpy()
    with np.load(a.ROOT/'measured_transport'/f'case{case}.npz') as z:sim=z['sim'];edges=z['edges']
    for state in range(4):
        for k in range(4):
            for g in [0,1]:
                use=common&(group==g)&cuts[state,:,k].astype(bool)
                np.testing.assert_allclose(sim[state,k,g].sum(0),[use.sum(),*values[state,use,:2].sum(0)],rtol=1e-12,atol=1e-7)
    # input G/M, membership G/M, cut, edge, group, intrinsic size, draw bank, quantity
    sums=np.zeros((2,2,4,4,2,5,3,7));start=time.monotonic();verified=False
    print('READY',case,device,'full parents',len(ids),'common',int(common.sum()),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),1024)):
            hi=min(len(ids),lo+1024);nb=hi-lo;keep=common[lo:hi];selected=np.flatnonzero(keep)
            if not len(selected):continue
            bid=ids[lo:hi][keep];gg=group[lo:hi][keep];sz=size[lo:hi][keep]
            seed=7301+10000019*case+1000003*block
            bank=portable.latent_bank(nb,64,4,seed)
            latent=torch.from_numpy(bank).to('cuda').reshape(nb*64,4)
            generated=[]
            for ctx in contexts:
                v,_=model.forward_physical(latent,ctx[lo:hi,None,:].expand(-1,64,-1).reshape(nb*64,-1))
                generated.append(v.reshape(nb,64,4)[keep])
            z=torch.from_numpy(bank[keep]).to('cuda').reshape(len(bid)*64,4)
            p0=np.searchsorted(pairids,bid[0]);p1=np.searchsorted(pairids,bid[-1],side='right');pp=pairs.iloc[p0:p1]
            def coefficients(v):
                bs=[a.predict_scene_draw_responses(emulator,pp.loc[pg[p0:p1]==g],bid,v.cpu().numpy())[0] for g in [0,1]]
                return torch.as_tensor(np.stack(bs,-1),dtype=torch.float64,device='cuda')
            bg=[coefficients(v) for v in generated]
            gmask=[e.output_cuts(v).double() for v in generated]
            measured=[];bm=[]
            groups=torch.as_tensor(gg,dtype=torch.long,device='cuda')
            for state in range(4):
                plus=torch.as_tensor(a.SIGNS[state],device='cuda')[groups]>0
                ctx=torch.where(plus[:,None],contexts[0][lo:hi][keep],contexts[1][lo:hi][keep])
                photo=torch.as_tensor(values[state,lo:hi][keep,2:],dtype=torch.float64,device='cuda')
                v=cp.conditional_physical(model,z,ctx[:,None,:].expand(-1,64,-1).reshape(len(bid)*64,-1),
                    photo[:,None,:].expand(-1,64,-1).reshape(len(bid)*64,2)).reshape(len(bid),64,4)
                torch.testing.assert_close(v[...,2:],photo[:,None,:].expand(-1,64,-1),rtol=0,atol=0)
                measured.append(v);bm.append(coefficients(v[:,:1]))
            if not verified:
                ctx=contexts[0][lo:hi][keep][:,None,:].expand(-1,64,-1).reshape(len(bid)*64,-1)
                photo=generated[0][...,2:].reshape(len(bid)*64,2)
                valid=(photo>0).all(-1);assert bool(valid.any())
                replay=cp.conditional_physical(model,z[valid],ctx[valid],photo[valid])
                torch.testing.assert_close(replay,generated[0].reshape(-1,4)[valid],rtol=2e-5,atol=2e-6)
                verified=True
            for edge,(s,t) in enumerate(a.EDGES):
                g=1 if edge<2 else 0;other=1-g;leg=edge%2
                change_g=edge_change(generated[leg][...,:2],bg[leg],s,t)
                change_m=.5*(edge_change(measured[s][...,:2],bm[s],s,t)+edge_change(measured[t][...,:2],bm[t],s,t))
                props_g=properties(generated[leg],bg[leg][...,other])
                props_m=.5*(properties(measured[s],bm[s][...,other])+properties(measured[t],bm[t][...,other]))
                wm=torch.as_tensor(.5*(cuts[s,lo:hi][keep]+cuts[t,lo:hi][keep]),device='cuda')[:,None,:].expand(-1,64,-1)
                for inp,(change,feature_values) in enumerate([(change_g,props_g),(change_m,props_m)]):
                    for membership,weight in enumerate([gmask[leg],wm]):
                        for bankid,draw_ids in enumerate(BANKS):
                            moment=weighted_statistics(weight[:,draw_ids],change[:,draw_ids],feature_values[:,draw_ids]).cpu().numpy()
                            for b in range(5):
                                rows=(gg==g)&(sz==b)
                                sums[inp,membership,:,edge,g,b,bankid]+=moment[rows].sum(0)
            if block%100==0:print('PROGRESS',case,block,round(time.monotonic()-start,1),flush=True)
    assert verified
    np.testing.assert_allclose(sums[...,0,:],.5*(sums[...,1,:]+sums[...,2,:]),rtol=1e-12,atol=2e-7)
    np.testing.assert_allclose(sums[0,...,0],sums[1,...,0],rtol=0,atol=1e-9)
    with np.load(a.ROOT/'fourstate/nine_weak'/f'case{case}.npz') as old:
        for edge,(s,t) in enumerate(a.EDGES):
            g=1 if edge<2 else 0
            current=sums[0,0,:,edge,g,:,0].sum(-2)
            previous=old['model'][2]
            np.testing.assert_allclose(current[:,0],previous[s,:,g,0],rtol=2e-10,atol=2e-7)
            np.testing.assert_allclose(current[:,1],previous[s,:,g,3]-previous[t,:,g,3],rtol=2e-10,atol=2e-7)
            actual=sums[1,1,:,edge,g,:,0].sum(-2)
            expected=.5*(sim[s,:,g,:,0].sum(-1)+sim[t,:,g,:,0].sum(-1))
            np.testing.assert_allclose(actual[:,0],expected,rtol=1e-12,atol=1e-7)
    np.savez(out,sums=sums,sim=sim,edges=edges,cuts=e.CUT_NAMES,case=case,quantities=QUANTITIES)
    out.with_suffix('.json').write_text(json.dumps({'case':case,'checkpoint':pin,'emulator':preset['emulator'],
        'source_sha256':e.sha256(__file__),'latent_source_sha256':e.sha256(HERE.parent/'alignment_target9_20261106/portable_draws.py'),
        'measured':receipts,'gpu':device,'draws':64,'full_parent_block_size':1024,
        'generated_replay_passed':True,'conditional_reconstruction_passed':True,'seconds':time.monotonic()-start},indent=2)+'\n')
    print('DONE',case,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
