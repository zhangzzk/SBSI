"""Resolve the validated factorial diagonal into photometry-bin moments."""
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
spec=importlib.util.spec_from_file_location('photo_factorial',HERE.parent/'photometry_selection_factorial_20261106/evaluate.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
a=f.a;e=f.e
ROOT=a.DATA/'sbsi_caches/photometry_response_bins_20261106'
RADIUS=np.array([0,1,2,2.5,2.8,3,3.1,3.2,3.4,3.6,4,4.5,5,6,8,12,20,np.inf])
MAG=np.array([-np.inf,20,21,22,23,23.5,24,24.25,24.5,24.75,25,25.25,25.5,26,27,np.inf])
NR=len(RADIUS)-1;NM=len(MAG)-1

def histogram(size,weight,change,properties):
    """Sum paired draw contributions; full endpoint support is retained."""
    size=np.asarray(size);p=np.asarray(properties);w=np.asarray(weight)/weight.shape[1]
    # A generated zero radius cannot pass the target cut. Preserve zero-weight
    # rows without allowing a numerical log(0) convention to move positive mass.
    rb=np.searchsorted(np.log(RADIUS[1:-1]),p[...,0],side='right')
    mb=np.searchsorted(MAG[1:-1],p[...,1],side='right')
    loc=((size[:,None]*NR+rb)*NM+mb).ravel()
    n=5*NR*NM
    return np.stack([np.bincount(loc,weights=v.ravel(),minlength=n) for v in
                     [w,w*change,w*p[...,2]]],-1).reshape(5,NR,NM,3)

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    index=int(os.environ['SLURM_ARRAY_TASK_ID']);case=a.CASES[index]
    device=torch.cuda.get_device_name();assert ['V100','A40','V100','RTX 2080 Ti'][index] in device
    ROOT.mkdir(parents=True,exist_ok=True);out=ROOT/f'case{case}.npz';assert not out.exists()
    torch.set_num_threads(4);e.CUTS=a.CUTS;e.CUT_NAMES=tuple(x[0] for x in a.CUTS)
    preset,paths,conditions=e.preset()
    pin=json.loads((f.OLD/'manifest.json').read_text())['arms']['nine_weak']
    assert e.sha256(pin['path'])==pin['sha256']
    model,condition,_,_=f.cp.fm.load(pin['path'])
    emulator=a.DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    with np.load(e.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context'].astype(float)
    with np.load(a.STUDY/f'groups_case{case}.npz') as z:lookup=pd.Series(z['group'],index=z['input_index'])
    group=lookup.loc[ids].to_numpy();values=[];usable=[];receipts=[]
    for root,h in [(e.SIMS,.02),(a.STUDY/'sims',.02),(a.STUDY/'sims',-.02),(e.SIMS,-.02)]:
        v,u,record=a.ev.measured_leg(root,case,h,ids);values.append(v);usable.append(u);receipts.append(record)
    common=np.logical_and.reduce(usable);values=np.array(values);cuts=e.output_cuts(values).astype(float)
    size=np.digitize(context[:,4],[.15,.25,.4,.7])
    contexts=[torch.as_tensor(condition.transform_frame(frame),dtype=torch.float32,device='cuda')
              for frame in axes.shear_frames(e.full_context(context),.02,1)]
    truth,_=e.truth(case,.02);np.testing.assert_array_equal(truth['index_input'],ids)
    pairs=e.pair_table(e.scene_frame(truth),np.flatnonzero(common),conditions)
    pairids=pairs['index_input_p'].to_numpy();pg=lookup.loc[pairs['index_input_s']].to_numpy()
    sums=np.zeros((2,4,5,NR,NM,3));start=time.monotonic();verified=False
    print('READY',case,device,len(ids),int(common.sum()),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),1024)):
            hi=min(len(ids),lo+1024);nb=hi-lo;keep=common[lo:hi]
            if not keep.any():continue
            bid=ids[lo:hi][keep];gg=group[lo:hi][keep];sz=size[lo:hi][keep]
            bank=f.portable.latent_bank(nb,64,4,7301+10000019*case+1000003*block)
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
            gmask=[e.output_cuts(v)[...,3].double() for v in generated]
            measured=[];bm=[];groups=torch.as_tensor(gg,dtype=torch.long,device='cuda')
            for state in range(4):
                plus=torch.as_tensor(a.SIGNS[state],device='cuda')[groups]>0
                ctx=torch.where(plus[:,None],contexts[0][lo:hi][keep],contexts[1][lo:hi][keep])
                photo=torch.as_tensor(values[state,lo:hi][keep,2:],dtype=torch.float64,device='cuda')
                v=f.cp.conditional_physical(model,z,ctx[:,None,:].expand(-1,64,-1).reshape(len(bid)*64,-1),
                    photo[:,None,:].expand(-1,64,-1).reshape(len(bid)*64,2)).reshape(len(bid),64,4)
                torch.testing.assert_close(v[...,2:],photo[:,None,:].expand(-1,64,-1),rtol=0,atol=0)
                measured.append(v);bm.append(coefficients(v[:,:1]))
            if not verified:
                ctx=contexts[0][lo:hi][keep][:,None,:].expand(-1,64,-1).reshape(len(bid)*64,-1)
                photo=generated[0][...,2:].reshape(len(bid)*64,2);valid=(photo>0).all(-1)
                replay=f.cp.conditional_physical(model,z[valid],ctx[valid],photo[valid])
                torch.testing.assert_close(replay,generated[0].reshape(-1,4)[valid],rtol=2e-5,atol=2e-6);verified=True
            for edge,(s,t) in enumerate(a.EDGES):
                g=1 if edge<2 else 0;other=1-g;leg=edge%2;rows=gg==g
                change=f.edge_change(generated[leg][...,:2],bg[leg],s,t)
                props=f.properties(generated[leg],bg[leg][...,other])
                sums[0,edge]+=histogram(sz[rows],gmask[leg][rows].cpu().numpy(),change[rows].cpu().numpy(),props[rows].cpu().numpy())
                wm=torch.as_tensor(.5*(cuts[s,lo:hi][keep,3]+cuts[t,lo:hi][keep,3]),device='cuda')[:,None].expand(-1,64)
                for source in [s,t]:
                    change=f.edge_change(measured[source][...,:2],bm[source],s,t)
                    props=f.properties(measured[source],bm[source][...,other])
                    sums[1,edge]+=.5*histogram(sz[rows],wm[rows].cpu().numpy(),change[rows].cpu().numpy(),props[rows].cpu().numpy())
            if block%100==0:print('PROGRESS',case,block,round(time.monotonic()-start,1),flush=True)
    assert verified
    with np.load(f.ROOT/f'case{case}.npz') as previous:
        for edge in range(4):
            g=1 if edge<2 else 0
            for branch in range(2):
                expected=previous['sums'][branch,branch,3,edge,g,:,0][:,[0,1,4]]
                np.testing.assert_allclose(sums[branch,edge].sum((1,2)),expected,rtol=2e-10,atol=2e-7)
    np.savez(out,sums=sums,radius_edges=RADIUS,magnitude_edges=MAG,case=case)
    out.with_suffix('.json').write_text(json.dumps({'case':case,'checkpoint':pin,'emulator':preset['emulator'],
        'source_sha256':e.sha256(__file__),'helper_sha256':e.sha256(f.__file__),'measured':receipts,'gpu':device,
        'draws':64,'factorial_replay_passed':True,'seconds':time.monotonic()-start},indent=2)+'\n')
    print('DONE',case,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
