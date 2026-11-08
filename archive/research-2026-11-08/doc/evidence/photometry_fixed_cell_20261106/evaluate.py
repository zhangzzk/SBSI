"""Hold emulator photometry exactly fixed within the preceding diagnostic bins."""
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
spec=importlib.util.spec_from_file_location('photo_bins',HERE.parent/'photometry_response_bins_20261106/evaluate.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
f=b.f;a=b.a;e=b.e
ROOT=a.DATA/'sbsi_caches/photometry_fixed_cell_20261106'
R_CENTRE=np.r_[.5,np.sqrt(b.RADIUS[1:-2]*b.RADIUS[2:-1]),25.]
M_CENTRE=np.r_[19.,.5*(b.MAG[1:-2]+b.MAG[2:-1]),28.]

def cells(props):
    return (np.searchsorted(np.log(b.RADIUS[1:-1]),props[...,0],side='right'),
            np.searchsorted(b.MAG[1:-1],props[...,1],side='right'))

def centre_draws(v,props):
    r,m=cells(props);out=np.array(v,copy=True)
    out[...,2]=R_CENTRE[r];out[...,3]=10.**(.4*(30-M_CENTRE[m]))
    rr=np.searchsorted(np.log(b.RADIUS[1:-1]),np.log(out[...,2]),side='right')
    mm=np.searchsorted(b.MAG[1:-1],30-2.5*np.log10(out[...,3]),side='right')
    np.testing.assert_array_equal(rr,r);np.testing.assert_array_equal(mm,m)
    return out

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    index=int(os.environ['SLURM_ARRAY_TASK_ID']);case=a.CASES[index]
    device=torch.cuda.get_device_name();assert ['V100','A40','V100','RTX 2080 Ti'][index] in device
    ROOT.mkdir(parents=True,exist_ok=True);out=ROOT/f'case{case}.npz';assert not out.exists()
    torch.set_num_threads(4);e.CUTS=a.CUTS;e.CUT_NAMES=tuple(x[0] for x in a.CUTS)
    preset,paths,conditions=e.preset();pin=json.loads((f.OLD/'manifest.json').read_text())['arms']['nine_weak']
    assert e.sha256(pin['path'])==pin['sha256']
    model,condition,_,_=f.cp.fm.load(pin['path'])
    emulator=a.DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    with np.load(e.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context'].astype(float)
    with np.load(a.STUDY/f'groups_case{case}.npz') as z:lookup=pd.Series(z['group'],index=z['input_index'])
    group=lookup.loc[ids].to_numpy();values=[];usable=[];receipts=[]
    for root,h in [(e.SIMS,.02),(a.STUDY/'sims',.02),(a.STUDY/'sims',-.02),(e.SIMS,-.02)]:
        v,u,record=a.ev.measured_leg(root,case,h,ids);values.append(v);usable.append(u);receipts.append(record)
    common=np.logical_and.reduce(usable);values=np.array(values);cuts=e.output_cuts(values).astype(float)
    size=np.digitize(context[:,4],[.15,.25,.4,.7]);common_index=np.cumsum(common)-1
    contexts=[torch.as_tensor(condition.transform_frame(frame),dtype=torch.float32,device='cuda')
              for frame in axes.shear_frames(e.full_context(context),.02,1)]
    truth,_=e.truth(case,.02);np.testing.assert_array_equal(truth['index_input'],ids)
    pairs=e.pair_table(e.scene_frame(truth),np.flatnonzero(common),conditions)
    pairids=pairs['index_input_p'].to_numpy();pg=lookup.loc[pairs['index_input_s']].to_numpy()
    sums=np.zeros((2,4,5,b.NR,b.NM,3));target=np.zeros((2,4,int(common.sum()),3))
    reference=np.zeros((int(common.sum()),2));start=time.monotonic()
    print('READY',case,device,len(ids),int(common.sum()),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),1024)):
            hi=min(len(ids),lo+1024);nb=hi-lo;keep=common[lo:hi]
            if not keep.any():continue
            bid=ids[lo:hi][keep];gg=group[lo:hi][keep];sz=size[lo:hi][keep];ci=common_index[lo:hi][keep]
            bank=f.portable.latent_bank(nb,64,4,7301+10000019*case+1000003*block)
            latent=torch.from_numpy(bank).to('cuda').reshape(nb*64,4)
            generated=[]
            for ctx in contexts:
                v,_=model.forward_physical(latent,ctx[lo:hi,None,:].expand(-1,64,-1).reshape(nb*64,-1))
                generated.append(v.reshape(nb,64,4)[keep])
            p0=np.searchsorted(pairids,bid[0]);p1=np.searchsorted(pairids,bid[-1],side='right');pp=pairs.iloc[p0:p1]
            def coefficients(v):
                return np.stack([a.predict_scene_draw_responses(emulator,pp.loc[pg[p0:p1]==g],bid,v)[0] for g in [0,1]],-1)
            def evaluate(v):
                raw=v.cpu().numpy();original=coefficients(raw)
                props=f.properties(v,torch.zeros(v.shape[:2],dtype=torch.float64,device='cuda')).cpu().numpy()
                fixed=coefficients(centre_draws(raw,props))
                return props,original,fixed
            gd=[evaluate(v) for v in generated]
            md=[evaluate(torch.as_tensor(values[state,lo:hi][keep,None,:],dtype=torch.float64,device='cuda')) for state in range(4)]
            ref=np.zeros((len(bid),1,4));ref[...,2]=3.5;ref[...,3]=10.**(.4*(30-24.5))
            reference[ci]=coefficients(ref)[:,0]
            gmask=[e.output_cuts(v)[...,3].double().cpu().numpy() for v in generated]
            for edge,(s,t) in enumerate(a.EDGES):
                g=1 if edge<2 else 0;other=1-g;leg=edge%2;rows=gg==g
                wm=.5*(cuts[s,lo:hi][keep,3]+cuts[t,lo:hi][keep,3])[:,None]
                sources=[(0,1.,gmask[leg],gd[leg]),(1,.5,wm,md[s]),(1,.5,wm,md[t])]
                for branch,scale,weight,(props,original,fixed) in sources:
                    props=props.copy();props[...,2]=fixed[...,other]
                    change=original[...,other]
                    sums[branch,edge]+=scale*b.histogram(sz[rows],weight[rows],change[rows],props[rows])
                    moments=np.stack([weight.mean(1),(weight*change).mean(1),(weight*fixed[...,other]).mean(1)],-1)
                    target[branch,edge,ci[rows]]+=scale*moments[rows]
            if block%100==0:print('PROGRESS',case,block,round(time.monotonic()-start,1),flush=True)
    with np.load(b.ROOT/f'case{case}.npz') as previous:
        np.testing.assert_allclose(sums[...,0],previous['sums'][...,0],rtol=2e-10,atol=2e-7)
        np.testing.assert_allclose(sums[...,1],previous['sums'][...,2],rtol=2e-10,atol=2e-7)
    np.testing.assert_allclose(target.sum(2),sums.sum((2,3,4)),rtol=2e-10,atol=2e-7)
    np.savez(out,sums=sums,radius_edges=b.RADIUS,magnitude_edges=b.MAG,radius_centres=R_CENTRE,magnitude_centres=M_CENTRE,case=case)
    np.savez(ROOT/f'targets{case}.npz',moments=target,reference_coefficient=reference,input_index=ids[common],
             context=context[common],group=group[common],case=case)
    out.with_suffix('.json').write_text(json.dumps({'case':case,'checkpoint':pin,'emulator':preset['emulator'],
        'source_sha256':e.sha256(__file__),'helper_sha256':e.sha256(b.__file__),'measured':receipts,'gpu':device,
        'draws':64,'original_histogram_replay_passed':True,'representative_bins_preserved':True,
        'target_moment_identity_passed':True,'seconds':time.monotonic()-start},indent=2)+'\n')
    print('DONE',case,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
