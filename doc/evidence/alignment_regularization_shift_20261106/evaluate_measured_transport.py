"""Conditional neighbour transport using observed endpoints; no flow draws."""
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import scripts.evaluate_smooth_classifier_joint as ev
from sbsi.output_conditioned_response import DiskResponseMoment, predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport

HERE=Path(__file__).resolve().parent
DATA=Path('/project/ls-gruen/users/zekang.zhang')
ROOT=DATA/'sbsi_caches/alignment_regularization_shift_20261106'
STUDY=DATA/'sbsi/galsbi_resim/constant_split_20261105'
CASES=[1000,1040,1080,1120]
CUTS=[('uncut',None,None),('mag25',None,25.),('radius3',('>',3.),None),('mag25_radius_gt0.60',('>',3.),25.)]
EDGES=[(0,2),(1,3),(0,1),(2,3)]
SIGNS=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
spec=importlib.util.spec_from_file_location('anchor_helpers',HERE.parent/'alignment_factorized_constant_20261105/helpers.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def transport_edge(shape, coefficients, source, target):
    """Predict target from source observation; coefficients stay source-fixed."""
    velocities=[]
    for state in [source,target]:
        v=.02*(coefficients*coefficients.new_tensor(SIGNS[state])).sum(-1)
        velocities.append(torch.stack([v,torch.zeros_like(v)],-1))
    base,_=disk_response_transport(shape,velocities[0],inverse=True)
    replay,_=disk_response_transport(base,velocities[0])
    torch.testing.assert_close(replay,shape,rtol=0,atol=2e-12)
    predicted,_=disk_response_transport(base,velocities[1])
    return predicted

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    case=CASES[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    dest=ROOT/'measured_transport';dest.mkdir(parents=True,exist_ok=True)
    out=dest/f'case{case}.npz';assert not out.exists(),out
    torch.set_num_threads(4);e.CUTS=CUTS;e.CUT_NAMES=tuple(x[0] for x in CUTS)
    preset,paths,conditions=e.preset()
    candidate=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    with np.load(e.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context']
    with np.load(STUDY/f'groups_case{case}.npz') as z:lookup=pd.Series(z['group'],index=z['input_index'])
    group=lookup.loc[ids].to_numpy()
    values=[];usable=[];receipts=[]
    for root,h in [(e.SIMS,.02),(STUDY/'sims',.02),(STUDY/'sims',-.02),(e.SIMS,-.02)]:
        v,u,r=ev.measured_leg(root,case,h,ids);values.append(v);usable.append(u);receipts.append(r)
    common=np.logical_and.reduce(usable);parent_rows=np.flatnonzero(common)
    values=np.array(values)[:,common];ids=ids[common];group=group[common]
    size=np.digitize(context[common,4],[.15,.25,.4,.7])
    truth,_=e.truth(case,.02)
    np.testing.assert_array_equal(truth['index_input'].to_numpy()[parent_rows],ids)
    pairs=e.pair_table(e.scene_frame(truth),parent_rows,conditions)
    pairids=pairs['index_input_p'].to_numpy();pg=lookup.loc[pairs['index_input_s']].to_numpy()
    cuts=e.output_cuts(values).astype(float)
    # state, cut, group, intrinsic size, [count, shape1, shape2]
    sim=np.zeros((4,4,2,5,3));edges=np.zeros((4,4,2,5,2));pred=np.zeros((4,4,2,5,3))
    for k in range(4):
        for g in [0,1]:
            for b in range(5):
                keep=(group==g)&(size==b)
                for state in range(4):
                    s=cuts[state,keep,k];v=values[state,keep,:2]
                    sim[state,k,g,b]=[s.sum(),*(s[:,None]*v).sum(0)]
                for edge,(a,z) in enumerate(EDGES):
                    sa=cuts[a,keep,k];sz=cuts[z,keep,k];ea=values[a,keep,0];ez=values[z,keep,0]
                    edges[k,edge,g,b]=[np.sum(.5*(sa+sz)*(ea-ez)),np.sum(.5*(ea+ez)*(sa-sz))]
    with np.load(HERE.parent/'constant_split_20261105/sums.npz') as old:
        ci=list(old['cases']).index(case)
        np.testing.assert_allclose(sim.sum(-2),old['sim'][ci,1].sum(-2),rtol=1e-12,atol=1e-7)
        np.testing.assert_allclose(edges.sum(-2),old['common_U_edges'][ci,:,:4],rtol=1e-12,atol=1e-7)
    start=time.monotonic();print('READY',case,'common targets',len(ids),'pairs',len(pairs),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),4096)):
            hi=min(len(ids),lo+4096);pp0=np.searchsorted(pairids,ids[lo]);pp1=np.searchsorted(pairids,ids[hi-1],side='right')
            pp=pairs.iloc[pp0:pp1];coeff=[]
            shapes=torch.as_tensor(values[:,lo:hi,:2],dtype=torch.float64,device='cuda')
            for state in range(4):
                bs=[predict_scene_draw_responses(candidate,pp.loc[pg[pp0:pp1]==g],ids[lo:hi],values[state,lo:hi,None,:])[0][:,0] for g in [0,1]]
                coeff.append(torch.as_tensor(np.stack(bs,-1),dtype=torch.float64,device='cuda'))
            for edge,(a,z) in enumerate(EDGES):
                target_group=1 if edge<2 else 0;changed_group=1-target_group
                da=shapes[a]-transport_edge(shapes[a],coeff[a],a,z)
                dz=transport_edge(shapes[z],coeff[z],z,a)-shapes[z]
                gain=1-shapes[...,0]**2+shapes[...,1]**2
                linear=.04*.5*(gain[a]*coeff[a][:,changed_group]+gain[z]*coeff[z][:,changed_group])
                change=torch.stack([da[:,0],dz[:,0],linear],-1).cpu().numpy()
                assert np.isfinite(change).all()
                sw=.5*(cuts[a,lo:hi]+cuts[z,lo:hi])
                for b in range(5):
                    keep=(group[lo:hi]==target_group)&(size[lo:hi]==b)
                    pred[:,edge,target_group,b]+=np.einsum('nk,nq->kq',sw[keep],change[keep])
            if block%20==0:print('PROGRESS',case,block,round(time.monotonic()-start,1),flush=True)
    np.savez(out,sim=sim,edges=edges,pred=pred,cuts=e.CUT_NAMES,case=case)
    out.with_suffix('.json').write_text(json.dumps({'case':case,'emulator':preset['emulator'],
        'source_sha256':e.sha256(__file__),'measured':receipts,'common_usable':len(ids),'pairs':len(pairs),
        'size_bins_arcsec':[0,.15,.25,.4,.7,'infinity'],'seconds':time.monotonic()-start,
        'definition':'Measured endpoint anchors; fixed output-conditioned coefficients during each predicted edge. Conditional diagnostic, not generative prediction.',
        'prediction_columns':['anchor_plus','anchor_minus','infinitesimal_endpoint_average'],
        'measured_replay_passed':True},indent=2)+'\n')
    print('DONE',case,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
