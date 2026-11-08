"""Measured photometry, conditional flow shapes, unchanged neighbour emulator."""
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
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=module('measured_anchor',HERE/'evaluate_measured_transport.py')
cp=module('conditional_shape',HERE.parent/'alignment_conditioned_photometry_20261105/evaluate.py')
portable=module('portable_conditional',HERE.parent/'alignment_target9_20261106/portable_draws.py')
e=a.e
ARMS=['baseline','nine_weak']

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    case=a.CASES[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    root=a.ROOT/'conditional_transport';root.mkdir(parents=True,exist_ok=True)
    out=root/f'case{case}.npz';assert not out.with_suffix('.json').exists()
    torch.set_num_threads(4);e.CUTS=a.CUTS;e.CUT_NAMES=tuple(x[0] for x in a.CUTS)
    preset,paths,conditions=e.preset()
    candidate=a.DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    pins=json.loads((HERE/'manifest.json').read_text())['arms']
    models={};preprocess={}
    for arm in ARMS:
        assert e.sha256(pins[arm]['path'])==pins[arm]['sha256']
        models[arm],preprocess[arm],_,_=cp.fm.load(pins[arm]['path'])
    with np.load(e.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context']
    with np.load(a.STUDY/f'groups_case{case}.npz') as z:lookup=pd.Series(z['group'],index=z['input_index'])
    group=lookup.loc[ids].to_numpy();values=[];usable=[];receipts=[]
    for r,h in [(e.SIMS,.02),(a.STUDY/'sims',.02),(a.STUDY/'sims',-.02),(e.SIMS,-.02)]:
        v,u,record=a.ev.measured_leg(r,case,h,ids);values.append(v);usable.append(u);receipts.append(record)
    common=np.logical_and.reduce(usable);rows=np.flatnonzero(common)
    values=np.array(values)[:,common];ids=ids[common];group=group[common];context=context[common]
    size=np.digitize(context[:,4],[.15,.25,.4,.7]);cuts=e.output_cuts(values).astype(float)
    contexts={arm:[torch.as_tensor(preprocess[arm].transform_frame(f),dtype=torch.float32,device='cuda')
        for f in axes.shear_frames(e.full_context(context),.02,1)] for arm in ARMS}
    verification={arm:cp.verify_conditioning(models[arm],contexts[arm][0][:32]) for arm in ARMS}
    truth,_=e.truth(case,.02);np.testing.assert_array_equal(truth['index_input'].to_numpy()[rows],ids)
    pairs=e.pair_table(e.scene_frame(truth),rows,conditions);pairids=pairs['index_input_p'].to_numpy();pg=lookup.loc[pairs['index_input_s']].to_numpy()
    # Reuse exact measured edge sums only after independently replaying counts/shapes.
    with np.load(a.ROOT/'measured_transport'/f'case{case}.npz') as z:
        sim=z['sim'];edges=z['edges'];measured_cuts=list(z['cuts'])
    assert measured_cuts==list(e.CUT_NAMES)
    for state in range(4):
        for k in range(4):
            for g in [0,1]:
                use=(group==g)&cuts[state,:,k].astype(bool)
                np.testing.assert_allclose(sim[state,k,g].sum(0),[use.sum(),*values[state,use,:2].sum(0)],rtol=1e-12,atol=1e-7)
    pred=np.zeros((2,4,4,2,5,3));start=time.monotonic()
    print('READY',case,len(ids),'two conditional models',flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),2048)):
            hi=min(len(ids),lo+2048);nb=hi-lo;groups=torch.as_tensor(group[lo:hi],dtype=torch.long,device='cuda')
            p0=np.searchsorted(pairids,ids[lo]);p1=np.searchsorted(pairids,ids[hi-1],side='right');pp=pairs.iloc[p0:p1]
            coeff=[]
            for state in range(4):
                bs=[a.predict_scene_draw_responses(candidate,pp.loc[pg[p0:p1]==g],ids[lo:hi],values[state,lo:hi,None,:])[0][:,0] for g in [0,1]]
                coeff.append(torch.as_tensor(np.stack(bs,-1),dtype=torch.float64,device='cuda'))
            seed=202611061800+10000019*case+1000003*block
            z=torch.from_numpy(portable.latent_bank(nb,64,4,seed)).to('cuda').reshape(nb*64,4)
            for ai,arm in enumerate(ARMS):
                shape=[]
                for state in range(4):
                    plus=torch.as_tensor(a.SIGNS[state],device='cuda')[groups]>0
                    ctx=torch.where(plus[:,None],contexts[arm][0][lo:hi],contexts[arm][1][lo:hi])
                    phot=torch.as_tensor(values[state,lo:hi,2:],dtype=torch.float64,device='cuda')
                    v=cp.conditional_physical(models[arm],z,ctx[:,None,:].expand(-1,64,-1).reshape(nb*64,-1),
                                             phot[:,None,:].expand(-1,64,-1).reshape(nb*64,2)).reshape(nb,64,4)
                    torch.testing.assert_close(v[...,2:],phot[:,None,:].expand(-1,64,-1),rtol=0,atol=0)
                    shape.append(v[...,:2])
                for edge,(s,t) in enumerate(a.EDGES):
                    g=1 if edge<2 else 0;changed=1-g;terms=[];linear=[]
                    for source in [s,t]:
                        shifted=[]
                        for state in [s,t]:
                            vel=.02*(coeff[source]*coeff[source].new_tensor(a.SIGNS[state])).sum(-1)
                            velocity=torch.stack([vel,vel*0],-1)[:,None,:]
                            shifted.append(a.disk_response_transport(shape[source],velocity)[0])
                        terms.append((shifted[0][...,0]-shifted[1][...,0]).mean(1))
                        observed=shifted[0 if source==s else 1]
                        linear.append(.04*coeff[source][:,changed]*(1-observed[...,0]**2+observed[...,1]**2).mean(1))
                    change=torch.stack([*terms,.5*(linear[0]+linear[1])],-1).cpu().numpy()
                    assert np.isfinite(change).all()
                    sw=.5*(cuts[s,lo:hi]+cuts[t,lo:hi])
                    for b in range(5):
                        keep=(group[lo:hi]==g)&(size[lo:hi]==b)
                        pred[ai,:,edge,g,b]+=np.einsum('nk,nq->kq',sw[keep],change[keep])
            if block%30==0:print('PROGRESS',case,block,round(time.monotonic()-start,1),flush=True)
    for ai,arm in enumerate(ARMS):
        dest=root/arm;dest.mkdir(exist_ok=True)
        np.savez(dest/f'case{case}.npz',sim=sim,edges=edges,pred=pred[ai],case=case,cuts=e.CUT_NAMES)
        (dest/f'case{case}.json').write_text(json.dumps({'case':case,'arm':arm,'checkpoint':pins[arm],
            'emulator':preset['emulator'],'verification':verification[arm],'measured_replay_passed':True,
            'source_sha256':e.sha256(__file__),'latent_source_sha256':e.sha256(HERE.parent/'alignment_target9_20261106/portable_draws.py'),
            'measured':receipts,'draws':64,'block_size':2048,'seed_formula':'202611061800+10000019*case+1000003*block',
            'definition':'Conditional flow shapes at measured photometry; endpoint-averaged opposite-group transport.'},indent=2)+'\n')
    out.with_suffix('.json').write_text(json.dumps({'case':case,'completed':ARMS,'seconds':time.monotonic()-start},indent=2)+'\n')
    print('DONE',case,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
