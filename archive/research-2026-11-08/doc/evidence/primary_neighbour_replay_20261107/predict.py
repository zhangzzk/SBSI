"""Physical four-corner predictions for sampled primaries, with coherent source replay."""
import importlib.util,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts import joint_response_axes as axes
from sbsi.disk_response_transport import disk_response_transport
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses

assert os.environ.get('SLURM_JOB_ID')
H=Path(__file__).resolve().parent
p=H.parent/'neighbour_factor_swap_20261107/common.py'
sp=importlib.util.spec_from_file_location('replay_pinned_common',p);c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
OUT=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107')
cfg=c.manifest();e=c.e;e.CUTS=cfg['cuts'];e.CUT_NAMES=[v[0] for v in e.CUTS]
asm=json.loads((c.ROOT/'assembly_completed.json').read_text());arms=['initial','control','weighted']
torch.set_num_threads(4);spec,paths,conditions=e.preset()
emu=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
models={}
for name in arms:
    path=c.ROOT/name/'final.pt';assert c.sha(path)==asm['arms'][name]['sha256']
    models[name]=c.m.load(path)[:2]
for case in cfg['cases']:
    path=OUT/f'model{case}.npz';receipt=path.with_suffix('.json')
    if receipt.exists():
        d=json.loads(receipt.read_text());assert d['source_sha256']==c.sha(Path(__file__)) and d['data_sha256']==c.sha(path);continue
    samplefile=OUT/f'sample_{case}.feather';sample=pd.read_feather(samplefile)
    sids=sample.index_input.to_numpy();parent=Path(cfg['parent_root'])
    with np.load(parent/f'parent/case{case}.npz') as z,np.load(parent/f'c8/case{case}.npz') as f:
        ids=z['input_index'];context=z['context'].astype(float)
        np.testing.assert_array_equal(ids,f['input_index']);prob=np.stack([f['probability_plus'],f['probability_minus']])
    loc=pd.Index(ids).get_indexer(sids);assert (loc>=0).all() and len(set(loc))==512
    truth,_=e.truth(case,.02);np.testing.assert_array_equal(ids,truth.index_input)
    pairs=e.pair_table(e.scene_frame(truth),np.arange(len(ids)),conditions);pids=pairs.index_input_p.to_numpy()
    score=c.c.score_pairs(emu,pairs,ids);frames=axes.shear_frames(e.full_context(context),.02,1)
    for f in frames:f[c.c.FEATURE]=score
    contexts={n:[torch.as_tensor(cond.transform_frame(f),dtype=torch.float32,device='cuda') for f in frames] for n,(_,cond) in models.items()}
    selected=np.full((3,4,len(sids),len(e.CUTS),5),np.nan);sums=np.zeros((3,2,len(e.CUTS),5))
    t0=time.monotonic();print('READY',case,len(pairs),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),1024)):
            hi=min(lo+1024,len(ids));ix=np.flatnonzero((loc>=lo)&(loc<hi));take=loc[ix]-lo
            pp=pairs.iloc[np.searchsorted(pids,ids[lo]):np.searchsorted(pids,ids[hi-1],side='right')]
            vs=[c.portable.physical_context_draws(models[n][0],[f[lo:hi] for f in contexts[n]],64,7301+10000019*case+1000003*block) for n in arms]
            joint=torch.cat([v for pair in vs for v in pair],dim=1).cpu().numpy()
            pooled,_=predict_scene_draw_responses(emu,pp,ids[lo:hi],joint)
            for a,pair in enumerate(vs):
                for leg,v in enumerate(pair):
                    response=torch.as_tensor(pooled[:,(2*a+leg)*64:(2*a+leg+1)*64],device='cuda')
                    vel=torch.stack((response*(.02 if leg==0 else -.02),torch.zeros_like(response)),-1)
                    g,_=disk_response_transport(v[...,:2],vel);keep=e.output_cuts(v)
                    mass,flow,blend=e.draw_moments(keep,v,g-v[...,:2])
                    mom=torch.cat((mass[...,None],flow,blend),-1).cpu().numpy()
                    sums[a,leg]+=np.einsum('n,nkq->kq',prob[leg,lo:hi],mom)
                    if len(ix):
                        selected[a,3*leg,ix]=mom[take]
                        g1,_=disk_response_transport(v[take,...,:2],-vel[take])
                        # A neighbour-only flip changes only disk transport. Keep
                        # the existing mass and flow numerator from the native block.
                        alt=mom[take].copy()
                        alt[...,3:]=((keep[take].to(v.dtype)[...,None])*(g1-v[take,...,:2])[:,:,None,:]).mean(1).cpu().numpy()
                        selected[a,1+leg,ix]=alt
            if block%200==0:print('PROGRESS',case,block,round(time.monotonic()-t0),flush=True)
    assert np.isfinite(selected).all()
    worst=0.
    def rr(x):return (x[0,:,1:]/x[0,:,:1]-x[1,:,1:]/x[1,:,:1])/.04
    for a,n in enumerate(arms):
        with np.load(c.ROOT/'constant'/n/'axis1'/f'case{case}.npz') as z:
            ref=z['model'][0].sum(-2)
        np.testing.assert_allclose(sums[a,...,0],ref[...,0],rtol=0,atol=1e-6)
        delta=float(np.max(abs(rr(sums[a])-rr(ref))));worst=max(worst,delta);assert delta<=cfg['reference_response_atol'],delta
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,moments=selected,probability=prob[:,loc],cuts=e.CUT_NAMES,arms=arms,input_index=sids,weights=sample.weight.to_numpy());tmp.replace(path)
    c.write(receipt,dict(case=case,source_sha256=c.sha(Path(__file__)),data_sha256=c.sha(path),sample_sha256=c.sha(samplefile),max_coherent_response_replay=worst,source_models={n:asm['arms'][n]['sha256'] for n in arms},emulator_sha256=spec['emulator']['sha256'],classifier_sha256=spec['classifier']['sha256'],states=['++','+-','-+','--']))
    print('DONE',case,worst,flush=True)
