"""Matched constant-shear tests with unchanged classifier and disk emulator."""
import importlib.util, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scripts.evaluate_smooth_classifier_joint as ev
from scripts import joint_response_axes as axes
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neighbour_eval_model',HERE/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.c;e=c.e
portable=c.module('neighbour_portable',HERE.parent/'alignment_target9_20261106/portable_draws.py')
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    task_index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    if task_index:assert (c.ROOT/'evaluation_pilot_passed.json').exists()
    manifest=json.loads((HERE/'manifest.json').read_text());task=manifest['tasks'][task_index]
    arm,case,axis=task['arm'],task['case'],task['axis'];base,stage=arm.split('_')
    completion=json.loads((c.ROOT/base/f'{stage}_completed.json').read_text());assert completion['reload_exact']
    checkpoint=c.ROOT/base/f'{stage}.pt';assert c.sha(checkpoint)==completion['sha256']
    model,condition,_,_=m.load(checkpoint)
    e.CUTS=manifest['cuts'];e.CUT_NAMES=tuple(x[0] for x in e.CUTS);center=e.CUT_NAMES.index('mag25_radius_gt0.60')
    e.SIMS=Path(manifest['sim_roots'][str(axis)])
    if axis==2:
        for leg in ['0.02','-0.02']:assert (e.SIMS.parent/f'case{case}_{leg}_verified.json').exists()
    dest=c.ROOT/'constant'/arm/f'axis{axis}';dest.mkdir(parents=True,exist_ok=True)
    out=dest/f'case{case}.npz';assert not out.exists()
    preset,paths,conditions=e.preset();emulator=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    parent=Path(manifest['parent_root']);flags=parent if axis==1 else c.ROOT.parent/'alignment_axis2_constant_20261105'
    with np.load(parent/f'parent/case{case}.npz') as z,np.load(flags/f'c8/case{case}.npz') as f:
        ids=z['input_index'];context=z['context'].astype(float);np.testing.assert_array_equal(ids,f['input_index'])
        probability=np.stack([f['probability_plus'],f['probability_minus']])
    truth,_=e.truth(case,.02);np.testing.assert_array_equal(truth['index_input'],ids)
    assert np.all(truth[f'gamma{axis}_input']==.02) and np.all(truth[f'gamma{3-axis}_input']==0)
    pairs=e.pair_table(e.scene_frame(truth),np.arange(len(truth)),conditions);pair_ids=pairs['index_input_p'].to_numpy()
    score=c.score_pairs(emulator,pairs,ids)
    # Reference score must replay the preceding four-scene diagnostic.
    with np.load(c.ROOT.parent/f'photometry_fixed_cell_20261106/targets{case}.npz') as z:
        ix=pd.Index(ids).get_indexer(z['input_index']);assert (ix>=0).all()
        np.testing.assert_allclose(score[ix],z['reference_coefficient'].sum(-1),rtol=2e-10,atol=2e-7)
    frames=axes.shear_frames(e.full_context(context),.02,axis)
    for frame in frames:frame[c.FEATURE]=score
    contexts=[torch.as_tensor(condition.transform_frame(frame),dtype=torch.float32,device='cuda') for frame in frames]
    measured=[];usable=[];selection=[];receipts=[]
    for shear in [.02,-.02]:
        v,u,r=ev.measured_leg(e.SIMS,case,shear,ids)
        measured.append(v);usable.append(u);selection.append(e.output_cuts(v)&u[:,None]);receipts.append(r)
    # Reproduce the common-four-U selection-score statistic only for axis1.
    common=None;target_pass=None
    if axis==1:
        all_values=[measured[0]];all_u=[usable[0]]
        split=e.DATA/'sbsi/galsbi_resim/constant_split_20261105/sims'
        for h in [.02,-.02]:
            v,u,_=ev.measured_leg(split,case,h,ids);all_values.append(v);all_u.append(u)
        all_values.append(measured[1]);all_u.append(usable[1])
        common=np.logical_and.reduce(all_u);target_pass=np.mean([e.output_cuts(v)[:,center] for v in all_values],axis=0)
    bins=np.digitize(context[:,4],manifest['intrinsic_size_edges_arcsec']);nc=len(e.CUTS)
    sums=np.zeros((3,2,nc,5,5));sim=np.zeros((2,nc,5,3));association=np.zeros(5)
    for leg in range(2):
        for k in range(nc):
            for b in range(5):
                keep=selection[leg][:,k]&(bins==b);sim[leg,k,b]=[keep.sum(),*measured[leg][keep,:2].sum(0)]
    started=time.monotonic();print('READY',arm,case,axis,len(ids),'pairs',len(pairs),'GPU',torch.cuda.get_device_name(),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),1024)):
            hi=min(len(ids),lo+1024)
            values=portable.physical_context_draws(model,[ctx[lo:hi] for ctx in contexts],64,7301+10000019*case+1000003*block)
            p0=np.searchsorted(pair_ids,ids[lo]);p1=np.searchsorted(pair_ids,ids[hi-1],side='right')
            pass_probability=np.zeros(hi-lo)
            for leg,v in enumerate(values):
                assert bool(torch.isfinite(v).all())
                response,_=predict_scene_draw_responses(emulator,pairs.iloc[p0:p1],ids[lo:hi],v.cpu().numpy())
                shear=torch.zeros(2,device='cuda',dtype=torch.float64);shear[axis-1]=.02 if leg==0 else -.02
                transformed,ld=disk_response_transport(v[...,:2],torch.as_tensor(response,device='cuda')[...,None]*shear)
                assert bool(torch.isfinite(transformed).all() and torch.isfinite(ld).all())
                shift=transformed-v[...,:2];keep=e.output_cuts(v)
                mass,flow,emu=e.draw_moments(keep,v,shift);own=torch.cat([mass[...,None],flow,emu],-1).cpu().numpy()
                uncut=torch.cat([torch.ones_like(mass[:,:1]),v[...,:2].mean(1),shift.mean(1)],-1).cpu().numpy()
                actual=selection[leg][lo:hi,:,None]*uncut[:,None,:]
                for mode,q,weight in [(0,own,probability[leg,lo:hi]),(1,own,usable[leg][lo:hi].astype(float)),(2,actual,np.ones(hi-lo))]:
                    for b in range(5):
                        take=bins[lo:hi]==b;sums[mode,leg,:,b]+=np.einsum('n,nkq->kq',weight[take],q[take])
                pass_probability+=.5*mass[:,center].cpu().numpy()
            if common is not None:
                take=common[lo:hi];y=target_pass[lo:hi][take];p=pass_probability[take];s=score[lo:hi][take]
                association+=[take.sum(),y.sum(),(y*s).sum(),p.sum(),(p*s).sum()]
            if block%100==0:print('PROGRESS',arm,case,axis,block,round(time.monotonic()-started,1),flush=True)
    np.testing.assert_array_equal(sums[2,...,0],sim[...,0]);assert np.isfinite(sums).all()
    with np.load(c.ROOT.parent/f'alignment_target9_20261106/axis{axis}/case{case}.npz') as z:
        np.testing.assert_array_equal(z['cuts'],e.CUT_NAMES);np.testing.assert_allclose(sim,z['sim'],rtol=1e-12,atol=1e-8)
    np.savez(out,model=sums,sim=sim,cuts=e.CUT_NAMES,case=case,axis=axis,association=association)
    c.write(out.with_suffix('.json'),{'arm':arm,'case':case,'axis':axis,'checkpoint_sha256':completion['sha256'],
        'model_source_sha256':c.sha(HERE/'model.py'),'source_sha256':c.sha(__file__),
        'emulator_sha256':preset['emulator']['sha256'],'classifier_sha256':preset['classifier']['sha256'],
        'latent_source_sha256':c.sha(HERE.parent/'alignment_target9_20261106/portable_draws.py'),
        'measured':receipts,'simulation_replay_passed':True,'reference_score_replay_passed':True,
        'true_selection_mass_exact':True,'seconds':time.monotonic()-started})
    print('DONE',arm,case,axis,time.monotonic()-started,flush=True)
if __name__=='__main__':main()
