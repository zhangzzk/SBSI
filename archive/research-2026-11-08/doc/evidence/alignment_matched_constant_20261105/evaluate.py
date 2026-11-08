"""Frozen-emulator constant-shear evaluation, including true-selection diagnostic."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import scripts.evaluate_smooth_classifier_joint as ev
from scripts import joint_response_axes as axes
from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_matched_constant_20261105')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
e=module('constant_helpers',HERE.parent/'alignment_factorized_constant_20261105/helpers.py')
fm=module('factorized_model',HERE.parent/'alignment_frame_full_20261105/frame_model.py')

def main():
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    assert Path(ev.__file__).resolve().parents[1]==e.DATA/'sbsi_flow_f8_20260924'
    manifest=json.loads((HERE/'manifest.json').read_text())
    if args.replay:arm,case='retained',1000
    else:
        assert (Path(manifest['factorized_root'])/'replay_passed.json').exists()
        task=manifest['tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])];arm,case=task['arm'],task['case']
    dest=ROOT/arm;dest.mkdir(exist_ok=True)
    path=dest/f'case{case}.npz'
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    spec,paths,conditions=e.preset()
    if args.replay:
        bundle=load_measurement_model(paths['flow'],device='cuda')
        model,condition=bundle.model,bundle.condition_preprocessor;checkpoint=paths['flow']
    else:
        pin=manifest['checkpoints'][arm];checkpoint=Path(pin['path'])
        assert e.sha256(checkpoint)==pin['sha256']
        if arm=='joint_cartesian':
            bundle=load_measurement_model(checkpoint,device='cuda');model,condition=bundle.model,bundle.condition_preprocessor
        else:model,condition,_,_=fm.load(checkpoint)
    model.eval();candidate=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    assert candidate.conditioned and candidate.response_coordinate=='disk_velocity'
    started=time.monotonic()
    parent=Path(manifest['parent_root'])
    with np.load(parent/f'parent/case{case}.npz') as z,np.load(parent/f'c8/case{case}.npz') as c:
        ids=z['input_index'];context=z['context'].astype(float)
        np.testing.assert_array_equal(ids,c['input_index'])
        probability=np.stack([c['probability_plus'],c['probability_minus']])
    assert np.all(np.diff(ids)>0)
    truth,_=e.truth(case,.02);np.testing.assert_array_equal(truth['index_input'],ids)
    pairs=e.pair_table(e.scene_frame(truth),np.arange(len(truth)),conditions)
    pair_ids=pairs['index_input_p'].to_numpy()
    frames=axes.shear_frames(e.full_context(context),.02,1)
    contexts=[torch.as_tensor(condition.transform_frame(f),dtype=torch.float32,device='cuda') for f in frames]
    measured=[];usable=[];selection=[];receipts=[]
    for shear in [.02,-.02]:
        v,u,r=ev.measured_leg(e.SIMS,case,shear,ids)
        measured.append(v);usable.append(u);selection.append(e.output_cuts(v)&u[:,None]);receipts.append(r)
    bins=np.digitize(context[:,3],[24,25.5,27.5]);nc=len(e.CUTS)
    # mode,leg,cut,intrinsic-mag bin,(mass,flow1,flow2,emu1,emu2)
    sums=np.zeros((3,2,nc,4,5));sim=np.zeros((2,nc,4,3))
    for l in range(2):
        for k in range(nc):
            for b in range(4):
                keep=selection[l][:,k]&(bins==b)
                sim[l,k,b]=[keep.sum(),*measured[l][keep,:2].sum(0)]
    baseline=Path(manifest['baseline_root'])/f'sums/case{case}.npz'
    with np.load(baseline) as z:
        ci=[list(z['cuts']).index(c) for c in e.CUT_NAMES]
        saved_sim=z['sim'][:,ci,:,:3];saved_model=z['model'][:,:,ci,:,:5]
    np.testing.assert_allclose(sim,saved_sim,rtol=1e-12,atol=1e-8)
    print('READY',arm,case,'parents',len(ids),'pairs',len(pairs),'GPU',torch.cuda.get_device_name(),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),1024)):
            hi=min(len(ids),lo+1024)
            values=physical_context_draws(model,[c[lo:hi] for c in contexts],64,7301+10000019*case+1000003*block)
            p0=np.searchsorted(pair_ids,ids[lo]);p1=np.searchsorted(pair_ids,ids[hi-1],side='right')
            for leg,v in enumerate(values):
                assert bool(torch.isfinite(v).all())
                response,_=predict_scene_draw_responses(candidate,pairs.iloc[p0:p1],ids[lo:hi],v.cpu().numpy())
                shear=torch.tensor([.02 if leg==0 else -.02,0.],device='cuda',dtype=torch.float64)
                transformed,ld=disk_response_transport(v[...,:2],torch.as_tensor(response,device='cuda')[...,None]*shear)
                assert bool(torch.isfinite(transformed).all() and torch.isfinite(ld).all())
                shift=transformed-v[...,:2]
                mass,flow,emu=e.draw_moments(e.output_cuts(v),v,shift)
                own=torch.cat([mass[...,None],flow,emu],-1).cpu().numpy()
                uncut=torch.cat([torch.ones_like(mass[:,:1]),v[...,:2].mean(1),shift.mean(1)],-1).cpu().numpy()
                actual=selection[leg][lo:hi,:,None]*uncut[:,None,:]
                for mode,q,weight in [(0,own,probability[leg,lo:hi]),(1,own,usable[leg][lo:hi].astype(float)),(2,actual,np.ones(hi-lo))]:
                    for b in range(4):
                        take=bins[lo:hi]==b
                        sums[mode,leg,:,b]+=np.einsum('n,nkq->kq',weight[take],q[take])
            if block%100==0:print('PROGRESS',arm,case,'block',block,'seconds',round(time.monotonic()-started,1),flush=True)
    np.testing.assert_array_equal(sums[2,...,0],sim[...,0])
    assert np.isfinite(sums).all()
    replay=None
    if args.replay:
        np.testing.assert_allclose(sums[:2],saved_model,rtol=2e-10,atol=2e-7)
        replay=float(np.max(np.abs(sums[:2]-saved_model)))
    np.savez(path,model=sums,sim=sim,cuts=np.array(e.CUT_NAMES),case=case)
    receipt={'arm':arm,'case':case,'seconds':time.monotonic()-started,'parents':len(ids),'pairs':len(pairs),
             'measured':receipts,'measured_replay_max_abs':float(np.max(np.abs(sim-saved_sim))),
             'retained_replay_max_abs':replay,'checkpoint':str(checkpoint),'checkpoint_sha256':e.sha256(checkpoint),
             'emulator_sha256':spec['emulator']['sha256'],'classifier_sha256':spec['classifier']['sha256'],
             'source_sha256':e.sha256(__file__),'helper_sha256':e.sha256(HERE.parent/'alignment_factorized_constant_20261105/helpers.py'),
             'parent_sha256':e.sha256(parent/f'parent/case{case}.npz'),'flags_sha256':e.sha256(parent/f'c8/case{case}.npz')}
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if args.replay:(ROOT/'replay_passed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('DONE',arm,case,receipt['seconds'],flush=True)

if __name__=='__main__':main()
