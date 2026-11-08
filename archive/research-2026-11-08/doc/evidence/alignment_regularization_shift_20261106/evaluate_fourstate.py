"""Reuse fixed four-state images to separate own/opposite-group responses."""
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
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport

HERE=Path(__file__).resolve().parent
DATA=Path('/project/ls-gruen/users/zekang.zhang')
ROOT=DATA/'sbsi_caches/alignment_regularization_shift_20261106'
STUDY=DATA/'sbsi/galsbi_resim/constant_split_20261105'
CASES=[1000,1040,1080,1120]
CUTS=[('uncut',None,None),('mag25',None,25.),('radius3',('>',3.),None),('mag25_radius_gt0.60',('>',3.),25.)]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
e=module('helpers',HERE.parent/'alignment_factorized_constant_20261105/helpers.py')
fm=module('factorized',HERE.parent/'alignment_factorized_20261105/frame_model.py')
portable=module('portable',HERE.parent/'alignment_target9_20261106/portable_draws.py')

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    idx=int(os.environ['SLURM_ARRAY_TASK_ID']);case=CASES[idx//2];arm=['baseline','nine_weak'][idx%2]
    assert json.loads((HERE.parent/'constant_split_20261105/control_report.json').read_text())['passed']
    check=json.loads((DATA/'sbsi_caches/alignment_target9_20261106/device_verification.json').read_text())
    assert check['portable_source_sha256']==e.sha256(HERE.parent/'alignment_target9_20261106/portable_draws.py')
    manifest=json.loads((HERE/'manifest.json').read_text());pin=manifest['arms'][arm]
    assert e.sha256(pin['path'])==pin['sha256'];e.CUTS=CUTS;e.CUT_NAMES=tuple(x[0] for x in CUTS)
    dest=ROOT/'fourstate'/arm;dest.mkdir(parents=True,exist_ok=True);out=dest/f'case{case}.npz'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(4);preset,paths,conditions=e.preset()
    model,condition,_,_=fm.load(pin['path']);model.eval()
    candidate=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    with np.load(e.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context'].astype(float)
    with np.load(e.OUT/f'c8/case{case}.npz') as z:
        np.testing.assert_array_equal(ids,z['input_index']);prob=np.stack([z['probability_plus'],z['probability_minus']])
    with np.load(STUDY/f'groups_case{case}.npz') as z:lookup=pd.Series(z['group'],index=z['input_index']);group=lookup.loc[ids].to_numpy()
    truth,_=e.truth(case,.02);np.testing.assert_array_equal(truth['index_input'],ids)
    pairs=e.pair_table(e.scene_frame(truth),np.arange(len(ids)),conditions);pairids=pairs['index_input_p'].to_numpy();pg=lookup.loc[pairs['index_input_s']].to_numpy()
    contexts=[torch.as_tensor(condition.transform_frame(f),dtype=torch.float32,device='cuda') for f in axes.shear_frames(e.full_context(context),.02,1)]
    values=[];usable=[];receipts=[]
    for root,h in [(e.SIMS,.02),(STUDY/'sims',.02),(STUDY/'sims',-.02),(e.SIMS,-.02)]:
        v,u,r=ev.measured_leg(root,case,h,ids);values.append(v);usable.append(u);receipts.append(r)
    common=np.logical_and.reduce(usable)
    sim=np.zeros((2,4,4,2,3));sums=np.zeros((3,4,4,2,5));direct_sums=np.zeros((2,2,4,5))
    for state,v in enumerate(values):
        cuts=e.output_cuts(v)
        for mode,u in enumerate([usable[state],common]):
            for k in range(4):
                for g in [0,1]:
                    use=u&cuts[:,k]&(group==g);sim[mode,state,k,g]=[use.sum(),*v[use,:2].sum(0)]
    with np.load(HERE.parent/'constant_split_20261105/sums.npz') as old:
        np.testing.assert_allclose(sim,old['sim'][CASES.index(case)].sum(-2),rtol=1e-12,atol=1e-7)
    started=time.monotonic();print('READY',arm,case,'parents',len(ids),torch.cuda.get_device_name(),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),1024)):
            hi=min(len(ids),lo+1024);gg=group[lo:hi];nb=hi-lo
            draws=portable.physical_context_draws(model,[x[lo:hi] for x in contexts],64,7301+10000019*case+1000003*block)
            p0=np.searchsorted(pairids,ids[lo]);p1=np.searchsorted(pairids,ids[hi-1],side='right');pp=pairs.iloc[p0:p1]
            moments=np.zeros((4,nb,4,5))
            for leg,v in enumerate(draws):
                b=[predict_scene_draw_responses(candidate,pp.loc[pg[p0:p1]==g],ids[lo:hi],v.cpu().numpy())[0] for g in [0,1]]
                direct=predict_scene_draw_responses(candidate,pp,ids[lo:hi],v.cpu().numpy())[0]
                np.testing.assert_allclose(b[0]+b[1],direct,rtol=1e-12,atol=1e-12)
                sign=1 if leg==0 else -1
                dshear=torch.tensor([.02*sign,0.],dtype=torch.float64,device='cuda')
                dshift,_=disk_response_transport(v[...,:2],torch.as_tensor(direct,device='cuda')[...,None]*dshear)
                direct_mom=e.draw_moments(e.output_cuts(v),v,dshift-v[...,:2])
                direct_values=torch.cat([direct_mom[0][...,None],direct_mom[1],direct_mom[2]],-1).cpu().numpy()
                for mode,weight in enumerate([prob[leg,lo:hi],usable[0 if leg==0 else 3][lo:hi].astype(float)]):
                    direct_sums[mode,leg]+=np.einsum('n,nkq->kq',weight,direct_values)
                velocities=[.02*sign*(b[0]+b[1]),.02*sign*np.where(gg[:,None]==0,1.,-1.)*(b[0]-b[1])]
                for mixed,velocity in enumerate(velocities):
                    vel=torch.stack([torch.as_tensor(velocity,device='cuda'),torch.zeros_like(v[...,0])],-1)
                    shifted,_=disk_response_transport(v[...,:2],vel)
                    mm=e.draw_moments(e.output_cuts(v),v,shifted-v[...,:2])
                    own=torch.cat([mm[0][...,None],mm[1],mm[2]],-1).cpu().numpy()
                    if mixed:moments[np.where((gg==0)==(leg==0),1,2),np.arange(nb)]=own
                    else:moments[0 if leg==0 else 3]=own
            for state in range(4):
                plus=np.ones(nb,bool) if state==0 else np.zeros(nb,bool) if state==3 else gg==(0 if state==1 else 1)
                probability=np.where(plus,prob[0,lo:hi],prob[1,lo:hi])
                for mode,weight in enumerate([probability,usable[state][lo:hi].astype(float),common[lo:hi].astype(float)]):
                    for g in [0,1]:
                        keep=gg==g;sums[mode,state,:,g]+=np.einsum('n,nkq->kq',weight[keep],moments[state,keep])
            if block%100==0:print('PROGRESS',arm,case,block,round(time.monotonic()-started,1),flush=True)
    cached=DATA/'sbsi_caches/alignment_target9_20261106/axis1' if arm=='baseline' else DATA/'sbsi_caches/alignment_regularized_20261106/constant'/arm/'axis1'
    replay=[]
    with np.load(cached/f'case{case}.npz') as z:
        for k,(name,_,_) in enumerate(CUTS):
            if name not in z['cuts']:continue
            j=list(z['cuts']).index(name)
            for leg,state in enumerate([0,3]):
                np.testing.assert_allclose(sim[0,state,k].sum(0),z['sim'][leg,j].sum(0),rtol=2e-10,atol=2e-7)
                for mode in [0,1]:
                    # Strict independent all-neighbour replay on this device.
                    np.testing.assert_allclose(sums[mode,state,k].sum(0),direct_sums[mode,leg,k],rtol=2e-10,atol=2e-7)
            for mode in [0,1]:
                current=direct_sums[mode,:,k];previous=z['model'][mode,:,j].sum(-2)
                cm=current[:,1:]/current[:,:1];pm=previous[:,1:]/previous[:,:1]
                response_change=((cm[0]-cm[1])-(pm[0]-pm[1]))/.04
                total_change=response_change[:2]+response_change[2:]
                replay.append({'cut':name,'mode':mode,'max_raw_sum_difference':float(np.max(np.abs(current-previous))),
                               'mass_fractional_change':(current[:,0]/previous[:,0]-1).tolist(),
                               'max_per_leg_mean_change':float(np.max(np.abs(cm-pm))),
                               'component_response_change':response_change.tolist(),'total_response_change':total_change.tolist()})
    (dest/f'replay_case{case}.json').write_text(json.dumps(replay,indent=2)+'\n')
    # Bound differences from hardware used for the historical cache in physical
    # response units. Never alter either prediction to make the two agree.
    for replay_check in replay:
        assert max(abs(x) for x in replay_check['mass_fractional_change'])<1e-7,replay_check
        assert replay_check['max_per_leg_mean_change']<1e-7,replay_check
        assert max(abs(x) for x in replay_check['component_response_change']+replay_check['total_response_change'])<5e-6,replay_check
    np.savez(out,sim=sim,model=sums,cuts=e.CUT_NAMES,case=case)
    out.with_suffix('.json').write_text(json.dumps({'case':case,'arm':arm,'checkpoint':pin,'source_sha256':e.sha256(__file__),'latent_source_sha256':check['portable_source_sha256'],'measured':receipts,'coherent_replay_passed':True,'strict_same_device_direct_replay':True,'cache_response_bound':5e-6,'cache_replay':replay,'seconds':time.monotonic()-started},indent=2)+'\n')
    print('DONE',arm,case,time.monotonic()-started,flush=True)

if __name__=='__main__':main()
