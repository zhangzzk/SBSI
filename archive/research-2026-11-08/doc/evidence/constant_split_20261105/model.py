"""Cache four-state predictions with exact original ++/-- replay."""
import os,sys,time,json,importlib.util
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
D=Path('/project/ls-gruen/users/zekang.zhang');F=D/'sbsi_flow_f8_20260924';sys.path.insert(0,str(F))
import pandas as pd
import torch
import scripts.evaluate_smooth_classifier_joint as ev
from scripts import joint_response_axes as axes
from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport
O=Path(__file__).resolve().parent;STUDY=D/'sbsi/galsbi_resim/constant_split_20261105'
spec=importlib.util.spec_from_file_location('egc',Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/scripts/evaluate_galsbi_constant.py'))
egc=importlib.util.module_from_spec(spec);spec.loader.exec_module(egc)
case=[1000,1040,1080,1120][int(os.environ['SLURM_ARRAY_TASK_ID'])]
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
_,paths,conditions=egc.preset();candidate=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
bundle=load_measurement_model(paths['flow'],device='cuda');bundle.model.eval()
with np.load(egc.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context'].astype(float)
with np.load(STUDY/f'groups_case{case}.npz') as z:
 lookup=pd.Series(z['group'],index=z['input_index']);group=lookup.loc[ids].to_numpy()
t,_=egc.truth(case,.02);assert np.array_equal(t['index_input'],ids)
pairs=egc.pair_table(egc.scene_frame(t),np.arange(len(t)),conditions);pair_ids=pairs['index_input_p'].to_numpy()
pg=lookup.loc[pairs['index_input_s']].to_numpy()
contexts=[bundle.context_tensor(f) for f in axes.shear_frames(egc.full_context(context),.02,1)]
def cuts(v):
 r=v[...,2]>3;m=v[...,3]>100
 return torch.stack([r==r,m,r,r&m],-1)
mom=np.full((4,len(ids),4,5),np.nan);start=time.monotonic()
print('START',case,torch.cuda.get_device_name(0),flush=True)
with torch.no_grad():
 for block,lo in enumerate(range(0,len(ids),1024)):
  hi=min(len(ids),lo+1024);draws=physical_context_draws(bundle.model,[x[lo:hi] for x in contexts],64,7301+10000019*case+1000003*block)
  p0=np.searchsorted(pair_ids,ids[lo]);p1=np.searchsorted(pair_ids,ids[hi-1],side='right');pp=pairs.iloc[p0:p1]
  rows=np.arange(lo,hi);gg=group[lo:hi]
  for leg,v in enumerate(draws):
   assert bool(torch.isfinite(v).all())
   b=[]
   for g in [0,1]:
    bg,_=predict_scene_draw_responses(candidate,pp.loc[pg[p0:p1]==g],ids[lo:hi],v.cpu().numpy());b.append(bg)
   # Independent direct all-neighbour replay on the first block checks split sums.
   if block==0:
    direct,_=predict_scene_draw_responses(candidate,pp,ids[lo:hi],v.cpu().numpy())
    np.testing.assert_allclose(b[0]+b[1],direct,atol=1e-12,rtol=1e-12)
   sign=1 if leg==0 else -1
   velocity_coherent=.02*sign*(b[0]+b[1])
   velocity_mixed=.02*sign*np.where(gg[:,None]==0,1.,-1.)*(b[0]-b[1])
   keep=cuts(v).double();mass=keep.mean(1);flow=(keep[...,None]*v[:,:,None,:2]).mean(1)
   for mixed,vel in enumerate([velocity_coherent,velocity_mixed]):
    velocity=torch.stack([torch.as_tensor(vel,device='cuda'),torch.zeros_like(v[...,0])],-1)
    transformed,_=disk_response_transport(v[...,:2],velocity);shift=transformed-v[...,:2]
    em=(keep[...,None]*shift[:,:,None,:]).mean(1)
    values=torch.cat([mass[...,None],flow,em],-1).cpu().numpy()
    if not mixed:mom[0 if leg==0 else 3,rows]=values
    else:
     states=np.where((gg==0)==(leg==0),1,2)
     mom[states,rows]=values
  if block%100==0:print('case',case,'block',block,'seconds',time.monotonic()-start,flush=True)
assert np.isfinite(mom).all()
with np.load(egc.OUT/f'c8/case{case}.npz') as z:
 np.testing.assert_array_equal(z['input_index'],ids);prob=np.stack([z['probability_plus'],z['probability_minus']])
oldroot=D/'sbsi_caches/galsbi_dev_constant_grid_20261104'
with np.load(oldroot/f'sums/case{case}.npz') as z:
 for leg,state in enumerate([0,3]):
  _,u,_=ev.measured_leg(egc.SIMS,case,[.02,-.02][leg],ids)
  for k,name in enumerate(['uncut','mag25','radius_gt0.60','mag25_radius_gt0.60']):
   j=list(z['cuts']).index(name)
   for flag,weight in enumerate([prob[leg],u.astype(float)]):
    np.testing.assert_allclose(weight@mom[state,:,k],z['model'][flag,leg,j,:,:5].sum(0),atol=2e-7,rtol=2e-10)
(STUDY/'model').mkdir(exist_ok=True)
np.savez(STUDY/f'model/case{case}.npz',moments=mom,input_index=ids,group=group,probability=prob,context=context)
(O/f'model_case{case}.json').write_text(json.dumps({'case':case,'seconds':time.monotonic()-start,'replay':'all original predicted-U and true-U masses/numerators replay for four cuts; split neighbour sums replay on first block','rows':len(ids)},indent=2)+'\n')
print('COMPLETE',case,time.monotonic()-start,flush=True)
