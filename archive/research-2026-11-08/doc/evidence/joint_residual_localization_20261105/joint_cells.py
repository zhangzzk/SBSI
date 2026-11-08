"""Pairing-free truth-cell localization of the unchanged joint response."""
import os,sys,time,json,importlib.util
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
D=Path('/project/ls-gruen/users/zekang.zhang');F=D/'sbsi_flow_f8_20260924';sys.path.insert(0,str(F))
import torch
import scripts.evaluate_smooth_classifier_joint as ev
from scripts import joint_response_axes as axes
from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport
O=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('egc',Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/scripts/evaluate_galsbi_constant.py'))
egc=importlib.util.module_from_spec(spec);spec.loader.exec_module(egc)
case=[1000,1040,1080,1120][int(os.environ['SLURM_ARRAY_TASK_ID'])]
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
_,paths,conditions=egc.preset();candidate=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
bundle=load_measurement_model(paths['flow'],device='cuda');bundle.model.eval()
with np.load(egc.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context'].astype(float)
t,_=egc.truth(case,.02);assert np.array_equal(t['index_input'],ids)
pairs=egc.pair_table(egc.scene_frame(t),np.arange(len(t)),conditions);pair_ids=pairs['index_input_p'].to_numpy()
frames=axes.shear_frames(egc.full_context(context),.02,1)
contexts=[bundle.context_tensor(f) for f in frames]
cell=(np.digitize(context[:,4],[.15,.25,.4,.7])*4+np.digitize(context[:,3],[24,25.5,27.5]))*5+np.digitize(context[:,6],[.1,.3,1,3])
def cuts(v):
 r=v[...,2]>3;m=v[...,3]>100
 return [r==r,m,r,r&m]
sim=np.zeros((2,4,100,2));model=np.zeros((2,4,100,3));usable=[];receipts=[]
for leg,h in enumerate([.02,-.02]):
 v,u,record=ev.measured_leg(egc.SIMS,case,h,ids);usable.append(u);receipts.append(record)
 for k,cut in enumerate(cuts(v)):
  keep=u&cut
  for q,values in enumerate([np.ones(keep.sum()),v[keep,0]]):sim[leg,k,:,q]=np.bincount(cell[keep],weights=values,minlength=100)
start=time.monotonic();print('START',case,torch.cuda.get_device_name(0),flush=True)
with torch.no_grad():
 for block,lo in enumerate(range(0,len(ids),1024)):
  hi=min(len(ids),lo+1024);generated=physical_context_draws(bundle.model,[c[lo:hi] for c in contexts],64,7301+10000019*case+1000003*block)
  for leg,v in enumerate(generated):
   ix=np.flatnonzero(usable[leg][lo:hi])
   if not len(ix):continue
   selected=ids[lo:hi][ix];v=v[ix]
   p0=np.searchsorted(pair_ids,ids[lo]);p1=np.searchsorted(pair_ids,ids[hi-1],side='right')
   pp=pairs.iloc[p0:p1];pp=pp[np.isin(pp['index_input_p'],selected)]
   b,_=predict_scene_draw_responses(candidate,pp,selected,v.cpu().numpy())
   shear=torch.tensor([[.02,0],[-.02,0]][leg],dtype=torch.float64,device='cuda')
   transformed,_=disk_response_transport(v[...,:2],torch.as_tensor(b,dtype=torch.float64,device='cuda')[...,None]*shear)
   shift=transformed[...,0]-v[...,0];cells=cell[lo:hi][ix]
   for k,keep in enumerate(cuts(v)):
    vals=torch.stack([keep.double().mean(1),(keep*v[...,0]).mean(1),(keep*shift).mean(1)],-1).cpu().numpy()
    for q in range(3):model[leg,k,:,q]+=np.bincount(cells,weights=vals[:,q],minlength=100)
  if block%200==0:print('case',case,'block',block,'seconds',time.monotonic()-start,flush=True)
with np.load(D/f'sbsi_caches/galsbi_dev_constant_grid_20261104/sums/case{case}.npz') as z:
 names=list(z['cuts'])
 for k,name in enumerate(['uncut','mag25','radius_gt0.60','mag25_radius_gt0.60']):
  # Store full names in the receipt to make any convention mismatch explicit.
  if name not in names:
   print('available cuts',names,flush=True);raise ValueError(name)
  j=names.index(name)
  np.testing.assert_allclose(model[:,k].sum(1),z['model'][1,:,j].sum(1)[:,[0,1,3]],atol=1e-7,rtol=2e-10)
  np.testing.assert_allclose(sim[:,k].sum(1),z['sim'][:,j].sum(1)[:,:2],atol=1e-7,rtol=2e-10)
np.savez(O/f'joint_case{case}.npz',sim=sim,model=model,case=case)
(O/f'joint_case{case}.json').write_text(json.dumps({'case':case,'seconds':time.monotonic()-start,'catalogues':receipts,'replay':'original true-U/model-own-selection masses and contributions reproduce for all four cuts'},indent=2)+'\n')
print('COMPLETE',case,time.monotonic()-start,flush=True)
