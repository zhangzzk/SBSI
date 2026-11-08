"""Four-case common-random-number emulator-input/selection/shape decomposition."""
import os,sys,time,json,importlib.util
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
D=Path('/project/ls-gruen/users/zekang.zhang');F=D/'sbsi_flow_f8_20260924'
sys.path.insert(0,str(F))
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
assert Path(ev.__file__).resolve().parents[1]==F
case=[1000,1040,1080,1120][int(os.environ['SLURM_ARRAY_TASK_ID'])]
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
_,paths,conditions=egc.preset();device=torch.device('cuda')
candidate=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
bundle=load_measurement_model(paths['flow'],device=device);bundle.model.eval()
with np.load(egc.OUT/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context'].astype(np.float64)
t,_=egc.truth(case,.02);assert np.array_equal(t['index_input'],ids)
pairs=egc.pair_table(egc.scene_frame(t),np.arange(len(t)),conditions);pair_ids=pairs['index_input_p'].to_numpy()
frames=dict(zip(('plus','minus'),axes.shear_frames(egc.full_context(context),.02,1)))
contexts=[bundle.context_tensor(frames[d]) for d in ('plus','minus')]
meas=[];usable=[];mkeep=[];receipts=[]
for h in [.02,-.02]:
 v,u,r=ev.measured_leg(egc.SIMS,case,h,ids);meas.append(v);usable.append(u);mkeep.append(u&(v[:,2]>3)&(v[:,3]>100));receipts.append(r)
# variant A original, B measured aux, C measured aux+mask, D all measured; leg, cohort, bin, q mass/flow/emu
sums=np.zeros((4,2,2,4,3));bins=np.digitize(context[:,3],[24,25.5,27.5]);started=time.monotonic()
print('START',case,torch.cuda.get_device_name(0),flush=True)
with torch.no_grad():
 for block,start in enumerate(range(0,len(ids),1024)):
  stop=min(len(ids),start+1024);generated=physical_context_draws(bundle.model,[x[start:stop] for x in contexts],64,7301+10000019*case+1000003*block)
  keeps=[(v[...,2]>3)&(v[...,3]>100) for v in generated]
  flowcommon=keeps[0]&keeps[1]
  for leg,v in enumerate(generated):
   u=usable[leg][start:stop];ix=np.flatnonzero(u)
   if not len(ix):continue
   selected_ids=ids[start:stop][ix];vv=v[ix];mm=meas[leg][start:stop][ix]
   lo=np.searchsorted(pair_ids,ids[start]);hi=np.searchsorted(pair_ids,ids[stop-1],side='right')
   pp=pairs.iloc[lo:hi];pp=pp[np.isin(pp['index_input_p'],selected_ids)]
   b0,_=predict_scene_draw_responses(candidate,pp,selected_ids,vv.cpu().numpy())
   bm,_=predict_scene_draw_responses(candidate,pp,selected_ids,mm[:,None,:])
   h=[.02,-.02][leg];shear=torch.tensor([h,0],dtype=torch.float64,device=device)
   shifts=[]
   for bb in [b0,bm]:
    transformed,_=disk_response_transport(vv[...,:2],torch.as_tensor(bb,dtype=torch.float64,device=device)[...,None]*shear)
    shifts.append(transformed[...,0]-vv[...,0])
   measured=torch.as_tensor(mm,dtype=torch.float64,device=device)
   td,_=disk_response_transport(measured[:,:2],torch.as_tensor(bm[:,0],dtype=torch.float64,device=device)[:,None]*shear)
   sd=td[:,0]-measured[:,0]
   bbins=bins[start:stop][ix]
   for cohort in range(2):
    fk=keeps[leg][ix] if cohort==0 else flowcommon[ix]&torch.as_tensor(usable[1-leg][start:stop][ix],device=device)[:,None]
    mk=mkeep[leg][start:stop][ix] if cohort==0 else (mkeep[0]&mkeep[1])[start:stop][ix]
    mk=torch.as_tensor(mk,dtype=torch.float64,device=device)
    fmass=fk.double().mean(1);fflow=(fk*vv[...,0]).mean(1)
    quantities=[torch.stack([fmass,fflow,(fk*shifts[0]).mean(1)],-1),
                torch.stack([fmass,fflow,(fk*shifts[1]).mean(1)],-1),
                torch.stack([mk,mk*vv[...,0].mean(1),mk*shifts[1].mean(1)],-1),
                torch.stack([mk,mk*measured[:,0],mk*sd],-1)]
    for variant,q in enumerate(quantities):
     q=q.cpu().numpy()
     for b in range(4):sums[variant,leg,cohort,b]+=q[bbins==b].sum(0)
  if block%100==0:print(f'case {case} block {block} seconds {time.monotonic()-started:.1f}',flush=True)
with np.load(D/f'sbsi_caches/galsbi_dev_constant_grid_20261104/sums/case{case}.npz') as z:
 k=list(z['cuts']).index('mag25_radius_gt0.60');old=z['model'][1,:,k];sim=z['sim'][:,k]
 for c,j in [(0,0),(1,5)]:
  expected=old[:,:,[j,j+1,j+3]]
  np.testing.assert_allclose(sums[0,:,c],expected,atol=1e-7,rtol=2e-10)
  np.testing.assert_allclose(sums[3,:,c,:,:2],sim[:,:,[0 if c==0 else 3,1 if c==0 else 4]],atol=1e-7,rtol=1e-10)
np.savez(O/f'replay_case{case}.npz',sums=sums,case=case)
(O/f'replay_case{case}.json').write_text(json.dumps({'case':case,'seconds':time.monotonic()-started,'gpu':torch.cuda.get_device_name(0),'replay':'original true-flag sums and measured counts/shapes match saved outputs','catalogues':receipts},indent=2)+'\n')
print('COMPLETE',case,'seconds',time.monotonic()-started,flush=True)
