"""Frozen joint Cartesian flow: both shear axes and nine cuts, paired measurements."""
import gc,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from scripts import joint_response_axes as axes
from sbsi.flow_guard_response import physical_context_draws as legacy_draws
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
D=ROOT.parent

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e=module('joint9_helpers',H.parent/'alignment_factorized_constant_20261105/helpers.py')
portable=module('joint9_portable',H.parent/'alignment_target9_20261106/portable_draws.py')
BANKS=[list(range(64)),list(range(16))+list(range(32,48)),list(range(16,32))+list(range(48,64))]
def check_pins():
 for p,h in json.loads((H/'source_pins.json').read_text()).items():assert e.sha256(p)==h,p

def normalized(x):return (x[0,:,1:]/x[0,:,:1]-x[1,:,1:]/x[1,:,:1])/.04
class Worker:
 def __init__(self):
  check_pins();self.cfg=json.loads((H/'manifest.json').read_text());cfg=self.cfg
  assert e.sha256(cfg['checkpoint']['path'])==cfg['checkpoint']['sha256']
  bundle=load_measurement_model(cfg['checkpoint']['path'],device='cuda');self.model=bundle.model;self.condition=bundle.condition_preprocessor;self.model.eval()
  self.preset,self.paths,self.conditions=e.preset();assert self.preset['emulator']['sha256']==cfg['emulator_sha256'] and self.preset['classifier']['sha256']==cfg['classifier_sha256']
  self.emulator=DiskResponseMoment(self.paths['emulator'].parent,nthread=4,device='cuda');assert self.emulator.conditioned and self.emulator.response_coordinate=='disk_velocity'
  e.CUTS=cfg['cuts'];e.CUT_NAMES=tuple(c[0] for c in e.CUTS)
  p=D/'selected_seed_calibration_20261108/case_sums.npz';assert e.sha256(p)==cfg['measurement_sums_sha256']
  with np.load(p) as z:
   np.testing.assert_array_equal(z['cases'],cfg['cases']);np.testing.assert_array_equal(z['cuts'],e.CUT_NAMES);self.old_sim=z['old'];self.new_sim=z['new']
  print('MODELS_READY',torch.cuda.get_device_name(),flush=True)
 def evaluate(self,index,pilot=False):
  cfg=self.cfg;case,axis=[cfg['tasks'][index][k] for k in ['case','axis']];out=ROOT/f'axis{axis}/case{case}.npz';out.parent.mkdir(exist_ok=True)
  if out.with_suffix('.json').exists():
   r=json.loads(out.with_suffix('.json').read_text());assert e.sha256(out)==r['data_sha256'] and r['source_sha256']==e.sha256(__file__) and r['checkpoint_sha256']==cfg['checkpoint']['sha256'];print('SKIP',case,axis,flush=True);return
  assert not out.exists();started=time.monotonic();parent=Path(cfg['parent_root']);e.SIMS=Path(cfg['sim_roots'][str(axis)])
  if axis==2:
   for leg in ['0.02','-0.02']:assert (e.SIMS.parent/f'case{case}_{leg}_verified.json').exists()
  parentfile=parent/f'parent/case{case}.npz';flagroot=parent if axis==1 else D/'alignment_axis2_constant_20261105';flags=flagroot/f'c8/case{case}.npz'
  with np.load(parentfile) as z,np.load(flags) as f:
   ids=z['input_index'];context=z['context'].astype(float);np.testing.assert_array_equal(ids,f['input_index']);prob=np.stack([f['probability_plus'],f['probability_minus']])
  assert (np.diff(ids)>0).all() and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
  historical=D/'alignment_matched_constant_20261105/joint_cartesian'/f'case{case}.npz';hist_rec=json.loads(historical.with_suffix('.json').read_text());assert hist_rec['checkpoint_sha256']==cfg['checkpoint']['sha256'] and hist_rec['parent_sha256']==e.sha256(parentfile)
  if axis==1:assert hist_rec['flags_sha256']==e.sha256(flags)
  else:
   fmeta=json.loads(flags.with_suffix('.json').read_text());assert fmeta['parent_sha256']==e.sha256(parentfile) and fmeta['classifier_sha256']==cfg['classifier_sha256'] and fmeta['axis']==2
  truth,truthpath=e.truth(case,.02);np.testing.assert_array_equal(ids,truth.index_input);assert (truth[f'gamma{axis}_input']==.02).all() and (truth[f'gamma{3-axis}_input']==0).all()
  pairs=e.pair_table(e.scene_frame(truth),np.arange(len(truth)),self.conditions);pair_ids=pairs.index_input_p.to_numpy()
  frames=axes.shear_frames(e.full_context(context),.02,axis);contexts=[torch.as_tensor(self.condition.transform_frame(f),dtype=torch.float32,device='cuda') for f in frames]
  ci=cfg['cases'].index(case);sl=slice((axis-1)*2,axis*2);sim=self.new_sim[ci,sl];oldsim=self.old_sim[ci,sl]
  with np.load(historical) as z:
   central=list(z['cuts']).index('mag25_radius_gt0.60');oldcentral=z['model'][0,:,central].sum(-2)[...,:5];np.testing.assert_allclose(z['sim'][:,central].sum(-2)[...,:3],self.old_sim[ci,:2,4],rtol=1e-12,atol=1e-8)
  # Canonical portable streams match the latest model. The old CUDA stream is replayed only in the first pilot.
  replay_legacy=pilot and axis==1
  if replay_legacy:assert torch.cuda.get_device_name()=='NVIDIA A40-8Q',torch.cuda.get_device_name()
  sums=np.zeros((3,2,9,5));legacy=np.zeros((2,1,5));pool_error=None
  print('READY',case,axis,'parents',len(ids),'pairs',len(pairs),'legacy_replay',replay_legacy,flush=True)
  with torch.no_grad():
   for block,lo in enumerate(range(0,len(ids),1024)):
    hi=min(len(ids),lo+1024);p0=np.searchsorted(pair_ids,ids[lo]);p1=np.searchsorted(pair_ids,ids[hi-1],side='right');pp=pairs.iloc[p0:p1]
    values=portable.physical_context_draws(self.model,[c[lo:hi] for c in contexts],64,7301+10000019*case+1000003*block)
    historical_values=legacy_draws(self.model,[c[lo:hi] for c in contexts],64,7301+10000019*case+1000003*block) if replay_legacy else []
    allvalues=values+historical_values
    pooled,_=predict_scene_draw_responses(self.emulator,pp,ids[lo:hi],torch.cat(allvalues,dim=1).cpu().numpy())
    if block==0:
     pool_error=0.
     for j,v in enumerate(allvalues):
      separate,_=predict_scene_draw_responses(self.emulator,pp,ids[lo:hi],v.cpu().numpy());part=pooled[:,64*j:64*(j+1)];pool_error=max(pool_error,float(np.abs(part-separate).max()));np.testing.assert_allclose(part,separate,rtol=2e-14,atol=2e-14)
     print('POOL_REPLAY_PASSED',case,axis,pool_error,flush=True)
    for j,v in enumerate(allvalues):
     leg=j%2;assert torch.isfinite(v).all();assert v[...,:2].square().sum(-1).max()<=1+16*torch.finfo(torch.float64).eps
     shear=torch.zeros(2,dtype=torch.float64,device='cuda');shear[axis-1]=.02 if leg==0 else -.02
     transformed,ld=disk_response_transport(v[...,:2],torch.as_tensor(pooled[:,64*j:64*(j+1)],device='cuda')[...,None]*shear);assert torch.isfinite(transformed).all() and torch.isfinite(ld).all();shift=transformed-v[...,:2];keep=e.output_cuts(v)
     for bank,draws in enumerate(BANKS if j<2 else [BANKS[0]]):
      selected=keep[:,draws] if j<2 else keep[:,draws,4:5]
      mass,flow,emu=e.draw_moments(selected,v[:,draws],shift[:,draws]);moment=torch.cat([mass[...,None],flow,emu],-1).cpu().numpy()
      weighted=np.einsum('n,nkq->kq',prob[leg,lo:hi],moment)
      if j<2:sums[bank,leg]+=weighted
      else:legacy[leg]+=weighted
    if block%100==0:print('PROGRESS',case,axis,block,round(time.monotonic()-started,1),flush=True)
  assert np.isfinite(sums).all();np.testing.assert_allclose(sums[0],(sums[1]+sums[2])/2,rtol=1e-12,atol=2e-7)
  replay=None
  if replay_legacy:
   np.testing.assert_allclose(legacy[:,0,0],oldcentral[:,0],rtol=2e-12,atol=2e-7)
   err=float(np.abs(normalized(legacy)-normalized(oldcentral[:,None])).max());assert err<=1e-6,err
   replay={'legacy_response_max_abs':err,'legacy_count_max_abs':float(np.abs(legacy[:,0,0]-oldcentral[:,0]).max()),'portable_minus_legacy_response_terms':(normalized(sums[0,:,4:5])-normalized(legacy)).tolist()};print('LEGACY_REPLAY_PASSED',replay,flush=True)
  tmp=out.with_name(out.stem+'.tmp.npz');np.savez(tmp,model=sums,sim=sim,old_sim=oldsim,case=case,axis=axis,cuts=e.CUT_NAMES);tmp.replace(out)
  r={'case':case,'axis':axis,'checkpoint_sha256':cfg['checkpoint']['sha256'],'emulator_sha256':cfg['emulator_sha256'],'classifier_sha256':cfg['classifier_sha256'],'data_sha256':e.sha256(out),'source_sha256':e.sha256(__file__),'parent_sha256':e.sha256(parentfile),'flags_sha256':e.sha256(flags),'measurement_sums_sha256':cfg['measurement_sums_sha256'],'pool_max_abs':pool_error,'half_bank_identity_passed':True,'legacy_replay':replay,'parent_count':len(ids),'pair_count':len(pairs),'gpu':torch.cuda.get_device_name(),'seconds':time.monotonic()-started}
  out.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print('DONE',case,axis,r['seconds'],flush=True);gc.collect()
def main():
 assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4);worker=Worker();cfg=worker.cfg;pilot=sys.argv[1]=='pilot'
 if not pilot:
  gate=json.loads((ROOT/'pilot_passed.json').read_text());assert gate['manifest_sha256']==e.sha256(H/'manifest.json')
 indices=cfg['pilot_indices'] if pilot else cfg['worker_tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
 for index in indices:worker.evaluate(index,pilot)
 if pilot:
  records=[json.loads((ROOT/f"axis{cfg['tasks'][i]['axis']}"/f"case{cfg['tasks'][i]['case']}.json").read_text()) for i in indices];assert records[0]['legacy_replay']['legacy_response_max_abs']<=1e-6
  (ROOT/'pilot_passed.json').write_text(json.dumps({'manifest_sha256':e.sha256(H/'manifest.json'),'records':records},indent=2)+'\n')
 print('DONE_PACKED',sys.argv[1],flush=True)
if __name__=='__main__':main()
