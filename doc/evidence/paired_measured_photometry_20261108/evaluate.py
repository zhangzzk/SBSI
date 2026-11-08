"""Matched measured-input expectations using validated signed membership updates."""
import importlib.util,json,os,time,gc,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sbsi.output_conditioned_response import predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
OLD=ROOT.parent/'current_measured_photometry_20261107'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prior=module('paired_photo_old',H.parent/OLD.name/'evaluate.py')
e,cp,nm,portable=prior.e,prior.cp,prior.nm,prior.portable

def predict(case,axis,ids,raw,truth,photo,vo,vn,cache,full=False):
 models,conditions,preset,emulator,observing,fixed=cache
 union=((vo|vn) if full else (vo!=vn)).any(1)
 output=np.zeros((3,3,2,9,5)) # old/new/difference, draw bank, leg, cut, moment
 if not union.any():return output,{'changed_rows':0,'feature_max_abs':0.,'checks':{},'full':full}
 pairs=e.pair_table(e.scene_frame(truth),np.flatnonzero(union),observing)
 score=nm.c.score_pairs(emulator,pairs,ids)
 selected=ids[np.flatnonzero(union)[np.linspace(0,int(union.sum())-1,min(32,int(union.sum())),dtype=int)]]
 subset=pairs.loc[pairs.index_input_p.isin(selected)]
 direct=emulator.predict_pairs(subset,np.full(len(subset),3.5),np.full(len(subset),10**(.4*(30-24.5))))
 grouped=pd.Series(direct,index=subset.index_input_p.to_numpy()).groupby(level=0).sum().reindex(selected,fill_value=0.).to_numpy()
 ref=score[pd.Index(ids).get_indexer(selected)];np.testing.assert_allclose(grouped,ref,rtol=2e-10,atol=2e-7)
 frames=prior.axes.shear_frames(e.full_context(raw),.02,axis)
 for f in frames:f[nm.c.FEATURE]=score
 contexts=[torch.as_tensor(conditions['control'].transform_frame(f),dtype=torch.float32,device='cuda') for f in frames]
 coeff=np.zeros((len(ids),2));eligible=(vo|vn) if full else (vo!=vn)
 for leg in range(2):
  k=eligible[:,leg]
  if not k.any():continue
  bid=ids[k];pp=pairs.loc[pairs.index_input_p.isin(bid)]
  values=np.concatenate([np.zeros((k.sum(),2)),photo[k,leg]],-1)[:,None,:]
  b,_=predict_scene_draw_responses(emulator,pp,bid,values);coeff[k,leg]=b[:,0]
 del pairs
 checks={};start=time.monotonic()
 for block,lo in enumerate(range(0,len(ids),1024)):
  hi=min(len(ids),lo+1024);nb=hi-lo
  if not eligible[lo:hi].any():continue
  latent=portable.latent_bank(nb,64,4,7301+10000019*case+1000003*block)
  for leg in range(2):
   k=eligible[lo:hi,leg]
   if not k.any():continue
   z=torch.from_numpy(latent[k]).to('cuda').reshape(-1,4)
   phot=torch.as_tensor(photo[lo:hi,leg][k],dtype=torch.float64,device='cuda')
   expanded=phot[:,None,:].expand(-1,64,-1).reshape(-1,2)
   ctx=contexts[leg][lo:hi][k,None,:].expand(-1,64,-1).reshape(len(z),-1)
   with torch.no_grad():
    v=cp.conditional_physical(models['control'],z,ctx,expanded).reshape(-1,64,4)
    if leg not in checks:
     other=cp.conditional_physical(models['weighted'],z,ctx,expanded).reshape(-1,64,4);torch.testing.assert_close(v,other,rtol=0,atol=0)
     joint,_=models['control'].forward_physical(z[:256],ctx[:256]);valid=(joint[:,2:]>0).all(-1);assert valid.any()
     recovered=cp.conditional_physical(models['control'],z[:256][valid],ctx[:256][valid],joint[valid,2:]);torch.testing.assert_close(recovered,joint[valid],rtol=2e-5,atol=2e-6)
     changed=z[:256][valid].clone();changed[:,2:]+=3;torch.testing.assert_close(cp.conditional_physical(models['control'],changed,ctx[:256][valid],joint[valid,2:]),recovered,rtol=0,atol=0)
     checks[leg]={'pair_conditional_exact':True,'reconstruction_max_abs':float((recovered-joint[valid]).abs().max()),'unused_photo_latents_exact':True}
    torch.testing.assert_close(v[...,2:],phot[:,None,:].expand(-1,64,-1),rtol=0,atol=0)
    assert torch.isfinite(v).all() and v[...,:2].square().sum(-1).max()<=1+16*torch.finfo(torch.float64).eps
    velocity=torch.zeros((len(v),1,2),dtype=torch.float64,device='cuda');velocity[:,:,axis-1]=torch.as_tensor(coeff[lo:hi,leg][k,None],device='cuda')*(.02 if leg==0 else -.02)
    transported,ld=disk_response_transport(v[...,:2],velocity);assert torch.isfinite(transported).all() and torch.isfinite(ld).all();shift=transported-v[...,:2]
    cut=e.output_cuts(v[:,0]).cpu().numpy();np.testing.assert_array_equal(cut,e.output_cuts(np.concatenate([np.zeros((k.sum(),2)),photo[lo:hi,leg][k]],-1)))
    wo=cut*vo[lo:hi,leg][k,None];wn=cut*vn[lo:hi,leg][k,None]
    weights=[wo.astype(float),wn.astype(float),wn.astype(float)-wo.astype(float)]
    for bank,draws in enumerate(prior.BANKS):
     moment=torch.cat([torch.ones((len(v),1),dtype=torch.float64,device='cuda'),v[:,draws,:2].mean(1),shift[:,draws].mean(1)],-1).cpu().numpy()
     for mode,w in enumerate(weights):output[mode,bank,leg]+=np.einsum('nk,nq->kq',w,moment)
  if block%100==0:print('PROGRESS',case,axis,'full',full,block,round(time.monotonic()-start,1),flush=True)
 np.testing.assert_allclose(output[2],output[1]-output[0],rtol=1e-10,atol=2e-7)
 np.testing.assert_allclose(output[:,0],.5*(output[:,1]+output[:,2]),rtol=1e-12,atol=2e-7)
 return output,{'feature_max_abs':float(np.abs(grouped-ref).max()),'checks':checks,'changed_rows':int(union.sum()),'full':full,'seconds':time.monotonic()-start}

def response(m):
 return (m[:,0,:,1:]/m[:,0,:,:1]-m[:,1,:,1:]/m[:,1,:,:1])/.04

def evaluate(index,cache,cfg,prepared,pilot=False):
 task=cfg['tasks'][index];case,axis=task['case'],task['axis'];dest=ROOT/f'axis{axis}';dest.mkdir(exist_ok=True);out=dest/f'case{case}.npz'
 if out.with_suffix('.json').exists():
  r=json.loads(out.with_suffix('.json').read_text());assert r['sha256']==e.sha256(out) and r['source_sha256']==e.sha256(__file__);print('SKIP',index,flush=True);return
 assert not out.exists();rec=next(r for r in prepared if r['case']==case);p=ROOT/f'case{case}.npz';assert e.sha256(p)==rec['sha256']
 with np.load(p) as z:data={k:z[k] for k in z.files}
 parent=Path(cfg['parent_root'])/'parent'/f'case{case}.npz'
 with np.load(parent) as z:ids=z['input_index'];raw=z['context'].astype(float)
 loc=pd.Index(ids).get_indexer(data['ids']);assert (loc>=0).all() and len(np.unique(loc))==len(loc)
 e.CUTS=cfg['cuts'];e.CUT_NAMES=tuple(c[0] for c in e.CUTS);e.SIMS=Path(cfg['sim_roots'][str(axis)])
 truth,_=e.truth(case,.02);np.testing.assert_array_equal(ids,truth.index_input);assert np.all(truth[f'gamma{axis}_input']==.02)
 photo=np.ones((len(ids),2,2));vo=np.zeros((len(ids),2),bool);vn=np.zeros_like(vo);sl=slice((axis-1)*2,axis*2)
 photo[loc]=data['photo'][:,sl];vo[loc]=data['old_valid'][:,sl];vn[loc]=data['new_valid'][:,sl]
 p=OLD/f'axis{axis}/case{case}.npz';oldreceipt=json.loads(p.with_suffix('.json').read_text());assert e.sha256(p)==next(x['sha256'] for x in rec['old_oracle_receipts'] if x['axis']==axis)
 with np.load(p) as z:old=z['model'][list(z['arms']).index('control')].sum(-2)[:,:,:9];np.testing.assert_allclose(z['sim'].sum(-2)[:,:9],data['old_sums'][sl],rtol=1e-12,atol=1e-8)
 signed,record=predict(case,axis,ids,raw,truth,photo,vo,vn,cache,False);new=old+signed[2]
 np.testing.assert_array_equal(new[...,0],np.broadcast_to(data['new_sums'][sl,:,0],new[...,0].shape))
 replay=None
 if pilot:
  full,full_record=predict(case,axis,ids,raw,truth,photo,vo,vn,cache,True)
  np.testing.assert_array_equal(full[0,...,0],old[...,0]);np.testing.assert_array_equal(full[1,...,0],new[...,0])
  olderr=float(np.abs(response(full[0])-response(old)).max());newerr=float(np.abs(response(full[1])-response(new)).max());assert olderr<=1e-6 and newerr<=1e-6,(olderr,newerr)
  replay={'old_full_response_max_abs':olderr,'new_full_vs_update_max_abs':newerr,'full_record':full_record};print('FULL_REPLAY_PASSED',case,axis,replay,flush=True)
 np.savez(out,model=new,old_model=old,delta=signed[2],sim=data['new_sums'][sl],old_sim=data['old_sums'][sl],cuts=e.CUT_NAMES,case=case,axis=axis)
 r={'case':case,'axis':axis,'sha256':e.sha256(out),'source_sha256':e.sha256(__file__),'prepared_sha256':rec['sha256'],'old_oracle_sha256':e.sha256(p),'checkpoints':cfg['checkpoints'],'emulator_sha256':cfg['emulator_sha256'],'selected_masses_exact':True,'matched_new_simulation_sums':True,'half_bank_identity_passed':True,'record':record,'full_replay':replay,'gpu':torch.cuda.get_device_name()}
 out.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print('DONE',case,axis,flush=True)

def main():
 assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
 for p,d in json.loads((H/'source_pins.json').read_text()).items():assert e.sha256(p)==d,p
 cfg=json.loads((H/'manifest.json').read_text());prepared=json.loads((ROOT/'prepared.json').read_text());cache=prior.load_models(cfg)
 pilot=sys.argv[1]=='pilot'
 if not pilot:assert (ROOT/'pilot_passed.json').exists()
 indices=cfg['pilot_indices'] if pilot else cfg['worker_tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
 for i in indices:evaluate(i,cache,cfg,prepared,pilot);gc.collect()
 if pilot:
  records=[json.loads((ROOT/f"axis{cfg['tasks'][i]['axis']}"/f"case{cfg['tasks'][i]['case']}.json").read_text()) for i in indices]
  assert all(r['full_replay'] is not None for r in records);(ROOT/'pilot_passed.json').write_text(json.dumps(records,indent=2)+'\n')
 print('DONE_PACKED',sys.argv[1],flush=True)
if __name__=='__main__':main()
