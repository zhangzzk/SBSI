"""Matched-object neighbour-shear amplitude comparison."""
import os,sys,json,importlib.util,time
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang');B=Path('/home/z/Zekang.Zhang/blendemu_galsbi_resim')
sys.path.insert(0,str(B))
import pandas as pd
import xgboost as xgb
from blendemu import response
from blendemu.data_utils import rescale
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prep=load('local_prepare',B/'scripts/prepare_output_conditioned_response.py')
gains=load('local_gains',B/'scripts/prepare_disk_response_moment.py')
root=D/'sbsi/galsbi_resim/training/sims'
model=D/'blendemu_runs/disk_response_moment_galsbi_20261103/model'
meta=json.loads((model/'metadata.json').read_text());booster=xgb.Booster(model_file=model/'model.json')
booster.set_param({'nthread':int(os.environ.get('SLURM_CPUS_PER_TASK',4))})
manifest=json.loads((D/'blendemu_runs/disk_response_moment_galsbi_cut_20261103/pairs/manifest.json').read_text())
with np.load(D/'sbsi_caches/galsbi_emulator_check_20261104/sums.npz') as z:
 old=z['sums'][:,:,:4,:,:].sum((1,2,3));oldq=list(z['quantities'])
KD=response.KDTree
class BoundedTree:
 def __init__(self,*a,**k):self.tree=KD(*a,**k)
 def query(self,*a,**k):k['workers']=int(os.environ.get('SLURM_CPUS_PER_TASK',4));return self.tree.query(*a,**k)
response.KDTree=BoundedTree
CUTS=['uncut','mag25','radius3','radius3_mag25']
Q=['n0','e01','e02','shape','model','selection_raw','mass1','mass2','pairs0','n1']
all_sums=[];receipts=[]
for case in range(40):
 start=time.monotonic()
 frame=response.retrieve_response(case,10,0,20,data_path=str(root),shear_cases=['0.0','0.2'])
 baseline,brec=prep.baseline_rows(root,case);frame,counts=prep.join_and_select(frame,baseline,30,False)
 joined=[];catalogue_receipts=[];truth=[]
 for tag in ['0.2','0.05']:
  cat=root/f'case{case}_{tag}/real0/catalogues'
  cross=pd.read_feather(cat/'CrossMatch/tile180.0_-0.5_rot0_matched.feather',columns=['id_input','id_detec'])
  shape=pd.read_feather(cat/'Shapes/shape_catalogue_detect_position_tile180.0_-0.5.feather',columns=['NUMBER','NGMIX_G1','NGMIX_G2','FLUX_RADIUS','MAG_AUTO','X_WORLD','Y_WORLD','FLUX_AUTO'])
  reject=response.utils.remove_detection_w_bright_neighbour(shape['X_WORLD'].array,shape['Y_WORLD'].array,shape['FLUX_AUTO'].array,ratio_max=5,r_min=0,r_max=3/3600)
  shape=shape.drop(reject)
  assert not cross['id_input'].duplicated().any() and not cross['id_detec'].duplicated().any() and not shape['NUMBER'].duplicated().any()
  j=cross.merge(shape,left_on='id_detec',right_on='NUMBER',validate='one_to_one').set_index('id_input')
  joined.append(j.reindex(frame['input_index']))
  catalogue_receipts.append({'leg':tag,'cross_without_retained_shape':len(cross)-len(j),'bright_excluded_detections':len(reject)})
  truth.append(pd.read_feather(cat/'input/gals_info_tile180.0_-0.5.feather',columns=['index_input','gamma1_input','gamma2_input']))
 np.testing.assert_array_equal(truth[0]['index_input'],truth[1]['index_input'])
 np.testing.assert_allclose(truth[0][['gamma1_input','gamma2_input']],4*truth[1][['gamma1_input','gamma2_input']],atol=1e-14,rtol=0)
 valid=np.ones(len(frame),bool)
 for j in joined:
  v=j[['NGMIX_G1','NGMIX_G2','FLUX_RADIUS','MAG_AUTO']].to_numpy()
  valid&=np.isfinite(v).all(1)&((v[:,:2]**2).sum(1)<1)&(v[:,2]>0)
 dropped=int((~valid).sum());frame=frame.loc[valid].copy();joined=[j.loc[valid] for j in joined]
 e0=frame[['measured_e1_m','measured_e2_m']].to_numpy(float)
 np.testing.assert_allclose(frame[['measured_e1_p','measured_e2_p']],joined[0][['NGMIX_G1','NGMIX_G2']],rtol=0,atol=1e-7)
 angle=np.deg2rad(frame['shear_angle'].to_numpy(float));u=np.column_stack([np.cos(2*angle),np.sin(2*angle)])
 scaled=rescale(frame.copy(),**manifest['rescale']);x=scaled[prep.FEATURES+prep.OUTPUT_FEATURES].to_numpy(np.float32)
 pred=booster.inplace_predict(np.ascontiguousarray(x)).astype(float)*meta['y_std']+meta['y_mean']
 gain,_=gains.response_gains(e0,frame['shear_angle'].to_numpy(np.float32));pred*=gain.astype(np.float32).astype(float)
 r0=baseline.reindex(frame['input_index'])['FLUX_RADIUS'].to_numpy();m0=frame['baseline_mag_auto'].to_numpy()
 masks0=[np.ones(len(frame),bool),m0<25,r0>3,(m0<25)&(r0>3)]
 _,unique=np.unique(frame['input_index'].to_numpy(),return_index=True)
 sums=np.zeros((2,4,len(Q)))
 for leg,(j,h) in enumerate(zip(joined,[.2,.05])):
  e1=j[['NGMIX_G1','NGMIX_G2']].to_numpy(float);r1=j['FLUX_RADIUS'].to_numpy();m1=j['MAG_AUTO'].to_numpy()
  label=np.sum((e1-e0)*u,1)/h
  if leg==0:np.testing.assert_allclose(label,frame['delta_et1'].to_numpy()/.2,atol=1e-9,rtol=1e-9)
  masks1=[np.ones(len(frame),bool),m1<25,r1>3,(m1<25)&(r1>3)]
  for k,(s0,s1) in enumerate(zip(masks0,masks1)):
   ds=s1.astype(float)-s0;sel=ds*np.sum(e1*u,1)/h;dm=ds[:,None]*u/h
   b0=s0[unique];b1=s1[unique]
   sums[leg,k]=[b0.sum(),*(b0[:,None]*e0[unique]).sum(0),(s0*label).sum(),(s0*pred).sum(),sel.sum(),*dm.sum(0),s0.sum(),b1.sum()]
 np.testing.assert_array_equal(sums[0,:,0:3],sums[1,:,0:3]);np.testing.assert_array_equal(sums[0,:,4],sums[1,:,4])
 all_sums.append(sums);receipts.append({'case':case,'drops':counts,'baseline':brec,'not_common_usable_auxiliary_pairs':dropped,'catalogues':catalogue_receipts,'seconds':time.monotonic()-start})
 np.savez(O/'neighbour_step_sums.npz',sums=np.array(all_sums),cuts=CUTS,quantities=Q,cases=np.arange(case+1),shears=[.2,.05])
 (O/'neighbour_step_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 print('case',case,'matched pair step check passed','seconds',round(time.monotonic()-start,1),flush=True)
a=np.array(all_sums);w=np.vstack([np.ones(40),np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)])
out={'cases':40,'cuts':{},'bootstrap':'10000 paired case resamples, seed20260925','definition':'Same primaries and pairs usable in g0,h0.05,h0.2; forward projected response per baseline-selected primary'}
def ent(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
for k,cut in enumerate(CUTS):
 t=np.einsum('bc,clq->blq',w,a[:,:,k]);mu=t[:,:,1:3]/t[:,:,0,None]
 shape=t[:,:,3]/t[:,:,0];model=t[:,:,4]/t[:,:,0];selection=(t[:,:,5]-(mu*t[:,:,6:8]).sum(-1))/t[:,:,0];full=shape+selection
 out['cuts'][cut]={}
 for l,h in enumerate([.2,.05]):
  out['cuts'][cut][str(h)]={name:ent(v[:,l]) for name,v in [('simulation_baseline_shape',shape),('model_shape',model),('selection_response',selection),('simulation_selected_total',full),('baseline_error',shape-model),('selected_error',full-model)]}
 out['cuts'][cut]['change_005_minus_02']={name:ent(v[:,1]-v[:,0]) for name,v in [('baseline_shape',shape),('selection_response',selection),('selected_total',full)]}
(O/'neighbour_step_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
