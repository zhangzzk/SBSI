"""Selection response omitted from the baseline-only neighbour-shape check."""
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
 baseline,brec=prep.baseline_rows(root,case)
 frame,counts=prep.join_and_select(frame,baseline,30,False)
 cat=root/f'case{case}_0.2/real0/catalogues'
 cross=pd.read_feather(cat/'CrossMatch/tile180.0_-0.5_rot0_matched.feather',columns=['id_input','id_detec'])
 shape=pd.read_feather(cat/'Shapes/shape_catalogue_detect_position_tile180.0_-0.5.feather',columns=['NUMBER','NGMIX_G1','NGMIX_G2','FLUX_RADIUS','MAG_AUTO'])
 assert not cross['id_input'].duplicated().any() and not cross['id_detec'].duplicated().any() and not shape['NUMBER'].duplicated().any()
 joined=cross.merge(shape,left_on='id_detec',right_on='NUMBER',validate='one_to_one').set_index('id_input')
 plus=joined.reindex(frame['input_index'])
 valid=np.isfinite(plus[['NGMIX_G1','NGMIX_G2','FLUX_RADIUS','MAG_AUTO']].to_numpy()).all(1)&(plus['FLUX_RADIUS'].to_numpy()>0)
 dropped=int((~valid).sum());frame=frame.loc[valid].copy();plus=plus.loc[valid]
 e0=frame[['measured_e1_m','measured_e2_m']].to_numpy(float);e1=frame[['measured_e1_p','measured_e2_p']].to_numpy(float)
 np.testing.assert_allclose(e1,plus[['NGMIX_G1','NGMIX_G2']].to_numpy(),rtol=0,atol=1e-7)
 angle=np.deg2rad(frame['shear_angle'].to_numpy(float));u=np.column_stack([np.cos(2*angle),np.sin(2*angle)])
 delta=np.sum((e1-e0)*u,1)/.2
 np.testing.assert_allclose(delta,frame['delta_et1'].to_numpy()/.2,rtol=1e-9,atol=1e-9)
 # Match the old float32 projected labels/model gains exactly for the baseline control.
 label=(frame['delta_et1'].to_numpy()/.2).astype(np.float32).astype(float)
 scaled=rescale(frame.copy(),**manifest['rescale']);x=scaled[prep.FEATURES+prep.OUTPUT_FEATURES].to_numpy(np.float32)
 pred=booster.inplace_predict(np.ascontiguousarray(x)).astype(float)*meta['y_std']+meta['y_mean']
 gain,_=gains.response_gains(e0,frame['shear_angle'].to_numpy(np.float32));pred*=gain.astype(np.float32).astype(float)
 r0=np.exp(frame['baseline_log_flux_radius'].to_numpy());m0=frame['baseline_mag_auto'].to_numpy()
 r1=plus['FLUX_RADIUS'].to_numpy();m1=plus['MAG_AUTO'].to_numpy()
 # Compare stored baseline radii directly, avoiding round-trip differences at r=3.
 r0=baseline.reindex(frame['input_index'])['FLUX_RADIUS'].to_numpy()
 masks0=[np.ones(len(frame),bool),m0<25,r0>3,(m0<25)&(r0>3)]
 masks1=[np.ones(len(frame),bool),m1<25,r1>3,(m1<25)&(r1>3)]
 ids=frame['input_index'].to_numpy();_,unique=np.unique(ids,return_index=True)
 bins=np.digitize(frame['Re_input_p'].to_numpy()*np.sqrt(frame['axis_ratio_input_p'].to_numpy()),[.15,.25,.4,.7])
 sums=np.zeros((4,5,len(Q)))
 for k,(s0,s1) in enumerate(zip(masks0,masks1)):
  ds=s1.astype(float)-s0;sel=ds*np.sum(e1*u,1)/.2;dm=ds[:,None]*u/.2
  for b in range(5):
   rows=bins==b;ur=unique[bins[unique]==b];b0=s0[ur];b1=s1[ur]
   sums[k,b]=[b0.sum(),*(b0[:,None]*e0[ur]).sum(0),(s0[rows]*label[rows]).sum(),(s0[rows]*pred[rows]).sum(),sel[rows].sum(),*dm[rows].sum(0),(s0&rows).sum(),b1.sum()]
 t=sums[-1].sum(0)
 np.testing.assert_equal(t[8],old[case,oldq.index('rows')])
 np.testing.assert_allclose(t[3:5],[old[case,oldq.index('label')],old[case,oldq.index('galsbi_dev_pred')]],rtol=1e-7,atol=1e-6)
 all_sums.append(sums);receipts.append({'case':case,'label_drops':counts,'baseline':brec,'missing_or_invalid_plus_auxiliary_pairs':dropped,'cross_without_shape':len(cross)-len(joined),'seconds':time.monotonic()-start})
 np.savez(O/'neighbour_sums.npz',sums=np.array(all_sums),cuts=CUTS,quantities=Q,cases=np.arange(case+1))
 (O/'neighbour_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 print('case',case,'baseline replay passed','seconds',round(time.monotonic()-start,1),flush=True)
a=np.array(all_sums);w=np.vstack([np.ones(40),np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)])
out={'cuts':{},'cases':40,'definition':'Half-trace projected forward neighbour-shear response per represented baseline-selected primary, h=0.2; common usable labels','bootstrap':'10000 paired whole-case resamples, seed20260925; baseline mass/mean recomputed','limitations':'Half of scene neighbours sheared in independent random directions. Common usability and bright-neighbour exclusions retained. Not an h=0.02 coherent-shear error budget.'}
for k,cut in enumerate(CUTS):
 z=np.einsum('bc,ckq->bkq',w,a[:,k]);t=z.sum(1);mu=t[:,1:3]/t[:,0,None]
 sel=z[:,:,5]-np.sum(mu[:,None,:]*z[:,:,6:8],-1)
 shape=t[:,3]/t[:,0];model=t[:,4]/t[:,0];selection=sel.sum(1)/t[:,0];full=shape+selection
 def ent(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
 out['cuts'][cut]={name:ent(v) for name,v in [('simulation_baseline_shape',shape),('model_shape',model),('selection_response',selection),('simulation_selected_total',full),('baseline_error',shape-model),('selected_error',full-model)]}
 out['cuts'][cut]['selection_contributions_by_true_size']=[ent(sel[:,b]/t[:,0]) for b in range(5)]
 out['cuts'][cut]['primaries']=int(t[0,0]);out['cuts'][cut]['pairs']=int(t[0,8])
(O/'neighbour_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
