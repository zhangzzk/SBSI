"""Frozen-model geometry and omitted-population check; CPU Slurm only."""
import os,sys,json,importlib.util,time
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang')
B=Path('/home/z/Zekang.Zhang/blendemu_galsbi_resim')
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
meta=json.loads((D/'blendemu_runs/disk_response_moment_galsbi_20261103/model/metadata.json').read_text())
booster=xgb.Booster(model_file=D/'blendemu_runs/disk_response_moment_galsbi_20261103/model/model.json')
booster.set_param({'nthread':int(os.environ.get('SLURM_CPUS_PER_TASK',4))})
manifest=json.loads((D/'blendemu_runs/disk_response_moment_galsbi_cut_20261103/pairs/manifest.json').read_text())
with np.load(D/'sbsi_caches/galsbi_emulator_check_20261104/sums.npz') as z:
 old=z['sums'][:,:,:4,:,:].sum((1,2)); oldq=list(z['quantities'])
original_reject=response.utils.remove_detection_w_bright_neighbour
KD=response.KDTree
class BoundedTree:
 def __init__(self,*a,**k):self.tree=KD(*a,**k)
 def query(self,*a,**k):k['workers']=int(os.environ.get('SLURM_CPUS_PER_TASK',4));return self.tree.query(*a,**k)
response.KDTree=BoundedTree
Q=['pairs','primaries','label','pred_measured_distance','pred_true_distance','true_distance_sum','measured_distance_sum']
all_sums=[];receipts=[]
for case in range(40):
 start=time.monotonic(); rejected=[]
 def capture(*a,**k):
  rejected.append(np.asarray(original_reject(*a,**k)));return np.array([],dtype=int)
 response.utils.remove_detection_w_bright_neighbour=capture
 frame=response.retrieve_response(case,10,0,20,data_path=str(root),shear_cases=['0.0','0.2'])
 assert len(rejected)==2
 bad=[]
 for leg,rej in zip(['0.0','0.2'],rejected):
  cross=pd.read_feather(root/f'case{case}_{leg}/real0/catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather',columns=['id_input','id_detec'])
  bad.append(cross.loc[np.isin(cross['id_detec'].to_numpy()-1,rej),'id_input'].to_numpy())
 bad=np.unique(np.concatenate(bad))
 baseline,brec=prep.baseline_rows(root,case)
 frame,counts=prep.join_and_select(frame,baseline,30,True)
 frame=frame[frame['baseline_mag_auto']<25].copy()
 originally_excluded=np.isin(frame['input_index'],bad)
 scaled=rescale(frame.copy(),**manifest['rescale'])
 x=scaled[prep.FEATURES+prep.OUTPUT_FEATURES].to_numpy(np.float32)
 pred=booster.inplace_predict(np.ascontiguousarray(x)).astype(float)*meta['y_std']+meta['y_mean']
 true_distance=np.hypot(frame['RA_input_p'].to_numpy()-frame['RA_input_s'].to_numpy(),frame['DEC_input_p'].to_numpy()-frame['DEC_input_s'].to_numpy())*3600
 scaled_true=rescale(frame.assign(distance=true_distance),**manifest['rescale'])
 xt=scaled_true[prep.FEATURES+prep.OUTPUT_FEATURES].to_numpy(np.float32)
 assert np.array_equal(x[:,:6],xt[:,:6]) and np.array_equal(x[:,7:],xt[:,7:])
 pred_true=booster.inplace_predict(np.ascontiguousarray(xt)).astype(float)*meta['y_std']+meta['y_mean']
 gain,_=gains.response_gains(frame[['measured_e1_m','measured_e2_m']].to_numpy(float),frame['shear_angle'].to_numpy(np.float32))
 gain=gain.astype(np.float32).astype(float)
 label=(frame['delta_et1'].to_numpy(float)/.2).astype(np.float32).astype(float)
 pred*=gain;pred_true*=gain
 bins=np.digitize(frame['r_input_p'],[24,25.5,27.5]); ids=frame['input_index'].to_numpy()
 sums=np.zeros((2,4,len(Q)))
 for exc in range(2):
  for b in range(4):
   keep=(originally_excluded==bool(exc))&(bins==b)
   sums[exc,b]=[keep.sum(),len(np.unique(ids[keep])),label[keep].sum(),pred[keep].sum(),pred_true[keep].sum(),true_distance[keep].sum(),frame['distance'].to_numpy()[keep].sum()]
 np.testing.assert_array_equal(sums[0,:,0],old[case,:,oldq.index('rows')])
 np.testing.assert_allclose(sums[0,:,2],old[case,:,oldq.index('label')],rtol=1e-10,atol=1e-7)
 np.testing.assert_allclose(sums[0,:,3],old[case,:,oldq.index('galsbi_dev_pred')],rtol=1e-7,atol=1e-7)
 all_sums.append(sums);receipts.append({'case':case,'drops':counts,'baseline':brec,'seconds':time.monotonic()-start})
 np.savez(O/'geometry_sums.npz',sums=np.array(all_sums),quantities=Q,cases=np.arange(case+1))
 (O/'geometry_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 print(f'case {case} replay passed; included/excluded primaries {sums[:,:,1].sum(1)}; seconds {time.monotonic()-start:.1f}',flush=True)
a=np.array(all_sums);w=np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)
out={}
for exc,sel in [('originally_included',[0]),('originally_excluded',[1]),('all',[0,1])]:
 for b,bs in [('all',list(range(4))),('<24',[0]),('24-25.5',[1]),('25.5-27.5',[2]),('>=27.5',[3])]:
  v=a[:,sel][:,:,bs].sum((1,2));t=v.sum(0);boot=w@v
  def metrics(t):
   return np.stack([t[...,2]/t[...,1],t[...,3]/t[...,1],t[...,4]/t[...,1],(t[...,2]-t[...,3])/t[...,1],(t[...,2]-t[...,4])/t[...,1],(t[...,4]-t[...,3])/t[...,1],100*(t[...,2]/t[...,3]-1),100*(t[...,2]/t[...,4]-1)],-1)
  point=metrics(t);se=metrics(boot).std(0,ddof=1)
  names=['simulation_per_primary','model_measured_distance','model_true_distance','error_measured_distance','error_true_distance','model_change_true_minus_measured','relative_error_measured_percent','relative_error_true_percent']
  out[f'{exc}/{b}']={'pairs':int(t[0]),'primaries':int(t[1]),**{name:[float(p),float(s)] for name,p,s in zip(names,point,se)}}
(O/'geometry_report.json').write_text(json.dumps(out,indent=2)+'\n')
for k,v in out.items():print(k,v,flush=True)
