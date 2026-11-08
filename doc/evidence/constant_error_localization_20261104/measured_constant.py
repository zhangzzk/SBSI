"""Measured-input emulator sensitivity on eight predeclared constant-shear cases."""
import os,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import xgboost as xgb
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
sys.path.insert(0,'/home/z/Zekang.Zhang/SBSI')
from sbsi.forward_catalogue import rescale_emulator_pairs
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang')
root=D/'sbsi/galsbi_resim/constant_g002/sims'
model=D/'blendemu_runs/disk_response_moment_galsbi_20261103/model'
meta=json.loads((model/'metadata.json').read_text());b=xgb.Booster(model_file=model/'model.json');b.set_param({'nthread':int(os.environ.get('SLURM_CPUS_PER_TASK',4))})
preset=json.loads(Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/configs/models_galsbi_dev.json').read_text())
cases=[1000,1001,1040,1041,1080,1081,1120,1121]
features=meta['features'];all_sums=[];records=[]
for case in cases:
 start=time.monotonic()
 t=pd.read_feather(root/f'case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather').sort_values('index_input').reset_index(drop=True)
 ids=t['index_input'].to_numpy();pos=t[['RA_input','DEC_input']].to_numpy();tree=KDTree(pos)
 legs=[];reports=[]
 for h in [.02,-.02]:
  cat=root/f'case{case}_{h:.2f}/real0/catalogues'
  cross=pd.read_feather(cat/'CrossMatch/tile180.0_-0.5_rot0_matched.feather',columns=['id_input','id_detec'])
  shape=pd.read_feather(cat/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather',columns=['NUMBER','NGMIX_G1','NGMIX_G2','MAG_AUTO','FLUX_RADIUS','X_WORLD','Y_WORLD'])
  assert not cross['id_input'].duplicated().any() and not cross['id_detec'].duplicated().any() and not shape['NUMBER'].duplicated().any()
  joined=cross.merge(shape,left_on='id_detec',right_on='NUMBER',validate='one_to_one').set_index('id_input')
  a=joined.reindex(ids);v=a[['NGMIX_G1','NGMIX_G2','FLUX_RADIUS','MAG_AUTO']].to_numpy(float)
  keep=np.isfinite(v).all(1)&((v[:,:2]**2).sum(1)<1)&(v[:,2]>3)&(v[:,3]<25)
  assert np.isfinite(a.loc[keep,['X_WORLD','Y_WORLD']]).all().all()
  reports.append({'cross_without_shape':len(cross)-len(joined),'selected':int(keep.sum()),'missing_parent_cross':int((~np.isin(ids,cross.index if False else cross['id_input'])).sum())})
  legs.append((a,v,keep))
 common=legs[0][2]&legs[1][2]
 # leg, cohort(own/common), true magnitude bin, quantity(count, shape, shift_base,shift_measured_distance,shift_k40,linear_base)
 sums=np.zeros((2,2,4,6))
 for l,(a,v,keep) in enumerate(legs):
  h=[.02,-.02][l];rows=np.flatnonzero(keep);n=len(rows)
  dist,ind=tree.query(pos[rows],k=41,distance_upper_bound=10/3600,workers=int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
  valid=(ind<len(t))&(ind!=rows[:,None])&(dist>0)
  rank=np.cumsum(valid,axis=1);valid&=rank<=40
  pr=np.broadcast_to(np.arange(n)[:,None],valid.shape)[valid];sr=ind[valid];rank=rank[valid];distance=dist[valid]*3600
  frame=pd.DataFrame({f'{name}_input_p':t[f'{name}_input'].to_numpy()[rows[pr]] for name in ['Re','r','sersic_n']})
  for name in ['Re','r','sersic_n']:frame[f'{name}_input_s']=t[f'{name}_input'].to_numpy()[sr]
  frame['distance']=distance
  scaled=rescale_emulator_pairs(frame,preset['observing_conditions'])
  scaled['baseline_log_flux_radius']=np.log(v[rows[pr],2]);scaled['baseline_mag_auto']=v[rows[pr],3]
  x=scaled[features].to_numpy(np.float32)
  pred=b.inplace_predict(np.ascontiguousarray(x)).astype(float)*meta['y_std']+meta['y_mean']
  base=rank<=19
  B0=np.bincount(pr[base],weights=pred[base],minlength=n)
  B40=np.bincount(pr,weights=pred,minlength=n)
  mpos=a[['X_WORLD','Y_WORLD']].to_numpy()[rows]
  mdist=np.hypot(*(mpos[pr]-pos[sr]).T)*3600
  scaled_m=rescale_emulator_pairs(frame.assign(distance=mdist),preset['observing_conditions'])
  xm=x[base].copy();xm[:,6]=scaled_m['distance_scaled'].to_numpy(np.float32)[base]
  pm=b.inplace_predict(np.ascontiguousarray(xm)).astype(float)*meta['y_std']+meta['y_mean']
  BM=np.bincount(pr[base],weights=pm,minlength=n)
  e=v[rows,0]+1j*v[rows,1]
  shifts=[]
  for B in [B0,BM,B40]:
   shift=np.tanh(h*B);transformed=(e+shift)/(1+shift*e)
   shifts.append(transformed.real-e.real)
  linear=h*B0*(1-v[rows,0]**2+v[rows,1]**2)
  bins=np.digitize(t['r_input'].to_numpy()[rows],[24,25.5,27.5])
  vals=np.column_stack([np.ones(n),e.real,*shifts,linear])
  for c,sel in enumerate([np.ones(n,dtype=bool),common[rows]]):
   for k in range(4):sums[l,c,k]=vals[sel&(bins==k)].sum(0)
 all_sums.append(sums);records.append({'case':case,'seconds':time.monotonic()-start,'catalogues':reports})
 np.savez(O/'measured_constant_sums.npz',sums=np.array(all_sums),cases=cases[:len(all_sums)])
 (O/'measured_constant_receipts.json').write_text(json.dumps(records,indent=2)+'\n')
 print(f'case {case} complete {time.monotonic()-start:.1f}s',flush=True)
# Compare same cases with saved true-flag joint response.
a=np.array(all_sums);old=[];sim=[]
for case in cases:
 with np.load(D/f'sbsi_caches/galsbi_dev_constant_grid_20261104/sums/case{case}.npz') as z:
  k=list(z['cuts']).index('mag25_radius_gt0.60');old.append(z['model'][1,:,k]);sim.append(z['sim'][:,k])
old=np.array(old);sim=np.array(sim)
np.testing.assert_allclose(a[:,0,0,:,:2],sim[:,0,:,:2],rtol=1e-12,atol=1e-8)
np.testing.assert_allclose(a[:,1,0,:,:2],sim[:,1,:,:2],rtol=1e-12,atol=1e-8)
w=np.random.default_rng(20260925).multinomial(8,np.full(8,1/8),size=10000)
out={}
for c,cohort in enumerate(['own_selection','common_selection']):
 for bn,bs in [('all',[0,1,2,3]),('<24',[0]),('24-25.5',[1]),('25.5-27.5',[2])]:
  av=a[:,:,c][:,:,bs].sum(2);mv=old[:,:,bs].sum(2);sv=sim[:,:,bs].sum(2)
  def calc(agg):
   x=agg(av);m=agg(mv);s=agg(sv)
   def resp(num,den):return (num[...,0]/den[...,0]-num[...,1]/den[...,1])/.04
   j=0 if c==0 else 5;sj=0 if c==0 else 3
   em=resp(m[...,j+3],m[...,j]);flow=resp(m[...,j+1],m[...,j]);rs=resp(s[...,sj+1],s[...,sj])
   proxy=[resp(x[...,i],x[...,0]) for i in range(2,6)]
   return np.stack([rs,flow,em,rs-flow-em,*proxy,proxy[0]-em,proxy[1]-proxy[0],proxy[2]-proxy[0],proxy[3]-proxy[0]],-1)
  p=calc(lambda x:x.sum(0));e=calc(lambda x:np.einsum('bc,c...->b...',w,x)).std(0,ddof=1)
  names=['R_sim','R_flow','R_emulator_original','joint_error','emulator_measured_inputs','emulator_measured_distance','emulator_k40','emulator_linear','change_measured_inputs','change_measured_distance','change_k40','change_linear']
  out[f'{cohort}/{bn}']={q:[float(v),float(se)] for q,v,se in zip(names,p,e)}
(O/'measured_constant_report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
