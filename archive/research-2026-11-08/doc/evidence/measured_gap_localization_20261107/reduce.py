"""Catalogue-only localization on the fixed four-exposure common population."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
BASE=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim')
def main():
 assert os.environ.get('SLURM_JOB_ID')
 cfg=json.loads((H.parent/'current_measured_photometry_20261107/manifest.json').read_text());all_sums=[];receipts=[];start=time.monotonic()
 for case in cfg['cases']:
  with np.load(Path(cfg['parent_root'])/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context']
  vals=[];masks=[];matching=[]
  for axis in [1,2]:
   for leg in ['0.02','-0.02']:
    cat=BASE/('constant_g002' if axis==1 else 'constant_axis2_20261105')/'sims'/f'case{case}_{leg}/real0/catalogues'
    c=pd.read_feather(cat/'CrossMatch/tile180.0_-0.5_rot0_matched.feather',columns=['id_input','id_detec'])
    s=pd.read_feather(cat/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather',columns=['NUMBER','NGMIX_G1','NGMIX_G2','FLUX_RADIUS','MAG_AUTO','X_IMAGE','Y_IMAGE'])
    assert not c.id_input.duplicated().any() and not c.id_detec.duplicated().any() and not s.NUMBER.duplicated().any()
    j=c.merge(s,left_on='id_detec',right_on='NUMBER',validate='one_to_one');matching.append([len(c)-len(j),len(s)-len(j)])
    v=j.set_index('id_input').reindex(ids)[['NGMIX_G1','NGMIX_G2','FLUX_RADIUS','MAG_AUTO','X_IMAGE','Y_IMAGE']].to_numpy(float)
    mask=np.isfinite(v[:,:4]).all(1)&(np.square(v[:,:2]).sum(1)<1)&(v[:,2]>3)&(v[:,3]<25)
    vals.append(v);masks.append(mask)
  keep=np.logical_and.reduce(masks);v=np.array(vals)[:,keep];ctx=context[keep];n=keep.sum()
  response=np.stack([(v[0,:,:2]-v[1,:,:2])/.04,(v[2,:,:2]-v[3,:,:2])/.04],-1)
  intrinsic=ctx[:,0]+1j*ctx[:,1];ideal=[]
  for shear in [.02,.02j]:
   z=((intrinsic+shear)/(1+shear.conjugate()*intrinsic)-(intrinsic-shear)/(1-shear.conjugate()*intrinsic))/.04
   ideal.append(np.stack([z.real,z.imag],-1))
  ideal=np.stack(ideal,-1)
  meanmag=v[:,:,3].mean(0);meanrad=v[:,:,2].mean(0);excursion=np.abs(response).max((1,2))
  # Statistics are count + measured four entries + ideal four entries.
  rows=np.column_stack([np.ones(n),response.reshape(n,4),ideal.reshape(n,4)])
  groups={'all':np.ones(n,bool)}
  for name,values,bins in [('mag',meanmag,[-np.inf,22,23,24,24.5,np.inf]),('radius',meanrad,[3,4,5,7,np.inf]),('response_max',excursion,[0,2,5,10,20,np.inf]),('intrinsic_abs',np.abs(intrinsic),[0,.2,.4,.6,1])]:
   for lo,hi in zip(bins[:-1],bins[1:]):groups[f'{name}_{lo}_{hi}']=(values>=lo)&(values<hi)
  # Median four-leg position anchors spatial bins, fixed before differencing.
  xy=np.median(v[:,:,4:6],axis=0);phase=xy-np.rint(xy)
  for dim in [0,1]:
   for lo,hi in [(-.5,-.25),(-.25,0),(0,.25),(.25,.50001)]:groups[f'phase{dim}_{lo}_{hi}']=(phase[:,dim]>=lo)&(phase[:,dim]<hi)
  sums=np.stack([rows[m].sum(0) for m in groups.values()]);all_sums.append(sums)
  receipts.append({'case':case,'common_count':int(n),'matching':matching})
  print('CASE_DONE',case,'count',int(n),'seconds',round(time.monotonic()-start,1),flush=True)
 data=np.array(all_sums);np.savez(ROOT/'sums.npz',sums=data,cases=cfg['cases'],groups=list(groups))
 out={'definition':'Same IDs selected in all four original exposures; response entries R11,R12,R21,R22. Intrinsic ideal response uses exact reduced-ellipticity shear composition and is not a model for noisy Gaussian-fit measurements. Groups depend on four-leg averages or measured response and are diagnostic, not acceptance cuts.','receipts':receipts,'groups':{}}
 n=len(data);w=np.vstack([np.ones(n),1-np.eye(n)]);pooled=np.einsum('bc,cgd->bgd',w,data)
 def stat(x):return {'value':float(x[0]),'jackknife_se':float(np.sqrt((n-1)/n*np.square(x[1:]-x[1:].mean()).sum()))}
 for i,name in enumerate(groups):
  p=pooled[:,i];r=p[:,1:5]/p[:,:1];ideal=p[:,5:9]/p[:,:1]
  out['groups'][name]={'count':int(p[0,0]),'fraction':float(p[0,0]/pooled[0,0,0]),'diagonal_gap':stat(r[:,0]-r[:,3]),'ideal_diagonal_gap':stat(ideal[:,0]-ideal[:,3]),'contribution_to_total_gap':stat((p[:,1]-p[:,4])/pooled[:,0,0]),'matrix':r[0].reshape(2,2).tolist()}
 ref=json.loads((H.parent/'measured_response_symmetry_20261107/membership_report.json').read_text())['cuts']['mag25_radius_gt0.60']['common_four']
 np.testing.assert_allclose(out['groups']['all']['matrix'],ref['matrix']['value'],atol=1e-12)
 out['common_matrix_replayed']=True
 (H/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('ALL',out['groups']['all'],flush=True);print('DONE_LOCALIZATION',flush=True)
if __name__=='__main__':main()
