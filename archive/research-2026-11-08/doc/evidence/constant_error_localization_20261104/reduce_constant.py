"""Re-express saved matched constant-shear diagnostics in absolute response units."""
from pathlib import Path
import json
import numpy as np
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_dev_constant_grid_20261104')
cases=[c for k in (0,4,8,12) for c in range(1000+10*k,1010+10*k)]
m=[];s=[]
for c in cases:
 with np.load(D/f'sums/case{c}.npz') as z:
  m.append(z['model'][1]);s.append(z['sim']);cuts=list(z['cuts'])
m=np.array(m);s=np.array(s)
w=np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)
def response(x):return (x[...,0,1]/x[...,0,0]-x[...,1,1]/x[...,1,0])/.04
def stats(v):return [float(v[0]),float(v[1:].std(ddof=1))]
out={'cases':cases,'definition':'simulation minus model; true usability flags; 10000 paired case bootstrap samples, seed20260925','cuts':{}}
for k,cut in enumerate(cuts):
 ms=m[:,:,k].sum(2);ss=s[:,:,k].sum(2)
 def calc(agg):
  mm=agg(ms);st=agg(ss)
  joint=np.stack([mm[...,0],mm[...,1]+mm[...,3]],-1)
  both=np.stack([mm[...,5],mm[...,6]+mm[...,8]],-1)
  rs,rm,bs,bm=response(st[...,:3]),response(joint),response(st[...,3:]),response(both)
  flow=response(np.stack([mm[...,5],mm[...,6]],-1))
  return np.stack([rs,rm,rs-rm,bs,bm,bs-bm,(rs-bs)-(rm-bm),flow,bm-flow],-1)
 v=np.concatenate([calc(lambda x:x.sum(0))[None],calc(lambda x:np.einsum('bc,c...->b...',w,x))])
 names=['R_sim','R_model','error_total','R_common_sim','R_common_model','error_common','error_selection_remainder','R_common_flow','R_common_emulator']
 out['cuts'][cut]={name:stats(v[:,j]) for j,name in enumerate(names)}
 np.testing.assert_allclose(v[:,2],v[:,5]+v[:,6],atol=1e-14)
# Per-bin common-selection response plus contribution to total common-selection error.
k=cuts.index('mag25_radius_gt0.60');mt=m[:,:,k];st=s[:,:,k]
def bin_calc(agg):
 mm=agg(mt);ss=agg(st)
 # ... leg, bin, quantity
 sn=ss[...,3];mn=mm[...,5]; se=ss[...,4];mf=mm[...,6];mb=mm[...,8]
 rs=(se[...,0,:]/sn[...,0,:]-se[...,1,:]/sn[...,1,:])/.04
 rf=(mf[...,0,:]/mn[...,0,:]-mf[...,1,:]/mn[...,1,:])/.04
 rb=(mb[...,0,:]/mn[...,0,:]-mb[...,1,:]/mn[...,1,:])/.04
 # Own-side total normalizers make contributions exactly sum to aggregate discrepancy.
 sc=(se[...,0,:]/sn[...,0,:].sum(-1)[...,None]-se[...,1,:]/sn[...,1,:].sum(-1)[...,None])/.04
 mc=((mf+mb)[...,0,:]/mn[...,0,:].sum(-1)[...,None]-(mf+mb)[...,1,:]/mn[...,1,:].sum(-1)[...,None])/.04
 return np.stack([rs,rf,rb,rs-rf-rb,sc-mc],-1)
v=np.concatenate([bin_calc(lambda x:x.sum(0))[None],bin_calc(lambda x:np.einsum('bc,c...->b...',w,x))])
out['target_true_magnitude_bins']={name:{q:stats(v[:,b,j]) for j,q in enumerate(['R_sim','R_flow','R_emulator','error_within_bin','contribution_to_common_error'])} for b,name in enumerate(['<24','24-25.5','25.5-27.5','>=27.5'])}
assert abs(v[0,:,4].sum()-out['cuts']['mag25_radius_gt0.60']['error_common'][0])<1e-12
original=json.loads(Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/doc/evidence/galsbi_bands40_constant_20261104/split.json').read_text())
for cut,row in out['cuts'].items():
 for oldkey,newkey in [('R_both_sim','R_common_sim'),('R_both_model','R_common_model')]:
  np.testing.assert_allclose(row[newkey],original['B_galsbi_dev/'+cut][oldkey],atol=1e-12,rtol=1e-12)
(O/'constant_report.json').write_text(json.dumps(out,indent=2)+'\n')
for cut in ['uncut','mag25','radius_gt0.60','mag25_radius_gt0.60','mag25_radius_ge0.75']:print(cut,out['cuts'][cut])
print(out['target_true_magnitude_bins'])
