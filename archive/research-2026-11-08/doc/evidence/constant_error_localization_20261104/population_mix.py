"""Exact symmetric within-bin / mixture decomposition of common-selection response."""
from pathlib import Path
import json
import numpy as np
O=Path(__file__).resolve().parent;D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_dev_constant_grid_20261104')
cases=[c for k in (0,4,8,12) for c in range(1000+10*k,1010+10*k)]
s=[];m=[]
for c in cases:
 with np.load(D/f'sums/case{c}.npz') as z:
  k=list(z['cuts']).index('mag25_radius_gt0.60');s.append(z['sim'][:,k,:,3:]);m.append(z['model'][1,:,k,:,5:])
s=np.array(s);m=np.array(m)
w=np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)
def calc(agg):
 ss=agg(s);mm=agg(m)
 np.testing.assert_allclose(ss[...,0,:,0],ss[...,1,:,0],atol=1e-8)
 np.testing.assert_allclose(mm[...,0,:,0],mm[...,1,:,0],atol=1e-8)
 fs=ss[...,0,:,0]/ss[...,0,:,0].sum(-1)[...,None];fm=mm[...,0,:,0]/mm[...,0,:,0].sum(-1)[...,None]
 rs=(ss[...,0,:,1]-ss[...,1,:,1])/(.04*ss[...,0,:,0])
 rm=(mm[...,0,:,1]+mm[...,0,:,3]-mm[...,1,:,1]-mm[...,1,:,3])/(.04*mm[...,0,:,0])
 within=.5*(fs+fm)*(rs-rm);mix=.5*(rs+rm)*(fs-fm)
 np.testing.assert_allclose((within+mix).sum(-1),(fs*rs-fm*rm).sum(-1),atol=1e-12)
 return np.stack([fs,fm,rs,rm,within,mix],-1)
p=calc(lambda x:x.sum(0));b=calc(lambda x:np.einsum('bc,c...->b...',w,x));se=b.std(0,ddof=1)
names=['sim_fraction','model_fraction','sim_response','model_response','within_bin_error_contribution','population_mix_contribution']
out={bn:{nm:[float(p[k,j]),float(se[k,j])] for j,nm in enumerate(names)} for k,bn in enumerate(['<24','24-25.5','25.5-27.5','>=27.5'])}
out['total']={nm:[float(p[:,j].sum()),float(b[:,:,j].sum(1).std(ddof=1))] for j,nm in [(4,'within_bin_error'),(5,'population_mix_error')]}
(O/'population_mix_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
