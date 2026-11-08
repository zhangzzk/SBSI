"""Direct selected-moment and jackknife replay from small saved case sums."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name

def main():
 r=json.loads((H/'report.json').read_text());p=ROOT/'sums.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==r['data_sha256']
 with np.load(p) as z:s=z['sums'];names=list(z['arms']);cuts=list(z['cuts']);cases=list(z['cases'])
 n=len(cases);ix=[list(range(n))]+[[k for k in range(n) if k!=j] for j in range(n)];count=0;maxerr=0
 for ai,name in enumerate(names):
  for ci,cut in enumerate(cuts):
   x=np.array([s[ai,jj,ci].sum(0) for jj in ix]);den=x[:,:1]
   got={'observed':x[:,1:7]/den,'predicted':(x[:,7:13]+x[:,13:19])/(2*den),'conditional_score':x[:,19:25]/den,'half_bank_difference':(x[:,7:13]-x[:,13:19])/den};got['error']=got['predicted']-got['observed']
   for k,v in got.items():
    jk=v[1:];entry={'value':v[0],'jackknife_se':np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0))}
    for stat,q in entry.items():
     expected=r['cuts'][cut]['arms'][name][k][stat];np.testing.assert_allclose(q,expected,rtol=1e-9,atol=1e-12);maxerr=max(maxerr,float(np.max(np.abs(q-expected))));count+=len(q)
 # Algebraic identity in observed/predicted quadrupoles for every saved scene/cut.
 for start in [1,7,13]:np.testing.assert_allclose(s[...,start+5],s[...,start+2]-s[...,start+4],rtol=1e-11,atol=1e-10)
 # Independent existing observed counts/shape sums on the exact same fixed rows.
 prior=ROOT.parent/'conditional_cut_probability_20261107/own_validation.npz'
 with np.load(prior) as z:
  np.testing.assert_array_equal(cases,z['cases']);np.testing.assert_array_equal(cuts,z['cuts']);v=z['probability_sums'][0,0,:,0]
  np.testing.assert_array_equal(s[0,...,0],v[...,1])
  np.testing.assert_allclose(s[0,...,1:3],v[...,5:7],rtol=1e-12,atol=1e-10)
 out={'statistic_scalars_checked':count,'maximum_absolute_difference':maxerr,'prior_selected_counts_and_shape_sums_replayed':True,'quadrupole_identity_passed':True,'prior_data_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (H/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
