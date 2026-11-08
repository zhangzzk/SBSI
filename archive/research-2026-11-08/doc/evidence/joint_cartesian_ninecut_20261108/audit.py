"""Independent scalar replay of matched-input matrices and paired errors."""
import hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
def main():
 r=json.loads((H/'report.json').read_text());p=ROOT/'case_sums.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==r['case_sums_sha256']
 with np.load(p) as z:sim=z['sim'];oracle=z['joint'][:,0];gen=z['latest'];cuts=list(z['cuts'])
 matrices=[];mb=[]
 for omit in [None,*range(40)]:
  keep=np.ones(40,bool)
  if omit is not None:keep[omit]=False
  allm=[]
  for data in [sim,oracle,gen]:
   s=data[keep].sum(0);mu=s[...,1:3]/s[...,:1]
   if s.shape[-1]==5:mu+=s[...,3:5]/s[...,:1]
   rr=np.empty((9,2,2))
   for cut in range(9):
    for a in range(2):
     for b in range(2):rr[cut,a,b]=(mu[2*b,cut,a]-mu[2*b+1,cut,a])/.04
   allm.append(rr)
  matrices.append(allm);mb.append([[100*(allm[0][ci,k,k]/allm[j][ci,k,k]-1) for k in range(2)] for j in [1,2] for ci in range(9)])
 matrices=np.array(matrices);m=np.array(mb).reshape(41,2,9,2);out=r;count=0
 def check(v,ref):
  nonlocal count
  jk=v[1:];se=np.sqrt(39*np.mean((jk-jk.mean(0))**2,0));np.testing.assert_allclose(v[0],ref['value'],rtol=1e-9,atol=1e-10);np.testing.assert_allclose(se,ref['jackknife_se'],rtol=1e-9,atol=1e-10);count+=2
 for ci,cut in enumerate(cuts):
  row=out['cuts'][cut]
  for j,key in enumerate(['measured_matrix','joint_matrix','latest_matrix']):check(matrices[:,j,ci],row[key])
  for j,key in enumerate(['joint_m_percent','latest_m_percent']):check(m[:,j,ci],row[key])
  check(m[:,0,ci]-m[:,1,ci],row['joint_minus_latest_pp']);check(m[:,0,ci].mean(-1),row['joint_mean_m_percent']);check(m[:,0,ci,0]-m[:,0,ci,1],row['joint_m1_minus_m2_pp'])
 for a,b in [(3,6),(4,7),(5,8)]:
  row=out['radius_contrasts'][cuts[b]+' minus '+cuts[a]];o=m[:,0,b]-m[:,0,a];g=m[:,1,b]-m[:,1,a]
  for v,k in [(o,'joint_delta_m_pp'),(g,'latest_delta_m_pp'),(o-g,'joint_minus_latest_delta_pp')]:check(v,row[k])
 ro=np.sqrt((m[:,0]**2).mean((1,2)));rg=np.sqrt((m[:,1]**2).mean((1,2)))
 for v,k in [(ro,'joint'),(rg,'latest'),(ro-rg,'joint_minus_latest_pp')]:check(v,out['joint_nine_cut_rms_percent'][k])
 result={'independent_scalar_point_jackknife_checks':count,'case_sums_sha256':r['case_sums_sha256'],'report_sha256':hashlib.sha256((H/'report.json').read_bytes()).hexdigest()};(H/'completed_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('INDEPENDENT_AUDIT_PASSED',count,flush=True)
if __name__=='__main__':main()
