"""Lightweight arithmetic on saved case sums; no model or catalogue evaluation."""
import json
from pathlib import Path
import numpy as np
P=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
O=Path(__file__).resolve().parent

def weights(n):return np.random.default_rng(20260925).multinomial(n,np.full(n,1/n),size=10000)
def entry(rs,rm,bs,bm,unit):return {'simulation':float(rs),'model':float(rm),'sim_minus_model':float(rs-rm),'paired_case_se':float((bs-bm).std(ddof=1)),'unit':unit}
out={'definition':'Signed response discrepancy: R_sim-R_model; positive means model underprediction.','bootstrap':'10000 whole-case paired resamples, seed20260925; fixed models and latent realizations','rows':{}}
F=P/'galsbi_target_guard_repeat_20261103/seed0';r=json.loads((F/'check_final.json').read_text())
k=next(i for i,c in enumerate(r['cuts']) if c['radius_min']==3 and c['magnitude_max']==25)
with np.load(F/'check_final_sums.npz') as a:
 s=a['sim'][:,k,[0,3]].sum(-1)/2;m=a['model_steps4'][:,k,[0,3]].sum(-1)/2;n=a['count'];w=weights(len(n))
 rs,rm=s.sum()/n.sum(),m.sum()/n.sum();bs,bm=(w@s)/(w@n),(w@m)/(w@n)
 out['rows']['flow_validation']=entry(rs,rm,bs,bm,'half-trace per galaxy; fixed-baseline soft-cut response on matched usable pairs')
 assert abs(100*(rs/rm-1)-r['flows']['steps4']['m_eq'][k])<1e-10
 assert abs(np.std(100*(bs/bm-1),ddof=1)-r['flows']['steps4']['m_eq_se'][k])<1e-10
with np.load(P/'galsbi_emulator_check_20261104/sums.npz') as a:
 x=a['sums'][:,:,:4,:,:].sum((1,2,3));q=list(a['quantities']);w=weights(len(x));b=w@x;t=x.sum(0)
 rs,rm=t[q.index('label')]/t[0],t[q.index('galsbi_dev_pred')]/t[0]
 bs,bm=b[:,q.index('label')]/b[:,0],b[:,q.index('galsbi_dev_pred')]/b[:,0]
 out['rows']['emulator_development']=entry(rs,rm,bs,bm,'projected neighbour-shear response per pair, measured g0 inputs')
 old=json.loads(Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/doc/evidence/galsbi_emulator_check_20261104/emulator_check.json').read_text())['radius>3 & mag<25 (target)']['all']['galsbi_dev']['d_percent']
 assert abs(100*(rs/rm-1)-old[0])<1e-10
 assert abs(np.std(100*(bs/bm-1),ddof=1)-old[1])<1e-10
j=json.loads((P/'constant_component_accounting_20261104/report.json').read_text())
for flag in ['C8','true flags']:
 v=j['flags'][flag];out['rows']['joint_'+flag]={'simulation':j['R_sim_total']['value'],'model':v['R_joint']['value'],'sim_minus_model':-v['R_joint_minus_sim']['value'],'paired_case_se':v['R_joint_minus_sim']['se'],'unit':'R11 per selected galaxy; independent own-leg hard cuts'}
out['cut']='radius>3 pixels and magnitude<25, using each original study selection convention'
out['limitations']='Component absolute errors have different denominators (galaxy versus pair), populations and response definitions; do not add them into a joint error budget.'
(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
