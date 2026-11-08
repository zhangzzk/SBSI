import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
r=json.loads((ROOT/'fit_stability.json').read_text());out={'definition':r['definition'],'groups':{}}
for group in ['seed_sensitive','stable_control']:
 rows=[v['fits'] for v in r['rows'] if v['group']==group];s={}
 for a,b in [('default_A','default_B'),('tight_A','tight_B'),('continue_A','continue_B'),('continue_A','default_A'),('continue_B','default_B')]:
  paired=[v for v in rows if v[a]['g'] is not None and v[b]['g'] is not None and v[a]['objective'] is not None and v[b]['objective'] is not None]
  delta=np.array([np.max(np.abs(np.array(v[a]['g'])-v[b]['g'])) for v in paired]);gain=np.array([v[b]['objective']-v[a]['objective'] for v in paired])
  s[a+'_vs_'+b]={'valid_pairs':len(paired),'shape_delta_quantiles':np.quantile(delta,[0,.5,.9,1]).tolist(),'shape_change_over_0.1':int((delta>.1).sum()),'objective_gain_quantiles':np.quantile(gain,[0,.5,.9,1]).tolist(),'objective_gain_gt1':int((gain>1).sum()),'nonzero_flags':sum(v[a]['flags']!=0 for v in rows)}
 psf=np.array([np.max(np.abs(np.array(v['default_A']['psf_pars'])-v['default_B']['psf_pars'])) for v in rows]);s['psf_pars_max_abs_diff_quantiles']=np.quantile(psf,[0,.5,.9,1]).tolist()
 s['cross_node_default_replay_max_abs']=max(v[k]['pilot_replay_max_abs'] for v in rows for k in ['default_A','default_B'])
 out['groups'][group]=s
(H/'stability_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
