import os,json,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((ROOT/'manifest.json').read_text())
assert sha(ROOT/'source/evaluate_galsbi_constant.py')==manifest['source_sha256']
assert sha(ROOT/'preset.json')==manifest['preset_sha256']
assert os.environ.get('SLURM_JOB_ID')
os.environ['EVAL_OUT']=str(ROOT);os.environ['EVAL_PRESET']=str(ROOT/'preset.json')
spec=importlib.util.spec_from_file_location('constant_eval_snapshot',ROOT/'source/evaluate_galsbi_constant.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
stage=sys.argv[1]
if stage=='draws':
 chunk=int(sys.argv[2]);assert chunk in manifest['chunks'];ev.draws(chunk)
elif stage=='summary':
 ev.CASES=tuple(manifest['cases']);ev.summary()
 summary=json.loads((ROOT/'summary.json').read_text());summary['case_list']=manifest['cases'];summary['preset_sha256']=manifest['preset_sha256'];(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 baseline=Path(manifest['baseline_root']);a=[];b=[];s=[]
 for case in manifest['cases']:
  with np.load(ROOT/f'sums/case{case}.npz') as x,np.load(baseline/f'sums/case{case}.npz') as y:
   assert np.array_equal(x['cuts'],y['cuts'])
   assert np.array_equal(x['sim'],y['sim']),f'simulation mismatch case {case}'
   assert np.isfinite(x['model']).all() and np.isfinite(x['sim']).all()
   a.append(x['model']);b.append(y['model']);s.append(x['sim'])
 a,b,s=map(np.asarray,(a,b,s));n=len(s)
 W=np.random.default_rng(20260925).multinomial(n,np.full(n,1/n),size=10000)
 def resp(x):return (x[...,0,1]/x[...,0,0]-x[...,1,1]/x[...,1,0])/.04
 def view(x,w,k):
  y=x[:,w,:,k,:,:5].sum(-2)
  return np.stack((y[...,0],y[...,1]+y[...,3]),-1)
 rows={}
 old=json.loads(Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/doc/evidence/galsbi_bands40_constant_20261104/compare.json').read_text())
 for w,label in enumerate(['C8','true flags']):
  for k,cut in enumerate(ev.CUT_NAMES):
   sims=s[:,:,k,:,:3].sum(-2)[...,:2];aa=view(a,w,k);bb=view(b,w,k)
   rs,ra,rb=[resp(v.sum(0)) for v in [sims,aa,bb]]
   bs,ba,bboot=[resp(np.einsum('bc,c...->b...',W,v)) for v in [sims,aa,bb]]
   ma,mb=100*(rs/ra-1),100*(rs/rb-1);bma,bmb=100*(bs/ba-1),100*(bs/bboot-1)
   key=f'{label}/{cut}';previous=old['cuts'][key]['B_galsbi_dev']
   assert abs(mb-previous[0])<1e-10 and abs(bmb.std(ddof=1)-previous[1])<1e-10
   assert abs(ma-summary['results'][key]['m_percent'])<1e-10
   rows[key]={'candidate_m_percent':float(ma),'candidate_case_se_pp':float(bma.std(ddof=1)),'baseline_m_percent':float(mb),'baseline_case_se_pp':float(bmb.std(ddof=1)),'change_pp':float(ma-mb),'paired_change_se_pp':float((bma-bmb).std(ddof=1))}
 result={'cases':manifest['cases'],'flow_sha256':manifest['candidate_sha256'],'cuts':rows,'simulation_exact_replay':True,'baseline_report_replay':True,'production_accepted':False}
 (ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 print('PRIMARY',json.dumps(rows['C8/mag25_radius_gt0.60']),flush=True)
else:raise ValueError(stage)
