"""Case bootstrap of the exact single-soft-cut measured response; CPU Slurm only."""
import json, os, hashlib
from pathlib import Path
import numpy as np
from scripts.train_fixed_g0_flow import load_split
from sbsi.flow_guard_response import numpy_guard_weights
assert os.environ.get('SLURM_JOB_ID')
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/target_response_uncertainty_20261104')
DOMAIN=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_f8c8_20261102/base/domain')
OLD=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_target_guard_repeat_20261103')
manifest=json.loads((DOMAIN/'manifest.json').read_text())
cut=({'name':'radius_gt_3_magnitude_lt_25','radius_min':3.,'magnitude_max':25.},)

def response(s):
    n,mass=s[...,0],s[...,1]
    mu=s[...,2:4]/mass[...,None]
    gram=s[...,4:8].reshape(*s.shape[:-1],2,2)
    cross=s[...,8:12].reshape(*s.shape[:-1],2,2)
    dwg=s[...,12:14]
    r=(cross-mu[..., :,None]*dwg[...,None,:])/(mass/n)[...,None,None] @ np.linalg.inv(gram)
    return np.trace(r,axis1=-2,axis2=-1)/2

results={};boots={};all_stats={};matching={}
for split in ('train','validation'):
    cases=manifest['split'][split+'_cases']; rows=[]; matching[split]=[]
    for k,c in enumerate(cases):
        d=load_split(DOMAIN,[c]); z,y,g=d['pair_zero_target'],d['pair_shear_target'],d['gamma']
        wz=numpy_guard_weights(z,cut,radius_softness=.05,magnitude_softness=.05)[:,0]
        wy=numpy_guard_weights(y,cut,radius_softness=.05,magnitude_softness=.05)[:,0]
        row=np.concatenate(([len(g),wz.sum()],(wz[:,None]*z[:,:2]).sum(0),(g.T@g).ravel(),
             ((wy[:,None]*y[:,:2]-wz[:,None]*z[:,:2]).T@g).ravel(),((wy-wz)[:,None]*g).sum(0)))
        rows.append(row);matching[split]+=d['per_case_matching']
        if (k+1)%20==0:print(split,k+1,'/',len(cases),flush=True)
    stats=np.asarray(rows); all_stats[split]=stats
    point=float(response(stats.sum(0))); expected=json.loads((OLD/f'{split}_target.json').read_text())['trace'][0][0]
    assert abs(point-expected)<1e-11,(point,expected)
    rng=np.random.default_rng(20260925 if split=='train' else 20260926)
    w=rng.multinomial(len(cases),np.full(len(cases),1/len(cases)),size=10000)
    boot=response(w@stats);boots[split]=boot
    jk=response(stats.sum(0)-stats)
    results[split]={'response':point,'case_bootstrap_se':float(boot.std(ddof=1)),
       'case_jackknife_se':float(np.sqrt((len(cases)-1)/len(cases)*np.square(jk-jk.mean()).sum())),
       'bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'cases':cases,'pairs':int(stats[:,0].sum()),
       'replay_absolute_error':abs(point-expected)}
    print(split,results[split],flush=True)
diff=boots['validation']-boots['train']; point=results['validation']['response']-results['train']['response']
ratio=100*(boots['validation']/boots['train']-1)
results['difference']={'value':point,'case_bootstrap_se':float(diff.std(ddof=1)),
    'z':point/float(diff.std(ddof=1)),'bootstrap_ci95':np.quantile(diff,[.025,.975]).tolist(),
    'relative_percent':100*(results['validation']['response']/results['train']['response']-1),
    'relative_se_pp':float(ratio.std(ddof=1))}
results['method']='10000 independent within-split whole-case bootstrap replicates; recompute baseline mass, baseline selected mean and shear normal matrix per replicate; unchanged matched usable pairs and soft cut r>3 pixels/mag<25, widths 0.05; R=(R11+R22)/2; no model evaluations'
results['job_id']=os.environ['SLURM_JOB_ID'];results['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
np.savez(ROOT/'case_sums.npz',**all_stats,train_bootstrap=boots['train'],validation_bootstrap=boots['validation'])
(ROOT/'matching.json').write_text(json.dumps(matching,indent=2)+'\n')
(ROOT/'report.json').write_text(json.dumps(results,indent=2)+'\n')
print('COMPLETE',json.dumps(results['difference']),flush=True)
