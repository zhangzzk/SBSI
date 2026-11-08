"""Paired-axis audit of completed constant-shear data, without new predictions."""
import importlib.util,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'joint_measured_photometry_20261106'
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_measured_photometry_20261106')
def main():
    assert os.environ.get('SLURM_JOB_ID')
    spec=importlib.util.spec_from_file_location('selection_completed_reducer',HERE.parent/'alignment_axis2_constant_20261105/report.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    manifest=json.loads((OLD/'manifest.json').read_text());cases=manifest['cases'];n=len(cases)
    w=np.vstack([np.ones(n),np.random.default_rng(202611069914).multinomial(n,np.full(n,1/n),10000),np.ones((n,n))-np.eye(n)])
    def ent(v):
        jk=v[10001:]
        return {'value':v[0].tolist(),'case_se':v[1:10001].std(0,ddof=1).tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0)).tolist()}
    values={}
    for axis in [1,2]:
        ss=[];mm=[]
        for case in cases:
            with np.load(ROOT/f'axis{axis}/case{case}.npz') as z:
                ci=list(z['cuts']).index('mag25_radius_gt0.60')
                ss.append(z['sim'].sum(-2)[:,ci]);mm.append(z['model'].sum(-2)[:,0,:,ci])
        for mi,name in enumerate(manifest['arms']):values[name,axis]=r.estimates(np.array(ss),np.array(mm)[:,mi],w,axis=axis)
    out={'cases':cases,'definition':'Paired whole-case bootstrap and jackknife across both shear axes; fixed fits and measurement realizations. m in percent, SE in percentage points.','arms':{}}
    for name in manifest['arms']:
        a,b=values[name,1],values[name,2]
        trace=100*((a['simulation_response'][:,0]+b['simulation_response'][:,1])/(a['model_response'][:,0]+b['model_response'][:,1])-1)
        out['arms'][name]={'trace_m_percent':ent(trace),'m2_minus_m1_pp':ent(b['m_percent']-a['m_percent']),
            'delta_R22_minus_R11':ent(b['delta_response'][:,1]-a['delta_response'][:,0])}
    out['regularization_change']={str(axis):{key:ent(values['nine_weak',axis][key]-values['baseline',axis][key]) for key in ['m_percent','model_flow_response','model_emulator_response']} for axis in [1,2]}
    (HERE/'completed_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print('COMPLETED_AUDIT',json.dumps(out),flush=True)
if __name__=='__main__':main()
