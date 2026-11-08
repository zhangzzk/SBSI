"""Paired joint calibration of all regularization endpoints on both shear axes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106')
spec=importlib.util.spec_from_file_location('axis_report',HERE.parent/'alignment_axis2_constant_20261105/report.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','interim','full']);args=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text());cases=manifest['cases'] if args.phase!='pilot' else manifest['cases'][:1]
    if args.phase=='interim':
        directories=[ROOT.parent/'alignment_target9_20261106'/f'axis{axis}' for axis in [1,2]]
        directories += [ROOT/'constant'/a['name']/f'axis{axis}' for a in manifest['arms'] for axis in [1,2]]
        # Freeze the common completed case set before reading any results.
        cases=[case for case in cases if all((directory/f'case{case}.npz').exists() and (directory/f'case{case}.json').exists() for directory in directories)]
        if len(cases)<2:raise RuntimeError('At least two cases complete for every arm and axis are required')
    n=len(cases)
    weights=np.vstack([np.ones(n),np.random.default_rng(202611068601).multinomial(n,np.full(n,1/n),10000)])
    names=['baseline']+[a['name'] for a in manifest['arms']]
    result={'cases':cases,'phase':args.phase,'planned_case_count':len(manifest['cases']),'definition':'m_j=100(Rsim_jj/Rmodel_jj-1); DeltaR=simulation-model. Fixed-model paired whole-case SE; no independent acceptance. Each arm uses its own selected normalization per leg.','axes':{}}
    for axis in [1,2]:
        arms={};samples={};reference=None
        for name in names:
            directory=ROOT.parent/'alignment_target9_20261106'/f'axis{axis}' if name=='baseline' else ROOT/'constant'/name/f'axis{axis}'
            sim=[];model=[]
            for case in cases:
                with np.load(directory/f'case{case}.npz') as z:
                    np.testing.assert_array_equal(z['cuts'],[c[0] for c in manifest['cuts']])
                    sim.append(z['sim'].sum(-2));model.append(z['model'].sum(-2))
            sim,model=np.array(sim),np.array(model)
            if reference is None:reference=sim
            else:np.testing.assert_allclose(sim,reference,rtol=1e-12,atol=1e-8)
            arms[name]={}
            for mode,label in enumerate(r.MODES[:2]):
                rows={};target_samples=[]
                for k,(cut,_,_) in enumerate(manifest['cuts']):
                    v=r.estimates(sim[:,:,k],model[:,mode,:,k],weights,axis=axis);samples[name,label,cut]=v
                    rows[cut]={key:r.ent(value) for key,value in v.items()}
                    if cut in manifest['target_cuts']:target_samples.append(v['m_percent'])
                ts=np.stack(target_samples,-1);assert ts.shape[-1]==9
                metrics={'rms_m_percent':np.sqrt((ts**2).mean(-1)),'max_abs_m_percent':np.abs(ts).max(-1),'range_m_percentage_points':ts.max(-1)-ts.min(-1)}
                samples[name,label,'summary']=metrics;rows['nine_cut_summary']={k:r.ent(v) for k,v in metrics.items()}
                arms[name][label]=rows
        comparisons={}
        for name in names[1:]:
            for ref in ['baseline','nll_control']:
                if name==ref:continue
                modes={}
                for label in r.MODES[:2]:
                    modes[label]={cut:{key:r.ent(samples[name,label,cut][key]-samples[ref,label,cut][key]) for key in ['m_percent','delta_response']} for cut in manifest['target_cuts']}
                    modes[label]['nine_cut_summary']={key:r.ent(samples[name,label,'summary'][key]-samples[ref,label,'summary'][key]) for key in samples[name,label,'summary']}
                comparisons[name+'_minus_'+ref]=modes
        result['axes'][str(axis)]={'arms':arms,'paired_changes':comparisons}
    if n<2:
        def null_se(x):
            if isinstance(x,dict):return {k:(None if k=='case_se' else null_se(v)) for k,v in x.items()}
            if isinstance(x,list):return [null_se(v) for v in x]
            return x
        result=null_se(result)
    (ROOT/f'{args.phase}_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DONE',args.phase,n,flush=True)
if __name__=='__main__':main()
