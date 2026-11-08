"""Paired scene-level constant-shear response and multiplicative bias."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_factorized_constant_20261105')
MODES=['predicted_usability_generated_selection','true_usability_generated_selection','true_usability_true_selection']

def response(x):
    return (x[...,0,1:]/x[...,0,:1]-x[...,1,1:]/x[...,1,:1])/.04

def ent(v):
    return {'value':v[0].tolist(),'case_se':np.std(v[1:],axis=0,ddof=1).tolist()}

def estimates(sim,model,weights):
    s=np.einsum('bc,clq->blq',weights,sim)
    x=np.einsum('bc,clq->blq',weights,model)
    flow=response(x[...,:3]);emu=response(np.concatenate([x[...,:1],x[...,3:5]],-1))
    total=response(np.concatenate([x[...,:1],x[...,1:3]+x[...,3:5]],-1))
    np.testing.assert_allclose(flow+emu,total,rtol=1e-12,atol=1e-12)
    rs=response(s);delta=rs-total
    m=100*delta[...,0]/total[...,0]
    return {'simulation_response':rs,'model_response':total,'model_flow_response':flow,
            'model_emulator_response':emu,'delta_response':delta,'m_percent':m,
            'selected_count_ratio':x[...,0]/s[...,0]}

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','full']);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    manifest=json.loads((HERE/'manifest.json').read_text())
    cases=manifest['pilot_cases' if args.phase=='pilot' else 'cases'];n=len(cases)
    cuts=[c[0] for c in manifest['cuts']]
    weights=np.vstack([np.ones(n),np.random.default_rng(202611057001).multinomial(n,np.full(n,1/n),10000)])
    loo=np.ones((n,n))-np.eye(n)
    result={'phase':args.phase,'cases':cases,'definitions':'DeltaR=simulation-model; m=100(Rsim/Rmodel-1). Whole-case paired bootstrap SE; fitted models and measured catalogues fixed. Model flow+emulator components add exactly but are not standalone component errors. True selection mode keeps generated emulator inputs. Development only.', 'arms':{}}
    samples={};reference=None
    for arm in ['retained','cartesian','frame']:
        models=[];simulations=[]
        for case in cases:
            path=(Path(manifest['baseline_root'])/'sums' if arm=='retained' else ROOT/arm)/f'case{case}.npz'
            with np.load(path) as z:
                ci=[list(z['cuts']).index(c) for c in cuts]
                models.append(z['model'][:,:,ci,:,:5].sum(-2))
                simulations.append(z['sim'][:,ci,:,:3].sum(-2))
        model,sim=np.array(models),np.array(simulations)
        if reference is None:reference=sim
        else:np.testing.assert_allclose(sim,reference,rtol=1e-12,atol=1e-8)
        result['arms'][arm]={}
        for mode in range(model.shape[1]):
            reports={}
            for k,cut in enumerate(cuts):
                v=estimates(sim[:,:,k],model[:,mode,:,k],weights)
                samples[arm,mode,cut]=v
                report={key:ent(val) for key,val in v.items()}
                percase=estimates(sim[:,:,k],model[:,mode,:,k],np.eye(n))
                jack=estimates(sim[:,:,k],model[:,mode,:,k],loo)
                report['per_case_m_percent']=percase['m_percent'].tolist()
                report['leave_one_case_out_m_range_percent']=[float(jack['m_percent'].min()),float(jack['m_percent'].max())]
                report['m_percent_bootstrap_ci95']=np.quantile(v['m_percent'][1:],[.025,.975]).tolist()
                reports[cut]=report
            result['arms'][arm][MODES[mode]]=reports
    result['paired_changes']={}
    for a,b in [('cartesian','retained'),('frame','retained'),('frame','cartesian')]:
        modes={}
        for mode in range(2 if b=='retained' else 3):
            modes[MODES[mode]]={cut:{key:ent(samples[a,mode,cut][key]-samples[b,mode,cut][key]) for key in ['m_percent','delta_response','model_flow_response','model_emulator_response']} for cut in cuts}
        result['paired_changes'][a+'_minus_'+b]=modes
    (ROOT/f'{args.phase}_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for a,modes in result['arms'].items():
        for mode,c in modes.items():
            for cut in ['mag25_radius_gt0.60','mag25.8_radius_gt0.60']:
                print(a,mode,cut,'m',c[cut]['m_percent'],'delta',c[cut]['delta_response'],flush=True)
    print('DONE',args.phase,flush=True)

if __name__=='__main__':main()
