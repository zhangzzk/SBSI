"""Exact paired response-change attribution from saved coherent sums."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
OUT=ROOT/'alignment_regularization_shift_20261106'

def ent(v):
    return {'value':v[0].tolist(),'case_se':v[1:].std(0,ddof=1).tolist()}

def diffleg(x):return (x[:,0]-x[:,1])/.04

def parts(x):
    mass=x[...,0];denom=mass.sum(-1)
    fraction=mass/denom[...,None]
    mean=np.divide(x[...,1:],mass[...,None],out=np.zeros_like(x[...,1:]),where=mass[...,None]>0)
    numerator=x[...,1:].sum(-2)
    response=diffleg(numerator/denom[...,None])
    contributions=diffleg(x[...,1:]/denom[...,None,None])
    np.testing.assert_allclose(contributions.sum(1),response,rtol=1e-11,atol=1e-12)
    return mass,denom,fraction,mean,numerator,response,contributions

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    m=json.loads((HERE/'manifest.json').read_text());cases=m['constant_cases'];n=len(cases)
    w=np.vstack([np.ones(n),np.random.default_rng(m['seed']).multinomial(n,np.full(n,1/n),m['bootstrap'])])
    result={'cases':cases,'definitions':'Model response changes new-minus-baseline. Four columns: flow1,flow2,emulator1,emulator2. Per-leg own selection/denominators. Exact descriptive identities; not standalone physical component errors. Case SEs condition on fixed models.','axes':{}}
    prior=json.loads((ROOT/'alignment_regularized_20261106/interim_report.json').read_text());assert cases==prior['cases']
    modes=['predicted_usability_generated_selection','true_usability_generated_selection']
    for axis in [1,2]:
        data={}
        for name in m['arms']:
            directory=ROOT/'alignment_target9_20261106'/f'axis{axis}' if name=='baseline' else ROOT/'alignment_regularized_20261106/constant'/name/f'axis{axis}'
            arrays=[]
            for case in cases:
                with np.load(directory/f'case{case}.npz') as z:
                    np.testing.assert_array_equal(z['cuts'],[c[0] for c in m['constant_cuts']]);arrays.append(z['model'])
            data[name]=np.asarray(arrays)
        results={}
        for mode,label in enumerate(modes):
            cuts={}
            for k,(cut,_,_) in enumerate(m['constant_cuts']):
                agg={name:(w@v[:,mode,:,k].reshape(n,-1)).reshape(len(w),2,5,5) for name,v in data.items()}
                bm,bd,bf,bu,bn,br,bc=parts(agg['baseline'])
                arms={}
                for name,x in agg.items():
                    mass,denom,f,u,num,response,contrib=parts(x)
                    delta=response-br
                    numterm=diffleg(.5*(1/denom+1/bd)[...,None]*(num-bn))
                    denterm=diffleg(.5*(num+bn)*(1/denom-1/bd)[...,None])
                    conditional=diffleg(.5*(f+bf)[...,None]*(u-bu))
                    mixture=diffleg(.5*(u+bu)*(f-bf)[...,None])
                    np.testing.assert_allclose(numterm+denterm,delta,rtol=1e-9,atol=2e-12)
                    np.testing.assert_allclose((conditional+mixture).sum(1),delta,rtol=1e-9,atol=2e-12)
                    old=prior['axes'][str(axis)]['arms'][name][label][cut]
                    np.testing.assert_allclose(response[0,:2],old['model_flow_response']['value'],atol=1e-12)
                    np.testing.assert_allclose(response[0,2:],old['model_emulator_response']['value'],atol=1e-12)
                    arms[name]={'response':ent(response),'change':ent(delta),'change_size_contributions':ent(contrib-bc),
                                'numerator_term':ent(numterm),'denominator_term':ent(denterm),
                                'conditional_mean_term_by_size':ent(conditional),'selected_fraction_term_by_size':ent(mixture),
                                'conditional_mean_term':ent(conditional.sum(1)),'selected_fraction_term':ent(mixture.sum(1)),
                                'selected_mass_ratio_to_baseline_per_leg':ent(denom/bd),
                                'joint_flow_fraction_of_diagonal_shift':None if name=='baseline' else ent(delta[:,axis-1]/(delta[:,axis-1]+delta[:,axis+1]))}
                cuts[cut]=arms
            results[label]=cuts
        result['axes'][str(axis)]=results
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'constant_decomposition.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DONE',len(cases),'cases, all identities and interim replay passed',flush=True)

if __name__=='__main__':main()
