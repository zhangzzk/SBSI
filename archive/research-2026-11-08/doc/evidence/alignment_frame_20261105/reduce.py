"""Paired whole-case evaluation of experimental flows versus the retained flow."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_frame_20261105')
spec=importlib.util.spec_from_file_location('response_reducer',HERE.parent/'fs2_galsbi_transfer_20261105/reduce_flow.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    result={'definition':'Simulation minus model response; model minus simulation alignment. 10000 paired whole-case bootstrap resamples; fixed fitted models, no training uncertainty. Developmental standalone data, not constant-shear joint acceptance.','arms':{}}
    reference=None;reference_cases=None;profile_errors={}
    for arm in ['retained','cartesian','frame']:
        directory=ROOT.parent/'fs2_galsbi_transfer_20261105/galsbi' if arm=='retained' else ROOT/(arm+'_eval')
        assert (directory/'completed.json').exists()
        protocol=json.loads((directory/'protocol.json').read_text());cases=protocol['cases']
        data={k:[] for k in ['base','cross','normal','count','profile']}
        for case in cases:
            with np.load(directory/f'case{case:03d}.npz') as z:
                for k in data:data[k].append(z[k])
        data={k:np.asarray(v) for k,v in data.items()}
        if reference is None:
            reference={k:v.copy() for k,v in data.items()};reference_cases=cases
        else:
            assert cases==reference_cases
            for k in ['base','cross','profile']:np.testing.assert_allclose(data[k][:,0],reference[k][:,0],rtol=1e-12,atol=1e-8)
            for k in ['normal','count']:np.testing.assert_allclose(data[k],reference[k],rtol=1e-12,atol=1e-8)
        for k in ['base','cross','profile']:data[k]=np.concatenate([data[k][:,:1],data[k][:,1:].mean(1,keepdims=True),data[k][:,1:]],axis=1)
        data['cuts']=protocol['cuts'];n=len(cases)
        w=np.vstack([np.ones(n),np.random.default_rng(202611053444).multinomial(n,np.full(n,1/n),10000)])
        response=r.reduce_family(data,w,np.ones(25))
        # Small intrinsic galaxies [0.15,0.25), measured 2.7--2.8px and 3.0--3.1px.
        p=data['profile'].reshape(n,4,2,5,32,7)[:,:,:,1]
        profiles={}
        for b in [8,11]:
            moments=np.einsum('nc,cbmk->nbmk',w,p[:,:,:,b])
            alignment=moments[...,2]/moments[...,1]
            delta=alignment[:,1]-alignment[:,0]
            profile_errors[(arm,b)]=delta
            profiles[str(b)]={'simulation':r.ent(alignment[:,0]),'model':r.ent(alignment[:,1]),'error':r.ent(delta),
                              'simulation_count':moments[0,0,:,0].tolist(),'model_count':moments[0,1,:,0].tolist()}
        full=p.sum(0)
        full_alignment=np.divide(full[...,2],full[...,1],out=np.zeros_like(full[...,2]),where=full[...,1]>0)
        result['arms'][arm]={'responses':response,'profiles':profiles,'cases':cases,
           'full_small_profiles':{'counts':full[...,0].tolist(),'alignment':full_alignment.tolist()}}
        if arm!='retained':
            result['arms'][arm]['training']=json.loads((ROOT/arm/'completed.json').read_text())
            result['arms'][arm]['validation_nll']=json.loads((directory/'validation_nll.json').read_text())
            result['arms'][arm]['measured_replay_max_abs']={k:float(np.max(np.abs(data[k][:,0]-reference[k][:,0]))) for k in ['base','cross','profile']}
    result['paired_alignment_changes']={f'{a}_minus_{b}_bin{k}':r.ent(profile_errors[(a,k)]-profile_errors[(b,k)])
          for a,b in [('frame','cartesian'),('frame','retained'),('cartesian','retained')] for k in [8,11]}
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for a,v in result['arms'].items():
        print(a,'alignment errors',v['profiles']['8']['error'],flush=True)
        for cut,c in v['responses']['own'].items():print(cut,'delta R11',c['delta_matrix']['value'][0][0],'+/-',c['delta_matrix']['case_se'][0][0],flush=True)
    plot_spec=importlib.util.spec_from_file_location('candidate_plot',HERE/'plot_comparison.py')
    plotting=importlib.util.module_from_spec(plot_spec);plot_spec.loader.exec_module(plotting);plotting.main()
    print('DONE',flush=True)
if __name__=='__main__':main()
