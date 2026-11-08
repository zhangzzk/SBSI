"""Matched-case full-budget representation comparison; no checkpoint selection."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
OUT=ROOT/'alignment_frame_full_20261105'
spec=importlib.util.spec_from_file_location('response_reducer',HERE.parent/'fs2_galsbi_transfer_20261105/reduce_flow.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    paths={'retained':ROOT/'fs2_galsbi_transfer_20261105/galsbi',
           'native':ROOT/'alignment_trajectory_20261105/galsbi_eval',
           'cartesian':ROOT/'alignment_trajectory_20261105/galsbi_fs2rate_eval',
           'frame':OUT/'frame_eval'}
    result={'definition':'Alignment=model-simulation; response=simulation-model. Paired whole-case SEs condition on fitted models; development only, not joint constant-shear m.', 'arms':{}}
    reference=None;errors={};matrix_errors={}
    for arm,directory in paths.items():
        assert (directory/'completed.json').exists()
        p=json.loads((directory/'protocol.json').read_text());cases=p['cases']
        data={k:[] for k in ['base','cross','normal','count','profile']}
        for case in cases:
            with np.load(directory/f'case{case:03d}.npz') as z:
                for k in data:data[k].append(z[k])
        data={k:np.asarray(v) for k,v in data.items()}
        if reference is None:reference=data;reference_cases=cases
        else:
            assert cases==reference_cases
            for k in ['base','cross','profile']:np.testing.assert_allclose(data[k][:,0],reference[k][:,0],rtol=1e-12,atol=1e-8)
            for k in ['normal','count']:np.testing.assert_allclose(data[k],reference[k],rtol=1e-12,atol=1e-8)
        # Rebinding prevents modification of the saved three-branch reference.
        data=dict(data)
        for k in ['base','cross','profile']:data[k]=np.concatenate([data[k][:,:1],data[k][:,1:].mean(1,keepdims=True),data[k][:,1:]],axis=1)
        data['cuts']=p['cuts'];nc=len(cases)
        w=np.vstack([np.ones(nc),np.random.default_rng(202611056100).multinomial(nc,np.full(nc,1/nc),10000)])
        response=r.reduce_family(data,w,np.ones(25))
        prof=data['profile'].reshape(nc,4,2,5,32,7)[:,:,:,1]
        profiles={}
        for b in [8,11]:
            moments=np.einsum('sc,cbmk->sbmk',w,prof[:,:,:,b])
            a=moments[...,2]/moments[...,1];errors[arm,b]=a[:,1]-a[:,0]
            profiles[str(b)]={'error':r.ent(errors[arm,b]),'simulation':r.ent(a[:,0]),'model':r.ent(a[:,1]),'counts':moments[0,:2,:,0].tolist()}
        full=prof.sum(0)
        alignment=np.divide(full[...,2],full[...,1],out=np.zeros_like(full[...,2]),where=full[...,1]>0)
        result['arms'][arm]={'responses':response,'profiles':profiles,
                            'full_small_profiles':{'counts':full[...,0].tolist(),'alignment':alignment.tolist()}}
        if (directory/'validation_nll.json').exists():result['arms'][arm]['validation_nll']=json.loads((directory/'validation_nll.json').read_text())
        # Retain resamples for paired full-matrix differences, including covariance.
        blocks=[]
        for lo in range(0,len(w),250):
            ww=w[lo:lo+250]
            b=np.einsum('sc,cb...->sb...',ww,data['base'][:,:2])
            c=np.einsum('sc,cb...->sb...',ww,data['cross'][:,:2])
            n=np.einsum('sc,c...->s...',ww,data['normal']);ct=ww@data['count']
            v=r.responses(b,c,n,ct).sum(2);blocks.append(v[:,0]-v[:,1])
        matrix_errors[arm]=np.concatenate(blocks)
        for j,cut in enumerate(p['cuts']):np.testing.assert_allclose(matrix_errors[arm][0,j],response['own'][cut]['delta_matrix']['value'],atol=1e-12,rtol=1e-10)
    result['paired_alignment_changes']={f'frame_minus_{arm}_bin{b}':r.ent(errors['frame',b]-errors[arm,b]) for arm in ['native','cartesian','retained'] for b in [8,11]}
    result['paired_response_changes']={f'frame_minus_{arm}':{cut:r.ent((matrix_errors['frame']-matrix_errors[arm])[:,j]) for j,cut in enumerate(p['cuts'])} for arm in ['native','cartesian','retained']}
    result['training']=json.loads((OUT/'frame/completed.json').read_text())
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for a,v in result['arms'].items():
        print(a,'alignment',v['profiles']['8']['error'],'NLL',v.get('validation_nll'),flush=True)
        for cut in ['radius3_mag25','radius3_mag25.8','radius3.75_mag25.8']:print(cut,v['responses']['own'][cut]['delta_matrix'],flush=True)
    print('DONE',flush=True)

if __name__=='__main__':main()
