"""Paired response curves and exact intrinsic-size contributions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_target9_20261106')
spec=importlib.util.spec_from_file_location('axis_reducer',HERE.parent/'alignment_axis2_constant_20261105/report.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','full']);args=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text())
    cases=manifest['cases'] if args.phase=='full' else manifest['cases'][:1]
    n=len(cases);weights=np.vstack([np.ones(n),np.random.default_rng(202611067101).multinomial(n,np.full(n,1/n),10000)])
    result={'cases':cases,'cuts':manifest['cuts'],'size_edges_arcsec':manifest['intrinsic_size_edges_arcsec'],
            'definition':'DeltaR=simulation-model; per-leg independent selection and denominator. '
            'Size contributions sum to full error; not bin-normalized.10000 whole-case bootstraps '
            'condition on fixed models and catalogues. Development only.','axes':{},'legacy_stream_comparison':{}}
    for axis in [1,2]:
        models=[];sims=[]
        for case in cases:
            with np.load(ROOT/f'axis{axis}/case{case}.npz') as z:
                np.testing.assert_array_equal(z['cuts'],[c[0] for c in manifest['cuts']])
                models.append(z['model']);sims.append(z['sim'])
        model,sim=np.array(models),np.array(sims)
        modes={};new_samples={}
        for mode,label in enumerate(r.MODES[:2]):
            rows={};target_samples=[]
            for k,(cut,_,_) in enumerate(manifest['cuts']):
                v=r.estimates(sim[:,:,k].sum(-2),model[:,mode,:,k].sum(-2),weights,axis=axis)
                new_samples[mode,cut]=v
                report={name:r.ent(val) for name,val in v.items()}
                s=np.einsum('bc,cliq->bliq',weights,sim[:,:,k])
                m=np.einsum('bc,cliq->bliq',weights,model[:,mode,:,k])
                ns=s[...,0].sum(-1);nm=m[...,0].sum(-1)
                sc=s[...,1:]/ns[...,None,None]
                mc=(m[...,1:3]+m[...,3:5])/nm[...,None,None]
                delta=(sc[:,0]-sc[:,1]-mc[:,0]+mc[:,1])/.04
                np.testing.assert_allclose(delta.sum(1),v['delta_response'],atol=1e-11,rtol=1e-11)
                report['intrinsic_size_delta_contributions']=r.ent(delta)
                report['simulation_selected_size_fraction_per_leg']=r.ent(s[...,0]/ns[...,None])
                report['model_selected_size_fraction_per_leg']=r.ent(m[...,0]/nm[...,None])
                rows[cut]=report
                if cut in manifest['target_cuts']:target_samples.append(v['m_percent'])
            target_samples=np.stack(target_samples,axis=-1)
            assert target_samples.shape[-1]==9
            rows['nine_cut_summary']={
                'rms_m_percent':r.ent(np.sqrt((target_samples**2).mean(-1))),
                'max_abs_m_percent':r.ent(np.abs(target_samples).max(-1)),
                'max_minus_min_m_percentage_points':r.ent(target_samples.max(-1)-target_samples.min(-1)),
                'note':'Overlapping cuts share paired case resamples; no nine-independent-cuts assumption.'}
            modes[label]=rows
        result['axes'][str(axis)]=modes
        old_directory=ROOT.parent/('alignment_factorized_constant_20261105' if axis==1 else 'alignment_axis2_constant_20261105')/'cartesian'
        overlap=['uncut','mag25','mag25.8','mag25_radius_gt0.60','mag25.8_radius_gt0.60','mag25.8_radius_ge0.75']
        old_models=[];old_sims=[]
        for case in cases:
            with np.load(old_directory/f'case{case}.npz') as z:
                ci=[list(z['cuts']).index(c) for c in overlap]
                old_models.append(z['model'][:2,:,ci,:,:5].sum(-2));old_sims.append(z['sim'][:,ci,:,:3].sum(-2))
        om,osim=np.asarray(old_models),np.asarray(old_sims);comparison={}
        for mode,label in enumerate(r.MODES[:2]):
            comparison[label]={}
            for k,cut in enumerate(overlap):
                previous=r.estimates(osim[:,:,k],om[:,mode,:,k],weights,axis=axis)
                comparison[label][cut]={key:r.ent(new_samples[mode,cut][key]-previous[key]) for key in ['m_percent','delta_response']}
        result['legacy_stream_comparison'][str(axis)]=comparison
    if n<2:
        def undefined_se(value):
            if isinstance(value,dict):
                return {k:(None if k=='case_se' else undefined_se(v)) for k,v in value.items()}
            if isinstance(value,list):return [undefined_se(v) for v in value]
            return value
        result=undefined_se(result)
    (ROOT/f'{args.phase}_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DONE',args.phase,'cases',n,flush=True)


if __name__=='__main__':main()
