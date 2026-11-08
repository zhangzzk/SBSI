"""Conditional fixed-image stamp comparison; never a calibration estimate."""
import json
import os
from pathlib import Path
import numpy as np

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/fs2_galsbi_transfer_20261105')
out={'definition':'Paired 48-subpixel versus centred65 fits on the same pixels/detections. True Re [0.15,0.25), mag<25, measured radius [2.4,3.6). Two seeds, common across stamps. Conditional object bootstrap within each radius bin; not population/case uncertainty and not a shear-response result.',
     'cases':{},'families':{},'case_profiles':{}}
all_data={}
for f,cases in [('fs2',[40,80]),('galsbi',[1000,1080])]:
    rows=[]
    for case in cases:
        with np.load(O/'stamps'/f'{f}_case{case}.npz') as z:
            d={k:z[k] for k in ['fits','radius_bin','intrinsic','original']}
        rows.append(d)
        all_data[f'{f}_case{case}']=d
        v=d['fits'];valid=np.isfinite(v[...,:2]).all(-1)&((v[...,:2]**2).sum(-1)<1)
        out['cases'][str(case)]={'family':f,'sample':len(v),'valid_fraction':valid.mean(0).tolist(),
             'negative_T_fraction':[float((v[:,s,:,2]<0)[valid[:,s]].mean()) for s in [0,1]]}
    all_data[f]={k:np.concatenate([d[k] for d in rows]) for k in rows[0]}
for j,(f,d) in enumerate(all_data.items()):
    v=d['fits'];intr=d['intrinsic'];power=(intr**2).sum(-1)
    valid=np.isfinite(v[...,:2]).all(-1)&((v[...,:2]**2).sum(-1)<1)
    common=valid.all((1,2));dot=(v[...,:2]*intr[:,None,None,:]).sum(-1).mean(-1)
    original_u=np.isfinite(d['original']).all(1)&((d['original']**2).sum(1)<1)
    orig_dot=(d['original']*intr).sum(-1)
    bins={};rng=np.random.default_rng(20261105099+j)
    for b in range(12):
        ix=np.flatnonzero((d['radius_bin']==b)&common)
        if not len(ix):continue
        # Independent object resamples are conditional on these two fixed images.
        w=np.vstack([np.ones(len(ix)),rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),2000)])
        a=(w@dot[ix])/(w@power[ix])[:,None];delta=a[:,1]-a[:,0]
        old=(d['radius_bin']==b)&original_u
        bins[str(b)]={'radius_low':round(2.4+.1*b,2),'common_objects':len(ix),
            'alignment_48':float(a[0,0]),'alignment_65':float(a[0,1]),
            'alignment_change_65_minus_48':float(delta[0]),'conditional_object_se':float(delta[1:].std(ddof=1)),
            'original_alignment':float(orig_dot[old].sum()/power[old].sum()),
            'negative_T_fraction':[float((v[ix,s,:,2]<0).mean()) for s in [0,1]],
            'median_inferred_noise_sigma':[float(np.median(1/np.sqrt(v[ix,s,:,5]))) for s in [0,1]]}
        # Exploratory split after the stamp comparison: these contributions sum
        # to the full alignment; T is never used to exclude a fit.
        branch=[]
        for s in [0,1]:
            dots=(v[ix,s,:,:2]*intr[ix,None,:]).sum(-1)
            neg=v[ix,s,:,2]<0
            den=2*power[ix].sum()
            terms=[float(dots[neg].sum()/den),float(dots[~neg].sum()/den)]
            np.testing.assert_allclose(sum(terms),a[0,s],atol=1e-12)
            branch.append({'negative_T_contribution':terms[0],
                           'nonnegative_T_contribution':terms[1]})
        bins[str(b)]['T_branch_alignment_contributions']=branch
    out['case_profiles' if '_case' in f else 'families'][f]=bins
(O/'stamp_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for f,bins in out['families'].items():
    print(f)
    for b,d in bins.items():print(d,flush=True)
