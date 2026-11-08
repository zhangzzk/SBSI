"""Descriptive summaries of a fixed paired image grid, not population estimates."""
import csv
import os
import json
from pathlib import Path
import numpy as np
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_frame_20261105/images')
HERE=Path(__file__).resolve().parent
MODE=os.environ.get('SBSI_PSF_CONTROL','moffat')
if MODE!='moffat':ROOT=ROOT.parent/('images_'+MODE)
PREFIX='images' if MODE=='moffat' else MODE

def main():
    zs=[np.load(ROOT/f'grid{i}.npz') for i in range(4)]
    x=np.concatenate([z['parameters'] for z in zs]);v=np.concatenate([z['fits'] for z in zs]);ids=np.concatenate([z['ids'] for z in zs])
    order=np.argsort(ids);ids=ids[order];x=x[order];v=v[order]
    np.testing.assert_array_equal(ids,np.arange(216))
    angle=np.deg2rad(2*x[:,3]);e=x[:,2,None]*np.stack([np.cos(angle),np.sin(angle)],-1)
    dot=(v[...,:2]*e[:,None,None,None,:]).sum(-1)
    power=np.broadcast_to((e**2).sum(-1)[:,None,None,None],dot.shape)
    valid=(v[...,3]==0)&np.isfinite(v[...,:3]).all(-1)&((v[...,:2]**2).sum(-1)<1)
    def summary(mask):
        answer=[]
        for psf in range(2):
            for noise in range(3):
                ok=valid[mask,psf,noise];d=dot[mask,psf,noise];t=v[mask,psf,noise,:,2];pw=power[mask,psf,noise]
                answer.append({'psf':(['fs2_r','galsbi_i'] if MODE!='gaussian' else ['gaussian_fs2_T','gaussian_galsbi_T'])[psf],'noise_sigma':[0.,.3115,.6159][noise],
                   'attempted':int(ok.size),'usable':int(ok.sum()),'alignment':float(d[ok].sum()/pw[ok].sum()),
                   'negative_T_fraction':float(np.mean(t[ok]<0)),'negative_dot_fraction':float(np.mean(d[ok]<0))})
        return answer
    def paired(p0,n0,p1,n1):
        ok=valid[:,p0,n0]&valid[:,p1,n1]
        first=dot[:,p0,n0];second=dot[:,p1,n1]
        return {'paired_usable':int(ok.sum()),'alignment_change':float((second-first)[ok].sum()/power[:,p0,n0][ok].sum()),
                'alignment_sign_flips':int(((first[ok]<0)!=(second[ok]<0)).sum()),
                'T_sign_flips':int(((v[:,p0,n0,:,2][ok]<0)!=(v[:,p1,n1,:,2][ok]<0)).sum())}
    result={'psf_family':MODE,'definition':'A=sum(e_measured dot e_intrinsic)/sum(|e_intrinsic|^2); equally weighted fixed contexts and four paired seeds. No inferred population weighting or sampling-error bars.',
       'limitations':'No added noise means sigma=0, while fitter weights still come from border MAD of the finite stamp. Isolated centred65 galaxies, no detection/FLUX_RADIUS cuts, no applied shear. This is not a calibration-bias estimate.',
       'all':summary(np.ones(len(x),bool)),'by_radius':{str(r):summary(x[:,0]==r) for r in [.15,.2,.25]},
       'paired':{'PSF_only_no_added_noise':paired(0,0,1,0),'PSF_only_r_noise':paired(0,1,1,1),'noise_only_r_PSF':paired(0,1,0,2),'noise_only_i_PSF':paired(1,1,1,2),'both_r_to_i':paired(0,1,1,2)},
       'receipts':[json.loads((ROOT/f'grid{i}.json').read_text()) for i in range(4)]}
    (HERE/f'{PREFIX}_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (HERE/f'{PREFIX}_contexts.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['Rc_arcsec','n','intrinsic_g','angle_deg','mag','psf','noise_sigma','alignment','negative_T_fraction','negative_dot_fraction','usable'])
        for j,row in enumerate(x):
            for s in summary(np.arange(len(x))==j):writer.writerow([*row,s['psf'],s['noise_sigma'],s['alignment'],s['negative_T_fraction'],s['negative_dot_fraction'],s['usable']])
    print(json.dumps(result['paired'],indent=2))
if __name__=='__main__':main()
