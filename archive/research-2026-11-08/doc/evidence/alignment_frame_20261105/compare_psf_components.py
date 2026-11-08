"""Compare only identical successful fits, while reporting all PSF failures."""
import json
from pathlib import Path
import numpy as np
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_frame_20261105')
HERE=Path(__file__).resolve().parent

def load(name):
    z=[np.load(ROOT/name/f'grid{i}.npz') for i in range(4)]
    o=np.argsort(np.concatenate([a['ids'] for a in z]))
    return np.concatenate([a['parameters'] for a in z])[o],np.concatenate([a['fits'] for a in z])[o]

def main():
    x,a=load('images');xx,b=load('images_moffat3g');np.testing.assert_array_equal(x,xx)
    e=x[:,2,None]*np.stack([np.cos(np.deg2rad(2*x[:,3])),np.sin(np.deg2rad(2*x[:,3]))],-1)
    power=np.broadcast_to((e**2).sum(-1)[:,None],(216,4));results=[]
    for psf in range(2):
        for noise in range(3):
            va=a[:,psf,noise];vb=b[:,psf,noise]
            ok=np.ones((216,4),bool)
            for v in [va,vb]:ok&=(v[...,3]==0)&np.isfinite(v[...,:3]).all(-1)&((v[...,:2]**2).sum(-1)<1)
            d1=(va[...,:2]*e[:,None]).sum(-1);d3=(vb[...,:2]*e[:,None]).sum(-1)
            results.append(dict(psf=psf,noise=noise,paired_count=int(ok.sum()),
                one_gauss_negative_T=float(np.mean(va[...,2][ok]<0)),three_gauss_negative_T=float(np.mean(vb[...,2][ok]<0)),
                one_gauss_alignment=float(d1[ok].sum()/power[ok].sum()),three_gauss_alignment=float(d3[ok].sum()/power[ok].sum())))
    errors={}
    for i in range(4):
        for k,v in json.loads((ROOT/'images_moffat3g'/f'grid{i}.json').read_text())['errors'].items():errors[k]=errors.get(k,0)+v
    (HERE/'paired_psf_components.json').write_text(json.dumps({'matched_only':results,'errors':errors},indent=2)+'\n')
if __name__=='__main__':main()
