"""Case uncertainty and a plot for the pairing-free intrinsic-shape alignment."""
import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/fs2_galsbi_transfer_20261105')
report={'definition':'alignment = sum(e_measured dot e_intrinsic) / sum(|e_intrinsic|^2); baseline g=0, each side bins its own measured radius. True circularized Re in [0.15,0.25) arcsec. No simulated/generated draw pairing.',
        'families':{}}
fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True,layout='constrained')
edges=np.arange(2,5.001,.1);centers=(edges[:-1]+edges[1:])/2
for j,family in enumerate(['fs2','galsbi']):
    prot=json.loads((O/family/'protocol.json').read_text())
    x=[]
    for case in prot['cases']:
        with np.load(O/family/f'case{case:03d}.npz') as z:
            a=z['profile'].reshape(3,2,5,32,7)[:,:,1]
            x.append(np.stack([a[0],a[1:].mean(0)]))
    x=np.array(x);n=len(x)
    w=np.vstack([np.ones(n),np.random.default_rng(20261105090+j).multinomial(n,np.full(n,1/n),10000)])
    align=[];diff=[]
    for lo in range(0,len(w),250):
        sums=np.einsum('nc,c...->n...',w[lo:lo+250],x)
        with np.errstate(invalid='ignore',divide='ignore'):
            a=sums[...,2]/sums[...,1]
        align.append(a)
    align=np.concatenate(align);total=x.sum(0)
    out={}
    for m,mag in enumerate([25.,25.8]):
        delta=align[:,1,m]-align[:,0,m]
        out[str(mag)]={'alignment':np.nan_to_num(align[0,:,m]).tolist(),
                      'case_se':np.nan_to_num(np.nanstd(align[1:,:,m],axis=0,ddof=1)).tolist(),
                      'model_minus_sim':np.nan_to_num(delta[0]).tolist(),
                      'difference_case_se':np.nan_to_num(np.nanstd(delta[1:],axis=0,ddof=1)).tolist(),
                      'counts':total[:,m,:,0].tolist()}
        ax=axs[m,j]
        for b,label,color in [(0,'Simulation','#252525'),(1,'Flow','#d35400')]:
            a=align[0,b,m,1:-1];se=np.nanstd(align[1:,b,m,1:-1],axis=0,ddof=1)
            ax.plot(centers,a,label=label,color=color,lw=1.8)
            ax.fill_between(centers,a-se,a+se,color=color,alpha=.2)
        ax.axvline(3,color='#4783a7',ls='--',lw=1)
        ax.axhline(0,color='.75',lw=.7)
        ax.set_xlim(2.4,3.8);ax.set_ylim(-.9,.9)
        ax.set_title(f'{family.upper()}: measured magnitude < {mag:g}')
        ax.set_ylabel('Measured–intrinsic alignment')
        ax.set_xlabel('Measured FLUX_RADIUS [pixels]')
        if m==0 and j==0:ax.legend(frameon=False)
    report['families'][family]=out
fig.suptitle('Same true-size range, different shape–size relation\nRe = 0.15–0.25 arcsec; shaded bands: case-bootstrap SE',fontsize=12)
fig.savefig(O/'shape_size_transfer.png',dpi=180)
(O/'profile_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('Saved profile_comparison.json and shape_size_transfer.png')
