"""Completed candidate results only; no interpolation of pending science."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_frame_20261105')

def main():
    d=json.loads((ROOT/'comparison.json').read_text())
    colors={'retained':'#777777','cartesian':'#0072B2','frame':'#D55E00'}
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    for arm in ['cartesian','frame']:
        h=json.loads((ROOT/arm/'history.json').read_text())
        axes[0].plot([x['epoch'] for x in h],[x['validation']['nll'] for x in h],color=colors[arm],label=arm)
    old=d['arms']['cartesian']['validation_nll']
    axes[0].axhline(old['retained']['nll'],color=colors['retained'],ls='--',label='retained')
    axes[0].set(xlabel='Epoch',ylabel='Validation NLL (lower is better)',title='Matched fresh fits')
    axes[0].set_ylim(-3.85,-2.1);axes[0].legend(frameon=False)
    centres=np.r_[1.95,np.arange(2.05,5.,.1),5.05]
    for mag,ax in enumerate(axes[1:]):
        for arm in ['retained','cartesian','frame']:
            prof=d['arms'][arm]['full_small_profiles'];a=np.asarray(prof['alignment']);count=np.asarray(prof['counts'])
            y=a[1,mag]-a[0,mag];keep=(count[0,mag]>0)&(count[1,mag]>0)
            ax.plot(centres,np.where(keep,y,np.nan),color=colors[arm],label=arm)
        ax.axhline(0,color='0.5',lw=.8);ax.axvline(3,color='0.5',ls='--',lw=.8)
        ax.set(xlim=(2.4,3.6),xlabel='Measured/generated radius [pixels]',ylabel='Alignment error: model − simulation',title=f'Magnitude < {[25,25.8][mag]}')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.suptitle('GalSBI flow representation test: held-out cases, intrinsic Re 0.15–0.25 arcsec',fontsize=13)
    fig.text(.5,.015,'Development comparison; different training budget from retained model. Profiles are point estimates; case SEs are in comparison.json.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.93))
    for ext in ['png','pdf']:fig.savefig(HERE/f'flow_comparison.{ext}',dpi=180)
    plt.close(fig)
if __name__=='__main__':main()
