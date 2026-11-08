"""Completed full-budget conditional-density comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
def main():
 d=json.loads((ROOT/'alignment_factorized_20261105/comparison.json').read_text())
 arms=['retained','joint_frame','cartesian','frame']
 labels={'retained':'Retained GalSBI','joint_frame':'Joint density, intrinsic frame','cartesian':'Conditional shape, Cartesian','frame':'Conditional shape, intrinsic frame'}
 colors={'retained':'#888888','joint_frame':'#D55E00','cartesian':'#0072B2','frame':'#009E73'}
 fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 for a in arms[1:]:
  folder=ROOT/'alignment_frame_full_20261105/frame' if a=='joint_frame' else ROOT/'alignment_factorized_20261105'/a
  h=json.loads((folder/'history.json').read_text());axes[0].plot([x['epoch'] for x in h],[x['validation']['nll'] for x in h],color=colors[a],label=labels[a])
 axes[0].set(xlabel='Epoch',ylabel='Validation NLL (lower is better)',ylim=(-3.98,-3.2),title='Same data, schedule and training budget')
 centers=np.r_[1.95,np.arange(2.05,5.,.1),5.05]
 for m,ax in enumerate(axes[1:]):
  for a in arms:
   v=d['arms'][a]['full_small_profiles'];moms=np.asarray(v['alignment']);ct=np.asarray(v['counts']);good=(ct[0,m]>0)&(ct[1,m]>0)
   ax.plot(centers,np.where(good,moms[1,m]-moms[0,m],np.nan),color=colors[a],label=labels[a],lw=1.8)
  ax.axhline(0,color='black',lw=.7);ax.axvline(3,color='0.5',ls='--',lw=.8)
  ax.set(xlabel='Measured/generated radius [pixels]',ylabel='Alignment error: model − simulation',xlim=(2.4,3.6),title=f'Magnitude < {[25,25.8][m]}')
 axes[1].legend(frameon=False,fontsize=8)
 for ax in axes:ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('GalSBI: conditioning shape on radius and flux recovers the alignment transition\nIntrinsic circularized Re 0.15–0.25 arcsec; held-out flow data, not joint constant-shear calibration',fontsize=12)
 for ext in ['png','pdf']:fig.savefig(HERE/f'comparison.{ext}',dpi=170)
 plt.close(fig)
if __name__=='__main__':main()
