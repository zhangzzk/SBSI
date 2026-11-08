"""Plots of completed native or final instrumented training runs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_trajectory_20261105')

def main(phase):
    d=json.loads((ROOT/f'{phase}_report.json').read_text())['trajectories']
    arms=['fs2','galsbi']+(['galsbi_fs2rate'] if phase=='final' else [])
    colors={'fs2':'#0072B2','galsbi':'#D55E00','galsbi_fs2rate':'#009E73'}
    labels={'fs2':'FS2 native replay','galsbi':'GalSBI native replay','galsbi_fs2rate':'GalSBI with FS2 rate schedule'}
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.4))
    for arm in arms:
        h=json.loads((ROOT/arm/'history.json').read_text());epochs=[x['epoch'] for x in h]
        ref=d[arm]['completion']['reference_nll']
        axes[0,0].plot(epochs,[x['validation']['nll']-ref for x in h],label=labels[arm],color=colors[arm])
        axes[1,0].step(epochs,[x['lr_used'] for x in h],where='post',color=colors[arm])
        tr=d[arm]['trajectory'];ep=[x['epoch'] for x in tr]
        for m in [0,1]:
            for row,split in enumerate(['train','validation']):
                y=[x[split]['alignment_error'][m][8] for x in tr]
                axes[row,m+1].plot(ep,y,color=colors[arm],marker='.',lw=1.7)
                base=d[arm]['reference'][split]['alignment_error'][m][8]
                axes[row,m+1].axhline(base,color=colors[arm],alpha=.4,ls=':',lw=1)
    axes[0,0].set_title('Validation NLL relative to own historical NLL model')
    axes[0,0].set_yscale('symlog',linthresh=.02);axes[0,0].set_ylabel('NLL difference [nats/object]')
    axes[0,0].legend(frameon=False,fontsize=8)
    axes[1,0].set_title('Learning rate consumed by each epoch');axes[1,0].set_yscale('log');axes[1,0].set_ylabel('Learning rate')
    for m in [0,1]:
        for row,split in enumerate(['Training','Validation']):
            axes[row,m+1].set_title(f'{split}: magnitude < {[25,25.8][m]}')
            axes[row,m+1].set_ylabel('Alignment error: flow − measurement')
            axes[row,m+1].axhline(0,color='0.4',lw=.7)
    for ax in axes.flat:
        ax.set_xlabel('Epoch');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Learning the size–shape relation with the original measurement pipelines',fontsize=14)
    fig.text(.5,.015,'Fixed 160k input-selected samples; intrinsic Re 0.15–0.25 arcsec, measured/generated radius 2.7–2.8px. Dotted lines: historical NLL model.\nNative labels differ between families. Trace points are development diagnostics, not independent shear-calibration results.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.95))
    for ext in ['png','pdf']:fig.savefig(HERE/f'{phase}_trajectories.{ext}',dpi=170)
    plt.close(fig)
