"""Plot preserved training/validation profile summaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
report=json.loads((root/'report.json').read_text())
centres=2.05+0.1*np.arange(30)
for mi,mag in enumerate([25,25.8]):
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,layout='constrained')
    for fi,family in enumerate(['fs2','galsbi']):
        for si,split in enumerate(['train','validation']):
            profile=report['families'][family][split]['groups']['all']['point']
            ax=axes[fi,si]
            for b,name,color,style in [(0,'Simulation','#252525','-'),(1,'NLL-only flow','#8e44ad','--'),(2,'Adopted flow','#d35400','-')]:
                ax.plot(centres,np.asarray(profile['alignment'][b][mi][1:-1],float),label=name,color=color,linestyle=style,lw=1.8)
            ax.axvline(3,color='#4783a7',ls=':',lw=1)
            ax.axhline(0,color='.75',lw=.7)
            ax.set(xlim=(2.4,3.8),ylim=(-.9,.9),title=f'{family.upper()}: {split}',xlabel='Measured FLUX_RADIUS [pixels]',ylabel='Measured–intrinsic alignment')
            if fi==0 and si==0:ax.legend(frameon=False,fontsize=9)
    fig.suptitle(f'MAG_AUTO < {mag:g}; intrinsic circularized Re = 0.15–0.25 arcsec\n40 training cases and 40 validation cases; frozen checkpoints',fontsize=12)
    tag=str(mag).replace('.','p')
    for suffix in ['png','pdf']:fig.savefig(root/f'train_validation_mag{tag}.{suffix}',dpi=180)
    plt.close(fig)
