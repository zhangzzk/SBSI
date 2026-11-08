"""Plot only completed, fixed-grid image interventions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
p=json.loads((HERE/'images_report.json').read_text())
radii=[.15,.2,.25]
fig,axes=plt.subplots(1,3,figsize=(11.7,4.3),sharey=True)
colors=['#0072B2','#D55E00']
for j,(ax,title) in enumerate(zip(axes,['No added noise','Fixed r-like noise: σ = 0.3115','Fixed i-like noise: σ = 0.6159'])):
    for k,name in enumerate(['FS2 PSF','GalSBI PSF']):
        a=[p['by_radius'][str(r)][3*k+j]['alignment'] for r in radii]
        ax.plot(radii,a,marker=['o','s'][k],color=colors[k],label=name,lw=2)
    ax.axhline(0,color='0.5',lw=.8,ls='--');ax.set_xticks(radii)
    ax.set_title(title,fontsize=11);ax.set_xlabel('Intrinsic circularized Re [arcsec]')
    ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Measured–intrinsic alignment A')
axes[0].legend(frameon=False,fontsize=10)
fig.suptitle('Identical galaxies: PSF changes move the fitted-shape transition',fontsize=14,y=.98)
fig.text(.5,.02,'Fixed grid; four paired seeds; isolated objects; no detection or measured-radius cuts. Not a shear-bias estimate.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.92))
for ext in ['png','pdf']:fig.savefig(HERE/f'identical_galaxies.{ext}',dpi=180)
plt.close(fig)
