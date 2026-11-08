"""Same-image PSF-component intervention on the common successful subset."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'paired_psf_components.json').read_text())['matched_only']
fig,axes=plt.subplots(1,2,figsize=(9.2,4.3))
for psf,color,name in [(0,'#0072B2','FS2 PSF (719 pairs)'),(1,'#D55E00','GalSBI PSF (703 pairs)')]:
    d=next(x for x in r if x['psf']==psf and x['noise']==0)
    axes[0].plot([0,1],[d['one_gauss_alignment'],d['three_gauss_alignment']],marker='o',color=color,label=name,lw=2)
    axes[1].plot([0,1],[100*d['one_gauss_negative_T'],100*d['three_gauss_negative_T']],marker='o',color=color,lw=2)
for ax in axes:
    ax.set_xticks([0,1],['1 Gaussian','3 Gaussians']);ax.set_xlabel('Components in fitted PSF model')
    ax.axhline(0,color='0.5',ls='--',lw=.8);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Measured–intrinsic alignment A');axes[0].legend(frameon=False,fontsize=9)
axes[1].set_ylabel('Fits with negative deconvolved T [%]')
fig.suptitle('Same Moffat images: changing the PSF approximation removes the sign pathology',fontsize=12)
fig.text(.5,.02,'No added noise; same successful rows. Three-component PSF fits fail in 17.7% of attempts.\nIsolated small-galaxy grid, no detection/radius cuts; not a calibrated replacement.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.93))
for ext in ['png','pdf']:fig.savefig(HERE/f'psf_components.{ext}',dpi=180)
plt.close(fig)
