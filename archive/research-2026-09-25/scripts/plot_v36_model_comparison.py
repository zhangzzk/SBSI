#!/usr/bin/env python
"""Plot verified saved-model comparisons; no model or selection changes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.root/'all_saved_models_v2.json').read_text())
    names = ['mag265_full_parent','unbounded_rows','fluxsize_ep08','guard8_ep02','closecrowd','closecrowd_guard_ep03']
    labels = ['Magnitude<26.5 training','Unbounded, 8 inputs','Flux/size refinement','8-input magnitude guard',
              'Close crowding, 11 inputs','Close crowding + guard']
    values = [summary['models'][n]['results']['mag25.8_radius_gt0.60/modeled_smooth'] for n in names]
    fig, axes = plt.subplots(1,2,figsize=(12,4.8), gridspec_kw={'width_ratios':[1.2,1]})
    ax = axes[0]
    ax.axvspan(-.3,.3,color='#d8eedf',label='Target band')
    ax.axvline(0,color='#777777',lw=.8)
    for i,v in enumerate(values):
        ax.errorbar(v['m_percent'], i, xerr=v['case_se_pp'], fmt='o',capsize=3,
                    color='#137a55' if names[i]=='closecrowd' else '#425b76')
        ax.annotate(f"{v['m_percent']:+.3f}%",(v['m_percent'],i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.set_yticks(range(len(names)),labels)
    ax.invert_yaxis()
    ax.set_xlim(-.8,5.4)
    ax.set_ylim(5.7,-.7)
    ax.set_xlabel('Normalized multiplicative bias (%)')
    ax.set_title('Same 80 scenes and measured cuts',fontsize=12)
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    ax = axes[1]
    x = np.arange(4)
    for shift,name,label,color in [(-.18,'closecrowd','Close crowding','#137a55'),(.18,'closecrowd_guard_ep03','After magnitude guard','#bc6334')]:
        d = json.loads((args.root/f'rows_{name}.json').read_text())['results']['modeled_smooth']
        ax.bar(x+shift,d['bin_contributions_pp'],width=.34,color=color,label=label,
               yerr=d['bin_case_se_pp'],capsize=3)
    ax.axhline(0,color='#777777',lw=.8)
    ax.set_xticks(x,['<24','24–25.5','25.5–27.5','≥27.5'])
    ax.set_xlabel('True magnitude — diagnostic bins; all retained')
    ax.set_ylabel('Contribution to total bias (percentage points)')
    ax.set_title('A small total still hides cancellation',fontsize=12)
    ax.legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('V3.6-related candidates: corrected evaluation',fontsize=15,y=1.01)
    fig.text(.5,-.025,'Radius >0.6 arcsec; MAG_AUTO <25.8; independent per-leg selection and normalization.\n'
             'Generated target-role parent with no truth-magnitude cut. Error bars: 1 scene-bootstrap SE; development evidence.',ha='center',fontsize=9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):
        path=args.output.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
