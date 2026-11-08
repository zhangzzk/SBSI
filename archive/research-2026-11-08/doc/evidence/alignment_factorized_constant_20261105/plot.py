"""Plot completed paired constant-shear comparison; run on a CPU allocation."""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert os.environ.get('SLURM_JOB_ID')
here=Path(__file__).resolve().parent
root=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_factorized_constant_20261105')
r=json.loads((root/'full_report.json').read_text())
cuts=['uncut','mag25','mag25.8','mag25_radius_gt0.60','mag25.8_radius_gt0.60','mag25.8_radius_ge0.75']
labels=['No photometric cut','mag < 25','mag < 25.8','radius > 3, mag < 25','radius > 3, mag < 25.8','radius ≥ 3.75, mag < 25.8']
fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharey=True,layout='constrained')
for ax,mode,title in zip(axes,['predicted_usability_generated_selection','true_usability_generated_selection'],['Predicted usability','Measured usability']):
 for j,(arm,label,color) in enumerate([('retained','Retained GalSBI','#666666'),('cartesian','Factorized Cartesian','#0072B2'),('frame','Factorized intrinsic frame','#D55E00')]):
  vals=[r['arms'][arm][mode][c]['m_percent'] for c in cuts]
  ax.errorbar([v['value'] for v in vals],[i+(j-1)*.18 for i in range(len(cuts))],xerr=[v['case_se'] for v in vals],fmt='o',ms=5,capsize=3,color=color,label=label)
 ax.axvline(0,color='black',lw=1);ax.grid(axis='x',alpha=.2)
 ax.set_title(title+'; generated selection');ax.set_xlabel('Multiplicative bias m (%)');ax.set_xlim(-1.8,3.8)
axes[0].set_yticks(range(len(cuts)),labels);axes[0].invert_yaxis();fig.legend(*axes[1].get_legend_handles_labels(),loc='outside lower center',ncols=3,fontsize=10)
fig.suptitle('Same 40 constant-shear scenes • fixed emulator and classifier\n1σ case-bootstrap errors; one training seed; radius in pixels (0.2 arcsec/pixel)',fontsize=12)
for ext in ['png','pdf']:fig.savefig(here/f'comparison.{ext}',dpi=180)
print('SAVED',here/'comparison.png')
