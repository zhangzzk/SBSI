"""Static figures from the completed audited report; no new model evaluation."""
from pathlib import Path
import json,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert os.environ.get('SLURM_JOB_ID')
H=Path(__file__).resolve().parent
r=json.loads((H/'report.json').read_text());v=r['splits']['validation'];groups=v['groups'];names=list(groups)
labels={'baseline':'Original NLL','nine_weak':'Original + response','neighbour_nine':'Neighbour + response','latest':'Latest'}
colors=['#0072B2','#E69F00','#009E73','#CC79A7']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
x=np.arange(1,6)
for ai,name in enumerate(names):
 for k in range(3):
  vals=[];ses=[]
  for q in x:
   g=groups[name][f'neighbour_score/q{q}']
   if k==2:z=g['joint_pass_error'];val=z['value'][4];se=z['jackknife_se'][4]
   else:
    # Individual marginal differences are point estimates; the stored joint
    # decomposition is different, so do not attach its error bars here.
    val=g['predicted_pass']['value'][4][k]-g['observed_pass']['value'][4][k];se=None
   vals.append(100*val);ses.append(None if se is None else 100*se)
  if k==2:axs[0,k].errorbar(x,vals,yerr=ses,marker='o',color=colors[ai],label=labels[name],capsize=2)
  else:axs[0,k].plot(x,vals,marker='o',color=colors[ai],label=labels[name])
for k,title in enumerate(['Radius > 3 pixels','Magnitude < 25','Joint cut']):
 axs[0,k].axhline(0,color='0.4',lw=.8);axs[0,k].set(title=title,xlabel='Neighbour-score quintile (low → high)',ylabel='Predicted − measured passing fraction (pp)',xticks=x)
axs[0,0].legend(fontsize=8)
xx=np.arange(len(names));width=.23
for k,label in enumerate(['Radius marginal','Flux marginal','Association']):
 yy=[100*groups[name]['all']['joint_pass_decomposition']['value'][4][k] for name in names]
 ee=[100*groups[name]['all']['joint_pass_decomposition']['jackknife_se'][4][k] for name in names]
 axs[1,0].bar(xx+(k-1)*width,yy,width,yerr=ee,capsize=2,label=label)
axs[1,0].axhline(0,color='0.4',lw=.8)
axs[1,0].set(title='Joint-pass error decomposition',ylabel='Contribution (pp)',xticks=xx,xticklabels=['NLL','+ response','Neighbour\n+ response','Latest'])
axs[1,0].legend(fontsize=8)
mats=[]
for name in ['baseline','latest']:
 cells=v['central_score_error_cell_contributions'][name]
 mats.append(np.array([[cells[f'true_radius_x_score/q{i+1}_q{j+1}']['value'] for j in range(5)] for i in range(5)])*1000)
lim=max(abs(a).max() for a in mats)
for ax,mat,name in zip(axs[1,1:],mats,['baseline','latest']):
 im=ax.imshow(mat,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim)
 ax.set(title=labels[name]+': selected-neighbour error',xlabel='Neighbour-score quintile',ylabel='Intrinsic-radius quintile',xticks=range(5),yticks=range(5),xticklabels=x,yticklabels=x)
 for i in range(5):
  for j in range(5):ax.text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(mat[i,j])>.65*lim else 'black')
fig.colorbar(im,ax=list(axs[1,1:]),label='Contribution to measured − predicted selected B (×1000)',shrink=.8)
fig.suptitle('Where conditional photometry fails: 40 validation scenes\nOriginal usable rows; central cut; reference B is a diagnostic, not shear response',fontsize=13)
fig.savefig(H/'localization.png',dpi=160);fig.savefig(H/'localization.pdf');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ai,name in enumerate(names):
 for axis,col in enumerate([3,4]):
  g=[groups[name][f'neighbour_score/q{q}']['diagnostics'] for q in x]
  axs[0,axis].errorbar(x,[a['value'][col] for a in g],yerr=[a['jackknife_se'][col] for a in g],color=colors[ai],marker='o',capsize=2,label=labels[name])
  # Extreme-rank sum lacks stored covariance; show point values without fabricated errors.
  j=5 if axis==0 else 7
  axs[1,axis].plot(x,[100*(a['value'][j]+a['value'][j+1]) for a in g],color=colors[ai],marker='o')
for i in range(2):
 for j,coord in enumerate(['radius','flux']):
  axs[i,j].axhline(0,color='0.4',lw=.8);axs[i,j].set(xlabel='Neighbour-score quintile',xticks=x)
  axs[i,j].set_title(coord.capitalize()+(': mean rank shift' if i==0 else ': extreme-rank excess'))
  axs[i,j].set_ylabel('Mean observation rank / 64 − 0.5' if i==0 else 'Excess extreme ranks (pp; null = 2/65)')
axs[0,0].legend(fontsize=8)
fig.suptitle('Conditional distribution checks with 64 IID draws\nPositive mean rank: measured values tend to exceed model draws',fontsize=13)
fig.savefig(H/'distribution_checks.png',dpi=160);fig.savefig(H/'distribution_checks.pdf');plt.close(fig)
print('PHOTOMETRY_DIAGNOSTIC_PLOTS_WRITTEN',flush=True)
