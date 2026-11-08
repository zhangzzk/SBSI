"""Independently audit saved small sums and plot the completed fixed comparison."""
from pathlib import Path
import os,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert os.environ.get('SLURM_JOB_ID'), 'Run via SLURM'
H=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
cfg=json.loads((H/'manifest.json').read_text()); out=Path(cfg['export_root'])
r=json.loads((H/'report.json').read_text());q=json.loads((H/'probability_report.json').read_text())
arms=cfg['arms']; cuts=cfg['target_cuts']; central='mag25_radius_gt0.60'; modes=list(r['axes']['1']['arms']['control'])
coefs={'loss_only':[-1,1,0,0],'input_only':[-1,0,1,0],'both_vs_control':[-1,0,0,1],'loss_with_input':[0,0,-1,1],'input_with_loss':[0,-1,0,1],'interaction':[1,-1,-1,1]}
checked=0

def stat(x):
 n=len(x)-1
 return dict(value=x[0],jackknife_se=np.sqrt((n-1)/n*np.square(x[1:]-x[1:].mean(0)).sum(0)))
def check(x,s):
 global checked
 for k,v in stat(x).items():np.testing.assert_allclose(v,s[k],atol=2e-11,rtol=2e-9);checked+=1

def response(x):return (x[:,0,1:]/x[:,0,:1]-x[:,1,1:]/x[:,1,:1])/.04
n=len(cfg['cases']); w=np.vstack([np.ones(n),np.ones((n,n))-np.eye(n)])
for ax in [1,2]:
 samples={}; simref=None
 for arm in arms:
  ss=[];mm=[]
  for case in cfg['cases']:
   path=D/'constant'/arm/f'axis{ax}'/f'case{case}.npz'
   receipt=json.loads(path.with_suffix('.json').read_text())
   assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['data_sha256']
   assert receipt['checkpoint_sha256']==r['training']['arms'][arm]['sha256']
   assert all(receipt[k] for k in ['simulation_replay_passed','reference_score_replay_passed','true_selection_mass_exact','pooled_emulator_roundoff_check_passed'])
   with np.load(path) as z:
    ss.append(z['sim'].sum(-2));mm.append(z['model'].sum(-2));allcuts=list(z['cuts'])
  ss=np.array(ss);mm=np.array(mm)
  if simref is None:simref=ss
  else:np.testing.assert_array_equal(simref,ss)
  for mode,name in enumerate(modes):
   for cut in cuts:
    ki=allcuts.index(cut);s=np.einsum('wc,clq->wlq',w,ss[:,:,ki]);m=np.einsum('wc,clq->wlq',w,mm[:,mode,:,ki])
    flow=response(m[:,:,:3]);emu=response(m[:,:,[0,3,4]]);total=flow+emu
    v={'m_percent':100*(response(s)[:,ax-1]/total[:,ax-1]-1),'model_flow_response':flow,'model_emulator_response':emu}
    samples[arm,name,cut]=v
    for key in v:check(v[key],r['axes'][str(ax)]['arms'][arm][name][cut][key])
   rms=np.sqrt(np.mean([samples[arm,name,c]['m_percent']**2 for c in cuts],axis=0));samples[arm,name,'rms']=rms
   check(rms,r['axes'][str(ax)]['arms'][arm][name]['nine_cut_rms_m_percent'])
 for con,c in coefs.items():
  for name in modes:
   for cut in cuts:
    for key in ['m_percent','model_flow_response','model_emulator_response']:
     check(sum(a*samples[arm,name,cut][key] for a,arm in zip(c,arms)),r['axes'][str(ax)]['paired_changes'][con][name][cut][key])
   check(sum(a*samples[arm,name,'rms'] for a,arm in zip(c,arms)),r['axes'][str(ax)]['paired_changes'][con][name]['nine_cut_rms_change'])
for split in ['train','validation']:
 with np.load(D/f'own_{split}.npz') as z:raw=z['probability_sums'];kinds=list(z['probability_kinds']);cc=list(z['cuts']);cases=z['cases']
 n=len(cases);w=np.vstack([np.ones(n),np.ones((n,n))-np.eye(n)])
 assert cc==cuts
 for kind,kindname in enumerate(kinds):
  for leg,pop in [(None,'pooled'),(0,'g0'),(1,'g05')]:
   data=raw[:,kind].sum(2) if leg is None else raw[:,kind,:,leg]
   vals=[]
   for ai,arm in enumerate(arms):
    totals=np.einsum('wc,ckv->wkv',w,data[ai]);score=totals[...,3]/totals[...,0];vals.append(score)
    for ci,cut in enumerate(cc):check(score[:,ci],q['splits'][split]['scores'][kindname][pop][arm][cut]['probability_score'])
    check(score.mean(1),q['splits'][split]['scores'][kindname][pop]['nine_cut_mean_score'][arm])
   for con,c in coefs.items():
    label='weighted_minus_control' if con=='loss_only' else con
    v=sum(a*b for a,b in zip(c,vals))
    for ci,cut in enumerate(cc):check(v[:,ci],q['splits'][split]['scores'][kindname][pop][label][cut]['probability_score'])
    check(v.mean(1),q['splits'][split]['scores'][kindname][pop]['nine_cut_mean_score'][label])
audit={'passed':True,'scalar_or_vector_statistics_checked':checked,'checks':['saved data hashes and checkpoint receipts','simulation equality across arms','independent reconstruction of pooled leg responses','all-cut biases and flow/emulator terms','all paired contrast and RMS jackknife statistics','hard and soft conditional scores and contrasts on both data splits'],'job_id':os.environ['SLURM_JOB_ID']}
for dest in [H,out]:(dest/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('AUDIT_PASSED',checked,flush=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
colors=['#0072B2','#D55E00']; contrasts=['loss_only','input_only','both_vs_control'];labels=['Extra loss','Extra neighbour input','Both'];x=np.arange(3)
fig,axs=plt.subplots(2,2,figsize=(13,8.6));fig.subplots_adjust(hspace=.58,wspace=.29,bottom=.14,top=.86)
fig.suptitle('Better cut probabilities did not improve shear calibration',fontsize=18,fontweight='bold',y=.97)
fig.text(.5,.918,'Four models with the same starting predictions; only photometry learns; 64 updates',ha='center',fontsize=12)
for a in axs.flat:a.axhline(0,color='.4',lw=1);a.grid(axis='y',alpha=.16)
sc=q['splits']['validation']['scores']['hard']['pooled']['nine_cut_mean_score']
a=axs[0,0]; vals=[sc['weighted_minus_control' if c=='loss_only' else c] for c in contrasts]
a.errorbar(x,[v['value']*1e4 for v in vals],yerr=[v['jackknife_se']*1e4 for v in vals],fmt='o',color='#009E73',capsize=4)
a.set(title='A  Cut-probability error: lower is better',ylabel='Change from control (×10⁻⁴)');a.set_xticks(x,labels);a.set_xlim(-.45,2.45)
for ai,ax in enumerate([1,2]):
 a=axs[0,1];vs=[r['axes'][str(ax)]['paired_changes'][c][modes[0]][central]['m_percent'] for c in contrasts]
 a.errorbar(x+(ai-.5)*.16,[v['value'] for v in vs],yerr=[v['jackknife_se'] for v in vs],fmt=['o','s'][ai],color=colors[ai],capsize=4,label=f'm{ax}')
 a=axs[1,0];vs=[r['axes'][str(ax)]['paired_changes'][c][modes[0]]['nine_cut_rms_change'] for c in contrasts]
 a.errorbar(x+(ai-.5)*.16,[v['value'] for v in vs],yerr=[v['jackknife_se'] for v in vs],fmt=['o','s'][ai],color=colors[ai],capsize=4,label=f'Axis {ax}')
 a=axs[1,1];vs=[r['axes'][str(ax)]['paired_changes']['loss_only'][mode][central]['m_percent'] for mode in modes]
 a.errorbar(x+(ai-.5)*.16,[v['value'] for v in vs],yerr=[v['jackknife_se'] for v in vs],fmt=['o','s'][ai],color=colors[ai],capsize=4,label=f'm{ax}')
axs[0,1].set(title='B  Central-cut shear bias becomes more positive',ylabel='Change in m (percentage points)');axs[0,1].set_xticks(x,labels);axs[0,1].legend()
axs[1,0].set(title='C  Error across all nine cuts: lower is better',ylabel='Change in nine-cut RMS (percentage points)');axs[1,0].set_xticks(x,labels);axs[1,0].legend()
axs[1,1].set(title='D  Extra loss changes the selection response',ylabel='Change in m (percentage points)');axs[1,1].set_xticks(x,['Predicted usability\nGenerated cuts','Measured usability\nGenerated cuts','Measured usability\nMeasured cuts']);axs[1,1].tick_params(axis='x',labelsize=9)
fig.text(.5,.035,'A: 40 validation scenes, mean over nine cuts. B–D: four reused development scenes; paired ±1 scene-jackknife SE.\nCentral cut: radius >3 px, magnitude <25. B–C use predicted usability. D retains generated model photometry in every mode.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(out/f'factorial_summary.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(13,4.5),sharey=True);fig.subplots_adjust(bottom=.25,top=.81,wspace=.12)
for ax,a in enumerate(axs,1):
 a.axhline(0,c='.4',lw=1)
 for con,col,lab in zip(contrasts,['#0072B2','#D55E00','#009E73'],labels):
  vs=[r['axes'][str(ax)]['paired_changes'][con][modes[0]][c]['absolute_m_change'] for c in cuts]
  a.errorbar(np.arange(9),[v['value'] for v in vs],yerr=[v['jackknife_se'] for v in vs],fmt='o-',ms=4,lw=1,capsize=2,color=col,label=lab)
 a.set_xticks(np.arange(9),[f'{radius:.1f}\n{mag:.1f}' for radius in [2.8,3.,3.2] for mag in [24.9,25.,25.1]])
 a.set_title(f'Axis {ax}');a.set_xlabel('Radius lower cut [px]\nMagnitude upper cut [mag]');a.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Change in |m| (percentage points)');axs[0].legend(fontsize=10)
fig.suptitle('All nine target cuts: negative means closer to zero bias',fontsize=16,y=.97)
fig.text(.5,.015,'Predicted usability and generated cuts; four development scenes; paired ±1 scene-jackknife SE.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(out/f'all_cut_changes.{ext}',dpi=180,bbox_inches='tight')
print('FIGURES_SAVED',out,flush=True)
