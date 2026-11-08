import os,json,csv,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert os.environ.get('SLURM_JOB_ID')
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches');R=D/'flow_photometry_information_20261107'
BASE=Path('/home/z/Zekang.Zhang/Documents/Codex/2026-11-07/size-and-flux-supplied-to-model');O=BASE/'outputs/information_test'
meta=json.loads((R/'completed.json').read_text());cases=meta['split']['test'];C=len(cases)
with np.load(R/'test.npz') as z:x=z['x'];y=z['y'];w=z['w'];case=z['case'];flow=z['flow']
with np.load(R/'train.npz') as z:train_x=z['x']
assert len(x)==163840 and C==40
arms=['existing_d4','existing_d8','score_d4','score_d8','shuffled_d8'];models=['original_flow','nine_cut_flow']+arms
preds=[flow[:,0],flow[:,1]];margins={}
for arm in arms:
    pp=np.column_stack([np.load(R/'models'/f'{arm}_{q}_test.npy') for q in ['radius','magnitude','joint_cut']])
    margins[arm]=pp[:,2].copy();pp[:,2]=np.exp(-np.logaddexp(0,-pp[:,2]));assert np.isfinite(pp).all();preds.append(pp)
pred=np.array(preds)
# Independent antithetic half-banks remove the quadratic Monte Carlo bias in flow MSE/Brier comparisons.
flow_half=np.empty_like(flow)
for cc in cases:
    ii=case==cc
    with np.load(D/'flow_truth_diagnostics_20261107'/f'case{cc}.npz') as z:
        np.testing.assert_array_equal(x[ii,:8],z['context'])
        flow_half[ii]=z['half'][:,:,[0,1,4]].transpose(1,0,2)
rows_bycase=[case==c for c in cases];den=np.array([w[t].sum() for t in rows_bycase])
def stats(sums,denom=den):
    sums=np.asarray(sums);d=np.asarray(denom)
    full=sums.sum(0)/d.sum(0)
    jk=(sums.sum(0)-sums)/(d.sum(0)-d)
    se=np.sqrt((C-1)/C*np.sum((jk-jk.mean(0))**2,axis=0))
    return {'value':float(full),'se':float(se),'by_scene':(sums/d).tolist()}
def mean(v):return stats(np.array([np.sum(w[t]*v[t]) for t in rows_bycase]))
losses={};results={}
for ai,name in enumerate(models):
    v=pred[ai];losses[name]={'radius_mse':(v[:,0]-y[:,0])**2,'magnitude_mse':(v[:,1]-y[:,1])**2,'cut_brier':(v[:,2]-y[:,2])**2}
    if name in margins:losses[name]['cut_logloss']=np.logaddexp(0,margins[name])-y[:,2]*margins[name]
    rr={k:mean(z) for k,z in losses[name].items()}
    if ai<2:
        first=flow_half[:,ai];second=2*v-first
        unbiased=(first-y)*(second-y)
        rr['mc_unbiased_losses']={k:mean(unbiased[:,j]) for j,k in enumerate(['radius_mse','magnitude_mse','cut_brier'])}
    rr['radius_bias']=mean(v[:,0]-y[:,0]);rr['magnitude_bias']=mean(v[:,1]-y[:,1]);rr['pass_rate_bias']=mean(v[:,2]-y[:,2])
    # Measured-minus-predicted selected-score contrast, each with its own selected denominator.
    masses=np.array([[np.sum(w[t]*y[t,2]),np.sum(w[t]*v[t,2])] for t in rows_bycase])
    scores=np.array([[np.sum(w[t]*y[t,2]*x[t,8]),np.sum(w[t]*v[t,2]*x[t,8])] for t in rows_bycase])
    pooled=scores.sum(0)/masses.sum(0);jk=(scores.sum(0)-scores)/(masses.sum(0)-masses);delta=jk[:,0]-jk[:,1]
    rr['selected_score_error']={'value':float(pooled[0]-pooled[1]),'se':float(np.sqrt((C-1)/C*np.sum((delta-delta.mean())**2)))}
    subset=(x[:,5]>=.00477071604691446)&(x[:,5]<.01859665252268315)
    dd=np.array([w[t&subset].sum() for t in rows_bycase])
    rr['close_crowding_bin']={q:stats(np.array([np.sum(w[t&subset]*(v[t&subset,k]-y[t&subset,k])) for t in rows_bycase]),dd) for k,q in enumerate(['radius_bias','magnitude_bias','pass_rate_bias'])}
    results[name]=rr
flow_comparisons={}
for arm in arms:
    rr={}
    for fi,fname in enumerate(['original_flow','nine_cut_flow']):
        first=flow_half[:,fi];second=2*flow[:,fi]-first;unbiased=(first-y)*(second-y)
        rr[fname]={key:mean(losses[arm][key]-unbiased[:,j]) for j,key in enumerate(['radius_mse','magnitude_mse','cut_brier'])}
    flow_comparisons[arm]=rr
comparisons=[('capacity','existing_d8','existing_d4'),('score_at_depth4','score_d4','existing_d4'),('score_at_depth8','score_d8','existing_d8'),('shuffle_at_depth8','shuffled_d8','existing_d8'),('score_vs_shuffle','score_d8','shuffled_d8')]
contrasts={}
for label,a,b in comparisons:
    rr={}
    for key in losses[a]:
        diff=losses[a][key]-losses[b][key];s=mean(diff);s['scenes_improved']=sum(v<0 for v in s['by_scene']);s['relative_percent']=100*s['value']/results[b][key]['value'];rr[key]=s
    contrasts[label]=dict(new=a,reference=b,metrics=rr)
output=dict(sample=len(x),population=float(w.sum()),cases=cases,results=results,contrasts=contrasts,flow_comparisons_mc_unbiased=flow_comparisons,training=meta['results'],definitions={'contrast_sign':'new minus reference; negative loss contrast is improvement','selected_score_error':'measured minus predicted selected-mean fixed-photometry reference coefficient','uncertainty':'paired delete-one-test-scene jackknife SE, fitted predictors fixed'})
(O/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');shutil.copy(R/'completed.json',O/'source_manifest.json')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
labels=['More capacity, same inputs','Add score, depth 4','Add score, depth 8','Shuffled score, depth 8']
fig,axs=plt.subplots(1,3,figsize=(15,5.2),layout='constrained')
fig.suptitle('Does additional neighbour information improve predictions on held-out scenes?',fontsize=16,fontweight='bold')
for ax,(key,title) in zip(axs,[('radius_mse','Size prediction'),('magnitude_mse','Magnitude prediction'),('cut_logloss','Joint-cut probability')]):
    vals=[];ses=[]
    for label,a,b in comparisons[:4]:
        m=contrasts[label]['metrics'][key];scale=100/results[b][key]['value'];vals.append(m['value']*scale);ses.append(m['se']*scale)
    ax.axvline(0,color='.5',lw=1);ax.errorbar(vals,np.arange(4),xerr=ses,fmt='o',color='#0072B2',capsize=4)
    ax.set(yticks=np.arange(4),yticklabels=labels,title=title,xlabel='Change in prediction loss [% of reference loss]');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
fig.supxlabel('Left of zero = improvement. Bars: paired 40-scene jackknife ±1 SE. Models fit on 128 scenes; early stopping on 32 different scenes.\nSame eight inputs at two tree depths; added score uses detailed truth neighbour information. This is a diagnostic predictor test, not a retrained flow.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(O/f'information_vs_capacity.{ext}',dpi=210,bbox_inches='tight')
plt.close(fig)
# Mean residuals versus close crowding, using the previously fixed diagnostic bins.
edges=np.array([0,.00477071604691446,.01859665252268315,.07174413204193122,np.inf])
fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained');fig.suptitle('Do the diagnostic predictors remove the close-neighbour residual?',fontsize=16,fontweight='bold')
for ai,name,col in [(1,'Nine-cut flow','#777777'),(3,'Existing inputs, depth 8','#0072B2'),(5,'Added score, depth 8','#D55E00')]:
    yy=[];ee=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        keep=(x[:,5]>=lo)&(x[:,5]<hi);dd=np.array([w[t&keep].sum() for t in rows_bycase]);vals=[];ers=[]
        for k in range(3):
            s=stats(np.array([np.sum(w[t&keep]*(pred[ai,t&keep,k]-y[t&keep,k])) for t in rows_bycase]),dd);vals.append(s['value']);ers.append(s['se'])
        yy.append(vals);ee.append(ers)
    yy=np.array(yy);ee=np.array(ee)
    for k,ax in enumerate(axs):
        scale=100 if k==2 else 1;ax.errorbar(np.arange(4),scale*yy[:,k],yerr=scale*ee[:,k],fmt='o-',color=col,label=name,capsize=3)
for ax,label in zip(axs,['Predicted − measured radius [pixels]','Predicted − measured magnitude [mag]','Predicted − measured passing fraction [pp]']):
    ax.axhline(0,color='.5',lw=1);ax.set(xticks=np.arange(4),xticklabels=['<.00477','.00477–.0186','.0186–.0717','≥.0717'],xlabel='Close-crowding score bin',ylabel=label);ax.tick_params(axis='x',rotation=20);ax.grid(alpha=.15)
axs[0].legend(frameon=False,fontsize=9)
fig.supxlabel('Same test galaxies and true-property bins; ±1 paired scene-jackknife SE. Positive magnitude error means too faint.\nThese subpopulation residuals supplement the full-population held-out prediction losses.',fontsize=10)
for ext in ['png','pdf']:fig.savefig(O/f'close_neighbour_residuals.{ext}',dpi=210,bbox_inches='tight')
plt.close(fig)
def fmt(s):return f"{s['value']:+.6g} ± {s['se']:.2g}"
lines=['# Completed information-versus-learning diagnostic','',f'Test: {len(x):,} galaxies in {C} scenes, disjoint from 128 training and 32 early-stopping scenes.','',
'## Held-out loss changes','', 'Negative means improvement. Errors are paired scene-jackknife standard errors, not training uncertainty.','',
'| Contrast | Radius MSE | Magnitude MSE | Joint-cut log loss |','|---|---:|---:|---:|']
for label,a,b in comparisons:
    rr=contrasts[label]['metrics'];lines.append('| '+label+' | '+' | '.join(fmt(rr[q]) for q in ['radius_mse','magnitude_mse','cut_logloss'])+' |')
lines+=['','## Selection association','', 'Reference-score errors are measured minus predicted; this is not R11 or m.','', '| Predictor | Selected-mean reference-score error |','|---|---:|']
for name in models:lines.append(f"| {name} | {fmt(results[name]['selected_score_error'])} |")
lines+=['','## Checks and limitations','', '- Hash-verified sources and feature files; zero unmatched IDs; unique samples; case-disjoint fits, early stopping and test.','- Every saved predictor reproduces its test predictions exactly on reload. Test data do not choose model rounds.','- Flow cut log loss is omitted: 64-draw probabilities may equal zero or one. Its Brier loss includes integration noise.','- Added-score improvement is evidence for useful additional conditioning, not proof of information-theoretic insufficiency or a uniquely missing physical feature.','- The held-out test scenes are reused development/validation scenes. Models are fixed for the jackknife; no training uncertainty or independent calibration claim.','- These regressors predict photometric means and one cut probability, not the full joint distribution. No production model was changed.']
(O/'results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(contrasts,indent=2));print('REPORT COMPLETE',flush=True)
