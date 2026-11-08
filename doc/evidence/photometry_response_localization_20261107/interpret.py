"""Audit the completed localization and distinguish reweighting from within-bin changes."""
from pathlib import Path
import os,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert os.environ.get('SLURM_JOB_ID')
H=Path(__file__).resolve().parent;cfg=json.loads((H/'manifest.json').read_text());D=Path(cfg['data_root']);O=Path(cfg['export_root'])
r=json.loads((H/'report.json').read_text());old=json.loads((Path(cfg['source_experiment'])/'report.json').read_text())
n=len(cfg['cases']);w=np.vstack([np.ones(n),np.ones((n,n))-np.eye(n)])

def stat(v):return {'value':v[0].tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(v[1:]-v[1:].mean(0)).sum(0)).tolist()}
checked=0

def check(v,expected):
 global checked
 for k,x in stat(v).items():np.testing.assert_allclose(x,expected[k],rtol=2e-8,atol=2e-12);checked+=1

def composition(raw):
 # raw: weight, arm, mode, leg, cut, bin, statistics (events summed).
 n0,n1=raw[...,0],raw[...,1];s0,s1=raw[...,2:6],raw[...,6:10]
 assert np.all(n0>0) and np.all(n1>0), 'No within-bin attribution for empty selected cells'
 f0=n0/n0.sum(-1,keepdims=True);f1=n1/n1.sum(-1,keepdims=True)
 mu0=s0/n0[...,None];mu1=s1/n1[...,None]
 mix=.5*(f1-f0)[...,None]*(mu1+mu0)
 within=.5*(f1+f0)[...,None]*(mu1-mu0)
 delta=f1[...,None]*mu1-f0[...,None]*mu0
 np.testing.assert_allclose(mix+within,delta,atol=2e-15,rtol=1e-10)
 return mix,within,f0,f1,mu0,mu1

out={'definition':'Exact alternative symmetric decomposition of the selected mean into changes of selected bin fractions and changes of within-bin means. Not a causal attribution; depends on the fixed truth-bin partition. Each shear leg retains its own measured/model selection normalization.','axes':{},'audit':{}}
plot={}
for axis in [1,2]:
 allraw=[];receipt_errors=[]
 for case in cfg['cases']:
  p=D/'localization'/f'axis{axis}'/f'case{case}.npz';receipt=json.loads(p.with_suffix('.json').read_text())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['data_sha256']
  assert receipt['partition_and_decomposition_passed']
  receipt_errors.append(receipt['double_accumulation_response_max_abs'])
  with np.load(p) as z:
   np.testing.assert_allclose(z['raw'].sum((4,5)),z['direct'],rtol=2e-10,atol=2e-7)
   allraw.append(z['raw'])
 raw=np.tensordot(w,np.array(allraw),axes=(1,0))
 # Independent reconstruction of three original terms.
 n0=raw[...,0].sum((-2,-1),keepdims=True);n1=raw[...,1].sum((-2,-1),keepdims=True)
 inv=.5*(1/n0+1/n1)[...,None]
 a=raw[...,10:14]*inv;b=raw[...,14:18]*inv;c=.5*(raw[...,2:6]+raw[...,6:10])*(1/n1-1/n0)[...,None]
 t=np.stack([a,b,c],-2);resp=(t[:,:,:,0]-t[:,:,:,1])/.04
 total=resp.sum((4,5,6))
 rows={}
 modes=['predicted_usability_generated_selection','true_usability_generated_selection']
 # Marginal truth-bin partitions use the same joint sufficient statistics.
 event_sum=raw.sum(-2);partitions={'joint':event_sum,'size':event_sum.reshape(*event_sum.shape[:-2],5,4,18).sum(-2),'neighbour':event_sum.reshape(*event_sum.shape[:-2],5,4,18).sum(-3)}
 compositions={key:composition(value) for key,value in partitions.items()}
 for ai,arm in enumerate(cfg['contrasts']):
  rows[arm]={}
  for mi,mode in enumerate(modes):
   rows[arm][mode]={}
   for ki,cut in enumerate(cfg['cuts']):
    z=resp[:,ai,mi,ki];diag=z[...,axis-1]+z[...,axis+1]
    existing=r['axes'][str(axis)]['arms'][arm][mode][cut]
    check(z.sum((1,2)),existing['all_components_by_term']);check(diag.sum((1,2,3)),existing['diagonal_total']);check(diag.sum((1,2)),existing['diagonal_by_term'])
    check(diag.reshape(n+1,5,4,5,3).sum((2,3)),existing['diagonal_by_true_size_and_term']);check(diag.reshape(n+1,5,4,5,3).sum((1,3)),existing['diagonal_by_neighbour_quartile_and_term']);check(diag.sum(1),existing['diagonal_by_event_and_term']);check(diag.sum((2,3)),existing['diagonal_joint_bin_total'])
    # Selected masses and relative changes for both legs.
    masses=raw[:,ai,mi,:,ki].sum((2,3))
    rel=(masses[...,1]/masses[...,0]-1)*100
    entry={'selected_mass_relative_change_percent_by_leg':stat(rel),'normalization_term':stat(diag.sum((1,2))[:,-1]),'partitions':{}}
    for part,(mix,within,f0,f1,mu0,mu1) in compositions.items():
     m=mix[:,ai,mi,:,ki];v=within[:,ai,mi,:,ki]
     m=(m[:,0]-m[:,1])/.04;v=(v[:,0]-v[:,1])/.04
     m=m[...,axis-1]+m[...,axis+1];v=v[...,axis-1]+v[...,axis+1]
     np.testing.assert_allclose((m+v).sum(-1),diag.sum((1,2,3)),atol=2e-12,rtol=1e-9)
     means0=mu0[:,ai,mi,:,ki];means1=mu1[:,ai,mi,:,ki]
     mu=((means1[:,0]-means1[:,1])-(means0[:,0]-means0[:,1]))/.04
     mu=mu[...,axis-1]+mu[...,axis+1]
     entry['partitions'][part]={'selected_fraction_change_pp_by_leg':stat(100*(f1-f0)[:,ai,mi,:,ki]),'composition_contribution':stat(m),'within_bin_contribution':stat(v),'total_contribution':stat(m+v),'within_bin_response_change':stat(mu),'pooled_composition':stat(m.sum(-1)),'pooled_within_bin':stat(v.sum(-1))}
    rows[arm][mode][cut]=entry
    if arm=='weighted' and mi==0 and cut=='mag25_radius_gt0.60':plot[axis]=entry
 out['axes'][str(axis)]={'arms':rows,'max_double_accumulation_response_error':max(receipt_errors)}
out['audit']={'passed':True,'reported_statistics_checked':checked,'partition_closure':True,'alternative_composition_closure':True,'job_id':os.environ['SLURM_JOB_ID']}
for dest in [H,O]:(dest/'interpretation.json').write_text(json.dumps(out,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12.5,8),sharex='col');fig.subplots_adjust(top=.85,bottom=.16,hspace=.42,wspace=.26)
labels={'size':['<0.15','0.15–0.25','0.25–0.4','0.4–0.7','≥0.7'],'neighbour':['Q1 (lowest)','Q2','Q3','Q4 (highest)']}
for row,axis in enumerate([1,2]):
 for col,part in enumerate(['size','neighbour']):
  ax=axs[row,col];e=plot[axis]['partitions'][part]
  for i,(key,lab,color) in enumerate([('composition_contribution','Changing population fractions','#D55E00'),('within_bin_contribution','Changing within-bin mean','#0072B2'),('total_contribution','Total','#222222')]):
   v=e[key];xx=np.arange(len(labels[part]))+(i-1)*.16
   ax.errorbar(xx,np.array(v['value'])*1e3,yerr=np.array(v['jackknife_se'])*1e3,fmt=['s','o','D'][i],capsize=3,color=color,label=lab,ms=5)
  ax.axhline(0,color='.5',lw=1);ax.grid(axis='y',alpha=.15);ax.set_xticks(np.arange(len(labels[part])),labels[part]);ax.set_ylabel(f'Axis {axis}: contribution to ΔR (×10⁻³)');ax.set_title('True radius [arcsec]' if part=='size' else 'Truth neighbour influence quartile')
  ax.tick_params(axis='x',labelbottom=True)
axs[0,0].legend(fontsize=9)
fig.suptitle('Population reweighting and changes within each group',fontsize=17,fontweight='bold',y=.97)
fig.text(.5,.915,'Extra loss minus control; central cut; predicted usability; each partition separately sums to total ΔR',ha='center')
fig.text(.5,.035,'Paired ±1 scene-jackknife SE; four development scenes. Group fractions and means are recomputed for each shear leg.\nThis alternative accounting separates population mix from within-group changes; neither decomposition is a causal proof.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(O/f'population_composition.{ext}',dpi=180,bbox_inches='tight')
print('AUDIT',out['audit'])
for axis,z in plot.items():
 print('AXIS',axis,'mass percent',z['selected_mass_relative_change_percent_by_leg'])
 for part,e in z['partitions'].items():
  print(part,'composition',e['pooled_composition'],'within',e['pooled_within_bin'])
  if part!='joint':print('within response',e['within_bin_response_change'])
