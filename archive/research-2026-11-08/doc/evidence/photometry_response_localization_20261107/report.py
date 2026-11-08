"""Whole-scene paired localization; global normalization precedes attribution."""
from pathlib import Path
import importlib.util,json,os,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;cfg=json.loads((H/'manifest.json').read_text());D=Path(cfg['data_root']);O=Path(cfg['export_root']);O.mkdir(parents=True,exist_ok=True)
s=importlib.util.spec_from_file_location('localization_math',H/'decompose.py');d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
def stat(x):
 n=len(x)-1;z=x[1:];return {'value':x[0].tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(z-z.mean(0)).sum(0)).tolist()}
def main():
 assert os.environ.get('SLURM_JOB_ID')
 old=json.loads((Path(cfg['source_experiment'])/'report.json').read_text());n=len(cfg['cases']);weights=np.vstack([np.ones(n),np.ones((n,n))-np.eye(n)])
 result={'definition':'New minus control model response; common latent draws, symmetric decomposition. Contributions use global per-leg normalization, not subgroup response. Errors are paired scene jackknife with models fixed.','cases':cfg['cases'],'terms':cfg['terms'],'components':cfg['components'],'arms':cfg['contrasts'],'cuts':cfg['cuts'],'axes':{}}
 plots={};maxerror=0.
 for axis in [1,2]:
  raws=[];quartiles=[]
  for case in cfg['cases']:
   path=D/'localization'/f'axis{axis}'/f'case{case}.npz';receipt=json.loads(path.with_suffix('.json').read_text())
   assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['data_sha256'] and receipt['partition_and_decomposition_passed']
   assert receipt['decompose_sha256']==hashlib.sha256((H/'decompose.py').read_bytes()).hexdigest()
   assert receipt['source']==hashlib.sha256((H/'evaluate.py').read_bytes()).hexdigest()
   assert receipt['model_pins']=={a:v['sha256'] for a,v in old['training']['arms'].items()}
   assert receipt['case']==case and receipt['axis']==axis
   with np.load(path) as z:raws.append(z['raw']);quartiles.append(z['score_quartiles'].tolist())
  pooled=np.tensordot(weights,np.array(raws),axes=(1,0));terms,direct=d.terms(pooled)
  # weights, arm, mode, leg, cut, joint-bin, event, term, component
  response=(terms[:,:,:,0]-terms[:,:,:,1])/.04
  total=response.sum((4,5,6))
  rows={}
  for ai,arm in enumerate(cfg['contrasts']):
   contrast={'weighted':'loss_only','neighbour':'input_only','both':'both_vs_control'}[arm]
   rows[arm]={}
   for mi,mode in enumerate(['predicted_usability_generated_selection','true_usability_generated_selection']):
    rows[arm][mode]={}
    for ki,cut in enumerate(cfg['cuts']):
     expected=old['axes'][str(axis)]['paired_changes'][contrast][mode][cut]
     got=total[:,ai,mi,ki];ref=np.array(expected['model_flow_response']['value']+expected['model_emulator_response']['value'])
     maxerror=max(maxerror,float(abs(got[0]-ref).max()));np.testing.assert_allclose(got[0],ref,atol=2e-7,rtol=0)
     z=response[:,ai,mi,ki] # weights, joint-bin, event, term, component
     diagonal=z[...,axis-1]+z[...,axis+1]
     bysize=diagonal.reshape(n+1,5,4,5,3).sum((2,3))
     byneighbour=diagonal.reshape(n+1,5,4,5,3).sum((1,3))
     row={'all_components_by_term':stat(z.sum((1,2))),'diagonal_total':stat(diagonal.sum((1,2,3))),'diagonal_by_term':stat(diagonal.sum((1,2))),'diagonal_by_true_size_and_term':stat(bysize),'diagonal_by_neighbour_quartile_and_term':stat(byneighbour),'diagonal_by_event_and_term':stat(diagonal.sum(1)),'diagonal_joint_bin_total':stat(diagonal.sum((2,3)))}
     rows[arm][mode][cut]=row
     if arm=='weighted' and mi==0 and cut=='mag25_radius_gt0.60':
      plots[axis]={'terms':stat(diagonal.sum((1,2))),'size':stat(bysize.sum(-1)),'neighbour':stat(byneighbour.sum(-1)),'boundary':stat(diagonal[:,:,1:4,0].sum(1))}
   # All per-scene terms retained for independent checks.
  if axis==2:np.testing.assert_allclose(quartiles,result['axes']['1']['scene_score_quartiles'],rtol=2e-10,atol=2e-7)
  result['axes'][str(axis)]={'arms':rows,'scene_score_quartiles':quartiles}
  np.savez_compressed(D/f'axis{axis}_jackknife_contributions.npz',response=response,weights=weights)
 result['replay_response_max_abs']=maxerror
 for dest in [H,O]:(dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 plt.rcParams.update({'font.size':11,'axes.spines.right':False,'axes.spines.top':False})
 fig,axs=plt.subplots(2,2,figsize=(12,8));fig.subplots_adjust(top=.86,bottom=.14,hspace=.5,wspace=.3)
 specs=[('terms',['Cut membership','Output value','Normalization'],'A  Three contributions to the response change'),('size',['<0.15','0.15–0.25','0.25–0.4','0.4–0.7','≥0.7'],'B  Contribution by true radius [arcsec]'),('neighbour',['Q1 (lowest)','Q2','Q3','Q4 (highest)'],'C  Contribution by neighbour influence'),('boundary',['Radius only','Flux only','Both'],'D  Cut-membership term by boundary')]
 for a,(key,labels,title) in zip(axs.flat,specs):
  a.axhline(0,color='.4',lw=1);a.grid(axis='y',alpha=.15)
  for ai,axis in enumerate([1,2]):
   val=plots[axis][key];xx=np.arange(len(labels))+(ai-.5)*.16
   a.errorbar(xx,np.array(val['value'])*1e3,yerr=np.array(val['jackknife_se'])*1e3,fmt=['o','s'][ai],capsize=3,label=f'Axis {axis}',color=['#0072B2','#D55E00'][ai])
  a.set_xticks(np.arange(len(labels)),labels,rotation=15 if key=='size' else 0);a.set_title(title,fontsize=12);a.set_ylabel('Contribution to ΔR (×10⁻³)')
 axs[0,0].legend()
 fig.suptitle('Where the extra loss changes the selected response',fontsize=17,fontweight='bold',y=.97)
 fig.text(.5,.915,'Extra-loss model minus matched control; radius >3 px, magnitude <25; predicted usability',ha='center',fontsize=11)
 fig.text(.5,.035,'Four development scenes; paired ±1 scene-jackknife SE. B and C each sum to the total ΔR; D sums to the cut-membership term.\nNeighbour quartiles use truth-only reference influence within each scene. Negative ΔR raises the already-positive shear bias.',ha='center',fontsize=10)
 for ext in ['png','pdf']:fig.savefig(O/f'response_localization.{ext}',dpi=180,bbox_inches='tight')
 lines=['# Selected-response localization completed','',result['definition'],'','Central cut; extra-loss model minus control. Entries below are response changes, not bias percentages.','', '| Contribution | Axis 1 | Axis 2 |','|---|---:|---:|']
 for i,label in enumerate(cfg['terms']):
  lines.append('| '+label+' | '+' | '.join(f"{plots[ax]['terms']['value'][i]:+.8f} ± {plots[ax]['terms']['jackknife_se'][i]:.8f}" for ax in [1,2])+' |')
 lines+=['','Full arm/cut/mode, true-size and neighbour-quartile tables are in report.json. Decomposition is exact for the common-draw pairing and uses symmetric averages; it is not a unique causal attribution. No new model training or simulated images.']
 for dest in [H,O]:(dest/'results.md').write_text('\n'.join(lines)+'\n')
 for name in ['manifest.json','rule.md','verification.json']:shutil.copy2(H/name,O/name)
 print('REPORT_COMPLETE replay_response_max_abs',maxerror,flush=True)
if __name__=='__main__':main()
