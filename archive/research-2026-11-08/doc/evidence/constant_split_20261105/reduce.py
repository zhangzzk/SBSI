"""Exact four-state own/opposite-group response-error decomposition."""
import os,sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from contrasts import contrasts, own, opposite
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang');STUDY=D/'sbsi/galsbi_resim/constant_split_20261105';OLD=D/'sbsi/galsbi_resim/constant_g002/sims'
sys.path.insert(0,str(D/'sbsi_flow_f8_20260924'))
from scripts.evaluate_smooth_classifier_joint import measured_leg
assert json.loads((O/'control_report.json').read_text())['passed'],'Unchanged-image control must pass'
CASES=[1000,1040,1080,1120];CUTS=['uncut','mag25','radius3','radius3_mag25']
sim=[];model=[];receipts=[];edges=[]
EDGE_STATES=[(0,2),(1,3),(0,1),(2,3),(0,3)]
for case in CASES:
 with np.load(STUDY/f'model/case{case}.npz') as z:
  ids=z['input_index'];group=z['group'];mom=z['moments'];prob=z['probability'];context=z['context']
 truth=pd.read_feather(OLD/f'case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather').sort_values('index_input').reset_index(drop=True)
 values=[];usable=[];records=[]
 for state,(root,h) in enumerate([(OLD,.02),(STUDY/'sims',.02),(STUDY/'sims',-.02),(OLD,-.02)]):
  v,u,record=measured_leg(root,case,h,ids);values.append(v);usable.append(u);records.append(record)
  if state in [1,2]:
   t=pd.read_feather(root/f'case{case}_{h:.2f}/real0/catalogues/input/gals_info_tile180.0_-0.5.feather').sort_values('index_input').reset_index(drop=True)
   derived_roundoff={}
   for col in truth.columns:
    if col in ['gamma1_input','gamma2_input']:continue
    if col.startswith(('e1_input_rot','e2_input_rot')):
     # Renderer computes these from identical q/position_angle inputs. Different
     # compute-node trig implementations can differ at float64 roundoff only.
     np.testing.assert_allclose(t[col],truth[col],atol=5e-16,rtol=0)
     derived_roundoff[col]=float(np.max(np.abs(t[col].to_numpy()-truth[col].to_numpy())))
    else:np.testing.assert_array_equal(t[col],truth[col])
   record['derived_intrinsic_ellipticity_roundoff']=derived_roundoff
   np.testing.assert_allclose(t['gamma1_input'],h*np.where(group==0,1.,-1.),atol=0,rtol=0)
   assert np.all(t['gamma2_input']==0)
 common=np.logical_and.reduce(usable)
 # Exact selected-mean shape/selection decomposition requires measured shapes
 # on both endpoints. Use only the explicitly declared common-all-four U cohort.
 ec=np.zeros((4,5,2,2))
 cut_masks=[]
 for v in values:
  r=v[:,2]>3;m=v[:,3]>100
  cut_masks.append([np.ones(len(ids),bool),m,r,r&m])
 for k in range(4):
  for edge,(a,b) in enumerate(EDGE_STATES):
   sa=cut_masks[a][k].astype(float);sb=cut_masks[b][k].astype(float)
   for g in [0,1]:
    mask=common&(group==g)
    ea=values[a][mask,0];eb=values[b][mask,0]
    ec[k,edge,g]=[np.sum(.5*(sa[mask]+sb[mask])*(ea-eb)),np.sum(.5*(ea+eb)*(sa[mask]-sb[mask]))]
 edges.append(ec)
 core=(context[:,4]>=.15)&(context[:,4]<.4)&(context[:,6]<.1)
 # actual/common usability, state, cut, group, population(core/rest), q count/e1/e2
 ss=np.zeros((2,4,4,2,2,3));mm=np.zeros((3,4,4,2,2,5))
 for state in range(4):
  v=values[state];r=v[:,2]>3;mag=v[:,3]>100;cuts=[np.ones(len(ids),bool),mag,r,r&mag]
  own_plus=np.ones(len(ids),bool) if state==0 else np.zeros(len(ids),bool) if state==3 else (group==0 if state==1 else group==1)
  p=np.where(own_plus,prob[0],prob[1])
  for k,cut in enumerate(cuts):
   for g in [0,1]:
    for pop,mask in enumerate([core,~core]):
     select=(group==g)&mask
     for mode,u in enumerate([usable[state],common]):
      rows=select&u&cut;ss[mode,state,k,g,pop]=[rows.sum(),*v[rows,:2].sum(0)]
     for mode,weight in enumerate([p,usable[state].astype(float),common.astype(float)]):
      mm[mode,state,k,g,pop]=weight[select]@mom[state,select,k]
 # Reproduce the original coherent totals, independently of group/population split.
 with np.load(D/f'sbsi_caches/galsbi_dev_constant_grid_20261104/sums/case{case}.npz') as old:
  for k,name in enumerate(['uncut','mag25','radius_gt0.60','mag25_radius_gt0.60']):
   j=list(old['cuts']).index(name)
   for leg,state in enumerate([0,3]):
    np.testing.assert_allclose(ss[0,state,k].sum((0,1)),old['sim'][leg,j,:,:3].sum(0),atol=2e-7,rtol=2e-10)
    for flag in [0,1]:np.testing.assert_allclose(mm[flag,state,k].sum((0,1)),old['model'][flag,leg,j,:,:5].sum(0),atol=2e-7,rtol=2e-10)
 sim.append(ss);model.append(mm);receipts.append({'case':case,'states':records,'common_usable':int(common.sum())})
 print('CASE_REDUCED_AND_REPLAYED',case,flush=True)
sim=np.array(sim);model=np.array(model);edges=np.array(edges)
np.savez(O/'sums.npz',sim=sim,model=model,common_U_edges=edges,cases=CASES,cuts=CUTS)
(O/'catalogue_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
w=np.vstack([np.ones(4),np.random.default_rng(20260925).multinomial(4,np.full(4,.25),size=10000)])
# State order ++,+-,-+,--; columns target group A,B.
def ent(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
out={'cases':CASES,'definition':'Exact four-state response decomposition; error=simulation minus model; own group includes same-group neighbours','bootstrap':'10000 paired whole-case resamples, seed20260925; fixed models/draws','comparisons':{},'limitations':'Four-case diagnostic. Global-denominator group contributions include selection-normalization effects. Opposite-group contrast holds target own shear fixed; no empirical doubling into pure neighbour response.'}
for name,si,mi in [('predicted_U',0,0),('actual_U',0,1),('common_U',1,2)]:
 out['comparisons'][name]={}
 for k,cut in enumerate(CUTS):
  s=np.einsum('bc,csgpq->bsgpq',w,sim[:,si,:,k]);m=np.einsum('bc,csgpq->bsgpq',w,model[:,mi,:,k])
  ds=s[...,0].sum((2,3));dm=m[...,0].sum((2,3));report={}
  for label,pops in [('all',[0,1]),('core',[0]),('outside_core',[1])]:
   ns=np.take(s[...,1],pops,axis=3).sum(3)
   nm=np.take(m[...,1]+m[...,3],pops,axis=3).sum(3)
   cs=contrasts(ns/ds[:,:,None]);cm=contrasts(nm/dm[:,:,None])
   report[label]={x:{'simulation':ent(cs[x]),'model':ent(cm[x]),'error':ent(cs[x]-cm[x])} for x in cs}
   cf=contrasts(np.take(m[...,1],pops,axis=3).sum(3)/dm[:,:,None])
   ce=contrasts(np.take(m[...,3],pops,axis=3).sum(3)/dm[:,:,None])
   for x in cs:
    np.testing.assert_allclose(cf[x]+ce[x],cm[x],atol=2e-12,rtol=0)
    report[label][x].update(model_flow=ent(cf[x]),model_emulator=ent(ce[x]))
  # Group-normalized responses isolate physical target groups from global mixture.
  sg=s.sum(3);mg=m.sum(3);Fs=sg[...,1]/sg[...,0];Fm=(mg[...,1]+mg[...,3])/mg[...,0]
  report['within_group']={}
  for g in [0,1]:
   report['within_group']['AB'[g]]={x:{'simulation':ent(Fs[:,:,g]@coeff[:,g]),'model':ent(Fm[:,:,g]@coeff[:,g]),'error':ent((Fs[:,:,g]-Fm[:,:,g])@coeff[:,g])} for x,coeff in [('own_group',own),('opposite_group',opposite)]}
  if name=='common_U':
   # Finite-difference product/ratio identity:
   # delta(mean)= [sum(Sbar*delta(e)) + sum(ebar*delta(S))
   #               - meanbar*delta(sum(S))] / mean(sum(S)).
   ed=np.einsum('bc,cegq->begq',w,edges[:,k]);parts=np.zeros((len(w),5,2,2))
   for edge,(a,b) in enumerate(EDGE_STATES):
    da=sg[:,a,:,0];db=sg[:,b,:,0];dbar=.5*(da+db)
    fbar=.5*(Fs[:,a]+Fs[:,b])
    parts[:,edge,:,0]=ed[:,edge,:,0]/dbar
    parts[:,edge,:,1]=(ed[:,edge,:,1]-fbar*(da-db))/dbar
    np.testing.assert_allclose(parts[:,edge].sum(-1),Fs[:,a]-Fs[:,b],atol=2e-12,rtol=0)
   for g in [0,1]:
    for x,ee in [('own_group',[0,1] if g==0 else [2,3]),('opposite_group',[2,3] if g==0 else [0,1])]:
     terms=parts[:,ee,g].sum(1)/.08
     expected=Fs[:,:,g]@({'own_group':own,'opposite_group':opposite}[x][:,g])
     np.testing.assert_allclose(terms.sum(1),expected,atol=2e-12,rtol=0)
     report['within_group']['AB'[g]][x].update(simulation_shape_term=ent(terms[:,0]),simulation_selection_term=ent(terms[:,1]))
     if x=='opposite_group':
      # In this fixed-U group, generated selection and flow depend only on the
      # target's own shear, so the model opposite-group term is all emulator.
      flow_opposite=(mg[:,:,g,1]/mg[:,:,g,0])@opposite[:,g]
      np.testing.assert_allclose(flow_opposite,0,atol=2e-12,rtol=0)
  out['comparisons'][name][cut]=report
(O/'report.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# Matched constant-shear decomposition results','', 'Signed absolute errors; four-case paired bootstrap SEs. Own-group includes same-group neighbours.','', '| Flags | Cut | Own-group error | Opposite-group error | Total |','|---|---|---:|---:|---:|']
for name,cuts in out['comparisons'].items():
 for cut,r in cuts.items():
  vals=[r['all'][q]['error'] for q in ['own_group','opposite_group','coherent_total']]
  lines.append('| '+name+' | '+cut+' | '+' | '.join(f"{v['value']:+.6f} +/- {v['se']:.6f}" for v in vals)+' |')
lines+=['','These contrasts add exactly within each comparison. Opposite-group changes neighbours while fixing target own shear. Do not identify own-group with pure self-response. See report.json for within-group normalization and core/complement contributions.']
(O/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
