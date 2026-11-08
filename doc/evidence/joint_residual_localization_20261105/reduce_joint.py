"""Exact response-unit residual contributions of fixed truth cells."""
import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
cases=[1000,1040,1080,1120]
s=[];m=[]
for c in cases:
 with np.load(O/f'joint_case{c}.npz') as z:s.append(z['sim']);m.append(z['model'])
s=np.array(s);m=np.array(m)
w=np.vstack([np.ones(4),np.random.default_rng(20260925).multinomial(4,np.full(4,.25),size=10000)])
def ent(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
def response(v):return (v[:,0]-v[:,1])/.04
def div(x,y):return np.divide(x,y,out=np.zeros_like(x),where=y!=0)
labels={'true_size':['<0.15','0.15-0.25','0.25-0.4','0.4-0.7','>=0.7'],
        'true_magnitude':['<24','24-25.5','25.5-27.5','>=27.5'],
        'crowding_log1p':['<0.1','0.1-0.3','0.3-1','1-3','>=3']}
out={'cases':cases,'sign':'simulation minus model','bootstrap':'10000 paired whole-case resamples, seed20260925','cut_reports':{},'limitations':'Four cases; fixed checkpoints/draws. Truth-cell contributions add to total error; they are not independent flow/emulator errors. Symmetric within/mix split depends on binning.'}
for k,cut in enumerate(['uncut','mag25','radius3','radius3_mag25']):
 bs=np.einsum('bc,clnq->blnq',w,s[:,:,k]).reshape(-1,2,5,4,5,2)
 bm=np.einsum('bc,clnq->blnq',w,m[:,:,k]).reshape(-1,2,5,4,5,3)
 ts=bs.sum((2,3,4));tm=bm.sum((2,3,4))
 rs=response(ts[...,1]/ts[...,0]);rf=response(tm[...,1]/tm[...,0]);re=response(tm[...,2]/tm[...,0]);err=rs-rf-re
 report={'R_sim':ent(rs),'R_flow':ent(rf),'R_emulator':ent(re),'error':ent(err),'axes':{}}
 for axis,sumaxes in [('true_size',(3,4)),('true_magnitude',(2,4)),('crowding_log1p',(2,3))]:
  ss=bs.sum(sumaxes);mm=bm.sum(sumaxes)
  ps=ss[...,0]/ts[...,0,None];pm=mm[...,0]/tm[...,0,None]
  es=div(ss[...,1],ss[...,0]);em=div(mm[...,1]+mm[...,2],mm[...,0])
  contributions=response(ss[...,1]/ts[...,0,None]-(mm[...,1]+mm[...,2])/tm[...,0,None])
  within=response(.5*(ps+pm)*(es-em));mix=response(.5*(es+em)*(ps-pm))
  np.testing.assert_allclose(contributions,within+mix,atol=2e-12,rtol=0)
  np.testing.assert_allclose(contributions.sum(1),err,atol=2e-12,rtol=0)
  report['axes'][axis]=[{'bin':label,'simulation_mass_fraction':float(ps[0,:,i].mean()),'model_mass_fraction':float(pm[0,:,i].mean()),
    'error_contribution':ent(contributions[:,i]),'within_contribution':ent(within[:,i]),'mixture_contribution':ent(mix[:,i]),
    'R_sim_contribution':ent(response(ss[...,1]/ts[...,0,None])[:,i]),
    'R_flow_contribution':ent(response(mm[...,1]/tm[...,0,None])[:,i]),
    'R_emulator_contribution':ent(response(mm[...,2]/tm[...,0,None])[:,i])} for i,label in enumerate(labels[axis])]
 # Full three-way cells, exact globally-normalized response contribution.
 sc=bs.reshape(-1,2,100,2);mc=bm.reshape(-1,2,100,3)
 ce=response(sc[...,1]/ts[...,0,None]-(mc[...,1]+mc[...,2])/tm[...,0,None])
 np.testing.assert_allclose(ce.sum(1),err,atol=2e-12,rtol=0)
 order=np.argsort(np.abs(ce[0]))[::-1][:15]
 report['largest_cells']=[{'true_size':labels['true_size'][i//20],'true_magnitude':labels['true_magnitude'][(i//5)%4],'crowding_log1p':labels['crowding_log1p'][i%5],'error_contribution':ent(ce[:,i])} for i in order]
 out['cut_reports'][cut]=report
old=json.loads((O.parent/'true_selection_flags_20261105/report.json').read_text())['rows']['true_U_generated_cut']
np.testing.assert_allclose(out['cut_reports']['radius3_mag25']['error']['value'],old['error']['value'],rtol=0,atol=2e-12)
(O/'joint_report.json').write_text(json.dumps(out,indent=2)+'\n')
for name,r in out['cut_reports'].items():
 print(name,r['error'],flush=True)
 for axis,rows in r['axes'].items():
  print(axis,[(x['bin'],x['error_contribution']) for x in rows],flush=True)
