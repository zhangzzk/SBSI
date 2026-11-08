"""Coherent-only precision from saved probability-sample fits; no mixed or model inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from report import CASES,CUTS,CENTER,CUT_NAMES,H,OUT,obs

def estimate(v,w):
    s=np.einsum('n,njcg->jcg',w,v.mean(1));F=s[...,1:]/s[...,:1]
    return (F[0]-F[1])/.04, F, s
def main():
    frames=[];v=[]
    for case in CASES:
        s=pd.read_feather(OUT/f'sample_{case}.feather');s['case']=case;frames.append(s)
        co=[json.loads((OUT/f'coherent{case}_{j}.json').read_text()) for j in [0,1]]
        assert all(r['passed'] for r in co)
        v.append(np.array([[obs(c['results'][str(ident)])[0] for c in co] for ident in s.index_input]).transpose(0,2,1,3,4))
    df=pd.concat(frames);v=np.concatenate(v);w=df.weight.to_numpy();c=df.case.to_numpy();st=df.stratum.to_numpy()
    val,F,s=estimate(v,w);jk=np.array([estimate(v,w*(c!=k))[0] for k in CASES])
    z=(v[...,1:]-v[...,:1]*F)/s[...,:1];influence=(z[:,:,0]-z[:,:,1])/.04;mean=influence.mean(1)
    vs=np.zeros_like(val)
    for k in CASES:
        for h in [0,1]:
            sel=(c==k)&(st==h);N=w[sel].sum();n=sel.sum();vs+=N*N*(1-n/N)*mean[sel].var(0,ddof=1)/n
    varfit=np.square((influence[:,0]-influence[:,1])*w[:,None,None]/2).sum(0)
    r=dict(cuts=CUT_NAMES,R=val.tolist(),case_jackknife_se=np.sqrt(.75*np.square(jk-jk.mean(0)).sum(0)).tolist(),stratified_sampling_se=np.sqrt(vs).tolist(),optimizer_mc_se=np.sqrt(varfit).tolist(),
           distinct_selected_by_leg_seed=v[:,:,:,CENTER,0].sum(0).tolist(),by_optimizer_seed=[estimate(v[:,i:i+1],w)[0].tolist() for i in [0,1]])
    (H/'coherent_summary.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Coherent measured R11 at central cut:',{k:r[k][CENTER][0] for k in ['R','case_jackknife_se','stratified_sampling_se','optimizer_mc_se']},r['distinct_selected_by_leg_seed'])
if __name__=='__main__':main()
