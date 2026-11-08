"""Physical primary/neighbour contrasts and paired uncertainty from the frozen probability sample."""
import json,os,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

H=Path(__file__).resolve().parent
OUT=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107')
CASES=[1000,1040,1080,1120]
ROLES=np.array([[1,1,-1,-1],[1,-1,1,-1],[2,0,0,-2]])/.08
CUTS=json.loads((H.parent/'neighbour_factor_swap_20261107/manifest.json').read_text())['cuts'][:9]
CUT_NAMES=[c[0] for c in CUTS];CENTER=CUT_NAMES.index('mag25_radius_gt0.60')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def obs(y):
    out=np.zeros((2,len(CUTS),3));u=np.zeros(2)
    if not y['detected'] or y['radius'] is None or y['mag'] is None:return out,u
    r,m=y['radius'],y['mag'];mask=np.array([r>cut[1][1] and m<cut[2] for cut in CUTS])
    for j,g in enumerate(y['g']):
        if g is not None and np.isfinite(g).all() and np.dot(g,g)<1 and r>0 and 0<10**(-.4*(m-30))<np.inf:
            u[j]=1;out[j]=mask[:,None]*np.array([1,*g])
    return out,u
def estimate(x,w,influences=False):
    # x: primary, optimizer repeat, corner, cut, (mass,e1 sum,e2 sum).
    s=np.einsum('n,nskf->skf',w,x.mean(1));mu=s[...,1:]/s[...,:1]
    val=np.einsum('rs,skg->rkg',ROLES,mu)
    np.testing.assert_allclose(val[0]+val[1],val[2],atol=2e-13,rtol=0)
    if not influences:return val
    z=(x[...,1:]-x[...,:1]*mu)/s[...,:1]
    return val,np.einsum('rs,nhskg->nhrkg',ROLES,z)
def entry(x):return np.asarray(x).tolist()
def summarize(x,y,w,case,strata,block):
    # x - y, or x alone. Keep case deletion, drawing uncertainties, optimizer
    # seeds and covariance between the physical roles paired throughout.
    val,inf=estimate(x,w,True)
    if y is not None:
        oth,ii=estimate(y,w,True);val-=oth;inf-=ii
    jk=[]
    for c in CASES:
        ww=w*(case!=c);a=estimate(x,ww)
        if y is not None:a-=estimate(y,ww)
        jk.append(a)
    jk=np.array(jk);d=jk-jk.mean(0);casevar=.75*np.sum(d*d,axis=0)
    casecov=.75*np.einsum('ir,is->rs',d[:,:,CENTER,0],d[:,:,CENTER,0])
    mean=inf.mean(1);drawvar=np.zeros_like(val);spatial=np.zeros_like(val);planning=[]
    drawcov=np.zeros((3,3))
    for c in CASES:
        for st in [0,1]:
            take=(case==c)&(strata==st);n=take.sum();N=w[take].sum();a=mean[take];factor=N*N*(1-n/N)/n
            drawvar+=factor*a.var(0,ddof=1);drawcov+=factor*np.cov(a[:,:,CENTER,0].T)
            planning.append((int(round(N)),n,a[:,:,CENTER,0].var(0,ddof=1)))
        block_sums=np.stack([np.einsum('n,nrkg->rkg',w[(case==c)&(block==b)],mean[(case==c)&(block==b)]) for b in range(16)])
        spatial+=16/15*np.square(block_sums-block_sums.mean(0)).sum(0)
    seedvar=np.square((inf[:,0]-inf[:,1])*w[:,None,None,None]/2).sum(0)
    for cov in [drawcov,casecov]:np.testing.assert_allclose(cov[2,2],cov[0,0]+cov[1,1]+2*cov[0,1],atol=1e-13)
    # Grow the original stratum allocation proportionally, stopping a stratum
    # at census. This scales only the estimated finite-frame sampling variance.
    budget=[]
    for role in range(3):
        def allocation(mult):
            ns=[min(N,max(n,int(np.ceil(n*mult)))) for N,n,_ in planning]
            var=sum(N*N*(1/nn-1/N)*v[role] for nn,(N,n,v) in zip(ns,planning))
            return ns,var
        lo,hi=1.,max(N/n for N,n,_ in planning)
        for _ in range(55):
            mid=(lo+hi)/2
            if allocation(mid)[1]>.003**2:lo=mid
            else:hi=mid
        ns,_=allocation(1 if allocation(1)[1]<=.003**2 else hi)
        budget.append(dict(total=sum(ns),counts_by_case_then_stratum=ns))
    return dict(value=entry(val),case_jackknife_se=entry(np.sqrt(casevar)),stratified_sampling_se=entry(np.sqrt(drawvar)),within_case_spatial_cluster_se=entry(np.sqrt(spatial)),optimizer_MC_se=entry(np.sqrt(seedvar)),central_R11_role_case_covariance=entry(casecov),central_R11_role_sampling_covariance=entry(drawcov),sampling_only_allocation_at_SE_0003=budget)
def main():
    assert os.environ.get('SLURM_JOB_ID')
    rows=[];allsim=[];mods={};receipts=[]
    for case in CASES:
        samp=OUT/f'sample_{case}.feather';s=pd.read_feather(samp);ids=s.index_input.to_numpy()
        co=[load(OUT/f'coherent{case}_{leg}.json') for leg in [0,1]]
        assert all(c['passed'] and c['matcher_checked']==len(s) for c in co)
        pg=load(OUT/f'pixel_gate{case}.json');assert pg['passed'];pix=dict(pg['cache'])
        sim=np.empty((len(s),2,4,9,3));us=np.empty((len(s),2,4));mixed_shas=set()
        for i,ident in enumerate(ids):
            mix=[load(OUT/f'mixed{case}_{l}'/f'{ident}.json') for l in [0,1]]
            for l,v in enumerate(mix):
                assert (v['case'],v['background_leg'],v['input_index'],v['pixel_sha256'])==(case,l,ident,pix[ident])
                mixed_shas.add((v['source_sha256'],v['sample_sha256']))
            corners=[co[0]['results'][str(ident)],mix[1]['result'],mix[0]['result'],co[1]['results'][str(ident)]]
            for k,v in enumerate(corners):sim[i,:,k],us[i,:,k]=obs(v)
        assert mixed_shas=={(co[0]['source_sha256'],sha(samp))} and co[0]['source_sha256']==co[1]['source_sha256']
        model=OUT/f'model{case}.npz';rec=load(model.with_suffix('.json'))
        assert rec['data_sha256']==sha(model) and rec['sample_sha256']==sha(samp)
        with np.load(model) as z:
            np.testing.assert_array_equal(z['input_index'],ids);np.testing.assert_array_equal(z['weights'],s.weight)
            ix=[list(z['cuts']).index(k) for k in CUT_NAMES];uncut=list(z['cuts']).index('uncut')
            for ai,arm in enumerate(z['arms']):
                moments=z['moments'][ai].transpose(1,0,2,3)
                merged=np.concatenate([moments[...,:1],moments[...,1:3]+moments[...,3:5]],-1)
                v=merged[:,:,ix]
                prob=z['probability'][[0,0,1,1]].T
                predicted=np.repeat((v*prob[:,:,None,None])[:,None],2,axis=1)
                actual=v[:,None]*us[:,:,:,None,None]
                measured=merged[:,:,uncut][:,None,: ,None,:]*sim[...,:1]
                for mode,x in [('predicted_generated',predicted),('actual_generated',actual),('actual_measured_selection',measured)]:
                    mods.setdefault((str(arm),mode),[]).append(x)
        allsim.append(sim);s['case']=case
        # Fixed sky quadrants subdivided 4x4; 18181-square source geometry.
        s['block']=np.clip((s.x0*4/18181).astype(int),0,3)+4*np.clip((s.y0*4/18181).astype(int),0,3)
        rows.append(s);receipts.append(dict(case=case,sample_sha256=sha(samp),coherent_model_replay=rec['max_coherent_response_replay'],measured_sha256=co[0]['source_sha256']))
    s=pd.concat(rows);w,case,st,block=[s[k].to_numpy() for k in ['weight','case','stratum','block']]
    sim=np.concatenate(allsim);args=(w,case,st,block)
    out=dict(cases=CASES,cuts=CUT_NAMES,roles=['primary','neighbours','coherent'],components=['R11','R21'],corners=['++','+-','-+','--'],receipts=receipts,sim=summarize(sim,None,*args),arms={})
    for (name,mode),xx in mods.items():
        x=np.concatenate(xx);v=summarize(sim,x,*args)
        rr=estimate(x,w);den=rr[2,CENTER,0]
        v['model_response']=entry(rr);v['coherent_m_percent_central']=100*v['value'][2][CENTER][0]/den
        m_jk=[]
        for c in CASES:
            a=estimate(sim,w*(case!=c))[2,CENTER,0];b=estimate(x,w*(case!=c))[2,CENTER,0];m_jk.append(100*(a/b-1))
        v['coherent_m_case_se_pp']=float(np.sqrt(.75*np.sum(np.square(m_jk-np.mean(m_jk)))))
        v['optimizer_rep_residuals_central_R11']=[entry((estimate(sim[:,i:i+1],w)-estimate(x[:,i:i+1],w))[:,CENTER,0]) for i in [0,1]]
        out['arms'].setdefault(name,{})[mode]=v
    # Precision planning only: sampling component on these finite frames,
    # proportional allocation across strata; no extrapolation of case scatter.
    out['precision_planning']='Pilot plug-in estimates with finite-population correction and capped stratum counts, ordered by case then detected-union/complement. SE 0.003 is in absolute response units. No extrapolation of case scatter, optimizer precision, or generalization; no larger run is authorized by an allocation estimate.'
    (H/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['# Independent primary and neighbour shear on four development fields','',
           '2,048 fixed probability-sampled primaries, two paired optimizer seeds. R11 residual = simulation minus model. Each ± is a whole-source-case jackknife SE; sampling SEs condition on the four original images, and must not be added to case SEs. Central cut: measured/generated radius > 3 pixels and magnitude < 25 as indicated.','',
           '| Flow | Usability / selection | Primary residual ± case SE | Neighbour residual ± case SE | Coherent residual ± case SE | Sampling SEs: primary / neighbour / coherent |',
           '|---|---|---:|---:|---:|---|']
    for name,modes in out['arms'].items():
        for mode,v in modes.items():
            cells=[f"{v['value'][j][CENTER][0]:+.6f} ± {v['case_jackknife_se'][j][CENTER][0]:.6f}" for j in range(3)]
            ss=' / '.join(f'{v["stratified_sampling_se"][j][CENTER][0]:.5f}' for j in range(3))
            lines.append('| '+' | '.join([name,mode,*cells,ss])+' |')
    lines.extend(['','See report.json for all cuts, R21, paired covariance, optimizer and spatial sampling diagnostics. This g1-only variance pilot cannot establish R22 or independent calibration. No model or larger sample is selected.'])
    (H/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
