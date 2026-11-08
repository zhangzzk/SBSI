"""Exact normalized retained/entering/exiting physical shear responses."""
import os,json,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('replay_reference',H.parent/'primary_neighbour_replay_20261107/report.py')
F=H.parent/'frozen_shape_neighbour_20261107';P=H.parent/'primary_selection_transport_20261107'
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=1,allow_nan=False)+'\n');tmp.replace(p)
def read_source(src):
    sim=[];us=[];frames=[]
    for case,rec in zip(r.CASES,src['receipts'],strict=True):
        sf=r.OUT/f'sample_{case}.feather';s=pd.read_feather(sf);ids=s.index_input.to_numpy()
        assert r.sha(sf)==rec['sample_sha256'] and rec['case']==case and len(set(ids))==len(ids)==512
        co=[r.load(r.OUT/f'coherent{case}_{l}.json') for l in (0,1)]
        assert all(c['passed'] and c['matcher_checked']==len(s) and c['source_sha256']==rec['measured_sha256'] for c in co)
        pg=r.load(r.OUT/f'pixel_gate{case}.json');assert pg['passed'];pix=dict(pg['cache'])
        x=np.empty((len(s),2,4,9,3));u=np.empty((len(s),2,4))
        for i,gid in enumerate(ids):
            mixes=[r.load(r.OUT/f'mixed{case}_{k}/{gid}.json') for k in (0,1)]
            for k,v in enumerate(mixes):
                assert (v['case'],v['input_index'],v['background_leg'],v['source_sha256'],v['sample_sha256'],v['pixel_sha256'])==(case,gid,k,rec['measured_sha256'],rec['sample_sha256'],pix[gid])
            for j,o in enumerate((co[0]['results'][str(gid)],mixes[1]['result'],mixes[0]['result'],co[1]['results'][str(gid)])):
                x[i,:,j],u[i,:,j]=r.obs(o)
        s['case']=case;sim.append(x);us.append(u);frames.append(s)
    return np.concatenate(sim),np.concatenate(us),pd.concat(frames,ignore_index=True)
terms=['total','retained','entries','exits','migration','both_usable_photometric_crossing','opposite_unusable_crossing']
def grouped(x,u):
    # row, optimizer seed, corner, cut, [full/common/cross-both-usable/cross-opposite-unusable] × [mass,e1,e2]
    other=x[...,0].take([1,0,3,2],axis=2);ou=u.take([1,0,3,2],axis=2)[...,None]
    assert np.all(other<=ou)
    groups=np.stack([x,x*other[...,None],x*((1-other)*ou)[...,None],x*(1-ou)[...,None]],-2)
    np.testing.assert_array_equal(groups[...,1:,:].sum(-2),x)
    return groups
def parts(t):
    den=t[...,0,0,None];cden=t[...,1,0,None]
    f=t[...,0,1:]/den;c=t[...,1,1:]/cden
    a=(t[...,2,1:]-c*t[...,2,0,None])/den
    b=(t[...,3,1:]-c*t[...,3,0,None])/den
    # corner, cut, term, shape. All differences signed later with original ROLES.
    k=np.array([1,0,1,0])[:,None,None]
    return np.stack([f,c,k*(a+b),(1-k)*(a+b),a+b,a,b],-2)
def evaluate(x,w,need=False):
    t=np.einsum('n,nsjkv->sjkv',w,x)
    assert (t[...,1,0]>0).all()
    f=parts(t);sg=r.ROLES[1]
    signed=f*sg[:,None,None,None];val=signed.sum(0)
    for v in [val,*signed]:
        np.testing.assert_allclose(v[:,0],v[:,1]+v[:,4],atol=3e-14)
        np.testing.assert_allclose(v[:,4],v[:,2]+v[:,3],atol=3e-14)
        np.testing.assert_allclose(v[:,4],v[:,5]+v[:,6],atol=3e-14)
    if not need:return val
    infl=np.zeros((len(w),*val.shape));step=1e-18
    for i in range(4):
        for j in range(3):
            z=t.astype(complex);z[:,:,i,j]+=step*1j
            der=sg[:,None,None,None]*np.imag(parts(z))/step
            infl+=np.einsum('nsc,sctv->nctv',x[:,:,:,i,j],der)
    np.testing.assert_allclose(np.einsum('n,nctv->ctv',w,infl),0,atol=2e-12)
    return val,infl,np.stack([2*signed[:2].sum(0),2*signed[2:].sum(0)])
def summary(xs,weights,case,st):
    x=xs.mean(1);v,inf,by_sign=evaluate(x,weights,True)
    jk=np.stack([evaluate(x,weights*(case!=c)) for c in r.CASES]);sd=np.sqrt(.75*np.square(jk-jk.mean(0)).sum(0))
    var=np.zeros_like(v)
    for c in r.CASES:
        for s in np.unique(st):
            a=(case==c)&(st==s);n=a.sum();N=weights[a].sum();assert n>1 and N>=n
            var+=N*N*(1-n/N)/n*inf[a].var(0,ddof=1)
    per=[evaluate(x,weights*(case==c)).tolist() for c in r.CASES]
    seeds=[evaluate(xs[:,i],weights).tolist() for i in range(xs.shape[1])]
    return dict(estimate=v.tolist(),case_jackknife_se=sd.tolist(),conditional_sampling_se=np.sqrt(var).tolist(),individual_cases=per,per_seed=seeds,fixed_primary_sign=by_sign.tolist(),leave_one_case_out=jk.tolist())
def main():
    assert os.environ.get('SLURM_JOB_ID');src=r.load(r.H/'report.json');flags=r.load(P/'neighbour_flags.json')
    assert flags['reference']==r.sha(r.H/'report.json')
    sim,us,df=read_source(src);w=df.weight.to_numpy();case=df.case.to_numpy();st=df.stratum.to_numpy()
    selected=sim[...,0].astype(bool);i0=r.CENTER
    counts=[]
    for k in range(9):
        rs=[]
        for j in range(2):
            both=selected[:,j,:,k];a=both[:,[0,2]];b=both[:,[1,3]]
            rs.append(dict(entries=int((a&~b).sum()),exits=int((~a&b).sum()),distinct=int(np.any(a!=b,axis=1).sum()),common_comparisons=int((a&b).sum())))
        counts.append(rs)
    for j in range(2):
        ref=flags['results']['neighbour'][f'usable_and_cut_seed{j+1}']
        assert all(counts[i0][j][k]==ref[k] for k in ('entries','exits','distinct'))
    fit=summary(grouped(sim,us),w,case,st)
    for key,old in [('estimate','value'),('case_jackknife_se','case_jackknife_se'),('conditional_sampling_se','stratified_sampling_se')]:
        np.testing.assert_allclose(np.array(fit[key])[:,0],src['sim'][old][1],rtol=1e-11,atol=4e-13)
    si=sim.copy();truth=df[['e1_input_rot0','e2_input_rot0']].to_numpy()
    si[...,1:]=si[...,:1]*truth[:,None,None,None,:]
    intrinsic=summary(grouped(si,us),w,case,st)
    np.testing.assert_allclose(np.array(intrinsic['estimate'])[:,1],0,atol=2e-13)
    out=dict(job=os.environ['SLURM_JOB_ID'],hashes={p.name:r.sha(p) for p in [Path(__file__),H/'rule.md',r.H/'report.json',P/'neighbour_flags.json']},cases=r.CASES,cuts=r.CUT_NAMES,terms=terms,components=['R11','R21'],source_sample=src['receipts'],n=len(df),fits=['original repeat 1','original repeat 2'],counts=counts,fitted=fit,intrinsic=intrinsic)
    write(H/'crossings.json',out)
    print('Verified original response, both reference errors, and all counts at the target cut.',flush=True)
    for i,n in enumerate(terms):
        print(n,*[round(np.array(fit[key])[i0,i,0],7) for key in ['estimate','case_jackknife_se','conditional_sampling_se']],flush=True)
if __name__=='__main__':main()
