"""Read-only crossing audit: fix the same intrinsic ellipticity across four existing image interventions."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
OLD=H.parent/'primary_neighbour_replay_20261107'
DATA=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    report=read(OLD/'report.json');cases=report['cases']
    cutcfg=read(H.parent/'neighbour_factor_swap_20261107/manifest.json')['cuts'][:9]
    cuts=[k[0] for k in cutcfg];assert cuts==report['cuts']
    xs=[];ss=[];cs=[];ws=[];es=[];oks=[];model_mass=[];recs=[]
    for case,rec in zip(cases,report['receipts'],strict=True):
        p=DATA/f'sample_{case}.feather';f=pd.read_feather(p)
        zpath=DATA/f'model{case}.npz';md=read(zpath.with_suffix('.json'))
        assert rec['case']==case and rec['sample_sha256']==sha(p)==md['sample_sha256'] and sha(zpath)==md['data_sha256']
        coh=[read(DATA/f'coherent{case}_{leg}.json') for leg in [0,1]]
        assert all(v['source_sha256']==rec['measured_sha256'] for v in coh)
        pix=dict(read(DATA/f'pixel_gate{case}.json')['cache'])
        arr=np.full((len(f),4,2),np.nan)
        for i,id in enumerate(f.index_input):
            mixes=[read(DATA/f'mixed{case}_{leg}/{id}.json') for leg in [0,1]]
            for k,v in enumerate(mixes):
                assert (v['case'],v['background_leg'],v['input_index'],v['pixel_sha256'],v['source_sha256'],v['sample_sha256'])==(case,k,id,pix[id],rec['measured_sha256'],rec['sample_sha256'])
            for j,obj in enumerate([coh[0]['results'][str(id)],mixes[1]['result'],mixes[0]['result'],coh[1]['results'][str(id)]]):
                if obj['detected'] and obj['radius'] is not None and obj['mag'] is not None and obj['radius']>0:
                    arr[i,j]=[obj['radius'],obj['mag']]
        ok=np.isfinite(arr).all((1,2))
        x=np.stack([(arr[:,:,0]>spec[1][1])&(arr[:,:,1]<spec[2]) for spec in cutcfg],axis=-1)*ok[:,None,None]
        with np.load(zpath) as z:
            np.testing.assert_array_equal(z['input_index'],f.index_input)
            ix=[list(z['cuts']).index(k) for k in cuts];m=z['moments'][...,0][:,:,:,ix]
            np.testing.assert_array_equal(m[:,0],m[:,1]);np.testing.assert_array_equal(m[:,2],m[:,3])
            names=z['arms'].tolist();model_mass.append(m.transpose(2,0,1,3)*ok[:,None,None,None])
        xs.append(x);oks.append(ok);ss.extend(f.stratum);cs.extend([case]*len(f));ws.extend(f.weight);es.extend(f.e1_input_rot0);recs.append(rec)
    x=np.concatenate(xs).astype(float);ok=np.concatenate(oks);mod=np.concatenate(model_mass)
    e=np.asarray(es);w=np.asarray(ws);c=np.asarray(cs);st=np.asarray(ss)
    roles=np.array([[1,1,-1,-1],[1,-1,1,-1],[2,0,0,-2]])/.08
    def estimate(m,wt):
        den=np.einsum('n,nsk->sk',wt,m);mu=np.einsum('n,n,nsk->sk',wt,e,m)/den
        value=roles@mu; np.testing.assert_allclose(value[0]+value[1],value[2],atol=2e-13)
        inf=np.einsum('rs,nsk->nrk',roles,m*(e[:,None,None]-mu)/den)
        return value,inf
    def stats(a,b=None):
        def val(wt):
            v,inf=estimate(a,wt)
            if b is not None:
                v2,inf2=estimate(b,wt);v-=v2;inf-=inf2
            return v,inf
        v,inf=val(w)
        loo=np.stack([val(w*(c!=cc))[0] for cc in cases]);jse=np.sqrt(.75*np.square(loo-loo.mean(0)).sum(0))
        var=np.zeros_like(v)
        for cc in cases:
            for s in [0,1]:
                take=(c==cc)&(st==s);n=take.sum();N=w[take].sum()
                var+=N*N*(1-n/N)/n*inf[take].var(0,ddof=1)
        return {'value':v.tolist(),'case_se':jse.tolist(),'conditional_sampling_se':np.sqrt(var).tolist()}
    edges={'primary':[(0,2),(1,3)],'neighbours':[(0,1),(2,3)]}
    counts={}
    for role,pairs in edges.items():
        counts[role]={}
        for k,name in enumerate(cuts):
            diffs=np.stack([x[:,a,k]-x[:,b,k] for a,b in pairs],1)
            counts[role][name]=dict(entries=int((diffs>0).sum()),exits=int((diffs<0).sum()),unique_changers=int(np.any(diffs!=0,axis=1).sum()),comparisons=int(2*ok.sum()))
    out=dict(definition='Common four detected/finite positive radius and finite magnitude. Intrinsic e1 kept constant; weighted cut membership alone changes. No optimizer fits, C8 or actual shear response of measured shapes. Errors: 4-field jackknife and conditional stratified-primary delta-method SE; do not add.',cuts=cuts,roles=list(edges)+['coherent'],sample=len(w),common=int(ok.sum()),common_by_case={str(cc):int(ok[c==cc].sum()) for cc in cases},receipts=recs,counts=counts,observed=stats(x),sim_minus_model={arm:stats(x,mod[:,i]) for i,arm in enumerate(names)},model={arm:stats(mod[:,i]) for i,arm in enumerate(names)})
    out['centre']=k=cuts.index('mag25_radius_gt0.60')
    for arm in names:np.testing.assert_allclose(np.array(out['model'][arm]['value'])[1],0,rtol=0,atol=2e-14)
    p=H/'boundary.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');q.replace(p)
    print('common',ok.sum(),'counts',{z:v[cuts[k]] for z,v in counts.items()},'intrinsic-selection', {k1:[round(y[k],6) for y in vs] for k1,vs in out['observed'].items()},flush=True)
if __name__=='__main__':main()
