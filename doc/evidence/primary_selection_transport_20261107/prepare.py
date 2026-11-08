import os,sys,runpy,time
from pathlib import Path
A=runpy.run_path(str(Path(__file__).with_name('shared.py')));globals().update({k:v for k,v in A.items() if not k.startswith('__')})
import pandas as pd
from scripts.train_fixed_g0_flow import load_split,FLOW_FEATURES,FLOW_TARGETS
def main():
    assert os.environ.get('SLURM_JOB_ID');start=time.monotonic()
    domain=C/'galsbi_f8c8_20261102/harmonized/domain';meta=json.loads((domain/'manifest.json').read_text());cases=meta['split']['validation_cases']
    assert tuple(FLOW_FEATURES[:2])==('e1_input_p','e2_input_p') and tuple(FLOW_TARGETS[-2:])==('measured_flux_radius','measured_flux_from_mag_auto')
    prior=json.loads((old/'completed.json').read_text());assert cases==prior['splits']['validation']['cases'] and sha(domain/'manifest.json')==prior['domain_sha256']
    ed=C/'neighbour_flow_selection_20261106/score_edges.json';edges=np.array(json.loads(ed.read_text())['edges'])
    score=np.load(prep/'validation/raw_score.npy',mmap_mode='r'); ctx=np.load(prep/'validation/probe_context.npy',mmap_mode='r')
    idx=np.load(old/'validation/probe_indices.npy');N=len(idx);assert len(cases)==40 and N==262144 and len(np.unique(idx))==N
    mod=load('transport_m',H.parent/'neighbour_conditioned_flow_20261106/model.py')
    pin=cfg['initial_checkpoint'];assert sha(pin['path'])==pin['sha256'];_,condition,_,_=mod.load(pin['path'],device='cpu')
    assert ctx.shape==(2*N,17)
    totals=np.zeros((2,40*5,9,len(labels)));match=[];base=off=0
    gg=np.zeros((N,2));uu=np.zeros(N);bb=np.zeros(N);groups=np.full(N,-1)
    for ci,case in enumerate(cases):
        d=load_split(domain,[case]);pairs=d['pairs'];n=len(pairs);nraw=len(d['target_raw']);raw=d['context_raw'];match+=d['per_case_matching']
        assert np.allclose(np.linalg.norm(d['gamma'],axis=1),h,atol=2e-8,rtol=0)
        direction=d['gamma']/np.linalg.norm(d['gamma'],axis=1)[:,None]
        u=(raw[pairs[:,0],:2]*direction).sum(-1);B=score[base+pairs[:,0]]
        np.testing.assert_array_equal(B,score[base+pairs[:,1]]);q=ci*5+np.digitize(B,edges)
        pos=np.flatnonzero((idx>=off)&(idx<off+n));ix=idx[pos]-off
        jj=pairs[ix].reshape(-1);frame=pd.DataFrame(raw[jj],columns=FLOW_FEATURES);frame[mod.c.FEATURE]=score[base+jj]
        np.testing.assert_array_equal(condition.transform_frame(frame).reshape(len(ix),2,-1),ctx.reshape(N,2,-1)[pos])
        gg[pos]=direction[ix];uu[pos]=u[ix];bb[pos]=B[ix];groups[pos]=q[ix]
        mask=np.zeros(n,bool);mask[ix]=True
        for lo in range(0,n,4096):
            sl=slice(lo,min(n,lo+4096));p0=d['pair_zero_target'][sl,None];p1=d['pair_shear_target'][sl,None]
            assert all(np.isfinite(p).all() and (p[:,:,2:]>0).all() for p in [p0,p1])
            val=features(p0,p1,direction[sl],u[sl],B[sl],np,lambda x:x[:,0,:])
            add(totals[0],q[sl],val)
            if mask[sl].any():add(totals[1],q[sl][mask[sl]],val[mask[sl]])
        base+=nraw;off+=n
        if ci%5==0 or ci==39:print('matched',ci+1,'/40',off,'pairs',round(time.monotonic()-start,1),flush=True)
    assert base==len(score) and off==prior['splits']['validation']['pairs'] and match==prior['splits']['validation']['matching']
    assert (groups>=0).all() and np.array_equal(totals[:, :,4,0].sum(1),[off,N])
    dest=ROOT/'matched.npz';np.savez(dest,measured=totals,group=groups,u=uu,B=bb,direction=gg,cases=cases)
    write(ROOT/'matched.json',dict(job=os.environ['SLURM_JOB_ID'],cases=cases,matching=match,full_pairs=off,probe_pairs=N,context_replay_exact=True,
        hashes={str(p):sha(p) for p in [Path(__file__),H/'shared.py',H/'rule.md',ed,domain/'manifest.json',old/'completed.json',prep/'completed.json',old/'validation/probe_indices.npy',prep/'validation/probe_context.npy',dest]}))
    print('measured and matched probe complete')
if __name__=='__main__':main()
