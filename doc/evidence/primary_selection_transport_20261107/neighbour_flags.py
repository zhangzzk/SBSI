"""Separate actual detection, usability and the photometric cut on fixed image interventions."""
import runpy,os
from pathlib import Path
A=runpy.run_path(str(Path(__file__).with_name('shared.py')));globals().update({k:v for k,v in A.items() if not k.startswith('__')})
import pandas as pd
r=load('source_observables',H.parent/'primary_neighbour_replay_20261107/report.py')
def main():
    assert os.environ.get('SLURM_JOB_ID')
    report=r.load(r.H/'report.json');allobs=[];allw=[];allcase=[];allstr=[]
    keys=['detected','usable_seed1','usable_seed2','detected_and_cut','usable_and_cut_seed1','usable_and_cut_seed2']
    for case,rec in zip(r.CASES,report['receipts']):
        sf=r.OUT/f'sample_{case}.feather';s=pd.read_feather(sf);co=[r.load(r.OUT/f'coherent{case}_{i}.json') for i in [0,1]]
        assert rec['case']==case and sha(sf)==rec['sample_sha256'] and all(c['passed'] and c['source_sha256']==rec['measured_sha256'] for c in co)
        pix=dict(r.load(r.OUT/f'pixel_gate{case}.json')['cache']);y=np.zeros((len(s),4,len(keys)),int)
        df=r.OUT/f'model{case}.npz';md=r.load(df.with_suffix('.json'));assert sha(df)==md['data_sha256'] and sha(sf)==md['sample_sha256']
        with np.load(df) as z:
            np.testing.assert_array_equal(z['input_index'],s.index_input);probs=z['probability'][[0,0,1,1]]
            np.testing.assert_array_equal(probs[0],probs[1]);np.testing.assert_array_equal(probs[2],probs[3])
            mass=z['moments'][...,0];np.testing.assert_array_equal(mass[:,0],mass[:,1]);np.testing.assert_array_equal(mass[:,2],mass[:,3])
        for i,ident in enumerate(s.index_input):
            mx=[r.load(r.OUT/f'mixed{case}_{j}/{ident}.json') for j in [0,1]]
            for j,o in enumerate(mx):
                assert (o['case'],o['background_leg'],o['input_index'],o['pixel_sha256'],o['sample_sha256'],o['source_sha256'])==(case,j,ident,pix[ident],rec['sample_sha256'],rec['measured_sha256'])
            objects=(co[0]['results'][str(ident)],mx[1]['result'],mx[0]['result'],co[1]['results'][str(ident)])
            for j,o in enumerate(objects):
                v,u=r.obs(o)
                dc=bool(o['detected'] and o['radius'] is not None and o['mag'] is not None and o['radius']>3 and o['mag']<25)
                y[i,j]=[o['detected'],*u,dc,*v[:,r.CENTER,0]]
        allobs.append(y);allw.extend(s.weight);allstr.extend(s.stratum);allcase.extend([case]*len(s))
    y=np.concatenate(allobs);w=np.array(allw);cs=np.array(allcase);st=np.array(allstr)
    out={}
    for role,edges in {'neighbour':[(0,1),(2,3)],'primary':[(0,2),(1,3)]}.items():
        delta=np.stack([y[:,a]-y[:,b] for a,b in edges],axis=1)
        how={}
        for i,key in enumerate(keys):
            d=delta[:,:,i];net=d.mean(1);cross=np.abs(d).mean(1)
            def summary(x):
                val=np.average(x,weights=w);loo=np.array([np.average(x[cs!=case],weights=w[cs!=case]) for case in r.CASES])
                var=0.
                for c in r.CASES:
                    for k in [0,1]:
                        ids=(cs==c)&(st==k);n=ids.sum();N=w[ids].sum();var+=N*N*(1-n/N)/n*x[ids].var(ddof=1)
                return [float(val),float(np.sqrt(.75*((loo-loo.mean())**2).sum())),float(np.sqrt(var)/w.sum())]
            how[key]=dict(entries=int((d>0).sum()),exits=int((d<0).sum()),distinct=int((d!=0).any(1).sum()),case_distinct=[int((d[cs==c]!=0).any(1).sum()) for c in r.CASES],
                weighted_cross_fraction=summary(cross),weighted_net_change=summary(net))
        out[role]=how
    write(H/'neighbour_flags.json',dict(job=os.environ['SLURM_JOB_ID'],source=sha(__file__),reference=sha(r.H/'report.json'),cases=r.CASES,
        sample=len(y),comparisons_per_role=2*len(y),central_cut=[3.,25.],counts='Unweighted unique IDs, entries and exits within the stratified 2048 sample. Weighted summaries are full-parent event fractions; [estimate,four-case jackknife,conditional sampling SE]. No division by shear; not ellipticity responses.',results=out))
    print(out['neighbour'],flush=True)
if __name__=='__main__':main()
