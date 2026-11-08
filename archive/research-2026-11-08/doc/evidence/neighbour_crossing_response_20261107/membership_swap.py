"""True membership diagnostic: use exactly the stored model means and compare on identical IDs."""
import importlib.util,os
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cross',H/'cross.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
r=c.r
def main():
    assert os.environ.get('SLURM_JOB_ID')
    src=r.load(r.H/'report.json');cross=r.load(H/'crossings.json')
    assert cross['hashes']['cross.py']==r.sha(H/'cross.py')
    sim,us,df=c.read_source(src);w=df.weight.to_numpy();cs=df.case.to_numpy();st=df.stratum.to_numpy()
    mod=[];pred=[];arm='control'
    for case,rec in zip(r.CASES,src['receipts'],strict=True):
        d=df[df.case==case];p=r.OUT/f'model{case}.npz';mr=r.load(p.with_suffix('.json'))
        assert mr['data_sha256']==r.sha(p) and mr['sample_sha256']==rec['sample_sha256']
        with np.load(p) as z:
            np.testing.assert_array_equal(z['input_index'],d.index_input);np.testing.assert_array_equal(z['weights'],d.weight)
            ix=z['arms'].tolist().index(arm);un=z['cuts'].tolist().index('uncut')
            cuts=[z['cuts'].tolist().index(k) for k in r.CUT_NAMES]
            m=z['moments'][ix].transpose(1,0,2,3)
            out=np.concatenate([m[...,:1],m[...,1:3]+m[...,3:5]],axis=-1)
            np.testing.assert_array_equal(out[:,:,un,0],1)
            q=out[:,:,un,1:];mask=sim[cs==case,:,:,:,:1];mod.append(np.concatenate([mask,mask*q[:,None,:,None,:]],axis=-1))
            pr=out[:,:,cuts]*z['probability'][[0,0,1,1]].T[:,:,None,None]
            np.testing.assert_array_equal(pr[:,0,:,0],pr[:,1,:,0]);np.testing.assert_array_equal(pr[:,2,:,0],pr[:,3,:,0])
            pred.append(pr)
    mod=np.concatenate(mod);pred=np.concatenate(pred);old=src['arms'][arm]
    M=c.grouped(mod,us);S=c.grouped(sim,us)
    def paired(wa,need=False):
        a=c.evaluate(S.mean(1),wa,need);b=c.evaluate(M.mean(1),wa,need)
        return tuple(x-y for x,y in zip(a,b)) if need else a-b
    v,inf,sg=paired(w,True);loo=np.array([paired(w*(cs!=z)) for z in r.CASES])
    var=np.zeros_like(v)
    for z in r.CASES:
        for k in np.unique(st):
            take=(cs==z)&(st==k);n=take.sum();N=w[take].sum();var+=N*N*(1-n/N)/n*inf[take].var(0,ddof=1)
    jk=np.sqrt(.75*np.square(loo-loo.mean(0)).sum(0));se=np.sqrt(var)
    stored=old['actual_measured_selection']
    for q,key in [(v,'value'),(jk,'case_jackknife_se'),(se,'stratified_sampling_se')]:
        np.testing.assert_allclose(q[:,0],np.asarray(stored[key])[1],atol=4e-13,rtol=1e-11)
    model=c.summary(M,w,cs,st)
    native=r.estimate(pred[:,None],w)[1]
    np.testing.assert_allclose(native,np.array(old['predicted_generated']['model_response'])[1],atol=4e-13,rtol=0)
    seeds=[(c.evaluate(S[:,i],w)-c.evaluate(M[:,i],w)).tolist() for i in range(2)]
    phys=np.array(cross['fitted']['estimate'])
    # The native model has no neighbour-flip membership changes; the complement uses different populations.
    np.testing.assert_allclose(phys[:,4]+(phys[:,1]-native),np.asarray(old['predicted_generated']['value'])[1],atol=4e-13)
    payload=dict(job=os.environ['SLURM_JOB_ID'],model='original neighbour-conditioned nine-cut start (physical replay control; not frozen continuation)',pins=mr['source_models'][arm],terms=c.terms,cuts=r.CUT_NAMES,components=['R11','R21'],cases=r.CASES,
        definition='Real corner membership, uncut unconditional model shape mean and disk transport at generated photometry, same saved actual_measured_selection mode. Paired measured minus model. This supplies true membership, not the missing conditional dependence of shapes on actual measured photometry.',
        source_sha256=r.sha(Path(__file__)),reference_sha256=r.sha(r.H/'report.json'),crossings_sha256=r.sha(H/'crossings.json'),model_with_actual_events=model,paired_error=dict(estimate=v.tolist(),case_jackknife_se=jk.tolist(),conditional_sampling_se=se.tolist(),per_seed=seeds,fixed_primary_sign=sg.tolist(),individual_cases=[paired(w*(cs==z)).tolist() for z in r.CASES]),native_predicted_neighbour_response=native.tolist())
    c.write(H/'membership_swap.json',payload)
    for i,n in enumerate(c.terms):
        print(n,round(np.array(model['estimate'])[4,i,0],6),*[round(z[4,i,0],6) for z in (v,jk,se)])
if __name__=='__main__':main()
