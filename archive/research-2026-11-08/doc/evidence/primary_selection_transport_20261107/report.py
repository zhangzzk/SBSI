import os,runpy
from pathlib import Path
A=runpy.run_path(str(Path(__file__).with_name('shared.py')));globals().update({k:v for k,v in A.items() if not k.startswith('__')})
def measures(v):
    x=dict(zip(labels,np.moveaxis(v,-1,0)));N=x['count'];u=x['u']/N;var=x['u2']/N-u*u
    assert (var>0).all() and (x['S0']>0).all() and (x['S1']>0).all()
    out={}
    def reg(z,uz):return (uz/N-u*z/N)/(h*var)
    for s in ('S','Sv','Su','SB'):
        y0=x[s+'0'];y1=x[s+'1'];out[s+'_mass0']=y0/N;out[s+'_mass1']=y1/N;out[s+'_raw_R']=(y1-y0)/(h*N)
        if s!='S':
            q0=y0/x['S0'];q1=y1/x['S1'];out[s+'_selected0']=q0;out[s+'_selected1']=q1;out[s+'_selected_R']=(q1-q0)/h
    for name in ('m','lnr','band_m','band_lnr'):
        z=x['delta_'+name];out[name+'_delta_R']=z/(N*h);out[name+'_slope']=reg(z,x['u_delta_'+name])
    out['S_slope']=reg(x['S1']-x['S0'],x['Su1']-x['Su0'])
    out['SB_slope']=reg(x['SB1']-x['SB0'],x['u_delta_SB'])
    out['v_minus_u_selected_R']=out['Sv_selected_R']-out['Su_selected_R']
    # Reconstruct the selected shape response including the normalizing mass term.
    np.testing.assert_allclose(out['Sv_selected_R'],(out['Sv_raw_R']-out['Sv_selected0']*out['S_raw_R'])/out['S_mass1'],atol=4e-13)
    return out
def main():
    assert os.environ.get('SLURM_JOB_ID')
    sources=[ROOT/'matched.json',ROOT/'models.json']
    for p in sources:
        receipt=json.loads(p.read_text())
        for fname,checksum in receipt['hashes'].items():assert sha(fname)==checksum,fname
    with np.load(ROOT/'matched.npz') as z:
        truth=z['measured'].reshape(2,40,5,9,-1)
    with np.load(ROOT/'models.npz') as z:
        val=z['sums'].reshape(3,2,40,5,9,-1)
    w=truth[0,:,:,:,:1]/truth[1,:,:,:,:1]
    assert np.isfinite(w).all() and (w>0).all()
    mc=np.stack([val[:,0],val[:,1],2*val[:,0]-val[:,1]],axis=1)
    modes=['matched_probe','full_measured_stratified_model']; bins=['all','Q1','Q2','Q3','Q4','Q5'];arms=['start','control','weighted']
    names=['measured',*arms,*['error_'+a for a in arms],'weighted_minus_control_error','weighted_minus_start_error','control_minus_start_error']
    path=H/'all_cuts.npz';values={};main={}
    for mi,mode in enumerate(modes):
        for bi,group in enumerate(bins):
            def project(x):
                a=x.sum(1) if bi==0 else x[:,bi-1]
                assert a.shape[:2]==(40,9)
                return measures(np.concatenate([a.sum(0)[None],a.sum(0)[None]-a]))
            t=project(truth[1-mi]);keys=list(t);t=np.stack(list(t.values()),axis=-1) # deletion,cut,metric
            def table(rows):
                a,b,c=[np.stack(list(project(v).values()),axis=-1) for v in rows]
                d,e,f=t-a,t-b,t-c
                return np.stack([t,a,b,c,d,e,f,f-e,f-d,e-d])
            banks=np.stack([table(mc[:,j]*(w if mi else 1)) for j in range(3)])
            jk=banks[0,:,1:];se=np.sqrt(39./40*((jk-jk.mean(1,keepdims=True))**2).sum(1))
            halves=(banks[1,:,0]-banks[2,:,0])/2 # signed paired half-difference, descriptive only
            v=np.stack([banks[0,:,0],se,halves],axis=-1)
            values[mode+'_'+group]=v
            if group=='all':
                main[mode]={name:{k:[float(p) for p in v[i,4,j]] for j,k in enumerate(keys)} for i,name in enumerate(names)}
    np.savez_compressed(path,**values,cuts=cuts,metrics=keys,series=names,statistics=['estimate','case_jackknife_SE','signed_MC_half_difference'])
    write(H/'central.json',main)
    write(H/'completed.json',dict(job=os.environ['SLURM_JOB_ID'],matched=json.loads(sources[0].read_text()),models=json.loads(sources[1].read_text()),
        output={str(p):sha(p) for p in [path,H/'central.json',Path(__file__)]}))
    for mode,t in main.items():
        print(mode,flush=True)
        for k in ['m_slope','lnr_slope','band_m_slope','band_lnr_slope','Su_selected_R','Sv_selected_R','SB_selected0','SB_slope']:
            print(k,'meas,start,ctl,wt,err_wt,err_wt-ctl:',[[round(v,6) for v in t[n][k][:2]] for n in ['measured','start','control','weighted','error_weighted','weighted_minus_control_error']])
if __name__=='__main__':main()
