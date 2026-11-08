"""Full-case standalone alignment and hard-selection response of fixed endpoints."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
OUT=ROOT/'alignment_regularization_shift_20261106'
spec=importlib.util.spec_from_file_location('reducer',HERE.parent/'fs2_galsbi_transfer_20261105/reduce_flow.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def load(directory):
    assert (directory/'completed.json').exists()
    p=json.loads((directory/'protocol.json').read_text());cases=p['cases'];data={}
    for case in cases:
        with np.load(directory/f'case{case:03d}.npz') as z:
            for key in ['base','cross','normal','count','profile','shape_cross','selection_cross']:
                if key in z:data.setdefault(key,[]).append(z[key])
    data={k:np.asarray(v) for k,v in data.items()}
    for key in ['base','cross','profile','shape_cross','selection_cross']:
        if key in data:data[key]=np.concatenate([data[key][:,:1],data[key][:,1:].mean(1,keepdims=True)],axis=1)
    return p,data

def reduce(p,data,w):
    out={};raw={};b=data['base'];c=data['cross'];n=len(b)
    profiles=data['profile'].reshape(n,2,2,5,32,7)[:,:,:,1,:,:3]
    agg=(w@profiles.reshape(n,-1)).reshape(len(w),2,2,32,3)
    align=np.divide(agg[...,2],agg[...,1],out=np.zeros_like(agg[...,2]),where=agg[...,1]>0)
    err=align[:,1]-align[:,0]
    out['small_alignment_profile']={'definition':'Intrinsic circularized Re[0.15,0.25), own output radius bins; A=sum(e dot e_intrinsic)/sum(intrinsic power); error=model-simulation',
        'magnitude_limits':[25.,25.8],'radius_edges':p['profile_radius_edges'],
        'error':r.ent(err),'simulation':r.ent(align[:,0]),'model':r.ent(align[:,1]),
        'counts':agg[0,...,0].tolist()}
    raw['alignment']=err
    matrices={key:[] for key in ['own','measured_reference','shape','selection','own_size']}
    for lo in range(0,len(w),250):
        ww=w[lo:lo+250]
        bb=np.einsum('sc,cb...->sb...',ww,b);cc=np.einsum('sc,cb...->sb...',ww,c)
        nn=np.einsum('sc,c...->s...',ww,data['normal']);ct=ww@data['count']
        own=r.responses(bb,cc,nn,ct);matrices['own'].append(own.sum(2))
        matrices['own_size'].append(own.reshape(len(ww),2,5,5,len(p['cuts']),2,2).sum(3))
        matrices['measured_reference'].append(r.responses(bb,cc,nn,ct,False).sum(2))
        if 'shape_cross' in data:
            sc=np.einsum('sc,cb...->sb...',ww,data['shape_cross'])
            se=np.einsum('sc,cb...->sb...',ww,data['selection_cross'])
            shape=r.responses(bb,sc,nn,ct).sum(2);selection=r.responses(bb,se,nn,ct).sum(2)
            np.testing.assert_allclose(shape+selection,own.sum(2),rtol=1e-9,atol=2e-12)
            matrices['shape'].append(shape);matrices['selection'].append(selection)
    matrices={k:np.concatenate(v) for k,v in matrices.items() if v}
    out['responses']={}
    for i,cut in enumerate(p['cuts']):
        one={}
        for name,x in matrices.items():
            xx=x[:,:,:,i] if name=='own_size' else x[:,:,i]
            one[name]={'simulation':r.ent(xx[:,0]),'model':r.ent(xx[:,1]),'error_sim_minus_model':r.ent(xx[:,0]-xx[:,1])}
        out['responses'][cut]=one
    raw.update(matrices)
    return out,raw

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    m=json.loads((HERE/'manifest.json').read_text());paths={a:OUT/(a+'_eval') for a in m['arms']}
    paths.update({'FS2_NLL':ROOT/'alignment_trajectory_20261105/fs2_eval',
                  'old_GalSBI_NLL':ROOT/'alignment_trajectory_20261105/galsbi_eval',
                  'improved_GalSBI_frame':ROOT/'alignment_factorized_20261105/frame_eval'})
    oldp,old=load(ROOT/'alignment_factorized_20261105/cartesian_eval')
    result={'definition':'Existing40 flow-validation cases; full hard-cut forward response. Alignment model-simulation, response simulation-model. Fixed-model whole-case SEs. FS2 is independent-population comparison. Shape/selection response split is an exact finite-pair identity.','arms':{},'paired_changes':{}}
    raw={};case_lists={}
    for a,path in paths.items():
        p,d=load(path);case_lists[a]=p['cases'];n=len(p['cases'])
        w=np.vstack([np.ones(n),np.random.default_rng(202611069201 if a!='FS2_NLL' else 202611069202).multinomial(n,np.full(n,1/n),10000)])
        if a!='FS2_NLL':
            assert p['cases']==oldp['cases']
            for key in ['base','cross','profile']:np.testing.assert_allclose(d[key][:,0],old[key][:,0],rtol=1e-12,atol=1e-8)
            for key in ['normal','count']:np.testing.assert_allclose(d[key],old[key],rtol=1e-12,atol=1e-8)
        out,v=reduce(p,d,w);out['protocol']=p;result['arms'][a]=out
        if a in m['arms']:raw[a]=v
        print('REDUCED',a,flush=True)
    result['cases']=case_lists
    for a in m['arms']:
        if a=='baseline':continue
        result['paired_changes'][a+'_minus_baseline']={
            'alignment':r.ent(raw[a]['alignment']-raw['baseline']['alignment']),
            'model_response':{k:r.ent(raw[a][k][:,1]-raw['baseline'][k][:,1]) for k in ['own','measured_reference','shape','selection','own_size']}}
    (OUT/'standalone_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DONE',flush=True)

if __name__=='__main__':main()
