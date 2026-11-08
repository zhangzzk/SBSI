"""Completed training trajectories, coarse population comparison and responses."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_trajectory_20261105')
spec=importlib.util.spec_from_file_location('response_reduce',HERE.parent/'fs2_galsbi_transfer_20261105/reduce_flow.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def safe(v):
    a=np.asarray(v);out=a.astype(object);out[~np.isfinite(a)]=None;return out.tolist()

def summarize(z):
    p=z['profile'].sum(2)
    # Pool sufficient statistics before taking any ratios.
    p=np.stack([p[0],p[1:].mean(0)])
    with np.errstate(divide='ignore',invalid='ignore'):
        a=p[...,2]/p[...,1];neg=p[...,3]/p[...,0]
        nll=-z['density'][...,1].sum(1)/z['density'][...,0].sum(1)
    return {'alignment':safe(a),'alignment_error':safe(a[1]-a[0]),'negative_fraction':safe(neg),
            'counts':p[...,0].tolist(),'power':p[...,1].tolist(),'local_nll_by_measured_bin':safe(nll),
            'local_nll_bin_counts':z['density'][...,0].sum(1).tolist()}

def bootstrap_error(z):
    p=z['profile'];nc=p.shape[2]
    w=np.vstack([np.ones(nc),np.random.default_rng(202611056001).multinomial(nc,np.full(nc,1/nc),10000)])
    # Only the predeclared bin, keeping case units rather than galaxies.
    total=np.einsum('sc,bmck->sbmk',w,p[:,:,:,8])
    sim=total[:,0];mod=total[:,1:].mean(1)
    delta=mod[...,2]/mod[...,1]-sim[...,2]/sim[...,1]
    return r.ent(delta)

def read_trajectories(arms):
    result={}
    for arm in arms:
        directory=ROOT/arm;assert (directory/'completed.json').exists()
        history=json.loads((directory/'history.json').read_text())
        records=[]
        for h in history:
            ep=h['epoch'];base=directory
            if arm=='galsbi_fs2rate' and ep<20:base=ROOT/'galsbi'
            paths=[base/f'{split}_epoch{ep:03d}.npz' for split in ['train','validation']]
            if not all(p.exists() for p in paths):continue
            item={'epoch':ep,'nll':h['validation']['nll'],'lr_used':h['lr_used'],'lr_next':h['lr_next']}
            for split,path in zip(['train','validation'],paths):
                with np.load(path) as z:item[split]=summarize(z)
            records.append(item)
        references={}
        for split in ['train','validation']:
            with np.load(directory/f'{split}_epoch-01.npz') as z:references[split]=summarize(z)
        original=Path(json.loads((directory/'protocol.json').read_text())['original_protocol']).parent
        historical=json.loads((original/'nll/history.json').read_text());historical=historical['history'] if isinstance(historical,dict) else historical
        first=next(h for h in history if h['epoch']==1)
        end={}
        for split in ['train','validation']:
            with np.load(directory/f'{split}_epoch140.npz') as z:end[split]=bootstrap_error(z)
        result[arm]={'completion':json.loads((directory/'completed.json').read_text()),'reference':references,
             'first_epoch_nll_difference_from_historical':first['validation']['nll']-historical[0]['validation']['nll'],
             'consumed_rate_changes':[{'epoch':h['epoch'],'rate':h['lr_used']} for i,h in enumerate(history) if i>0 and h['lr_used']!=history[i-1]['lr_used']],
             'lr_changes':[{'epoch':h['epoch'],'used':h['lr_used'],'next':h['lr_next']} for h in history if h['epoch']>0 and h['lr_used']!=h['lr_next']],
             'trajectory':records,'endpoint_alignment_error_bin8':end}
    # Equal mixture of normalized input-stratum masses, restricted to common support.
    with np.load(ROOT/'fs2/validation_epoch140.npz') as f,np.load(ROOT/'galsbi/validation_epoch140.npz') as g:
        nf=f['input_counts'].astype(float);ng=g['input_counts'].astype(float);pf=nf/nf.sum();pg=ng/ng.sum()
        common=(nf>0)&(ng>0);q=(pf+pg)/2;q[~common]=0;q/=q.sum()
        matched={}
        for name,z,pop in [('fs2',f,pf),('galsbi',g,pg)]:
            ratio=np.divide(q,pop,out=np.zeros_like(q),where=pop>0)
            p=(z['strata']*ratio[None,None,:,None,None]).sum(2)
            p=np.stack([p[0],p[1:].mean(0)])
            with np.errstate(divide='ignore',invalid='ignore'):a=p[...,2]/p[...,1]
            matched[name]={'alignment_error':safe(a[1]-a[0]),'counts':p[...,0].tolist(),'unsupported_input_fraction':float(pop[~common].sum()),'weights':ratio.tolist()}
    result['coarse_common_input_mixture']={'definition':'48 intrinsic n/magnitude/ellipticity cells; common input proportions, not equal measured-bin composition or identical physical SEDs','families':matched}
    if 'galsbi_fs2rate' in arms:
        with np.load(ROOT/'galsbi/validation_epoch020.npz') as a,np.load(ROOT/'galsbi_fs2rate/validation_epoch020.npz') as b:
            np.testing.assert_allclose(a['profile'],b['profile'],rtol=1e-12,atol=1e-8)
            result['fork_epoch20_profile_max_abs']=float(np.max(np.abs(a['profile']-b['profile'])))
    return result

def response_reports(arms):
    reports={}
    dirs={'retained_fs2':ROOT.parent/'fs2_galsbi_transfer_20261105/fs2','retained_galsbi':ROOT.parent/'fs2_galsbi_transfer_20261105/galsbi'}
    dirs.update({a:ROOT/(a+'_eval') for a in arms})
    references={}
    for arm,directory in dirs.items():
        assert (directory/'completed.json').exists()
        protocol=json.loads((directory/'protocol.json').read_text());cases=protocol['cases']
        data={k:[] for k in ['base','cross','normal','count','profile']}
        for case in cases:
            with np.load(directory/f'case{case:03d}.npz') as z:
                for k in data:data[k].append(z[k])
        data={k:np.array(v) for k,v in data.items()}
        family='fs2' if arm in ['retained_fs2','fs2'] else 'galsbi'
        if arm.startswith('retained'):references[family]=(cases,{k:v.copy() for k,v in data.items()})
        else:
            rc,ref=references[family];assert cases==rc
            for k in ['base','cross','profile']:np.testing.assert_allclose(data[k][:,0],ref[k][:,0],rtol=1e-12,atol=1e-8)
            for k in ['normal','count']:np.testing.assert_allclose(data[k],ref[k],rtol=1e-12,atol=1e-8)
        for k in ['base','cross','profile']:data[k]=np.concatenate([data[k][:,:1],data[k][:,1:].mean(1,keepdims=True),data[k][:,1:]],axis=1)
        data['cuts']=protocol['cuts'];nc=len(cases)
        w=np.vstack([np.ones(nc),np.random.default_rng(202611056100).multinomial(nc,np.full(nc,1/nc),10000)])
        prof=data['profile'].reshape(nc,4,2,5,32,7)[:,:,:,1]
        moments=np.einsum('sc,cbmk->sbmk',w,prof[:,:,:,8])
        alignment=moments[...,2]/moments[...,1]
        reports[arm]={'response':r.reduce_family(data,w,np.ones(25)),
                      'selected_alignment_error_bin8':r.ent(alignment[:,1]-alignment[:,0]),
                      'selected_bin_counts':moments[0,:2,:,0].tolist()}
        if not arm.startswith('retained'):reports[arm]['validation_nll']=json.loads((directory/'validation_nll.json').read_text())
    return reports

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['native','final'],required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    arms=['fs2','galsbi']+(['galsbi_fs2rate'] if a.phase=='final' else [])
    result={'definition':'Fixed original measurement pipelines. Case SEs omit retraining uncertainty; differing family NLL reference measures. Development only.',
            'trajectories':read_trajectories(arms)}
    if a.phase=='final':result['responses']=response_reports(arms)
    (ROOT/f'{a.phase}_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for arm in arms:print(arm,result['trajectories'][arm]['completion'],result['trajectories'][arm]['endpoint_alignment_error_bin8'],flush=True)
    plot_spec=importlib.util.spec_from_file_location('trajectory_plot',HERE/'plot.py')
    plotting=importlib.util.module_from_spec(plot_spec);plot_spec.loader.exec_module(plotting);plotting.main(a.phase)
    print('DONE',a.phase,flush=True)
if __name__=='__main__':main()
