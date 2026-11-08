"""Separate marginal pass errors, radius/flux association, and scene weighting."""
import json,os,types
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
c=types.ModuleType('photo_distribution_report_common');c.__file__=str(H/'common.py')
exec(compile((H/'common.py').read_bytes(),c.__file__,'exec'),c.__dict__)

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.floating)):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def moments(data,weights):return np.einsum('bc,ckq->bkq',weights,data)

def decompose(p,o):
    # Each is [radius pass, flux pass, joint pass], for all nine cuts.
    dr=p[...,0]-o[...,0];df=p[...,1]-o[...,1]
    radius=.5*(p[...,1]+o[...,1])*dr
    flux=.5*(p[...,0]+o[...,0])*df
    association=(p[...,2]-p[...,0]*p[...,1])-(o[...,2]-o[...,0]*o[...,1])
    np.testing.assert_allclose(radius+flux+association,p[...,2]-o[...,2],rtol=1e-10,atol=2e-15)
    return np.stack([radius,flux,association],-1)

def summary(v,n):
    jk=v[1:];se=np.sqrt((n-1)/n*((jk-jk.mean(0))**2).sum(0))
    return clean(dict(value=v[0],jackknife_se=se))

def main():
    assert os.environ.get('SLURM_JOB_ID');c.verify();cfg=c.config();names=list(cfg['arms']);out={'definition':'Model-minus-measured photometry diagnostics on original usable training-domain rows. Not shear bias.','splits':{}}
    with np.load(c.ROOT/'train/rows.npz') as z:train=z['raw']
    edges={str(j):np.unique(np.quantile(train[:,j],[.2,.4,.6,.8])).tolist() for j in [2,3,4,5,6,7,8]}
    c.write(H/'bin_edges.json',{'training_only_edges':edges,'columns':cfg['raw_features']});del train
    for split in ['train','validation']:
        root=c.ROOT/split;receipt=json.loads((root/'completed.json').read_text());assert receipt['provenance']['manifest_sha256']==c.sha(H/'manifest.json')
        for k,v in receipt['outputs'].items():assert c.sha(root/k)==v
        with np.load(root/'rows.npz') as z:raw=z['raw'];obs=z['observed'];case=z['case'];leg=z['leg'];cases=z['cases'].tolist()
        n=len(cases);ci=np.searchsorted(np.array(sorted(cases)),case);assert len(np.unique(ci))==n
        weights=np.vstack([np.ones(n),1-np.eye(n)])
        prob=np.load(root/'probabilities.npy',mmap_mode='r');rank=np.load(root/'rank.npy',mmap_mode='r');latent=np.load(root/'latent.npy',mmap_mode='r')
        mm=np.load(root/'moments.npy',mmap_mode='r');nll=np.load(root/'photo_nll.npy',mmap_mode='r')
        rr=obs[:,0,None]>np.array([2.8,3.,3.2]);ff=obs[:,1,None]>10**(.4*(30-np.array([24.9,25.,25.1])))
        observed=np.concatenate([rr,ff,(rr[:,:,None]&ff[:,None,:]).reshape(len(obs),9)],-1).astype(float)
        obs3=np.stack([np.repeat(observed[:,:3],3,axis=1),np.tile(observed[:,3:6],(1,3)),observed[:,6:]],-1)
        groups=[('all',np.ones(len(obs),bool)),('g0',leg==0),('g05',leg==1)]
        bins={j:np.digitize(raw[:,j],edges[str(j)]) for j in [2,3,4,5,6,7,8]}
        for j in bins:
            for b in range(len(edges[str(j)])+1):groups.append((cfg['raw_features'][j]+f'/q{b+1}',bins[j]==b))
        # Joint true-size/environment cells refine the association localization.
        for a in range(len(edges['4'])+1):
            for b in range(len(edges['8'])+1):groups.append((f'true_radius_x_score/q{a+1}_q{b+1}',(bins[4]==a)&(bins[8]==b)))
        result={};source_checks=[]
        for ai,name in enumerate(names):
            result[name]={};p0=np.asarray(prob[ai,:,0],dtype=float)
            pred3=np.stack([np.repeat(p0[:,:3],3,axis=1),np.tile(p0[:,3:6],(1,3)),p0[:,6:]],-1)
            predicted_B=p0[:,6:]*raw[:,8,None];observed_B=observed[:,6:]*raw[:,8,None]
            # Independent antithetic half-banks for MC-corrected joint-cut Brier score.
            qa=np.asarray(prob[ai,:,1,6:],float);qb=np.asarray(prob[ai,:,2,6:],float)
            brier=(qa-observed[:,6:])*(qb-observed[:,6:]);half=qa-qb
            rankvalues=np.array(rank[ai],dtype=float);zvalues=np.array(latent[ai],dtype=float)
            diagnostic=np.column_stack([np.asarray(nll[ai]),mm[ai,:,0]-obs[:,0],mm[ai,:,1]-(30-2.5*np.log10(obs[:,1])),
                rankvalues[:,0]/64-.5,rankvalues[:,1]/64-.5,
                (rankvalues[:,0]==0)-1/65,(rankvalues[:,0]==64)-1/65,
                (rankvalues[:,1]==0)-1/65,(rankvalues[:,1]==64)-1/65,
                zvalues[:,0],zvalues[:,1],zvalues[:,0]**2-1,zvalues[:,1]**2-1,zvalues[:,0]*zvalues[:,1],
                (np.square(zvalues).sum(-1)>9.210340371976184)-.01])
            for group,take in groups:
                count=np.bincount(ci[take],minlength=n).astype(float)
                if count.sum()<500:continue
                def aggregate(v):
                    v=np.asarray(v);v=v.reshape(len(v),-1)
                    return np.stack([np.bincount(ci[take],weights=v[take,k],minlength=n) for k in range(v.shape[1])],-1)
                countw=weights@count
                p=(weights@aggregate(pred3))/countw[:,None];o=(weights@aggregate(obs3))/countw[:,None]
                p=p.reshape(n+1,9,3);o=o.reshape(n+1,9,3)
                dec=decompose(p,o)
                ob=weights@aggregate(observed_B);pb=weights@aggregate(predicted_B)
                masso=o[:,:,2]*countw[:,None];massp=p[:,:,2]*countw[:,None]
                with np.errstate(divide='ignore',invalid='ignore'):db=ob/masso-pb/massp
                d=(weights@aggregate(diagnostic))/countw[:,None]
                bs=(weights@aggregate(brier))/countw[:,None];hs=(weights@aggregate(half))/countw[:,None]
                result[name][group]=dict(rows=int(count.sum()),case_count=int((count>0).sum()),
                    predicted_pass=summary(p,n),observed_pass=summary(o,n),joint_pass_error=summary(p[:,:,2]-o[:,:,2],n),
                    joint_pass_decomposition=summary(dec,n),selected_score_measured_minus_model=summary(db,n),
                    diagnostics=summary(d,n),joint_brier=summary(bs,n),half_bank_pass_difference=summary(hs,n))
                if group=='all':
                    # Direct row-level sums independently reproduce pooled reductions.
                    np.testing.assert_allclose(p[0],pred3.mean(0),rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(o[0],obs3.mean(0),rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(db[0],observed_B.sum(0)/observed[:,6:].sum(0)-predicted_B.sum(0)/p0[:,6:].sum(0),rtol=1e-9,atol=1e-11)
                    np.testing.assert_allclose(d[0],diagnostic.mean(0),rtol=1e-10,atol=1e-10)
            # Verify canonical original draw stream against existing saved per-case statistics.
            oldroot=c.ROOT.parent/'neighbour_flow_selection_20261106';oldpath=oldroot/f'{split}.npz'
            if name in ['baseline','nine_weak','neighbour_nine']:
                oldreceipt=json.loads(oldpath.with_suffix('.json').read_text());assert c.sha(oldpath)==oldreceipt['data_sha256']
                with np.load(oldpath) as zz:
                    ix=zz['arms'].tolist().index(name);q=zz['sums'][ix,0].sum((0,1,2))
                    assert q.shape==(9,5)
                    # fields: row count, measured mass, measured B numerator, generated mass, generated B numerator
                    actual=np.column_stack([np.full(9,len(obs)),observed[:,6:].sum(0),observed_B.sum(0),p0[:,6:].sum(0),predicted_B.sum(0)])
                    np.testing.assert_allclose(actual,q,rtol=1e-10,atol=2e-6)
                source_checks.append(name)
            elif name=='latest':
                oldpath=c.ROOT.parent/'conditional_cut_probability_20261107'/f'own_{split}.npz'
                oldreceipt=json.loads(oldpath.with_suffix('.json').read_text());assert c.sha(oldpath)==oldreceipt['data_sha256']
                with np.load(oldpath) as zz:
                    ix=zz['arms'].tolist().index('weighted');q=zz['sums'][ix,0,0].sum((0,1))
                    actual=np.column_stack([np.full(9,len(obs)),observed[:,6:].sum(0),observed_B.sum(0),p0[:,6:].sum(0),predicted_B.sum(0)])
                    np.testing.assert_allclose(actual,q,rtol=1e-10,atol=2e-6)
                source_checks.append(name)
            print('REDUCED',split,name,'groups',len(result[name]),flush=True)
        # Exact scene-cell contributions to selected B discrepancy, using global normalizations.
        contributions={}
        for ai,name in enumerate(names):
            p=np.asarray(prob[ai,:,0,6:],float);o=observed[:,6:]
            nb=np.stack([np.bincount(ci,weights=raw[:,8]*v[:,4],minlength=n) for v in [o,p]],-1)
            mass=np.stack([np.bincount(ci,weights=v[:,4],minlength=n) for v in [o,p]],-1)
            nn=weights@nb;dd=weights@mass;mid_b=.5*(nn[:,0]/dd[:,0]+nn[:,1]/dd[:,1]);mid_d=.5*(dd[:,0]+dd[:,1])
            cells={};total=np.zeros(n+1)
            for group,take in groups:
                if not group.startswith('true_radius_x_score/'):continue
                dm=np.bincount(ci[take],weights=(o[take,4]-p[take,4]),minlength=n)
                dn=np.bincount(ci[take],weights=raw[take,8]*(o[take,4]-p[take,4]),minlength=n)
                v=((weights@dn)-mid_b*(weights@dm))/mid_d;total+=v;cells[group]=summary(v,n)
            np.testing.assert_allclose(total,nn[:,0]/dd[:,0]-nn[:,1]/dd[:,1],rtol=1e-9,atol=1e-12)
            contributions[name]=cells
        out['splits'][split]=dict(groups=result,central_score_error_cell_contributions=contributions,canonical_replay_passed=source_checks,underflow=receipt['underflow'])
    out['diagnostic_columns']=['photo_nll','radius_mean_error_pixels','magnitude_mean_error','radius_mean_rank_minus_half','flux_mean_rank_minus_half',
        'radius_low_rank_excess','radius_high_rank_excess','flux_low_rank_excess','flux_high_rank_excess','latent_z0_mean','latent_z1_mean',
        'latent_z0_second_moment_minus_one','latent_z1_second_moment_minus_one','latent_cross_moment','latent_radial_tail_excess']
    out['decomposition_columns']=['radius_marginal','flux_marginal','radius_flux_association']
    out['limitations']=['Association is within each specified population, including remaining context heterogeneity; not an identified per-object conditional copula.',
        'One measured realization per row; individual prediction residuals are not individual systematic errors.',
        'Original usable training-domain populations; no classifier-error or coherent-shear attribution.',
        'Fixed models and samples; scene jackknife omits retraining variability. Original data are not newly paired-seed shape fits.']
    c.write(H/'report.json',clean(out))
    lines=['# Conditional photometry distribution audit','','## Validation: central cut','','Passing-rate errors are model minus measured, percentage points. B error is measured minus model in reference-neighbour-coefficient units. These are not shear biases.','',
        '| Model | Radius contribution (pp) | Flux contribution (pp) | Association contribution (pp) | Joint passing error (pp) | Selected B error |','|---|---:|---:|---:|---:|---:|']
    for name in names:
        g=out['splits']['validation']['groups'][name]['all'];d=g['joint_pass_decomposition']['value'][4];ds=g['joint_pass_decomposition']['jackknife_se'][4]
        q=g['joint_pass_error'];b=g['selected_score_measured_minus_model'];fmt=lambda v,s:f'{v:+.4f} ±{s:.4f}'
        lines.append('| '+name+' | '+' | '.join(fmt(100*v,100*s) for v,s in zip(d,ds))+f' | {fmt(100*q["value"][4],100*q["jackknife_se"][4])} | {fmt(b["value"][4],b["jackknife_se"][4])} |')
    lines+=['','## Where the central selected-neighbour error comes from','',
        'Largest absolute contributions are descriptive rankings on these validation scenes. Each model’s 25 cells add exactly to its full selected-B error.','']
    for name in names:
        cells=out['splits']['validation']['central_score_error_cell_contributions'][name]
        lines.append(f'### {name}\n')
        for key,v in sorted(cells.items(),key=lambda x:abs(x[1]['value']),reverse=True)[:5]:lines.append(f'- {key}: {v["value"]:+.6f} ±{v["jackknife_se"]:.6f}')
        lines.append('')
    lines+=['## Limits','']+['- '+v for v in out['limitations']]
    (H/'results.md').write_text('\n'.join(lines)+'\n')
    print('PHOTOMETRY_DISTRIBUTION_REPORT_DONE',flush=True)
if __name__=='__main__':main()
