"""Case-resampled matched-flow response and profile comparisons."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/fs2_galsbi_transfer_20261105')


def responses(base, cross, normal, count, own=True):
    """Return exactly additive cell contributions; leading dimension is resample."""
    total = base.sum(2)  # sample, branch, cut, (mass,e1,e2)
    if not own:
        total = np.broadcast_to(total[:, :1], total.shape)
    mean = total[...,1:]/total[...,:1]
    centered = cross[...,1:,:] - mean[:,:,None,:,:,None]*cross[...,:1,:]
    inverse = np.linalg.inv(normal.sum(1))
    result = np.einsum('nbckoi,nij->nbckoj',centered,inverse)
    result *= (count.sum(1)[:,None,None]/total[...,0])[:,:,None,:,None,None]
    return result


def ent(x):
    return {'value':np.asarray(x[0]).tolist(),'case_se':np.std(x[1:],axis=0,ddof=1).tolist()}


def reduce_family(data, weights, reweight):
    cols = {k:v*reweight.reshape((1,1,25)+(1,)*(v.ndim-3))
            for k,v in data.items() if k in ['base','cross']}
    norm = data['normal']*reweight[None,:,None,None]
    count = data['count']*reweight[None,:]
    results = {mode:[] for mode in ['own','measured_reference']}
    for start in range(0,len(weights),250):
        w = weights[start:start+250]
        b = np.einsum('nc,cb...->nb...',w,cols['base'])
        c = np.einsum('nc,cb...->nb...',w,cols['cross'])
        n = np.einsum('nc,c...->n...',w,norm)
        ct = w@count
        for mode in results:
            results[mode].append(responses(b,c,n,ct,mode=='own'))
    report = {}
    for mode, blocks in results.items():
        r = np.concatenate(blocks)
        total = r.sum(2)
        # Branches: measured, mean of model banks, bank0, bank1.
        delta = r[:,0]-r[:,1]
        dr = total[:,0]-total[:,1]
        cuts = {}
        for k,name in enumerate(data['cuts']):
            rs,rm = total[:,0,k],total[:,1,k]
            cuts[name] = {
                'simulation_matrix':ent(rs),'model_matrix':ent(rm),'delta_matrix':ent(dr[:,k]),
                'm_R11_percent':ent(100*(rs[:,0,0]/rm[:,0,0]-1)),
                'm_halftrace_percent':ent(100*(np.trace(rs,axis1=-2,axis2=-1)/np.trace(rm,axis1=-2,axis2=-1)-1)),
                'delta_R11_size_contribution':ent(delta[:,:,k,0,0].reshape(-1,5,5).sum(2)),
                'delta_R11_crowd_contribution':ent(delta[:,:,k,0,0].reshape(-1,5,5).sum(1)),
                'model_bank_R11_difference':float(total[0,2,k,0,0]-total[0,3,k,0,0]),
                'selected_fraction_sim_size':(cols['base'][:,0,:,k,0].sum(0).reshape(5,5).sum(1)/cols['base'][:,0,:,k,0].sum()).tolist(),
            }
        # Exact bin closure is an algebraic invariant on every resample.
        np.testing.assert_allclose(delta.sum(1),dr,atol=1e-12,rtol=1e-10)
        report[mode] = cuts
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:
        # Constant selected mass with known 2x2 response, split over two cells.
        b=np.zeros((1,1,2,1,3));b[...,0]=[1.]
        norm=np.array([[[[1.,0.],[0.,2.]],[[2.,0.],[0.,1.]]]])
        R=np.array([[.7,.02],[-.03,.8]])
        c=np.zeros((1,1,2,1,3,2))
        c[0,0,:,0,1:]=R@norm[0]
        got=responses(b,c,norm,np.array([[1.,1.]]))
        np.testing.assert_allclose(got.sum(2)[0,0,0],R)
        print('Known matrix and disjoint-cell sum passed.');return
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    data={};protocols={}
    for family in ['fs2','galsbi']:
        root=ROOT/family
        protocol=json.loads((root/'protocol.json').read_text());protocols[family]=protocol
        assert (root/'completed.json').exists()
        d={k:[] for k in ['base','cross','normal','count','profile']}
        invalid=[]
        for case in protocol['cases']:
            with np.load(root/f'case{case:03d}.npz') as z:
                for k in d:d[k].append(z[k])
                invalid.append(z['nonpositive_radius'] if 'nonpositive_radius' in z else np.zeros((3,2),int))
        d={k:np.array(v) for k,v in d.items()}
        for k in ['base','cross','profile']:
            d[k]=np.concatenate([d[k][:,:1],d[k][:,1:].mean(1,keepdims=True),d[k][:,1:]],axis=1)
        d['cuts']=protocol['cuts'];d['nonpositive_radius']=np.sum(invalid,axis=0).tolist()
        data[family]=d
    result={'definitions':'simulation minus model; 10000 whole-case bootstrap draws, independent across families; fixed models; 2 banks of 16 draws; no training uncertainty',
            'families':{},'profiles':{},'protocols':protocols}
    for j,(family,d) in enumerate(data.items()):
        n=len(d['count']);w=np.vstack([np.ones(n),np.random.default_rng(20261105+j).multinomial(n,np.full(n,1/n),10000)])
        f=d['count'].sum(0);f=f/f.sum()
        other=data['galsbi' if family=='fs2' else 'fs2']['count'].sum(0);other=other/other.sum()
        rw=np.divide(other,f,out=np.zeros_like(f),where=f>0)
        result['families'][family]={'native':reduce_family(d,w,np.ones(25)),
            'other_family_size_crowding_mix':reduce_family(d,w,rw),
            'fixed_reweighting':rw.tolist(),'other_mass_without_support':float(other[f==0].sum()),
            'matched_pair_count':int(d['count'].sum()),'nonpositive_generated_radius':d['nonpositive_radius']}
        # Profile comparisons use baseline-only, pairing-free moments.
        prof=d['profile'].sum(0).reshape(4,2,5,32,7)
        with np.errstate(invalid='ignore',divide='ignore'):
            align=prof[...,2]/prof[...,1]
            cov=prof[...,5]/prof[...,0]-(prof[...,2]/prof[...,0])*(prof[...,3]/prof[...,0])
        result['profiles'][family]={'alignment_slope':np.nan_to_num(align).tolist(),
            'dot_logradius_covariance':np.nan_to_num(cov).tolist(),'counts':prof[...,0].tolist()}
    (ROOT/'flow_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for family in data:
        print(family,'pairs',result['families'][family]['matched_pair_count'])
        for name,v in result['families'][family]['native']['own'].items():
            print(name,'delta R11',v['delta_matrix']['value'][0][0],v['delta_matrix']['case_se'][0][0],
                  'm',v['m_R11_percent'],'draw bank diff',v['model_bank_R11_difference'])


if __name__=='__main__':main()
