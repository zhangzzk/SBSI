"""Summarize seeded replay, optimizer flags and where repeat scatter occurs."""
import os,json
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/constant_split_20261105/fit_audit')
raw=json.loads((D/'fit_flags.json').read_text());rows=raw['records'];delta=[];good=[];nfev=[];flags={}
for r in rows:
    a,b=r['fits'][0],r['fits'][2]
    for x in [a,b]:
        flag=str(x.get('flags','missing'));flags[flag]=flags.get(flag,0)+1
        if isinstance(x.get('nfev'),(int,float)):nfev.append(x['nfev'])
    ga=np.array(a['g']);gb=np.array(b['g'])
    if np.isfinite([ga,gb]).all() and ga@ga<1 and gb@gb<1:
        delta.append(ga-gb);good.append(a.get('flags')==0 and b.get('flags')==0)
delta=np.array(delta);good=np.array(good)
out={'seeded_replay_passed':raw['seeded_replay_passed'],'flag_sample_rows':len(rows),'flags':flags,'flag_sample_both_success_fraction':float(good.mean()),'both_success_component_rms':float(np.sqrt(np.mean(delta[good]**2))) if good.any() else None,'nfev_quantiles':dict(zip(['min','median','p90','max'],np.quantile(nfev,[0,.5,.9,1]).tolist())) if nfev else {},'all_sample':{}}
records=[]
for case in [1000,1040,1080,1120]:
    for leg in [0,1]:
        with np.load(D/f'case{case}_leg{leg}.npz') as z:
            a,b=z['new'];valid=np.isfinite(a).all(1)&np.isfinite(b).all(1)&((a*a).sum(1)<1)&((b*b).sum(1)<1)&(z['radius']>3)&(z['mag']<25)
            records.append(np.column_stack([z['mag'][valid],z['radius'][valid],(a-b)[valid]]))
x=np.concatenate(records);diff=x[:,2:];norm=np.linalg.norm(diff,axis=1)
out['all_sample']={'n':len(x),'component_rms':float(np.sqrt(np.mean(diff**2))),'vector_change_quantiles':dict(zip(['p25','p50','p75','p90','p99'],np.quantile(norm,[.25,.5,.75,.9,.99]).tolist())),'fraction_vector_change_gt_0.01':float((norm>.01).mean()),'fraction_vector_change_gt_0.1':float((norm>.1).mean()),'bins':[]}
for label,col,edges in [('MAG_AUTO',0,[-np.inf,23,24,24.5,25]),('radius',1,[3,3.2,4,6,np.inf])]:
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(x[:,col]>=lo)&(x[:,col]<hi)
        out['all_sample']['bins'].append({'variable':label,'lo':str(lo),'hi':str(hi),'n':int(m.sum()),'component_rms':float(np.sqrt(np.mean(diff[m]**2))) if m.any() else None,'share_squared_g1_change':float(np.square(diff[m,0]).sum()/np.square(diff[:,0]).sum())})
(O/'fit_details_report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
