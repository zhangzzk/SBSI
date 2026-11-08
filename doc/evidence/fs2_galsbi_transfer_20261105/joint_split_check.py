"""Saved FS2 joint result by its flow and classifier training membership."""
import json
import os
from pathlib import Path
import numpy as np

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Slurm required')
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
O=D/'fs2_galsbi_transfer_20261105'
S=D/'fs2_galsbi_split_20261103/fs2'
axes=json.loads((S/'axes.json').read_text());q={name:i for i,name in enumerate(axes['Q'])}
cases=np.arange(40,120)
split=json.loads((D/'v36_f8_20260924_v1/harmonized/domain/manifest.json').read_text())['split']
rows=[]
for case in cases:
    with np.load(S/f'case{case:03d}.npz') as z:rows.append(z['size'].sum(2))
rows=np.array(rows) # case,leg,cut,quantity
groups={'all':np.ones(80,bool),'flow_train':np.isin(cases,split['train_cases']),
        'flow_validation':np.isin(cases,split['validation_cases']),
        'classifier_train':cases<100,'classifier_tune':cases>=100}
report={'definition':'Saved FS2 V3.9 joint response, 80 reused scenes; groups do not provide an independent test.',
        'groups':{}}
for j,(name,mask) in enumerate(groups.items()):
    x=rows[mask];n=len(x)
    w=np.vstack([np.ones(n),np.random.default_rng(20261105+j).multinomial(n,np.full(n,1/n),10000)])
    v=np.einsum('sc,clkp->slkp',w,x)
    def response(numerator,denominator):
        mean=v[...,q[numerator]]/v[...,q[denominator]]
        return (mean[:,0]-mean[:,1])/.04
    sim=response('sim_e1','sim_n');mod=response('c8_e1','c8_n');true=response('tf_e1','tf_n')
    result={}
    for k,cut in enumerate(['uncut','radius3_mag25.8','radius3.75_mag25.8']):
        result[cut]={}
        for label,array in [('delta_R',sim-mod),('m_percent',100*(sim/mod-1)),('true_U_m_percent',100*(sim/true-1))]:
            result[cut][label]={'value':float(array[0,k]),'case_se':float(array[1:,k].std(ddof=1))}
    report['groups'][name]={'cases':cases[mask].tolist(),'results':result}
np.testing.assert_allclose(report['groups']['all']['results']['radius3_mag25.8']['m_percent']['value'],
                           .020769635769424255,atol=1e-8)
(O/'joint_membership.json').write_text(json.dumps(report,indent=2)+'\n')
for name,v in report['groups'].items():
    print(name,len(v['cases']),v['results']['radius3_mag25.8'],flush=True)
