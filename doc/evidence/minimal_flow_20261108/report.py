"""Paired development pilot; bootstrap bands do not establish independent acceptance."""
import json
import os
import types
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
c=types.ModuleType('minimal_report_common');c.__file__=str(H/'common.py')
exec(compile((H/'common.py').read_bytes(),c.__file__,'exec'),c.__dict__)

def response(data,weights):
    q=np.einsum('bc,clkq->blkq',weights,data)
    mean=(q[...,1:3]+q[...,3:5])/q[...,:1] if q.shape[-1]==5 else q[...,1:3]/q[...,:1]
    return np.stack([(mean[:,0]-mean[:,1])/.04,(mean[:,2]-mean[:,3])/.04],-1)

def scalar_response(data,weights):
    # Independent Python sums and explicit per-leg normalizations.
    out=np.zeros((9,2,2))
    for k in range(9):
        for axis in range(2):
            for shape in range(2):
                means=[]
                for leg in (2*axis,2*axis+1):
                    denominator=sum(float(w)*float(row[leg,k,0]) for row,w in zip(data,weights))
                    numerator=sum(float(w)*(float(row[leg,k,1+shape])+(float(row[leg,k,3+shape]) if data.shape[-1]==5 else 0.)) for row,w in zip(data,weights))
                    means.append(numerator/denominator)
                out[k,shape,axis]=(means[0]-means[1])/.04
    return out

def main():
    assert os.environ.get('SLURM_JOB_ID');c.verify();cfg=c.config()
    cases=cfg['pilot_cases'];names=[x['name'] for x in cfg['arms']]+['teacher'];models=[];sim=[];receipts=[]
    for case in cases:
        legs=[];ss=[]
        for axis in (1,2):
            p=c.ROOT/'constant'/f'axis{axis}'/f'case{case}.npz';r=json.loads(p.with_suffix('.json').read_text())
            assert c.sha(p)==r['data_sha256'] and r['source_sha256']==c.sha(H/'evaluate.py') and r['manifest_sha256']==c.sha(H/'manifest.json')
            assert r['teacher_replay_max_abs']<1e-9
            for arm in cfg['arms']:assert r['models'][arm['name']]==c.sha(c.ROOT/arm['name']/'final.pt')
            with np.load(p) as z:
                np.testing.assert_array_equal(z['names'],names);np.testing.assert_array_equal(z['cuts'],[x[0] for x in cfg['cuts']]);legs.append(z['model']);ss.append(z['sim'])
            receipts.append(r)
        models.append(np.concatenate(legs,axis=1));sim.append(np.concatenate(ss,axis=0))
    model=np.array(models);sim=np.array(sim);n=len(cases)
    weights=np.vstack([np.ones(n),np.random.default_rng(202611081950).multinomial(n,np.full(n,1/n),10000),1-np.eye(n)])
    rs=response(sim,weights);r=[response(model[:,i],weights) for i in range(len(names))]
    checks=0
    for data,vector in [(sim,rs)]+[(model[:,i],r[i]) for i in range(len(names))]:
        for wi in [0]+list(range(len(weights)-n,len(weights))):
            np.testing.assert_allclose(scalar_response(data,weights[wi]),vector[wi],rtol=1e-10,atol=1e-11);checks+=1
    bias=[100*(np.diagonal(rs,axis1=-2,axis2=-1)/np.diagonal(v,axis1=-2,axis2=-1)-1) for v in r]
    def summary(v):
        jk=v[-n:];return dict(value=v[0].tolist(),jackknife_se=np.sqrt((n-1)/n*((jk-jk.mean(0))**2).sum(0)).tolist(),bootstrap_ci95=np.quantile(v[1:10001],[.025,.975],axis=0).tolist())
    teacher=bias[-1];rms0=np.sqrt((teacher**2).mean((1,2)));arms={}
    for i,arm in enumerate(cfg['arms']):
        b=bias[i];d=b-teacher;degrade=np.abs(b)-np.abs(teacher);rms=np.sqrt((b**2).mean((1,2)))
        # Simultaneous one-sided bootstrap band over the 18 absolute-bias differences.
        critical=float(np.quantile((degrade[1:10001]-degrade[0]).max((1,2)),.95))
        upper=degrade[0]+max(0.,critical)
        rms_upper=float((rms-rms0)[0]+max(0.,np.quantile((rms-rms0)[1:10001]-(rms-rms0)[0],.95)))
        passes=bool(upper.max()<=cfg['tolerances']['per_component_cut_pp'] and rms_upper<=cfg['tolerances']['rms_pp'])
        arms[arm['name']]=dict(matrix=summary(r[i]),m_percent=summary(b),delta_m_pp=summary(d),absolute_bias_degradation_pp=summary(degrade),
            nine_cut_rms_percent=summary(rms),rms_delta_pp=summary(rms-rms0),simultaneous_upper_degradation_pp=upper.tolist(),rms_upper_degradation_pp=rms_upper,
            pilot_screen_pass=passes,training=json.loads((c.ROOT/arm['name']/'completed.json').read_text()))
    result=dict(cases=cases,cuts=cfg['cuts'],measured_matrix=summary(rs),teacher_matrix=summary(r[-1]),teacher_m_percent=summary(teacher),
                teacher_nine_cut_rms_percent=summary(rms0),arms=arms,receipts=receipts,independent_scalar_matrix_checks=checks,
                limitation='Four reused development scenes: preliminary screen only. Full 40-scene paired evaluation required before replacing baseline.',
                manifest_sha256=c.sha(H/'manifest.json'))
    c.write(H/'report.json',result)
    lines=['# Minimal-flow preliminary comparison','','## Four-scene screen','',result['limitation'],'',
           'm=100*(Rmeasured_ii/Rmodel_ii−1), percent. Errors: whole-scene jackknife. Independent per-leg selection and normalization.',
           '','| Model | Parameters | Central m1 (%) | Central m2 (%) | Nine-cut RMS (%) | Provisional screen |','|---|---:|---:|---:|---:|---|']
    def f(v,se):return f'{v:+.4f} ±{se:.4f}'
    for i,name in enumerate(names):
        b=summary(bias[i]);rms=summary(np.sqrt((bias[i]**2).mean((1,2))))
        count=arms[name]['training']['counts']['student'] if name!='teacher' else next(iter(arms.values()))['training']['counts']['teacher']
        flag=str(arms[name]['pilot_screen_pass']) if name!='teacher' else 'Reference'
        lines.append(f'| {name} | {count} | {f(b["value"][4][0],b["jackknife_se"][4][0])} | {f(b["value"][4][1],b["jackknife_se"][4][1])} | {f(rms["value"],rms["jackknife_se"])} | {flag} |')
    lines.extend(['','## All nine cuts','','| Model | Radius > pixels | Magnitude < | m1 (%) | m2 (%) |','|---|---:|---:|---:|---:|'])
    for i,name in enumerate(names):
        b=summary(bias[i])
        for k,cut in enumerate(cfg['cuts']):
            lines.append(f'| {name} | {cut[1][1]} | {cut[2]} | {f(b["value"][k][0],b["jackknife_se"][k][0])} | {f(b["value"][k][1],b["jackknife_se"][k][1])} |')
    lines.extend(['',f'Independent scalar point/leave-one-out matrix checks passed: {checks}. All live teacher replays passed.',
                  '[Protocol](rule.md#performance-contract) and [response conventions](../../CONVENTIONS.md#6-response-estimators).'])
    (H/'results.md').write_text('\n'.join(lines)+'\n')
    print('MINIMAL_PILOT_REPORT_DONE', {name:arm['pilot_screen_pass'] for name,arm in arms.items()},flush=True)
if __name__=='__main__':main()
