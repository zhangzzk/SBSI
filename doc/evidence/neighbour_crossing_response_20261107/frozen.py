"""Small reduction of completed per-case sufficient statistics; no model draws."""
from pathlib import Path
import hashlib,importlib.util,json,os
import numpy as np
H=Path(__file__).resolve().parent;F=H.parent/'frozen_shape_neighbour_20261107'
C=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/F.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    assert os.environ.get('SLURM_JOB_ID')
    cfg=read(F/'manifest.json');ref=read(F/'report.json');pins=read(C/'training_completed.json')
    cases=cfg['cases'];cuts=cfg['target_cuts'];mid=cuts.index('mag25_radius_gt0.60')
    spec=importlib.util.spec_from_file_location('existing_estimator',H.parent/'alignment_axis2_constant_20261105/report.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    ws=np.vstack([np.ones(4),np.eye(4),1-np.eye(4)]);out={}
    for axis in [1,2]:
        raw={}
        for arm in ('control','weighted'):
            sim=[];model=[]
            for case in cases:
                p=C/'constant'/arm/f'axis{axis}'/f'case{case}.npz';rec=read(p.with_suffix('.json'))
                assert rec['case']==case and rec['data_sha256']==sha(p) and rec['checkpoint_sha256']==pins['arms'][arm]['sha256']
                with np.load(p) as z:
                    assert list(z['cuts'])[:9]==cuts
                    sim.append(z['sim'].sum(-2));model.append(z['model'].sum(-2))
            raw[arm]=(np.array(sim),np.array(model))
        np.testing.assert_array_equal(raw['control'][0],raw['weighted'][0]);out[str(axis)]={}
        for j,mode in enumerate(r.MODES):
            rows={};rr={}
            def ent(v):
                jk=v[5:];return dict(pooled=float(v[0]),cases=v[1:5].tolist(),case_jackknife_se=float(np.sqrt(.75*np.square(jk-jk.mean()).sum())))
            for k,cut in enumerate(cuts):
                res={}
                for arm,(s,m) in raw.items():
                    rr[arm,k]=v=r.estimates(s[:,:,k],m[:,j,:,k],ws,axis=axis)
                    res[arm]=dict(m_percent=ent(v['m_percent']),residual=ent(v['delta_response'][:,axis-1]))
                a,b=rr['weighted',k],rr['control',k]
                for key in ['model_flow_response','model_emulator_response','delta_response','m_percent']:
                    d=a[key]-b[key]
                    if d.ndim>1:d=d[:,axis-1]
                    t=ent(d);res['change_'+key]=t
                    saved=ref['axes'][str(axis)]['paired_changes']['weighted_minus_control'][mode][cut][key]
                    ix=() if key=='m_percent' else (axis-1,)
                    np.testing.assert_allclose([t['pooled'],t['case_jackknife_se']],[np.array(saved['value'])[ix],np.array(saved['jackknife_se'])[ix]],rtol=1e-9,atol=1e-13)
                res['change_absolute_residual']=ent(abs(a['delta_response'][:,axis-1])-abs(b['delta_response'][:,axis-1]))
                rows[cut]=res
            rms={a:np.sqrt(np.mean([rr[a,k]['m_percent']**2 for k in range(9)],0)) for a in raw}
            out[str(axis)][mode]={'cuts':rows,'nine_cut_rms_m_percent':{**{a:ent(q) for a,q in rms.items()},'paired_change':ent(rms['weighted']-rms['control'])}}
    (H/'frozen_scenes.json').write_text(json.dumps(dict(cases=cases,sign='weighted minus frozen control',job=os.environ['SLURM_JOB_ID'],cuts=cuts,source_report=sha(F/'report.json'),source_code=sha(Path(__file__)),axes=out),indent=1,allow_nan=False)+'\n')
    for ax,modes in out.items():
        for mode,s in modes.items():
            rows=s['cuts'];v=[z['change_delta_response']['pooled'] for z in rows.values()]
            print(ax,mode,'range',min(v),max(v),'num positive',sum(q>0 for q in v),'abs improved',sum(z['change_absolute_residual']['pooled']<0 for z in rows.values()),'center',rows[cuts[mid]]['change_delta_response'],'rms',s['nine_cut_rms_m_percent']['paired_change'])
if __name__=='__main__':main()
