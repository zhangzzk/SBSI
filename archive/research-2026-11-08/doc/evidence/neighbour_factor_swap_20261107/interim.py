"""Lightweight jackknife reduction of small saved sums for fully finished axes."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_factor_swap_20261107')
MODES=['predicted_usability_generated_selection','true_usability_generated_selection','true_usability_true_selection']
cfg=json.loads((HERE/'manifest.json').read_text());cases=cfg['cases'];n=len(cases)
assembly=json.loads((ROOT/'assembly_completed.json').read_text())
source=hashlib.sha256((HERE/'evaluate.py').read_bytes()).hexdigest()
weights=np.vstack([np.ones(n),np.ones((n,n))-np.eye(n)])
def entry(v):
    return {'value':v[0].tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(v[1:]-v[1:].mean(0)).sum(0)).tolist()}
def response(v):return (v[:,0,1:]/v[:,0,:1]-v[:,1,1:]/v[:,1,:1])/.04
def estimate(sim,model,axis):
    s=np.einsum('bc,clq->blq',weights,sim);m=np.einsum('bc,clq->blq',weights,model)
    rs=response(s);flow=response(m[:,:,:3]);emu=response(m[:,:,[0,3,4]]);total=flow+emu
    return {'simulation_response':rs,'model_response':total,'model_flow_response':flow,
            'model_emulator_response':emu,'delta_response':rs-total,
            'm_percent':100*(rs[:,axis-1]/total[:,axis-1]-1)}
out={'definition':'Only fully completed axes, all four development cases. Whole-case delete-one jackknife SE; fixed models and draws. DeltaR=simulation-model; m=100(Rsim/Rmodel-1). Component contrasts are model changes, not standalone errors.',
     'cases':cases,'completed_axes':[],'pending':{},'axes':{}}
for axis in [1,2]:
    missing=[case for case in cases if not (ROOT/f'audit_case{case}_axis{axis}.json').exists()]
    if missing:out['pending'][str(axis)]=missing;continue
    for case in cases:
        audit=json.loads((ROOT/f'audit_case{case}_axis{axis}.json').read_text())
        assert audit['same_photo_selection_exact'] and audit['source_sha256']==source
    out['completed_axes'].append(axis);samples={};arms={};ref=None
    for name in cfg['arms']:
        simulations=[];models=[]
        for case in cases:
            p=ROOT/'constant'/name/f'axis{axis}'/f'case{case}.npz';receipt=json.loads(p.with_suffix('.json').read_text())
            assert receipt['source_sha256']==source and receipt['checkpoint_sha256']==assembly['arms'][name]['sha256']
            assert receipt['data_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
            assert all(receipt[k] for k in ['simulation_replay_passed','reference_score_replay_passed','true_selection_mass_exact','pooled_emulator_roundoff_check_passed'])
            with np.load(p) as z:
                np.testing.assert_array_equal(z['cuts'],[x[0] for x in cfg['cuts']]);simulations.append(z['sim'].sum(-2));models.append(z['model'].sum(-2))
        sim=np.array(simulations);model=np.array(models)
        if ref is None:ref=sim
        else:np.testing.assert_array_equal(sim,ref)
        arms[name]={}
        for mode,label in enumerate(MODES):
            arms[name][label]={}
            for ci,(cut,_,_) in enumerate(cfg['cuts']):
                v=estimate(sim[:,:,ci],model[:,mode,:,ci],axis);samples[name,mode,cut]=v
                arms[name][label][cut]={key:entry(value) for key,value in v.items()}
    contrasts={}
    for contrast in cfg['factorial_comparisons']:
        contrasts[contrast['name']]={}
        for mode,label in enumerate(MODES):
            rows={}
            for cut in cfg['target_cuts']:
                a,b,ba,ab=[samples[contrast[k],mode,cut] for k in ['old','new','photo_only','shape_only']]
                rows[cut]={}
                for key in ['model_response','model_flow_response','model_emulator_response','delta_response','m_percent']:
                    terms={'photo_at_old_shape':ba[key]-a[key],'shape_at_old_photo':ab[key]-a[key],
                        'interaction':b[key]-ba[key]-ab[key]+a[key],'total':b[key]-a[key],
                        'photo_at_new_shape':b[key]-ab[key],'shape_at_new_photo':b[key]-ba[key]}
                    np.testing.assert_allclose(terms['photo_at_old_shape']+terms['shape_at_old_photo']+terms['interaction'],terms['total'],rtol=1e-10,atol=1e-12)
                    rows[cut][key]={term:entry(value) for term,value in terms.items()}
            contrasts[contrast['name']][label]=rows
    out['axes'][str(axis)]={'arms':arms,'factorial':contrasts}
previous=json.loads((HERE.parent/'neighbour_moment_guard_20261106/report.json').read_text())
replay_max=0.;replay_count=0
for axis in out['completed_axes']:
    for name in ['initial','control','weighted']:
        for mode,rows in out['axes'][str(axis)]['arms'][name].items():
            for cut,values in rows.items():
                for key,value in values.items():
                    for field in ['value','jackknife_se']:
                        x=np.asarray(value[field]);y=np.asarray(previous['axes'][str(axis)]['arms'][name][mode][cut][key][field])
                        np.testing.assert_allclose(x,y,atol=1e-8,rtol=0)
                        replay_max=max(replay_max,float(np.max(abs(x-y))));replay_count+=1
out['unchanged_source_summary_replay']={'entries':replay_count,'max_abs_difference':replay_max,'tolerance':1e-8}
for p in [ROOT/'interim.json',HERE/'interim.json']:p.write_text(json.dumps(out,indent=2)+'\n')
lines=['# Completed-axis interim factor comparison','',out['definition'],'',f"Complete axes: {out['completed_axes']}. Pending case(s) by axis: {out['pending']}.",'',
       'Target: radius>3px and magnitude<25. Actual usability with generated photometry/selection.','']
mode='true_usability_generated_selection';cut='mag25_radius_gt0.60'
for axis in out['completed_axes']:
    d=out['axes'][str(axis)];lines += [f'Axis {axis}:','', '| Product | m (%) ± jackknife SE (pp) |','|---|---:|']
    for name in cfg['arms']:
        v=d['arms'][name][mode][cut]['m_percent'];lines.append(f"| {name} | {v['value']:+.6f} ± {v['jackknife_se']:.6f} |")
    lines+=['','| Contrast / contribution | Photo at old shapes | Shapes at old photo | Interaction | Total |','|---|---:|---:|---:|---:|']
    for contrast in cfg['factorial_comparisons']:
        rows=d['factorial'][contrast['name']][mode][cut]
        for key in ['model_flow_response','model_emulator_response','delta_response']:
            values=[rows[key][term] for term in ['photo_at_old_shape','shape_at_old_photo','interaction','total']]
            cells=[f"{v['value'][axis-1]:+.6f} ± {v['jackknife_se'][axis-1]:.6f}" for v in values]
            lines.append('| '+contrast['name']+' / '+key+' | '+' | '.join(cells)+' |')
    lines.append('')
(HERE/'interim.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
