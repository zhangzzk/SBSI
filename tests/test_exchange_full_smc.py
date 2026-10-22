import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('exchange_full',Path(__file__).resolve().parents[1]/'scripts/benchmark_exchange_full_smc.py')
m=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def test_array_bytes_dtype_shape_and_field_guards(tmp_path):
    a,b=tmp_path/'a.npz',tmp_path/'b.npz'
    np.savez(a,x=np.array([0.,np.nan]),y=np.array([1],dtype='i8'))
    np.savez(b,x=np.array([0.,np.nan]),y=np.array([1],dtype='i8'))
    assert set(m.compare_npz(a,b))=={'x','y'}
    for fields in [dict(x=np.array([-0.,np.nan]),y=np.array([1],dtype='i8')),
                   dict(x=np.array([0.,np.nan]),y=np.array([1],dtype='i4')),
                   dict(x=np.array([[0.,np.nan]]),y=np.array([1],dtype='i8')),
                   dict(x=np.array([0.,np.nan]))]:
        np.savez(b,**fields)
        with pytest.raises(ValueError): m.compare_npz(a,b)


@pytest.mark.parametrize('kind,a,b',[
    ('result',{'score_mean':[0.]},{'score_mean':[-0.]}),
    ('result',{'log_normalizer':[1.]},{'log_normalizer':[1.0000000000000002]}),
    ('progress',{'stages':[{'stage':1,'elapsed_seconds':1.}]},{'stages':[{'stage':2,'elapsed_seconds':2.}]}),
    ('protocol',{'runtime':{'numpy':'2.2.6'}},{'runtime':{'numpy':'2.2.7'}}),
    ('result',{'new_seconds':1.},{'new_seconds':2.}),
    ('result',{'x':1},{'x':True}),
])
def test_unlisted_scientific_difference_fails(kind,a,b):
    with pytest.raises(ValueError): m.compare_json(a,b,kind)


def test_only_explicit_timing_and_job_id_exceptions():
    m.compare_json({'timing_seconds':{'total':1.}}, {'timing_seconds':{'total':2.}},'result')
    m.compare_json({'stages':[{'stage':1,'elapsed_seconds':1.}]},
                   {'stages':[{'stage':1,'elapsed_seconds':2.}]},'progress')
    with pytest.raises(ValueError):
        m.compare_json({'timing_seconds':{'total':1.}},{'timing_seconds':{'total':float('nan')}},'result')


def test_source_difference_gate_and_external_hash(tmp_path,monkeypatch):
    d=dict(arms={},external_files={},harness_files={},seeds=m.SEEDS,observations=m.OBSERVATIONS)
    for arm,payload in [('baseline','baseline'),('optimized','optimized')]:
        s=tmp_path/arm; (s/'sbsi').mkdir(parents=True)
        (s/'sbsi/scene_exchange.py').write_text(payload)
        (s/'other.py').write_text('identical')
        (s/'SHA256SUMS').write_text(''.join(m.digest(s/n)+'  '+n+'\n' for n in ['sbsi/scene_exchange.py','other.py']))
        d['arms'][arm]=dict(source=str(s),manifest_sha256=m.digest(s/'SHA256SUMS'))
    monkeypatch.setattr(m,'BASE_HASH',m.digest(tmp_path/'baseline/sbsi/scene_exchange.py'))
    monkeypatch.setattr(m,'OPT_HASH',m.digest(tmp_path/'optimized/sbsi/scene_exchange.py'))
    assert m.verify(d)
    (tmp_path/'optimized/other.py').write_text('mutation')
    with pytest.raises(ValueError): m.verify(d)


def test_command_preserves_full_fixed_recipe(tmp_path):
    d=dict(arms={'baseline':{'source':str(tmp_path)}},reference='/reference',
           initial_proposal='/initial.json',physical_geometry='/geometry.json')
    c=m.driver_command(d,'baseline',2026103600,tmp_path/'out')
    assert c[c.index('--moves')+1]=='64' and c[c.index('--particles')+1]=='256'
    assert c[c.index('--observation-indices')+1:c.index('--particles')]==list(map(str,m.OBSERVATIONS))
    assert c[c.index('--density-batch-size')+1]=='4096'
    assert '--relocate-exchanges' not in c and '--informed-environment' not in c


def test_failed_arm_is_retained_and_other_arm_still_attempted(tmp_path,monkeypatch):
    d=dict(output=str(tmp_path/'runs'),arms={a:dict(source=str(tmp_path)) for a in ('baseline','optimized')})
    declaration=tmp_path/'declaration.json'; m.save(declaration,d)
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setattr(m,'verify',lambda d: {})
    monkeypatch.setattr(m,'driver_command',lambda d,a,s,o:[a])
    calls=[]
    def fake(command,**kwargs):
        calls.append(command[0]); return type('Process',(),{'returncode':1})()
    monkeypatch.setattr(m.subprocess,'run',fake)
    with pytest.raises(ValueError): m.run_pair(declaration,1)
    r=m.load(tmp_path/'runs/pair_1/receipt.json')
    assert calls==['optimized','baseline'] and r['status']=='failed'
    assert set(r['arms'])=={'baseline','optimized'}
    with pytest.raises(FileExistsError): m.run_pair(declaration,1)


def test_final_report_adds_common_process_cost(tmp_path):
    p,c=tmp_path/'report.json',tmp_path/'cost.json'
    m.save(p,dict(status='pending_report_process_cost',pairs=[dict(charged_seconds={'baseline':10.,'optimized':8.})]*4))
    m.save(c,dict(seconds=6.,exit_code=0))
    m.finalize(p,c)
    assert m.load(p)['all_inclusive_seconds']==dict(baseline=43.,optimized=35.)
    with pytest.raises(ValueError): m.finalize(p,c)


def test_saved_output_comparison_verifies_archives_sources_and_audit(tmp_path):
    d={'arms':{}}; sources={}
    for arm in ('baseline','optimized'):
        source=tmp_path/('src_'+arm); source.mkdir()
        sources[arm]={'sbsi/scene_exchange.py':arm,'scripts/check_galsbi_particles.py':'same'}
        d['arms'][arm]={'source':str(source)}
        out=tmp_path/arm; out.mkdir()
        np.savez(out/'particles.npz',coordinates=np.array([[1.,-0.]]),extinct=np.array([False]))
        np.savez(out/'scenes.npz',indptr=np.array([0,0]),offset_arcsec=np.empty((0,2)))
        m.save(out/'protocol.json',dict(implementation_sha256={str(source/k):v for k,v in sources[arm].items()},
            runtime={'slurm_job_id':arm,'numpy':'2.2.6'},physical_geometry={'path':'/same/artifact'}))
        m.save(out/'result.json',dict(status='completed',audit={'likelihood_rows':4},
            timing_seconds={'initialization':1.,'sampling':2.,'total':3.},
            **{key:m.digest(out/(key[:-7]+('.json' if key=='protocol_sha256' else '.npz')))
               for key in m.ARTIFACT_HASH_KEYS}))
        m.save(out/'progress.json',{'stages':[{'stage':1,'elapsed_seconds':1.,'logZ':[2.]}],'requested':1})
    assert m.compare_runs(d,tmp_path,sources)
    path=tmp_path/'optimized/result.json'; result=m.load(path); result['audit']['likelihood_rows']=5; m.save(path,result)
    with pytest.raises(ValueError): m.compare_runs(d,tmp_path,sources)
    result['audit']['likelihood_rows']=4; m.save(path,result)
    with (tmp_path/'optimized/scenes.npz').open('ab') as f: f.write(b'changed zip bytes')
    with pytest.raises(ValueError): m.compare_runs(d,tmp_path,sources)
