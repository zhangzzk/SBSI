import copy
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
    monkeypatch.setattr(m,'expected_protocol',lambda *args: {})
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
    assert m.load(p)['new_work_seconds_by_arm']==dict(baseline=43.,optimized=35.)
    with pytest.raises(ValueError): m.finalize(p,c)


def fixed_protocol():
    return dict(seed=m.SEEDS[0],particles=256,stages=64,moves=64,scale=1.,
        observations=m.OBSERVATIONS,fixed_shear=[0.,0.],density_batch_size=4096,
        betas=[(i/64)**2 for i in range(65)],reference='/reference',
        initial_proposal={'path':'/initial'},physical_geometry={'path':'/geometry'},
        runtime=dict(python='3.10.14',numpy='2.2.6',torch='2.9.1+cu126',device='cuda',
            cuda_device_name='Tesla V100-SXM2-32GB',torch_num_threads=2,
            slurm_node_list='th-cl-nv02',slurm_job_id='123'))


def fixed_saved_values():
    protocol=fixed_protocol()
    stages=[dict(stage=i,beta=protocol['betas'][i],elapsed_seconds=float(i)) for i in range(1,65)]
    result=dict(status='completed',particles=256,observations=m.OBSERVATIONS,final_stage=stages[-1])
    return result,protocol,dict(stages=stages,requested=64)


@pytest.mark.parametrize('case',['python','numpy','torch','device','cuda_device_name','torch_num_threads',
    'slurm_node_list','seed','particles','stages','moves','scale','observations','fixed_shear',
    'density_batch_size','requested','missing_stage','stage_index','beta','endpoint','final_stage'])
def test_shared_wrong_recipe_or_short_history_fails(case):
    result,protocol,progress=copy.deepcopy(fixed_saved_values())
    expected=fixed_protocol()
    if case in protocol['runtime']: protocol['runtime'][case]='wrong'
    elif case in protocol: protocol[case]='wrong'
    elif case=='requested': progress['requested']=63
    elif case=='missing_stage': progress['stages'].pop()
    elif case=='stage_index': progress['stages'][4]['stage']=6
    elif case=='beta': progress['stages'][4]['beta']+=.0001
    elif case=='endpoint': progress['stages'][-1]['beta']=.999
    else: result['final_stage']=copy.deepcopy(progress['stages'][-2])
    with pytest.raises(ValueError): m.validate_saved_run(result,protocol,progress,expected)


def test_complete_stage_recipe_passes():
    values=fixed_saved_values()
    m.validate_saved_run(*values,fixed_protocol())


def test_saved_output_comparison_verifies_archives_sources_and_audit(tmp_path):
    d=dict(arms={},reference='/reference',initial_proposal='/initial',physical_geometry='/geometry',
        settings=dict(particles=256,stages=64,moves=64,power=2.,scale=1.,shear=[0.,0.],density_batch_size=4096))
    sources={}
    for arm in ('baseline','optimized'):
        source=tmp_path/('src_'+arm); source.mkdir()
        sources[arm]={'sbsi/scene_exchange.py':arm,'scripts/check_galsbi_particles.py':'same'}
        d['arms'][arm]={'source':str(source)}
        out=tmp_path/arm; out.mkdir()
        np.savez(out/'particles.npz',coordinates=np.array([[1.,-0.]]),extinct=np.array([False]))
        np.savez(out/'scenes.npz',indptr=np.array([0,0]),offset_arcsec=np.empty((0,2)))
        result,protocol,progress=copy.deepcopy(fixed_saved_values())
        protocol['implementation_sha256']={str(source/k):v for k,v in sources[arm].items()}
        m.save(out/'protocol.json',protocol)
        result.update(audit={'likelihood_rows':4},timing_seconds={'initialization':1.,'sampling':2.,'total':3.},
            **{key:m.digest(out/(key[:-7]+('.json' if key=='protocol_sha256' else '.npz')))
               for key in m.ARTIFACT_HASH_KEYS})
        m.save(out/'result.json',result); m.save(out/'progress.json',progress)
    template=tmp_path/'template.json'; template.write_bytes((tmp_path/'baseline/protocol.json').read_bytes())
    d['protocol_template']=dict(path=str(template),sha256=m.digest(template),source=d['arms']['baseline']['source'])
    d['expected_runtime']={k:v for k,v in fixed_protocol()['runtime'].items() if k!='slurm_job_id'}
    assert m.compare_runs(d,tmp_path,sources,0)
    wrong=copy.deepcopy(d); wrong['expected_runtime']['numpy']='wrong'
    with pytest.raises(ValueError): m.compare_runs(wrong,tmp_path,sources,0)
    path=tmp_path/'optimized/result.json'; result=m.load(path); result['audit']['likelihood_rows']=5; m.save(path,result)
    with pytest.raises(ValueError): m.compare_runs(d,tmp_path,sources,0)
    result['audit']['likelihood_rows']=4; m.save(path,result)
    with (tmp_path/'optimized/scenes.npz').open('ab') as f: f.write(b'changed zip bytes')
    with pytest.raises(ValueError): m.compare_runs(d,tmp_path,sources,0)
