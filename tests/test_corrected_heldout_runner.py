"""Synthetic workflow checks; no real heldout outputs, models or GPUs are read."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
SPEC = importlib.util.spec_from_file_location('corrected_heldout_runner', ROOT/'scripts/run_corrected_particle_heldout.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bind(path):
    return dict(path=str(path), sha256=m.digest(path))


@pytest.fixture
def declared(tmp_path):
    source, output = tmp_path/'source', tmp_path/'output'
    entries = {}
    for name in m.REQUIRED_SOURCE:
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen '+name)
        entries[name] = m.digest(path)
    (source/'SHA256SUMS').write_text(''.join(f'{checksum}  {name}\n' for name, checksum in sorted(entries.items())))
    evidence = tmp_path/'development_input.json'; write(evidence, {'synthetic': True})
    development = tmp_path/'development.json'
    write(development, dict(schema='sbsi-n23-corrected-comparison-v1', development_checks_pass=True,
        accepted_sampler=False, comparisons={k:dict(all_raw_gates=True, all_corrected_phase_gates=True)
        for k in ('small_minus_large', 'small_minus_broad', 'large_minus_broad')},
        sources={str(evidence): m.digest(evidence)}))
    preparation = tmp_path/'preparation.json'
    write(preparation, dict(kind='scheduler_allocation_wall_upper_bound',
        process_wall_seconds=10., cpu_allocated_seconds=20., gpu_allocated_seconds=0.))
    groups = [list(range(20+4*i, 24+4*i)) for i in range(5)]
    panel = tmp_path/'panel.json'; write(panel, {'bands':[dict(observations=g) for g in groups]})
    reference = tmp_path/'reference'
    write(reference/'protocol.json', {'synthetic': True})
    write(reference/'result.json', {'synthetic': True})
    write(reference/'observations.parquet', {'synthetic_not_parquet': True})
    common = dict(stages=64, blocks=m.BLOCKS, scale=1., power=2.)
    proposal = dict(component_family='gaussian', faint_magnitude_shift=5., weights=[.5,.5],
        phase='independent uniform periodic', means=np.zeros((2,9)).tolist(),
        scales=np.ones((2,9)).tolist(), uses_generating_truth=False, retry_or_clip_draws=False)
    broad = dict(proposal, component_family='student_t', student_df=5.)
    d = dict(schema='sbsi-heldout-corrected-declaration-v1', source=str(source),
        source_manifest_sha256=m.digest(source/'SHA256SUMS'),
        source_files={str(source/name): checksum for name,checksum in entries.items()},
        external_files={str(reference/name): m.digest(reference/name) for name in ('protocol.json','observations.parquet','result.json')},
        output=str(output), panel=str(panel), panel_sha256=m.digest(panel), groups=groups,
        development_ids=[0,1,8,58165], shear=[0.,0.], density_batch_size=4096,
        reference=str(reference), reference_protocol_sha256=m.digest(reference/'protocol.json'),
        observation_sha256=m.digest(reference/'observations.parquet'),
        pilot=dict(common, particles=256, moves=32, replicas=4, seed_base=1000),
        candidate_settings=dict(common, particles=256, moves=64, replicas=8, seed_base=2000),
        reference_settings=dict(common, particles=1024, moves=64, replicas=8, seed_base=3000),
        pilot_proposal_metadata=[[copy.deepcopy(proposal) for _ in range(4)] for _ in range(5)],
        reference_proposal_metadata=[[copy.deepcopy(broad) for _ in range(4)] for _ in range(5)],
        runtime=dict(python='fixture', numpy='fixture', torch='fixture', cuda_device_name='Tesla V100-SXM2-32GB',
            slurm_node_list='th-cl-nv02', torch_num_threads=2),
        development_receipt=bind(development), preparation_cost_receipt=bind(preparation))
    path = output/'declaration.json'; write(path,d)
    return path,d,entries


def test_declaration_binds_complete_inputs_before_outputs(declared):
    path,d,entries = declared
    actual,manifest = m.validate_declaration(path, check_execution=False)
    assert actual == d and manifest == entries
    assert not (Path(d['output'])/'group_0').exists()


@pytest.mark.parametrize('change', ['development_false','development_pair','evidence','panel','source',
    'source_map','external','seed_overlap','effort','blocks','proposal','cost','groups'])
def test_bad_declaration_rejects_before_any_population(declared,change):
    path,d,_=declared
    if change.startswith('development'):
        p=Path(d['development_receipt']['path']); r=m.read(p)
        if change=='development_false': r['development_checks_pass']=False
        else: r['comparisons']['small_minus_broad']['all_corrected_phase_gates']=False
        write(p,r); d['development_receipt']=bind(p)
    elif change=='evidence':
        p=next(iter(m.read(d['development_receipt']['path'])['sources'])); write(Path(p),{'changed':True})
    elif change=='panel': write(Path(d['panel']),{'changed':True})
    elif change=='source': Path(next(iter(d['source_files']))).write_text('changed')
    elif change=='source_map': d['source_files'].pop(next(iter(d['source_files'])))
    elif change=='external': d['external_files']={}
    elif change=='seed_overlap': d['reference_settings']['seed_base']=d['pilot']['seed_base']
    elif change=='effort': d['candidate_settings']['moves']=32
    elif change=='blocks': d['pilot']['blocks']=['physical']
    elif change=='proposal': d['reference_proposal_metadata'][0][0]['student_df']=7.
    elif change=='cost':
        p=Path(d['preparation_cost_receipt']['path']); r=m.read(p);r['gpu_allocated_seconds']=-1
        write(p,r);d['preparation_cost_receipt']=bind(p)
    else: d['groups'].reverse()
    write(path,d)
    with pytest.raises(ValueError):m.validate_declaration(path,check_execution=False)


def populate_pilot(d,entries):
    run,phase=m.run_paths(d,0,0,'pilot');run.mkdir(parents=True)
    s=d['pilot'];seed=m.run_seed(d,0,0,'pilot')
    p=dict(seed=seed, observations=d['groups'][0], particles=s['particles'], stages=s['stages'],moves=s['moves'],
        blocks=s['blocks'],scale=s['scale'],betas=(np.linspace(0,1,65)**2).tolist(),fixed_shear=d['shear'],
        density_batch_size=4096,reference=d['reference'],reference_protocol_sha256=d['reference_protocol_sha256'],
        observation_sha256=d['observation_sha256'],selection=m.SELECTION,bridge=m.BRIDGE,endpoint_target=m.ENDPOINT,
        proposals=d['pilot_proposal_metadata'][0],runtime=dict(d['runtime'],device='cuda'),
        implementation_sha256={str(Path(d['source'])/name):entries[name] for name in m.PROTOCOL_SOURCES})
    write(run/'protocol.json',p)
    np.savez(run/'particles.npz',summaries=np.zeros((4,256,9)),scores=np.zeros((4,256,2)),log_kernel=np.zeros((4,256)))
    np.savez(run/'scenes.npz',fixture=np.arange(4))
    final=dict(stage=64,beta=1.,log_normalizer=[0.]*4,target_within_two_magnitudes=[.5]*4,
        target_over_three_magnitudes_fainter=[.2]*4)
    result=dict(status='completed',observations=d['groups'][0],particles=256,final_stage=final,
        log_normalizer=[0.]*4,score_mean=[[0.,0.]]*4,score_variance=[[1.,1.]]*4,
        timing_seconds=dict(initialization=1.,sampling=2.,total=3.),
        **{name+'_sha256':m.digest(run/file) for name,file in [('protocol','protocol.json'),('particles','particles.npz'),('scenes','scenes.npz')]})
    write(run/'result.json',result)
    identity={k:p[k] for k in ('observations','fixed_shear','reference_protocol_sha256','observation_sha256','density_batch_size','selection')}
    identity.update(native_source_sha256={Path(k).name:v for k,v in entries.items()
        if k.startswith(('sbsi/','configs/')) and Path(k).name in m.NATIVE_SOURCES},phase_derivative_protocol=m.PHASE_PROTOCOL)
    np.savez(phase.with_suffix('.npz'),raw_scores=np.zeros((4,256,2)),phase_control=np.zeros((4,256)),delta_log_kernel=np.zeros((4,256)))
    report=dict(schema='sbsi-phase-diagnostic-v1',numerical_checks_pass=True,stride=1,
        arrays_sha256=m.digest(phase.with_suffix('.npz')),run=str(run),seed=seed,observations=d['groups'][0],
        retained_particles=256,log_normalizer=[0.]*4,device='cuda',device_name=d['runtime']['cuda_device_name'],
        derivative_protocol=m.PHASE_PROTOCOL,target_identity=identity,
        source_hashes={str(run/name):m.digest(run/name) for name in ('protocol.json','result.json','particles.npz')},
        implementation_sha256={str(Path(d['source'])/name):entries[name] for name in m.DIAGNOSTIC_SOURCES})
    write(phase,report)
    return run,phase,p,result,report


def test_complete_pilot_and_diagnostic_content_are_bound(declared):
    _,d,entries=declared;populate_pilot(d,entries)
    report,arrays=m.validate_phase(d,entries,0,0,'pilot')
    assert arrays['raw_scores'].shape==(4,256,2) and report['seed']==1000


@pytest.mark.parametrize('change',['numeric_false','stride','substituted_source','native_identity','implementation',
    'kernel_target','proposal_scale','extinction','hardware','normalizer','particle_count'])
def test_run_and_phase_validation_fail_closed(declared,change):
    _,d,e=declared;run,phase,p,r,report=populate_pilot(d,e)
    if change=='numeric_false':report['numerical_checks_pass']=False
    elif change=='stride':report['stride']=2
    elif change=='substituted_source':report['source_hashes']={str(phase.with_suffix('.npz')):m.digest(phase.with_suffix('.npz'))}
    elif change=='native_identity':report['target_identity']['native_source_sha256']={}
    elif change=='implementation':report['implementation_sha256']={}
    elif change=='kernel_target':p['bridge']='other target'
    elif change=='proposal_scale':p['proposals']=copy.deepcopy(p['proposals']);p['proposals'][0]['scales'][0][0]=3.
    elif change=='extinction':r['status']='completed_with_extinction'
    elif change=='hardware':p['runtime']['torch_num_threads']=1
    elif change=='normalizer':report['log_normalizer'][0]=1.
    else:report['retained_particles']=255
    write(run/'protocol.json',p);r['protocol_sha256']=m.digest(run/'protocol.json');write(run/'result.json',r)
    if change!='substituted_source':report['source_hashes']={str(run/name):m.digest(run/name) for name in ('protocol.json','result.json','particles.npz')}
    write(phase,report)
    with pytest.raises(ValueError):m.validate_phase(d,e,0,0,'pilot')


def test_arguments_and_counterbalanced_fixed_order(declared):
    _,d,_=declared
    assert m.production_order(0)==('candidate','reference')
    assert m.production_order(1)==('reference','candidate')
    for kind in ('pilot','candidate','reference'):
        args=m.population_arguments(d,2,1,kind)
        assert args[args.index('--seed')+1]==str(m.run_seed(d,2,1,kind))
        assert args[args.index('--moves')+1]==str(m.settings(d,kind)['moves'])
        assert ('--initial-proposal' in args)==(kind=='candidate')
        assert ('--physical-geometry' in args)==(kind!='pilot')
        phase=m.phase_arguments(d,2,1,kind)
        assert phase[phase.index('--stride')+1]=='1'
        assert ('--coefficients' in phase)==(kind!='pilot')


def test_production_runs_then_checks_each_phase_in_fixed_order(declared,monkeypatch):
    path,d,e=declared;events=[]
    stage=SimpleNamespace(declaration_path=path,timed=lambda directory,tag,script,args:events.append(tag))
    monkeypatch.setattr(m,'validate_warmup',lambda *a:events.append('warmup_verified'))
    monkeypatch.setattr(m,'validate_run',lambda *a:events.append('run_verified'))
    monkeypatch.setattr(m,'validate_phase',lambda *a:events.append('phase_verified'))
    m.production(stage,d,e,9)
    assert events==['warmup_verified','reference_1','run_verified','reference_1_phase','phase_verified',
                    'candidate_1','run_verified','candidate_1_phase','phase_verified']


def test_failed_step_preserves_cost_and_prevents_retry(declared,monkeypatch):
    path,d,_=declared;directory=Path(d['output'])/'group_0';directory.mkdir()
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=7))
    stage=m.Stage(d,path,'warmup_0',dict(cpus=2,gpus=1),time.monotonic())
    with pytest.raises(RuntimeError):stage.timed(directory,'pilot_0',m.SCRIPTS['run'],['--fixture'])
    receipt=m.read(directory/'pilot_0_timing.json')
    assert receipt['returncode']==7 and receipt['cost']['process_wall_seconds']>0
    failed=stage.finish(RuntimeError('fixture'))
    assert failed['status']=='failed' and failed['cost']['gpu_allocated_seconds']>0
    with pytest.raises(FileExistsError):m.Stage(d,path,'warmup_0',dict(cpus=2,gpus=1),time.monotonic())
    with pytest.raises(ValueError):m.validate_stage_receipt(failed,d,path,'warmup_0')


def step(tag,seconds):return dict(tag=tag,cost=m.measured_cost(seconds,dict(cpus=2,gpus=1)))
def staged(steps):
    overhead=m.measured_cost(1.,dict(cpus=2,gpus=1))
    return dict(steps=steps,verification_overhead=overhead,cost=m.add_cost(overhead,*(s['cost'] for s in steps)))


def all_costs():
    warmups=[staged([step(f'pilot_{i}{suffix}',1.) for i in range(4) for suffix in ('','_phase')]+
        [step('geometry_fit',1.),step('proposal_fit',2.),step('coefficient_fit',1.)]) for _ in range(5)]
    production=[staged([step(f'{kind}_{index%8}{suffix}',2. if kind=='candidate' else 4.)
        for kind in m.production_order(index%8) for suffix in ('','_phase')]) for index in range(40)]
    report=staged([step(f'group_{g}_{kind}_{label}',1.) for g in range(5)
        for kind in ('candidate','reference') for label in ('raw_report','phase_report')])
    prep=dict(process_wall_seconds=10.,cpu_allocated_seconds=20.,gpu_allocated_seconds=0.)
    return warmups,production,report,prep


def test_all_costs_charged_and_actual_work_not_double_counted():
    warm,production,report,prep=all_costs();cost=m.recipe_costs(warm,production,report,prep)
    # Shared prep:50, candidate-only proposal fits:10; process overhead46 plus preflight10 split equally.
    assert cost['recipes']['candidate']['process_wall_seconds']==50+10+160+10+28
    assert cost['recipes']['reference']['process_wall_seconds']==50+320+10+28
    assert cost['actual_work']['process_wall_seconds']==60+480+20+46+10
    groups=[dict(group=i,accuracy_pass=True) for i in range(5)]
    assert m.panel_result(groups,cost)['accepted_sampler']
    groups[2]['accuracy_pass']=False
    assert not m.panel_result(groups,cost)['accepted_sampler']
    cost['recipes']['candidate']=cost['recipes']['reference'].copy()
    assert not m.panel_result([dict(group=i,accuracy_pass=True) for i in range(5)],cost)['heldout_cost_gate']


@pytest.mark.parametrize('missing',['pilot','production','phase_report'])
def test_missing_cost_cannot_be_dropped(missing):
    warm,production,report,prep=all_costs()
    if missing=='pilot':warm[0]['steps'].pop()
    elif missing=='production':production.pop()
    else:report['steps'].pop()
    with pytest.raises(ValueError):m.recipe_costs(warm,production,report,prep)


def test_phase_gate_requires_both_measures_and_raw_precision(monkeypatch):
    monkeypatch.setattr(m,'comparison_target_identity',lambda *a:None)
    score=dict(mean=[0.,0.],mcse=[.1,.1],log_mean_normalizer=0.,normalizer_relative_mcse=.01)
    row=dict(observation=0,precision_gate=True,score=score,region=dict(mean=[.5,.2],mcse=[.01,.01]),
        posterior_score_variance_estimate=[1.,1.])
    raw=dict(observations=[dict(copy.deepcopy(row),observation=i) for i in range(4)],seeds=list(range(8)))
    other=copy.deepcopy(raw);other['seeds']=list(range(10,18))
    value=dict(mean=[0.]*5,mcse=[.1]*5)
    phase=dict(target_identity={'same':True},coefficients={'same':True},
        observations=[dict(observation=i,**{k:copy.deepcopy(value) for k in m.MEASURES}) for i in range(4)])
    assert m.compare_group(raw,other,phase,copy.deepcopy(phase))['accuracy_pass']
    changed=copy.deepcopy(phase);changed['observations'][0][m.MEASURES[1]]['mean'][3]=1.
    assert not m.compare_group(raw,other,phase,changed)['accuracy_pass']
    raw['observations'][1]['precision_gate']=False
    assert not m.compare_group(raw,other,phase,phase)['accuracy_pass']
    raw['observations'].pop()
    with pytest.raises(ValueError):m.compare_group(raw,other,phase,phase)
