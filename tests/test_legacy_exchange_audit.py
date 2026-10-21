"""Pairing, historical decisions, owner subsets, and complete replay gates."""
from copy import deepcopy
from types import SimpleNamespace
import json
import importlib.util
import os
from pathlib import Path
import sys
from copy import copy

import numpy as np
import pytest

from sbsi.legacy_exchange_audit import LegacyExchangeAudit, compare_complete_outputs, file_digest
from audit_legacy_particle_trajectory import replay_plan, BLOCKS


def toy_sampler(*, owner=None, counts=None, log_prior=None):
    owner = np.array([0, 0, 1, 1, 2] if owner is None else owner)
    counts = np.array([1, 2, 0, 1, 1] if counts is None else counts)
    log_prior = np.array([0., -np.inf, 0., 0., 0.] if log_prior is None else log_prior)
    exchange = SimpleNamespace()
    def original(prior, data):
        return np.zeros(len(data['below']))
    def fixed(prior, data, *, return_support_audit):
        assert return_support_audit
        below = np.asarray(data['below'], bool)
        return np.where(below, -np.inf, 0.), dict(legacy_finite=np.ones(len(below), bool),
                                                below_uniform_clamp=below)
    exchange.numerical_over_analytic = original
    def proposal(prior, scene, probability, rng, **kwargs):
        active = scene.counts > 0
        if active.any():
            old = exchange.numerical_over_analytic(prior, {'below':scene.old[active]})
            new = exchange.numerical_over_analytic(prior, {'below':scene.new[active]})
            assert np.isfinite(old).all() and np.isfinite(new).all()  # Old ratios really return.
        return scene, np.zeros(len(active)), active
    exchange.propose_native_exchange = proposal
    module = SimpleNamespace(propose_native_exchange=proposal)
    class Target:
        def __init__(self):
            self.owner = owner
        def mutate(self, state, beta, rng, *, block, **kwargs):
            noise = rng.normal(size=len(owner))
            active = np.ones(len(owner), bool)
            if block in ('exchange', 'exchange_relocate'):
                rows = np.flatnonzero(np.isfinite(state.log_prior))
                active[:] = False
                if len(rows):
                    scene = SimpleNamespace(counts=counts[rows], old=(rows % 4 == 0), new=(rows % 4 == 3))
                    _, _, active[rows] = module.propose_native_exchange(None, scene, None, rng)
            accepted = (np.arange(len(owner)) % 2 == (block == 'exchange_relocate')) & active
            moved = SimpleNamespace(x=state.x+np.where(accepted, noise, 0)[:, None],
                                    log_prior=state.log_prior)
            return moved, accepted, active
    module.GalSBIParticleTarget = Target
    class Populations:
        def __init__(self, target): self.target = target
        def mutate(self, state, beta, rng, **kwargs): return self.target.mutate(state, beta, rng, **kwargs)
        def reweight_resample(self, state, delta, stage, rng):
            rng.random()
            return state, []
    state = SimpleNamespace(x=np.zeros((len(owner), 1)), log_prior=log_prior)
    return module, exchange, Populations, fixed, state


def test_pairs_active_native_rows_and_counts_actual_legacy_decisions_across_blocks():
    module, exchange, populations, fixed, initial = toy_sampler()
    target = module.GalSBIParticleTarget()
    sequence = [(.001, 'physical'), (.001, 'exchange'), (.001, 'exchange_relocate'),
                (.4, 'exchange'), (1., 'exchange')]
    expected, state, rng = [], initial, np.random.default_rng(29)
    for beta, block in sequence:
        state, accepted, active = target.mutate(state, beta, rng, block=block)
        expected.append((state.x.copy(), accepted.copy(), active.copy(), deepcopy(rng.bit_generator.state)))
    original_helper = exchange.numerical_over_analytic
    audit = LegacyExchangeAudit([58165, 0, 8], fixed)
    state, rng = initial, np.random.default_rng(29)
    with audit.instrument(module, exchange, populations):
        for (beta, block), (x, a, b, generator) in zip(sequence, expected):
            state, accepted, active = target.mutate(state, beta, rng, block=block)
            np.testing.assert_array_equal(state.x, x)
            np.testing.assert_array_equal(accepted, a)
            np.testing.assert_array_equal(active, b)
            assert rng.bit_generator.state == generator
    assert exchange.numerical_over_analytic is original_helper
    records = {(r['beta'],r['observation_index'],r['block']):r for r in audit.records()}
    for beta in (.001, .4, 1.):
        assert records[beta,0,'exchange']['old_accepted_target_into_zero'] == 1
        assert records[beta,1,'exchange']['proposed_selected_neighbour_zero'] == 1
        assert records[beta,1,'exchange']['old_accepted_off_support'] == 0
        assert records[beta,2,'exchange']['old_accepted_off_support'] == 1
    assert records[.001,1,'exchange_relocate']['old_accepted_selected_neighbour_zero'] == 1
    assert records[.001,0,'exchange_relocate']['old_accepted_off_support'] == 0
    assert audit.proposal_calls == 4 and audit.support_calls == 8
    stages = []
    for beta in (.001,.4,1.):
        acceptance = {}
        for block in ('exchange','exchange_relocate'):
            rows = [records.get((beta,i,block),{}) for i in range(3)]
            acceptance[block] = dict(attempted=[r.get('attempted',0) for r in rows],
                                    accepted=[r.get('old_accepted',0) for r in rows])
        stages.append(dict(beta=beta,acceptance=acceptance))
    audit.validate_progress(dict(stages=stages))
    stages[0]['acceptance']['exchange']['accepted'][1] += 1
    with pytest.raises(ValueError, match='disagree'): audit.validate_progress(dict(stages=stages))


@pytest.mark.parametrize('counts,log_prior,expected_calls', [([0,0],[0.,0.],0),([1,1],[-np.inf,-np.inf],0),([1,1],[0.,0.],2)])
def test_owner_subset_empty_neighbours_and_offprior_rows(counts,log_prior,expected_calls):
    module, exchange, populations, fixed, state = toy_sampler(owner=[2,2],counts=counts,log_prior=log_prior)
    audit = LegacyExchangeAudit([7,8,9],fixed)
    rng = np.random.default_rng(9)
    with audit.instrument(module,exchange,populations):
        tracker=populations(module.GalSBIParticleTarget())
        state,_=tracker.reweight_resample(state,.4,1,rng)
        state,_,_=tracker.mutate(state,.4,rng,block='exchange')
    assert audit.support_calls==expected_calls
    assert {r['observation_index'] for r in audit.records()} == {2}
    assert audit.last_state is state and audit.last_populations is tracker and audit.last_rng is rng
    rows={r['observation_index']:r for r in audit.records()}
    audit.validate_progress(dict(stages=[dict(beta=.4,acceptance={'exchange':dict(
        attempted=[0,0,rows[2]['attempted']],accepted=[0,0,rows[2]['old_accepted']])})]))
    assert len(audit.records())==3
    assert all(r['proposed_off_support']==0 for r in audit.records() if r['observation_index']!=2)


def test_unpaired_helper_calls_fail_closed_and_restore_patches():
    module,exchange,populations,fixed,state=toy_sampler()
    original=exchange.propose_native_exchange
    def broken(*args,**kwargs):
        value=original(*args,**kwargs)
        exchange.numerical_over_analytic(None,{'below':np.array([False])})
        return value
    exchange.propose_native_exchange=module.propose_native_exchange=broken
    helper=exchange.numerical_over_analytic
    audit=LegacyExchangeAudit([1,2,3],fixed)
    with pytest.raises(ValueError,match='paired'):
        with audit.instrument(module,exchange,populations):
            module.GalSBIParticleTarget().mutate(state,.4,np.random.default_rng(1),block='exchange')
    assert exchange.numerical_over_analytic is helper
    assert module.propose_native_exchange is broken and audit.frame is None


def write_json(path,data): path.write_text(json.dumps(data,allow_nan=False))


def artifacts(root, *, informed=False, rng=False):
    root.mkdir()
    protocol=dict(runtime=dict(python='3.10',numpy='2',torch='2',device='cpu',
        torch_num_threads=2,cuda_device_name=None,slurm_job_id=root.name,slurm_node_list='node'),
        stages=2,betas=[0.,.25,1.])
    stage=lambda n:dict(stage=n,beta=protocol['betas'][n],acceptance={'exchange':dict(attempted=[1],accepted=[1])},elapsed_seconds=float(n))
    progress=dict(stages=[stage(1),stage(2)],requested=2)
    particles={k:np.array([1.]) for k in ('coordinates','summaries','log_prior','log_kernel','scores','ancestor_labels','extinct','log_normalizer')}
    native=('type','z','M_B','c','R_kpc','n','a','psi','m_r','m_i','M_i')
    scenes={prefix+k:np.array([1.]) for prefix in ('target_','neighbour_') for k in native}
    scenes.update(indptr=np.array([0,1]),offset_arcsec=np.zeros((1,2)))
    if informed:
        protocol['informed_environment']={}
        environment={k:np.array([1.]) for k in ('log_tilt','valid','augmented','raw_coordinates','raw_types','raw_offsets','insertion_index','classification')}
        particles.update({'environment_'+k:v for k,v in environment.items()})
        np.savez_compressed(root/'initial_environment.npz',**environment,target_coordinates=np.zeros((1,10)),log_potential=np.zeros(1))
    np.savez_compressed(root/'particles.npz',**particles)
    np.savez_compressed(root/'scenes.npz',**scenes)
    write_json(root/'protocol.json',protocol);write_json(root/'progress.json',progress)
    result=dict(status='completed',score_mean=[[2.,3.]],final_stage=stage(2),timing_seconds={'total':9.})
    if informed: result['informed_environment_initialization']=dict(sha256=file_digest(root/'initial_environment.npz'))
    for name in ('protocol','particles','scenes'):
        result[name+'_sha256']=file_digest(root/(name+('.json' if name=='protocol' else '.npz')))
    write_json(root/'result.json',result)
    if rng: write_json(root/'rng_state.json',dict(bit_generator='PCG64',state=123))


def update_archive(root, name, mutate):
    with np.load(root/name) as f: arrays={k:f[k] for k in f.files}
    mutate(arrays);np.savez_compressed(root/name,**arrays)
    result=json.loads((root/'result.json').read_text())
    if name=='initial_environment.npz': result['informed_environment_initialization']['sha256']=file_digest(root/name)
    else: result[name.removesuffix('.npz')+'_sha256']=file_digest(root/name)
    write_json(root/'result.json',result)


def test_complete_outputs_accept_only_timing_and_scheduler_execution_changes(tmp_path):
    a,b=tmp_path/'original',tmp_path/'replay';artifacts(a,informed=True,rng=True);artifacts(b,informed=True,rng=True)
    result=json.loads((b/'result.json').read_text());result['timing_seconds']['total']=100.;result['final_stage']['elapsed_seconds']=99.;write_json(b/'result.json',result)
    progress=json.loads((b/'progress.json').read_text());progress['stages'][0]['elapsed_seconds']=8.;write_json(b/'progress.json',progress)
    comparison=compare_complete_outputs(a,b)
    assert comparison['original_rng_verified'] and comparison['status']=='all_persisted_outputs_bitwise_equal'


@pytest.mark.parametrize('change', ['score','normalizer','scene','environment','initial','history','rng','runtime','unknown','missing','both_missing','incomplete_history'])
def test_complete_replay_rejects_every_material_change(tmp_path,change):
    a,b=tmp_path/'original',tmp_path/'replay';artifacts(a,informed=True,rng=True);artifacts(b,informed=True,rng=True)
    if change in ('score','normalizer','environment'):
        key={'score':'scores','normalizer':'log_normalizer','environment':'environment_raw_offsets'}[change]
        update_archive(b,'particles.npz',lambda d:d[key].__setitem__(0,2.))
    elif change=='scene':update_archive(b,'scenes.npz',lambda d:d['offset_arcsec'].__setitem__((0,0),1.))
    elif change=='initial':update_archive(b,'initial_environment.npz',lambda d:d['raw_coordinates'].__setitem__(0,2.))
    elif change in ('history','incomplete_history'):
        p=json.loads((b/'progress.json').read_text())
        if change=='history':p['stages'][0]['acceptance']['exchange']['accepted'][0]=0
        else:p['stages'].pop(0)
        write_json(b/'progress.json',p)
    elif change=='rng':write_json(b/'rng_state.json',dict(bit_generator='PCG64',state=124))
    elif change=='runtime':
        p=json.loads((b/'protocol.json').read_text());p['runtime']['torch_num_threads']=1;write_json(b/'protocol.json',p)
        r=json.loads((b/'result.json').read_text());r['protocol_sha256']=file_digest(b/'protocol.json');write_json(b/'result.json',r)
    elif change=='unknown':write_json(a/'unreviewed.json',{})
    elif change=='missing':(b/'initial_environment.npz').unlink()
    elif change=='both_missing':
        for p in (a,b):update_archive(p,'particles.npz',lambda d:d.pop('log_normalizer'))
    with pytest.raises(ValueError):compare_complete_outputs(a,b)


def test_plan_uses_only_startup_protocol_and_preserves_declared_law(tmp_path):
    run=tmp_path/'dev_v8_informed_2';run.mkdir()
    protocol=dict(implementation_sha256={str(tmp_path/'source_v8/scripts/check_galsbi_particles.py'):'a'*64},
        particles=256,stages=64,moves=32,betas=[(i/64)**2 for i in range(65)],blocks=BLOCKS,
        initial_proposal={'path':'initial.json'},physical_geometry={'path':'geometry.json'},
        informed_environment={'path':'initial.json','metadata':{'mixture_probability':.5}},
        runtime={'device':'cuda'},reference='reference',scale=1.,seed=2026102702,
        density_batch_size=4096,observations=[58165,0,1,8],fixed_shear=[0.,0.])
    write_json(run/'protocol.json',protocol)
    plan=replay_plan(run,tmp_path/'audit')
    assert plan['cost']['mutation_calls']==2048 and plan['cost']['extra_likelihood_calls']==0
    assert plan['cost']['diagnostic_native_evaluations_upper_bound']==256
    assert '--informed-environment' in plan['driver_arguments']
    assert not (run/'result.json').exists()  # Planning must not inspect outcomes.
    with pytest.raises(ValueError,match='heldout'):replay_plan(tmp_path/'heldout_v7_0',tmp_path/'audit')


def load_frozen(name,source_name=None):
    directory=os.environ.get('SBSI_LEGACY_AUDIT_SOURCE')
    if not directory:pytest.skip('set untouched source_v7 for real frozen-code instrumentation checks')
    directory=Path(directory)
    if source_name is not None:directory=directory.parent/source_name
    path=directory/'sbsi'/name
    spec=importlib.util.spec_from_file_location('sbsi._audit_fixture_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def frozen_fixture(target,state,source_name=None):
    old=load_frozen('galsbi_particle_target.py',source_name)
    exchange=load_frozen('scene_exchange.py',source_name)
    populations=load_frozen('particle_populations.py',source_name)
    old.propose_native_exchange=exchange.propose_native_exchange
    class Legacy(type(target),old.GalSBIParticleTarget):
        # Keep the fixture's synthetic summary; exercise frozen evaluate/mutate.
        evaluate=old.GalSBIParticleTarget.evaluate
        initialize=old.GalSBIParticleTarget.initialize
        def mutate(self,*args,**kwargs):
            return old.GalSBIParticleTarget.mutate(self,*args,**kwargs)
    target=copy(target);target.__class__=Legacy
    return old,exchange,populations.ParticlePopulations,target,target.evaluate(state.x,state.scene)


@pytest.mark.parametrize('accepted',[False,True])
@pytest.mark.parametrize('block',['exchange','exchange_relocate'])
def test_real_legacy_support_proposals_follow_old_acceptance_unchanged(monkeypatch,accepted,block):
    from test_scene_exchange_support import tail_state
    from test_informed_environment import assert_state_equal
    from sbsi.scene_exchange import numerical_over_analytic
    target,state=tail_state()
    old,exchange,populations,target,state=frozen_fixture(target,state)
    monkeypatch.setattr(old,'metropolis_accept',lambda a,b,u:np.full(len(a),accepted,bool))
    baseline_rng=np.random.default_rng(2)
    expected,a,b=target.mutate(state,.4,baseline_rng,block=block)
    audit=LegacyExchangeAudit([58165],numerical_over_analytic)
    rng=np.random.default_rng(2)
    with audit.instrument(old,exchange,populations):
        moved,c,d=target.mutate(state,.4,rng,block=block)
    assert_state_equal(moved,expected)
    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    assert rng.bit_generator.state==baseline_rng.bit_generator.state
    row=audit.records()[0]
    assert row['proposed_off_support']==row['proposed_target_into_zero']==1
    assert row['old_accepted_off_support']==int(accepted)


def test_real_frozen_full_block_sequence_is_bitwise_with_instrumentation():
    from test_informed_environment import fixture,assert_state_equal
    from sbsi.scene_exchange import numerical_over_analytic
    target,state,_=fixture(4)
    old,exchange,populations,target,initial=frozen_fixture(target,state)
    sequence=[(beta,block) for beta in (.001,.4,1.) for block in
        ('physical','nuisance','sed','environment','positions','exchange','exchange_relocate','independence')]
    baseline_rng=np.random.default_rng(928);state=initial;expected=[]
    for beta,block in sequence:
        state,a,b=target.mutate(state,beta,baseline_rng,block=block)
        expected.append((deepcopy(state),a.copy(),b.copy(),deepcopy(baseline_rng.bit_generator.state)))
    rng=np.random.default_rng(928);state=initial
    audit=LegacyExchangeAudit([0],numerical_over_analytic)
    with audit.instrument(old,exchange,populations):
        for (beta,block),(saved,a,b,generator) in zip(sequence,expected):
            state,c,d=target.mutate(state,beta,rng,block=block)
            assert_state_equal(state,saved)
            np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
            assert rng.bit_generator.state==generator
    assert audit.support_calls==12


@pytest.mark.parametrize('block',['exchange','exchange_relocate'])
def test_frozen_v8_informed_guard_counts_proposal_but_not_old_acceptance(block):
    from test_scene_exchange_support import tail_state
    from test_informed_environment import assert_state_equal,assert_payload_equal
    from sbsi.scene_exchange import numerical_over_analytic
    target,state=tail_state(informed=True)
    old,exchange,populations,target,state=frozen_fixture(target,state,'source_v8')
    rng=np.random.default_rng(2)
    audit=LegacyExchangeAudit([58165],numerical_over_analytic)
    with audit.instrument(old,exchange,populations):
        moved,accepted,active=target.mutate(state,.4,rng,block=block)
    assert active.all() and not accepted.any()
    assert_state_equal(moved,state);assert_payload_equal(moved.environment_payload,state.environment_payload)
    assert audit.records()[0]['proposed_off_support']==1
    assert audit.records()[0]['old_accepted_off_support']==0
