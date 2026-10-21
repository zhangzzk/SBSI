"""Per-object zero-mass populations: driver, ragged state, receipts and report."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbsi.annealed_particles import reweight, stratified_resample
from sbsi.galsbi_scene_prior import GalSBISceneBatch
from sbsi.particle_populations import (EXTINCTION_SCHEMA, ParticlePopulations,
    replace_particles, survivor_diagnostics)
from test_galsbi_particle_target import setup_target
from test_particle_report import reporter, make_runs, run_report, write_json, digest


def population_fixture(particles=4, extinct=()):
    target, state = setup_target(2*particles)
    target.observations = pd.DataFrame([[.1,.2,3.,2.], [.2,.1,4.,5.]],
        columns=['e1','e2','radius','flux'])
    target.owner = np.repeat([0,1], particles)
    target.observed = target.observations.iloc[target.owner].reset_index(drop=True)
    target.auxiliary = target.observed.iloc[:, 2:4].to_numpy()
    target.proposals = [target.proposals[0], deepcopy(target.proposals[0])]
    x = state.x.copy()
    for observation in extinct:
        x[observation*particles:(observation+1)*particles, 0] = np.log(4.)
    state = target.evaluate(x, state.scene)
    return target, state


def assert_states_equal(left, right):
    for name in ('x','log_prior','log_kernel','log_proposal','type_probability','summaries'):
        np.testing.assert_array_equal(getattr(left,name), getattr(right,name))
    np.testing.assert_array_equal(left.scene.indptr, right.scene.indptr)
    np.testing.assert_array_equal(left.scene.offset_arcsec, right.scene.offset_arcsec)
    for field in ('targets','neighbours'):
        for key, values in getattr(left.scene,field).items():
            np.testing.assert_array_equal(values, getattr(right.scene,field)[key])


def test_no_extinction_preserves_original_numeric_and_rng_path_bitwise():
    target, state = population_fixture()
    original = state.take(np.arange(8))
    tracker = ParticlePopulations(target, 2, 4)
    new_rng, old_rng = np.random.default_rng(924), np.random.default_rng(924)
    old_logz, old_labels = np.zeros(2), np.tile(np.arange(4),(2,1))
    previous = 0.
    for stage,beta in enumerate([.001,.2,.5,1.],1):
        rows = []
        for i in range(2):
            weights, increment = reweight(np.full(4,-np.log(4)), original.potential.reshape(2,4)[i], beta-previous)
            old_logz[i] += increment
            chosen = stratified_resample(weights, old_rng)
            old_labels[i] = old_labels[i,chosen]
            rows.extend(i*4+chosen)
        original = original.take(np.asarray(rows))
        state, _ = tracker.reweight_resample(state,beta-previous,stage,new_rng)
        for block in ('physical','nuisance','sed','positions','environment','exchange'):
            original, old_decision, old_active = target.mutate(original,beta,old_rng,block=block)
            state, decision, active = tracker.mutate(state,beta,new_rng,block=block)
            assert_states_equal(state,original)
            np.testing.assert_array_equal(decision,old_decision)
            np.testing.assert_array_equal(active,old_active)
        np.testing.assert_array_equal(tracker.logz,old_logz)
        np.testing.assert_array_equal(tracker.labels,old_labels)
        assert new_rng.bit_generator.state == old_rng.bit_generator.state
        previous = beta


def test_extinction_is_absorbing_and_other_observation_keeps_original_owner():
    target, state = population_fixture(extinct=(0,))
    extinct_state = state.take(np.arange(4))
    tracker, rng = ParticlePopulations(target,2,4), np.random.default_rng(926)
    state, ess = tracker.reweight_resample(state,.1,1,rng)
    assert tracker.alive.tolist() == [False,True] and ess[0] == 0
    for beta in (.1,.4,1.):
        state, decision, active = tracker.mutate(state,beta,rng,block='physical')
        assert not decision[:4].any() and not active[:4].any()
        assert_states_equal(state.take(np.arange(4)),extinct_state)
    # Even an artificially repaired inactive payload must not revive its Zhat.
    state.log_prior[:4] = 0.; state.log_kernel[:4] = 0.
    state, _ = tracker.reweight_resample(state,.2,2,rng)
    assert tracker.logz[0] == -np.inf and tracker.extinction_stage.tolist() == [1,-1]
    seen = []
    def score(full, observed):
        seen.append(observed.copy())
        return full[:,:2]
    full, scores, error = survivor_diagnostics(target,state,tracker.alive,4,target.likelihood,score)
    assert error < 1e-12 and len(seen) == 1
    np.testing.assert_array_equal(seen[0].iloc[:,2:4], np.tile([4.,5.],(4,1)))
    assert not full[:4].any() and not scores[:4].any()


def test_scatter_replaces_complete_ragged_scenes_without_aliases():
    _, state = population_fixture()
    rows = np.array([6,1])
    replacement = state.take(rows)
    replacement.scene = GalSBISceneBatch(replacement.scene.targets,
        {k:np.repeat(v[:1],3,axis=0) for k,v in replacement.scene.neighbours.items()},
        np.array([0,3,3]), np.array([[.1,.2],[.3,.4],[.5,.6]]))
    replacement.x += .001
    result = replace_particles(state,rows,replacement)
    for row in range(8):
        expected = replacement.take(np.array([list(rows).index(row)])) if row in rows else state.take(np.array([row]))
        assert_states_equal(result.take(np.array([row])),expected)
    result.scene.neighbours['z'][0] = -1
    assert (state.scene.neighbours['z'] > 0).all()
    with pytest.raises(ValueError,match='unique'):
        replace_particles(state,np.array([1,1]),replacement)


def test_all_extinct_skips_mutation_rng_and_all_model_diagnostics(monkeypatch):
    target,state = population_fixture(extinct=(0,1))
    tracker,rng = ParticlePopulations(target,2,4),np.random.default_rng(928)
    before = deepcopy(rng.bit_generator.state)
    state,_ = tracker.reweight_resample(state,.1,1,rng)
    def forbidden(*args,**kwargs):
        raise AssertionError('extinct populations must not evaluate models or mutate')
    monkeypatch.setattr(target,'mutate',forbidden)
    monkeypatch.setattr(target,'full_summaries',forbidden)
    state,decision,active = tracker.mutate(state,.1,rng,block='physical')
    full,score,error = survivor_diagnostics(target,state,tracker.alive,4,None,forbidden)
    assert not decision.any() and not active.any() and error is None
    assert not full.any() and not score.any() and rng.bit_generator.state == before
    assert tracker.receipt()['log_normalizer'] == [None,None]
    json.dumps(tracker.receipt(),allow_nan=False)


@pytest.fixture
def driver(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('extinction_driver',scripts/'check_galsbi_particles.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('extinct', [(0,), (0,1), ()])
def test_driver_writes_complete_finite_receipt_and_artifacts_for_every_observation(tmp_path,monkeypatch,driver,reporter,extinct):
    target,state=population_fixture(extinct=extinct)
    reference=tmp_path/'reference'; reference.mkdir()
    target.observations.to_parquet(reference/'observations.parquet')
    write_json(reference/'protocol.json',dict(conditions=dict(zero_point=30.)))
    write_json(reference/'result.json',dict(preparation_checks_pass=True,
        observations_sha256=digest(reference/'observations.parquet')))
    doubles=[]
    class Model:
        def double(self): doubles.append(1)
    monkeypatch.setattr(driver,'V39_LIKE',SimpleNamespace(validate=lambda:None,flow_checkpoints=['fixture']))
    monkeypatch.setattr(driver,'load_measurement_model',lambda *a,**k:SimpleNamespace(model=Model()))
    monkeypatch.setattr(driver,'load_detection_classifier',lambda *a,**k:SimpleNamespace(model=Model()))
    monkeypatch.setattr(driver,'load_disk_response',lambda *a,**k:None)
    monkeypatch.setattr(driver,'SummaryLikelihood',lambda *a,**k:target.likelihood)
    monkeypatch.setattr(driver,'build_prior',lambda:(SimpleNamespace(p={}),None,None))
    monkeypatch.setattr(driver,'GalSBIScenePrior',lambda *a:None)
    monkeypatch.setattr(driver,'GalSBITargetCoordinates',lambda *a:None)
    monkeypatch.setattr(driver,'MeasurementParticleProposal',SimpleNamespace(
        from_measurement=lambda *a,**k:SimpleNamespace(metadata={'fixture':True})))
    monkeypatch.setattr(driver,'GalSBIParticleTarget',lambda *a,**k:target)
    monkeypatch.setattr(target,'initialize',lambda rng:state)
    class Scores:
        def __init__(self,*a,**k): pass
        def evaluate(self,full,observed,*a,**k): return np.zeros(len(full)),full[:,:2]
    monkeypatch.setattr(driver,'SummaryGradients',Scores)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['check_galsbi_particles.py','--reference',str(reference),
        '--output',str(output),'--observation-indices','0','1','--particles','4',
        '--stages','2','--moves','2','--seed','929','--device','cpu'])
    monkeypatch.chdir(Path(driver.__file__).resolve().parents[1])
    driver.main()
    protocol=json.loads((output/'protocol.json').read_text())
    result=json.loads((output/'result.json').read_text())
    decoded=reporter.validate_completed_result(protocol,result)
    assert decoded[4].tolist() == [i in extinct for i in range(2)]
    assert result['status'] == ('completed_with_extinction' if extinct else 'completed')
    assert result['final_stage']['stage'] == 2 and result['final_stage']['beta'] == 1.
    with np.load(output/'particles.npz') as stored:
        np.testing.assert_array_equal(stored['extinct'],decoded[4])
        np.testing.assert_array_equal(stored['log_normalizer'],decoded[0])
        assert np.isfinite(stored['scores']).all() and np.isfinite(stored['summaries']).all()
    if len(extinct)==2:
        assert not doubles and result['final_state_replay_max'] is None
    else:
        assert len(doubles)==2 and result['final_state_replay_max'] < 1e-12
    assert not (output/'failure.json').exists()


def explicit_runs(root,reporter,dead,*,seed_start=100):
    def alter(index,protocol,result):
        extinct=bool(dead[index]); protocol['extinction_schema']=EXTINCTION_SCHEMA
        protocol['implementation_sha256'][f'/snapshot_{index}/particle_populations.py']='d'*64
        receipt=dict(extinction_schema=EXTINCTION_SCHEMA,extinct=[extinct],
            extinction_stage=[1 if extinct else None],log_normalizer=[None if extinct else 0.])
        result.update(receipt,status='completed_with_extinction' if extinct else 'completed')
        result['final_stage'].update(receipt)
        if extinct:
            result.update(score_mean=[[None,None]],score_variance=[[None,None]])
            result['final_stage'].update(target_within_two_magnitudes=[None],target_over_three_magnitudes_fainter=[None])
        path=root/f'run{index}'
        with np.load(path/'particles.npz') as stored: scores=stored['scores'].copy()
        np.savez(path/'particles.npz',scores=np.zeros_like(scores) if extinct else scores,
            extinct=np.array([extinct]),log_normalizer=np.array([-np.inf if extinct else 0.]))
        np.savez(path/'scenes.npz',placeholder=np.array([0]))
        result['scenes_sha256']=digest(path/'scenes.npz')
    return make_runs(root,reporter,count=len(dead),seed_start=seed_start,alter=alter)


def test_report_retains_extinct_seeds_in_normalizer_and_error_denominators(tmp_path,reporter,monkeypatch):
    paths,raw=explicit_runs(tmp_path/'runs',reporter,[True,False,False,True])
    report=run_report(reporter,monkeypatch,paths,tmp_path/'report.json')
    row=report['observations'][0]; score=row['score']
    assert score['independent_populations']==4 and score['extinct_populations']==2
    np.testing.assert_allclose(score['weights'],[0,.5,.5,0])
    np.testing.assert_allclose(score['mean'],raw[[1,2]].mean((0,1)))
    assert np.exp(score['log_mean_normalizer'])==pytest.approx(.5)
    influence=np.zeros((4,2)); influence[[1,2]]=2*(raw[[1,2]].mean(1)-np.asarray(score['mean']))
    np.testing.assert_allclose(score['covariance'],influence.T@influence/12)
    assert not row['precision_gate'] and not row['all_populations_extinct']


def test_all_extinct_report_and_comparison_are_explicitly_undefined(tmp_path,reporter,monkeypatch):
    paths,_=explicit_runs(tmp_path/'dead',reporter,[True]*4)
    first=tmp_path/'dead.json'; report=run_report(reporter,monkeypatch,paths,first)
    row=report['observations'][0]
    assert row['all_populations_extinct'] and not report['all_precision_gates']
    assert row['score']['mean']==[None,None] and row['score']['log_mean_normalizer'] is None
    paths,_=explicit_runs(tmp_path/'live',reporter,[False]*4,seed_start=200)
    compared=run_report(reporter,monkeypatch,paths,tmp_path/'live.json',first)
    assert not compared['comparison_gate'] and not compared['comparison']['observations'][0]['agreement']


@pytest.mark.parametrize('failure',['status','mask','stage','non_null_dead','null_live','artifact_mask','missing_artifact','source'])
def test_explicit_receipts_do_not_hide_unknown_or_inconsistent_failures(tmp_path,reporter,monkeypatch,failure):
    paths,_=explicit_runs(tmp_path/'runs',reporter,[True,False,False,False])
    path=paths[0]; receipt=json.loads((path/'result.json').read_text())
    if failure=='status': receipt['status']='failed'
    elif failure=='mask': receipt['extinct']=[False]
    elif failure=='stage': receipt['extinction_stage']=[None]
    elif failure=='non_null_dead': receipt['score_mean']=[[0.,0.]]
    elif failure=='null_live':
        path=paths[1]; receipt=json.loads((path/'result.json').read_text()); receipt['log_normalizer']=[None]
    elif failure=='missing_artifact': (path/'scenes.npz').unlink()
    elif failure=='artifact_mask':
        np.savez(path/'particles.npz',extinct=np.array([False]),log_normalizer=np.array([-np.inf]))
        receipt['particles_sha256']=digest(path/'particles.npz')
    else:
        protocol=json.loads((path/'protocol.json').read_text())
        protocol['implementation_sha256']['/snapshot_0/particle_populations.py']='e'*64
        write_json(path/'protocol.json',protocol); receipt['protocol_sha256']=digest(path/'protocol.json')
    write_json(path/'result.json',receipt)
    with pytest.raises((ValueError,FileNotFoundError)):
        run_report(reporter,monkeypatch,paths,tmp_path/'report.json')
