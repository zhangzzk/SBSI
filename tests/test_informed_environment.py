"""Optional environment law, raw invalid draws, and full bridge/state checks."""
from copy import copy, deepcopy
from dataclasses import fields
from itertools import product
import importlib.util
import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import poisson, multivariate_t

import sbsi.galsbi_particle_target as target_module
import sbsi.informed_environment as environment_module
from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
from sbsi.galsbi_scene_prior import ARCSEC2_PER_SR, GalSBISceneBatch
from sbsi.informed_environment import EnvironmentPayload, InformedEnvironmentProposal
from sbsi.numerical_native_intensity import numerical_native_log_intensity
from sbsi.particle_populations import ParticlePopulations, replace_particles
from sbsi.poisson_augmentation import poisson_augmentation_log_tilt
from sbsi.scene_exchange import relocation_position_log_density, numerical_over_analytic
from sbsi.galsbi_prior import upper_gamma
from test_galsbi_particle_target import setup_target, PeriodicGaussian


def fixture(n=4, objects=1):
    target, state = setup_target(n=n)
    x = state.x.copy(); x[:, 0] = np.log(.5); x[:, 1] = 20.
    target.proposals = [PeriodicGaussian(x[0], sigma=.03) for _ in range(objects)]
    if objects > 1:
        target.observations = pd.concat([target.observations]*objects, ignore_index=True)
        target.owner = np.repeat(np.arange(objects), n//objects)
        target.observed = target.observations.iloc[target.owner].reset_index(drop=True)
        target.auxiliary = target.observed.iloc[:, 2:4].to_numpy()
    state = target.evaluate(x, state.scene)
    proposals = [MeasurementParticleProposal(np.tile(x[0, :9], (2, 1)),
        np.full((2, 9), .05), [.4, .6], component_family='student_t',
        phase_centres=[.3, .8], phase_concentrations=[9., 11.],
        metadata={'phase_centres':[99.,99.], 'phase_concentrations':[99.,99.]})
        for _ in range(objects)]
    frozen = SimpleNamespace(proposals=proposals,
        metadata={'observations':list(range(objects)), 'pilot_seeds':[10,11,12,13]})
    informed = InformedEnvironmentProposal(target.prior, target.coordinates, frozen, 'a'*64)
    return target, state, informed


def assert_payload_equal(left, right):
    for f in fields(left):
        np.testing.assert_array_equal(getattr(left, f.name), getattr(right, f.name))


def assert_state_equal(left, right):
    for name in ('x','log_prior','log_kernel','log_proposal','type_probability','summaries'):
        np.testing.assert_array_equal(getattr(left,name), getattr(right,name))
    np.testing.assert_array_equal(left.scene.indptr,right.scene.indptr)
    np.testing.assert_array_equal(left.scene.offset_arcsec,right.scene.offset_arcsec)
    for name in ('targets','neighbours'):
        for key,value in getattr(left.scene,name).items():
            np.testing.assert_array_equal(value,getattr(right.scene,name)[key])


def test_ordered_poisson_mixture_direct_density_and_normalization():
    intensity, h, a = np.array([.6, 1.4]), np.array([.8, .2]), .5
    base = lambda marks: np.exp(-2.)*np.prod(intensity[list(marks)])/math.factorial(len(marks))
    total = 0.
    for n in range(9):
        for marks in product(range(2), repeat=n):
            augmented = sum(h[mark]*base(marks[:j]+marks[j+1:])/n for j,mark in enumerate(marks)) if n else 0.
            expected = (1-a)*base(marks)+a*augmented
            tilt = poisson_augmentation_log_tilt(np.log(h[list(marks)]/intensity[list(marks)]),np.array([0,n]),a)[0]
            np.testing.assert_allclose(np.exp(tilt)*base(marks),expected,rtol=3e-15)
            total += expected
    np.testing.assert_allclose(total,(1-a)*poisson.cdf(8,2.)+a*poisson.cdf(7,2.),atol=2e-15)


def test_all_neighbour_density_has_native_units_type_jacobian_and_uniform_phase():
    target, state, informed = fixture(2)
    scene = GalSBISceneBatch({k:v[:1] for k,v in state.scene.targets.items()},
        state.scene.neighbours, np.array([0,2]), state.scene.offset_arcsec)
    x = target.coordinates.encode(scene.neighbours)
    _, jac = target.coordinates.decode(x,scene.neighbours['type'])
    proposal = informed.proposals[0]
    terms = np.column_stack([np.log(w)+multivariate_t.logpdf(x[:,:9],loc=mu,shape=np.diag(sd**2),df=5.)
        for w,mu,sd in zip(proposal.weights,proposal.means,proposal.scales)])
    h = (logsumexp(terms,axis=1)-np.log(2*np.pi)-np.log(2)-jac
        +relocation_position_log_density(scene.offset_arcsec,**informed.position_parameters))
    intensity = numerical_native_log_intensity(target.prior.galaxy,scene.neighbours)-np.log(ARCSEC2_PER_SR)
    expected = np.logaddexp(np.log(.5),np.log(.5)+logsumexp(h-intensity))
    np.testing.assert_allclose(informed.log_tilt(scene,np.array([0])),expected,atol=2e-12,rtol=0)
    metadata = informed.metadata['proposals'][0]
    assert metadata['phase'] == 'independent uniform periodic'
    assert 'phase_centres' not in metadata and 'phase_concentrations' not in metadata
    changed=x.copy();changed[:,9]+=.713
    np.testing.assert_array_equal(proposal.log_prob(x),proposal.log_prob(changed))
    assert metadata['source_artifact_sha256']=='a'*64


class AugmentEveryRow:
    def __init__(self, seed=123): self.delegate=np.random.default_rng(seed)
    def __getattr__(self,name): return getattr(self.delegate,name)
    def random(self,n): return np.zeros(n)


def test_raw_invalid_draws_skip_likelihood_keep_denominator_and_resample_normally(monkeypatch):
    target,state,informed=fixture()
    raw=np.tile(state.x[0],(4,1));raw[1,0]=np.log(4.);raw[2,1]=40.
    monkeypatch.setattr(informed.proposals[0],'sample',lambda n,rng:raw.copy())
    monkeypatch.setattr(target.proposals[0],'sample',lambda n,rng:np.tile(state.x[0],(n,1)))
    calls=[];photo=target.prior.galaxy.photometry
    def checked(z,M,c):
        assert (z<=3.).all() and (z>0).all();calls.append(len(z));return photo(z,M,c)
    target.prior.galaxy.photometry=checked
    target.informed_environment=informed
    before=target.audit['likelihood_rows']
    initial=target.initialize(AugmentEveryRow())
    payload=initial.environment_payload
    assert payload.valid.tolist()==[True,False,False,True]
    assert payload.classification.tolist()==['valid','type_redshift_support','intrinsic_photometric_cut','valid']
    np.testing.assert_array_equal(payload.raw_coordinates,raw)
    assert target.audit['likelihood_rows']-before==2
    assert np.isneginf(initial.log_kernel[~payload.valid]).all()
    with pytest.raises(ValueError,match='positive-beta'):
        target.mutate(initial,.1,np.random.default_rng(3),block='physical')
    tracker=ParticlePopulations(target,1,4)
    expected=logsumexp(.2*initial.potential)-np.log(4)
    moved,_=tracker.reweight_resample(initial,.2,1,np.random.default_rng(4))
    np.testing.assert_allclose(tracker.logz,[expected],atol=1e-12)
    assert moved.environment_payload.valid.all()
    np.testing.assert_allclose(moved.environment_payload.log_tilt,informed.log_tilt(moved.scene,target.owner),atol=1e-12)


def test_classified_numerical_luminosity_support_and_unknown_decode_errors(monkeypatch):
    target,state,informed=fixture()
    x=state.x[:1].copy();x[:,1]=23.
    target.prior.galaxy.p['lum_fct_m_max']=-18.373
    kept,data,reason=informed._decode_once(x,np.array([0]))
    assert not len(kept) and reason[0]=='numerical_luminosity_support'
    for column,value in [(0,-1000.),(2,1000.),(6,1000.),(7,1000.),(8,1000.)]:
        bad=state.x[:1].copy();bad[:,column]=value
        with pytest.raises(FloatingPointError): informed._decode_once(bad,np.array([0]))
    def failure(*args): raise RuntimeError('unknown photometry failure')
    monkeypatch.setattr(target.prior.galaxy,'photometry',failure)
    with pytest.raises(RuntimeError,match='unknown photometry'):
        informed._decode_once(state.x[:1],np.array([0]))
    target.prior.galaxy.z_max['blue']=.1
    outside=state.x[:1].copy();outside[:,0]=np.log(.2)
    _,_,reason=informed._decode_once(outside,np.array([1]))
    assert reason[0]=='type_redshift_support'  # no photometry call even inside the other type's domain


def test_clamp_atoms_and_unknown_generator_failures_raise(monkeypatch):
    target,state,informed=fixture()
    coverage=environment_module.luminosity_table_coverage
    def atom(prior,name,z):
        v=coverage(prior,name,z)
        v['clamp_atom_mass']=np.ones(len(z))
        # Each fixture neighbour has a unique z; use its unchanged M_B.
        data=state.scene.neighbours
        v['clamp_magnitude']=np.array([data['M_B'][np.flatnonzero(data['z']==zz)[0]] for zz in z])
        return v
    monkeypatch.setattr(environment_module,'luminosity_table_coverage',atom)
    with pytest.raises(FloatingPointError,match='clamp atom'):
        informed.log_tilt(state.scene,target.owner)
    target.informed_environment=informed
    def failed(*args):raise RuntimeError('conditional luminosity inversion outside table support')
    monkeypatch.setattr(target.prior,'sample_given_targets',failed)
    with pytest.raises(RuntimeError,match='outside table support'):
        target.initialize(np.random.default_rng(2))


@pytest.mark.parametrize('block',['physical','nuisance','sed','independence','environment','positions','exchange','exchange_relocate'])
@pytest.mark.parametrize('beta',[.001,.4,1.])
def test_actual_mutation_ratios_add_tilt_for_every_environment_changing_block(monkeypatch,block,beta):
    target,old,informed=fixture(1)
    if block=='environment':
        original=target.prior.refresh
        monkeypatch.setattr(target.prior,'refresh',lambda scene,fraction,seed:original(scene,1.,seed))
    optional=copy(target);optional.informed_environment=informed
    state=optional.evaluate(old.x,old.scene)
    ratios=[]
    monkeypatch.setattr(target_module,'metropolis_accept',lambda a,b,u:ratios.append(b-a) or np.ones(len(a),bool))
    plain,_,_=target.mutate(old,beta,np.random.default_rng(134),block=block)
    changed,_,_=optional.mutate(state,beta,np.random.default_rng(134),block=block)
    assert_state_equal(plain,changed)
    new_tilt=informed.log_tilt(changed.scene,optional.owner)
    delta=(1-beta)*(new_tilt-state.environment_payload.log_tilt)
    assert np.isfinite(ratios).all()
    np.testing.assert_allclose(ratios[1]-ratios[0],delta,atol=2e-11,rtol=0)
    np.testing.assert_allclose(changed.potential,plain.potential-new_tilt,atol=2e-12,rtol=0)


@pytest.mark.parametrize('block',['physical','nuisance','sed','independence'])
def test_target_only_moves_cache_tilt_and_keep_target_hastings_separate(monkeypatch,block):
    target,state,informed=fixture(1);target.informed_environment=informed
    state=target.evaluate(state.x,state.scene)
    monkeypatch.setattr(informed,'log_tilt',lambda *args:pytest.fail('E unchanged: use cached tilt'))
    moved,_,_=target.mutate(state,.4,np.random.default_rng(9),block=block)
    assert_payload_equal(state.environment_payload,moved.environment_payload)


def test_known_off_support_exchange_rejects_without_evaluating_bad_environment(monkeypatch):
    target,state,informed=fixture(1);target.informed_environment=informed
    state=target.evaluate(state.x,state.scene)
    def unsupported(prior,scene,probability,rng,**kwargs):
        bad=scene.take(np.array([0]));bad.neighbours['M_B'][:]=100.
        return bad,np.array([-np.inf]),np.array([True])
    monkeypatch.setattr(target_module,'propose_native_exchange',unsupported)
    moved,accepted,active=target.mutate(state,.4,np.random.default_rng(2),block='exchange')
    assert not accepted.any() and active.all()
    assert_state_equal(moved,state);assert_payload_equal(moved.environment_payload,state.environment_payload)


def test_exchange_helper_continuous_tail_strip_gets_known_zero_rejection():
    target,state,informed=fixture(1)
    galaxy=target.prior.galaxy
    galaxy.p['gals_mag_min']=0.  # Expose the actual finite-table tail in the toy prior.
    z=state.scene.targets['z']
    ms,_,alpha=galaxy.luminosity_parameters(z,'red')
    logmax=np.log(upper_gamma(alpha+1,10**(.4*(ms-galaxy.magnitude_cap(z)))))
    tab=galaxy.tables['red'];logtiny=np.log(np.finfo(float).tiny)
    assert tab['log_g'][0]-logmax[0] < logtiny
    logu=(tab['log_g'][0]-logmax[0]+logtiny)/2
    M=ms-np.interp(logmax+logu,tab['log_g'],tab['inverse_y'])
    x=state.x.copy();x[:,1]=M+40.
    target.informed_environment=informed
    state=target.evaluate(x,state.scene)
    assert np.isfinite(state.log_prior).all()
    ratio,audit=numerical_over_analytic(galaxy,state.scene.targets,return_support_audit=True)
    assert audit['legacy_finite'].all() and audit['below_uniform_clamp'].all()
    assert np.isneginf(ratio).all()
    assert np.isneginf(numerical_native_log_intensity(galaxy,state.scene.targets)).all()
    moved,accepted,active=target.mutate(state,.4,np.random.default_rng(2),block='exchange')
    assert not accepted.any() and active.all()
    assert_state_equal(moved,state);assert_payload_equal(moved.environment_payload,state.environment_payload)


def test_payload_take_scatter_retention_and_absorbing_extinction(monkeypatch):
    target,state,informed=fixture(8,objects=2);target.informed_environment=informed
    raw=np.tile(state.x[0],(4,1));bad=raw.copy();bad[:,0]=np.log(4.)
    monkeypatch.setattr(informed.proposals[0],'sample',lambda n,rng:bad.copy())
    monkeypatch.setattr(informed.proposals[1],'sample',lambda n,rng:raw.copy())
    for p in target.proposals:monkeypatch.setattr(p,'sample',lambda n,rng:np.tile(state.x[0],(n,1)))
    state=target.initialize(AugmentEveryRow())
    repeat=np.array([6,6,1]);sample=state.take(repeat)
    assert_payload_equal(sample.environment_payload,state.environment_payload.take(repeat))
    replacement=state.take(np.array([7,4]));replacement.environment_payload.log_tilt+=1
    scattered=replace_particles(state,np.array([6,5]),replacement)
    assert_payload_equal(scattered.environment_payload.take(np.array([6,5])),replacement.environment_payload)
    retained=state.environment_payload.retained(scattered.environment_payload,np.arange(8)%2==0)
    assert_payload_equal(retained.take(np.array([6])),scattered.environment_payload.take(np.array([6])))
    tracker=ParticlePopulations(target,2,4);rng=np.random.default_rng(39)
    state,_=tracker.reweight_resample(state,.1,1,rng)
    assert tracker.alive.tolist()==[False,True]
    extinct=state.take(np.arange(4))
    state,decision,active=tracker.mutate(state,.1,rng,block='physical')
    assert not active[:4].any() and not decision[:4].any()
    assert_payload_equal(state.environment_payload.take(np.arange(4)),extinct.environment_payload)
    assert np.isneginf(tracker.logz[0]) and state.environment_payload.valid[4:].all()


def test_disabled_default_replays_frozen_original_bitwise(monkeypatch):
    path=os.environ.get('SBSI_ENVIRONMENT_BASELINE_TARGET')
    if not path:pytest.skip('set frozen pre-extension target path for independent bitwise replay')
    spec=importlib.util.spec_from_file_location('sbsi._environment_baseline_target',path)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    target,state,_=fixture(4)
    class Legacy(type(target),module.GalSBIParticleTarget):
        evaluate=module.GalSBIParticleTarget.evaluate
        initialize=module.GalSBIParticleTarget.initialize
        mutate=module.GalSBIParticleTarget.mutate
    old=copy(target);old.__class__=Legacy
    initial_rng,baseline_rng=np.random.default_rng(37),np.random.default_rng(37)
    assert_state_equal(target.initialize(initial_rng),old.initialize(baseline_rng))
    assert initial_rng.bit_generator.state==baseline_rng.bit_generator.state
    legacy=old.evaluate(state.x,state.scene)
    newrng,oldrng=np.random.default_rng(928),np.random.default_rng(928)
    for beta in [.001,.4,1.]:
        for block in ['physical','nuisance','sed','environment','positions','exchange','exchange_relocate','independence']:
            state,a,b=target.mutate(state,beta,newrng,block=block)
            legacy,c,d=old.mutate(legacy,beta,oldrng,block=block)
            assert_state_equal(state,legacy)
            np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
            assert newrng.bit_generator.state==oldrng.bit_generator.state
            assert state.environment_payload is None
