"""CPU-only driver receipts for optional raw environment draws and extinction."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

import sbsi.informed_environment as environment_module
from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
from sbsi.particle_initial_proposal import NATIVE_TARGET_SOURCES, fit_initial_proposal, target_identity
from test_galsbi_particle_target import ToyTarget
from test_informed_environment import fixture, AugmentEveryRow
from test_particle_extinction import driver
from test_particle_report import write_json, digest


@pytest.mark.parametrize('reuse_seed',[False,True])
def test_driver_preserves_all_initial_draws_and_fingerprints_optional_law(tmp_path,monkeypatch,driver,reuse_seed):
    target,state,_=fixture(8,objects=2)
    reference=tmp_path/'reference';reference.mkdir()
    target.observations.to_parquet(reference/'observations.parquet')
    write_json(reference/'protocol.json',dict(conditions=dict(zero_point=30.)))
    write_json(reference/'result.json',dict(preparation_checks_pass=True,
        observations_sha256=digest(reference/'observations.parquet')))
    root=Path(driver.__file__).resolve().parents[1]
    identity=dict(observations=[0,1],fixed_shear=[0.,0.],
        reference_protocol_sha256=digest(reference/'protocol.json'),
        observation_sha256=digest(reference/'observations.parquet'),density_batch_size=4096,
        bridge='q_environment(E) * q0(x)**(1-beta) * (p_analytic(x)*K(y|x,E))**beta',
        selection='none in fixed-y posterior; no population shear fit',
        implementation_sha256={str(root/('configs' if n.endswith('.json') else 'sbsi')/n):
            digest(root/('configs' if n.endswith('.json') else 'sbsi')/n) for n in NATIVE_TARGET_SOURCES})
    broad=[MeasurementParticleProposal(np.tile(state.x[4*i,:9],(2,1)),np.full(9,.03),[.5,.5]) for i in range(2)]
    frozen=fit_initial_proposal([state.x.reshape(2,4,10).copy() for _ in range(4)],broad,
        dict(target_identity(identity),pilot_seeds=[801,802,803,804]))
    artifact=tmp_path/'environment.json';frozen.save(artifact)
    actual_factory=environment_module.InformedEnvironmentProposal
    def make_informed(*args,**kwargs):
        informed=actual_factory(*args,**kwargs)
        for i,p in enumerate(informed.proposals):
            raw=state.x[4*i].copy()
            if i==0:raw[0]=np.log(4.)
            monkeypatch.setattr(p,'sample',lambda n,rng,raw=raw:np.tile(raw,(n,1)))
        original=informed.augment
        monkeypatch.setattr(informed,'augment',lambda scene,owner,rng:original(scene,owner,AugmentEveryRow()))
        return informed
    monkeypatch.setattr(environment_module,'InformedEnvironmentProposal',make_informed)
    class Model:
        def double(self):return self
    loaded=[]
    monkeypatch.setattr(driver,'V39_LIKE',SimpleNamespace(validate=lambda:None,flow_checkpoints=['fixture']))
    monkeypatch.setattr(driver,'load_measurement_model',lambda *a,**k:loaded.append(1) or SimpleNamespace(model=Model()))
    monkeypatch.setattr(driver,'load_detection_classifier',lambda *a,**k:SimpleNamespace(model=Model()))
    monkeypatch.setattr(driver,'load_disk_response',lambda *a,**k:None)
    monkeypatch.setattr(driver,'SummaryLikelihood',lambda *a,**k:target.likelihood)
    cosmo=SimpleNamespace(background=SimpleNamespace(dist_ang_a=lambda a:np.full_like(a,1000.)))
    monkeypatch.setattr(driver,'build_prior',lambda:(target.prior.galaxy,cosmo,None))
    monkeypatch.setattr(driver,'MeasurementParticleProposal',SimpleNamespace(from_measurement=lambda *a,**k:broad[0]))
    monkeypatch.setattr(driver,'GalSBIParticleTarget',ToyTarget)
    class Scores:
        def __init__(self,*a,**k):pass
        def evaluate(self,full,observed,*a,**k):return np.zeros(len(full)),full[:,:2]
    monkeypatch.setattr(driver,'SummaryGradients',Scores)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['check_galsbi_particles.py','--reference',str(reference),'--output',str(output),
        '--observation-indices','0','1','--particles','4','--stages','2','--moves','2',
        '--seed',str(801 if reuse_seed else 929),'--device','cpu','--informed-environment',str(artifact)])
    monkeypatch.chdir(root)
    if reuse_seed:
        with pytest.raises(ValueError,match='reuses an independent pilot seed'):driver.main()
        assert not loaded
        return
    driver.main()
    protocol=json.loads((output/'protocol.json').read_text())
    result=json.loads((output/'result.json').read_text())
    optional=protocol['informed_environment']
    assert optional['sha256']==digest(artifact)
    assert optional['metadata']['pilot_seeds']==[801,802,803,804]
    assert optional['metadata']['mixture_probability']==.5
    assert 'R_frozen_environment' in protocol['bridge']
    for p in optional['metadata']['proposals']:
        assert p['phase']=='independent uniform periodic'
        assert 'phase_centres' not in p and 'phase_concentrations' not in p
    recorded={Path(p).name for p in protocol['implementation_sha256']}
    assert {'informed_environment.py','numerical_native_intensity.py','poisson_augmentation.py','galsbi_prior.py','galsbi_scene_prior.py'} <= recorded
    assert result['extinct']==[True,False]
    receipt=result['informed_environment_initialization']
    assert receipt['original_particles_per_observation']==4
    assert receipt['zero_target_environment']==[4,0]
    assert receipt['sha256']==digest(output/'initial_environment.npz')
    with np.load(output/'initial_environment.npz',allow_pickle=False) as initial:
        assert initial['valid'].tolist()==[False]*4+[True]*4
        np.testing.assert_array_equal(initial['raw_coordinates'][:4,0],np.log(4.))
        assert np.isneginf(initial['log_potential'][:4]).all()
        assert initial['classification'][:4].tolist()==['type_redshift_support']*4
    with np.load(output/'particles.npz',allow_pickle=False) as final:
        assert final['environment_valid'].shape==(2,4)
        assert not final['environment_valid'][0].any() and final['environment_valid'][1].all()
        assert final['environment_raw_coordinates'].shape==(2,4,10)
        assert final['environment_raw_types'].shape==(2,4)
        assert final['environment_insertion_index'].shape==(2,4)
        assert np.isneginf(final['log_normalizer'][0])


def test_augmentation_probability_requires_optional_artifact(tmp_path,monkeypatch,driver):
    monkeypatch.setattr(sys,'argv',['check_galsbi_particles.py','--reference',str(tmp_path/'ref'),
        '--output',str(tmp_path/'out'),'--seed','1','--environment-mixture-probability','.5'])
    with pytest.raises(SystemExit):driver.main()
    assert not (tmp_path/'out').exists()
