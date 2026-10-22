"""Target-law identities must survive proposal fitting, reporting and controls."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from sbsi.particle_target_measure import (
    NUMERICAL_BRIDGE, NUMERICAL_ENDPOINT, NUMERICAL_MEASURE, NUMERICAL_SOURCES,
    ENVIRONMENT_TILT, numerical_identity,
)
from sbsi.particle_initial_proposal import FrozenParticleProposal, target_identity
from test_particle_initial_proposal import identity_protocol, fitted_fixture
from test_particle_report import make_runs

ROOT = Path(__file__).resolve().parents[1]


def module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('test_measure_'+name, ROOT/'scripts'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def numerical(protocol=None):
    result = deepcopy(identity_protocol() if protocol is None else protocol)
    result.update(target_prior='numerical', numerical_prior_measure=NUMERICAL_MEASURE,
                  bridge=NUMERICAL_BRIDGE, endpoint_target=NUMERICAL_ENDPOINT)
    result['implementation_sha256'].update({f'/source/{name}':hashlib.sha256(name.encode()).hexdigest()
                                            for name in NUMERICAL_SOURCES})
    return result


def test_legacy_analytic_identity_remains_unextended():
    p = identity_protocol()
    assert numerical_identity(p) == {}
    assert 'target_prior' not in target_identity(p)
    fitted, _ = fitted_fixture()
    fitted.validate_production(9, p)


@pytest.mark.parametrize('field', ['target_prior', 'numerical_prior_measure', 'bridge', 'endpoint_target'])
def test_partial_numerical_identity_fails(field):
    p = numerical();p.pop(field)
    with pytest.raises(ValueError):numerical_identity(p)


@pytest.mark.parametrize('name', NUMERICAL_SOURCES)
def test_every_numerical_density_dependency_is_mandatory(name):
    p = numerical();p['implementation_sha256'].pop('/source/'+name)
    with pytest.raises(ValueError, match='source hashes incomplete'):numerical_identity(p)


def test_numerical_source_identity_cannot_disagree_or_alias():
    p = numerical();p.update(numerical_identity(p))
    p['implementation_sha256']['/source/'+NUMERICAL_SOURCES[0]] = 'b'*64
    with pytest.raises(ValueError, match='differs'):numerical_identity(p)
    p = numerical();p['implementation_sha256']['/second/'+NUMERICAL_SOURCES[0]] = 'b'*64
    with pytest.raises(ValueError, match='duplicate'):numerical_identity(p)


def test_numerical_proposal_roundtrip_and_both_cross_law_rejections(tmp_path):
    old, _ = fitted_fixture();p = numerical()
    metadata = dict(old.metadata, **target_identity(p))
    new = FrozenParticleProposal(old.proposals, metadata)
    file = tmp_path/'proposal.json';new.save(file)
    restored = FrozenParticleProposal.load(file)
    assert restored.metadata == new.metadata
    restored.validate_production(9, p)
    with pytest.raises(ValueError, match='target_prior'):old.validate_production(9, p)
    with pytest.raises(ValueError, match='target_prior'):restored.validate_production(9, identity_protocol())
    p['implementation_sha256']['/source/numerical_native_intensity.py'] = 'c'*64
    with pytest.raises(ValueError, match='numerical_prior_source'):restored.validate_production(9, p)


def test_actual_driver_records_numerical_measure_before_model_loading(tmp_path, monkeypatch):
    driver = module('check_galsbi_particles', monkeypatch)
    reference = tmp_path/'reference';reference.mkdir()
    observations = reference/'observations.parquet'
    pd.DataFrame(dict(a=[0., 1., 2., 3.])).to_parquet(observations)
    (reference/'protocol.json').write_text('{}')
    (reference/'result.json').write_text(json.dumps(dict(preparation_checks_pass=True,
        observations_sha256=hashlib.sha256(observations.read_bytes()).hexdigest())))
    class BeforeModels(Exception):pass
    def stop(*args, **kwargs):raise BeforeModels
    monkeypatch.setattr(driver.torch, 'set_num_threads', stop)
    monkeypatch.chdir(ROOT)
    for law in ['analytic', 'numerical']:
        output = tmp_path/law
        monkeypatch.setattr(sys, 'argv', ['check_galsbi_particles', '--reference',str(reference),
            '--output',str(output),'--observation-indices','0','1','2','3','--seed','5500',
            '--device','cpu','--target-prior',law])
        with pytest.raises(BeforeModels):driver.main()
        saved = json.loads((output/'protocol.json').read_text())
        identity = numerical_identity(saved)
        assert bool(identity) == (law == 'numerical')
        if law == 'numerical':
            for name, checksum in identity['numerical_prior_source_sha256'].items():
                assert checksum == hashlib.sha256((ROOT/'sbsi'/name).read_bytes()).hexdigest()
    # Exercise the actual driver's frozen-proposal validation, before model load.
    fitted, _ = fitted_fixture()
    for pilot_law in ['analytic', 'numerical']:
        pilot = json.loads((tmp_path/pilot_law/'protocol.json').read_text())
        metadata = dict(fitted.metadata, **target_identity(pilot))
        proposal = FrozenParticleProposal(fitted.proposals*4, metadata)
        artifact = tmp_path/f'{pilot_law}_proposal.json';proposal.save(artifact)
        output = tmp_path/f'numerical_from_{pilot_law}'
        monkeypatch.setattr(sys, 'argv', ['check_galsbi_particles','--reference',str(reference),
            '--output',str(output),'--observation-indices','0','1','2','3','--seed','5500',
            '--device','cpu','--target-prior','numerical','--initial-proposal',str(artifact)])
        if pilot_law == 'numerical':
            with pytest.raises(BeforeModels):driver.main()
        else:
            with pytest.raises(ValueError, match='target_prior'):driver.main()
            assert not (output/'protocol.json').exists()


def test_full_raw_report_preserves_numerical_identity_and_rejects_analytic_comparison(tmp_path, monkeypatch):
    reporter = module('report_galsbi_particles', monkeypatch)
    def adapt(index, p, r):p.update(numerical(p))
    runs, _ = make_runs(tmp_path/'runs', reporter, alter=adapt)
    output = tmp_path/'report.json'
    monkeypatch.setattr(sys, 'argv', ['report_galsbi_particles', '--runs',*[str(x) for x in runs], '--output',str(output)])
    reporter.main()
    saved = json.loads(output.read_text())
    assert numerical_identity(saved, saved['target_source_sha256'])['target_prior'] == 'numerical'
    assert reporter.comparison_target_identity(saved, saved)['endpoint_target'] == NUMERICAL_ENDPOINT
    tilted = deepcopy(saved);tilted['bridge'] += ENVIRONMENT_TILT
    assert reporter.comparison_target_identity(saved, tilted)['mode'] == 'identical_explicit_endpoint'
    inconsistent = deepcopy(saved);inconsistent.pop('target_prior')
    with pytest.raises(ValueError, match='explicit numerical'):reporter.comparison_target_identity(saved, inconsistent)
    analytic = deepcopy(saved)
    for key in numerical_identity(saved, saved['target_source_sha256']):analytic.pop(key)
    analytic['bridge'] = 'analytic bridge'
    with pytest.raises(ValueError, match='prior identities'):reporter.comparison_target_identity(saved, analytic)


def test_phase_control_identity_binds_prior_law_and_density_bytes(tmp_path, monkeypatch):
    phase = module('check_particle_phase_control', monkeypatch)
    p = numerical()
    for name in (*phase.NATIVE_SOURCES, *NUMERICAL_SOURCES):
        path = ROOT/('configs' if name.endswith('.json') else 'sbsi')/name
        p['implementation_sha256'] = {k:v for k,v in p['implementation_sha256'].items() if Path(k).name != name}
        p['implementation_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = phase.phase_target_identity(p, ROOT)
    assert identity['target_prior'] == 'numerical'
    assert 'uniform numerical' in identity['phase_control']
    p['implementation_sha256'][str(ROOT/'sbsi/numerical_native_intensity.py')] = 'a'*64
    with pytest.raises(ValueError, match='source identity changed'):phase.phase_target_identity(p, ROOT)
