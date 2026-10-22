import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from sbsi.particle_geometry import FrozenParticleGeometry
from sbsi.particle_initial_proposal import FrozenParticleProposal, NATIVE_TARGET_SOURCES
from test_particle_initial_proposal import fitted_fixture, identity_protocol
from test_particle_extinction import driver

ROOT = Path(__file__).resolve().parents[1]
def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

p = module('prior_reuse_provenance_test', 'prior_reuse_proposal_provenance.py')
h = module('prior_reuse_harness_test', 'benchmark_prior_reuse_smc.py')
startup = module('prior_reuse_startup_test', 'check_prior_reuse_startup.py')


def fixture(tmp_path, before=None, after=None, changes=None, identity=None):
    changes = changes or {name: ('c' * 64, 'd' * 64) for name in h.HASHES}
    before = before or {name: pair[0] for name, pair in changes.items()}
    for name in NATIVE_TARGET_SOURCES:
        key = ('configs/' if name.endswith('.json') else 'sbsi/') + name
        before.setdefault(key, 'c' * 64)
    after = after or dict(before, **{name: pair[1] for name, pair in changes.items()})
    sources = dict(baseline=before, optimized=after)
    arms = {arm: dict(source=str(tmp_path / arm), manifest_sha256=('a' if arm == 'baseline' else 'b') * 64)
            for arm in sources}
    fitted, _ = fitted_fixture()
    original = tmp_path / 'original.json'
    fitted.save(original)
    payload = p.load(original)
    if identity:
        payload['metadata'].update(identity)
    payload['metadata']['native_target_source_sha256'] = p.native_sources(before)
    original.write_text(json.dumps(payload))
    evidence = tmp_path / 'native.json'
    evidence.write_text(json.dumps(dict(status='passed', all_exactly_equal=True, post_run_hashes_match=True,
        development_rows=73728, probe_rows=15,
        source_sha256={f'/{arm}/{name}': sources[arm][name] for arm in sources for name in changes})))
    derived = tmp_path / 'derived.json'
    derived.write_text(json.dumps(p.derived_payload(original, evidence, arms, sources, changes)))
    d = dict(schema_version=2, initial_proposal=str(original), arms=arms,
        initial_proposals=dict(baseline=str(original), optimized=str(derived)),
        proposal_migration=dict(original=str(original), derived=str(derived),
            native_replay=str(evidence), native_replay_sha256=p.digest(evidence)))
    return d, sources, changes


def test_derived_artifact_preserves_numerical_law_rng_and_all_pilot_fields(tmp_path):
    d, sources, changes = fixture(tmp_path)
    original, derived = p.validate_migration(d, sources, changes)
    assert original['proposals'] == derived['proposals']
    assert original['coordinate_names'] == derived['coordinate_names']
    for key in original['metadata']:
        if key != 'native_target_source_sha256':
            assert original['metadata'][key] == derived['metadata'][key]
    a = FrozenParticleProposal.load(d['initial_proposal'])
    b = FrozenParticleProposal.load(d['initial_proposals']['optimized'])
    left, right = np.random.default_rng(735), np.random.default_rng(735)
    x, y = a.proposals[0].sample(128, left), b.proposals[0].sample(128, right)
    assert x.tobytes() == y.tobytes()
    assert a.proposals[0].log_prob(x).tobytes() == b.proposals[0].log_prob(y).tobytes()
    assert left.bit_generator.state == right.bit_generator.state
    current = identity_protocol()
    current['implementation_sha256']['/snapshot/galsbi_target_coordinates.py'] = 'd' * 64
    with pytest.raises(ValueError, match='native_target_source_sha256'):
        a.validate_production(2026103900, current)
    b.validate_production(2026103900, current)
    with pytest.raises(ValueError, match='pilot seed'):
        b.validate_production(1, current)


@pytest.mark.parametrize('case', ['numerical', 'pilot', 'source', 'extra', 'evidence'])
def test_only_explicit_bound_provenance_migration_allowed(tmp_path, case):
    d, sources, changes = fixture(tmp_path)
    path = Path(d['initial_proposals']['optimized'])
    value = p.load(path)
    if case == 'numerical': value['proposals'][0]['means'][2][0] += .001
    elif case == 'pilot': value['metadata']['pilot_seeds'][0] = 42
    elif case == 'source': value['metadata']['native_target_source_sha256']['galsbi_summary.py'] = 'f' * 64
    elif case == 'extra': value['metadata']['undocumented_allowance'] = True
    else: value['metadata'][p.MIGRATION_KEY]['native_replay']['sha256'] = '0' * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='beyond the validated'):
        p.validate_migration(d, sources, changes)


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('post_run_hashes_match', False),
    ('all_exactly_equal', False), ('development_rows', 73727), ('probe_rows', 14)])
def test_incomplete_native_replay_cannot_authorize_migration(tmp_path, field, value):
    d, sources, changes = fixture(tmp_path)
    path = Path(d['proposal_migration']['native_replay'])
    evidence = p.load(path); evidence[field] = value; path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='complete successful'):
        p.derived_payload(d['initial_proposal'], path, d['arms'], sources, changes)


def test_protocol_normalization_first_checks_complete_raw_provenance(tmp_path):
    d, sources, changes = fixture(tmp_path)
    derived = p.load(d['initial_proposals']['optimized'])
    raw = dict(initial_proposal=dict(path=d['initial_proposals']['optimized'],
        sha256=p.digest(d['initial_proposals']['optimized']), metadata=derived['metadata'],
        effective_law='unchanged'), science=[1., -0.])
    result = p.normalize_initial_protocol(raw, d, 'optimized', sources, changes)
    assert result['initial_proposal']['path'] == d['initial_proposal']
    assert result['initial_proposal']['metadata'] == p.load(d['initial_proposal'])['metadata']
    assert raw['initial_proposal']['path'] == d['initial_proposals']['optimized']
    assert result['science'] == raw['science']
    bad = copy.deepcopy(raw); bad['initial_proposal']['metadata']['pilot_seeds'][0] = 99
    with pytest.raises(ValueError, match='raw production proposal'):
        p.normalize_initial_protocol(bad, d, 'optimized', sources, changes)


def test_real_production_startup_rejects_original_and_accepts_derived_before_gpu(tmp_path, monkeypatch, driver):
    reference = tmp_path / 'reference'; reference.mkdir()
    pd.DataFrame(np.tile([.1, .2, 3., 200.], (12, 1)), columns=['e1', 'e2', 'radius', 'flux']).to_parquet(reference / 'observations.parquet')
    (reference / 'protocol.json').write_text(json.dumps(dict(conditions=dict(zero_point=30.))))
    (reference / 'result.json').write_text(json.dumps(dict(preparation_checks_pass=True,
        observations_sha256=p.digest(reference / 'observations.parquet'))))
    before = {name: values[0] for name, values in h.HASHES.items()}
    after = {name: values[1] for name, values in h.HASHES.items()}
    for name in NATIVE_TARGET_SOURCES:
        key = ('configs/' if name.endswith('.json') else 'sbsi/') + name
        if key not in before:
            before[key] = after[key] = p.digest(ROOT / key)
    identity = dict(observations=[11], fixed_shear=[0., 0.],
        reference_protocol_sha256=p.digest(reference / 'protocol.json'),
        observation_sha256=p.digest(reference / 'observations.parquet'), density_batch_size=4096,
        bridge='q_environment(E) * q0(x)**(1-beta) * (p_analytic(x)*K(y|x,E))**beta',
        selection='none in fixed-y posterior; no population shear fit')
    d, sources, changes = fixture(tmp_path, before, after, h.HASHES, identity)
    geometry_path = tmp_path / 'geometry.json'
    FrozenParticleGeometry(np.eye(5)[None], np.zeros(1), dict(observations=[11], pilot_seeds=[1, 2, 3, 4])).save(geometry_path)
    class ReachedPreModel(Exception): pass
    def stop(threads): raise ReachedPreModel()
    monkeypatch.setattr(driver.torch, 'set_num_threads', stop)
    def forbidden(*args, **kwargs): raise AssertionError('model/GPU loading must not happen in startup preflight')
    monkeypatch.setattr(driver, 'load_measurement_model', forbidden)
    seen_geometry = []
    real_geometry_validation = FrozenParticleGeometry.validate_production
    def checked_geometry(self, seed, observations):
        seen_geometry.append((seed, observations))
        return real_geometry_validation(self, seed, observations)
    monkeypatch.setattr(FrozenParticleGeometry, 'validate_production', checked_geometry)
    monkeypatch.chdir(ROOT)
    def run(arm, seed=2026103900):
        output = tmp_path / f'run_{arm}_{seed}'
        monkeypatch.setattr(sys, 'argv', ['check_galsbi_particles.py', '--reference', str(reference),
            '--output', str(output), '--observation-indices', '11', '--particles', '256',
            '--stages', '64', '--moves', '64', '--seed', str(seed), '--device', 'cuda',
            '--physical-geometry', str(geometry_path), '--initial-proposal', d['initial_proposals'][arm]])
        driver.main()
    with pytest.raises(ValueError, match='native_target_source_sha256'):
        run('baseline')
    assert not any((tmp_path / 'run_baseline_2026103900').iterdir())
    with pytest.raises(ReachedPreModel):
        run('optimized')
    assert seen_geometry == [(2026103900, [11]), (2026103900, [11])]
    output = tmp_path / 'run_optimized_2026103900'
    assert {v.name for v in output.iterdir()} == {'protocol.json'}
    raw = p.load(output / 'protocol.json')
    assert raw['initial_proposal']['metadata'][p.MIGRATION_KEY]['schema'] == p.MIGRATION_SCHEMA
    assert raw['physical_geometry']['sha256'] == p.digest(geometry_path)
    with pytest.raises(ValueError, match='pilot seed'):
        run('optimized', seed=1)


def test_geometry_still_rejects_wrong_observation_order():
    geometry = FrozenParticleGeometry(np.tile(np.eye(5), (2, 1, 1)), np.zeros(2),
        dict(observations=[11, 12], pilot_seeds=[1, 2, 3, 4]))
    with pytest.raises(ValueError, match='order'):
        geometry.validate_production(2026103900, [12, 11])


def test_failed_v1_cost_is_retained_and_no_optimized_samples_can_be_hidden(tmp_path):
    optimized = tmp_path / 'optimized'; optimized.mkdir()
    receipt = tmp_path / 'failed.json'
    receipt.write_text(json.dumps(dict(status='failed', replica=0, seed=2026103900,
        order=['baseline', 'optimized'], resources=dict(slurm_job_id='17446115'),
        arms=dict(baseline=dict(workers={'0': dict(returncode=0)}),
                  optimized=dict(workers={'0': dict(returncode=1)})))))
    log = tmp_path / 'failure.log'; log.write_text(startup.ERROR)
    cost = tmp_path / 'cost.json'
    process_cost = dict(wall_seconds=248.06, exit_code=1)
    cost.write_text(json.dumps(process_cost))
    item = dict(optimized_output=str(optimized), process_cost=process_cost)
    for key, path in [('receipt', receipt), ('log', log), ('cost', cost)]:
        item[key] = dict(path=str(path), sha256=p.digest(path))
    assert p.validate_failed_startup(item) == process_cost
    (optimized / 'particles.npz').write_bytes(b'must not be hidden')
    with pytest.raises(ValueError, match='must not contain any sampled'):
        p.validate_failed_startup(item)


def test_unknown_native_source_change_still_rejected_by_public_validator(tmp_path):
    d, _, _ = fixture(tmp_path)
    proposal = FrozenParticleProposal.load(d['initial_proposals']['optimized'])
    current = identity_protocol()
    current['implementation_sha256']['/snapshot/galsbi_target_coordinates.py'] = 'd' * 64
    current['implementation_sha256']['/snapshot/galsbi_summary.py'] = 'e' * 64
    with pytest.raises(ValueError, match='native_target_source_sha256'):
        proposal.validate_production(2026103900, current)
