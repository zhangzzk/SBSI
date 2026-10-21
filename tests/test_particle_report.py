"""Receipt compatibility and independent-population report calculations."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def reporter(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('particle_report_audit', scripts/'report_galsbi_particles.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def make_runs(root, reporter, *, seed_start=100, count=8, alter=None):
    paths, raw = [], []
    for r in range(count):
        path = root/f'run{r}'
        path.mkdir(parents=True)
        # Duplicate particles intentionally share ancestry. Their raw second
        # moments are empirical identities, not independent-sample estimates.
        scores = np.array([[-2., -1.], [-2., -1.], [2., 1.], [2., 1.]])
        scores += [.01*r, .02*r]
        raw.append(scores)
        np.savez(path/'particles.npz', scores=scores[None])
        protocol = dict(seed=seed_start+r, observations=[11], fixed_shear=[0., 0.],
            reference_protocol_sha256='a'*64, observation_sha256='b'*64,
            density_batch_size=4096, particles=4, stages=8, moves=4,
            betas=np.linspace(0, 1, 9).tolist(), scale=1., proposals=[{'family':'gaussian'}],
            blocks=['physical', 'nuisance'], algorithm='fixed annealed particles',
            bridge='q0**(1-beta)*(p*K)**beta', selection='none',
            # Each root differs while the source identity remains identical.
            implementation_sha256={f'/snapshot_{r}/{name}':sha256(name.encode()).hexdigest()
                                   for name in (*reporter.TARGET_SOURCES, 'annealed_particles.py')})
        result = dict(status='completed', observations=[11], particles=4,
            log_normalizer=[0.], score_mean=scores.mean(0)[None].tolist(),
            score_variance=scores.var(0, ddof=1)[None].tolist(),
            final_stage=dict(stage=8, beta=1., log_normalizer=[0.],
                             target_within_two_magnitudes=[.5], target_over_three_magnitudes_fainter=[.25]),
            timing_seconds=dict(initialization=1., sampling=2., total=3.5))
        if alter is not None:
            alter(r, protocol, result)
        write_json(path/'protocol.json', protocol)
        result.update(protocol_sha256=digest(path/'protocol.json'), particles_sha256=digest(path/'particles.npz'))
        write_json(path/'result.json', result)
        paths.append(path)
    return paths, np.asarray(raw)


def run_report(reporter, monkeypatch, paths, output, comparison=None):
    argv = ['report_galsbi_particles.py', '--runs', *map(str, paths), '--output', str(output)]
    if comparison is not None:
        argv += ['--comparison', str(comparison)]
    monkeypatch.setattr(sys, 'argv', argv)
    reporter.main()
    return json.loads(output.read_text())


def test_report_matches_pooled_empirical_variance_and_independent_mean_covariance(tmp_path, reporter, monkeypatch):
    paths, raw = make_runs(tmp_path/'runs', reporter)
    report = run_report(reporter, monkeypatch, paths, tmp_path/'report.json')
    row = report['observations'][0]
    pooled = raw.reshape(-1, 2)
    np.testing.assert_allclose(row['score']['mean'], pooled.mean(0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(row['posterior_score_variance_estimate'], pooled.var(0), rtol=0, atol=2e-15)
    covariance = np.cov(raw.mean(1).T, ddof=1)/len(raw)
    np.testing.assert_allclose(row['score']['covariance'], covariance, rtol=0, atol=1e-16)
    np.testing.assert_allclose(row['score_effective_count_from_replicates'], pooled.var(0)/np.diag(covariance))
    assert report['all_precision_gates'] is True
    assert report['accepted_sampler'] is False and report['heldout_gate'] is False


@pytest.mark.parametrize('kind', ['blocks', 'target', 'geometry', 'result_observations'])
def test_changed_group_configuration_or_receipt_is_rejected(tmp_path, reporter, monkeypatch, kind):
    def alter(index, protocol, result):
        if index != 1:
            return
        if kind == 'blocks':
            protocol['blocks'] = ['nuisance', 'physical']
        elif kind == 'target':
            protocol['implementation_sha256'][f'/snapshot_{index}/galsbi_particle_target.py'] = 'f'*64
        elif kind == 'geometry':
            protocol['physical_geometry'] = dict(path='/geometry.json', sha256='c'*64, metadata={})
        else:
            result['observations'] = [12]
    paths, _ = make_runs(tmp_path/'runs', reporter, alter=alter)
    output = tmp_path/'report.json'
    with pytest.raises(ValueError):
        run_report(reporter, monkeypatch, paths, output)
    assert not output.exists()


def test_same_frozen_geometry_at_different_paths_is_compatible(tmp_path, reporter, monkeypatch):
    def alter(index, protocol, result):
        protocol['physical_geometry'] = dict(path=f'/copy{index}/geometry.json', sha256='c'*64,
                                             metadata={'pilot_seeds':[12]})
    paths, _ = make_runs(tmp_path/'runs', reporter, alter=alter)
    report = run_report(reporter, monkeypatch, paths, tmp_path/'report.json')
    assert report['physical_geometry'] == dict(sha256='c'*64, metadata={'pilot_seeds':[12]})


def test_comparison_requires_matching_target_sources_but_allows_changed_proposal(tmp_path, reporter, monkeypatch):
    paths, _ = make_runs(tmp_path/'first', reporter)
    first = tmp_path/'first.json'
    run_report(reporter, monkeypatch, paths, first)

    def change_proposal(index, protocol, result):
        protocol['proposals'] = [{'family':'student_t'}]
    paths, _ = make_runs(tmp_path/'second', reporter, seed_start=200, alter=change_proposal)
    second = run_report(reporter, monkeypatch, paths, tmp_path/'second.json', first)
    assert second['comparison_gate'] is True and second['accepted_sampler'] is False

    def change_target(index, protocol, result):
        protocol['implementation_sha256'][f'/snapshot_{index}/galsbi_particle_target.py'] = 'f'*64
    paths, _ = make_runs(tmp_path/'third', reporter, seed_start=300, alter=change_target)
    with pytest.raises(ValueError, match='comparison targets'):
        run_report(reporter, monkeypatch, paths, tmp_path/'third.json', first)


@pytest.mark.parametrize('failure', ['missing', 'unknown', 'null_normalizer'])
def test_missing_failed_or_nonfinite_run_is_never_replaced_by_zero(tmp_path, reporter, monkeypatch, failure):
    paths, _ = make_runs(tmp_path/'runs', reporter)
    final = paths[-1]/'result.json'
    if failure == 'missing':
        final.unlink()
    else:
        result = json.loads(final.read_text())
        if failure == 'unknown':
            result['status'] = 'failed'
        else:
            result['log_normalizer'] = [None]
        write_json(final, result)
    output = tmp_path/'report.json'
    with pytest.raises((ValueError, FileNotFoundError)):
        run_report(reporter, monkeypatch, paths, output)
    assert not output.exists()


def test_duplicate_source_basename_is_rejected(reporter):
    protocol = dict(implementation_sha256={f'/source/{name}':'a'*64 for name in reporter.TARGET_SOURCES})
    protocol['implementation_sha256']['/other/galsbi_summary.py'] = 'a'*64
    with pytest.raises(ValueError, match='unique source basenames'):
        reporter.implementation_fingerprint(protocol)


def test_comparison_rejects_different_normalizers_even_with_identical_scores(tmp_path, reporter, monkeypatch):
    paths, _ = make_runs(tmp_path/'first', reporter)
    first = tmp_path/'first.json'
    run_report(reporter, monkeypatch, paths, first)

    def change_mass(index, protocol, result):
        result['log_normalizer'] = [np.log(2.)]
        result['final_stage']['log_normalizer'] = [np.log(2.)]
    paths, _ = make_runs(tmp_path/'second', reporter, seed_start=200, alter=change_mass)
    report = run_report(reporter, monkeypatch, paths, tmp_path/'second.json', first)
    comparison = report['comparison']['observations'][0]
    assert report['all_precision_gates'] is True
    assert comparison['score_difference'] == [0., 0.]
    assert comparison['normalizer_scaled_difference'] == .5
    assert comparison['normalizer_scaled_combined_mcse'] == 0.
    assert comparison['normalizer_agreement'] is False
    assert report['comparison_gate'] is False


@pytest.mark.parametrize('common_log_scale', [-1000., 0., 1000.])
def test_normalizer_comparison_matches_mass_standard_errors(reporter, common_log_scale):
    first = dict(log_mean_normalizer=common_log_scale+np.log(2.), normalizer_relative_mcse=.05)
    second = dict(log_mean_normalizer=common_log_scale, normalizer_relative_mcse=.1)
    result = reporter.compare_normalizers(first, second)
    np.testing.assert_allclose(result['normalizer_scaled_difference'], .5, atol=3e-14)
    np.testing.assert_allclose(result['normalizer_scaled_combined_mcse'], np.sqrt(.02)/2, atol=3e-14)
    assert result['normalizer_agreement'] is False
    first['normalizer_relative_mcse'] = .2
    assert reporter.compare_normalizers(first, second)['normalizer_agreement'] is True


@pytest.mark.parametrize('artifact', ['physical_geometry', 'initial_proposal', 'informed_environment'])
def test_report_rejects_production_pilot_overlap(tmp_path, reporter, monkeypatch, artifact):
    def add_pilot(index, protocol, result):
        protocol[artifact] = dict(sha256='c'*64, metadata=dict(pilot_seeds=[101]))
    paths, _ = make_runs(tmp_path/'runs', reporter, alter=add_pilot)
    with pytest.raises(ValueError, match='production seeds overlap'):
        run_report(reporter, monkeypatch, paths, tmp_path/'report.json')


@pytest.mark.parametrize('artifact', ['physical_geometry', 'initial_proposal', 'informed_environment'])
@pytest.mark.parametrize('pilot_in_first', [True, False])
def test_comparison_rejects_other_groups_pilot_seeds(tmp_path, reporter, monkeypatch, artifact, pilot_in_first):
    def add_pilot(index, protocol, result):
        protocol[artifact] = dict(sha256='c'*64, metadata=dict(pilot_seeds=[201 if pilot_in_first else 101]))
    paths, _ = make_runs(tmp_path/'first', reporter, alter=add_pilot if pilot_in_first else None)
    first = tmp_path/'first.json'
    run_report(reporter, monkeypatch, paths, first)
    paths, _ = make_runs(tmp_path/'second', reporter, seed_start=200,
                         alter=None if pilot_in_first else add_pilot)
    with pytest.raises(ValueError, match='production seeds overlap'):
        run_report(reporter, monkeypatch, paths, tmp_path/'second.json', first)


def test_changed_initial_proposal_is_rejected_within_group(tmp_path, reporter, monkeypatch):
    def change_proposal(index, protocol, result):
        protocol['initial_proposal'] = dict(sha256=('c' if index else 'd')*64,
                                          metadata=dict(pilot_seeds=[12]))
    paths, _ = make_runs(tmp_path/'runs', reporter, alter=change_proposal)
    with pytest.raises(ValueError, match='same frozen initial proposal'):
        run_report(reporter, monkeypatch, paths, tmp_path/'report.json')


def informed_environment_record(path='/environment.json'):
    return dict(path=path, sha256='c'*64,
        source_metadata=dict(pilot_seeds=[12, 13, 14, 15], observations=[11]),
        source_pilot_validation_bridge='q0**(1-beta)*(p*K)**beta',
        metadata=dict(pilot_seeds=[12, 13, 14, 15], observations=[11],
            source_artifact_sha256='c'*64, mixture_probability=.5,
            position=dict(scale=.1, aperture_arcsec=10., central_weight=.9),
            phase_override='independent uniform', proposals=[dict(student_df=5.)]))


def test_copied_environment_artifact_preserves_full_fingerprint(tmp_path, reporter, monkeypatch):
    def add_environment(index, protocol, result):
        protocol['informed_environment'] = informed_environment_record(f'/copy{index}/environment.json')
    paths, _ = make_runs(tmp_path/'runs', reporter, alter=add_environment)
    report = run_report(reporter, monkeypatch, paths, tmp_path/'report.json')
    expected = informed_environment_record()
    del expected['path']
    assert report['informed_environment'] == expected


@pytest.mark.parametrize('change', ['artifact', 'mixture', 'position', 'source', 'bridge', 'proposal', 'missing'])
def test_environment_configuration_cannot_change_within_group(tmp_path, reporter, monkeypatch, change):
    def alter(index, protocol, result):
        environment = informed_environment_record()
        protocol['informed_environment'] = environment
        if index != 1:
            return
        if change == 'artifact':
            environment['sha256'] = 'd'*64
        elif change == 'mixture':
            environment['metadata']['mixture_probability'] = .6
        elif change == 'position':
            environment['metadata']['position']['central_weight'] = .8
        elif change == 'source':
            environment['source_metadata']['observations'] = [12]
        elif change == 'bridge':
            environment['source_pilot_validation_bridge'] += ' * R'
        elif change == 'proposal':
            environment['metadata']['proposals'][0]['student_df'] = 6.
        else:
            del protocol['informed_environment']
    paths, _ = make_runs(tmp_path/'runs', reporter, alter=alter)
    output = tmp_path/'report.json'
    with pytest.raises(ValueError, match='same frozen informed environment'):
        run_report(reporter, monkeypatch, paths, output)
    assert not output.exists()


@pytest.mark.parametrize('change', ['missing', 'inconsistent'])
def test_environment_pilot_provenance_is_required(tmp_path, reporter, monkeypatch, change):
    def alter(index, protocol, result):
        environment = informed_environment_record()
        protocol['informed_environment'] = environment
        if change == 'missing':
            del environment['metadata']['pilot_seeds']
        else:
            environment['source_metadata']['pilot_seeds'] = [101, 102, 103, 104]
    paths, _ = make_runs(tmp_path/'runs', reporter, alter=alter)
    with pytest.raises(ValueError, match='pilot seeds'):
        run_report(reporter, monkeypatch, paths, tmp_path/'report.json')


@pytest.mark.parametrize('mass_factor', [1., 2.])
def test_cross_bridge_comparison_requires_same_declared_endpoint_and_sources(tmp_path, reporter, monkeypatch, mass_factor):
    def original(index, protocol, result):
        protocol['endpoint_target'] = reporter.ENDPOINT_TARGET
    paths, _ = make_runs(tmp_path/'first', reporter, alter=original)
    first = tmp_path/'first.json'
    run_report(reporter, monkeypatch, paths, first)

    def informed(index, protocol, result):
        original(index, protocol, result)
        protocol['informed_environment'] = informed_environment_record()
        protocol['bridge'] += ' * R_frozen_environment(E)**(1-beta)'
        result['log_normalizer'] = [np.log(mass_factor)]
        result['final_stage']['log_normalizer'] = [np.log(mass_factor)]
    paths, _ = make_runs(tmp_path/'second', reporter, seed_start=200, alter=informed)
    report = run_report(reporter, monkeypatch, paths, tmp_path/'second.json', first)
    assert report['comparison_gate'] is (mass_factor == 1.)
    assert report['comparison']['observations'][0]['normalizer_agreement'] is (mass_factor == 1.)
    compatibility = report['comparison']['bridge_compatibility']
    assert compatibility['mode'] == 'identical_explicit_endpoint'
    assert compatibility['endpoint_target'] == reporter.ENDPOINT_TARGET
    assert compatibility['current_bridge'] != compatibility['comparison_bridge']
    assert report['target_source_sha256'] == json.loads(first.read_text())['target_source_sha256']


@pytest.mark.parametrize('failure', ['endpoint', 'missing_first', 'missing_second', 'target_source', 'old_report'])
def test_cross_bridge_comparison_rejects_unproven_identity(tmp_path, reporter, monkeypatch, failure):
    def original(index, protocol, result):
        if failure != 'missing_first':
            protocol['endpoint_target'] = reporter.ENDPOINT_TARGET
    paths, _ = make_runs(tmp_path/'first', reporter, alter=original)
    first = tmp_path/'first.json'
    run_report(reporter, monkeypatch, paths, first)
    if failure == 'old_report':
        report = json.loads(first.read_text())
        report['diagnostics_version'] = 'independent-populations-v4-normalizer-comparison'
        write_json(first, report)

    def changed(index, protocol, result):
        if failure != 'missing_second':
            protocol['endpoint_target'] = reporter.ENDPOINT_TARGET
        if failure == 'endpoint':
            protocol['endpoint_target'] = 'a different posterior'
        if failure == 'target_source':
            protocol['implementation_sha256'][f'/snapshot_{index}/galsbi_particle_target.py'] = 'f'*64
        protocol['bridge'] += ' * R_frozen_environment(E)**(1-beta)'
    paths, _ = make_runs(tmp_path/'second', reporter, seed_start=200, alter=changed)
    output = tmp_path/'second.json'
    with pytest.raises(ValueError, match='comparison|endpoint'):
        run_report(reporter, monkeypatch, paths, output, first)
    assert not output.exists()


def test_same_bridge_does_not_require_new_endpoint_field(tmp_path, reporter, monkeypatch):
    paths, _ = make_runs(tmp_path/'first', reporter)
    first = tmp_path/'first.json'
    run_report(reporter, monkeypatch, paths, first)
    def declare_endpoint(index, protocol, result):
        protocol['endpoint_target'] = reporter.ENDPOINT_TARGET
    paths, _ = make_runs(tmp_path/'second', reporter, seed_start=200, alter=declare_endpoint)
    report = run_report(reporter, monkeypatch, paths, tmp_path/'second.json', first)
    assert report['comparison_gate'] is True
    assert report['comparison']['bridge_compatibility']['mode'] == 'identical_literal_bridge'


def test_endpoint_declaration_cannot_change_within_group(tmp_path, reporter, monkeypatch):
    def alter(index, protocol, result):
        if index:
            protocol['endpoint_target'] = reporter.ENDPOINT_TARGET
    paths, _ = make_runs(tmp_path/'runs', reporter, alter=alter)
    with pytest.raises(ValueError, match='same endpoint target'):
        run_report(reporter, monkeypatch, paths, tmp_path/'report.json')


def test_literal_bridge_match_cannot_hide_conflicting_endpoint(tmp_path, reporter, monkeypatch):
    def declare(index, protocol, result):
        protocol['endpoint_target'] = reporter.ENDPOINT_TARGET
    paths, _ = make_runs(tmp_path/'first', reporter, alter=declare)
    first = tmp_path/'first.json'
    run_report(reporter, monkeypatch, paths, first)
    def conflict(index, protocol, result):
        protocol['endpoint_target'] = 'q_environment(E) * p_other(x) * K(y|x,E)'
    paths, _ = make_runs(tmp_path/'second', reporter, seed_start=200, alter=conflict)
    with pytest.raises(ValueError, match='endpoint target identities'):
        run_report(reporter, monkeypatch, paths, tmp_path/'second.json', first)
