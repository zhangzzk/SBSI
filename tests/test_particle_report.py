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
