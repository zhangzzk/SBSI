"""Guard against declaring missing or individually trapped replicates converged."""
import json
import hashlib
import numpy as np
from _script_loader import load_script_module

REPORT = load_script_module('report_galsbi_sampler_runs.py')


def test_missing_runs_cannot_pass_report(tmp_path, monkeypatch):
    output = tmp_path/'report'
    monkeypatch.setattr('sys.argv', ['report', str(tmp_path/'missing_a'), str(tmp_path/'missing_b'),
                                   '--output', str(output)])
    REPORT.main()
    result = json.loads((output/'report.json').read_text())
    assert result['status'] == 'incomplete'
    assert not result['all_checks_pass'] and not result['compatible_independent_runs']
    assert all(not r['complete'] for r in result['runs'])


def test_interrupted_receipt_cannot_pass_report(tmp_path, monkeypatch):
    broken = tmp_path/'broken'
    broken.mkdir()
    for name in ('result.json', 'protocol.json', 'chain.npz', 'local_profiles.npz', 'initial_replay.json'):
        (broken/name).write_text('{')
    output = tmp_path/'report'
    monkeypatch.setattr('sys.argv', ['report', str(broken), '--output', str(output)])
    REPORT.main()
    result = json.loads((output/'report.json').read_text())
    assert not result['all_checks_pass']
    assert result['runs'][0]['error']


def test_combining_individually_mixed_but_disagreeing_replicates_fails():
    rng = np.random.default_rng(111)
    a, b = rng.normal(-3, 1, (4000, 4)), rng.normal(3, 1, (4000, 4))
    assert REPORT.passing(REPORT.profile_summary(a)['diagnostics'])
    assert REPORT.passing(REPORT.profile_summary(b)['diagnostics'])
    assert not REPORT.passing(REPORT.profile_summary(np.concatenate((a, b), axis=1))['diagnostics'])


def test_likelihood_scale_mcse_matches_iid_lognormal_oracle():
    rng = np.random.default_rng(113)
    ratios = .2*rng.normal(size=(8000, 4))-.5*.2**2
    report = REPORT.profile_summary(ratios)
    assert report['valid']
    expected = np.sqrt(np.expm1(.2**2)/ratios.size)
    assert .7*expected < report['relative_mcse'] < 1.3*expected
    assert abs(report['log_mean_ratio']) < 5*report['relative_mcse']


def test_complete_independent_runs_produce_combined_prefix_report(tmp_path, monkeypatch):
    paths = []
    for seed in (301, 302):
        root = tmp_path/str(seed)
        root.mkdir()
        paths.append(str(root))
        rng = np.random.default_rng(seed)
        protocol = dict(seed=seed, steps=1200, warmup=200, observations=[3], chains_per_observation=4,
                        environments=1, environment_update='persistent')
        encoded = json.dumps(protocol).encode()
        (root/'protocol.json').write_bytes(encoded)
        (root/'result.json').write_text(json.dumps(dict(status='completed_diagnostic',
            protocol_sha256=hashlib.sha256(encoded).hexdigest(), all_exploration_checks_pass=True,
            sed_replay=dict(valid_proposals=20, maximum_summary_change=0., maximum_log_kernel_change=0.))))
        (root/'initial_replay.json').write_text(json.dumps(dict(max_abs_error=0., tolerance=1e-6)))
        np.savez_compressed(root/'chain.npz', target_coordinates=rng.normal(size=(1200, 4, 10)),
            summaries=rng.normal(size=(1200, 4, 1, 9)), type_probability=rng.random((1200, 4, 2)),
            neighbour_counts=rng.poisson(20, size=(1200, 4)))
        np.savez_compressed(root/'local_profiles.npz', **{
            f'observation_3_g1_{g:g}': rng.normal(0, .02, (1000, 4)) for g in (.018, .022)})
    output = tmp_path/'report'
    monkeypatch.setattr('sys.argv', ['report', *paths, '--output', str(output), '--prefixes', '700', '1200'])
    REPORT.main()
    result = json.loads((output/'report.json').read_text())
    assert result['status'] == 'completed_diagnostic' and result['compatible_independent_runs']
    assert result['all_checks_pass']
    assert [p['total_steps'] for p in result['prefixes']] == [700, 1200]
