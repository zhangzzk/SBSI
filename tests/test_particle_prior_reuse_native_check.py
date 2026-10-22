"""Small metadata/byte-comparison tests; never load real native data/models."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1]/'scripts/check_particle_prior_reuse_native.py'
    spec = importlib.util.spec_from_file_location('native_reuse_checker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_detects_signed_zero_nonfinite_payload_dtype_and_shape(checker):
    baseline = np.array([0., 1., -np.inf, np.inf, np.nan])
    assert checker.compare_array(baseline.copy(), baseline)['matches']
    changed = baseline.copy()
    changed[0] = -0.
    report = checker.compare_array(changed, baseline)
    assert not report['matches'] and report['mismatched_elements'] == 1
    assert report['examples'][0]['index'] == [0]
    assert report['baseline']['finite'] == 2
    assert report['baseline']['negative_infinite'] == 1
    changed = baseline.copy()
    changed.view(np.uint64)[-1] ^= np.uint64(1)
    assert np.isnan(changed[-1])
    report = checker.compare_array(changed, baseline)
    assert not report['matches'] and report['mismatched_elements'] == 1
    assert not checker.compare_array(baseline.astype(np.float32), baseline)['matches']
    assert not checker.compare_array(baseline[None, :], baseline)['matches']
    assert checker.compare_array(np.empty((0, 2)), np.empty((0, 2)))['matches']
    json.dumps(report, allow_nan=False)


def test_native_comparison_keeps_individual_field_failures(checker):
    original = (np.array([-np.inf]), {'z': np.array([.5])}, np.zeros((1, 2)))
    changed = (np.array([-np.inf]), {'z': np.array([.6]), 'extra': np.array([1.])}, np.zeros((1, 2)))
    report = checker.compare_native(changed, original)
    assert not report['matches']
    assert report['fields']['log_prior']['matches']
    assert not report['fields']['native_schema']['matches']
    assert not report['fields']['native/z']['matches']


def fake_panel(checker, root, monkeypatch):
    sources = {}
    for arm, particles, seed in checker.ARMS:
        for replica in range(8):
            folder = root/f'dev_v10_{arm}_{replica}'
            folder.mkdir(parents=True)
            (folder/'particles.npz').write_bytes(b'only metadata, not opened as arrays')
            protocol = dict(observations=checker.OBSERVATIONS, particles=particles,
                            seed=seed+replica, fixed_shear=[0., 0.])
            (folder/'protocol.json').write_text(json.dumps(protocol))
            receipt = dict(status='completed', particles_sha256=checker.digest(folder/'particles.npz'),
                           protocol_sha256=checker.digest(folder/'protocol.json'))
            (folder/'result.json').write_text(json.dumps(receipt))
            for path in folder.iterdir():
                sources[str(path)] = checker.digest(path)
    audit = root/'v10_corrected_comparison_20261022_v3.json'
    audit.write_text(json.dumps(dict(schema='sbsi-n23-corrected-comparison-v1',
                                    development_checks_pass=True, sources=sources)))
    monkeypatch.setattr(checker, 'AUDIT_SHA256', checker.digest(audit))
    return audit


def test_complete_panel_identity_and_hashes_are_mandatory(checker, tmp_path, monkeypatch):
    root = tmp_path/'development'
    fake_panel(checker, root, monkeypatch)
    bound = {}
    populations = checker.panel_inputs(root, bound)
    assert len(populations) == 24 and len(bound) == 73
    assert sum(4*row['particles'] for row in populations) == 73728
    changed = root/'dev_v10_large_7/particles.npz'
    changed.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed or unbound'):
        checker.panel_inputs(root, {})
    with pytest.raises(ValueError, match='no held-out'):
        checker.panel_inputs(tmp_path/'heldout_data', {})


def test_rebound_wrong_seed_is_still_rejected(checker, tmp_path, monkeypatch):
    root = tmp_path/'development'
    audit_path = fake_panel(checker, root, monkeypatch)
    folder = root/'dev_v10_small_0'
    protocol_path = folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['seed'] += 1
    protocol_path.write_text(json.dumps(protocol))
    receipt_path = folder/'result.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['protocol_sha256'] = checker.digest(protocol_path)
    receipt_path.write_text(json.dumps(receipt))
    audit = json.loads(audit_path.read_text())
    for path in (protocol_path, receipt_path):
        audit['sources'][str(path)] = checker.digest(path)
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(checker, 'AUDIT_SHA256', checker.digest(audit_path))
    with pytest.raises(ValueError, match='population identity'):
        checker.panel_inputs(root, {})


def test_fixed_support_probes_cover_both_type_edges_without_randomness(checker):
    from types import SimpleNamespace
    prior = SimpleNamespace(types=['red', 'blue'], z_max={'red': 1., 'blue': 3.},
                            p={'gals_mag_min': 14., 'gals_mag_max': 28.})
    seed_row = np.zeros(10)
    probes, labels = checker.support_probes(seed_row, prior)
    assert probes.shape == (15, 10) and len(set(labels)) == 15
    assert np.isfinite(probes).all()
    np.testing.assert_array_equal(probes[0], seed_row)
    for kind in prior.types:
        positions = [labels.index(f'log_redshift_{kind}_{side}') for side in ('below', 'at', 'above')]
        assert np.all(np.diff(probes[positions, 0]) > 0)
    assert probes[labels.index('intrinsic_bright'), 1] < prior.p['gals_mag_min']
    assert probes[labels.index('intrinsic_faint'), 1] > prior.p['gals_mag_max']
