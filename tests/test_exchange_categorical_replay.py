"""Bitwise categorical/proposal/trajectory replay against corrected 55f1220."""
import ast
import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pytest

import sbsi.scene_exchange as candidate
import sbsi.galsbi_particle_target as particle_module
from sbsi.galsbi_scene_prior import GalSBISceneBatch
from test_galsbi_prior import prior_fixture
from test_galsbi_particle_target import setup_target
from test_informed_environment import fixture as informed_fixture, assert_state_equal, assert_payload_equal


BASELINE = Path(__file__).resolve().parents[1]/'doc/evidence/scene_exchange_baseline_55f1220.py'


def load_baseline():
    spec = importlib.util.spec_from_file_location('sbsi._exchange_before_categorical', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_rng_equal(a, b):
    assert pickle.dumps(a.bit_generator.state) == pickle.dumps(b.bit_generator.state)
    np.testing.assert_array_equal(a.random(20), b.random(20))


def assert_scene_equal(a, b):
    for name in ('targets', 'neighbours'):
        assert getattr(a, name).keys() == getattr(b, name).keys()
        for key in getattr(a, name):
            left, right = getattr(a, name)[key], getattr(b, name)[key]
            assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
    for name in ('indptr', 'offset_arcsec'):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()


def make_scene(counts, *, support_zero=False):
    counts = np.asarray(counts, int)
    prior = prior_fixture()
    native, _ = prior.sample(256, 18271)
    keys = [k for k in native if k != 'proposal_index']
    n = len(native['z'])
    targets = {k: native[k][np.arange(len(counts)) % n].copy() for k in keys}
    neighbours = {k: native[k][(np.arange(counts.sum())+17) % n].copy() for k in keys}
    if support_zero:
        targets['z'][::3] = 4.
        neighbours['M_B'][::5] = 0.
    rng = np.random.default_rng(181)
    offsets = rng.normal(size=(counts.sum(), 2))
    offsets *= (9.9*rng.random(counts.sum())/np.maximum(np.linalg.norm(offsets, axis=1), 1e-30))[:, None]
    return prior, GalSBISceneBatch(targets, neighbours, np.r_[0, counts.cumsum()], offsets)


@pytest.mark.parametrize('bitgen', [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('categories', [1, 2, 3, 17, 128])
def test_batched_cdf_choices_and_rng_match_scalar_numpy(bitgen, dtype, categories):
    raw = np.random.default_rng(120).lognormal(0, 6, size=(513, categories))
    if categories > 1:
        raw[::4, 0] = 0
        raw[::7, -1] = 0
        raw[0] = 0
        raw[0, categories//2] = 1
        raw[np.flatnonzero(raw.sum(1) == 0), categories//2] = 1
    probability = (raw/raw.sum(1)[:, None]).astype(dtype)
    a, b = np.random.Generator(bitgen(8123)), np.random.Generator(bitgen(8123))
    expected = np.array([a.choice(categories, p=p) for p in probability])
    actual = candidate._draw_type_categories(probability, b)
    np.testing.assert_array_equal(expected, actual)
    assert_rng_equal(a, b)


def test_right_sided_boundary_and_empty_rows_keep_rng():
    a, b = np.random.default_rng(891), np.random.default_rng(891)
    u = np.random.default_rng(891).random()
    p = np.array([[0., u, 0., 1-u, 0.]])
    assert a.choice(5, p=p[0]) == 3
    assert candidate._draw_type_categories(p, b)[0] == 3
    assert_rng_equal(a, b)
    a, b = np.random.default_rng(90), np.random.default_rng(90)
    assert len(candidate._draw_type_categories(np.empty((0, 2)), a)) == 0
    assert_rng_equal(a, b)


@pytest.mark.parametrize('counts', [[0]*7, [1]*7, [0, 1, 2, 7, 0, 33, 129]])
@pytest.mark.parametrize('relocation', [False, True])
@pytest.mark.parametrize('support_zero', [False, True])
def test_full_ragged_proposed_payload_and_dynamic_probabilities_replay(counts, relocation, support_zero):
    baseline = load_baseline()
    prior, original = make_scene(counts, support_zero=support_zero)
    old, new = original, original
    a, b = np.random.default_rng(7291), np.random.default_rng(7291)
    options = dict(radius_scale=.03)
    if relocation:
        options.update(relocation_scale=.1, aperture_arcsec=10., central_weight=.9)
    for step in range(5):
        probability = np.random.default_rng(72+step).dirichlet([.3, .7], size=len(counts))
        probability[::3] = [1., 0.] if step % 2 else [0., 1.]
        old, oldratio, oldactive = baseline.propose_native_exchange(prior, old, probability, a, **options)
        new, newratio, newactive = candidate.propose_native_exchange(prior, new, probability, b, **options)
        assert_scene_equal(old, new)
        assert oldratio.tobytes() == newratio.tobytes()
        np.testing.assert_array_equal(oldactive, newactive)
        assert_rng_equal(a, b)


@pytest.mark.parametrize('informed', [False, True])
def test_complete_mh_trajectories_acceptance_payload_and_rng(monkeypatch, informed):
    baseline = load_baseline()
    if informed:
        target, initial, geometry = informed_fixture(4)
        target.informed_environment = geometry
        initial = target.evaluate(initial.x, initial.scene)
    else:
        target, initial = setup_target(4)
    a, b = np.random.default_rng(2026103101), np.random.default_rng(2026103101)
    left = right = initial
    blocks = ('exchange', 'physical', 'exchange_relocate', 'positions', 'nuisance', 'sed', 'environment', 'independence')
    for step in range(32):
        beta = (.001, .4, 1.)[step % 3]
        block = blocks[step % len(blocks)]
        monkeypatch.setattr(particle_module, 'propose_native_exchange', baseline.propose_native_exchange)
        left, accepted_a, active_a = target.mutate(left, beta, a, block=block)
        monkeypatch.setattr(particle_module, 'propose_native_exchange', candidate.propose_native_exchange)
        right, accepted_b, active_b = target.mutate(right, beta, b, block=block)
        assert_state_equal(left, right)
        for name in ('x','log_prior','log_kernel','log_proposal','type_probability','summaries'):
            assert getattr(left,name).tobytes() == getattr(right,name).tobytes()
        if informed:
            assert_payload_equal(left.environment_payload, right.environment_payload)
            for name in vars(left.environment_payload):
                assert getattr(left.environment_payload,name).tobytes() == getattr(right.environment_payload,name).tobytes()
        assert_scene_equal(left.scene, right.scene)
        np.testing.assert_array_equal(accepted_a, accepted_b)
        np.testing.assert_array_equal(active_a, active_b)
        assert accepted_a.tobytes() == accepted_b.tobytes() and active_a.tobytes() == active_b.tobytes()
        assert_rng_equal(a, b)


def test_support_correction_unchanged_including_unconditional_clamp_fix():
    def node(path):
        tree = ast.parse(Path(path).read_text())
        return next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'numerical_over_analytic')
    assert ast.dump(node(BASELINE)) == ast.dump(node(candidate.__file__))
    assert 'below_uniform_clamp' in ast.unparse(node(candidate.__file__))
