"""Frozen proposal geometry tests; no posterior-convergence assertions."""

import json
import hashlib
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from sbsi.particle_geometry import (
    BROAD_WIDTHS, EIGENVALUE_FLOOR, ORIGINAL_WIDTHS, PHYSICAL_COLUMNS,
    PILOT_SCALE, FrozenParticleGeometry, fit_particle_geometry, phase_residual,
)


def metadata(objects=2):
    return {"observations": [58165, 8][:objects], "pilot_seeds": [11, 12]}


def array_from_physical(physical):
    result = np.zeros((*physical.shape[:-1], 10))
    result[..., PHYSICAL_COLUMNS] = physical
    return result


def toy_geometry(multiscale=False):
    rng = np.random.default_rng(2026102160)
    runs = [array_from_physical(rng.normal(size=(2, 1024, 5)) * ORIGINAL_WIDTHS)
            for _ in range(2)]
    return fit_particle_geometry(runs, metadata(), multiscale=multiscale)


def test_covariance_pools_within_run_and_keeps_objects_separate():
    rng = np.random.default_rng(2026102161)
    base = rng.normal(size=(2, 2048, 5)) * ORIGINAL_WIDTHS
    base[1] *= 2
    shifted = base.copy()
    shifted[..., :4] += np.array([40, -20, 30, 50])
    runs = [array_from_physical(base), array_from_physical(shifted)]
    saved = [x.copy() for x in runs]
    geometry = fit_particle_geometry(runs, metadata())
    expected = np.stack([np.cov(x, rowvar=False) for x in base]) * PILOT_SCALE
    np.testing.assert_allclose(geometry.covariance(1), expected, rtol=1e-10, atol=1e-12)
    assert geometry.covariance(1)[1, 0, 0] > 3 * geometry.covariance(1)[0, 0, 0]
    for before, after in zip(saved, runs):
        np.testing.assert_array_equal(before, after)


def test_covariance_equal_run_weight_even_when_particle_counts_differ():
    rng = np.random.default_rng(2026102162)
    a = rng.normal(size=(1, 20, 5)) * ORIGINAL_WIDTHS
    b = 2 * rng.normal(size=(1, 200, 5)) * ORIGINAL_WIDTHS
    geometry = fit_particle_geometry([array_from_physical(a), array_from_physical(b)], metadata(1))
    expected = PILOT_SCALE * (np.cov(a[0], rowvar=False) + np.cov(b[0], rowvar=False)) / 2
    np.testing.assert_allclose(geometry.covariance(1)[0], expected, rtol=1e-10, atol=1e-12)


def test_phase_wrap_preserves_covariance_and_cross_correlations():
    rng = np.random.default_rng(2026102163)
    x = rng.normal(size=(1, 2000, 5)) * ORIGINAL_WIDTHS
    x[..., -1] = 0.5 * x[..., 0] + 0.01 * rng.normal(size=(1, 2000))
    wrapped = x.copy()
    wrapped[..., -1] = (wrapped[..., -1] + 2 * np.pi - 0.003) % (2 * np.pi)
    direct = fit_particle_geometry([array_from_physical(x)], metadata(1))
    crossing = fit_particle_geometry([array_from_physical(wrapped)], metadata(1))
    np.testing.assert_allclose(direct.covariance(1), crossing.covariance(1), rtol=1e-10, atol=1e-12)
    assert crossing.covariance(1)[0, 0, -1] > 0
    np.testing.assert_allclose(phase_residual([2 * np.pi - 0.1, 0.1], 0), [-0.1, 0.1])


def test_regularization_and_temperature_endpoints():
    geometry = fit_particle_geometry([np.zeros((2, 20, 10))], metadata())
    expected = PILOT_SCALE * EIGENVALUE_FLOOR * np.diag(ORIGINAL_WIDTHS**2)
    np.testing.assert_allclose(geometry.covariance(1), np.tile(expected, (2, 1, 1)))
    np.testing.assert_allclose(geometry.covariance(0), np.tile(np.diag(BROAD_WIDTHS**2), (2, 1, 1)))
    for beta in (0.001, 0.3, 0.9):
        wanted = np.linalg.inv(beta * np.linalg.inv(expected) + (1-beta) * np.diag(1 / BROAD_WIDTHS**2))
        np.testing.assert_allclose(geometry.covariance(beta), np.tile(wanted, (2, 1, 1)))


def test_increment_distribution_per_object_and_fixed_multiscale():
    geometry = toy_geometry(multiscale=True)
    rng = np.random.default_rng(2026102164)
    owner = np.repeat(np.arange(2), 40000)
    draws = geometry.increments(owner, 0.6, rng)
    mixture_second = 0.2 * 0.25**2 + 0.6 + 0.2 * 4**2
    mixture_fourth = 0.2 * 0.25**4 + 0.6 + 0.2 * 4**4
    for i in range(2):
        x = draws[owner == i]
        covariance = geometry.covariance(0.6)[i] * mixture_second
        # This scale mixture has kurtosis >10, not Gaussian kurtosis3. Use the
        # exact variance of the unbiased sample variance, including its fourth
        # moment, instead of an arbitrary percentage or Gaussian approximation.
        variance = np.diag(covariance)
        fourth_moment = 3 * np.diag(geometry.covariance(0.6)[i])**2 * mixture_fourth
        variance_mcse = np.sqrt((fourth_moment - (len(x)-3)/(len(x)-1)*variance**2) / len(x))
        assert np.all(np.abs(np.diag(np.cov(x, rowvar=False)) - variance) < 5 * variance_mcse)
        assert np.all(np.abs(x.mean(0)) < 5 * np.sqrt(np.diag(covariance) / len(x)))


def test_wrapped_correlated_gaussian_mixture_is_symmetric():
    geometry = toy_geometry(multiscale=True)
    covariance = geometry.covariance(0.4)[0]
    precision = np.linalg.inv(covariance)

    def wrapped_density(delta):
        total = 0.0
        for factor, probability in zip([0.25, 1, 4], [0.2, 0.6, 0.2]):
            for winding in range(-12, 13):
                value = delta.copy()
                value[-1] += 2 * np.pi * winding
                total += probability * np.exp(-0.5 * value @ precision @ value / factor**2) / factor**5
        return total

    for delta in (np.array([0.01, -0.02, 0.07, -0.04, 0.03]),
                  np.array([0.01, 0.02, -0.03, 0.04, 2*np.pi-0.01])):
        assert wrapped_density(delta) == pytest.approx(wrapped_density(-delta), rel=1e-12)


def test_artifact_roundtrip_seed_guard_and_readonly_metadata(tmp_path):
    geometry = toy_geometry(multiscale=True)
    path = tmp_path / "geometry.json"
    geometry.save(path)
    loaded = FrozenParticleGeometry.load(path)
    owner = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(geometry.increments(owner, 0.4, np.random.default_rng(1)),
                                  loaded.increments(owner, 0.4, np.random.default_rng(1)))
    with pytest.raises(FileExistsError):
        geometry.save(path)
    data = loaded.metadata
    data["pilot_seeds"][0] = 99
    assert loaded.metadata["pilot_seeds"] == [11, 12]
    loaded.validate_production(99, [58165, 8])
    with pytest.raises(ValueError, match="pilot seed"):
        loaded.validate_production(11, [58165, 8])
    with pytest.raises(ValueError, match="order"):
        loaded.validate_production(99, [8, 58165])
    malformed = json.loads(path.read_text())
    malformed["physical_columns"][0] = 0
    path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="schema"):
        FrozenParticleGeometry.load(path)


@pytest.mark.parametrize("beta", [-0.1, 1.1, np.nan, np.inf, [0.5]])
def test_bad_temperature_is_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        toy_geometry().increments(np.array([0]), beta, np.random.default_rng(1))


@pytest.mark.parametrize("owner", [[-1], [2], [0.5], [[0]]])
def test_bad_owner_is_rejected(owner):
    with pytest.raises(ValueError, match="owner"):
        toy_geometry().increments(owner, 0.5, np.random.default_rng(1))


def test_bad_pilots_and_covariances_are_rejected():
    for runs in ([], [np.zeros((2, 1, 10))], [np.zeros((2, 3, 9))],
                 [np.full((2, 3, 10), np.nan)]):
        with pytest.raises(ValueError):
            fit_particle_geometry(runs, metadata())
    with pytest.raises(ValueError, match="positive definite"):
        FrozenParticleGeometry(np.zeros((2, 5, 5)), np.zeros(2), metadata())
    with pytest.raises(ValueError, match="pilot seeds"):
        FrozenParticleGeometry(np.tile(np.eye(5), (2, 1, 1)), np.zeros(2),
                               {"observations": [58165, 8], "pilot_seeds": [11, 11]})


def test_fit_cli_checks_receipts_and_ignores_scores(tmp_path, monkeypatch):
    runs, source_hashes = [], []
    for i in range(2):
        path = tmp_path / f"run{i}"
        path.mkdir()
        np.savez(path / "particles.npz", coordinates=np.zeros((2, 32, 10)),
                 scores=np.full((2, 32, 2), np.nan))
        protocol = {"observations": [58165, 8], "particles": 32, "seed": 11+i,
                    "fixed_shear": [0, 0], "reference_protocol_sha256": "a" * 64,
                    "observation_sha256": "b" * 64}
        (path / "protocol.json").write_text(json.dumps(protocol))
        particle_hash = hashlib.sha256((path / "particles.npz").read_bytes()).hexdigest()
        result = {"status": "completed", "observations": [58165, 8],
                  "particles_sha256": particle_hash,
                  "protocol_sha256": hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest()}
        (path / "result.json").write_text(json.dumps(result))
        source_hashes.append(particle_hash)
        runs.append(path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "fit_particle_geometry.py"
    output = tmp_path / "frozen.json"
    args = [str(script), "--runs", *map(str, runs), "--output", str(output), "--multiscale"]
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(str(script), run_name="__main__")
    geometry = FrozenParticleGeometry.load(output)
    assert geometry.metadata["pilot_seeds"] == [11, 12]
    assert geometry.metadata["uses_scores_or_posterior_weights"] is False
    assert [x["particles_sha256"] for x in geometry.metadata["pilot_runs"]] == source_hashes
    assert [hashlib.sha256((p / "particles.npz").read_bytes()).hexdigest() for p in runs] == source_hashes
    (runs[0] / "protocol.json").write_text("{}")
    args[args.index("--output") + 1] = str(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="receipt"):
        runpy.run_path(str(script), run_name="__main__")
