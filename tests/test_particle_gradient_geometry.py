"""Independent physical chain-rule and frozen-gradient-metric checks."""

import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import scene_model_inputs
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.particle_geometry import BROAD_WIDTHS, PHYSICAL_COLUMNS, PILOT_SCALE, FrozenParticleGeometry
from sbsi.particle_gradient_geometry import fit_gradient_geometry, physical_likelihood_gradients
from test_galsbi_prior import prior_fixture


def forward_fixture():
    prior = prior_fixture()
    distance = lambda z: 1000*z/(1+z)
    coordinates = GalSBITargetCoordinates(prior, distance)
    scenes_prior = GalSBIScenePrior(prior, aperture_arcsec=10)
    x = np.array([
        [-0.7, 20, 0.1, -0.2, 0.3, -0.4, -0.5, -0.8, 0.3, 2*np.pi-1e-8],
        [-0.4, 24, 0.2, -0.3, 0.4, -0.5, 0.2, 0.7, -0.9, 1e-8],
        [-0.9, 27, 0.3, -0.4, 0.5, -0.6, 0.8, 1.1, 0.4, 1.8],
    ], dtype=float)
    neighbours = dict(z=np.full(5, 0.3), m_r=np.array([21, 18, 25, 26, 17]))
    indptr = np.array([0, 3, 3, 5])
    offsets = np.array([[0.2, 0.1], [1.2, 0.3], [12, 0], [0.3, 0.4], [2, 0]])

    def forward(values):
        target, _ = coordinates.decode(values, np.zeros(len(values), dtype=int))
        scene = GalSBISceneBatch(target, neighbours, indptr, offsets)
        context, _, _ = scene_model_inputs(scene, scenes_prior, distance, {},
                                           response_radius_arcsec=10, response_neighbours=20,
                                           response=False)
        # Response may vary with target parameters; at g=0 its product b*g
        # vanishes. A nonconstant placeholder makes that exclusion explicit.
        return np.column_stack((context.to_numpy(), np.sin(values[:, 1])))

    parameters = {key: prior.p[key] for key in ("sersic_n_min", "sersic_n_max", "ell_beta_emax")}
    return x, forward, parameters


def test_physical_chain_rule_matches_native_forward_finite_differences():
    x, forward, parameters = forward_fixture()
    summary = forward(x)
    rng = np.random.default_rng(2026102180)
    a = rng.normal(size=(8, 8))
    a = (a+a.T)/2
    b = rng.normal(size=8)

    def log_kernel(values):
        v = forward(values)[:, :8]
        return 0.5*np.einsum("ni,ij,nj->n", v, a, v) + v @ b

    raw_gradient = summary[:, :8] @ a + b
    actual = physical_likelihood_gradients(x, summary, raw_gradient, **parameters)
    finite = []
    for column in PHYSICAL_COLUMNS:
        step = np.zeros_like(x)
        step[:, column] = 2e-5
        finite.append((log_kernel(x+step)-log_kernel(x-step))/4e-5)
    np.testing.assert_allclose(actual, np.array(finite).T, atol=2e-7, rtol=2e-7)


def test_crowding_magnitude_derivative_includes_all_channels_and_empty_scene():
    x, forward, parameters = forward_fixture()
    summary = forward(x)
    gradient = np.zeros((len(x), 8))
    gradient[:, 5:8] = [1.0, -2.0, 0.7]
    actual = physical_likelihood_gradients(x, summary, gradient, **parameters)
    step = np.zeros_like(x)
    step[:, 1] = 1e-5
    finite = ((forward(x+step)[:, 5:8]-forward(x-step)[:, 5:8]) @ [1, -2, 0.7])/2e-5
    np.testing.assert_allclose(actual[:, 0], finite, atol=1e-9, rtol=1e-8)
    np.testing.assert_array_equal(actual[:, 1:], 0)
    assert actual[1, 0] == 0
    assert np.any(np.abs(actual[:, 0]) > 0.01)


@pytest.mark.parametrize("distortion", [0.0, 0.999e-7, 1.001e-7, 0.3])
def test_renderer_circle_threshold_and_phase_chain_rule(distortion):
    x, forward, parameters = forward_fixture()
    fraction = distortion / parameters["ell_beta_emax"]
    x[:, 8] = -1000 if distortion == 0 else np.log(fraction)-np.log1p(-fraction)
    summary = forward(x)
    gradient = np.zeros((len(x), 8))
    gradient[:, :2] = [1.3, -0.7]
    actual = physical_likelihood_gradients(x, summary, gradient, **parameters)
    for index, column in [(3, 8), (4, 9)]:
        step = np.zeros_like(x)
        step[:, column] = 1e-6
        finite = ((forward(x+step)[:, :2]-forward(x-step)[:, :2]) @ [1.3, -0.7])/2e-6
        np.testing.assert_allclose(actual[:, index], finite, atol=1e-10, rtol=2e-7)
    if distortion <= 1e-7:
        np.testing.assert_array_equal(actual[:, 3:], 0)


def test_nonzero_shear_stale_summary_negative_crowding_are_rejected():
    x, forward, parameters = forward_fixture()
    summary = forward(x)
    gradient = np.ones((len(x), 8))
    with pytest.raises(ValueError, match="zero shear"):
        physical_likelihood_gradients(x, summary, gradient, shear=[0.02, 0], **parameters)
    stale = summary.copy()
    stale[0, 4] += 0.1
    with pytest.raises(ValueError, match="replay"):
        physical_likelihood_gradients(x, stale, gradient, **parameters)
    stale = summary.copy()
    stale[0, 5] = -0.1
    with pytest.raises(ValueError, match="crowding"):
        physical_likelihood_gradients(x, stale, gradient, **parameters)
    with pytest.raises(ValueError, match="aligned"):
        physical_likelihood_gradients(x[:, :9], summary, gradient, **parameters)
    with pytest.raises(ValueError, match="bounds"):
        physical_likelihood_gradients(x, summary, gradient,
                                      **{**parameters, "ell_beta_emax": 1.0})


def metadata():
    return {"observations": [58165, 8], "pilot_seeds": [13, 14]}


def test_uncentred_metric_matches_rank_one_sherman_morrison_result():
    gradient = np.array([2.0, -3.0, 4.0, -1.0, 0.5])
    runs = [np.broadcast_to(gradient, (2, 7, 5)), np.broadcast_to(gradient, (2, 13, 5))]
    geometry = fit_gradient_geometry(runs, metadata())
    broad = np.diag(BROAD_WIDTHS**2)
    weighted = broad @ gradient
    exact = PILOT_SCALE * (broad - np.outer(weighted, weighted)/(1+gradient @ weighted))
    np.testing.assert_allclose(geometry.covariance(1), np.tile(exact, (2, 1, 1)), atol=1e-14, rtol=1e-12)
    # Constant nonzero gradients have zero covariance but nonzero second moment.
    assert not np.allclose(exact, PILOT_SCALE*broad)
    assert np.all(np.linalg.eigvalsh(geometry.covariance(1)) > 0)


def test_zero_gradients_give_broad_regularized_covariance():
    geometry = fit_gradient_geometry([np.zeros((2, 4, 5))], metadata())
    np.testing.assert_allclose(geometry.covariance(1),
                              np.tile(PILOT_SCALE*np.diag(BROAD_WIDTHS**2), (2, 1, 1)))


def test_equal_run_weight_and_particle_object_permutation_equivariance():
    rng = np.random.default_rng(2026102181)
    runs = [rng.normal(size=(2, 17, 5)), 3*rng.normal(size=(2, 61, 5))]
    saved = [x.copy() for x in runs]
    geometry = fit_gradient_geometry(runs, metadata())
    metric = (np.einsum("oni,onj->oij", runs[0], runs[0])/17
              + np.einsum("oni,onj->oij", runs[1], runs[1])/61)/2
    metric += np.diag(1/BROAD_WIDTHS**2)
    np.testing.assert_allclose(geometry.covariance(1), PILOT_SCALE*np.linalg.inv(metric), atol=1e-14, rtol=1e-12)
    permuted = [run[::-1, rng.permutation(run.shape[1])] for run in runs[::-1]]
    alternate = fit_gradient_geometry(permuted, {"observations": [8, 58165], "pilot_seeds": [14, 13]})
    np.testing.assert_allclose(alternate.covariance(1), geometry.covariance(1)[::-1], atol=1e-14, rtol=1e-12)
    for original, after in zip(saved, runs):
        np.testing.assert_array_equal(original, after)


def test_metric_artifact_metadata_and_production_seed_contract(tmp_path):
    geometry = fit_gradient_geometry([np.ones((2, 6, 5))], metadata(), phase_centres=[0.2, 6.1])
    path = tmp_path / "gradient_geometry.json"
    geometry.save(path)
    loaded = FrozenParticleGeometry.load(path)
    np.testing.assert_array_equal(loaded.covariance(0.7), geometry.covariance(0.7))
    assert loaded.metadata["native_prior_in_metric"] is False
    assert loaded.metadata["likelihood_gradient_shear"] == [0, 0]
    assert loaded.metadata["pilot_seeds"] == [13, 14]
    with pytest.raises(ValueError, match="pilot seed"):
        loaded.validate_production(13, [58165, 8])
    loaded.validate_production(15, [58165, 8])


@pytest.mark.parametrize("runs", [[], [np.zeros((2, 0, 5))], [np.zeros((2, 3, 4))],
                                   [np.full((2, 3, 5), np.nan)],
                                   [np.zeros((2, 3, 5)), np.zeros((3, 3, 5))]])
def test_metric_rejects_bad_gradient_inputs(runs):
    with pytest.raises(ValueError):
        fit_gradient_geometry(runs, metadata())


def test_pilot_reader_checks_hashes_shear_and_required_summary_fields(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "fit_particle_gradient_geometry.py"
    loader = runpy.run_path(str(script))["load_pilots"]
    paths = []
    for index in range(2):
        path = tmp_path / str(index)
        path.mkdir()
        np.savez(path / "particles.npz", coordinates=np.zeros((2, 4, 10)), summaries=np.zeros((2, 4, 9)),
                 scores=np.full((2, 4, 2), np.nan))
        protocol = dict(observations=[58165, 8], seed=13+index, particles=4, fixed_shear=[0, 0],
                        reference="/fixed/reference", reference_protocol_sha256="a"*64, observation_sha256="b"*64)
        (path / "protocol.json").write_text(json.dumps(protocol))
        result = dict(status="completed", observations=[58165, 8],
                      particles_sha256=hashlib.sha256((path / "particles.npz").read_bytes()).hexdigest(),
                      protocol_sha256=hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest())
        (path / "result.json").write_text(json.dumps(result))
        paths.append(path)
    arrays, provenance = loader(paths)
    assert len(arrays) == 2 and provenance["pilot_seeds"] == [13, 14]
    assert provenance["pilot_runs"][0]["path"] == str(paths[0].resolve())
    protocol = json.loads((paths[0] / "protocol.json").read_text())
    protocol["fixed_shear"] = [0.02, 0]
    (paths[0] / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="receipt"):
        loader(paths)
    result = json.loads((paths[0] / "result.json").read_text())
    result["protocol_sha256"] = hashlib.sha256((paths[0] / "protocol.json").read_bytes()).hexdigest()
    (paths[0] / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="zero shear"):
        loader(paths)
