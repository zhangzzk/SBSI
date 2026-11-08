from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm

from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.catalogue_closure import MockCatalogue, generate_mock_catalogue, run_exact_closure
from sbsi.catalogue_likelihood import (
    CatalogueLikelihood,
    CatalogueModelCache,
    CatalogueSelection,
)
from sbsi.scene_prior import ScenePrior
from sbsi.catalogue_likelihood import OutputCut
from sbsi.selection_normalization import (
    ExactPopulationNormalization,
    QuadraticPopulationNormalization,
    detected_selected_mass_shard,
    load_population_normalization,
)


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


def _quadratic_normalization(**updates):
    values = {
        "center": np.array([0.01, -0.02]),
        "log_mass_at_center": -0.4,
        "gradient": np.array([0.3, -0.2]),
        "hessian": np.array([[2.0, 0.5], [0.5, -1.0]]),
        "trust_min": np.array([-0.02, -0.04]),
        "trust_max": np.array([0.04, 0.02]),
        "finite_difference_step": 0.001,
        "identity": {"scene": "test"},
        "validation": {"maximum_absolute_gradient_error": 1e-4},
        "source": {"sha256": "test"},
    }
    values.update(updates)
    return QuadraticPopulationNormalization(**values)


def test_quadratic_population_normalization_evaluates_and_refuses_extrapolation():
    model = _quadratic_normalization()
    point = np.array([0.013, -0.016])
    offset = point - model.center
    expected = -0.4 + model.gradient @ offset + 0.5 * offset @ model.hessian @ offset
    assert model.log_mass(*point) == pytest.approx(expected)
    with pytest.raises(ValueError, match="refuses extrapolation"):
        model.log_mass(0.041, 0.0)


def test_exact_population_normalization_round_trip_and_missing_point(tmp_path):
    model = ExactPopulationNormalization(
        points={(0.01, -0.02): 0.4, (0.011, -0.02): 0.41},
        finite_difference_step=0.001,
        identity={"scene": "test"},
        source={"n_shards": 2},
    )
    path = tmp_path / "exact.json"
    model.save(path)
    restored = load_population_normalization(path)
    assert restored.log_mass(0.01, -0.02) == pytest.approx(np.log(0.4))
    assert restored.available_shears == ((0.01, -0.02), (0.011, -0.02))
    with pytest.raises(ValueError, match="has no value"):
        restored.log_mass(0.0, 0.0)




class GaussianShapeFlow:
    def __init__(self, sigma=0.12):
        self.sigma = float(sigma)
        self.condition_preprocessor = SimpleNamespace(feature_names=["e1_input_p"])
        self.target_transform = SimpleNamespace(target_names=["measured_e1"])

    def log_prob(self, frame):
        residual = frame["measured_e1"].to_numpy(float) - frame["e1_input_p"].to_numpy(float)
        return -0.5 * (residual / self.sigma) ** 2 - np.log(self.sigma * np.sqrt(2.0 * np.pi))

    def sample(self, condition_frame, n_samples=1, batch_size=None, qmc=False):
        mean = torch.as_tensor(condition_frame["e1_input_p"].to_numpy(float), dtype=torch.float32)
        noise = torch.randn(len(condition_frame), n_samples) * self.sigma
        return (mean[:, None] + noise).numpy()[..., None]


class RandomShiftShapeFlow(GaussianShapeFlow):
    """Small flow with the production QMC stream shape for shard tests."""

    device = torch.device("cpu")

    def sample(self, condition_frame, n_samples=1, batch_size=None, qmc=False):
        mean = torch.as_tensor(condition_frame["e1_input_p"].to_numpy(float))
        if not qmc:
            return super().sample(
                condition_frame, n_samples=n_samples, batch_size=batch_size, qmc=qmc
            )
        shift = torch.rand(len(condition_frame), 1, 1).squeeze(-1)
        noise = (shift - 0.5).expand(len(condition_frame), n_samples) * self.sigma
        return (mean[:, None] + noise).numpy()[..., None]


class GaussianTwoShapeFlow:
    def __init__(self, sigma=0.12, deterministic_samples=False):
        self.sigma = float(sigma)
        self.deterministic_samples = bool(deterministic_samples)
        self.condition_preprocessor = SimpleNamespace(feature_names=["e1_input_p", "e2_input_p"])
        self.target_transform = SimpleNamespace(target_names=["measured_ngmix_g1", "measured_ngmix_g2"])

    def log_prob(self, frame):
        residual = frame[["measured_ngmix_g1", "measured_ngmix_g2"]].to_numpy(float) - frame[
            ["e1_input_p", "e2_input_p"]
        ].to_numpy(float)
        return -0.5 * np.square(residual / self.sigma).sum(axis=1) - 2.0 * np.log(
            self.sigma * np.sqrt(2.0 * np.pi)
        )

    def sample(self, condition_frame, n_samples=1, batch_size=None, qmc=False):
        mean = torch.as_tensor(
            condition_frame[["e1_input_p", "e2_input_p"]].to_numpy(float),
            dtype=torch.float32,
        )
        if self.deterministic_samples:
            return np.broadcast_to(
                mean.numpy()[:, None, :],
                (len(condition_frame), n_samples, 2),
            ).copy()
        noise = torch.randn(len(condition_frame), n_samples, 2) * self.sigma
        return (mean[:, None, :] + noise).numpy()


def _catalogue():
    arcsec = 1.0 / 3600.0
    return pd.DataFrame(
        {
            "RA": [10.0, 10.0 + arcsec, 10.0 + 2 * arcsec, 10.0 + 3 * arcsec],
            "DEC": [0.0, 0.0, 0.0, 0.0],
            "redshift": [0.4, 0.5, 0.6, 0.7],
            "r": [22.0, 23.0, 24.0, 25.0],
            "Re": [0.7, 0.6, 0.8, 0.5],
            "sersic_n": [1.0, 2.0, 3.0, 1.5],
            "axis_ratio": [0.65, 0.75, 0.85, 0.7],
            "position_angle": [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            "prior_weight": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _detector(frame):
    # Nonconstant on purpose: the hand calculation below checks that P_det is
    # inside each atom and that the catalogue likelihood divides by B(g).
    return 1.0 / (1.0 + np.exp((frame["r_input_p"].to_numpy(float) - 24.0)))


class SpinZeroDetector:
    def __init__(self):
        self.preprocessor = SimpleNamespace(
            feature_names=[
                "Re_input_p_scaled",
                "Re_input_s_scaled",
                "r_input_p_scaled",
                "r_input_s_scaled",
                "sersic_n_input_p",
                "sersic_n_input_s",
                "distance_scaled",
            ]
        )

    def predict_proba(self, frame):
        return 1.0 / (1.0 + np.exp(frame["r_input_p_scaled"].to_numpy(float)))


class MinimalRBlendDetector:
    feature_names = [
        "e1_input_p",
        "e2_input_p",
        "sersic_n_input_p",
        "r_input_p",
        "circularized_Re_input_p",
        "nbr_flux_near",
        "nbr_flux_far",
        "nbr_flux_max",
        "R_blend",
    ]

    def __init__(self):
        self.preprocessor = SimpleNamespace(feature_names=self.feature_names)
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append(frame.loc[:, self.feature_names].copy())
        logit = frame["e1_input_p"].to_numpy(float) + frame["R_blend"].to_numpy(float)
        return 1.0 / (1.0 + np.exp(-logit))


def _likelihood():
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=8.0, weight_column="prior_weight")
    cache = CatalogueModelCache(
        prior,
        detector=_detector,
        conditions=CONDITIONS,
        detection_radius_arcsec=3.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
    )
    return CatalogueLikelihood(GaussianShapeFlow(), cache)


def _two_shape_likelihood(*, blend_values=None, flow=None, selection=None):
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=8.0, weight_column="prior_weight")
    response = None
    if blend_values is not None:
        response = CatalogueBlendResponse(
            np.asarray(blend_values, dtype=float), metadata={"test": True}, report={}
        )
    cache = CatalogueModelCache(
        prior,
        detector=_detector,
        conditions=CONDITIONS,
        detection_radius_arcsec=3.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
        blend_response=response,
    )
    return CatalogueLikelihood(flow or GaussianTwoShapeFlow(), cache, selection=selection)


class FixedSelection:
    """Known per-atom P_pass for a direct normalization check."""

    def __init__(self, probability):
        self.output_cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, 0.2)])
        self._probability = np.asarray(probability, dtype=float)

    def probability(self, flow_model, view, *, active_indices=None):
        assert len(view.flow) == len(self._probability)
        return self._probability


class AnalyticGaussianSelection:
    """Exact P_pass for the toy Gaussian flow and an upper measured cut."""

    def __init__(self, upper=0.05):
        self.upper = float(upper)
        self.output_cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, self.upper)])

    def probability(self, flow_model, view, *, active_indices=None):
        mean = view.flow["e1_input_p"].to_numpy(float)
        return norm.cdf((self.upper - mean) / flow_model.sigma)


def test_exact_catalogue_likelihood_matches_direct_finite_sum():
    likelihood = _likelihood()
    observed = pd.DataFrame({"measured_e1": [0.05, -0.1]})
    actual = likelihood.log_likelihood(observed, 0.0, 0.0)

    view = likelihood.cache.get(0.0, 0.0)
    means = view.flow["e1_input_p"].to_numpy(float)
    sigma = likelihood.flow_model.sigma
    density = np.exp(-0.5 * ((observed["measured_e1"].to_numpy()[:, None] - means) / sigma) ** 2) / (
        sigma * np.sqrt(2 * np.pi)
    )
    mass = likelihood.cache.prior.weights * view.detection_probability
    expected = np.log(density @ mass) - np.log(mass.sum())
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_measured_selection_enters_only_the_population_normalization():
    base = _likelihood()
    p_pass = np.array([0.2, 0.4, 0.7, 0.9])
    likelihood = CatalogueLikelihood(
        base.flow_model,
        base.cache,
        selection=FixedSelection(p_pass),
    )
    observed = pd.DataFrame({"measured_e1": [0.05, -0.1]})
    actual = likelihood.log_likelihood(observed, 0.0, 0.0)

    view = likelihood.cache.get(0.0, 0.0)
    means = view.flow["e1_input_p"].to_numpy(float)
    sigma = likelihood.flow_model.sigma
    density = np.exp(-0.5 * ((observed["measured_e1"].to_numpy()[:, None] - means) / sigma) ** 2) / (
        sigma * np.sqrt(2 * np.pi)
    )
    detected_mass = likelihood.cache.prior.weights * view.detection_probability
    expected = np.log(density @ detected_mass) - np.log(np.sum(detected_mass * p_pass))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    with pytest.raises(ValueError, match="fail the declared measured cut"):
        likelihood.log_likelihood(pd.DataFrame({"measured_e1": [0.25]}), 0.0, 0.0)


def test_selection_probability_cache_round_trip_and_common_seed(tmp_path):
    base = _likelihood()
    cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, 0.05)])
    selection = CatalogueSelection(cut, n_samples=32, seed=91, row_chunk=2)
    selection.metadata = {"identity": "toy"}
    view = base.cache.get(0.0, 0.0)
    first = selection.probability(base.flow_model, view).copy()
    selection.save(tmp_path / "selection")

    restored = CatalogueSelection.load(tmp_path / "selection", output_cut=cut)
    assert restored.metadata == {"identity": "toy"}
    np.testing.assert_array_equal(restored.probability(base.flow_model, view), first)
    assert restored.available_shears == ((0.0, 0.0),)


def test_selection_cache_extension_keeps_existing_shear_file_stable(tmp_path):
    base = _likelihood()
    cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, 0.05)])
    selection = CatalogueSelection(cut, n_samples=32, seed=91, row_chunk=2)
    selection.probability(base.flow_model, base.cache.get(0.0, 0.0))
    root = tmp_path / "selection"
    selection.save(root)
    first_manifest = json.loads((root / "manifest.json").read_text())
    first_name = first_manifest["entries"][0]["probability"]
    first_bytes = (root / first_name).read_bytes()

    selection.probability(base.flow_model, base.cache.get(0.02, -0.01))
    selection.save(root)
    second_manifest = json.loads((root / "manifest.json").read_text())
    zero_entry = next(
        entry for entry in second_manifest["entries"] if entry["g1"] == 0.0 and entry["g2"] == 0.0
    )
    assert zero_entry["probability"] == first_name
    assert (root / first_name).read_bytes() == first_bytes
    assert not tuple(root.glob(".*.tmp"))


@pytest.mark.parametrize(
    "corrupt_probability",
    (
        np.array([0.1, 0.2, np.nan, 0.4]),
        np.array([0.1, 0.2, 1.1, 0.4]),
        np.array([0.1, 0.2, 0.3]),
    ),
)
def test_selection_cache_rejects_invalid_probability_arrays(tmp_path, corrupt_probability):
    base = _likelihood()
    cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, 0.05)])
    selection = CatalogueSelection(cut, n_samples=32, seed=91, row_chunk=2)
    selection.probability(base.flow_model, base.cache.get(0.0, 0.0))
    root = tmp_path / "selection"
    selection.save(root)
    manifest = json.loads((root / "manifest.json").read_text())
    np.save(root / manifest["entries"][0]["probability"], corrupt_probability)

    with pytest.raises(ValueError, match="selection probabilit"):
        CatalogueSelection.load(root, output_cut=cut)


def test_selection_probability_skips_zero_mass_scene_rows():
    base = _likelihood()
    cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, 0.05)])
    selection = CatalogueSelection(cut, n_samples=32, seed=96, row_chunk=2)
    probability = selection.probability(
        base.flow_model,
        base.cache.get(0.0, 0.0),
        active_indices=np.array([1, 3]),
    )
    np.testing.assert_array_equal(probability[[0, 2]], 0.0)
    assert ((probability[[1, 3]] > 0) & (probability[[1, 3]] < 1)).all()


def test_distributed_selection_mass_matches_full_probability_scan():
    base = _likelihood()
    base = CatalogueLikelihood(RandomShiftShapeFlow(), base.cache)
    cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, 0.05)])
    selection = CatalogueSelection(cut, n_samples=32, seed=96, row_chunk=2)
    view = base.cache.get(0.0, 0.0)
    active = np.flatnonzero(base.cache.prior.weights > 0).astype(np.int64)
    probability = selection.probability(base.flow_model, view, active_indices=active)
    expected = np.sum(
        base.cache.prior.weights * view.detection_probability * probability
    )
    first = detected_selected_mass_shard(
        selection,
        base.flow_model,
        view,
        base.cache.prior.weights,
        active_indices=active[:2],
        random_offset_rows=0,
    )
    second = detected_selected_mass_shard(
        selection,
        base.flow_model,
        view,
        base.cache.prior.weights,
        active_indices=active[2:],
        random_offset_rows=2,
    )
    assert first + second == pytest.approx(expected, rel=0, abs=1e-15)


def test_selected_mock_generation_applies_the_same_output_cut():
    base = _likelihood()
    cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, 0.05)])
    likelihood = CatalogueLikelihood(
        base.flow_model,
        base.cache,
        selection=CatalogueSelection(cut, n_samples=32, seed=92, row_chunk=4),
    )
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=200,
        g1=0.0,
        g2=0.0,
        scene_seed=93,
        detection_seed=94,
        flow_seed=95,
    )
    assert (mock.measurements["measured_e1"] < 0.05).all()


def test_selected_likelihood_profile_recovers_shear_under_strong_cut():
    base = _likelihood()
    likelihood = CatalogueLikelihood(
        base.flow_model,
        base.cache,
        selection=AnalyticGaussianSelection(upper=0.05),
    )
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=8000,
        g1=0.02,
        g2=0.0,
        scene_seed=101,
        detection_seed=201,
        flow_seed=301,
        max_candidates=100_000,
    )
    grid = np.linspace(0.0, 0.04, 17)
    log_likelihood = np.array(
        [
            likelihood.log_likelihood(
                mock.measurements,
                shear,
                0.0,
                object_chunk=2000,
                atom_chunk=4,
            ).sum()
            for shear in grid
        ]
    )
    # A finite mock's MLE need not land exactly on the injected grid point.
    # This fixed realization peaks one cell high; the direct normalization
    # oracle above is the exact regression for the selection formula.
    assert abs(grid[np.argmax(log_likelihood)] - 0.02) <= grid[1] - grid[0]


def test_importance_sum_with_prior_proposal_matches_exact_when_enumerated():
    likelihood = _likelihood()
    observed = pd.DataFrame({"measured_e1": [0.02]})
    exact = likelihood.log_likelihood(observed, 0.0, 0.0)
    rows = np.arange(len(likelihood.cache.prior.weights))
    sampled = likelihood.log_likelihood(
        observed,
        0.0,
        0.0,
        atom_indices=rows,
        proposal_probability=np.full(len(rows), 1.0 / len(rows)),
    )
    np.testing.assert_allclose(sampled, exact, rtol=0, atol=1e-12)


def test_target_specific_importance_matrix_matches_exact_when_enumerated():
    likelihood = _likelihood()
    observed = pd.DataFrame({"measured_e1": [0.02, -0.08]})
    exact = likelihood.log_likelihood(observed, 0.0, 0.0)
    rows = np.arange(len(likelihood.cache.prior.weights))
    indices = np.broadcast_to(rows, (len(observed), len(rows)))
    proposal = np.full(indices.shape, 1.0 / len(rows))
    sampled = likelihood.log_likelihood(
        observed,
        0.0,
        0.0,
        atom_indices=indices,
        proposal_probability=proposal,
    )
    np.testing.assert_allclose(sampled, exact, rtol=0, atol=1e-12)


def test_model_cache_round_trip_reuses_views_without_detector(tmp_path):
    likelihood = _likelihood()
    observed = pd.DataFrame({"measured_e1": [0.02]})
    expected = likelihood.log_likelihood(observed, 0.0, 0.0)
    likelihood.cache.save(tmp_path / "model-cache", metadata={"model": "fake"})

    restored = CatalogueModelCache.load(tmp_path / "model-cache", prior=likelihood.cache.prior)
    assert restored.detector is None
    assert restored.metadata == {"model": "fake"}
    actual = CatalogueLikelihood(likelihood.flow_model, restored).log_likelihood(observed, 0.0, 0.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    restored.attach_detector(_detector)
    restored.get(0.01, 0.0)
    assert (0.01, 0.0) in restored.available_shears
    restored.discard_views(((0.01, 0.0),))
    assert (0.01, 0.0) not in restored.available_shears


@pytest.mark.parametrize("corruption", ("reorder", "drop"))
def test_model_cache_rejects_flow_rows_not_aligned_with_scene(tmp_path, corruption):
    likelihood = _likelihood()
    root = tmp_path / "model-cache"
    likelihood.cache.save(root)
    manifest = json.loads((root / "manifest.json").read_text())
    flow_path = root / manifest["views"][0]["flow"]
    flow = pd.read_parquet(flow_path)
    if corruption == "reorder":
        flow = flow.iloc[::-1]
    else:
        flow = flow.iloc[:-1]
    flow.to_parquet(flow_path, index=False)

    with pytest.raises(ValueError, match="flow rows are not aligned"):
        CatalogueModelCache.load(root, prior=likelihood.cache.prior)


def test_model_cache_rejects_detection_rows_not_aligned_with_scene(tmp_path):
    likelihood = _likelihood()
    root = tmp_path / "model-cache"
    likelihood.cache.save(root)
    manifest = json.loads((root / "manifest.json").read_text())
    detection_path = root / manifest["views"][0]["detection"]
    detection = pd.read_parquet(detection_path).iloc[::-1]
    detection.to_parquet(detection_path, index=False)

    with pytest.raises(ValueError, match="detection rows are not aligned"):
        CatalogueModelCache.load(root, prior=likelihood.cache.prior)


def test_shape_only_compact_cache_stores_spin0_conditions_once(tmp_path):
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=8.0, weight_column="prior_weight")
    features = (
        "e1_input_p",
        "e2_input_p",
        "sersic_n_input_p",
        "r_input_p",
        "Re_input_p",
        "nbr_flux_near",
        "nbr_flux_far",
        "nbr_flux_max",
    )
    detector = SpinZeroDetector()
    cache = CatalogueModelCache(
        prior,
        detector=detector,
        conditions=CONDITIONS,
        detection_radius_arcsec=3.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
        flow_features=features,
    )
    zero = cache.get(0.0, 0.0)
    shifted = cache.get(0.005, 0.0)
    reference_cache = CatalogueModelCache(
        prior,
        detector=detector,
        conditions=CONDITIONS,
        detection_radius_arcsec=3.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
    )
    reference = reference_cache.get(0.005, 0.0)
    np.testing.assert_allclose(
        shifted.flow.loc[:, features], reference.flow.loc[:, features], rtol=0, atol=1e-14
    )
    np.testing.assert_array_equal(zero.flow["nbr_flux_near"], shifted.flow["nbr_flux_near"])
    assert not np.array_equal(zero.flow["e1_input_p"], shifted.flow["e1_input_p"])
    np.testing.assert_array_equal(zero.detection_probability, shifted.detection_probability)
    assert cache.validate_detection_shear_invariance() == tuple(detector.preprocessor.feature_names)

    cache.save(tmp_path / "compact")
    manifest = json.loads((tmp_path / "compact" / "manifest.json").read_text())
    assert manifest["storage"] == "shape_only_zero_base"
    assert len(manifest["views"]) == 1
    restored = CatalogueModelCache.load(tmp_path / "compact", prior=prior)
    assert restored.available_shears == ((0.0, 0.0),)
    restored.attach_detector(detector)
    np.testing.assert_allclose(
        restored.get(0.005, 0.0).flow["e1_input_p"],
        shifted.flow["e1_input_p"],
    )


def test_minimal_rblend_detector_gets_aligned_invariant_and_sheared_features(tmp_path, monkeypatch):
    prior = ScenePrior.from_catalogue(
        _catalogue(), guard_radius_arcsec=8.0, weight_column="prior_weight"
    )
    detector = MinimalRBlendDetector()
    response = CatalogueBlendResponse(
        np.array([0.1, 0.2, 0.3, 0.4]), metadata={"test": True}, report={}
    )
    flow_features = tuple(detector.feature_names[:-1])
    cache = CatalogueModelCache(
        prior,
        detector=detector,
        conditions=CONDITIONS,
        detection_radius_arcsec=3.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
        blend_response=response,
        flow_features=flow_features,
    )

    zero = cache.get(0.0, 0.0)
    cache.save(tmp_path / "coherent")
    restored = CatalogueModelCache.load(tmp_path / "coherent", prior=prior, blend_response=response)
    restored.attach_detector(MinimalRBlendDetector())
    # A fresh full-scene view is the oracle. The restored compact cache must
    # agree without consulting a graph at any nonzero shear.
    reference = CatalogueModelCache(
        prior, detector=MinimalRBlendDetector(), conditions=CONDITIONS,
        detection_radius_arcsec=3.0, flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0), blend_response=response,
    ).get(0.01, 0.0)
    def no_graph(*args, **kwargs):
        raise AssertionError("compact view unexpectedly used the neighbour graph")
    monkeypatch.setattr(ScenePrior, "shear", no_graph)
    shifted = cache.get(0.01, 0.0)
    recovered = restored.get(0.01, 0.0)
    np.testing.assert_allclose(recovered.detection, reference.detection, rtol=0, atol=1e-14)
    np.testing.assert_allclose(recovered.detection_probability, reference.detection_probability)
    np.testing.assert_allclose(recovered.blend_shift, reference.blend_shift, rtol=0, atol=1e-14)
    expected_radius = _catalogue()["Re"].to_numpy() * np.sqrt(
        _catalogue()["axis_ratio"].to_numpy()
    )

    assert tuple(zero.detection.columns) == tuple(detector.feature_names)
    np.testing.assert_allclose(zero.flow["circularized_Re_input_p"], expected_radius)
    np.testing.assert_array_equal(
        zero.flow["circularized_Re_input_p"],
        shifted.flow["circularized_Re_input_p"],
    )
    np.testing.assert_array_equal(zero.detection["R_blend"], response.values)
    np.testing.assert_array_equal(shifted.detection["R_blend"], response.values)
    for name in ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max"):
        np.testing.assert_array_equal(zero.detection[name], zero.flow[name])
    assert not np.array_equal(zero.detection["e1_input_p"], shifted.detection["e1_input_p"])
    assert not np.array_equal(zero.detection_probability, shifted.detection_probability)
    assert len(detector.calls) == 2


def test_minimal_rblend_detector_requires_blend_response():
    prior = ScenePrior.from_catalogue(
        _catalogue(), guard_radius_arcsec=8.0, weight_column="prior_weight"
    )
    cache = CatalogueModelCache(
        prior,
        detector=MinimalRBlendDetector(),
        conditions=CONDITIONS,
        detection_radius_arcsec=3.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
    )
    with pytest.raises(KeyError, match="R_blend requires"):
        cache.get(0.0, 0.0)


def test_detection_neighbour_rule_is_applied_and_persisted(tmp_path):
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=8.0, weight_column="prior_weight")
    cache = CatalogueModelCache(
        prior,
        detector=_detector,
        conditions=CONDITIONS,
        detection_radius_arcsec=3.0,
        detection_neighbour_selection="impact",
        detection_impact_exponent=1.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
    )
    actual = cache.get(0.0, 0.0).detection
    expected = prior.shear(0.0, 0.0).detection_view(
        conditions=CONDITIONS,
        radius_arcsec=3.0,
        neighbour_selection="impact",
        impact_exponent=1.0,
    )
    pd.testing.assert_frame_equal(actual, expected)

    cache.save(tmp_path / "impact")
    restored = CatalogueModelCache.load(tmp_path / "impact", prior=prior)
    assert restored.detection_neighbour_selection == "impact"
    assert restored.detection_impact_exponent == 1.0


def test_score_can_be_evaluated_about_nonzero_shear():
    likelihood = _likelihood()
    observed = pd.DataFrame({"measured_e1": [0.02, -0.08]})
    center = 0.01
    delta = 0.0025
    score = likelihood.score_and_information(
        observed,
        center=(center, 0.0),
        delta=delta,
        richardson=False,
    )
    middle = likelihood.log_likelihood(observed, center, 0.0)
    plus = likelihood.log_likelihood(observed, center + delta, 0.0)
    minus = likelihood.log_likelihood(observed, center - delta, 0.0)
    np.testing.assert_allclose(score.log_likelihood, middle)
    np.testing.assert_allclose(score.score, (plus - minus) / (2 * delta))
    np.testing.assert_allclose(
        score.information,
        -(plus - 2 * middle + minus) / delta**2,
    )


def test_mock_catalogue_round_trip_is_exact(tmp_path):
    mock = generate_mock_catalogue(
        _likelihood(),
        n_detected=12,
        g1=0.02,
        g2=0.0,
        scene_seed=50,
        detection_seed=51,
        flow_seed=52,
    )
    mock.save(tmp_path / "mock")
    restored = MockCatalogue.load(tmp_path / "mock")
    pd.testing.assert_frame_equal(restored.measurements, mock.measurements)
    pd.testing.assert_frame_equal(restored.truth, mock.truth)


def test_mock_generation_and_exact_small_catalogue_closure():
    likelihood = _likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=4000,
        g1=0.02,
        g2=0.0,
        scene_seed=10,
        detection_seed=11,
        flow_seed=12,
    )
    result = run_exact_closure(
        likelihood,
        mock,
        direction=(1.0, 0.0),
        delta=0.005,
        richardson=True,
        object_chunk=1000,
        atom_chunk=4,
    )
    assert result.n_detected == 4000
    assert abs(result.estimated_shear - result.injected_shear) < 0.012
    assert np.isfinite(result.robust_standard_error)
    assert result.robust_standard_error > 0
    assert np.isfinite(result.model_standard_error)
    np.testing.assert_allclose(
        result.closure_pull,
        (result.estimated_shear - result.injected_shear) / result.robust_standard_error,
    )
    assert {"scene_row", "detection_uniform", "flow_seed"}.issubset(mock.truth)


def test_blend_response_likelihood_matches_direct_shifted_finite_sum():
    response = np.array([0.4, -0.2, 0.1, 0.7])
    likelihood = _two_shape_likelihood(blend_values=response)
    observed = pd.DataFrame(
        {
            "measured_ngmix_g1": [0.05, -0.1],
            "measured_ngmix_g2": [-0.03, 0.08],
        }
    )
    g1, g2 = 0.025, -0.015
    actual = likelihood.log_likelihood(observed, g1, g2)

    view = likelihood.cache.get(g1, g2)
    mean = view.flow[["e1_input_p", "e2_input_p"]].to_numpy(float)
    mean = mean + view.blend_shift
    residual = observed.to_numpy(float)[:, None, :] - mean[None, :, :]
    sigma = likelihood.flow_model.sigma
    density = np.exp(-0.5 * np.square(residual / sigma).sum(axis=2)) / (2.0 * np.pi * sigma**2)
    mass = likelihood.cache.prior.weights * view.detection_probability
    expected = np.log(density @ mass) - np.log(mass.sum())
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_mock_generator_adds_exact_atom_blend_shift_and_records_truth():
    response = np.array([0.4, -0.2, 0.1, 0.7])
    base = _two_shape_likelihood()
    blended = _two_shape_likelihood(blend_values=response)
    kwargs = dict(
        n_detected=128,
        g1=0.03,
        g2=-0.02,
        scene_seed=401,
        detection_seed=402,
        flow_seed=403,
    )
    base_mock = generate_mock_catalogue(base, **kwargs)
    blend_mock = generate_mock_catalogue(blended, **kwargs)

    np.testing.assert_array_equal(blend_mock.truth["scene_row"], base_mock.truth["scene_row"])
    difference = blend_mock.measurements.to_numpy() - base_mock.measurements.to_numpy()
    recorded = blend_mock.truth[["blend_shift_g1", "blend_shift_g2"]].to_numpy()
    np.testing.assert_allclose(difference, recorded, rtol=0, atol=2e-8)
    rows = blend_mock.truth["scene_row"].to_numpy(dtype=int)
    np.testing.assert_array_equal(blend_mock.truth["r_blend"], response[rows])
    np.testing.assert_array_equal(base_mock.truth["r_blend"], 0.0)


def test_selection_qmc_applies_blend_shift_before_measured_cut():
    flow = GaussianTwoShapeFlow(deterministic_samples=True)
    cut = OutputCut(
        flow.target_transform.target_names,
        bounds=[("measured_ngmix_g1", None, 0.15)],
    )
    response = np.full(4, 3.0)
    base = _two_shape_likelihood(flow=flow)
    blended = _two_shape_likelihood(blend_values=response, flow=flow)
    base_selection = CatalogueSelection(cut, n_samples=8, seed=501, row_chunk=2)
    blend_selection = CatalogueSelection(cut, n_samples=8, seed=501, row_chunk=2)
    g1 = 0.04
    base_probability = base_selection.probability(flow, base.cache.get(g1, 0.0))
    blend_view = blended.cache.get(g1, 0.0)
    blend_probability = blend_selection.probability(flow, blend_view)

    expected_draws = blend_view.flow[["e1_input_p", "e2_input_p"]].to_numpy(float) + blend_view.blend_shift
    expected = (expected_draws[:, 0] < 0.15).astype(float)
    np.testing.assert_array_equal(blend_probability, expected)
    assert np.any(blend_probability != base_probability)


def test_selection_common_random_numbers_do_not_depend_on_shear_order():
    flow = GaussianTwoShapeFlow()
    cut = OutputCut(
        flow.target_transform.target_names,
        bounds=[("measured_ngmix_g1", None, 0.15)],
    )
    likelihood = _two_shape_likelihood(
        blend_values=[0.6, -0.1, 0.3, 0.8],
        flow=flow,
    )
    first = CatalogueSelection(cut, n_samples=32, seed=502, row_chunk=2)
    second = CatalogueSelection(cut, n_samples=32, seed=502, row_chunk=2)
    points = ((0.02, -0.01), (-0.015, 0.025), (0.0, 0.0))
    forward = {point: first.probability(flow, likelihood.cache.get(*point)).copy() for point in points}
    reverse = {
        point: second.probability(flow, likelihood.cache.get(*point)).copy() for point in reversed(points)
    }
    for point in points:
        np.testing.assert_array_equal(forward[point], reverse[point])


def test_model_cache_round_trip_reconstructs_blend_shift(tmp_path):
    response = np.array([0.4, -0.2, 0.1, 0.7])
    likelihood = _two_shape_likelihood(blend_values=response)
    expected = likelihood.cache.get(0.02, -0.01).blend_shift.copy()
    likelihood.cache.save(tmp_path / "model-cache")

    restored = CatalogueModelCache.load(
        tmp_path / "model-cache",
        prior=likelihood.cache.prior,
        blend_response=likelihood.cache.blend_response,
    )
    np.testing.assert_allclose(restored.get(0.02, -0.01).blend_shift, expected, rtol=0, atol=0)


def test_blend_cache_reuses_spin0_views_and_only_updates_shape_and_shift():
    likelihood = _two_shape_likelihood(blend_values=[0.4, -0.2, 0.1, 0.7])
    cache = likelihood.cache
    zero = cache.get(0.0, 0.0)
    cache.flow_features = ("e1_input_p", "e2_input_p")
    cache.detection_features = ("r_input_p_scaled",)

    def fail_if_recomputed(frame):
        raise AssertionError("spin-0 detection was recomputed at nonzero shear")

    cache.detector = fail_if_recomputed
    g1, g2 = 0.02, -0.01
    view = cache.get(g1, g2)
    assert view.detection is zero.detection
    assert view.detection_probability is zero.detection_probability
    expected_scene = cache.prior.shear(g1, g2)
    expected_flow = expected_scene.flow_view(
        conditions=cache.conditions,
        neighbour_radius_arcsec=cache.flow_neighbour_radius_arcsec,
        crowding_radii_arcsec=cache.crowding_radii_arcsec,
    )
    np.testing.assert_allclose(
        view.flow[["e1_input_p", "e2_input_p"]],
        expected_flow[["e1_input_p", "e2_input_p"]],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        view.blend_shift,
        cache.blend_response.shape_shift(expected_scene),
        rtol=0,
        atol=0,
    )


def test_nonzero_blend_response_closure_profile_recovers_injected_shear():
    likelihood = _two_shape_likelihood(blend_values=[0.6, -0.1, 0.3, 0.8])
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=6000,
        g1=0.02,
        g2=0.0,
        scene_seed=601,
        detection_seed=602,
        flow_seed=603,
    )
    grid = np.linspace(0.0, 0.04, 17)
    profile = np.array(
        [
            likelihood.log_likelihood(
                mock.measurements,
                shear,
                0.0,
                object_chunk=2000,
                atom_chunk=4,
            ).sum()
            for shear in grid
        ]
    )
    assert abs(grid[np.argmax(profile)] - 0.02) <= 0.005
