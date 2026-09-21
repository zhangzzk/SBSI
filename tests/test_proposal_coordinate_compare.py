"""Unit tests for the proposal-coordinate comparison helpers.

These cover the pure helpers on small synthetic arrays.  The whole-catalogue
reproduction of the recorded production proposal is asserted inside the job
itself, against the 24m cached coordinate table, and is not repeated here.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.compare_proposal_coordinates import (
    FOUR_D,
    PRODUCTION_DELTA,
    SIZE_FLUX_2D,
    build_variants,
    check_variants,
    component,
    coverage_metrics,
    local_component,
    variant_dispersion,
)
from sbsi.catalogue_sampling import ProposalCoordinateTable, WholeCatalogueProxy


def test_variant_dispersion_scales_only_named_columns():
    dispersion = np.arange(1.0, 13.0).reshape(3, 4)
    scaled = variant_dispersion(dispersion, {0: 3.0, 1: 10.0})
    np.testing.assert_allclose(scaled[:, 0], dispersion[:, 0] * 3.0)
    np.testing.assert_allclose(scaled[:, 1], dispersion[:, 1] * 10.0)
    np.testing.assert_allclose(scaled[:, 2:], dispersion[:, 2:])
    # The input must not be mutated in place.
    np.testing.assert_allclose(dispersion, np.arange(1.0, 13.0).reshape(3, 4))


@pytest.mark.parametrize("factor", [0.0, -1.0, np.nan, np.inf])
def test_variant_dispersion_rejects_invalid_inflation(factor):
    with pytest.raises(ValueError):
        variant_dispersion(np.ones((2, 4)), {0: factor})


def test_coverage_metrics_match_hand_computation():
    weights = np.array([0.5, 0.25, 0.05])
    probabilities = np.array([0.8, 1.0e-3, 4.0e-9])
    metrics = coverage_metrics(weights, probabilities, floor=4.0e-9, n_draws=16384)
    assert metrics["capture"] == pytest.approx(0.80)
    assert metrics["n_at_defensive_floor"] == 1
    assert metrics["max_proposal_probability"] == pytest.approx(0.8)
    expected_second = 0.5**2 / 0.8 + 0.25**2 / 1.0e-3 + 0.05**2 / 4.0e-9
    assert metrics["second_moment"] == pytest.approx(expected_second)
    assert metrics["ess_upper_bound"] == pytest.approx(16384 / expected_second)
    assert metrics["expected_hits_excluding_largest"] == pytest.approx(
        16384 * (probabilities.sum() - 0.8)
    )


def test_coverage_metrics_rejects_zero_probability():
    with pytest.raises(ValueError):
        coverage_metrics(np.array([1.0]), np.array([0.0]), floor=1e-9)


def test_build_variants_includes_production_and_two_dimensional():
    variants = {v["name"]: v for v in build_variants()}
    production = variants["prod_4d"]["components"]
    assert len(production) == 1
    assert production[0]["columns"] == FOUR_D
    assert production[0]["temperature"] == 1.0
    assert production[0]["inflation"] == {}
    # The owner's idea: size and brightness only, dropping both shape columns.
    assert variants["rf_2d"]["components"][0]["columns"] == SIZE_FLUX_2D
    # The mixture keeps both, at equal weight.
    mixture = variants["mix_4d_rf2d"]["components"]
    assert [c["columns"] for c in mixture] == [FOUR_D, SIZE_FLUX_2D]
    assert [c["weight"] for c in mixture] == [0.5, 0.5]
    check_variants(build_variants())


def test_check_variants_rejects_non_convex_weights():
    with pytest.raises(ValueError, match="sum to"):
        check_variants(
            [
                {
                    "name": "bad",
                    "components": [
                        component(FOUR_D, weight=0.5),
                        component(SIZE_FLUX_2D, weight=0.9),
                    ],
                    "note": "",
                }
            ]
        )


def test_check_variants_rejects_duplicate_names():
    duplicate = {"name": "x", "components": [component(FOUR_D)], "note": ""}
    with pytest.raises(ValueError, match="unique"):
        check_variants([duplicate, dict(duplicate)])


def test_coordinate_table_exposes_the_attributes_the_job_reads():
    """Guard the real attribute names the comparison job depends on."""

    table = ProposalCoordinateTable(
        np.zeros((4, 2)),
        ("measured_flux_radius", "measured_flux_from_mag_auto"),
        np.zeros(2),
        np.ones(2),
        dispersion=np.ones((4, 2)),
    )
    assert list(table.target_names) == [
        "measured_flux_radius",
        "measured_flux_from_mag_auto",
    ]
    assert table.values.shape == (4, 2)
    assert table.dispersion.shape == (4, 2)
    assert not hasattr(table, "names")


def test_report_metrics_are_json_serializable_without_nan():
    """write_json uses allow_nan=False, so no metric may be inf or nan."""

    metrics = coverage_metrics(
        np.array([0.0, 0.0]), np.array([0.5, 0.5]), floor=1e-9
    )
    assert metrics["second_moment"] == 0.0
    assert metrics["ess_upper_bound"] is None
    json.dumps(metrics, allow_nan=False)


def _proxy(values, dispersion, prior, detection):
    shim = SimpleNamespace(
        coordinates=SimpleNamespace(values=values, dispersion=dispersion),
        active_indices=np.arange(len(values)),
        local_base_weights=detection,
        prior_weights=prior,
    )
    return WholeCatalogueProxy(shim, torch.device("cpu"), score_dtype=torch.float64)


def test_proxy_shim_reproduces_explicit_gaussian_mixture():
    """The shim drives the real production proxy class, so check its algebra."""

    values = np.array([[0.0, 1.0], [2.0, -1.0], [-3.0, 0.5]])
    dispersion = np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.25]])
    prior = np.full(3, 1.0 / 3.0)
    detection = np.array([0.5, 1.0, 0.25])
    observation = np.array([[0.3, 0.8]])

    mixture = _proxy(values, dispersion, prior, detection).mixture(
        observation, delta=PRODUCTION_DELTA
    )

    score = (
        np.log(detection)
        - np.log(dispersion).sum(axis=1)
        - 0.5 * np.square((observation[0] - values) / dispersion).sum(axis=1)
    )
    local = np.exp(score - score.max())
    local /= local.sum()
    expected = PRODUCTION_DELTA * prior + (1.0 - PRODUCTION_DELTA) * local
    np.testing.assert_allclose(mixture.numpy()[0], expected, rtol=1e-12, atol=0)
    assert mixture.numpy().sum() == pytest.approx(1.0)


def test_tempering_flattens_and_dropping_columns_changes_ranking():
    values = np.array([[0.0, 0.0], [0.4, 0.0], [0.0, 5.0]])
    dispersion = np.array([[0.05, 1.0], [0.05, 1.0], [0.05, 1.0]])
    prior = np.full(3, 1.0 / 3.0)
    detection = np.ones(3)
    observation = np.array([[0.0, 0.0]])
    proxy = _proxy(values, dispersion, prior, detection)

    cold = proxy.mixture(observation, delta=PRODUCTION_DELTA, temperature=1.0).numpy()[0]
    warm = proxy.mixture(observation, delta=PRODUCTION_DELTA, temperature=4.0).numpy()[0]
    # Tempering moves mass off the winner and onto the starved atoms.
    assert warm.max() < cold.max()
    assert warm[1] > cold[1]

    # Dropping the first column removes the coordinate that separated atom 1,
    # which is exactly the mechanism the two-dimensional variant relies on.
    narrow = _proxy(
        values[:, 1:], dispersion[:, 1:], prior, detection
    ).mixture(observation[:, 1:], delta=PRODUCTION_DELTA).numpy()[0]
    assert narrow[1] > cold[1]
    assert narrow[0] == pytest.approx(narrow[1], rel=1e-12)


def test_local_component_inverts_the_defensive_blend():
    """local_component must return the bare softmax, summing to one."""

    values = np.array([[0.0, 1.0], [2.0, -1.0], [-3.0, 0.5]])
    dispersion = np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.25]])
    prior = np.full(3, 1.0 / 3.0)
    proxy = _proxy(values, dispersion, prior, np.array([0.5, 1.0, 0.25]))
    observation = np.array([[0.3, 0.8]])

    local = local_component(proxy, observation, 1.0).numpy()[0]
    assert local.sum() == pytest.approx(1.0, rel=1e-12)
    assert (local >= 0).all()

    rebuilt = PRODUCTION_DELTA * prior + (1.0 - PRODUCTION_DELTA) * local
    expected = proxy.mixture(observation, delta=PRODUCTION_DELTA).numpy()[0]
    np.testing.assert_allclose(rebuilt, expected, rtol=1e-12, atol=0)


def test_equal_mixture_lies_between_its_two_components():
    """A 50/50 mixture must be the mean of the components it blends."""

    values = np.array([[0.0, 0.0], [0.4, 0.0], [0.0, 5.0]])
    dispersion = np.full((3, 2), 0.05)
    dispersion[:, 1] = 1.0
    prior = np.full(3, 1.0 / 3.0)
    detection = np.ones(3)
    observation = np.array([[0.0, 0.0]])

    wide = _proxy(values, dispersion, prior, detection)
    narrow = _proxy(values[:, 1:], dispersion[:, 1:], prior, detection)
    a = local_component(wide, observation, 1.0).numpy()[0]
    b = local_component(narrow, observation[:, 1:], 1.0).numpy()[0]
    blended = 0.5 * a + 0.5 * b

    assert blended.sum() == pytest.approx(1.0, rel=1e-12)
    # The mixture never starves an atom either component reaches.
    assert blended.min() >= min(a.min(), b.min())
    assert blended[1] > a[1]
