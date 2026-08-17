import numpy as np

from scripts.plot_anchor_high_response_population import (
    finite_stat,
    normalized_histogram,
)


def test_normalized_histogram_mass_and_density() -> None:
    values = np.array([0.2, 0.8, 1.2])
    weights = np.array([0.5, 0.5, 1.0])
    bins = np.array([0.0, 1.0, 3.0])
    mass = normalized_histogram(
        values, bins, weights=weights, discrete=True,
    )
    density = normalized_histogram(
        values, bins, weights=weights, discrete=False,
    )
    np.testing.assert_allclose(mass, [0.5, 0.5])
    np.testing.assert_allclose(density, [0.5, 0.25])
    np.testing.assert_allclose(np.sum(density * np.diff(bins)), 1.0)


def test_equal_anchor_neighbour_weighting() -> None:
    # The first anchor has two neighbours and the second has one. Each anchor
    # contributes total weight one rather than the first counting twice.
    values = np.array([0.2, 0.8, 1.2])
    weights = np.array([0.5, 0.5, 1.0])
    bins = np.array([0.0, 1.0, 2.0])
    result = normalized_histogram(
        values, bins, weights=weights, discrete=True,
    )
    np.testing.assert_allclose(result, [0.5, 0.5])


def test_finite_stat_uses_case_sem() -> None:
    result = finite_stat(np.array([1.0, 2.0, 3.0, 4.0]))
    assert result["n_cases"] == 4
    np.testing.assert_allclose(result["mean"], 2.5)
    np.testing.assert_allclose(
        result["case_sem"], np.std([1.0, 2.0, 3.0, 4.0], ddof=1) / 2.0,
    )
