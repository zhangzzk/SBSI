import numpy as np

from scripts.compute_anchorblend_outer_features import outer_summaries


def test_outer_summaries_obey_annulus_boundaries_and_extent():
    distance = np.array([9.9, 10.0, 14.9, 15.0, 19.9, 20.0, 29.9, 30.0])
    flux = np.arange(1.0, 9.0)
    re_value = np.ones(8)
    got = outer_summaries(distance, flux, re_value)
    assert np.isclose(got["logflux_outer_10_15"], np.log10(1 + 2 + 3))
    assert np.isclose(got["logflux_outer_15_20"], np.log10(1 + 4 + 5))
    assert np.isclose(got["logflux_outer_20_30"], np.log10(1 + 6 + 7))
    assert got["n_outer_10_30"] == 6
    assert np.isclose(got["max_re_over_d_outer_10_30"], 0.1)


def test_outer_summaries_empty():
    got = outer_summaries(np.array([]), np.array([]), np.array([]))
    assert got["n_outer_10_30"] == 0
    assert got["max_re_over_d_outer_10_30"] == 0.0
