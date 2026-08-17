import numpy as np

from scripts.analyze_v22_rblend_within_bin_slopes import (
    assign_edge_bins,
    grouped_slope_with_case_jackknife,
    slope_from_sums,
)


def test_slope_from_sums_controls_group_intercepts():
    x = np.asarray([0.0, 1.0, 0.0, 1.0])
    y = np.asarray([2.0, 5.0, 12.0, 15.0])
    group = np.asarray([0, 0, 1, 1])
    n = np.bincount(group, minlength=2)
    sx = np.bincount(group, weights=x, minlength=2)
    sy = np.bincount(group, weights=y, minlength=2)
    sxx = np.bincount(group, weights=x * x, minlength=2)
    sxy = np.bincount(group, weights=x * y, minlength=2)

    assert slope_from_sums(n, sx, sy, sxx, sxy) == 3.0


def test_grouped_case_jackknife_recovers_known_slope():
    cases = np.repeat(np.arange(4), 4)
    groups = np.tile(np.asarray([0, 0, 1, 1]), 4)
    x = np.tile(np.asarray([0.0, 1.0, 0.0, 1.0]), 4)
    y = 3.0 * x + 10.0 * groups + np.repeat(np.asarray([0.0, 1.0, -1.0, 0.5]), 4)
    slope, sem, ncase = grouped_slope_with_case_jackknife(x, y, cases, groups, 2)

    assert slope == 3.0
    assert sem < 1e-12
    assert ncase == 4


def test_assign_edge_bins_excludes_upper_edge():
    ids = assign_edge_bins(
        np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0]), np.asarray([0.0, 1.0, 2.0]),
    )
    assert ids.tolist() == [-1, 0, 0, 1, -1]
