import numpy as np

from _script_loader import load_script_module

PLOT = load_script_module("plot_flow_theta_validation.py")


def test_orientation_fit_recovers_slope_without_dividing_by_zero_shape():
    x = np.tile([-0.5, 0.0, 0.7], 4)
    case = np.repeat(np.arange(4), 3)
    weights = np.random.default_rng(17).multinomial(4, np.full(4, .25), size=30)
    # Include an intercept: an origin-forced fit would fail on asymmetric x.
    y = 2.3*x + 7.0
    value, se, ncase = PLOT.grouped_estimate(x, y, case, weights, slope=True)
    np.testing.assert_allclose(value, 2.3, atol=1e-12)
    assert se < 1e-12
    assert ncase == 4


def test_case_bootstrap_uses_pooled_counts_and_preserves_paired_residual():
    case = np.array([0, 1, 1, 1])
    x = np.array([0., 0., 1., 2.])
    y = np.array([8., 0., 0., 0.])
    weights = np.array([[2, 0], [1, 1], [0, 2]])
    value, se, _ = PLOT.grouped_estimate(x, y, case, weights, slope=False)
    assert value == 2.0  # Galaxy pooled, not the mean of the two case means (4).
    np.testing.assert_allclose(se, np.std([8., 2., 0.], ddof=1))
    value, se, _ = PLOT.grouped_estimate(x, (y+3)-y, case, weights, slope=False)
    assert value == 3.0
    assert se == 0.0
