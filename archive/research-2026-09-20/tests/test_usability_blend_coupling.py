import numpy as np

from scripts.analyze_usability_blend_coupling import decompose


def test_order_symmetric_exact_response_decomposition():
    na = np.array([[[2., 4.], [1., 3.]]])
    npred = np.array([[[3., 5.], [1., 4.]]])
    da, dp = np.array([[4., 5.]]), np.array([[5., 6.]])
    x = decompose(npred, dp, na, da, .02)
    np.testing.assert_allclose(x['difference'], x['numerator_attribution']+x['denominator_attribution'])
    reverse = decompose(na, da, npred, dp, .02)
    for name in ['difference', 'numerator_attribution', 'denominator_attribution']:
        np.testing.assert_allclose(x[name], -reverse[name])


def test_equal_masses_isolate_numerator_and_components_add():
    na = np.array([[[2., 4.], [1., 3.]]])
    npred = 1.1*na
    d = np.array([[4., 5.]])
    x = decompose(npred, d, na, d, .02)
    np.testing.assert_allclose(x['denominator_attribution'], 0.)
    pieces = [decompose(f*npred, d, f*na, d, .02) for f in [.3, .7]]
    for key in x:
        np.testing.assert_allclose(x[key], pieces[0][key]+pieces[1][key])
