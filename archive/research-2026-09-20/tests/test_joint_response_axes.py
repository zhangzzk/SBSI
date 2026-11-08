import numpy as np
import pytest

from plots.constgold_property_bias import make_case_weights, summarize_branch
from scripts.evaluate_constgold_fixed_g0_response import shear_frames as old_frames
from scripts.joint_response_axes import shear_frames, shear_vector, summarize_axis_branch


def test_g1_contexts_remain_bitwise_identical_and_g2_matches_complex_map():
    context = np.array([[.13, -.27, 1., 25., 1., 0., 0., 0.],
                        [-.1, .07, 1., 24., 1., 0., 0., 0.]])
    for actual, expected in zip(shear_frames(context, .02, 1), old_frames(context, .02)):
        np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
    for sign, frame in zip((1, -1), shear_frames(context, .02, 2)):
        e = context[:, 0]+1j*context[:, 1]
        g = sign*.02j
        expected = (e+g)/(1+np.conj(g)*e)
        np.testing.assert_allclose(frame.iloc[:, 0]+1j*frame.iloc[:, 1], expected, atol=1e-15)
        np.testing.assert_array_equal(frame.iloc[:, 2:], context[:, 2:])


def synthetic_stats():
    result = {}
    for source, response in [('measured', [.1, .8]), ('model', [.2, .7])]:
        result[source] = {}
        for direction, sign in [('plus', 1), ('minus', -1)]:
            denominator = np.array([[30.], [40.], [50.]])
            means = np.array([[.3, -.1]])+sign*.02*np.array([response])
            result[source][direction] = {'denominator': denominator,
                                         'numerator': denominator[..., None]*means}
    return result


def test_bias_uses_correct_diagonal_and_vectors_keep_sky_order():
    stats = synthetic_stats()
    weights = make_case_weights(20, 3, 100)
    expected = summarize_branch(stats, weights, .02)
    g1 = summarize_axis_branch(stats, weights, .02, 1)
    for key in expected:
        np.testing.assert_array_equal(g1[key], expected[key])
    g2 = summarize_axis_branch(stats, weights, .02, 2)
    np.testing.assert_allclose(g2['measured_response'], [[.1, .8]])
    np.testing.assert_allclose(g2['model_response'], [[.2, .7]])
    np.testing.assert_allclose(g1['m_percent'], [-50.])
    np.testing.assert_allclose(g2['m_percent'], [100*(.8/.7-1)])
    pilot = summarize_axis_branch(stats, None, .02, 2)
    np.testing.assert_array_equal(pilot['m_percent'], g2['m_percent'])
    assert np.isnan(pilot['m_ci95_percent']).all()


@pytest.mark.parametrize('h,axis', [(np.nan, 1), (1., 1), (.02, 0), (.02, 3)])
def test_invalid_shear_rejected(h, axis):
    with pytest.raises(ValueError):
        shear_vector(h, axis)
