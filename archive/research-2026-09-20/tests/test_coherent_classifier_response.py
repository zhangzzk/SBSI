import numpy as np
import pytest

from scripts.audit_coherent_classifier_response import sufficient, response_summary


def example():
    context = np.zeros((4, 8))
    context[:, :2] = [[.1, .2], [.2, -.1], [-.3, .4], [.4, -.2]]
    context[:, 3] = [23., 24.5, 25.5, 25.9]
    minus = {'p': np.array([0., 1., 1., 1.]), 'u': np.array([0, 1, 1, 1]), 'd': np.ones(4)}
    plus = {'p': np.array([1., 0., 1., 0.]), 'u': np.array([1, 0, 1, 0]), 'd': np.ones(4)}
    return context, plus, minus


def test_enter_leave_events_and_truth_shape_derivative():
    context, plus, minus = example()
    row = sufficient(context, plus, minus, .02)
    result = response_summary([row, row])
    expected = np.column_stack((np.ones(4), context[:, :2])).T@(plus['u']-minus['u'])/(.04*4)
    np.testing.assert_allclose(result['response']['estimate'][0, 1], expected)
    np.testing.assert_allclose(result['prediction_minus_U']['estimate'], 0.)
    np.testing.assert_allclose(result['U_minus_D']['estimate'][0], expected)
    np.testing.assert_array_equal(row['n'], [4, 1, 1, 1, 1])


def test_pooled_parent_normalization_and_single_case_uncertainty():
    context, plus, minus = example()
    first = sufficient(context, plus, minus, .02)
    zero = sufficient(context, minus, minus, .02)
    zero = {k: 3*v for k, v in zero.items()}
    result = response_summary([first, zero])
    np.testing.assert_allclose(result['response']['estimate'], first['response_sum']/first['n'][:, None, None]/4)
    assert response_summary([first])['response']['case_bootstrap_se'] is None


def test_reject_invalid_events_and_parent_cut():
    context, plus, minus = example()
    plus['d'][0] = 0
    with pytest.raises(ValueError, match='implies'):
        sufficient(context, plus, minus, .02)
    context[0, 3] = 26
    with pytest.raises(ValueError, match='strict'):
        sufficient(context, minus, minus, .02)
