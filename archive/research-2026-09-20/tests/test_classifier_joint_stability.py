import numpy as np
import pytest

from scripts.summarize_classifier_joint_stability import bias_replicas


def stats(responses, h=.02):
    r = np.asarray(responses)
    return {leg: {'denominator': np.ones((len(r), 1)),
                  'numerator': np.stack((sign*h*r, np.zeros_like(r)), axis=-1)[:, None, :]}
            for leg, sign in (('plus', 1), ('minus', -1))}


def test_pooled_bias_and_all_leave_one_out_cases_are_retained():
    sufficient = {'measured': stats([1, 2]), 'model': stats([2, 1])}
    out = bias_replicas(sufficient, np.array([[1, 1], [0, 1], [1, 0]]), .02, 1)
    np.testing.assert_allclose(out, [0, 100, -50], atol=1e-12)
    # The pooled estimate is not the mean of individual-scene ratios (25%).
    assert out[0] != np.mean(out[1:])


def test_zero_model_response_is_rejected():
    with pytest.raises(ValueError, match='nonzero model response'):
        bias_replicas({'measured': stats([1, 2]), 'model': stats([0, 0])}, np.ones((1, 2)), .02, 1)
