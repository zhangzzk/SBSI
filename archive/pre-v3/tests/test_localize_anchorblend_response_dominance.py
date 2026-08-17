import numpy as np


def test_absolute_response_dominance_uses_no_sign_cancellation():
    response = np.array([0.8, 0.1, -0.1])
    fraction = np.max(np.abs(response)) / np.sum(np.abs(response))
    np.testing.assert_allclose(fraction, 0.8)
