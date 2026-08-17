import numpy as np

from scripts.build_anchorblend_response import project_anchor_response


def test_project_anchor_response_parallel_and_null():
    g = 0.05
    u1 = np.array([1.0, 0.0])
    u2 = np.array([0.0, 1.0])
    de1 = 2 * g * np.array([3.0, -2.0])
    de2 = 2 * g * np.array([4.0, 5.0])
    truth, null = project_anchor_response(de1, de2, u1, u2, g)
    np.testing.assert_allclose(truth, [3.0, 5.0])
    np.testing.assert_allclose(null, [4.0, 2.0])
