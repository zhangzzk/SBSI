import numpy as np
import pytest
from _script_loader import load_script_module


M = load_script_module('v39_bright_math.py')


def test_carry_updates_nonquadratic_normalizer_and_full_cross_component():
    """Analytic numerator and cubic log-normalizer provide an independent oracle."""
    center = np.array([.02, -.01])
    target = np.array([.018, .004])
    numerator_information = np.array([[[8., 3.], [3., 5.]], [[9., -2.], [-2., 6.]]])
    offset = np.array([[.03, -.01], [.04, .02]])

    def numerator_score(point):
        return offset-np.einsum('nij,j->ni', numerator_information, point)

    def norm(point):
        x, y = point
        return np.array([6*x*x+y, x+3*y*y]), np.array([[12*x, 1.], [1., 6*y]])

    old_g, old_h = norm(center)
    new_g, new_h = norm(target)
    score, info = M.carry(numerator_score(center)-old_g, numerator_information+old_h,
                          target-center, old_g, old_h, new_g, new_h)
    np.testing.assert_allclose(score, numerator_score(target)-new_g, atol=1.e-14)
    np.testing.assert_allclose(info, numerator_information+new_h, atol=1.e-14)
    # A later target must still carry from the same original reference.
    final = np.array([.019, .001])
    g, h = norm(final)
    score, info = M.carry(numerator_score(center)-old_g, numerator_information+old_h,
                          final-center, old_g, old_h, g, h)
    np.testing.assert_allclose(score, numerator_score(final)-g, atol=1.e-14)
    np.testing.assert_allclose(info, numerator_information+h, atol=1.e-14)


def test_carry_rejects_diagonal_only_or_misaligned_information():
    with pytest.raises(ValueError, match='aligned'):
        M.carry(np.zeros((3, 2)), np.zeros((3, 2)), [0, 0], [0, 0], np.eye(2), [0, 0], np.eye(2))
