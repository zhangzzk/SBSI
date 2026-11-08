import numpy as np
import pytest
from _script_loader import load_script_module


M = load_script_module('v39_recenter_math.py')


def test_quadratic_recovers_joint_maximum_from_distant_center():
    center = np.array([.011, -.003])
    optimum = np.array([.02, .001])
    information = np.broadcast_to([[300., 70.], [70., 200.]], (4, 2, 2)).copy()
    noise = np.array([[.2, .1], [-.2, -.1], [.1, -.2], [-.1, .2]])
    score = np.einsum('nij,j->ni', information, optimum-center)+noise
    result, _ = M.summarize(center, score, information)
    np.testing.assert_allclose(M.next_center(result), optimum)
    assert not result['small_step']
    settled, _ = M.summarize(optimum, noise, information)
    assert settled['small_step']


def test_paired_identical_estimates_have_zero_difference_uncertainty():
    result, influence = M.summarize([0, 0], [[1., 2.], [2., 1.], [0., 0.]],
                                  np.broadcast_to(np.eye(2), (3, 2, 2)))
    delta = M.paired(result, result, influence, influence)
    np.testing.assert_array_equal(delta['difference'], [0, 0])
    np.testing.assert_array_equal(delta['standard_error'], [0, 0])


def test_indefinite_information_and_unsafe_step_stop_without_repair():
    result, influence = M.summarize([0, 0], np.ones((3, 2)),
                                  np.broadcast_to(np.diag([1., -1.]), (3, 2, 2)))
    assert influence is None
    with pytest.raises(ValueError, match='non-positive'):
        M.next_center(result)
    with pytest.raises(ValueError, match='guard'):
        M.next_center(dict(status='estimated', estimate=[.2, 0], step=[.1, 0]))
