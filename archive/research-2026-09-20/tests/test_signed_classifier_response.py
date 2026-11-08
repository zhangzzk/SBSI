import itertools

import numpy as np
import pytest

from scripts.audit_signed_classifier_response import event_moment


def test_independent_sign_average_recovers_self_and_neighbour_moments():
    jacobian = np.array([[.2, -.3], [.4, -.1]])
    basis = np.array([[1., .3], [1., -.2]])
    h = .02
    probes = []
    for values in itertools.product((-1, 1), repeat=2):
        signs = np.array(values)
        delta = h*jacobian@signs
        probes.append(event_moment(.5+delta, .5-delta, basis, h, signs))
    expected_self = np.mean(np.diag(jacobian)[:, None]*basis, axis=0)
    np.testing.assert_allclose(np.mean(probes, axis=0), expected_self, atol=1e-14)
    delta = h*jacobian@np.ones(2)
    coherent = event_moment(.5+delta, .5-delta, basis, h)
    expected_neighbour = np.mean((jacobian.sum(1)-np.diag(jacobian))[:, None]*basis, axis=0)
    np.testing.assert_allclose(coherent-expected_self, expected_neighbour, atol=1e-14)


def test_binary_entering_and_leaving_events_are_retained():
    basis = np.array([[1., .3], [1., -.2]])
    np.testing.assert_allclose(event_moment([1, 0], [0, 1], basis, .02), [0., 6.25])


def test_reject_invalid_probabilities_and_signs():
    with pytest.raises(ValueError, match='probabilities'):
        event_moment([1.1], [0.], [[1.]], .02)
    with pytest.raises(ValueError, match='Rademacher'):
        event_moment([1.], [0.], [[1.]], .02, [0])
