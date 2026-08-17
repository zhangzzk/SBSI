import numpy as np
import pytest

from scripts.compute_response_target_antithetic_self import (
    central_response,
    exact_match,
    packed_key,
)


def test_central_response_recovers_known_linear_response():
    g1 = np.array([0.02, 0.0, -0.012])
    g2 = np.array([0.0, 0.02, 0.016])
    response = np.array([0.7, 0.9, 1.1])
    e01 = np.array([0.2, -0.1, 0.05])
    e02 = np.array([-0.3, 0.4, 0.1])
    ep1, ep2 = e01 + response * g1, e02 + response * g2
    em1, em2 = e01 - response * g1, e02 - response * g2

    got, gmag, guard = central_response(
        ep1, ep2, em1, em2, g1, g2, -g1, -g2
    )
    np.testing.assert_allclose(got, response, rtol=0, atol=1e-13)
    np.testing.assert_allclose(gmag, 0.02)
    assert guard == 0.0


def test_central_response_rejects_non_antithetic_shears():
    with pytest.raises(RuntimeError, match="not antithetic"):
        central_response(
            [0.1], [0.2], [0.0], [0.0],
            [0.02], [0.0], [-0.019], [0.0],
        )


def test_exact_match_preserves_reference_order():
    reference = packed_key([2, 1, 3], [20, 10, 30])
    candidate = packed_key([3, 2], [30, 20])
    hit, pos = exact_match(reference, candidate)
    np.testing.assert_array_equal(hit, [True, False, True])
    np.testing.assert_array_equal(candidate[pos[hit]], reference[hit])


def test_exact_match_rejects_duplicate_candidate_keys():
    reference = packed_key([1], [10])
    candidate = packed_key([1, 1], [10, 10])
    with pytest.raises(RuntimeError, match="duplicate"):
        exact_match(reference, candidate)
