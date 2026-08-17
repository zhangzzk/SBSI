import numpy as np
import pytest

from scripts.toy_random_half_response import sum_conditional_pair_means, unit


def test_unit_vectors_cover_spin2_circle():
    got = unit(np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]))
    np.testing.assert_allclose(
        got,
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
        atol=1e-15,
    )


def test_sum_conditional_pair_means_handles_random_half_rows():
    active = np.array([
        [True, False, True],
        [False, True, True],
        [True, True, False],
    ])
    labels = np.array([
        [1.0, 99.0, 3.0],
        [99.0, 2.0, 5.0],
        [7.0, 4.0, 99.0],
    ])
    total, means, counts = sum_conditional_pair_means(labels, active)
    np.testing.assert_allclose(means, [4.0, 3.0, 4.0])
    np.testing.assert_array_equal(counts, [2, 2, 2])
    assert total == 11.0


def test_sum_conditional_pair_means_rejects_missing_neighbour():
    with pytest.raises(ValueError, match="every neighbour"):
        sum_conditional_pair_means(
            np.ones((2, 2)), np.array([[True, False], [True, False]]),
        )
