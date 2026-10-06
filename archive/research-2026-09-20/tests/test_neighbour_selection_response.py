import numpy as np

from scripts.audit_neighbour_selection_response import decomposition_sums, response_terms


def test_exact_decomposition_separates_boundary_and_paired_shape_response():
    midpoint = np.array([[-0.2, 0.1], [0.2, -0.1]])
    plus = np.column_stack((midpoint + [0.002, 0], [2.5, 4.0], [100, 100]))
    minus = np.column_stack((midpoint - [0.002, 0], [4.0, 2.5], [100, 100]))
    sums = decomposition_sums(plus, minus)
    uncut, selected = response_terms(sums[0]), response_terms(sums[1])
    np.testing.assert_allclose(uncut["boundary"], 0, atol=1e-14)
    np.testing.assert_allclose(uncut["total"], [0.1, 0], atol=1e-14)
    np.testing.assert_allclose(selected["boundary"], [10, -5])
    np.testing.assert_allclose(selected["within"], [0.1, 0], atol=1e-14)
    np.testing.assert_allclose(selected["total"], [10.1, -5])


def test_stratum_weight_changes_sums_but_not_single_stratum_response():
    plus = np.array([[0.11, 0, 4, 100], [0.21, 0.1, 4, 100]])
    minus = plus.copy()
    minus[:, 0] -= 0.02
    ordinary = decomposition_sums(plus, minus)
    weighted = decomposition_sums(plus, minus, weight=7.0)
    assert weighted[0]["plus"]["weight"] == 7 * ordinary[0]["plus"]["weight"]
    for a, b in zip(ordinary, weighted):
        for name, value in response_terms(a).items():
            np.testing.assert_allclose(value, response_terms(b)[name], atol=1e-13)
