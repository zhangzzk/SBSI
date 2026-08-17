import numpy as np

from scripts.audit_v22_conditional_target_stability import (
    assign_conditional_cells,
    holm_rejections,
)


def test_holm_stops_after_first_non_rejection():
    got = holm_rejections(np.array([1e-6, 0.03, 0.04]), alpha=0.05)
    np.testing.assert_array_equal(got, [True, False, False])


def test_assign_conditional_cells_uses_primary_specific_edges():
    mag_edges = np.array([0.0, 1.0, 2.0])
    size_edges = np.array([0.0, 1.0])
    crowd_edges = np.array([[[-1.0, 0.0, 1.0]], [[-1.0, 0.5, 1.0]]])
    got = assign_conditional_cells(
        np.array([0.5, 1.5]), np.array([0.5, 0.5]), np.array([0.25, 0.25]),
        mag_edges, size_edges, crowd_edges,
    )
    np.testing.assert_array_equal(got, [1, 2])
