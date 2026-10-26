"""Focused tests for the training-radius residual plot."""

from __future__ import annotations

import numpy as np

from plots.flow_training_residual_vs_flux_radius import (
    case_mean_and_sem,
    radius_edges_pixels,
)


def test_radius_edges_resolve_selection_boundary_at_quarter_tenth_pixel():
    edges = radius_edges_pixels()
    np.testing.assert_allclose(edges[:5], [3.0, 3.025, 3.05, 3.075, 3.1])
    assert np.isinf(edges[-1])
    assert np.all(np.diff(edges) > 0)


def test_case_mean_and_sem_uses_paired_finite_cases():
    values = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    means, sems, counts = case_mean_and_sem(values)
    np.testing.assert_allclose(means, [3.0, 3.0])
    np.testing.assert_allclose(sems, [2.0 / np.sqrt(3.0), 1.0])
    np.testing.assert_array_equal(counts, [3, 2])
