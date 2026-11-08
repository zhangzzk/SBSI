import numpy as np

from scripts.audit_measurement_symmetry_pilot import summarize_case


def test_exact_transformed_back_shapes_have_zero_marginal_difference():
    original = np.array([[.1, .2], [-.2, .1]])
    shapes = np.broadcast_to(original[:, None, None, :], (2, 4, 2, 2)).copy()
    result = summarize_case(shapes, [[4., 25.], [5., 24.]])
    for rows in result.values():
        for summary in rows.values():
            np.testing.assert_array_equal(summary['moment_difference'], 0)
            assert summary['original_selected_usable'] == summary['candidate_selected_usable'] == 2


def test_transformed_fit_failure_does_not_remove_identity_measurement():
    shapes = np.broadcast_to(np.array([[.1, .2], [-.2, .1]])[:, None, None, :], (2, 4, 2, 2)).copy()
    shapes[0, 1, 0] = np.nan
    result = summarize_case(shapes, [[4., 25.], [5., 24.]])
    row = result['mag25.8_radius_gt0.60']['flip_x/stream0']
    assert row['original_selected_usable'] == 2
    assert row['candidate_selected_usable'] == 1
    assert row['original_only'] == 1
    np.testing.assert_allclose(row['candidate_moments'][:2], [-.2, .1])
