import numpy as np

from scripts.audit_v36_cut_transfer import decompose, pooled_response


def test_cut_decomposition_includes_selection_migration():
    # Each region has a shear-independent mean, but its selected weight changes.
    high = np.array([[2., 1., -.2], [4., 2., -.4]])
    shell = np.array([[8., -4., .8], [6., -3., .6]])
    total = high+shell
    result = decompose(total, high, .02)
    np.testing.assert_allclose(result['high_response'], [0., 0.])
    np.testing.assert_allclose(result['shell_response'], [0., 0.], atol=1e-14)
    np.testing.assert_allclose(result['selection_migration_contribution'], pooled_response(total, .02))
    assert np.linalg.norm(result['selection_migration_contribution']) > 0
