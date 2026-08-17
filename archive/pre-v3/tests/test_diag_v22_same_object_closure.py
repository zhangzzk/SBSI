import numpy as np

from scripts.diag_v22_same_object_closure import decompose, equality_summary


def test_decomposition_identity():
    rc = np.array([0.7, 0.8])
    rh = np.array([0.69, 0.79])
    rb = np.array([0.1, 0.2])
    rs = np.array([0.82, 1.03])
    sh = np.array([0.71, 0.81])
    terms = decompose(rc, rh, rb, rs, sh)
    np.testing.assert_allclose(terms["identity_error"], 0.0, atol=1e-15)


def test_equality_summary_reports_exact_and_numeric_difference():
    row = equality_summary(np.array([1.0, 2.0]), np.array([1.0, 2.5]))
    assert row["exact_equal_fraction"] == 0.5
    assert row["mean_abs_difference"] == 0.25
    assert row["max_abs_difference"] == 0.5
