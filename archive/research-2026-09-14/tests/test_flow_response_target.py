import numpy as np

from sbsi.flow_response_target import estimate_response_matrix


def test_response_matrix_estimator_recovers_cross_terms():
    angle = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    applied = 0.05 * np.column_stack([np.cos(angle), np.sin(angle)])
    truth = np.array([[0.82, 0.03], [-0.02, 0.79]])
    delta = applied @ truth.T

    np.testing.assert_allclose(
        estimate_response_matrix(delta, applied), truth, atol=1.0e-12
    )


def test_response_matrix_estimator_rejects_degenerate_shear_directions():
    applied = np.column_stack([np.full(20, 0.05), np.zeros(20)])
    delta = applied.copy()

    try:
        estimate_response_matrix(delta, applied)
    except ValueError as error:
        assert "do not span" in str(error)
    else:
        raise AssertionError("expected a degenerate shear design to be rejected")
