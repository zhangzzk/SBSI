import numpy as np

from scripts.audit_response_label_tensor import directional_labels


def test_random_direction_tensor_recovers_full_matrix_and_scalar_trace():
    angles = np.arange(360, dtype=float)/2
    u = np.column_stack((np.cos(np.deg2rad(2*angles)), np.sin(np.deg2rad(2*angles))))
    response = np.array([[.2, -.03], [.07, .1]])
    delta = u @ response.T
    parallel = (delta*u).sum(1)
    cross = -delta[:, 0]*u[:, 1] + delta[:, 1]*u[:, 0]
    labels, directions = directional_labels(parallel, cross, angles)
    np.testing.assert_allclose(directions, u, atol=1e-14)
    np.testing.assert_allclose(labels.mean(0), response, atol=1e-14)
    np.testing.assert_allclose(np.trace(labels, axis1=1, axis2=2)/2, parallel, atol=1e-14)
