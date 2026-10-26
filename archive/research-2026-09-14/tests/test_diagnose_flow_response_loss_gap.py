import numpy as np

from scripts.diagnose_flow_response_loss_gap import loss_decomposition


def test_loss_decomposition_closes_with_weighting_and_cross_term():
    model = np.array(
        [
            [[1.0, 0.2], [-0.1, 0.8]],
            [[0.4, -0.2], [0.3, 1.2]],
        ]
    )
    target = np.array(
        [
            [[0.9, 0.0], [0.0, 0.9]],
            [[0.5, -0.1], [0.1, 1.0]],
        ]
    )
    simulation = np.array(
        [
            [[0.7, 0.1], [0.0, 1.0]],
            [[0.6, 0.0], [0.2, 0.7]],
        ]
    )
    result = loss_decomposition(model, target, simulation, np.array([20, 40]))
    assert result["cells"] == 2
    assert result["rows"] == 60
    assert abs(result["closure_error"]) < 1.0e-14
    assert result["model_vs_simulation"] > 0.0


def test_loss_decomposition_respects_minimum_count():
    matrices = np.zeros((2, 2, 2))
    simulation = matrices.copy()
    simulation[0, 0, 0] = 10.0
    result = loss_decomposition(
        matrices, matrices, simulation, np.array([19, 20]), min_count=20
    )
    assert result["cells"] == 1
    assert result["model_vs_simulation"] == 0.0
