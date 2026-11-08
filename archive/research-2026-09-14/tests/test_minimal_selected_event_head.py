import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/train_minimal_selected_event_head.py"
SPEC = importlib.util.spec_from_file_location("minimal_selected_event", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_probability_metrics_matches_manual_binary_log_loss():
    probability = np.asarray([[0.2, 0.8], [0.7, 0.1]])
    label = np.asarray([[False, True], [True, False]])
    result = MODULE.probability_metrics(probability, label)
    expected = -np.mean(np.log([0.8, 0.8, 0.7, 0.9]))
    assert result["bce"] == pytest.approx(expected)
    assert result["event_fraction"] == 0.5


def test_probability_metrics_rejects_unresolved_qmc_zero_for_observed_event():
    with pytest.raises(ValueError, match="impossible observed event"):
        MODULE.probability_metrics(np.asarray([0.0, 0.2]), np.asarray([True, False]))


def test_flow_cut_is_magnitude_radius_without_shape():
    samples = np.zeros((2, 3, 4))
    samples[..., :2] = 0.99  # deliberately outside the retired measured-|e| cut
    samples[..., 2] = [[3.75, 3.74, 4.0], [4.0, 4.0, 4.0]]
    samples[..., 3] = [[100.0, 100.0, 1.0], [100.0, 1.0, 100.0]]
    assert MODULE._flow_pass_probability(samples).tolist() == [
        [True, False, False],
        [True, False, True],
    ]


def test_minimal_architecture_has_4865_parameters():
    model = MODULE.SelectionMLP(
        input_dim=9, hidden_dim=64, n_layers=2, dropout=0.0, activation="silu"
    )
    assert sum(value.numel() for value in model.parameters()) == 4865
