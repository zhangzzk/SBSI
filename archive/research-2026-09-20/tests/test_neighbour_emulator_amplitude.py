import numpy as np
import pandas as pd
import pytest

from scripts.audit_neighbour_emulator_amplitude import aggregate_predictions, add_predictions
from scripts.audit_neighbour_selection_response import decomposition_sums


def example():
    truth = pd.DataFrame({"index": [10, 20, 30, 40], "gamma1": [0, 0, .02, 0],
                          "gamma2": [0]*4, "axis_ratio": [.5, 1, 1, 1], "position_angle": [0]*4})
    pairs = pd.DataFrame({"index_input_p": [10, 10], "index_input_s": [20, 30], "response": [.3, .2]})
    return truth, pairs


def test_stationary_neighbour_mask_preserves_existing_pairs_and_isolated_rows():
    truth, pairs = example()
    result, counts = aggregate_predictions(pairs, truth, [40, 10])
    np.testing.assert_allclose(result["native_full_scene"], [[0, 0], [.5, 0]])
    np.testing.assert_allclose(result["native_sheared_neighbours"], [[0, 0], [.2, 0]])
    expected = .2 * (1 - (1/3)**2) / (1 - .02**2 * (1/3)**2)
    np.testing.assert_allclose(result["primary_shape_proxy"], [[0, 0], [expected, 0]])
    assert counts["stationary_secondary_pairs"] == 1
    assert counts["isolated_primaries"] == 1


def test_prediction_uses_same_per_leg_cut_and_stratum_weight():
    truth, pairs = example()
    predictions, _ = aggregate_predictions(pairs, truth, [40, 10])
    plus = np.array([[.1, 0, 2, 100], [.2, 0, 4, 100]])
    minus = plus.copy()
    minus[:, 2] = [4, 2]
    sums = decomposition_sums(plus, minus, weight=7)
    add_predictions(sums, plus, minus, predictions, 7)
    np.testing.assert_allclose(sums[1]["plus"]["native_sheared_neighbours"], [1.4, 0])
    np.testing.assert_allclose(sums[1]["minus"]["native_sheared_neighbours"], [0, 0])


def test_duplicate_pairs_rejected():
    truth, pairs = example()
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_predictions(pd.concat([pairs, pairs]), truth, [10])
