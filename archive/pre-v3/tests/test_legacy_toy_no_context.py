import numpy as np

from scripts.run_legacy_toy_no_context import (
    assemble_draws,
    closepair_configs,
    sweep_configs,
)
from scripts.analyze_legacy_toy_no_context import (
    CLOSEPAIR_REFERENCE,
    SWEEP_REFERENCE,
)


def test_exact_legacy_configuration_grids():
    sweep = sweep_configs()
    close = closepair_configs()
    assert len(sweep) == 12
    assert [item["config_id"] for item in sweep] == list(range(12))
    assert [item["legacy_nreal"] for item in sweep] == [300] * 12
    assert [len(item["neighbours"]) for item in sweep] == [1, 2, 4, 4] * 3
    assert [item["target"]["flux"] for item in sweep] == [800] * 4 + [2000] * 4 + [500] * 4
    assert len(close) == 26
    assert [item["config_id"] for item in close] == list(range(26))
    assert [item["legacy_nreal"] for item in close] == [400] * 26
    assert [len(item["neighbours"]) for item in close[-8:]] == [1, 2, 4, 8] * 2
    assert len(SWEEP_REFERENCE) == len(sweep)
    assert len(CLOSEPAIR_REFERENCE) == len(close)


def test_three_closures_and_context_deltas_are_exact():
    self_context = np.array([0.3, 0.4, 0.5])
    neighbour_context = [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]
    self_isolated = np.array([0.4, 0.5, 0.6])
    neighbour_pairs = [np.array([0.05, 0.1, 0.15]), np.array([0.1, 0.2, 0.3])]
    full = np.array([0.8, 1.0, 1.2])
    frame = assemble_draws(
        self_context, neighbour_context, self_isolated, neighbour_pairs, full
    )
    np.testing.assert_allclose(
        frame.excess_pair_contextself,
        frame.excess_context + frame.delta_neighbour_context_pair,
    )
    np.testing.assert_allclose(
        frame.excess_isolated,
        frame.excess_pair_contextself + frame.delta_self_context_isolated,
    )
    np.testing.assert_allclose(frame.R_neighbour_context_sum, [0.3, 0.5, 0.7])
    np.testing.assert_allclose(frame.R_neighbour_pair_sum, [0.15, 0.3, 0.45])
