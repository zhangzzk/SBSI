import numpy as np
import pandas as pd

from scripts.analyze_localized_anchor_noise_repeat_toys import (
    draw_stat,
    scene_summary,
)
from scripts.prepare_localized_anchor_noise_repeat_toys import (
    SELECTION_FEATURES,
    rank_design,
    select_typical_scenes,
)
from scripts.run_localized_anchor_noise_repeat_toy_scene import repeat_seed


def synthetic_tail(n_per_stratum=40):
    rows = []
    rng = np.random.RandomState(12)
    for compact in (True, False):
        for index in range(n_per_stratum):
            row = {
                "case": 700 + len(rows), "input_index": 10_000 + len(rows),
                "compact_dominant_secondary": compact,
                "dominant_secondary_size": 0.3 if compact else 0.7,
                "n_deployed_pairs": 8 + index % 9,
                "bias_truth_minus_model": 1e6 * (-1) ** index,
            }
            for feature in SELECTION_FEATURES:
                row[feature] = float(rng.normal() + 0.03 * index)
            row["scene_prediction"] = 0.5 + 0.001 * index
            rows.append(row)
    return pd.DataFrame(rows)


def test_rank_design_excludes_marginal_extreme():
    frame = pd.DataFrame({"x": np.arange(100), "y": np.arange(100)})
    selected, design = rank_design(frame, ["x", "y"], 0.02)
    assert 0 not in selected.x.to_numpy()
    assert 99 not in selected.x.to_numpy()
    assert design.shape == (96, 2)


def test_typical_selection_is_balanced_case_unique_and_outcome_blind():
    frame = synthetic_tail()
    first, _ = select_typical_scenes(
        frame, n_per_stratum=3, central_quantile=0.0, seed=44,
    )
    changed = frame.copy()
    changed["bias_truth_minus_model"] *= -123.0
    second, _ = select_typical_scenes(
        changed, n_per_stratum=3, central_quantile=0.0, seed=44,
    )
    assert first[["case", "input_index"]].equals(second[["case", "input_index"]])
    assert first.selection_stratum.value_counts().to_dict() == {
        "compact": 3, "noncompact": 3,
    }
    assert not first.case.duplicated().any()


def test_repeat_seeds_are_deterministic_and_scene_specific():
    assert repeat_seed(700, 12, 4, 17) == repeat_seed(700, 12, 4, 17)
    assert repeat_seed(700, 12, 4, 17) != repeat_seed(701, 12, 4, 17)
    assert repeat_seed(700, 12, 4, 17) != repeat_seed(700, 12, 5, 17)
    assert repeat_seed(700, 12, 4, 17) != repeat_seed(700, 12, 4, 18)


def test_draw_stat_and_scene_summary_respect_technical_replication():
    values = np.array([-0.01, 0.00, 0.01, 0.00])
    stat = draw_stat(values)
    assert stat["n_draws"] == 4
    np.testing.assert_allclose(stat["mean"], 0.0)
    draws = pd.DataFrame({
        "success": [True] * 4,
        "R11_coherent": 0.6 + values,
        "R11_individual_sum": np.full(4, 0.6),
        "additivity_gap_g1": values,
        "coherent_truth_minus_model_g1": 0.1 + values,
        "individual_truth_minus_model_g1": np.full(4, 0.1),
        "R_trace_coherent": 0.6 + values,
        "R_trace_individual_sum": np.full(4, 0.6),
        "additivity_gap_trace": values,
        "coherent_truth_minus_model_trace": 0.1 + values,
        "individual_truth_minus_model_trace": np.full(4, 0.1),
    })
    row = pd.Series({
        "scene_id": 0, "case": 700, "input_index": 1,
        "selection_stratum": "compact", "compact_dominant_secondary": True,
        "dominant_secondary_size": 0.3, "n_deployed_pairs": 8,
        "scene_prediction": 0.5,
    })
    summary, _ = scene_summary(draws, row, sesoi=0.02, target_halfwidth=0.02)
    assert summary["n_success"] == 4
    np.testing.assert_allclose(summary["additivity_gap_trace"], 0.0)
