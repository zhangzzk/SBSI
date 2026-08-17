import numpy as np
import pandas as pd

from scripts.analyze_v22_validation_tail_scenes import (
    aggregate_pair_batch,
    combine_scene_partials,
    selection_mask,
    summarize_selection,
)


def toy_pairs() -> pd.DataFrame:
    return pd.DataFrame({
        "case": [40, 40, 40, 41, 41, 41],
        "input_index": [1, 1, 2, 1, 1, 2],
        "r_input_p": [24.2, 24.2, 23.0, 24.7, 24.7, 23.5],
        "Re_input_p": [0.7, 0.7, 0.8, 0.6, 0.6, 0.9],
        "r_input_s": [24.0, 25.0, 26.0, 23.0, 26.0, 27.0],
        "distance": [0.8, 1.4, 2.0, 0.6, 1.8, 2.5],
        "delta_et1": [0.04, 0.02, 0.01, 0.05, 0.03, -0.01],
        "delta_et2": [0.00, 0.01, 0.00, -0.01, 0.00, 0.01],
    })


def test_scene_partial_combination_preserves_pair_sums():
    pairs = toy_pairs()
    prediction = np.array([0.08, 0.05, 0.01, 0.09, 0.04, -0.02])
    validation = np.array([True, False, True, True, False, True])
    first = aggregate_pair_batch(
        pairs.iloc[[0, 2, 3, 5]], prediction[[0, 2, 3, 5]],
        validation[[0, 2, 3, 5]], 0.2,
    )
    second = aggregate_pair_batch(
        pairs.iloc[[1, 4]], prediction[[1, 4]], validation[[1, 4]], 0.2,
    )
    scenes = combine_scene_partials([first, second]).set_index(
        ["case", "input_index"]
    )
    assert len(scenes) == 4
    assert scenes.loc[(40, 1), "n_pairs"] == 2
    assert np.isclose(scenes.loc[(40, 1), "scene_prediction"], 0.13)
    assert scenes.loc[(40, 1), "validation_pair_count"] == 1
    expected = 0.04 / 0.2 - 0.08
    assert np.isclose(
        scenes.loc[(40, 1), "validation_residual_sum"], expected
    )


def test_tail_selection_and_inverse_probability_scene_residual():
    pairs = toy_pairs()
    prediction = np.array([0.08, 0.05, 0.01, 0.09, 0.04, -0.02])
    validation = np.array([True, False, True, True, False, True])
    scenes = combine_scene_partials([
        aggregate_pair_batch(pairs, prediction, validation, 0.2)
    ])
    keep = selection_mask(scenes, response_threshold=0.1, primary_mag_min=24.0)
    assert keep.sum() == 2
    summary, cases = summarize_selection(
        scenes,
        keep,
        name="tail",
        response_threshold=0.1,
        primary_mag_min=24.0,
        case_min=40,
        n_cases=2,
        validation_scale=2.0,
        eligible_primary_count=np.array([2, 2]),
    )
    selected = scenes.loc[keep].sort_values("case")
    expected = 2.0 * selected.validation_residual_sum.to_numpy(float)
    assert np.allclose(
        cases.residual_per_selected_primary.to_numpy(float), expected
    )
    assert np.isclose(
        summary["scene_scale"][
            "label_minus_prediction_per_selected_primary"
        ]["mean"],
        expected.mean(),
    )
    assert np.isclose(
        summary["scene_scale"][
            "residual_contribution_per_all_eligible_primary"
        ]["mean"],
        0.5 * expected.mean(),
    )
