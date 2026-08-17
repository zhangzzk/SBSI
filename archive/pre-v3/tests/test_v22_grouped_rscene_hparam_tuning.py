import numpy as np

from scripts.evaluate_v22_grouped_rscene_hparam_candidates import select_candidate
from scripts.train_v22_grouped_rscene_hparam_candidate import residual_summary
from scripts.v22_grouped_rscene_hparam_common import (
    CANDIDATES,
    candidate_by_id,
    prediction_edges,
    select_case_rows,
    xgb_params,
)


def test_candidate_grid_is_unique_and_keeps_exact_baseline_first() -> None:
    names = [item["name"] for item in CANDIDATES]
    assert len(names) == 18
    assert len(set(names)) == len(names)
    baseline = candidate_by_id(0)
    assert baseline["max_depth"] == 5
    assert baseline["min_child_weight"] == 2000.0
    assert baseline["reg_lambda"] == 10.0
    assert baseline["learning_rate"] == 0.05
    params = xgb_params(baseline)
    assert params["base_score"] == 0.0
    assert params["objective"] == "reg:squarederror"


def test_case_selection_keeps_groups_disjoint_and_optional_row_mask() -> None:
    case = np.repeat(np.arange(4), 3)
    official_train = np.tile([True, False, False], 4)
    heldout = select_case_rows(
        case, official_train, (1, 2), official_validation_only=True
    )
    np.testing.assert_array_equal(heldout, [4, 5, 7, 8])
    all_rows = select_case_rows(
        case, official_train, (1, 2), official_validation_only=False
    )
    np.testing.assert_array_equal(all_rows, [3, 4, 5, 6, 7, 8])


def test_prediction_edges_are_label_free_strict_quantiles() -> None:
    values = np.linspace(-0.2, 0.5, 1001)
    edges = prediction_edges(values, n_bins=10)
    assert len(edges) == 11
    assert np.all(np.diff(edges) > 0)
    np.testing.assert_allclose(edges[[0, -1]], values[[0, -1]])


def test_residual_summary_reports_physical_mse_and_bin_means() -> None:
    before = np.asarray([1.0, 3.0, -2.0, 0.0])
    correction = np.asarray([0.5, 0.5, -1.0, -1.0])
    bins = np.asarray([0, 0, 1, 1])
    got = residual_summary(before, correction, bins)
    after = before - correction
    np.testing.assert_allclose(got["mse_before"], np.mean(before**2))
    np.testing.assert_allclose(got["mse_after"], np.mean(after**2))
    expected_bin_means = np.asarray([after[:2].mean(), after[2:].mean()])
    np.testing.assert_allclose(
        got["scene_pair_mean_rms_after"],
        np.sqrt(np.mean(expected_bin_means**2)),
    )


def test_selection_uses_worst_population_score_with_mse_guardrail() -> None:
    populations = (
        "internal_selection_c160_199",
        "external_development_c0_19",
    )
    candidates = [
        {"id": 0, "name": "a", "summary": {"training": {"best_rounds": 100}}},
        {"id": 1, "name": "b", "summary": {"training": {"best_rounds": 80}}},
        {"id": 2, "name": "c", "summary": {"training": {"best_rounds": 60}}},
    ]
    summary = {population: {} for population in populations}
    scores = {
        "a": (0.50, 0.80, -0.02, -0.03),
        "b": (0.65, 0.65, -0.02, -0.03),
        # Better calibration but ineligible because external pair MSE worsens.
        "c": (0.40, 0.40, -0.02, 0.001),
    }
    for model, (internal_score, external_score, internal_mse, external_mse) in scores.items():
        for population, score, mse in (
            (populations[0], internal_score, internal_mse),
            (populations[1], external_score, external_mse),
        ):
            summary[population][model] = {
                "calibration_score": score,
                "pair_mse_percent_change_from_v22": {"mean": mse},
            }
    got = select_candidate(summary, candidates)
    assert got["selected_candidate_id"] == 1
    assert got["selected_candidate_name"] == "b"
