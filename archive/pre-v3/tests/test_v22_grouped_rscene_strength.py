import json

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_v22_grouped_rscene_hparam_final import load_strength_selection
from scripts.evaluate_v22_grouped_rscene_strength import (
    POPULATIONS,
    build_strength_metrics,
    mse_at_strength,
    residual_at_strength,
    select_strength,
)


def test_residual_and_mse_moments_match_direct_evaluation() -> None:
    residual = np.asarray([1.0, -2.0, 0.5, 3.0])
    correction = np.asarray([0.2, -0.4, 0.7, 0.1])
    strength = 1.3
    direct = residual - strength * correction
    np.testing.assert_allclose(
        residual_at_strength(residual, correction, strength), direct
    )
    got = mse_at_strength(
        np.mean(residual**2),
        np.mean(residual * correction),
        np.mean(correction**2),
        strength,
    )
    np.testing.assert_allclose(got, np.mean(direct**2))


def _toy_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_rows = []
    pair_bins = []
    scene_bins = []
    for population_index, population in enumerate(POPULATIONS):
        for case in (0, 1):
            residual = np.asarray([0.4, -0.2]) + 0.02 * population_index
            correction = np.asarray([0.3, -0.1]) + 0.01 * case
            case_rows.append({
                "population": population,
                "case": case,
                "selection": "pair",
                "baseline_residual": float(residual.mean()),
                "correction": float(correction.mean()),
                "baseline_mse": float(np.mean(residual**2)),
                "residual_correction_moment": float(np.mean(residual * correction)),
                "correction_square_moment": float(np.mean(correction**2)),
            })
            for selection, multiplier in (
                ("all", 2.0),
                ("tail_gt_0p1", 3.0),
                ("tail_gt_0p2", 4.0),
            ):
                case_rows.append({
                    "population": population,
                    "case": case,
                    "selection": selection,
                    "baseline_residual": multiplier * float(residual.mean()),
                    "correction": multiplier * float(correction.mean()),
                    "baseline_mse": np.nan,
                    "residual_correction_moment": np.nan,
                    "correction_square_moment": np.nan,
                })
            for bin_index in range(20):
                pair_bins.append({
                    "population": population,
                    "case": case,
                    "bin": bin_index,
                    "baseline_residual": 0.02 * (bin_index - 9.5),
                    "correction": 0.015 * (bin_index - 9.5),
                })
                scene_bins.append({
                    "population": population,
                    "case": case,
                    "bin": bin_index,
                    "baseline_residual": 0.03 * (bin_index - 9.5),
                    "correction": 0.02 * (bin_index - 9.5),
                })
    return pd.DataFrame(case_rows), pd.DataFrame(pair_bins), pd.DataFrame(scene_bins)


def test_strength_metrics_use_exact_casewise_mse_and_curve_interpolation() -> None:
    case_frame, pair_bins, scene_bins = _toy_frames()
    strengths = np.asarray([0.0, 0.5, 1.0])
    got = build_strength_metrics(case_frame, pair_bins, scene_bins, strengths)
    assert len(got) == len(POPULATIONS) * len(strengths)
    baseline = got.loc[np.isclose(got.strength, 0.0)]
    np.testing.assert_allclose(baseline.calibration_score, 1.0)
    local = got.loc[
        (got.population == next(iter(POPULATIONS)))
        & np.isclose(got.strength, 0.5)
    ].iloc[0]
    pair = case_frame.loc[
        (case_frame.population == next(iter(POPULATIONS)))
        & (case_frame.selection == "pair")
    ]
    expected = np.mean([
        mse_at_strength(row.baseline_mse, row.residual_correction_moment,
                        row.correction_square_moment, 0.5)
        for row in pair.itertuples()
    ])
    np.testing.assert_allclose(local.pair_mse, expected)


def test_selection_uses_worst_population_and_pair_mse_guardrail() -> None:
    populations = list(POPULATIONS)
    rows = []
    scores = {
        populations[0]: [1.0, 0.70, 0.55],
        populations[1]: [1.0, 0.80, 0.60],
    }
    mse = {
        populations[0]: [0.0, -0.03, -0.04],
        populations[1]: [0.0, -0.02, +0.01],
    }
    for population in populations:
        for strength, score, change in zip((0.0, 1.0, 1.5), scores[population], mse[population]):
            rows.append({
                "population": population,
                "strength": strength,
                "calibration_score": score,
                "pair_mse_percent_change_from_v22": change,
            })
    selected = select_strength(pd.DataFrame(rows))
    assert selected["selected_strength"] == 1.0
    assert selected["worst_population_calibration_score"] == 0.80


def test_final_loader_enforces_gain_firewall(tmp_path) -> None:
    path = tmp_path / "gain.json"
    payload = {
        "model_name": "shallow_d3_m2000_l10",
        "selection": {"selected_strength": 1.18},
        "search": {"strength_min": 0.0, "strength_max": 1.5},
        "provenance": {
            "external_final_cases_20_39_opened": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    got = load_strength_selection(
        str(path), {"selected_candidate_name": "shallow_d3_m2000_l10"}
    )
    assert got["strength"] == 1.18
    payload["provenance"]["constgold_opened"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="firewall"):
        load_strength_selection(
            str(path), {"selected_candidate_name": "shallow_d3_m2000_l10"}
        )
