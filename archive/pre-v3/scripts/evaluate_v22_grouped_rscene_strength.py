#!/usr/bin/env python3
"""Tune one global gain on the selected V2.2 pair-residual correction.

The gain is selected only from the two half-shear selection populations used
by the preceding tree search.  Every statistic is formed per rendered case;
coherent-anchor and ConstGold products are never read.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.v22_grouped_rscene_common import (
    CONDITIONAL_FEATURES,
    VALIDATION_SCALE,
    case_offsets,
    conditional_matrix,
    json_clean,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    sha256,
    strict_json,
)
from scripts.v22_grouped_rscene_hparam_common import (
    DEVELOPMENT_CASES,
    N_CURVE_BINS,
    SELECTION_CASES,
    TUNE_TRAIN_CASES,
    physical_prediction,
)


POPULATIONS = {
    "internal_selection_c160_199": (*SELECTION_CASES, True),
    "external_development_c0_19": (*DEVELOPMENT_CASES, False),
}
STRENGTHS = np.round(np.linspace(0.0, 1.5, 151), 8)
PAIR_MSE_TOLERANCE_PERCENT = 0.0
BOOTSTRAP_REPETITIONS = 2000
MODEL_NAME = "shallow_d3_m2000_l10"


def finite_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("SEM requires at least two finite values")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def residual_at_strength(
    baseline_residual: np.ndarray,
    correction: np.ndarray,
    strength: float | np.ndarray,
) -> np.ndarray:
    """Return label minus (V2.2 plus strength times correction)."""
    return np.asarray(baseline_residual) - np.asarray(strength) * np.asarray(correction)


def mse_at_strength(
    baseline_mse: np.ndarray,
    residual_correction_moment: np.ndarray,
    correction_square_moment: np.ndarray,
    strength: float | np.ndarray,
) -> np.ndarray:
    """Evaluate E[(r0 - strength*c)^2] from sufficient moments."""
    strength = np.asarray(strength)
    return (
        np.asarray(baseline_mse)
        - 2.0 * strength * np.asarray(residual_correction_moment)
        + np.square(strength) * np.asarray(correction_square_moment)
    )


def _case_bin_matrices(
    frame: pd.DataFrame, population: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = frame.loc[frame.population == population]
    baseline = local.pivot(index="case", columns="bin", values="baseline_residual")
    correction = local.pivot(index="case", columns="bin", values="correction")
    if (
        baseline.shape[1] != N_CURVE_BINS
        or correction.shape != baseline.shape
        or baseline.isna().any().any()
        or correction.isna().any().any()
    ):
        raise RuntimeError(f"{population}: incomplete case-by-bin matrices")
    baseline = baseline.sort_index()
    correction = correction.reindex(index=baseline.index, columns=baseline.columns)
    return (
        baseline.index.to_numpy(int),
        baseline.to_numpy(float),
        correction.to_numpy(float),
    )


def _case_rows(
    frame: pd.DataFrame, population: str, selection: str
) -> pd.DataFrame:
    local = frame.loc[
        (frame.population == population) & (frame.selection == selection)
    ].sort_values("case")
    if local.empty or local.case.duplicated().any():
        raise RuntimeError(f"{population}/{selection}: invalid case rows")
    return local


def build_strength_metrics(
    case_frame: pd.DataFrame,
    pair_case_bins: pd.DataFrame,
    scene_case_bins: pd.DataFrame,
    strengths: np.ndarray = STRENGTHS,
) -> pd.DataFrame:
    """Evaluate exact case-averaged metrics for every proposed gain."""
    strengths = np.asarray(strengths, dtype=np.float64)
    if strengths.ndim != 1 or len(strengths) < 2 or not np.isfinite(strengths).all():
        raise ValueError("strength grid must be a finite one-dimensional array")
    rows: list[dict[str, Any]] = []
    for population in POPULATIONS:
        pair_cases, pair0, pair_correction = _case_bin_matrices(
            pair_case_bins, population
        )
        scene_cases, scene0, scene_correction = _case_bin_matrices(
            scene_case_bins, population
        )
        if not np.array_equal(pair_cases, scene_cases):
            raise RuntimeError(f"{population}: pair and scene case coverage differs")
        pair_curve0 = pair0.mean(axis=0)
        scene_curve0 = scene0.mean(axis=0)
        pair_curve = (
            pair_curve0[None, :]
            - strengths[:, None] * pair_correction.mean(axis=0)[None, :]
        )
        scene_curve = (
            scene_curve0[None, :]
            - strengths[:, None] * scene_correction.mean(axis=0)[None, :]
        )
        pair_rms0 = float(np.sqrt(np.mean(np.square(pair_curve0))))
        scene_rms0 = float(np.sqrt(np.mean(np.square(scene_curve0))))
        pair_ratio = np.sqrt(np.mean(np.square(pair_curve), axis=1)) / pair_rms0
        scene_ratio = np.sqrt(np.mean(np.square(scene_curve), axis=1)) / scene_rms0

        pair = _case_rows(case_frame, population, "pair")
        if not np.array_equal(pair.case.to_numpy(int), pair_cases):
            raise RuntimeError(f"{population}: pair moments and bins cover different cases")
        pair_residual_cases = (
            pair.baseline_residual.to_numpy(float)[:, None]
            - pair.correction.to_numpy(float)[:, None] * strengths[None, :]
        )
        pair_mse_cases = mse_at_strength(
            pair.baseline_mse.to_numpy(float)[:, None],
            pair.residual_correction_moment.to_numpy(float)[:, None],
            pair.correction_square_moment.to_numpy(float)[:, None],
            strengths[None, :],
        )
        pair_mse_change_cases = 100.0 * (
            pair_mse_cases / pair.baseline_mse.to_numpy(float)[:, None] - 1.0
        )
        scene_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for selection in ("all", "tail_gt_0p1", "tail_gt_0p2"):
            local = _case_rows(case_frame, population, selection)
            if not np.array_equal(local.case.to_numpy(int), pair_cases):
                raise RuntimeError(f"{population}/{selection}: case coverage differs")
            values = (
                local.baseline_residual.to_numpy(float)[:, None]
                - local.correction.to_numpy(float)[:, None] * strengths[None, :]
            )
            scene_values[selection] = (
                values.mean(axis=0),
                values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]),
            )

        pair_mean = pair_residual_cases.mean(axis=0)
        pair_sem = pair_residual_cases.std(axis=0, ddof=1) / np.sqrt(
            pair_residual_cases.shape[0]
        )
        mse_mean = pair_mse_cases.mean(axis=0)
        mse_change_mean = pair_mse_change_cases.mean(axis=0)
        mse_change_sem = pair_mse_change_cases.std(axis=0, ddof=1) / np.sqrt(
            pair_mse_change_cases.shape[0]
        )
        for index, strength in enumerate(strengths):
            rows.append({
                "population": population,
                "strength": float(strength),
                "n_cases": int(len(pair_cases)),
                "pair_residual": float(pair_mean[index]),
                "pair_residual_case_sem": float(pair_sem[index]),
                "pair_mse": float(mse_mean[index]),
                "pair_mse_percent_change_from_v22": float(mse_change_mean[index]),
                "pair_mse_percent_change_case_sem": float(mse_change_sem[index]),
                "pair_prediction_curve_rms_ratio_to_v22": float(pair_ratio[index]),
                "scene_prediction_curve_rms_ratio_to_v22": float(scene_ratio[index]),
                "calibration_score": float(0.5 * (pair_ratio[index] + scene_ratio[index])),
                "scene_gap_all": float(scene_values["all"][0][index]),
                "scene_gap_all_case_sem": float(scene_values["all"][1][index]),
                "scene_gap_tail_gt_0p1": float(scene_values["tail_gt_0p1"][0][index]),
                "scene_gap_tail_gt_0p1_case_sem": float(scene_values["tail_gt_0p1"][1][index]),
                "scene_gap_tail_gt_0p2": float(scene_values["tail_gt_0p2"][0][index]),
                "scene_gap_tail_gt_0p2_case_sem": float(scene_values["tail_gt_0p2"][1][index]),
            })
    return pd.DataFrame(rows)


def _choose_index(
    strengths: np.ndarray,
    worst_score: np.ndarray,
    mean_score: np.ndarray,
    eligible: np.ndarray,
) -> int:
    index = np.flatnonzero(eligible)
    if len(index) == 0:
        raise RuntimeError("no strength satisfies the pair-MSE guardrail")
    order = np.lexsort((
        strengths[index],
        np.abs(strengths[index] - 1.0),
        mean_score[index],
        worst_score[index],
    ))
    return int(index[order[0]])


def select_strength(metrics: pd.DataFrame) -> dict[str, Any]:
    """Minimize worst-population calibration subject to pair-MSE safety."""
    populations = list(POPULATIONS)
    strengths = np.sort(metrics.strength.unique().astype(float))
    score = np.vstack([
        metrics.loc[metrics.population == population]
        .sort_values("strength").calibration_score.to_numpy(float)
        for population in populations
    ])
    mse_change = np.vstack([
        metrics.loc[metrics.population == population]
        .sort_values("strength").pair_mse_percent_change_from_v22.to_numpy(float)
        for population in populations
    ])
    if score.shape != mse_change.shape or score.shape[1] != len(strengths):
        raise RuntimeError("strength metric grid is not rectangular")
    eligible = np.all(mse_change <= PAIR_MSE_TOLERANCE_PERCENT, axis=0)
    worst = score.max(axis=0)
    mean = score.mean(axis=0)
    selected_index = _choose_index(strengths, worst, mean, eligible)
    population_optima: dict[str, Any] = {}
    for row, population in enumerate(populations):
        local_eligible = mse_change[row] <= PAIR_MSE_TOLERANCE_PERCENT
        local_index = _choose_index(
            strengths,
            score[row],
            score[row],
            local_eligible,
        )
        population_optima[population] = {
            "strength": float(strengths[local_index]),
            "calibration_score": float(score[row, local_index]),
            "pair_mse_percent_change_from_v22": float(mse_change[row, local_index]),
            "other_population_scores": {
                other: float(score[other_row, local_index])
                for other_row, other in enumerate(populations)
                if other != population
            },
        }
    return {
        "selected_strength": float(strengths[selected_index]),
        "selected_grid_index": selected_index,
        "worst_population_calibration_score": float(worst[selected_index]),
        "mean_population_calibration_score": float(mean[selected_index]),
        "population_calibration_scores": {
            population: float(score[row, selected_index])
            for row, population in enumerate(populations)
        },
        "pair_mse_percent_changes": {
            population: float(mse_change[row, selected_index])
            for row, population in enumerate(populations)
        },
        "population_specific_optima": population_optima,
        "criterion": (
            "Lowest worst-population half-shear calibration score among gains "
            "whose mean casewise pair-MSE change is <= 0 in both populations; "
            "ties prefer the lower mean score, then proximity to unit gain."
        ),
    }


def bootstrap_strength(
    case_frame: pd.DataFrame,
    pair_case_bins: pd.DataFrame,
    scene_case_bins: pd.DataFrame,
    selected_strength: float,
    strengths: np.ndarray = STRENGTHS,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = 20260815,
) -> dict[str, Any]:
    """Case-bootstrap the full guarded minimax gain selection."""
    strengths = np.asarray(strengths, dtype=np.float64)
    selected_index = int(np.flatnonzero(np.isclose(strengths, selected_strength))[0])
    unit_index = int(np.flatnonzero(np.isclose(strengths, 1.0))[0])
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for population in POPULATIONS:
        cases, pair0, pair_correction = _case_bin_matrices(pair_case_bins, population)
        scene_cases, scene0, scene_correction = _case_bin_matrices(
            scene_case_bins, population
        )
        pair = _case_rows(case_frame, population, "pair")
        if not (
            np.array_equal(cases, scene_cases)
            and np.array_equal(cases, pair.case.to_numpy(int))
        ):
            raise RuntimeError(f"{population}: bootstrap case coverage differs")
        prepared[population] = {
            "pair0": pair0,
            "pair_correction": pair_correction,
            "scene0": scene0,
            "scene_correction": scene_correction,
            "mse0": pair.baseline_mse.to_numpy(float),
            "cross": pair.residual_correction_moment.to_numpy(float),
            "correction2": pair.correction_square_moment.to_numpy(float),
        }
    generator = np.random.default_rng(seed)
    selected = np.empty(repetitions, dtype=np.float64)
    score_delta = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        population_scores = []
        population_mse_changes = []
        for population in POPULATIONS:
            data = prepared[population]
            draw = generator.integers(0, len(data["mse0"]), size=len(data["mse0"]))
            pair0 = data["pair0"][draw].mean(axis=0)
            pair_correction = data["pair_correction"][draw].mean(axis=0)
            scene0 = data["scene0"][draw].mean(axis=0)
            scene_correction = data["scene_correction"][draw].mean(axis=0)
            pair_curve = pair0[None, :] - strengths[:, None] * pair_correction[None, :]
            scene_curve = scene0[None, :] - strengths[:, None] * scene_correction[None, :]
            pair_ratio = np.sqrt(np.mean(np.square(pair_curve), axis=1)) / np.sqrt(
                np.mean(np.square(pair0))
            )
            scene_ratio = np.sqrt(np.mean(np.square(scene_curve), axis=1)) / np.sqrt(
                np.mean(np.square(scene0))
            )
            population_scores.append(0.5 * (pair_ratio + scene_ratio))
            mse0 = data["mse0"][draw, None]
            mse = mse_at_strength(
                mse0,
                data["cross"][draw, None],
                data["correction2"][draw, None],
                strengths[None, :],
            )
            population_mse_changes.append(np.mean(100.0 * (mse / mse0 - 1.0), axis=0))
        score = np.vstack(population_scores)
        mse_change = np.vstack(population_mse_changes)
        worst = score.max(axis=0)
        mean = score.mean(axis=0)
        eligible = np.all(mse_change <= PAIR_MSE_TOLERANCE_PERCENT, axis=0)
        chosen = _choose_index(strengths, worst, mean, eligible)
        selected[repetition] = strengths[chosen]
        score_delta[repetition] = worst[selected_index] - worst[unit_index]
    return {
        "repetitions": int(repetitions),
        "resampling_unit": "rendered case, independently within each population",
        "selected_strength_mean": float(selected.mean()),
        "selected_strength_sd": float(selected.std(ddof=1)),
        "selected_strength_p16": float(np.quantile(selected, 0.16)),
        "selected_strength_median": float(np.median(selected)),
        "selected_strength_p84": float(np.quantile(selected, 0.84)),
        "fraction_selecting_unit_or_lower": float(np.mean(selected <= 1.0)),
        "fraction_at_search_boundaries": float(
            np.mean((selected == strengths[0]) | (selected == strengths[-1]))
        ),
        "selected_minus_unit_worst_score": {
            "mean": float(score_delta.mean()),
            "p16": float(np.quantile(score_delta, 0.16)),
            "p84": float(np.quantile(score_delta, 0.84)),
            "fraction_below_zero": float(np.mean(score_delta < 0.0)),
        },
    }


def load_selected_candidate(
    summary_path_string: str, tuning_path_string: str
) -> dict[str, Any]:
    summary_path = Path(summary_path_string).resolve()
    tuning_path = Path(tuning_path_string).resolve()
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with tuning_path.open(encoding="utf-8") as handle:
        tuning = json.load(handle)
    selected_id = int(tuning["selection"]["selected_candidate_id"])
    selected_name = str(tuning["selection"]["selected_candidate_name"])
    if summary["candidate_id"] != selected_id or summary["candidate_name"] != selected_name:
        raise RuntimeError("candidate summary is not the frozen tree-search selection")
    if summary["candidate_name"] != MODEL_NAME:
        raise RuntimeError("selected candidate name drifted from the preregistered gain scan")
    if summary["feature_names"] != CONDITIONAL_FEATURES:
        raise RuntimeError("candidate feature definition drifted")
    if summary["training"]["case_window"] != list(TUNE_TRAIN_CASES):
        raise RuntimeError("gain candidate was not fit on c40--139")
    candidate_entries = {
        int(item["candidate_id"]): item for item in tuning["candidate_models"]
    }
    entry = candidate_entries[selected_id]
    if entry["summary_sha256"] != sha256(summary_path):
        raise RuntimeError("candidate summary hash differs from tree-search result")
    model_path = Path(summary["model"]).resolve()
    if sha256(model_path) != summary["model_sha256"] or entry["model_sha256"] != summary["model_sha256"]:
        raise RuntimeError("candidate model hash drifted")
    provenance = tuning["provenance"]
    if provenance["external_final_cases_20_39_opened"]:
        raise RuntimeError("tree search unexpectedly opened c20--39")
    if provenance["coherent_anchor_truth_opened"] or provenance["constgold_opened"]:
        raise RuntimeError("tree search crossed an evaluation firewall")
    booster = xgb.Booster({
        "device": "cpu",
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    })
    booster.load_model(model_path)
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "tuning_path": str(tuning_path),
        "tuning_sha256": sha256(tuning_path),
        "booster": booster,
    }


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_figure(metrics: pd.DataFrame, selection: dict[str, Any], output_prefix: Path) -> None:
    configure_style()
    colors = {
        "internal_selection_c160_199": "#0072B2",
        "external_development_c0_19": "#D55E00",
    }
    labels = {
        "internal_selection_c160_199": "Internal c160–199",
        "external_development_c0_19": "External dev c0–19",
    }
    linestyles = {
        "internal_selection_c160_199": "-",
        "external_development_c0_19": "--",
    }
    selected = float(selection["selected_strength"])
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex=True)
    for population in POPULATIONS:
        local = metrics.loc[metrics.population == population].sort_values("strength")
        style = dict(
            color=colors[population],
            linestyle=linestyles[population],
            linewidth=1.4,
            label=labels[population],
        )
        axes[0, 0].plot(local.strength, local.calibration_score, **style)
        axes[0, 1].plot(
            local.strength,
            local.pair_mse_percent_change_from_v22,
            **style,
        )
        axes[1, 0].plot(local.strength, local.scene_gap_all, **style)
        axes[1, 0].fill_between(
            local.strength,
            local.scene_gap_all - local.scene_gap_all_case_sem,
            local.scene_gap_all + local.scene_gap_all_case_sem,
            color=colors[population],
            alpha=0.12,
            linewidth=0,
        )
        axes[1, 1].plot(local.strength, local.scene_gap_tail_gt_0p1, **style)
        axes[1, 1].fill_between(
            local.strength,
            local.scene_gap_tail_gt_0p1 - local.scene_gap_tail_gt_0p1_case_sem,
            local.scene_gap_tail_gt_0p1 + local.scene_gap_tail_gt_0p1_case_sem,
            color=colors[population],
            alpha=0.12,
            linewidth=0,
        )
    axes[0, 0].axhline(1.0, color="0.45", linestyle=":", linewidth=0.8)
    axes[0, 0].set_ylabel("Conditional calibration score / V2.2")
    axes[0, 0].set_title("Bias-focused validation loss")
    axes[0, 1].axhline(0.0, color="0.45", linestyle=":", linewidth=0.8)
    axes[0, 1].set_ylabel("Pair MSE change from V2.2 (%)")
    axes[0, 1].set_title("Ordinary validation loss")
    for axis, title in zip(
        axes[1], ("Global accumulated scene gap", r"Tail gap for frozen $P_s>0.1$"),
    ):
        axis.axhline(0.0, color="0.45", linestyle=":", linewidth=0.8)
        axis.set_ylabel("Label − prediction")
        axis.set_title(title)
        axis.set_xlabel("Correction gain")
    for axis in axes.ravel():
        axis.axvline(1.0, color="0.35", linestyle="--", linewidth=0.9)
        axis.axvline(selected, color="#009E73", linestyle="-.", linewidth=1.1)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes,
                  fontweight="bold", fontsize=11, va="top")
    fig.suptitle(
        "Half-shear-only gain tuning of the frozen scene-informed pair correction\n"
        "Dashed vertical: unit gain; green dash-dot: guarded minimax selection",
        fontsize=10.5,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def metrics_at(metrics: pd.DataFrame, strength: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for population in POPULATIONS:
        local = metrics.loc[
            (metrics.population == population)
            & np.isclose(metrics.strength, strength)
        ]
        if len(local) != 1:
            raise RuntimeError(f"missing unique metrics for {population} gain {strength}")
        output[population] = {
            key: float(local.iloc[0][key])
            for key in (
                "pair_residual",
                "pair_residual_case_sem",
                "pair_mse",
                "pair_mse_percent_change_from_v22",
                "pair_mse_percent_change_case_sem",
                "pair_prediction_curve_rms_ratio_to_v22",
                "scene_prediction_curve_rms_ratio_to_v22",
                "calibration_score",
                "scene_gap_all",
                "scene_gap_all_case_sem",
                "scene_gap_tail_gt_0p1",
                "scene_gap_tail_gt_0p1_case_sem",
                "scene_gap_tail_gt_0p2",
                "scene_gap_tail_gt_0p2_case_sem",
            )
        }
    return output


def write_markdown(
    path: Path,
    selection: dict[str, Any],
    bootstrap: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    selected = float(selection["selected_strength"])
    lines = [
        "# Half-shear-only correction-gain tuning",
        "",
        "The selected tree correction is multiplied by one global gain. The gain "
        "is selected on c160--199 and c0--19 only; c20--39, coherent anchors, and "
        "ConstGold are not opened.",
        "",
        f"Selected gain: **{selected:.2f}**. {selection['criterion']}",
        "",
        "| population | gain | pair residual | pair MSE d% | calibration / V2.2 | global scene gap | P_s>0.1 gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    label = {
        "internal_selection_c160_199": "internal c160--199",
        "external_development_c0_19": "external c0--19",
    }
    for strength_key, strength in (("unit", 1.0), ("selected", selected)):
        for population in POPULATIONS:
            item = comparison[strength_key][population]
            lines.append(
                f"| {label[population]} | {strength:.2f} | "
                f"{item['pair_residual']:+.6f} +- {item['pair_residual_case_sem']:.6f} | "
                f"{item['pair_mse_percent_change_from_v22']:+.5f} | "
                f"{item['calibration_score']:.4f} | "
                f"{item['scene_gap_all']:+.6f} +- {item['scene_gap_all_case_sem']:.6f} | "
                f"{item['scene_gap_tail_gt_0p1']:+.6f} +- "
                f"{item['scene_gap_tail_gt_0p1_case_sem']:.6f} |"
            )
    lines.extend([
        "",
        "Population-specific preferred gains: "
        + ", ".join(
            f"{label[population]} {item['strength']:.2f}"
            for population, item in selection["population_specific_optima"].items()
        )
        + ".",
        "",
        f"Case bootstrap ({bootstrap['repetitions']} repetitions): selected gain "
        f"median {bootstrap['selected_strength_median']:.2f}, 16--84% "
        f"[{bootstrap['selected_strength_p16']:.2f}, "
        f"{bootstrap['selected_strength_p84']:.2f}]. The selected gain beats unit "
        f"gain in {100.0 * bootstrap['selected_minus_unit_worst_score']['fraction_below_zero']:.1f}% "
        "of bootstrap draws on the worst-population score.",
        "",
        "Error bars are one SEM across rendered cases. The MSE guardrail is "
        "evaluated exactly from row-level residual/correction moments.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--tree-tuning-result", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix).resolve()
    suffixes = (
        ".json", ".md", ".png", ".pdf", ".strengths.csv",
        ".case_moments.csv", ".pair_case_bins.csv", ".scene_case_bins.csv",
    )
    outputs = [output_prefix.with_suffix(suffix) for suffix in suffixes]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing gain-tuning outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    if scene_meta["source_metadata_sha256"] != sha256(source_cache / "metadata.json"):
        raise RuntimeError("scene/source cache provenance differs")
    candidate = load_selected_candidate(args.candidate_summary, args.tree_tuning_result)
    with Path(args.tree_tuning_result).resolve().open(encoding="utf-8") as handle:
        tree_tuning = json.load(handle)
    pair_edges = np.asarray(tree_tuning["pair_prediction_edges"], dtype=np.float64)
    if len(pair_edges) != N_CURVE_BINS + 1 or not np.all(np.diff(pair_edges) > 0.0):
        raise RuntimeError("tree-tuning pair edges are invalid")

    offsets = case_offsets(source_meta)
    case_all = mmap_array(source_cache, source_meta, "case")
    primary_all = mmap_array(source_cache, source_meta, "input_index")
    official_train_all = mmap_array(source_cache, source_meta, "official_train")
    label_all = mmap_array(source_cache, source_meta, "label")
    prediction_all = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction_all = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs_all = mmap_array(scene_cache, scene_meta, "scene_n_pairs")
    scene_bin_all = mmap_array(scene_cache, scene_meta, "scene_bin")

    case_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    for population, (case_min, case_max, sampled_labels) in POPULATIONS.items():
        for case_value in range(case_min, case_max + 1):
            start, stop = int(offsets[case_value]), int(offsets[case_value + 1])
            row_slice = slice(start, stop)
            if not np.all(np.asarray(case_all[row_slice]) == case_value):
                raise RuntimeError(f"case {case_value}: source-cache boundary mismatch")
            primary = np.asarray(primary_all[row_slice], dtype=np.int64)
            change = np.empty(len(primary), dtype=bool)
            change[0] = True
            change[1:] = primary[1:] != primary[:-1]
            starts = np.flatnonzero(change)
            prediction = np.asarray(prediction_all[row_slice], dtype=np.float64)
            label = np.asarray(label_all[row_slice], dtype=np.float64)
            residual0 = label - prediction
            official_validation = ~np.asarray(official_train_all[row_slice], dtype=bool)
            pair_mask = official_validation if sampled_labels else np.ones(
                len(primary), dtype=bool
            )
            scene_prediction_rows = np.asarray(
                scene_prediction_all[row_slice], dtype=np.float64
            )
            scene_prediction = scene_prediction_rows[starts]
            if np.max(np.abs(np.add.reduceat(prediction, starts) - scene_prediction)) > 2.0e-6:
                raise RuntimeError(f"case {case_value}: frozen scene sum does not replay")
            if sampled_labels:
                sampled = np.where(pair_mask, residual0, 0.0)
                baseline_scene_gap = VALIDATION_SCALE * np.add.reduceat(sampled, starts)
            else:
                baseline_scene_gap = np.add.reduceat(residual0, starts)
            features = conditional_matrix(
                x_scaled,
                x_raw,
                prediction_all,
                scene_prediction_all,
                scene_n_pairs_all,
                row_slice,
            )
            correction = physical_prediction(
                candidate["booster"], features, float(candidate["summary"]["target_scale"])
            )
            selected_residual = residual0[pair_mask]
            selected_correction = correction[pair_mask]
            case_records.append({
                "population": population,
                "case": case_value,
                "selection": "pair",
                "n_items": int(pair_mask.sum()),
                "baseline_residual": float(selected_residual.mean()),
                "correction": float(selected_correction.mean()),
                "baseline_mse": float(np.mean(np.square(selected_residual))),
                "residual_correction_moment": float(np.mean(selected_residual * selected_correction)),
                "correction_square_moment": float(np.mean(np.square(selected_correction))),
            })
            pair_bins = np.searchsorted(pair_edges[1:-1], prediction, side="right")
            for bin_index in range(N_CURVE_BINS):
                local = pair_mask & (pair_bins == bin_index)
                if not local.any():
                    raise RuntimeError(f"case {case_value}: empty pair bin {bin_index}")
                pair_records.append({
                    "population": population,
                    "case": case_value,
                    "bin": bin_index,
                    "coordinate": float(prediction[local].mean()),
                    "n_items": int(local.sum()),
                    "baseline_residual": float(residual0[local].mean()),
                    "correction": float(correction[local].mean()),
                })
            scene_correction = np.add.reduceat(correction, starts)
            for selection_name, mask in {
                "all": np.ones(len(starts), dtype=bool),
                "tail_gt_0p1": scene_prediction > 0.1,
                "tail_gt_0p2": scene_prediction > 0.2,
            }.items():
                if not mask.any():
                    raise RuntimeError(f"case {case_value}: empty scene selection {selection_name}")
                case_records.append({
                    "population": population,
                    "case": case_value,
                    "selection": selection_name,
                    "n_items": int(mask.sum()),
                    "baseline_residual": float(baseline_scene_gap[mask].mean()),
                    "correction": float(scene_correction[mask].mean()),
                    "baseline_mse": np.nan,
                    "residual_correction_moment": np.nan,
                    "correction_square_moment": np.nan,
                })
            scene_bins = np.asarray(scene_bin_all[row_slice], dtype=np.int64)[starts]
            for bin_index in range(N_CURVE_BINS):
                local = scene_bins == bin_index
                if not local.any():
                    raise RuntimeError(f"case {case_value}: empty scene bin {bin_index}")
                scene_records.append({
                    "population": population,
                    "case": case_value,
                    "bin": bin_index,
                    "coordinate": float(scene_prediction[local].mean()),
                    "n_items": int(local.sum()),
                    "baseline_residual": float(baseline_scene_gap[local].mean()),
                    "correction": float(scene_correction[local].mean()),
                })
            print(
                f"{population}: case {case_value} pairs={len(primary):,} "
                f"scenes={len(starts):,}",
                flush=True,
            )

    case_frame = pd.DataFrame(case_records)
    pair_case_frame = pd.DataFrame(pair_records)
    scene_case_frame = pd.DataFrame(scene_records)
    metrics = build_strength_metrics(case_frame, pair_case_frame, scene_case_frame)
    selection = select_strength(metrics)
    bootstrap = bootstrap_strength(
        case_frame,
        pair_case_frame,
        scene_case_frame,
        selection["selected_strength"],
    )
    comparison = {
        "v22": metrics_at(metrics, 0.0),
        "unit": metrics_at(metrics, 1.0),
        "selected": metrics_at(metrics, selection["selected_strength"]),
    }
    payload = {
        "schema_version": 1,
        "experiment": "half-shear-only gain tuning of selected V2.2 pair-residual correction",
        "model_name": candidate["summary"]["candidate_name"],
        "selection": selection,
        "bootstrap": bootstrap,
        "metrics_at_strengths": comparison,
        "search": {
            "strength_min": float(STRENGTHS[0]),
            "strength_max": float(STRENGTHS[-1]),
            "strength_step": float(STRENGTHS[1] - STRENGTHS[0]),
            "n_strengths": int(len(STRENGTHS)),
            "pair_mse_tolerance_percent": PAIR_MSE_TOLERANCE_PERCENT,
        },
        "uncertainty": "one SEM across rendered cases; gain interval from case bootstrap",
        "provenance": {
            "source_cache": str(source_cache),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "scene_cache": str(scene_cache),
            "scene_metadata_sha256": sha256(scene_cache / "metadata.json"),
            "candidate_summary": candidate["summary_path"],
            "candidate_summary_sha256": candidate["summary_sha256"],
            "tree_tuning_result": candidate["tuning_path"],
            "tree_tuning_result_sha256": candidate["tuning_sha256"],
            "gain_selection_populations": {
                population: [values[0], values[1]]
                for population, values in POPULATIONS.items()
            },
            "external_final_cases_20_39_opened": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
            "labels_used_to_define_bins": False,
        },
    }
    case_frame.to_csv(output_prefix.with_suffix(".case_moments.csv"), index=False)
    pair_case_frame.to_csv(output_prefix.with_suffix(".pair_case_bins.csv"), index=False)
    scene_case_frame.to_csv(output_prefix.with_suffix(".scene_case_bins.csv"), index=False)
    metrics.to_csv(output_prefix.with_suffix(".strengths.csv"), index=False)
    strict_json(output_prefix.with_suffix(".json"), payload)
    write_markdown(output_prefix.with_suffix(".md"), selection, bootstrap, comparison)
    make_figure(metrics, selection, output_prefix)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print("V22_RSCENE_STRENGTH_TUNING_DONE", flush=True)


if __name__ == "__main__":
    main()
