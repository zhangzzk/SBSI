#!/usr/bin/env python3
"""Select V2.2 correction hyperparameters using half-shear data only."""

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
    load_scene_metadata,
    load_source_metadata,
    json_clean,
    mmap_array,
    sha256,
    strict_json,
)
from scripts.v22_grouped_rscene_hparam_common import (
    CANDIDATES,
    DEVELOPMENT_CASES,
    N_CURVE_BINS,
    SELECTION_CASES,
    TUNE_TRAIN_CASES,
    candidate_by_id,
    physical_prediction,
    prediction_edges,
    rms,
    select_case_rows,
)


POPULATIONS = {
    "internal_selection_c160_199": (*SELECTION_CASES, True),
    "external_development_c0_19": (*DEVELOPMENT_CASES, False),
}
PAIR_MSE_TOLERANCE_PERCENT = 0.0
BOOTSTRAP_REPETITIONS = 1000


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("statistic requires at least two finite case values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def load_candidates(paths: list[str]) -> list[dict[str, Any]]:
    candidates = []
    ids: set[int] = set()
    for value in paths:
        path = Path(value).resolve()
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        candidate_id = int(summary["candidate_id"])
        if candidate_id in ids:
            raise RuntimeError(f"duplicate candidate id {candidate_id}")
        ids.add(candidate_id)
        if summary["configuration"] != candidate_by_id(candidate_id):
            raise RuntimeError(f"candidate configuration drift in {path}")
        if summary["feature_names"] != CONDITIONAL_FEATURES:
            raise RuntimeError(f"feature definition drift in {path}")
        if summary["training"]["case_window"] != list(TUNE_TRAIN_CASES):
            raise RuntimeError(f"candidate training cases drift in {path}")
        model_path = Path(summary["model"]).resolve()
        if sha256(model_path) != summary["model_sha256"]:
            raise RuntimeError(f"candidate model hash drift in {path}")
        booster = xgb.Booster({
            "device": "cpu",
            "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        })
        booster.load_model(model_path)
        if booster.num_boosted_rounds() != summary["training"]["best_rounds"]:
            raise RuntimeError(f"candidate tree-count drift in {path}")
        candidates.append({
            "id": candidate_id,
            "name": summary["candidate_name"],
            "summary_path": str(path),
            "summary_sha256": sha256(path),
            "summary": summary,
            "booster": booster,
        })
    candidates.sort(key=lambda item: item["id"])
    expected = list(range(len(CANDIDATES)))
    if [item["id"] for item in candidates] != expected:
        raise RuntimeError("candidate summaries do not cover the predefined search exactly")
    return candidates


def aggregate_curve(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    for (population, model, bin_index), local in frame.groupby(
        ["population", "model", "bin"], sort=True
    ):
        stat = finite_stat(local.residual.to_numpy(float))
        coordinate = float(np.average(local.coordinate, weights=local.n_items))
        rows.append({
            "kind": kind,
            "population": population,
            "model": model,
            "bin": int(bin_index),
            "coordinate": coordinate,
            "n_items": int(local.n_items.sum()),
            "n_cases": int(local.case.nunique()),
            "residual": stat["mean"],
            "residual_case_sem": stat["case_sem"],
        })
    return pd.DataFrame(rows)


def curve_rms(frame: pd.DataFrame, population: str, model: str) -> float:
    local = frame.loc[
        (frame.population == population) & (frame.model == model)
    ].sort_values("bin")
    if len(local) != N_CURVE_BINS:
        raise RuntimeError(f"{population}/{model} does not have {N_CURVE_BINS} bins")
    return rms(local.residual.to_numpy(float))


def bootstrap_composite_score(
    pair_case_bins: pd.DataFrame,
    scene_case_bins: pd.DataFrame,
    population: str,
    model: str,
    seed: int,
) -> dict[str, float | int]:
    def matrix(frame: pd.DataFrame, which_model: str) -> np.ndarray:
        local = frame.loc[
            (frame.population == population) & (frame.model == which_model)
        ]
        pivot = local.pivot(index="case", columns="bin", values="residual")
        if pivot.shape[1] != N_CURVE_BINS or pivot.isna().any().any():
            raise RuntimeError("bootstrap curve matrix is incomplete")
        return pivot.sort_index().to_numpy(float)

    pair = matrix(pair_case_bins, model)
    pair0 = matrix(pair_case_bins, "V2.2")
    scene = matrix(scene_case_bins, model)
    scene0 = matrix(scene_case_bins, "V2.2")
    if not (pair.shape == pair0.shape == scene.shape == scene0.shape):
        raise RuntimeError("bootstrap curve matrices have different shapes")
    generator = np.random.default_rng(seed)
    n_cases = pair.shape[0]
    score = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        draw = generator.integers(0, n_cases, size=n_cases)
        pair_ratio = rms(pair[draw].mean(axis=0)) / rms(pair0[draw].mean(axis=0))
        scene_ratio = rms(scene[draw].mean(axis=0)) / rms(scene0[draw].mean(axis=0))
        score[index] = 0.5 * (pair_ratio + scene_ratio)
    return {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_sd": float(score.std(ddof=1)),
        "bootstrap_p16": float(np.quantile(score, 0.16)),
        "bootstrap_p84": float(np.quantile(score, 0.84)),
    }


def build_summary(
    case_metrics: pd.DataFrame,
    pair_curve: pd.DataFrame,
    scene_curve: pd.DataFrame,
    pair_case_bins: pd.DataFrame,
    scene_case_bins: pd.DataFrame,
    model_names: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for population in POPULATIONS:
        baseline_pair_rms = curve_rms(pair_curve, population, "V2.2")
        baseline_scene_rms = curve_rms(scene_curve, population, "V2.2")
        output[population] = {}
        for model_index, model in enumerate(model_names):
            local = case_metrics.loc[
                (case_metrics.population == population)
                & (case_metrics.model == model)
            ]
            pair_rows = local.loc[local.selection == "pair"]
            pair_rms = curve_rms(pair_curve, population, model)
            scene_rms = curve_rms(scene_curve, population, model)
            pair_ratio = pair_rms / baseline_pair_rms
            scene_ratio = scene_rms / baseline_scene_rms
            item: dict[str, Any] = {
                "n_cases": int(pair_rows.case.nunique()),
                "n_label_pairs": int(pair_rows.n_items.sum()),
                "pair_residual": finite_stat(pair_rows.residual.to_numpy(float)),
                "pair_mse": finite_stat(pair_rows.mse.to_numpy(float)),
                "pair_mse_percent_change_from_v22": finite_stat(
                    pair_rows.mse_percent_change.to_numpy(float)
                ),
                "pair_prediction_curve_rms": pair_rms,
                "pair_prediction_curve_rms_ratio_to_v22": pair_ratio,
                "scene_prediction_curve_rms": scene_rms,
                "scene_prediction_curve_rms_ratio_to_v22": scene_ratio,
                "calibration_score": 0.5 * (pair_ratio + scene_ratio),
                "scenes": {},
            }
            for selection in ("all", "tail_gt_0p1", "tail_gt_0p2"):
                rows = local.loc[local.selection == selection]
                item["scenes"][selection] = {
                    "n_scenes": int(rows.n_items.sum()),
                    "truth_minus_prediction": finite_stat(
                        rows.residual.to_numpy(float)
                    ),
                    "applied_correction": finite_stat(
                        rows.correction.to_numpy(float)
                    ),
                }
            item["calibration_score_uncertainty"] = bootstrap_composite_score(
                pair_case_bins,
                scene_case_bins,
                population,
                model,
                seed=20260815 + 1000 * model_index + 100 * list(POPULATIONS).index(population),
            )
            output[population][model] = item
    return output


def select_candidate(
    summary: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    populations = list(POPULATIONS)
    for candidate in candidates:
        name = candidate["name"]
        population_scores = {
            population: float(summary[population][name]["calibration_score"])
            for population in populations
        }
        mse_changes = {
            population: float(
                summary[population][name]
                ["pair_mse_percent_change_from_v22"]["mean"]
            )
            for population in populations
        }
        eligible = all(
            value <= PAIR_MSE_TOLERANCE_PERCENT for value in mse_changes.values()
        )
        rows.append({
            "candidate_id": candidate["id"],
            "candidate_name": name,
            "eligible": eligible,
            "worst_population_calibration_score": max(population_scores.values()),
            "mean_population_calibration_score": float(
                np.mean(list(population_scores.values()))
            ),
            "population_calibration_scores": population_scores,
            "pair_mse_percent_changes": mse_changes,
            "best_rounds": int(
                candidate["summary"]["training"]["best_rounds"]
            ),
        })
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return {
            "selected_candidate_id": None,
            "selected_candidate_name": None,
            "reason": (
                "No candidate met the predeclared no-validation-MSE-worsening "
                "guardrail in both half-shear populations."
            ),
            "candidate_rows": rows,
        }
    chosen = min(
        eligible,
        key=lambda row: (
            row["worst_population_calibration_score"],
            row["mean_population_calibration_score"],
            row["best_rounds"],
            row["candidate_id"],
        ),
    )
    return {
        "selected_candidate_id": chosen["candidate_id"],
        "selected_candidate_name": chosen["candidate_name"],
        "reason": (
            "Lowest worst-population conditional-calibration score among "
            "candidates that do not worsen ordinary pair MSE in either "
            "half-shear selection population. The score equally weights the "
            "R_blend-pair residual curve and accumulated scene-gap curve, each "
            "normalized to frozen V2.2."
        ),
        "pair_mse_tolerance_percent": PAIR_MSE_TOLERANCE_PERCENT,
        "candidate_rows": rows,
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


def make_figure(
    candidates: list[dict[str, Any]],
    summary: dict[str, Any],
    selection: dict[str, Any],
    output_prefix: Path,
) -> None:
    configure_style()
    colors = {
        "internal_selection_c160_199": "#0072B2",
        "external_development_c0_19": "#D55E00",
    }
    markers = {
        "internal_selection_c160_199": "o",
        "external_development_c0_19": "s",
    }
    labels = {
        "internal_selection_c160_199": "Internal c160–199",
        "external_development_c0_19": "External dev c0–19",
    }
    x = np.asarray([item["id"] for item in candidates])
    selected_id = selection["selected_candidate_id"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex=True)

    train_rmse = np.asarray([
        item["summary"]["training_diagnostics"]["rmse_after"]
        for item in candidates
    ])
    early_rmse = np.asarray([
        item["summary"]["early_stopping_diagnostics"]["rmse_after"]
        for item in candidates
    ])
    axes[0, 0].plot(x, train_rmse, color="#009E73", marker="^", label="Fit c40–139")
    axes[0, 0].plot(x, early_rmse, color="#CC79A7", marker="v", label="Early stop c140–159")
    axes[0, 0].set_ylabel("Pair RMSE")
    axes[0, 0].set_title("Direct correction loss")
    axes[0, 0].legend(frameon=False)

    for population in POPULATIONS:
        mse = np.asarray([
            summary[population][item["name"]]
            ["pair_mse_percent_change_from_v22"]["mean"]
            for item in candidates
        ])
        mse_sem = np.asarray([
            summary[population][item["name"]]
            ["pair_mse_percent_change_from_v22"]["case_sem"]
            for item in candidates
        ])
        axes[0, 1].errorbar(
            x,
            mse,
            yerr=mse_sem,
            color=colors[population],
            marker=markers[population],
            linewidth=1.2,
            capsize=2,
            label=labels[population],
        )
    axes[0, 1].axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylabel("Pair MSE change from V2.2 (%)")
    axes[0, 1].set_title("Held-out ordinary loss")
    axes[0, 1].legend(frameon=False)

    for population in POPULATIONS:
        score = np.asarray([
            summary[population][item["name"]]["calibration_score"]
            for item in candidates
        ])
        score_sd = np.asarray([
            summary[population][item["name"]]["calibration_score_uncertainty"]
            ["bootstrap_sd"]
            for item in candidates
        ])
        axes[1, 0].errorbar(
            x,
            score,
            yerr=score_sd,
            color=colors[population],
            marker=markers[population],
            linewidth=1.2,
            capsize=2,
            label=labels[population],
        )
    axes[1, 0].axhline(1.0, color="0.45", linestyle="--", linewidth=0.8)
    axes[1, 0].set_ylabel("Conditional calibration score / V2.2")
    axes[1, 0].set_title("Bias-focused selection loss (one case-bootstrap SD)")

    for population in POPULATIONS:
        tail = np.asarray([
            summary[population][item["name"]]["scenes"]["tail_gt_0p1"]
            ["truth_minus_prediction"]["mean"]
            for item in candidates
        ])
        tail_sem = np.asarray([
            summary[population][item["name"]]["scenes"]["tail_gt_0p1"]
            ["truth_minus_prediction"]["case_sem"]
            for item in candidates
        ])
        axes[1, 1].errorbar(
            x,
            tail,
            yerr=tail_sem,
            color=colors[population],
            marker=markers[population],
            linewidth=1.2,
            capsize=2,
            label=labels[population],
        )
    axes[1, 1].axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
    axes[1, 1].set_ylabel(r"Scene gap for frozen $P_s>0.1$")
    axes[1, 1].set_title("Half-shear tail diagnostic")

    for axis in axes.ravel():
        if selected_id is not None:
            axis.axvspan(selected_id - 0.35, selected_id + 0.35, color="#F0E442", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(x)
    axes[1, 0].set_xlabel("Candidate ID")
    axes[1, 1].set_xlabel("Candidate ID")
    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes,
                  fontweight="bold", fontsize=11, va="top")
    fig.suptitle(
        "Half-shear-only tuning of the scene-informed V2.2 pair correction\n"
        "Yellow band marks the frozen selection; error bars use rendered cases",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    path: Path,
    candidates: list[dict[str, Any]],
    summary: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    internal = "internal_selection_c160_199"
    external = "external_development_c0_19"
    lines = [
        "# Half-shear-only correction hyperparameter tuning",
        "",
        "Candidates fit c40--139, use c140--159 only for early stopping, and "
        "are selected on c160--199 plus c0--19. Cases c20--39, coherent "
        "anchors, and ConstGold are not opened by selection.",
        "",
        f"Selected: `{selection['selected_candidate_name']}` (candidate "
        f"{selection['selected_candidate_id']}). {selection['reason']}",
        "",
        "| id | candidate | trees | train RMSE | early-stop RMSE | c160--199 MSE d% | c0--19 MSE d% | c160--199 cal | c0--19 cal | worst cal |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    row_by_id = {
        row["candidate_id"]: row for row in selection["candidate_rows"]
    }
    for candidate in candidates:
        item = row_by_id[candidate["id"]]
        train = candidate["summary"]["training_diagnostics"]["rmse_after"]
        early = candidate["summary"]["early_stopping_diagnostics"]["rmse_after"]
        lines.append(
            f"| {candidate['id']} | {candidate['name']} | {item['best_rounds']} | "
            f"{train:.6f} | {early:.6f} | "
            f"{item['pair_mse_percent_changes'][internal]:+.5f} | "
            f"{item['pair_mse_percent_changes'][external]:+.5f} | "
            f"{item['population_calibration_scores'][internal]:.4f} | "
            f"{item['population_calibration_scores'][external]:.4f} | "
            f"{item['worst_population_calibration_score']:.4f} |"
        )
    lines.extend([
        "",
        "Calibration score is the mean of two RMS ratios relative to V2.2: "
        "the pair-residual curve versus frozen pair R_blend and the accumulated "
        "scene-gap curve versus frozen scene sum. Lower is better; 1 is V2.2.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--candidate-summary", action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix).resolve()
    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (
            ".json", ".md", ".png", ".pdf", ".case_metrics.csv",
            ".pair_bins.csv", ".scene_bins.csv", ".pair_case_bins.csv",
            ".scene_case_bins.csv",
        )
    ]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing tuning outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    if scene_meta["source_metadata_sha256"] != sha256(source_cache / "metadata.json"):
        raise RuntimeError("scene/source cache provenance differs")
    candidates = load_candidates(args.candidate_summary)

    offsets = case_offsets(source_meta)
    case_all = mmap_array(source_cache, source_meta, "case")
    primary_all = mmap_array(source_cache, source_meta, "input_index")
    official_train_all = mmap_array(source_cache, source_meta, "official_train")
    label_all = mmap_array(source_cache, source_meta, "label")
    pair_prediction_all = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction_all = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs_all = mmap_array(scene_cache, scene_meta, "scene_n_pairs")
    scene_bin_all = mmap_array(scene_cache, scene_meta, "scene_bin")

    edge_index = select_case_rows(
        case_all,
        official_train_all,
        TUNE_TRAIN_CASES,
        official_validation_only=True,
    )
    pair_edges = prediction_edges(
        np.asarray(pair_prediction_all[edge_index], dtype=np.float64)
    )
    model_names = ["V2.2", *[item["name"] for item in candidates]]
    candidate_by_name = {item["name"]: item for item in candidates}
    case_records: list[dict[str, Any]] = []
    pair_bin_records: list[dict[str, Any]] = []
    scene_bin_records: list[dict[str, Any]] = []

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
            prediction = np.asarray(pair_prediction_all[row_slice], dtype=np.float64)
            label = np.asarray(label_all[row_slice], dtype=np.float64)
            residual0 = label - prediction
            official_validation = ~np.asarray(
                official_train_all[row_slice], dtype=bool
            )
            pair_mask = official_validation if sampled_labels else np.ones(
                len(primary), dtype=bool
            )
            scene_prediction_rows = np.asarray(
                scene_prediction_all[row_slice], dtype=np.float64
            )
            scene_prediction = scene_prediction_rows[starts]
            replay = np.add.reduceat(prediction, starts)
            if np.max(np.abs(replay - scene_prediction)) > 2.0e-6:
                raise RuntimeError(f"case {case_value}: frozen scene sum does not replay")
            if sampled_labels:
                sampled = np.where(pair_mask, residual0, 0.0)
                baseline_scene_gap = VALIDATION_SCALE * np.add.reduceat(
                    sampled, starts
                )
            else:
                baseline_scene_gap = np.add.reduceat(residual0, starts)
            pair_bins = np.searchsorted(pair_edges[1:-1], prediction, side="right")
            scene_bins_rows = np.asarray(
                scene_bin_all[row_slice], dtype=np.int64
            )
            scene_bins = scene_bins_rows[starts]
            features = conditional_matrix(
                x_scaled,
                x_raw,
                pair_prediction_all,
                scene_prediction_all,
                scene_n_pairs_all,
                row_slice,
            )
            corrections = {"V2.2": np.zeros(len(primary), dtype=np.float64)}
            for name, candidate in candidate_by_name.items():
                corrections[name] = physical_prediction(
                    candidate["booster"],
                    features,
                    float(candidate["summary"]["target_scale"]),
                )

            baseline_pair_mse = float(np.mean(np.square(residual0[pair_mask])))
            for model in model_names:
                correction = corrections[model]
                residual = residual0 - correction
                selected = residual[pair_mask]
                mse = float(np.mean(np.square(selected)))
                case_records.append({
                    "population": population,
                    "model": model,
                    "case": case_value,
                    "selection": "pair",
                    "n_items": int(pair_mask.sum()),
                    "residual": float(selected.mean()),
                    "correction": float(correction[pair_mask].mean()),
                    "mse": mse,
                    "mse_percent_change": 100.0 * (mse / baseline_pair_mse - 1.0),
                })
                for bin_index in range(N_CURVE_BINS):
                    local = pair_mask & (pair_bins == bin_index)
                    if not local.any():
                        raise RuntimeError(
                            f"case {case_value}: empty pair bin {bin_index}"
                        )
                    pair_bin_records.append({
                        "population": population,
                        "model": model,
                        "case": case_value,
                        "bin": bin_index,
                        "coordinate": float(prediction[local].mean()),
                        "n_items": int(local.sum()),
                        "residual": float(residual[local].mean()),
                    })
                scene_correction = np.add.reduceat(correction, starts)
                scene_gap = baseline_scene_gap - scene_correction
                for selection, mask in {
                    "all": np.ones(len(starts), dtype=bool),
                    "tail_gt_0p1": scene_prediction > 0.1,
                    "tail_gt_0p2": scene_prediction > 0.2,
                }.items():
                    if not mask.any():
                        raise RuntimeError(
                            f"case {case_value}: empty scene selection {selection}"
                        )
                    case_records.append({
                        "population": population,
                        "model": model,
                        "case": case_value,
                        "selection": selection,
                        "n_items": int(mask.sum()),
                        "residual": float(scene_gap[mask].mean()),
                        "correction": float(scene_correction[mask].mean()),
                        "mse": np.nan,
                        "mse_percent_change": np.nan,
                    })
                for bin_index in range(N_CURVE_BINS):
                    local = scene_bins == bin_index
                    if not local.any():
                        raise RuntimeError(
                            f"case {case_value}: empty scene bin {bin_index}"
                        )
                    scene_bin_records.append({
                        "population": population,
                        "model": model,
                        "case": case_value,
                        "bin": bin_index,
                        "coordinate": float(scene_prediction[local].mean()),
                        "n_items": int(local.sum()),
                        "residual": float(scene_gap[local].mean()),
                    })
            print(
                f"{population}: case {case_value} pairs={len(primary):,} "
                f"scenes={len(starts):,}",
                flush=True,
            )

    case_frame = pd.DataFrame(case_records)
    pair_case_frame = pd.DataFrame(pair_bin_records)
    scene_case_frame = pd.DataFrame(scene_bin_records)
    pair_curve = aggregate_curve(pair_case_frame, "pair_prediction")
    scene_curve = aggregate_curve(scene_case_frame, "scene_prediction")
    summary = build_summary(
        case_frame,
        pair_curve,
        scene_curve,
        pair_case_frame,
        scene_case_frame,
        model_names,
    )
    selection = select_candidate(summary, candidates)

    payload = {
        "schema_version": 1,
        "experiment": "half-shear-only hyperparameter tuning of V2.2 scene-informed pair correction",
        "search_space": [dict(item) for item in CANDIDATES],
        "candidate_models": [
            {
                "candidate_id": item["id"],
                "candidate_name": item["name"],
                "summary": item["summary_path"],
                "summary_sha256": item["summary_sha256"],
                "model": item["summary"]["model"],
                "model_sha256": item["summary"]["model_sha256"],
            }
            for item in candidates
        ],
        "metrics": summary,
        "selection": selection,
        "pair_prediction_edges": pair_edges,
        "scene_prediction_edges": scene_meta["scene_bin_definition"]["edges"],
        "uncertainty": {
            "reported_error_bars": "one SEM across rendered cases",
            "calibration_score_interval": (
                f"16th--84th percentiles from {BOOTSTRAP_REPETITIONS} "
                "case-bootstrap repetitions"
            ),
        },
        "provenance": {
            "source_cache": str(source_cache),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "scene_cache": str(scene_cache),
            "scene_metadata_sha256": sha256(scene_cache / "metadata.json"),
            "training_cases": list(TUNE_TRAIN_CASES),
            "early_stopping_cases": [140, 159],
            "internal_selection_cases": list(SELECTION_CASES),
            "external_development_cases": list(DEVELOPMENT_CASES),
            "external_final_cases_20_39_opened": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
            "labels_used_to_define_bins": False,
        },
    }

    case_frame.to_csv(output_prefix.with_suffix(".case_metrics.csv"), index=False)
    pair_curve.to_csv(output_prefix.with_suffix(".pair_bins.csv"), index=False)
    scene_curve.to_csv(output_prefix.with_suffix(".scene_bins.csv"), index=False)
    pair_case_frame.to_csv(
        output_prefix.with_suffix(".pair_case_bins.csv"), index=False
    )
    scene_case_frame.to_csv(
        output_prefix.with_suffix(".scene_case_bins.csv"), index=False
    )
    strict_json(output_prefix.with_suffix(".json"), payload)
    write_markdown(output_prefix.with_suffix(".md"), candidates, summary, selection)
    make_figure(candidates, summary, selection, output_prefix)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print("V22_RSCENE_HPARAM_EVALUATION_DONE", flush=True)


if __name__ == "__main__":
    main()
