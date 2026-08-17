#!/usr/bin/env python3
"""Evaluate the fixed grouped residual model on sealed half-shear and anchors."""

from __future__ import annotations

import argparse
import json
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
    EXTERNAL_FINAL_CASE_MAX,
    EXTERNAL_FINAL_CASE_MIN,
    assign_scene_bins,
    case_offsets,
    conditional_matrix,
    finite_stat,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    physical_correction,
    scene_edges_from_metadata,
    sha256,
    strict_json,
)


SELECTIONS = ("all", "outside_le_0p1", "tail_gt_0p1", "tail_gt_0p2")


def selection_masks(prediction: np.ndarray) -> dict[str, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "all": np.ones(len(prediction), dtype=bool),
        "outside_le_0p1": prediction <= 0.1,
        "tail_gt_0p1": prediction > 0.1,
        "tail_gt_0p2": prediction > 0.2,
    }


def aggregate_bin_records(records: list[dict[str, Any]], kind: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    rows = []
    for (dataset, model, bin_index), local in frame.groupby(
        ["dataset", "model", "bin"], sort=True
    ):
        residual = finite_stat(local.residual.to_numpy(float))
        baseline = finite_stat(local.baseline_residual.to_numpy(float))
        correction = finite_stat(local.correction.to_numpy(float))
        rows.append({
            "kind": kind,
            "dataset": dataset,
            "model": model,
            "bin": int(bin_index),
            "coordinate": float(np.average(
                local.coordinate, weights=local.n_items
            )),
            "n_items": int(local.n_items.sum()),
            "n_cases": int(local.case.nunique()),
            "residual": residual["mean"],
            "residual_sem": residual["case_sem"],
            "baseline_residual": baseline["mean"],
            "baseline_residual_sem": baseline["case_sem"],
            "correction": correction["mean"],
            "correction_sem": correction["case_sem"],
        })
    return pd.DataFrame(rows)


def summarize_cases(case_frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (dataset, model), local in case_frame.groupby(
        ["dataset", "model"], sort=False
    ):
        item: dict[str, Any] = {"scenes": {}}
        pair = local.loc[local.selection == "pair"]
        if len(pair):
            item["pairs"] = {
                "n_pairs": int(pair.n_items.sum()),
                "residual": finite_stat(pair.residual.to_numpy(float)),
                "mse": finite_stat(pair.mse.to_numpy(float)),
                "mse_percent_change_from_v22": finite_stat(
                    pair.mse_percent_change.to_numpy(float)
                ),
            }
        for selection in SELECTIONS:
            scene = local.loc[local.selection == selection]
            item["scenes"][selection] = {
                "n_scenes": int(scene.n_items.sum()),
                "truth_minus_prediction": finite_stat(
                    scene.residual.to_numpy(float)
                ),
                "v22_truth_minus_prediction": finite_stat(
                    scene.baseline_residual.to_numpy(float)
                ),
                "applied_scene_correction": finite_stat(
                    scene.correction.to_numpy(float)
                ),
            }
        output.setdefault(dataset, {})[model] = item
    return output


def configure_plot() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def curve(axis: plt.Axes, frame: pd.DataFrame, dataset: str, title: str) -> None:
    colors = {"V2.2": "#0072B2", "Grouped residual": "#D55E00"}
    markers = {"V2.2": "o", "Grouped residual": "s"}
    for model in ("V2.2", "Grouped residual"):
        local = frame.loc[(frame.dataset == dataset) & (frame.model == model)]
        axis.errorbar(
            local.coordinate, local.residual, yerr=local.residual_sem,
            color=colors[model], marker=markers[model], linewidth=1.5,
            capsize=2, ms=3, label=model,
        )
    axis.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axis.axvline(0.1, color="0.7", linestyle=":", linewidth=1)
    axis.set_xscale("symlog", linthresh=1.0e-3)
    axis.set_xlabel(r"Frozen V2.2 scene sum $P_s$")
    axis.set_ylabel("Truth − prediction")
    axis.set_title(title)


def make_figure(
    pair_bins: pd.DataFrame,
    scene_bins: pd.DataFrame,
    summary: dict[str, Any],
    output_prefix: Path,
) -> None:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    curve(
        axes[0, 0], pair_bins, "half_shear_final_c20_39",
        "Half-shear final: pair residual",
    )
    axes[0, 0].set_ylabel("Mean pair residual")
    curve(
        axes[0, 1], scene_bins, "half_shear_final_c20_39",
        "Half-shear final: scene gap",
    )
    curve(
        axes[1, 0], scene_bins, "coherent_anchor_c400_899",
        "Coherent anchors: scene gap",
    )
    axes[0, 0].legend(frameon=False)

    categories = [
        ("half_shear_final_c20_39", "all", "Half-shear\nall"),
        ("half_shear_final_c20_39", "tail_gt_0p1", "Half-shear\n$P_s>0.1$"),
        ("coherent_anchor_c400_899", "all", "Anchors\nall"),
        ("coherent_anchor_c400_899", "tail_gt_0p1", "Anchors\n$P_s>0.1$"),
    ]
    x = np.arange(len(categories), dtype=float)
    for offset, model, color, marker in (
        (-0.10, "V2.2", "#0072B2", "o"),
        (+0.10, "Grouped residual", "#D55E00", "s"),
    ):
        mean, sem = [], []
        for dataset, selection, _ in categories:
            stat = summary[dataset][model]["scenes"][selection][
                "truth_minus_prediction"
            ]
            mean.append(stat["mean"])
            sem.append(stat["case_sem"])
        axes[1, 1].errorbar(
            x + offset, mean, yerr=sem, color=color, marker=marker,
            linestyle="none", capsize=3, label=model,
        )
    axes[1, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(x, [item[2] for item in categories])
    axes[1, 1].set_ylabel("Truth − prediction")
    axes[1, 1].set_title("Transfer summary")
    axes[1, 1].legend(frameon=False)
    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes,
                  fontsize=12, fontweight="bold", va="top")
    fig.suptitle(
        "Frozen pair-aware grouped residual correction\n"
        "Error bars are one SEM across rendered cases",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt(item: dict[str, Any]) -> str:
    return f"{item['mean']:+.6f} ± {item['case_sem']:.6f}"


def write_markdown(path: Path, summary: dict[str, Any], model: dict[str, Any]) -> None:
    lines = [
        "# Final grouped scene-residual transfer",
        "",
        f"Selected grouped-loss strength: `{model['training']['group_strength']}`.",
        "The model was fixed using half-shear development data, retrained on "
        "official-validation pairs in cases 40–199, and then evaluated once on "
        "sealed half-shear cases 20–39 and coherent anchors 400–899.",
        "",
        "| dataset / selection | V2.2 gap | grouped-residual gap |",
        "|---|---:|---:|",
    ]
    for dataset, selection in (
        ("half_shear_final_c20_39", "all"),
        ("half_shear_final_c20_39", "tail_gt_0p1"),
        ("half_shear_final_c20_39", "tail_gt_0p2"),
        ("coherent_anchor_c400_899", "all"),
        ("coherent_anchor_c400_899", "tail_gt_0p1"),
        ("coherent_anchor_c400_899", "tail_gt_0p2"),
    ):
        baseline = summary[dataset]["V2.2"]["scenes"][selection]
        corrected = summary[dataset]["Grouped residual"]["scenes"][selection]
        lines.append(
            f"| {dataset}: {selection} | "
            f"{fmt(baseline['truth_minus_prediction'])} | "
            f"{fmt(corrected['truth_minus_prediction'])} |"
        )
    pair = summary["half_shear_final_c20_39"]["Grouped residual"]["pairs"]
    lines.extend([
        "",
        "## Ordinary pair loss on sealed half-shear cases 20–39",
        "",
        f"- Pair residual: {fmt(pair['residual'])}.",
        f"- Pair MSE change from V2.2: "
        f"{fmt(pair['mse_percent_change_from_v22'])}%.",
        "",
        "Errors are one SEM across rendered cases.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--model-summary", required=True)
    parser.add_argument("--anchor-score-dir", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    model_summary_path = Path(args.model_summary).resolve()
    anchor_score_dir = Path(args.anchor_score_dir).resolve()
    anchor_feature_path = Path(args.anchor_features).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (
            ".json", ".md", ".png", ".pdf", ".case_metrics.csv",
            ".pair_bins.csv", ".scene_bins.csv",
        )
    ]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing outputs: {existing}")

    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    with model_summary_path.open(encoding="utf-8") as handle:
        model_summary = json.load(handle)
    if model_summary["feature_names"] != CONDITIONAL_FEATURES:
        raise RuntimeError("final model feature definition drifted")
    if model_summary["training"]["case_window"] != [40, 199]:
        raise RuntimeError("final model was not retrained on c40--199")
    model_path = Path(model_summary["model"])
    if sha256(model_path) != model_summary["model_sha256"]:
        raise RuntimeError("final model hash differs from summary")
    booster = xgb.Booster({"device": "cpu", "n_jobs": 8})
    booster.load_model(model_path)

    offsets = case_offsets(source_meta)
    case_all = mmap_array(source_cache, source_meta, "case")
    primary_all = mmap_array(source_cache, source_meta, "input_index")
    label_all = mmap_array(source_cache, source_meta, "label")
    pair_prediction_all = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction_all = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs_all = mmap_array(scene_cache, scene_meta, "scene_n_pairs")
    scene_bin_all = mmap_array(scene_cache, scene_meta, "scene_bin")
    edges = scene_edges_from_metadata(scene_meta)
    n_bins = len(edges) - 1

    case_records: list[dict[str, Any]] = []
    pair_bin_records: list[dict[str, Any]] = []
    scene_bin_records: list[dict[str, Any]] = []

    for case in range(EXTERNAL_FINAL_CASE_MIN, EXTERNAL_FINAL_CASE_MAX + 1):
        start, stop = int(offsets[case]), int(offsets[case + 1])
        row_slice = slice(start, stop)
        if not np.all(np.asarray(case_all[row_slice]) == case):
            raise RuntimeError(f"case {case}: source-cache boundary mismatch")
        primary = np.asarray(primary_all[row_slice], dtype=np.int64)
        change = np.empty(len(primary), dtype=bool)
        change[0] = True
        change[1:] = primary[1:] != primary[:-1]
        starts = np.flatnonzero(change)
        prediction = np.asarray(pair_prediction_all[row_slice], dtype=float)
        label = np.asarray(label_all[row_slice], dtype=float)
        residual0 = label - prediction
        scene_prediction_rows = np.asarray(
            scene_prediction_all[row_slice], dtype=float
        )
        scene_prediction = scene_prediction_rows[starts]
        pair_bins = np.asarray(scene_bin_all[row_slice], dtype=int)
        scene_bins = pair_bins[starts]
        features = conditional_matrix(
            x_scaled, x_raw, pair_prediction_all, scene_prediction_all,
            scene_n_pairs_all, row_slice,
        )
        correction = physical_correction(
            booster, features, float(model_summary["target_scale"])
        )
        correction_by_model = {
            "V2.2": np.zeros(len(primary), dtype=float),
            "Grouped residual": correction,
        }
        baseline_scene_gap = np.add.reduceat(residual0, starts)
        masks = selection_masks(scene_prediction)
        baseline_mse = float(np.mean(np.square(residual0)))
        for model_name, local_correction in correction_by_model.items():
            residual = residual0 - local_correction
            mse = float(np.mean(np.square(residual)))
            case_records.append({
                "dataset": "half_shear_final_c20_39", "model": model_name,
                "case": case, "selection": "pair", "n_items": len(primary),
                "residual": float(residual.mean()),
                "baseline_residual": float(residual0.mean()),
                "correction": float(local_correction.mean()), "mse": mse,
                "baseline_mse": baseline_mse,
                "mse_percent_change": 100.0 * (mse / baseline_mse - 1.0),
            })
            for bin_index in range(n_bins):
                local = pair_bins == bin_index
                if not local.any():
                    continue
                pair_bin_records.append({
                    "dataset": "half_shear_final_c20_39", "model": model_name,
                    "case": case, "bin": bin_index,
                    "coordinate": float(scene_prediction_rows[local].mean()),
                    "n_items": int(local.sum()),
                    "residual": float(residual[local].mean()),
                    "baseline_residual": float(residual0[local].mean()),
                    "correction": float(local_correction[local].mean()),
                })
            scene_correction = np.add.reduceat(local_correction, starts)
            scene_gap = baseline_scene_gap - scene_correction
            for selection, mask in masks.items():
                case_records.append({
                    "dataset": "half_shear_final_c20_39", "model": model_name,
                    "case": case, "selection": selection,
                    "n_items": int(mask.sum()),
                    "residual": float(scene_gap[mask].mean()),
                    "baseline_residual": float(baseline_scene_gap[mask].mean()),
                    "correction": float(scene_correction[mask].mean()),
                    "mse": np.nan, "baseline_mse": np.nan,
                    "mse_percent_change": np.nan,
                })
            for bin_index in range(n_bins):
                local = scene_bins == bin_index
                if not local.any():
                    continue
                scene_bin_records.append({
                    "dataset": "half_shear_final_c20_39", "model": model_name,
                    "case": case, "bin": bin_index,
                    "coordinate": float(scene_prediction[local].mean()),
                    "n_items": int(local.sum()),
                    "residual": float(scene_gap[local].mean()),
                    "baseline_residual": float(
                        baseline_scene_gap[local].mean()
                    ),
                    "correction": float(scene_correction[local].mean()),
                })
        print(f"half-shear final case {case}: pairs={len(primary):,}", flush=True)

    score_parts = []
    for case in range(400, 900):
        path = anchor_score_dir / f"case{case}.feather"
        if not path.exists():
            raise FileNotFoundError(path)
        score_parts.append(pd.read_feather(path))
    anchor_score = pd.concat(score_parts, ignore_index=True)
    del score_parts
    anchor = pd.read_feather(
        anchor_feature_path,
        columns=[
            "case", "input_index", "R_blend_truth", "scene_prediction",
            "log1p_n_pairs",
        ],
    )
    joined = anchor.merge(
        anchor_score, on=["case", "input_index"], how="left",
        validate="one_to_one",
    )
    if joined.R_blend_grouped.isna().any() or len(joined) != 1_703_884:
        raise RuntimeError("anchor score coverage differs from frozen truth table")
    replay_error = np.max(np.abs(
        joined.R_blend_v22_replay.to_numpy(float)
        - joined.scene_prediction.to_numpy(float)
    ))
    if replay_error > 2.0e-6:
        raise RuntimeError(f"anchor V2.2 replay differs by {replay_error:.3e}")
    expected_n = np.rint(np.expm1(joined.log1p_n_pairs)).astype(np.int16)
    if not np.array_equal(expected_n, joined.n_pairs.to_numpy(np.int16)):
        raise RuntimeError("anchor pair multiplicity differs from feature table")
    joined["scene_bin"] = assign_scene_bins(joined.scene_prediction, edges)
    joined["baseline_gap"] = joined.R_blend_truth - joined.scene_prediction
    joined["grouped_gap"] = joined.R_blend_truth - joined.R_blend_grouped
    joined["scene_correction"] = joined.grouped_pair_correction_sum
    for case, local_case in joined.groupby("case", sort=True):
        prediction = local_case.scene_prediction.to_numpy(float)
        baseline_gap = local_case.baseline_gap.to_numpy(float)
        grouped_gap = local_case.grouped_gap.to_numpy(float)
        scene_correction = local_case.scene_correction.to_numpy(float)
        scene_bins = local_case.scene_bin.to_numpy(int)
        masks = selection_masks(prediction)
        for model_name, gap, correction in (
            ("V2.2", baseline_gap, np.zeros(len(local_case))),
            ("Grouped residual", grouped_gap, scene_correction),
        ):
            for selection, mask in masks.items():
                case_records.append({
                    "dataset": "coherent_anchor_c400_899", "model": model_name,
                    "case": int(case), "selection": selection,
                    "n_items": int(mask.sum()), "residual": float(gap[mask].mean()),
                    "baseline_residual": float(baseline_gap[mask].mean()),
                    "correction": float(correction[mask].mean()),
                    "mse": np.nan, "baseline_mse": np.nan,
                    "mse_percent_change": np.nan,
                })
            for bin_index in range(n_bins):
                mask = scene_bins == bin_index
                if not mask.any():
                    continue
                scene_bin_records.append({
                    "dataset": "coherent_anchor_c400_899", "model": model_name,
                    "case": int(case), "bin": bin_index,
                    "coordinate": float(prediction[mask].mean()),
                    "n_items": int(mask.sum()), "residual": float(gap[mask].mean()),
                    "baseline_residual": float(baseline_gap[mask].mean()),
                    "correction": float(correction[mask].mean()),
                })
        if int(case) % 25 == 0:
            print(f"anchor case {int(case)}: scenes={len(local_case):,}", flush=True)

    case_frame = pd.DataFrame(case_records)
    pair_bin_frame = aggregate_bin_records(pair_bin_records, "pair")
    scene_bin_frame = aggregate_bin_records(scene_bin_records, "scene")
    summary = summarize_cases(case_frame)
    payload = {
        "schema_version": 1,
        "experiment": "final pair-aware grouped scene-residual transfer",
        "model_summary": str(model_summary_path),
        "model_summary_sha256": sha256(model_summary_path),
        "selected_group_strength": model_summary["training"]["group_strength"],
        "metrics": summary,
        "audit": {
            "anchor_v22_replay_max_abs": float(replay_error),
            "n_anchor_rows": int(len(joined)),
            "n_anchor_cases": int(joined.case.nunique()),
        },
        "uncertainty": "one SEM across rendered cases",
        "provenance": {
            "final_training_cases": [40, 199],
            "final_half_shear_cases": [20, 39],
            "coherent_anchor_cases": [400, 899],
            "strength_selected_without_cases_20_39_or_anchor_truth": True,
            "constgold_opened": False,
        },
    }
    case_frame.to_csv(output_prefix.with_suffix(".case_metrics.csv"), index=False)
    pair_bin_frame.to_csv(output_prefix.with_suffix(".pair_bins.csv"), index=False)
    scene_bin_frame.to_csv(output_prefix.with_suffix(".scene_bins.csv"), index=False)
    strict_json(output_prefix.with_suffix(".json"), payload)
    write_markdown(output_prefix.with_suffix(".md"), summary, model_summary)
    make_figure(pair_bin_frame, scene_bin_frame, summary, output_prefix)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("V22_GROUPED_RSCENE_FINAL_TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
