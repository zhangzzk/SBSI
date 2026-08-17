#!/usr/bin/env python3
"""Compare V2.2, its old P_s correction, and the new physical-proxy correction."""

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
    CONDITIONAL_FEATURES as OLD_FEATURES,
    case_offsets,
    conditional_matrix,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    physical_correction,
    sha256,
    strict_json,
)
from scripts.v22_proxy_correction_common import (
    CONDITIONAL_FEATURES as PROXY_FEATURES,
    aligned_full_scene_context,
    load_full_metadata,
    proxy_conditional_matrix,
)


MODELS = ("V2.2", "Old $P_s$ correction", "Physical $Q$ correction")
COLORS = {
    "V2.2": "#0072B2",
    "Old $P_s$ correction": "#D55E00",
    "Physical $Q$ correction": "#009E73",
}
MARKERS = {
    "V2.2": "o",
    "Old $P_s$ correction": "s",
    "Physical $Q$ correction": "^",
}


def stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))) if len(values) > 1 else float("nan"),
        "n_cases": int(len(values)),
    }


def finite_edges(raw: list[float | None]) -> np.ndarray:
    if raw[0] is not None or raw[-1] is not None:
        raise RuntimeError("proxy edges lack infinite endpoint sentinels")
    return np.asarray([-np.inf, *raw[1:-1], np.inf], dtype=float)


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = edges[1] - 1.0
    bins = np.searchsorted(edges, values, side="right") - 1
    if np.any((bins < 0) | (bins >= len(edges) - 1)):
        raise RuntimeError("proxy-bin assignment escaped support")
    return bins


def case_curve(
    case: np.ndarray,
    coordinate: np.ndarray,
    values: dict[str, np.ndarray],
    edges: np.ndarray,
    dataset: str,
    kind: str,
) -> pd.DataFrame:
    bins = bin_index(coordinate, edges)
    records = []
    for local_case in np.unique(case):
        cmask = case == local_case
        for local_bin in range(len(edges) - 1):
            mask = cmask & (bins == local_bin)
            if not mask.any():
                continue
            for model, value in values.items():
                records.append({
                    "dataset": dataset,
                    "kind": kind,
                    "case": int(local_case),
                    "bin": local_bin,
                    "model": model,
                    "coordinate": float(np.mean(coordinate[mask])),
                    "value": float(np.mean(value[mask])),
                    "n_items": int(mask.sum()),
                })
    frame = pd.DataFrame(records)
    output = []
    for (model, local_bin), local in frame.groupby(["model", "bin"], sort=True):
        value = stat(local.value.to_numpy(float))
        output.append({
            "dataset": dataset,
            "kind": kind,
            "model": model,
            "bin": int(local_bin),
            "coordinate": float(np.average(local.coordinate, weights=local.n_items)),
            "n_items": int(local.n_items.sum()),
            "n_cases": int(local.case.nunique()),
            "value": value["mean"],
            "sem": value["case_sem"],
        })
    return pd.DataFrame(output)


def scene_selections(ps: np.ndarray, logq: np.ndarray, q70: float) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(ps), dtype=bool),
        "v22_ps_gt_0p1": ps > 0.1,
        "v22_ps_gt_0p2": ps > 0.2,
        "q_upper30_train": logq >= q70,
    }


def summarize_case_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (dataset, model), local in frame.groupby(["dataset", "model"], sort=False):
        item: dict[str, Any] = {"scenes": {}}
        pair = local[local.selection == "pair"]
        if len(pair):
            item["pairs"] = {
                "n_pairs": int(pair.n_items.sum()),
                "residual": stat(pair.residual.to_numpy(float)),
                "mse": stat(pair.mse.to_numpy(float)),
                "mse_percent_change_from_v22": stat(
                    pair.mse_percent_change.to_numpy(float)
                ),
            }
        for selection in (
            "all", "v22_ps_gt_0p1", "v22_ps_gt_0p2", "q_upper30_train"
        ):
            scene = local[local.selection == selection]
            item["scenes"][selection] = {
                "n_scenes": int(scene.n_items.sum()),
                "truth_minus_prediction": stat(scene.residual.to_numpy(float)),
                "applied_scene_correction": stat(scene.correction.to_numpy(float)),
            }
        result.setdefault(dataset, {})[model] = item
    return result


def load_booster(summary_path: Path, expected_features: list[str]) -> tuple[dict, xgb.Booster]:
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["feature_names"] != expected_features:
        raise RuntimeError(f"feature definition drifted for {summary_path}")
    model_path = Path(summary["model"])
    if sha256(model_path) != summary["model_sha256"]:
        raise RuntimeError(f"model hash differs for {summary_path}")
    booster = xgb.Booster({"device": "cpu", "n_jobs": 8})
    booster.load_model(model_path)
    return summary, booster


def make_figure(
    pair_curve: pd.DataFrame,
    scene_curve: pd.DataFrame,
    summary: dict[str, Any],
    output: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    for model in MODELS:
        local = pair_curve[pair_curve.model == model]
        axes[0, 0].errorbar(
            local.coordinate, local.value, yerr=local["sem"],
            color=COLORS[model], marker=MARKERS[model], linewidth=1.4,
            markersize=3, capsize=2, label=model,
        )
    axes[0, 0].set_xscale("symlog", linthresh=1.0e-3)
    axes[0, 0].set_xlabel(r"Frozen V2.2 pair prediction $p$")
    axes[0, 0].set_ylabel("Mean pair residual")
    axes[0, 0].set_title("Half-shear final cases 20--39")
    axes[0, 0].legend(frameon=False)

    for axis, dataset, title in (
        (axes[0, 1], "half_shear_final_c20_39", "Half-shear scene gap"),
        (axes[1, 0], "coherent_anchor_c400_899", "Coherent-anchor scene gap"),
    ):
        for model in MODELS:
            local = scene_curve[
                (scene_curve.dataset == dataset) & (scene_curve.model == model)
            ]
            axis.errorbar(
                local.coordinate, local.value, yerr=local["sem"],
                color=COLORS[model], marker=MARKERS[model], linewidth=1.4,
                markersize=3, capsize=2, label=model,
            )
        axis.axhline(0.0, color="0.45", linestyle="--", linewidth=1)
        axis.set_xlabel(r"Physical scene proxy $\log_{10}Q_{d^{-2}}$")
        axis.set_ylabel("Truth − prediction")
        axis.set_title(title)

    categories = [
        ("half_shear_final_c20_39", "all", "Half-shear\nall"),
        ("half_shear_final_c20_39", "q_upper30_train", "Half-shear\nupper 30% $Q$"),
        ("half_shear_final_c20_39", "v22_ps_gt_0p1", "Half-shear\n$P_s>0.1$"),
        ("coherent_anchor_c400_899", "all", "Anchors\nall"),
        ("coherent_anchor_c400_899", "q_upper30_train", "Anchors\nupper 30% $Q$"),
        ("coherent_anchor_c400_899", "v22_ps_gt_0p1", "Anchors\n$P_s>0.1$"),
    ]
    x = np.arange(len(categories), dtype=float)
    for offset, model in zip((-0.16, 0.0, 0.16), MODELS):
        means, sems = [], []
        for dataset, selection, _ in categories:
            item = summary[dataset][model]["scenes"][selection][
                "truth_minus_prediction"
            ]
            means.append(item["mean"])
            sems.append(item["case_sem"])
        axes[1, 1].errorbar(
            x + offset, means, yerr=sems, color=COLORS[model],
            marker=MARKERS[model], linestyle="none", capsize=3, label=model,
        )
    axes[1, 1].axhline(0.0, color="0.45", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(x, [item[2] for item in categories], rotation=12)
    axes[1, 1].set_ylabel("Truth − prediction")
    axes[1, 1].set_title("Fixed-population transfer summary")
    axes[1, 1].legend(frameon=False)
    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes,
                  fontsize=12, fontweight="bold", va="top")
    fig.suptitle(
        r"V2.2 residual correction: learned scene response $P_s$ versus physical $Q_{d^{-2}}$"
        "\nError bars are one SEM across rendered cases",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--old-scene-cache", required=True)
    parser.add_argument("--full-neighbour-cache", required=True)
    parser.add_argument("--old-model-summary", required=True)
    parser.add_argument("--proxy-model-summary", required=True)
    parser.add_argument("--old-anchor-score-dir", required=True)
    parser.add_argument("--proxy-anchor-score-dir", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    output = Path(args.output_prefix).resolve()
    outputs = [output.with_suffix(s) for s in (
        ".json", ".png", ".pdf", ".case_metrics.csv",
        ".pair_curve.csv", ".scene_curve.csv",
    )]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_cache = Path(args.source_cache).resolve()
    old_scene_cache = Path(args.old_scene_cache).resolve()
    full_cache = Path(args.full_neighbour_cache).resolve()
    source_meta = load_source_metadata(source_cache)
    old_scene_meta = load_scene_metadata(old_scene_cache)
    full_meta = load_full_metadata(full_cache, source_cache)
    old_summary, old_booster = load_booster(Path(args.old_model_summary), OLD_FEATURES)
    proxy_summary, proxy_booster = load_booster(
        Path(args.proxy_model_summary), PROXY_FEATURES
    )
    q_edges = finite_edges(proxy_summary["scene_proxy"]["quantile_edges"])
    q70 = float(proxy_summary["scene_proxy"]["upper_30_percent_threshold"])
    offsets = case_offsets(source_meta)
    case_all = mmap_array(source_cache, source_meta, "case")
    primary_all = mmap_array(source_cache, source_meta, "input_index")
    label_all = mmap_array(source_cache, source_meta, "label")
    prediction_all = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled_all = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw_all = mmap_array(source_cache, source_meta, "x_raw")
    old_scene_prediction_all = mmap_array(
        old_scene_cache, old_scene_meta, "scene_prediction"
    )
    old_scene_n_all = mmap_array(old_scene_cache, old_scene_meta, "scene_n_pairs")

    metric_records: list[dict[str, Any]] = []
    half_scene_case: list[np.ndarray] = []
    half_scene_logq: list[np.ndarray] = []
    half_scene_gaps: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    pair_cases, pair_x, pair_res0, pair_res_old, pair_res_proxy = [], [], [], [], []
    for case in range(20, 40):
        start, stop = int(offsets[case]), int(offsets[case + 1])
        row_slice = slice(start, stop)
        if not np.all(np.asarray(case_all[row_slice]) == case):
            raise RuntimeError(f"case {case}: source boundary mismatch")
        primary = np.asarray(primary_all[row_slice], dtype=np.int64)
        change = np.empty(len(primary), dtype=bool)
        change[0] = True
        change[1:] = primary[1:] != primary[:-1]
        starts = np.flatnonzero(change)
        prediction = np.asarray(prediction_all[row_slice], dtype=float)
        label = np.asarray(label_all[row_slice], dtype=float)
        residual0 = label - prediction
        logq_rows, _, full_count = aligned_full_scene_context(
            primary, full_cache, full_meta, case
        )
        proxy_features = proxy_conditional_matrix(
            np.asarray(x_scaled_all[row_slice]), np.asarray(x_raw_all[row_slice]),
            prediction, logq_rows, full_count,
        )
        proxy_correction = proxy_booster.inplace_predict(proxy_features).astype(float)
        proxy_correction *= float(proxy_summary["target_scale"])
        old_features = conditional_matrix(
            x_scaled_all, x_raw_all, prediction_all, old_scene_prediction_all,
            old_scene_n_all, row_slice,
        )
        old_correction = physical_correction(
            old_booster, old_features, float(old_summary["target_scale"])
        )
        pair_values = {
            "V2.2": residual0,
            "Old $P_s$ correction": residual0 - old_correction,
            "Physical $Q$ correction": residual0 - proxy_correction,
        }
        baseline_mse = float(np.mean(np.square(residual0)))
        for model, residual in pair_values.items():
            mse = float(np.mean(np.square(residual)))
            metric_records.append({
                "dataset": "half_shear_final_c20_39", "model": model,
                "case": case, "selection": "pair", "n_items": len(primary),
                "residual": float(residual.mean()), "correction": float(
                    residual0.mean() - residual.mean()
                ), "mse": mse,
                "mse_percent_change": 100.0 * (mse / baseline_mse - 1.0),
            })
        scene_ps = np.asarray(old_scene_prediction_all[row_slice], dtype=float)[starts]
        scene_logq = logq_rows[starts]
        baseline_gap = np.add.reduceat(residual0, starts)
        scene_gaps = {
            "V2.2": baseline_gap,
            "Old $P_s$ correction": baseline_gap - np.add.reduceat(old_correction, starts),
            "Physical $Q$ correction": baseline_gap - np.add.reduceat(proxy_correction, starts),
        }
        masks = scene_selections(scene_ps, scene_logq, q70)
        for model, gap in scene_gaps.items():
            correction = baseline_gap - gap
            for selection, mask in masks.items():
                metric_records.append({
                    "dataset": "half_shear_final_c20_39", "model": model,
                    "case": case, "selection": selection,
                    "n_items": int(mask.sum()), "residual": float(gap[mask].mean()),
                    "correction": float(correction[mask].mean()),
                    "mse": np.nan, "mse_percent_change": np.nan,
                })
        half_scene_case.append(np.full(len(starts), case, dtype=np.int16))
        half_scene_logq.append(scene_logq)
        for model in MODELS:
            half_scene_gaps[model].append(scene_gaps[model])
        pair_cases.append(np.full(len(primary), case, dtype=np.int16))
        pair_x.append(prediction)
        pair_res0.append(pair_values["V2.2"])
        pair_res_old.append(pair_values["Old $P_s$ correction"])
        pair_res_proxy.append(pair_values["Physical $Q$ correction"])
        print(f"half-shear case {case}: pairs={len(primary):,}", flush=True)

    pair_case = np.concatenate(pair_cases)
    pair_coordinate = np.concatenate(pair_x)
    pair_edges = np.quantile(pair_coordinate, np.linspace(0.0, 1.0, 21))
    pair_edges[0], pair_edges[-1] = -np.inf, np.inf
    pair_curve = case_curve(
        pair_case, pair_coordinate,
        {
            "V2.2": np.concatenate(pair_res0),
            "Old $P_s$ correction": np.concatenate(pair_res_old),
            "Physical $Q$ correction": np.concatenate(pair_res_proxy),
        }, pair_edges, "half_shear_final_c20_39", "pair",
    )
    half_scene_curve = case_curve(
        np.concatenate(half_scene_case), np.concatenate(half_scene_logq),
        {model: np.concatenate(half_scene_gaps[model]) for model in MODELS},
        q_edges, "half_shear_final_c20_39", "scene",
    )

    old_parts, proxy_parts = [], []
    old_dir = Path(args.old_anchor_score_dir)
    proxy_dir = Path(args.proxy_anchor_score_dir)
    for case in range(400, 900):
        old_parts.append(pd.read_feather(old_dir / f"case{case}.feather"))
        proxy_parts.append(pd.read_feather(proxy_dir / f"case{case}.feather"))
    old_score = pd.concat(old_parts, ignore_index=True)[
        ["case", "input_index", "R_blend_grouped"]
    ]
    proxy_score = pd.concat(proxy_parts, ignore_index=True)[[
        "case", "input_index", "log10_scene_proxy_q_d2",
        "R_blend_v22_replay", "R_blend_proxy_corrected",
    ]]
    anchor = pd.read_feather(
        args.anchor_features,
        columns=["case", "input_index", "R_blend_truth", "scene_prediction"],
    )
    joined = anchor.merge(
        old_score, on=["case", "input_index"], how="left", validate="one_to_one"
    ).merge(
        proxy_score, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    if len(joined) != 1_703_884 or joined.isna().any().any():
        raise RuntimeError("anchor score join does not match frozen truth population")
    replay = float(np.max(np.abs(
        joined.R_blend_v22_replay - joined.scene_prediction
    )))
    if replay > 2.0e-6:
        raise RuntimeError(f"anchor V2.2 replay differs by {replay:.3e}")
    joined.loc[
        ~np.isfinite(joined.log10_scene_proxy_q_d2), "log10_scene_proxy_q_d2"
    ] = q_edges[1] - 1.0
    anchor_gaps = {
        "V2.2": (joined.R_blend_truth - joined.scene_prediction).to_numpy(float),
        "Old $P_s$ correction": (
            joined.R_blend_truth - joined.R_blend_grouped
        ).to_numpy(float),
        "Physical $Q$ correction": (
            joined.R_blend_truth - joined.R_blend_proxy_corrected
        ).to_numpy(float),
    }
    for case, local in joined.groupby("case", sort=True):
        index = local.index.to_numpy(np.int64)
        ps = local.scene_prediction.to_numpy(float)
        logq = local.log10_scene_proxy_q_d2.to_numpy(float)
        baseline_gap = anchor_gaps["V2.2"][index]
        masks = scene_selections(ps, logq, q70)
        for model in MODELS:
            gap = anchor_gaps[model][index]
            correction = baseline_gap - gap
            for selection, mask in masks.items():
                metric_records.append({
                    "dataset": "coherent_anchor_c400_899", "model": model,
                    "case": int(case), "selection": selection,
                    "n_items": int(mask.sum()), "residual": float(gap[mask].mean()),
                    "correction": float(correction[mask].mean()),
                    "mse": np.nan, "mse_percent_change": np.nan,
                })
    anchor_curve = case_curve(
        joined.case.to_numpy(np.int16),
        joined.log10_scene_proxy_q_d2.to_numpy(float), anchor_gaps,
        q_edges, "coherent_anchor_c400_899", "scene",
    )
    scene_curve = pd.concat([half_scene_curve, anchor_curve], ignore_index=True)
    metric_frame = pd.DataFrame(metric_records)
    summary = summarize_case_metrics(metric_frame)
    payload = {
        "schema_version": 1,
        "experiment": "V2.2 physical full-scene-proxy pair-residual correction transfer",
        "models": list(MODELS),
        "metrics": summary,
        "proxy_definition": proxy_summary["scene_proxy"],
        "uncertainty": "one SEM across rendered-case means",
        "audit": {
            "anchor_v22_replay_max_abs": replay,
            "n_anchor_rows": int(len(joined)),
            "n_anchor_cases": int(joined.case.nunique()),
        },
        "provenance": {
            "proxy_training_cases": [40, 199],
            "half_shear_validation_cases": [20, 39],
            "coherent_anchor_transfer_cases": [400, 899],
            "hyperparameter_tuning_performed": False,
            "constgold_opened": False,
        },
    }
    metric_frame.to_csv(output.with_suffix(".case_metrics.csv"), index=False)
    pair_curve.to_csv(output.with_suffix(".pair_curve.csv"), index=False)
    scene_curve.to_csv(output.with_suffix(".scene_curve.csv"), index=False)
    strict_json(output.with_suffix(".json"), payload)
    make_figure(pair_curve, scene_curve, summary, output)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("V22_PROXY_CORRECTION_TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
