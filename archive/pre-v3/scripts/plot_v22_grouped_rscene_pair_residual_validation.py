#!/usr/bin/env python3
"""Plot half-shear pair residual versus frozen V2.2 pair prediction.

The original V2.2 prediction and the frozen scene-informed pair correction are
compared in the same 20 V2.2-prediction quantile bins on external half-shear
cases 20--39.  The x coordinate is deliberately shared: it is always the
frozen V2.2 pair prediction, so vertical movement shows only the learned
correction.  Labels never define bins.
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

from scripts.plot_v22_emulator_label_calibration import quantile_edges
from scripts.v22_grouped_rscene_common import (
    CONDITIONAL_FEATURES,
    EXTERNAL_FINAL_CASE_MAX,
    EXTERNAL_FINAL_CASE_MIN,
    case_offsets,
    conditional_matrix,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    physical_correction,
    sha256,
    strict_json,
)


MODEL_ORDER = ("V2.2", "Scene-informed correction")
COLORS = {"V2.2": "#0072B2", "Scene-informed correction": "#D55E00"}
MARKERS = {"V2.2": "o", "Scene-informed correction": "s"}
LINESTYLES = {"V2.2": "--", "Scene-informed correction": "-"}


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


def summarize_shared_bins(
    case: np.ndarray,
    label: np.ndarray,
    baseline_prediction: np.ndarray,
    corrected_prediction: np.ndarray,
    edges: np.ndarray,
    case_min: int,
    case_max: int,
) -> dict[str, Any]:
    """Summarize both models in common baseline-prediction bins."""
    case = np.asarray(case, dtype=np.int64)
    label = np.asarray(label, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    corrected = np.asarray(corrected_prediction, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if not (case.shape == label.shape == baseline.shape == corrected.shape):
        raise ValueError("pair arrays have different shapes")
    if case.ndim != 1 or len(case) == 0:
        raise ValueError("pair arrays must be non-empty and one-dimensional")
    if not np.isfinite(np.column_stack((label, baseline, corrected))).all():
        raise ValueError("pair arrays contain non-finite values")
    if np.any((case < case_min) | (case > case_max)):
        raise ValueError("case lies outside the requested validation window")
    if edges.ndim != 1 or len(edges) < 3 or np.any(np.diff(edges) <= 0):
        raise ValueError("prediction edges are not strictly increasing")

    n_cases = case_max - case_min + 1
    n_bins = len(edges) - 1
    bin_index = np.searchsorted(edges[1:-1], baseline, side="right")
    case_index = case - case_min
    flat = case_index * n_bins + bin_index
    minlength = n_cases * n_bins

    def sums(values: np.ndarray) -> np.ndarray:
        return np.bincount(
            flat, weights=values, minlength=minlength
        ).reshape(n_cases, n_bins)

    counts = np.bincount(flat, minlength=minlength).reshape(n_cases, n_bins)
    label_sum = sums(label)
    baseline_sum = sums(baseline)
    corrected_sum = sums(corrected)
    if np.any(counts.sum(axis=1) == 0):
        raise RuntimeError("validation case has no supported pairs")

    prediction_sums = {
        "V2.2": baseline_sum,
        "Scene-informed correction": corrected_sum,
    }
    pooled_count = int(counts.sum())
    models: dict[str, Any] = {}
    for model in MODEL_ORDER:
        prediction_sum = prediction_sums[model]
        bins = []
        for index in range(n_bins):
            present = counts[:, index] > 0
            if int(present.sum()) < 2:
                raise RuntimeError(f"bin {index} occurs in fewer than two cases")
            local_count = counts[present, index]
            case_label = label_sum[present, index] / local_count
            case_baseline = baseline_sum[present, index] / local_count
            case_prediction = prediction_sum[present, index] / local_count
            bins.append({
                "bin": int(index),
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "n_pairs": int(local_count.sum()),
                "pair_fraction": float(local_count.sum() / pooled_count),
                "n_cases_with_pairs": int(present.sum()),
                "v22_coordinate": finite_stat(case_baseline),
                "label": finite_stat(case_label),
                "prediction": finite_stat(case_prediction),
                "label_minus_prediction": finite_stat(
                    case_label - case_prediction
                ),
                "applied_correction": finite_stat(
                    case_prediction - case_baseline
                ),
            })
        total_count = counts.sum(axis=1)
        case_label = label_sum.sum(axis=1) / total_count
        case_baseline = baseline_sum.sum(axis=1) / total_count
        case_prediction = prediction_sum.sum(axis=1) / total_count
        models[model] = {
            "bins": bins,
            "global_pair_label": finite_stat(case_label),
            "global_pair_prediction": finite_stat(case_prediction),
            "global_pair_label_minus_prediction": finite_stat(
                case_label - case_prediction
            ),
            "global_applied_correction": finite_stat(
                case_prediction - case_baseline
            ),
        }
    return {
        "prediction_edges": edges.tolist(),
        "n_pairs": pooled_count,
        "n_cases": n_cases,
        "models": models,
        "maximum_bin_fraction_deviation_from_equal": float(
            np.max(np.abs(counts.sum(axis=0) / pooled_count - 1.0 / n_bins))
        ),
    }


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def make_plot(payload: dict[str, Any], output_prefix: Path) -> None:
    configure_style()
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    handles = []
    labels = []
    lower_values = []
    upper_values = []
    x_values = []
    for model in MODEL_ORDER:
        summary = payload["summary"]["models"][model]
        bins = summary["bins"]
        x = np.asarray([item["v22_coordinate"]["mean"] for item in bins])
        residual = np.asarray([
            item["label_minus_prediction"]["mean"] for item in bins
        ])
        sem = np.asarray([
            item["label_minus_prediction"]["case_sem"] for item in bins
        ])
        handle = axis.errorbar(
            x,
            residual,
            yerr=sem,
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            markersize=4.2,
            linewidth=1.25,
            capsize=1.8,
            alpha=0.95,
            zorder=4 if model != "V2.2" else 3,
        )
        global_residual = summary["global_pair_label_minus_prediction"]["mean"]
        handles.append(handle)
        labels.append(f"{model}  (global {global_residual:+.5f})")
        x_values.append(x)
        lower_values.append(residual - sem)
        upper_values.append(residual + sem)

    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9, zorder=2)
    axis.set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
    all_x = np.concatenate(x_values)
    x_pad = 0.05 * max(float(all_x.max() - all_x.min()), 1.0e-3)
    axis.set_xlim(float(all_x.min() - x_pad), float(all_x.max() + x_pad))
    lower = float(np.min(np.concatenate(lower_values)))
    upper = float(np.max(np.concatenate(upper_values)))
    y_pad = 0.08 * max(upper - lower, 1.0e-3)
    axis.set_ylim(lower - y_pad, upper + y_pad)
    axis.set_xlabel(r"Frozen V2.2 pair prediction $p$")
    axis.set_ylabel("Mean pair label − prediction")
    axis.spines[["top", "right"]].set_visible(False)

    baseline_bins = payload["summary"]["models"]["V2.2"]["bins"]
    edges = np.asarray(payload["summary"]["prediction_edges"], dtype=float)
    fractions = np.asarray(
        [item["pair_fraction"] for item in baseline_bins], dtype=float
    )
    transformed_edges = axis.xaxis.get_transform().transform(edges)
    transformed_widths = np.diff(transformed_edges)
    if np.any(transformed_widths <= 0.0):
        raise RuntimeError("prediction histogram edges are invalid")
    density = fractions / transformed_widths
    density /= density.max()
    histogram_axis = axis.twinx()
    histogram = histogram_axis.stairs(
        density,
        edges,
        baseline=0.0,
        fill=True,
        color="0.58",
        alpha=0.22,
        linewidth=0.7,
        label="V2.2 prediction density",
    )
    histogram_axis.set_ylim(0.0, 1.65)
    histogram_axis.set_yticks([])
    histogram_axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    histogram_axis.set_zorder(axis.get_zorder() - 1)
    axis.patch.set_alpha(0.0)

    axis.legend(
        [*handles, histogram],
        [*labels, "V2.2 prediction density"],
        title=r"Model  (global $\langle y-p\rangle$)",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        borderaxespad=0.0,
        handlelength=2.5,
    )
    fig.suptitle(
        "Independent half-shear validation pairs (cases 20–39)\n"
        "20 shared V2.2-prediction bins; error bars are one SEM across rendered cases",
        fontsize=9.5,
        y=0.98,
    )
    fig.subplots_adjust(left=0.105, right=0.70, bottom=0.16, top=0.79)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def flatten_csv(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for item in payload["summary"]["models"][model]["bins"]:
            rows.append({
                "model": model,
                "bin": item["bin"],
                "lower": item["lower"],
                "upper": item["upper"],
                "n_pairs": item["n_pairs"],
                "pair_fraction": item["pair_fraction"],
                "v22_prediction_mean": item["v22_coordinate"]["mean"],
                "v22_prediction_case_sem": item["v22_coordinate"]["case_sem"],
                "model_prediction_mean": item["prediction"]["mean"],
                "label_mean": item["label"]["mean"],
                "residual_mean": item["label_minus_prediction"]["mean"],
                "residual_case_sem": item["label_minus_prediction"]["case_sem"],
                "correction_mean": item["applied_correction"]["mean"],
                "correction_case_sem": item["applied_correction"]["case_sem"],
            })
    return pd.DataFrame(rows)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Scene-informed pair correction on half-shear validation",
        "",
        "Both curves use the same 20 quantile bins of frozen V2.2 pair prediction. "
        "The x-axis therefore stays fixed while the residual changes. Labels do not "
        "enter binning. Errors are one SEM across 20 rendered cases.",
        "",
        f"- Validation pairs: `{payload['summary']['n_pairs']:,}`.",
        "- Cases: `20--39` (external to correction-model training cases).",
        "- ConstGold opened by this plot: `false`.",
        "",
        "| model | global residual | lowest-prediction 5% | highest-prediction 5% |",
        "|---|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        summary = payload["summary"]["models"][model]
        global_stat = summary["global_pair_label_minus_prediction"]
        low = summary["bins"][0]["label_minus_prediction"]
        high = summary["bins"][-1]["label_minus_prediction"]
        lines.append(
            f"| {model} | {global_stat['mean']:+.6f} +- "
            f"{global_stat['case_sem']:.6f} | {low['mean']:+.6f} +- "
            f"{low['case_sem']:.6f} | {high['mean']:+.6f} +- "
            f"{high['case_sem']:.6f} |"
        )
    lines.extend(["", "Residual means pair label minus model prediction.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--model-summary", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-bins", type=int, default=20)
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (".json", ".csv", ".md", ".pdf", ".png")
    ]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing outputs: {existing}")

    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    if scene_meta["source_metadata_sha256"] != sha256(source_cache / "metadata.json"):
        raise RuntimeError("scene/source cache provenance differs")

    model_summary_path = Path(args.model_summary).resolve()
    with model_summary_path.open(encoding="utf-8") as handle:
        model_summary = json.load(handle)
    if model_summary["feature_names"] != CONDITIONAL_FEATURES:
        raise RuntimeError("correction-model feature definition drifted")
    if model_summary["training"]["case_window"] != [40, 199]:
        raise RuntimeError("plot requires the final c40--199 correction fit")
    model_path = Path(model_summary["model"]).resolve()
    if sha256(model_path) != model_summary["model_sha256"]:
        raise RuntimeError("correction-model hash differs from summary")
    booster = xgb.Booster({
        "device": "cpu",
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    })
    booster.load_model(model_path)

    offsets = case_offsets(source_meta)
    case_all = mmap_array(source_cache, source_meta, "case")
    label_all = mmap_array(source_cache, source_meta, "label")
    prediction_all = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction_all = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs_all = mmap_array(scene_cache, scene_meta, "scene_n_pairs")

    case_parts = []
    label_parts = []
    prediction_parts = []
    corrected_parts = []
    for case in range(EXTERNAL_FINAL_CASE_MIN, EXTERNAL_FINAL_CASE_MAX + 1):
        start, stop = int(offsets[case]), int(offsets[case + 1])
        row_slice = slice(start, stop)
        case_values = np.asarray(case_all[row_slice], dtype=np.int64)
        if not np.all(case_values == case):
            raise RuntimeError(f"case {case}: source-cache boundary mismatch")
        prediction = np.asarray(prediction_all[row_slice], dtype=np.float64)
        features = conditional_matrix(
            x_scaled,
            x_raw,
            prediction_all,
            scene_prediction_all,
            scene_n_pairs_all,
            row_slice,
        )
        correction = physical_correction(
            booster, features, float(model_summary["target_scale"])
        )
        case_parts.append(case_values.astype(np.int16))
        label_parts.append(np.asarray(label_all[row_slice], dtype=np.float64))
        prediction_parts.append(prediction)
        corrected_parts.append(prediction + correction)
        print(f"case {case}: pairs={len(prediction):,}", flush=True)

    case = np.concatenate(case_parts).astype(np.int64, copy=False)
    label = np.concatenate(label_parts)
    prediction = np.concatenate(prediction_parts)
    corrected = np.concatenate(corrected_parts)
    edges = quantile_edges(prediction, args.n_bins)
    summary = summarize_shared_bins(
        case,
        label,
        prediction,
        corrected,
        edges,
        EXTERNAL_FINAL_CASE_MIN,
        EXTERNAL_FINAL_CASE_MAX,
    )

    reference_path = Path(args.reference).resolve()
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    reference_metrics = reference["metrics"]["half_shear_final_c20_39"]
    checks = {
        "n_pairs": summary["n_pairs"]
        == reference_metrics["V2.2"]["pairs"]["n_pairs"],
        "v22_global_residual": np.isclose(
            summary["models"]["V2.2"]["global_pair_label_minus_prediction"]["mean"],
            reference_metrics["V2.2"]["pairs"]["residual"]["mean"],
            rtol=0.0,
            atol=2.0e-12,
        ),
        "corrected_global_residual": np.isclose(
            summary["models"]["Scene-informed correction"]
            ["global_pair_label_minus_prediction"]["mean"],
            reference_metrics["Grouped residual"]["pairs"]["residual"]["mean"],
            rtol=0.0,
            atol=2.0e-12,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"final-transfer reference closure failed: {checks}")

    payload = {
        "schema_version": 1,
        "design": {
            "case_window": [EXTERNAL_FINAL_CASE_MIN, EXTERNAL_FINAL_CASE_MAX],
            "validation_status": "external to correction-model training cases",
            "binning": (
                f"{len(edges) - 1} pooled quantile bins of frozen V2.2 pair "
                "prediction shared by both curves"
            ),
            "x_coordinate": "frozen V2.2 pair prediction for both models",
            "residual": "half-shear pair label minus model prediction",
            "labels_used_in_binning": False,
            "uncertainty_unit": "rendered simulation case",
            "histogram": "V2.2 prediction density in displayed symlog coordinate",
            "constgold_opened": False,
            "coherent_anchor_truth_opened": False,
        },
        "provenance": {
            "source_cache": str(source_cache),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "scene_cache": str(scene_cache),
            "scene_metadata_sha256": sha256(scene_cache / "metadata.json"),
            "model_summary": str(model_summary_path),
            "model_summary_sha256": sha256(model_summary_path),
            "model": str(model_path),
            "model_sha256": model_summary["model_sha256"],
            "reference": str(reference_path),
            "reference_checks": {name: bool(value) for name, value in checks.items()},
        },
        "summary": summary,
    }
    strict_json(output_prefix.with_suffix(".json"), payload)
    flatten_csv(payload).to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    make_plot(payload, output_prefix)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_GROUPED_RSCENE_PAIR_RESIDUAL_VALIDATION_PLOT_DONE", flush=True)


if __name__ == "__main__":
    main()
