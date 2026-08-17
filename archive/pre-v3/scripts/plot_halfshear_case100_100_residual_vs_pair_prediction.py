#!/usr/bin/env python3
"""Plot 100/100 model pair residuals versus each model's pair prediction."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from scripts.plot_halfshear_case100_100_residual_vs_measured_scene import (
    CASE_MAX,
    CASE_MIN,
    MODELS,
    N_BINS,
    atomic_csv,
    atomic_json,
    configure_style,
    finite_stat,
    load_stack,
)
from scripts.train_newbase_oldway_fullneighbour_160_40 import (
    build_full_base_context,
    load_full_metadata,
    physical_base_prediction,
    scene_index,
)
from scripts.v22_grouped_rscene_common import (
    conditional_matrix,
    load_source_metadata,
    mmap_array,
    sha256,
)


PREDICTION_COLUMNS = {
    "V2.2": "prediction_v22",
    "Raw 100/100 base": "prediction_raw",
    "100/100 base + correction": "prediction_corrected",
}


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite pair prediction")
    inner = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    if len(inner) != n_bins - 1:
        raise RuntimeError("pair prediction has tied quantile edges")
    return np.r_[-np.inf, inner, np.inf]


def summarize_model(
    case: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    model: str,
) -> tuple[list[dict], np.ndarray]:
    """Return case-balanced residual curve in own-prediction quantile bins."""
    case = np.asarray(case, dtype=np.int64)
    label = np.asarray(label, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if not (len(case) == len(label) == len(prediction)):
        raise RuntimeError("pair arrays have inconsistent lengths")
    local_case = case - CASE_MIN
    if local_case.min() != 0 or local_case.max() != CASE_MAX - CASE_MIN:
        raise RuntimeError("unexpected case range")
    edges = quantile_edges(prediction, N_BINS)
    bin_index = np.searchsorted(edges[1:-1], prediction, side="right")
    flat = local_case * N_BINS + bin_index
    length = (CASE_MAX - CASE_MIN + 1) * N_BINS
    counts = np.bincount(flat, minlength=length).reshape(-1, N_BINS)
    prediction_sums = np.bincount(
        flat, weights=prediction, minlength=length
    ).reshape(-1, N_BINS)
    residual_sums = np.bincount(
        flat, weights=label - prediction, minlength=length
    ).reshape(-1, N_BINS)
    rows: list[dict] = []
    for index in range(N_BINS):
        present = counts[:, index] > 0
        if present.sum() < 2:
            raise RuntimeError(f"pair-prediction bin {index} has too few cases")
        local_counts = counts[present, index]
        case_prediction = prediction_sums[present, index] / local_counts
        case_residual = residual_sums[present, index] / local_counts
        prediction_stat = finite_stat(case_prediction)
        residual_stat = finite_stat(case_residual)
        cell_prediction = prediction[bin_index == index]
        rows.append({
            "model": model,
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "prediction_median": float(np.median(cell_prediction)),
            "prediction_case_mean": float(prediction_stat["mean"]),
            "prediction_case_sem": float(prediction_stat["case_sem"]),
            "residual_mean": float(residual_stat["mean"]),
            "residual_case_sem": float(residual_stat["case_sem"]),
            "residual_case_sd": float(residual_stat["case_sd"]),
            "n_pairs": int(local_counts.sum()),
            "n_cases": int(present.sum()),
        })
    return rows, edges


def global_pair_stat(
    case: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    frame = pd.DataFrame({
        "case": np.asarray(case, dtype=np.int16),
        "residual": np.asarray(label, dtype=np.float64)
        - np.asarray(prediction, dtype=np.float64),
    })
    return finite_stat(
        frame.groupby("case", sort=True).residual.mean().to_numpy(float)
    )


def plot_panel(curves: pd.DataFrame, stem: Path, n_pairs: int) -> None:
    configure_style()
    styles = {
        "V2.2": ("#0072B2", "o", "-", "V2.2"),
        "Raw 100/100 base": (
            "#009E73", "^", "-.", "Raw 100/100 base",
        ),
        "100/100 base + correction": (
            "#D55E00", "s", "--", "100/100 base + correction",
        ),
    }
    fig, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    for model, _ in MODELS:
        line = curves.loc[curves.model == model].sort_values("bin")
        color, marker, linestyle, label = styles[model]
        axis.errorbar(
            line.prediction_median,
            line.residual_mean,
            yerr=line.residual_case_sem,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.15,
            markersize=4.0,
            capsize=2.2,
            label=label,
        )
    axis.axhline(0.0, color="0.60", linestyle=":", linewidth=0.7)
    axis.set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
    axis.set_xlabel(r"Each model's pair prediction $R_{\rm blend}$")
    axis.set_ylabel("Mean pair residual (label - prediction)")
    axis.set_title(
        "100/100 model pair calibration on half-shear cases 100–199\n"
        f"n={n_pairs:,} labelled pairs; error bars: one case SEM"
    )
    axis.legend(frameon=False, loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    for suffix, options in ((".png", {"dpi": 300}), (".pdf", {})):
        output = stem.with_suffix(suffix)
        if output.exists():
            raise FileExistsError(output)
        fig.savefig(output, bbox_inches="tight", **options)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--full-neighbour-cache", required=True)
    parser.add_argument("--stack-summary", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    full_cache = Path(args.full_neighbour_cache).resolve()
    summary_path = Path(args.stack_summary).resolve()
    stem = Path(args.output_stem).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".csv", ".json", ".png", ".pdf"):
        if stem.with_suffix(suffix).exists():
            raise FileExistsError(stem.with_suffix(suffix))

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    summary, base, correction = load_stack(summary_path, n_jobs)
    if summary["training"]["base"]["case_window"] != [0, 99]:
        raise RuntimeError("stack base is not the requested c0--99 model")
    if summary["training"]["correction"]["case_window"] != [100, 199]:
        raise RuntimeError("stack correction is not the requested c100--199 model")

    source_meta = load_source_metadata(source_cache)
    full_meta = load_full_metadata(full_cache, source_cache)
    case = mmap_array(source_cache, source_meta, "case")
    primary = mmap_array(source_cache, source_meta, "input_index")
    label = mmap_array(source_cache, source_meta, "label")
    v22_prediction = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    starts, counts, _, scene_offsets = scene_index(case, primary)
    selected = np.flatnonzero(
        (np.asarray(case) >= CASE_MIN) & (np.asarray(case) <= CASE_MAX)
    ).astype(np.int32)
    expected_rows = int(summary["training"]["correction"]["n_rows"])
    if len(selected) != expected_rows:
        raise RuntimeError(f"selected row mismatch: {len(selected)} != {expected_rows}")

    target_mean = float(summary["source"]["target_mean"])
    target_scale = float(summary["source"]["target_scale"])
    all_base_prediction = physical_base_prediction(
        base, x_scaled, target_mean, target_scale
    )
    full_scene_base, full_scene_count = build_full_base_context(
        base=base,
        full_cache=full_cache,
        full_meta=full_meta,
        source_primary=primary,
        source_scene_starts=starts,
        source_scene_offsets=scene_offsets,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    row_scene_base = np.repeat(full_scene_base, counts)
    row_scene_count = np.repeat(full_scene_count, counts).astype(np.int32)
    features = conditional_matrix(
        x_scaled,
        x_raw,
        all_base_prediction,
        row_scene_base,
        row_scene_count,
        selected,
    )
    correction_pair = (
        correction.inplace_predict(features).astype(np.float64) * target_scale
    )
    del features, row_scene_base, row_scene_count
    gc.collect()

    selected_case = np.asarray(case[selected], dtype=np.int16)
    selected_label = np.asarray(label[selected], dtype=np.float64)
    predictions = {
        "V2.2": np.asarray(v22_prediction[selected], dtype=np.float64),
        "Raw 100/100 base": np.asarray(
            all_base_prediction[selected], dtype=np.float64
        ),
    }
    predictions["100/100 base + correction"] = (
        predictions["Raw 100/100 base"] + correction_pair
    )
    del correction_pair, all_base_prediction
    gc.collect()
    if selected_case.min() != CASE_MIN or selected_case.max() != CASE_MAX:
        raise RuntimeError("incomplete half-shear case coverage")
    if not np.isfinite(selected_label).all() or any(
        not np.isfinite(value).all() for value in predictions.values()
    ):
        raise RuntimeError("non-finite pair label or prediction")

    rows: list[dict] = []
    edge_payload: dict[str, list[float]] = {}
    global_stats: dict[str, dict[str, float | int]] = {}
    for model, prediction in predictions.items():
        model_rows, edges = summarize_model(
            selected_case, selected_label, prediction, model
        )
        rows.extend(model_rows)
        edge_payload[model] = [float(value) for value in edges[1:-1]]
        global_stats[model] = global_pair_stat(
            selected_case, selected_label, prediction
        )
        print(f"summarized {model}", flush=True)
    curves = pd.DataFrame(rows)
    if len(curves) != len(MODELS) * N_BINS:
        raise RuntimeError("unexpected curve row count")

    atomic_csv(stem.with_suffix(".csv"), curves)
    plot_panel(curves, stem, len(selected))
    payload = {
        "schema_version": 1,
        "dataset": "half_shear_c100_199_correction_fit_block",
        "n_pairs": int(len(selected)),
        "n_cases": int(np.unique(selected_case).size),
        "residual_definition": "pair label - matching pair prediction",
        "conditioning": (
            "each model uses its own final pair prediction as the x coordinate"
        ),
        "fit_status": (
            "base is case-out-of-sample on c100--199; correction is fit "
            "in-sample on c100--199"
        ),
        "uncertainty": "one SEM across per-case conditional pair means",
        "binning": {
            "n_bins": N_BINS,
            "method": "separate equal-count quantile bins for each model",
            "internal_edges": edge_payload,
            "labels_used_to_define_bins": False,
        },
        "global_case_balanced_pair_residual": global_stats,
        "audit": {
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
        "provenance": {
            "source_cache": str(source_cache),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "full_neighbour_cache": str(full_cache),
            "full_neighbour_metadata_sha256": sha256(
                full_cache / "metadata.json"
            ),
            "stack_summary": str(summary_path),
            "stack_summary_sha256": sha256(summary_path),
        },
        "outputs": {
            "curves_csv": str(stem.with_suffix(".csv")),
            "figure_png": str(stem.with_suffix(".png")),
            "figure_pdf": str(stem.with_suffix(".pdf")),
        },
    }
    atomic_json(stem.with_suffix(".json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("PLOT_HALFSHEAR_CASE100_100_PAIR_PREDICTION_DONE", flush=True)


if __name__ == "__main__":
    main()
