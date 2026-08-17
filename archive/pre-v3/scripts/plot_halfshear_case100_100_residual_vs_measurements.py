#!/usr/bin/env python3
"""Plot pair and scene residuals versus measured half-shear responses."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.plot_halfshear_case100_100_residual_vs_measured_scene import (
    CASE_MAX,
    CASE_MIN,
    N_BINS,
    atomic_csv,
    atomic_json,
    configure_style,
    finite_stat,
)
from scripts.train_newbase_oldway_fullneighbour_160_40 import (
    build_full_base_context,
    load_full_metadata,
    physical_base_prediction,
    scene_index,
)
from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
)
from scripts.v22_grouped_rscene_common import (
    conditional_matrix,
    load_source_metadata,
    mmap_array,
    sha256,
)


MODELS = (
    "V2.2",
    "Raw 100/100 base",
    "Ordinary pair-MSE correction",
    "Binned scene-MSE correction",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["features"]["base"] != BASE_FEATURES:
        raise RuntimeError(f"base feature drift in {path}")
    if summary["features"]["correction"] != CONDITIONAL_FEATURES:
        raise RuntimeError(f"correction feature drift in {path}")
    if summary["training"]["base"]["case_window"] != [0, 99]:
        raise RuntimeError(f"wrong base window in {path}")
    if summary["training"]["correction"]["case_window"] != [100, 199]:
        raise RuntimeError(f"wrong correction window in {path}")
    deployment = summary["deployment_stack"]
    for key, hash_key in (
        ("base_model", "base_sha256"),
        ("correction_model", "correction_sha256"),
    ):
        model_path = Path(deployment[key])
        if file_sha256(model_path) != deployment[hash_key]:
            raise RuntimeError(f"model hash mismatch for {model_path}")
    return summary


def load_booster(path: str, n_jobs: int) -> xgb.Booster:
    booster = xgb.Booster({"device": "cpu", "n_jobs": n_jobs})
    booster.load_model(path)
    return booster


def quantile_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite measured response")
    inner = np.unique(np.quantile(values, np.arange(1, N_BINS) / N_BINS))
    if len(inner) != N_BINS - 1:
        raise RuntimeError("measured-response quantile edges are tied")
    return np.r_[-np.inf, inner, np.inf]


def conditional_rows(
    panel: str,
    case: np.ndarray,
    measurement: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[list[dict], np.ndarray]:
    """Case-balanced residual summaries in common measured-response bins."""
    case = np.asarray(case, dtype=np.int64)
    measurement = np.asarray(measurement, dtype=np.float64)
    n_case = CASE_MAX - CASE_MIN + 1
    local_case = case - CASE_MIN
    if local_case.min() != 0 or local_case.max() != n_case - 1:
        raise RuntimeError(f"incomplete case range for {panel}")
    edges = quantile_edges(measurement)
    bin_index = np.searchsorted(edges[1:-1], measurement, side="right")
    flat = local_case * N_BINS + bin_index
    length = n_case * N_BINS
    counts = np.bincount(flat, minlength=length).reshape(n_case, N_BINS)
    measurement_sums = np.bincount(
        flat, weights=measurement, minlength=length
    ).reshape(n_case, N_BINS)
    rows: list[dict] = []
    for model in MODELS:
        prediction = np.asarray(predictions[model], dtype=np.float64)
        if len(prediction) != len(measurement):
            raise RuntimeError(f"prediction length mismatch for {model}")
        residual_sums = np.bincount(
            flat, weights=measurement - prediction, minlength=length
        ).reshape(n_case, N_BINS)
        for index in range(N_BINS):
            present = counts[:, index] > 0
            if present.sum() < 2:
                raise RuntimeError(f"too few cases in {panel} bin {index}")
            local_counts = counts[present, index]
            case_measurement = (
                measurement_sums[present, index] / local_counts
            )
            case_residual = residual_sums[present, index] / local_counts
            measurement_stat = finite_stat(case_measurement)
            residual_stat = finite_stat(case_residual)
            cell = measurement[bin_index == index]
            rows.append({
                "panel": panel,
                "model": model,
                "bin": int(index),
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "measurement_median": float(np.median(cell)),
                "measurement_case_mean": float(measurement_stat["mean"]),
                "measurement_case_sem": float(measurement_stat["case_sem"]),
                "residual_mean": float(residual_stat["mean"]),
                "residual_case_sem": float(residual_stat["case_sem"]),
                "residual_case_sd": float(residual_stat["case_sd"]),
                "n_objects": int(local_counts.sum()),
                "n_cases": int(present.sum()),
            })
    return rows, edges


def global_stats(
    case: np.ndarray,
    measurement: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    frame = pd.DataFrame({
        "case": np.asarray(case, dtype=np.int16),
        **{
            model: np.asarray(measurement, dtype=np.float64)
            - np.asarray(prediction, dtype=np.float64)
            for model, prediction in predictions.items()
        },
    })
    by_case = frame.groupby("case", sort=True)[list(MODELS)].mean()
    return {
        model: finite_stat(by_case[model].to_numpy(float))
        for model in MODELS
    }


def plot(curves: pd.DataFrame, stem: Path, n_pairs: int, n_scenes: int) -> None:
    configure_style()
    styles = {
        "V2.2": ("#0072B2", "o", "-"),
        "Raw 100/100 base": ("#009E73", "^", "-."),
        "Ordinary pair-MSE correction": ("#D55E00", "s", "--"),
        "Binned scene-MSE correction": ("#CC79A7", "D", ":"),
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(10.0, 3.8), constrained_layout=True
    )
    panels = (
        (
            "pair",
            axes[0],
            r"Measured pair response $R_i$",
            "Mean pair residual (label - prediction)",
            f"Pair level (n={n_pairs:,})",
        ),
        (
            "scene",
            axes[1],
            "Measured sheared-neighbour scene response",
            "Mean scene residual (label sum - prediction sum)",
            f"Scene level (n={n_scenes:,})",
        ),
    )
    for panel, axis, xlabel, ylabel, title in panels:
        local = curves.loc[curves.panel == panel]
        for model in MODELS:
            line = local.loc[local.model == model].sort_values("bin")
            color, marker, linestyle = styles[model]
            axis.errorbar(
                line.measurement_median,
                line.residual_mean,
                yerr=line.residual_case_sem,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.15,
                markersize=3.8,
                capsize=2.0,
                label=model,
            )
        axis.axhline(0.0, color="0.60", linestyle=":", linewidth=0.7)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].text(
        -0.14, 1.08, "A", transform=axes[0].transAxes,
        fontsize=11, fontweight="bold", va="top",
    )
    axes[1].text(
        -0.14, 1.08, "B", transform=axes[1].transAxes,
        fontsize=11, fontweight="bold", va="top",
    )
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(
        "Residuals conditioned on measured half-shear response, cases 100–199\n"
        "common label-defined quantile bins; error bars: one SEM across cases",
        fontsize=10,
    )
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
    parser.add_argument("--ordinary-summary", required=True)
    parser.add_argument("--binned-scene-summary", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    full_cache = Path(args.full_neighbour_cache).resolve()
    ordinary_path = Path(args.ordinary_summary).resolve()
    binned_path = Path(args.binned_scene_summary).resolve()
    stem = Path(args.output_stem).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".csv", ".json", ".png", ".pdf"):
        if stem.with_suffix(suffix).exists():
            raise FileExistsError(stem.with_suffix(suffix))

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    ordinary = load_summary(ordinary_path)
    binned = load_summary(binned_path)
    if ordinary["deployment_stack"]["base_sha256"] != binned[
        "deployment_stack"
    ]["base_sha256"]:
        raise RuntimeError("the two stacks do not share the same frozen base")
    if ordinary["source"]["target_mean"] != binned["source"]["target_mean"]:
        raise RuntimeError("target mean differs between stacks")
    if ordinary["source"]["target_scale"] != binned["source"]["target_scale"]:
        raise RuntimeError("target scale differs between stacks")

    base = load_booster(ordinary["deployment_stack"]["base_model"], n_jobs)
    ordinary_correction = load_booster(
        ordinary["deployment_stack"]["correction_model"], n_jobs
    )
    binned_correction = load_booster(
        binned["deployment_stack"]["correction_model"], n_jobs
    )
    source_meta = load_source_metadata(source_cache)
    full_meta = load_full_metadata(full_cache, source_cache)
    case = mmap_array(source_cache, source_meta, "case")
    primary = mmap_array(source_cache, source_meta, "input_index")
    label = mmap_array(source_cache, source_meta, "label")
    v22 = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    starts, counts, scene_case, scene_offsets = scene_index(case, primary)
    selected = np.flatnonzero(
        (np.asarray(case) >= CASE_MIN) & (np.asarray(case) <= CASE_MAX)
    ).astype(np.int32)
    expected = int(ordinary["training"]["correction"]["n_rows"])
    if len(selected) != expected or len(selected) != int(
        binned["training"]["correction"]["n_rows"]
    ):
        raise RuntimeError("selected pair count does not match model summaries")

    target_mean = float(ordinary["source"]["target_mean"])
    target_scale = float(ordinary["source"]["target_scale"])
    base_all = physical_base_prediction(
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
    features = conditional_matrix(
        x_scaled,
        x_raw,
        base_all,
        np.repeat(full_scene_base, counts),
        np.repeat(full_scene_count, counts).astype(np.int32),
        selected,
    )
    ordinary_delta = (
        ordinary_correction.inplace_predict(features).astype(np.float64)
        * target_scale
    )
    binned_delta = (
        binned_correction.inplace_predict(features).astype(np.float64)
        * target_scale
    )
    del features
    gc.collect()

    pair_case = np.asarray(case[selected], dtype=np.int16)
    pair_truth = np.asarray(label[selected], dtype=np.float64)
    pair_predictions = {
        "V2.2": np.asarray(v22[selected], dtype=np.float64),
        "Raw 100/100 base": np.asarray(base_all[selected], dtype=np.float64),
    }
    pair_predictions["Ordinary pair-MSE correction"] = (
        pair_predictions["Raw 100/100 base"] + ordinary_delta
    )
    pair_predictions["Binned scene-MSE correction"] = (
        pair_predictions["Raw 100/100 base"] + binned_delta
    )

    scene_mask = (scene_case >= CASE_MIN) & (scene_case <= CASE_MAX)
    scene_counts = np.asarray(counts[scene_mask], dtype=np.int64)
    local_starts = np.r_[0, np.cumsum(scene_counts[:-1])]
    if int(scene_counts.sum()) != len(pair_truth):
        raise RuntimeError("selected scene counts do not close to pair rows")
    scene_truth = np.add.reduceat(pair_truth, local_starts)
    scene_predictions = {
        model: np.add.reduceat(prediction, local_starts)
        for model, prediction in pair_predictions.items()
    }
    selected_scene_case = np.asarray(scene_case[scene_mask], dtype=np.int16)

    pair_rows, pair_edges = conditional_rows(
        "pair", pair_case, pair_truth, pair_predictions
    )
    scene_rows, scene_edges = conditional_rows(
        "scene", selected_scene_case, scene_truth, scene_predictions
    )
    curves = pd.DataFrame(pair_rows + scene_rows)
    atomic_csv(stem.with_suffix(".csv"), curves)
    plot(curves, stem, len(pair_truth), len(scene_truth))

    payload = {
        "schema_version": 1,
        "dataset": "half_shear_c100_199_correction_fit_block",
        "n_pairs": int(len(pair_truth)),
        "n_scenes": int(len(scene_truth)),
        "n_cases": int(np.unique(pair_case).size),
        "models": list(MODELS),
        "definitions": {
            "pair": "x=pair label; y=pair label minus pair prediction",
            "scene": (
                "x=sum of labelled/sheared-neighbour pair labels per primary; "
                "y=that sum minus the sum of predictions for those same pairs; "
                "not doubled and excludes the unsheared neighbour half"
            ),
            "conditioning_warning": (
                "the noisy measurement appears in both x and residual y, "
                "which induces a strong mechanical correlation"
            ),
            "fit_status": (
                "base is case-out-of-sample on c100--199; both corrections "
                "are in-sample on their c100--199 fit block"
            ),
            "uncertainty": "one SEM across per-case conditional means",
        },
        "binning": {
            "n_quantile_bins": N_BINS,
            "common_bins_across_models": True,
            "labels_used_to_define_bins": True,
            "pair_internal_edges": [float(x) for x in pair_edges[1:-1]],
            "scene_internal_edges": [float(x) for x in scene_edges[1:-1]],
        },
        "neighbour_counts": {
            "mean_sheared_labelled": float(scene_counts.mean()),
            "mean_all_context": float(full_scene_count[scene_mask].mean()),
        },
        "global_case_balanced_residual": {
            "pair": global_stats(pair_case, pair_truth, pair_predictions),
            "scene": global_stats(
                selected_scene_case, scene_truth, scene_predictions
            ),
        },
        "audit": {
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
            "shared_base_sha256": ordinary["deployment_stack"]["base_sha256"],
        },
        "provenance": {
            "source_cache": str(source_cache),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "full_neighbour_cache": str(full_cache),
            "full_neighbour_metadata_sha256": sha256(
                full_cache / "metadata.json"
            ),
            "ordinary_summary": str(ordinary_path),
            "ordinary_summary_sha256": sha256(ordinary_path),
            "binned_scene_summary": str(binned_path),
            "binned_scene_summary_sha256": sha256(binned_path),
        },
        "outputs": {
            "curves_csv": str(stem.with_suffix(".csv")),
            "figure_png": str(stem.with_suffix(".png")),
            "figure_pdf": str(stem.with_suffix(".pdf")),
        },
    }
    atomic_json(stem.with_suffix(".json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("PLOT_HALFSHEAR_CASE100_100_RESIDUAL_VS_MEASUREMENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
