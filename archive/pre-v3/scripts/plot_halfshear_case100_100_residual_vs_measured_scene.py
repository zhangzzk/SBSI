#!/usr/bin/env python3
"""Plot 100/100 model scene residuals versus measured half-shear scene response."""

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


CASE_MIN = 100
CASE_MAX = 199
N_BINS = 12
MODELS = (
    ("V2.2", "residual_v22"),
    ("Raw 100/100 base", "residual_raw"),
    ("100/100 base + correction", "residual_corrected"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise RuntimeError("need at least two finite case values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def case_stat(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    by_case = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    return finite_stat(by_case)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite half-shear scene response")
    inner = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    if len(inner) != n_bins - 1:
        raise RuntimeError("half-shear scene response has tied quantile edges")
    return np.r_[-np.inf, inner, np.inf]


def conditional_curves(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    edges = quantile_edges(frame.scene_truth.to_numpy(float), N_BINS)
    residual_columns = [column for _, column in MODELS]
    local = frame[
        ["case", "scene_truth", *residual_columns]
    ].copy()
    local["bin"] = np.digitize(
        local.scene_truth.to_numpy(float), edges[1:-1]
    )
    rows = []
    for index in range(N_BINS):
        cell = local.loc[local.bin == index]
        if cell.empty or cell.case.nunique() < 2:
            raise RuntimeError(f"empty or one-case scene-response bin {index}")
        by_case = cell.groupby("case", sort=True)[residual_columns].mean()
        common = {
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "scene_truth_median": float(cell.scene_truth.median()),
            "scene_truth_mean": float(cell.scene_truth.mean()),
            "n_scenes": int(len(cell)),
            "n_cases": int(len(by_case)),
        }
        for model, column in MODELS:
            stats = finite_stat(by_case[column].to_numpy(float))
            rows.append({
                **common,
                "model": model,
                "residual_mean": stats["mean"],
                "residual_case_sem": stats["case_sem"],
                "residual_case_sd": stats["case_sd"],
            })
    return pd.DataFrame(rows), edges


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def configure_style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_panel(
    curves: pd.DataFrame,
    stem: Path,
    n_scenes: int,
    n_cases: int,
) -> None:
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
    bounds = []
    for row in curves.itertuples():
        bounds.extend([
            abs(float(row.residual_mean) - float(row.residual_case_sem)),
            abs(float(row.residual_mean) + float(row.residual_case_sem)),
        ])
    bound = max(0.01, 1.08 * max(bounds))
    for model, _, in MODELS:
        line = curves.loc[curves.model == model].sort_values("bin")
        color, marker, linestyle, label = styles[model]
        axis.errorbar(
            line.scene_truth_median,
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
    axis.set_ylim(-bound, bound)
    axis.set_xlabel(
        "Measured half-shear scene response "
        "(sum of sheared-neighbour labels)"
    )
    axis.set_ylabel("truth - prediction")
    axis.set_title(
        "100/100 model residual versus measured half-shear scene response\n"
        f"correction-fit cases 100–199; n={n_scenes:,}; "
        "error bars: one case SEM"
    )
    axis.legend(frameon=False, loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    for suffix, options in ((".png", {"dpi": 300}), (".pdf", {})):
        output = stem.with_suffix(suffix)
        if output.exists():
            raise FileExistsError(output)
        fig.savefig(output, bbox_inches="tight", **options)
    plt.close(fig)


def load_stack(
    summary_path: Path,
    n_jobs: int,
) -> tuple[dict, xgb.Booster, xgb.Booster]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["features"]["base"] != BASE_FEATURES:
        raise RuntimeError("base feature drift")
    if summary["features"]["correction"] != CONDITIONAL_FEATURES:
        raise RuntimeError("correction feature drift")
    deployment = summary["deployment_stack"]
    base_path = Path(deployment["base_model"])
    correction_path = Path(deployment["correction_model"])
    if file_sha256(base_path) != deployment["base_sha256"]:
        raise RuntimeError("base model hash mismatch")
    if file_sha256(correction_path) != deployment["correction_sha256"]:
        raise RuntimeError("correction model hash mismatch")
    options = {"device": "cpu", "n_jobs": n_jobs}
    base = xgb.Booster(options)
    base.load_model(base_path)
    correction = xgb.Booster(options)
    correction.load_model(correction_path)
    return summary, base, correction


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
    v22_prediction = mmap_array(
        source_cache, source_meta, "v22_prediction"
    )
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    starts, counts, scene_case, scene_offsets = scene_index(case, primary)
    correction_index = np.flatnonzero(
        (np.asarray(case) >= CASE_MIN) & (np.asarray(case) <= CASE_MAX)
    ).astype(np.int32)
    expected_rows = int(summary["training"]["correction"]["n_rows"])
    if len(correction_index) != expected_rows:
        raise RuntimeError(
            f"correction row mismatch: {len(correction_index)} != {expected_rows}"
        )

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
        correction_index,
    )
    correction_pair = (
        correction.inplace_predict(features).astype(np.float64)
        * target_scale
    )
    del features, row_scene_base, row_scene_count
    gc.collect()

    correction_all = np.zeros(len(case), dtype=np.float32)
    correction_all[correction_index] = correction_pair.astype(np.float32)
    scene_truth_all = np.add.reduceat(
        np.asarray(label, dtype=np.float64), starts
    )
    scene_v22_all = np.add.reduceat(
        np.asarray(v22_prediction, dtype=np.float64), starts
    )
    scene_raw_all = np.add.reduceat(
        all_base_prediction.astype(np.float64), starts
    )
    scene_corrected_all = scene_raw_all + np.add.reduceat(
        correction_all.astype(np.float64), starts
    )
    del correction_all, correction_pair
    gc.collect()

    mask = (scene_case >= CASE_MIN) & (scene_case <= CASE_MAX)
    frame = pd.DataFrame({
        "case": np.asarray(scene_case[mask], dtype=np.int16),
        "scene_truth": scene_truth_all[mask],
        "scene_v22": scene_v22_all[mask],
        "scene_raw": scene_raw_all[mask],
        "scene_corrected": scene_corrected_all[mask],
        "n_sheared_neighbours": np.asarray(counts[mask], dtype=np.int16),
        "n_all_neighbours": np.asarray(
            full_scene_count[mask], dtype=np.int16
        ),
    })
    if frame.case.nunique() != 100 or not np.isfinite(
        frame.drop(columns=["case"]).to_numpy(float)
    ).all():
        raise RuntimeError("half-shear scene population is incomplete")
    frame["residual_v22"] = frame.scene_truth - frame.scene_v22
    frame["residual_raw"] = frame.scene_truth - frame.scene_raw
    frame["residual_corrected"] = (
        frame.scene_truth - frame.scene_corrected
    )
    curves, edges = conditional_curves(frame)

    reference = summary[
        "correction_block_diagnostic_not_final_validation"
    ]["scene"]
    raw_reference = reference[
        "base_conditioned_on_full_base_sum"
    ]["case_balanced_residual_mean"]
    corrected_reference = reference[
        "combined_conditioned_on_full_base_sum"
    ]["case_balanced_residual_mean"]
    raw_stat = case_stat(frame, "residual_raw")
    corrected_stat = case_stat(frame, "residual_corrected")
    if abs(float(raw_stat["mean"]) - float(raw_reference)) > 2.0e-7:
        raise RuntimeError("raw scene residual does not replay training summary")
    if (
        abs(float(corrected_stat["mean"]) - float(corrected_reference))
        > 2.0e-7
    ):
        raise RuntimeError(
            "corrected scene residual does not replay training summary"
        )

    atomic_csv(stem.with_suffix(".csv"), curves)
    plot_panel(curves, stem, len(frame), frame.case.nunique())
    payload = {
        "schema_version": 1,
        "dataset": "half_shear_c100_199_correction_fit_block",
        "n_scenes": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "gap_definition": "summed pair label - summed pair prediction",
        "scene_truth_definition": (
            "sum of labelled sheared-neighbour pair responses per primary; "
            "not doubled and does not include the unsheared neighbour half"
        ),
        "prediction_population": (
            "same labelled sheared-neighbour pairs as the scene truth"
        ),
        "conditioning_warning": (
            "the noisy scene truth appears in both the x coordinate and the "
            "residual y coordinate, inducing mechanical correlation"
        ),
        "fit_status": (
            "base is case-out-of-sample on c100--199; correction is fit "
            "in-sample on c100--199"
        ),
        "uncertainty": "one SEM across per-case conditional means",
        "neighbour_counts": {
            "mean_sheared_labelled": float(
                frame.n_sheared_neighbours.mean()
            ),
            "mean_all_context": float(frame.n_all_neighbours.mean()),
        },
        "binning": {
            "n_bins": N_BINS,
            "internal_edges": [float(value) for value in edges[1:-1]],
            "outer_bounds": "negative and positive infinity",
            "labels_used_to_define_bins": True,
        },
        "global_case_balanced_residual": {
            "V2.2": case_stat(frame, "residual_v22"),
            "Raw 100/100 base": raw_stat,
            "100/100 base + correction": corrected_stat,
        },
        "audit": {
            "raw_summary_replay_abs": abs(
                float(raw_stat["mean"]) - float(raw_reference)
            ),
            "corrected_summary_replay_abs": abs(
                float(corrected_stat["mean"])
                - float(corrected_reference)
            ),
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
        "provenance": {
            "source_cache": str(source_cache),
            "source_metadata_sha256": sha256(
                source_cache / "metadata.json"
            ),
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
    print("PLOT_HALFSHEAR_CASE100_100_MEASURED_SCENE_DONE", flush=True)


if __name__ == "__main__":
    main()
