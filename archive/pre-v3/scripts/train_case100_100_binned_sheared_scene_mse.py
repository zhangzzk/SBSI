#!/usr/bin/env python3
"""Fit a 100/100 correction using pure binned sheared-scene MSE.

The exact frozen c0--99 base from the existing 100/100 stack is reused.  The
new correction is fitted on every labelled/sheared pair in cases 100--199.
For each case and each label-free bin of the base full-neighbour scene sum, the
loss squares the mean residual of the sheared-pair scene sum.  There is no
ordinary pair-MSE term.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.train_newbase_oldway_fullneighbour_160_40 import (
    CORRECTION_PARAMS,
    CORRECTION_TREES,
    N_SCENE_BINS,
    build_full_base_context,
    case_balanced_pair_metrics,
    corrected_full_coordinate,
    load_full_metadata,
    physical_base_prediction,
    scene_index,
    scene_metrics,
)
from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
    atomic_json,
    atomic_model,
    json_clean,
)
from scripts.v22_grouped_rscene_common import (
    assign_scene_bins,
    conditional_matrix,
    load_source_metadata,
    make_scene_edges,
    mmap_array,
    sha256,
)


BASE_CASE_MIN = 0
BASE_CASE_MAX = 99
CORRECTION_CASE_MIN = 100
CORRECTION_CASE_MAX = 199


def load_frozen_base(
    summary_path: Path, n_jobs: int
) -> tuple[dict[str, Any], xgb.Booster, Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["features"]["base"] != BASE_FEATURES:
        raise RuntimeError("base feature drift")
    if summary["training"]["base"]["case_window"] != [0, 99]:
        raise RuntimeError("source stack does not contain the c0--99 base")
    if summary["training"]["correction"]["case_window"] != [100, 199]:
        raise RuntimeError("source stack is not the controlled 100/100 split")
    deployment = summary["deployment_stack"]
    base_path = Path(deployment["base_model"])
    if sha256(base_path) != deployment["base_sha256"]:
        raise RuntimeError("frozen base hash mismatch")
    base = xgb.Booster({"device": "cpu", "n_jobs": n_jobs})
    base.load_model(base_path)
    return summary, base, base_path


def build_groups(
    selected_case: np.ndarray,
    scene_case: np.ndarray,
    scene_count: np.ndarray,
    scene_coordinate: np.ndarray,
    scene_mask: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """Build immutable case-by-scene-bin groups for the custom objective."""
    local_scene_case = np.asarray(scene_case[scene_mask], dtype=np.int64)
    local_scene_count = np.asarray(scene_count[scene_mask], dtype=np.int64)
    local_coordinate = np.asarray(scene_coordinate[scene_mask], dtype=np.float64)
    edges = make_scene_edges(
        local_coordinate, n_quantile_bins=N_SCENE_BINS
    )
    scene_bin = assign_scene_bins(local_coordinate, edges).astype(np.int64)
    scene_group = (
        (local_scene_case - CORRECTION_CASE_MIN) * N_SCENE_BINS + scene_bin
    ).astype(np.int32)
    n_groups = (
        (CORRECTION_CASE_MAX - CORRECTION_CASE_MIN + 1) * N_SCENE_BINS
    )
    group_scene_count = np.bincount(
        scene_group, minlength=n_groups
    ).astype(np.float64)
    group_row_count = np.bincount(
        scene_group,
        weights=local_scene_count,
        minlength=n_groups,
    ).astype(np.float64)
    if np.any(group_scene_count <= 0) or np.any(group_row_count <= 0):
        raise RuntimeError("one or more case-by-scene-bin cells are empty")
    row_group = np.repeat(scene_group, local_scene_count).astype(np.uint16)
    if len(row_group) != len(selected_case):
        raise RuntimeError("row-group expansion does not close")
    expected_case = (
        row_group.astype(np.int64) // N_SCENE_BINS + CORRECTION_CASE_MIN
    )
    if not np.array_equal(expected_case, np.asarray(selected_case, dtype=np.int64)):
        raise RuntimeError("row-group case assignment differs from source rows")

    # Diagonal approximation to the exact block Hessian.  For a leaf that
    # contains a complete case-bin cell, its summed Hessian is exact.  The
    # common normalization restores mean row Hessian=1 so the frozen tree
    # regularization and min-child-weight retain their usual scale.
    hessian_by_group = group_row_count / np.square(group_scene_count)
    raw_hessian_sum = float(np.dot(group_row_count, hessian_by_group))
    normalization = float(len(row_group) / raw_hessian_sum)
    hessian = (
        normalization * hessian_by_group[row_group.astype(np.int64)]
    ).astype(np.float32)
    if not np.isfinite(hessian).all() or not np.isclose(
        float(hessian.mean()), 1.0, rtol=2.0e-6, atol=2.0e-6
    ):
        raise RuntimeError("custom-objective Hessian normalization failed")
    return {
        "edges": edges,
        "scene_bin": scene_bin,
        "scene_group": scene_group,
        "scene_count": local_scene_count,
        "row_group": row_group,
        "group_scene_count": group_scene_count,
        "group_row_count": group_row_count,
        "normalization": normalization,
        "hessian": hessian,
        "n_groups": n_groups,
    }


def scene_group_error(
    prediction: np.ndarray,
    target: np.ndarray,
    scene_starts: np.ndarray,
    scene_group: np.ndarray,
    group_scene_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scene and case-bin means of prediction minus target."""
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    scene_error = np.add.reduceat(error, scene_starts)
    group_error = np.bincount(
        scene_group.astype(np.int64),
        weights=scene_error,
        minlength=len(group_scene_count),
    )
    group_mean = group_error / group_scene_count
    return scene_error, group_mean


def make_objective(
    scene_starts: np.ndarray,
    groups: dict[str, np.ndarray | float | int],
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[np.ndarray, np.ndarray]]:
    row_group = np.asarray(groups["row_group"], dtype=np.int64)
    scene_group = np.asarray(groups["scene_group"], dtype=np.int64)
    group_scene_count = np.asarray(
        groups["group_scene_count"], dtype=np.float64
    )
    normalization = float(groups["normalization"])
    hessian = np.asarray(groups["hessian"], dtype=np.float32)

    def objective(
        prediction: np.ndarray, matrix: xgb.DMatrix
    ) -> tuple[np.ndarray, np.ndarray]:
        target = matrix.get_label()
        if len(prediction) != len(row_group) or len(target) != len(row_group):
            raise RuntimeError("custom-objective row order changed")
        _, group_mean = scene_group_error(
            prediction,
            target,
            scene_starts,
            scene_group,
            group_scene_count,
        )
        gradient_by_group = (
            normalization * group_mean / group_scene_count
        )
        gradient = gradient_by_group[row_group].astype(np.float32)
        return gradient, hessian

    return objective


def objective_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    scene_starts: np.ndarray,
    groups: dict[str, np.ndarray | float | int],
    target_scale: float,
) -> dict[str, float | int]:
    scene_group = np.asarray(groups["scene_group"], dtype=np.int64)
    group_scene_count = np.asarray(
        groups["group_scene_count"], dtype=np.float64
    )
    _, group_mean = scene_group_error(
        prediction,
        target,
        scene_starts,
        scene_group,
        group_scene_count,
    )
    physical = group_mean * target_scale
    return {
        "n_case_bin_cells": int(len(group_mean)),
        "standardized_mse": float(np.mean(np.square(group_mean))),
        "standardized_rmse": float(np.sqrt(np.mean(np.square(group_mean)))),
        "physical_mse": float(np.mean(np.square(physical))),
        "physical_rmse": float(np.sqrt(np.mean(np.square(physical)))),
        "max_abs_physical_cell_residual": float(np.max(np.abs(physical))),
    }


def atomic_case_bin_table(
    path: Path,
    groups: dict[str, np.ndarray | float | int],
    before_prediction: np.ndarray,
    after_prediction: np.ndarray,
    target: np.ndarray,
    scene_starts: np.ndarray,
    target_scale: float,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    scene_group = np.asarray(groups["scene_group"], dtype=np.int64)
    group_scene_count = np.asarray(
        groups["group_scene_count"], dtype=np.float64
    )
    _, before_error = scene_group_error(
        before_prediction,
        target,
        scene_starts,
        scene_group,
        group_scene_count,
    )
    _, after_error = scene_group_error(
        after_prediction,
        target,
        scene_starts,
        scene_group,
        group_scene_count,
    )
    group = np.arange(len(group_scene_count), dtype=np.int64)
    frame = pd.DataFrame({
        "case": group // N_SCENE_BINS + CORRECTION_CASE_MIN,
        "bin": group % N_SCENE_BINS,
        "n_scenes": group_scene_count.astype(np.int64),
        "n_pairs": np.asarray(
            groups["group_row_count"], dtype=np.int64
        ),
        "base_truth_minus_prediction": -before_error * target_scale,
        "corrected_truth_minus_prediction": -after_error * target_scale,
    })
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--full-neighbour-cache", required=True)
    parser.add_argument("--source-stack-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    full_cache = Path(args.full_neighbour_cache).resolve()
    source_summary_path = Path(args.source_stack_summary).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"Binned-scene-MSE correction complete: {summary_path}", flush=True)
        return

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    source_summary, base, base_path = load_frozen_base(
        source_summary_path, n_jobs
    )
    source_meta = load_source_metadata(source_cache)
    full_meta = load_full_metadata(full_cache, source_cache)
    if sha256(source_cache / "metadata.json") != source_summary["source"][
        "metadata_sha256"
    ]:
        raise RuntimeError("source cache differs from frozen base training")
    case = mmap_array(source_cache, source_meta, "case")
    primary = mmap_array(source_cache, source_meta, "input_index")
    label = mmap_array(source_cache, source_meta, "label")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    starts, counts, scene_case, scene_offsets = scene_index(case, primary)
    correction_mask = (
        (np.asarray(case) >= CORRECTION_CASE_MIN)
        & (np.asarray(case) <= CORRECTION_CASE_MAX)
    )
    correction_index = np.flatnonzero(correction_mask).astype(np.int32)
    if (
        len(correction_index)
        != int(source_summary["training"]["correction"]["n_rows"])
        or int(correction_index[-1]) - int(correction_index[0]) + 1
        != len(correction_index)
    ):
        raise RuntimeError("correction rows differ from the controlled 100/100 stack")
    correction_scene_mask = (
        (scene_case >= CORRECTION_CASE_MIN)
        & (scene_case <= CORRECTION_CASE_MAX)
    )
    correction_scene_starts = (
        starts[correction_scene_mask] - int(correction_index[0])
    ).astype(np.int64)
    correction_scene_counts = counts[correction_scene_mask]
    if (
        correction_scene_starts[0] != 0
        or int(correction_scene_counts.sum()) != len(correction_index)
    ):
        raise RuntimeError("correction scene indexing does not close")

    target_mean = float(source_summary["source"]["target_mean"])
    target_scale = float(source_summary["source"]["target_scale"])
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
    groups = build_groups(
        np.asarray(case[correction_index], dtype=np.int16),
        scene_case,
        counts,
        full_scene_base,
        correction_scene_mask,
    )
    row_scene_base = np.repeat(full_scene_base, counts)
    row_scene_count = np.repeat(full_scene_count, counts).astype(np.int32)
    correction_features = conditional_matrix(
        x_scaled,
        x_raw,
        all_base_prediction,
        row_scene_base,
        row_scene_count,
        correction_index,
    )
    del row_scene_base, row_scene_count
    gc.collect()
    correction_physical_target = (
        np.asarray(label[correction_index], dtype=np.float64)
        - np.asarray(all_base_prediction[correction_index], dtype=np.float64)
    )
    correction_target = (
        correction_physical_target / target_scale
    ).astype(np.float32)
    matrix = xgb.DMatrix(
        correction_features,
        label=correction_target,
        feature_names=CONDITIONAL_FEATURES,
    )
    params = CORRECTION_PARAMS | {
        "n_jobs": n_jobs,
        "subsample": 1.0,
        "disable_default_eval_metric": 1,
    }
    objective = make_objective(correction_scene_starts, groups)
    zero = np.zeros(len(correction_target), dtype=np.float32)
    before_objective = objective_metrics(
        zero,
        correction_target,
        correction_scene_starts,
        groups,
        target_scale,
    )
    started = time.time()
    correction = xgb.train(
        params,
        matrix,
        obj=objective,
        num_boost_round=CORRECTION_TREES,
        verbose_eval=False,
    )
    fit_seconds = time.time() - started
    standardized_correction = correction.predict(matrix).astype(np.float64)
    after_objective = objective_metrics(
        standardized_correction,
        correction_target,
        correction_scene_starts,
        groups,
        target_scale,
    )
    if after_objective["physical_mse"] >= before_objective["physical_mse"]:
        raise RuntimeError("binned sheared-scene MSE did not decrease")
    correction_prediction = standardized_correction * target_scale
    correction_base = np.asarray(
        all_base_prediction[correction_index], dtype=np.float64
    )
    correction_combined = correction_base + correction_prediction

    scene_truth = np.add.reduceat(np.asarray(label, dtype=np.float64), starts)
    scene_base_active = np.add.reduceat(
        all_base_prediction.astype(np.float64), starts
    )
    scene_combined_active = scene_base_active.copy()
    scene_combined_active[correction_scene_mask] += np.add.reduceat(
        correction_prediction, correction_scene_starts
    )
    corrected_coordinate = corrected_full_coordinate(
        cases=range(CORRECTION_CASE_MIN, CORRECTION_CASE_MAX + 1),
        base=base,
        correction=correction,
        full_cache=full_cache,
        full_meta=full_meta,
        source_primary=primary,
        source_scene_starts=starts,
        source_scene_offsets=scene_offsets,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    if not np.isfinite(corrected_coordinate[correction_scene_mask]).all():
        raise RuntimeError("corrected full-neighbour coordinate is incomplete")

    case_bin_path = output / "fit_case_bin_residuals.csv"
    atomic_case_bin_table(
        case_bin_path,
        groups,
        zero,
        standardized_correction,
        correction_target,
        correction_scene_starts,
        target_scale,
    )
    correction_path = output / "correction.json"
    atomic_model(correction, correction_path)
    pair_diagnostic = {
        "base_before_correction": case_balanced_pair_metrics(
            label[correction_index],
            correction_base,
            case[correction_index],
        ),
        "combined_in_sample": case_balanced_pair_metrics(
            label[correction_index],
            correction_combined,
            case[correction_index],
        ),
    }
    scene_diagnostic = {
        "base_conditioned_on_full_base_sum": scene_metrics(
            scene_truth[correction_scene_mask],
            scene_base_active[correction_scene_mask],
            full_scene_base[correction_scene_mask],
            scene_case[correction_scene_mask],
        ),
        "combined_conditioned_on_full_base_sum": scene_metrics(
            scene_truth[correction_scene_mask],
            scene_combined_active[correction_scene_mask],
            full_scene_base[correction_scene_mask],
            scene_case[correction_scene_mask],
        ),
        "combined_conditioned_on_full_corrected_sum": scene_metrics(
            scene_truth[correction_scene_mask],
            scene_combined_active[correction_scene_mask],
            corrected_coordinate[correction_scene_mask],
            scene_case[correction_scene_mask],
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "frozen 100/100 base plus pure binned sheared-scene-MSE correction",
        "tuning_performed": False,
        "features": {
            "base": BASE_FEATURES,
            "correction": CONDITIONAL_FEATURES,
        },
        "source": {
            "cache": str(source_cache),
            "metadata_sha256": sha256(source_cache / "metadata.json"),
            "full_neighbour_cache": str(full_cache),
            "full_neighbour_metadata_sha256": sha256(
                full_cache / "metadata.json"
            ),
            "source_stack_summary": str(source_summary_path),
            "source_stack_summary_sha256": sha256(source_summary_path),
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        "training": {
            "base": {
                "case_window": [BASE_CASE_MIN, BASE_CASE_MAX],
                "reused_exact_frozen_model": True,
                "model": str(base_path),
                "sha256": sha256(base_path),
            },
            "correction": {
                "case_window": [CORRECTION_CASE_MIN, CORRECTION_CASE_MAX],
                "n_rows": int(len(correction_index)),
                "n_scenes": int(correction_scene_mask.sum()),
                "n_case_bin_cells": int(groups["n_groups"]),
                "scene_bins": N_SCENE_BINS,
                "scene_bin_edges": groups["edges"],
                "bin_coordinate": (
                    "sum frozen-base predictions over all deployed neighbours"
                ),
                "supervised_scene_response": (
                    "sum labels and corrected predictions over only the "
                    "labelled/sheared neighbours"
                ),
                "loss": (
                    "equal-weight MSE of the mean sheared-scene residual in "
                    "each case-by-scene-bin cell; no pair-MSE term"
                ),
                "hessian_approximation": (
                    "case-bin block curvature distributed over its rows and "
                    "globally normalized to mean row Hessian one"
                ),
                "hessian_normalization": float(groups["normalization"]),
                "rounds": CORRECTION_TREES,
                "params": params,
                "fit_seconds": fit_seconds,
                "objective_before": before_objective,
                "objective_after": after_objective,
                "case_bin_table": str(case_bin_path),
                "case_bin_table_sha256": sha256(case_bin_path),
            },
        },
        "correction_block_diagnostic_not_final_validation": {
            "pair": pair_diagnostic,
            "scene": scene_diagnostic,
            "note": (
                "base predictions are case-out-of-sample, but correction and "
                "combined predictions are fit in-sample on c100--199"
            ),
        },
        "deployment_stack": {
            "base_model": str(base_path),
            "base_sha256": sha256(base_path),
            "correction_model": str(correction_path),
            "correction_sha256": sha256(correction_path),
            "base_is_the_exact_frozen_model_used_for_correction_targets": True,
        },
        "protocol": {
            "base_and_correction_cases_are_disjoint": True,
            "full_neighbours_used_for_context_and_bin_coordinate": True,
            "only_sheared_neighbours_used_in_supervised_scene_loss": True,
            "ordinary_pair_loss_weight": 0.0,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True), flush=True)
    print("CASE100_100_BINNED_SHEARED_SCENE_MSE_DONE", flush=True)


if __name__ == "__main__":
    main()
