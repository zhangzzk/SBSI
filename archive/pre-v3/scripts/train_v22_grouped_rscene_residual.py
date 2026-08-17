#!/usr/bin/env python3
"""Train one pair-aware residual model with an explicit scene-bin mean loss."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import xgboost as xgb

from scripts.v22_grouped_rscene_common import (
    CASE_MAX,
    CONDITIONAL_FEATURES,
    FIT_CASE_MIN,
    TRAIN_CASE_MAX,
    conditional_matrix,
    grouped_loss_value,
    grouped_objective,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    model_params,
    physical_correction,
    scene_edges_from_metadata,
    sha256,
    strict_json,
)


DEFAULT_TREES = 180


def atomic_model(booster: xgb.Booster, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.json")
    booster.save_model(temporary)
    os.replace(temporary, path)


def bin_summary(
    bin_index: np.ndarray,
    residual_before: np.ndarray,
    residual_after: np.ndarray,
    n_bins: int,
) -> list[dict[str, float | int]]:
    bins = np.asarray(bin_index, dtype=np.int64)
    count = np.bincount(bins, minlength=n_bins).astype(np.int64)
    before = np.bincount(
        bins, weights=np.asarray(residual_before, dtype=np.float64),
        minlength=n_bins,
    ) / count
    after = np.bincount(
        bins, weights=np.asarray(residual_after, dtype=np.float64),
        minlength=n_bins,
    ) / count
    return [
        {
            "bin": index,
            "n_pairs": int(count[index]),
            "residual_before": float(before[index]),
            "residual_after": float(after[index]),
        }
        for index in range(n_bins)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--group-strength", type=float, required=True)
    parser.add_argument("--train-case-min", type=int, default=FIT_CASE_MIN)
    parser.add_argument("--train-case-max", type=int, default=TRAIN_CASE_MAX)
    parser.add_argument("--trees", type=int, default=DEFAULT_TREES)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    model_path = Path(args.output_model).resolve()
    summary_path = Path(args.output_summary).resolve()
    if args.group_strength < 0 or not np.isfinite(args.group_strength):
        raise ValueError("group strength must be finite and non-negative")
    if not (
        FIT_CASE_MIN <= args.train_case_min <= args.train_case_max <= CASE_MAX
    ):
        raise ValueError("training cases must lie inside 40--199")
    if args.trees <= 0:
        raise ValueError("tree count must be positive")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite grouped residual outputs")

    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    if scene_meta["source_metadata_sha256"] != sha256(source_cache / "metadata.json"):
        raise RuntimeError("scene augmentation does not match the source cache")
    case = mmap_array(source_cache, source_meta, "case")
    official_train = mmap_array(source_cache, source_meta, "official_train")
    label = mmap_array(source_cache, source_meta, "label")
    pair_prediction = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs = mmap_array(scene_cache, scene_meta, "scene_n_pairs")
    scene_bin = mmap_array(scene_cache, scene_meta, "scene_bin")

    selection = (
        (np.asarray(case) >= args.train_case_min)
        & (np.asarray(case) <= args.train_case_max)
        & ~np.asarray(official_train)
    )
    index = np.flatnonzero(selection).astype(np.int32)
    del selection
    if len(index) < 1_000_000:
        raise RuntimeError(f"unexpectedly small training sample: {len(index):,}")
    observed_cases = np.unique(case[index]).astype(int)
    expected_cases = np.arange(args.train_case_min, args.train_case_max + 1)
    if not np.array_equal(observed_cases, expected_cases):
        raise RuntimeError("training sample does not cover every requested case")

    features = conditional_matrix(
        x_scaled, x_raw, pair_prediction, scene_prediction, scene_n_pairs, index
    )
    target_scale = float(source_meta["source_standardization"]["std"])
    physical_target = (
        np.asarray(label[index], dtype=np.float64)
        - np.asarray(pair_prediction[index], dtype=np.float64)
    )
    target = (physical_target / target_scale).astype(np.float32)
    bins = np.asarray(scene_bin[index], dtype=np.uint8)
    edges = scene_edges_from_metadata(scene_meta)
    n_bins = len(edges) - 1
    counts = np.bincount(bins.astype(np.int64), minlength=n_bins)
    if np.any(counts == 0):
        raise RuntimeError(f"empty training scene bins: {np.flatnonzero(counts == 0)}")

    matrix = xgb.DMatrix(
        features, label=target, feature_names=CONDITIONAL_FEATURES
    )
    params = model_params()
    initial_loss = grouped_loss_value(
        np.zeros(len(target), dtype=np.float32), target, bins,
        args.group_strength,
    )
    started = time.time()
    booster = xgb.train(
        params,
        matrix,
        obj=grouped_objective(bins, args.group_strength),
        num_boost_round=args.trees,
        verbose_eval=False,
    )
    elapsed = time.time() - started
    standardized_prediction = booster.predict(matrix).astype(np.float64)
    final_loss = grouped_loss_value(
        standardized_prediction, target, bins, args.group_strength
    )
    correction = standardized_prediction * target_scale
    residual_after = physical_target - correction
    table = bin_summary(bins, physical_target, residual_after, n_bins)
    before_bin = np.asarray([item["residual_before"] for item in table])
    after_bin = np.asarray([item["residual_after"] for item in table])
    tail = np.asarray(scene_prediction[index], dtype=np.float64) > 0.1

    atomic_model(booster, model_path)
    payload = {
        "schema_version": 1,
        "kind": "additive_pair_residual_booster_with_equal_scene_bin_mean_loss",
        "base_model": "lsst_r_extnbr_v22",
        "source_cache": str(source_cache),
        "source_metadata_sha256": sha256(source_cache / "metadata.json"),
        "scene_cache": str(scene_cache),
        "scene_metadata_sha256": sha256(scene_cache / "metadata.json"),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "feature_names": CONDITIONAL_FEATURES,
        "target": "half-shear pair label minus frozen V2.2 pair prediction",
        "target_center": 0.0,
        "target_scale": target_scale,
        "training": {
            "case_window": [args.train_case_min, args.train_case_max],
            "split": "official frozen-V2.2 random-row validation rows only",
            "n_rows": int(len(index)),
            "n_cases": int(len(observed_cases)),
            "group_strength": float(args.group_strength),
            "trees": int(args.trees),
            "params": params,
            "fit_seconds": elapsed,
            "scene_bin_edges": edges,
            "scene_bin_pair_counts": counts,
            "objective": (
                "0.5*mean(pair_error^2) + "
                "0.5*group_strength*mean_b(mean_pair_error_in_scene_bin_b^2)"
            ),
            "hessian": (
                "positive block-curvature approximation distributed over rows; "
                "exact curvature for a leaf containing one complete scene bin"
            ),
            "initial_standardized_loss": initial_loss,
            "final_standardized_loss": final_loss,
        },
        "in_sample_diagnostics": {
            "residual_before_mean": float(physical_target.mean()),
            "residual_after_mean": float(residual_after.mean()),
            "mse_before": float(np.mean(np.square(physical_target))),
            "mse_after": float(np.mean(np.square(residual_after))),
            "mse_percent_change": float(
                100.0 * (
                    np.mean(np.square(residual_after))
                    / np.mean(np.square(physical_target)) - 1.0
                )
            ),
            "scene_bin_mean_rms_before": float(np.sqrt(np.mean(before_bin**2))),
            "scene_bin_mean_rms_after": float(np.sqrt(np.mean(after_bin**2))),
            "scene_bin_max_abs_mean_before": float(np.max(np.abs(before_bin))),
            "scene_bin_max_abs_mean_after": float(np.max(np.abs(after_bin))),
            "tail_pair_residual_before": float(physical_target[tail].mean()),
            "tail_pair_residual_after": float(residual_after[tail].mean()),
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std(ddof=1)),
            "scene_bins": table,
        },
        "protocol": {
            "response_labels_used_to_define_scene_coordinate_or_bins": False,
            "cases_160_199_used_for_training": bool(
                args.train_case_max >= 160
            ),
            "external_cases_0_39_opened": False,
            "constgold_opened": False,
            "anchor_truth_opened": False,
        },
    }
    strict_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("V22_GROUPED_RSCENE_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
