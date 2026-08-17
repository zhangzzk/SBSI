#!/usr/bin/env python3
"""Train a V2.2 pair-residual correction conditioned on a physical scene proxy.

This deliberately reuses the early grouped-residual model's frozen recipe:
official V2.2 validation rows from cases 40--199, 180 conservative trees, and
group strength zero.  The sole scene-coordinate change is replacing the V2.2
scene prediction with log10(sum flux-ratio / angular-separation**2), evaluated
over all deployed neighbours.  The base V2.2 pair prediction remains a pair
feature and remains the additive base prediction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import xgboost as xgb

from scripts.v22_grouped_rscene_common import (
    assign_scene_bins,
    case_offsets,
    grouped_loss_value,
    grouped_objective,
    load_source_metadata,
    make_scene_edges,
    mmap_array,
    model_params,
    sha256,
    strict_json,
)
from scripts.v22_proxy_correction_common import (
    CONDITIONAL_FEATURES,
    PROXY_DEFINITION,
    aligned_full_scene_context,
    load_full_metadata,
    proxy_conditional_matrix,
)


def atomic_model(booster: xgb.Booster, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.json")
    booster.save_model(temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--full-neighbour-cache", required=True)
    parser.add_argument("--train-case-min", type=int, default=40)
    parser.add_argument("--train-case-max", type=int, default=199)
    parser.add_argument("--edge-case-max", type=int, default=159)
    parser.add_argument("--n-proxy-bins", type=int, default=20)
    parser.add_argument("--trees", type=int, default=180)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    if not 40 <= args.train_case_min <= args.edge_case_max <= args.train_case_max <= 199:
        raise ValueError("expected training c40--199 and edge cases within that window")
    if args.trees <= 0 or args.n_proxy_bins < 2:
        raise ValueError("trees and proxy-bin count must be positive")

    source_cache = Path(args.source_cache).resolve()
    full_cache = Path(args.full_neighbour_cache).resolve()
    model_path = Path(args.output_model).resolve()
    summary_path = Path(args.output_summary).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite proxy-correction outputs")

    source_meta = load_source_metadata(source_cache)
    full_meta = load_full_metadata(full_cache, source_cache)
    offsets = case_offsets(source_meta)
    case_all = mmap_array(source_cache, source_meta, "case")
    primary_all = mmap_array(source_cache, source_meta, "input_index")
    label_all = mmap_array(source_cache, source_meta, "label")
    prediction_all = mmap_array(source_cache, source_meta, "v22_prediction")
    official_train_all = mmap_array(source_cache, source_meta, "official_train")
    x_scaled_all = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw_all = mmap_array(source_cache, source_meta, "x_raw")

    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    proxy_parts: list[np.ndarray] = []
    edge_scene_parts: list[np.ndarray] = []
    case_rows: dict[str, int] = {}
    for case in range(args.train_case_min, args.train_case_max + 1):
        start, stop = int(offsets[case]), int(offsets[case + 1])
        if not np.all(np.asarray(case_all[start:stop]) == case):
            raise RuntimeError(f"case {case}: source boundary mismatch")
        source_primary = np.asarray(primary_all[start:stop], dtype=np.int64)
        log_proxy, proxy, full_count = aligned_full_scene_context(
            source_primary, full_cache, full_meta, case
        )
        keep = ~np.asarray(official_train_all[start:stop], dtype=bool)
        if not keep.any():
            raise RuntimeError(f"case {case}: no official-validation rows")
        feature_parts.append(proxy_conditional_matrix(
            np.asarray(x_scaled_all[start:stop])[keep],
            np.asarray(x_raw_all[start:stop])[keep],
            np.asarray(prediction_all[start:stop])[keep],
            log_proxy[keep], full_count[keep],
        ))
        residual = (
            np.asarray(label_all[start:stop], dtype=np.float64)
            - np.asarray(prediction_all[start:stop], dtype=np.float64)
        )
        target_parts.append(residual[keep].astype(np.float32))
        proxy_parts.append(log_proxy[keep].astype(np.float32))
        if case <= args.edge_case_max:
            change = np.empty(len(source_primary), dtype=bool)
            change[0] = True
            change[1:] = source_primary[1:] != source_primary[:-1]
            edge_scene_parts.append(log_proxy[np.flatnonzero(change)].astype(np.float32))
        case_rows[str(case)] = int(keep.sum())
        if case % 10 == 0 or case == args.train_case_max:
            print(
                f"case {case}: train_rows={keep.sum():,} "
                f"full_Q=[{proxy.min():.3e},{proxy.max():.3e}]",
                flush=True,
            )

    features = np.concatenate(feature_parts)
    physical_target = np.concatenate(target_parts).astype(np.float64)
    log_proxy_rows = np.concatenate(proxy_parts).astype(np.float64)
    edge_scenes = np.concatenate(edge_scene_parts).astype(np.float64)
    del feature_parts, target_parts, proxy_parts, edge_scene_parts
    expected_rows = int(source_meta["official_random_row_split"]["validation_rows"])
    if len(features) != expected_rows:
        raise RuntimeError(
            f"training row count {len(features):,} != expected {expected_rows:,}"
        )
    edges = make_scene_edges(
        edge_scenes, n_quantile_bins=args.n_proxy_bins, forced_edges=()
    )
    bins = assign_scene_bins(log_proxy_rows, edges)
    target_scale = float(source_meta["source_standardization"]["std"])
    target = (physical_target / target_scale).astype(np.float32)
    matrix = xgb.DMatrix(features, label=target, feature_names=CONDITIONAL_FEATURES)
    params = model_params()
    initial_loss = grouped_loss_value(
        np.zeros(len(target), dtype=np.float32), target, bins, 0.0
    )
    started = time.time()
    booster = xgb.train(
        params, matrix, obj=grouped_objective(bins, 0.0),
        num_boost_round=args.trees, verbose_eval=False,
    )
    elapsed = time.time() - started
    correction = booster.predict(matrix).astype(np.float64) * target_scale
    residual_after = physical_target - correction
    final_loss = grouped_loss_value(
        correction / target_scale, target, bins, 0.0
    )

    counts = np.bincount(bins.astype(np.int64), minlength=args.n_proxy_bins)
    before_bin = np.bincount(
        bins, weights=physical_target, minlength=args.n_proxy_bins
    ) / counts
    after_bin = np.bincount(
        bins, weights=residual_after, minlength=args.n_proxy_bins
    ) / counts
    table = [
        {
            "bin": int(index),
            "n_pairs": int(counts[index]),
            "log10_proxy_lower": None if index == 0 else float(edges[index]),
            "log10_proxy_upper": (
                None if index == args.n_proxy_bins - 1 else float(edges[index + 1])
            ),
            "residual_before": float(before_bin[index]),
            "residual_after": float(after_bin[index]),
        }
        for index in range(args.n_proxy_bins)
    ]

    atomic_model(booster, model_path)
    mse_before = float(np.mean(np.square(physical_target)))
    mse_after = float(np.mean(np.square(residual_after)))
    payload = {
        "schema_version": 1,
        "kind": "additive V2.2 pair-residual booster with physical full-scene proxy",
        "base_model": "lsst_r_extnbr_v22",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "feature_names": CONDITIONAL_FEATURES,
        "target": "half-shear pair label minus frozen V2.2 pair prediction",
        "target_scale": target_scale,
        "source_cache": str(source_cache),
        "source_metadata_sha256": sha256(source_cache / "metadata.json"),
        "full_neighbour_cache": str(full_cache),
        "full_neighbour_metadata_sha256": sha256(full_cache / "metadata.json"),
        "scene_proxy": {
            "definition": PROXY_DEFINITION,
            "feature_transform": "log10",
            "population": "all deployed neighbours, sheared and unsheared",
            "distance_unit": "arcsec",
            "response_labels_used": False,
            "edge_fit_cases": [args.train_case_min, args.edge_case_max],
            "quantile_edges": edges,
            "upper_30_percent_threshold": float(
                edges[int(round(0.70 * args.n_proxy_bins))]
            ),
        },
        "training": {
            "case_window": [args.train_case_min, args.train_case_max],
            "split": "official frozen-V2.2 random-row validation rows only",
            "n_rows": int(len(features)),
            "n_cases": int(len(case_rows)),
            "case_rows": case_rows,
            "trees": int(args.trees),
            "group_strength": 0.0,
            "params": params,
            "fit_seconds": elapsed,
            "initial_standardized_loss": initial_loss,
            "final_standardized_loss": final_loss,
        },
        "in_sample_diagnostics": {
            "residual_before_mean": float(physical_target.mean()),
            "residual_after_mean": float(residual_after.mean()),
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std(ddof=1)),
            "mse_before": mse_before,
            "mse_after": mse_after,
            "mse_percent_change": 100.0 * (mse_after / mse_before - 1.0),
            "proxy_bin_mean_rms_before": float(np.sqrt(np.mean(before_bin**2))),
            "proxy_bin_mean_rms_after": float(np.sqrt(np.mean(after_bin**2))),
            "proxy_bins": table,
        },
        "protocol": {
            "base_model_retrained": False,
            "correction_hyperparameters_tuned": False,
            "recipe_copied_from_early_v22_grouped_residual_final": True,
            "external_cases_0_39_opened": False,
            "anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    strict_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("V22_PROXY_CORRECTION_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
