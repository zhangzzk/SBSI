#!/usr/bin/env python3
"""Train a case-disjoint 100/100 base-plus-correction R_blend stack.

All labelled half-shear pairs in cases 0--99 fit the base.  The frozen base is
then scored on cases 100--199, and every labelled pair in those cases fits the
residual correction.  Full-neighbour scene context includes both half-shear
neighbour populations.  No half-shear cases remain for final validation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import xgboost as xgb

from scripts.train_newbase_oldway_fullneighbour_160_40 import (
    CORRECTION_PARAMS,
    CORRECTION_TREES,
    N_CASES,
    build_full_base_context,
    case_balanced_pair_metrics,
    corrected_full_coordinate,
    load_base_recipe,
    load_full_metadata,
    metrics,
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
    xgb_common,
)
from scripts.v22_grouped_rscene_common import (
    conditional_matrix,
    load_source_metadata,
    mmap_array,
    sha256,
)


BASE_CASE_MIN = 0
BASE_CASE_MAX = 99
CORRECTION_CASE_MIN = 100
CORRECTION_CASE_MAX = 199


def partition_indices(
    case: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    case_array = np.asarray(case, dtype=np.int64)
    base_mask = (case_array >= BASE_CASE_MIN) & (case_array <= BASE_CASE_MAX)
    correction_mask = (
        (case_array >= CORRECTION_CASE_MIN)
        & (case_array <= CORRECTION_CASE_MAX)
    )
    if np.any(base_mask & correction_mask):
        raise RuntimeError("base and correction case masks overlap")
    if not np.all(base_mask | correction_mask):
        raise RuntimeError("100/100 case masks do not cover all source rows")
    base_index = np.flatnonzero(base_mask).astype(np.int32)
    correction_index = np.flatnonzero(correction_mask).astype(np.int32)
    if not len(base_index) or not len(correction_index):
        raise RuntimeError("empty 100/100 stage partition")
    return base_index, correction_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--full-neighbour-cache", required=True)
    parser.add_argument("--base-parameter-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    full_cache = Path(args.full_neighbour_cache).resolve()
    parameter_summary = Path(args.base_parameter_summary).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"Case-disjoint 100/100 stack complete: {summary_path}", flush=True)
        return

    source_meta = load_source_metadata(source_cache)
    full_meta = load_full_metadata(full_cache, source_cache)
    if int(source_meta.get("n_rows", -1)) != 47_310_214:
        raise RuntimeError("unexpected source-cache population")

    case = mmap_array(source_cache, source_meta, "case")
    primary = mmap_array(source_cache, source_meta, "input_index")
    label = mmap_array(source_cache, source_meta, "label")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    starts, counts, scene_case, scene_offsets = scene_index(case, primary)
    base_index, correction_index = partition_indices(case)
    if set(np.unique(case[base_index]).astype(int)) != set(range(0, 100)):
        raise RuntimeError("base stage does not contain exactly cases 0--99")
    if set(np.unique(case[correction_index]).astype(int)) != set(range(100, 200)):
        raise RuntimeError("correction stage does not contain exactly cases 100--199")

    standardization = source_meta["source_standardization"]
    target_mean = float(standardization["mean"])
    target_scale = float(standardization["std"])
    recipe = load_base_recipe(parameter_summary)
    base_params = xgb_common() | recipe["params"]
    base_target = (
        np.asarray(label[base_index], dtype=np.float64) - target_mean
    ) / target_scale
    dbase = xgb.DMatrix(
        np.asarray(x_scaled[base_index], dtype=np.float32),
        label=base_target.astype(np.float32),
        feature_names=BASE_FEATURES,
    )
    started = time.time()
    base = xgb.train(
        base_params,
        dbase,
        num_boost_round=recipe["rounds"],
        verbose_eval=False,
    )
    base_fit_seconds = time.time() - started
    base_train_prediction = (
        base.predict(dbase).astype(np.float64) * target_scale + target_mean
    )
    base_train_metrics = metrics(label[base_index], base_train_prediction)
    del dbase, base_target, base_train_prediction
    gc.collect()

    all_base_prediction = physical_base_prediction(
        base, x_scaled, target_mean, target_scale
    )
    scene_base, scene_count = build_full_base_context(
        base=base,
        full_cache=full_cache,
        full_meta=full_meta,
        source_primary=primary,
        source_scene_starts=starts,
        source_scene_offsets=scene_offsets,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    row_scene_base = np.repeat(scene_base, counts)
    row_scene_count = np.repeat(scene_count, counts).astype(np.int32)
    if len(row_scene_base) != len(case) or len(row_scene_count) != len(case):
        raise RuntimeError("full-neighbour context expansion failed")

    correction_features = conditional_matrix(
        x_scaled,
        x_raw,
        all_base_prediction,
        row_scene_base,
        row_scene_count,
        correction_index,
    )
    correction_physical_target = (
        np.asarray(label[correction_index], dtype=np.float64)
        - np.asarray(all_base_prediction[correction_index], dtype=np.float64)
    )
    correction_target = (
        correction_physical_target / target_scale
    ).astype(np.float32)
    correction_params = CORRECTION_PARAMS | {
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    }
    dcorrection = xgb.DMatrix(
        correction_features,
        label=correction_target,
        feature_names=CONDITIONAL_FEATURES,
    )
    started = time.time()
    correction = xgb.train(
        correction_params,
        dcorrection,
        num_boost_round=CORRECTION_TREES,
        verbose_eval=False,
    )
    correction_fit_seconds = time.time() - started
    correction_prediction = (
        correction.predict(dcorrection).astype(np.float64) * target_scale
    )
    correction_base = np.asarray(
        all_base_prediction[correction_index], dtype=np.float64
    )
    correction_combined = correction_base + correction_prediction
    correction_pair = {
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
    del correction_features, correction_physical_target, correction_target
    del dcorrection
    gc.collect()

    scene_truth = np.add.reduceat(np.asarray(label, dtype=np.float64), starts)
    scene_base_active = np.add.reduceat(
        all_base_prediction.astype(np.float64), starts
    )
    combined_active = all_base_prediction.astype(np.float64)
    combined_active[correction_index] = correction_combined
    scene_combined_active = np.add.reduceat(combined_active, starts)
    del combined_active
    correction_scene_mask = (
        (scene_case >= CORRECTION_CASE_MIN)
        & (scene_case <= CORRECTION_CASE_MAX)
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
    correction_scene = {
        "base_conditioned_on_full_base_sum": scene_metrics(
            scene_truth[correction_scene_mask],
            scene_base_active[correction_scene_mask],
            scene_base[correction_scene_mask],
            scene_case[correction_scene_mask],
        ),
        "combined_conditioned_on_full_base_sum": scene_metrics(
            scene_truth[correction_scene_mask],
            scene_combined_active[correction_scene_mask],
            scene_base[correction_scene_mask],
            scene_case[correction_scene_mask],
        ),
        "combined_conditioned_on_full_corrected_sum": scene_metrics(
            scene_truth[correction_scene_mask],
            scene_combined_active[correction_scene_mask],
            corrected_coordinate[correction_scene_mask],
            scene_case[correction_scene_mask],
        ),
    }

    base_path = output / "base.json"
    correction_path = output / "correction.json"
    atomic_model(base, base_path)
    atomic_model(correction, correction_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "case-disjoint 100/100 sequential base-plus-correction stack",
        "tuning_performed": False,
        "base_parameter_source": str(parameter_summary),
        "base_parameter_source_sha256": sha256(parameter_summary),
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
            "scene_context_population": "full",
            "full_neighbour_cache_used": True,
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        "training": {
            "base": {
                "case_window": [BASE_CASE_MIN, BASE_CASE_MAX],
                "split": "all labelled pair rows in cases 0--99",
                "n_rows": int(len(base_index)),
                "rounds": int(recipe["rounds"]),
                "params": base_params,
                "fit_seconds": base_fit_seconds,
                "in_sample_metrics": base_train_metrics,
            },
            "correction": {
                "case_window": [CORRECTION_CASE_MIN, CORRECTION_CASE_MAX],
                "split": "all labelled pair rows in cases 100--199",
                "n_rows": int(len(correction_index)),
                "target": (
                    "labelled sheared-pair response minus frozen base "
                    "prediction from a case-disjoint base"
                ),
                "scene_context": (
                    "sum frozen-base predictions over all deployed neighbours, "
                    "including the unsheared half"
                ),
                "multiplicity_context": (
                    "all deployed neighbours, including the unsheared half"
                ),
                "rounds": CORRECTION_TREES,
                "params": correction_params,
                "loss": "ordinary pair residual MSE; no scene-level term",
                "fit_seconds": correction_fit_seconds,
                "diagnostic_in_sample_metrics": {
                    "before": metrics(
                        label[correction_index], correction_base
                    ),
                    "after": metrics(
                        label[correction_index], correction_combined
                    ),
                },
            },
        },
        "correction_block_diagnostic_not_final_validation": {
            "case_window": [CORRECTION_CASE_MIN, CORRECTION_CASE_MAX],
            "pair": correction_pair,
            "scene": correction_scene,
            "note": (
                "base predictions are case-out-of-sample, but combined "
                "predictions are correction-fit in-sample"
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
            "base_case_count": 100,
            "correction_case_count": 100,
            "all_200_cases_used_for_fitting": True,
            "half_shear_final_validation_cases_remaining": 0,
            "base_retrained_after_correction_targets": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True), flush=True)
    print("NEWBASE_OLDWAY_FULLNEIGHBOUR_CASE100_100_DONE", flush=True)


if __name__ == "__main__":
    main()
