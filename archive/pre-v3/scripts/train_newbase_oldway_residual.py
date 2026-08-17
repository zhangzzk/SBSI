#!/usr/bin/env python3
"""Train the new R_blend base followed by the original residual recipe.

This deliberately mirrors the successful early V2.2 correction protocol:

* fit one base model on the official random-row training split in cases 40--199;
* freeze that exact base;
* predict every cached half-shear pair and sum those predictions per primary;
* fit the original 180-tree residual booster on the disjoint official
  validation rows in cases 40--199;
* save the same frozen base and correction for deployment.

Only the base recipe is replaced by the selected newer recipe.  There is no
cross-fitting, no joint training, no scene-loss term, and no tuning.  Coherent
anchors and ConstGold are not opened.
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

from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
    atomic_json,
    atomic_model,
    json_clean,
    r2_score,
    xgb_common,
)
from scripts.v22_grouped_rscene_common import (
    conditional_matrix,
    load_source_metadata,
    mmap_array,
    sha256,
)


FIT_CASE_MIN = 40
FIT_CASE_MAX = 199
CORRECTION_TREES = 180
CORRECTION_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "booster": "gbtree",
    "eval_metric": "rmse",
    "disable_default_eval_metric": 0,
    "max_bin": 256,
    "base_score": 0.0,
    "seed": 20260815,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 2000,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}


def load_base_recipe(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    recipe = payload["fixed_params"]
    if len(recipe["base"]) != 8 or int(recipe["base_rounds"]) <= 0:
        raise RuntimeError("new-base parameter summary is incomplete")
    return {
        "params": recipe["base"],
        "rounds": int(recipe["base_rounds"]),
    }


def scene_index(
    case: np.ndarray, primary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    change = np.empty(len(case), dtype=bool)
    change[0] = True
    change[1:] = (case[1:] != case[:-1]) | (primary[1:] != primary[:-1])
    starts = np.flatnonzero(change).astype(np.int64)
    counts = np.diff(np.append(starts, len(case))).astype(np.int32)
    if int(counts.sum()) != len(case):
        raise RuntimeError("scene counts do not close")
    return starts, counts


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    return {
        "r2": r2_score(target, prediction),
        "mse": float(np.mean(np.square(residual))),
        "residual_mean": float(residual.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--base-parameter-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    parameter_summary = Path(args.base_parameter_summary).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"New-base old-way stack already complete: {summary_path}")
        return

    source_meta = load_source_metadata(source_cache)
    case = mmap_array(source_cache, source_meta, "case")
    primary = mmap_array(source_cache, source_meta, "input_index")
    label = mmap_array(source_cache, source_meta, "label")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    official_train = mmap_array(source_cache, source_meta, "official_train")
    fit_case = (
        (np.asarray(case) >= FIT_CASE_MIN)
        & (np.asarray(case) <= FIT_CASE_MAX)
    )
    base_index = np.flatnonzero(
        fit_case & np.asarray(official_train)
    ).astype(np.int32)
    correction_index = np.flatnonzero(
        fit_case & ~np.asarray(official_train)
    ).astype(np.int32)
    if len(base_index) != 30_281_914 or len(correction_index) != 7_570_479:
        raise RuntimeError(
            "official old-way split drifted: "
            f"base={len(base_index):,} correction={len(correction_index):,}"
        )
    del fit_case

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

    all_base_prediction = base.inplace_predict(
        np.asarray(x_scaled, dtype=np.float32)
    ).astype(np.float64)
    all_base_prediction = (
        all_base_prediction * target_scale + target_mean
    ).astype(np.float32)
    starts, counts = scene_index(np.asarray(case), np.asarray(primary))
    scene_base = np.add.reduceat(
        all_base_prediction.astype(np.float64), starts
    ).astype(np.float32)
    row_scene_base = np.repeat(scene_base, counts)
    row_scene_count = np.repeat(counts, counts).astype(np.int32)
    if len(row_scene_base) != len(case) or len(row_scene_count) != len(case):
        raise RuntimeError("scene context expansion failed")

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
    base_correction_prediction = np.asarray(
        all_base_prediction[correction_index], dtype=np.float64
    )
    combined_correction_prediction = (
        base_correction_prediction + correction_prediction
    )
    correction_before_metrics = metrics(
        label[correction_index], base_correction_prediction
    )
    correction_after_metrics = metrics(
        label[correction_index], combined_correction_prediction
    )

    external_index = np.flatnonzero(
        (np.asarray(case) >= 20) & (np.asarray(case) <= 39)
    ).astype(np.int32)
    external_features = conditional_matrix(
        x_scaled,
        x_raw,
        all_base_prediction,
        row_scene_base,
        row_scene_count,
        external_index,
    )
    external_correction = (
        correction.inplace_predict(external_features).astype(np.float64)
        * target_scale
    )
    external_base = np.asarray(
        all_base_prediction[external_index], dtype=np.float64
    )
    external_metrics = {
        "base": metrics(label[external_index], external_base),
        "combined": metrics(
            label[external_index], external_base + external_correction
        ),
        "n_rows": int(len(external_index)),
    }

    base_path = output / "base.json"
    correction_path = output / "correction.json"
    atomic_model(base, base_path)
    atomic_model(correction, correction_path)
    payload = {
        "schema_version": 1,
        "kind": "frozen new base plus exact early residual-correction protocol",
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
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        "training": {
            "case_window": [FIT_CASE_MIN, FIT_CASE_MAX],
            "base": {
                "split": "official frozen-V2.2 random-row training rows",
                "n_rows": int(len(base_index)),
                "rounds": int(recipe["rounds"]),
                "params": base_params,
                "fit_seconds": base_fit_seconds,
                "in_sample_metrics": base_train_metrics,
            },
            "correction": {
                "split": "disjoint official frozen-V2.2 random-row validation rows",
                "n_rows": int(len(correction_index)),
                "target": "pair label minus frozen new-base pair prediction",
                "scene_context": (
                    "sum frozen new-base predictions over all cached half-shear "
                    "pair rows for the primary"
                ),
                "rounds": CORRECTION_TREES,
                "params": correction_params,
                "loss": "ordinary pair residual MSE; no grouped scene term",
                "fit_seconds": correction_fit_seconds,
                "before_metrics": correction_before_metrics,
                "after_metrics": correction_after_metrics,
            },
        },
        "deployment_stack": {
            "base_model": str(base_path),
            "base_sha256": sha256(base_path),
            "correction_model": str(correction_path),
            "correction_sha256": sha256(correction_path),
            "base_and_correction_are_the_same_frozen_models_used_to_make_targets": True,
        },
        "external_half_shear_c20_39_diagnostic_not_used_for_training_or_selection": (
            external_metrics
        ),
        "protocol": {
            "base_and_correction_splits_are_row_disjoint": True,
            "base_retrained_after_correction_targets": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True), flush=True)
    print("NEWBASE_OLDWAY_RESIDUAL_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
