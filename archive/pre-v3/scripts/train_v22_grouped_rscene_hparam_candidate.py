#!/usr/bin/env python3
"""Train one case-separated hyperparameter candidate for the V2.2 correction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import xgboost as xgb

from scripts.v22_grouped_rscene_common import (
    CONDITIONAL_FEATURES,
    conditional_matrix,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    sha256,
    strict_json,
)
from scripts.v22_grouped_rscene_hparam_common import (
    CANDIDATES,
    EARLY_STOPPING_ROUNDS,
    EARLY_STOP_CASES,
    TUNE_TRAIN_CASES,
    candidate_by_id,
    physical_prediction,
    rms,
    select_case_rows,
    xgb_params,
)


def atomic_model(booster: xgb.Booster, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.json")
    booster.save_model(temporary)
    os.replace(temporary, path)


def residual_summary(
    residual_before: np.ndarray,
    correction: np.ndarray,
    scene_bin: np.ndarray,
) -> dict[str, float]:
    before = np.asarray(residual_before, dtype=np.float64)
    after = before - np.asarray(correction, dtype=np.float64)
    bins = np.asarray(scene_bin, dtype=np.int64)
    count = np.bincount(bins).astype(np.float64)
    if np.any(count == 0):
        raise RuntimeError("candidate diagnostic has an empty scene bin")
    before_bin = np.bincount(bins, weights=before) / count
    after_bin = np.bincount(bins, weights=after) / count
    mse_before = float(np.mean(np.square(before)))
    mse_after = float(np.mean(np.square(after)))
    return {
        "residual_before_mean": float(before.mean()),
        "residual_after_mean": float(after.mean()),
        "mse_before": mse_before,
        "mse_after": mse_after,
        "rmse_before": float(np.sqrt(mse_before)),
        "rmse_after": float(np.sqrt(mse_after)),
        "mse_percent_change": float(100.0 * (mse_after / mse_before - 1.0)),
        "scene_pair_mean_rms_before": rms(before_bin),
        "scene_pair_mean_rms_after": rms(after_bin),
        "scene_pair_mean_max_abs_before": float(np.max(np.abs(before_bin))),
        "scene_pair_mean_max_abs_after": float(np.max(np.abs(after_bin))),
        "correction_mean": float(np.mean(correction)),
        "correction_sd": float(np.std(correction, ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--candidate-id", type=int, required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()

    config = candidate_by_id(args.candidate_id)
    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    model_path = Path(args.output_model).resolve()
    summary_path = Path(args.output_summary).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite hyperparameter outputs")

    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    if scene_meta["source_metadata_sha256"] != sha256(source_cache / "metadata.json"):
        raise RuntimeError("scene augmentation does not match source cache")

    case = mmap_array(source_cache, source_meta, "case")
    official_train = mmap_array(source_cache, source_meta, "official_train")
    label = mmap_array(source_cache, source_meta, "label")
    pair_prediction = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs = mmap_array(scene_cache, scene_meta, "scene_n_pairs")
    scene_bin = mmap_array(scene_cache, scene_meta, "scene_bin")

    train_index = select_case_rows(
        case, official_train, TUNE_TRAIN_CASES, official_validation_only=True
    )
    early_index = select_case_rows(
        case, official_train, EARLY_STOP_CASES, official_validation_only=True
    )
    if np.intersect1d(train_index, early_index).size:
        raise RuntimeError("training and early-stopping rows overlap")

    target_scale = float(source_meta["source_standardization"]["std"])

    def build(index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = conditional_matrix(
            x_scaled,
            x_raw,
            pair_prediction,
            scene_prediction,
            scene_n_pairs,
            index,
        )
        physical_target = (
            np.asarray(label[index], dtype=np.float64)
            - np.asarray(pair_prediction[index], dtype=np.float64)
        )
        standardized_target = (physical_target / target_scale).astype(np.float32)
        return features, physical_target, standardized_target

    train_features, train_physical, train_target = build(train_index)
    early_features, early_physical, early_target = build(early_index)
    dtrain = xgb.DMatrix(
        train_features, label=train_target, feature_names=CONDITIONAL_FEATURES
    )
    dearly = xgb.DMatrix(
        early_features, label=early_target, feature_names=CONDITIONAL_FEATURES
    )

    params = xgb_params(config)
    history: dict[str, dict[str, list[float]]] = {}
    started = time.time()
    full_booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(config["max_rounds"]),
        evals=[(dtrain, "train"), (dearly, "early_validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=history,
        verbose_eval=50,
    )
    elapsed = time.time() - started
    best_iteration = int(full_booster.best_iteration)
    best_rounds = best_iteration + 1
    booster = full_booster[:best_rounds]
    if booster.num_boosted_rounds() != best_rounds:
        raise RuntimeError("sliced booster does not end at the best iteration")

    train_correction = physical_prediction(booster, train_features, target_scale)
    early_correction = physical_prediction(booster, early_features, target_scale)
    train_diagnostics = residual_summary(
        train_physical,
        train_correction,
        np.asarray(scene_bin[train_index], dtype=np.uint8),
    )
    early_diagnostics = residual_summary(
        early_physical,
        early_correction,
        np.asarray(scene_bin[early_index], dtype=np.uint8),
    )

    atomic_model(booster, model_path)
    payload = {
        "schema_version": 1,
        "kind": "half-shear-only V2.2 additive pair-residual hyperparameter candidate",
        "base_model": "lsst_r_extnbr_v22",
        "candidate_id": int(args.candidate_id),
        "candidate_count": len(CANDIDATES),
        "candidate_name": config["name"],
        "configuration": config,
        "feature_names": CONDITIONAL_FEATURES,
        "target": "half-shear pair label minus frozen V2.2 pair prediction",
        "target_scale": target_scale,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "source_cache": str(source_cache),
        "source_metadata_sha256": sha256(source_cache / "metadata.json"),
        "scene_cache": str(scene_cache),
        "scene_metadata_sha256": sha256(scene_cache / "metadata.json"),
        "training": {
            "case_window": list(TUNE_TRAIN_CASES),
            "early_stopping_case_window": list(EARLY_STOP_CASES),
            "split": "official frozen-V2.2 random-row validation rows only",
            "n_rows": int(len(train_index)),
            "n_early_stopping_rows": int(len(early_index)),
            "params": params,
            "maximum_rounds": int(config["max_rounds"]),
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration_zero_based": best_iteration,
            "best_rounds": best_rounds,
            "best_early_standardized_rmse": float(full_booster.best_score),
            "fit_seconds": elapsed,
            "saved_model_contains_only_best_rounds": True,
        },
        "training_diagnostics": train_diagnostics,
        "early_stopping_diagnostics": early_diagnostics,
        "learning_curve": {
            "train_standardized_rmse": history["train"]["rmse"],
            "early_validation_standardized_rmse": history["early_validation"]["rmse"],
        },
        "protocol": {
            "selection_cases_160_199_opened": False,
            "external_development_cases_0_19_opened": False,
            "external_final_cases_20_39_opened": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    strict_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"V22_RSCENE_HPARAM_CANDIDATE_DONE id={args.candidate_id} "
        f"name={config['name']} best_rounds={best_rounds}",
        flush=True,
    )


if __name__ == "__main__":
    main()
