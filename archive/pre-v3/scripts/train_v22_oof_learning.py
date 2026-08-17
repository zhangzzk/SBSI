#!/usr/bin/env python3
"""Train the cross-fitted V2.2 residual/variance experiment.

Stages are deliberately separate so expensive work can run as SLURM arrays:

``base-fold``
    Refit the frozen V2.2 recipe on three of four rendered-case folds and
    predict the omitted fold.
``merge-base``
    Assemble complete case-held-out base predictions for cases 40--199.
``bias-fold`` / ``bias-final`` / ``merge-bias``
    Learn E[label - base_oof | pair conditions], with fold predictions used
    to centre the variance target and a final model used on external cases.
``variance``
    Learn E[(residual - bias_oof)^2 | pair conditions].
``variance-weighted``
    Refit the original V2.2 mean model using stabilized inverse predicted
    conditional variance on the exact original random-row training split.

Only half-shear pairs are read.  Constgold and coherent-anchor truth are never
opened by this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import xgboost as xgb


FIT_CASE_MIN = 40
CASE_MAX = 199
N_FOLDS = 4
BASE_TREES = 271
BIAS_TREES = 180
VARIANCE_TREES = 140
CONDITIONAL_FEATURES = [
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
    "base_prediction",
    "log10_flux_s_over_flux_p",
    "log10_Re_s_over_Re_p",
]


def strict_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def atomic_model(booster: xgb.Booster, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.json")
    booster.save_model(temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def load_metadata(cache: Path) -> dict[str, Any]:
    with (cache / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["source_tag"] != "lsst_r_extnbr_v22":
        raise RuntimeError("cache is not the frozen V2.2 population")
    if metadata["case_crossfit"]["n_folds"] != N_FOLDS:
        raise RuntimeError("cache fold count differs from training code")
    return metadata


def mmap(cache: Path, metadata: dict[str, Any], name: str) -> np.ndarray:
    return np.load(cache / metadata["arrays"][name], mmap_mode="r")


def indices_for(
    case: np.ndarray,
    fold_by_case: np.ndarray,
    *,
    heldout_fold: int | None,
    training: bool,
) -> np.ndarray:
    fit = (case >= FIT_CASE_MIN) & (case <= CASE_MAX)
    if heldout_fold is None:
        mask = fit
    else:
        if heldout_fold < 0 or heldout_fold >= N_FOLDS:
            raise ValueError(f"fold must be in [0,{N_FOLDS}), got {heldout_fold}")
        row_fold = fold_by_case[case]
        mask = fit & ((row_fold != heldout_fold) if training else (row_fold == heldout_fold))
    return np.flatnonzero(mask).astype(np.int32)


def conditional_matrix(
    x_scaled: np.ndarray,
    x_raw: np.ndarray,
    prediction: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    """Build the fixed pair-local conditional coordinate in float32."""
    index = np.asarray(index)
    raw = np.asarray(x_raw[index], dtype=np.float32)
    scaled = np.asarray(x_scaled[index], dtype=np.float32)
    base = np.asarray(prediction[index], dtype=np.float32)
    if scaled.ndim != 2 or scaled.shape[1] != 7 or raw.shape != scaled.shape:
        raise ValueError("unexpected cached feature shape")
    if base.shape != (len(index),):
        raise ValueError("prediction/index shape mismatch")
    log_flux = -0.4 * (raw[:, 3] - raw[:, 2])
    if np.any(raw[:, :2] <= 0):
        raise RuntimeError("supported pair has non-positive intrinsic size")
    log_size = np.log10(raw[:, 1] / raw[:, 0])
    output = np.empty((len(index), len(CONDITIONAL_FEATURES)), dtype=np.float32)
    output[:, :7] = scaled
    output[:, 7] = base
    output[:, 8] = log_flux
    output[:, 9] = log_size
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite conditional feature")
    return output


def standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise RuntimeError("invalid target standardization")
    return ((values - mean) / std).astype(np.float32), mean, std


def physical_predict(
    booster: xgb.Booster,
    features: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    prediction = booster.inplace_predict(features).astype(np.float64)
    prediction = prediction * std + mean
    if not np.isfinite(prediction).all():
        raise RuntimeError("model returned non-finite physical prediction")
    return prediction


def base_params(metadata: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(metadata["source_metadata"])
    with source_path.open(encoding="utf-8") as handle:
        task = json.load(handle)["tasks"]["regression"]
    params = dict(task["params"])
    params.update(
        objective="reg:squarederror",
        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        device=os.environ.get("XGB_DEVICE", "cpu"),
        tree_method="hist",
        booster="gbtree",
        disable_default_eval_metric=1,
        seed=20260814,
    )
    return params


def bias_params() -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cpu"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        "max_depth": 5,
        "min_child_weight": 2000,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 10.0,
        "seed": 20260815,
        "disable_default_eval_metric": 1,
    }


def variance_params() -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cpu"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        "max_depth": 4,
        "min_child_weight": 10000,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 50.0,
        "seed": 20260816,
        "disable_default_eval_metric": 1,
    }


def fit_booster(
    features: np.ndarray,
    target: np.ndarray,
    params: dict[str, Any],
    trees: int,
    *,
    weights: np.ndarray | None = None,
) -> tuple[xgb.Booster, float]:
    matrix = xgb.DMatrix(
        features,
        label=np.asarray(target, dtype=np.float32),
        weight=None if weights is None else np.asarray(weights, dtype=np.float32),
        feature_names=(
            CONDITIONAL_FEATURES
            if features.shape[1] == len(CONDITIONAL_FEATURES)
            else None
        ),
    )
    start = time.time()
    booster = xgb.train(params, matrix, num_boost_round=trees, verbose_eval=False)
    return booster, float(time.time() - start)


def summarize_residual(label: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    label = np.asarray(label, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = label - prediction
    return {
        "n_rows": int(len(label)),
        "label_mean": float(label.mean()),
        "prediction_mean": float(prediction.mean()),
        "residual_mean": float(residual.mean()),
        "mse": float(np.mean(np.square(residual))),
        "residual_std_ddof1": float(residual.std(ddof=1)),
    }


def stage_base_fold(args: argparse.Namespace) -> None:
    cache, output = Path(args.cache), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(cache)
    case = mmap(cache, metadata, "case")
    fold_by_case = mmap(cache, metadata, "fold_by_case")
    x_scaled = mmap(cache, metadata, "x_scaled")
    label = mmap(cache, metadata, "label")
    train = indices_for(case, fold_by_case, heldout_fold=args.fold, training=True)
    heldout = indices_for(case, fold_by_case, heldout_fold=args.fold, training=False)
    source = metadata["source_standardization"]
    target = (
        (np.asarray(label[train], dtype=np.float64) - float(source["mean"]))
        / float(source["std"])
    ).astype(np.float32)
    booster, elapsed = fit_booster(
        np.asarray(x_scaled[train], dtype=np.float32),
        target,
        base_params(metadata),
        BASE_TREES,
    )
    model_path = output / f"base_fold{args.fold}.json"
    atomic_model(booster, model_path)
    prediction = physical_predict(
        booster,
        np.asarray(x_scaled[heldout], dtype=np.float32),
        float(source["mean"]),
        float(source["std"]),
    )
    pred_path = output / f"base_fold{args.fold}_prediction.npz"
    atomic_npz(
        pred_path,
        index=heldout.astype(np.int32),
        prediction=prediction.astype(np.float32),
    )
    payload = {
        "stage": "base_fold",
        "fold": int(args.fold),
        "train_rows": int(len(train)),
        "heldout_rows": int(len(heldout)),
        "train_cases": sorted(np.unique(case[train]).astype(int).tolist()),
        "heldout_cases": sorted(np.unique(case[heldout]).astype(int).tolist()),
        "trees": BASE_TREES,
        "params": base_params(metadata),
        "fit_seconds": elapsed,
        "heldout": summarize_residual(label[heldout], prediction),
        "model": str(model_path),
        "prediction_file": str(pred_path),
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    strict_json(output / f"base_fold{args.fold}_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_OOF_BASE_FOLD_DONE", flush=True)


def merge_fold_predictions(
    cache: Path,
    output: Path,
    *,
    prefix: str,
    destination: str,
) -> tuple[Path, np.ndarray]:
    metadata = load_metadata(cache)
    case = mmap(cache, metadata, "case")
    n_rows = int(metadata["n_rows"])
    destination_path = output / destination
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite {destination_path}")
    temporary = output / f".{destination}.{os.getpid()}.tmp.npy"
    assembled = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(n_rows,)
    )
    assembled[:] = np.nan
    coverage = np.zeros(n_rows, dtype=np.uint8)
    for fold in range(N_FOLDS):
        path = output / f"{prefix}_fold{fold}_prediction.npz"
        with np.load(path) as data:
            index = data["index"].astype(np.int64, copy=False)
            prediction = data["prediction"].astype(np.float32, copy=False)
        if len(index) != len(prediction) or np.any(coverage[index]):
            raise RuntimeError(f"overlap or shape error in {path}")
        assembled[index] = prediction
        coverage[index] = 1
    fit = np.asarray(case) >= FIT_CASE_MIN
    if not np.all(coverage[fit] == 1) or np.any(coverage[~fit]):
        raise RuntimeError("fold predictions do not cover fit rows exactly once")
    if not np.isfinite(assembled[fit]).all() or not np.isnan(assembled[~fit]).all():
        raise RuntimeError("assembled prediction has invalid finite coverage")
    assembled.flush()
    os.replace(temporary, destination_path)
    return destination_path, fit


def stage_merge_base(args: argparse.Namespace) -> None:
    cache, output = Path(args.cache), Path(args.output_dir)
    destination, fit = merge_fold_predictions(
        cache, output, prefix="base", destination="base_oof.npy"
    )
    metadata = load_metadata(cache)
    label = mmap(cache, metadata, "label")
    prediction = np.load(destination, mmap_mode="r")
    payload = {
        "stage": "merge_base",
        "prediction_file": str(destination),
        "fit": summarize_residual(label[fit], prediction[fit]),
        "fold_summaries": [
            json.load((output / f"base_fold{fold}_summary.json").open(encoding="utf-8"))
            for fold in range(N_FOLDS)
        ],
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    strict_json(output / "base_oof_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_OOF_BASE_MERGE_DONE", flush=True)


def train_bias(args: argparse.Namespace, final: bool) -> None:
    cache, output = Path(args.cache), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(cache)
    case = mmap(cache, metadata, "case")
    fold_by_case = mmap(cache, metadata, "fold_by_case")
    x_scaled = mmap(cache, metadata, "x_scaled")
    x_raw = mmap(cache, metadata, "x_raw")
    label = mmap(cache, metadata, "label")
    base_oof = np.load(output / "base_oof.npy", mmap_mode="r")
    fold = None if final else int(args.fold)
    train = indices_for(case, fold_by_case, heldout_fold=fold, training=True)
    residual = np.asarray(label[train], dtype=np.float64) - np.asarray(
        base_oof[train], dtype=np.float64
    )
    target, target_mean, target_std = standardize(residual)
    features = conditional_matrix(x_scaled, x_raw, base_oof, train)
    booster, elapsed = fit_booster(features, target, bias_params(), BIAS_TREES)
    suffix = "final" if final else f"fold{fold}"
    model_path = output / f"bias_{suffix}.json"
    atomic_model(booster, model_path)
    payload: dict[str, Any] = {
        "stage": "bias_final" if final else "bias_fold",
        "fold": fold,
        "train_rows": int(len(train)),
        "train_cases": sorted(np.unique(case[train]).astype(int).tolist()),
        "target_standardization": {"mean": target_mean, "std": target_std},
        "conditional_features": CONDITIONAL_FEATURES,
        "trees": BIAS_TREES,
        "params": bias_params(),
        "fit_seconds": elapsed,
        "model": str(model_path),
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    if final:
        correction = physical_predict(booster, features, target_mean, target_std)
        payload["training_prediction"] = summarize_residual(residual, correction)
        payload["mean_alignment_offset"] = float(target_mean - correction.mean())
    else:
        heldout = indices_for(
            case, fold_by_case, heldout_fold=int(fold), training=False
        )
        held_features = conditional_matrix(x_scaled, x_raw, base_oof, heldout)
        correction = physical_predict(
            booster, held_features, target_mean, target_std
        )
        pred_path = output / f"bias_fold{fold}_prediction.npz"
        atomic_npz(
            pred_path,
            index=heldout.astype(np.int32),
            prediction=correction.astype(np.float32),
        )
        held_residual = np.asarray(label[heldout], dtype=np.float64) - np.asarray(
            base_oof[heldout], dtype=np.float64
        )
        payload["heldout_rows"] = int(len(heldout))
        payload["heldout_cases"] = sorted(
            np.unique(case[heldout]).astype(int).tolist()
        )
        payload["heldout_correction"] = summarize_residual(
            held_residual, correction
        )
        payload["prediction_file"] = str(pred_path)
    strict_json(output / f"bias_{suffix}_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_OOF_BIAS_FINAL_DONE" if final else "V22_OOF_BIAS_FOLD_DONE", flush=True)


def stage_merge_bias(args: argparse.Namespace) -> None:
    cache, output = Path(args.cache), Path(args.output_dir)
    destination, fit = merge_fold_predictions(
        cache, output, prefix="bias", destination="bias_oof.npy"
    )
    metadata = load_metadata(cache)
    label = mmap(cache, metadata, "label")
    base = np.load(output / "base_oof.npy", mmap_mode="r")
    correction = np.load(destination, mmap_mode="r")
    before = np.asarray(label[fit], dtype=np.float64) - np.asarray(
        base[fit], dtype=np.float64
    )
    payload = {
        "stage": "merge_bias",
        "prediction_file": str(destination),
        "before": summarize_residual(label[fit], base[fit]),
        "after": summarize_residual(before, correction[fit]),
        "fold_summaries": [
            json.load((output / f"bias_fold{fold}_summary.json").open(encoding="utf-8"))
            for fold in range(N_FOLDS)
        ],
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    strict_json(output / "bias_oof_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_OOF_BIAS_MERGE_DONE", flush=True)


def stage_variance(args: argparse.Namespace) -> None:
    cache, output = Path(args.cache), Path(args.output_dir)
    metadata = load_metadata(cache)
    case = mmap(cache, metadata, "case")
    fold_by_case = mmap(cache, metadata, "fold_by_case")
    x_scaled = mmap(cache, metadata, "x_scaled")
    x_raw = mmap(cache, metadata, "x_raw")
    label = mmap(cache, metadata, "label")
    base = np.load(output / "base_oof.npy", mmap_mode="r")
    bias = np.load(output / "bias_oof.npy", mmap_mode="r")
    train = indices_for(case, fold_by_case, heldout_fold=None, training=True)
    centered = (
        np.asarray(label[train], dtype=np.float64)
        - np.asarray(base[train], dtype=np.float64)
        - np.asarray(bias[train], dtype=np.float64)
    )
    square = np.square(centered)
    target, target_mean, target_std = standardize(square)
    features = conditional_matrix(x_scaled, x_raw, base, train)
    booster, elapsed = fit_booster(
        features, target, variance_params(), VARIANCE_TREES
    )
    model_path = output / "variance_final.json"
    atomic_model(booster, model_path)
    predicted = physical_predict(booster, features, target_mean, target_std)
    quantiles = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    payload = {
        "stage": "variance",
        "train_rows": int(len(train)),
        "target": "(label - base_oof - bias_oof)^2",
        "target_standardization": {"mean": target_mean, "std": target_std},
        "target_quantiles": dict(zip(
            map(str, quantiles), np.quantile(square, quantiles).astype(float).tolist()
        )),
        "prediction_quantiles": dict(zip(
            map(str, quantiles), np.quantile(predicted, quantiles).astype(float).tolist()
        )),
        "negative_prediction_fraction": float(np.mean(predicted <= 0.0)),
        "conditional_features": CONDITIONAL_FEATURES,
        "trees": VARIANCE_TREES,
        "params": variance_params(),
        "fit_seconds": elapsed,
        "model": str(model_path),
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    strict_json(output / "variance_final_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_OOF_VARIANCE_DONE", flush=True)


def capped_mean_one_inverse(
    variance: np.ndarray,
    floor_fraction: float = 0.1,
    lower: float = 0.25,
    upper: float = 4.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return clipped inverse-variance weights with mean exactly one."""
    variance = np.asarray(variance, dtype=np.float64)
    positive = variance[np.isfinite(variance) & (variance > 0.0)]
    if not len(positive):
        raise ValueError("variance model has no positive finite prediction")
    floor = float(floor_fraction * np.median(positive))
    effective = np.maximum(np.where(np.isfinite(variance), variance, floor), floor)
    inverse = 1.0 / effective
    lo_scale, hi_scale = 0.0, 1.0 / np.mean(inverse)
    while float(np.clip(hi_scale * inverse, lower, upper).mean()) < 1.0:
        hi_scale *= 2.0
        if hi_scale > 1.0e12:
            raise RuntimeError("could not bracket mean-one inverse-variance scale")
    for _ in range(80):
        scale = 0.5 * (lo_scale + hi_scale)
        mean = float(np.clip(scale * inverse, lower, upper).mean())
        if mean < 1.0:
            lo_scale = scale
        else:
            hi_scale = scale
    weights = np.clip(hi_scale * inverse, lower, upper).astype(np.float32)
    weight_mean = float(weights.astype(np.float64).mean())
    if not np.isclose(weight_mean, 1.0, atol=2.0e-6, rtol=0.0):
        raise RuntimeError("failed to normalize capped inverse-variance weights")
    return weights, {
        "variance_floor": floor,
        "floor_fraction_of_positive_median": float(floor_fraction),
        "weight_lower_cap": float(lower),
        "weight_upper_cap": float(upper),
        "weight_mean": weight_mean,
        "weight_min": float(weights.min()),
        "weight_median": float(np.median(weights)),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(weights.max()),
    }


def stage_variance_weighted(args: argparse.Namespace) -> None:
    cache, output = Path(args.cache), Path(args.output_dir)
    metadata = load_metadata(cache)
    case = mmap(cache, metadata, "case")
    x_scaled = mmap(cache, metadata, "x_scaled")
    x_raw = mmap(cache, metadata, "x_raw")
    label = mmap(cache, metadata, "label")
    official = mmap(cache, metadata, "official_train")
    base_oof = np.load(output / "base_oof.npy", mmap_mode="r")
    train = np.flatnonzero(np.asarray(official, dtype=bool)).astype(np.int32)
    if np.any(case[train] < FIT_CASE_MIN):
        raise RuntimeError("official training mask includes external evaluation cases")

    with (output / "variance_final_summary.json").open(encoding="utf-8") as handle:
        variance_summary = json.load(handle)
    variance_booster = xgb.Booster({"device": "cpu", "n_jobs": -1})
    variance_booster.load_model(output / "variance_final.json")
    variance_features = conditional_matrix(x_scaled, x_raw, base_oof, train)
    target_spec = variance_summary["target_standardization"]
    variance_prediction = physical_predict(
        variance_booster,
        variance_features,
        float(target_spec["mean"]),
        float(target_spec["std"]),
    )
    weights, weight_summary = capped_mean_one_inverse(variance_prediction)
    del variance_features, variance_booster

    source = metadata["source_standardization"]
    target = (
        (np.asarray(label[train], dtype=np.float64) - float(source["mean"]))
        / float(source["std"])
    ).astype(np.float32)
    booster, elapsed = fit_booster(
        np.asarray(x_scaled[train], dtype=np.float32),
        target,
        base_params(metadata),
        BASE_TREES,
        weights=weights,
    )
    model_path = output / "variance_weighted_mean.json"
    atomic_model(booster, model_path)
    payload = {
        "stage": "variance_weighted",
        "train_rows": int(len(train)),
        "training_split": metadata["official_random_row_split"],
        "source_standardization": source,
        "weight_source": "conditional variance of case-OOF bias-centered residual",
        "weight_summary": weight_summary,
        "variance_prediction_quantiles": {
            str(q): float(np.quantile(variance_prediction, q))
            for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "trees": BASE_TREES,
        "params": base_params(metadata),
        "fit_seconds": elapsed,
        "model": str(model_path),
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    strict_json(output / "variance_weighted_mean_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_OOF_VARIANCE_WEIGHTED_DONE", flush=True)


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument(
        "stage",
        choices=[
            "base-fold",
            "merge-base",
            "bias-fold",
            "bias-final",
            "merge-bias",
            "variance",
            "variance-weighted",
        ],
    )
    output.add_argument("--cache", required=True)
    output.add_argument("--output-dir", required=True)
    output.add_argument("--fold", type=int)
    return output


def main() -> None:
    args = parser().parse_args()
    if args.stage in {"base-fold", "bias-fold"} and args.fold is None:
        raise SystemExit(f"{args.stage} requires --fold")
    if args.stage == "base-fold":
        stage_base_fold(args)
    elif args.stage == "merge-base":
        stage_merge_base(args)
    elif args.stage == "bias-fold":
        train_bias(args, final=False)
    elif args.stage == "bias-final":
        train_bias(args, final=True)
    elif args.stage == "merge-bias":
        stage_merge_bias(args)
    elif args.stage == "variance":
        stage_variance(args)
    elif args.stage == "variance-weighted":
        stage_variance_weighted(args)
    else:  # pragma: no cover
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
