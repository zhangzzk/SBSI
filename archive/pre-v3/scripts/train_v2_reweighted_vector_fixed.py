#!/usr/bin/env python3
"""Transfer the frozen V2.2 vector-tuning recipe to the larger historical V2 domain.

No parameter is searched here.  The XGBoost parameters, positive-response weight,
tree count, cap, and seed are read from the completed V2.2 Optuna summary.  Both a
fresh ordinary base and its weighted from-scratch refit use half-shear cases 40--199
inside the V2 primary box (r < 26, Re > 0.3 arcsec).  ConstGold is never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU, BLENDEMU / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blendemu.config import load_config  # noqa: E402
import retrain_extnbr as RE  # noqa: E402


FEATURES = [
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
]
V2_TAG = "lsst_r_extnbr_indom_tuned"
V2_CUTS = [[13.0, 29.0], [18.0, 26.0], [0.0, 10.0], [0.3, 1.5], [0.0, 10.0]]
FIT_CASES = [40, 199]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def save_model(booster: xgb.Booster, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.json")
    booster.save_model(temporary)
    os.replace(temporary, path)


def physical_prediction(
    booster: xgb.Booster, matrix: xgb.DMatrix, mean: float, scale: float
) -> np.ndarray:
    return (booster.predict(matrix).astype(np.float64) * scale + mean).astype(np.float32)


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    centered = np.asarray(target, dtype=np.float64) - float(np.mean(target))
    return 1.0 - float(np.dot(residual, residual) / np.dot(centered, centered))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--recipe-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"Training already finalized: {summary_path}", flush=True)
        return

    recipe_path = Path(args.recipe_summary).resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    params = dict(recipe["best_params"])
    alpha = float(recipe["best_alpha"])
    cap = float(recipe["best_metrics"]["weight"]["cap"])
    n_trees = int(recipe["best_metrics"]["n_trees"])
    seed = int(recipe["study"]["seed"])
    if recipe["features"] != FEATURES or recipe["study"]["best_trial"] != 15:
        raise RuntimeError("unexpected source recipe provenance")

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    training = cfg["training"]
    cuts = np.asarray(training["regression_cuts"], dtype=float)
    if training["model_tag"] != V2_TAG:
        raise RuntimeError(f"expected V2 tag {V2_TAG}, got {training['model_tag']}")
    if cuts.shape != (5, 2) or not np.array_equal(cuts, np.asarray(V2_CUTS)):
        raise RuntimeError(f"V2 cuts drifted: {cuts.tolist()}")
    if list(training["features"]) != FEATURES:
        raise RuntimeError("V2 feature order differs from the frozen recipe")
    if int(os.environ.get("HELDOUT_MIN_CASE", "0")) != FIT_CASES[0]:
        raise RuntimeError("HELDOUT_MIN_CASE must be exactly 40")
    if float(os.environ.get("WEIGHT_CLOSE", "0")) != 0.0:
        raise RuntimeError("WEIGHT_CLOSE must remain disabled")

    print("### FIXED V2 REWEIGHTED-VECTOR TRANSFER ###", flush=True)
    print(
        f"domain r<26, Re>0.3; cases {FIT_CASES[0]}--{FIT_CASES[1]}; "
        f"alpha={alpha:.12g}; cap={cap:g}; trees={n_trees}; seed={seed}",
        flush=True,
    )
    dm_train0, dm_validation0, x_train, x_validation, y_train, y_validation, mean, scale = (
        RE.load_regression_data_lowmem(cfg)
    )
    mean, scale = float(mean), float(scale)
    source_metadata_path = BLENDEMU / "models" / f"emulator_metadata_{V2_TAG}.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_standardization = source_metadata["tasks"]["regression"]["standardization"]
    if not np.isclose(mean, float(source_standardization["mean"]), rtol=0.0, atol=1e-12):
        raise RuntimeError("V2 target mean differs from frozen V2 metadata")
    if not np.isclose(scale, float(source_standardization["std"]), rtol=0.0, atol=1e-12):
        raise RuntimeError("V2 target scale differs from frozen V2 metadata")

    fit_params = {
        **params,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cuda"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "12")),
        "booster": "gbtree",
        "verbosity": 0,
        "max_bin": 256,
        "seed": seed,
    }
    started = time.time()
    base_evals: dict[str, dict[str, list[float]]] = {}
    base = xgb.train(
        fit_params,
        dm_train0,
        evals=[(dm_train0, "train"), (dm_validation0, "validation")],
        evals_result=base_evals,
        num_boost_round=n_trees,
        verbose_eval=50,
    )
    base_train = physical_prediction(base, dm_train0, mean, scale)
    base_validation = physical_prediction(base, dm_validation0, mean, scale)
    train_signal = np.maximum(base_train.astype(np.float64), 0.0) ** 2
    validation_signal = np.maximum(base_validation.astype(np.float64), 0.0) ** 2
    mean_power = float(train_signal.mean())
    train_ratio = np.minimum(train_signal / mean_power, cap)
    validation_ratio = np.minimum(validation_signal / mean_power, cap)
    normalization = float((1.0 + alpha * train_ratio).mean())
    train_weight = ((1.0 + alpha * train_ratio) / normalization).astype(np.float32)
    validation_weight = ((1.0 + alpha * validation_ratio) / normalization).astype(np.float32)
    y_validation_physical = (
        np.asarray(y_validation, dtype=np.float64) * scale + mean
    ).astype(np.float32)
    base_validation_r2 = r2(y_validation_physical, base_validation)
    del dm_train0, dm_validation0, train_signal, validation_signal, train_ratio, validation_ratio

    dm_train = xgb.DMatrix(x_train, label=y_train, weight=train_weight, feature_names=FEATURES)
    dm_validation = xgb.DMatrix(
        x_validation, label=y_validation, weight=validation_weight, feature_names=FEATURES
    )
    weighted_evals: dict[str, dict[str, list[float]]] = {}
    weighted = xgb.train(
        fit_params,
        dm_train,
        evals=[(dm_train, "train"), (dm_validation, "validation")],
        evals_result=weighted_evals,
        num_boost_round=n_trees,
        verbose_eval=50,
    )
    weighted_validation = physical_prediction(weighted, dm_validation, mean, scale)
    weighted_validation_r2 = r2(y_validation_physical, weighted_validation)

    base_path = output / "base_model.json"
    weighted_path = output / "weighted_model.json"
    save_model(base, base_path)
    save_model(weighted, weighted_path)
    payload = {
        "schema_version": 1,
        "kind": "fixed V2-domain transfer of V2.2 reweighted-vector Optuna winner",
        "domain": {
            "name": "V2 rectangular primary domain",
            "primary_mag_max": 26.0,
            "primary_re_min_arcsec": 0.3,
            "regression_cuts": V2_CUTS,
            "fit_cases": FIT_CASES,
        },
        "fixed_recipe": {
            "source_summary": str(recipe_path),
            "source_summary_sha256": sha256(recipe_path),
            "source_best_trial": int(recipe["study"]["best_trial"]),
            "params": params,
            "alpha": alpha,
            "cap": cap,
            "n_trees": n_trees,
            "seed": seed,
            "weight_formula": (
                "(1 + alpha * min(max(fresh_base_prediction,0)^2 / "
                "train_mean_power, cap)) / train_mean_raw_weight"
            ),
            "weight_uses_label": False,
            "both_models_trained_from_scratch": True,
            "parameter_search_on_v2_domain": False,
        },
        "training": {
            "config": str(config_path),
            "n_train_rows": int(len(x_train)),
            "n_validation_rows": int(len(x_validation)),
            "target_mean": mean,
            "target_scale": scale,
            "elapsed_seconds": float(time.time() - started),
            "base_final_validation_rmse_standardized": float(
                base_evals["validation"]["rmse"][-1]
            ),
            "weighted_final_validation_rmse_standardized": float(
                weighted_evals["validation"]["rmse"][-1]
            ),
            "base_ordinary_validation_r2": base_validation_r2,
            "weighted_ordinary_validation_r2": weighted_validation_r2,
            "weight": {
                "train_mean_prediction_power": mean_power,
                "raw_weight_normalization": normalization,
                "train_weight_p50": float(np.quantile(train_weight, 0.50)),
                "train_weight_p99": float(np.quantile(train_weight, 0.99)),
                "train_weight_max": float(train_weight.max()),
            },
        },
        "features": FEATURES,
        "artifacts": {
            "base_model": str(base_path),
            "base_model_sha256": sha256(base_path),
            "weighted_model": str(weighted_path),
            "weighted_model_sha256": sha256(weighted_path),
        },
        "provenance": {
            "source_v2_metadata": str(source_metadata_path),
            "source_v2_metadata_sha256": sha256(source_metadata_path),
            "constgold_opened": False,
            "coherent_anchor_truth_opened": False,
        },
    }
    write_json(summary_path, payload)
    print(json.dumps(clean(payload), indent=2, sort_keys=True), flush=True)
    print("V2_REWEIGHTED_VECTOR_FIXED_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
