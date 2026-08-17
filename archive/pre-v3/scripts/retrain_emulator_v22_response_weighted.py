"""Retrain a BlendEMU response model while allocating capacity to response-carrying pairs.

The only change from the source recipe is a feature-only sample weight derived
from the *frozen* source-model prediction.  Labels are never used to construct
weights.  This preserves the population conditional mean in the infinite-model
limit, while testing whether ordinary row-wise MSE under-allocates tree capacity
to the rare pairs that carry nearly all scene-response power.

The defaults reproduce the original V2.2 experiment.  ``RESPONSE_WEIGHT_SOURCE_TAG``
and ``RESPONSE_WEIGHT_OUTPUT_TAG`` deliberately make the same fixed recipe reusable
for a domain-transfer bridge test without copying or subtly drifting the training code.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import xgboost as xgb


SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BE = "/home/z/Zekang.Zhang/blendemu"
for _path in (SBSI_ROOT, BE, os.path.join(BE, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402
import retrain_extnbr as RE  # noqa: E402
import train_emulator as TE  # noqa: E402
DEFAULT_SOURCE_TAG = "lsst_r_extnbr_v22"


def main() -> None:
    alpha = float(os.environ["RESPONSE_WEIGHT_ALPHA"])
    cap = float(os.environ.get("RESPONSE_WEIGHT_CAP", "50"))
    token = str(os.environ["RESPONSE_WEIGHT_TOKEN"])
    mode = str(os.environ.get("RESPONSE_WEIGHT_MODE", "square"))
    n_trees = int(os.environ.get("RESPONSE_WEIGHT_N_TREES", "271"))
    gamma_override = os.environ.get("RESPONSE_WEIGHT_GAMMA")
    min_child_override = os.environ.get("RESPONSE_WEIGHT_MIN_CHILD_WEIGHT")
    source_tag = os.environ.get("RESPONSE_WEIGHT_SOURCE_TAG", DEFAULT_SOURCE_TAG)
    if alpha < 0 or cap <= 1 or n_trees <= 0:
        raise SystemExit("alpha must be >=0, cap >1, and tree count >0")
    if mode not in {"square", "positive_square"}:
        raise SystemExit(f"unsupported RESPONSE_WEIGHT_MODE={mode!r}")

    cfg = load_config(os.environ["CONFIG_PATH"])
    cuts = np.asarray(cfg["training"]["regression_cuts"], dtype=float)
    frozen_meta_path = os.path.join(BE, "models", f"emulator_metadata_{source_tag}.json")
    frozen_meta = json.load(open(frozen_meta_path, encoding="utf-8"))
    frozen_task = frozen_meta["tasks"]["regression"]
    source_cuts = np.asarray(frozen_task["cuts"], dtype=float)
    if cuts.shape != (5, 2) or not np.array_equal(cuts, source_cuts):
        raise SystemExit(
            "configuration/source regression cuts disagree: "
            f"config={cuts.tolist()} source={source_cuts.tolist()}"
        )
    tag = os.environ.get("RESPONSE_WEIGHT_OUTPUT_TAG", f"{source_tag}_rpow{token}")
    cfg["training"]["model_tag"] = tag
    metadata_path = os.path.join(BE, "models", f"emulator_metadata_{tag}.json")
    model_path = os.path.join(BE, "models", f"regression_model_{tag}.json")
    for path in (metadata_path, model_path):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    print(
        f"### RESPONSE-POWER-WEIGHTED source={source_tag} tag={tag} "
        f"alpha={alpha} cap={cap} ###"
    )
    dm_train0, dm_test0, x_train, x_test, y_train, y_test, y_mean, y_std = (
        RE.load_regression_data_lowmem(cfg)
    )
    frozen = xgb.Booster({"device": "cpu", "n_jobs": -1})
    frozen.load_model(os.path.join(BE, "models", frozen_task["model_file"]))
    frozen_standardization = frozen_task["standardization"]
    frozen_mean = float(frozen_standardization["mean"])
    frozen_std = float(frozen_standardization["std"])
    train_phys = frozen.predict(dm_train0) * frozen_std + frozen_mean
    test_phys = frozen.predict(dm_test0) * frozen_std + frozen_mean
    if mode == "square":
        train_signal = train_phys * train_phys
        test_signal = test_phys * test_phys
    else:
        train_signal = np.maximum(train_phys, 0.0) ** 2
        test_signal = np.maximum(test_phys, 0.0) ** 2
    mean_power = float(np.mean(train_signal))
    train_ratio = np.minimum(train_signal / mean_power, cap)
    test_ratio = np.minimum(test_signal / mean_power, cap)
    train_weight = 1.0 + alpha * train_ratio
    normalization = float(np.mean(train_weight))
    train_weight /= normalization
    test_weight = (1.0 + alpha * test_ratio) / normalization
    print(
        "  frozen prediction power: "
        f"mean={mean_power:.8g}; weight train mean={train_weight.mean():.6f} "
        f"p50={np.quantile(train_weight, .5):.4f} "
        f"p99={np.quantile(train_weight, .99):.4f} "
        f"max={train_weight.max():.4f}"
    )
    signal_formula = (
        "prediction_physical^2" if mode == "square"
        else "max(prediction_physical, 0)^2"
    )
    del dm_train0, dm_test0, train_phys, test_phys, train_signal, test_signal
    del train_ratio, test_ratio
    dm_train = xgb.DMatrix(x_train, y_train, weight=train_weight)
    dm_test = xgb.DMatrix(x_test, y_test, weight=test_weight)

    params = dict(frozen_task["params"])
    if gamma_override is not None:
        params["gamma"] = float(gamma_override)
    if min_child_override is not None:
        params["min_child_weight"] = float(min_child_override)
    params.update({
        "objective": "reg:squarederror", "n_jobs": -1,
        "device": os.environ.get("XGB_DEVICE", "cpu"), "tree_method": "hist",
        "booster": "gbtree", "disable_default_eval_metric": 0,
    })
    evals_result = {}
    t0 = time.time()
    booster = xgb.train(
        params, dm_train, evals=[(dm_train, "train"), (dm_test, "eval")],
        evals_result=evals_result, num_boost_round=n_trees, verbose_eval=50,
    )
    elapsed = time.time() - t0
    print(f"  trained fixed {n_trees} trees in {elapsed:.1f}s")

    features = cfg["training"]["features"]
    boundary = np.array([[x_train[f].min(), x_train[f].max()] for f in features])
    curve_path = TE._fname(cfg["training"]["model_dir"], "regression_train_curve.npz", cfg)
    booster.save_model(model_path)
    np.savez(
        curve_path,
        train_rmse=np.asarray(evals_result["train"]["rmse"], float),
        eval_rmse=np.asarray(evals_result["eval"]["rmse"], float),
    )
    TE._update_metadata(
        cfg, "regression", model_path, features, boundary, params,
        standardization=(y_mean, y_std), train_curve_path=curve_path,
        metrics={
            "best_iteration": n_trees - 1,
            "best_trees": n_trees,
            "score_name": "weighted_rmse",
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_test)),
            "response_power_weight": {
                "source_model": source_tag,
                "formula": (
                    f"(1 + alpha * min({signal_formula} / train_mean_power, cap)) "
                    "/ train_mean_raw_weight"
                ),
                "alpha": alpha,
                "cap": cap,
                "mode": mode,
                "train_mean_prediction_power": mean_power,
                "source_prediction_standardization": {
                    "mean": frozen_mean, "std": frozen_std,
                },
                "train_raw_weight_normalization": normalization,
                "label_used_in_weight": False,
                "minimum_training_case": int(os.environ.get("HELDOUT_MIN_CASE", "0")),
                "fixed_tree_count": n_trees,
                "parameter_overrides": {
                    "gamma": None if gamma_override is None else float(gamma_override),
                    "min_child_weight": (
                        None if min_child_override is None else float(min_child_override)
                    ),
                },
            },
        },
    )
    metadata = json.load(open(metadata_path, encoding="utf-8"))
    for task_name in ("classification", "self_response"):
        metadata["tasks"][task_name] = frozen_meta["tasks"][task_name]
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"saved model={model_path}\nmetadata={metadata_path}")
    print("V22_RESPONSE_WEIGHTED_TRAIN_DONE")


if __name__ == "__main__":
    main()
