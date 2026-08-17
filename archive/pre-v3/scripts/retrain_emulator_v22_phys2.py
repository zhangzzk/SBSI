"""Retrain V2.2 R_blend using two dimensionless physical pair coordinates.

This is a feature-only ablation of ``lsst_r_extnbr_v22``.  It preserves the
V2.2 response catalogue, cases 40--199, labels, source cuts, row split,
standardization, secondary support, and inherited XGBoost hyperparameters.
The seven baseline inputs are replaced by exactly:

* log10[(Re_primary + Re_secondary) / distance]
* log10(F_secondary / F_primary)

The distance is the same response-catalogue distance used by V2.2.  Constgold
is not read and remains evaluation-only.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BE = "/home/z/Zekang.Zhang/blendemu"
for _path in (SBSI_ROOT, BE, os.path.join(BE, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402

V22_TAG = "lsst_r_extnbr_v22"
PHYS2_TAG = "lsst_r_extnbr_v22_phys2"
PHYS2_FEATURES = [
    "log10_re_sum_over_distance",
    "log10_flux_s_over_flux_p",
]
V22_FEATURES = [
    "Re_input_p_scaled", "Re_input_s_scaled",
    "r_input_p_scaled", "r_input_s_scaled",
    "sersic_n_input_p", "sersic_n_input_s",
    "distance_scaled",
]
V22_MAG_MAX = 25.8
V22_RE_MIN = 0.5
EXPECTED_CUTS = [
    [13.0, 29.0], [18.0, V22_MAG_MAX], [0.0, 10.0],
    [V22_RE_MIN, 1.5], [0.0, 10.0],
]
FEATURE_DEFINITIONS = {
    "log10_re_sum_over_distance": (
        "log10((Re_input_p + Re_input_s) / distance), using arcsec"
    ),
    "log10_flux_s_over_flux_p": (
        "log10(F_s/F_p) = -0.4 * (r_input_s - r_input_p)"
    ),
}


def assert_phys2_config(cfg: dict) -> None:
    """Fail closed if anything besides the intended feature ablation drifted."""
    training = cfg["training"]
    if training["model_tag"] != PHYS2_TAG:
        raise SystemExit(
            f"expected model_tag={PHYS2_TAG!r}, got {training['model_tag']!r}"
        )
    if list(training["features"]) != PHYS2_FEATURES:
        raise SystemExit(
            f"expected exactly the two physical features {PHYS2_FEATURES}, "
            f"got {training['features']}"
        )
    cuts = np.asarray(training["regression_cuts"], dtype=float)
    expected = np.asarray(EXPECTED_CUTS, dtype=float)
    if cuts.shape != (5, 2) or not np.array_equal(cuts, expected):
        raise SystemExit(
            f"V2.2 regression cuts drifted: {cuts.tolist()} != {EXPECTED_CUTS}"
        )


def assert_finite_features(name: str, frame) -> None:
    values = frame[PHYS2_FEATURES].to_numpy(dtype=float)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise RuntimeError(f"{name} contains {bad} non-finite conditioning values")


def main() -> None:
    # Keep the testable scientific guards importable in the lightweight py31
    # test environment, which intentionally lacks Optuna (imported by TE).
    import train_emulator as TE
    import retrain_extnbr as RE

    cfg = load_config(os.environ["CONFIG_PATH"])
    assert_phys2_config(cfg)
    training = cfg["training"]
    print(f"### V2.2 PHYS2 EMULATOR RETRAIN tag={training['model_tag']} ###")
    print(f"  features: {training['features']}")
    print(f"  definitions: {FEATURE_DEFINITIONS}")
    print(f"  regression cuts: {training['regression_cuts']}")
    print(f"  HELDOUT_MIN_CASE={os.environ.get('HELDOUT_MIN_CASE', '0')}")

    loaded = RE.load_regression_data_lowmem(cfg)
    (dm_train, dm_test, x_train, x_test, y_train, y_test,
     y_mean, y_std) = loaded
    assert_finite_features("training split", x_train)
    assert_finite_features("validation split", x_test)
    print(f"  split rows: train={len(x_train):,}, validation={len(x_test):,}")
    for feature in PHYS2_FEATURES:
        print(
            f"  {feature}: train=[{x_train[feature].min():.6f}, "
            f"{x_train[feature].max():.6f}]",
            flush=True,
        )

    baseline_path = os.path.join(
        BE, f"models/emulator_metadata_{V22_TAG}.json"
    )
    with open(baseline_path, encoding="utf-8") as handle:
        baseline_task = json.load(handle)["tasks"]["regression"]
    if list(baseline_task["features"]) != V22_FEATURES:
        raise SystemExit(
            f"baseline V2.2 features drifted: {baseline_task['features']}"
        )
    if not np.array_equal(
        np.asarray(baseline_task["cuts"], dtype=float),
        np.asarray(EXPECTED_CUTS, dtype=float),
    ):
        raise SystemExit("baseline V2.2 cuts differ from the phys2 training cuts")
    params = dict(baseline_task["params"])
    params.update({
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "device": os.environ.get("XGB_DEVICE", "cpu"),
        "tree_method": "hist",
        "booster": "gbtree",
        "disable_default_eval_metric": 1,
    })
    print(f"  params inherited from {V22_TAG}: {params}")

    metric = lambda predt, matrix: (  # noqa: E731
        "r2", r2_score(matrix.get_label(), predt)
    )
    evals_result = {}
    started = time.time()
    booster = xgb.train(
        params,
        dm_train,
        evals=[(dm_train, "train"), (dm_test, "eval")],
        evals_result=evals_result,
        num_boost_round=2000,
        verbose_eval=100,
        custom_metric=metric,
        maximize=True,
        callbacks=[
            TE._early_stopping_callback(
                cfg, "regression", "r2", maximize=True
            )
        ],
    )
    best_trees = data_utils.get_xgb_iteration_range(booster)[1]
    print(
        f"  time {time.time() - started:.1f}s best_iter={booster.best_iteration} "
        f"trees={best_trees} R2={booster.best_score:.6f}"
    )

    model_dir = training["model_dir"]
    model_path = TE._fname(model_dir, "regression_model.json", cfg)
    curve_path = TE._fname(model_dir, "regression_train_curve.npz", cfg)
    boundary = np.array([
        [x_train[feature].min(), x_train[feature].max()]
        for feature in PHYS2_FEATURES
    ])
    booster.save_model(model_path)
    np.savez(
        curve_path,
        train_r2=evals_result["train"]["r2"],
        eval_r2=evals_result["eval"]["r2"],
    )
    metadata_path = TE._update_metadata(
        cfg,
        "regression",
        model_path,
        PHYS2_FEATURES,
        boundary,
        params,
        standardization=(y_mean, y_std),
        train_curve_path=curve_path,
        metrics={
            "best_iteration": booster.best_iteration,
            "best_trees": best_trees,
            "best_score": booster.best_score,
            "score_name": "r2",
            "feature_ablation_of": V22_TAG,
            "feature_definitions": FEATURE_DEFINITIONS,
            "conditioning_features_only": True,
            "distance_source": (
                "response catalogue distance; identical source column to V2.2"
            ),
            "sbsi_domain": {
                "domain": "v2.2",
                "primary_mag_max": V22_MAG_MAX,
                "primary_re_min_arcsec": V22_RE_MIN,
                "description": "rectangular intrinsic primary cut",
            },
            "params_inherited_from": V22_TAG,
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_test)),
        },
    )
    print(f"  saved model={model_path}\n  metadata={metadata_path}")
    print(
        f"V22_PHYS2_EMU_TRAIN_DONE trees={best_trees} "
        f"R2={booster.best_score:.6f}"
    )


if __name__ == "__main__":
    main()
