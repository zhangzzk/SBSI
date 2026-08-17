"""Retrain V2.2 with pair-specific absolute other-galaxy flux.

The response labels, cases 40--199, support cuts, fixed random split,
standardization, and inherited V2.2 XGBoost parameters are unchanged.  The
only added inputs are absolute intrinsic other-galaxy flux in three shells,
after excluding the primary and each row's designated secondary.  Constgold
is never read.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BE = "/home/z/Zekang.Zhang/blendemu"
for path in (ROOT, BE, os.path.join(BE, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402
from blendemu.scene_features import PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS  # noqa: E402
import train_emulator as TE  # noqa: E402
import retrain_extnbr as RE  # noqa: E402


BASE_TAG = "lsst_r_extnbr_v22"
TAG = "lsst_r_extnbr_v22_other3abs"
EXPECTED_CUTS = np.asarray([[13, 29], [18, 25.8], [0, 10], [0.5, 1.5], [0, 10]], float)


def summarize(name, booster, frame, target, mean, std):
    pred = data_utils.reverse_standardize(
        booster.predict(
            xgb.DMatrix(frame),
            iteration_range=data_utils.get_xgb_iteration_range(booster),
        ), mean, std,
    )
    label = data_utils.reverse_standardize(np.asarray(target), mean, std)
    residual = pred - label
    print(
        f"  {name}: N={len(label):,} label={label.mean():.6f} pred={pred.mean():.6f} "
        f"pred/label-1={100 * (pred.mean() / label.mean() - 1):+.3f}% "
        f"residual_std={residual.std(ddof=1):.6f} R2={r2_score(label, pred):.6f}",
        flush=True,
    )


def main():
    cfg = load_config(os.environ["CONFIG_PATH"])
    training = cfg["training"]
    if training["model_tag"] != TAG:
        raise SystemExit(f"wrong tag {training['model_tag']!r}")
    if not np.array_equal(np.asarray(training["regression_cuts"], float), EXPECTED_CUTS):
        raise SystemExit(f"V2.2 cuts drifted: {training['regression_cuts']}")
    if tuple(training["features"][-3:]) != PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS:
        raise SystemExit(f"wrong other-flux features: {training['features'][-3:]}")
    if os.environ.get("HELDOUT_MIN_CASE", "0") != "40":
        raise SystemExit("HELDOUT_MIN_CASE must be 40")

    RE._NEED = list(dict.fromkeys([*RE._NEED, *PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS]))
    print(
        f"### V2.2 PAIR-OTHER-ABS EMULATOR tag={TAG} ###\n"
        f"  features={list(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS)}",
        flush=True,
    )
    loaded = RE.load_regression_data_lowmem(cfg)
    dm_train, dm_test, x_train, x_test, y_train, y_test, y_mean, y_std = loaded
    base_task = json.load(open(
        os.path.join(BE, f"models/emulator_metadata_{BASE_TAG}.json"),
        encoding="utf-8",
    ))["tasks"]["regression"]
    params = dict(base_task["params"])
    params.update(
        objective="reg:squarederror", n_jobs=-1,
        device=os.environ.get("XGB_DEVICE", "cpu"), tree_method="hist",
        booster="gbtree", disable_default_eval_metric=1,
    )
    print(f"  parameters inherited from {BASE_TAG}: {params}", flush=True)
    metric = lambda prediction, data: ("r2", r2_score(data.get_label(), prediction))
    history = {}
    start = time.time()
    booster = xgb.train(
        params, dm_train, evals=[(dm_train, "train"), (dm_test, "eval")],
        evals_result=history, num_boost_round=2000, verbose_eval=100,
        custom_metric=metric, maximize=True,
        callbacks=[TE._early_stopping_callback(cfg, "regression", "r2", maximize=True)],
    )
    trees = data_utils.get_xgb_iteration_range(booster)[1]
    print(
        f"  time={time.time() - start:.1f}s best_iter={booster.best_iteration} trees={trees}",
        flush=True,
    )
    summarize("train", booster, x_train, y_train, y_mean, y_std)
    summarize("validation", booster, x_test, y_test, y_mean, y_std)

    model = TE._fname(training["model_dir"], "regression_model.json", cfg)
    curve = TE._fname(training["model_dir"], "regression_train_curve.npz", cfg)
    boundary = np.asarray([
        [x_train[feature].min(), x_train[feature].max()]
        for feature in training["features"]
    ])
    booster.save_model(model)
    np.savez(curve, train_r2=history["train"]["r2"], eval_r2=history["eval"]["r2"])
    metadata = TE._update_metadata(
        cfg, "regression", model, training["features"], boundary, params,
        standardization=(y_mean, y_std), train_curve_path=curve,
        metrics={
            "best_iteration": booster.best_iteration,
            "best_trees": trees,
            "best_score": booster.best_score,
            "score_name": "r2",
            "params_inherited_from": BASE_TAG,
            "train_rows": len(x_train),
            "validation_rows": len(x_test),
            "scene_features": list(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS),
            "scene_feature_source": (
                "intrinsic full rendered input scene; primary and designated secondary "
                "excluded; absolute simulation-count flux"
            ),
            "sbsi_domain": {
                "domain": "v2.2", "primary_mag_max": 25.8,
                "primary_re_min_arcsec": 0.5,
            },
        },
    )
    print(
        f"saved model={model}\nmetadata={metadata}\nV22_OTHER3ABS_EMU_TRAIN_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
