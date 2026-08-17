"""Retrain the V2 blending-response emulator on the V2.2 rectangular primary domain.

The only scientific lever relative to V2 is the primary population: true r < 25.8 and true
Re > 0.5 arcsec. The catalogue, features, fixed split, preprocessing, secondary support and
hyperparameters are inherited from the tuned V2 emulator. No constgold measurement is read.
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
for _p in (SBSI_ROOT, BE, os.path.join(BE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402
import train_emulator as TE  # noqa: E402
import retrain_extnbr as RE  # noqa: E402

FIDUCIAL_TAG = "lsst_r_extnbr_indom_tuned"
V22_TAG = "lsst_r_extnbr_v22"
V22_MAG_MAX = 25.8
V22_RE_MIN = 0.5
EXPECTED_CUTS = [[13.0, 29.0], [18.0, V22_MAG_MAX], [0.0, 10.0],
                 [V22_RE_MIN, 1.5], [0.0, 10.0]]


def assert_v22_config(cfg):
    tr = cfg["training"]
    if tr["model_tag"] != V22_TAG:
        raise SystemExit(f"expected model_tag={V22_TAG!r}, got {tr['model_tag']!r}")
    cuts = np.asarray(tr["regression_cuts"], dtype=float)
    if cuts.shape != (5, 2) or not np.array_equal(cuts, np.asarray(EXPECTED_CUTS)):
        raise SystemExit(f"V2.2 regression cuts drifted: {cuts.tolist()} != {EXPECTED_CUTS}")


def main():
    cfg = load_config(os.environ["CONFIG_PATH"])
    assert_v22_config(cfg)
    tr = cfg["training"]
    print(f"### V2.2 EMULATOR RETRAIN tag={tr['model_tag']} ###")
    print(f"  primary true 18 < r < {V22_MAG_MAX}; {V22_RE_MIN} < Re < 1.5 arcsec")
    print(f"  regression cuts: {tr['regression_cuts']}")
    print(f"  HELDOUT_MIN_CASE={os.environ.get('HELDOUT_MIN_CASE', '0')}")

    DMtrain, DMtest, x_train, x_test, y_train, y_test, y_mean, y_std = \
        RE.load_regression_data_lowmem(cfg)
    print(f"  split rows: train={len(x_train):,}, validation={len(x_test):,}")

    fid_meta = os.path.join(BE, f"models/emulator_metadata_{FIDUCIAL_TAG}.json")
    fid_task = json.load(open(fid_meta))["tasks"]["regression"]
    params = dict(fid_task["params"])
    params.update({"objective": "reg:squarederror", "n_jobs": -1,
                   "device": os.environ.get("XGB_DEVICE", "cpu"), "tree_method": "hist",
                   "booster": "gbtree", "disable_default_eval_metric": 1})
    if list(fid_task["features"]) != list(tr["features"]):
        raise SystemExit("V2.2 emulator features differ from the V2 fiducial")
    print(f"  params inherited from {FIDUCIAL_TAG}: {params}")

    metric = lambda predt, d: ("r2", r2_score(d.get_label(), predt))  # noqa: E731
    evals_result = {}
    t0 = time.time()
    bst = xgb.train(
        params, DMtrain, evals=[(DMtrain, "train"), (DMtest, "eval")],
        evals_result=evals_result, num_boost_round=2000, verbose_eval=100,
        custom_metric=metric, maximize=True,
        callbacks=[TE._early_stopping_callback(cfg, "regression", "r2", maximize=True)])
    best_trees = data_utils.get_xgb_iteration_range(bst)[1]
    print(f"  time {time.time() - t0:.1f}s best_iter={bst.best_iteration} "
          f"trees={best_trees} R2={bst.best_score:.6f}")

    model_dir = tr["model_dir"]
    features = tr["features"]
    model_path = TE._fname(model_dir, "regression_model.json", cfg)
    curve_path = TE._fname(model_dir, "regression_train_curve.npz", cfg)
    boundary = np.array([[x_train[f].min(), x_train[f].max()] for f in features])
    bst.save_model(model_path)
    np.savez(curve_path, train_r2=evals_result["train"]["r2"],
             eval_r2=evals_result["eval"]["r2"])
    meta_path = TE._update_metadata(
        cfg, "regression", model_path, features, boundary, params,
        standardization=(y_mean, y_std), train_curve_path=curve_path,
        metrics={"best_iteration": bst.best_iteration, "best_trees": best_trees,
                 "best_score": bst.best_score, "score_name": "r2",
                 "sbsi_domain": {"domain": "v2.2", "primary_mag_max": V22_MAG_MAX,
                                 "primary_re_min_arcsec": V22_RE_MIN,
                                 "description": "rectangular intrinsic primary cut"},
                 "params_inherited_from": FIDUCIAL_TAG,
                 "train_rows": int(len(x_train)), "validation_rows": int(len(x_test))})
    print(f"  saved model={model_path}\n  metadata={meta_path}")
    print(f"V22_EMU_TRAIN_DONE trees={best_trees} R2={bst.best_score:.6f}")


if __name__ == "__main__":
    main()
