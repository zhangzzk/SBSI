"""Retrain the blending-response emulator on the V2.1 domain.

ONE LEVER vs the fiducial `lsst_r_extnbr_indom_tuned`: the training POPULATION. Same catalogue,
same features, same preprocessing, same fixed-seed split, same hyperparameters (see below), same
`HELDOUT_MIN_CASE` firewall. Only the rows change.

WHY A DRIVER AND NOT ANOTHER CONFIG. blendemu selects its training rows with box cuts --
`source_select_reg` compares five columns against five [min, max] pairs. Half the V2.1 domain is a
box (Re > 0.5") and fits there; the other half, true S/N > 10, is a CURVE in (mag, Re) and does
not. So this script wraps `source_select_reg` with `sbs_shear.domain.select_frame` and then reuses
blendemu's own loader, which is where every other detail of the recipe lives. Nothing is
reimplemented, and blendemu is not modified.

HYPERPARAMETERS ARE INHERITED FROM THE TUNED EMULATOR, NOT FROM PRODUCTION. `retrain_extnbr.py`
reads production's params; using those here would move two levers at once (params AND population)
relative to the fiducial. So we read the tuned emulator's own params. Note what this does and does
not claim: those params were tuned on the V2 in-domain population, so on V2.1 they are INHERITED,
not tuned. That is fine and deliberate -- AGENTS.md records that Optuna tuning is a null on
constgold m (+0.010 pts, sd 0.000, ~20x below seed error), so re-searching would buy nothing and
would break the one-lever comparison. It does mean this emulator must not be described as "tuned".

THE STORED BOX MATTERS AT INFERENCE. The emulator applies its own `cuts` when scoring, and returns
nothing outside them (AGENTS.md, "Two traps" #1: the in-domain emulator covers only 43.4% of the
wide population and silently collapses <R_blend>). The config's PRIMARY box is therefore set to the
V2.1 BOUNDING box, and this script asserts those YAML numbers against `sbs_shear.domain` rather
than trusting them -- a box looser than the domain lets the emulator extrapolate, and one tighter
drops rows that belong. The S/N curve itself is applied by SBSI at both train and score time.

FIREWALL: HELDOUT_MIN_CASE=40 -> trains on cases 40-199; gold cases 0-39 excluded. R_blend never
sees constgold.
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

from sbs_shear import domain as sbs_domain  # noqa: E402

FIDUCIAL_TAG = "lsst_r_extnbr_indom_tuned"


def install_domain_filter():
    """Wrap blendemu's row selection so every batch is narrowed to the V2.1 domain.

    Patched at module level because `retrain_extnbr.load_regression_data_lowmem` looks the function
    up on the module at call time. Counts are accumulated so the log states how many rows the curve
    removed on top of the box -- if that is ever 0, the patch silently did nothing.
    """
    original = data_utils.source_select_reg
    stats = {"box": 0, "domain": 0}

    def select_v21(dataset, cuts=None, **kw):
        out = original(dataset, cuts=cuts, **kw) if cuts is not None else original(dataset, **kw)
        stats["box"] += len(out)
        out = sbs_domain.select_frame(out)
        stats["domain"] += len(out)
        return out

    data_utils.source_select_reg = select_v21
    return stats


def assert_config_box_matches_domain(cfg):
    """The YAML box must be exactly the V2.1 bounding box. Checked, not trusted.

    cut order = [r_s, r_p, Re_s, Re_p, distance]; index 1 and 3 are the PRIMARY.
    """
    cuts = cfg["training"]["regression_cuts"]
    re_lo = float(cuts[3][0])
    mag_hi = float(cuts[1][1])
    want_re = sbs_domain.V21_RE_MIN
    # The faintest magnitude anywhere in the domain: the S/N = 10 limit at the SMALLEST kept size,
    # since the limit gets brighter as objects grow.
    want_mag = float(sbs_domain.sn_limiting_mag(sbs_domain.V21_RE_MIN))
    print(f"  config PRIMARY box: Re > {re_lo}, mag < {mag_hi}")
    print(f"  V2.1 bounding box : Re > {want_re}, mag < {want_mag:.3f}")
    if abs(re_lo - want_re) > 1e-9:
        raise SystemExit(f"config Re_p floor {re_lo} != domain V21_RE_MIN {want_re}")
    if not (want_mag - 0.02 <= mag_hi <= want_mag + 0.05):
        raise SystemExit(
            f"config r_p ceiling {mag_hi} is not the V2.1 bounding magnitude {want_mag:.3f}. "
            f"Looser lets the emulator extrapolate at inference; tighter drops domain rows.")
    # Neighbours must stay full-population -- that is the deliverable definition.
    if float(cuts[0][0]) > 13.0 or float(cuts[2][1]) < 10.0:
        raise SystemExit("SECONDARY cuts were narrowed; neighbours must stay full-population")


def main():
    cfg = load_config(os.environ["CONFIG_PATH"])
    tr = cfg["training"]
    tag = tr["model_tag"]
    if tag in (FIDUCIAL_TAG, "lsst_r_extnbr_ho", "lsst_r"):
        raise SystemExit(f"refusing to overwrite the certified emulator {tag!r}")

    print(f"### V2.1 EMULATOR RETRAIN  tag={tag} ###")
    print(f"  {sbs_domain.describe()}")
    assert_config_box_matches_domain(cfg)
    print(f"  HELDOUT_MIN_CASE={os.environ.get('HELDOUT_MIN_CASE', '0')}")

    stats = install_domain_filter()
    DMtrain, DMtest, x_train, x_test, y_train, y_test, y_mean, y_std = \
        RE.load_regression_data_lowmem(cfg)
    print(f"  rows after the BOX cuts       : {stats['box']:,}")
    print(f"  rows after the V2.1 S/N CURVE : {stats['domain']:,} "
          f"({100.0 * stats['domain'] / max(stats['box'], 1):.2f}% kept)")
    if stats["domain"] == stats["box"]:
        raise SystemExit("the S/N curve removed NOTHING -- the domain filter is not being applied")

    # Hyperparameters inherited from the tuned emulator (see the module docstring).
    fid_meta = os.path.join(BE, f"models/emulator_metadata_{FIDUCIAL_TAG}.json")
    params = dict(json.load(open(fid_meta))["tasks"]["regression"]["params"])
    params.update({"objective": "reg:squarederror", "n_jobs": -1,
                   "device": os.environ.get("XGB_DEVICE", "cpu"), "tree_method": "hist",
                   "booster": "gbtree", "disable_default_eval_metric": 1})
    print(f"  params inherited from {FIDUCIAL_TAG}: {params}")

    metric = lambda predt, d: ("r2", r2_score(d.get_label(), predt))   # noqa: E731
    evals_result = {}
    t0 = time.time()
    bst = xgb.train(params, DMtrain, evals=[(DMtrain, "train"), (DMtest, "eval")],
                    evals_result=evals_result, num_boost_round=2000, verbose_eval=100,
                    custom_metric=metric, maximize=True,
                    callbacks=[TE._early_stopping_callback(cfg, "regression", "r2", maximize=True)])
    best_trees = data_utils.get_xgb_iteration_range(bst)[1]
    print(f"  time {time.time() - t0:.1f}s  best_iter={bst.best_iteration}  trees={best_trees}  "
          f"R2={bst.best_score:.6f}")

    model_dir = tr["model_dir"]
    features = tr["features"]
    model_path = TE._fname(model_dir, "regression_model.json", cfg)
    curve_path = TE._fname(model_dir, "regression_train_curve.npz", cfg)
    boundary = np.array([[x_train[f].min(), x_train[f].max()] for f in features])
    bst.save_model(model_path)
    np.savez(curve_path, train_r2=evals_result["train"]["r2"], eval_r2=evals_result["eval"]["r2"])
    meta_path = TE._update_metadata(
        cfg, "regression", model_path, features, boundary, params,
        standardization=(y_mean, y_std), train_curve_path=curve_path,
        metrics={"best_iteration": bst.best_iteration, "best_trees": best_trees,
                 "best_score": bst.best_score, "score_name": "r2",
                 # The population is part of the model. Recorded so a later reader does not have to
                 # infer it from the tag, and so a mismatch is visible rather than latent.
                 "sbsi_domain": sbs_domain.metadata(),
                 "params_inherited_from": FIDUCIAL_TAG,
                 "rows_after_box": int(stats["box"]),
                 "rows_after_domain": int(stats["domain"])})
    print(f"  saved model={model_path}\n  metadata={meta_path}")
    print(f"V21_EMU_TRAIN_DONE trees={best_trees} R2={bst.best_score:.6f}")


if __name__ == "__main__":
    main()
