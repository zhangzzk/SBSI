"""Optuna hyperparameter search for the IN-DOMAIN blending-response emulator.

WHY THIS EXISTS. Every blend emulator in the project -- production `lsst_r`, the certified
`lsst_r_extnbr_ho`, and `lsst_r_extnbr_indom` -- carries BYTE-IDENTICAL XGBoost hyperparameters
(max_depth 8, min_child_weight 154, gamma 3.19, lr 0.0384, subsample 0.970, colsample 0.869),
inherited from a single Optuna search run on the ORIGINAL full-population fit. `retrain_extnbr.py`
says so in its own docstring ("the EXACT production hyperparameters ... any difference vs production
is the domain extension, not tuning"), and `models/studies/` contains no study for any `_extnbr*`
tag. The `n_trials: 100` in the configs is inert on that code path.

So when WORKLOG 28l concluded the -41.5% close-pair deficit is a "REPRESENTATIONAL limit", the
capacity knobs had never been re-searched on the restricted population that conclusion is about.

PARTIAL COVER FROM THE EXISTING EVIDENCE, stated so this is not oversold. XGBoost's
`min_child_weight` and `gamma` both scale with sample weight, so 28l's WEIGHT_CLOSE=5/20 sweep did
indirectly relax them (154 -> ~8 effective at K=20) and bought nothing beyond the first ~4 points.
That is real evidence those two are not binding. `max_depth` is the exception: it is weight-INVARIANT,
so no reweighting experiment has ever probed it. Depth 8 was chosen to fit the full population; here
the model must carve out a regime holding 0.09% of the rows using those same eight levels. Depth,
learning rate, and the sampling fractions are the genuinely untested axes.

PRE-REGISTERED EXPECTATION (recorded before the run, per the project's standing habit). The search
objective is GLOBAL R2 with an overfit penalty -- the same objective blendemu uses. Close pairs are
0.09% of the training rows, so global R2 is almost blind to them. I therefore expect the search to
improve global R2 by a small amount and to move the close-pair deficit LITTLE OR NOT AT ALL. If that
is what happens, "representational limit" survives with the depth loophole closed, and the next lever
is a close-pair-weighted OBJECTIVE (not just weighted samples). If instead the deficit moves
materially, 28l's conclusion needs revising and hyperparameter inheritance was the confound.

FIREWALL. `HELDOUT_MIN_CASE=40` restricts training AND the Optuna validation split to cases 40-199;
constgold (cases 0-39) is never read, so no hyperparameter is selected on constgold m. The
tuned model is written under a NEW tag (`lsst_r_extnbr_indom_tuned`) and touches neither the
certified `_ho` nor the existing `_indom`. Promotion, if any, must still be argued on the per-pair
ruler (`scripts/eval_rblend_gap.py`), never on constgold m.

Data path, cuts, rescale, standardization and the fixed-seed split are `retrain_extnbr`'s, reused
unchanged, so the ONLY difference from the existing `_indom` model is the hyperparameters.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

BE = "/home/z/Zekang.Zhang/blendemu"
sys.path.insert(0, BE)
sys.path.insert(0, os.path.join(BE, "scripts"))

from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402
import train_emulator as TE  # noqa: E402
import retrain_extnbr as RE  # noqa: E402


def close_pair_r2(bst, x_test, y_test, frac=0.2):
    """R2 on the CLOSEST `frac` of test pairs -- a diagnostic, never a selection criterion.

    `distance_scaled` is a monotone rescaling of separation, so the lowest quantile is the closest
    pairs. Reported alongside global R2 so we can see whether the two agree; the authoritative
    close-pair number is the per-pair ruler, not this.
    """
    d = x_test["distance_scaled"].to_numpy(float)
    thr = float(np.quantile(d, frac))
    m = d <= thr
    if m.sum() < 1000:
        return float("nan"), int(m.sum()), thr
    pred = bst.predict(xgb.DMatrix(x_test[m]))
    return float(r2_score(np.asarray(y_test)[m], pred)), int(m.sum()), thr


def main():
    n_trials = int(os.environ.get("N_TRIALS", "0")) or None
    cfg = load_config(os.environ["CONFIG_PATH"])
    tr = cfg["training"]
    n_trials = n_trials or int(tr["n_trials"])
    tag = tr["model_tag"]
    if "tuned" not in tag:
        raise SystemExit(f"refusing to run: model_tag {tag!r} would overwrite an existing emulator")

    print(f"### OPTUNA SEARCH  tag={tag}  n_trials={n_trials} ###")
    print(f"  regression_cuts: {tr['regression_cuts']}")
    print(f"  HELDOUT_MIN_CASE={os.environ.get('HELDOUT_MIN_CASE', '0')}  "
          f"WEIGHT_CLOSE={os.environ.get('WEIGHT_CLOSE', '0 (none)')}")

    DMtrain, DMtest, x_train, x_test, y_train, y_test, y_mean, y_std = \
        RE.load_regression_data_lowmem(cfg)

    # Baseline: the inherited production hyperparameters, refit here so the comparison is on the
    # SAME split and the same rows as every trial. This is the number the search must beat.
    prod = json.load(open(os.path.join(BE, "models/emulator_metadata_lsst_r.json")))
    base = dict(prod["tasks"]["regression"]["params"])
    base.update({"objective": "reg:squarederror", "n_jobs": -1,
                 "device": os.environ.get("XGB_DEVICE", "cuda"), "tree_method": "hist",
                 "booster": "gbtree", "disable_default_eval_metric": 1})
    metric = lambda predt, d: ("r2", r2_score(d.get_label(), predt))
    ev0 = {}
    t0 = time.time()
    bst0 = xgb.train(base, DMtrain, evals=[(DMtrain, "train"), (DMtest, "eval")],
                     evals_result=ev0, num_boost_round=2000, verbose_eval=False,
                     custom_metric=metric, maximize=True,
                     callbacks=[TE._early_stopping_callback(cfg, "regression", "r2", maximize=True)])
    cp0, ncp, thr = close_pair_r2(bst0, x_test, y_test)
    print(f"\n  BASELINE (inherited production params): R2={bst0.best_score:.6f}  "
          f"trees={data_utils.get_xgb_iteration_range(bst0)[1]}  {time.time() - t0:.0f}s")
    print(f"    close-pair R2 (closest 20%, n={ncp:,}, distance_scaled<={thr:.4f}): {cp0:.6f}")

    # The search itself: blendemu's own space and objective (R2 penalized by the train/eval gap),
    # unchanged, so this is "the same recipe, tuned" and not a different recipe.
    #
    # FINALIZE_ONLY=1 skips the search and refits/saves from the trials already in the study. That is
    # how the parallel `tune_emulator_worker.py` fan-out terminates: K workers fill the shared study
    # and save nothing, then exactly ONE finalizer writes the model, so there is no race on
    # models/regression_model_*.json.
    if os.environ.get("FINALIZE_ONLY") == "1":
        study = TE._load_or_create_study("regression", cfg, direction="maximize")
        n_done = len([t for t in study.get_trials(deepcopy=False)
                      if t.state.name in ("COMPLETE", "PRUNED")])
        n_complete = len([t for t in study.get_trials(deepcopy=False)
                          if t.state.name == "COMPLETE"])
        print(f"\n  FINALIZE-ONLY: reusing the existing study, adding no trials "
              f"({n_complete} complete, {n_done - n_complete} pruned)")
        if n_complete < 10:
            raise SystemExit(f"  REFUSING to finalize: only {n_complete} completed trials -- that is "
                             "not a search. Run the workers first.")
        n_trials = n_complete   # what gets recorded in the metadata, and what the promotion
                                # provenance guard reads. Must be the REAL count, never the request.
    else:
        study = TE.tune_regression(cfg, DMtrain, DMtest, n_trials)

    best = dict(base)
    best.update(study.best_params)
    print(f"\n  best params: {json.dumps(study.best_params, sort_keys=True)}")

    ev1 = {}
    t0 = time.time()
    bst = xgb.train(best, DMtrain, evals=[(DMtrain, "train"), (DMtest, "eval")],
                    evals_result=ev1, num_boost_round=2000, verbose_eval=200,
                    custom_metric=metric, maximize=True,
                    callbacks=[TE._early_stopping_callback(cfg, "regression", "r2", maximize=True)])
    best_trees = data_utils.get_xgb_iteration_range(bst)[1]
    cp1, _, _ = close_pair_r2(bst, x_test, y_test)
    print(f"\n  TUNED: R2={bst.best_score:.6f}  trees={best_trees}  {time.time() - t0:.0f}s")
    print(f"    close-pair R2 (closest 20%): {cp1:.6f}")
    print(f"\n  DELTA vs inherited:  global R2 {bst.best_score - bst0.best_score:+.6f}   "
          f"close-pair R2 {cp1 - cp0:+.6f}")
    print("  (global R2 is dominated by the 99.9% of rows that are NOT close pairs; the")
    print("   authoritative close-pair verdict is scripts/eval_rblend_gap.py on the per-pair ruler.)")

    model_dir = tr["model_dir"]
    features = tr["features"]
    model_path = TE._fname(model_dir, "regression_model.json", cfg)
    curve_path = TE._fname(model_dir, "regression_train_curve.npz", cfg)
    boundary = np.array([[x_train[f].min(), x_train[f].max()] for f in features])
    bst.save_model(model_path)
    np.savez(curve_path, train_r2=ev1["train"]["r2"], eval_r2=ev1["eval"]["r2"])
    meta_path = TE._update_metadata(
        cfg, "regression", model_path, features, boundary, best,
        standardization=(y_mean, y_std), train_curve_path=curve_path,
        metrics={"best_iteration": bst.best_iteration, "best_trees": best_trees,
                 "best_score": bst.best_score, "score_name": "r2",
                 "baseline_inherited_r2": float(bst0.best_score),
                 "close_pair_r2_tuned": cp1, "close_pair_r2_inherited": cp0,
                 "n_trials": n_trials})
    print(f"  saved model={model_path}\n  metadata={meta_path}")
    print(f"TUNE_INDOM_DONE R2={bst.best_score:.6f} baseline={bst0.best_score:.6f}")


if __name__ == "__main__":
    main()
