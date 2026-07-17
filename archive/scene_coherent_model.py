"""Scene-level COHERENT-response forward model (lever-2, principled).

The production decomposition R_total = R_flow(self) + sum_j R_blend_pair(emulator) UNDER-supplies
the coherent response in crowded scenes because (a) the per-pair emulator drops faint/OOD neighbours
and (b) the true coherent (whole-field-sheared) response is SUPER-ADDITIVE -- larger than the sum of
independent per-pair marginals (toy_blend_linearity: +56%..+569% excess). A sum of per-pair terms
cannot represent that collective boost.

This script instead learns the coherent response R_full DIRECTLY as a forward model of a galaxy's
own properties + AGGREGATE g=0 scene-crowding features (n neighbours, total neighbour flux, closeness,
light-weighted trace), trained on the constant renders and CROSS-VALIDATED across independent case
realizations (k-fold by case). The field shear is coherent in a real survey, so R_full is the
physically relevant response; predicting it from g=0-computable scene features is a legitimate
forward model (same status as the emulator), not an empirical m-subtraction.

Data: constant_response_catalogue_train.feather (one row per (target primary, neighbour secondary)
pair; per-object coherent `response` = delta_et/(2g) repeated across the target's pair-rows).

Metric: held-out multiplicative bias m = <R_true>/<R_pred> - 1 (the ensemble shear calibration bias),
globally and per r_blend bin; additive c from sum_et. Ablation: own-features-only vs own+scene.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf

CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
ZP = 30.0
SBSI = "/home/z/Zekang.Zhang/SBSI"


def flux(mag):
    return np.power(10.0, -0.4 * (mag - ZP))


def build_per_object(cat_path, max_rows):
    cols = ["case", "input_index", "input_index_sec", "neighbored", "distance",
            "r_input_p", "Re_input_p", "sersic_n_input_p", "axis_ratio_input_p",
            "redshift_input_p", "r_input_s", "Re_input_s", "sersic_n_input_s",
            "response", "sum_et"]
    t = pf.read_table(cat_path, columns=cols)
    if max_rows and t.num_rows > max_rows:
        t = t.slice(0, max_rows)
    df = t.to_pandas()
    df["nbr"] = df["neighbored"].astype(bool) & np.isfinite(df["distance"].to_numpy())
    d = df["distance"].to_numpy(float)
    fs = flux(df["r_input_s"].to_numpy(float))
    df["flux_s"] = np.where(df["nbr"], fs, 0.0)
    df["flux_s_d2"] = np.where(df["nbr"], fs / np.clip(d, 0.1, None) ** 2, 0.0)
    df["flux_s_d1"] = np.where(df["nbr"], fs / np.clip(d, 0.1, None), 0.0)
    df["flux_s_Re2"] = np.where(df["nbr"], fs * df["Re_input_s"].to_numpy(float) ** 2, 0.0)
    df["dist_nbr"] = np.where(df["nbr"], d, np.nan)

    g = df.groupby(["case", "input_index"], sort=False)
    own = g.agg(
        response=("response", "mean"),
        sum_et=("sum_et", "mean"),
        r_p=("r_input_p", "first"),
        Re_p=("Re_input_p", "first"),
        n_p=("sersic_n_input_p", "first"),
        q_p=("axis_ratio_input_p", "first"),
        z_p=("redshift_input_p", "first"),
        n_nbr=("nbr", "sum"),
        sum_flux_s=("flux_s", "sum"),
        sum_flux_s_d2=("flux_s_d2", "sum"),
        sum_flux_s_d1=("flux_s_d1", "sum"),
        sum_flux_s_Re2=("flux_s_Re2", "sum"),
        min_dist=("dist_nbr", "min"),
        mean_dist=("dist_nbr", "mean"),
    ).reset_index()

    own["flux_p"] = flux(own["r_p"].to_numpy(float))
    own["flux_ratio"] = own["sum_flux_s"] / np.clip(own["flux_p"], 1e-12, None)
    own["min_dist"] = own["min_dist"].fillna(20.0)
    own["mean_dist"] = own["mean_dist"].fillna(20.0)
    # light-weighted scene trace proxy ~ (own size + sum of neighbour size*flux)/total flux
    tot_f = own["flux_p"] + own["sum_flux_s"]
    own["scene_trace"] = (own["flux_p"] * own["Re_p"] ** 2 + own["sum_flux_s_Re2"]) / np.clip(tot_f, 1e-12, None)
    return own


def merge_rblend(own, rb_lookup):
    if rb_lookup and os.path.exists(rb_lookup):
        rb = pf.read_table(rb_lookup).to_pandas()
        kcol = "R_blend" if "R_blend" in rb.columns else [c for c in rb.columns if c not in ("case", "input_index")][0]
        rb = rb[["case", "input_index", kcol]].rename(columns={kcol: "r_blend"})
        own = own.merge(rb, on=["case", "input_index"], how="left")
        own["r_blend"] = own["r_blend"].fillna(0.0)
    else:
        own["r_blend"] = np.nan
    return own


OWN_FEATURES = ["r_p", "Re_p", "n_p", "q_p", "z_p"]
SCENE_FEATURES = ["n_nbr", "sum_flux_s", "sum_flux_s_d2", "sum_flux_s_d1",
                  "flux_ratio", "min_dist", "mean_dist", "scene_trace", "flux_p"]


def kfold_predict(own, feats, n_folds, seed, xgb_kwargs):
    import xgboost as xgb
    cases = np.sort(own["case"].unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cases)
    folds = np.array_split(perm, n_folds)
    pred = np.full(len(own), np.nan)
    y = own["response"].to_numpy(float)
    X = own[feats].to_numpy(float)
    case_arr = own["case"].to_numpy()
    n_rounds = xgb_kwargs.pop("n_estimators", 400)
    for fi, hold in enumerate(folds):
        te = np.isin(case_arr, hold)
        tr = ~te
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dte = xgb.DMatrix(X[te])
        bst = xgb.train(xgb_kwargs, dtr, num_boost_round=n_rounds)
        pred[te] = bst.predict(dte)
        print(f"  fold {fi}: held-out cases {list(hold)}  n_te={te.sum():,}")
    return pred


def report(own, pred, tag):
    y = own["response"].to_numpy(float)
    m = float(np.mean(y) / np.mean(pred) - 1.0)
    c = float(np.mean(own["sum_et"].to_numpy(float)) / 2.0)
    print(f"\n[{tag}] GLOBAL held-out  <R_true>={np.mean(y):.4f}  <R_pred>={np.mean(pred):.4f}  "
          f"m = <R_true>/<R_pred>-1 = {m:+.2%}   c(sum_et/2)={c:+.5f}   N={len(y):,}")
    if own["r_blend"].notna().any():
        rb = own["r_blend"].to_numpy(float)
        edges = [-1e9, 0.02, 0.05, 0.10, 0.25, 1e9]
        labs = ["ISO(<0.02)", "[0.02,0.05)", "[0.05,0.10)", "[0.10,0.25)", ">=0.25"]
        print(f"  {'r_blend-bin':>12} {'R_true':>7} {'R_pred':>7} {'m':>8} {'N':>10}")
        for i in range(len(labs)):
            sel = (rb >= edges[i]) & (rb < edges[i + 1])
            if sel.sum() < 2000:
                continue
            mi = np.mean(y[sel]) / np.mean(pred[sel]) - 1.0
            print(f"  {labs[i]:>12} {np.mean(y[sel]):7.4f} {np.mean(pred[sel]):7.4f} {mi:+8.2%} {sel.sum():10,}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CBASE + "/constant_response_catalogue_train.feather")
    ap.add_argument("--rb-lookup", default=SBSI + "/results/blend_lookup_extnbrho_c0-39.feather")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    print("### Scene-level coherent-response forward model (k-fold by case) ###")
    print(f"catalogue={args.catalogue}")
    own = build_per_object(args.catalogue, args.max_rows)
    own = merge_rblend(own, args.rb_lookup)
    print(f"per-object rows: {len(own):,}  cases: {own['case'].nunique()}  "
          f"<R_full>={own['response'].mean():.4f}  <n_nbr>={own['n_nbr'].mean():.2f}")

    base_kwargs = dict(max_depth=args.max_depth, eta=args.lr, subsample=0.8,
                       colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist")

    print("\n--- baseline: predict global mean R for everyone (no features) ---")
    report(own, np.full(len(own), own["response"].mean()), "global-mean")

    print("\n--- ablation A: OWN features only (~ self-response; no scene) ---")
    predA = kfold_predict(own, OWN_FEATURES, args.n_folds, args.seed,
                          {**base_kwargs, "n_estimators": args.n_estimators})
    report(own, predA, "own-only")

    print("\n--- ablation B: OWN + SCENE features (coherent forward model) ---")
    predB = kfold_predict(own, OWN_FEATURES + SCENE_FEATURES, args.n_folds, args.seed,
                          {**base_kwargs, "n_estimators": args.n_estimators})
    report(own, predB, "own+scene")

    out = SBSI + "/results/scene_coherent_heldout.npz"
    np.savez(out, case=own["case"].to_numpy(), input_index=own["input_index"].to_numpy(),
             R_true=own["response"].to_numpy(), R_pred_ownscene=predB, R_pred_own=predA,
             r_blend=own["r_blend"].to_numpy())
    print(f"\nsaved held-out predictions -> {out}")
    print("SCENE_COHERENT_DONE")


if __name__ == "__main__":
    main()
