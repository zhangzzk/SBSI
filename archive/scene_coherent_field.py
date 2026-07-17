"""Scene-level coherent-response forward model v2 -- scene features from the FULL INPUT FIELD.

v1 (scene_coherent_model.py) failed because the constant response catalogue records at most ONE
neighbour per target (max n_nbr=1), so its "scene" features carry no multi-neighbour crowding -- the
exact information R_full's super-additive coherent boost depends on. Here we rebuild the scene features
from each case's input `gals_info` field (~700k galaxies with full positions/fluxes/sizes), via a KDTree
over ALL neighbours in the aperture (INCLUDING faint ones the emulator's r<28 cut drops). Target R_full
still comes from the measured constant-render per-object response. k-fold by case, held-out m per r_blend
bin. If the crowded/high-r_blend bin flattens now -> g=0 multi-neighbour scene info predicts the coherent
response (lever-2 solvable as a forward model); if not -> it genuinely needs the render.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
ZP = 30.0
SBSI = "/home/z/Zekang.Zhang/SBSI"
TILE = "tile180.0_-0.5"
SHELLS = [2.0, 4.0, 7.0, 10.0]


def flux(mag):
    return np.power(10.0, -0.4 * (mag - ZP))


def per_object_Rfull(cat_path, cases):
    t = pf.read_table(cat_path, columns=["case", "input_index", "response", "sum_et"])
    df = t.to_pandas()
    df = df[df["case"].isin(cases)]
    g = df.groupby(["case", "input_index"], sort=False).agg(
        response=("response", "mean"), sum_et=("sum_et", "mean")).reset_index()
    return g


def field_scene_features(case, target_idx, base=CBASE, sign="0.02", r_max=10.0):
    """Multi-neighbour scene aggregates for target_idx (input indices) from the full input field."""
    ipath = f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"
    inp = pf.read_table(ipath, columns=["index_input", "RA_input", "DEC_input", "r_input",
                                        "Re_input", "sersic_n_input", "axis_ratio_input"]).to_pandas()
    ra = inp["RA_input"].to_numpy(float); dec = inp["DEC_input"].to_numpy(float)
    ra0 = ra.mean(); dec0 = dec.mean()
    x = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - dec0) * 3600.0
    f_all = flux(inp["r_input"].to_numpy(float))
    Re_all = inp["Re_input"].to_numpy(float)
    idx_all = inp["index_input"].to_numpy()
    pos = np.column_stack([x, y])
    tree = cKDTree(pos)

    row_of = {int(i): r for r, i in enumerate(idx_all)}
    trows = np.array([row_of.get(int(i), -1) for i in target_idx])
    ok = trows >= 0
    trows_ok = trows[ok]
    tpos = pos[trows_ok]

    nbr_lists = tree.query_ball_point(tpos, r=r_max, workers=-1)
    nT = len(target_idx)
    feats = {k: np.zeros(nT) for k in
             ["own_flux", "n_nbr", "sum_f", "sum_f_d2", "sum_f_d1", "sum_f_Re2",
              "near_d", "near_f", "flux_ratio", "scene_trace"]}
    for s in SHELLS:
        feats[f"n_{int(s)}"] = np.zeros(nT)
        feats[f"sf_{int(s)}"] = np.zeros(nT)
    feats["near_d"][:] = r_max
    pos_ok_i = np.where(ok)[0]
    for jj, ti in enumerate(pos_ok_i):
        trow = trows_ok[jj]
        own_f = f_all[trow]; own_Re = Re_all[trow]
        feats["own_flux"][ti] = own_f
        nbr = [n for n in nbr_lists[jj] if n != trow]
        if not nbr:
            feats["scene_trace"][ti] = own_Re ** 2
            continue
        nbr = np.asarray(nbr)
        d = np.hypot(pos[nbr, 0] - tpos[jj, 0], pos[nbr, 1] - tpos[jj, 1])
        fn = f_all[nbr]; Ren = Re_all[nbr]
        dc = np.clip(d, 0.1, None)
        feats["n_nbr"][ti] = len(nbr)
        feats["sum_f"][ti] = fn.sum()
        feats["sum_f_d2"][ti] = (fn / dc ** 2).sum()
        feats["sum_f_d1"][ti] = (fn / dc).sum()
        feats["sum_f_Re2"][ti] = (fn * Ren ** 2).sum()
        feats["near_d"][ti] = d.min()
        feats["near_f"][ti] = fn[np.argmin(d)]
        feats["flux_ratio"][ti] = fn.sum() / max(own_f, 1e-12)
        feats["scene_trace"][ti] = (own_f * own_Re ** 2 + (fn * Ren ** 2).sum()) / max(own_f + fn.sum(), 1e-12)
        for s in SHELLS:
            in_s = d < s
            feats[f"n_{int(s)}"][ti] = in_s.sum()
            feats[f"sf_{int(s)}"][ti] = fn[in_s].sum()
    out = pd.DataFrame(feats)
    out["input_index"] = np.asarray(target_idx)
    out["case"] = case
    out["_matched"] = ok
    return out


FEATS = (["own_flux", "n_nbr", "sum_f", "sum_f_d2", "sum_f_d1", "sum_f_Re2",
          "near_d", "near_f", "flux_ratio", "scene_trace"]
         + [f"n_{int(s)}" for s in SHELLS] + [f"sf_{int(s)}" for s in SHELLS])
OWN = ["own_r", "own_Re", "own_n", "own_q"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default=CBASE + "/constant_response_catalogue_train.feather")
    ap.add_argument("--rb-lookup", default=SBSI + "/results/blend_lookup_extnbrho_c0-39.feather")
    ap.add_argument("--cases", type=int, nargs="+", default=list(range(40)))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import xgboost as xgb

    print("### Scene-level coherent forward model v2 (field-based multi-neighbour features) ###")
    robj = per_object_Rfull(args.cat, set(args.cases))
    print(f"per-object R_full rows: {len(robj):,}  cases: {robj['case'].nunique()}")

    # own props from the input field (r, Re, n, q) + scene features
    parts = []
    for c in args.cases:
        tgt = robj[robj["case"] == c]["input_index"].to_numpy()
        sf = field_scene_features(c, tgt, sign="0.02", r_max=SHELLS[-1])
        parts.append(sf)
        print(f"  case {c}: {len(tgt):,} targets, matched {sf['_matched'].mean():.3f}, "
              f"<n_nbr>={sf['n_nbr'].mean():.1f}")
    scene = pd.concat(parts, ignore_index=True)

    # attach own morphology from input field (r,Re,n,q) via one more read per case merged already? add here:
    own_parts = []
    for c in args.cases:
        ipath = f"{CBASE}/case{c}_0.02/real0/catalogues/input/gals_info_{TILE}.feather"
        inp = pf.read_table(ipath, columns=["index_input", "r_input", "Re_input",
                                            "sersic_n_input", "axis_ratio_input"]).to_pandas()
        inp = inp.rename(columns={"index_input": "input_index", "r_input": "own_r",
                                  "Re_input": "own_Re", "sersic_n_input": "own_n",
                                  "axis_ratio_input": "own_q"})
        inp["case"] = c
        own_parts.append(inp)
    ownmorph = pd.concat(own_parts, ignore_index=True)

    df = robj.merge(scene, on=["case", "input_index"], how="inner")
    df = df.merge(ownmorph, on=["case", "input_index"], how="left")
    df = df[df["_matched"]].reset_index(drop=True)
    if args.rb_lookup and os.path.exists(args.rb_lookup):
        rb = pf.read_table(args.rb_lookup).to_pandas()
        kcol = "R_blend" if "R_blend" in rb.columns else [x for x in rb.columns if x not in ("case", "input_index")][0]
        df = df.merge(rb[["case", "input_index", kcol]].rename(columns={kcol: "r_blend"}),
                      on=["case", "input_index"], how="left")
        df["r_blend"] = df["r_blend"].fillna(0.0)
    else:
        df["r_blend"] = np.nan
    print(f"\nmerged rows: {len(df):,}  <R_full>={df['response'].mean():.4f}  <n_nbr>={df['n_nbr'].mean():.2f}")
    cache = SBSI + "/results/scene_field_features.feather"
    df.to_feather(cache)
    print(f"cached feature table -> {cache}")

    y = df["response"].to_numpy(float)
    case_arr = df["case"].to_numpy()
    cases = np.sort(df["case"].unique())
    rng = np.random.default_rng(args.seed)
    folds = np.array_split(rng.permutation(cases), args.n_folds)
    base = dict(max_depth=args.max_depth, eta=args.lr, subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", tree_method="hist")

    def cv_predict(feats, weight_tail=0.0):
        X = df[feats].to_numpy(float)
        pred = np.full(len(df), np.nan)
        for hold in folds:
            te = np.isin(case_arr, hold); tr = ~te
            w = None
            if weight_tail > 0:  # up-weight high-R_full tail so squared-error stops regressing it in
                w = 1.0 + weight_tail * np.clip(y[tr] - 0.5, 0, None) / 0.5
            dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w)
            bst = xgb.train(base, dtr, num_boost_round=args.n_estimators)
            pred[te] = bst.predict(xgb.DMatrix(X[te]))
        return pred

    def rep(tag, p):
        m = np.mean(y) / np.mean(p) - 1
        print(f"\n[{tag}] GLOBAL held-out m=<R_true>/<R_pred>-1 = {m:+.2%}  <R_true>={y.mean():.4f} <R_pred>={p.mean():.4f}")
        rb = df["r_blend"].to_numpy(float)
        edges = [-1e9, 0.02, 0.05, 0.10, 0.25, 1e9]
        labs = ["ISO(<0.02)", "[0.02,0.05)", "[0.05,0.10)", "[0.10,0.25)", ">=0.25"]
        print(f"  {'r_blend-bin':>12} {'R_true':>7} {'R_pred':>7} {'m':>8} {'N':>10}")
        for i in range(len(labs)):
            s = (rb >= edges[i]) & (rb < edges[i + 1])
            if s.sum() < 2000:
                continue
            print(f"  {labs[i]:>12} {y[s].mean():7.4f} {p[s].mean():7.4f} {y[s].mean()/p[s].mean()-1:+8.2%} {s.sum():10,}")

    p_scene = cv_predict(OWN + FEATS)
    rep("A own+field-scene", p_scene)
    p_stack = cv_predict(OWN + FEATS + ["r_blend"])
    rep("B own+field-scene+emulator_rblend", p_stack)
    p_tail = cv_predict(OWN + FEATS + ["r_blend"], weight_tail=3.0)
    rep("C stack + tail-weighted", p_tail)

    np.savez(SBSI + "/results/scene_coherent_field_heldout.npz",
             case=case_arr, input_index=df["input_index"].to_numpy(),
             R_true=y, R_pred=p_scene, R_pred_stack=p_stack, R_pred_tail=p_tail,
             r_blend=df["r_blend"].to_numpy())
    print("\nSCENE_COHERENT_FIELD_DONE")


if __name__ == "__main__":
    main()
