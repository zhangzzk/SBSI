#!/usr/bin/env python -B
"""GOAL 3 -- collider de-risk: does inferring TRUE properties from MEASURED observables and cutting
on the INFERRED truth dissolve the shear-dependent selection bias that makes direct measured cuts
catastrophic?

For a FIXED a-priori family of realistic windows we compute the certified m THREE ways, all on the
SAME window in true-property units:

  TARGET  (oracle)  : cut on the TRUE (shear-invariant) property r_input_p / Re_input_p
                      -> the Goal-1 clean bound (what we could achieve with perfect knowledge)
  CURE    (proposal): cut on thetahat = E[true | ALL measured observables]  (OOS multivariate
                      regression -- a proxy for the flow's posterior mean P(theta|x))
                      -> m_CURE is the COLLIDER FLOOR: the residual an *ideal* inference leaves,
                         because thetahat still inherits the shear-correlated part of the noise in x
  DISEASE (naive)   : cut on an OOS *isotonic* univariate calibration of the single raw measured
                      observable (measured_mag_auto / measured_flux_radius) onto true units.
                      Isotonic => this selection is IDENTICAL to a raw measured cut, just on-scale,
                      so DISEASE vs CURE isolates the value of MULTIVARIATE inference over naive
                      single-observable selection.

Reading:
  m_CURE ~ m_TARGET  << m_DISEASE  => inferring truth dissolves measured selection; collider small
                                      -> greenlight the principled flow-based inversion / head.
  m_CURE ~ m_DISEASE >> m_TARGET   => collider is fundamental; measurement noise is itself shear-
                                      correlated and inference cannot strip it -> need a different
                                      strategy (measured-selection classifier with its own response).

DIAGNOSTIC / feasibility ONLY.  The regressions are proxies for P(theta|x); NOTHING is wired into
the certified estimator, NO certified flow/classifier is trained, the window family is fixed here a
priori and never adjusted by watching |m|.  Firewall-safe.
"""
import argparse, os, sys, time, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_selection_robustness import load, join_measured, apply_override, MEAS_COLS, CB, DUMP

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

t0 = time.time(); _o = []
def emit(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _o.append(s)
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

SCENE_RBLEND = CB + "derisk/rblend_scene_prodrflow_c40-139.npz"   # pairs with certified per-bin R_flow
OUTNPZ = CB + "derisk/goal3_collider_{tag}.npz"
OUTTXT = CB + "derisk/goal3_collider_{tag}.txt"

# FIXED a-priori window family (true-property units).  mag = r_input_p, size = Re_input_p [arcsec].
MAG_WINDOWS  = [(24.0, 25.0), (25.0, 26.0), (24.0, 26.0), (26.0, 27.0), (24.5, 26.5)]
SIZE_WINDOWS = [(0.3, 0.5), (0.5, 1.0), (1.0, 1.5), (0.5, 1.5), (0.2, 0.3)]


def oof_multivariate(X, y, groups, n_splits=5):
    """Out-of-fold E[y | X] via HistGradientBoostingRegressor, GroupKFold by case (no leakage)."""
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(max_iter=400, max_leaf_nodes=63,
                                          learning_rate=0.05, l2_regularization=1.0,
                                          early_stopping=True, random_state=0)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred


def oof_isotonic(x, y, groups, n_splits=5):
    """Out-of-fold monotone-increasing calibration of a SINGLE observable onto true units.
    Isotonic => cutting thetahat_uni in [a,b] is identical to a raw measured cut (on true scale)."""
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(x.reshape(-1, 1), y, groups):
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ir.fit(x[tr], y[tr]); pred[te] = ir.predict(x[te])
    return pred


def r2(y, yhat):
    good = np.isfinite(y) & np.isfinite(yhat)
    ss = np.sum((y[good] - yhat[good]) ** 2); tot = np.sum((y[good] - np.mean(y[good])) ** 2)
    return 1.0 - ss / tot if tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--scene-rblend", default=SCENE_RBLEND)
    ap.add_argument("--min-n", type=int, default=50000)
    ap.add_argument("--min-cases", type=int, default=30)
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--resp-floor", type=float, default=0.05)
    args = ap.parse_args()

    log("loading dump + joins (certified r_sim/R_flow/R_blend, true mag+size, measured observables)")
    df = load(args.dump)
    df = join_measured(df)
    if args.scene_rblend and os.path.exists(args.scene_rblend):
        df = apply_override(df, args.scene_rblend, "R_blend")           # honest scene R_blend (Goal-1 parity)
        log(f"applied scene R_blend override: {args.scene_rblend.split('/')[-1]}")
    else:
        log("WARNING: scene R_blend not found; using dump (certified density lookup) R_blend")

    # keep rows with the full measured vector + both true properties finite (fair, common support)
    need = MEAS_COLS + ["r_input_p", "Re_input_p", "r_sim", "R_flow", "R_blend"]
    have = df[need].apply(lambda c: np.isfinite(c.to_numpy(float)))
    keep = have.all(axis=1).to_numpy()
    log(f"common-support rows (all measured + true mag/size + certified pieces finite): "
        f"{int(keep.sum()):,}/{len(df):,} ({100*keep.mean():.1f}%)")
    df = df.loc[keep].reset_index(drop=True)

    case = df.case.to_numpy(np.int64)
    ucase, cidx = np.unique(case, return_inverse=True); ncase = len(ucase)
    X = df[MEAS_COLS].to_numpy(float)
    truemag = df.r_input_p.to_numpy(float); truesize = df.Re_input_p.to_numpy(float)

    log("fitting OOS inference (GroupKFold by case) -- multivariate CURE + univariate DISEASE")
    hat_mag_multi  = oof_multivariate(X, truemag,  case);  log("  hat_mag_multi done")
    hat_size_multi = oof_multivariate(X, truesize, case);  log("  hat_size_multi done")
    hat_mag_uni  = oof_isotonic(df.measured_mag_auto.to_numpy(float),    truemag,  case); log("  hat_mag_uni done")
    hat_size_uni = oof_isotonic(df.measured_flux_radius.to_numpy(float), truesize, case); log("  hat_size_uni done")

    emit("=" * 108)
    emit(f"GOAL-3 COLLIDER DE-RISK  [tag={args.tag}]  -- infer truth from measured observables, cut on inferred truth")
    emit(f"  dump={args.dump.split('/')[-1]}  scene_rblend={os.path.basename(args.scene_rblend)}  ncase={ncase}  n={len(df):,}")
    emit("=" * 108)
    emit("OOS inference quality (R^2 of E[true|measured]):")
    emit(f"   mag : multivariate={r2(truemag,hat_mag_multi):.4f}   univariate(meas_mag_auto)={r2(truemag,hat_mag_uni):.4f}")
    emit(f"   size: multivariate={r2(truesize,hat_size_multi):.4f}   univariate(flux_radius)={r2(truesize,hat_size_uni):.4f}")
    emit(f"   residual std |thetahat-true|: mag multi={np.nanstd(hat_mag_multi-truemag):.3f} "
         f"uni={np.nanstd(hat_mag_uni-truemag):.3f} ; size multi={np.nanstd(hat_size_multi-truesize):.3f} "
         f"uni={np.nanstd(hat_size_uni-truesize):.3f}")
    emit("")

    rsim = df.r_sim.to_numpy(float); rflow = df.R_flow.to_numpy(float); rbl = df.R_blend.to_numpy(float)
    rng = np.random.default_rng(7)
    pick = rng.integers(0, ncase, size=(args.nboot, ncase))

    def compute_m(mask):
        n = int(mask.sum())
        ci = cidx[mask]
        sc_s = np.bincount(ci, weights=rsim[mask], minlength=ncase)
        sc_f = np.bincount(ci, weights=rflow[mask], minlength=ncase)
        sc_b = np.bincount(ci, weights=rbl[mask], minlength=ncase)
        sc_n = np.bincount(ci, minlength=ncase)
        nc = int((sc_n > 0).sum())
        denom = sc_f.sum() + sc_b.sum()
        if not denom or n < args.min_n or nc < args.min_cases:
            return dict(m=np.nan, sem=np.nan, z=np.nan, resp=float(denom/n) if n else np.nan, n=n, nc=nc)
        m = sc_s.sum() / denom - 1
        bm = sc_s[pick].sum(1) / (sc_f[pick].sum(1) + sc_b[pick].sum(1)) - 1
        sem = float(bm.std())
        return dict(m=float(m), sem=sem, z=float(abs(m)/sem) if sem > 0 else np.nan,
                    resp=float(denom/n), n=n, nc=nc)

    rows = []
    families = [("mag", MAG_WINDOWS, truemag, hat_mag_multi, hat_mag_uni),
                ("size", SIZE_WINDOWS, truesize, hat_size_multi, hat_size_uni)]
    for prop, wins, true_v, hat_multi, hat_uni in families:
        for lo, hi in wins:
            variants = {
                "TARGET":  np.isfinite(true_v)  & (true_v  >= lo) & (true_v  < hi),
                "CURE":    np.isfinite(hat_multi) & (hat_multi >= lo) & (hat_multi < hi),
                "DISEASE": np.isfinite(hat_uni)  & (hat_uni  >= lo) & (hat_uni  < hi),
            }
            res = {k: compute_m(v) for k, v in variants.items()}
            rows.append(dict(prop=prop, lo=lo, hi=hi, **{f"{k}_{f}": res[k][f]
                        for k in res for f in ("m", "sem", "z", "resp", "n", "nc")}))

    emit(f"THREE-WAY m PER WINDOW  (m%% +/- sem%%; collider FLOOR = CURE; disease = naive measured cut)")
    emit(f"   {'window':16s} {'n_true':>9s} | {'TARGET(true)':>16s} {'CURE(infer)':>16s} {'DISEASE(meas)':>16s} | {'cure/dis':>8s}")
    for r in rows:
        w = f"{r['prop']}{r['lo']:g}-{r['hi']:g}"
        def cell(k):
            m, s, z = r[f"{k}_m"], r[f"{k}_sem"], r[f"{k}_z"]
            if not np.isfinite(m): return f"{'n/a':>16s}"
            return f"{100*m:>+7.2f}+/-{100*s:<4.2f}"
        ratio = (abs(r["CURE_m"]) / abs(r["DISEASE_m"])) if np.isfinite(r["CURE_m"]) and np.isfinite(r["DISEASE_m"]) and r["DISEASE_m"] != 0 else np.nan
        emit(f"   {w:16s} {int(r['TARGET_n']):>9,d} | {cell('TARGET'):>16s} {cell('CURE'):>16s} {cell('DISEASE'):>16s} | {ratio:>8.2f}")
    emit("")

    # summary verdict metrics (non-degenerate windows only: mean total response above floor)
    def absm(k): return np.array([abs(r[f"{k}_m"]) for r in rows if np.isfinite(r[f"{k}_m"]) and abs(r[f"{k}_resp"]) > args.resp_floor])
    for k in ("TARGET", "CURE", "DISEASE"):
        a = absm(k)
        if len(a):
            emit(f"   {k:8s} |m| over non-degenerate windows: median={100*np.median(a):.2f}%%  worst={100*a.max():.2f}%%  (n={len(a)})")
    emit("")
    emit("READING: CURE~TARGET<<DISEASE => inference dissolves measured selection (collider small) -> build it.")
    emit("         CURE~DISEASE>>TARGET => collider fundamental; inference insufficient -> selection-response head.")

    outtxt = OUTTXT.format(tag=args.tag); outnpz = OUTNPZ.format(tag=args.tag)
    pd.DataFrame(rows).to_json(outtxt.replace(".txt", "_rows.json"), orient="records")
    np.savez(outnpz, rows=json.dumps(rows),
             r2_mag_multi=r2(truemag, hat_mag_multi), r2_mag_uni=r2(truemag, hat_mag_uni),
             r2_size_multi=r2(truesize, hat_size_multi), r2_size_uni=r2(truesize, hat_size_uni))
    open(outtxt, "w").write("\n".join(_o) + "\n")
    log(f"wrote {outtxt} and {outnpz}")


if __name__ == "__main__":
    main()
