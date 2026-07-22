#!/usr/bin/env python
"""DIAGNOSTIC (read-only): scan the certified estimator m on a FINE true-size grid to test whether
the large-size bias in the sz6eqw acceptance eval is a WITHIN-BIN response gradient (the response
target grid's top size bin [0.592,1.5] is too wide -> fixable with a finer tail grid) or a flat
calibration offset (capacity/target problem).  Reuses eval_selection_robustness.load/apply_override
so the numbers reproduce the acceptance harness exactly.  Tunes NOTHING.
"""
import sys, numpy as np
sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/scripts")
from eval_selection_robustness import load, apply_override, DUMP

CB = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
RFLOW = CB + "rflow_joint_sz6eqw_ens3_c40-139.npz"
RBLEND = CB + "rblend_scene_sz6eqwrflow_c40-139.npz"

def mstat(rsim, rflow, rbl, mask):
    n = int(mask.sum())
    if n == 0:
        return None
    s = rsim[mask].sum(); f = rflow[mask].sum(); b = rbl[mask].sum()
    denom = f + b
    m = s / denom - 1 if denom else np.nan
    resp = denom / n
    return dict(n=n, m=m, resp=resp, add=m * resp,
                rsim=s / n, rflow=f / n, rbl=b / n, rmod=(f + b) / n)

def main():
    print("loading dump + sz6eqw overrides ...", flush=True)
    df = load(DUMP)
    df = apply_override(df, RFLOW, "R_flow")
    df = apply_override(df, RBLEND, "R_blend")
    rsim = df.r_sim.to_numpy(float); rflow = df.R_flow.to_numpy(float); rbl = df.R_blend.to_numpy(float)
    size = df.Re_input_p.to_numpy(float)
    mag = df.r_input_p.to_numpy(float)
    fin = np.isfinite(size)
    print(f"rows={len(size):,}  finite size={int(fin.sum()):,}  "
          f"size q[1,5,50,95,99]={np.round(np.nanquantile(size,[.01,.05,.5,.95,.99]),3).tolist()}")

    # 1) FINE contiguous true-size windows spanning the full range.
    edges = [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
             0.65, 0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.5]
    print("\n=== FINE true-size WINDOWS  (all mag) ===")
    print(f"{'window':>14s} {'n':>10s} {'<rsim>':>8s} {'<Rmod>':>8s} {'m%':>9s} {'resp':>6s} {'add%':>8s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        r = mstat(rsim, rflow, rbl, fin & (size >= lo) & (size < hi))
        if r:
            print(f"  [{lo:.2f},{hi:.2f}) {r['n']:>10,d} {r['rsim']:>+8.4f} {r['rmod']:>+8.4f} "
                  f"{100*r['m']:>+8.2f} {r['resp']:>6.3f} {100*r['add']:>+7.3f}")

    # 2) Zoom INSIDE the current coarse top quantile bin [0.592, 1.5] -> is m a gradient?
    print("\n=== INSIDE coarse top grid bin [0.592,1.5]: sub-windows ===")
    sub = [0.592, 0.65, 0.72, 0.80, 0.90, 1.05, 1.25, 1.50]
    print(f"{'window':>14s} {'n':>10s} {'<rsim>':>8s} {'<Rmod>':>8s} {'m%':>9s} {'resp':>6s} {'add%':>8s}")
    for lo, hi in zip(sub[:-1], sub[1:]):
        r = mstat(rsim, rflow, rbl, fin & (size >= lo) & (size < hi))
        if r:
            print(f"  [{lo:.3f},{hi:.2f}) {r['n']:>10,d} {r['rsim']:>+8.4f} {r['rmod']:>+8.4f} "
                  f"{100*r['m']:>+8.2f} {r['resp']:>6.3f} {100*r['add']:>+7.3f}")

    # 3) Same but WITHIN the certified mag window 24-26 (the realistic regime).
    mw = fin & (mag >= 24.0) & (mag < 26.0)
    print("\n=== FINE true-size windows WITHIN mag 24-26 ===")
    print(f"{'window':>14s} {'n':>10s} {'<rsim>':>8s} {'<Rmod>':>8s} {'m%':>9s} {'resp':>6s} {'add%':>8s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        r = mstat(rsim, rflow, rbl, mw & (size >= lo) & (size < hi))
        if r and r['n'] > 5000:
            print(f"  [{lo:.2f},{hi:.2f}) {r['n']:>10,d} {r['rsim']:>+8.4f} {r['rmod']:>+8.4f} "
                  f"{100*r['m']:>+8.2f} {r['resp']:>6.3f} {100*r['add']:>+7.3f}")

    # 4) ISO vs BLEND split of the gap: decompose <rsim> vs <R_flow> vs <R_blend> for large size.
    ngh = df.neighbored.to_numpy().astype(int)
    iso = fin & (ngh == 0)
    bl = fin & (ngh == 1)
    print("\n=== LARGE-size gap decomposition: ISOLATED (R_blend~0) vs BLENDED ===")
    print(f"{'window':>14s} {'grp':>5s} {'n':>10s} {'<rsim>':>8s} {'<Rflow>':>8s} {'<Rbl>':>7s} "
          f"{'<Rmod>':>8s} {'flow_gap':>8s} {'m%':>8s}")
    for lo, hi in [(0.50, 0.60), (0.60, 0.75), (0.75, 1.00), (1.00, 1.50), (0.50, 1.50)]:
        for gl, gm in [("iso", iso), ("bl", bl)]:
            mk = gm & (size >= lo) & (size < hi)
            n = int(mk.sum())
            if n < 2000:
                continue
            rs = rsim[mk].mean(); rf = rflow[mk].mean(); rb = rbl[mk].mean()
            rmod = rf + rb
            print(f"  [{lo:.2f},{hi:.2f}) {gl:>5s} {n:>10,d} {rs:>+8.4f} {rf:>+8.4f} {rb:>+7.4f} "
                  f"{rmod:>+8.4f} {rs-rf:>+8.4f} {100*(rs/rmod-1) if rmod else 0:>+7.2f}")

    # 5) Harvested R_flow (ISOLATED) binned on the grid's (flux,size) edges vs the grid target.
    import numpy as _np
    g = _np.load("results/response_target_isoblend_snc_c0-99_6x6x5.npz")
    Rg = g["Rsim"]; ef = g["edges_flux"]; esz = g["edges_size"]
    fi = _np.clip(_np.digitize(mag, ef) - 1, 0, len(ef) - 2)
    si = _np.clip(_np.digitize(size, esz) - 1, 0, len(esz) - 2)
    print("\n=== ISOLATED: harvested <R_flow> vs grid target Rsim[flux,size,0]  (top 2 size bins) ===")
    print(f"{'flux(mag)':>12s} {'size':>10s} {'n':>9s} {'<Rflow>':>8s} {'grid_tgt':>9s} {'flow-tgt':>9s}")
    for f in range(len(ef) - 1):
        for s in [4, 5]:
            mk = iso & (fi == f) & (si == s)
            n = int(mk.sum())
            if n < 1000:
                continue
            rf = rflow[mk].mean()
            print(f"  {ef[f]:.1f}-{ef[f+1]:.1f} {esz[s]:.2f}-{esz[s+1]:.2f} {n:>9,d} "
                  f"{rf:>+8.3f} {Rg[f,s,0]:>+9.3f} {rf-Rg[f,s,0]:>+9.3f}")

    print("\nINTERPRETATION: (4) if ISOLATED <rsim> >> <Rflow> (flow_gap large, positive) the flow head "
          "under-predicts a correct target -> capacity/weighting fix. (5) flow-tgt < 0 in bright-large "
          "cells = the head cannot reach the high-response target it was trained on (saturation).")

if __name__ == "__main__":
    main()
