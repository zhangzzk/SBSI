#!/usr/bin/env python
"""Read-only decomposition of the residual size-cut bias into ISOLATED (R_flow's domain) vs
BLENDED (the size-BLIND scene R_blend's domain), and into fine size sub-bins.

Additive residual of the certified estimator over a selection S is exactly the mean per-object
response mismatch:   m_S * resp_S = mean_{S}( r_sim - R_flow - R_blend ).
(Since m = sum(r_sim)/sum(R_flow+R_blend) - 1 and resp = sum(R_flow+R_blend)/n.)

So we report add = 100 * mean(r_sim - R_flow - R_blend) [%], case-bootstrapped for significance.
Purpose: after sz13cg, is the stuck size1.0-1.5 residual in the ISOLATED population (=> R_flow still
under-resolves, keep refining the flow's size target) or the BLENDED population (=> a size-BLIND
R_blend cannot carry the size dependence => the binned-R_blend floor, an owner-gated design question)?

TUNES NOTHING: reuses eval_selection_robustness.load/apply_override; pure diagnostic read.
"""
import argparse, numpy as np
import eval_selection_robustness as E


def stats(rsim, rflow, rbl, case, mask, nboot=400, seed=7):
    """additive residual = mean(r_sim - R_flow - R_blend); case-bootstrap sem."""
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, add=np.nan, sem=np.nan, z=np.nan, resp=np.nan, m=np.nan)
    resid = (rsim - rflow - rbl)[mask]
    denom = (rflow + rbl)[mask].sum()
    c = case[mask]
    uc, ci = np.unique(c, return_inverse=True); ncase = len(uc)
    # per-case sums for bootstrap
    s_resid = np.bincount(ci, weights=resid, minlength=ncase)
    s_n = np.bincount(ci, minlength=ncase).astype(float)
    add = float(resid.mean())
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, ncase, size=(nboot, ncase))
    bs = s_resid[pick].sum(1); bn = s_n[pick].sum(1)
    badd = bs / np.where(bn > 0, bn, np.nan)
    sem = float(np.nanstd(badd))
    resp = float(denom / n)
    m = float(rsim[mask].sum() / denom - 1) if denom else np.nan
    return dict(n=n, add=100 * add, sem=100 * sem, z=abs(add) / sem if sem > 0 else np.nan,
                resp=resp, m=100 * m)


def row(label, st):
    z = st["z"]
    zs = f"{z:5.1f}" if np.isfinite(z) else "   nan"
    print(f"  {label:22s} n={st['n']:>9,} | add={st['add']:+7.3f}% +/-{st['sem']:5.3f}  z={zs}  "
          f"| m={st['m']:+7.2f}%  resp={st['resp']:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=E.DUMP)
    ap.add_argument("--rflow-override", default=None)
    ap.add_argument("--rblend-override", default=None)
    ap.add_argument("--tag", default="diag")
    args = ap.parse_args()

    E.log("loading dump + joins")
    df = E.load(args.dump)
    E.log(f"loaded {len(df):,} rows, cases {int(df.case.min())}-{int(df.case.max())}")
    if args.rflow_override:
        df = E.apply_override(df, args.rflow_override, "R_flow")
    if args.rblend_override:
        df = E.apply_override(df, args.rblend_override, "R_blend")

    rsim = df.r_sim.to_numpy(float); rflow = df.R_flow.to_numpy(float); rbl = df.R_blend.to_numpy(float)
    case = df.case.to_numpy(np.int64)
    ngh = df.neighbored.to_numpy().astype(int)
    size = df.Re_input_p.to_numpy(float)
    fin = np.isfinite(size)

    print("=" * 96)
    print(f"SIZE ISO/BLEND DECOMPOSITION  [tag={args.tag}]  additive = mean(r_sim - R_flow - R_blend)")
    print(f"  rflow_override={args.rflow_override}")
    print(f"  rblend_override={args.rblend_override}")
    print("=" * 96)

    # --- the two stuck realistic windows, split iso/blend ---
    for lo, hi in [(0.5, 1.0), (1.0, 1.5)]:
        base = fin & (size >= lo) & (size < hi)
        nb = int(base.sum()); niso = int((base & (ngh == 0)).sum()); nbl = int((base & (ngh == 1)).sum())
        print(f"\nsize[{lo},{hi})  (N={nb:,}  iso={niso:,}={100*niso/nb:.0f}%  blend={nbl:,}={100*nbl/nb:.0f}%)")
        row("ALL", stats(rsim, rflow, rbl, case, base))
        row("ISOLATED (R_flow)", stats(rsim, rflow, rbl, case, base & (ngh == 0)))
        row("BLENDED  (R_blend)", stats(rsim, rflow, rbl, case, base & (ngh == 1)))

    # --- fine size sub-bins across [0.5,1.5], iso/blend, to localize the gradient ---
    edges = [0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
    print("\nFINE size sub-bins (additive, iso vs blend):")
    print(f"  {'window':22s} {'ALL':>10s} {'ISO':>10s} {'BLEND':>10s}   (add% ; iso%=frac isolated)")
    for lo, hi in zip(edges[:-1], edges[1:]):
        base = fin & (size >= lo) & (size < hi)
        if base.sum() < 20000:
            continue
        a_all = stats(rsim, rflow, rbl, case, base)
        a_iso = stats(rsim, rflow, rbl, case, base & (ngh == 0))
        a_bl = stats(rsim, rflow, rbl, case, base & (ngh == 1))
        fiso = 100 * a_iso["n"] / a_all["n"]
        print(f"  size[{lo:.2f},{hi:.2f})       {a_all['add']:+7.2f}   {a_iso['add']:+7.2f}   "
              f"{a_bl['add']:+7.2f}    (iso={fiso:2.0f}%, z_all={a_all['z']:.1f})")


if __name__ == "__main__":
    main()
