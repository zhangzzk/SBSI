#!/usr/bin/env python
"""FIREWALL / LEAK AUDIT: recompute the acceptance on IN-SAMPLE vs OUT-OF-SAMPLE cases.

The response target grid is built from constgold cases 0-99; the flow trains on cases <=80; the
acceptance dump spans cases 40-139. So cases 40-99 are IN-SAMPLE (target built partly from the same
objects the metric evaluates) and cases 100-139 are OUT-OF-SAMPLE (never used to build target/flow).

If the sub-percent result is REAL (not an in-sample / shared-noise artifact), the OOS cases must show
the same behavior as the in-sample cases. This reuses the certified estimator verbatim, restricting the
per-case sums to a case range. Reports both multiplicative m and additive residual (m*resp) per split.

TUNES NOTHING: read-only; reuses eval_selection_robustness.load/apply_override/realistic_selections.
"""
import argparse, numpy as np
import eval_selection_robustness as E


def est(rsim, rflow, rbl, mask):
    n = int(mask.sum())
    if n == 0:
        return None
    Ss = rsim[mask].sum(); Sf = rflow[mask].sum(); Sb = rbl[mask].sum()
    denom = Sf + Sb
    m = Ss / denom - 1 if denom else np.nan
    resp = denom / n
    add = float((rsim[mask] - rflow[mask] - rbl[mask]).mean())   # = m*resp exactly
    return dict(n=n, m=100 * m, add=100 * add, resp=resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=E.DUMP)
    ap.add_argument("--rflow-override", required=True)
    ap.add_argument("--rblend-override", required=True)
    ap.add_argument("--is-lo", type=int, default=40); ap.add_argument("--is-hi", type=int, default=100)
    ap.add_argument("--oos-lo", type=int, default=100); ap.add_argument("--oos-hi", type=int, default=140)
    args = ap.parse_args()

    E.log("loading dump + joins")
    df = E.load(args.dump)
    df = E.apply_override(df, args.rflow_override, "R_flow")
    df = E.apply_override(df, args.rblend_override, "R_blend")
    rsim = df.r_sim.to_numpy(float); rflow = df.R_flow.to_numpy(float); rbl = df.R_blend.to_numpy(float)
    case = df.case.to_numpy(np.int64)
    IS = (case >= args.is_lo) & (case < args.is_hi)
    OOS = (case >= args.oos_lo) & (case < args.oos_hi)
    E.log(f"IN-SAMPLE cases [{args.is_lo},{args.is_hi}): {IS.sum():,} rows, {len(np.unique(case[IS]))} cases; "
          f"OOS cases [{args.oos_lo},{args.oos_hi}): {OOS.sum():,} rows, {len(np.unique(case[OOS]))} cases")

    print("=" * 104)
    print("IN-SAMPLE (cases 40-99, target/flow saw these) vs OUT-OF-SAMPLE (cases 100-139, unseen)")
    print("  add = additive residual m*resp (physical bias, %); m = multiplicative (%)")
    print("=" * 104)
    print(f"{'selection':22s} {'kind':11s} | {'IS add':>7s} {'OOS add':>7s} {'d':>6s} | {'IS m%':>7s} {'OOS m%':>7s} | {'resp':>5s}")
    print("-" * 104)
    worst = []
    for label, kind, mask in E.realistic_selections(df):
        si = est(rsim, rflow, rbl, mask & IS)
        so = est(rsim, rflow, rbl, mask & OOS)
        if si is None or so is None:
            continue
        d = so["add"] - si["add"]
        print(f"{label:22s} {kind:11s} | {si['add']:+7.2f} {so['add']:+7.2f} {d:+6.2f} | "
              f"{si['m']:+7.2f} {so['m']:+7.2f} | {so['resp']:5.2f}")
        worst.append((label, kind, si, so, d))
    print("-" * 104)
    # summary: OOS additive distribution + biggest IS->OOS degradations
    oos_add = np.array([abs(w[3]["add"]) for w in worst])
    print(f"\nOOS additive |resid|: median={np.median(oos_add):.2f}%  90th={np.quantile(oos_add,0.9):.2f}%  max={oos_add.max():.2f}%")
    print(f"OOS selections with |add|>1.0%: {[w[0] for w in worst if abs(w[3]['add'])>1.0]}")
    print("\nLargest IN-SAMPLE -> OOS additive degradations (|d|):")
    for w in sorted(worst, key=lambda x: -abs(x[4]))[:8]:
        print(f"  {w[0]:22s} IS {w[2]['add']:+.2f}% -> OOS {w[3]['add']:+.2f}%  (d={w[4]:+.2f}%)")


if __name__ == "__main__":
    main()
