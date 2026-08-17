"""WHY does np7 drop half the detected primaries? The answer decides whether the isolated-row fix exists.

`eval_iso_rows_target.py` established that np7 keeps 632,586 of the 1,265,725 detected in-domain
primaries in the ruler -- a 49.98%/50.02% split -- and that the SNC g=0 lookup covers 99.65% of the
kept ones and **0.00%** of the dropped ones. The lookup's row count also matches np7's almost exactly.
Two very different explanations fit that pattern, and they have opposite consequences:

  (H1) GEOMETRY. np7 emits no row for a primary with no neighbour inside 7". The dropped rows are
       isolated galaxies with perfectly good measured shapes, wrongly absent from the target.
       => The fix is real: add the rows with zero neighbour flux (and rebuild the g0 lookup, which
          inherited the same restriction).

  (H2) MEASUREMENT. np7 was built `--flow-only` (detected AND finite ngmix). ngmix converges on only
       about half of detected objects, and the g=0 lookup independently drops ngmix failures
       ("if a == -1.0 or (a == 0.0 and b == 0.0): continue"). The dropped rows have NO measured shape
       at all.
       => There is nothing to add. A galaxy with no shape cannot enter a shape-response target, and
          its absence is a correct exclusion, not a population defect.

The two are trivially separable: look at whether the dropped rows have a finite `measured_ngmix_g1`
in the ruler, which carries every primary regardless of what np7 did.

The decisive table is `finite ngmix` x `in np7`. Under H1 the dropped rows are mostly finite; under H2
they are mostly not. The script also reports whether np7's key set is exactly "detected AND finite
ngmix", which would settle it outright.

FIREWALL: half-shear catalogues only. Trains nothing, selects nothing, reads no constgold.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.feather as pf
from sbs_shear.paths import catalogue

KEY = ["case", "input_index"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler-cat", default=catalogue("det_meas_ngmix_g0.05_val.feather"))
    ap.add_argument("--np7-cat", default=catalogue("det_meas_ngmix_np7_g0.05_val.feather"))
    ap.add_argument("--g0-lookup", default="/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather")
    ap.add_argument("--max-case", type=int, default=4)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    args = ap.parse_args()

    cols = KEY + ["r_input_p", "Re_input_p", "neighbored", "distance", "detected",
                  "measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto"]
    d = ds.dataset(args.ruler_cat, format="feather")
    R = d.to_table(columns=[c for c in cols if c in d.schema.names],
                   filter=(ds.field("case") <= args.max_case)).to_pandas()
    R = R.drop_duplicates(KEY).reset_index(drop=True)
    print(f"ruler cases 0..{args.max_case}: {len(R):,} unique primaries")

    n7 = ds.dataset(args.np7_cat, format="feather")
    N7 = n7.to_table(columns=KEY, filter=(ds.field("case") <= args.max_case)).to_pandas()
    N7 = N7.drop_duplicates(KEY).reset_index(drop=True)
    in7 = np.asarray(pd.MultiIndex.from_frame(R[KEY]).isin(pd.MultiIndex.from_frame(N7[KEY])))
    print(f"np7   cases 0..{args.max_case}: {len(N7):,} unique primaries; "
          f"{in7.sum():,} of the ruler's rows are in it")

    det = R["detected"].astype(bool).to_numpy()
    fin = (np.isfinite(R["measured_ngmix_g1"].to_numpy(float))
           & np.isfinite(R["measured_ngmix_g2"].to_numpy(float)))
    # blendemu writes an ngmix failure as the sentinel -1.0, which IS finite -- treat it as a failure
    g1 = R["measured_ngmix_g1"].to_numpy(float)
    g2 = R["measured_ngmix_g2"].to_numpy(float)
    sent = (g1 == -1.0) | ((g1 == 0.0) & (g2 == 0.0))
    ok_shape = fin & ~sent
    dom = (det & (R["r_input_p"].to_numpy(float) < args.true_mag_max)
           & (R["Re_input_p"].to_numpy(float) > args.true_re_min))

    print(f"\n--- the decisive table: detected+in-domain primaries ---")
    print(f"{'':>22} {'in np7':>12} {'NOT in np7':>12}")
    for nm, m in (("usable ngmix shape", ok_shape), ("no usable shape", ~ok_shape)):
        a = int((dom & m & in7).sum())
        b = int((dom & m & ~in7).sum())
        print(f"{nm:>22} {a:>12,} {b:>12,}")
    nd = int(dom.sum())
    print(f"{'total':>22} {int((dom & in7).sum()):>12,} {int((dom & ~in7).sum()):>12,}   "
          f"(all detected in-domain = {nd:,})")

    drop = dom & ~in7
    print(f"\n  of the {int(drop.sum()):,} dropped detected in-domain primaries:")
    print(f"    with a usable ngmix shape : {int((drop & ok_shape).sum()):,} "
          f"({100 * (drop & ok_shape).sum() / max(drop.sum(), 1):.2f}%)")
    print(f"    sentinel / non-finite     : {int((drop & ~ok_shape).sum()):,} "
          f"({100 * (drop & ~ok_shape).sum() / max(drop.sum(), 1):.2f}%)")

    # is np7 EXACTLY "detected and usable shape"?
    pred = det & ok_shape
    print(f"\n--- is np7's key set exactly (detected AND usable ngmix)? ---")
    print(f"  predicted keeps : {int(pred.sum()):,}")
    print(f"  actual np7 keeps: {int(in7.sum()):,}")
    print(f"  agree on        : {100 * (pred == in7).mean():.2f}% of rows  "
          f"(in np7 but predicted out: {int((in7 & ~pred).sum()):,}; "
          f"predicted in but not in np7: {int((pred & ~in7).sum()):,})")

    # geometry check: if H1 were true the dropped rows would be the FAR ones
    print(f"\n--- geometry: is the drop a distance cut? ---")
    dist = R["distance"].to_numpy(float)
    for nm, m in (("in np7", dom & in7), ("dropped", dom & ~in7)):
        dd = dist[m]
        f_ = np.isfinite(dd)
        print(f"  {nm:>8}: finite distance {100 * f_.mean():5.2f}%  "
              f"median {np.median(dd[f_]) if f_.any() else float('nan'):.3f}\"  "
              f"neighbored {100 * R['neighbored'].to_numpy(bool)[m].mean():5.2f}%")
    print("  (a 7\" geometric cut would leave the dropped rows with systematically LARGER or")
    print("   non-finite neighbour distances; near-identical distributions rule H1 out.)")

    # g0 lookup, cross-tabbed against shape usability rather than np7 membership
    g0 = pf.read_table(args.g0_lookup, columns=KEY).to_pandas()
    g0 = g0[g0["case"] <= args.max_case]
    hasg0 = np.asarray(pd.MultiIndex.from_frame(R[KEY]).isin(pd.MultiIndex.from_frame(g0[KEY])))
    print(f"\n--- g0 lookup coverage vs shape usability (not vs np7) ---")
    for nm, m in (("usable shape", dom & ok_shape), ("no usable shape", dom & ~ok_shape)):
        print(f"  {nm:>16}: g0 coverage {100 * hasg0[m].mean() if m.any() else 0:.2f}%  "
              f"(n={int(m.sum()):,})")
    print("  If coverage tracks SHAPE USABILITY rather than np7 membership, the g0 lookup is not")
    print("  restricted by geometry either -- both simply require a converged ngmix measurement.")

    print("\n=== VERDICT ===")
    frac = (drop & ok_shape).sum() / max(drop.sum(), 1)
    if frac > 0.5:
        print(f"  H1 GEOMETRY: {100 * frac:.1f}% of dropped rows have a usable shape. They are real")
        print("  galaxies wrongly excluded -- the isolated-row fix is available, but the g0 lookup")
        print("  must be rebuilt to cover them before an SNC target can use them.")
    else:
        print(f"  H2 MEASUREMENT: only {100 * frac:.1f}% of dropped rows have a usable shape. The np7")
        print("  build is dropping ngmix FAILURES, not isolated galaxies. There is no population to")
        print("  add back: a galaxy with no measured shape cannot enter a shape-response target.")


if __name__ == "__main__":
    main()
