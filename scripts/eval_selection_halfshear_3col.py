"""The same three-shape-column ladder as eval_selection_constgold_3col.py, on the HALF-SHEAR legs.

WHY. The delivered figure reports `m_sel = R_sim(cut)/R_sim(no cut) - 1`, which sums TWO effects:
    SUBPOPULATION  the selected galaxies simply respond differently. NOT a bias.
    SELECTION      the boundary moves with shear, so the two legs keep different objects. IS a bias.
So "+14.98% at R>0.90\"" is an upper bound on the selection part until the two are separated. This
separates them the same way the constgold version does:

  (1) UNSHEARED INTRINSIC   both legs project the SAME raw intrinsic shape -> the shape response is
      identically zero, R(no cut) == 0, and the ONLY way to get a non-zero number is for the
      selection to differ between legs. => PURE SELECTION.
  (2) SHEARED INTRINSIC     leg g projects the analytically sheared shape. => selection + subpop.
      This is the column the figure's m_sel comes from.
  (3) MEASURED              per-leg measured ngmix shape. => adds measurement error.

Column (1) divided by R(no cut) of column (2) is the selection bias in the standard normalisation,
directly comparable to the figure's m_sel. If the two agree, m_sel was already almost pure selection
and the figure stands as read; if column (1) is much smaller, the figure's large size-cut numbers are
mostly subpopulation and must be relabelled.

FIREWALL: half-shear legs only, truth-only. No model, no emulator, no constgold. CPU.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_selection_response import (  # noqa: E402
    CAT, CROWD, NN, build_base, apply_shear_to_ellipticity,
)

PX = 0.2


def two_means(p0, pg, gmed, pass0, passg):
    if int(pass0.sum()) == 0 or int(passg.sum()) == 0:
        return np.nan
    return (float(np.mean(pg[passg])) - float(np.mean(p0[pass0]))) / gmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--size-cuts", type=float, nargs="+", default=[2.5, 2.9, 3.0, 3.5, 4.0, 4.5])
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[24.5, 25.0, 25.5, 26.0])
    args = ap.parse_args()

    t0 = time.time()
    ru = build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                    args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]

    i1 = base["e1_input_rot0_p"].to_numpy(float)
    i2 = base["e2_input_rot0_p"].to_numpy(float)
    s1, s2 = apply_shear_to_ellipticity(i1, i2, gmed * gh1, gmed * gh2)
    m1_0 = base["measured_ngmix_g1_0"].to_numpy(float)
    m2_0 = base["measured_ngmix_g2_0"].to_numpy(float)
    m1_g = base["measured_ngmix_g1_g"].to_numpy(float)
    m2_g = base["measured_ngmix_g2_g"].to_numpy(float)
    pr = lambda a, b: a * gh1 + b * gh2

    pu = pr(i1, i2)                       # identical in both legs
    kinds = [("unsheared intrinsic", pu, pu),
             ("sheared intrinsic", pu, pr(s1, s2)),
             ("measured", pr(m1_0, m2_0), pr(m1_g, m2_g))]

    fin = iso.copy()
    for _, a, b in kinds:
        fin &= np.isfinite(a) & np.isfinite(b)
    print(f"\n  ISOLATED, finite in all three shape kinds: {int(fin.sum()):,}", flush=True)

    R0 = {k: two_means(a, b, gmed, fin, fin) for k, a, b in kinds}
    print("  R(no cut):  " + "   ".join(f"{k}={R0[k]:+.5f}" for k in R0))
    print("  (the unsheared column is 0 by construction; that is the point -- it removes the")
    print("   subpopulation term, so anything it reports under a cut is PURE selection.)")
    Rnorm = R0["sheared intrinsic"]

    rows = ([(f'R>{c*PX:.2f}"', "measured_flux_radius", c, True) for c in args.size_cuts] +
            [(f"mag<{c:g}", "measured_mag_auto", c, False) for c in args.mag_cuts])

    print("\n" + "=" * 112)
    print("HALF-SHEAR: shift m in three shape columns (cut applied PER LEG on the measured value)")
    print("=" * 112)
    print(f"  {'cut':>10} {'frac':>6} | {'(1) PURE SELECTION':^24} | {'(2) SHEARED INTR':^24} | "
          f"{'(3) MEASURED':^22}")
    print(f"  {'':>10} {'':>6} | {'R_sel':>10} {'R_sel/R':>12} | {'R':>10} {'m_sel':>12} | "
          f"{'R':>10} {'shift m':>10}")
    for lab, col, thr, keep_high in rows:
        x0 = base[col + "_0"].to_numpy(float)
        xg = base[col + "_g"].to_numpy(float)
        p0 = fin & ((x0 > thr) if keep_high else (x0 < thr))
        pg = fin & ((xg > thr) if keep_high else (xg < thr))
        vals = [two_means(a, b, gmed, p0, pg) for _, a, b in kinds]
        fr = 0.5 * (p0.mean() + pg.mean()) / max(fin.mean(), 1e-9)
        print(f"  {lab:>10} {fr:>6.3f} | {vals[0]:>+10.5f} {100*vals[0]/Rnorm:>+11.3f}% | "
              f"{vals[1]:>+10.5f} {100*(vals[1]/R0['sheared intrinsic']-1):>+11.3f}% | "
              f"{vals[2]:>+10.5f} {100*(vals[2]/R0['measured']-1):>+9.3f}%")
    print("\n  Compare column (1) 'R_sel/R' against column (2) 'm_sel'. Column (2) is what the figure")
    print("  plots. Any gap between them is the SUBPOPULATION term, which is not a selection bias.")
    print("HALFSHEAR_3COL_DONE", flush=True)


if __name__ == "__main__":
    main()
