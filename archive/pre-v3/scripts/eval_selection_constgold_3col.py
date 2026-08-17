"""Shift m on constgold in THREE shape columns: unsheared intrinsic | sheared intrinsic | measured.

Owner's request. The three columns are a LADDER -- each adds exactly one ingredient, so differences
between adjacent columns are attributable:

  (1) UNSHEARED INTRINSIC   both legs project the SAME raw intrinsic shape.
      The shape response is identically zero, so R(no cut) == 0 and R at a FIXED boundary == 0 too.
      The only way to get a non-zero number is for the selection to DIFFER between the +g and -g
      legs. => this column is the PURE MOVING-BOUNDARY (selection) term, with nothing else in it.

  (2) SHEARED INTRINSIC     each leg projects its own analytically sheared intrinsic shape.
      Adds the true shape response, noise-free. => selection + SUBPOPULATION.

  (3) MEASURED              each leg projects the stored measured ngmix shape.
      Adds measurement error and its shear response. => the realistic number.

WHY THIS MATTERS -- AND A CAVEAT ON THE HALF-SHEAR m_sel. `m_sel = R(cut)/R(no cut) - 1` mixes TWO
distinct things:
    SUBPOPULATION  the selected galaxies simply respond differently (big galaxies are not a random
                   subset). This is NOT a bias -- a calibrated analysis of that subsample sees it.
    SELECTION      the boundary MOVES with shear, so the +g and -g legs keep different objects.
                   This IS the bias.
Column (1) isolates the second with no contamination, because the first is zero there by
construction. So this table is the check on how much of the half-shear `m_sel` (which reports the
sum) is really selection. Columns (2) and (3) additionally split R at a FIXED boundary (pass in BOTH
legs) from R at the MOVING boundary, which separates the same two effects a second way.

CUT AXIS. constgold stores a per-leg MEASURED brightness (`S/N_plus` / `S/N_minus`) but NO per-leg
measured size, so S/N is the only measured cut available here. That is useful rather than limiting:
S/N is the axis real analyses actually cut on, and the half-shear table could not test it.

NULL. A cut on a TRUE property (Re_input_p) has pass_plus == pass_minus by construction, so the
boundary cannot move and column (1) must return EXACTLY 0. Printed as a built-in check.

FIREWALL: constgold is read for EVALUATION only. Nothing is trained, fitted, or selected on it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow.feather as pf

CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")
COLS = ["case", "neighbored", "Re_input_p", "r_input_p",
        "axis_ratio_input_p", "position_angle_input_p",
        "applied_g1", "applied_g2", "shear_magnitude", "shear_angle",
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "et_plus", "response", "S/N_plus", "S/N_minus"]


def apply_shear(e1, e2, g1, g2):
    """Reduced-shear addition e' = (e + g) / (1 + g* e) for spin-2 ellipticity."""
    e = e1 + 1j * e2
    g = g1 + 1j * g2
    es = (e + g) / (1.0 + np.conj(g) * e)
    return es.real, es.imag


def leg_avg(etp, etm, g, passp, passm):
    """Two-means leg-average response: (<et>_+ over pass+ , <et>_- over pass-) / 2g."""
    if int(passp.sum()) == 0 or int(passm.sum()) == 0:
        return np.nan
    return (float(np.mean(etp[passp])) - float(np.mean(etm[passm]))) / (2.0 * g)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat", default=CG)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--sn-cuts", type=float, nargs="+", default=[5, 7, 10, 15, 20, 30])
    ap.add_argument("--re-null", type=float, nargs="+", default=[0.3, 0.5])
    args = ap.parse_args()

    print(f"loading {args.cat}", flush=True)
    t = pf.read_table(args.cat, columns=COLS, memory_map=True).to_pandas()
    t = t[t.case >= args.min_case].reset_index(drop=True)
    g = float(np.median(t.shear_magnitude.to_numpy(float)))
    print(f"N={len(t):,}  cases={t.case.min()}..{t.case.max()}  g={g:.4f}", flush=True)

    q = t.axis_ratio_input_p.to_numpy(float)
    pa = np.deg2rad(t.position_angle_input_p.to_numpy(float))
    e = (1.0 - q) / (1.0 + q)
    e1i, e2i = e * np.cos(2 * pa), e * np.sin(2 * pa)
    ag1, ag2 = t.applied_g1.to_numpy(float), t.applied_g2.to_numpy(float)
    e1p, e2p = apply_shear(e1i, e2i, ag1, ag2)
    e1m, e2m = apply_shear(e1i, e2i, -ag1, -ag2)

    sa = t.shear_angle.to_numpy(float)
    c2, s2 = np.cos(2 * sa), np.sin(2 * sa)
    me1p, me2p = t.measured_e1_plus.to_numpy(float), t.measured_e2_plus.to_numpy(float)
    me1m, me2m = t.measured_e1_minus.to_numpy(float), t.measured_e2_minus.to_numpy(float)
    # Lock the projection sign to the stored `et` convention, as eval_selection_constgold.py does.
    myp = me1p * c2 + me2p * s2
    st = t.et_plus.to_numpy(float)
    ok = np.isfinite(myp) & np.isfinite(st)
    sign = 1.0 if np.nanmean((myp * st)[ok]) > 0 else -1.0
    proj = lambda a, b: sign * (a * c2 + b * s2)

    eu = proj(e1i, e2i)                                   # (1) same in BOTH legs
    kinds = [("unsheared intrinsic", eu, eu),
             ("sheared intrinsic", proj(e1p, e2p), proj(e1m, e2m)),
             ("measured", proj(me1p, me2p), proj(me1m, me2m))]

    fin = np.ones(len(t), bool)
    for _, a, b in kinds:
        fin &= np.isfinite(a) & np.isfinite(b)
    snp, snm = t["S/N_plus"].to_numpy(float), t["S/N_minus"].to_numpy(float)
    fin &= np.isfinite(snp) & np.isfinite(snm)
    Re = t.Re_input_p.to_numpy(float)
    print(f"finite in all three shape kinds: {int(fin.sum()):,}", flush=True)

    R0 = {k: leg_avg(a, b, g, fin, fin) for k, a, b in kinds}
    print("\n  R(no cut):  " + "   ".join(f"{k}={R0[k]:+.5f}" for k in R0))
    print(f"  cross-check measured vs stored mean(response) = "
          f"{np.nanmean(t.response.to_numpy(float)):+.5f}")
    print("  NOTE R(no cut) for the unsheared column is 0 BY CONSTRUCTION (identical shapes, "
          "identical objects),")
    print("  so its shift m = R(cut)/R(no cut) - 1 is 0/0 and UNDEFINED. It is reported instead as")
    print("  R_sel and as R_sel / R(no cut, sheared intrinsic) -- the standard normalisation.")
    Rnorm = R0["sheared intrinsic"]

    print("\n" + "=" * 118)
    print("SHIFT m ON CONSTGOLD, THREE SHAPE COLUMNS   (cut on per-leg MEASURED S/N; boundary moves)")
    print("=" * 118)
    print(f"  {'cut':>8} {'frac':>6} | {'(1) UNSHEARED INTR':^26} | {'(2) SHEARED INTR':^26} | "
          f"{'(3) MEASURED':^26}")
    print(f"  {'':>8} {'':>6} | {'R_sel':>10} {'R_sel/R':>14} | {'R':>10} {'shift m':>14} | "
          f"{'R':>10} {'shift m':>14}")
    for thr in args.sn_cuts:
        pp, pm = fin & (snp > thr), fin & (snm > thr)
        both = pp & pm
        cells = []
        for k, a, b in kinds:
            Rmov = leg_avg(a, b, g, pp, pm)
            Rfix = leg_avg(a, b, g, both, both)
            cells.append((Rmov, Rfix))
        (Ru, Ru_f), (Rs, Rs_f), (Rm, Rm_f) = cells
        fr = 0.5 * (pp.mean() + pm.mean()) / max(fin.mean(), 1e-9)
        print(f"  {'S/N>%g' % thr:>8} {fr:>6.3f} | {Ru:>+10.5f} {100*Ru/Rnorm:>+13.3f}% | "
              f"{Rs:>+10.5f} {100*(Rs/R0['sheared intrinsic']-1):>+13.3f}% | "
              f"{Rm:>+10.5f} {100*(Rm/R0['measured']-1):>+13.3f}%")

    print("\n" + "-" * 118)
    print("SPLIT of the shift m into SUBPOPULATION (fixed boundary) vs SELECTION (moving boundary)")
    print("  subpop = R_fixed/R_nocut - 1   selection = R_moving/R_fixed - 1   (cols 2 and 3)")
    print("-" * 118)
    print(f"  {'cut':>8} | {'(2) subpop':>12} {'(2) selection':>15} | "
          f"{'(3) subpop':>12} {'(3) selection':>15} | {'(1) R_sel/R':>13}")
    for thr in args.sn_cuts:
        pp, pm = fin & (snp > thr), fin & (snm > thr)
        both = pp & pm
        row = []
        for k, a, b in kinds:
            Rmov = leg_avg(a, b, g, pp, pm)
            Rfix = leg_avg(a, b, g, both, both)
            row.append((Rfix / R0[k] - 1.0 if R0[k] else np.nan,
                        Rmov / Rfix - 1.0 if Rfix else np.nan, Rmov))
        print(f"  {'S/N>%g' % thr:>8} | {100*row[1][0]:>+11.3f}% {100*row[1][1]:>+14.3f}% | "
              f"{100*row[2][0]:>+11.3f}% {100*row[2][1]:>+14.3f}% | "
              f"{100*row[0][2]/Rnorm:>+12.3f}%")

    print("\n  --- NULL: cut on TRUE Re (boundary CANNOT move) -> column (1) must be exactly 0 ---")
    for thr in args.re_null:
        keep = fin & (Re > thr)
        Ru = leg_avg(eu, eu, g, keep, keep)
        Rs = leg_avg(kinds[1][1], kinds[1][2], g, keep, keep)
        print(f"  {'Re>%.2f' % thr:>8}  frac={keep.mean()/max(fin.mean(),1e-9):.3f}  "
              f"(1) R_sel={Ru:+.3e}  |  (2) shift m={100*(Rs/R0['sheared intrinsic']-1):+.3f}% "
              f"<- SUBPOPULATION only, not a bias")
    print("CONSTGOLD_3COL_DONE", flush=True)


if __name__ == "__main__":
    main()
