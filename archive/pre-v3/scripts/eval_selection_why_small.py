"""Why is m_sel ~0 for magnitude cuts and for size cuts below the PSF? Is it physics or a dead estimator?

The owner flagged the near-zero m_sel as surprising. An estimator that returns ~0 is worth doubting:
it could be correct, or it could be structurally unable to see the effect. This measures the MECHANISM
rather than re-measuring the answer.

SELECTION BIAS REQUIRES A MOVING BOUNDARY. m_sel can only be non-zero if the set of objects passing
the cut CHANGES between the g=0 and g=gS legs, and if the objects that change are shape-biased. So:

  1. FLIP FRACTIONS. Count objects entering (pass under shear, not at g=0) and leaving. If these are
     ~0, the boundary does not move and m_sel MUST be ~0 -- that is physics, not a bug. If they are
     large but m_sel is still ~0, the flippers are shape-unbiased and something is wrong upstream.
  2. THE SHEAR RESPONSE OF THE CUT VARIABLE ITSELF. A cut can only move if its variable responds to
     shear. For a size cut the expectation is dT/T = 2 g.e * T_gal/(T_gal + T_psf): it VANISHES for
     galaxies far below the PSF, which is the predicted origin of the 0.53" turnover. For magnitude
     the expectation is ~0 because shear conserves flux (area x surface brightness).
  3. SHAPE BIAS OF THE FLIPPERS. <e.ghat> for objects entering vs leaving. This is the actual engine
     of selection bias; if it is zero the cut is shear-blind.

SCOPE -- READ BEFORE CONCLUDING "NO SELECTION BIAS". This sample is already restricted to
BOTH-DETECTED, ISOLATED pairs that pass TRUE-property cuts (Re>0.3", mag<26). So detection selection
and blend-driven selection are excluded BY CONSTRUCTION, and the smallest galaxies are already gone.
Whatever this reports is the measured-property selection that REMAINS on top of all that.

FIREWALL: half-shear legs only. No emulator, no constgold, no model. Reads catalogues; trains nothing.
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

from eval_selection_response import CAT, CROWD, NN, build_base  # noqa: E402

PSF_R50 = 0.5268
PX = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--size-cuts", type=float, nargs="+",
                    default=[1.0, 2.0, 2.5, 2.6, 2.9, 3.0, 3.5, 4.0, 4.5])
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[24.5, 25.0, 25.5, 26.0])
    args = ap.parse_args()

    t0 = time.time()
    ru = build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                    args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]

    i1 = base["e1_input_rot0_p"].to_numpy(float)
    i2 = base["e2_input_rot0_p"].to_numpy(float)
    p = i1 * gh1 + i2 * gh2                     # intrinsic shape projected on the shear direction
    sel = iso & np.isfinite(p)

    print("\n" + "=" * 96)
    print("SAMPLE SCOPE (everything below is measured ON TOP of these restrictions)")
    print("=" * 96)
    print(f"  matched both-detected pairs passing TRUE cuts : {len(base):,}")
    print(f"  of which ISOLATED (nn_bright > {args.iso_radius:g}\")     : {int(sel.sum()):,}"
          f"  ({sel.mean():.1%})")
    print("  -> detection selection and blend-driven selection are EXCLUDED BY CONSTRUCTION,")
    print(f"  -> and galaxies with true Re < {args.true_re_min}\" are already gone before we start.")

    # ---- 2. does the cut VARIABLE respond to shear at all? -----------------------------------
    print("\n" + "=" * 96)
    print("SHEAR RESPONSE OF THE CUT VARIABLE  (a cut cannot move if its variable does not)")
    print("=" * 96)
    for col, lab in (("measured_flux_radius", "size"), ("measured_mag_auto", "mag")):
        x0 = base[col + "_0"].to_numpy(float)
        xg = base[col + "_g"].to_numpy(float)
        ok = sel & np.isfinite(x0) & np.isfinite(xg)
        d = xg[ok] - x0[ok]
        # The engine of selection bias is the part of the response CORRELATED with shape.
        cov = float(np.mean(d * p[ok]) - np.mean(d) * np.mean(p[ok]))
        print(f"  {lab:>5}: <x_g - x_0> = {np.mean(d):+.5f}   sd = {np.std(d):.5f}   "
              f"cov(dx, e.ghat) = {cov:+.3e}")
    # size response vs size: the predicted PSF dilution T_gal/(T_gal+T_psf)
    x0 = base["measured_flux_radius_0"].to_numpy(float)
    xg = base["measured_flux_radius_g"].to_numpy(float)
    ok = sel & np.isfinite(x0) & np.isfinite(xg)
    a0 = x0 * PX
    print("\n  size response SPLIT BY measured size  (cov of dR with e.ghat; the selection engine):")
    edges = [0.0, 0.40, 0.50, 0.53, 0.60, 0.70, 0.80, 9.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (a0 >= lo) & (a0 < hi)
        if m.sum() < 500:
            continue
        d = (xg[m] - x0[m]) * PX
        cov = float(np.mean(d * p[m]) - np.mean(d) * np.mean(p[m]))
        print(f"    {lo:.2f}-{hi:.2f}\"  n={int(m.sum()):>9,}  cov(dR, e.ghat) = {cov:+.3e}"
              f"   {'  <-- below PSF' if hi <= PSF_R50 else ''}")

    # ---- 1 & 3. flip fractions and the shape bias of the flippers ----------------------------
    print("\n" + "=" * 96)
    print("MOVING BOUNDARY: who enters / leaves the selection when the shear is applied")
    print("=" * 96)
    print(f"  {'cut':>14} {'kept':>10} {'enter':>8} {'leave':>8} {'net/kept':>10} "
          f"{'<e.g> enter':>12} {'<e.g> leave':>12} {'<e.g> kept':>11}")
    rows = []
    for c in args.size_cuts:
        rows.append(("size", f'R>{c*PX:.2f}"', "measured_flux_radius", c, True))
    for c in args.mag_cuts:
        rows.append(("mag", f"mag<{c:g}", "measured_mag_auto", c, False))
    for kind, lab, col, thr, keep_high in rows:
        x0 = base[col + "_0"].to_numpy(float)
        xg = base[col + "_g"].to_numpy(float)
        p0 = sel & ((x0 > thr) if keep_high else (x0 < thr))
        pg = sel & ((xg > thr) if keep_high else (xg < thr))
        enter = pg & ~p0
        leave = p0 & ~pg
        kept = p0 & pg
        ne, nl, nk = int(enter.sum()), int(leave.sum()), int(kept.sum())
        f = lambda m, n: (float(np.mean(p[m])) if n else np.nan)
        print(f"  {lab:>14} {nk:>10,} {ne:>8,} {nl:>8,} "
              f"{(ne - nl) / max(nk, 1):>+10.2e} {f(enter, ne):>+12.4f} {f(leave, nl):>+12.4f} "
              f"{f(kept, nk):>+11.4f}")
    print("\n  If 'enter' and 'leave' are ~0, the boundary does not move and m_sel MUST be ~0 --")
    print("  that is physics. If they are large but <e.g> of enter and leave are EQUAL, the cut is")
    print("  shear-blind. Selection bias needs BOTH a moving boundary AND shape-biased movers.")
    print("WHY_SMALL_DONE", flush=True)


if __name__ == "__main__":
    main()
