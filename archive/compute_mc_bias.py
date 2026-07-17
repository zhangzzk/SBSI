"""Fit the shear calibration bias g_hat = (1+m) g + c from recovery scans.

Multiplicative m: slope of recovered s_hat vs true |g| across the available shears.
Additive c1, c2: recovered shear at g=0 along fixed spin-2 axes (fiducial angle 0 / 45 deg).
Errors on each peak come from a Monte-Carlo over the per-point SEM of the likelihood curve.
Compares to the LSST-era (Stage-IV/"Stage-VI") requirement.
"""

from __future__ import annotations

import os
import sys

import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "heldout_shear_recovery")

# LSST Y10 / Stage-IV-era shear-calibration requirement (benchmark).
REQ_M = 3.0e-3
REQ_C = 1.0e-3


def peak_with_err(tag, half_window=0.025):
    """Peak of a quadratic fit, error from the weighted-polyfit coefficient covariance.

    Stable alternative to MC: fit y=a s^2 + b s + c over a FIXED +/- half_window in s
    around the argmax, weighting by 1/SEM^2; propagate cov(a,b) to s_hat = -b/(2a)."""
    p = os.path.join(D, f"recovery_{tag}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    grid, mean, sem = d["grid"], d["mean_logprob"], d["sem"]
    n = int(d["n_rows"]); nominal = float(d["nominal_shear"])
    im = int(np.argmax(mean))
    win = np.abs(grid - grid[im]) <= half_window
    g, y, s = grid[win], mean[win], np.maximum(sem[win], 1e-9)
    coef, cov = np.polyfit(g, y, 2, w=1.0 / s, cov=True)
    a, b = coef[0], coef[1]
    sh = -b / (2 * a)
    # d s_hat/d a = b/(2 a^2), d s_hat/d b = -1/(2 a)
    da, db = b / (2 * a**2), -1.0 / (2 * a)
    var = da**2 * cov[0, 0] + db**2 * cov[1, 1] + 2 * da * db * cov[0, 1]
    return dict(s_hat=float(sh), err=float(np.sqrt(max(var, 0.0))), n=n, nominal=nominal)


def m_for_suffix(suffix=""):
    """m and its error from the two-shear slope for a given cut suffix."""
    r1 = peak_with_err(f"g0.05_sheared{suffix}")
    r2 = peak_with_err(f"g0.2_sheared{suffix}")
    if r1 is None or r2 is None:
        return None
    slope = (r2["s_hat"] - r1["s_hat"]) / (r2["nominal"] - r1["nominal"])
    m = slope - 1.0
    m_err = np.hypot(r1["err"], r2["err"]) / abs(r2["nominal"] - r1["nominal"])
    return dict(m=m, m_err=m_err, s1=r1["s_hat"], s2=r2["s_hat"])


def main():
    # --- m as a function of quality cut ---
    print("=== m vs quality cut ===")
    for label, suf in [("no cut", ""), ("SNR>20", "_snr20"), ("SNR>40", "_snr40"), ("mag<23.5", "_mag23.5")]:
        r = m_for_suffix(suf)
        if r:
            print(f"  {label:10s}: m = {r['m']:+.4f} +/- {r['m_err']:.4f}  "
                  f"(s_hat: {r['s1']:.4f}, {r['s2']:.4f})  {'PASS' if abs(r['m'])<REQ_M else 'FAIL'}")
    print()

    # --- multiplicative: s_hat vs g ---
    pts = []
    for tag in ["g0.05_sheared", "g0.2_sheared"]:
        r = peak_with_err(tag)
        if r:
            pts.append(r)
            print(f"  recovery {tag}: s_hat={r['s_hat']:.5f} +/- {r['err']:.5f}  (g={r['nominal']}, n={r['n']:,})")
    if len(pts) >= 2:
        g1, s1, e1 = pts[0]["nominal"], pts[0]["s_hat"], pts[0]["err"]
        g2, s2, e2 = pts[1]["nominal"], pts[1]["s_hat"], pts[1]["err"]
        slope = (s2 - s1) / (g2 - g1)
        m = slope - 1.0
        m_err = np.hypot(e1, e2) / abs(g2 - g1)
        c_proj = s1 - slope * g1
        c_proj_err = np.hypot(e1 * (1 - g1 / (g1 - g2)), e2 * (g1 / (g1 - g2)))
        print(f"\n=== MULTIPLICATIVE BIAS ===")
        print(f"  m = {m:+.4f} +/- {m_err:.4f}   (slope {slope:.4f})")
        print(f"  c_proj (additive along applied dir) = {c_proj:+.5f} +/- {c_proj_err:.5f}")
    else:
        m = m_err = None

    # --- additive: g=0 along fixed axes ---
    print(f"\n=== ADDITIVE BIAS (g=0 recovery along fixed axes) ===")
    cvals = {}
    for comp, tag in [("c1", "g0.05_unsheared_ang0"), ("c2", "g0.05_unsheared_ang45")]:
        r = peak_with_err(tag)
        if r:
            cvals[comp] = (r["s_hat"], r["err"])
            print(f"  {comp} = {r['s_hat']:+.5f} +/- {r['err']:.5f}  (n={r['n']:,})")

    # --- verdict ---
    print(f"\n=== vs LSST-era requirement (|m|<{REQ_M:.0e}, |c|<{REQ_C:.0e}) ===")
    if m is not None:
        ok_m = abs(m) < REQ_M
        print(f"  |m|={abs(m):.4f}  -> {'PASS' if ok_m else 'FAIL'} (need <{REQ_M:.0e}; off by {abs(m)/REQ_M:.0f}x)")
    for comp, (v, e) in cvals.items():
        ok_c = abs(v) < REQ_C
        print(f"  |{comp}|={abs(v):.5f}  -> {'PASS' if ok_c else 'FAIL'} (need <{REQ_C:.0e})")
    print("\nNote: this is the RAW (uncalibrated) bias of the framework estimator. The constant")
    print("multiplicative m is calibratable from the sim response catalogues; quality cuts and a")
    print("responsivity correction are the path to the requirement.")


if __name__ == "__main__":
    main()
