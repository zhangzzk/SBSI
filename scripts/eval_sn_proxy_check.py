"""Can measured mag + size stand in for S/N? Gate before asking the flow to predict an S/N cut.

THE PROPOSAL (owner). The flow outputs measured shape, magnitude and size. If S/N is an analytic
function of measured magnitude and size, the flow can predict the S/N-cut selection probability for
every true galaxy without ever being trained on S/N.

WHY THIS NEEDS CHECKING FIRST. The flow does NOT output `measured_flux_auto` or
`measured_fluxerr_auto`, so it cannot produce SExtractor's S/N directly -- only a PROXY built from
mag and size. If that proxy tracks the real S/N poorly, then any sim-vs-model agreement is about a
FICTITIOUS cut and says nothing about selection bias. So this script asks two questions, in order:

  Q1  HOW WELL IS S/N DETERMINED BY MAG AND SIZE?  Regress log10(real S/N) on
      (mag_auto, log measured_flux_radius) and report R^2 and the fitted exponents. The sky-limited
      expectation is log10 S/N = -0.4 mag - 1.0 log10 R + const; deviations say which noise regime
      dominates. A LOW R^2 here kills the proposal outright -- mag and size would simply not carry
      the information.

  Q2  DOES CUTTING ON THE PROXY REPRODUCE THE SELECTION BIAS OF CUTTING ON THE REAL S/N?
      This is the question that actually matters, and Q1 does not answer it: selection bias depends
      on the SHEAR RESPONSE of the cut variable, not on how well it is predicted on average. A proxy
      can have R^2 = 0.99 and still miss the bias if the missing 1% is the shear-correlated part.
      Both cuts are placed at MATCHED KEEP-FRACTIONS so the comparison is like-for-like.

  We report m_sel with EXACT intrinsic shapes (pure selection, no shape response) for both.

DETECTION IS SEPARATED BY CONSTRUCTION, as requested: `build_base` keeps only objects detected in
BOTH legs, so nothing here mixes in detection bias.

FIREWALL: half-shear legs, truth-only. No model, no emulator, no constgold. CPU.
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

import eval_selection_response as ESR  # noqa: E402
from eval_selection_response import CAT, CROWD, NN, apply_shear_to_ellipticity  # noqa: E402

EXTRA = ["measured_flux_auto", "measured_fluxerr_auto"]


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
    ap.add_argument("--keep-fracs", type=float, nargs="+",
                    default=[0.95, 0.85, 0.70, 0.50, 0.30, 0.20])
    args = ap.parse_args()

    # The harness loads a fixed MEAS list; add the flux columns needed for the REAL S/N.
    ESR.MEAS = list(dict.fromkeys(ESR.MEAS + EXTRA))
    print(f"MEAS columns: {ESR.MEAS}", flush=True)

    t0 = time.time()
    ru = ESR.build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                        args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]

    def col(name, leg):
        return base[f"{name}_{leg}"].to_numpy(float)

    out = {}
    for leg in ("0", "g"):
        f, fe = col("measured_flux_auto", leg), col("measured_fluxerr_auto", leg)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[("real", leg)] = np.where(fe > 0, f / fe, np.nan)
        out[("mag", leg)] = col("measured_mag_auto", leg)
        out[("rad", leg)] = col("measured_flux_radius", leg)

    fin = iso.copy()
    for k in out:
        fin &= np.isfinite(out[k])
    fin &= (out[("rad", "0")] > 0) & (out[("rad", "g")] > 0) & (out[("real", "0")] > 0) \
        & (out[("real", "g")] > 0)
    n = int(fin.sum())
    print(f"  usable (isolated, both legs finite, positive): {n:,}", flush=True)

    # ---------------- Q1: is S/N determined by mag and size? ---------------------------------
    y = np.log10(out[("real", "0")][fin])
    x1 = out[("mag", "0")][fin]
    x2 = np.log10(out[("rad", "0")][fin])
    A = np.column_stack([x1, x2, np.ones_like(x1)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    print("\n" + "=" * 96)
    print("Q1  log10(S/N) = a*mag + b*log10(R) + c     [sky-limited expectation: a=-0.4, b=-1.0]")
    print("=" * 96)
    print(f"  fitted   a={coef[0]:+.4f}   b={coef[1]:+.4f}   c={coef[2]:+.4f}")
    print(f"  R^2 = {r2:.5f}   residual scatter = {np.std(y - pred):.4f} dex")
    print(f"  corr(proxy, real) in log10 = {np.corrcoef(pred, y)[0,1]:.5f}")

    # proxy per leg, using the SAME fitted coefficients on each leg's own mag/size
    for leg in ("0", "g"):
        out[("proxy", leg)] = 10.0 ** (coef[0] * out[("mag", leg)]
                                       + coef[1] * np.log10(np.maximum(out[("rad", leg)], 1e-6))
                                       + coef[2])

    # ---------------- Q2: same selection bias at matched keep-fraction? -----------------------
    i1 = base["e1_input_rot0_p"].to_numpy(float)
    i2 = base["e2_input_rot0_p"].to_numpy(float)
    s1, s2 = apply_shear_to_ellipticity(i1, i2, gmed * gh1, gmed * gh2)
    p_int = i1 * gh1 + i2 * gh2
    p_shr = s1 * gh1 + s2 * gh2
    fin &= np.isfinite(p_int) & np.isfinite(p_shr)
    R_nocut = two_means(p_int, p_shr, gmed, fin, fin)
    print(f"\n  R(no cut, sheared intrinsic) = {R_nocut:+.5f}")

    print("\n" + "=" * 96)
    print("Q2  m_sel from cutting on REAL S/N vs on the mag+size PROXY, at MATCHED keep-fraction")
    print("    (exact intrinsic shapes -> pure selection; detection excluded: both-detected only)")
    print("=" * 96)
    print(f"  {'keep':>6} | {'thr real':>10} {'m_sel real':>12} | {'thr proxy':>10} "
          f"{'m_sel proxy':>12} | {'difference':>11}")
    for kf in args.keep_fracs:
        row = []
        for which in ("real", "proxy"):
            v0, vg = out[(which, "0")], out[(which, "g")]
            thr = float(np.quantile(v0[fin], 1.0 - kf))
            p0 = fin & (v0 > thr)
            pg = fin & (vg > thr)
            R = two_means(p_int, p_shr, gmed, p0, pg)
            row.append((thr, (R / R_nocut - 1.0) * 100.0))
        print(f"  {kf:>6.2f} | {row[0][0]:>10.2f} {row[0][1]:>+11.3f}% | {row[1][0]:>10.2f} "
              f"{row[1][1]:>+11.3f}% | {row[1][1]-row[0][1]:>+10.3f}")
    print("\n  If the two m_sel columns track each other, the flow's mag+size CAN stand in for an")
    print("  S/N cut and the prediction test is meaningful. If they diverge, the proxy misses the")
    print("  shear-correlated part of S/N and the flow would be predicting the wrong cut.")
    print("SN_PROXY_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
