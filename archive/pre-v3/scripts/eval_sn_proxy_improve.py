"""Can a BETTER S/N proxy close the 0.25-0.52 pt selection-bias gap? And what is the ceiling?

CONTEXT. `eval_sn_proxy_check.py` (job 15364971) fitted log10 S/N = a*mag + b*log10(R) + c, got
R^2 = 0.9944, and yet cutting on that proxy OVERSTATED the real-S/N selection bias by 0.25-0.52 pts
at every keep-fraction. Owner asked whether the proxy can be improved.

WHY IT SHOULD BE IMPROVABLE -- the missing piece is identifiable, not mysterious.
`mag_auto` IS `flux_auto` up to a zero point, so the proxy already has the S/N NUMERATOR exactly.
The whole approximation sits in the denominator: it models `fluxerr_auto` as a power law in
`flux_radius`. But SExtractor's AUTO error comes from the KRON ELLIPTICAL aperture, whose area goes
as a*b. Shear changes ellipticity -> changes that aperture -> changes fluxerr -> changes S/N. A
CIRCULARIZED flux_radius cannot represent that, and it is exactly a shear-correlated term -- i.e.
precisely the part selection bias depends on. So the failure mode and the fix are the same object.

CRUCIALLY, THE FIX IS FLOW-REACHABLE: the flow outputs measured g1,g2, so |e| can enter the proxy
without any retrain. Models M2/M3 below use ONLY quantities the flow emits.

MODELS (all least-squares on log10 real S/N; "flow-reachable" means buildable from flow outputs):
  M1  a*mag + b*log10R + c                       flow-reachable   (the current proxy)
  M2  M1 + d*|e|^2                               flow-reachable   (aperture ellipticity term)
  M3  M2 + quadratic mag + mag*log10R            flow-reachable   (noise-regime curvature)
  M4  M1 + log10(a_image*b_image) terms          ORACLE, NOT flow-reachable -- it uses SExtractor's
      own aperture axes. Its purpose is to BOUND what any proxy could achieve, so we can tell
      "the proxy is improvable" from "this information is simply not in the flow's outputs".

THE METRIC THAT MATTERS IS NOT R^2. A proxy can have R^2 = 0.99 and still miss the bias, because
selection bias depends only on the SHEAR-CORRELATED part of the cut variable. So for every model we
report BOTH R^2 AND the m_sel gap against the real-S/N cut at matched keep-fractions. We also report
the shear-correlated part of the residual directly: cov(residual_g - residual_0, e.ghat), which is
the quantity that should shrink if the fix is real.

DETECTION SEPARATED: both-detected pairs only. FIREWALL: half-shear legs, truth-only, no model. CPU.
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

EXTRA = ["measured_flux_auto", "measured_fluxerr_auto", "measured_a_image", "measured_b_image"]


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
    ap.add_argument("--keep-fracs", type=float, nargs="+", default=[0.95, 0.85, 0.70, 0.50, 0.30])
    args = ap.parse_args()

    ESR.MEAS = list(dict.fromkeys(ESR.MEAS + EXTRA))
    t0 = time.time()
    ru = ESR.build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                        args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]
    C = lambda n, leg: base[f"{n}_{leg}"].to_numpy(float)

    D = {}
    for leg in ("0", "g"):
        f, fe = C("measured_flux_auto", leg), C("measured_fluxerr_auto", leg)
        with np.errstate(divide="ignore", invalid="ignore"):
            D[("sn", leg)] = np.where(fe > 0, f / fe, np.nan)
        D[("mag", leg)] = C("measured_mag_auto", leg)
        D[("lr", leg)] = np.log10(np.maximum(C("measured_flux_radius", leg), 1e-6))
        e1, e2 = C("measured_ngmix_g1", leg), C("measured_ngmix_g2", leg)
        D[("e2", leg)] = np.clip(e1 ** 2 + e2 ** 2, 0.0, 0.998)
        a_, b_ = C("measured_a_image", leg), C("measured_b_image", leg)
        D[("lab", leg)] = np.log10(np.maximum(a_ * b_, 1e-6))

    fin = iso.copy()
    for k in D:
        fin &= np.isfinite(D[k])
    fin &= (D[("sn", "0")] > 0) & (D[("sn", "g")] > 0)
    print(f"  usable: {int(fin.sum()):,}", flush=True)

    i1 = base["e1_input_rot0_p"].to_numpy(float)
    i2 = base["e2_input_rot0_p"].to_numpy(float)
    s1, s2 = apply_shear_to_ellipticity(i1, i2, gmed * gh1, gmed * gh2)
    p_int, p_shr = i1 * gh1 + i2 * gh2, s1 * gh1 + s2 * gh2
    fin &= np.isfinite(p_int) & np.isfinite(p_shr)
    R_nc = two_means(p_int, p_shr, gmed, fin, fin)

    def design(leg, model):
        m, lr, e2, lab = D[("mag", leg)], D[("lr", leg)], D[("e2", leg)], D[("lab", leg)]
        one = np.ones_like(m)
        if model == "M1":
            return np.column_stack([m, lr, one])
        if model == "M2":
            return np.column_stack([m, lr, e2, one])
        if model == "M3":
            return np.column_stack([m, lr, e2, m * m, m * lr, one])
        if model == "M4":
            return np.column_stack([m, lr, lab, one])
        raise ValueError(model)

    models = [("M1  mag+logR            (flow)", "M1"),
              ("M2  + |e|^2             (flow)", "M2"),
              ("M3  + mag^2, mag*logR   (flow)", "M3"),
              ("M4  + log(a*b)        (ORACLE)", "M4")]

    y = np.log10(D[("sn", "0")][fin])
    print("\n" + "=" * 104)
    print("FIT QUALITY and SHEAR-CORRELATED RESIDUAL  (the second is what selection bias cares about)")
    print("=" * 104)
    print(f"  {'model':>32} {'R^2':>9} {'resid dex':>10} {'cov(dresid, e.ghat)':>21}")
    fits = {}
    for lab, key in models:
        A0 = design("0", key)[fin]
        coef, *_ = np.linalg.lstsq(A0, y, rcond=None)
        pred0 = A0 @ coef
        r2 = 1.0 - float(np.sum((y - pred0) ** 2)) / float(np.sum((y - y.mean()) ** 2))
        # residual in BOTH legs -> its shear-correlated part is the thing that biases the cut
        rg = np.log10(D[("sn", "g")][fin]) - design("g", key)[fin] @ coef
        r0 = y - pred0
        dres = rg - r0
        cov = float(np.mean(dres * p_int[fin]) - np.mean(dres) * np.mean(p_int[fin]))
        fits[key] = coef
        print(f"  {lab:>32} {r2:>9.5f} {np.std(r0):>10.4f} {cov:>+21.3e}")

    print("\n" + "=" * 104)
    print("m_sel: REAL S/N cut vs each PROXY, at MATCHED keep-fraction (exact intrinsic shapes)")
    print("=" * 104)
    hdr = f"  {'keep':>6} {'real':>9} |"
    for lab, _ in models:
        hdr += f" {lab.split()[0]:>8} {'gap':>8} |"
    print(hdr)
    for kf in args.keep_fracs:
        v0, vg = D[("sn", "0")], D[("sn", "g")]
        thr = float(np.quantile(v0[fin], 1.0 - kf))
        Rr = two_means(p_int, p_shr, gmed, fin & (v0 > thr), fin & (vg > thr))
        m_real = (Rr / R_nc - 1.0) * 100.0
        line = f"  {kf:>6.2f} {m_real:>+8.3f}% |"
        for lab, key in models:
            u0 = design("0", key) @ fits[key]
            ug = design("g", key) @ fits[key]
            th = float(np.quantile(u0[fin], 1.0 - kf))
            Rp = two_means(p_int, p_shr, gmed, fin & (u0 > th), fin & (ug > th))
            m_p = (Rp / R_nc - 1.0) * 100.0
            line += f" {m_p:>+7.3f}% {m_p-m_real:>+7.3f} |"
        print(line)
    print("\n  Judge on the GAP column, not R^2. M4 is an ORACLE (uses SExtractor's own aperture axes,")
    print("  which the flow does not output): if M2/M3 approach M4, the proxy route is viable; if only")
    print("  M4 closes the gap, the missing information is genuinely outside the flow's outputs and")
    print("  predicting a real S/N cut needs flux_auto/fluxerr_auto as extra flow dimensions.")
    print("SN_PROXY_IMPROVE_DONE", flush=True)


if __name__ == "__main__":
    main()
