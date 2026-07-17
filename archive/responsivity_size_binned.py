"""Test the covariate-shift fix: size-CONDITIONED vs MARGINAL shape responsivity.

The sheared population shifts shape independently of size (shear adds a uniform shape shift,
holding size fixed). So the response that governs <chi_par>_sheared is the size-conditioned
within-bin shape slope, averaged over the population:

    R_conditional = < dE[chi|shape,size]/dshape >_pop   (slope within a size bin)

The MARGINAL OLS slope (shape-only forward) instead is

    R_marginal    = dE[chi|shape]/dshape                (mixes in the g=0 shape<->size correlation)

If shape and size are correlated at g=0, R_marginal != R_conditional, and the shape-only forward
estimator is biased by exactly that confounding.  R_conditional should equal the data response
<chi_par>_sheared / g (to the nonlinearity floor) and give m -> 0.  This script computes both from
g=0 (+ the exact distortion map) and validates m on the held-out sheared catalogues.  No sheared
shear truth enters either responsivity.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import add_measurement_target_features  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

CD = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"
REQ_M = 3.0e-3
RAW = ["detected", "gamma1_input_p", "gamma2_input_p", "e1_input_rot0_p", "e2_input_rot0_p",
       "measured_a_image", "measured_b_image", "measured_theta_image",
       "measured_flux_auto", "measured_fluxerr_auto", "measured_mag_auto",
       "r_input_p", "Re_input_p", "distance", "neighbored"]


def eps_to_distortion(e1, e2):
    d = 1.0 + e1 * e1 + e2 * e2
    return 2.0 * e1 / d, 2.0 * e2 / d


def _stream(path, max_rows, snr_min, seed, max_batches=None):
    rng = np.random.default_rng(seed)
    reservoir, raw = None, 0
    with ipc.open_file(path) as reader:
        avail = set(reader.schema.names)
        cols = [c for c in RAW if c in avail]
        n = reader.num_record_batches
        order = (sorted(set(int(x) for x in np.linspace(0, n - 1, max_batches)))
                 if max_batches and max_batches < n else range(n))
        for bi in order:
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            if snr_min is not None:
                snr = b["measured_flux_auto"].to_numpy(float) / b["measured_fluxerr_auto"].to_numpy(float)
                b = b[np.isfinite(snr) & (snr > snr_min)].reset_index(drop=True)
            if len(b) == 0:
                continue
            b = add_measurement_target_features(b)
            keep = np.isfinite(b["measured_e1_image"]) & np.isfinite(b["measured_e2_image"])
            b = b[keep].reset_index(drop=True)
            if len(b) == 0:
                continue
            b["__k"] = rng.random(len(b))
            reservoir = b if reservoir is None else pd.concat([reservoir, b], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__k").reset_index(drop=True)
    reservoir = reservoir.nlargest(min(max_rows, len(reservoir)), "__k").reset_index(drop=True)
    return reservoir, raw


def slope_on_distortion(e1, e2, y1, y2):
    """Diagonal-mean OLS slope of measured chi on the scene distortion (no intercept bias)."""
    x1, x2 = eps_to_distortion(e1, e2)
    s1 = np.polyfit(x1, y1, 1)[0]
    s2 = np.polyfit(x2, y2, 1)[0]
    return 0.5 * (s1 + s2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g0", default=os.path.join(CD, "det_meas_g0.0_train.feather"))
    ap.add_argument("--sheared", nargs="+",
                    default=[f"{CD}/det_meas_g0.05_val.feather:0.05",
                             f"{CD}/det_meas_g0.2_val.feather:0.2"])
    ap.add_argument("--g0-rows", type=int, default=3_000_000)
    ap.add_argument("--sheared-rows", type=int, default=2_000_000)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--snr-min", type=float, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--shear-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    g0, g0raw = _stream(args.g0, args.g0_rows, args.snr_min, args.seed, args.max_batches)
    print(f"[g0] rows={len(g0):,} (raw {g0raw:,})")
    e1 = g0["e1_input_rot0_p"].to_numpy(float); e2 = g0["e2_input_rot0_p"].to_numpy(float)
    y1 = g0["measured_e1_image"].to_numpy(float); y2 = g0["measured_e2_image"].to_numpy(float)
    logRe = np.log10(g0["Re_input_p"].to_numpy(float))

    eabs = np.hypot(e1, e2)
    print(f"[g0] corr(|e_scene|, logRe) = {np.corrcoef(eabs, logRe)[0,1]:+.3f}")

    # rho_S: scene-distortion responsivity to applied shear, from the analytic Mobius (g=0).
    ds = 0.01
    p1, _ = eps_to_distortion(*apply_shear_to_ellipticity(e1, e2, ds, 0.0))
    m1, _ = eps_to_distortion(*apply_shear_to_ellipticity(e1, e2, -ds, 0.0))
    rho_S = (p1.mean() - m1.mean()) / (2 * ds)

    # MARGINAL slope (shape-only).
    M_marg = slope_on_distortion(e1, e2, y1, y2)

    # CONDITIONAL slope: within size-quantile bins, count-weighted average local slope.
    edges = np.quantile(logRe, np.linspace(0, 1, args.nbins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    binid = np.digitize(logRe, edges) - 1
    slopes, counts = [], []
    for b in range(args.nbins):
        m = binid == b
        if m.sum() < 2000:
            continue
        slopes.append(slope_on_distortion(e1[m], e2[m], y1[m], y2[m]))
        counts.append(int(m.sum()))
    slopes = np.array(slopes); counts = np.array(counts)
    M_cond = float(np.average(slopes, weights=counts))

    R_marg = M_marg * rho_S
    R_cond = M_cond * rho_S
    print(f"\nrho_S(distortion)={rho_S:.4f}")
    print(f"M_marginal (shape-only OLS slope)      = {M_marg:.4f}  -> R_marg = {R_marg:.4f}")
    print(f"M_conditional (<within-size-bin slope>) = {M_cond:.4f}  -> R_cond = {R_cond:.4f}")
    print(f"  per-bin slope range: {slopes.min():.3f}..{slopes.max():.3f} over {len(slopes)} bins")
    print(f"  confounding = R_marg/R_cond - 1 = {R_marg/R_cond-1:+.4f}")

    print(f"\n=== validate m on held-out shears (g_hat=<chi_par>/R) vs Stage-IV |m|<{REQ_M:.0e} ===")
    print(f"{'g':>5s} {'<chi_par>':>10s} {'m(marginal)':>12s} {'m(conditional)':>15s}")
    for spec in args.sheared:
        path, nominal = spec.rsplit(":", 1); nominal = float(nominal)
        sh, _ = _stream(path, args.sheared_rows, args.snr_min, args.seed + 1, args.max_batches)
        g1 = sh["gamma1_input_p"].to_numpy(float); g2 = sh["gamma2_input_p"].to_numpy(float)
        gm = np.hypot(g1, g2); msk = gm > args.shear_threshold
        gh1 = g1[msk] / gm[msk]; gh2 = g2[msk] / gm[msk]
        e1m = sh["measured_e1_image"].to_numpy(float)[msk]; e2m = sh["measured_e2_image"].to_numpy(float)[msk]
        chi_par = (e1m * gh1 + e2m * gh2).mean()
        m_marg = chi_par / R_marg / nominal - 1.0
        m_cond = chi_par / R_cond / nominal - 1.0
        print(f"{nominal:5.2f} {chi_par:10.5f} {m_marg:+12.4f} {m_cond:+15.4f}"
              f"   {'<-- PASS' if abs(m_cond)<REQ_M else ''}")
    print("\nIf m(conditional) << m(marginal), the g=0 shape<->size correlation (covariate shift)")
    print("was the multiplicative-bias source, and the size-conditioned responsivity is the fix.")


if __name__ == "__main__":
    main()
