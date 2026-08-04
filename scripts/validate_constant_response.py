"""GOLD validation: does the trained measurement flow PREDICT the constant-shear response
that the simulation actually produces, and what MULTIPLICATIVE (m) and ADDITIVE (c) bias
remains after we calibrate with the flow?

The two-sided (constant_two_sided) sim renders the SAME galaxies + SAME noise at +g and -g
along a FIXED axis.  Split the two measurements into difference and sum:
  DIFFERENCE (e(+g)-e(-g))/2 = R.g      -> intrinsic shape cancels -> the RESPONSE (m)
  SUM        (e(+g)+e(-g))/2 = e_int+c  -> the shear cancels        -> the ADDITIVE bias (c)

Response we compare to the flow's induced response on the SAME galaxies (condition on their
true properties, shear the intrinsic shape by the analytic map S_{+-g}, read predicted mean):
  R_sim   = <(e(+g)-e(-g)).ghat>/(2g)            R_model = <(mu(S_{+g})-mu(S_{-g})).ghat>/(2g)

After calibrating the estimator as ghat = e / R_model:
  m = R_sim / R_model - 1     residual MULTIPLICATIVE bias (m>0 = we over-estimate shear
                              because the flow under-predicted R).  ~0 = flow is accurate.
  c = <(e(+g)+e(-g))/2> / R_model   ADDITIVE bias (shear units), split parallel/cross to the
                              shear axis.  PSF is ROUND here, so c ~ 0 is the expected/good
                              result; nonzero c flags an additive (e.g. PSF-leakage) systematic.

Convention: the sim's measured_e1/e2 are ngmix G (reduced-shear/epsilon convention) -- the
SAME quantity the flow now predicts (measured_ngmix_g1/g2), so NO conversion is applied.  We
report m and c globally and per (true flux x size x blend) cell.  m ~ 0 and c ~ 0 = strong
evidence the model is accurate.
"""
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

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from scripts.response_ratio_diagnostic import model_mean_proj  # noqa: E402
# NOTE: this `BASE` is the CONSTANT (constgold) tree, not the half-shear `SIM_BASE`.
from sbs_shear.paths import CONST_SIM_BASE as BASE



def load(cat, max_rows):
    need = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "neighbored", "distance",
            "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
            "position_angle_input_p", "position_angle_input_s", "r_input_p", "r_input_s",
            "redshift_input_p", "redshift_input_s", "sersic_n_input_p", "sersic_n_input_s",
            "polarization_angle"]
    with ipc.open_file(cat) as r:
        avail = set(r.schema.names); cols = [c for c in need if c in avail]
        parts = []; n = 0
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            parts.append(b); n += len(b)
            if n >= max_rows:
                break
    df = pd.concat(parts, ignore_index=True)
    # add the columns the flow expects but the constant catalogue stores parameterically
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i; df["e2_input_rot0_p"] = e2i      # intrinsic shape for S_delta
    df["gamma1_input_p"] = 0.0; df["gamma2_input_p"] = 0.0        # analytic map carries the shear
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue", default=BASE + "constant_response_catalogue_train.feather")
    ap.add_argument("--n-dist", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=8_000_000)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pixel-rms", type=float, default=0.312)
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--zero-mag", type=float, default=30.0)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--moffat-beta", type=float, default=2.224)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.measurement_model, device=device)
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)

    df = load(args.catalogue, args.max_rows)
    g = float(np.median(np.hypot(df["applied_g1"], df["applied_g2"])))
    gh1 = (df["applied_g1"].to_numpy(float) / np.hypot(df["applied_g1"], df["applied_g2"]))
    gh2 = (df["applied_g2"].to_numpy(float) / np.hypot(df["applied_g1"], df["applied_g2"]))

    # SIM antithetic split: difference -> response (m); sum -> additive (c).
    # measured_e1/2 are ngmix G (reduced-shear/epsilon convention) -- SAME convention the
    # flow now predicts (measured_ngmix_g1/g2), so NO chi->eps conversion (that was the bug).
    e1p, e2p = df["measured_e1_plus"].to_numpy(float), df["measured_e2_plus"].to_numpy(float)
    e1m, e2m = df["measured_e1_minus"].to_numpy(float), df["measured_e2_minus"].to_numpy(float)
    gp1, gp2 = -gh2, gh1                                          # cross (perpendicular) direction
    r_sim_i = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / (2 * g)   # per-object response (-> m)
    cpar_i = ((e1p + e1m) * gh1 + (e2p + e2m) * gh2) / 2.0        # additive shape, parallel to shear
    ccrs_i = ((e1p + e1m) * gp1 + (e2p + e2m) * gp2) / 2.0        # additive shape, cross to shear

    # FLOW induced response: shear the intrinsic shape by +-g, difference the predicted mean
    intr = (df["e1_input_rot0_p"].to_numpy(float).copy(), df["e2_input_rot0_p"].to_numpy(float).copy())
    mp, _ = model_mean_proj(bundle, df, +g, gh1, gh2, intr, rk, args.n_samples, args.batch_size)
    mm, _ = model_mean_proj(bundle, df, -g, gh1, gh2, intr, rk, args.n_samples, args.batch_size)
    R_model_global = (mp - mm) / (2 * g)
    R_sim_global = float(np.mean(r_sim_i))
    m_global = R_sim_global / R_model_global - 1.0               # residual mult. bias after flow calib
    cpar = float(np.mean(cpar_i)) / R_model_global               # additive bias (shear units), parallel
    ccrs = float(np.mean(ccrs_i)) / R_model_global               # additive bias (shear units), cross
    print(f"model={os.path.basename(args.measurement_model)}  |g|={g:.4f}  N={len(df):,}  blended_frac={df['neighbored'].astype(bool).mean():.3f}")
    print(f"\nGLOBAL (epsilon conv):  R_sim={R_sim_global:.4f}  R_model={R_model_global:.4f}")
    print(f"  m = R_sim/R_model - 1 = {m_global:+.2%}   (residual multiplicative bias after flow calibration)")
    print(f"  c_parallel = {cpar:+.5f}   c_cross = {ccrs:+.5f}   (additive bias, shear units; ~0 expected, round PSF)")

    # per blend cell (isolated + distance tertiles): re-run the flow response within each cell.
    nbf = df["neighbored"].astype(bool).to_numpy(); dist = df["distance"].to_numpy(float)
    db = dist[nbf & np.isfinite(dist)]; ed = np.quantile(db, np.linspace(0, 1, args.n_dist + 1)); ed[0] -= 1e-6; ed[-1] += 1e-6
    di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, args.n_dist - 1), 0)
    print(f"\nby blend cell (epsilon conv):")
    print(f"{'blend':>10} {'R_sim':>8} {'R_model':>8} {'m':>8} {'c_par':>9} {'c_crs':>9} {'N':>9}")
    nbl = args.n_dist + 1
    for c in range(nbl):
        m = di == c
        if m.sum() < 5000:
            continue
        sub = df[m].reset_index(drop=True)
        intrc = (intr[0][m], intr[1][m])
        mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
        mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
        Rm = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m]))
        mcell = Rs / Rm - 1.0
        cp = float(np.mean(cpar_i[m])) / Rm; cc = float(np.mean(ccrs_i[m])) / Rm
        tag = "ISOLATED" if c == 0 else f"blend d{c}"
        print(f"{tag:>10} {Rs:>8.4f} {Rm:>8.4f} {mcell:>+8.2%} {cp:>+9.5f} {cc:>+9.5f} {int(m.sum()):>9,}")
    print("\nm ~ 0 and c ~ 0 => the flow predicts the gold response accurately (sub-percent m).")


if __name__ == "__main__":
    main()
