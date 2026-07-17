"""Is the cut-dependence of m a SELECTION effect (from cutting on a noisy, shear-affected
variable) or a real response-calibration trend?  MODEL-FREE test, simulation only.

For a sequence of cuts that keep the brightest fraction f of the detected sample, cutting
either on
  * MEASURED SNR  (measured_flux_auto/measured_fluxerr_auto)  -- noisy & shear-affected, so
    the cut boundary itself is a shear-dependent selection; OR
  * TRUE magnitude (r_input_p)                                -- shear-INDEPENDENT intrinsic
    property, so the cut injects no selection bias,
we measure the sim's own first-moment shear response and the held-out multiplicative bias:

  R_sim(g, cut) = < e_meas . ghat / |g| >_kept           (response of the kept sample at shear g)
  m_sim(cut)    = R_sim(0.02, cut) / R_sim(0.05, cut) - 1 (calibrate response @0.05, predict @0.02)

m_sim is LEGITIMATE (response calibrated at one shear, tested at an independent held-out
shear; no test-set m removed).  Reading:
  * if m_sim is ~flat vs the TRUE-magnitude cut but trends vs the MEASURED-SNR cut
    -> the trend is a shear-dependent SELECTION effect from cutting on the noisy measured
       variable, NOT a property-dependent calibration error.
  * if m_sim trends for BOTH -> a genuine response transfer/calibration trend with brightness.
"""
import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    add_measurement_target_features,
    raw_columns_for_measurement_targets,
)
from sbs_shear.sim_stream import stream_reservoir  # noqa: E402


def load(cat, max_rows, seed):
    """Detected, source-selected sheared rows; return (proj, snr, true_mag, gmag)."""
    need = set(raw_columns_for_measurement_targets(["measured_e1_image", "measured_e2_image"]))
    need |= {"detected", "gamma1_input_p", "gamma2_input_p",
             "measured_flux_auto", "measured_fluxerr_auto",
             "r_input_p", "Re_input_p", "distance", "neighbored"}
    res = stream_reservoir(cat, need, max_rows, seed=seed, detected=True, shear_threshold=1e-6)

    meas = add_measurement_target_features(res.copy())
    g1 = res["gamma1_input_p"].to_numpy(float)
    g2 = res["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    gh1, gh2 = g1 / gmag, g2 / gmag
    proj = (meas["measured_e1_image"].to_numpy(float) * gh1
            + meas["measured_e2_image"].to_numpy(float) * gh2)
    snr = res["measured_flux_auto"].to_numpy(float) / res["measured_fluxerr_auto"].to_numpy(float)
    true_mag = res["r_input_p"].to_numpy(float)
    fin = np.isfinite(proj) & np.isfinite(snr) & np.isfinite(true_mag) & np.isfinite(gmag) & (gmag > 0)
    print(f"  {os.path.basename(cat)}: kept={fin.sum():,}  <|g|>={np.mean(gmag[fin]):.4f}")
    return proj[fin], snr[fin], true_mag[fin], gmag[fin]


def response(proj, gmag, mask):
    r = proj[mask] / gmag[mask]
    n = int(mask.sum())
    return float(np.mean(r)), float(np.std(r) / max(np.sqrt(n), 1)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-005", required=True)
    ap.add_argument("--cat-002", required=True)
    ap.add_argument("--max-rows", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print("Loading g=0.05 ..."); p5, snr5, tm5, gm5 = load(args.cat_005, args.max_rows, args.seed)
    print("Loading g=0.02 ..."); p2, snr2, tm2, gm2 = load(args.cat_002, args.max_rows, args.seed)

    fracs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    for var in ("measured_snr", "true_mag"):
        print(f"\n================ CUT VARIABLE: {var} ================")
        print(f"  (keep the brightest fraction f; "
              f"{'higher SNR' if var=='measured_snr' else 'lower r_input_p (= brighter)'} kept)")
        print(f"{'keep_f':>7} {'thr05':>9} {'R_sim@0.05':>11} {'R_sim@0.02':>11} "
              f"{'m_sim(%)':>9} {'+/-(%)':>7} {'N05':>10}")
        for f in fracs:
            if var == "measured_snr":
                thr5 = np.quantile(snr5, 1 - f); k5 = snr5 >= thr5
                thr2 = np.quantile(snr2, 1 - f); k2 = snr2 >= thr2
            else:  # true_mag: brighter = smaller r_input_p
                thr5 = np.quantile(tm5, f); k5 = tm5 <= thr5
                thr2 = np.quantile(tm2, f); k2 = tm2 <= thr2
            R5, e5, n5 = response(p5, gm5, k5)
            R2, e2, n2 = response(p2, gm2, k2)
            m = R2 / R5 - 1.0
            # error propagation on the ratio
            me = abs(m + 1) * np.sqrt((e5 / R5) ** 2 + (e2 / R2) ** 2)
            print(f"{f:>7.2f} {thr5:>9.3f} {R5:>11.4f} {R2:>11.4f} "
                  f"{m*100:>+9.2f} {me*100:>7.2f} {n5:>10,}")


if __name__ == "__main__":
    main()
