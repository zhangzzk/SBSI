"""Quick sim first-moment shear response R_sim = <e_meas . ghat>/g for one catalogue.

Standalone, model-free: validates a built catalogue and reports the measured
responsivity. Used to sanity-check the g=0.02 render and, on the full catalogue,
to compute the held-out first-moment m vs a calibration shear.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import add_measurement_target_features  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    source_select_selection,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--calib-R", type=float, default=None,
                    help="responsivity calibrated at another shear; if set, print the "
                         "held-out first-moment m = R_sim(this g)/calib_R - 1")
    ap.add_argument("--calib-R-sem", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_feather(args.catalogue)
    n_raw = len(df)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    df = df[df["detected"].astype(bool)].reset_index(drop=True)
    if args.max_rows and len(df) > args.max_rows:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    g1 = df["gamma1_input_p"].to_numpy(float)
    g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    keep = gmag > 1e-6
    df, g1, g2, gmag = df[keep].reset_index(drop=True), g1[keep], g2[keep], gmag[keep]
    ghat1, ghat2 = g1 / gmag, g2 / gmag

    meas = add_measurement_target_features(df.copy())
    e1m = meas["measured_e1_image"].to_numpy(float)
    e2m = meas["measured_e2_image"].to_numpy(float)
    proj = e1m * ghat1 + e2m * ghat2
    m_sim = float(np.mean(proj))
    sem = float(np.std(proj) / np.sqrt(len(proj)))
    g = args.nominal_g
    print(f"catalogue : {os.path.basename(args.catalogue)}")
    print(f"rows      : raw={n_raw:,}  detected+selected+sheared={len(proj):,}")
    print(f"median |g|: {np.median(gmag):.4f}  (nominal {g})")
    R = m_sim / g
    R_sem = sem / g
    print(f"m_sim     : {m_sim:+.5f} +/- {sem:.5f}")
    print(f"R_sim     : {R:+.4f} +/- {R_sem:.4f}")
    if args.calib_R is not None:
        m_held = R / args.calib_R - 1.0
        err = abs(R / args.calib_R) * np.hypot(
            R_sem / R, (args.calib_R_sem / args.calib_R) if args.calib_R else 0.0
        )
        print(f"\n--- held-out FIRST-MOMENT m (calibrate R at another shear, predict g={g}) ---")
        print(f"  calib_R   : {args.calib_R:.4f} +/- {args.calib_R_sem:.4f}")
        print(f"  HELD-OUT m: {m_held:+.4f} +/- {err:.4f}   "
              f"({'SUB-PERCENT' if abs(m_held) < 0.01 else 'above 1%'})")


if __name__ == "__main__":
    main()
