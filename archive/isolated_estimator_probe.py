"""Is the ISOLATED-galaxy -2.8% gold gap a MODEL issue or a SIM issue?

The flow is trained to emulate the SExtractor windowed-moment ellipticity
measured_e1_image = (a-b)/(a+b) . [cos2t, sin2t]  (from measured_a/b/theta_image).
The constant "gold" render instead stores measured_e1/e2 = NGMIX_G1/G2 (an ngmix
model-fit shear estimator) -- a DIFFERENT estimator with a different shear response.

This probe measures the ISOLATED (neighbored=False), detected, source-selected
first-moment shear response for each estimator, so we can attribute the -2.8%:

  det_meas SExtractor eps response, g=0.05 and g=0.02   (D5, D2)  <- flow's estimator
  const    NGMIX     response, raw and chi->eps          (Rraw, Reps)
  flow R_model (isolated, +-0.02)                        = 0.2392  (known, prints for ref)

Reading:
  if D2 ~ D5 ~ 0.239 (~ flow)  AND  Reps ~ 0.2325 (< D)
     -> the flow faithfully reproduces its SExtractor estimator; the gold gap is the
        SExtractor-vs-NGMIX estimator difference -> a SIM/measurement-definition issue.
  if D2 ~ 0.2325 (~ const)     -> the SExtractor response itself is 0.2325 and the flow
        (0.239) over-predicts it -> a MODEL issue.
"""
import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import add_measurement_target_features  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402

DET = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g{g}_val.feather"
CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather"


def chi_to_eps(e1, e2):
    chi = np.hypot(e1, e2)
    safe = chi > 1e-9
    eps_mag = np.where(safe, chi / (1.0 + np.sqrt(np.clip(1.0 - chi**2, 0, 1))), 0.0)
    scale = np.where(safe, eps_mag / np.where(safe, chi, 1.0), 0.0)
    return e1 * scale, e2 * scale


def read(path, cols, max_rows):
    with ipc.open_file(path) as r:
        avail = set(r.schema.names); use = [c for c in cols if c in avail]
        parts = []; n = 0
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            parts.append(b); n += len(b)
            if n >= max_rows:
                break
    return pd.concat(parts, ignore_index=True)


def det_isolated_response(g, max_rows):
    """SExtractor eps response of isolated detected source-selected galaxies at shear g."""
    cols = ["measured_a_image", "measured_b_image", "measured_theta_image",
            "measured_flux_auto", "measured_fluxerr_auto", "measured_mag_auto",
            "gamma1_input_p", "gamma2_input_p", "detected", "neighbored", "distance",
            "r_input_p", "Re_input_p"]
    df = read(DET.format(g=g), cols, max_rows)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    df = df[df["detected"].astype(bool) & ~df["neighbored"].astype(bool)].reset_index(drop=True)
    df = add_measurement_target_features(df)
    g1 = df["gamma1_input_p"].to_numpy(float); g2 = df["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2); ok = gm > 1e-6
    gh1 = np.where(ok, g1 / np.where(ok, gm, 1), 0.0); gh2 = np.where(ok, g2 / np.where(ok, gm, 1), 0.0)
    proj = df["measured_e1_image"].to_numpy(float) * gh1 + df["measured_e2_image"].to_numpy(float) * gh2
    fin = np.isfinite(proj) & ok
    r = proj[fin] / gm[fin]
    return float(np.mean(r)), float(np.std(r) / np.sqrt(fin.sum())), int(fin.sum())


def const_isolated_response(max_rows):
    """NGMIX antithetic response of isolated galaxies, raw and chi->eps."""
    cols = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "neighbored", "distance",
            "axis_ratio_input_p", "position_angle_input_p", "r_input_p", "Re_input_p"]
    df = read(CONST, cols, max_rows)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    df = df[~df["neighbored"].astype(bool)].reset_index(drop=True)
    g = float(np.median(np.hypot(df["applied_g1"], df["applied_g2"])))
    gm = np.hypot(df["applied_g1"], df["applied_g2"]).to_numpy(float)
    gh1 = df["applied_g1"].to_numpy(float) / gm; gh2 = df["applied_g2"].to_numpy(float) / gm
    e1p = df["measured_e1_plus"].to_numpy(float); e2p = df["measured_e2_plus"].to_numpy(float)
    e1m = df["measured_e1_minus"].to_numpy(float); e2m = df["measured_e2_minus"].to_numpy(float)
    raw = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / (2 * g)
    E1p, E2p = chi_to_eps(e1p, e2p); E1m, E2m = chi_to_eps(e1m, e2m)
    eps = ((E1p - E1m) * gh1 + (E2p - E2m) * gh2) / (2 * g)
    fin = np.isfinite(raw) & np.isfinite(eps)
    return (float(np.mean(raw[fin])), float(np.std(raw[fin]) / np.sqrt(fin.sum())),
            float(np.mean(eps[fin])), float(np.std(eps[fin]) / np.sqrt(fin.sum())),
            int(fin.sum()), g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=20_000_000)
    args = ap.parse_args()

    print("ISOLATED-galaxy shear response by estimator (model-vs-sim decomposition)\n")
    print("FLOW R_model (isolated, +-0.02, SExtractor eps target) = 0.2392  [from gold run]\n")

    for g in ("0.05", "0.02"):
        R, e, n = det_isolated_response(g, args.max_rows)
        print(f"det_meas SExtractor eps  g={g}:  R={R:.4f} +/- {e:.4f}   (N={n:,})")

    Rraw, eraw, Reps, eeps, n, g = const_isolated_response(args.max_rows)
    print(f"\nconst NGMIX antithetic  |g|={g:.3f}:")
    print(f"  raw (measured_e1/e2 as stored):  R={Rraw:.4f} +/- {eraw:.4f}   (N={n:,})")
    print(f"  chi->eps converted:              R={Reps:.4f} +/- {eeps:.4f}")
    print("\nINTERPRETATION:")
    print("  det_meas SExtractor ~ 0.239 == flow  -> flow faithful; gold gap is SExtractor-vs-NGMIX")
    print("  (a SIM/estimator-definition issue). det_meas ~ 0.2325 -> a MODEL issue.")


if __name__ == "__main__":
    main()
