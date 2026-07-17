"""Precompute the property-resolved response target R_sim(bin) for Phase-1.5 training.

Bins the g=0.05 detected+selected sample by TRUE flux (r_input_p) x TRUE size (Re_input_p)
quantiles and measures R_sim(bin) = <e_meas . ghat>/g in each cell.  Saves bin edges + the
R_sim table to npz, which train_measurement_model.py consumes to supervise the model's
induced response PER BIN (so the flow resolves the response across parameter space rather
than collapsing to the population mean).  Binned by TRUE properties because that is what
the model conditions on and can learn; the per-noise-realization part of the SNR spread is
not resolvable and sets the ceiling.
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
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True, help="sheared catalogue (e.g. g=0.05 val)")
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--n-flux", type=int, default=6)
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_feather(args.catalogue)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    df = df[df["detected"].astype(bool)].reset_index(drop=True)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=7).reset_index(drop=True)

    g1 = df["gamma1_input_p"].to_numpy(float); g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2); keep = gmag > 1e-6
    df = df[keep].reset_index(drop=True); g1, g2, gmag = g1[keep], g2[keep], gmag[keep]
    ghat1, ghat2 = g1 / gmag, g2 / gmag

    meas = add_measurement_target_features(df.copy())
    proj = (meas["measured_e1_image"].to_numpy(float) * ghat1
            + meas["measured_e2_image"].to_numpy(float) * ghat2)
    g = args.nominal_g

    flux = df["r_input_p"].to_numpy(float)   # true magnitude (smaller = brighter)
    size = df["Re_input_p"].to_numpy(float)  # true half-light radius
    fin = np.isfinite(flux) & np.isfinite(size) & np.isfinite(proj)
    flux, size, proj = flux[fin], size[fin], proj[fin]

    ef = np.quantile(flux, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-6; ef[-1] += 1e-6
    es = np.quantile(size, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-6; es[-1] += 1e-6
    fi = np.clip(np.digitize(flux, ef) - 1, 0, args.n_flux - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, args.n_size - 1)

    Rsim = np.full((args.n_flux, args.n_size), np.nan)
    cnt = np.zeros((args.n_flux, args.n_size), dtype=np.int64)
    for a in range(args.n_flux):
        for b in range(args.n_size):
            m = (fi == a) & (si == b)
            cnt[a, b] = m.sum()
            if cnt[a, b] > 200:
                Rsim[a, b] = float(np.mean(proj[m])) / g
    # fill any sparse cell with the global mean (safe fallback)
    gm = float(np.mean(proj)) / g
    Rsim = np.where(np.isfinite(Rsim), Rsim, gm)

    np.savez(args.output, edges_flux=ef, edges_size=es, Rsim=Rsim, counts=cnt,
             nominal_g=g, global_R=gm)
    print(f"R_sim(flux x size) target, g={g}, N={len(proj):,}, global R={gm:.4f}")
    print("flux\\size  " + "  ".join(f"s{b}" for b in range(args.n_size)))
    for a in range(args.n_flux):
        print(f"  f{a}  " + "  ".join(f"{Rsim[a,b]:.3f}" for b in range(args.n_size))
              + f"   (n={cnt[a].sum():,})")
    print(f"R_sim spans {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f} "
          f"(x{np.nanmax(Rsim)/np.nanmin(Rsim):.2f})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
