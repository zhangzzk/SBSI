"""Is the model's shear response RESOLVED across parameter space, or a single global
number?  Bin the g=0.05 sample by measured SNR (and magnitude) and compare, per bin:

  R_sim(bin)   = <e_meas . ghat>/g            (the TRUE response of that subpopulation)
  R_model(bin) = induced flow first-moment response on that subpopulation

A LINEAR mean head has a constant shape-Jacobian, so R_model(bin) is identical in every
bin BY CONSTRUCTION -- the model cannot resolve how the response varies with brightness/
SNR/size.  If R_sim(bin) varies strongly while R_model(bin) is flat, the model is
mis-resolved: it is only right for the exact training selection, and any cut on measured
parameter space exposes a large response mismatch.  An expressive (MLP/interaction) mean
head + property-resolved response supervision is required to fix it.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model, add_measurement_target_features  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, rescale, source_select_selection  # noqa: E402
from scripts.response_ratio_diagnostic import model_mean_proj, load_sheared_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=3_000_000)
    ap.add_argument("--max-read-batches", type=int, default=None,
                    help="cap record batches streamed from the (17GB) feather; None=all")
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--bin-by", default="snr", choices=["snr", "mag"])
    ap.add_argument("--n-samples", type=int, default=64)
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
    rescale_kwargs = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size,
                          zero_mag=args.zero_mag, psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)

    # Stream only the needed columns (the feather is ~17GB; read_feather OOMs at 64G).
    df = load_sheared_sample(args.catalogue, bundle, args.max_rows,
                             shear_threshold=1e-6, seed=7,
                             max_read_batches=args.max_read_batches)
    g = args.nominal_g
    g1 = df["gamma1_input_p"].to_numpy(float); g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2); keep = gmag > 1e-6
    df = df[keep].reset_index(drop=True); g1, g2, gmag = g1[keep], g2[keep], gmag[keep]
    ghat1, ghat2 = g1 / gmag, g2 / gmag

    df = rescale(df, **rescale_kwargs)
    df = add_measurement_target_features(df)
    e1m = df["measured_e1_image"].to_numpy(float); e2m = df["measured_e2_image"].to_numpy(float)
    proj = e1m * ghat1 + e2m * ghat2

    snr = df["measured_flux_auto"].to_numpy(float) / df["measured_fluxerr_auto"].to_numpy(float)
    mag = df["measured_mag_auto"].to_numpy(float)
    key = snr if args.bin_by == "snr" else mag
    finite = np.isfinite(key) & np.isfinite(proj)
    edges = np.quantile(key[finite], np.linspace(0, 1, args.n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    binidx = np.clip(np.digitize(key, edges) - 1, 0, args.n_bins - 1)

    intrinsic = (df["e1_input_rot0_p"].to_numpy(float).copy(),
                 df["e2_input_rot0_p"].to_numpy(float).copy())

    print(f"model={os.path.basename(args.measurement_model)}  bin_by={args.bin_by}  "
          f"g={g}  N={len(df):,}")
    print(f"{'bin':>3} {'SNR_range':>16} {'<mag>':>7} {'N':>9} "
          f"{'R_sim':>8} {'R_model':>8} {'ratio':>7}")
    rows = []
    for b in range(args.n_bins):
        sel = (binidx == b) & finite
        n = int(sel.sum())
        if n < 1000:
            continue
        R_sim = float(np.mean(proj[sel])) / g
        sub = df[sel].reset_index(drop=True)
        gh1, gh2 = ghat1[sel], ghat2[sel]
        intr = (intrinsic[0][sel], intrinsic[1][sel])
        m0, _ = model_mean_proj(bundle, sub, 0.0, gh1, gh2, intr, rescale_kwargs, args.n_samples, args.batch_size)
        mg, _ = model_mean_proj(bundle, sub, g, gh1, gh2, intr, rescale_kwargs, args.n_samples, args.batch_size)
        R_model = (mg - m0) / g
        lo, hi = key[sel].min(), key[sel].max()
        print(f"{b:>3} {lo:7.1f}-{hi:<8.1f} {np.mean(mag[sel]):7.2f} {n:>9,} "
              f"{R_sim:8.4f} {R_model:8.4f} {R_model/R_sim:7.3f}")
        rows.append((b, R_sim, R_model))
    if rows:
        Rs = np.array([r[1] for r in rows]); Rm = np.array([r[2] for r in rows])
        print(f"\nR_sim spans {Rs.min():.3f}..{Rs.max():.3f} (x{Rs.max()/Rs.min():.2f} across bins)")
        print(f"R_model spans {Rm.min():.3f}..{Rm.max():.3f} (x{Rm.max()/Rm.min():.2f} across bins)")
        print("=> model " + ("RESOLVES the response" if Rm.max()/Rm.min() > 1.3
                              else "does NOT resolve the response (flat -> single global value)"))


if __name__ == "__main__":
    main()
