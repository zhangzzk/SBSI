"""Probe 3: is the shear response RESOLVED within blends -- does it vary with blend severity
(neighbour distance), and does the flow track that variation?

Bin the BLENDED detected g=0.05 sample by neighbour `distance` (closer = more severe blend)
and compare per bin:
  R_sim(bin)   = <e_meas . ghat / |g|>                         (truth)
  R_model(bin) = [<E[e_hat|S_g(x)].ghat> - <E[e_hat|x].ghat>]/g  (flow induced response)

Probe 2 showed the flow's neighbour adjustment has the WRONG sign; here we see whether the
true response actually depends on blend severity and how badly the flow tracks it across it.
"""
import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model, add_measurement_target_features  # noqa
from sbs_shear.preprocessing import rescale  # noqa
from scripts.response_ratio_diagnostic import load_sheared_sample, model_mean_proj  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=2_500_000)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--bin-by", default="distance", choices=["distance", "flux_ratio"])
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
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    g = args.nominal_g

    base = load_sheared_sample(args.catalogue, bundle, args.max_rows, shear_threshold=1e-6, seed=7)
    g1 = base["gamma1_input_p"].to_numpy(float); g2 = base["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2); gh1, gh2 = g1 / gmag, g2 / gmag
    meas = add_measurement_target_features(rescale(base.copy(), **rk))
    proj = (meas["measured_e1_image"].to_numpy(float) * gh1
            + meas["measured_e2_image"].to_numpy(float) * gh2) / gmag
    nb = base["neighbored"].astype(bool).to_numpy()
    intr = (base["e1_input_rot0_p"].to_numpy(float).copy(), base["e2_input_rot0_p"].to_numpy(float).copy())

    if args.bin_by == "distance":
        key = base["distance"].to_numpy(float)
    else:
        key = (base["r_input_s"].to_numpy(float) - base["r_input_p"].to_numpy(float)
               if "r_input_s" in base.columns else base["distance"].to_numpy(float))
    bl = nb & np.isfinite(key)
    print(f"model={os.path.basename(args.measurement_model)}  blended N={int(bl.sum()):,}  bin_by={args.bin_by}")
    edges = np.quantile(key[bl], np.linspace(0, 1, args.n_bins + 1)); edges[0] -= 1e-9; edges[-1] += 1e-9
    binidx = np.clip(np.digitize(key, edges) - 1, 0, args.n_bins - 1)

    print(f"{'bin':>3} {'range':>16} {'N':>9} {'R_sim':>8} {'R_model':>8} {'ratio':>7}")
    Rs, Rm = [], []
    for b in range(args.n_bins):
        sel = bl & (binidx == b)
        if sel.sum() < 2000:
            continue
        sub = base[sel].reset_index(drop=True)
        gh1m, gh2m = gh1[sel], gh2[sel]
        intrm = (intr[0][sel], intr[1][sel])
        R_sim = float(np.mean(proj[sel]))
        m0, _ = model_mean_proj(bundle, sub, 0.0, gh1m, gh2m, intrm, rk, args.n_samples, args.batch_size)
        mg, _ = model_mean_proj(bundle, sub, g, gh1m, gh2m, intrm, rk, args.n_samples, args.batch_size)
        R_model = (mg - m0) / g
        lo, hi = key[sel].min(), key[sel].max()
        print(f"{b:>3} {lo:7.2f}-{hi:<8.2f} {int(sel.sum()):>9,} {R_sim:>8.4f} {R_model:>8.4f} {R_model/R_sim:>7.3f}")
        Rs.append(R_sim); Rm.append(R_model)
    if Rs:
        Rs, Rm = np.array(Rs), np.array(Rm)
        print(f"\nR_sim   spans {Rs.min():.3f}..{Rs.max():.3f}  (does the TRUE response vary with blend severity?)")
        print(f"R_model spans {Rm.min():.3f}..{Rm.max():.3f}  (does the flow track it -- and in the right direction?)")


if __name__ == "__main__":
    main()
