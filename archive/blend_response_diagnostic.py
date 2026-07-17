"""Probe 1 of the blending investigation: does blending change the shear response, and does
the flow capture it?

Split the detected g=0.05 sample by the catalogue `neighbored` flag (isolated vs blended) and
compare, per subset:
  R_sim(subset)   = <e_meas . ghat / |g|>                       (model-free truth)
  R_model(subset) = [<E[e_hat|S_g(x)].ghat> - <E[e_hat|x].ghat>]/g   (flow induced response)

  * R_sim(isolated) vs R_sim(blended)   -> does blending change the TRUE response?
  * R_model/R_sim per subset            -> does the flow (which conditions on neighbour
                                           features) reproduce the blended vs isolated response?
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
    ap.add_argument("--max-rows", type=int, default=2_000_000)
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
            + meas["measured_e2_image"].to_numpy(float) * gh2) / gmag  # per-galaxy R_sim
    nb = base["neighbored"].astype(bool).to_numpy()
    intr = (base["e1_input_rot0_p"].to_numpy(float).copy(), base["e2_input_rot0_p"].to_numpy(float).copy())

    print(f"model={os.path.basename(args.measurement_model)}  g={g}  N={len(base):,}  "
          f"blended_frac={nb.mean():.3f}")
    print(f"{'subset':>10} {'N':>10} {'frac':>6} {'R_sim':>8} {'R_model':>8} {'ratio':>7}")
    for name, mask in (("all", np.ones(len(base), bool)), ("isolated", ~nb), ("blended", nb)):
        sub = base[mask].reset_index(drop=True)
        gh1m, gh2m = gh1[mask], gh2[mask]
        intrm = (intr[0][mask], intr[1][mask])
        Rsim = float(np.mean(proj[mask]))
        m0, _ = model_mean_proj(bundle, sub, 0.0, gh1m, gh2m, intrm, rk, args.n_samples, args.batch_size)
        mg, _ = model_mean_proj(bundle, sub, g, gh1m, gh2m, intrm, rk, args.n_samples, args.batch_size)
        Rmodel = (mg - m0) / g
        print(f"{name:>10} {int(mask.sum()):>10,} {mask.mean():>6.3f} "
              f"{Rsim:>8.4f} {Rmodel:>8.4f} {Rmodel/Rsim:>7.3f}")
    print("\nR_sim(blended) vs R_sim(isolated): does blending change the true response?")
    print("ratio R_model/R_sim per subset: does the flow reproduce it (it conditions on neighbours)?")


if __name__ == "__main__":
    main()
