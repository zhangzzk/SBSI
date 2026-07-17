"""Probe 2: how much does the flow USE neighbour info, and in which direction?

The neighbour features are gated by `neighbored` (set to the neighbour's value when blended,
else 0).  So flipping neighbored->False on a blended galaxy makes the flow see it as ISOLATED
(all *_blend features -> 0).  We compute the flow's induced response on BLENDED galaxies:

  R_nb_on  : true features (neighbored=True)
  R_nb_off : neighbour info ablated (neighbored=False -> *_blend gate to 0)
  shift    = R_nb_on - R_nb_off   (how much the neighbour features move the response)

Reference truths: R_sim(blended)=0.228, R_sim(isolated)=0.239 (probe 1) -- isolated has the
HIGHER true response.  So a correct flow should make R_nb_off (isolated-like) > R_nb_on.
If instead R_nb_on > R_nb_off, the flow's neighbour adjustment has the WRONG SIGN.
Also ablate the distance feature alone (push neighbour far away) as a softer de-blend.
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


def response(bundle, sub, g, gh1, gh2, intr, rk, ns, bs):
    m0, _ = model_mean_proj(bundle, sub, 0.0, gh1, gh2, intr, rk, ns, bs)
    mg, _ = model_mean_proj(bundle, sub, g, gh1, gh2, intr, rk, ns, bs)
    return (mg - m0) / g


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
            + meas["measured_e2_image"].to_numpy(float) * gh2) / gmag
    nb = base["neighbored"].astype(bool).to_numpy()
    intr = (base["e1_input_rot0_p"].to_numpy(float).copy(), base["e2_input_rot0_p"].to_numpy(float).copy())

    Rsim_iso = float(np.mean(proj[~nb])); Rsim_bl = float(np.mean(proj[nb]))
    print(f"model={os.path.basename(args.measurement_model)}  N={len(base):,}  blended_frac={nb.mean():.3f}")
    print(f"TRUTH:  R_sim(isolated)={Rsim_iso:.4f}   R_sim(blended)={Rsim_bl:.4f}   "
          f"(isolated {'>' if Rsim_iso>Rsim_bl else '<'} blended)\n")

    # work on BLENDED galaxies
    sub = base[nb].reset_index(drop=True)
    gh1b, gh2b = gh1[nb], gh2[nb]
    intrb = (intr[0][nb], intr[1][nb])

    R_on = response(bundle, sub, g, gh1b, gh2b, intrb, rk, args.n_samples, args.batch_size)

    sub_off = sub.copy(); sub_off["neighbored"] = False           # ablate ALL neighbour info
    R_off = response(bundle, sub_off, g, gh1b, gh2b, intrb, rk, args.n_samples, args.batch_size)

    print("BLENDED galaxies, flow induced response:")
    print(f"  R_model (neighbour info ON)  = {R_on:.4f}   ratio to R_sim(blended) = {R_on/Rsim_bl:.3f}")
    print(f"  R_model (neighbour info OFF) = {R_off:.4f}   ratio to R_sim(isolated)= {R_off/Rsim_iso:.3f}")
    print(f"  shift from neighbour features = {R_on - R_off:+.4f}  "
          f"({'flow RAISES response for blended (WRONG sign: truth lowers it)' if R_on>R_off else 'flow lowers response for blended (correct sign)'})")
    print(f"\n  Interpretation: |shift|={abs(R_on-R_off):.4f} = how strongly the flow uses neighbour features.")
    print(f"  If R_off ~ R_sim(isolated) and R_on != R_sim(blended), the gate works but the "
          f"magnitude/sign of the blend adjustment is mis-learned.")


if __name__ == "__main__":
    main()
