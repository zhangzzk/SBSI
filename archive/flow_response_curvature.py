"""Does the FLOW reproduce the simulation's brightness-dependent response CURVATURE?

The simulation shows m_sim(cut) = R_sim(0.02,cut)/R_sim(0.05,cut) - 1 trending to ~-3% for
bright galaxies -- a real shear-nonlinearity of the response.  The flow's induced response
also curves with shear (the analytic Mobius map S_gamma is nonlinear in gamma; the MLP mean
head mu is nonlinear in shape).  This asks whether the flow's OWN curvature matches the sim's
-- i.e. whether a model whose response was calibrated at a SINGLE shear (0.05) already
PREDICTS the held-out 0.02 response, including for bright-selected samples.

For the detected sample, binned/cut by TRUE magnitude r_input_p (shear-independent):
  R_sim(g)   = <e_meas . ghat>/g                                   (model-free)
  R_model(g) = <[mu(S_g(intrinsic)) - mu(intrinsic)] . ghat>/g     (flow, via the analytic map)
             (the residual flow is shape-blind, so the induced first moment is exactly the
              mean-head mu response -- computed deterministically, no sampling.)
  m_sim   = R_sim(0.02)/R_sim(0.05) - 1      m_model = R_model(0.02)/R_model(0.05) - 1

Reading:
  m_model ~ m_sim          -> flow HAS the right curvature; single-shear calib predicts 0.02.
  m_model ~ 0 (flat)       -> flow MISSES the curvature; needs MULTI-shear response supervision.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model, add_measurement_target_features  # noqa
from sbs_shear.preprocessing import rescale  # noqa
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa
from scripts.response_ratio_diagnostic import load_sheared_sample  # noqa


def mu_proj(bundle, base, s, ghat1, ghat2, intrinsic, rescale_kwargs, device, batch=200000):
    """Per-galaxy projected mean-head response: mu(S_s(intrinsic)) . ghat  (deterministic)."""
    frame = base.copy()
    e1p, e2p = apply_shear_to_ellipticity(intrinsic[0], intrinsic[1], s * ghat1, s * ghat2)
    frame["e1_input_rot0_p"] = e1p
    frame["e2_input_rot0_p"] = e2p
    frame = rescale(frame, **rescale_kwargs)
    ctxnp = bundle.condition_preprocessor.transform_frame(frame)
    out = np.empty(len(frame), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(frame), batch):
            ctx = torch.as_tensor(ctxnp[i:i + batch], dtype=torch.float32, device=device)
            mu = bundle.model._mu(ctx).cpu().numpy()
            out[i:i + batch] = mu[:, 0] * ghat1[i:i + batch] + mu[:, 1] * ghat2[i:i + batch]
    return out


def sim_and_model(bundle, cat, gnom, max_rows, rescale_kwargs, device, want_model):
    base = load_sheared_sample(cat, bundle, max_rows, shear_threshold=1e-6, seed=7)
    g1 = base["gamma1_input_p"].to_numpy(float); g2 = base["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2); gh1, gh2 = g1 / gmag, g2 / gmag
    meas = add_measurement_target_features(rescale(base.copy(), **rescale_kwargs))
    proj_sim = (meas["measured_e1_image"].to_numpy(float) * gh1
                + meas["measured_e2_image"].to_numpy(float) * gh2) / gmag  # per-galaxy R_sim
    tmag = base["r_input_p"].to_numpy(float)
    r_model_g = None
    if want_model:
        intr = (base["e1_input_rot0_p"].to_numpy(float).copy(),
                base["e2_input_rot0_p"].to_numpy(float).copy())
        m0 = mu_proj(bundle, base, 0.0, gh1, gh2, intr, rescale_kwargs, device)
        mg = mu_proj(bundle, base, gnom, gh1, gh2, intr, rescale_kwargs, device)
        r_model_g = (mg - m0) / gnom  # per-galaxy R_model(gnom)
    return tmag, proj_sim, r_model_g, gh1, gh2, base, intr if want_model else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--cat-005", required=True)
    ap.add_argument("--cat-002", required=True)
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pixel-rms", type=float, default=0.312)
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--zero-mag", type=float, default=30.0)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--moffat-beta", type=float, default=2.224)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.measurement_model, device=device)
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    print(f"model={os.path.basename(args.measurement_model)}  device={device}")

    # g=0.05 sample: gives R_sim@0.05 AND the flow's R_model@0.05 and R_model@0.02 (curvature).
    print("Loading g=0.05 + computing flow response @0.05 and @0.02 ...")
    tm5, projsim5, rmod5, gh1, gh2, base5, intr = sim_and_model(
        bundle, args.cat_005, 0.05, args.max_rows, rk, device, want_model=True)
    # flow R_model@0.02 on the SAME g=0.05 galaxies (model curvature is sample-population only):
    m0 = mu_proj(bundle, base5, 0.0, gh1, gh2, intr, rk, device)
    m2 = mu_proj(bundle, base5, 0.02, gh1, gh2, intr, rk, device)
    rmod2 = (m2 - m0) / 0.02
    # g=0.02 sample: gives R_sim@0.02 (model-free).
    print("Loading g=0.02 (sim R@0.02) ...")
    tm2, projsim2, _, _, _, _, _ = sim_and_model(
        bundle, args.cat_002, 0.02, args.max_rows, rk, device, want_model=False)

    fracs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    print(f"\n{'keep_f':>7} {'Rsim05':>8} {'Rsim02':>8} {'m_sim%':>7} | "
          f"{'Rmod05':>8} {'Rmod02':>8} {'m_model%':>9}  (cut on TRUE mag)")
    for f in fracs:
        thr5 = np.quantile(tm5, f); k5 = tm5 <= thr5
        thr2 = np.quantile(tm2, f); k2 = tm2 <= thr2
        Rs5 = projsim5[k5].mean(); Rs2 = projsim2[k2].mean()
        msim = Rs2 / Rs5 - 1
        Rm5 = rmod5[k5].mean(); Rm2 = rmod2[k5].mean()  # both on g=0.05 galaxies, same cut
        mmod = Rm2 / Rm5 - 1
        print(f"{f:>7.2f} {Rs5:>8.4f} {Rs2:>8.4f} {msim*100:>+7.2f} | "
              f"{Rm5:>8.4f} {Rm2:>8.4f} {mmod*100:>+9.2f}")
    print("\nm_sim = the real curvature; m_model = the flow's curvature.")
    print("If m_model tracks m_sim -> flow predicts the held-out shear from a single-shear calib.")
    print("If m_model ~ 0 while m_sim trends -> curvature is unconstrained; needs multi-shear supervision.")


if __name__ == "__main__":
    main()
