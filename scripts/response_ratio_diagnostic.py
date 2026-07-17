"""Response-ratio gating diagnostic (CLI) + back-compat shim for the response library.

The response-evaluation helpers (``load_sheared_sample``, ``_shape_target_indices``,
``model_mean_proj``, and the ``flow_response`` secant) now live in
``sbs_shear/response.py``.  They are re-exported here unchanged so existing
``from scripts.response_ratio_diagnostic import model_mean_proj, _shape_target_indices``
imports keep working byte-for-byte.  The CLI ``main()`` below is the original
response-ratio gating diagnostic for the response-aware (Sobolev) plan
(``SBI_shear_response.md``).

SBI_shear_response.md proposes supervising the forward model's first-order shear
response (Jacobian) with paired sims, rather than only its likelihood.  That fixes
the +3% multiplicative bias m ONLY IF the model-vs-sim response mismatch is a
shear-INDEPENDENT multiplicative constant: then a response correction calibrated at
small shear transfers out to finite shear (0.05, 0.2).  This script measures, at each
held-out shear g in {0.05, 0.2}, R_sim(g) and the model induced R_model(g), and the
ratio(g) = R_model(g)/R_sim(g): if ratio(0.05) ~= ratio(0.2) the mismatch is a
shear-independent constant and response-aware training transfers.  No retraining;
pure forward evaluation of the existing trained flow.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

# Re-exported for back-compat: these used to be defined in this module.
from sbs_shear.response import (  # noqa: E402,F401
    load_sheared_sample,
    _shape_target_indices,
    model_mean_proj,
    flow_response,
)
from sbs_shear.measurement_model import (  # noqa: E402
    load_measurement_model,
    add_measurement_target_features,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue-005", required=True)
    ap.add_argument("--catalogue-020", required=True)
    ap.add_argument("--max-rows", type=int, default=400_000,
                    help="rows kept for the (cheap, column-only) sim first moment R_sim")
    ap.add_argument("--model-max-rows", type=int, default=400_000,
                    help="subsample for the (GPU) model induced-response eval R_model")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--delta", type=float, default=0.01, help="small shear for the near-0 local slope")
    ap.add_argument("--shear-threshold", type=float, default=0.01)
    ap.add_argument("--snr-min", type=float, default=None,
                    help="measured-quality cut: keep measured_flux_auto/fluxerr_auto > this "
                         "(a shear-dependent selection on the measured sample).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default=os.path.join(SBSI_ROOT, "results/response_ratio"))
    ap.add_argument("--pixel-rms", type=float, default=0.312)
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--zero-mag", type=float, default=30.0)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--moffat-beta", type=float, default=2.224)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    bundle = load_measurement_model(args.measurement_model, device=device)
    tag = os.path.splitext(os.path.basename(args.measurement_model))[0]
    print(f"Model: {tag}  targets={bundle.target_transform.target_names}")

    rescale_kwargs = dict(
        pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
        psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta,
    )

    rows = []
    for g, cat in ((0.05, args.catalogue_005), (0.20, args.catalogue_020)):
        print(f"\n########## nominal g={g}  catalogue={os.path.basename(cat)} ##########", flush=True)
        base = load_sheared_sample(cat, bundle, args.max_rows, args.shear_threshold, args.seed,
                                   snr_min=args.snr_min)

        g1 = base["gamma1_input_p"].to_numpy(float)
        g2 = base["gamma2_input_p"].to_numpy(float)
        gmag = np.hypot(g1, g2)
        ghat1 = g1 / gmag
        ghat2 = g2 / gmag
        print(f"  applied |g|: median={np.median(gmag):.4f}")

        intrinsic = (base["e1_input_rot0_p"].to_numpy(float).copy(),
                     base["e2_input_rot0_p"].to_numpy(float).copy())

        # --- simulation first moment (full sample, column-only -> precise R_sim) ---
        meas = add_measurement_target_features(base.copy())
        e1m = meas["measured_e1_image"].to_numpy(float)
        e2m = meas["measured_e2_image"].to_numpy(float)
        proj = e1m * ghat1 + e2m * ghat2
        m_sim = float(np.mean(proj))
        m_sim_sem = float(np.std(proj) / np.sqrt(len(proj)))
        R_sim = m_sim / g
        R_sim_sem = m_sim_sem / g

        # --- model induced first moment at s=0, delta, g (subsample -> cheap GPU eval) ---
        t0 = time.time()
        ns = min(args.model_max_rows, len(base))
        sub = base.iloc[:ns].reset_index(drop=True)
        gh1s, gh2s = ghat1[:ns], ghat2[:ns]
        intr_s = (intrinsic[0][:ns], intrinsic[1][:ns])
        m_mod_0, _ = model_mean_proj(bundle, sub, 0.0, gh1s, gh2s, intr_s,
                                     rescale_kwargs, args.n_samples, args.batch_size)
        m_mod_d, _ = model_mean_proj(bundle, sub, args.delta, gh1s, gh2s, intr_s,
                                     rescale_kwargs, args.n_samples, args.batch_size)
        m_mod_g, m_mod_g_sem = model_mean_proj(bundle, sub, g, gh1s, gh2s, intr_s,
                                               rescale_kwargs, args.n_samples, args.batch_size)
        # responses (subtract the s=0 baseline to cancel any residual intrinsic projection)
        R_model = (m_mod_g - m_mod_0) / g
        R_local = (m_mod_d - m_mod_0) / args.delta
        ratio = R_model / R_sim if R_sim != 0 else float("nan")
        print(f"  [sim]   m_sim={m_sim:+.5f} +/-{m_sim_sem:.5f}   R_sim={R_sim:+.4f} +/-{R_sim_sem:.4f}  (n={len(base):,})")
        print(f"  [model] m0={m_mod_0:+.5f} m_delta={m_mod_d:+.5f} m_g={m_mod_g:+.5f}")
        print(f"  [model] R_model(g)={R_model:+.4f}   R_local(0->{args.delta})={R_local:+.4f}")
        print(f"  ratio R_model/R_sim = {ratio:+.4f}    (eval {time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(g=g, m_sim=m_sim, m_sim_sem=m_sim_sem, R_sim=R_sim, R_sim_sem=R_sim_sem,
                         m_mod_0=m_mod_0, m_mod_delta=m_mod_d, m_mod_g=m_mod_g, R_model=R_model,
                         R_local=R_local, ratio=ratio, n=len(base)))

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out = os.path.join(args.output_dir, f"response_ratio_{tag}.csv")
    df.to_csv(out, index=False)

    print("\n================ SUMMARY ================")
    print(df.to_string(index=False))
    if len(df) == 2:
        rs = df["R_sim"].to_numpy()
        rm = df["R_model"].to_numpy()
        rt = df["ratio"].to_numpy()
        print(f"\nR_sim:   0.05 -> {rs[0]:+.4f}   0.2 -> {rs[1]:+.4f}   "
              f"(sim response { 'LINEAR' if abs(rs[0]-rs[1])/abs(rs[0])<0.05 else 'shear-DEPENDENT' })")
        print(f"R_model: 0.05 -> {rm[0]:+.4f}   0.2 -> {rm[1]:+.4f}")
        print(f"ratio:   0.05 -> {rt[0]:+.4f}   0.2 -> {rt[1]:+.4f}   "
              f"(mismatch { 'CONSTANT -> response-aware transfers -> GO' if abs(rt[0]-rt[1])/abs(rt[0])<0.05 else 'shear-DEPENDENT -> near-0 supervision will NOT fix finite-shear m' })")

        # --- held-out FIRST-MOMENT m: calibrate responsivity at one shear, predict the
        # other.  s_hat(g) = M_data(g)/R_cal; m = s_hat/g - 1 = R_sim(g)/R_sim(g_cal) - 1.
        # This is the legitimate response-aware estimate: the responsivity is calibrated
        # from the sim's near-0 response and TESTED at the held-out shear (no test-set m
        # divide-out).  Selection-consistent (detected sample on both sides).
        rss = df["R_sim_sem"].to_numpy()
        m_held_02 = rs[1] / rs[0] - 1.0   # calibrate@0.05, test@0.2
        m_held_005 = rs[0] / rs[1] - 1.0  # calibrate@0.2,  test@0.05
        # error propagation (R_sim errors independent)
        e02 = abs(rs[1] / rs[0]) * np.hypot(rss[1] / rs[1], rss[0] / rs[0])
        e005 = abs(rs[0] / rs[1]) * np.hypot(rss[0] / rs[0], rss[1] / rs[1])
        print("\n--- held-out FIRST-MOMENT m (response-aware, selection-consistent) ---")
        print(f"  calibrate R at 0.05, predict 0.2:  m = {m_held_02:+.4f} +/- {e02:.4f}")
        print(f"  calibrate R at 0.2,  predict 0.05: m = {m_held_005:+.4f} +/- {e005:.4f}")
        print(f"  (pure-forward first-moment m, no sim response: "
              f"0.05 {rs[0]/rm[0]-1:+.3f}, 0.2 {rs[1]/rm[1]-1:+.3f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
