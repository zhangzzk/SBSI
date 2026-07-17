"""Response-ratio gating diagnostic for the response-aware (Sobolev) plan.

SBI_shear_response.md proposes supervising the forward model's first-order shear
response (Jacobian) with paired sims, rather than only its likelihood.  That fixes
the +3% multiplicative bias m ONLY IF the model-vs-sim response mismatch is a
shear-INDEPENDENT multiplicative constant: then a response correction calibrated at
small shear transfers out to finite shear (0.05, 0.2).  If the mismatch varies with
shear, near-0 supervision will NOT fix finite-shear m, and the heavy retrain is not
worth launching.

This script measures, at each held-out shear g in {0.05, 0.2}:

  m_sim(g)   = < e_meas . ghat >                 (measured shape projected on the
                                                  per-object applied-shear direction)
  R_sim(g)   = m_sim(g) / g                       (sim first-moment shear response)

  m_model(s) = < E[e_hat | S_{s*ghat}(x)] . ghat> (induced flow mean, conditioning
                                                  truth sheared by the analytic map)
  R_model(g) = m_model(g) / g                     (model induced response at g)
  R_local    = m_model(delta) / delta             (model induced response near 0)

Decisive outputs:
  * R_sim(0.05) vs R_sim(0.2)        -> is the TRUE response linear in shear?
  * R_model(0.05) vs R_model(0.2)    -> is the model induced response linear?
  * ratio(g) = R_model(g)/R_sim(g)   -> is the mismatch a shear-independent constant?
    If ratio(0.05) ~= ratio(0.2) -> response-aware training transfers -> GO heavy.
    If they differ                -> local supervision won't fix finite-shear m.

No retraining; pure forward evaluation of the existing trained flow.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    load_measurement_model,
    add_measurement_target_features,
    raw_columns_for_measurement_targets,
)
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402


def load_sheared_sample(catalogue, bundle, max_rows, shear_threshold, seed, max_read_batches=None,
                        snr_min=None):
    """Stream a sheared catalogue, apply the standard cuts + detected, and reservoir-
    sample.  Keep intrinsic shape, applied-shear truth, and the measured targets.
    snr_min applies a measured-quality cut (measured_flux_auto/fluxerr_auto > snr_min)
    on the SHEARED measured quantity -- a shear-dependent selection."""
    rng = np.random.default_rng(seed)
    condition_features = bundle.condition_preprocessor.feature_names
    target_features = bundle.target_transform.target_names

    with ipc.open_file(catalogue) as reader:
        available = set(reader.schema.names)
        needed = set()
        needed |= raw_columns_for_selection_features(condition_features, available_columns=available)
        needed |= raw_columns_for_measurement_targets(target_features)
        needed |= {"detected", "gamma1_input_p", "gamma2_input_p"}
        needed |= {"e1_input_rot0_p", "e2_input_rot0_p"}
        needed |= {"r_input_p", "Re_input_p", "distance", "neighbored"}
        needed |= {"measured_flux_auto", "measured_fluxerr_auto", "measured_mag_auto"}
        read_columns = sorted(c for c in needed if c in available)

        reservoir = None
        raw_rows = 0
        for bi in range(reader.num_record_batches):
            if max_read_batches is not None and bi >= max_read_batches:
                break
            batch = pa.Table.from_batches([reader.get_batch(bi)]).select(read_columns).to_pandas()
            raw_rows += len(batch)
            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
            if len(batch) == 0:
                continue
            batch = batch[batch["detected"].astype(bool)].reset_index(drop=True)
            if len(batch) == 0:
                continue
            if snr_min is not None and "measured_flux_auto" in batch.columns:
                snr = batch["measured_flux_auto"].to_numpy(float) / batch["measured_fluxerr_auto"].to_numpy(float)
                batch = batch[np.isfinite(snr) & (snr > snr_min)].reset_index(drop=True)
                if len(batch) == 0:
                    continue
            gmag = np.hypot(batch["gamma1_input_p"].to_numpy(float), batch["gamma2_input_p"].to_numpy(float))
            batch = batch[gmag > shear_threshold].reset_index(drop=True)
            if len(batch) == 0:
                continue
            batch = batch.copy()
            batch["__key"] = rng.random(len(batch))
            reservoir = batch if reservoir is None else pd.concat([reservoir, batch], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)

    if reservoir is None:
        raise SystemExit(f"No sheared rows selected from {catalogue}")
    if len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    reservoir = reservoir.drop(columns="__key").reset_index(drop=True)
    print(f"  raw scanned={raw_rows:,}  kept (sheared)={len(reservoir):,}")
    return reservoir


def _shape_target_indices(names):
    """Locate the (e1,e2)-like shape target pair among the flow's target names,
    supporting both SExtractor (measured_e1_image/e2_image) and ngmix
    (measured_ngmix_g1/g2) conventions."""
    for c1, c2 in (("measured_e1_image", "measured_e2_image"),
                   ("measured_ngmix_g1", "measured_ngmix_g2"),
                   ("measured_galsim_g1", "measured_galsim_g2")):
        if c1 in names and c2 in names:
            return names.index(c1), names.index(c2)
    raise KeyError(f"No known shape target pair in {names}")


def model_mean_proj(bundle, base, s, ghat1, ghat2, intrinsic, rescale_kwargs,
                    n_samples, batch_size, return_proj=False):
    """< E[e_hat | S_{s*ghat}(intrinsic)] . ghat >  -- induced flow first moment
    projected onto the per-object applied-shear direction.

    Returns (global_mean, sem).  With return_proj=True also returns the per-object
    projection array `proj` (shape N,), letting callers form a per-object response
    (proj_{+g} - proj_{-g})/(2g).  Because the global mean is exactly np.mean(proj),
    the scalar response is identical whether taken from the two means or from the
    per-object array -- so exposing proj never changes the certified global R_flow."""
    frame = base.copy()
    e1p, e2p = apply_shear_to_ellipticity(intrinsic[0], intrinsic[1], s * ghat1, s * ghat2)
    frame["e1_input_rot0_p"] = e1p
    frame["e2_input_rot0_p"] = e2p
    frame = rescale(frame, **rescale_kwargs)
    draws = bundle.sample(frame, n_samples=n_samples, batch_size=batch_size)  # (N, n_samples, dim)
    mean = draws.mean(axis=1)  # (N, dim) in engineered target units
    i1, i2 = _shape_target_indices(bundle.target_transform.target_names)
    proj = mean[:, i1] * ghat1 + mean[:, i2] * ghat2
    gmean, sem = float(np.mean(proj)), float(np.std(proj) / np.sqrt(len(proj)))
    if return_proj:
        return gmean, sem, proj
    return gmean, sem


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
