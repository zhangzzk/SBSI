"""Held-out-shear recovery validation (SBI_shear.md §6, primary end-to-end test).

The forward model p_meas(x_hat | x, n) is trained at g=0 only.  Shear is applied to
the *true scene before rendering*, so for a sheared object the rendered scene shape is
S_g(intrinsic).  The g=0 model, fed the sheared truth S_s(intrinsic), should explain the
measured observables x_hat best when the trial magnitude s equals the true applied shear.

This script implements that as a marginal-likelihood scan, which is frame-agnostic (the
targets x_hat are left untouched; only the conditioning truth is shifted by the analytic
S_gamma map in the model's own learned space):

  * Each sheared object has a known per-object applied shear gamma_input (random
    orientation, magnitude = the case value 0.05 / 0.2; see catalog.generate_catalog_
    realization).  ghat = gamma_input / |gamma_input| is its direction.
  * Replace the primary intrinsic shape e_input_rot0_p with S_{s*ghat}(e_input_rot0_p)
    for a grid of magnitudes s, keeping the secondary at its true applied shear.
  * Recompute features (preprocessing.rescale) and the mean log p_meas over the set.
  * The curve mean_logprob(s) should peak at s = |g_applied|.

Run separately on the unsheared half (true |g|=0 -> peak near 0, the null) and the
sheared half (peak near the nominal g) of a held-out catalogue.
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
    raw_columns_for_measurement_targets,
)
from sbs_shear.selection_model import load_selection_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402


SHAPE_COLS = ["e1_input_rot0_p", "e2_input_rot0_p", "e1_input_rot0_s", "e2_input_rot0_s"]


def load_sample(args, bundle):
    """Stream the held-out catalogue, apply cuts + detected, keep intrinsic shape and
    applied-shear truth, reservoir-sample to args.max_rows."""
    rng = np.random.default_rng(args.seed)
    condition_features = bundle.condition_preprocessor.feature_names
    target_features = bundle.target_transform.target_names

    with ipc.open_file(args.catalogue) as reader:
        available = set(reader.schema.names)
        needed = set()
        needed |= raw_columns_for_selection_features(condition_features, available_columns=available)
        needed |= raw_columns_for_measurement_targets(target_features)
        needed |= {"detected", "gamma1_input_p", "gamma2_input_p"}
        needed |= set(SHAPE_COLS)
        # source-cut columns
        needed |= {"r_input_p", "Re_input_p", "distance", "neighbored"}
        # measured-quality-cut columns
        needed |= {"measured_flux_auto", "measured_fluxerr_auto", "measured_mag_auto"}
        read_columns = sorted(c for c in needed if c in available)

        reservoir = None
        raw_rows = 0
        for bi in range(reader.num_record_batches):
            if args.max_read_batches is not None and bi >= args.max_read_batches:
                break
            batch = pa.Table.from_batches([reader.get_batch(bi)]).select(read_columns).to_pandas()
            raw_rows += len(batch)
            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
            if len(batch) == 0:
                continue
            batch = batch[batch["detected"].astype(bool)].reset_index(drop=True)
            if len(batch) == 0:
                continue
            # Measured-quality cut on the TEST sample (observational SNR / magnitude).
            if args.snr_min is not None and "measured_flux_auto" in batch.columns:
                snr = batch["measured_flux_auto"].to_numpy(float) / batch["measured_fluxerr_auto"].to_numpy(float)
                batch = batch[np.isfinite(snr) & (snr > args.snr_min)].reset_index(drop=True)
            if args.mag_max is not None and "measured_mag_auto" in batch.columns:
                mag = batch["measured_mag_auto"].to_numpy(float)
                batch = batch[np.isfinite(mag) & (mag < args.mag_max)].reset_index(drop=True)
            if len(batch) == 0:
                continue
            if args.blend_subset != "all" and "neighbored" in batch.columns:
                nb = batch["neighbored"].astype(bool).to_numpy()
                batch = batch[nb if args.blend_subset == "blended" else ~nb].reset_index(drop=True)
                if len(batch) == 0:
                    continue
            gmag = np.hypot(batch["gamma1_input_p"].to_numpy(float), batch["gamma2_input_p"].to_numpy(float))
            if args.which == "sheared":
                batch = batch[gmag > args.shear_threshold].reset_index(drop=True)
            elif args.which == "unsheared":
                batch = batch[gmag <= args.shear_threshold].reset_index(drop=True)
            if len(batch) == 0:
                continue
            batch = batch.copy()
            batch["__key"] = rng.random(len(batch))
            reservoir = batch if reservoir is None else pd.concat([reservoir, batch], ignore_index=True)
            if len(reservoir) > 2 * args.max_rows:
                reservoir = reservoir.nlargest(args.max_rows, "__key").reset_index(drop=True)

    if reservoir is None:
        raise SystemExit("No rows selected from held-out catalogue")
    if len(reservoir) > args.max_rows:
        reservoir = reservoir.nlargest(args.max_rows, "__key").reset_index(drop=True)
    reservoir = reservoir.drop(columns="__key").reset_index(drop=True)
    print(f"  raw scanned={raw_rows:,}  kept ({args.which})={len(reservoir):,}")
    return reservoir


def mean_logprob_at_shear(bundle, base, s, ghat1, ghat2, intrinsic, rescale_kwargs, batch_size,
                          selection_bundle=None):
    """Return mean per-object log-likelihood with the primary shape sheared by s*ghat.

    Uses log p_meas(x_hat|S_s(x),n); if ``selection_bundle`` is given, adds
    log P(s=1|S_s(x),n) so the objective is the detected-object catalogue density
    log p_cat = log p_meas + log P(s=1) (the selection response is then included)."""
    frame = base.copy()
    e1p, e2p = apply_shear_to_ellipticity(intrinsic[0], intrinsic[1], s * ghat1, s * ghat2)
    frame["e1_input_rot0_p"] = e1p
    frame["e2_input_rot0_p"] = e2p
    # secondary kept at its intrinsic shape (a small approximation); recompute features.
    frame = rescale(frame, **rescale_kwargs)
    lp = bundle.log_prob(frame, batch_size=batch_size)
    if selection_bundle is not None:
        p = selection_bundle.predict_proba(frame, batch_size=batch_size)
        lp = lp + np.log(np.clip(p, 1e-6, 1.0))
    return float(np.mean(lp)), float(np.std(lp) / max(np.sqrt(len(lp)), 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--selection-model", default=None,
                        help="If given, add log P(s=1|S_s(x),n) to the likelihood so the "
                        "objective is the full detected catalogue density p_cat = p_meas*P(s=1).")
    parser.add_argument("--catalogue", required=True, help="Held-out-shear detection+measured catalogue")
    parser.add_argument("--nominal-shear", type=float, required=True, help="Case shear magnitude, e.g. 0.05 or 0.2")
    parser.add_argument("--which", default="sheared", choices=["sheared", "unsheared"])
    parser.add_argument("--shear-threshold", type=float, default=0.01)
    parser.add_argument("--fiducial-angle-deg", type=float, default=0.0,
                        help="Spin-2 fiducial direction (deg) for direction-less (unsheared) "
                        "objects: ghat=(cos2a, sin2a). Use 0 -> measures c1, 45 -> measures c2.")
    parser.add_argument("--blend-subset", default="all", choices=["all", "isolated", "blended"],
                        help="Restrict by neighbour status: isolated=neighbored False, "
                        "blended=neighbored True. Tests whether the ~6% residual is blend-related.")
    parser.add_argument("--snr-min", type=float, default=None,
                        help="Measured-quality cut: keep measured_flux_auto/fluxerr_auto > this.")
    parser.add_argument("--mag-max", type=float, default=None,
                        help="Measured-quality cut: keep measured_mag_auto < this.")
    parser.add_argument("--closure-shear", type=float, default=None,
                        help="CLOSURE TEST: ignore the real measured targets and instead "
                        "draw x_hat from the trained flow at conditioning S_{s0}(intrinsic) "
                        "(s0 = this value, along each object's direction). A correctly "
                        "specified recovery MUST peak at s0; deviation = estimator/model bias.")
    parser.add_argument("--output-dir", default=os.path.join(SBSI_ROOT, "results/heldout_shear_recovery"))
    parser.add_argument("--grid-min", type=float, default=-0.05)
    parser.add_argument("--grid-max", type=float, default=0.30)
    parser.add_argument("--grid-n", type=int, default=36)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-mag", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    bundle = load_measurement_model(args.measurement_model, device=device)
    print(f"Loaded measurement model: cond={len(bundle.condition_preprocessor.feature_names)} "
          f"targets={bundle.target_transform.target_names}")
    selection_bundle = None
    if args.selection_model is not None:
        selection_bundle = load_selection_model(args.selection_model, device=device)
        print(f"Loaded selection model: log P(s=1|x,n) added to likelihood (p_cat)")

    rescale_kwargs = dict(
        pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
        psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta,
    )

    base = load_sample(args, bundle)
    # Per-object applied-shear direction (sky basis) from gamma_input.
    g1 = base["gamma1_input_p"].to_numpy(float)
    g2 = base["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    safe = gmag > 1e-8
    # Direction-less (unsheared) objects get a FIXED fiducial spin-2 direction so the scan
    # measures the recovered shear along that axis. With g_true=0 the recovered value is the
    # additive bias component: angle 0 -> c1, angle 45deg -> c2.
    a = np.deg2rad(args.fiducial_angle_deg)
    fid1, fid2 = np.cos(2 * a), np.sin(2 * a)
    ghat1 = np.where(safe, g1 / np.where(safe, gmag, 1.0), fid1)
    ghat2 = np.where(safe, g2 / np.where(safe, gmag, 1.0), fid2)
    print(f"  applied |g|: median={np.median(gmag[safe]) if safe.any() else 0:.4f} "
          f"frac_sheared={safe.mean():.3f}  (direction-less objects use fiducial ghat=(1,0))")

    intrinsic = (base["e1_input_rot0_p"].to_numpy(float).copy(),
                 base["e2_input_rot0_p"].to_numpy(float).copy())

    if args.closure_shear is not None:
        # Replace real measured targets with model draws at a KNOWN shear s0 along each
        # object's direction.  The scan below must then peak at s0 if the method is sound.
        frame0 = base.copy()
        e1c, e2c = apply_shear_to_ellipticity(
            intrinsic[0], intrinsic[1], args.closure_shear * ghat1, args.closure_shear * ghat2)
        frame0["e1_input_rot0_p"] = e1c
        frame0["e2_input_rot0_p"] = e2c
        frame0 = rescale(frame0, **rescale_kwargs)
        samp = bundle.sample(frame0, n_samples=1).reshape(len(base), -1)
        for j, name in enumerate(bundle.target_transform.target_names):
            base[name] = samp[:, j]
        print(f"  CLOSURE: targets replaced by model draws at s0={args.closure_shear} "
              f"(expect recovered peak at s0)")

    grid = np.linspace(args.grid_min, args.grid_max, args.grid_n)
    t0 = time.time()
    means = np.empty_like(grid)
    sems = np.empty_like(grid)
    for i, s in enumerate(grid):
        means[i], sems[i] = mean_logprob_at_shear(
            bundle, base, s, ghat1, ghat2, intrinsic, rescale_kwargs, args.batch_size,
            selection_bundle=selection_bundle,
        )
        print(f"  s={s:+.4f}  mean_logprob={means[i]:.5f} +/- {sems[i]:.5f}", flush=True)

    # Quadratic fit around the argmax for a sub-grid peak estimate.
    imax = int(np.argmax(means))
    lo, hi = max(0, imax - 3), min(len(grid), imax + 4)
    coef = np.polyfit(grid[lo:hi], means[lo:hi], 2)
    s_hat = -coef[1] / (2 * coef[0]) if coef[0] < 0 else grid[imax]

    os.makedirs(args.output_dir, exist_ok=True)
    cut_tag = ""
    if args.snr_min is not None:
        cut_tag += f"_snr{args.snr_min:g}"
    if args.mag_max is not None:
        cut_tag += f"_mag{args.mag_max:g}"
    if args.closure_shear is not None:
        cut_tag += f"_closure{args.closure_shear:g}"
    if args.selection_model is not None:
        cut_tag += "_pcat"
    if args.blend_subset != "all":
        cut_tag += f"_{args.blend_subset}"
    if args.which == "unsheared":
        cut_tag += f"_ang{args.fiducial_angle_deg:g}"
    tag = f"g{args.nominal_shear}_{args.which}{cut_tag}"
    np.savez(os.path.join(args.output_dir, f"recovery_{tag}.npz"),
             grid=grid, mean_logprob=means, sem=sems, s_hat=s_hat,
             nominal_shear=args.nominal_shear, which=args.which, n_rows=len(base))

    print("\n=== RESULT ===")
    print(f"  which={args.which}  nominal |g|={args.nominal_shear}")
    print(f"  grid-argmax s={grid[imax]:+.4f}  quad-peak s_hat={s_hat:+.4f}")
    print(f"  (expect ~{args.nominal_shear} for sheared, ~0 for unsheared)")
    print(f"  rows={len(base):,}  scan time={time.time()-t0:.0f}s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(grid, means, yerr=sems, fmt="o-", ms=3, lw=1)
        ax.axvline(args.nominal_shear, color="C3", ls="--", label=f"applied |g|={args.nominal_shear}")
        ax.axvline(s_hat, color="C2", ls=":", label=f"recovered s_hat={s_hat:.4f}")
        ax.axvline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel("trial shear magnitude s (along applied direction)")
        ax.set_ylabel("mean log p_meas(x_hat | S_s(x), n)")
        ax.set_title(f"Held-out-shear recovery ({args.which}, nominal |g|={args.nominal_shear})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"recovery_{tag}.png"), dpi=130)
        print(f"  wrote {os.path.join(args.output_dir, f'recovery_{tag}.png')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (plot skipped: {exc})")


if __name__ == "__main__":
    main()
