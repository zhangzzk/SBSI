"""Forward-model first-moment shear estimator, calibrated from g=0 ONLY.

This is the weak-shear linearization of the SBI marginal likelihood (SBI_shear.md §2):
the responsivity is derived from the g=0 forward model, never fit to the sheared data.
It is the robust, density-shape-insensitive alternative to the flow-MLE recovery (whose
~3-6% m is an estimator artifact of imperfect conditional-density shape, not of the
underlying response, which the worklog showed is linear to ~0.4%).

Construction (all calibration quantities are g=0 + analytic S_gamma; NO sheared truth):

  1. M_data : OLS 2x2 response of the measured image shape chi=(measured_e1,e2_image) to
     the intrinsic scene ellipticity eps=(e1,e2_input_p), measured on the g=0 sample.
     This carries the PSF dilution / measurement convention.
  2. Forward first moment mu_chi(s) : apply the analytic reduced-shear map S_{s}(eps) to the
     g=0 scene-ellipticity population (a fixed probe direction; isotropic so direction is
     immaterial) and push it through M_data:  mu_chi(s) = < M_data @ S_{s e1hat}(eps) >_1 .
     d mu_chi/ds at 0 is the forward-model responsivity R_g0 = M_data * rho_S, with rho_S
     the analytic scene responsivity d<eps_par>/ds -- the whole calibration is g=0-derived.
  3. Estimate : on a held-out sheared catalogue, measure the mean measured shape projected
     on each object's own applied direction, <chi_par>_sheared, and invert the (monotone)
     forward curve mu_chi(s)=<chi_par>_sheared for s_hat.  m = s_hat/g - 1.

Validation only (not calibration) compares s_hat to the known nominal g.  Additive c1,c2
come from the unsheared half's <chi>/R_g0 along fixed axes.

This is NOT responsivity_bias.py: that one sets R = <chi_par>_sheared / g using the known
applied shear (circular, illegitimate). Here R comes only from the g=0 model + S_gamma.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import add_measurement_target_features  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    rescale,
    source_select_selection,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

REQ_M, REQ_C = 3.0e-3, 1.0e-3
CD = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"

# Raw columns needed to build eps (scene shape), chi (measured shape), cuts, shear, quality.
RAW = [
    "detected", "gamma1_input_p", "gamma2_input_p",
    "e1_input_rot0_p", "e2_input_rot0_p",
    "measured_a_image", "measured_b_image", "measured_theta_image",
    "measured_flux_auto", "measured_fluxerr_auto", "measured_mag_auto",
    "r_input_p", "Re_input_p", "distance", "neighbored",
]


def _stream(path, max_rows, snr_min, mag_max, seed, max_batches=None):
    """Detected, source-cut rows with finite eps and chi, reservoir-sampled to max_rows.

    Each case is ordered [unsheared half][sheared half], so a capped read must SPREAD its
    batches across the whole file (not take the first k) or it would see only unsheared rows.
    """
    rng = np.random.default_rng(seed)
    reservoir = None
    raw_rows = 0
    with ipc.open_file(path) as reader:
        avail = set(reader.schema.names)
        cols = [c for c in RAW if c in avail]
        n = reader.num_record_batches
        if max_batches is not None and max_batches < n:
            order = sorted(set(int(x) for x in np.linspace(0, n - 1, max_batches)))
        else:
            order = list(range(n))
        for bi in order:
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw_rows += len(b)
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            if len(b) == 0:
                continue
            if snr_min is not None:
                snr = b["measured_flux_auto"].to_numpy(float) / b["measured_fluxerr_auto"].to_numpy(float)
                b = b[np.isfinite(snr) & (snr > snr_min)].reset_index(drop=True)
            if mag_max is not None:
                mag = b["measured_mag_auto"].to_numpy(float)
                b = b[np.isfinite(mag) & (mag < mag_max)].reset_index(drop=True)
            if len(b) == 0:
                continue
            # scene ellipticity in the sky basis == the rot0 components (preprocessing
            # _add_shape_components sets e1_input_p = e1_input_rot0_p); no full rescale needed.
            b["e1_input_p"] = b["e1_input_rot0_p"].to_numpy(float)
            b["e2_input_p"] = b["e2_input_rot0_p"].to_numpy(float)
            b = add_measurement_target_features(b)  # builds measured_e1_image, measured_e2_image
            keep = (np.isfinite(b["e1_input_p"]) & np.isfinite(b["e2_input_p"])
                    & np.isfinite(b["measured_e1_image"]) & np.isfinite(b["measured_e2_image"]))
            b = b[keep].reset_index(drop=True)
            if len(b) == 0:
                continue
            sub = b[["e1_input_p", "e2_input_p", "measured_e1_image", "measured_e2_image",
                     "gamma1_input_p", "gamma2_input_p", "Re_input_p", "r_input_p"]].copy()
            sub["__k"] = rng.random(len(sub))
            reservoir = sub if reservoir is None else pd.concat([reservoir, sub], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__k").reset_index(drop=True)
    if reservoir is None:
        raise SystemExit(f"no rows from {path}")
    reservoir = reservoir.nlargest(min(max_rows, len(reservoir)), "__k").reset_index(drop=True)
    return reservoir, raw_rows


def eps_to_distortion(e1, e2):
    """Scene distortion chi=(1-q^2)/(1+q^2) from reduced-shear ellipticity eps=(1-q)/(1+q):
    chi = 2 eps / (1 + |eps|^2), componentwise (the exact ellipse identity)."""
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    denom = 1.0 + e1 * e1 + e2 * e2
    return 2.0 * e1 / denom, 2.0 * e2 / denom


def _design(e1, e2, mode):
    """Regressor columns predicting the measured distortion from the scene reduced-shear eps.

    reduced    : [e1, e2]                       linear (misses distortion responsivity)
    distortion : [chi1, chi2]                   exact ellipse map chi=2eps/(1+|eps|^2)
    cubic      : [e1, e2, e1*r2, e2*r2]         free isotropic cubic conditional mean E[chi|eps],
                 r2=|eps|^2                      captures the nonlinear/regression response slope
                                                 the population OLS slope misses (~6%).
    """
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    if mode == "reduced":
        return [e1, e2]
    if mode == "distortion":
        c1, c2 = eps_to_distortion(e1, e2)
        return [c1, c2]
    if mode == "cubic":
        r2 = e1 * e1 + e2 * e2
        return [e1, e2, e1 * r2, e2 * r2]
    raise ValueError(mode)


def fit_response(g0, mode):
    """OLS of measured chi=(measured_e1,e2_image) on the g=0 scene-shape design. Returns coef
    (n_feat+1, 2) with the intercept last."""
    e1 = g0["e1_input_p"].to_numpy(float); e2 = g0["e2_input_p"].to_numpy(float)
    cols = _design(e1, e2, mode) + [np.ones(len(g0))]
    X = np.column_stack(cols)
    Y = np.column_stack([g0["measured_e1_image"], g0["measured_e2_image"]])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coef


def forward_curve(g0, coef, s_grid, mode):
    """mu_chi_par(s): mean predicted measured e1 when the g=0 scene eps population is sheared by
    s along a fixed probe direction (1,0) [isotropic, so direction is immaterial].  Per object:
    shear eps by the exact Mobius S_s, build the g=0-fit response design, predict measured chi."""
    e1 = g0["e1_input_p"].to_numpy(float)
    e2 = g0["e2_input_p"].to_numpy(float)
    out = np.empty_like(s_grid)
    for i, s in enumerate(s_grid):
        se1, se2 = apply_shear_to_ellipticity(e1, e2, s, 0.0)
        X = np.column_stack(_design(se1, se2, mode) + [np.ones(len(se1))])
        chi1 = X @ coef[:, 0]                                   # predicted measured e1
        out[i] = chi1.mean()
    return out


def invert(s_grid, mu_grid, target):
    """Solve mu(s)=target by local linear interpolation on the monotone forward curve."""
    j = int(np.argmin(np.abs(mu_grid - target)))
    lo = max(0, j - 1); hi = min(len(s_grid) - 1, j + 1)
    sl = np.polyfit(s_grid[lo:hi + 1], mu_grid[lo:hi + 1], 1)  # local slope a,b
    return (target - sl[1]) / sl[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g0-catalogue", default=os.path.join(CD, "det_meas_g0.0_train.feather"))
    ap.add_argument("--sheared", nargs="+",
                    default=[f"{CD}/det_meas_g0.05_val.feather:0.05",
                             f"{CD}/det_meas_g0.2_val.feather:0.2"],
                    help="catalogue:nominal pairs")
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--g0-max-rows", type=int, default=2_000_000)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--snr-min", type=float, default=None)
    ap.add_argument("--mag-max", type=float, default=None)
    ap.add_argument("--shear-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    print(f"[g0] loading {args.g0_catalogue}")
    g0, g0raw = _stream(args.g0_catalogue, args.g0_max_rows, args.snr_min, args.mag_max,
                        args.seed, args.max_batches)
    print(f"[g0] rows={len(g0):,} (raw {g0raw:,})")
    e1g = g0["e1_input_p"].to_numpy(float); e2g = g0["e2_input_p"].to_numpy(float)
    print(f"[g0] intrinsic <eps^2>={np.mean(e1g**2+e2g**2):.4f}  <1-eps^2>={np.mean(1-e1g**2-e2g**2):.4f}")

    # Pre-load the sheared samples once (they are the data being calibrated; their applied-shear
    # MAGNITUDE is used only to VALIDATE m, never in the responsivity).
    sheared_data = []
    for spec in args.sheared:
        path, nominal = spec.rsplit(":", 1)
        sh, _ = _stream(path, args.max_rows, args.snr_min, args.mag_max, args.seed + 1, args.max_batches)
        g1 = sh["gamma1_input_p"].to_numpy(float); g2 = sh["gamma2_input_p"].to_numpy(float)
        gm = np.hypot(g1, g2)
        msk = gm > args.shear_threshold
        e1m = sh["measured_e1_image"].to_numpy(float); e2m = sh["measured_e2_image"].to_numpy(float)
        gh1 = np.where(msk, g1 / np.where(msk, gm, 1.0), 0.0)
        gh2 = np.where(msk, g2 / np.where(msk, gm, 1.0), 0.0)
        chi_par = (e1m * gh1 + e2m * gh2)[msk]
        sheared_data.append(dict(
            nominal=float(nominal), chi_par=chi_par,
            c1=e1m[~msk].mean(), c2=e2m[~msk].mean(),
            # intrinsic shape + applied shear of the SHEARED objects (for per-object fidelity)
            ei1=sh["e1_input_p"].to_numpy(float)[msk], ei2=sh["e2_input_p"].to_numpy(float)[msk],
            g1=g1[msk], g2=g2[msk], gh1=gh1[msk], gh2=gh2[msk],
            logsize=np.log10(sh["Re_input_p"].to_numpy(float))[msk]))
        print(f"[sheared g={nominal}] sheared rows={chi_par.size:,}  <chi_par>={chi_par.mean():+.5f}")

    s_grid = np.linspace(-0.30, 0.30, 241)
    i0 = int(np.argmin(np.abs(s_grid)))
    for mode in ("cubic", "distortion", "reduced"):
        coef = fit_response(g0, mode)
        mu = forward_curve(g0, coef, s_grid, mode)
        R_g0 = (mu[i0 + 1] - mu[i0 - 1]) / (s_grid[i0 + 1] - s_grid[i0 - 1])
        print(f"\n================  MODE = {mode}  ================")
        print(f"  forward R_g0 = d mu/ds|0 = {R_g0:.4f}  (linear-response slope coef[0,0]={coef[0,0]:.4f})")
        print(f"  vs Stage-IV (|m|<{REQ_M:.0e}, |c|<{REQ_C:.0e}):")
        worst_m = 0.0
        for d in sheared_data:
            mean_par = d["chi_par"].mean()
            sem = d["chi_par"].std(ddof=1) / np.sqrt(d["chi_par"].size)
            s_hat = invert(s_grid, mu, mean_par)
            s_err = abs(invert(s_grid, mu, mean_par + sem) - s_hat)
            m = s_hat / d["nominal"] - 1.0
            worst_m = max(worst_m, abs(m))
            print(f"    g={d['nominal']}: s_hat={s_hat:.5f}+/-{s_err:.5f}  "
                  f"m={m:+.5f}+/-{s_err/d['nominal']:.5f}  {'PASS' if abs(m)<REQ_M else 'FAIL'}")
        c1 = float(np.mean([d["c1"] for d in sheared_data])) / R_g0
        c2 = float(np.mean([d["c2"] for d in sheared_data])) / R_g0
        print(f"    additive c1={c1:+.5f} {'PASS' if abs(c1)<REQ_C else 'FAIL'}  "
              f"c2={c2:+.5f} {'PASS' if abs(c2)<REQ_C else 'FAIL'}")
        print(f"    worst |m| = {worst_m:.5f}  ->  "
              f"{'STAGE-IV PASS' if worst_m < REQ_M and abs(c1) < REQ_C and abs(c2) < REQ_C else 'NOT YET'}")

    # --- Per-object forward-model FIDELITY check (decisive: where does the ~6% live?) ---
    # For each sheared object, shear ITS OWN intrinsic shape by ITS OWN applied gamma (exact
    # Mobius), predict the measured shape via the g=0 cubic conditional mean, and compare the
    # predicted <chi_par> to the ACTUAL <chi_par>.  If predicted == actual, the g=0 forward model
    # is faithful and the responsivity-inversion gap is an estimator/projection artifact; if
    # predicted OVER-shoots, the gap is a real sheared-vs-g0 population effect (selection / noise
    # rectification) NOT captured by conditioning on the true scene shape alone.
    # g=0 size standardization (shared by the size-conditioned design).
    z_g0 = np.log10(g0["Re_input_p"].to_numpy(float))
    zmu, zsd = float(np.nanmean(z_g0)), float(np.nanstd(z_g0) + 1e-9)

    def design_sz(e1, e2, logsize, use_size):
        """cubic shape design, optionally with size-interacted linear shape terms e1*z, e2*z."""
        cols = _design(e1, e2, "cubic")
        if use_size:
            z = (np.asarray(logsize, float) - zmu) / zsd
            cols = cols + [e1 * z, e2 * z]
        return cols

    def fit_sz(use_size):
        cols = design_sz(g0["e1_input_p"].to_numpy(float), g0["e2_input_p"].to_numpy(float),
                         z_g0, use_size) + [np.ones(len(g0))]
        X = np.column_stack(cols)
        Y = np.column_stack([g0["measured_e1_image"], g0["measured_e2_image"]])
        cf, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return cf

    print("\n================  per-object forward-model fidelity (E[chi|scene]) ================")
    print("  predicted = g=0 conditional mean applied to S_gamma(own intrinsic); pred/act->1 is faithful")
    for use_size in (False, True):
        cf = fit_sz(use_size)
        tag = "cubic+size" if use_size else "cubic (shape only)"
        for d in sheared_data:
            se1, se2 = apply_shear_to_ellipticity(d["ei1"], d["ei2"], d["g1"], d["g2"])  # scene=S_g(intrinsic)
            Xp = np.column_stack(design_sz(se1, se2, d["logsize"], use_size) + [np.ones(len(se1))])
            pred_par = (Xp @ cf[:, 0]) * d["gh1"] + (Xp @ cf[:, 1]) * d["gh2"]
            act = d["chi_par"].mean(); pred = pred_par.mean()
            print(f"  [{tag:18s}] g={d['nominal']}: predicted={pred:+.5f}  actual={act:+.5f}  "
                  f"pred/act={pred/act:.4f}  m_implied={pred/act-1:+.4f}")
    print("\n  NOTE: R_g0 is derived only from the g=0 model + analytic S_gamma; the sheared")
    print("  catalogues are used only to measure <chi_par> (the data being calibrated) and to")
    print("  VALIDATE m. No sheared applied-shear value enters the responsivity.")


if __name__ == "__main__":
    main()
