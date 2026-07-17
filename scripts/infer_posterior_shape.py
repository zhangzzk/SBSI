"""Posterior-mean shape estimator etilde = E[e | ehat, theta_hat] -- driver (WORKLOG cont.23).

The deployment estimator of WORKLOG cont.22: per galaxy, the posterior-mean corrected
shape under the trained measurement flow (true-neighbour conditional; the dtheta_b
marginalization and the selection factor are explicitly NOT included yet).  Direct
end-to-end shear test with NO R decomposition: the antithetic constant-gold render
injects +/-g, so  m_etilde = <(etilde_plus - etilde_minus) . ghat> / (2g) - 1.

Modes
-----
closure : synthetic unit test. Draw e_int from the prior itself, shear by +/-g, draw
          ehat FROM THE FLOW at the true lensed context, invert with the matching
          sheared prior. Tower rule holds by construction -> recovers +/-g to grid
          resolution. Includes a brute-force per-galaxy check of the batched grid math.
g0      : null + calibration on the g=0 training catalogue (real sim measurements):
          <etilde> ~ 0, binned <e_true | etilde> on the identity, MSE(etilde) < MSE(ehat).
gold    : the headline. Constant-gold antithetic render, per-case bootstrap m_etilde,
          cross/additive projections, tables by true magnitude and by R_blend quantile,
          under three priors:
            intrinsic -- empirical g=0 population (mean-0): <etilde>/g measures the
                         shrinkage K = sig_prior^2/(sig_prior^2+sig_meas,eff^2); biased
                         low BY DESIGN (prior misspecification, cont.22 ingredient (a)).
            sheared   -- prior = intrinsic population Mobius-sheared by the KNOWN +/-g
                         of each case (exact pullback, no KDE): the tower rule applies
                         exactly -> the direct test of the posterior formula.
            selfcal   -- deployable: per case+sign, iterate gamma_hat = <etilde>,
                         re-shear the prior at gamma_hat (empirical Bayes fixed point).
                         The map gamma_prior -> <etilde> is affine with contraction
                         1-K ~ 0.81, so plain iteration needs ~30 sweeps AND still
                         undershoots by ~0.2% in m; the default 'aitken' mode instead
                         extrapolates the fixed point from 3 sweeps (Aitken Delta^2)
                         and confirms with 1 more.

The likelihood grid is evaluated ONCE per (galaxy, sign) and stored fp16; every prior
mode (and every selfcal sweep) is a cheap reweighting of the same grid (GPU-chunked).
With --loglike-cache the fp16 grids are persisted to disk, so later prior/selfcal
experiments skip the flow evaluations entirely.

Completing the cont.22 posterior (this file implements all of it):
  --prior-conditioning rmag : pi(e | r-mag bin) instead of the marginal pi(e) --
      conditioning the prior on theta_hat covariates (targets the per-magnitude
      residual pattern of cont.30; exact under the tower rule).
  --marginal-m M            : integral dtheta_b -- the blend conditioning
      (nbr_flux_near/far/max) marginalized over the empirical detected-population
      prior with M per-galaxy draws (deployment: true neighbour fluxes unknown).
  Selection factor          : needs NO extra term -- the posterior is over the
      DETECTED population throughout: the flow is trained on detected galaxies
      (p(ehat|e,theta,det)) and the prior sample is detected+selected (pi(e|det)),
      so Bayes on the detected subpopulation is already selection-consistent.
      (Residual caveat: shearing pi(e|det) assumes detection isotropy in e.)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    rescale,
    source_select_selection,
)
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
from sbs_shear.posterior_shape import (  # noqa: E402
    PosteriorShapeEstimator,
    RadialShapePrior,
    make_e_grid,
    shear_prior_matrix,
)

CATBASE = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
G0_CAT = CATBASE + "det_meas_crowd_conc_g0.0_train_full.feather"
GOLD_CAT = CBASE + "constant_response_catalogue_c40-139.feather"
CROWD_LOOKUP = "results/crowd_flux_conc_c0-199.feather"

# raw ingredients of the g0_crowd_flux_conc conditioning + cuts + rescale inputs
COND_RAW = ["Re_input_p", "r_input_p", "sersic_n_input_p", "redshift_input_p",
            "Re_input_s", "r_input_s", "distance", "neighbored"]
NBR_COLS = ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"]

# magnitude-conditioned prior pi(e | r-mag) bins == the gold diagnostic table edges
MAG_EDGES = np.array([18.0, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 28.1])


def stream_feather(path, columns, max_rows, stride=1, start=0):
    """Stream record batches (optionally strided for even case coverage), select the
    available subset of `columns`, stop after max_rows raw rows (0 = all)."""
    parts, n = [], 0
    with ipc.open_file(path) as r:
        avail = set(r.schema.names)
        cols = [c for c in columns if c in avail]
        for bi in range(start, r.num_record_batches, stride):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            parts.append(b)
            n += len(b)
            if max_rows and n >= max_rows:
                break
    df = pd.concat(parts, ignore_index=True)
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    return df


def auto_stride(path, max_rows):
    """Stride so a max_rows read still covers the full case range of the file.
    Reads at least ~40 batches spread across the file (cases are stored contiguously,
    so a head-only read would collapse the case coverage and the case bootstrap)."""
    if not max_rows:
        return 1
    with ipc.open_file(path) as r:
        nb = r.num_record_batches
        per = r.get_batch(0).num_rows
    need = max(1, int(np.ceil(max_rows / max(per, 1))))
    need = max(need, min(nb, 40))
    return max(1, nb // need)


def get_prior(args, need_r=False):
    """Fit the empirical radial prior; build+cache the (e1,e2[,r]) sample if needed.
    Returns (global prior, sample dataframe) -- the sample df feeds the magnitude-
    conditioned priors when --prior-conditioning rmag is requested."""
    cache = args.prior_sample
    if os.path.exists(cache) and need_r:
        if "r_input_p" not in pf.read_table(cache).schema.names:
            print("prior sample cache lacks r_input_p (needed for rmag conditioning) -> rebuilding")
            os.remove(cache)
    if not os.path.exists(cache):
        print(f"prior sample cache missing -> building from {os.path.basename(args.prior_catalogue)} "
              f"({args.prior_batches} batches)", flush=True)
        cols = ["e1_input_rot0_p", "e2_input_rot0_p", "detected",
                "r_input_p", "Re_input_p", "distance", "neighbored"]
        parts = []
        with ipc.open_file(args.prior_catalogue) as r:
            avail = set(r.schema.names)
            sel = [c for c in cols if c in avail]
            for bi in range(min(args.prior_batches, r.num_record_batches)):
                parts.append(pa.Table.from_batches([r.get_batch(bi)]).select(sel).to_pandas())
        df = pd.concat(parts, ignore_index=True)
        df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
        df = df[df["detected"].astype(bool)].reset_index(drop=True)
        out = df[["e1_input_rot0_p", "e2_input_rot0_p", "r_input_p"]].astype(np.float32)
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        out.reset_index(drop=True).to_feather(cache)
        print(f"  cached {len(out):,} detected+selected intrinsic shapes -> {cache}", flush=True)
    s = pf.read_table(cache).to_pandas()
    prior = RadialShapePrior(s["e1_input_rot0_p"].to_numpy(),
                             s["e2_input_rot0_p"].to_numpy(), n_bins=args.prior_bins)
    print(f"prior: {prior.n_samples:,} shapes, r_max={prior.r_max:.3f}, "
          f"per-comp std={s[['e1_input_rot0_p', 'e2_input_rot0_p']].std().mean():.4f}", flush=True)
    return prior, s


def binned_priors(sample, args):
    """pi(e | r-mag bin): one radial prior per MAG_EDGES bin from the g0 sample."""
    r = sample["r_input_p"].to_numpy(float)
    b = np.clip(np.digitize(r, MAG_EDGES) - 1, 0, len(MAG_EDGES) - 2)
    out = []
    for k in range(len(MAG_EDGES) - 1):
        msk = b == k
        e1 = sample.loc[msk, "e1_input_rot0_p"].to_numpy()
        e2 = sample.loc[msk, "e2_input_rot0_p"].to_numpy()
        out.append(RadialShapePrior(e1, e2, n_bins=args.prior_bins, min_samples=3000))
        print(f"  pi(e|r in [{MAG_EDGES[k]:g},{MAG_EDGES[k+1]:g}]): {int(msk.sum()):,} shapes, "
              f"per-comp std={np.std(np.concatenate([e1, e2])):.4f}", flush=True)
    return out


def loglike_verbose(est, frame, ehat, chunk, tag, pool=None, sample_idx=None):
    """est.log_likelihood (or the marginal variant when pool/sample_idx are given)
    with progress prints (slabs of ~128 chunks)."""
    n = len(frame)
    slab = chunk * 128
    out = np.empty((n, est.G), dtype=np.float16)
    t0 = time.time()
    for s0 in range(0, n, slab):
        s1 = min(s0 + slab, n)
        if pool is None:
            out[s0:s1] = est.log_likelihood(frame.iloc[s0:s1], ehat[s0:s1], chunk=chunk,
                                            row_offset=s0)
        else:
            out[s0:s1] = est.log_likelihood_marginal(frame.iloc[s0:s1], ehat[s0:s1],
                                                     pool, sample_idx[s0:s1],
                                                     chunk=max(64, chunk // 4))
        el = time.time() - t0
        print(f"  [{tag}] {s1:,}/{n:,} ({el:.0f}s, ETA {el/(s1)*(n-s1):.0f}s)", flush=True)
    return out


def mu_correction_grid_table(path, grid):
    """(K,G,2) raw-unit location offsets: the g0-fitted U(e; r-mag cell) polynomials
    (build_mu_correction.py) evaluated on the estimator grid.  Grid points beyond the
    fit's e-range are clipped componentwise (flat extrapolation -- prior support ends
    at |e|~0.9 anyway)."""
    z = np.load(path)
    expo, coeffs = z["expo"], z["coeffs"]                    # (nterms,2), (K,2,nterms)
    g1 = np.clip(grid[:, 0], -0.85, 0.85)
    g2 = np.clip(grid[:, 1], -0.85, 0.85)
    X = np.stack([g1 ** i * g2 ** j for i, j in expo], axis=1)   # (G,nterms)
    return np.einsum("gt,kct->kgc", X, coeffs)


def _aitken(x1, x2, x3, gmax=0.2):
    """Componentwise Aitken Delta^2 extrapolation of a linearly convergent sequence
    x_{n+1} = F(x_n): for an affine F this lands exactly on the fixed point.

    Near-cancelling increments (|den| small from reweight noise) make the jump
    unbounded; any component beyond gmax (10x the applied |g|) falls back to the
    plain iterate x3 -- an overshoot past |g|=1 would otherwise NaN the sheared
    prior's log(1-|g|^2) Jacobian."""
    d1, d2 = x2 - x1, x3 - x2
    den = d2 - d1
    safe = np.abs(den) > 1e-14
    jump = np.where(safe, x3 - d2 * d2 / np.where(safe, den, 1.0), x3)
    return np.where(np.abs(jump) <= gmax, jump, x3)


def boot_mean(values, cases, n_boot, seed=0):
    """Per-case bootstrap std of the global mean of `values`."""
    uc = np.unique(cases)
    per = {c: (float(values[cases == c].sum()), int((cases == c).sum())) for c in uc}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        s = sum(per[c][0] for c in pick)
        n = sum(per[c][1] for c in pick)
        out.append(s / n)
    return float(np.std(out))


# ----------------------------------------------------------------------------- closure
def run_closure(args, bundle, est, prior, grid, rk):
    g1, g2 = args.closure_g1, args.closure_g2
    g = float(np.hypot(g1, g2))
    print(f"\n=== CLOSURE: synthetic ehat from the flow itself, injected g=({g1},{g2}) ===")
    cols = COND_RAW + NBR_COLS + ["detected", "e1_input_rot0_p", "e2_input_rot0_p"]
    df = stream_feather(args.catalogue or G0_CAT, cols, max_rows=4 * args.max_rows)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    df = df[df["detected"].astype(bool)].reset_index(drop=True)
    missing_nbr = [c for c in NBR_COLS if c not in df.columns]
    if missing_nbr:
        raise SystemExit(f"context catalogue lacks {missing_nbr}")
    df = df.iloc[:args.max_rows].reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(args.seed)
    ei1, ei2 = prior.sample(n, rng)  # intrinsic draws FROM the estimation prior
    print(f"contexts N={n:,}  <|e_int|>={np.hypot(ei1, ei2).mean():.4f}")

    pool = df[NBR_COLS].to_numpy(float) if args.marginal_m else None
    sidx = (np.random.default_rng(args.seed + 101)
            .integers(0, len(df), size=(n, args.marginal_m)) if args.marginal_m else None)
    if args.marginal_m:
        print(f"theta_b MARGINAL: M={args.marginal_m} per-galaxy draws from the pool of "
              f"{len(df):,} detected rows (ehat still generated at the TRUE theta_b -> "
              "tower rule over pi(theta_b) applies)")

    et = {}
    ll_cache = {}
    for sign in (+1, -1):
        el1, el2 = apply_shear_to_ellipticity(ei1, ei2, sign * g1, sign * g2)
        fr = df.copy()
        fr["e1_input_rot0_p"] = el1
        fr["e2_input_rot0_p"] = el2
        fr = rescale(fr, **rk)
        ehat = bundle.sample(fr, n_samples=1, batch_size=args.chunk * 8)[:, 0, :]
        ll = loglike_verbose(est, fr, ehat, args.chunk, f"closure {sign:+d}",
                             pool=pool, sample_idx=sidx)
        lp = prior.sheared_log_prob(grid[:, 0], grid[:, 1], sign * g1, sign * g2)
        et[sign], _ = est.posterior_mean(ll, lp)
        ll_cache[sign] = (ll, fr, ehat)

    gh1, gh2 = g1 / g, g2 / g
    dproj = 0.5 * ((et[1][:, 0] - et[-1][:, 0]) * gh1 + (et[1][:, 1] - et[-1][:, 1]) * gh2)
    m = dproj.mean() / g - 1.0
    sem = dproj.std() / np.sqrt(n) / g
    print(f"\nCLOSURE (sheared prior): m_etilde = {m:+.4%} +/- {sem:.4%}   "
          f"(<etilde+.ghat>={((et[1][:,0]*gh1+et[1][:,1]*gh2)).mean():+.5f}, target {g:+.5f})")
    # intrinsic-prior shrinkage on the same likelihoods (reweighting only)
    lp0 = prior.log_prob(grid[:, 0], grid[:, 1])
    e0p, _ = est.posterior_mean(ll_cache[1][0], lp0)
    e0m, _ = est.posterior_mean(ll_cache[-1][0], lp0)
    d0 = 0.5 * ((e0p[:, 0] - e0m[:, 0]) * gh1 + (e0p[:, 1] - e0m[:, 1]) * gh2)
    print(f"CLOSURE (intrinsic prior): <etilde>/g = K = {d0.mean()/g:.4f} (the shrinkage; "
          f"m_K = {d0.mean()/g-1:+.2%} expected < 0)")

    if args.marginal_m:  # brute-force row path below is conditional-only
        return m, sem
    # brute-force hand check: pandas/bundle.log_prob path vs the batched grid math
    ll, fr, ehat = ll_cache[1]
    lp = prior.sheared_log_prob(grid[:, 0], grid[:, 1], g1, g2)
    worst = 0.0
    for i in range(min(3, n)):
        rep = fr.iloc[[i] * len(grid)].reset_index(drop=True).copy()
        rep["e1_input_p"] = grid[:, 0]
        rep["e2_input_p"] = grid[:, 1]
        for j, name in enumerate(bundle.target_transform.target_names):
            rep[name] = ehat[i, j]
        lb = bundle.log_prob(rep)
        w = np.exp(lb + lp - (lb + lp).max())
        ref = (w[:, None] * grid).sum(axis=0) / w.sum()
        bat, _ = est.posterior_mean(ll[i:i + 1], lp)
        worst = max(worst, float(np.abs(ref - bat[0]).max()))
    print(f"brute-force vs batched posterior mean: max |diff| = {worst:.2e} "
          f"(fp16 loglike storage -> expect ~1e-4)")
    return m, sem


# ----------------------------------------------------------------------------- g0 null
def run_g0(args, bundle, est, prior, grid, rk, bpriors=None):
    print("\n=== G0 NULL + CALIBRATION: real measured shapes at g=0 ===")
    tnames = bundle.target_transform.target_names
    cols = COND_RAW + NBR_COLS + ["detected", "e1_input_rot0_p", "e2_input_rot0_p", *tnames]
    df = stream_feather(args.catalogue or G0_CAT, cols, max_rows=3 * args.max_rows)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    df = df[df["detected"].astype(bool)].reset_index(drop=True)
    ehat = df[list(tnames)].to_numpy(np.float32)
    fin = np.isfinite(ehat).all(axis=1)
    df = df[fin].reset_index(drop=True).iloc[:args.max_rows].reset_index(drop=True)
    ehat = df[list(tnames)].to_numpy(np.float32)
    n = len(df)
    fr = rescale(df, **rk)
    print(f"N={n:,} detected+selected, finite {list(tnames)}")

    ll = loglike_verbose(est, fr, ehat, args.chunk, "g0")
    e_true = df[["e1_input_rot0_p", "e2_input_rot0_p"]].to_numpy(float)
    magbin = np.clip(np.digitize(df["r_input_p"].to_numpy(float), MAG_EDGES) - 1,
                     0, len(MAG_EDGES) - 2).astype(np.int64)

    for cond in args.prior_conditioning:
        if cond == "global":
            et, _ = est.posterior_mean(ll, prior.log_prob(grid[:, 0], grid[:, 1]))
        else:
            mat = np.stack([bp.log_prob(grid[:, 0], grid[:, 1]) for bp in bpriors])
            et, _ = est.posterior_mean(ll, mat, case_idx=magbin)
        print(f"\n  --- prior conditioning: {cond} ---")
        for k, lab in ((0, "e1"), (1, "e2")):
            print(f"  <etilde_{lab}> = {et[:,k].mean():+.5f} +/- {et[:,k].std()/np.sqrt(n):.5f}   "
                  f"(<ehat_{lab}> = {ehat[:,k].mean():+.5f})")
        mse_t = float(np.mean((et - e_true) ** 2))
        mse_h = float(np.mean((ehat - e_true) ** 2))
        print(f"  MSE(etilde vs e_true) = {mse_t:.5f}   MSE(ehat vs e_true) = {mse_h:.5f}   "
              f"ratio = {mse_t/mse_h:.3f} (MMSE -> should be < 1)")
        print("\n  calibration <e_true | etilde> (10 quantile bins; identity = calibrated):")
        print(f"  {'bin':>4} {'<etilde_1>':>11} {'<e_true_1>':>11} {'N':>9}    "
              f"{'<etilde_2>':>11} {'<e_true_2>':>11}")
        q = np.quantile(et[:, 0], np.linspace(0, 1, 11))
        q[0] -= 1e-9; q[-1] += 1e-9
        b1 = np.clip(np.digitize(et[:, 0], q) - 1, 0, 9)
        q2 = np.quantile(et[:, 1], np.linspace(0, 1, 11))
        q2[0] -= 1e-9; q2[-1] += 1e-9
        b2 = np.clip(np.digitize(et[:, 1], q2) - 1, 0, 9)
        for c in range(10):
            m1, m2 = b1 == c, b2 == c
            print(f"  {c:>4} {et[m1,0].mean():>11.4f} {e_true[m1,0].mean():>11.4f} "
                  f"{int(m1.sum()):>9,}    {et[m2,1].mean():>11.4f} {e_true[m2,1].mean():>11.4f}")


# ----------------------------------------------------------------------------- gold
def gold_stats(tag, et_p, et_m, gh1, gh2, g, cases, n_boot):
    dp1 = 0.5 * (et_p[:, 0] - et_m[:, 0])
    dp2 = 0.5 * (et_p[:, 1] - et_m[:, 1])
    dproj = dp1 * gh1 + dp2 * gh2
    cross = -dp1 * gh2 + dp2 * gh1
    a1 = 0.5 * (et_p[:, 0] + et_m[:, 0])
    a2 = 0.5 * (et_p[:, 1] + et_m[:, 1])
    err = boot_mean(dproj, cases, n_boot) / g
    print(f"  [{tag:>10}] m_etilde = {dproj.mean()/g - 1:+.4%} +/- {err:.4%}   "
          f"cross = {cross.mean()/g:+.4%}   c_add = ({a1.mean():+.5f},{a2.mean():+.5f})")
    return dproj


def gold_binned(label, edges_or_bins, values, dproj, g, cases, n_boot, quantile=False, iso_eps=None):
    print(f"\n  --- m_etilde by {label} ---")
    v = np.asarray(values, dtype=float)
    if quantile:
        hi = v >= iso_eps
        if hi.sum() < 5000:
            print("  (too few blended objects for quantile bins)")
            return
        qe = np.quantile(v[hi], np.linspace(0, 1, edges_or_bins + 1))
        qe[0] -= 1e-9; qe[-1] += 1e-9
        bins = np.where(~hi, 0, 1 + np.clip(np.digitize(v, qe) - 1, 0, edges_or_bins - 1))
        labels = ["ISO(~0)"] + [f"q{c}" for c in range(1, edges_or_bins + 1)]
        nb = edges_or_bins + 1
    else:
        e = np.asarray(edges_or_bins, dtype=float)
        bins = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
        labels = [f"{e[i]:g}-{e[i+1]:g}" for i in range(len(e) - 1)]
        nb = len(e) - 1
    print(f"  {'bin':>10} {'m_etilde':>9} {'+/-':>7} {'<val>':>8} {'N':>10}")
    for c in range(nb):
        msk = bins == c
        if msk.sum() < 2000:
            continue
        err = boot_mean(dproj[msk], cases[msk], n_boot) / g
        print(f"  {labels[c]:>10} {dproj[msk].mean()/g-1:>+9.2%} {err:>7.2%} "
              f"{v[msk].mean():>8.3f} {int(msk.sum()):>10,}")


def run_gold(args, bundle, est, prior, grid, rk, bpriors=None):
    cat = args.catalogue or GOLD_CAT
    print(f"\n=== GOLD: antithetic constant render {os.path.basename(cat)} ===")
    meas = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus"]
    cache = args.loglike_cache
    if args.mu_correction and args.marginal_m:
        raise SystemExit("--mu-correction is not supported with --marginal-m: the "
                         "marginal likelihood path does not apply the armed correction "
                         "(sbs_shear/posterior_shape.set_mu_correction)")
    cmeta = dict(model=os.path.basename(args.measurement_model),
                 catalogue=os.path.basename(cat), G=int(len(grid)),
                 grid_n=args.grid_n, grid_emax=args.grid_emax, grid_rmax=args.grid_rmax,
                 max_rows=args.max_rows, seed=args.seed, marginal_m=args.marginal_m,
                 crowd=os.path.basename(args.crowd_flux_lookup),
                 mu_corr=os.path.basename(args.mu_correction) if args.mu_correction
                 else None,
                 # likelihood-changing physics + cache format (rows are per-row
                 # max-shifted fp16 on BOTH paths since rowshift-v2)
                 fmt="rowshift-v2", **{k: float(v) for k, v in rk.items()})
    paths = (cache + "_llp.npy", cache + "_llm.npy",
             cache + "_rows.feather", cache + "_meta.json") if cache else None
    cached = bool(cache) and all(os.path.exists(p) for p in paths)
    if cached:
        with open(paths[3]) as fh:
            got = json.load(fh)
        bad = {k: (got.get(k), v) for k, v in cmeta.items() if got.get(k) != v}
        if bad:
            raise SystemExit(f"--loglike-cache mismatch {bad}; delete {cache}_* or change the prefix")
        df = pf.read_table(paths[2]).to_pandas()
        ll_p = np.load(paths[0], mmap_mode="r")
        ll_m = np.load(paths[1], mmap_mode="r")
        if ll_p.shape != (len(df), len(grid)) or ll_m.shape != ll_p.shape:
            raise SystemExit(f"--loglike-cache shape mismatch: {ll_p.shape}/{ll_m.shape} "
                             f"vs (N={len(df):,}, G={len(grid)})")
        print(f"loglike cache HIT: {cache}_*  (N={len(df):,}, G={len(grid)}) -> "
              "skipping catalogue stream + flow evaluations")
    else:
        need = [*meas, "applied_g1", "applied_g2", "case", "input_index",
                "axis_ratio_input_p", "position_angle_input_p", *COND_RAW]
        stride = auto_stride(cat, args.max_rows * 2)  # ~50% headroom for cuts; keep case coverage
        print(f"batch stride={stride} for max_rows={args.max_rows:,}")
        # no early break: read every strided batch (case coverage), subsample rows below
        df = stream_feather(cat, need, max_rows=0, stride=stride)
        e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                     df["position_angle_input_p"].to_numpy(float))
        df["e1_input_rot0_p"] = e1i  # placeholder for rescale/transform; overwritten on the grid
        df["e2_input_rot0_p"] = e2i
        df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
        fin = np.isfinite(df[meas].to_numpy(float)).all(axis=1)
        df = df[fin].reset_index(drop=True)
        if len(df) > args.max_rows:  # random (not head) subsample -> keep case coverage even
            df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

        # nbr_flux_near/far/max from the conc crowding lookup (same merge as the validator)
        cf = pf.read_table(args.crowd_flux_lookup).to_pandas()[["case", "input_index", *NBR_COLS]]
        cm = df[["case", "input_index"]].merge(cf, on=["case", "input_index"], how="left")
        for c in NBR_COLS:
            df[c] = cm[c].fillna(0.0).to_numpy(float)
        print(f"crowd-flux lookup: matched {np.mean(cm['nbr_flux_near'].notna()):.1%} of rows")
        if args.blend_lookup and os.path.exists(args.blend_lookup):
            rb = pf.read_table(args.blend_lookup).to_pandas()[["case", "input_index", "R_blend"]]
            bm = df[["case", "input_index"]].merge(rb, on=["case", "input_index"], how="left")
            df["r_blend"] = bm["R_blend"].fillna(0.0).to_numpy(float)
            print(f"blend lookup (binning only): matched {np.mean(bm['R_blend'].notna()):.1%}")

    cases = df["case"].to_numpy(np.int64)
    ucases = np.unique(cases)
    gmag = np.hypot(df["applied_g1"], df["applied_g2"]).to_numpy(float)
    g = float(np.median(gmag))
    gh1 = df["applied_g1"].to_numpy(float) / gmag
    gh2 = df["applied_g2"].to_numpy(float) / gmag
    gc = df.groupby("case")[["applied_g1", "applied_g2"]].agg(["mean", "std"])
    max_std = float(np.nanmax(gc.xs("std", axis=1, level=1).to_numpy()))
    print(f"N={len(df):,}  cases={len(ucases)}  |g|={g:.4f}  "
          f"max within-case shear std={max_std:.2e} (coherent render check)")
    g1c = gc[("applied_g1", "mean")].reindex(ucases).to_numpy()
    g2c = gc[("applied_g2", "mean")].reindex(ucases).to_numpy()
    case_row = np.searchsorted(ucases, cases)

    ehat_p = df[["measured_e1_plus", "measured_e2_plus"]].to_numpy(np.float32)
    ehat_m = df[["measured_e1_minus", "measured_e2_minus"]].to_numpy(np.float32)
    ts = bundle.target_transform
    print(f"ehat sanity: std(e1_plus)={ehat_p[:,0].std():.4f} vs trained target scale "
          f"{ts.scales[0]:.4f} (should be comparable)")
    # anchor: raw measured response (validator's R_sim) for reference
    r_sim = np.mean(((ehat_p[:, 0] - ehat_m[:, 0]) * gh1 + (ehat_p[:, 1] - ehat_m[:, 1]) * gh2) / (2 * g))
    print(f"raw R_sim = {r_sim:.4f} (reference; the posterior divides this gain out)")

    if not cached:
        fr = rescale(df, **rk)
        if args.mu_correction:
            mb = np.clip(np.digitize(df["r_input_p"].to_numpy(float), MAG_EDGES) - 1,
                         0, len(MAG_EDGES) - 2).astype(np.int64)
            est.set_mu_correction(mu_correction_grid_table(args.mu_correction, grid), mb)
            print(f"mu-correction ARMED: {os.path.basename(args.mu_correction)} "
                  f"({len(MAG_EDGES) - 1} r-mag cells on G={len(grid)}; likelihood "
                  "location = mu + U(e_grid; cell), WORKLOG cont.33)")
        pool = df[NBR_COLS].to_numpy(float) if args.marginal_m else None
        sidx = (np.random.default_rng(args.seed + 101)
                .integers(0, len(df), size=(len(df), args.marginal_m))
                if args.marginal_m else None)
        if args.marginal_m:
            # same per-galaxy quadrature for both signs -> the finite-M noise cancels
            # in the antithetic difference like the intrinsic-shape part does
            print(f"theta_b MARGINAL likelihood: M={args.marginal_m} per-galaxy draws "
                  "from the detected+selected pool (deployment estimator: true "
                  "neighbour fluxes NOT used)")
        ll_p = loglike_verbose(est, fr, ehat_p, args.chunk, "gold +", pool=pool, sample_idx=sidx)
        ll_m = loglike_verbose(est, fr, ehat_m, args.chunk, "gold -", pool=pool, sample_idx=sidx)
        if cache:
            np.save(paths[0], ll_p)
            np.save(paths[1], ll_m)
            keep = [c for c in ("case", "input_index", "applied_g1", "applied_g2",
                                "r_input_p", "r_blend", *meas) if c in df.columns]
            df[keep].reset_index(drop=True).to_feather(paths[2])
            with open(paths[3], "w") as fh:
                json.dump({**cmeta, "n_rows": int(len(df))}, fh)
            print(f"loglike cache SAVED -> {cache}_*  "
                  f"({(ll_p.nbytes + ll_m.nbytes) / 2**30:.1f} GiB fp16)")

    results = {}
    magbin = np.clip(np.digitize(df["r_input_p"].to_numpy(float), MAG_EDGES) - 1,
                     0, len(MAG_EDGES) - 2).astype(np.int64)
    nmb = len(MAG_EDGES) - 1

    def cond_tools(cond):
        """Per-conditioning prior plumbing: (intrinsic (K,G) matrix, its per-row index,
        sheared-matrix builder from per-case shear vectors, its per-row index).
        rmag conditioning uses pi(e | r-mag bin): matrix rows ordered (case, magbin)."""
        if cond == "global":
            intr = prior.log_prob(grid[:, 0], grid[:, 1])[None, :]

            def sheared_mat(a1, a2):
                return shear_prior_matrix(prior, grid, a1, a2)
            return intr, np.zeros(len(df), np.int64), sheared_mat, case_row
        intr = np.stack([bp.log_prob(grid[:, 0], grid[:, 1]) for bp in bpriors])

        def sheared_mat(a1, a2):
            rows = [bp.sheared_log_prob(grid[:, 0], grid[:, 1], float(a1[k]), float(a2[k]))
                    for k in range(len(ucases)) for bp in bpriors]
            return np.stack(rows)
        return intr, magbin, sheared_mat, case_row * nmb + magbin

    def reweight(ll, mat, idx):
        et, _ = est.posterior_mean(ll, mat, case_idx=idx)
        return et

    nk = len(ucases)
    cnt = np.bincount(case_row, minlength=nk).astype(float)

    def case_means(et):
        return np.stack([np.bincount(case_row, et[:, 0], nk),
                         np.bincount(case_row, et[:, 1], nk)], axis=1) / cnt[:, None]

    for cond in args.prior_conditioning:
        intr_mat, intr_idx, sheared_mat, sh_idx = cond_tools(cond)
        tagc = cond[:4]

        if "intrinsic" in args.priors:
            et_p = reweight(ll_p, intr_mat, intr_idx)
            et_m = reweight(ll_m, intr_mat, intr_idx)
            print(f"\nGLOBAL (intrinsic prior | {cond} -- measures the shrinkage K, "
                  "biased low by design):")
            dproj = gold_stats(f"int|{tagc}", et_p, et_m, gh1, gh2, g, cases, args.n_boot)
            print(f"  [{f'int|{tagc}':>10}] K = <dproj>/g = {dproj.mean()/g:.4f}")
            results[f"intrinsic_{cond}"] = (et_p, et_m, dproj)

        if "sheared" in args.priors:
            et_p = reweight(ll_p, sheared_mat(+g1c, +g2c), sh_idx)
            et_m = reweight(ll_m, sheared_mat(-g1c, -g2c), sh_idx)
            print(f"\nGLOBAL (sheared prior at KNOWN g | {cond} -- the direct "
                  "posterior-formula test):")
            dproj = gold_stats(f"she|{tagc}", et_p, et_m, gh1, gh2, g, cases, args.n_boot)
            results[f"sheared_{cond}"] = (et_p, et_m, dproj)

        if "selfcal" in args.priors:
            print(f"\nGLOBAL (selfcal prior | {cond} -- deployable empirical-Bayes "
                  "fixed point):")
            npass = [0]

            def sweep(gp, gm):
                """One reweight pass: etilde under priors sheared at the per-case gamma_hat."""
                npass[0] += 1
                et_p = reweight(ll_p, sheared_mat(gp[:, 0], gp[:, 1]), sh_idx)
                et_m = reweight(ll_m, sheared_mat(gm[:, 0], gm[:, 1]), sh_idx)
                dq = 0.5 * ((et_p[:, 0] - et_m[:, 0]) * gh1 + (et_p[:, 1] - et_m[:, 1]) * gh2)
                return et_p, et_m, case_means(et_p), case_means(et_m), float(dq.mean() / g - 1.0)

            if args.selfcal_mode == "plain":
                gp, gm = np.zeros((nk, 2)), np.zeros((nk, 2))
                for _ in range(args.selfcal_iters):
                    et_p, et_m, gp, gm, m = sweep(gp, gm)
                    print(f"    pass {npass[0]:>2}: m = {m:+.4%}", flush=True)
            else:
                # affine map (contraction 1-K ~ 0.81, stable across sweeps): 3 sweeps ->
                # Aitken Delta^2 jump -> confirming sweep; plain-30 both costs 6x more
                # passes and undershoots the fixed point by ~0.2% (cont.29/30)
                xs_p, xs_m = [np.zeros((nk, 2))], [np.zeros((nk, 2))]
                for _ in range(3):
                    et_p, et_m, gp, gm, m = sweep(xs_p[-1], xs_m[-1])
                    xs_p.append(gp)
                    xs_m.append(gm)
                    print(f"    pass {npass[0]:>2}: m = {m:+.4%}", flush=True)
                resid = float("inf")
                for cyc in range(args.selfcal_cycles):
                    sp = _aitken(xs_p[-3], xs_p[-2], xs_p[-1])
                    sm = _aitken(xs_m[-3], xs_m[-2], xs_m[-1])
                    et_p, et_m, gp, gm, m = sweep(sp, sm)
                    resid = max(np.abs(gp - sp).max(), np.abs(gm - sm).max())
                    print(f"    pass {npass[0]:>2} (Aitken jump {cyc + 1}): m = {m:+.4%}   "
                          f"|F(x*)-x*|_max = {resid:.2e} (gamma units, tol {args.selfcal_tol:g})",
                          flush=True)
                    if resid < args.selfcal_tol:
                        break
                    et_p, et_m, gp2, gm2, m = sweep(gp, gm)
                    print(f"    pass {npass[0]:>2}: m = {m:+.4%}", flush=True)
                    xs_p, xs_m = [sp, gp, gp2], [sm, gm, gm2]
                else:
                    print(f"    WARNING: selfcal did NOT converge after "
                          f"{args.selfcal_cycles} Aitken cycles (last |F(x*)-x*|_max = "
                          f"{resid:.2e} >= tol {args.selfcal_tol:g}); reporting the last "
                          "iterate", flush=True)
                if not (np.isfinite(et_p).all() and np.isfinite(et_m).all()):
                    print("    WARNING: selfcal produced non-finite etilde values -- "
                          "the reported scal numbers are invalid (diverged iterate?)",
                          flush=True)
            dproj = gold_stats(f"scal|{tagc}", et_p, et_m, gh1, gh2, g, cases, args.n_boot)
            results[f"selfcal_{cond}"] = (et_p, et_m, dproj)

    # binned tables: one set per conditioning, on its headline prior (sheared first)
    for cond in args.prior_conditioning:
        head = next((f"{p}_{cond}" for p in ("sheared", "selfcal", "intrinsic")
                     if f"{p}_{cond}" in results), None)
        if head is None:
            continue
        et_p, et_m, dproj = results[head]
        print(f"\nBINNED (prior = {head}):")
        gold_binned("true r-mag r_input_p", MAG_EDGES,
                    df["r_input_p"].to_numpy(float), dproj, g, cases, args.n_boot)
        if "r_blend" in df.columns:
            gold_binned("emulator R_blend quantile", 4, df["r_blend"].to_numpy(float),
                        dproj, g, cases, args.n_boot, quantile=True, iso_eps=args.blend_eps)

    if args.dump:
        out = pd.DataFrame(dict(
            case=cases, input_index=df["input_index"].to_numpy(np.int64),
            r_input_p=df["r_input_p"].to_numpy(float),
            r_blend=df["r_blend"].to_numpy(float) if "r_blend" in df.columns else 0.0,
            ghat1=gh1, ghat2=gh2))
        for tag, (ep, em, _) in results.items():
            out[f"etilde1_plus_{tag}"] = ep[:, 0]
            out[f"etilde2_plus_{tag}"] = ep[:, 1]
            out[f"etilde1_minus_{tag}"] = em[:, 0]
            out[f"etilde2_minus_{tag}"] = em[:, 1]
        out.to_feather(args.dump)
        print(f"\nper-object dump -> {args.dump} ({len(out):,} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["closure", "g0", "gold"])
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue", default=None,
                    help=f"default: {os.path.basename(G0_CAT)} (closure/g0) or "
                         f"{os.path.basename(GOLD_CAT)} (gold)")
    ap.add_argument("--crowd-flux-lookup", default=CROWD_LOOKUP,
                    help="per-(case,input_index) nbr_flux_near/far/max feather (gold mode)")
    ap.add_argument("--blend-lookup", default="results/blend_lookup_extnbrho_c40-139.feather",
                    help="per-(case,input_index) R_blend feather -- BINNING ONLY, no additive term")
    ap.add_argument("--blend-eps", type=float, default=0.02)
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--grid-n", type=int, default=41)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--priors", nargs="+", default=["intrinsic", "sheared", "selfcal"],
                    choices=["intrinsic", "sheared", "selfcal"])
    ap.add_argument("--prior-conditioning", nargs="+", default=["global"],
                    choices=["global", "rmag"],
                    help="prior families to reweight under: the marginal pi(e) and/or "
                         "the magnitude-conditioned pi(e | r-mag bin) (MAG_EDGES bins; "
                         "targets the per-magnitude residual pattern found in cont.30)")
    ap.add_argument("--marginal-m", type=int, default=0,
                    help="if >0, marginalize the blend conditioning theta_b "
                         "(nbr_flux_near/far/max) over M per-galaxy draws from the "
                         "empirical detected-population prior instead of plugging in "
                         "true values (the cont.22 dtheta_b stage = deployment realism; "
                         "closure/gold modes). Costs M x the flow evaluations")
    ap.add_argument("--selfcal-mode", choices=["aitken", "plain"], default="aitken",
                    help="aitken: 3 sweeps + Delta^2 jump to the fixed point + confirming "
                         "sweep (exact for an affine map); plain: legacy iteration")
    ap.add_argument("--selfcal-iters", type=int, default=30,
                    help="plain mode only: number of fixed-point sweeps (contraction "
                         "1-K~0.81 needs ~30, and even then ~0.2%% truncation remains in m)")
    ap.add_argument("--selfcal-cycles", type=int, default=4,
                    help="aitken mode: max Delta^2 jump cycles (1 suffices when affine)")
    ap.add_argument("--selfcal-tol", type=float, default=2e-5,
                    help="aitken mode: fixed-point residual tolerance |F(x*)-x*| in gamma "
                         "units (2e-5 = 0.1%% of g=0.02)")
    ap.add_argument("--loglike-cache", default=None,
                    help="gold mode: path PREFIX to persist the fp16 likelihood grids + row "
                         "table (<prefix>_llp.npy/_llm.npy/_rows.feather/_meta.json; ~40 GiB "
                         "at 8M rows x 41x41 -- put it under $DATA_DIR). A later run with "
                         "matching model/grid/rows/seed skips catalogue streaming and ALL "
                         "flow evaluations: prior/selfcal experiments in minutes, not hours")
    ap.add_argument("--mu-correction", default=None,
                    help="npz from build_mu_correction.py: g0-measured location-vs-e "
                         "misfit U(e; r-mag cell) added to the likelihood location "
                         "(gold mode; population-mean-free so the residual flow's own "
                         "mean stays aligned)")
    ap.add_argument("--no-tf32", action="store_true",
                    help="disable TF32 matmuls (only affects A100+; TF32 error is far below "
                         "the fp16 loglike storage rounding)")
    ap.add_argument("--prior-sample", default="results/etilde_prior_e_samples.feather",
                    help="cached feather of detected+selected intrinsic (e1,e2); built if missing")
    ap.add_argument("--prior-catalogue", default=G0_CAT)
    ap.add_argument("--prior-batches", type=int, default=16)
    ap.add_argument("--prior-bins", type=int, default=60)
    ap.add_argument("--closure-g1", type=float, default=0.02)
    ap.add_argument("--closure-g2", type=float, default=0.0)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump", default=None)
    import inspect
    optics = {k: float(p.default)  # canonical values: rescale()'s own signature
              for k, p in inspect.signature(rescale).parameters.items()
              if p.default is not inspect.Parameter.empty}
    for k, v in optics.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)  # bundle.sample draws (closure) reproducible across runs
    if not args.no_tf32:
        torch.set_float32_matmul_precision("high")  # TF32 on A100+; no-op on v100/cpu
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={os.path.basename(args.measurement_model)}")
    bundle = load_measurement_model(args.measurement_model, device=device)
    grid, cell = make_e_grid(args.grid_n, args.grid_emax, args.grid_rmax)
    print(f"e-grid: {args.grid_n}x{args.grid_n} masked to |e|<={args.grid_rmax} -> G={len(grid)}")
    est = PosteriorShapeEstimator(bundle, grid, device=device)
    need_r = "rmag" in args.prior_conditioning
    prior, psample = get_prior(args, need_r=need_r)
    bpriors = binned_priors(psample, args) if need_r else None
    rk = {k: getattr(args, k) for k in optics}

    if args.mode == "closure":
        run_closure(args, bundle, est, prior, grid, rk)
    elif args.mode == "g0":
        run_g0(args, bundle, est, prior, grid, rk, bpriors=bpriors)
    else:
        run_gold(args, bundle, est, prior, grid, rk, bpriors=bpriors)


if __name__ == "__main__":
    main()
