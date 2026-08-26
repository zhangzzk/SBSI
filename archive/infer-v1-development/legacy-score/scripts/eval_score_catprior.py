#!/usr/bin/env python -B
# Archived Infer V1 development diagnostic.
"""Legacy §5B quadrature study with catalogue-drawn conditioning variables.

This script is retained for reproducibility of the finite-M study.  It is not
the current catalogue-prior closure runner; use ``run_catalogue_closure.py``.

THE QUESTION.  `eval_score_select.py --mode closure` integrates over the primary's true
shape and pins every other conditioning variable -- true magnitude, size, Sersic index and
the three neighbour fluxes -- at the scored row's own catalogue value.  That is exact for a
closure test and unavailable in deployment, where those are latent and `INFERENCE.md`
§5B.1(i) requires them marginalised over a scene prior.  This script replaces the pinned
conditioning by an empirical draw from the catalogue and asks whether the estimator still
returns the injected shear.

WHY BOTH LIMITS ARE EXACT, WHICH IS WHAT MAKES THIS A CLOSURE TEST.  Row `i` carries fixed
true properties `theta_i`; the true shape `e_i` is drawn from the estimation prior,
independently of the row; `xhat_i` comes from the flow at `(S_g e_i, theta_i)`.  So

  * conditioning on `theta_i` (the "pin" column) is exact -- model and data agree per row;
  * marginalising over the EMPIRICAL distribution of `theta` across the same rows is also
    exact, because averaging over which row we are looking at is precisely that
    distribution, and `e` was drawn independently of it.

Two different likelihood models, one truth.  Their difference is therefore not physics: it
is the finite-draw quadrature error, and it must go to zero as the draw count grows.  That
is the measurement.

WHAT IS EXPECTED TO GO WRONG, AND AT WHAT RATE.  `(1/M) sum_m p` is unbiased for the
marginal likelihood but the estimator consumes its log, so the leading error is
`-Var/(2 M L^2)` -- a BIAS falling as `1/M`, not noise that averages away over galaxies.
The ladder is nested, so consecutive points are paired and the DIFFERENCE is measured far
better than either point; the reported `d(m)` bars are jackknifed as differences, block by
block, which is the only way that pairing is honest.

TIERS.  `--features nbr` marginalises the three neighbour fluxes alone: the latent scene,
weakly constrained by the measurement, where the draw should work.  `--features all` adds
true magnitude, size and Sersic index, which the measured magnitude and size very nearly
pin -- so a blind catalogue draw puts almost every atom at negligible weight and the
effective sample size collapses.  `--beta` then switches on the bounded-weight mixture
proposal of `sbsi/catalogue_prior.py`, which is the fix for that and not for anything else.

Usage (Slurm, GPU):
    python scripts/eval_score_catprior.py --features nbr --ladder 1,2,4,8,16
    python scripts/eval_score_catprior.py --features all --beta 0.8 --cells 8
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbsi.catalogue_prior import (  # noqa: E402
    cell_index, draw_indices, marginal_score_pass, pool_half_indices,
    pool_shard_indices, quantile_edges,
    standardized_pool,
)
from sbsi.measurement_model import load_measurement_model  # noqa: E402
from sbsi.models import get_model  # noqa: E402
from sbsi.posterior_shape import PosteriorShapeEstimator, make_e_grid  # noqa: E402
from sbsi.precision import precision_region  # noqa: E402
from sbsi.preprocessing import rescale  # noqa: E402
from sbsi.score_inference import (  # noqa: E402
    ShapeScoreNodes, blocked_sums, jackknife_blocks, jackknife_sigma,
)
from sbsi.shear_map import apply_shear_to_ellipticity  # noqa: E402

from eval_score_response import G0_CAT, PRIOR_CACHE, build_prior, load_g0  # noqa: E402
from eval_score_select import primary_domain_cuts  # noqa: E402

MODEL = str(get_model("V3").flow_checkpoints[0])

# The flow's conditioning set, split by how tightly the MEASUREMENT constrains it.  This is
# the axis the two tiers are cut along, and the reason the second one needs a proposal.
FEATURE_SETS = {
    "nbr": ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"],
    "struct": ["sersic_n_input_p", "r_input_p", "Re_input_p"],
    "all": ["sersic_n_input_p", "r_input_p", "Re_input_p",
            "nbr_flux_near", "nbr_flux_far", "nbr_flux_max"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default=MODEL)
    ap.add_argument("--g0-catalogue", default=G0_CAT)
    ap.add_argument("--prior-sample", default=PRIOR_CACHE)
    ap.add_argument("--prior-catalogue", default=G0_CAT)
    ap.add_argument("--prior-rows", type=int, default=2_000_000)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=8)
    ap.add_argument("--prior-knot-margin", type=float, default=0.10)
    ap.add_argument("--grid-n", type=int, default=61)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--max-rows", type=int, default=120_000)
    ap.add_argument("--load-oversample", type=float, default=7.0)
    ap.add_argument("--closure-g", type=float, default=0.05)
    ap.add_argument("--closure-g2", type=float, default=0.0)
    ap.add_argument("--features", default="nbr",
                    help="one of " + "/".join(FEATURE_SETS) + ", or a comma list of "
                         "conditioning column names to marginalise")
    ap.add_argument("--ladder", default="1,2,4,8,16",
                    help="nested draw counts to report.  Nested, so the whole ladder "
                         "costs max(ladder) flow evaluations and consecutive points are "
                         "PAIRED -- which is what makes d(m) precise and what stops a "
                         "single point's bar from being called independent")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="mixture weight on the galaxy's own MEASURED cell (0 = uniform "
                         "catalogue draw).  Importance weights are bounded by 1/(1-beta)")
    ap.add_argument("--cells", type=int, default=8,
                    help="quantile bins per axis for the measured-space strata "
                         "(magnitude x log flux radius), so cells = this squared")
    ap.add_argument("--ring", choices=["none", "rot90"], default="rot90")
    ap.add_argument("--shape-reps", type=int, default=1)
    ap.add_argument("--jk-blocks", type=int, default=200)
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--info-delta", type=float, default=0.0025)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flow-seed", type=int, default=1234)
    ap.add_argument("--draw-seed", type=int, default=99,
                    help="seed for the CATALOGUE DRAWS alone.  Re-running with a different "
                         "value and the same --seed/--flow-seed gives an INDEPENDENT "
                         "quadrature realisation on identical data, which is the only "
                         "honest error bar on a ladder point (the nested bars are paired)")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--catalogue-shapes", action="store_true",
                    help="take the TRUE shapes from the catalogue instead of drawing them "
                         "from the estimation prior.  The likelihood stays exact (data "
                         "still come from the flow), so this isolates PRIOR "
                         "misspecification: the estimator uses the marginal pi(e) where "
                         "the correct prior is pi(e | true size, magnitude, neighbours), "
                         "and on a catalogue those correlate (rounder objects are small "
                         "and faint).  It also breaks the factorisation the default run "
                         "relies on, so d(m) vs pin still isolates the finite-draw term "
                         "while BOTH columns now carry the factorisation cost")
    ap.add_argument("--self-pool-check", type=int, default=2048,
                    help="rows for the exactness check run before the real pass: draw "
                         "every atom from the galaxy's OWN row, where the marginal "
                         "likelihood collapses to the pinned one identically.  0 skips it")
    ap.add_argument("--pool-half", choices=["none", "A", "B"], default="none",
                    help="restrict the DRAW POOL to one disjoint half of the catalogue "
                         "while scoring the SAME rows.  A and B are exact complements, so "
                         "running both and differencing asks whether the answer depends on "
                         "WHICH half of the catalogue supplies the prior -- if the pool is "
                         "a fair sample of the scene prior and large enough, d_A - d_B = 0. "
                         "Both arms are half-size, so neither is comparable to a full-pool "
                         "run; only A against B is.  This is the k = 2 special case of "
                         "--pool-shard/--pool-nshards and is kept because it names the two "
                         "arms; prefer the general form, which is cheaper per unit "
                         "information (doc/WORKLOG.md cont.189 §6c)")
    ap.add_argument("--pool-shard", type=int, default=-1,
                    help="restrict the DRAW POOL to shard i of --pool-nshards disjoint "
                         "shards, scoring the SAME rows.  Running all k shards gives k arm "
                         "values whose scatter has k-1 degrees of freedom -- from k jobs -- "
                         "where k = 2 halves give 1 degree of freedom from 2 jobs.  Since "
                         "the finite-catalogue term goes as 1/N_pool, a k-way arm also "
                         "carries k times as much of it, so the jackknife floor matters "
                         "less.  Both effects favour large k.  -1 disables")
    ap.add_argument("--pool-nshards", type=int, default=2,
                    help="number of disjoint pool shards k (used with --pool-shard).  Two "
                         "DIFFERENT k values are what test the a ~ 1/N_pool extrapolation "
                         "that a single k has to assume")
    ap.add_argument("--pool-split-seed", type=int, default=4242,
                    help="seed for the A/B split.  Deliberately INDEPENDENT of --draw-seed "
                         "so the two arms share their draw randomness and the difference "
                         "is paired; changing --draw-seed must not reshuffle the halves")
    ap.add_argument("--save", default=None, help="write per-key block sums + summary here")
    for k, v in dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0,
                     psf_fwhm=0.73, moffat_beta=2.224).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    feats = FEATURE_SETS.get(args.features) or [s for s in args.features.split(",") if s]
    ladder = sorted({int(v) for v in args.ladder.split(",") if v})
    gvec = np.array([float(args.closure_g), float(args.closure_g2)])
    gn = float(np.hypot(*gvec))
    if gn == 0:
        raise SystemExit("--closure-g/--closure-g2 must not both be zero")
    gdir = gvec / gn

    print(f"device={args.device}  torch={torch.__version__}", flush=True)
    bundle = load_measurement_model(args.measurement_model, device=args.device)
    tnames = list(bundle.target_transform.target_names)
    print(f"model: {os.path.basename(args.measurement_model)}")
    print(f"  outputs    {tnames}")
    print(f"  conditions {list(bundle.condition_preprocessor.feature_names)}")
    cuts, dom, domain_tag = primary_domain_cuts(bundle.metadata)
    print(f"  domain     primary true mag < {dom[0]}, Re > {dom[1]}  "
          f"(prior cache tag {domain_tag or 'default'})")
    print(f"MARGINALISE  {feats}   ladder {ladder}   beta={args.beta}", flush=True)

    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size,
              zero_mag=args.zero_mag, psf_fwhm=args.psf_fwhm,
              moffat_beta=args.moffat_beta)
    prior = build_prior(args, cuts=cuts, tag=domain_tag,
                        oversample=args.load_oversample)
    grid, cell_area = make_e_grid(n=args.grid_n, emax=args.grid_emax, rmax=args.grid_rmax)
    nodes = ShapeScoreNodes(grid, prior, delta=args.fd_delta, info_delta=args.info_delta)
    print(f"grid: G={len(grid)} nodes, cell {cell_area:.3e}, "
          f"|u_fd-u_closed|/rms={nodes.closed_form_residual():.2e}", flush=True)

    df = load_g0(args.g0_catalogue, args.max_rows, cuts=cuts,
                 oversample=args.load_oversample)
    print(f"rows: {len(df):,} from {os.path.basename(args.g0_catalogue)}", flush=True)

    est = PosteriorShapeEstimator(bundle, grid, device=args.device)
    rng = np.random.default_rng(args.seed)
    drng = np.random.default_rng(args.draw_seed)

    # The measured columns the strata are built on: the flow's own magnitude and log-radius
    # outputs.  Both galaxy and pool cells come from the SAME xhat within a leg, so the
    # proposal and the pool speak one language and no cross-catalogue matching is needed.
    try:
        c_mag = tnames.index("measured_mag_auto")
        c_rad = tnames.index("measured_log_flux_radius")
    except ValueError:
        c_mag = c_rad = None
    if args.beta > 0 and c_mag is None:
        raise SystemExit("--beta needs a flow predicting measured_mag_auto and "
                         "measured_log_flux_radius to define the measured-space strata")

    def shear_and_sample(e1, e2, seed):
        f = df.iloc[:len(e1)].copy()
        f["e1_input_rot0_p"], f["e2_input_rot0_p"] = apply_shear_to_ellipticity(
            e1, e2, gvec[0] * np.ones(len(e1)), gvec[1] * np.ones(len(e1)))
        f = rescale(f, **rk)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return f, bundle.sample(f, n_samples=1, batch_size=args.batch_size)[:, 0, :]

    cat_e1 = df["e1_input_rot0_p"].to_numpy(float)
    cat_e2 = df["e2_input_rot0_p"].to_numpy(float)
    if args.catalogue_shapes:
        print("TRUE shapes: from the CATALOGUE -- the shape prior is now deliberately "
              "misspecified.\n  |e| pairs with the row's own size and magnitude, and "
              "rot90 preserves |e|, so the ring\n  pairing leaves that correlation "
              "intact.", flush=True)

    def draw_shapes(n_rows, r):
        if args.catalogue_shapes:
            return cat_e1[:n_rows].copy(), cat_e2[:n_rows].copy()
        return prior.sample(n_rows, rng)

    def legs():
        for r in range(args.shape_reps):
            e1i, e2i = draw_shapes(len(df), r)
            base = args.flow_seed + 10 * r
            yield shear_and_sample(e1i, e2i, base) + (np.arange(len(df)),)
            if args.ring == "rot90":
                yield shear_and_sample(-e1i, -e2i, base + 1) + (np.arange(len(df)),)

    n_legs = args.shape_reps * (2 if args.ring == "rot90" else 1)
    keys = ["pin"] + ladder
    acc = {k: [[], [], []] for k in keys}
    ess_all, ess_rung_all = [], []
    # A PAIRED COMPARISON THAT SILENTLY STOPPED PAIRING LOOKS EXACTLY LIKE A NULL.  Two runs
    # that differ only in `grid_n` are supposed to see the SAME mock measurements, so that
    # differencing them cancels the galaxy noise.  If anything in one arm consumes the CUDA
    # RNG differently -- a different card, a different chunking path, an allocator retry --
    # the arms draw different `xhat`, the difference degrades to two independent runs, and
    # it comes back consistent with zero at a bar wide enough to hide the whole effect.
    # So record a bitwise digest of every leg's `xhat` and make the pairing CHECKABLE rather
    # than assumed: two runs whose digests match saw identical data, full stop.
    ehat_digest = []
    print(f"\n{n_legs} leg(s) x {len(df):,} rows -> {n_legs*len(df):,} objects; "
          f"{max(ladder)+1} flow passes per (row, node)", flush=True)

    t0 = time.time()
    with precision_region(args.precision, args.device):
        if args.self_pool_check:
            # EXACTNESS, END TO END, ON THE REAL FLOW.  Point every draw at the galaxy's own
            # row and the marginalisation degenerates: logsumexp of M copies of one term,
            # minus log M, is that term.  So `s` and `I` must come back identical to the
            # pinned column -- not close, identical to float round-off.  What this actually
            # certifies is the join between `standardized_pool` and `_grid_tiled_context`:
            # if the feature indices, the column order, or anything `rescale()` does had
            # drifted between the pool and the conditioning it overwrites, the substituted
            # values would no longer be the row's own and this would break.  A silent
            # mismatch there would look exactly like a physical result.
            k = min(int(args.self_pool_check), len(df))
            e1c, e2c = (cat_e1[:k].copy(), cat_e2[:k].copy()) if args.catalogue_shapes \
                else prior.sample(k, np.random.default_rng(args.seed + 555))
            frc, ehc = shear_and_sample(e1c, e2c, args.flow_seed + 5000)
            pc, fic = standardized_pool(est, frc, feats)
            self_idx = np.repeat(np.arange(k)[:, None], 4, axis=1)
            # THE WEIGHTS MUST BE EXERCISED, NOT STUBBED OUT.  This check used to pass
            # `np.zeros(...)` as `log_w`, which certifies the marginal path only where every
            # weight is one -- so the whole `--beta > 0` plumbing (the log_w tensor, the
            # `lw_run` normaliser, the `acc - lw_run` snapshot) went untested at run time.
            # When every atom is the SAME row the self-normalised mixture collapses to that
            # row's likelihood WHATEVER the weights are, because the normaliser divides them
            # straight back out.  So the check stays exact while now covering the weighted
            # path end to end.  Two weight sets, because they fail differently:
            #   `varied`  -- ordinary spread over the bounded-mixture range;
            #   `extreme` -- one atom carrying essentially all the weight, which is what
            #                catches a normaliser applied twice or dropped.
            wrng = np.random.default_rng(args.seed + 777)
            wmax = np.log(1.0 / (1.0 - args.beta)) if 0.0 < args.beta < 1.0 else 3.0
            varied = (wrng.uniform(-wmax, wmax, size=(k, 4))).astype(np.float32)
            extreme = np.zeros((k, 4), dtype=np.float32)
            extreme[:, 0] = 30.0
            resc = None
            for wname, wmat in (("unit", np.zeros((k, 4), dtype=np.float32)),
                                ("varied", varied), ("extreme", extreme)):
                rw, _, _ = marginal_score_pass(
                    est, nodes, frc, ehc, pc, fic, self_idx, wmat, [1, 4],
                    chunk=args.chunk, tag=f"self-pool/{wname}", progress_every=0)
                if resc is None:
                    resc = rw
                else:
                    dsw = float(np.abs(rw[4][0] - resc[4][0]).max())
                    print(f"  self-pool weight variant '{wname}': max|ds| vs unit weights "
                          f"= {dsw:.3e}", flush=True)
                    if not np.isfinite(dsw) or dsw > 1e-3:
                        raise SystemExit(
                            f"self-pool WEIGHTED check FAILED for '{wname}': with every "
                            f"draw the row itself the weights must normalise away, so this "
                            f"must reproduce the unit-weight answer.  The log_w plumbing "
                            f"or the acc/lw_run normaliser is wrong; no --beta result from "
                            f"this run is meaningful.")
            # TOLERANCES, DERIVED RATHER THAN DIALLED.  At M=1 the arithmetic is the same
            # operation in the same order, so the agreement is BIT EXACT and the tolerance
            # is only there to keep the comparison from being brittle.  At M>1 the marginal
            # path costs M logaddexps and a -log M that the pinned path never performs, so
            # float32 round-off enters the score at the 1e-6 level -- and the information is
            # a stencil of the score divided by `2 * info_delta`, which AMPLIFIES that by
            # 1/(2*0.0025) = 200x.  So the information tolerance is the score tolerance
            # propagated through the stencil, not an independent guess.  A genuinely wrong
            # column would move `I` by order unity and is nowhere near either bound.
            s_tol = 1e-5 * max(float(np.abs(resc["pin"][0]).mean()), 1e-3)
            i_tol = 10.0 * s_tol / (2.0 * args.info_delta)
            for m in (1, 4):
                ds = float(np.abs(resc[m][0] - resc["pin"][0]).max())
                di = float(np.abs(resc[m][1] - resc["pin"][1]).max())
                scale = float(np.abs(resc["pin"][0]).mean())
                print(f"  self-pool check M={m} on {k:,} rows: max|ds| = {ds:.3e} "
                      f"(tol {s_tol:.1e}), max|dI| = {di:.3e} (tol {i_tol:.1e})  "
                      f"<|s|> = {scale:.3f}", flush=True)
                if ds > s_tol or di > i_tol:
                    raise SystemExit(
                        f"self-pool check FAILED at M={m}: the marginal path does not "
                        f"reproduce the pinned likelihood when every draw is the row "
                        f"itself.  The pool and the conditioning it overwrites have "
                        f"drifted apart; no result from this run is meaningful.")
            del frc, ehc, resc
        for li, (fr, ehat, pair) in enumerate(legs()):
            pool_std, feat_idx = standardized_pool(est, fr, feats)
            cell_row = cell_pool = None
            n_cells = 0
            if args.beta > 0:
                e1 = quantile_edges(ehat[:, c_mag], args.cells)
                e2 = quantile_edges(ehat[:, c_rad], args.cells)
                n_cells = (len(e1) - 1) * (len(e2) - 1)
                cell_row = cell_index(ehat[:, c_mag], ehat[:, c_rad], e1, e2)
                cell_pool = cell_row            # pool IS the scored rows (see module doc)
            # POOL HALVES.  The pool IS the scored rows, so restricting it to half the
            # catalogue leaves the galaxies being scored completely untouched -- only the
            # atoms drawn to marginalise over change.  The split is a permutation under its
            # OWN seed so that A and B are exact complements and stay put when --draw-seed
            # moves; that is what makes d_A - d_B a paired difference rather than two
            # independent runs.  Note the pin arm never reads the pool at all, so `pin` must
            # come out bit-identical between the halves: that is the validity check.
            # --pool-half is the k = 2 alias; both spellings funnel through the same
            # subsetting so there is exactly one place the pool can be restricted.
            take, tag = None, ""
            if args.pool_shard >= 0:
                take = pool_shard_indices(len(pool_std), args.pool_shard,
                                          args.pool_nshards, args.pool_split_seed)
                tag = f"shard {args.pool_shard} of {args.pool_nshards}"
            elif args.pool_half != "none":
                take = pool_half_indices(len(pool_std), args.pool_half,
                                         args.pool_split_seed)
                tag = f"half {args.pool_half}"
            if take is not None:
                pool_std = pool_std[take]
                if cell_pool is not None:
                    # cells of the POOL rows, which are no longer the cells of the galaxy
                    # rows -- indexing makes a copy, so `cell_row` is left intact
                    cell_pool = cell_row[take]
                if li == 0:
                    print(f"pool {tag}: {len(pool_std):,} of "
                          f"{len(fr):,} rows drawable (split seed "
                          f"{args.pool_split_seed})", flush=True)
            idx, logw = draw_indices(len(fr), max(ladder), len(pool_std), drng,
                                     cell_of_row=cell_row, cell_of_pool=cell_pool,
                                     n_cells=n_cells, beta=args.beta)
            res, ess, ess_r = marginal_score_pass(
                est, nodes, fr, ehat, pool_std, feat_idx, idx, logw, ladder,
                chunk=args.chunk, tag=f"leg {li+1}/{n_legs}")
            ess_all.append(ess)
            ess_rung_all.append(ess_r)
            ehat_digest.append(hashlib.sha1(
                np.ascontiguousarray(ehat, dtype=np.float32).tobytes()).hexdigest()[:16])
            blk = pair % args.jk_blocks
            for k in keys:
                s, i = res[k]
                for dst, v in zip(acc[k], (s, i, blk)):
                    dst.append(v)
            del fr, ehat, res
    print(f"score pass done in {time.time()-t0:.0f}s", flush=True)
    print(f"xhat digest per leg: {' '.join(ehat_digest)}\n  ^ two runs differing only in "
          f"grid_n MUST show identical digests, or their difference is not paired and the "
          f"error bar on it is meaningless.", flush=True)

    ess_rung = np.concatenate(ess_rung_all, axis=0)
    ess = np.concatenate(ess_all)
    print(f"\nimportance-weight ESS at M={max(ladder)}: mean {ess.mean():.2f}  "
          f"median {np.median(ess):.2f}  p10 {np.percentile(ess,10):.2f}  "
          f"p90 {np.percentile(ess,90):.2f}   (out of {max(ladder)}; "
          f"efficiency {ess.mean()/max(ladder):.1%})")
    print("  ESS is for the EVIDENCE integral -- how many of the drawn scenes actually "
          "carry\n  weight for this galaxy.  Efficiency far below 100% is the measured "
          "magnitude and\n  size pinning the true ones, and it is what --beta exists to fix.")

    # PER-ROW INFORMATION, KEPT SO THE ROW-WEIGHTING APPROXIMATION CAN BE BOUNDED.
    # The finite-draw abscissa is a mean of reciprocals over rows, and the exact version is
    # weighted by each row's information rather than flat:
    #     <1/ESS>_I  =  <1/ESS>  +  Cov(I_i, 1/ESS_i) / <I>
    # so the residual error is ONE COVARIANCE, computable per rung -- but only if I_i
    # survives to the file.  It did not: `ns`/`ni` are per-BLOCK sums.  A block-level proxy
    # is worthless here and would have been actively misleading, because blocks are assigned
    # `pair % jk_blocks` (interleaved), so every block has near-identical composition by
    # construction and the between-block covariance is ~0 whatever the row-level truth is.
    # Expect the covariance to be POSITIVE: rows whose measured mag and size pin their true
    # values tightly are both high-information and low-ESS, so a flat mean UNDERSTATES the
    # abscissa, in the same direction as the Jensen gap and on top of it.
    i_row = {k: np.concatenate(acc[k][1]) for k in keys}
    blk = {k: blocked_sums(np.concatenate(acc[k][0]), i_row[k],
                           np.concatenate(acc[k][2]), args.jk_blocks) for k in keys}
    full, reps = {}, {}
    for k in keys:
        f, _, r = jackknife_blocks(*blk[k])
        full[k], reps[k] = f, r

    def m_of(k):
        return float(full[k] @ gdir) / gn - 1.0

    def sig_of(k):
        return float(jackknife_sigma((reps[k] @ gdir)[:, None])[0]) / gn

    def dm(k, ref):
        d = (reps[k] - reps[ref]) @ gdir
        v = (float(full[k] @ gdir) - float(full[ref] @ gdir)) / gn
        return v, float(jackknife_sigma(d[:, None])[0]) / gn

    def rho_of(k, ref):
        """Correlation between two arms, from the three bars they already carry.

        The pairing GAIN is a soft diagnostic because its "good" value depends on how
        correlated the two arms have any right to be.  rho is hard.  Two INDEPENDENT arms
        of equal width give gain 1/sqrt(2) = 0.707x at rho = 0; gain 1.0x is rho = 0.5.
        So the alarm condition is not "gain near 1x", it is rho < 0 -- the arms
        ANTI-correlating on identical data, for which there is no benign explanation.
        """
        sa, sb = sig_of(ref), sig_of(k)
        sd = dm(k, ref)[1]
        den = 2.0 * sa * sb
        return (sa ** 2 + sb ** 2 - sd ** 2) / den if den > 0 else float("nan")

    n_obj = int(blk["pin"][0].sum())
    print(f"\nN = {n_obj:,} objects, {args.jk_blocks} jackknife blocks, "
          f"truth g = [{gvec[0]:+.4f}, {gvec[1]:+.4f}]")
    print(f"{'model':>10} {'<I>':>9} {'ghat_1':>10} {'ghat_2':>10} "
          f"{'m':>10} {'+/-':>8}  {'d(m) vs pin':>12} {'+/-':>8} {'nsig':>6} {'pair':>8}"
          f" {'rho':>6}")
    summary = []
    for k in keys:
        mi = float(gdir @ (blk[k][2].sum(axis=0) / n_obj) @ gdir)
        line = (f"{str(k):>10} {mi:>9.4f} {full[k][0]:>10.6f} {full[k][1]:>10.6f} "
                f"{m_of(k):>10.3%} {sig_of(k):>8.3%}")
        row = dict(key=str(k), info=mi, m=m_of(k), sigma=sig_of(k))
        if k != "pin":
            d, sd = dm(k, "pin")
            # PAIRING GAIN: the single-arm bar divided into the paired one.  This IS the
            # shared galaxy noise cancelling, so it is also the cheapest detector for a
            # pairing that quietly stopped working -- a gain near 1x means the two columns
            # share nothing and the tight bar below is fiction.
            gain = sig_of(k) / max(sd, 1e-12)
            rho = rho_of(k, "pin")
            line += (f"  {d:>12.3%} {sd:>8.3%} {abs(d)/max(sd,1e-12):>6.1f}"
                     f" {gain:>7.1f}x {rho:>6.2f}")
            row.update(dm_pin=d, dm_pin_sigma=sd, pair_gain=gain, rho_pin=rho)
        summary.append(row)
        print(line, flush=True)

    print("\n  d(m) vs pin is the whole test: the pinned and the marginalised likelihood "
          "are\n  BOTH exact for these data, so their difference is pure finite-draw "
          "quadrature error\n  and must fall to zero.  It is jackknifed AS a difference -- "
          "same blocks, same rows,\n  same xhat.")
    print("  READ rho, NOT THE BAR.  Unlike a same-estimator/different-bank comparison "
          "(which\n  gains 48x-273x, cont.185/186), this one gains ~1x -- CORRECT rather "
          "than broken: the\n  ring pairing already removed the intrinsic shape noise "
          "inside each arm, so the\n  difference has little left to cancel, and the two "
          "likelihoods are genuinely different\n  functions of the same xhat.  d(m) here "
          "therefore costs raw sample size; it is NOT\n  bought cheaply by the pairing.")
    print("  THE ALARM CONDITION IS rho < 0, NOT a small gain.  Two INDEPENDENT arms of "
          "equal\n  width already give gain 1/sqrt(2) = 0.71x; gain 1.0x is rho = 0.5.  So "
          "0.9x is nowhere\n  near a boundary -- it says the arms share about a third of "
          "their fluctuation, which is\n  what pinned vs marginalised on shared xhat should "
          "do.  rho < 0 would mean the arms\n  ANTI-correlate on identical data, and there "
          "is no benign story for that.")
    print("  rho is necessary but NOT sufficient: it cannot tell 'these estimators differ' "
          "from\n  'these arms saw different data'.  Only the xhat digest above does that, "
          "and rho is\n  uninterpretable at any value if the digests disagree.")

    if len(ladder) >= 2:
        # SECOND, TIGHTER OBSERVABLE.  The ladder is NESTED -- M=2 reuses M=1's atom and
        # adds one -- so consecutive rungs share most of their draws and this comparison
        # DOES pair, unlike the pin comparison above.  Under d(M) = d_inf + a/M the rung
        # differences vs the deepest rung are a(1/M - 1/Mmax), which contains no d_inf at
        # all: it measures the SLOPE alone, at far better precision than the intercept.
        # That is the mechanism test.  If these are consistent with a single `a` the 1/M
        # law is confirmed and the extrapolation below is trustworthy; if they are not,
        # the fit is being applied to something that is not a 1/M convergence and the
        # intercept means nothing regardless of how good its chi2 looks.
        top = ladder[-1]
        print(f"\n  nested-ladder slope check -- each rung vs the deepest (M={top}), which "
              f"IS paired\n  (shared atoms), so these bars are the tight ones.  Expect "
              f"a*(1/M - 1/{top}).")
        print(f"{'M':>10} {'d(m) vs Mmax':>14} {'+/-':>9} {'gain':>7} {'rho':>6} "
              f"{'implied a':>11}")
        for k in ladder[:-1]:
            d, sd = dm(k, top)
            gain = sig_of(k) / max(sd, 1e-12)
            a_imp = d / (1.0 / k - 1.0 / top)
            print(f"{k:>10} {d:>14.3%} {sd:>9.3%} {gain:>6.1f}x "
                  f"{rho_of(k, top):>6.2f} {a_imp:>11.3%}", flush=True)
            summary.append(dict(key=f"{k}_vs_{top}", dm_top=d, dm_top_sigma=sd,
                                pair_gain_top=gain, rho_top=rho_of(k, top),
                                a_implied=a_imp))
        print("  HERE a gain near 1x IS the warning: these rungs are the SAME estimator "
              "family on\n  nested draws, so rho should be close to 1 and the gain well "
              "above 1x.  If it is not,\n  the draw indices are not prefix-shared across "
              "rungs, the nesting is broken, and the\n  ladder -- not just this table -- "
              "is invalid.")

    if len(ladder) >= 3:
        # d(M) = d_inf + a/M.  The 1/M law is the leading term of the log-of-an-unbiased-
        # estimate bias, so the fit is a PREDICTION being tested, not a curve chosen to fit:
        # a good chi2 supports the diagnosis, and d_inf consistent with zero is closure.
        x = np.array([1.0 / m for m in ladder])
        y = np.array([dm(m, "pin")[0] for m in ladder])
        e = np.array([dm(m, "pin")[1] for m in ladder])
        w = 1.0 / np.maximum(e, 1e-12) ** 2
        A = np.stack([np.ones_like(x), x], axis=1)
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        beta_hat = cov @ (A.T @ (w * y))
        chi2 = float(np.sum(w * (y - A @ beta_hat) ** 2))
        print(f"\n  fit d(m) = d_inf + a/M over M = {ladder}:")
        print(f"    d_inf = {beta_hat[0]:+.3%} +/- {np.sqrt(cov[0,0]):.3%}   "
              f"a = {beta_hat[1]:+.3%} +/- {np.sqrt(cov[1,1]):.3%}   "
              f"chi2/dof = {chi2/max(len(ladder)-2,1):.2f}")
        print(f"    -> draws needed for |bias| < 0.1%: M ~ "
              f"{abs(beta_hat[1])/0.001:.0f}" if abs(beta_hat[1]) > 0 else "")
        summary.append(dict(key="fit", d_inf=float(beta_hat[0]),
                            d_inf_sigma=float(np.sqrt(cov[0, 0])),
                            a=float(beta_hat[1]), a_sigma=float(np.sqrt(cov[1, 1])),
                            chi2=chi2, dof=len(ladder) - 2))

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        np.savez(args.save,
                 summary=json.dumps(summary),
                 config=json.dumps({k: (v if isinstance(v, (int, float, str, bool))
                                        else str(v)) for k, v in vars(args).items()}),
                 features=np.array(feats), ladder=np.array(ladder), ess=ess,
                 ess_rung=ess_rung,
                 ehat_digest=np.array(ehat_digest),
                 **{f"{k}_{name}": arr for k in keys
                    for name, arr in zip(("cnt", "ns", "ni"), blk[k])},
                 **{f"{k}_i_row": i_row[k].astype(np.float32) for k in keys})
        print(f"\nsaved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
