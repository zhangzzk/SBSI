#!/usr/bin/env python -B
"""§5B end to end WITH selection: the full (5.3), on the real flow, against a known shear.

WHAT THIS CLOSES.  `WORKLOG.md` cont.164 lists three defects in the §5B implementation.  This
script exercises the fix for the second and most consequential one: the estimator had no
`I_sel`, so it centred the score's numerator and left the denominator alone.  `INFERENCE.md`
A.7 prices that at `m = -64%` for a cut on the median, and `tests/test_population_terms.py`
now reproduces that number from the production code.  Here it runs on the real flow.

THE CUT.  It must be a function of the flow's OUTPUT, or `P_pass` does not exist (§4.7) -- the
reason a measured size/mag cut is unavailable on this V1-shaped model, whose only output is the
measured shape.  So the cut is `|xhat| < c`, which is not a workaround but the interesting
case: it is ISOTROPIC, and §5B.2 predicts that for a spin-2 shear an isotropic cut kills the
numerator term by orientation averaging and leaves `I_sel` as the only surviving selection
correction.  This run therefore tests the document's central claim about selection directly:

    <s>_sel  ~ 0     (both components, by isotropy)
    I_sel    ~ isotropic, and NOT small
    ghat with both corrections -> the injected g;  without I_sel -> biased low.

CLOSURE, SO THERE IS A TRUTH.  Data are drawn FROM the flow at a known shear, exactly as
`eval_score_response.py --mode closure` does, so the likelihood is exact and any departure of
`ghat` from `g` is the estimator's own.  The cut is then applied to those draws.

Pi_k, CHEAPLY.  The population block needs `Pi_k = P(xhat in S | true shape = e_k)` averaged
over the population of the OTHER true properties (`population_log_pi`'s docstring explains why
that average is needed at all -- this bank is per-galaxy in its conditioning).  Estimating it
costs one flow sample per (subsample row, node), not a nested integral: substitute the node's
shape into a subsample of real rows, sample, and count what passes.  With a few hundred rows
and a few samples each that is a few million draws.

Usage (Slurm, GPU):
    python scripts/eval_score_select.py --closure-g 0.02 --cut-abs-ehat 0.6
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.posterior_shape import PosteriorShapeEstimator, make_e_grid  # noqa: E402
from sbs_shear.preprocessing import rescale  # noqa: E402
from sbs_shear.score_inference import (  # noqa: E402
    ShapeScoreNodes, full_shear_estimate, population_terms,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

from eval_score_response import (  # noqa: E402
    G0_CAT, LL_DTYPE, PRIOR_CACHE, build_prior, load_g0, score_pass,
)

MODEL = "models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt"


def pass_fraction_by_node(bundle, df_sub, grid, rk, cut, n_samples, batch_size, seeds):
    """`Pi_k = P(xhat in S | true shape = e_k)`, averaged over the population.

    One flow sample per (row, node, draw).  The node's shape is substituted into real rows,
    so every other conditioning variable keeps its catalogue value and the average over rows
    IS the marginalisation over the rest of the scene prior.

    COMMON RANDOM NUMBERS ACROSS NODES, which is not a refinement but the difference between
    a usable `Pi` and an unusable one.  `<s>_sel = E_Pi[u]` is a ratio of two integrals that
    very nearly cancel -- for an isotropic cut it should vanish outright -- so INDEPENDENT
    Bernoulli noise at each node does not average away, it lands directly on the answer.
    Measured on a 16-draw pilot it produced a spurious `<s>_sel = 0.049` that removed two
    thirds of the estimator's numerator.  Re-seeding identically for every node makes `Pi_k`
    a smooth function of `k`: the latents are shared, so neighbouring nodes differ by the
    signal rather than by noise.

    ERROR BARS COME FROM INDEPENDENT REALISATIONS, not from splitting the rows.  Sharing
    latents across nodes is what makes `Pi` smooth, but it also means one realisation
    imprints a fixed pattern on the whole node bank -- and a split-half over ROWS cannot
    see it, because both halves carry the same latents.  Measured, that fake error bar
    reported a 34-sigma detection of a quantity that is zero by symmetry.  This is exactly
    the trap §5B.4 records for nested ladders, met again one level down, and the cure is
    the same: independent replicates.  `seeds` therefore drives `n_rep` full sweeps, and
    the scatter ACROSS them is the honest uncertainty.

    Returns `Pi (G, n_rep)`.
    """
    g = np.asarray(grid, float)
    out = np.empty((len(g), len(seeds)), dtype=np.float64)
    for k in range(len(g)):
        f = df_sub.copy()
        f["e1_input_rot0_p"] = g[k, 0]
        f["e2_input_rot0_p"] = g[k, 1]
        fr = rescale(f, **rk)
        for j, sd in enumerate(seeds):
            torch.manual_seed(sd)                  # SAME latents at every node -> CRN
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sd)
            xh = bundle.sample(fr, n_samples=n_samples, batch_size=batch_size)  # (m,ns,2)
            out[k, j] = cut(xh).mean()
        if k % 500 == 0:
            print(f"    Pi nodes {k:>6,}/{len(g):,}  <Pi>={out[:k+1].mean():.4f}",
                  flush=True)
    return out


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
    ap.add_argument("--max-rows", type=int, default=400_000)
    ap.add_argument("--closure-g", type=float, default=0.02)
    ap.add_argument("--cut-abs-ehat", type=float, default=0.6,
                    help="keep |xhat| < this.  Isotropic on purpose (see module docstring)")
    ap.add_argument("--pi-rows", type=int, default=256, help="subsample for Pi_k")
    ap.add_argument("--pi-samples", type=int, default=8)
    ap.add_argument("--uncut-control", action="store_true",
                    help="also run the estimator with NO cut, where the population terms "
                         "vanish identically.  Not optional for interpretation: the cut "
                         "result is only meaningful relative to what the SAME estimator "
                         "returns on the same rows without one.")
    ap.add_argument("--pi-reps", type=int, default=4,
                    help="independent Pi realisations; their scatter is the error bar")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--grad-chunk", type=int, default=256)
    ap.add_argument("--grad-delta", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--slab-mult", type=int, default=64)
    ap.add_argument("--ll-dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--info-delta", type=float, default=0.0025)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    for k, v in dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0,
                     psf_fwhm=0.73, moffat_beta=2.224).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={args.device}  torch={torch.__version__}", flush=True)

    prior = build_prior(args)
    grid, cell = make_e_grid(n=args.grid_n, emax=args.grid_emax, rmax=args.grid_rmax)
    nodes = ShapeScoreNodes(grid, prior, delta=args.fd_delta, info_delta=args.info_delta)
    print(f"grid: G={len(grid)} nodes, cell {cell:.3e}, "
          f"|u_fd-u_closed|/rms={nodes.closed_form_residual():.2e}", flush=True)

    bundle = load_measurement_model(args.measurement_model, device=args.device)
    df = load_g0(args.g0_catalogue, args.max_rows)
    print(f"rows: {len(df):,} from {os.path.basename(args.g0_catalogue)}", flush=True)

    # ---- data: draw true shapes from the prior, shear, push through the flow ----------
    rng = np.random.default_rng(args.seed)
    e1i, e2i = prior.sample(len(df), rng)
    g = float(args.closure_g)
    gh1, gh2 = np.ones(len(df)), np.zeros(len(df))
    e1l, e2l = apply_shear_to_ellipticity(e1i, e2i, g * gh1, g * gh2)
    fr = df.copy()
    fr["e1_input_rot0_p"], fr["e2_input_rot0_p"] = e1l, e2l
    fr = rescale(fr, **rk)
    torch.manual_seed(args.flow_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.flow_seed)
    ehat = bundle.sample(fr, n_samples=1, batch_size=args.batch_size)[:, 0, :]
    print(f"closure data at g={g:+.4f}: {len(ehat):,} measured shapes", flush=True)

    # ---- the cut, on a flow OUTPUT ---------------------------------------------------
    c = float(args.cut_abs_ehat)
    keep = np.hypot(ehat[:, 0], ehat[:, 1]) < c
    print(f"cut |xhat| < {c}: keeps {keep.sum():,}/{len(keep):,} = {keep.mean():.2%}",
          flush=True)
    fr_keep = fr.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    ehat_keep = ehat[keep]

    # ---- per-object block on the KEPT rows -------------------------------------------
    s, info = score_pass(est := PosteriorShapeEstimator(bundle, grid, device=args.device),
                         nodes, fr_keep, ehat_keep, args.chunk, "kept",
                         grad_chunk=args.grad_chunk, grad_delta=args.grad_delta,
                         slab_mult=args.slab_mult, ll_dtype=LL_DTYPE[args.ll_dtype])
    del est

    # ---- uncut control: the same estimator, same rows, no selection at all --------
    if args.uncut_control:
        s_u, info_u = score_pass(
            PosteriorShapeEstimator(bundle, grid, device=args.device), nodes, fr, ehat,
            args.chunk, "uncut", grad_chunk=args.grad_chunk, grad_delta=args.grad_delta,
            slab_mult=args.slab_mult, ll_dtype=LL_DTYPE[args.ll_dtype])
        gh_u, _, den_u = full_shear_estimate(s_u, info_u)
        err_u = 1.0 / np.sqrt(max(den_u[0, 0], 1e-30))
        print(f"\nUNCUT CONTROL (Pi == 1, so both population terms vanish by construction):")
        print(f"  ghat = [{gh_u[0]:+.6f} +/- {err_u:.6f}, {gh_u[1]:+.6f}]   "
              f"m = {gh_u[0]/g - 1:+.2%} +/- {err_u/abs(g):.2%}")
        print(f"  Any bias here is the BASELINE estimator's, not the cut's.  The cut rows")
        print(f"  below must be read against this number, not against zero.")

    # ---- population block ------------------------------------------------------------
    sub = df.iloc[:args.pi_rows].reset_index(drop=True)
    print(f"\nPi_k from {len(sub)} rows x {len(grid):,} nodes x {args.pi_samples} draws "
          f"x {args.pi_reps} reps = {len(sub)*len(grid)*args.pi_samples*args.pi_reps:,} draws", flush=True)
    seeds = [args.seed + 1000 * j for j in range(args.pi_reps)]
    pi_reps = pass_fraction_by_node(
        bundle, sub, grid, rk,
        cut=lambda x: np.hypot(x[:, :, 0], x[:, :, 1]) < c,
        n_samples=args.pi_samples, batch_size=args.batch_size, seeds=seeds)

    def terms(p):
        return population_terms(nodes, np.log(np.maximum(p, 1e-12)))

    per_rep = [terms(pi_reps[:, j]) for j in range(pi_reps.shape[1])]
    s_all = np.stack([t[0] for t in per_rep])                 # (n_rep, 2)
    i_all = np.stack([t[1] for t in per_rep])                 # (n_rep, 2, 2)
    pi = pi_reps.mean(axis=1)
    s_sel, i_sel = terms(pi)
    r = max(len(seeds), 2)
    es = s_all.std(axis=0, ddof=1) / np.sqrt(r)               # error on the mean
    ei = i_all.std(axis=0, ddof=1) / np.sqrt(r)
    print(f"  {len(seeds)} independent Pi realisations; scatter across them is the error")

    w = nodes.prior_weights()
    print(f"\nPi_k: min={pi.min():.4f} max={pi.max():.4f} "
          f"prior-weighted mean={float(w @ pi):.4f}")
    print(f"<s>_sel = [{s_sel[0]:+.6f} +/- {es[0]:.6f}, "
          f"{s_sel[1]:+.6f} +/- {es[1]:.6f}]")
    print(f"          isotropy (§5B.2) predicts BOTH are 0 for a spin-2 shear; "
          f"|s|/sigma = {abs(s_sel[0])/max(es[0],1e-12):.1f}, "
          f"{abs(s_sel[1])/max(es[1],1e-12):.1f}")
    print(f"I_sel   = [[{i_sel[0,0]:+.5f} +/- {ei[0,0]:.5f}, {i_sel[0,1]:+.5f}], "
          f"[{i_sel[1,0]:+.5f}, {i_sel[1,1]:+.5f} +/- {ei[1,1]:.5f}]]")
    off = 0.5 * (abs(i_sel[0, 1]) + abs(i_sel[1, 0]))
    dia = 0.5 * (i_sel[0, 0] + i_sel[1, 1])
    print(f"          isotropy of I_sel: off-diagonal / diagonal = {off/max(abs(dia),1e-12):.3f}"
          f"  (0 if the cut is exactly isotropic)")

    # ---- the estimator, three ways ---------------------------------------------------
    n = len(s)
    mean_i = info.sum(axis=0) / n
    print(f"\nper-object: N={n:,}  <I> = [[{mean_i[0,0]:+.5f}, {mean_i[0,1]:+.5f}], "
          f"[{mean_i[1,0]:+.5f}, {mean_i[1,1]:+.5f}]]")
    print(f"I_sel / <I> (component 11) = {i_sel[0,0]/mean_i[0,0]:+.4f}   "
          f"-- A.7's 2/pi analogue")

    rows = (("none      (sum s / sum I)", None, None),
            ("numerator only", s_sel, None),
            ("FULL (5.3)", s_sel, i_sel))
    print(f"\n  {'correction':<24} {'ghat_1':>10} {'ghat_2':>10} {'m = ghat/g - 1':>14}")
    res = {}
    for name, ss, ii in rows:
        gh, num, den = full_shear_estimate(s, info, ss, ii)
        res[name] = gh
        err = 1.0 / np.sqrt(max(den[0, 0], 1e-30))
        print(f"  {name:<24} {gh[0]:>10.6f} {gh[1]:>10.6f} {gh[0]/g - 1:>13.2%} "
              f"+/- {err/abs(g):.2%}")

    print(f"\n  truth g = {g:+.6f}")
    print("  §5B.2 predicts the numerator correction does nothing for an isotropic cut")
    print("  and that I_sel carries the whole effect.  Compare rows 1-2 (should agree)")
    print("  against row 3 (should move, and toward the truth).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
