#!/usr/bin/env python
"""How much of the uncut-control bias can the SCORE grid alone explain?  Flow-free.

cont.182 established that the Bartlett curvature residual -- the identity
`E_0[du] + Var_0(u) = 0`, which any normalised prior must satisfy -- fails at the production
grid by 3.6e-3 of `Var_0(u)`, and that this is a spurious information floor with the right
sign to push `m` negative.  Converted at face value it allows ~2%, comfortably more than the
measured -0.665%.  But that conversion is the FLAT-likelihood limit: it is the floor carried
by an object whose posterior is the prior itself.  Real posteriors are peaked, they weight a
different part of the node bank, and the quadrature error they see is a different number.
"Right sign and enough room" is therefore an upper bound, not a prediction.

This script turns it into a prediction, by running the closure test with the flow replaced by
the simplest measurement model that can stand in for it:

    true shape   e ~ p_gamma            (prior samples, Moebius-sheared by g)
    measurement  xhat = e + N(0, sigma^2 I)
    estimator    w_k  propto  p_0(e_k) N(xhat; e_k, sigma^2),  s = E_w[u],
                 I = -E_w[du] - Var_w(u)

That estimator is correctly specified for that model, so `ghat` would return `g` exactly on a
perfect grid, and whatever it returns instead IS the grid's error -- the same statement the
real uncut control makes, with the flow taken out. Nothing here is fitted to the measured
bias; the ONE free number, sigma, is pinned by matching the measured `<I>` (--target-info),
which is why the sweep prints `<I>` next to every sigma.

What this can and cannot do. It CAN say how strongly the answer depends on the node bank at a
realistic posterior width, which is the paired job's observable and is testable against it.
It CANNOT reproduce the absolute -0.665%: a Gaussian shape likelihood is not the flow, and
the flow's posterior is neither isotropic nor of uniform width. Treat a matching
`d(m)` between grids as support for the grid explanation, and a null as evidence against it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbsi.posterior_shape import make_e_grid            # noqa: E402
from sbsi.score_inference import ShapeScoreNodes        # noqa: E402
from sbsi.shear_map import apply_shear_to_ellipticity   # noqa: E402

from check_quadrature import _list                      # noqa: E402
from eval_score_response import G0_CAT, PRIOR_CACHE, build_prior  # noqa: E402


def score_and_info(nodes, xhat, sigma, chunk=256, block=None):
    """Per-BLOCK `s` and `I` for objects whose likelihood is N(xhat; e_node, sigma^2).

    Chunked over OBJECTS, not nodes: the weight block is (chunk, G) and the reference grid
    has G ~ 3e4, so a few hundred objects at a time keeps it in cache while still being one
    BLAS call per chunk.  Sums are kept per jackknife block because the number that matters
    is a DIFFERENCE between two grids evaluated on the same objects, and that difference has
    to carry an error bar -- computed the same paired way as diff_score_caches.py.
    """

    lp, u, du = nodes.log_prior, nodes.u, nodes.du
    grid = nodes.grid
    nb = 1 if block is None else int(block.max()) + 1
    s_tot = np.zeros((nb, 2))
    i_tot = np.zeros((nb, 2, 2))
    inv2s2 = 0.5 / sigma ** 2
    for a in range(0, len(xhat), chunk):
        x = xhat[a:a + chunk]
        d2 = ((x[:, None, 0] - grid[None, :, 0]) ** 2
              + (x[:, None, 1] - grid[None, :, 1]) ** 2)
        logw = lp[None, :] - inv2s2 * d2
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)          # a node off support has lp = -inf -> w = 0
        s = w @ u                                   # (m, 2)
        mean_du = np.einsum("mk,kab->mab", w, du)
        second = np.einsum("mk,ka,kb->mab", w, u, u)
        info = -(mean_du + second - np.einsum("ma,mb->mab", s, s))
        b = np.zeros(len(x), dtype=int) if block is None else block[a:a + chunk]
        np.add.at(s_tot, b, s)
        np.add.at(i_tot, b, info)
    return s_tot, i_tot


def jackknife_dm(s_a, i_a, s_b, i_b, g):
    """Jackknifed m_b - m_a over blocks, formed INSIDE each replicate so noise cancels."""

    nb = len(s_a)
    reps = np.array([(ghat(s_b[m].sum(0), i_b[m].sum(0))[0]
                      - ghat(s_a[m].sum(0), i_a[m].sum(0))[0]) / g
                     for m in (np.arange(nb) != b for b in range(nb))])
    return np.sqrt((nb - 1) / nb * ((reps - reps.mean()) ** 2).sum())


def ghat(s, i):
    return np.linalg.solve(i, s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--grid-ns", default="61:81:101:141")
    ap.add_argument("--reference-n", type=int, default=201)
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--info-delta", type=float, default=0.0025)
    ap.add_argument("--closure-g", type=float, default=0.05)
    ap.add_argument("--objects", type=int, default=20000)
    ap.add_argument("--sigma-sweep", default="0.15:0.20:0.25:0.30:0.40")
    ap.add_argument("--target-info", type=float, default=6.243,
                    help="the measured per-object <I> the toy's sigma is pinned by "
                         "(cont.181, cut 0.6)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--jk-blocks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--prior-sample", default=PRIOR_CACHE)
    ap.add_argument("--prior-catalogue", default=G0_CAT)
    ap.add_argument("--prior-rows", type=int, default=2_000_000)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=8)
    ap.add_argument("--prior-knot-margin", type=float, default=0.10)
    args = ap.parse_args()

    prior = build_prior(args)
    g = args.closure_g
    rng = np.random.default_rng(args.seed)

    # The SAME objects for every grid and every sigma: the comparison across grids is then
    # paired, so the Monte-Carlo noise on the difference is far below the noise on either
    # side.  (The measurement noise is redrawn per sigma because its scale IS sigma, but it
    # is drawn from a fixed unit-normal so the realisation is common across sigmas too.)
    e1, e2 = prior.sample(args.objects, rng)
    e1s, e2s = apply_shear_to_ellipticity(e1, e2, g * np.ones_like(e1), np.zeros_like(e1))
    unit_noise = rng.standard_normal((args.objects, 2))

    ns = [int(x) for x in _list(args.grid_ns)]
    banks = {}
    for n in ns + [args.reference_n]:
        grid, _ = make_e_grid(n=n, emax=args.grid_emax, rmax=args.grid_rmax)
        banks[n] = ShapeScoreNodes(grid, prior, delta=args.fd_delta,
                                   info_delta=args.info_delta)
        print(f"  bank grid_n={n:4d}  G={len(grid):,}", flush=True)

    print("\n" + "=" * 78)
    print(f"CLOSURE with a Gaussian shape likelihood, g={g}, {args.objects:,} objects.")
    print(f"`m` is the toy's own bias at that grid; d(m) is vs grid_n={args.reference_n}.")
    print("Pin sigma by <I>: the row whose <I> matches the measured "
          f"{args.target_info:.3f} is the")
    print("one to read.  Rows far from it describe a population that is not ours.")
    print("=" * 78)
    block = np.arange(args.objects) % args.jk_blocks
    for sigma in [float(x) for x in _list(args.sigma_sweep)]:
        xhat = np.stack([e1s, e2s], axis=1) + sigma * unit_noise
        blocks = {}
        for n in list(banks):
            blocks[n] = score_and_info(banks[n], xhat, sigma, args.chunk, block)
        def totals(n):
            s, i = blocks[n]
            return ghat(s.sum(0), i.sum(0))[0], i.sum(0)[0, 0] / args.objects
        ref_g, ref_i = totals(args.reference_n)
        mref = ref_g / g - 1
        print(f"\n  sigma={sigma:.3f}   <I>={ref_i:.3f}"
              f"   (reference m = {100*mref:+.3f}%)")
        print(f"    grid_n        <I>          m          d(m) vs reference")
        for n in ns:
            gn, iu = totals(n)
            dsig = jackknife_dm(*blocks[args.reference_n], *blocks[n], g)
            print(f"    {n:6d}   {iu:8.3f}   {100*(gn/g-1):+9.3f}%   "
                  f"{100*(gn/g - ref_g/g):+8.3f}% +/- {100*dsig:.3f}%")
        if len(ns) >= 2:
            lo, hi = ns[0], ns[1]
            d = (totals(hi)[0] - totals(lo)[0]) / g
            dsig = jackknife_dm(*blocks[lo], *blocks[hi], g)
            print(f"    => the paired job measures grid_n {lo} -> {hi}, "
                  f"predicted d(m) = {100*d:+.3f}% +/- {100*dsig:.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
