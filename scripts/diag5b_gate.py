#!/usr/bin/env python -B
"""Does §5B's integrand carry the pathology that killed §5C?  Run this BEFORE trusting (5.3).

§5C died on variance non-existence: its integrand `phi' = d_gamma log p_flow` has a Hill tail
index near 1.3 (< 2 => no finite second moment), and its denominator GREW with bank size
(fitted exponent +0.20 against -1 for honest Monte Carlo) while a tame integrand on the
identical weights averaged down.  The weights were fine; the integrand was not (WORKLOG
cont.170).  §5B divides by the same kind of object -- `I_i` contains `Var_{w_i}(u)` -- so the
question transfers even though the integrand does not.

WHAT TRANSFERS, AND WHAT DOES NOT.  §5C's node bank is a Monte Carlo SAMPLE from the scene
prior, so "does the answer settle as the bank grows" is a statement about sampling noise, and
disjoint equal blocks measure it.  §5B's node bank is a deterministic GRID with flat
quadrature (`make_e_grid`), so there is no bank realisation scatter to measure at all: two
runs at the same settings agree bit for bit.  The failure mode is not noise, it is
NON-CONVERGENCE, and the honest analogue of the bank ladder is a pair of grid ladders --
refinement at fixed reach, and reach at fixed refinement.  Reporting a disjoint-block ladder
here would be theatre; the reach ladder is the one that can actually fail.

The reach ladder matters for a specific reason.  `make_e_grid` defaults to `rmax = 0.95` while
the population tops out near 0.90, so the default node bank never visits the edge of the
ellipticity disc.  If the information secretly lives at the edge, that truncation is doing
hidden work and every §5B number is a function of an arbitrary cutoff.

THE ANALYTIC CONDITION, CORRECTED.  INFERENCE.md (5.3d) used to argue that with
`p_0 ~ (1-|eps|^2)^a` the generator behaves as `u ~ a/(1-|eps|^2)`, so that the information
exists only for `a > 1`.  That is wrong, and wrong in an instructive way: it assumes the
shear velocity is O(1) at the edge.  It is not.  The Mobius map preserves the unit disc, so
the velocity field is TANGENT to the boundary and its normal component vanishes like
`1 - |eps|^2` -- exactly cancelling the divergence of `grad log p_0`.  In the closed form
carried by `score_inference.py`,

    u_a = e_a [ 4 - 2 psi'(t) (1 - t) ],     t = |eps|^2,   psi(t) = log p_0,

the prior enters only through the product `(1-t) psi'(t)`.  For the power law
`psi = a log(1-t)` that product is the constant `-a`, giving `u_a = e_a (4 + 2a)`: bounded for
every `a`, with `E_0[u^2] = 4(a+2)` finite even at `a = 0`, a prior that does not vanish at
the edge at all.  Verified against the exact Mobius pullback to six digits at edge distances
down to 1e-7.  The real condition is therefore

    E_0[u^2] < inf   <=>   (1-t) psi'(t)  is square-integrable against p_0,

i.e. the log-density's slope may not blow up FASTER than 1/(1-t).  The whole power-law family
passes.  `SmoothRadialPrior` continues `psi` linearly in `t` past the last populated bin, so
its `psi'` is asymptotically constant and `(1-t) psi' -> 0`: safe with room to spare.  This leg
checks that on the prior actually fitted, rather than on the family it belongs to.

OUTCOME MAP (fixed in advance, so this cannot be read after the fact):
  u bounded at the edge, Hill >> 2, Var_0(u) flat in BOTH ladders
        -> §5C's variance non-existence does not arise in §5B's shape channel.  The gate
           passes and the open question moves to the OTHER channels (size, flux, separation),
           whose geometry is different and is NOT covered by this test.
  Var_0(u) climbs with rmax and does not settle
        -> the information lives at the edge, the 0.95 truncation is load-bearing, and every
           §5B number to date is a function of an arbitrary cutoff.
  Hill < 2 on prior samples
        -> §5B has §5C's disease after all, from a prior we wrote down, and the cure is to
           choose a better-behaved prior rather than to retrain anything.

Usage (negligible; login node is fine -- 11 MB cache + numpy grids):
    python scripts/diag5b_gate.py
"""

import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyarrow.feather as pf  # noqa: E402

from sbs_shear.posterior_shape import RadialShapePrior, make_e_grid  # noqa: E402
from sbs_shear.score_inference import (  # noqa: E402
    ShapeScoreNodes, SmoothRadialPrior, generator_closed_form,
)

PRIOR_CACHE = os.path.join(SBSI_ROOT, "results", "etilde_prior_e_samples.feather")


class PowerLawDisc:
    """`p_0 propto (1-|eps|^2)^a`; the analytic control for the edge leg."""

    def __init__(self, a):
        self.a = float(a)

    def log_prob(self, e1, e2):
        t = np.asarray(e1, float) ** 2 + np.asarray(e2, float) ** 2
        return np.where(t < 1.0, self.a * np.log1p(-np.minimum(t, 1.0 - 1e-16)), -np.inf)

    sheared_log_prob = RadialShapePrior.sheared_log_prob      # only touches self.log_prob


def hill(x, frac=0.01):
    """Hill tail index of |x| on its top `frac`.  alpha < 2 => no finite second moment."""
    a = np.sort(np.abs(np.asarray(x, float).ravel()))
    a = a[np.isfinite(a) & (a > 0)]
    k = max(10, int(frac * len(a)))
    top, xmin = a[-k:], a[-k - 1]
    return float(1.0 / np.mean(np.log(top / xmin)))


def u_fd(prior, e1, e2, delta=1e-3, richardson=True):
    """`u = d_gamma log p_gamma(e)` at fixed `e` -- ShapeScoreNodes' own definition."""
    def grad(d):
        return np.stack([(prior.sheared_log_prob(e1, e2, +d, 0.0)
                          - prior.sheared_log_prob(e1, e2, -d, 0.0)) / (2 * d),
                         (prior.sheared_log_prob(e1, e2, 0.0, +d)
                          - prior.sheared_log_prob(e1, e2, 0.0, -d)) / (2 * d)], axis=1)
    u = grad(delta)
    if richardson:
        u = (4.0 * grad(0.5 * delta) - u) / 3.0
    return u


# ------------------------------------------------------------------------------------
# legs
# ------------------------------------------------------------------------------------

def leg_edge_analytic(taus):
    """E1a: the power-law control.  `u_1 -> e_1 (4 + 2a)`, bounded, for every `a`."""
    print("\n[E1a] ANALYTIC CONTROL: p_0 ~ (1-|eps|^2)^a, ray into the edge along e1")
    print("      the retracted (5.3d) predicted u1 ~ a/tau, i.e. u1*tau/a -> 1")
    print("      the closed form predicts u1 = e1 (4+2a),  i.e. u1/closed -> 1")
    r = np.sqrt(1.0 - taus)
    print(f"      {'a':>4} {'tau':>9} {'u1':>12} {'u1*tau/a':>10} {'u1/closed':>10}")
    for a in (0.0, 1.0, 2.0):
        u = u_fd(PowerLawDisc(a), r, np.zeros_like(r))
        for i, t in enumerate(taus):
            doc = (u[i, 0] * t / a) if a > 0 else float("nan")
            print(f"      {a:>4.1f} {t:>9.1e} {u[i,0]:>12.6f} {doc:>10.4f} "
                  f"{u[i,0]/(r[i]*(4.0+2.0*a)):>10.6f}")
    print("      E_0[u^2] = 4(a+2) exactly: "
          + ", ".join(f"a={a}->{4*(a+2)}" for a in (0.0, 1.0, 2.0)))


def leg_edge_fitted(prior, taus):
    """E1b: the SAME ray on the prior actually fitted.  `(1-t) psi'` is what must stay bounded."""
    print("\n[E1b] FITTED PRIOR: same ray.  The prior enters u only via (1-t) psi'(t).")
    r = np.sqrt(1.0 - taus)
    u = u_fd(prior, r, np.zeros_like(r))
    closed = generator_closed_form(prior, np.stack([r, np.zeros_like(r)], axis=1))
    dpsi = prior._dpsi(1.0 - taus)
    lbl = "(1-t)dpsi"
    print(f"      {'tau':>9} {'u1(fd)':>12} {'u1(closed)':>12} {lbl:>12} "
          f"{'fd/closed':>10}")
    for i, t in enumerate(taus):
        print(f"      {t:>9.1e} {u[i,0]:>12.6f} {closed[i,0]:>12.6f} "
              f"{t*dpsi[i]:>12.6f} {u[i,0]/closed[i,0]:>10.6f}")
    print(f"      spline tail slope psi' beyond t={prior._t1:.4f} is the constant "
          f"{prior._d1:+.4f} -> (1-t)psi' -> 0 at the edge")
    print(f"      NaN in the fd column is the prior's own hard cut at r_hard="
          f"{prior.r_hard} (log_prob = -inf beyond it, so the stencil differences -inf),")
    print(f"      NOT a divergence of u: the closed form has no cutoff and stays finite.")
    print(f"      That cut sits far outside both the grid (rmax 0.95) and the population.")


def leg_hill(prior, n, seed):
    """E2: Hill index of |u| on PRIOR SAMPLES -- the number comparable to §5C's 1.32/1.38."""
    print(f"\n[E2] HILL INDEX of |u| on {n:,} prior samples (§5C's phi' gave 1.32-1.38)")
    rng = np.random.default_rng(seed)
    e1, e2 = prior.sample(n, rng)
    u = generator_closed_form(prior, np.stack([e1, e2], axis=1))
    mag = np.hypot(u[:, 0], u[:, 1])
    for frac in (0.05, 0.01, 0.002):
        print(f"      top {frac:>6.1%}: Hill alpha = {hill(mag, frac):>7.3f}")
    print(f"      max|u| = {mag.max():.4f}   rms|u| = {np.sqrt(np.mean(mag**2)):.4f}   "
          f"max|eps| drawn = {np.hypot(e1,e2).max():.4f}")
    print("      alpha >> 2 means the second moment exists and Var(u) is a real number.")


def leg_grid_ladders(prior, refine, reaches, emax, fd_delta):
    """E3: the two grid ladders.  Deterministic quadrature: non-convergence, not noise."""
    print("\n[E3a] REFINEMENT ladder (rmax fixed at 0.95): does Var_0(u) settle in n?")
    _ladder(prior, [(n, 0.95, emax) for n in refine], fd_delta)
    print("\n[E3b] REACH ladder: does Var_0(u) settle as the grid approaches the edge?")
    print("      The default rmax=0.95 never visits it.  The grid STEP is held fixed")
    print("      (n scaled with emax), so this varies reach ALONE -- at fixed n, a wider")
    print("      emax would silently coarsen the spacing and confound the two.")
    base_n, base_emax = 121, 0.96
    step = 2.0 * base_emax / (base_n - 1)
    rungs = []
    for rm in reaches:
        em = max(base_emax, rm + 0.01)
        rungs.append((int(round(2.0 * em / step)) + 1, rm, em))
    _ladder(prior, rungs, fd_delta, step=step)


def _ladder(prior, settings, fd_delta, step=None):
    print(f"      {'n':>5} {'rmax':>6} {'G':>7} {'step':>8} {'Var_0(u)':>11} "
          f"{'E_0[u]':>11} {'curv/Var':>10} {'|u_fd-u_cf|':>12}")
    for row in settings:
        n, rmax, em = row if len(row) == 3 else (row[0], row[1], 0.96)
        gr, _ = make_e_grid(n=n, emax=em, rmax=rmax)
        nodes = ShapeScoreNodes(gr, prior, delta=fd_delta)
        b = nodes.bartlett()
        w = nodes.prior_weights()
        var = float(np.einsum("k,ka,ka->", w, nodes.u, nodes.u)
                    - np.sum((w @ nodes.u) ** 2))
        curv = float(np.max(np.abs(b["curvature"])))
        st = 2.0 * em / (n - 1)
        print(f"      {n:>5} {rmax:>6.3f} {len(gr):>7,} {st:>8.5f} {var:>11.5f} "
              f"{float(np.max(np.abs(b['mean_u']))):>11.2e} {curv/max(var,1e-30):>10.2e} "
              f"{nodes.closed_form_residual():>12.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-sample", default=PRIOR_CACHE)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=8)
    ap.add_argument("--prior-knot-margin", type=float, default=0.10)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--refine", default="41,61,81,121,161,201")
    ap.add_argument("--reaches", default="0.85,0.90,0.95,0.975,0.99,0.995")
    ap.add_argument("--hill-n", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    if not os.path.exists(args.prior_sample):
        print(f"prior cache missing: {args.prior_sample}\n"
              f"build it first with scripts/eval_score_response.py (mode unit).")
        return 2
    s = pf.read_table(args.prior_sample).to_pandas()
    prior = SmoothRadialPrior(s["e1_input_rot0_p"].to_numpy(), s["e2_input_rot0_p"].to_numpy(),
                              n_bins=args.prior_bins, n_knots=args.prior_knots,
                              knot_margin=args.prior_knot_margin)
    print(f"prior: {prior.n_samples:,} shapes, r_max={prior.r_max:.4f}, "
          f"{prior.n_knots} knots, chi2/dof={prior.fit_chi2_dof:.2f}, "
          f"norm err={prior.norm_error:+.2e}")

    taus = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    leg_edge_analytic(taus)
    leg_edge_fitted(prior, taus)
    leg_hill(prior, args.hill_n, args.seed)
    leg_grid_ladders(prior, [int(x) for x in args.refine.split(",")],
                     [float(x) for x in args.reaches.split(",")],
                     args.grid_emax, args.fd_delta)

    print("\n  Read [E2]'s alpha against 2 and [E3b]'s Var_0(u) column against itself.")
    print("  This gate covers the SHAPE channel only.  The size, flux and separation")
    print("  channels have different geometry -- the disc-tangency argument that makes the")
    print("  shape generator bounded does NOT apply to them, and they need their own test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
