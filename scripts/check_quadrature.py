"""Is the §5B machinery itself biased?  A flow-free, data-free convergence test.

WHY THIS IS THE RIGHT TEST RIGHT NOW.  The closure test draws its `xhat` FROM THE FLOW and
then uses that same flow's response to invert it.  The model is therefore perfectly specified
by construction, and `d(m)` ought to be zero up to numerics.  It is not: +0.640 +/- 0.280% at
cut 0.6.  With physics ruled out by construction, every remaining candidate is arithmetic:

  (a) the population terms are a QUADRATURE on G = 2765 nodes, not an exact integral;
  (b) `I_sel` is a FINITE DIFFERENCE (`info_delta`), and `u` is another one (`delta`);
  (c) `ghat = sum s / sum I` is a RATIO, unbiased only to first order;
  (d) the estimator is derived at `gamma = 0` and evaluated at `g = 0.05`.

(c) and (d) need the flow and are being tested by the g=0 null and the g=0.10 linearity runs.
(a) and (b) need neither, and that is what this script measures -- exactly, with no Monte
Carlo noise anywhere, which is what makes it sharper than cont.175's attempt.  That one
doubled the node bank against a MONTE-CARLO `Pi` and got `-0.349% +/- 0.71%`: a 0.5 sigma
null, but at a precision 2x COARSER than the effect now being chased, so it never cleared the
grid at the 0.3% level.  It was recorded as "grid refinement: null" and I had been treating it
as settled.  Replacing `Pi` with an ANALYTIC function removes the noise entirely and turns the
same comparison into a convergence measurement good to machine precision.

TWO PROBES, both flow-free:

  1. `bartlett()` -- the two identities that hold for ANY normalised prior, `E_0[u] = 0` and
     `E_0[du] + Var_0(u) = 0`.  They follow from `integral p_gamma = 1` alone, so any residual
     is pure numerics: prior, generator, shear map and quadrature together.  This is the
     machinery's own error with nothing else mixed in.

  2. `population_terms()` on an ANALYTIC `Pi` -- the actual quantities that enter (5.3),
     computed at increasing grid resolution and compared against a fine reference.  This is
     the one that converts directly into a bias on `m`.

CONVERTING TO `m`, so the numbers mean something.  `ghat` has `I_sel` in its denominator:
`ghat = (sum s - N<s>_sel) / (sum I - N I_sel)`.  A RELATIVE error `eps` on `I_sel` moves the
denominator by `eps * I_sel / (<I> - I_sel)`, and with the measured `I_sel/<I> = 0.274` at cut
0.6 that is `0.377 * eps`, landing on `m` with the opposite sign.  So **a 1% error on `I_sel`
is a 0.38% bias on `m`** -- which is the size of the residual we are chasing.  (The same
factor is implicit in the driver's own note that a 0.63% `I_sel` error is +/-0.24% on `m`.)
The threshold that matters is therefore about 0.8% on `I_sel` for a 0.3% deliverable.
"""

import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbsi.posterior_shape import make_e_grid  # noqa: E402
from sbsi.score_inference import ShapeScoreNodes, population_terms  # noqa: E402

from eval_score_response import G0_CAT, PRIOR_CACHE, build_prior  # noqa: E402

# `I_sel / <I>` measured at cut 0.6; converts a relative I_sel error into a bias on m.
I_SEL_FRAC = 0.2742
M_PER_RELATIVE_I_SEL = I_SEL_FRAC / (1.0 - I_SEL_FRAC)


def analytic_pi(grid, a2=0.006, a4=0.035):
    """A smooth, realistic stand-in for `Pi` with NO Monte Carlo noise.

    Radial part is fitted by eye to the measured cut-0.6 profile (0.803 at |e|=0.1 down to
    0.459 at 0.9).  The angular part carries m=2 and m=4 terms of roughly the measured
    amplitudes, growing with radius as they do in the data, so the test exercises the same
    angular structure the real `Pi` has rather than an artificially smooth one.
    """
    g = np.asarray(grid, float)
    r = np.hypot(g[:, 0], g[:, 1])
    phi = np.arctan2(g[:, 1], g[:, 0])
    radial = 0.81 - 0.42 * r ** 2
    angular = 1.0 + a2 * r ** 2 * np.cos(2 * phi) + a4 * r ** 4 * np.cos(4 * phi)
    return np.clip(radial * angular, 1e-6, 1.0 - 1e-9)


def build_nodes(args, n, delta, info_delta, prior):
    grid, _ = make_e_grid(n=n, emax=args.grid_emax, rmax=args.grid_rmax)
    return grid, ShapeScoreNodes(grid, prior, delta=delta, info_delta=info_delta)


def _list(text):
    """Split a CLI list on commas OR colons.

    `sbatch --export` splits its OWN argument on commas, so `--export=ALL,EXTRA="--grid-ns
    61,101,141"` delivers `--grid-ns 61` and drops the rest -- the job then runs one rung,
    exits 0 and reads as a converged sweep.  It cost a run here.  Colons survive, and
    `jobs/job_pi_grid_ladder.sh` already uses that convention, so accept both.
    """
    return [x for x in text.replace(":", ",").split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--grid-ns", default="31,45,61,81,101,141")
    ap.add_argument("--reference-n", type=int, default=201,
                    help="the fine grid every coarser one is compared against")
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--info-delta", type=float, default=0.0025)
    ap.add_argument("--delta-sweep", default="0.04,0.02,0.01,0.005,0.0025")
    ap.add_argument("--prior-sample", default=PRIOR_CACHE)
    ap.add_argument("--prior-catalogue", default=G0_CAT)
    ap.add_argument("--prior-rows", type=int, default=2_000_000)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=8)
    ap.add_argument("--prior-knot-margin", type=float, default=0.10)
    args = ap.parse_args()

    prior = build_prior(args)
    ns = [int(x) for x in _list(args.grid_ns)]

    # ---- probe 1: Bartlett, the machinery's own identities --------------------------
    print("=" * 78)
    print("PROBE 1: Bartlett identities -- E_0[u] = 0 and E_0[du] + Var_0(u) = 0.")
    print("Pure numerics: prior + generator + shear map + quadrature, no flow, no data.")
    print("'curvature' is reported RELATIVE to Var_0(u), i.e. as a fractional error on I.")
    print("=" * 78)
    print(f"\n  grid_n        G     |E_0[u]|/scale    |curv|/Var(u)   => bias on m")
    print("  " + "-" * 68)
    for n in ns + [args.reference_n]:
        grid, nodes = build_nodes(args, n, args.fd_delta, args.info_delta, prior)
        b = nodes.bartlett()
        w = nodes.prior_weights()
        var_u = np.einsum("k,ka,kb->ab", w, nodes.u, nodes.u) - np.outer(b["mean_u"],
                                                                        b["mean_u"])
        rel_c = np.abs(b["curvature"]).max() / max(np.abs(np.diag(var_u)).mean(), 1e-30)
        print(f"  {n:6d} {len(grid):8,}   {np.abs(b['mean_u']).max()/b['scale']:13.3e}   "
              f"{rel_c:13.3e}   {rel_c*M_PER_RELATIVE_I_SEL:+9.3%}")

    # ---- probe 2: the actual (5.3) terms on an analytic Pi ---------------------------
    print("\n" + "=" * 78)
    print("PROBE 2: <s>_sel and I_sel on an ANALYTIC Pi, vs a fine reference grid.")
    print("These are the quantities (5.3) actually uses.  No Monte Carlo anywhere, so the")
    print("difference IS the quadrature error -- not a noise-limited upper bound on it.")
    print("=" * 78)
    gref, nref = build_nodes(args, args.reference_n, args.fd_delta, args.info_delta, prior)
    sref, iref = population_terms(nref, np.log(analytic_pi(gref)))
    print(f"\n  reference: grid_n={args.reference_n} G={len(gref):,}  "
          f"<s>_sel=[{sref[0]:+.6f}, {sref[1]:+.6f}]  I_sel_00={iref[0,0]:+.6f}")
    print(f"\n  grid_n        G     d<s>_sel_1     d<s>_sel_2    dI_sel/I_sel   => bias on m")
    print("  " + "-" * 74)
    for n in ns:
        grid, nodes = build_nodes(args, n, args.fd_delta, args.info_delta, prior)
        s, i = population_terms(nodes, np.log(analytic_pi(grid)))
        rel_i = (i[0, 0] - iref[0, 0]) / abs(iref[0, 0])
        print(f"  {n:6d} {len(grid):8,}   {s[0]-sref[0]:+11.3e}   {s[1]-sref[1]:+11.3e}   "
              f"{rel_i:+12.3e}   {-rel_i*M_PER_RELATIVE_I_SEL:+9.3%}")

    # ---- probe 3: the finite-difference deltas ---------------------------------------
    print("\n" + "=" * 78)
    print("PROBE 3: finite-difference truncation.  `delta` builds `u`, `info_delta` builds")
    print("I_sel by differencing the score.  Both are truncation errors, both scale as the")
    print("step SQUARED for a centred difference -- so halving the step should quarter the")
    print("residual.  A term that does NOT shrink that way is not truncation.")
    print("=" * 78)
    n_fix = 61
    print(f"\n  at grid_n={n_fix}, varying `delta` (the generator step):")
    print(f"    delta      <s>_sel_1      I_sel_00      |E_0[u]|/scale")
    base = None
    for d in [float(x) for x in _list(args.delta_sweep)]:
        grid, nodes = build_nodes(args, n_fix, d, args.info_delta, prior)
        s, i = population_terms(nodes, np.log(analytic_pi(grid)))
        b = nodes.bartlett()
        tag = "" if base is None else f"   (dI vs prev {i[0,0]-base:+.3e})"
        base = i[0, 0]
        print(f"    {d:<8.4f}  {s[0]:+.6f}    {i[0,0]:+.6f}    "
              f"{np.abs(b['mean_u']).max()/b['scale']:.3e}{tag}")

    print(f"\n  at grid_n={n_fix}, varying `info_delta` (the information step):")
    print(f"    info_delta   <s>_sel_1      I_sel_00")
    for d in [float(x) for x in _list(args.delta_sweep)]:
        grid, nodes = build_nodes(args, n_fix, args.fd_delta, d, prior)
        s, i = population_terms(nodes, np.log(analytic_pi(grid)))
        print(f"    {d:<10.4f}   {s[0]:+.6f}    {i[0,0]:+.6f}")

    print("\n  READING THIS.  The production grid is grid_n=61 (G=2765) with delta=0.01 and")
    print("  info_delta=0.0025.  If probe 2's 'bias on m' at grid_n=61 is a sizeable")
    print("  fraction of the +0.640% residual, the quadrature is the explanation and the fix")
    print("  is a finer grid -- cost is exactly linear in G.  If it is far below, the")
    print("  arithmetic is clean and the residual must come from the ratio bias or from")
    print("  evaluating a gamma=0 estimator at finite gamma, which the g=0 and g=0.10 runs")
    print("  are testing.  These probes cannot see those two: they have no flow and no data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
