"""`INFERENCE.md` §5C.5 cross-check (i): the Eulerian and Lagrangian scores, object by object.

Both (2.2) and (5.5) equal `d_gamma log p(xhat | gamma)`, so on any data where both are
computable they must agree PER GALAXY -- not just in the population sum.  That makes this
the sharpest available test of the §5C machinery on the real flow, and it needs no new
prior, no new model and no truth:

    Eulerian (5B)     s_i = E_w[u],       u_k    = -( v . grad log p_0 + div v )(e_k)
    Lagrangian (5C)   s_i = E_w[phi'],    phi_k(g) = log p_flow( ehat_i | S_g e_k )

with the SAME weights `w_k propto L_k p_0(e_k)`, because `S_0 = id`.  The identity relating
the two is the integration by parts that produced (2.7); if they disagree, either the
prior's radial spline is wrong (Eulerian side) or the sheared-grid likelihood is (Lagrangian
side), and the per-object scatter says which.

The Louis partners must agree too: `-E_w[du] - Var_w(u)` versus `-E_w[phi''] - Var_w(phi')`.

Scope.  This is the SHAPE channel with no detection factor, which is the only place the two
forms are both available -- `score_inference.py` has no `P_det` and no `P_pass`, and adding
them is exactly what §5C buys (§5C.2's table).  So a pass here validates the reparametrization,
not the full estimator (5.8).

Cost.  The Lagrangian side re-evaluates the flow on a sheared grid at each stencil point:
3 passes without Richardson, 5 with -- the "2-3x" of §5C.5.  Rows are slabbed so only
`slab x G` floats are ever resident.

Usage:
    python scripts/check_lagrangian_agreement.py --max-rows 100000 --grid-n 61
"""

import argparse
import os
import sys

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(SBSI_ROOT, "scripts")
for _p in (SBSI_ROOT, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.lagrangian_score import (  # noqa: E402
    curve_derivatives,
    score_and_information,
)
from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.posterior_shape import PosteriorShapeEstimator, make_e_grid  # noqa: E402
from sbs_shear.preprocessing import rescale  # noqa: E402
from sbs_shear.score_inference import ShapeScoreNodes, scores_from_loglike  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

import eval_score_response as esr  # noqa: E402  -- reuse the loader and prior verbatim


class ShearedGridBank:
    """`PosteriorShapeEstimator` per stencil offset, so the flow sees `S_gamma e_k`.

    One estimator per gamma, built once and reused for every slab.  Construction is cheap
    (it only standardises the grid); the cost is the `log_likelihood` call each one makes.
    Keyed on the rounded offset so `curve_derivatives` hits the cache for repeated `t`.
    """

    def __init__(self, bundle, grid, axis, device=None):
        self.bundle, self.grid, self.axis, self.device = bundle, grid, axis, device
        self._cache = {}

    def __call__(self, t):
        key = round(float(t), 12)
        if key not in self._cache:
            g = [0.0, 0.0]
            g[self.axis] = key
            e1, e2 = apply_shear_to_ellipticity(self.grid[:, 0], self.grid[:, 1], g[0], g[1])
            sheared = np.stack([e1, e2], axis=1)
            self._cache[key] = PosteriorShapeEstimator(self.bundle, sheared,
                                                       device=self.device)
        return self._cache[key]


def lagrangian_slab(bank, frame, ehat, log_prior, delta, richardson, chunk):
    """`(s, I)` along one shear axis for one slab of rows, from the curve (5.5b)."""
    def f(t):
        # shift_rows=False is mandatory: the per-row max moves with gamma, and
        # differencing shifted rows would put -dM/dgamma straight into s_i.
        return bank(t).log_likelihood(frame, ehat, chunk=chunk,
                                      out_dtype=np.float32, shift_rows=False)

    p0, d1, d2 = curve_derivatives(f, delta=delta, richardson=richardson)
    return score_and_information(p0, d1, d2, log_prior=log_prior)


def report(axis, s_e, i_e, s_l, i_l):
    lbl = f"gamma{axis + 1}"
    ds = s_l - s_e
    scale = float(np.std(s_e))
    print(f"\n  --- {lbl} ---")
    print(f"    <s>   Eulerian {np.mean(s_e):+.6f}   Lagrangian {np.mean(s_l):+.6f}   "
          f"diff {np.mean(ds):+.3e}")
    print(f"    sd(s) Eulerian {scale:.6f}   Lagrangian {np.std(s_l):.6f}")
    print(f"    per-object |Ds|/sd(s):  rms {np.sqrt(np.mean(ds ** 2)) / scale:.3e}   "
          f"max {np.max(np.abs(ds)) / scale:.3e}")
    print(f"    correlation r = {np.corrcoef(s_e, s_l)[0, 1]:.10f}")
    di = i_l - i_e
    iscale = float(np.mean(i_e))
    print(f"    <I>   Eulerian {iscale:+.5f}   Lagrangian {np.mean(i_l):+.5f}   "
          f"rel diff {np.mean(di) / iscale:+.3e}")
    print(f"    per-object |DI|/<I>:    rms {np.sqrt(np.mean(di ** 2)) / abs(iscale):.3e}")
    return float(np.sqrt(np.mean(ds ** 2)) / scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measurement-model",
                    default="models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt")
    ap.add_argument("--g0-catalogue", default=esr.G0_CAT)
    ap.add_argument("--prior-sample", default=esr.PRIOR_CACHE)
    ap.add_argument("--prior-catalogue", default=esr.G0_CAT)
    ap.add_argument("--prior-rows", type=int, default=2_000_000)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=12)
    ap.add_argument("--prior-knot-margin", type=float, default=0.02)
    ap.add_argument("--max-rows", type=int, default=100_000)
    ap.add_argument("--grid-n", type=int, default=61)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--info-delta", type=float, default=0.0025)
    ap.add_argument("--lag-delta", type=float, default=0.01,
                    help="stencil half-width for the Lagrangian curve (5.5b)")
    ap.add_argument("--no-richardson", action="store_true",
                    help="3 flow passes per axis instead of 5; coarser but cheaper")
    ap.add_argument("--slab", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="max rms per-object |Ds|/sd(s) treated as agreement")
    ap.add_argument("--device", default=None)
    for k, v in dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0,
                     psf_fwhm=0.73, moffat_beta=2.224).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={args.device}  torch={torch.__version__}")

    prior = esr.build_prior(args)
    grid, cell = make_e_grid(n=args.grid_n, emax=args.grid_emax, rmax=args.grid_rmax)
    print(f"grid: n={args.grid_n} -> G={len(grid)} nodes, cell area {cell:.3e}")

    bundle = load_measurement_model(args.measurement_model, device=args.device)
    print(f"model: {os.path.basename(args.measurement_model)}")

    df = esr.load_g0(args.g0_catalogue, args.max_rows)
    ehat = df[["measured_ngmix_g1", "measured_ngmix_g2"]].to_numpy(float)
    good = np.isfinite(ehat).all(axis=1)
    df, ehat = df[good].reset_index(drop=True), ehat[good]
    print(f"rows: {len(df):,} detected+selected, finite shapes")

    nodes = ShapeScoreNodes(grid, prior, delta=args.fd_delta, info_delta=args.info_delta)
    print(f"node bank: G={len(grid)}, supported={int(nodes.support.sum())}, "
          f"|u_fd-u_closed|/rms={nodes.closed_form_residual():.2e}")
    bart = nodes.bartlett()
    print(f"bartlett: E[u]={bart['mean_u']}, curvature diag="
          f"{np.diag(bart['curvature'])}, scale={bart['scale']:.4f}")

    log_prior = np.where(nodes.support, nodes.log_prior, -np.inf)
    fr = rescale(df.copy(), **rk)
    n = len(df)
    banks = [ShearedGridBank(bundle, grid, a, device=args.device) for a in (0, 1)]

    s_e = np.empty((n, 2)); i_e = np.empty((n, 2, 2))
    s_l = np.empty((n, 2)); i_l = np.empty((n, 2))
    n_pass = 3 if args.no_richardson else 5
    print(f"\nLagrangian stencil: delta={args.lag_delta}, "
          f"{'no ' if args.no_richardson else ''}richardson -> {n_pass} flow passes/axis")

    for start in range(0, n, args.slab):
        stop = min(start + args.slab, n)
        sl_fr, sl_e = fr.iloc[start:stop], ehat[start:stop]

        # Eulerian: one pass, the row shift is harmless here (softmax is shift-invariant)
        ll = banks[0](0.0).log_likelihood(sl_fr, sl_e, chunk=args.chunk,
                                          out_dtype=np.float32)
        s_e[start:stop], i_e[start:stop], _ = scores_from_loglike(
            ll, nodes, device=args.device, analytic_info=True)
        del ll

        for a in (0, 1):
            s, info = lagrangian_slab(banks[a], sl_fr, sl_e, log_prior,
                                      args.lag_delta, not args.no_richardson, args.chunk)
            s_l[start:stop, a], i_l[start:stop, a] = s, info
        print(f"  rows {start:,}-{stop:,} done", flush=True)

    print("\n=== §5C.5 cross-check (i): Eulerian (2.2) vs Lagrangian (5.5) ===")
    worst = max(report(a, s_e[:, a], i_e[:, a, a], s_l[:, a], i_l[:, a]) for a in (0, 1))

    print(f"\n  worst per-object rms |Ds|/sd(s) = {worst:.3e}  (tolerance {args.tol:.1e})")
    if worst <= args.tol:
        print("  AGREE -- the reparametrization is exact on the real flow.")
    else:
        print("  DISAGREE -- one of the two sides is wrong.  A per-object scatter that is "
              "flat in |e| points at the sheared-grid likelihood; one that grows with |e| "
              "points at the prior's radial spline (the Eulerian side).")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
