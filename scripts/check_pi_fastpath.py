"""Regression check: the rewritten `pass_fraction_by_node` returns the OLD numbers.

The rewrite claims the residual flow cannot see the node -- `flow_ctx` index-selects away
`flow_drop_indices = [0, 1, 8, 9]`, which are `e1`, `e2` and their missing indicators -- so
under the common random numbers the loop already imposed, one flow pass serves the whole
node bank and only the 16->128->2 mean head has to be re-evaluated.  That is an exactness
claim about a function whose output feeds `<s>_sel` and `I_sel`, i.e. straight into the
denominator of (5.3).  It is one array comparison to check, so check it.

The reference below is a literal transcription of the loop as it stood before the rewrite:
rebuild the frame per node, reseed, call `bundle.sample`, apply the cut in numpy.  Anything
but agreement to float noise means the reuse is invalid for this checkpoint and the fast
path must not be used with it.

Small by construction -- a handful of nodes and a few thousand rows is enough to falsify an
exactness claim, and the point is to run it often, not to run it long.
"""

import argparse
import os
import sys

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.posterior_shape import make_e_grid  # noqa: E402
from sbs_shear.precision import MODES, precision_region  # noqa: E402
from sbs_shear.preprocessing import rescale  # noqa: E402

from eval_score_response import G0_CAT, PRIOR_CACHE, load_g0  # noqa: E402
from eval_score_select import MODEL, pass_fraction_by_node  # noqa: E402


def reference(bundle, df, grid, rk, cut, n_samples, batch_size, seeds, ladder, rng):
    """`pass_fraction_by_node` exactly as it read before the rewrite."""
    g = np.asarray(grid, float)
    ladder = list(ladder)
    m_max = max(ladder)
    subs = [df.iloc[rng.choice(len(df), m_max, replace=False)].reset_index(drop=True)
            for _ in seeds]
    out = np.empty((len(ladder), len(g), len(seeds)), dtype=np.float64)
    for j, (sd, sub) in enumerate(zip(seeds, subs)):
        for k in range(len(g)):
            f = sub.copy()
            f["e1_input_rot0_p"] = g[k, 0]
            f["e2_input_rot0_p"] = g[k, 1]
            fr = rescale(f, **rk)
            torch.manual_seed(sd)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sd)
            xh = bundle.sample(fr, n_samples=n_samples, batch_size=batch_size)
            cs = np.cumsum(cut(xh).mean(axis=1))
            for t, mm in enumerate(ladder):
                out[t, k, j] = cs[mm - 1] / mm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default=MODEL)
    ap.add_argument("--g0-catalogue", default=G0_CAT)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--pi-samples", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--grid-n", type=int, default=61)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--cut-abs-ehat", type=float, default=0.6)
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--info-delta", type=float, default=0.02)
    ap.add_argument("--prior-sample", default=PRIOR_CACHE)
    ap.add_argument("--prior-catalogue", default=G0_CAT)
    ap.add_argument("--prior-rows", type=int, default=2_000_000)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=8)
    ap.add_argument("--prior-knot-margin", type=float, default=0.10)
    ap.add_argument("--precision", default="fp32", choices=list(MODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    for k, v in dict(pixel_rms=6.0, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.8,
                     moffat_beta=3.5).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    bundle = load_measurement_model(args.measurement_model, device=args.device)
    df = load_g0(args.g0_catalogue, args.rows)
    full, _ = make_e_grid(n=args.grid_n, emax=args.grid_emax, rmax=args.grid_rmax)
    # spread the probe nodes across the bank: a cluster near the origin would not exercise
    # the mean head's dependence on e, which is the only thing the fast path recomputes
    grid = full[np.linspace(0, len(full) - 1, args.nodes).astype(int)]
    c = args.cut_abs_ehat
    seeds = [args.seed + 1000 * r for r in range(args.reps)]
    ladder = [args.rows // 2, args.rows]
    print(f"device={args.device}  precision={args.precision}  rows={args.rows:,}  "
          f"nodes={len(grid)}  reps={args.reps}  ladder={ladder}", flush=True)

    kw = dict(n_samples=args.pi_samples, batch_size=args.batch_size, seeds=seeds,
              ladder=ladder)
    with precision_region(args.precision, args.device):
        fast = pass_fraction_by_node(
            bundle, df, grid, rk,
            cut=lambda x: torch.hypot(x[..., 0], x[..., 1]) < c,
            rng=np.random.default_rng(args.seed + 77), **kw)
        ref = reference(
            bundle, df, grid, rk,
            cut=lambda x: np.hypot(x[:, :, 0], x[:, :, 1]) < c,
            rng=np.random.default_rng(args.seed + 77), **kw)

    d = np.abs(fast - ref)
    one_draw = 1.0 / (min(ladder) * args.pi_samples)
    print(f"\nPi shape {fast.shape}   <Pi>_ref={ref.mean():.6f}   <Pi>_fast={fast.mean():.6f}")
    print(f"max |fast - ref| = {d.max():.3e}  ({d.max()/one_draw:.2f} flipped draws)")
    print(f"mean |fast - ref| = {d.mean():.3e}   one flipped draw = {one_draw:.3e}", flush=True)

    # WHY THE TOLERANCE IS NOT ZERO.  The two paths reach the same sample by different
    # arithmetic -- `(resid + mu) * scale + mean` in torch against `flow.sample() + mu` then
    # `inverse_transform_array` in numpy -- so a draw whose |xhat| sits within float rounding
    # of the cut can land on the other side of it.  Measured at 65536 rows x 64 nodes: ONE
    # flipped draw in 67 million, mean |dPi| = 6e-08.  That is rounding, not a logic error;
    # demanding literal zero at scale would be demanding bit-identical float pipelines.
    ok = d.mean() < 1e-6 and d.max() / one_draw < 10

    # BUT Pi IS NOT THE NUMBER THAT MATTERS.  `<s>_sel` is a near-cancellation -- for an
    # isotropic cut it should vanish outright -- so it amplifies whatever survives in Pi,
    # and a small dPi does NOT imply a small d<s>_sel.  Compare what actually enters (5.3).
    # This needs the whole bank to mean anything, so it only runs when the probe nodes ARE
    # the bank; with a sparse bank `population_terms` is not merely noisy but meaningless.
    verdict = f"PI FASTPATH {'OK' if ok else 'MISMATCH'}"
    if args.nodes >= len(full):
        from sbs_shear.score_inference import ShapeScoreNodes, population_terms
        from eval_score_response import build_prior
        nd = ShapeScoreNodes(grid, build_prior(args), delta=args.fd_delta,
                             info_delta=args.info_delta)
        term = lambda p: population_terms(nd, np.log(np.maximum(p, 1e-12)))
        sf, if_ = term(fast[-1].mean(axis=1))
        sr, ir = term(ref[-1].mean(axis=1))
        print(f"\n<s>_sel  ref=[{sr[0]:+.6f},{sr[1]:+.6f}]  fast=[{sf[0]:+.6f},{sf[1]:+.6f}]"
              f"  d=[{sf[0]-sr[0]:+.2e},{sf[1]-sr[1]:+.2e}]")
        print(f"I_sel00  ref={ir[0,0]:+.6f}  fast={if_[0,0]:+.6f}  d={if_[0,0]-ir[0,0]:+.2e}")
        print("  <s>_sel's replicate error in the definitive runs is ~3e-4.  A difference "
              "well below\n  that is invisible to any result; one comparable to it is not.",
              flush=True)
        verdict += f"; d<s>_sel = [{sf[0]-sr[0]:+.1e},{sf[1]-sr[1]:+.1e}]"
    else:
        verdict += f"; <s>_sel NOT tested (needs --nodes {len(full)}, got {args.nodes})"
    print(f"\n### {verdict} ###", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
