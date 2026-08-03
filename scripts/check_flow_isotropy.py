"""Is the trained flow isotropic in the shape context?  `<s>_sel` says probably not.

WHY.  §5B.2 predicts `<s>_sel = 0` for an isotropic cut.  The measured value is not zero
and, as of cont.176's deep ladders, is CONVERGED and enormous in units of its own error:
`<s>_sel,2 = +0.0108 +/- 0.000078` at cut 0.6 (138 sigma) and `+0.0179 +/- 0.000106` at
cut 0.4 (169 sigma), while component 1 is ~6x smaller and FLIPS SIGN between the two cuts.
Three things could break the prediction:

  (a) the prior is anisotropic  -- CHECKED AND RULED OUT.  The 928 900-shape prior sample
      has <e1> = +0.000356 +/- 0.000248 and <e2> = +0.000363 +/- 0.000247 (both ~1.4 sigma,
      and equal to each other), with per-component widths 0.23858 vs 0.23838, i.e. matched
      to 0.1%.  It is also fitted as a radial spline in |e|, so it is isotropic by
      construction as well as in the sample.
  (b) the node grid is anisotropic -- a 61x61 Cartesian grid clipped to a disc is symmetric
      under e2 -> -e2, so this would have to be a bug rather than a design choice.
  (c) THE FLOW ITSELF treats `e1` and `e2` differently.  Nothing in the architecture or the
      loss imposes the spin-2 rotational symmetry that the physics has; it is only ever
      learned from data.  The supporting tell is already in the reports: `I_sel` has
      diag 1.011 (e1) vs 1.026 (e2), a 1.5% asymmetry, plus off-diag/diag = 0.007.

This script tests (c) directly, and separates it from (b) by construction.  `Pi_k` is
`P(xhat in S | true shape = e_k)` marginalised over the population.  If the model is
isotropic then `Pi` depends on `|e|` ALONE, so on a ring of constant `|e|` it must be flat.
So: build rings instead of a grid, hand them to the SAME `pass_fraction_by_node` the
estimator uses, and look at the variation around each ring.

Reading the output.  Per ring we report the spread of `Pi` around it against the Monte
Carlo error on a single ring point, `sqrt(Pi(1-Pi)/(M*n_samples))`.  Flat to within that is
consistent with isotropy.  We also report the two lowest angular harmonics, since they say
WHICH symmetry is broken and are what `<s>_sel` actually integrates:

  m=2 (cos/sin 2phi)  -- a genuine e1-vs-e2 axis asymmetry, the spin-2 symmetry itself.
  m=4 (cos/sin 4phi)  -- a FOUR-FOLD pattern.  Do not read this as "network bug" by reflex:
                         the simulation is rendered on SQUARE PIXELS and measured in square
                         postage stamps, so the measurement genuinely has C4 symmetry rather
                         than full SO(2), and an m=4 term in `Pi` may be the flow correctly
                         reproducing pixelisation.  m=2 has no such excuse -- a square grid
                         cannot produce it -- so an m=2 term is either a learned-symmetry
                         failure or a real e1-vs-e2 asymmetry in the sim (an elliptical PSF
                         would do it; the Moffat used here is parameterised round).

WHICH HARMONIC MATTERS IS NOT OBVIOUS, and this script does not decide it.  `<s>_sel` is
`E_Pi[u]` with `u` the analytic prior score, and `u` is NOT purely spin-2: the shear map is
nonlinear in `e`, so `u` carries m=0, m=2 and m=4 content and `Pi`'s m=4 term can couple to
it. Deciding the question needs the harmonic decomposition of `u` -- or, more directly, one
cheap numerical experiment: recompute `<s>_sel` with `Pi` replaced by its azimuthal average,
so that `Pi` depends on `|e|` alone by construction. If the anomaly vanishes the anisotropy
is the cause; if it survives, it is not.

A NULL IS INFORMATIVE HERE.  If the rings come out flat, then (c) is ruled out too and the
`<s>_sel` anomaly lives in the score/prior machinery -- `ShapeScoreNodes`, `population_terms`
or the grid quadrature -- not in the model.  Either answer localises the problem, which is
the point; this is a diagnostic, not an acceptance test.
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
from sbs_shear.precision import MODES, precision_region  # noqa: E402

from eval_score_response import G0_CAT, load_g0  # noqa: E402
from eval_score_select import MODEL, pass_fraction_by_node  # noqa: E402


def ring_grid(radii, n_phi):
    """Nodes on concentric rings: `(G, 2)` laid out ring-major, plus the angles."""
    phi = np.arange(n_phi) * (2.0 * np.pi / n_phi)
    pts = np.concatenate([np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
                          for r in radii], axis=0)
    return pts, phi


def harmonics(pi, phi, m):
    """Amplitude of the `cos(m phi)` / `sin(m phi)` components of `Pi` around one ring."""
    c = 2.0 * np.mean(pi * np.cos(m * phi))
    s = 2.0 * np.mean(pi * np.sin(m * phi))
    return np.hypot(c, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default=MODEL)
    ap.add_argument("--g0-catalogue", default=G0_CAT)
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--pi-rows", type=int, default=262_144,
                    help="population sample per ring point.  The signal we are chasing is "
                         "1e-2 in <s>_sel, which Pi resolves only if its own error is well "
                         "below the ring-to-ring spread; 262k x 8 draws gives ~3e-4")
    ap.add_argument("--pi-samples", type=int, default=8)
    ap.add_argument("--reps", type=int, default=4,
                    help="independent row draws; the spread across them is the error bar "
                         "that matters, since a single draw shares rows around the ring")
    ap.add_argument("--n-phi", type=int, default=32)
    ap.add_argument("--radii", default="0.1,0.2,0.3,0.45,0.6,0.75,0.9")
    ap.add_argument("--cut-abs-ehat", type=float, default=0.6)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default="fp32", choices=list(MODES))
    for k, v in dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0,
                     psf_fwhm=0.73, moffat_beta=2.224).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    radii = [float(r) for r in args.radii.split(",")]
    grid, phi = ring_grid(radii, args.n_phi)

    bundle = load_measurement_model(args.measurement_model, device=args.device)
    df = load_g0(args.g0_catalogue, args.max_rows)
    print(f"device={args.device}  rows={len(df):,}  "
          f"{len(radii)} rings x {args.n_phi} angles = {len(grid)} nodes  "
          f"M={args.pi_rows:,} x {args.pi_samples} draws x {args.reps} reps", flush=True)
    print(f"cut |xhat| < {args.cut_abs_ehat}", flush=True)

    seeds = [args.seed + 1000 * j for j in range(args.reps)]
    with precision_region(args.precision, args.device):
        pi = pass_fraction_by_node(
            bundle, df, grid, rk,
            cut=lambda x: torch.hypot(x[..., 0], x[..., 1]) < args.cut_abs_ehat,
            n_samples=args.pi_samples, batch_size=args.batch_size, seeds=seeds,
            ladder=[args.pi_rows], rng=np.random.default_rng(args.seed + 77))[0]

    # `pi` is (G, reps).  Average over reps for the shape of the ring; use the scatter
    # ACROSS reps for the error, since within one rep the ring points share rows and their
    # Monte Carlo noise is correlated -- the binomial formula would understate it.
    mean = pi.mean(axis=1)
    err = pi.std(axis=1, ddof=1) / np.sqrt(pi.shape[1])

    print("\n  |e|     <Pi>    ring spread   err/point   spread/err     A2        A4")
    print("  " + "-" * 68)
    flagged = []
    for i, r in enumerate(radii):
        sl = slice(i * args.n_phi, (i + 1) * args.n_phi)
        p, e = mean[sl], err[sl]
        spread = p.std(ddof=1)
        epoint = float(np.mean(e))
        a2, a4 = harmonics(p, phi, 2), harmonics(p, phi, 4)
        ratio = spread / epoint if epoint > 0 else np.inf
        print(f"  {r:.2f}  {p.mean():7.5f}   {spread:9.2e}   {epoint:9.2e}   "
              f"{ratio:8.1f}   {a2:8.2e}  {a4:8.2e}")
        if ratio > 3:
            flagged.append((r, ratio, a2, a4))

    print("\n  A2 = |cos2phi, sin2phi| -- an e1-vs-e2 axis asymmetry.  A square pixel grid")
    print("       CANNOT make this, so it is a learned-symmetry failure or a real sim")
    print("       asymmetry (e.g. an elliptical PSF).")
    print("  A4 = |cos4phi, sin4phi| -- a four-fold pattern, which square pixels and square")
    print("       postage stamps genuinely do have; may be the flow reproducing the sim.")
    print("  'spread/err' compares the ring's variation to the error on one ring point;")
    print("  isotropy predicts ~1.  Rings share rows within a replicate, so the error bar")
    print("  is the scatter ACROSS replicates, not the binomial formula.")
    print("  Which harmonic drives <s>_sel is NOT settled here -- u is not purely spin-2.")
    if flagged:
        worst = max(flagged, key=lambda t: t[1])
        kind = "m=2" if worst[2] > worst[3] else "m=4"
        print(f"\n### FLOW ANISOTROPIC: {len(flagged)}/{len(radii)} rings exceed 3x; "
              f"worst |e|={worst[0]:.2f} at {worst[1]:.0f}x, largest harmonic {kind} ###")
    else:
        print("\n### FLOW ISOTROPIC on every ring: the <s>_sel anomaly is NOT the model. "
              "Look at ShapeScoreNodes / population_terms / the grid quadrature. ###")
    # Exit 0 for EITHER verdict.  This is a diagnostic, not an acceptance test: both answers
    # are results, and returning non-zero on a finding makes `sacct` report FAILED, which
    # reads as a crash.  A genuine error still raises and exits non-zero on its own.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
