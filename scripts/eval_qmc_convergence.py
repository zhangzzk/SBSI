"""Does randomized QMC let us cut n_samples in the selection harness? 1 seed, ISOLATED.

WHY. The selection estimator needs, per galaxy, an expectation over the flow's 4D predictive
distribution restricted to the passing side of a MEASURED cut. There is no closed form (the coupling
layers mix all four dims nonlinearly, so there is no marginal CDF), so it is a Monte Carlo integral
and `n_samples` is the dominant cost: the 16-seed run is 16 ckpt x 2 legs x 1.02M gal x 128 draws =
4.16 BILLION forward passes. Plain `torch.randn` converges as 1/sqrt(n). A randomly-shifted scrambled
Sobol set converges nearer 1/n on smooth integrands, so the same accuracy may need far fewer draws.

DESIGN -- measure the SCATTER, not the distance to some assumed truth. For each (mode, n) we repeat
the whole estimate with R independent sampling seeds and report the spread of R_model. That is a
direct, assumption-free measurement of the estimator's own noise: whichever mode has the smaller
spread at a given n is the better estimator, and the n at which QMC's spread matches plain MC at 128
is the answer to "how far can we cut n_samples".

We also report each mode's MEAN, because a variance reduction that shifts the central value is not a
variance reduction -- it is a bug. RQMC is unbiased BY CONSTRUCTION only because each object gets its
own random shift (see _qmc_normal); if the means separate, that construction is wrong.

n_samples are powers of two: Sobol's equidistribution holds on 2^k points and degrades off it.

FIREWALL: half-shear legs only, ISOLATED (R_blend ~ 0). No emulator, no constgold. Scores one
existing checkpoint; trains nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from eval_selection_response import (  # noqa: E402
    CAT, CROWD, NN, build_base, model_selected_response, truth_selected_response,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--n-list", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--size-cuts", type=float, nargs="+", default=[3.5, 4.4])
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[24.0])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device}  ckpt={os.path.basename(args.ckpt)}  "
          f"n_list={args.n_list}  repeats={args.repeats}", flush=True)

    ru = build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                    args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]

    cuts = ([dict(name=f"size>{c:g}", dim=3, thr=float(np.log(c)), keep_high=True, raw=c, kind="size")
             for c in args.size_cuts] +
            [dict(name=f"mag<{c:g}", dim=2, thr=float(c), keep_high=False, raw=c, kind="mag")
             for c in args.mag_cuts])
    names = ["__nocut__"] + [c["name"] for c in cuts]

    # sim truth, for context only -- it does not depend on the sampler
    sim = {}
    R_nc, _, _, _ = truth_selected_response(base, gh1, gh2, gmed, iso, "measured_flux_radius", -1e9, True)
    sim["__nocut__"] = R_nc
    for c in cuts:
        xcol = "measured_flux_radius" if c["kind"] == "size" else "measured_mag_auto"
        sim[c["name"]], _, _, _ = truth_selected_response(base, gh1, gh2, gmed, iso, xcol,
                                                          c["raw"], c["keep_high"])

    bundle = load_measurement_model(args.ckpt, device=device)
    res = {}   # (mode, n) -> {name: [R per repeat]}
    for n in args.n_list:
        for mode in ("mc", "qmc"):
            acc = {k: [] for k in names}
            for r in range(args.repeats):
                seed = 12345 + 1000 * r          # independent sampling streams per repeat
                out = model_selected_response(bundle, base, gh1, gh2, gmed, iso, cuts, n,
                                              args.batch_size, seed, device, qmc=(mode == "qmc"))
                for k in names:
                    acc[k].append(out[k][0])
            res[(mode, n)] = acc
            nc = np.array(acc["__nocut__"])
            print(f"  n={n:>4} {mode:>3}: nocut R={nc.mean():+.5f} sd={nc.std(ddof=1):.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 96)
    print("SCATTER of R_model across independent sampling seeds  (lower = better estimator)")
    print("=" * 96)
    for k in names:
        print(f"\n  --- {k}   (sim R = {sim[k]:+.4f}) ---")
        print(f"  {'n':>5} | {'MC mean':>10} {'MC sd':>9} | {'QMC mean':>10} {'QMC sd':>9} | "
              f"{'sd ratio':>9} {'mean shift':>11}")
        for n in args.n_list:
            a = np.array(res[("mc", n)][k]); b = np.array(res[("qmc", n)][k])
            sa, sb = a.std(ddof=1), b.std(ddof=1)
            ratio = (sa / sb) if sb > 0 else np.inf
            shift = b.mean() - a.mean()
            print(f"  {n:>5} | {a.mean():>+10.5f} {sa:>9.5f} | {b.mean():>+10.5f} {sb:>9.5f} | "
                  f"{ratio:>8.2f}x {shift:>+11.5f}")

    # Headline. NOTE: the no-cut row is USELESS as a reference -- common random numbers across the
    # two legs make it deterministic (sd = 0 at every n), so any "does QMC beat it" test is vacuous.
    # The question that matters is whether the sampling noise is small COMPARED TO THE EFFECT, so
    # report it in the units the result is quoted in: percentage points of m.
    nmax = max(args.n_list)
    print("\n" + "=" * 96)
    tight = names[-1] if len(names) > 1 else "__nocut__"
    for k in names:
        if k == "__nocut__":
            continue
        Rs = sim[k]
        print(f"  {k}: sampling noise as points of m  (effect here = "
              f"{abs(Rs/np.mean(res[('mc', nmax)][k]) - 1)*100:.2f} pts)")
        for n in args.n_list:
            a = np.array(res[("mc", n)][k]); b = np.array(res[("qmc", n)][k])
            f = lambda sd, R: 100.0 * Rs * sd / R**2
            print(f"      n={n:>4}: MC {f(a.std(ddof=1), a.mean()):.4f} pts   "
                  f"QMC {f(b.std(ddof=1), b.mean()):.4f} pts")
    print("  -> if these are orders of magnitude below the effect, n_samples is OVERKILL and the")
    print("     right move is to CUT n, regardless of which sampler wins.")
    print("CAVEAT: with only %d repeats each sd is itself uncertain by ~%.0f%%; treat a ratio near 1 "
          "as 'no difference'." % (args.repeats, 100.0 / np.sqrt(2 * (args.repeats - 1))))

    if args.output:
        np.savez(args.output, sim=np.array([sim[k] for k in names]), names=np.array(names),
                 **{f"{m}_{n}_{k}": np.array(v[k]) for (m, n), v in res.items() for k in names})
        print(f"saved {args.output}")
    print("QMC_CONVERGENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
