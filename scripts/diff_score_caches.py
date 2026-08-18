#!/usr/bin/env python
"""Difference the UNCUT control of two banked score caches, block by block.

Two score passes that differ only in a quadrature setting -- the node bank, a
finite-difference stencil, the likelihood dtype -- score the SAME objects: the drawn data
depend on `flow_seed`, `rows`, `ring` and `shape_reps`, none of which the setting touches.
Their jackknife blocks therefore align by construction, and the DIFFERENCE of their biases
is far better determined than either bias alone, because the shape noise that dominates
both cancels row for row.

That is the whole point of running the pair. Comparing the two published error bars instead
would be comparing two numbers whose noise is ~100% correlated, and would overstate the
uncertainty on the difference by an order of magnitude.

The uncut control is the clean place to do this: `Pi == 1` there, so both population terms
vanish identically and `ghat = (sum s) / (sum I)` with nothing else in it.

Usage:
    python scripts/diff_score_caches.py A.npz B.npz
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# Fields a QUADRATURE comparison is allowed to vary.  Everything else must match, or the two
# caches are not scoring the same objects and the paired difference is meaningless -- most
# dangerously `rows`, `flow_seed`, `ring` and `shape_reps`, which change the data themselves.
QUADRATURE_KEYS = frozenset({
    "grid_n", "grid_rmax", "grid_emax", "fd_delta", "info_delta", "grad_delta",
    "ll_dtype", "precision",
})


def ratio_estimate(ns, ni):
    """ghat from summed per-block score and information, as the estimator forms it."""

    return np.linalg.solve(ni.sum(axis=0), ns.sum(axis=0))


def jackknife(fn, n_blocks):
    """Delete-one jackknife of `fn(mask)`; returns (mean of replicates, sigma)."""

    reps = np.stack([fn(np.arange(n_blocks) != b) for b in range(n_blocks)])
    factor = (n_blocks - 1) / n_blocks
    return reps.mean(axis=0), np.sqrt(factor * ((reps - reps.mean(axis=0)) ** 2).sum(axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_a")
    ap.add_argument("cache_b")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="report a differing non-quadrature key instead of refusing.  The "
                         "difference is then NOT paired and its error bar is wrong; use only "
                         "to inspect two unrelated caches")
    args = ap.parse_args()

    names = [args.cache_a, args.cache_b]
    data = [np.load(p, allow_pickle=True) for p in names]
    keys = [json.loads(str(d["key"])) for d in data]

    differing = {k for k in set(keys[0]) | set(keys[1])
                 if keys[0].get(k) != keys[1].get(k)}
    print(f"A: {names[0]}\nB: {names[1]}\n")
    for k in sorted(differing):
        flag = "quadrature" if k in QUADRATURE_KEYS else "*** NOT A QUADRATURE KEY ***"
        print(f"  differs: {k:12s} {keys[0].get(k)!r} -> {keys[1].get(k)!r}   {flag}")
    if not differing:
        print("  the two keys are identical -- nothing to compare")
    bad = differing - QUADRATURE_KEYS
    if bad and not args.allow_mismatch:
        sys.exit(f"\nrefusing: {sorted(bad)} differ, so the two runs do not score the same "
                 f"objects and their blocks do not align.  Pass --allow-mismatch to override.")

    g = float(keys[0]["closure_g"])
    nb = int(keys[0]["jk_blocks"])
    ns = [d["ns_u"] for d in data]
    ni = [d["ni_u"] for d in data]
    if any(x.shape[0] != nb for x in ns):
        sys.exit("block counts disagree with the cache key")

    print(f"\ninjected g = {g}, {nb} jackknife blocks, uncut control (Pi == 1)\n")
    for name, s, i in zip("AB", ns, ni):
        _, sig = jackknife(lambda m, s=s, i=i: ratio_estimate(s[m], i[m]), nb)
        ghat = ratio_estimate(s, i)
        print(f"  {name}: ghat = [{ghat[0]:+.6f} +/- {sig[0]:.6f}, "
              f"{ghat[1]:+.6f} +/- {sig[1]:.6f}]   "
              f"m = {100 * (ghat[0] / g - 1):+.3f}% +/- {100 * sig[0] / g:.3f}%")

    # The paired difference: form B - A INSIDE each jackknife replicate, so the shape noise
    # common to both cancels before the variance is taken.
    def diff(mask):
        return (ratio_estimate(ns[1][mask], ni[1][mask])
                - ratio_estimate(ns[0][mask], ni[0][mask]))

    dg = ratio_estimate(ns[1], ni[1]) - ratio_estimate(ns[0], ni[0])
    _, dsig = jackknife(diff, nb)
    dm, dm_sig = 100 * dg[0] / g, 100 * dsig[0] / g
    print(f"\n  B - A: d(ghat_1) = {dg[0]:+.6f} +/- {dsig[0]:.6f}    d(m) = {dm:+.4f}%"
          f" +/- {dm_sig:.4f}%" + (f"   ({abs(dm) / dm_sig:.1f} sigma)" if dm_sig else ""))
    _, sig_a = jackknife(lambda m: ratio_estimate(ns[0][m], ni[0][m]), nb)
    if dm_sig:
        print(f"  pairing gains a factor {100 * sig_a[0] / g / dm_sig:.1f} over A's own bar "
              f"({100 * sig_a[0] / g:.3f}%) -- that factor IS the shared shape noise.")
    else:
        print("  the difference is identically zero in every block: the same cache twice, "
              "or two runs\n  whose differing setting does not reach the score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
