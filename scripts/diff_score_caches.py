#!/usr/bin/env python
"""Difference two banked score caches, block by block, on either arm.

Two score passes that differ only in a quadrature setting -- the node bank, a
finite-difference stencil, the likelihood dtype -- score the SAME objects: the drawn data
depend on `flow_seed`, `rows`, `ring` and `shape_reps`, none of which the setting touches.
Their jackknife blocks therefore align by construction, and the DIFFERENCE of their biases
is far better determined than either bias alone, because the shape noise that dominates
both cancels row for row.

That is the whole point of running the pair. Comparing the two published error bars instead
would be comparing two numbers whose noise is ~100% correlated, and would overstate the
uncertainty on the difference by an order of magnitude.

The uncut control is the cleanest place to do this: `Pi == 1` there, so both population terms
vanish identically and `ghat = (sum s) / (sum I)` with nothing else in it.

The KEPT arm needs the two population terms as well, and those are NOT in the cache --
they come from the population block, which runs on its own node bank and so does not move
with the score grid.  They are read out of the run LOGS rather than retyped, and the two
logs must agree exactly: a quadrature comparison that silently used two different `<s>_sel`
would attribute a population difference to the score pass.  Retyping them is the same
failure by hand.

Usage:
    python scripts/diff_score_caches.py A.npz B.npz
    python scripts/diff_score_caches.py A.npz B.npz --arm kept --pop-log A.out B.out
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

# Fields a QUADRATURE comparison is allowed to vary.  Everything else must match, or the two
# caches are not scoring the same objects and the paired difference is meaningless -- most
# dangerously `rows`, `flow_seed`, `ring` and `shape_reps`, which change the data themselves.
QUADRATURE_KEYS = frozenset({
    "grid_n", "grid_rmax", "grid_emax", "fd_delta", "info_delta", "grad_delta",
    "ll_dtype", "precision",
})


# `<s>_sel = [-0.000345 +/- 0.000001, +0.000035 +/- 0.000002]   |s|/sigma = ...`
# `I_sel   = [[+0.01116 +/- 0.00006, +0.00010], [+0.00011, +0.01080 +/- 0.00006]]`
# Only the central values are taken; the quoted errors are the population block's own and are
# reported separately, not propagated through a paired difference in which they are common mode.
_NUM = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
_POP_S = re.compile(r"<s>_sel\s*=\s*\[" + _NUM + r"\s*\+/-\s*" + _NUM +
                    r"\s*,\s*" + _NUM + r"\s*\+/-")
_POP_I = re.compile(r"I_sel\s*=\s*\[\[" + _NUM + r"\s*\+/-\s*" + _NUM +
                    r"\s*,\s*" + _NUM + r"\]\s*,\s*\[" + _NUM +
                    r"\s*,\s*" + _NUM + r"\s*\+/-")


def population_terms_from_log(path):
    """Read `(<s>_sel, I_sel)` back out of a run log.

    Parsed rather than retyped, and returned with the text they came from so a caller can
    show what it matched.  A log that does not carry both terms is an error, not a zero:
    silently defaulting to no correction would turn a missing population block into a
    plausible-looking uncorrected number.
    """
    text = open(path).read()
    ms, mi = _POP_S.search(text), _POP_I.search(text)
    if ms is None or mi is None:
        missing = [n for n, m in (("<s>_sel", ms), ("I_sel", mi)) if m is None]
        raise SystemExit(f"{path}: no {' and no '.join(missing)} line -- not a cut run log?")
    s_sel = np.array([float(ms.group(1)), float(ms.group(3))])
    i_sel = np.array([[float(mi.group(1)), float(mi.group(3))],
                      [float(mi.group(4)), float(mi.group(5))]])
    return s_sel, i_sel, ms.group(0), mi.group(0)


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
    ap.add_argument("--arm", choices=("uncut", "kept"), default="uncut",
                    help="which arm to difference.  'uncut' (default) is the clean control, "
                         "where Pi == 1 and both population terms vanish identically.  'kept' "
                         "is the cut estimate and needs --pop-log")
    ap.add_argument("--vs-uncut", action="store_true",
                    help="with --arm kept, also report the DIFFERENCE OF DIFFERENCES: how "
                         "much the cut's offset from its own uncut control moves.  That is "
                         "the quantity a selection correction is judged on, and it is far "
                         "less sensitive to the setting than either absolute m, because a "
                         "shift common to both arms cancels")
    ap.add_argument("--pop-log", nargs=2, metavar=("LOG_A", "LOG_B"),
                    help="the two run logs to read <s>_sel and I_sel from; required for "
                         "--arm kept, and they must agree exactly")
    args = ap.parse_args()
    if args.arm == "kept" and not args.pop_log:
        ap.error("--arm kept needs --pop-log LOG_A LOG_B (the population terms are not cached)")
    if args.pop_log and args.arm != "kept":
        ap.error("--pop-log only applies to --arm kept; the uncut arm has no population terms")
    if args.vs_uncut and args.arm != "kept":
        ap.error("--vs-uncut compares the kept arm against its uncut control; use --arm kept")

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
    suf = "_u" if args.arm == "uncut" else "_k"
    cnt = [d["cnt" + suf] for d in data]
    ns = [d["ns" + suf] for d in data]
    ni = [d["ni" + suf] for d in data]
    if any(x.shape[0] != nb for x in ns):
        sys.exit("block counts disagree with the cache key")

    s_sel = np.zeros(2)
    i_sel = np.zeros((2, 2))
    if args.arm == "kept":
        got = [population_terms_from_log(p) for p in args.pop_log]
        for path, (_, _, ls, li) in zip(args.pop_log, got):
            print(f"\n  population terms from {path}\n    {ls}\n    {li}")
        if not (np.array_equal(got[0][0], got[1][0]) and np.array_equal(got[0][1], got[1][1])):
            sys.exit("\nrefusing: the two logs carry DIFFERENT population terms, so the two "
                     "runs\ndo not share <s>_sel / I_sel and their difference is not the "
                     "score pass alone.")
        s_sel, i_sel = got[0][0], got[0][1]
        print("  the two logs agree exactly -- the difference below is the score pass alone")

    def est(n, sum_s, sum_i):
        """(5.3): the population terms are subtracted per galaxy, so they scale with `n`."""
        return np.linalg.solve(sum_i - n * i_sel, sum_s - n * s_sel)

    def estimate(k, mask):
        return est(cnt[k][mask].sum(), ns[k][mask].sum(axis=0), ni[k][mask].sum(axis=0))

    arm = ("uncut control (Pi == 1, both population terms vanish identically)"
           if args.arm == "uncut" else "KEPT arm, FULL (5.3) with both population terms")
    print(f"\ninjected g = {g}, {nb} jackknife blocks, {arm}\n")
    all_blocks = np.ones(nb, dtype=bool)
    for k, name in enumerate("AB"):
        _, sig = jackknife(lambda m, k=k: estimate(k, m), nb)
        ghat = estimate(k, all_blocks)
        print(f"  {name}: ghat = [{ghat[0]:+.6f} +/- {sig[0]:.6f}, "
              f"{ghat[1]:+.6f} +/- {sig[1]:.6f}]   "
              f"m = {100 * (ghat[0] / g - 1):+.3f}% +/- {100 * sig[0] / g:.3f}%")

    # The paired difference: form B - A INSIDE each jackknife replicate, so the shape noise
    # common to both cancels before the variance is taken.
    def diff(mask):
        return estimate(1, mask) - estimate(0, mask)

    dg = estimate(1, all_blocks) - estimate(0, all_blocks)
    _, dsig = jackknife(diff, nb)
    dm, dm_sig = 100 * dg[0] / g, 100 * dsig[0] / g
    print(f"\n  B - A: d(ghat_1) = {dg[0]:+.6f} +/- {dsig[0]:.6f}    d(m) = {dm:+.4f}%"
          f" +/- {dm_sig:.4f}%" + (f"   ({abs(dm) / dm_sig:.1f} sigma)" if dm_sig else ""))
    _, sig_a = jackknife(lambda m: estimate(0, m), nb)
    if dm_sig:
        print(f"  pairing gains a factor {100 * sig_a[0] / g / dm_sig:.1f} over A's own bar "
              f"({100 * sig_a[0] / g:.3f}%) -- that factor IS the shared shape noise.")
    else:
        print("  the difference is identically zero in every block: the same cache twice, "
              "or two runs\n  whose differing setting does not reach the score.")

    if args.vs_uncut:
        # `d(m) vs uncut` is what the cut rows are actually read on.  Both arms of the cache
        # carry their own uncut control on the SAME blocks, so the double difference is paired
        # twice over and any shift common to cut and uncut -- which is most of the grid
        # effect -- cancels before the variance is taken.
        ucnt = [d["cnt_u"] for d in data]
        uns = [d["ns_u"] for d in data]
        uni = [d["ni_u"] for d in data]

        def offset(k, mask):
            """The cut estimate minus its own uncut control, within one cache."""
            return estimate(k, mask) - np.linalg.solve(uni[k][mask].sum(axis=0),
                                                       uns[k][mask].sum(axis=0))

        print("\n  cut MINUS its own uncut control, per cache "
              "(this is the `d(m) vs uncut` column):")
        for k, name in enumerate("AB"):
            _, osig = jackknife(lambda m, k=k: offset(k, m), nb)
            o = offset(k, all_blocks)
            print(f"    {name}: d(m) vs uncut = {100 * o[0] / g:+.3f}% "
                  f"+/- {100 * osig[0] / g:.3f}%")
        do = offset(1, all_blocks) - offset(0, all_blocks)
        _, dosig = jackknife(lambda m: offset(1, m) - offset(0, m), nb)
        ddm, ddm_sig = 100 * do[0] / g, 100 * dosig[0] / g
        print(f"    B - A: {ddm:+.4f}% +/- {ddm_sig:.4f}%"
              + (f"   ({abs(ddm) / ddm_sig:.1f} sigma)" if ddm_sig else "")
              + "\n    ^ the grid moves the cut and the uncut control together, so this is"
                "\n      much smaller than either arm's own shift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
