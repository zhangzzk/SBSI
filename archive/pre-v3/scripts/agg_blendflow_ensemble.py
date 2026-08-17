"""Aggregate the 16-seed flow-#2 ensemble into the two statistics that are actually different.

Every flow-#2 number quoted before this was a single checkpoint, so its error bar was a lower bound
by construction. This script turns the per-seed evaluations into:

  (1) THE SPREAD OF PER-SEED METRICS -- compute the metric inside each seed, then take the spread
      across seeds. This says how much ONE run can be trusted, and it is the number to attach to any
      single-checkpoint result quoted in the past.

  (2) THE METRICS OF THE SEED-AVERAGED RESPONSE -- average `R_blend` across seeds row by row FIRST,
      then compute the metric once. This is what an ensemble prediction would actually deliver in
      deployment.

**These are not the same number and must not be conflated.** (1) is a statement about run-to-run
variability; (2) is a statement about a specific, better predictor. Averaging cancels the seed-random
part of each model's error but leaves any bias they SHARE untouched, so (2) improves on the mean of
(1) only to the extent the errors are independent. Reporting (2) with (1)'s error bar -- or quoting
(1)'s mean as if it were the ensemble -- would overstate the result in opposite directions.

The same distinction AGENTS.md draws for flow #1: form the quantity inside each seed, then take the
spread across seeds; never build it from ensemble means and propagate one term's scatter.

Reads the npz files written by `scripts/eval_blend_flow.py --save-npz`.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from eval_blend_flow import MAG_EDGES, SEP_EDGES, SIZE_EDGES, cluster_sem  # noqa: E402

K_EDGES = [1, 2, 3, 4, 6, 9, 99]


def cluster_sem_fast(vals, inv, n_prim, mask):
    """`eval_blend_flow.cluster_sem` with the primary index precomputed.

    Algebraically IDENTICAL -- residuals are summed within each primary before squaring, and
    primaries absent from the mask contribute a zero term either way. The only change is that the
    caller supplies the global `pid -> 0..n_prim-1` map instead of this function re-deriving it with
    a sort on every call. That re-derivation is what made the aggregate over 16 seeds x 5 metrics x
    ~6 bins intractable; `test_cluster_sem_fast_matches_the_original` pins the equivalence.
    """
    v = vals[mask]
    n = v.size
    if n < 2:
        return np.nan
    per_primary = np.bincount(inv[mask], weights=v - v.mean(), minlength=n_prim)
    return float(np.sqrt((per_primary ** 2).sum()) / n)


def chi2_dof(truth, pred, key, edges, inv, n_prim, min_n=200):
    """chi2/dof of the binned mean prediction against the label, cluster-robust errors."""
    chi2, dof = 0.0, 0
    for i in range(len(edges) - 1):
        m = (key >= edges[i]) & (key < edges[i + 1])
        if int(m.sum()) < min_n:
            continue
        a = float(truth[m].mean())
        s = cluster_sem_fast(truth, inv, n_prim, m)
        if not np.isfinite(s) or s <= 0:
            continue
        chi2 += ((float(pred[m].mean()) - a) / s) ** 2
        dof += 1
    return chi2 / max(dof, 1), dof


def summed_bias(truth, pred, inv, n_prim):
    """Percent bias of R_blend SUMMED over each primary's neighbours -- the form `m` uses."""
    n = n_prim
    st = np.bincount(inv, weights=truth, minlength=n)
    sp = np.bincount(inv, weights=pred, minlength=n)
    a = st.mean()
    return (sp.mean() / a - 1) * 100 if a else np.nan, st.std(ddof=1) / np.sqrt(n) / abs(a) * 100


def metrics(truth, pred, d, key_re, key_mag, key_k, inv, n_prim):
    out = {}
    out["sep"], _ = chi2_dof(truth, pred, d, SEP_EDGES, inv, n_prim)
    out["Re_p"], _ = chi2_dof(truth, pred, key_re, SIZE_EDGES, inv, n_prim)
    out["r_p"], _ = chi2_dof(truth, pred, key_mag, MAG_EDGES, inv, n_prim)
    out["k"], _ = chi2_dof(truth, pred, key_k, K_EDGES, inv, n_prim)
    out["summed%"], _ = summed_bias(truth, pred, inv, n_prim)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True, help="e.g. '<cache>/eval_ens_s*.npz'")
    ap.add_argument("--expect", type=int, default=16, help="how many seeds SHOULD be present")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"REFUSING: no files matched {args.glob}")
    seeds = [int(re.search(r"_s(\d+)\.npz$", p).group(1)) for p in paths]
    print(f"found {len(paths)} seed evaluations: {seeds}")
    if len(paths) != args.expect:
        # Not fatal, but it MUST be visible: a silently short ensemble would report a tighter or
        # looser spread than the design and nothing downstream could tell.
        print(f"  *** WARNING: expected {args.expect} seeds, found {len(paths)}. Every number below "
              f"describes the {len(paths)} present, NOT the intended ensemble. ***")

    ref = np.load(paths[0])
    truth, pid = ref["truth"], ref["pid"]
    dist, tre, tmag, emu = ref["dist"], ref["tre"], ref["tmag"], ref["emu"]
    _, kinv = np.unique(pid, return_inverse=True)
    n_prim = int(kinv.max()) + 1
    kcol = np.bincount(kinv)[kinv].astype(float)

    flows = []
    for p, sd in zip(paths, seeds):
        z = np.load(p)
        # The rows must be the same rows in the same order, or averaging across seeds would mix
        # different pairs together. Checked, not assumed.
        if not (np.array_equal(z["pid"], pid) and np.allclose(z["truth"], truth, atol=0, rtol=0)):
            raise SystemExit(
                f"REFUSING: {p} does not have the same rows as {paths[0]}. The per-row average "
                "across seeds is only meaningful if every file describes the identical pair list.")
        if not np.allclose(z["emu"], emu, rtol=0, atol=0):
            raise SystemExit(f"REFUSING: {p} has a different emulator prediction; the emulator is "
                             "deterministic, so this means the runs are not comparable.")
        flows.append(z["flow"])
    F = np.vstack(flows)

    # ---- (1) per-seed metrics, spread ACROSS seeds ----
    per = [metrics(truth, F[i], dist, tre, tmag, kcol, kinv, n_prim) for i in range(len(paths))]
    keys = ["summed%", "sep", "Re_p", "r_p", "k"]
    print(f"\n{'='*104}")
    print("(1) PER-SEED metrics -- the metric is formed INSIDE each seed, then the spread is taken")
    print("    across seeds. This is the error bar that belongs on any single-checkpoint result.")
    print(f"{'='*104}")
    print(f"  {'seed':>6}" + "".join(f"{k:>12}" for k in keys))
    for sd, mrow in zip(seeds, per):
        print(f"  {sd:>6}" + "".join(f"{mrow[k]:>12.3f}" for k in keys))
    print("  " + "-" * 70)
    stats = {}
    for k in keys:
        v = np.array([m[k] for m in per], float)
        stats[k] = (v.mean(), v.std(ddof=1), v.std(ddof=1) / np.sqrt(len(v)))
    print(f"  {'mean':>6}" + "".join(f"{stats[k][0]:>12.3f}" for k in keys))
    print(f"  {'sd':>6}" + "".join(f"{stats[k][1]:>12.3f}" for k in keys))
    print(f"  {'sem':>6}" + "".join(f"{stats[k][2]:>12.3f}" for k in keys))

    # ---- (2) metrics of the seed-AVERAGED response ----
    fbar = F.mean(axis=0)
    ens = metrics(truth, fbar, dist, tre, tmag, kcol, kinv, n_prim)
    emum = metrics(truth, emu, dist, tre, tmag, kcol, kinv, n_prim)
    print(f"\n{'='*104}")
    print("(2) The SEED-AVERAGED response -- R_blend averaged across seeds row by row, THEN scored.")
    print("    This is the predictor an ensemble would actually deploy.")
    print(f"{'='*104}")
    print(f"  {'model':>22}" + "".join(f"{k:>12}" for k in keys))
    print(f"  {'ensemble mean':>22}" + "".join(f"{ens[k]:>12.3f}" for k in keys))
    print(f"  {'mean of single seeds':>22}" + "".join(f"{stats[k][0]:>12.3f}" for k in keys))
    print(f"  {'BlendEMU':>22}" + "".join(f"{emum[k]:>12.3f}" for k in keys))

    # ---- how much of the single-seed error is seed-random rather than shared ----
    print(f"\n{'='*104}")
    print("PER-PAIR SEED SCATTER of R_blend itself")
    print(f"{'='*104}")
    sd_pair = F.std(axis=0, ddof=1)
    print(f"  mean |R_blend| = {np.abs(fbar).mean():.5f}")
    print(f"  mean per-pair seed sd = {sd_pair.mean():.5f} "
          f"({100*sd_pair.mean()/max(np.abs(fbar).mean(),1e-12):.1f}% of the mean response)")
    print(f"  seed sd of the POPULATION mean response = {F.mean(axis=1).std(ddof=1):.6f} "
          f"on a mean of {fbar.mean():.5f} "
          f"({100*F.mean(axis=1).std(ddof=1)/abs(fbar.mean()):.2f}%)")
    print("\n  The first is per-pair noise, which the ensemble average suppresses by ~sqrt(N_seeds).")
    print("  The second is what survives into a POPULATION average and therefore into `m`: seed")
    print("  scatter that is coherent across pairs is NOT reduced by having many pairs, only by")
    print("  having many seeds. If it is small, single-seed `m` was never seed-limited; if it is")
    print("  comparable to the quoted errors, every past single-checkpoint `m` was understated.")
    print("\nAGG_BLENDFLOW_ENSEMBLE_DONE")


if __name__ == "__main__":
    main()
