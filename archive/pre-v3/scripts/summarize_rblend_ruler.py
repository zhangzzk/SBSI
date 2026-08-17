"""Compare `eval_rblend_gap.py` runs across shear legs and emulators, with PAIRED statistics.

WHY A SEPARATE SUMMARISER. The ruler prints a `sem` column, but that is the sem of TRUTH, which is
the wrong error for the quantity actually of interest. Truth and prediction are evaluated on the
SAME rows, so the emulator's error is a paired difference and its error is the scatter of
`pred - truth`, not of `truth`. And when two emulators are compared, truth cancels EXACTLY -- that
comparison carries almost no noise at all and deserves to be reported as the sharp number it is.

WHAT IT SETTLES. 2026-08-05a left the emulator neither convicted nor exonerated because the g=0.05
ruler pins blend response only to +-6.2%. The g=0.2 leg carries 4x the shear signal at similar shape
noise. Two things must be checked before that extra precision may be used:

  (1) LINEARITY. R_blend at g=0.2 is the same quantity as at g=0.05 only if the response is linear
      out to 0.2. Comparing <truth> between the legs tests exactly that, and it is a real risk --
      0.2 is a large shear.
  (2) SELECTION. The both-detected requirement selects differently at g=0.2, because shear changes
      which objects are detected. So the two legs do NOT hold the same rows. This script therefore
      reports both the full-sample numbers and the numbers on the INTERSECTION of (case,
      input_index), where the populations are identical by construction and the leg is the only
      difference left.

APERTURE FAMILIES MUST NOT BE MIXED. A 7"-capped (`ap7`) run and a non-ap7 run see different pair
lists, and AGENTS.md's pair-list rule says that handicaps whichever model relies on the excluded
pairs. This script will compare whatever npz files it is given -- it cannot detect the family from
the file -- so the CALLER is responsible for passing same-family runs. The job script enforces it.

FIREWALL: reads only ruler outputs, which are built from half-shear legs. No constgold, no fitting.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in ("truth", "pred", "case", "input_index", "distance")}
    good = np.isfinite(d["truth"]) & np.isfinite(d["pred"])
    return {k: v[good] for k, v in d.items()}


def paired(truth, pred, label, note=""):
    """Mean emulator error with the PAIRED sem (scatter of pred-truth), not truth's own sem."""
    diff = pred - truth
    sem = diff.std(ddof=1) / np.sqrt(len(diff))
    rel = 100 * (pred.mean() / truth.mean() - 1) if truth.mean() else np.nan
    relsem = 100 * sem / abs(truth.mean()) if truth.mean() else np.nan
    print(f"  {label:<34}{truth.mean():>9.4f}{pred.mean():>10.4f}{diff.mean():>+9.4f}"
          f"{rel:>+9.2f}%{relsem:>8.2f}%{abs(diff.mean()) / sem:>7.1f}s{len(truth):>12,}  {note}")
    return diff.mean(), sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g005", nargs=2, required=True, metavar=("V21_NPZ", "FID_NPZ"))
    ap.add_argument("--g02", nargs=2, required=True, metavar=("V21_NPZ", "FID_NPZ"))
    ap.add_argument("--r-blend", type=float, default=0.11786, help="V2.1 emulator <R_blend>")
    ap.add_argument("--r-sim", type=float, default=0.99027)
    ap.add_argument("--r-flow", type=float, default=0.8647)
    args = ap.parse_args()

    runs = {("g=0.05", "V2.1"): load(args.g005[0]), ("g=0.05", "fiducial"): load(args.g005[1]),
            ("g=0.20", "V2.1"): load(args.g02[0]), ("g=0.20", "fiducial"): load(args.g02[1])}

    print(f"\n{'='*118}\nEMULATOR vs HALF-SHEAR TRUTH, paired errors (ap7 family throughout)\n{'='*118}")
    print(f"  {'leg / emulator':<34}{'truth':>9}{'emu':>10}{'diff':>9}{'rel':>10}{'+-':>8}"
          f"{'sig':>7}{'N':>12}")
    for (leg, emu), d in runs.items():
        paired(d["truth"], d["pred"], f"{leg}  {emu}")

    # (1) LINEARITY. Same quantity at both legs? Compare truth on the rows BOTH legs contain, so the
    # selection difference below cannot masquerade as a nonlinearity.
    a, b = runs[("g=0.05", "V2.1")], runs[("g=0.20", "V2.1")]
    ka = a["case"].astype(np.int64) * 1_000_003 + a["input_index"].astype(np.int64)
    kb = b["case"].astype(np.int64) * 1_000_003 + b["input_index"].astype(np.int64)
    common = np.intersect1d(ka, kb)
    ia = np.isin(ka, common)
    ib = np.isin(kb, common)
    ta, tb = a["truth"][ia], b["truth"][ib]
    # paired at the ROW level requires the same order; sort both by key
    oa = np.argsort(ka[ia]); ob = np.argsort(kb[ib])
    ta, tb = ta[oa], tb[ob]
    dt = tb - ta
    print(f"\n{'='*118}\n(1) LINEARITY -- is g=0.2 the same quantity as g=0.05?\n{'='*118}")
    print(f"  matched rows in BOTH legs: {len(common):,} "
          f"(of {len(ka):,} at g=0.05, {len(kb):,} at g=0.2)")
    print(f"  <truth> g=0.05 = {ta.mean():.4f}   g=0.20 = {tb.mean():.4f}   "
          f"difference = {dt.mean():+.5f} +- {dt.std(ddof=1)/np.sqrt(len(dt)):.5f} "
          f"({abs(dt.mean())/(dt.std(ddof=1)/np.sqrt(len(dt))):.1f} sigma)")
    print("  A null here means blend response is LINEAR to g=0.2 and the 4x-signal leg is usable.")

    print(f"\n{'='*118}\n(2) SELECTION -- the legs do not hold the same rows\n{'='*118}")
    print(f"  g=0.05 only: {int((~np.isin(ka, common)).sum()):,}    "
          f"g=0.20 only: {int((~np.isin(kb, common)).sum()):,}")
    print("  Shear changes which objects are detected, so the both-detected cut selects differently.")
    for (leg, emu), d in runs.items():
        k = d["case"].astype(np.int64) * 1_000_003 + d["input_index"].astype(np.int64)
        m = np.isin(k, common)
        paired(d["truth"][m], d["pred"][m], f"{leg}  {emu}", note="[matched rows only]")

    print(f"\n{'='*118}\n(3) HEAD-TO-HEAD: V2.1 emulator vs fiducial, IDENTICAL rows (truth cancels)"
          f"\n{'='*118}")
    for leg in ("g=0.05", "g=0.20"):
        dp = runs[(leg, "V2.1")]["pred"] - runs[(leg, "fiducial")]["pred"]
        print(f"  {leg}: V2.1 - fiducial = {dp.mean():+.5f} +- {dp.std(ddof=1)/np.sqrt(len(dp)):.5f}"
              f"   ({'V2.1 predicts LESS' if dp.mean() < 0 else 'V2.1 predicts MORE'} blend response)")

    print(f"\n{'='*118}\n(4) WHAT IT WOULD DO TO m, IF the per-pair relative error carried over to "
          f"the SUMMED R_blend\n{'='*118}")
    print("  UNVERIFIED SCALING: the ruler is one pair per primary; R_blend sums over the aperture.")
    print(f"  baseline: R_sim={args.r_sim:.5f}  R_flow={args.r_flow:.4f}  R_blend={args.r_blend:.5f}"
          f"  -> m={100*(args.r_sim/(args.r_flow+args.r_blend)-1):+.3f}%")
    for (leg, emu), d in runs.items():
        if emu != "V2.1":
            continue
        diff = d["pred"] - d["truth"]
        sem = diff.std(ddof=1) / np.sqrt(len(diff))
        for lab, delta in (("central", diff.mean()), ("+1sig", diff.mean() + sem),
                           ("-1sig", diff.mean() - sem)):
            fac = d["truth"].mean() / (d["truth"].mean() + delta)
            rb = args.r_blend * fac
            print(f"    {leg} {lab:>8}: R_blend -> {rb:.5f}   "
                  f"m -> {100*(args.r_sim/(args.r_flow+rb)-1):+.3f}%")
    print("\nRBLEND_RULER_SUMMARY_DONE", flush=True)


if __name__ == "__main__":
    main()
