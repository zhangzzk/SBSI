"""Merge per-seed dumps from the near-domain array job into the table + npz.

WHY THIS EXISTS. Scoring 16 seeds sequentially is ~2h20 of wall clock (setup ~8 min + ~8 min per
seed, at the full 11.67M-row population). The seeds are completely independent given the prepared
population, so they fan out perfectly: each array task rebuilds the population and scores ONE
checkpoint, and wall clock collapses to setup + one seed (~16 min). The price is rebuilding the
population in every task -- pure duplicated CPU, no correctness cost.

WHAT IS CHECKED, AND WHY IT IS NOT OPTIONAL. Fanning out is only sound if every task built the SAME
population. That is true today (the build is deterministic; the only RNG is the `--max-rows`
subsample, which is seeded), but "deterministic" is an assumption about code that changes -- a
different catalogue path, a different domain cut, or a re-ordered join would produce accumulators
that are individually fine and meaningless to average. Every dump therefore carries a fingerprint
(row count, case/index checksums, g, R_blend sum) and a copy of the sim side, and this script
REFUSES to merge unless they all agree. It also refuses on duplicate checkpoints, which would
silently double-weight a seed.

The aggregation itself is NOT reimplemented here: it calls the same `report()` the single-process
path calls, so the two cannot drift.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_selection_constgold_neardomain import _fp_equal, report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True,
                    help="glob for the per-seed JSON dumps, e.g. 'results/nd_seeds/*.json'")
    ap.add_argument("--save-npz", default="results/constgold_neardomain_table.npz")
    ap.add_argument("--expect-seeds", type=int, default=16,
                    help="refuse if fewer dumps than this are found (0 disables). Default 16: `m` "
                         "is an e-response quantity and the e-response standard is 16 seeds.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.dumps))
    if not paths:
        raise SystemExit(f"REFUSING: no dumps matched {args.dumps!r}")

    per, ref, seen = [], None, {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        ck = d["ckpt"]
        if ck in seen:
            raise SystemExit(
                f"REFUSING: checkpoint {ck} appears in both {seen[ck]} and {p}. Merging it twice "
                "would double-weight that seed and silently shrink the ensemble spread.")
        seen[ck] = p
        if ref is None:
            ref = d
        elif not _fp_equal(ref["fingerprint"], d["fingerprint"]):
            raise SystemExit(
                f"REFUSING: {p} was built from a DIFFERENT population than {seen[ref['ckpt']]}.\n"
                f"  {seen[ref['ckpt']]}: {ref['fingerprint']}\n  {p}: {d['fingerprint']}\n"
                "Averaging accumulators across different populations is meaningless. Re-run the "
                "array with one consistent configuration.")
        # `rk` (model-side keep fraction) and `rx` (the per-leg split: keep_plus/keep_minus,
        # b_plus/b_minus, sel_plus/sel_minus) both postdate the first dumps; pass None when absent so
        # an older dump set still merges, just without the mkeep column / the leg-split block.
        per.append((d["rf"], d["rb"], d.get("rk"), d.get("rx")))

    n = len(per)
    print(f"merging {n} per-seed dumps from {args.dumps}")
    print(f"  population fingerprint: {ref['fingerprint']}")
    print("  checkpoints: " + ", ".join(sorted(seen)))
    if args.expect_seeds and n < args.expect_seeds:
        raise SystemExit(
            f"REFUSING: found {n} dumps but --expect-seeds={args.expect_seeds}. A short ensemble is "
            "usually a silently failed array task, and `m` at fewer than 16 seeds is under-powered "
            "(AGENTS.md). Pass --expect-seeds 0 to override deliberately.")

    report(ref["sim"], per, ref["group_rows"], ref["cuts"], args.save_npz,
           ref["dom_mag_max"], ref["dom_re_min"])
    print("CG_ND_MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
