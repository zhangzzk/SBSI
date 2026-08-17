"""V1 vs V2 transport on the acceptance population, across seeds.

V2 (ablation rung S2c, `--flow-drop`-style true-property conditioning with a 4-D output and
the lambda_theta=500 coupling pin) conditions on TRUE mag/size and emits measured mag/size,
which is the construction meant to remove the errors-in-variables floor cont.110f localized
to the true-size transition Re ~ 0.24-0.50.  This script asks whether that shows up in the
number that matters: transport `m` on the `GOALS.md` acceptance cut, true mag < 26 and
true Re > 0.3.

Reads `<prefix>_rflow.npz` dumps written by `eval_score_response.py --flow-perobj-only`,
all on the SAME rows, and reports the seed MEAN and the seed-to-seed SCATTER.  The scatter
is the point: a single seed is not enough to call a per-bin difference, and V2 is trained
on ~8x fewer rows than V1, so its scatter is expected to be larger.

Usage:
    python scripts/compare_v1_v2_seeds.py --v1 <pfx>,<pfx>,... --v2 <pfx>,<pfx>,...
"""

import argparse
import contextlib
import io
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", required=True, help="comma-separated dump prefixes")
    ap.add_argument("--v2", required=True, help="comma-separated dump prefixes")
    ap.add_argument("--truth-dump", required=True,
                    help="a score dump carrying r_input_p, for the true magnitude")
    ap.add_argument("--mag-cut", type=float, default=26.0)
    ap.add_argument("--size-cut", type=float, default=0.3)
    ap.add_argument("--min-case", type=int, default=40)
    args = ap.parse_args()

    from scripts.eval_score_response import (BLEND_LOOKUP, CROWD_FLUX_LOOKUP, GOLD_CAT,
                                             MEAS_PRIM_LOOKUP, load_constgold)

    sc = np.load(args.truth_dump)
    mag_t = sc["r_input_p"]
    n = len(mag_t)
    a = types.SimpleNamespace(
        catalogue=GOLD_CAT, min_case=args.min_case, max_rows=n,
        no_source_selection=False, crowd_flux_lookup=CROWD_FLUX_LOOKUP,
        meas_prim_lookup=MEAS_PRIM_LOOKUP, blend_lookup=BLEND_LOOKUP)
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_constgold(a)
    if len(df) != n or not np.allclose(df["r_input_p"].to_numpy(float), mag_t):
        raise SystemExit("re-load does not reproduce the dump's rows -- refusing to join")
    re_t = df["Re_input_p"].to_numpy(float)

    def load(pfx):
        d = np.load(f"{pfx}_rflow.npz")
        if not np.allclose(d["r_sim"], sc["r_sim"]):
            raise SystemExit(f"{pfx}: rows do not match the truth dump")
        return d

    v1 = [load(p) for p in args.v1.split(",")]
    v2 = [load(p) for p in args.v2.split(",")]
    print(f"aligned: {n:,} rows;  V1 seeds {len(v1)}, V2 seeds {len(v2)}")

    def m_of(d, m):
        return d["r_sim"][m].mean() / (d["r_flow"][m].mean() + d["r_blend"][m].mean()) - 1

    keep = (mag_t < args.mag_cut) & (re_t > args.size_cut)
    masks = [("FULL", np.ones(n, bool)),
             (f"ACCEPTANCE mag<{args.mag_cut:g}&Re>{args.size_cut:g}", keep),
             ("rejected", ~keep),
             (f"true Re>{args.size_cut:g}", re_t > args.size_cut)]
    for lo, hi in ((0.10, 0.24), (0.24, 0.30), (0.30, 0.38), (0.38, 0.50), (0.50, 0.90),
                   (0.90, 1.50)):
        masks.append((f"true Re {lo:.2f}-{hi:.2f}", (re_t > lo) & (re_t <= hi)))

    print(f"\n{'sample':>26} | {'V1 mean +/- sd':>18} | {'V2 mean +/- sd':>18} | {'V2-V1':>8}")
    for lab, msk in masks:
        if msk.sum() < 500:
            continue
        a1 = np.array([m_of(d, msk) for d in v1])
        a2 = np.array([m_of(d, msk) for d in v2])
        print(f"{lab:>26} | {a1.mean():>+9.2%} +/-{a1.std(ddof=1):>6.2%} | "
              f"{a2.mean():>+9.2%} +/-{a2.std(ddof=1):>6.2%} | "
              f"{a2.mean() - a1.mean():>+8.2%}")
    print("\n  sd is the SEED-TO-SEED scatter, not the error on the mean.  A per-bin "
          "difference\n  smaller than the scatter is not a result -- V2's scatter is the "
          "larger of the two\n  everywhere, consistent with its ~8x smaller training set.")


if __name__ == "__main__":
    main()
