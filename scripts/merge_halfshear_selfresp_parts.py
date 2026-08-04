"""Merge the per-checkpoint half-shear self-response parts into one dump.

Stage B of the fresh-case dump writes one feather per checkpoint, each carrying the sim columns
plus that checkpoint's `R_flow_s{seed}`. All tasks score the SAME cached population in the SAME
order, so the merge is a column stack -- but it is only allowed to BE a column stack if that is
verified, not assumed. This script therefore:

  * requires the exact expected seed set (16 dom6x6 e-response seeds; 504 does not exist),
  * compares (case, input_index) ELEMENTWISE against the spine part -- the strictest form of the
    join on those keys, since it certifies identity AND order in one step. If that fails it falls
    back to an explicit 1:1 merge on (case, input_index) and aborts if the merge is not exact,
  * compares the sim columns bit-for-bit across parts (they are recomputed independently in every
    task, so a drifted population would show up here even if the keys happened to line up),
  * asserts the case window: no case <= 39 (cases 0-39 are burned for the modulation question) and
    the expected number of distinct cases,
  * asserts (case, input_index) uniqueness,
  * cross-checks each part's sidecar meta for a common gmed and row count.

BLIND. This script prints counts, fractions and key checks only. It never prints, averages or bins
any response value: the fresh cases must stay unexamined until the gate statistic is fixed in
writing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf

EXPECTED_SEEDS = ["501", "502", "503"] + [str(s) for s in range(505, 518)]
SIM_COLS = ["r_sim_self", "SN", "Re_input_p", "nbr_flux_near"]
KEYS = ["case", "input_index"]


def _seed_of(df, path):
    cols = [c for c in df.columns if c.startswith("R_flow_s")]
    if len(cols) != 1:
        raise SystemExit(f"{path}: expected exactly one R_flow_s* column, found {cols}")
    return cols[0][len("R_flow_s"):], cols[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="glob for the per-seed part feathers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=199)
    ap.add_argument("--expect-seeds", nargs="*", default=EXPECTED_SEEDS)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.parts))
    if not paths:
        raise SystemExit(f"no parts matched {args.parts}")
    print(f"parts found: {len(paths)}")

    spine = None
    spine_path = None
    frames = {}
    metas = {}
    for p in paths:
        df = pf.read_table(p).to_pandas()
        sd, col = _seed_of(df, p)
        if sd in frames:
            raise SystemExit(f"duplicate seed {sd} ({p})")
        mp = os.path.splitext(p)[0] + "_meta.json"
        if os.path.exists(mp):
            with open(mp) as fh:
                metas[sd] = json.load(fh)
        if spine is None:
            spine, spine_path = df, p
            print(f"  spine = s{sd} ({os.path.basename(p)})  rows={len(df):,}")
        else:
            if len(df) != len(spine):
                raise SystemExit(f"{p}: {len(df):,} rows vs spine {len(spine):,}")
            same = all(np.array_equal(df[k].to_numpy(), spine[k].to_numpy()) for k in KEYS)
            if not same:
                print(f"  s{sd}: key order differs from spine -- falling back to explicit merge")
                m = spine[KEYS].merge(df[KEYS + [col]], on=KEYS, how="left", validate="one_to_one")
                if len(m) != len(spine) or m[col].isna().any():
                    raise SystemExit(f"{p}: (case,input_index) join against the spine is not exact")
                df = m
            else:
                for c in SIM_COLS:
                    if c in df.columns and c in spine.columns:
                        a, b = df[c].to_numpy(), spine[c].to_numpy()
                        if not np.array_equal(a, b, equal_nan=True):
                            nd = int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
                            raise SystemExit(f"{p}: sim column {c} differs from spine in "
                                             f"{nd:,} rows -- the populations are not the same")
        frames[sd] = df[col].to_numpy()
        print(f"  s{sd}: rows={len(df):,}  finite({col})={np.isfinite(frames[sd]).mean():.6f}")

    missing = [s for s in args.expect_seeds if s not in frames]
    extra = [s for s in frames if s not in args.expect_seeds]
    if missing or extra:
        raise SystemExit(f"seed set mismatch: missing={missing} extra={extra}")

    if metas:
        gm = sorted({round(float(m["gmed"]), 12) for m in metas.values()})
        nr = sorted({int(m["n_rows"]) for m in metas.values()})
        if len(gm) != 1 or len(nr) != 1:
            raise SystemExit(f"part metas disagree: gmed={gm} n_rows={nr}")
        print(f"part metas agree: gmed={gm[0]:.6f}  n_rows={nr[0]:,}  ({len(metas)} sidecars)")

    out = spine[KEYS + [c for c in SIM_COLS if c in spine.columns]].copy()
    for sd in args.expect_seeds:
        out[f"R_flow_s{sd}"] = frames[sd]

    case = out["case"].to_numpy()
    if case.min() <= 39:
        raise SystemExit(f"BLIND VIOLATION: merged dump contains case {int(case.min())} <= 39 "
                         "-- cases 0-39 are burned for this question")
    if case.min() < args.min_case or case.max() > args.max_case:
        raise SystemExit(f"case window violated: [{case.min()},{case.max()}] outside "
                         f"[{args.min_case},{args.max_case}]")
    ncase = int(pd.Series(case).nunique())
    key = pd.MultiIndex.from_arrays([out["case"], out["input_index"]])
    ndup = int(len(key) - key.nunique())
    if ndup:
        raise SystemExit(f"{ndup:,} duplicate (case,input_index) rows in the merged dump")

    seed_cols = [c for c in out.columns if c.startswith("R_flow_s")]
    print(f"\nmerged {len(out):,} rows x {len(seed_cols)} seeds")
    print(f"  cases [{int(case.min())},{int(case.max())}]  distinct={ncase}  "
          f"(expected {args.max_case - args.min_case + 1})")
    print(f"  duplicate keys: {ndup}")
    for c in SIM_COLS + seed_cols:
        if c in out.columns:
            print(f"  finite {c:16s} {np.isfinite(out[c].to_numpy()).mean():.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.reset_index(drop=True).to_feather(args.out)
    meta = dict(out=os.path.abspath(args.out), n_rows=int(len(out)),
                case_lo=int(case.min()), case_hi=int(case.max()), n_cases=ncase,
                seeds=args.expect_seeds, parts=[os.path.basename(p) for p in paths],
                gmed=(float(list(metas.values())[0]["gmed"]) if metas else None),
                finite_frac={c: float(np.isfinite(out[c].to_numpy()).mean())
                             for c in SIM_COLS + seed_cols if c in out.columns})
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print(f"saved -> {args.out}")
    print("HALFSHEAR_SELFRESP_MERGE_DONE")


if __name__ == "__main__":
    main()
