"""Split the Gold-V2 per-seed constgold dumps into the GOALS.md acceptance population.

`figv2_fig3_bias_true_neighbours.png` reports ensemble m = -0.46 +- 0.35% over 8 seeds of the
coupling-pinned tabular flow, and every seed sits inside +-1.2%.  That figure is computed on the
FULL, UNCUT constgold population (its own annotation, R_sim = 0.4534, is the global number).
cont.161 showed for V1 that a global number of that kind can be a CANCELLATION -- accepted rows
biased one way, rejected rows the other, the two averaging to something that looks like closure.
This script tests whether the same is true here, using the SAME dumps the figure is drawn from,
so nothing but the row split changes:

    m(band) = <r_sim>_band / (<R_flow>_band + <R_blend>_band) - 1

The dumps carry true magnitude but not true size, so `Re_input_p` is joined from the constgold
catalogue on (case, input_index) by sorted key lookup.  Every dump row must find exactly one
catalogue row or the script aborts -- a partial join would silently redefine the population,
which is the failure mode that made an earlier no-source-selection run uninterpretable.

Row order is verified identical across seeds before the join is reused, again by abort rather
than by assumption.

Usage:
    python scripts/split_v2_dumps_acceptance.py --dump-dir <dir> --seeds 501,502,...
"""

import argparse
import glob
import os
import time

import numpy as np
import pyarrow.feather as pf
import pyarrow.ipc as ipc
import pyarrow as pa

CAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
       "constant_response_catalogue_train.feather")
ACC_RE_MIN, ACC_MAG_MAX = 0.30, 26.0


def load_catalogue_size(min_case):
    """(key, Re) for every catalogue row with case >= min_case, key sorted ascending."""
    cases, idxs, res = [], [], []
    with ipc.open_file(CAT) as r:
        cols = ["case", "input_index", "Re_input_p"]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[b["case"] >= min_case]
            if len(b):
                cases.append(b["case"].to_numpy(np.int64))
                idxs.append(b["input_index"].to_numpy(np.int64))
                res.append(b["Re_input_p"].to_numpy(np.float64))
    case = np.concatenate(cases); idx = np.concatenate(idxs); re = np.concatenate(res)
    stride = int(idx.max()) + 1
    key = case * stride + idx
    o = np.argsort(key, kind="stable")
    return key[o], re[o], stride


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_constgold_dumps")
    ap.add_argument("--seeds", default="501,502,503,505,506,507,508,509")
    ap.add_argument("--min-case", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()

    seeds = [s.strip() for s in args.seeds.split(",")]
    paths = [os.path.join(args.dump_dir, f"v2_perobj_s{s}.feather") for s in seeds]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing dumps: {missing}")

    ckey, cre, stride = load_catalogue_size(args.min_case)
    print(f"catalogue: {len(ckey):,} rows with case >= {args.min_case}  ({time.time()-t0:.1f}s)",
          flush=True)

    ref_key = None
    rows = []
    for s, p in zip(seeds, paths):
        t = pf.read_table(p)
        d = {c: t.column(c).to_numpy(zero_copy_only=False) for c in
             ("case", "input_index", "r_input_p", "r_sim", "R_flow", "R_blend")}
        key = d["case"].astype(np.int64) * stride + d["input_index"].astype(np.int64)
        if ref_key is None:
            pos = np.searchsorted(ckey, key)
            if pos.max() >= len(ckey) or not np.array_equal(ckey[pos], key):
                raise SystemExit(f"seed {s}: dump rows not found in the catalogue -- refusing to join")
            re_t = cre[pos]
            ref_key = key
            print(f"joined true size for {len(key):,} rows, all matched  ({time.time()-t0:.1f}s)",
                  flush=True)
        elif not np.array_equal(key, ref_key):
            raise SystemExit(f"seed {s}: row order differs from the first dump -- refusing to reuse the join")

        mag = d["r_input_p"]
        acc = (re_t > ACC_RE_MIN) & (mag < ACC_MAG_MAX)
        out = {}
        for name, sel in (("GLOBAL", np.ones(len(key), bool)), ("ACCEPTED", acc),
                          ("REJECTED", ~acc)):
            rs = d["r_sim"][sel].mean()
            rm = d["R_flow"][sel].mean() + d["R_blend"][sel].mean()
            out[name] = (rs / rm - 1) * 100
            out[name + "_R"] = (rs, d["R_flow"][sel].mean(), d["R_blend"][sel].mean())
        rows.append((s, out, int(acc.sum()), len(key)))
        print(f"  s{s}: GLOBAL {out['GLOBAL']:+6.2f}%   ACCEPTED {out['ACCEPTED']:+6.2f}%   "
              f"REJECTED {out['REJECTED']:+7.2f}%   ({time.time()-t0:.1f}s)", flush=True)
        del d, t

    print("\n" + "=" * 78)
    print("Gold-V2 dumps (same rows as figv2_fig3), split on GOALS.md true cut Re>0.3 & mag<26")
    print("=" * 78)
    n, na = rows[0][3], rows[0][2]
    print(f"N = {n:,}   accepted {na:,} ({na/n:.1%})   rejected {n-na:,} ({1-na/n:.1%})\n")
    print(f"  {'band':>9} {'ensemble m':>12} {'seed sd':>9} {'sem':>8}   "
          f"{'<R_sim>':>8} {'<R_flow>':>9} {'<R_blend>':>10}")
    for band in ("GLOBAL", "ACCEPTED", "REJECTED"):
        v = np.array([r[1][band] for r in rows])
        rs, rf, rb = rows[0][1][band + "_R"]
        print(f"  {band:>9} {v.mean():>+11.2f}% {v.std(ddof=1):>8.2f}% "
              f"{v.std(ddof=1)/np.sqrt(len(v)):>7.2f}%   {rs:>8.4f} {rf:>9.4f} {rb:>10.4f}")
    print("\n  <R_flow> shown for the FIRST seed only; <R_sim>/<R_blend> are seed-independent.")
    print("  GLOBAL should reproduce the published figure; if ACCEPTED and REJECTED straddle it,\n"
          "  the figure's closure is a cancellation and not a statement about the deliverable.")


if __name__ == "__main__":
    main()
