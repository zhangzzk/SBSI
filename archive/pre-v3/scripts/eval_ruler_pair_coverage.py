"""What fraction of the PAIRS the deployed `R_blend` sums over does the ruler actually see?

WHY THIS EXISTS
---------------
`scripts/eval_rblend_ruler_coverage.py` shows the RADIAL coverage is nearly complete (7" holds ~98%
of the emulator's 10" sum). But a second, larger hole opened while measuring it: on the half-shear
input fields the emulator's summed `<R_blend>` inside 7" is ~0.19 per in-domain primary, whereas the
same emulator summed over the pairs the `det_meas_ngmix_ap7_*` catalogue ANNOTATES gives ~0.072 --
and the per-pair mean is the same (0.0174) in both. So the difference is purely the NUMBER of
neighbours: the ap7 catalogue carries 4.16 rows per primary inside 7" while the input field contains
several times that.

AGENTS.md already warns about exactly this ("only represents the intended all-neighbour scene if the
source blendemu catalogue was built with `k` large enough to cover that aperture"). This script
measures it instead of assuming it, and characterises WHICH neighbours are missing (they are
presumably the faintest, if the annotation is k-capped by distance rank).

Counts the true neighbours inside 7" of each in-domain primary straight from the input field
(KD-tree on RA/DEC), and compares with the annotated row count per primary in the ruler dump, for the
same (case, input_index) primaries.

FIREWALL: half-shear input fields + the half-shear ruler dump. constgold is never opened. Nothing is
trained, fitted or tuned; this is pure bookkeeping.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

HSBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
TILE = "tile180.0_-0.5"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--base", default=HSBASE)
    ap.add_argument("--sign", default="0.05")
    ap.add_argument("--ruler-npz", required=True)
    ap.add_argument("--radius", type=float, default=7.0, help="the ruler catalogue's annotation radius")
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    args = ap.parse_args()

    d = np.load(args.ruler_npz, allow_pickle=True)
    case_r = d["case"].astype(np.int64)
    sel = np.isin(case_r, np.array(args.cases, np.int64))
    key_r = (case_r[sel] << 32) + d["input_index"][sel].astype(np.int64)
    dist_r = d["dist"][sel].astype(float)
    print(f"ruler rows in cases {args.cases}: {sel.sum():,}   "
          f"annotated separation: max={dist_r.max():.4f}\"  median={np.median(dist_r):.3f}\"",
          flush=True)
    ann = pd.Series(1, index=key_r).groupby(level=0).sum()
    del d

    tot_ann = tot_true = 0
    magparts = []
    for case in args.cases:
        fp = f"{args.base}/case{case}_{args.sign}/real0/catalogues/input/gals_info_{TILE}.feather"
        if not os.path.exists(fp):
            print(f"case{case}: MISSING {fp}")
            continue
        t = pf.read_table(fp).to_pandas()
        ra = t["RA_input"].to_numpy(float)
        dec = t["DEC_input"].to_numpy(float)
        mag = t["r_input"].to_numpy(float)
        re = t["Re_input"].to_numpy(float)
        idx = t["index_input"].to_numpy(np.int64)
        # small field -> a local tangent-plane approximation is exact to <1e-4 at these separations
        cosd = np.cos(np.deg2rad(np.median(dec)))
        xy = np.column_stack([(ra - ra.mean()) * 3600.0 * cosd, (dec - dec.mean()) * 3600.0])
        tree = cKDTree(xy)
        pairs = tree.query_ball_point(xy, r=args.radius)
        ncount = np.array([len(p) - 1 for p in pairs])          # exclude self

        keys = (np.int64(case) << 32) + idx
        a = ann.reindex(keys).to_numpy(float)
        have = np.isfinite(a) & (a > 0)                          # the primaries the ruler actually uses
        dom = (mag < args.true_mag_max) & (re > args.true_re_min)
        m = have & dom
        print(f"  case{case}: {len(t):,} input gals | ruler primaries in-domain {m.sum():,} | "
              f"TRUE neighbours <{args.radius}\" per primary = {ncount[m].mean():.3f} | "
              f"ANNOTATED rows per primary = {a[m].mean():.3f} | "
              f"pair coverage = {a[m].sum()/ncount[m].sum():.1%}", flush=True)
        tot_ann += a[m].sum()
        tot_true += ncount[m].sum()

        # which neighbours go missing? compare the mag distribution of ALL neighbours of ruler
        # primaries against the annotated count -- if the annotation is a distance-rank cap, the
        # dropped ones are the FAR ones; if it is a flux cut, they are the FAINT ones.
        sub = np.flatnonzero(m)[:20000]
        nb_mag, nb_d = [], []
        for i in sub:
            for j in pairs[i]:
                if j == i:
                    continue
                nb_mag.append(mag[j])
                nb_d.append(np.hypot(*(xy[j] - xy[i])))
        magparts.append(pd.DataFrame({"mag": nb_mag, "d": nb_d}))

    print(f"\nOVERALL pair coverage of the {args.radius}\" ruler aperture: "
          f"{tot_ann:,.0f} annotated / {tot_true:,.0f} true = {tot_ann/tot_true:.1%}")

    nb = pd.concat(magparts, ignore_index=True)
    print(f"\n[TRUE neighbour population inside {args.radius}\" of ruler primaries "
          f"(sample N={len(nb):,})]")
    print(f"  {'neighbour r mag':>18} {'frac':>8} {'cum frac':>9}  "
          f"(the ANNOTATED share is {tot_ann/tot_true:.1%} of all of them)")
    edges = [0, 22, 24, 25, 26, 27, 28, 29, 99]
    c = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        f = float(((nb["mag"] >= lo) & (nb["mag"] < hi)).mean())
        c += f
        print(f"  {f'[{lo},{hi})':>18} {f:>8.3%} {c:>9.3%}")
    print(f"\n[TRUE neighbour separation distribution]")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 7)):
        f = float(((nb["d"] >= lo) & (nb["d"] < hi)).mean())
        lab = '[{},{})"'.format(lo, hi)
        print(f"  {lab:>18} {f:>8.3%}")
    print("PAIR_COVERAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
