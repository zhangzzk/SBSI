"""Per-(case, input_index) TRUTH nearest-neighbour distance (arcsec).

For a genuinely blend-free isolation cut that does NOT depend on the emulator: for each
input-field galaxy, the distance to its nearest neighbour of ANY magnitude, and to its
nearest BRIGHTER neighbour (r_neighbour < r_self). Keyed by (case, input_index) so it joins
the constant/det_meas catalogues like the blend/crowd lookups.

Columns: case, input_index, nn_dist_any, nn_dist_bright (arcsec; inf if none within MAXR).
"""
import argparse, os
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

BASE_CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
TILE = "tile180.0_-0.5"
MAXR = 15.0 / 3600.0   # deg; brighter-neighbour search radius (beyond this = effectively isolated)


def input_feather(case, sign, base):
    return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.0")
    ap.add_argument("--base", default=BASE_CONST)
    args = ap.parse_args()

    parts = []
    for c in args.cases:
        fp = input_feather(c, args.sign, args.base)
        if not os.path.exists(fp):
            print(f"case{c}: MISSING {fp}", flush=True); continue
        t = pf.read_table(fp, columns=["index_input", "RA_input", "DEC_input", "r_input"]).to_pandas()
        pos = t[["RA_input", "DEC_input"]].to_numpy(float)
        rmag = t["r_input"].to_numpy(float)
        n = len(t)
        tree = cKDTree(pos)
        # nearest neighbour of ANY magnitude (k=2: self + nearest other)
        dd, _ = tree.query(pos, k=2)
        nn_any = dd[:, 1] * 3600.0
        # nearest BRIGHTER neighbour within MAXR (a bright close neighbour blends most)
        nn_bright = np.full(n, np.inf)
        pairs = tree.query_pairs(r=MAXR, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) * 3600.0
            jb = rmag[j] < rmag[i]           # j brighter than i -> candidate for i
            np.minimum.at(nn_bright, i[jb], d[jb])
            ib = rmag[i] < rmag[j]           # i brighter than j -> candidate for j
            np.minimum.at(nn_bright, j[ib], d[ib])
        parts.append(pd.DataFrame({
            "case": c, "input_index": t["index_input"].to_numpy(np.int64),
            "nn_dist_any": nn_any, "nn_dist_bright": nn_bright}))
        print(f"case{c}: {n:,} gals  median nn_any={np.median(nn_any):.2f}\"  "
              f"frac(nn>3\")={np.mean(nn_any > 3):.3f} frac(nn>7\")={np.mean(nn_any > 7):.3f}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases", flush=True)


if __name__ == "__main__":
    main()
