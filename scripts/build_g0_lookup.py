"""Extract the g=0 (unsheared) per-galaxy ngmix shape, keyed by (case, input_index), for
shape-noise cancellation (SNC): pair each sheared galaxy with its OWN unsheared measurement so
the intrinsic shape cancels in e(g)-e(0).

Reads the raw g=0.0 Shapes (secondaries) + CrossMatch per case (light: ~58 MB/case).
Output feather: case, input_index, ngmix0_g1, ngmix0_g2  (ngmix-converged rows only).
"""
import argparse, os
import numpy as np
import pandas as pd
import pyarrow.feather as f

BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
TILE = "tile180.0_-0.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    for c in args.cases:
        base = f"{BASE}/case{c}_0.0/real0/catalogues"
        sp = f"{base}/Shapes/shape_catalogue_detect_position_secondaries_{TILE}.feather"
        cm = f"{base}/CrossMatch/{TILE}_rot0_matched.feather"
        if not (os.path.exists(sp) and os.path.exists(cm)):
            print(f"case{c}: missing g0 shapes/crossmatch, skip"); continue
        sh = f.read_table(sp, columns=["NUMBER", "NGMIX_G1", "NGMIX_G2"]).to_pandas()
        m = f.read_table(cm, columns=["id_detec", "id_input"]).to_pandas()
        e1 = sh["NGMIX_G1"].to_numpy(float); e2 = sh["NGMIX_G2"].to_numpy(float)
        num = sh["NUMBER"].to_numpy()
        row_of = {int(n): i for i, n in enumerate(num)}
        d = m["id_detec"].to_numpy(); iid = m["id_input"].to_numpy()
        cc = []; ii = []; g1 = []; g2 = []
        for dd, jj in zip(d, iid):
            r = row_of.get(int(dd))
            if r is None:
                continue
            a, b = e1[r], e2[r]
            if a == -1.0 or (a == 0.0 and b == 0.0):     # ngmix failed -> no g0 measurement
                continue
            cc.append(c); ii.append(int(jj)); g1.append(a); g2.append(b)
        rows.append(pd.DataFrame({"case": cc, "input_index": ii, "ngmix0_g1": g1, "ngmix0_g2": g2}))
        print(f"case{c}: {len(cc):,} g0 ngmix-converged targets")
    df = pd.concat(rows, ignore_index=True)
    df.to_feather(args.output)
    print(f"wrote {args.output}: {len(df):,} (case,target) g0 measurements over {df['case'].nunique()} cases")


if __name__ == "__main__":
    main()
