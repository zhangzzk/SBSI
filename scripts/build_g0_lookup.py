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
    ap.add_argument("--extra-cols", nargs="*", default=[],
                    help="extra raw Shapes columns to carry through, as SRC:DST pairs "
                         "(e.g. MAG_AUTO:measured_mag_auto FLUX_RADIUS:measured_flux_radius). "
                         "Default empty => the output is byte-identical to the historical lookup. "
                         "Rows are filtered by the SAME ngmix-convergence condition, so an extended "
                         "lookup is a strict superset of columns on the same row set.")
    ap.add_argument("--shape-suffix", default="_secondaries",
                    choices=["_secondaries", ""],
                    help="'_secondaries' (historical default) reads the SECONDARY-target shape "
                         "catalogue; '' reads the PRIMARY-target one. These measure different "
                         "objects; pick the one whose targets match the half-shear catalogue's "
                         "`_p` object, verified empirically -- do not assume.")
    args = ap.parse_args()
    extra = []
    for spec in args.extra_cols:
        if ":" not in spec:
            raise ValueError(f"--extra-cols entry {spec!r} must be SRC:DST")
        src, dst = spec.split(":", 1)
        extra.append((src, dst))
    if extra:
        print(f"extra columns: {[f'{s}->{d}' for s, d in extra]}")
    rows = []
    for c in args.cases:
        base = f"{BASE}/case{c}_0.0/real0/catalogues"
        # WHICH TARGET SET the g=0 shapes come from. blendemu writes two files per case at the same
        # detection positions: no suffix measures the PRIMARIES (first-half input ids), `_secondaries`
        # measures the SECONDARIES (second-half). They are NOT the same objects -- measured on
        # matched NUMBER they correlate -0.80. Using the wrong one makes SNC subtract a different
        # galaxy's shape, which ANTI-cancels instead of cancelling and leaves the mean response
        # nearly right while the per-object scatter explodes. Default keeps the historical
        # `_secondaries` so existing lookups reproduce; see WORKLOG 2026-08-03v/w.
        sp = f"{base}/Shapes/shape_catalogue_detect_position{args.shape_suffix}_{TILE}.feather"
        cm = f"{base}/CrossMatch/{TILE}_rot0_matched.feather"
        if not (os.path.exists(sp) and os.path.exists(cm)):
            print(f"case{c}: missing g0 shapes/crossmatch, skip"); continue
        sh = f.read_table(sp, columns=["NUMBER", "NGMIX_G1", "NGMIX_G2",
                                       *[s for s, _ in extra]]).to_pandas()
        m = f.read_table(cm, columns=["id_detec", "id_input"]).to_pandas()
        e1 = sh["NGMIX_G1"].to_numpy(float); e2 = sh["NGMIX_G2"].to_numpy(float)
        xarr = [sh[s].to_numpy(float) for s, _ in extra]
        num = sh["NUMBER"].to_numpy()
        row_of = {int(n): i for i, n in enumerate(num)}
        d = m["id_detec"].to_numpy(); iid = m["id_input"].to_numpy()
        cc = []; ii = []; g1 = []; g2 = []; rr = []
        for dd, jj in zip(d, iid):
            r = row_of.get(int(dd))
            if r is None:
                continue
            a, b = e1[r], e2[r]
            if a == -1.0 or (a == 0.0 and b == 0.0):     # ngmix failed -> no g0 measurement
                continue
            cc.append(c); ii.append(int(jj)); g1.append(a); g2.append(b); rr.append(r)
        out = {"case": cc, "input_index": ii, "ngmix0_g1": g1, "ngmix0_g2": g2}
        if extra:
            ridx = np.asarray(rr, dtype=np.int64)
            for (src, dst), arr in zip(extra, xarr):
                out[dst] = arr[ridx] if len(ridx) else np.empty(0, dtype=float)
        rows.append(pd.DataFrame(out))
        print(f"case{c}: {len(cc):,} g0 ngmix-converged targets")
    df = pd.concat(rows, ignore_index=True)
    df.to_feather(args.output)
    print(f"wrote {args.output}: {len(df):,} (case,target) g0 measurements over {df['case'].nunique()} cases")


if __name__ == "__main__":
    main()
