"""Does the measured ellipticity depend on where the galaxy sits inside its 48 px stamp?

Production cutout (BlendEMU ``shape.cutout``, 48 px) puts a detection at 1-indexed (x, y) at stamp position
24 + frac(x), 24 + frac(y): offset ox = 0.5 + frac(x), oy = 0.5 + frac(y) from the geometric centre, always
towards +x/+y. The ngmix jacobian is centred on the source (subpixel), but the fit window still holds more pixels on
the low side. Light spread evenly over the window then has mean dx*dy = ox*oy > 0 about the source, i.e. a positive
e2 and no e1. Intrinsic shapes do not depend on frac(x), frac(y), so any slope of g2 on ox*oy is a measurement effect.

Per magnitude bin (MAG_AUTO) fits g_k = a + b*ox*oy + c*e_k,true (k = 1, 2) on all ConstGold case-legs. Errors:
leave-one-case-leg-out jackknife. Unmatched detections and failed fits are dropped and counted.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.feather as feather

TILE = "tile180.0_-0.5"
NPIX = 18181
EDGE = 26
MAG_EDGES = [0.0, 23.0, 24.0, 25.0, 25.8, 99.0]
NBIN = len(MAG_EDGES) - 1
NGRID = 4


def case_stats(case_dir):
    real = os.path.join(case_dir, "real0", "catalogues")
    s = feather.read_table(f"{real}/Shapes/shape_catalogue_detect_position_all_{TILE}.feather",
                           columns=["X_IMAGE", "Y_IMAGE", "MAG_AUTO", "NGMIX_G1", "NGMIX_G2"]).to_pandas()
    m = feather.read_table(f"{real}/CrossMatch/{TILE}_rot0_matched.feather").to_pandas()
    t = feather.read_table(f"{real}/input/gals_info_{TILE}.feather",
                           columns=["index_input", "e1_input_rot0", "e2_input_rot0"]).to_pandas()
    row = np.full(max(t["index_input"].max(), m["id_input"].max()) + 1, -1)
    row[t["index_input"].to_numpy()] = np.arange(len(t))
    no_truth = row[m["id_input"].to_numpy()] < 0
    m = m[~no_truth]
    det = m["id_detec"].to_numpy() - 1
    inp = row[m["id_input"].to_numpy()]
    n_det = len(s)
    x = s["X_IMAGE"].to_numpy()[det]
    y = s["Y_IMAGE"].to_numpy()[det]
    mag = s["MAG_AUTO"].to_numpy()[det]
    g = np.stack([s["NGMIX_G1"].to_numpy()[det], s["NGMIX_G2"].to_numpy()[det]], 1)
    e = np.stack([t["e1_input_rot0"].to_numpy()[inp], t["e2_input_rot0"].to_numpy()[inp]], 1)
    ok_fit = np.all(np.isfinite(g), 1) & ~np.all(g == -1.0, 1) & (np.sum(g ** 2, 1) < 1.0)
    ok_edge = (x > EDGE) & (x < NPIX - EDGE) & (y > EDGE) & (y < NPIX - EDGE)
    ok = ok_fit & ok_edge & np.isfinite(mag)
    ox = 0.5 + (x - np.floor(x))
    oy = 0.5 + (y - np.floor(y))
    p = ox * oy
    ib = np.digitize(mag, MAG_EDGES) - 1
    gx = np.minimum((ox - 0.5) * NGRID, NGRID - 1).astype(int)
    gy = np.minimum((oy - 0.5) * NGRID, NGRID - 1).astype(int)
    xtx = np.zeros((2, NBIN, 3, 3))
    xty = np.zeros((2, NBIN, 3))
    grid_n = np.zeros((NBIN, NGRID, NGRID))
    grid_sum = np.zeros((2, NBIN, NGRID, NGRID))
    grid_esum = np.zeros((2, NBIN, NGRID, NGRID))
    for b in range(NBIN):
        sel = ok & (ib == b)
        for k in range(2):
            X = np.stack([np.ones(sel.sum()), p[sel], e[sel, k]], 1)
            xtx[k, b] = X.T @ X
            xty[k, b] = X.T @ g[sel, k]
            np.add.at(grid_sum[k, b], (gy[sel], gx[sel]), g[sel, k])
            np.add.at(grid_esum[k, b], (gy[sel], gx[sel]), e[sel, k])
        np.add.at(grid_n[b], (gy[sel], gx[sel]), 1)
    counts = dict(detections=int(n_det), matched=int(len(det)), unmatched_detections=int(n_det - len(det)),
                  matched_without_truth_row=int(no_truth.sum()),
                  failed_fit=int((~ok_fit).sum()), edge=int((ok_fit & ~ok_edge).sum()), used=int(ok.sum()))
    return os.path.basename(case_dir), xtx, xty, grid_n, grid_sum, grid_esum, counts


def solve(xtx, xty):
    return np.array([[np.linalg.solve(xtx[k, b], xty[k, b]) for b in range(NBIN)] for k in range(2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cases = sorted(d for d in glob.glob(f"{a.root}/case*_0.02") + glob.glob(f"{a.root}/case*_-0.02")
                   if os.path.isfile(f"{d}/real0/catalogues/Shapes/shape_catalogue_detect_position_all_{TILE}.feather"))
    with Pool(a.workers) as pool:
        res = pool.map(case_stats, cases)
    names = [r[0] for r in res]
    XTX = np.array([r[1] for r in res])
    XTY = np.array([r[2] for r in res])
    full = solve(XTX.sum(0), XTY.sum(0))
    n = len(res)
    jk = np.array([solve(XTX.sum(0) - XTX[i], XTY.sum(0) - XTY[i]) for i in range(n)])
    err = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
    gn = sum(r[3] for r in res)
    gs = sum(r[4] for r in res)
    ge = sum(r[5] for r in res)
    counts = {k: int(sum(r[6][k] for r in res)) for k in res[0][6]}
    out = dict(
        description=__doc__.strip().splitlines()[0],
        model="g_k = a + b * ox*oy + c * e_k,true; ox = 0.5 + frac(X_IMAGE), oy = 0.5 + frac(Y_IMAGE)",
        mean_oxoy_production=1.0,
        n_case_legs=n, case_legs=names, mag_edges_MAG_AUTO=MAG_EDGES, counts=counts,
        bins=[],
        grid_note=f"{NGRID}x{NGRID} grid in (frac y, frac x); mean measured g minus mean true e (raw, no slope c)",
    )
    for b in range(NBIN):
        row = dict(mag=[MAG_EDGES[b], MAG_EDGES[b + 1]], n=int(gn[b].sum()))
        for k in range(2):
            row[f"g{k+1}"] = dict(a=full[k, b, 0], a_err=err[k, b, 0], b_slope=full[k, b, 1], b_err=err[k, b, 1],
                                  c=full[k, b, 2], c_err=err[k, b, 2],
                                  centred_minus_production=-full[k, b, 1], centred_minus_production_err=err[k, b, 1])
            row[f"grid_g{k+1}_minus_e"] = ((gs[k, b] - ge[k, b]) / np.maximum(gn[b], 1)).round(6).tolist()
        row["grid_n"] = gn[b].astype(int).tolist()
        out["bins"].append(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    for row in out["bins"]:
        print(row["mag"], row["n"], "g2 slope %.2e +- %.2e" % (row["g2"]["b_slope"], row["g2"]["b_err"]),
              "g1 slope %.2e +- %.2e" % (row["g1"]["b_slope"], row["g1"]["b_err"]))
    print(counts)


if __name__ == "__main__":
    main()
