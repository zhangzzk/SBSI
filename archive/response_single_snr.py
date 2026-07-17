"""Single-render (the flow's actual R_sim) isolated ngmix response, binned by measured S/N,
for const@0.02, half@0.05, half@0.2 -- to separate a BUG from noise-bias from shear magnitude.

R = <e . ghat>/g  on isolated matched ngmix-converged detections.  ghat=(1,0) for const
(single common axis), per-galaxy gamma direction for half.  If half@0.05 ~ half@0.2 at each
S/N -> no shear-magnitude effect in the half pipeline.  If half@0.05 ~ const@0.02 at each
S/N -> no render/bug difference (the 0.30-vs-0.39 is then purely the const GOLD using the
noise-cancelling two-sided estimator).  If half < const at every S/N -> real render/bug gap.
"""
import argparse
import numpy as np
import pyarrow.feather as f
from scipy.spatial import cKDTree

CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
HALF  = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
TILE  = "tile180.0_-0.5"
EDGES = [0, 10, 15, 20, 30, 50, 100, 1e9]


def isolated_set(inp, r_iso):
    ra = inp["RA_input"].to_numpy(float); dec = inp["DEC_input"].to_numpy(float)
    dec0 = np.median(dec)
    x = (ra - np.median(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - np.median(dec)) * 3600.0
    d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    return d[:, 1] > r_iso


def gather(base, casedir, mode, gnom, const_axis, r_iso):
    sh = f.read_table(f"{base}/{casedir}/real0/catalogues/Shapes/"
                      f"shape_catalogue_detect_position_{mode}_{TILE}.feather").to_pandas()
    cm = f.read_table(f"{base}/{casedir}/real0/catalogues/CrossMatch/{TILE}_rot0_matched.feather").to_pandas()
    inp = f.read_table(f"{base}/{casedir}/real0/catalogues/input/gals_info_{TILE}.feather").to_pandas()
    iso = isolated_set(inp, r_iso)
    g1 = inp["gamma1_input"].to_numpy(float); g2 = inp["gamma2_input"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    num = sh["NUMBER"].to_numpy()
    row_of = {int(n): i for i, n in enumerate(num)}
    e1 = sh["NGMIX_G1"].to_numpy(float); e2 = sh["NGMIX_G2"].to_numpy(float)
    sn = sh["FLUX_AUTO"].to_numpy(float) / sh["FLUXERR_AUTO"].to_numpy(float)
    P = []; S = []
    for d, iid in zip(cm["id_detec"].to_numpy(), cm["id_input"].to_numpy()):
        r = row_of.get(int(d)); iid = int(iid)
        if r is None or not iso[iid] or gmag[iid] <= 1e-6:
            continue
        a, b = e1[r], e2[r]
        if a == -1.0 or (a == 0.0 and b == 0.0):
            continue
        if const_axis:
            gh1, gh2 = 1.0, 0.0
        else:
            gh1, gh2 = g1[iid] / gmag[iid], g2[iid] / gmag[iid]
        P.append(a * gh1 + b * gh2); S.append(sn[r])
    return np.array(P), np.array(S)


def report(label, P, S, gnom):
    R = P.mean() / gnom; e = P.std() / np.sqrt(len(P)) / gnom
    print(f"\n>>> {label}: global R={R:.4f} +/- {e:.4f}  (N={len(P):,})")
    for i in range(len(EDGES) - 1):
        m = (S >= EDGES[i]) & (S < EDGES[i + 1])
        if m.sum() < 200:
            continue
        r = P[m].mean() / gnom; ee = P[m].std() / np.sqrt(m.sum()) / gnom
        print(f"     S/N [{EDGES[i]:5.0f},{EDGES[i+1]:5.0f}): R={r:.3f} +/- {ee:.3f}  (N={int(m.sum()):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-iso", type=float, default=3.0)
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    configs = [
        ("CONST single @0.02", CONST, "case{c}_0.02", "all",         0.02, True),
        ("HALF  single @0.02", HALF,  "case{c}_0.02", "secondaries", 0.02, False),
        ("HALF  single @0.05", HALF,  "case{c}_0.05", "secondaries", 0.05, False),
    ]
    for label, base, cdir, mode, gnom, axis in configs:
        P = []; S = []
        for c in args.cases:
            try:
                p, s = gather(base, cdir.format(c=c), mode, gnom, axis, args.r_iso)
                P.append(p); S.append(s)
            except Exception as e:
                print(f"   {label} case{c}: SKIP ({type(e).__name__}: {e})")
        if P:
            report(label, np.concatenate(P), np.concatenate(S), gnom)


if __name__ == "__main__":
    main()
