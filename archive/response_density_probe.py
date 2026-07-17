"""Coherent-blend test via LOCAL DENSITY (the field is too dense to carve a clean isolated
sample).  Hypothesis: const (coherent shear) response is boosted by coherently-sheared
neighbour light; half (incoherent) is not.  So the const-minus-half gap should GROW with
local galaxy density and be absent for truly isolated galaxies.

For const@0.02 and half@0.02, restrict to LOW S/N (<15, where the gap lives) and bin by the
number of input galaxies within 5 arcsec (local density).  Report R per density bin.
"""
import argparse
import numpy as np
import pyarrow.feather as f
from scipy.spatial import cKDTree

CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
HALF  = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
TILE  = "tile180.0_-0.5"


def neighbor_count(inp, r_count):
    ra = inp["RA_input"].to_numpy(float); dec = inp["DEC_input"].to_numpy(float)
    dec0 = np.median(dec)
    x = (ra - np.median(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - np.median(dec)) * 3600.0
    tree = cKDTree(np.c_[x, y])
    # neighbours within r_count (excludes self -> subtract 1)
    cnts = tree.query_ball_point(np.c_[x, y], r_count, return_length=True) - 1
    return cnts


def gather(base, casedir, mode, gnom, const_axis, r_count, snmax):
    b = f"{base}/{casedir}/real0/catalogues"
    sh = f.read_table(f"{b}/Shapes/shape_catalogue_detect_position_{mode}_{TILE}.feather").to_pandas()
    cm = f.read_table(f"{b}/CrossMatch/{TILE}_rot0_matched.feather").to_pandas()
    inp = f.read_table(f"{b}/input/gals_info_{TILE}.feather").to_pandas()
    nc = neighbor_count(inp, r_count)
    g1 = inp["gamma1_input"].to_numpy(float); g2 = inp["gamma2_input"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    num = sh["NUMBER"].to_numpy(); row_of = {int(n): i for i, n in enumerate(num)}
    e1 = sh["NGMIX_G1"].to_numpy(float); e2 = sh["NGMIX_G2"].to_numpy(float)
    sn = sh["FLUX_AUTO"].to_numpy(float) / sh["FLUXERR_AUTO"].to_numpy(float)
    P = []; NC = []
    for d, iid in zip(cm["id_detec"].to_numpy(), cm["id_input"].to_numpy()):
        r = row_of.get(int(d)); iid = int(iid)
        if r is None or iid >= len(gmag) or gmag[iid] <= 1e-6:
            continue
        if sn[r] >= snmax:            # low-S/N only
            continue
        a, c = e1[r], e2[r]
        if a == -1.0 or (a == 0.0 and c == 0.0):
            continue
        gh1, gh2 = (1.0, 0.0) if const_axis else (g1[iid] / gmag[iid], g2[iid] / gmag[iid])
        P.append((a * gh1 + c * gh2) / gnom); NC.append(nc[iid])
    return np.array(P), np.array(NC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--r-count", type=float, default=5.0)
    ap.add_argument("--snmax", type=float, default=15.0)
    args = ap.parse_args()
    dbins = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 999)]  # neighbours within r_count
    for label, base, cdir, mode, gnom, axis in [
        ("CONST @0.02 lowSN", CONST, "case{c}_0.02", "all",         0.02, True),
        ("HALF  @0.02 lowSN", HALF,  "case{c}_0.02", "secondaries", 0.02, False),
    ]:
        P = []; NC = []
        for c in args.cases:
            try:
                p, nc = gather(base, cdir.format(c=c), mode, gnom, axis, args.r_count, args.snmax)
                P.append(p); NC.append(nc)
            except Exception as e:
                pass
        P = np.concatenate(P); NC = np.concatenate(NC)
        print(f"\n>>> {label}  (S/N<{args.snmax}, neighbours within {args.r_count}\")  N={len(P):,}  global R={P.mean():.3f}+/-{P.std()/np.sqrt(len(P)):.3f}")
        for lo, hi in dbins:
            m = (NC >= lo) & (NC <= hi)
            if m.sum() < 300:
                continue
            R = P[m].mean(); e = P[m].std() / np.sqrt(m.sum())
            tag = f"{lo}" if lo == hi else f"{lo}-{hi}"
            print(f"     neigh={tag:>5}: R={R:.3f} +/- {e:.3f}  (N={int(m.sum()):,})")


if __name__ == "__main__":
    main()
