"""Is the isolated ngmix response ANISOTROPIC w.r.t. the pixel grid?  The constant render
shears every galaxy along +x (pixel axis); the half render uses random directions.  If R
depends on the shear position angle, then const (locked on x) is the special case and the
half (direction-averaged) is the true cosmological response -- which would explain
CONST@0.02 (0.39) > HALF (0.27) at the same shear, with NO bug.

Within the HALF render (all shear directions present), bin isolated sheared secondaries by
their spin-2 shear angle 2phi = atan2(g2,g1) and measure R = <e.ghat>/g per bin.  Also
report the sub-sample sheared ~along +x (|2phi|<22.5deg, like const) vs ~along the diagonal.
"""
import argparse
import numpy as np
import pyarrow.feather as f
from scipy.spatial import cKDTree

HALF = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
TILE = "tile180.0_-0.5"


def isolated_set(inp, r_iso):
    ra = inp["RA_input"].to_numpy(float); dec = inp["DEC_input"].to_numpy(float)
    dec0 = np.median(dec)
    x = (ra - np.median(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - np.median(dec)) * 3600.0
    d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    return d[:, 1] > r_iso


def gather(casedir, gnom, r_iso):
    base = f"{HALF}/{casedir}/real0/catalogues"
    sh = f.read_table(f"{base}/Shapes/shape_catalogue_detect_position_secondaries_{TILE}.feather").to_pandas()
    cm = f.read_table(f"{base}/CrossMatch/{TILE}_rot0_matched.feather").to_pandas()
    inp = f.read_table(f"{base}/input/gals_info_{TILE}.feather").to_pandas()
    iso = isolated_set(inp, r_iso)
    g1 = inp["gamma1_input"].to_numpy(float); g2 = inp["gamma2_input"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    num = sh["NUMBER"].to_numpy(); row_of = {int(n): i for i, n in enumerate(num)}
    e1 = sh["NGMIX_G1"].to_numpy(float); e2 = sh["NGMIX_G2"].to_numpy(float)
    sn = sh["FLUX_AUTO"].to_numpy(float) / sh["FLUXERR_AUTO"].to_numpy(float)
    P = []; A = []; S = []
    for d, iid in zip(cm["id_detec"].to_numpy(), cm["id_input"].to_numpy()):
        r = row_of.get(int(d)); iid = int(iid)
        if r is None or iid >= len(gmag) or not iso[iid] or gmag[iid] <= 1e-6:
            continue
        a, b = e1[r], e2[r]
        if a == -1.0 or (a == 0.0 and b == 0.0):
            continue
        gh1, gh2 = g1[iid] / gmag[iid], g2[iid] / gmag[iid]
        P.append((a * gh1 + b * gh2) / gnom)
        A.append(np.arctan2(g2[iid], g1[iid]))     # spin-2 shear angle 2phi
        S.append(sn[r])
    return np.array(P), np.array(A), np.array(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-iso", type=float, default=3.0)
    ap.add_argument("--gnom", type=float, default=0.05)
    ap.add_argument("--shear", default="0.05")
    ap.add_argument("--cases", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--nbins", type=int, default=8)
    args = ap.parse_args()
    P = []; A = []; S = []
    for c in args.cases:
        try:
            p, a, s = gather(f"case{c}_{args.shear}", args.gnom, args.r_iso)
            P.append(p); A.append(a); S.append(s)
        except Exception as e:
            print(f"  case{c}: SKIP ({type(e).__name__}: {e})")
    P = np.concatenate(P); A = np.concatenate(A); S = np.concatenate(S)
    print(f"\nHALF @{args.shear}  isolated sheared, N={len(P):,}, global R={P.mean():.4f}+/-{P.std()/np.sqrt(len(P)):.4f}")
    print("\nR vs spin-2 shear angle 2phi=atan2(g2,g1)  (isotropic => flat):")
    edges = np.linspace(-np.pi, np.pi, args.nbins + 1)
    for i in range(args.nbins):
        m = (A >= edges[i]) & (A < edges[i + 1])
        if m.sum() < 200:
            continue
        R = P[m].mean(); e = P[m].std() / np.sqrt(m.sum())
        print(f"  2phi [{np.degrees(edges[i]):+7.1f},{np.degrees(edges[i+1]):+7.1f}) deg: R={R:.3f} +/- {e:.3f}  (N={int(m.sum()):,})")
    # along-x (like const: g1>0,g2~0 -> 2phi~0) vs diagonal (2phi~+/-90)
    near_x = np.abs(np.arctan2(np.sin(A), np.cos(A))) < np.deg2rad(22.5)         # 2phi ~ 0  (shear along +x)
    near_y = np.abs(np.abs(np.arctan2(np.sin(A), np.cos(A))) - np.pi) < np.deg2rad(22.5)  # 2phi ~ +/-180 (along -x/y-ish)
    near_diag = np.abs(np.abs(np.arctan2(np.sin(A), np.cos(A))) - np.pi / 2) < np.deg2rad(22.5)  # 2phi ~ +/-90 (diagonal)
    print("\n  sheared ~along pixel +x axis (2phi~0, like CONST): "
          f"R={P[near_x].mean():.3f} +/- {P[near_x].std()/np.sqrt(max(near_x.sum(),1)):.3f} (N={int(near_x.sum()):,})")
    print(f"  sheared ~along 2phi~180:                            "
          f"R={P[near_y].mean():.3f} +/- {P[near_y].std()/np.sqrt(max(near_y.sum(),1)):.3f} (N={int(near_y.sum()):,})")
    print(f"  sheared ~along diagonal (2phi~90):                  "
          f"R={P[near_diag].mean():.3f} +/- {P[near_diag].std()/np.sqrt(max(near_diag.sum(),1)):.3f} (N={int(near_diag.sum()):,})")


if __name__ == "__main__":
    main()
