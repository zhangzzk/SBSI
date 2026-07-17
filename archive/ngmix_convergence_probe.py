"""Measure the ngmix CONVERGENCE RATE in the raw Shapes catalogues, comparing the
constant-shear render (--targets all) with the half-shear render (--targets secondaries).

For each detected source we know (via CrossMatch -> input index) whether it was a
PRIMARY (first half of the input list) or SECONDARY (second half).  In the half-shear
render only the SECONDARIES are ngmix-measured; primaries carry the -1 sentinel by
DESIGN (not a convergence failure).  We separate:

  measured set   = sources the mode actually asked ngmix to fit
  converged      = NGMIX_G1 != -1 and not (0,0)
  convergence    = converged / measured   (the honest rate)

Isolation is computed from the INPUT positions (byte-identical across renders), so the
isolated sub-sample is identical -> a clean apples-to-apples comparison.
"""
import argparse, os
import numpy as np
import pyarrow.feather as f
from scipy.spatial import cKDTree


def load_tile(shapes_path, cm_path, input_path):
    sh = f.read_table(shapes_path).to_pandas()
    cm = f.read_table(cm_path).to_pandas()
    inp = f.read_table(input_path).to_pandas()
    return sh, cm, inp


def isolated_flag(inp, r_iso_arcsec):
    """isolated = no OTHER input galaxy within r_iso arcsec (nearest-neighbour)."""
    ra = inp["RA_input"].to_numpy(float); dec = inp["DEC_input"].to_numpy(float)
    # small-angle tangent plane around field centre; positions in arcsec
    dec0 = np.median(dec)
    x = (ra - np.median(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - np.median(dec)) * 3600.0
    tree = cKDTree(np.c_[x, y])
    d, _ = tree.query(np.c_[x, y], k=2)          # k=2: self + nearest other
    nn = d[:, 1]
    return nn > r_iso_arcsec                       # True = isolated


def analyse(tag, shapes_path, cm_path, input_path, r_iso):
    sh, cm, inp = load_tile(shapes_path, cm_path, input_path)
    n_input = len(inp)
    half = n_input // 2
    # per-input |g| and isolation
    g1 = inp["gamma1_input"].to_numpy(float); g2 = inp["gamma2_input"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    iso_in = isolated_flag(inp, r_iso)             # indexed by input row (0..n_input-1)
    is_secondary_in = np.arange(n_input) >= half   # second half = secondaries

    # map detection NUMBER -> input index via CrossMatch (id_detec is 1-based NUMBER)
    det_num = sh["NUMBER"].to_numpy()
    ng1 = sh["NGMIX_G1"].to_numpy(float); ng2 = sh["NGMIX_G2"].to_numpy(float)
    cm_detec = cm["id_detec"].to_numpy(); cm_input = cm["id_input"].to_numpy()
    # build detec->input map
    d2i = {int(d): int(i) for d, i in zip(cm_detec, cm_input)}
    inp_idx = np.array([d2i.get(int(n), -1) for n in det_num])
    matched = inp_idx >= 0

    conv = (ng1 != -1.0) & ~((ng1 == 0.0) & (ng2 == 0.0))   # ngmix converged
    # attach per-detection input properties
    sec = np.zeros(len(sh), bool); iso = np.zeros(len(sh), bool); gm = np.full(len(sh), np.nan)
    sec[matched] = is_secondary_in[inp_idx[matched]]
    iso[matched] = iso_in[inp_idx[matched]]
    gm[matched] = gmag[inp_idx[matched]]

    print(f"\n===== {tag} =====")
    print(f"  detections={len(sh):,}  matched-to-input={matched.sum():,}")
    print(f"  overall: NGMIX converged = {conv.mean()*100:.1f}%  (converged {conv.sum():,}/{len(sh):,})")
    print(f"  among PRIMARIES (1st half): converged = {conv[matched & ~sec].mean()*100:.1f}%  (N={int((matched & ~sec).sum()):,})")
    print(f"  among SECONDARIES(2nd half): converged = {conv[matched & sec].mean()*100:.1f}%  (N={int((matched & sec).sum()):,})")
    # ISOLATED sub-sample
    for lbl, mm in [("ISOLATED all", matched & iso),
                    ("ISOLATED secondaries", matched & iso & sec),
                    ("ISOLATED primaries", matched & iso & ~sec)]:
        if mm.sum() > 0:
            print(f"  [{lbl:22s}] converged = {conv[mm].mean()*100:.1f}%  (N={int(mm.sum()):,})")
    # sheared-only (|g|>0) isolated -- the sample the response actually uses
    sheared = matched & iso & (gm > 1e-6)
    if sheared.sum() > 0:
        print(f"  [ISOLATED & |g|>0        ] converged = {conv[sheared].mean()*100:.1f}%  (N={int(sheared.sum()):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-iso", type=float, default=3.0, help="isolation radius arcsec")
    args = ap.parse_args()
    CB = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case0_0.02/real0/catalogues"
    HB = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/case0_0.05/real0/catalogues"
    tile = "tile180.0_-0.5"
    analyse("CONST all g=0.02",
            f"{CB}/Shapes/shape_catalogue_detect_position_all_{tile}.feather",
            f"{CB}/CrossMatch/{tile}_rot0_matched.feather",
            f"{CB}/input/gals_info_{tile}.feather", args.r_iso)
    analyse("HALF secondaries g=0.05",
            f"{HB}/Shapes/shape_catalogue_detect_position_secondaries_{tile}.feather",
            f"{HB}/CrossMatch/{tile}_rot0_matched.feather",
            f"{HB}/input/gals_info_{tile}.feather", args.r_iso)


if __name__ == "__main__":
    main()
