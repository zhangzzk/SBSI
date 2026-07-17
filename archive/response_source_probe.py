"""Localize the 0.30-vs-0.39 isolated-response gap: is it (a) upstream join/shear-direction
dilution, or (b) noise bias / shear magnitude?

Discriminator: the Shapes catalogue carries BOTH NGMIX_G and GALSIM_G, joined through the
SAME detection->input->shear-direction chain.  A join/direction bug dilutes BOTH estimators
in the half-shear render (random per-galaxy ghat) while leaving the constant render (single
common ghat=(1,0)) intact.  Genuine noise bias hits only the noisier estimator.

We compute, for ISOLATED matched detected sources, the SINGLE-RENDER projected response
  R = <e . ghat> / g          (spin-2 dot; ghat = gamma_input/|gamma_input|)
for NGMIX and GALSIM in each render.  For the half render we also difference against the
0.0 render (the actual self_response construction) to see if the 0.0-subtraction matters.
"""
import argparse, os
import numpy as np
import pyarrow.feather as f
from scipy.spatial import cKDTree

CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
HALF  = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
TILE  = "tile180.0_-0.5"


def load(shapes, cm, inp):
    return (f.read_table(shapes).to_pandas(),
            f.read_table(cm).to_pandas(),
            f.read_table(inp).to_pandas())


def isolated_flag(inp, r_iso):
    ra = inp["RA_input"].to_numpy(float); dec = inp["DEC_input"].to_numpy(float)
    dec0 = np.median(dec)
    x = (ra - np.median(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - np.median(dec)) * 3600.0
    d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    return d[:, 1] > r_iso


def det_to_input(sh, cm):
    """Map each Shapes row (SExtractor NUMBER) -> input index via CrossMatch."""
    num = sh["NUMBER"].to_numpy()
    d2i = {int(d): int(i) for d, i in zip(cm["id_detec"].to_numpy(), cm["id_input"].to_numpy())}
    inp_idx = np.array([d2i.get(int(n), -1) for n in num])
    return inp_idx


def proj_response(e1, e2, gh1, gh2, gnom, mask):
    """R = <e.ghat>/g over mask, with jackknife-free s.e."""
    p = e1[mask] * gh1[mask] + e2[mask] * gh2[mask]
    p = p[np.isfinite(p)]
    R = p.mean() / gnom
    se = p.std() / np.sqrt(len(p)) / gnom
    return R, se, len(p)


def analyse_single(tag, shapes, cm, inp, gnom, iso_arr, sheared_only):
    sh, cmt, inpt = load(shapes, cm, inp)
    inp_idx = det_to_input(sh, cmt)
    matched = inp_idx >= 0
    g1 = inpt["gamma1_input"].to_numpy(float); g2 = inpt["gamma2_input"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    # per-detection shear direction & isolation
    gd1 = np.full(len(sh), np.nan); gd2 = np.full(len(sh), np.nan)
    gm = np.full(len(sh), np.nan); iso = np.zeros(len(sh), bool)
    ok = matched
    with np.errstate(invalid="ignore", divide="ignore"):
        gd1[ok] = g1[inp_idx[ok]] / gmag[inp_idx[ok]]
        gd2[ok] = g2[inp_idx[ok]] / gmag[inp_idx[ok]]
    gm[ok] = gmag[inp_idx[ok]]
    iso[ok] = iso_arr[inp_idx[ok]]

    ng1 = sh["NGMIX_G1"].to_numpy(float); ng2 = sh["NGMIX_G2"].to_numpy(float)
    gs1 = sh["GALSIM_G1"].to_numpy(float); gs2 = sh["GALSIM_G2"].to_numpy(float)
    ng_ok = (ng1 != -1.0) & ~((ng1 == 0.0) & (ng2 == 0.0))
    gs_ok = (gs1 != -1.0) & ~((gs1 == 0.0) & (gs2 == 0.0))

    base = matched & iso
    if sheared_only:
        base = base & (gm > 1e-6)

    print(f"\n===== {tag}  (isolated, g_nom={gnom}) =====")
    print(f"   isolated matched detections: {int(base.sum()):,}")
    for est, e1, e2, eok in [("NGMIX", ng1, ng2, ng_ok), ("GALSIM", gs1, gs2, gs_ok)]:
        m = base & eok
        if m.sum() < 50:
            print(f"   {est}: too few ({int(m.sum())})"); continue
        R, se, n = proj_response(e1, e2, gd1, gd2, gnom, m)
        print(f"   {est:6s} single-render R = {R:.4f} +/- {se:.4f}   (N={n:,}, valid={eok[base].mean()*100:.0f}%)")
    return sh, cmt, inpt, inp_idx, iso, gd1, gd2, gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-iso", type=float, default=3.0)
    ap.add_argument("--cases", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    for case in args.cases:
        print(f"\n################  CASE {case}  ################")
        # constant +0.02 (all sheared, ghat=(1,0))
        cdir = f"{CONST}/case{case}_0.02/real0/catalogues"
        inp_c = f"{cdir}/input/gals_info_{TILE}.feather"
        iso_c = isolated_flag(f.read_table(inp_c).to_pandas(), args.r_iso)
        analyse_single("CONST all +0.02",
                       f"{cdir}/Shapes/shape_catalogue_detect_position_all_{TILE}.feather",
                       f"{cdir}/CrossMatch/{TILE}_rot0_matched.feather", inp_c,
                       0.02, iso_c, sheared_only=False)

        # half 0.05 (secondaries sheared, random ghat)
        hdir = f"{HALF}/case{case}_0.05/real0/catalogues"
        inp_h = f"{hdir}/input/gals_info_{TILE}.feather"
        iso_h = isolated_flag(f.read_table(inp_h).to_pandas(), args.r_iso)
        analyse_single("HALF secondaries 0.05",
                       f"{hdir}/Shapes/shape_catalogue_detect_position_secondaries_{TILE}.feather",
                       f"{hdir}/CrossMatch/{TILE}_rot0_matched.feather", inp_h,
                       0.05, iso_h, sheared_only=True)


if __name__ == "__main__":
    main()
