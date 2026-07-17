"""Decisive bug-vs-noise-bias test for the 0.30(half) vs 0.39(const) isolated ngmix response.

For each render we pair the two shear cases the response is built from and, on ISOLATED
matched detections common to both, measure:

  (1) NOISE SHARING: corr of ngmix e between the two paired renders.  Constant pairs
      +0.02/-0.02 (SAME galaxies+noise -> corr ~1 -> noise cancels in the difference).
      Half pairs 0.0/0.05 (if independent noise -> lower corr -> ngmix noise bias survives).
  (2) E-mode (tangential) and B-mode (cross) response via the DIFFERENCE method:
        const: [e(+)-e(-)]/(2g)   ghat=(1,0)
        half : [e(0.05)-e(0.0)]/g  ghat = gamma_dir per galaxy
      B-mode ~ 0 rules out a systematic shear-direction rotation in the join.
  (3) E-mode response BINNED by measured S/N.  If the gap is NOISE BIAS it vanishes at
      high S/N (the two converge); a real upstream/join bug would persist at all S/N.
"""
import argparse, os
import numpy as np
import pyarrow.feather as f
from scipy.spatial import cKDTree

CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
HALF  = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
TILE  = "tile180.0_-0.5"


def shp_path(base, casedir, mode):
    return f"{base}/{casedir}/real0/catalogues/Shapes/shape_catalogue_detect_position_{mode}_{TILE}.feather"

def cm_path(base, casedir):
    return f"{base}/{casedir}/real0/catalogues/CrossMatch/{TILE}_rot0_matched.feather"

def inp_path(base, casedir):
    return f"{base}/{casedir}/real0/catalogues/input/gals_info_{TILE}.feather"


def isolated_set(inp, r_iso):
    ra = inp["RA_input"].to_numpy(float); dec = inp["DEC_input"].to_numpy(float)
    dec0 = np.median(dec)
    x = (ra - np.median(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - np.median(dec)) * 3600.0
    d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    return d[:, 1] > r_iso     # boolean over input rows


def render_map(base, casedir, mode):
    """Return dict input_id -> (e1,e2,sn) for ngmix-converged matched detections."""
    sh = f.read_table(shp_path(base, casedir, mode)).to_pandas()
    cm = f.read_table(cm_path(base, casedir)).to_pandas()
    num = sh["NUMBER"].to_numpy()
    row_of_num = {int(n): i for i, n in enumerate(num)}
    e1 = sh["NGMIX_G1"].to_numpy(float); e2 = sh["NGMIX_G2"].to_numpy(float)
    sn = (sh["FLUX_AUTO"].to_numpy(float) / sh["FLUXERR_AUTO"].to_numpy(float))
    out = {}
    for d, iid in zip(cm["id_detec"].to_numpy(), cm["id_input"].to_numpy()):
        r = row_of_num.get(int(d))
        if r is None:
            continue
        a, b = e1[r], e2[r]
        if a == -1.0 or (a == 0.0 and b == 0.0):
            continue
        out[int(iid)] = (a, b, sn[r])
    return out


def analyse(tag, base, dir_lo, dir_hi, mode, g_lo, g_hi, gnom, r_iso):
    """dir_lo/dir_hi paired renders; e_hi - e_lo is the response.  For const, dir_lo=-0.02."""
    inp = f.read_table(inp_path(base, dir_hi)).to_pandas()
    iso = isolated_set(inp, r_iso)
    g1 = inp["gamma1_input"].to_numpy(float); g2 = inp["gamma2_input"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    m_lo = render_map(base, dir_lo, mode)
    m_hi = render_map(base, dir_hi, mode)
    common = np.array(sorted(set(m_lo) & set(m_hi)), dtype=int)
    # isolated + sheared (|g|>0 in the hi render)
    keep = np.array([iso[i] and gmag[i] > 1e-6 for i in common])
    common = common[keep]
    e_lo = np.array([m_lo[i][:2] for i in common])   # (N,2)
    e_hi = np.array([m_hi[i][:2] for i in common])
    sn = np.array([m_hi[i][2] for i in common])
    gd = np.array([[g1[i], g2[i]] for i in common])
    gd = gd / np.linalg.norm(gd, axis=1, keepdims=True)     # ghat per galaxy
    gperp = np.c_[-gd[:, 1], gd[:, 0]]
    de = e_hi - e_lo                                          # response (hi - lo)
    et = (de * gd).sum(1)                                     # E-mode (tangential)
    ex = (de * gperp).sum(1)                                  # B-mode (cross)
    RE = et.mean() / gnom;  RE_e = et.std() / np.sqrt(len(et)) / gnom
    RB = ex.mean() / gnom;  RB_e = ex.std() / np.sqrt(len(ex)) / gnom
    # noise sharing
    c1 = np.corrcoef(e_lo[:, 0], e_hi[:, 0])[0, 1]
    c2 = np.corrcoef(e_lo[:, 1], e_hi[:, 1])[0, 1]
    print(f"\n===== {tag}  (N_iso_sheared={len(common):,}, gnom={gnom}) =====")
    print(f"   noise-share corr(e_lo,e_hi): e1={c1:.3f} e2={c2:.3f}   (->1 = same noise, cancels)")
    print(f"   E-mode R = {RE:.4f} +/- {RE_e:.4f}    B-mode(cross) R = {RB:+.4f} +/- {RB_e:.4f}")
    return common, et, ex, sn, gnom


def sn_binned(label, et, sn, gnom, edges):
    print(f"   S/N-binned E-mode R [{label}]:")
    for i in range(len(edges) - 1):
        m = (sn >= edges[i]) & (sn < edges[i + 1])
        if m.sum() < 200:
            continue
        R = et[m].mean() / gnom; e = et[m].std() / np.sqrt(m.sum()) / gnom
        print(f"      S/N [{edges[i]:5.0f},{edges[i+1]:5.0f}): R={R:.3f} +/- {e:.3f}  (N={int(m.sum()):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-iso", type=float, default=3.0)
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    edges = [0, 10, 15, 20, 30, 50, 100, 1e9]

    # aggregate across cases
    for label, base, lo, hi, mode, gnom in [
        ("CONST two-sided +/-0.02", CONST, "case{c}_-0.02", "case{c}_0.02", "all", 0.04),
        ("HALF  0.05 minus 0.0",    HALF,  "case{c}_0.0",   "case{c}_0.05", "secondaries", 0.05),
    ]:
        ET = []; EX = []; SN = []
        for c in args.cases:
            try:
                _, et, ex, sn, _ = analyse(f"{label} [case{c}]", base,
                                           lo.format(c=c), hi.format(c=c), mode,
                                           None, None, gnom, args.r_iso)
                ET.append(et); EX.append(ex); SN.append(sn)
            except Exception as e:
                print(f"   case{c}: SKIP ({type(e).__name__}: {e})")
        if ET:
            ET = np.concatenate(ET); EX = np.concatenate(EX); SN = np.concatenate(SN)
            RE = ET.mean()/gnom; RE_e = ET.std()/np.sqrt(len(ET))/gnom
            RB = EX.mean()/gnom; RB_e = EX.std()/np.sqrt(len(EX))/gnom
            print(f"\n>>> AGG {label}: E-mode R={RE:.4f}+/-{RE_e:.4f}  B-mode R={RB:+.4f}+/-{RB_e:.4f}  (N={len(ET):,})")
            sn_binned(label, ET, SN, gnom, edges)


if __name__ == "__main__":
    main()
