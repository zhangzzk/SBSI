"""Per-(case, input_index) OUT-OF-DOMAIN neighbour flux: light from neighbours the blend
emulator CANNOT score (r>28, or Re outside [0.1,1.5]) within 10". Used to test whether the
constant-gold residual comes from blend the emulator misses (Option 1).

Output columns: case, input_index, ood_flux (log10(1+F_ood/aperture_rms)), ind_flux (in-domain,
for reference).  Same units/scale as build_crowding_lookup so it is directly comparable.
"""
import argparse, os
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

TILE = "tile180.0_-0.5"
PIXEL_RMS, PIXEL_SIZE, PSF_FWHM, MOFFAT_BETA, ZERO_MAG = 0.312, 0.2, 0.73, 2.224, 30.0
R_MAX = 10.0 / 3600.0                          # emulator aperture (degrees)
R_LO, R_HI, RE_LO, RE_HI = 18.0, 28.0, 0.1, 1.5   # emulator in-domain cuts


def _aperture_rms():
    factor = np.sqrt((2 ** (1 / (MOFFAT_BETA - 1)) - 1) / (2 ** (1 / MOFFAT_BETA) - 1)) / 2
    psf_size = PSF_FWHM * factor
    return PIXEL_RMS * (psf_size / PIXEL_SIZE) ** 2 * np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--base", default="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant")
    args = ap.parse_args()
    ap_rms = _aperture_rms()

    parts = []
    for c in args.cases:
        fp = f"{args.base}/case{c}_{args.sign}/real0/catalogues/input/gals_info_{TILE}.feather"
        if not os.path.exists(fp):
            print(f"case{c}: MISSING {fp}", flush=True); continue
        t = pf.read_table(fp, columns=["index_input", "RA_input", "DEC_input", "r_input", "Re_input"]).to_pandas()
        pos = t[["RA_input", "DEC_input"]].to_numpy(float)
        flux = 10 ** (-0.4 * (t["r_input"].to_numpy(float) - ZERO_MAG))
        r = t["r_input"].to_numpy(float); Re = t["Re_input"].to_numpy(float)
        in_domain = (r > R_LO) & (r < R_HI) & (Re > RE_LO) & (Re < RE_HI)
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=R_MAX, output_type="ndarray")
        f_ood = np.zeros(len(t)); f_ind = np.zeros(len(t))
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            # neighbour j contributes to i (and vice versa), split by the NEIGHBOUR's domain status
            for a, b in ((i, j), (j, i)):   # a = target, b = neighbour
                np.add.at(f_ind, a[in_domain[b]], flux[b[in_domain[b]]])
                np.add.at(f_ood, a[~in_domain[b]], flux[b[~in_domain[b]]])
        parts.append(pd.DataFrame({
            "case": c, "input_index": t["index_input"].to_numpy(np.int64),
            "ood_flux": np.log10(1.0 + f_ood / ap_rms),
            "ind_flux": np.log10(1.0 + f_ind / ap_rms)}))
        print(f"case{c}: {len(t):,} gals  <ood>={parts[-1]['ood_flux'].mean():.3f} "
              f"<ind>={parts[-1]['ind_flux'].mean():.3f}  ood/in gals frac={np.mean(f_ood>0):.3f}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases", flush=True)


if __name__ == "__main__":
    main()
