"""Per-(case, input_index) OUT-OF-DOMAIN neighbour flux, SPLIT by neighbour brightness.

The emulator in-domain cut is r in (18,28), Re in (0.1,1.5). OOD neighbours split cleanly:
  - BRIGHT ood  (r < 24): the ~0.02% r<18 objects (24% of field flux) + bright Re-extreme.
                 A bright uncounted neighbour -> LARGE true blend the emulator sets to 0.
  - FAINT  ood  (r >= 24, i.e. r>28 or faint Re-extreme): ~0 true blend (toy: -0.005 at 2.5 mag
                 fainter). A cut on this is a SELECTION CONTROL: removes similarly-crowded
                 primaries but no real uncounted blend.

If cutting BRIGHT-ood moves m but cutting FAINT-ood does not, the bright uncounted blend is a
REAL under-count (mechanism B), not a crowding-selection artifact.

Columns: case, input_index, ood_flux_bright, ood_flux_faint, ood_flux (total, back-compat).
Same log10(1+F/aperture_rms) scaling as build_ood_lookup.
"""
import argparse, os
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

TILE = "tile180.0_-0.5"
PIXEL_RMS, PIXEL_SIZE, PSF_FWHM, MOFFAT_BETA, ZERO_MAG = 0.312, 0.2, 0.73, 2.224, 30.0
R_MAX = 10.0 / 3600.0
R_LO, R_HI, RE_LO, RE_HI = 18.0, 28.0, 0.1, 1.5
BRIGHT_SPLIT = 24.0   # OOD neighbour with r < this = "bright" (large uncounted blend)


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
        r = np.clip(t["r_input"].to_numpy(np.float64), 10.0, 40.0)
        Re = t["Re_input"].to_numpy(np.float64)
        flux = 10.0 ** (-0.4 * (r - ZERO_MAG))
        in_domain = (r > R_LO) & (r < R_HI) & (Re > RE_LO) & (Re < RE_HI)
        ood_bright_nbr = (~in_domain) & (r < BRIGHT_SPLIT)   # bright uncounted neighbour
        ood_faint_nbr = (~in_domain) & (r >= BRIGHT_SPLIT)   # faint uncounted neighbour
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=R_MAX, output_type="ndarray")
        f_b = np.zeros(len(t)); f_f = np.zeros(len(t))
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            for a, b in ((i, j), (j, i)):   # a = target primary, b = neighbour
                np.add.at(f_b, a[ood_bright_nbr[b]], flux[b[ood_bright_nbr[b]]])
                np.add.at(f_f, a[ood_faint_nbr[b]], flux[b[ood_faint_nbr[b]]])
        ob = np.log10(1.0 + f_b / ap_rms); of = np.log10(1.0 + f_f / ap_rms)
        tot = np.log10(1.0 + (f_b + f_f) / ap_rms)
        parts.append(pd.DataFrame({
            "case": c, "input_index": t["index_input"].to_numpy(np.int64),
            "ood_flux_bright": ob, "ood_flux_faint": of, "ood_flux": tot}))
        print(f"case{c}: {len(t):,} gals  <ood_bright>={ob.mean():.3f} (gals w/ bright-ood {np.mean(f_b>0):.3%})  "
              f"<ood_faint>={of.mean():.3f} (gals w/ faint-ood {np.mean(f_f>0):.3%})", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases", flush=True)


if __name__ == "__main__":
    main()
