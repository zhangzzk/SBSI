"""Per-(case, input_index) CROWDING summary: total neighbour flux in distance shells.

For each galaxy, sum the flux of ALL input-field neighbours within two shells (0-3", 3-7")
and store as log10(1 + F_shell / aperture_rms) -- a well-behaved, noise-relative crowding
feature (0 for isolated, growing with crowding), consistent with the pipeline's r_*_scaled.

Keyed by (case, input_index) so it joins the det_meas catalogues like the blend lookup.
Output columns: case, input_index, nbr_flux_near, nbr_flux_far, nbr_flux_max.

nbr_flux_max = log10(1 + F_brightest_neighbour / aperture_rms): the flux of the SINGLE
brightest neighbour within FAR. Together with the near/far SUMS it gives the flow a flux
CONCENTRATION signal (one bright close blend vs many faint) that summed flux alone hides --
the confirm-first orthogonality test (WORKLOG cont.15) showed the flow's self-response deficit
depends on this at fixed nbr_flux, so a bright-neighbour dominance feature is the root-cause fix.
"""
import argparse, os
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

BASE_CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
TILE = "tile180.0_-0.5"
# rescale constants (LSST r) -> aperture noise scale, matching sbs_shear.preprocessing.rescale
PIXEL_RMS, PIXEL_SIZE, PSF_FWHM, MOFFAT_BETA, ZERO_MAG = 0.312, 0.2, 0.73, 2.224, 30.0
NEAR, FAR = 3.0 / 3600.0, 7.0 / 3600.0   # degrees


def _aperture_rms():
    factor = np.sqrt((2 ** (1 / (MOFFAT_BETA - 1)) - 1) / (2 ** (1 / MOFFAT_BETA) - 1)) / 2
    psf_size = PSF_FWHM * factor
    return PIXEL_RMS * (psf_size / PIXEL_SIZE) ** 2 * np.pi


def input_feather(case, sign, base):
    return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.0")
    ap.add_argument("--base", default=BASE_CONST)
    args = ap.parse_args()
    ap_rms = _aperture_rms()

    parts = []
    for c in args.cases:
        fp = input_feather(c, args.sign, args.base)
        if not os.path.exists(fp):
            print(f"case{c}: MISSING {fp}", flush=True); continue
        t = pf.read_table(fp, columns=["index_input", "RA_input", "DEC_input", "r_input"]).to_pandas()
        pos = t[["RA_input", "DEC_input"]].to_numpy(float)
        flux = 10 ** (-0.4 * (t["r_input"].to_numpy(float) - ZERO_MAG))   # mag2flux
        tree = cKDTree(pos)
        # all pairs within FAR (i<j), vectorized -> symmetric flux accumulation
        pairs = tree.query_pairs(r=FAR, output_type="ndarray")
        f_near = np.zeros(len(t)); f_far = np.zeros(len(t)); f_max = np.zeros(len(t))
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            near = d < NEAR; far = ~near
            np.add.at(f_near, i[near], flux[j[near]]); np.add.at(f_near, j[near], flux[i[near]])
            np.add.at(f_far, i[far], flux[j[far]]); np.add.at(f_far, j[far], flux[i[far]])
            # brightest single neighbour (over ALL neighbours within FAR) -> flux concentration
            np.maximum.at(f_max, i, flux[j]); np.maximum.at(f_max, j, flux[i])
        parts.append(pd.DataFrame({
            "case": c, "input_index": t["index_input"].to_numpy(np.int64),
            "nbr_flux_near": np.log10(1.0 + f_near / ap_rms),
            "nbr_flux_far": np.log10(1.0 + f_far / ap_rms),
            "nbr_flux_max": np.log10(1.0 + f_max / ap_rms)}))
        print(f"case{c}: {len(t):,} gals  <nbr_flux_near>={parts[-1]['nbr_flux_near'].mean():.3f} "
              f"<far>={parts[-1]['nbr_flux_far'].mean():.3f} "
              f"<max>={parts[-1]['nbr_flux_max'].mean():.3f}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases", flush=True)


if __name__ == "__main__":
    main()
