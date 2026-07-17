"""DETECTED-ONLY crowding lookup: nbr_flux_near/far/max summed over DETECTED neighbours.

Step 1/2 of the etilde probabilistic-blending ladder (WORKLOG cont.32, mirroring the
validated R-space PROB_BLENDING ladder): the production crowd features
(build_crowding_lookup.py) sum the flux of ALL input-field neighbours -- deployment
only sees detected ones.  This builder tags every neighbour with its g=0 detection
truth and accumulates only detected neighbours:

  --flux true      : detected neighbours at their TRUE flux (step 1 -- isolates the
                     undetected-census effect; the R-space analogue cost Dm=+3.29%)
  --flux measured  : detected neighbours at their MEASURED flux_auto (step 2 --
                     adds deployment flux noise; needs --meas-catalogue)

Same keying/columns as the production lookup, so the etilde driver consumes it via
--crowd-flux-lookup with no code change.  Undetected/unmapped neighbours contribute 0
(the step-3 forward-modelled undetected component is added separately).
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc
from scipy.spatial import cKDTree

BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
DET = f"{BASE}/detection_catalogue_train.feather"
MEAS = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.0_train_full.feather"
TILE = "tile180.0_-0.5"
PIXEL_RMS, PIXEL_SIZE, PSF_FWHM, MOFFAT_BETA, ZERO_MAG = 0.312, 0.2, 0.73, 2.224, 30.0
NEAR, FAR = 3.0 / 3600.0, 7.0 / 3600.0   # degrees


def _aperture_rms():
    factor = np.sqrt((2 ** (1 / (MOFFAT_BETA - 1)) - 1) / (2 ** (1 / MOFFAT_BETA) - 1)) / 2
    psf_size = PSF_FWHM * factor
    return PIXEL_RMS * (psf_size / PIXEL_SIZE) ** 2 * np.pi


def stream_keyed(path, columns, cases):
    """Stream a big keyed feather, keeping only the requested cases."""
    parts = []
    want = set(int(c) for c in cases)
    with ipc.open_file(path) as r:
        cols = [c for c in columns if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[b["case"].isin(want)]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--flux", choices=["true", "measured"], default="true")
    ap.add_argument("--det-catalogue", default=DET)
    ap.add_argument("--meas-catalogue", default=MEAS)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--sign", default="0.0")
    args = ap.parse_args()
    ap_rms = _aperture_rms()

    print(f"detection flags <- {args.det_catalogue}", flush=True)
    det = stream_keyed(args.det_catalogue, ["case", "input_index", "detected"], args.cases)
    det = det.drop_duplicates(["case", "input_index"])
    print(f"  {len(det):,} rows over {det['case'].nunique()} cases, "
          f"det rate {det['detected'].mean():.3f}", flush=True)
    meas = None
    if args.flux == "measured":
        print(f"measured fluxes <- {args.meas_catalogue}", flush=True)
        meas = stream_keyed(args.meas_catalogue,
                            ["case", "input_index", "detected", "measured_mag_auto"],
                            args.cases)
        meas = meas[meas["detected"].astype(bool)].drop_duplicates(["case", "input_index"])
        print(f"  {len(meas):,} measured rows", flush=True)

    parts = []
    for c in args.cases:
        fp = f"{args.base}/case{c}_{args.sign}/real0/catalogues/input/gals_info_{TILE}.feather"
        if not os.path.exists(fp):
            print(f"case{c}: MISSING {fp}", flush=True)
            continue
        t = pf.read_table(fp, columns=["index_input", "RA_input", "DEC_input",
                                       "r_input"]).to_pandas()
        idx = t["index_input"].to_numpy(np.int64)
        pos = t[["RA_input", "DEC_input"]].to_numpy(float)
        flux_true = 10 ** (-0.4 * (t["r_input"].to_numpy(float) - ZERO_MAG))

        dc = det[det["case"] == c]
        dmap = pd.Series(dc["detected"].to_numpy(bool), index=dc["input_index"]).reindex(idx)
        mapped = float(dmap.notna().mean())
        det_ok = dmap.fillna(False).to_numpy(bool)  # unmapped -> undetected (below det limit)
        if args.flux == "measured":
            mc = meas[meas["case"] == c]
            mmag = pd.Series(mc["measured_mag_auto"].to_numpy(float),
                             index=mc["input_index"]).reindex(idx)
            cflux = 10 ** (-0.4 * (mmag.to_numpy(float) - ZERO_MAG))
            good = det_ok & np.isfinite(cflux)
            cflux = np.where(good, cflux, 0.0)
            det_ok = good
        else:
            cflux = np.where(det_ok, flux_true, 0.0)

        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=FAR, output_type="ndarray")
        f_near = np.zeros(len(t)); f_far = np.zeros(len(t)); f_max = np.zeros(len(t))
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            near = d < NEAR
            for a, b in ((i, j), (j, i)):     # only DETECTED neighbours b contribute to a
                ok = det_ok[b]
                sn = near & ok
                sf = (~near) & ok
                np.add.at(f_near, a[sn], cflux[b[sn]])
                np.add.at(f_far, a[sf], cflux[b[sf]])
                np.maximum.at(f_max, a[ok], cflux[b[ok]])
        parts.append(pd.DataFrame({
            "case": c, "input_index": idx,
            "nbr_flux_near": np.log10(1.0 + f_near / ap_rms),
            "nbr_flux_far": np.log10(1.0 + f_far / ap_rms),
            "nbr_flux_max": np.log10(1.0 + f_max / ap_rms)}))
        print(f"case{c}: {len(t):,} gals  mapped={mapped:.1%} det(nbr pool)={det_ok.mean():.2%}  "
              f"<near>={parts[-1]['nbr_flux_near'].mean():.3f} "
              f"<far>={parts[-1]['nbr_flux_far'].mean():.3f} "
              f"<max>={parts[-1]['nbr_flux_max'].mean():.3f}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases", flush=True)


if __name__ == "__main__":
    main()
