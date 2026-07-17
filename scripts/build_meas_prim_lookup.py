"""Per-(case, input_index) MEASURED PRIMARY observables for the constant-gold validation, so the
realistic flow (WORKLOG cont.19, feature sets g0_meas_crowd_conc_*) can be SCORED on constgold.

The constgold `constant_response_catalogue_*` carry only measured shape (e1/e2_plus/minus) + S/N, NOT
the measured own-property observables (mag_auto, flux_radius, class_star, ...) the realistic flow now
conditions on. Those raw measurements DO exist per case/sign under
  case{c}_{sign}/real0/catalogues/SExtractor/{TILE}_bandr_rot0.feather   (NUMBER + measured cols)
  case{c}_{sign}/real0/catalogues/CrossMatch/{TILE}_rot0_matched.feather (id_detec<->id_input)
and the training det_meas pipeline's measured_* columns are DIRECT renames of these raw SExtractor
columns (verified: identical mag_auto/flux_radius/class_star distributions), so no transform is needed.

Shear-symmetry (verify fix): constgold has only +/-g renders (no g=0). A conditioner held fixed across
the +/-delta response derivative must be shear-EVEN, so we AVERAGE each measured value over the +0.02
and -0.02 renders per (case, input_index) -- mirroring the S/N = 0.5*(S/N_plus+S/N_minus) convention.

Output columns (lowercase measured_* to match the training feature names): case, input_index,
measured_mag_auto, measured_flux_radius, measured_class_star, measured_flux_auto, measured_fluxerr_auto,
measured_mag_aper, measured_fwhm_image, measured_isoarea_image.
"""
import argparse, os
import numpy as np
import pandas as pd
import pyarrow.feather as pf

BASE_CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
TILE = "tile180.0_-0.5"
# raw SExtractor col -> output measured_* name
COLMAP = {
    "MAG_AUTO": "measured_mag_auto",
    "FLUX_RADIUS": "measured_flux_radius",
    "CLASS_STAR": "measured_class_star",
    "FLUX_AUTO": "measured_flux_auto",
    "FLUXERR_AUTO": "measured_fluxerr_auto",
    "MAG_APER": "measured_mag_aper",
    "FWHM_IMAGE": "measured_fwhm_image",
    "ISOAREA_IMAGE": "measured_isoarea_image",
}


def sex_path(case, sign, base):
    return f"{base}/case{case}_{sign}/real0/catalogues/SExtractor/{TILE}_bandr_rot0.feather"


def cm_path(case, sign, base):
    return f"{base}/case{case}_{sign}/real0/catalogues/CrossMatch/{TILE}_rot0_matched.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base", default=BASE_CONST)
    ap.add_argument("--signs", nargs="+", default=["-0.02", "0.02"])
    args = ap.parse_args()

    per_sign = []            # one (case, input_index, measured_*) frame per case/sign
    for c in args.cases:
        got = 0
        for sign in args.signs:
            sp, cp = sex_path(c, sign, args.base), cm_path(c, sign, args.base)
            if not (os.path.exists(sp) and os.path.exists(cp)):
                continue
            sx = pf.read_table(sp, columns=["NUMBER", *COLMAP.keys()]).to_pandas()
            cm = pf.read_table(cp, columns=["id_detec", "id_input"]).to_pandas()
            m = cm.merge(sx, left_on="id_detec", right_on="NUMBER", how="inner")
            out = m[["id_input"]].rename(columns={"id_input": "input_index"}).copy()
            for raw, name in COLMAP.items():
                out[name] = m[raw].to_numpy(float)
            out["case"] = c
            per_sign.append(out)
            got += len(out)
        print(f"case{c}: {got:,} matched detections over signs {args.signs}", flush=True)

    allrows = pd.concat(per_sign, ignore_index=True)
    # shear-symmetric: mean over both signs (and any duplicate matches) per (case, input_index)
    agg = allrows.groupby(["case", "input_index"], as_index=False)[list(COLMAP.values())].mean()
    agg.to_feather(args.output)
    print(f"wrote {args.output}: {len(agg):,} (case,input_index) rows over {agg['case'].nunique()} cases",
          flush=True)
    for name in ("measured_mag_auto", "measured_flux_radius", "measured_class_star"):
        v = agg[name].to_numpy(float); v = v[np.isfinite(v)]
        print(f"  {name}: med={np.median(v):.3f} mean={v.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
