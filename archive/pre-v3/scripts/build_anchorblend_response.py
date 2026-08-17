"""Measure neighbour-only coherent response on sparse unsheared anchors.

For each case, anchor shapes are matched between the +g and -g renders.  The
anchors themselves have zero input shear.  In the original coherent design the
response is the g1 central difference.  With ``--anchor-directions``, each
manifest anchor instead carries its own stable spin-2 direction; the difference
is projected on that direction and an orthogonal null is retained.  Existing
pairwise BlendEMU predictions are evaluated on the same input field.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from blendemu.response import retrieve_constant_shear  # noqa: E402
from sbs_shear.population import primary_mask  # noqa: E402

COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"


def label(g):
    return str(float(g))


def project_anchor_response(de1, de2, u1, u2, g):
    """Project an antithetic shape difference on a unit spin-2 direction/null."""
    scale = 2.0 * abs(float(g))
    return (
        (np.asarray(de1) * np.asarray(u1) + np.asarray(de2) * np.asarray(u2)) / scale,
        (-np.asarray(de1) * np.asarray(u2) + np.asarray(de2) * np.asarray(u1)) / scale,
    )


def emulator_sum(predictor, base, case, sign, anchors):
    path = os.path.join(
        base, f"case{case}_{label(sign)}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    frame = frame.rename(columns={c: c.replace("_input", "") for c in frame.columns})
    pairs = predictor.predict_response(frame, frame)
    primary = next(c for c in pairs.columns if c.startswith("index") and c.endswith("_p"))
    summed = pairs.groupby(primary, sort=False)["response"].sum()
    return summed.reindex(anchors, fill_value=0.0).to_numpy(float)


def unsupported_near_anchor(base, case, sign, anchor_ids, radius_arcsec=10.0):
    """Flag anchors whose response aperture contains an out-of-support source."""
    from scipy.spatial import cKDTree

    frame = pd.read_feather(
        os.path.join(base, f"gals{case}_{label(sign)}.feather"),
        columns=["index", "RA", "DEC", "r", "Re"],
    )
    supported = (
        np.isfinite(frame["r"]) & np.isfinite(frame["Re"])
        & (frame["r"] > 13.0) & (frame["r"] < 29.0)
        & (frame["Re"] > 0.0) & (frame["Re"] < 10.0)
    ).to_numpy()
    if supported.all():
        return np.zeros(len(anchor_ids), dtype=bool)
    by_id = frame.set_index("index", verify_integrity=True)
    anchors = by_id.loc[np.asarray(anchor_ids, dtype=np.int64)]
    dec0 = float(np.median(frame["DEC"]))
    cosdec = np.cos(np.deg2rad(dec0))
    source_xy = np.column_stack([
        frame.loc[~supported, "RA"].to_numpy(float) * cosdec * 3600.0,
        frame.loc[~supported, "DEC"].to_numpy(float) * 3600.0,
    ])
    anchor_xy = np.column_stack([
        anchors["RA"].to_numpy(float) * cosdec * 3600.0,
        anchors["DEC"].to_numpy(float) * 3600.0,
    ])
    distance = cKDTree(source_xy).query(anchor_xy, k=1)[0]
    return distance <= float(radius_arcsec)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.02)
    ap.add_argument("--tags", nargs="+", default=["lsst_r_extnbr_v21", "lsst_r_extnbr_ho"])
    ap.add_argument(
        "--primary-domain", choices=("original", "v2", "v2-complement"),
        default="original",
        help="truth-level primary population retained after both-leg matching",
    )
    ap.add_argument(
        "--anchor-directions", action="store_true",
        help="project each anchor on manifest columns u1/u2 and write a 45-degree null",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    predictors = {
        tag: BlendingPredictor.load(
            "/home/z/Zekang.Zhang/blendemu/models", tag=tag,
            conditions=COND, device="cpu",
        )
        for tag in args.tags
    }
    parts = []
    for case in args.cases:
        manifest = pd.read_feather(os.path.join(args.base, f"anchors_case{case}.feather"))
        if args.anchor_directions:
            missing_direction = {"u1", "u2"} - set(manifest.columns)
            if missing_direction:
                raise RuntimeError(
                    f"case {case}: manifest lacks direction columns {sorted(missing_direction)}"
                )
            norm = np.hypot(manifest["u1"].to_numpy(float), manifest["u2"].to_numpy(float))
            if not np.allclose(norm, 1.0, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"case {case}: non-unit anchor direction")
        anchors = manifest["index"].to_numpy(np.int64)
        legs = []
        for sign in (args.g, -args.g):
            d = retrieve_constant_shear(
                case, label(sign), r_max=10.0, r_min=0.0, k=20,
                data_path=args.base, shape_suffix="_all", include_isolated=True,
            )
            d = d[d["input_index"].isin(anchors)][
                ["input_index", "measured_e1", "measured_e2", "r_input_p", "Re_input_p"]
            ].copy()
            if d["input_index"].duplicated().any():
                raise RuntimeError(f"case {case} sign {sign}: duplicate anchor ids")
            legs.append(d)
        joined = legs[0].merge(
            legs[1], on="input_index", suffixes=("_plus", "_minus"),
            how="inner", validate="one_to_one",
        )
        coverage = len(joined) / max(len(anchors), 1)
        if coverage < 0.8:
            raise RuntimeError(f"case {case}: both-leg anchor coverage {coverage:.2%} < 80%")
        # Enforce the exact shared intrinsic primary population again at
        # measurement so this truth and the deployed constgold score have the
        # same ordinary support box + V2.1 Re/SN domain.
        for name in ("r_input_p", "Re_input_p"):
            if not np.array_equal(
                joined[f"{name}_plus"].to_numpy(), joined[f"{name}_minus"].to_numpy()
            ):
                raise RuntimeError(f"case {case}: +/- primary truth column {name} differs")
        pop_frame = pd.DataFrame({
            "r_input_p": joined["r_input_p_plus"],
            "Re_input_p": joined["Re_input_p_plus"],
        })
        original_primary = primary_mask(pop_frame)
        mag = pop_frame["r_input_p"].to_numpy(float)
        size = pop_frame["Re_input_p"].to_numpy(float)
        v2_primary = (
            np.isfinite(mag) & np.isfinite(size)
            & (mag > 18.0) & (mag < 26.0)
            & (size > 0.3) & (size < 1.5)
        )
        if args.primary_domain == "original":
            shared_primary = original_primary
        elif args.primary_domain == "v2":
            shared_primary = v2_primary
        else:
            shared_primary = v2_primary & ~original_primary
        joined = joined.loc[shared_primary].copy()
        unsupported_near = unsupported_near_anchor(
            args.base, case, args.g, joined["input_index"].to_numpy(np.int64),
        )
        unsupported_removed = int(unsupported_near.sum())
        joined = joined.loc[~unsupported_near].copy()
        if len(joined) < 100:
            raise RuntimeError(f"case {case}: only {len(joined)} shared-population anchors")
        matched_shared = len(joined)
        de1 = (
            joined["measured_e1_plus"].to_numpy(float)
            - joined["measured_e1_minus"].to_numpy(float)
        )
        de2 = (
            joined["measured_e2_plus"].to_numpy(float)
            - joined["measured_e2_minus"].to_numpy(float)
        )
        if args.anchor_directions:
            direction = manifest.set_index("index", verify_integrity=True)[["u1", "u2"]]
            u1 = joined["input_index"].map(direction["u1"]).to_numpy(float)
            u2 = joined["input_index"].map(direction["u2"]).to_numpy(float)
            if not np.isfinite(u1).all() or not np.isfinite(u2).all():
                raise RuntimeError(f"case {case}: missing joined anchor direction")
            truth, null = project_anchor_response(de1, de2, u1, u2, args.g)
            joined["R_blend_truth"] = truth
            joined["R_blend_null"] = null
        else:
            joined["R_blend_truth"] = de1 / (2.0 * abs(args.g))
        for tag, predictor in predictors.items():
            values = emulator_sum(predictor, args.base, case, args.g, anchors)
            lookup = pd.Series(values, index=anchors)
            joined[f"R_blend_{tag}"] = lookup.reindex(joined["input_index"]).to_numpy(float)
        response_cols = ["R_blend_truth"]
        if args.anchor_directions:
            response_cols.append("R_blend_null")
        response_cols.extend(f"R_blend_{t}" for t in args.tags)
        finite = np.isfinite(joined[response_cols].to_numpy(float)).all(axis=1)
        invalid = int((~finite).sum())
        joined = joined.loc[finite].copy()
        if len(joined) < 0.99 * matched_shared:
            raise RuntimeError(
                f"case {case}: more than 1% of matched responses are non-finite "
                f"({invalid}/{matched_shared})"
            )
        joined["case"] = case
        parts.append(joined)
        means = " ".join(f"{c}={joined[c].mean():+.5f}" for c in response_cols)
        print(
            f"case {case}: anchors={len(anchors):,}, matched={len(joined):,}, "
            f"unsupported-near={unsupported_removed}, invalid={invalid} {means}",
            flush=True,
        )

    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    case_means = out.groupby("case")[["R_blend_truth", *[f"R_blend_{t}" for t in args.tags]]].mean()
    print("\nCASE-MEAN SUMMARY")
    for col in case_means:
        values = case_means[col].to_numpy(float)
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        print(f"  {col}: {values.mean():+.6f} +/- {sem:.6f}")
    print(f"wrote {args.output}: {len(out):,} matched anchors")
    print("ANCHORBLEND_RESPONSE_DONE", flush=True)


if __name__ == "__main__":
    main()
