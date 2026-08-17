"""Build projected random-neighbour labels on matched anchor scenes."""
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


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"


def input_frame(base: str, case: int, sign: float) -> pd.DataFrame:
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={column: column.replace("_input", "") for column in frame})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.tag, conditions=COND, device="cpu",
    )
    parts = []
    for case in args.cases:
        manifest = pd.read_feather(os.path.join(args.base, f"anchors_case{case}.feather"))
        anchor_ids = manifest["index"].to_numpy(np.int64)
        legs = []
        for sign in (args.g, -args.g):
            leg = retrieve_constant_shear(
                case, sign, r_max=10.0, r_min=0.0, k=20, data_path=args.base,
                shape_suffix="_all", include_isolated=True,
            )
            legs.append(leg.loc[leg.input_index.isin(anchor_ids), [
                "input_index", "measured_e1", "measured_e2",
                "r_input_p", "Re_input_p",
            ]])
        measured = legs[0].merge(
            legs[1], on="input_index", suffixes=("_plus", "_minus"),
            how="inner", validate="one_to_one",
        )
        finite = np.isfinite(measured[[
            "measured_e1_plus", "measured_e2_plus",
            "measured_e1_minus", "measured_e2_minus",
        ]].to_numpy(float)).all(axis=1)
        measured = measured.loc[finite].copy()
        frame = input_frame(args.base, case, args.g)
        pairs = predictor.predict_response(frame, frame)
        primary = next(
            column for column in pairs if column.startswith("index") and column.endswith("_p")
        )
        pairs = pairs[pairs[primary].isin(measured.input_index)].copy()
        u1 = pairs.gamma1_input_s.to_numpy(float) / args.g
        u2 = pairs.gamma2_input_s.to_numpy(float) / args.g
        norm = np.hypot(u1, u2)
        if not np.allclose(norm, 1.0, rtol=0.0, atol=1e-10):
            raise RuntimeError(
                f"case {case}: predictor includes non-local/non-unit sheared pair, "
                f"max |norm-1|={np.max(np.abs(norm-1)):.3e}"
            )
        pairs["u1"] = u1
        pairs["u2"] = u2
        pairs["response_u1"] = pairs.response.to_numpy(float) * u1
        pairs["response_u2"] = pairs.response.to_numpy(float) * u2
        aggregate = pairs.groupby(primary, sort=False).agg(
            prediction=("response", "sum"), n_pairs=("response", "size"),
            sum_u1=("u1", "sum"), sum_u2=("u2", "sum"),
            prediction_u1=("response_u1", "sum"),
            prediction_u2=("response_u2", "sum"),
        )
        joined = measured.set_index("input_index").join(aggregate, how="inner").reset_index()
        de1 = joined.measured_e1_plus - joined.measured_e1_minus
        de2 = joined.measured_e2_plus - joined.measured_e2_minus
        scale = 2.0 * args.g
        joined["label_sum"] = (de1 * joined.sum_u1 + de2 * joined.sum_u2) / scale
        joined["null_sum"] = (-de1 * joined.sum_u2 + de2 * joined.sum_u1) / scale
        joined["projected_prediction"] = (
            joined.prediction_u1 * joined.sum_u1
            + joined.prediction_u2 * joined.sum_u2
        )
        joined["desired_gap"] = joined.prediction - joined.label_sum
        joined["projected_gap"] = joined.projected_prediction - joined.label_sum
        joined["case"] = int(case)
        parts.append(joined)
        print(
            f"case {case}: manifest={len(anchor_ids):,} measured={len(measured):,} "
            f"paired={len(joined):,} pairs={len(pairs):,} "
            f"pred={joined.prediction.mean():+.6f} label={joined.label_sum.mean():+.6f} "
            f"projected={joined.projected_prediction.mean():+.6f}", flush=True,
        )
    output = pd.concat(parts, ignore_index=True)
    output.to_feather(args.output)
    print(f"wrote {args.output}: rows={len(output):,}")
    print("ANCHORBLEND_INDEPENDENT_RESPONSE_DONE", flush=True)


if __name__ == "__main__":
    main()
