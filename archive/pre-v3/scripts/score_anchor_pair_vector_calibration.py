"""Score a frozen half-shear pair calibration on coherent anchor scenes."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from build_anchorblend_independent_response import COND, input_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-offset", type=int, required=True)
    parser.add_argument("--n-cases", type=int, default=1)
    parser.add_argument("--sign", type=float, default=0.05)
    args = parser.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if rank >= args.n_cases:
        return
    case = args.case_offset + rank
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"case{case}.feather")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    calibration = json.load(open(args.calibration, encoding="utf-8"))
    boundary = np.asarray([
        -np.inf if value is None and index == 0 else
        np.inf if value is None else float(value)
        for index, value in enumerate(calibration["response_bin_edges"])
    ])
    coefficient = np.asarray(calibration["coefficients"], float)
    if len(boundary) != len(coefficient) + 1:
        raise RuntimeError("calibration edge/coefficient mismatch")
    reference = pd.read_feather(args.reference)[
        ["case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22"]
    ]
    reference = reference.loc[reference.case == case].copy()
    if reference.empty or reference.input_index.duplicated().any():
        raise RuntimeError(f"case {case}: invalid reference")
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    frame = input_frame(args.base, case, args.sign)
    pairs = predictor.predict_response(frame, frame)
    primary = next(
        column for column in pairs if column.startswith("index") and column.endswith("_p")
    )
    pairs = pairs.loc[pairs[primary].isin(reference.input_index)].copy()
    bin_index = np.clip(
        np.searchsorted(boundary, pairs.response.to_numpy(float), side="right") - 1,
        0, len(coefficient) - 1,
    )
    pairs["corrected_response"] = pairs.response.to_numpy(float) * coefficient[bin_index]
    aggregate = pairs.groupby(primary, sort=False).agg(
        prediction_raw=("response", "sum"),
        prediction_corrected=("corrected_response", "sum"),
        n_pairs=("response", "size"),
    )
    aggregate.index.name = "input_index"
    scored = reference.set_index("input_index").join(aggregate, how="left").reset_index()
    missing = scored.prediction_raw.isna()
    if np.any(missing & (np.abs(scored.R_blend_lsst_r_extnbr_v22) > 2e-7)):
        raise RuntimeError(f"case {case}: nonzero reference lacks predicted pairs")
    scored.loc[missing, ["prediction_raw", "prediction_corrected", "n_pairs"]] = 0.0
    replay = np.max(np.abs(scored.prediction_raw - scored.R_blend_lsst_r_extnbr_v22))
    if replay > 2e-7:
        raise RuntimeError(f"case {case}: raw replay mismatch {replay:.3e}")
    scored["gap_raw"] = scored.prediction_raw - scored.R_blend_truth
    scored["gap_corrected"] = scored.prediction_corrected - scored.R_blend_truth
    scored["correction"] = scored.prediction_corrected - scored.prediction_raw
    scored.to_feather(output)
    print(
        f"case {case}: anchors={len(scored):,} raw={scored.prediction_raw.mean():+.6f} "
        f"corrected={scored.prediction_corrected.mean():+.6f} "
        f"truth={scored.R_blend_truth.mean():+.6f}", flush=True,
    )
    print("ANCHOR_PAIR_VECTOR_CALIBRATION_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
