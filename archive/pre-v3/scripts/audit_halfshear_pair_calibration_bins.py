"""Direct held-out pair-label audit in the frozen vector-calibration bins."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from analyze_halfshear_vector_closure import stat  # noqa: E402


FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
NEED = ["case", "input_index", "delta_et1", "delta_et2", *FEATURES]
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)


def summarize(frame: pd.DataFrame, n_bins: int) -> dict:
    result = {}
    for index in range(n_bins):
        subset = frame.loc[frame.bin == index].copy()
        case = subset.groupby("case", sort=True)[["label_sum", "prediction_sum", "count"]].sum()
        case["label_mean"] = case.label_sum / case["count"]
        case["prediction_mean"] = case.prediction_sum / case["count"]
        case["gap"] = case.prediction_mean - case.label_mean
        result[str(index)] = {
            "n_pairs": int(case["count"].sum()),
            "label": stat(case.label_mean),
            "prediction": stat(case.prediction_mean),
            "prediction_minus_label": stat(case.gap),
            "pooled_label_over_prediction": float(case.label_sum.sum() / case.prediction_sum.sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--shear", type=float, default=0.2)
    parser.add_argument("--case-min", type=int, default=0)
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--model-tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--bin-model-tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    calibration = json.load(open(args.calibration, encoding="utf-8"))
    boundary = np.asarray([
        -np.inf if value is None and index == 0 else
        np.inf if value is None else float(value)
        for index, value in enumerate(calibration["response_bin_edges"])
    ])
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.model_tag,
        conditions=COND, device="cpu",
    )
    bin_predictor = predictor if args.bin_model_tag == args.model_tag else BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.bin_model_tag,
        conditions=COND, device="cpu",
    )
    cuts, _, _ = predictor._select("regression")
    pieces = []
    with ipc.open_file(args.catalogue) as reader:
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            frame = pa.Table.from_batches([batch]).select(NEED).to_pandas()
            case_values = frame.case.to_numpy(int)
            frame = frame.loc[
                (case_values >= args.case_min) & (case_values < args.n_cases)
            ]
            if frame.empty:
                batch_cases = batch.column(reader.schema.get_field_index("case")).to_numpy()
                if len(batch_cases) and int(np.min(batch_cases)) >= args.n_cases:
                    break
                continue
            finite = np.isfinite(frame[["delta_et1", "delta_et2", *FEATURES]]).all(axis=1)
            selected = finite & (
                (frame.r_input_s > cuts[0][0]) & (frame.r_input_s < cuts[0][1])
                & (frame.r_input_p > cuts[1][0]) & (frame.r_input_p < cuts[1][1])
                & (frame.Re_input_s > cuts[2][0]) & (frame.Re_input_s < cuts[2][1])
                & (frame.Re_input_p > cuts[3][0]) & (frame.Re_input_p < cuts[3][1])
                & (frame.distance > cuts[4][0]) & (frame.distance < cuts[4][1])
            )
            pairs = frame.loc[selected]
            if pairs.empty:
                continue
            prediction = predictor.predict_on_pairs(
                pairs[FEATURES], task="response",
            ).response.to_numpy(float)
            bin_prediction = (
                prediction if bin_predictor is predictor else
                bin_predictor.predict_on_pairs(
                    pairs[FEATURES], task="response",
                ).response.to_numpy(float)
            )
            bin_index = np.clip(
                np.searchsorted(boundary, bin_prediction, side="right") - 1,
                0, len(boundary) - 2,
            )
            scored = pd.DataFrame({
                "case": pairs.case.to_numpy(int), "bin": bin_index,
                "label_sum": pairs.delta_et1.to_numpy(float) / args.shear,
                "prediction_sum": prediction, "count": 1,
            })
            pieces.append(scored.groupby(["case", "bin"], as_index=False).sum())
    totals = pd.concat(pieces, ignore_index=True).groupby(
        ["case", "bin"], as_index=False,
    ).sum()
    development = totals.case <= args.development_max
    development_name = f"development_c{args.case_min}_{args.development_max}"
    validation_name = f"validation_c{args.development_max + 1}_{args.n_cases - 1}"
    payload = {
        "design": "direct per-pair projected labels in response-power bin edges frozen by c0--19 aggregate-vector fit",
        "case_window": [args.case_min, args.n_cases - 1],
        "response_bin_edges": calibration["response_bin_edges"],
        "model_tag": args.model_tag,
        "bin_model_tag": args.bin_model_tag,
        "fitted_vector_coefficients": calibration["coefficients"],
        development_name: summarize(totals.loc[development], len(boundary) - 1),
        validation_name: summarize(totals.loc[~development], len(boundary) - 1),
        "constgold_opened": False, "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_PAIR_CALIBRATION_BIN_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
