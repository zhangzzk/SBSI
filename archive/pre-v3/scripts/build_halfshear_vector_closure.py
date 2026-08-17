"""Build a case-balanced V2.2 vector-closure table from half-shear pairs."""
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
from analyze_halfshear_vector_closure import fit  # noqa: E402


FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
NEED = [
    "case", "input_index", "shear_angle", "delta_et1", "delta_et2", *FEATURES,
]
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--case-min", type=int, default=0)
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--shear", type=float, required=True)
    parser.add_argument("--model-tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.model_tag,
        conditions=COND, device="cpu",
    )
    cuts, _, _ = predictor._select("regression")
    parts = []
    with ipc.open_file(args.catalogue) as reader:
        missing = set(NEED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
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
            finite = np.isfinite(
                frame[["delta_et1", "delta_et2", "shear_angle", *FEATURES]].to_numpy(float)
            ).all(axis=1)
            selected = finite & (
                (frame.r_input_s > cuts[0][0]) & (frame.r_input_s < cuts[0][1])
                & (frame.r_input_p > cuts[1][0]) & (frame.r_input_p < cuts[1][1])
                & (frame.Re_input_s > cuts[2][0]) & (frame.Re_input_s < cuts[2][1])
                & (frame.Re_input_p > cuts[3][0]) & (frame.Re_input_p < cuts[3][1])
                & (frame.distance > cuts[4][0]) & (frame.distance < cuts[4][1])
            )
            pairs = frame.loc[selected].copy()
            if pairs.empty:
                continue
            prediction = predictor.predict_on_pairs(
                pairs[FEATURES], task="response",
            ).response.to_numpy(float)
            phase = np.deg2rad(2.0 * pairs.shear_angle.to_numpy(float))
            cosine, sine = np.cos(phase), np.sin(phase)
            scored = pairs[["case", "input_index"]].copy()
            scored["label"] = pairs.delta_et1.to_numpy(float) / args.shear
            scored["null"] = pairs.delta_et2.to_numpy(float) / args.shear
            scored["cos_shear"] = cosine
            scored["sin_shear"] = sine
            scored["prediction"] = prediction
            scored["prediction_cos"] = prediction * cosine
            scored["prediction_sin"] = prediction * sine
            parts.append(scored.groupby(["case", "input_index"], as_index=False).agg(
                label=("label", "sum"), null=("null", "sum"),
                cos_shear=("cos_shear", "sum"), sin_shear=("sin_shear", "sum"),
                prediction=("prediction", "sum"),
                prediction_cos=("prediction_cos", "sum"),
                prediction_sin=("prediction_sin", "sum"),
                n_pairs=("label", "size"),
            ))
            if len(parts) % 100 == 0:
                print(f"processed {len(parts)} selected batches", flush=True)
    table = pd.concat(parts, ignore_index=True).groupby(
        ["case", "input_index"], as_index=False,
    ).sum()
    direction_power = table.cos_shear**2 + table.sin_shear**2
    if (direction_power <= 1e-8).any():
        raise RuntimeError("encountered a numerically vanishing direction sum")
    table["y1"] = (
        table.label * table.cos_shear - table.null * table.sin_shear
    ) / direction_power
    table["y2"] = (
        table.label * table.sin_shear + table.null * table.cos_shear
    ) / direction_power
    table["power"] = table.prediction_cos**2 + table.prediction_sin**2
    table["dot"] = table.prediction_cos * table.y1 + table.prediction_sin * table.y2
    table["cross"] = -table.prediction_sin * table.y1 + table.prediction_cos * table.y2
    table.to_feather(args.table_output)
    development = table.case <= args.development_max
    payload = {
        "design": "streamed exact V2.2-domain half-shear pairs; measured response vector reconstructed per primary and fitted on frozen aggregate model vector",
        "model_tag": args.model_tag,
        "shear": float(args.shear),
        "all": fit(table),
        f"development_c{args.case_min}_{args.development_max}": fit(table.loc[development]),
        f"validation_c{args.development_max + 1}_{args.n_cases - 1}": fit(table.loc[~development]),
        "case_window": [args.case_min, args.n_cases - 1],
        "n_rows": int(len(table)),
        "n_cases": int(table.case.nunique()),
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_VECTOR_CLOSURE_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
