"""Fit a rotation-equivariant pair-response calibration on half-shear scenes.

Frozen V2.2 pair predictions are divided into six signed-response bins carrying
equal prediction-squared weight on cases 0--19.  For each primary, every bin
contributes a two-component vector sum.  Six multipliers are fitted jointly to
the measured primary response vector, with cases balanced equally, then scored
once on cases 20--39.  This is an exploratory, reported calibration model; it
does not open anchors or constgold and writes no deployable BlendEMU model.
"""
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
NEED = [
    "case", "input_index", "shear_angle", "delta_et1", "delta_et2", *FEATURES,
]
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)


def weighted_edges(values: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(values)
    x, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    result = np.unique(np.interp(np.linspace(0, 1, count + 1), cumulative, x))
    if len(result) != count + 1:
        raise RuntimeError("response-power edges collapsed")
    result[0], result[-1] = -np.inf, np.inf
    return result


def vector_fit(frame: pd.DataFrame, prefix: str) -> dict:
    p1, p2 = frame[f"{prefix}_cos"], frame[f"{prefix}_sin"]
    work = pd.DataFrame({
        "case": frame.case,
        "power": p1**2 + p2**2,
        "dot": p1 * frame.y1 + p2 * frame.y2,
        "cross": -p2 * frame.y1 + p1 * frame.y2,
        "squared_error": (frame.y1 - p1)**2 + (frame.y2 - p2)**2,
    })
    case_sum = work.groupby("case", sort=True)[["power", "dot", "cross"]].sum()
    slopes = case_sum["dot"] / case_sum["power"]
    nulls = case_sum["cross"] / case_sum["power"]
    case_mse = work.groupby("case", sort=True).squared_error.mean()
    return {
        "slope_measured_on_predicted": stat(slopes),
        "slope_minus_one": stat(slopes - 1.0),
        "orthogonal_slope_null": stat(nulls),
        "mean_squared_vector_residual": stat(case_mse),
        "pooled_slope": float(case_sum["dot"].sum() / case_sum["power"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--shear", type=float, default=0.2)
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument(
        "--fixed-zero-split", action="store_true",
        help="use exactly two bins split at response zero",
    )
    parser.add_argument("--ridge-fraction", type=float, default=1e-6)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    cuts, _, _ = predictor._select("regression")
    pieces = []
    with ipc.open_file(args.catalogue) as reader:
        missing = set(NEED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            frame = pa.Table.from_batches([batch]).select(NEED).to_pandas()
            frame = frame.loc[frame.case.to_numpy(int) < args.n_cases]
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
            pairs = frame.loc[selected]
            if pairs.empty:
                continue
            prediction = predictor.predict_on_pairs(
                pairs[FEATURES], task="response",
            ).response.to_numpy(float)
            phase = np.deg2rad(2.0 * pairs.shear_angle.to_numpy(float))
            pieces.append(pd.DataFrame({
                "case": pairs.case.to_numpy(np.int16),
                "input_index": pairs.input_index.to_numpy(np.int64),
                "label": pairs.delta_et1.to_numpy(float) / args.shear,
                "null": pairs.delta_et2.to_numpy(float) / args.shear,
                "cos": np.cos(phase), "sin": np.sin(phase),
                "prediction": prediction,
            }))
            if len(pieces) % 100 == 0:
                print(f"scored {len(pieces)} selected batches", flush=True)
    pairs = pd.concat(pieces, ignore_index=True)
    development_pair = pairs.case.to_numpy(int) <= args.development_max
    if args.fixed_zero_split:
        if args.n_bins != 2:
            raise RuntimeError("--fixed-zero-split requires --n-bins 2")
        boundary = np.asarray([-np.inf, 0.0, np.inf])
    else:
        boundary = weighted_edges(
            pairs.loc[development_pair, "prediction"].to_numpy(float),
            pairs.loc[development_pair, "prediction"].to_numpy(float) ** 2,
            args.n_bins,
        )
    pair_bin = np.clip(
        np.searchsorted(boundary, pairs.prediction.to_numpy(float), side="right") - 1,
        0, args.n_bins - 1,
    )
    for index in range(args.n_bins):
        in_bin = pair_bin == index
        pairs[f"b{index}_cos"] = np.where(
            in_bin, pairs.prediction.to_numpy(float) * pairs.cos.to_numpy(float), 0.0,
        )
        pairs[f"b{index}_sin"] = np.where(
            in_bin, pairs.prediction.to_numpy(float) * pairs.sin.to_numpy(float), 0.0,
        )
    aggregations = {
        "label": ("label", "sum"), "null": ("null", "sum"),
        "cos": ("cos", "sum"), "sin": ("sin", "sum"),
        "n_pairs": ("label", "size"),
    }
    for index in range(args.n_bins):
        aggregations[f"b{index}_cos"] = (f"b{index}_cos", "sum")
        aggregations[f"b{index}_sin"] = (f"b{index}_sin", "sum")
    primary = pairs.groupby(["case", "input_index"], as_index=False).agg(**aggregations)
    direction_power = primary.cos**2 + primary.sin**2
    if (direction_power <= 1e-8).any():
        raise RuntimeError("numerically vanishing direction sum")
    primary["y1"] = (primary.label * primary.cos - primary.null * primary.sin) / direction_power
    primary["y2"] = (primary.label * primary.sin + primary.null * primary.cos) / direction_power
    cosine_columns = [f"b{i}_cos" for i in range(args.n_bins)]
    sine_columns = [f"b{i}_sin" for i in range(args.n_bins)]
    primary["raw_cos"] = primary[cosine_columns].sum(axis=1)
    primary["raw_sin"] = primary[sine_columns].sum(axis=1)

    development = primary.case.to_numpy(int) <= args.development_max
    train = primary.loc[development]
    x = np.vstack([train[cosine_columns].to_numpy(float), train[sine_columns].to_numpy(float)])
    y = np.concatenate([train.y1.to_numpy(float), train.y2.to_numpy(float)])
    count_by_case = train.groupby("case").size()
    row_weight = 1.0 / train.case.map(count_by_case).to_numpy(float)
    weight = np.concatenate([row_weight, row_weight])
    weight *= len(weight) / weight.sum()
    xw, yw = x * np.sqrt(weight[:, None]), y * np.sqrt(weight)
    gram, rhs = xw.T @ xw, xw.T @ yw
    ridge = float(args.ridge_fraction * np.trace(gram) / args.n_bins)
    coefficient = np.linalg.solve(
        gram + ridge * np.eye(args.n_bins), rhs + ridge * np.ones(args.n_bins),
    )
    primary["corrected_cos"] = primary[cosine_columns].to_numpy(float) @ coefficient
    primary["corrected_sin"] = primary[sine_columns].to_numpy(float) @ coefficient
    primary.to_feather(args.table_output)
    validation = ~development
    raw_dev = vector_fit(primary.loc[development], "raw")
    corrected_dev = vector_fit(primary.loc[development], "corrected")
    raw_val = vector_fit(primary.loc[validation], "raw")
    corrected_val = vector_fit(primary.loc[validation], "corrected")
    payload = {
        "candidate": (
            "two-bin sign-split rotation-equivariant pair-response calibration"
            if args.fixed_zero_split else
            f"{args.n_bins}-bin rotation-equivariant pair-response calibration"
        ),
        "response_bin_edges": [None if not np.isfinite(x) else float(x) for x in boundary],
        "coefficients": [float(x) for x in coefficient],
        "ridge_fraction": float(args.ridge_fraction), "ridge_absolute": ridge,
        "gram_condition_number": float(np.linalg.cond(gram + ridge * np.eye(args.n_bins))),
        "development_cases": [0, args.development_max],
        "validation_cases": [args.development_max + 1, args.n_cases - 1],
        "development": {"raw": raw_dev, "corrected": corrected_dev},
        "validation": {"raw": raw_val, "corrected": corrected_val},
        "validation_improves_abs_slope_error": bool(
            abs(corrected_val["slope_minus_one"]["mean"])
            < abs(raw_val["slope_minus_one"]["mean"])
        ),
        "n_pairs": int(len(pairs)), "n_primaries": int(len(primary)),
        "deployable_model_written": False,
        "constgold_opened": False, "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_PAIR_VECTOR_CALIBRATION_DONE", flush=True)


if __name__ == "__main__":
    main()
