"""Score V2.2 whole-case bootstrap retrains on clean coherent anchors.

The anchor cases 200--299 have no corresponding V2.2 training field.  The
stored V2.2 prediction is replayed exactly before bootstrap models are scored.
Uncertainty is summarized over rendered cases for anchor noise and over
whole-case bootstrap retrains for training-sample variance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import data_utils  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402


BASE_TAG = "lsst_r_extnbr_v22"
KEY = ["case", "input_index"]
TILE = "tile180.0_-0.5"
COND = dict(
    pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
    moffat_beta=2.224, pixel_rms=0.312,
)


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
        "n": int(len(values)),
    }


def input_frame(base: str, case: int, sign: float) -> pd.DataFrame:
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={column: column.replace("_input", "") for column in frame})


def load_models(model_dir: str, n_bootstrap: int) -> tuple[list[xgb.Booster], list[dict]]:
    boosters = []
    summaries = []
    for replicate in range(n_bootstrap + 1):
        summary_path = os.path.join(model_dir, f"summary_rep{replicate:02d}.json")
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary["replicate"] != replicate:
            raise RuntimeError(f"replicate mismatch in {summary_path}")
        booster = xgb.Booster()
        booster.load_model(summary["model_path"])
        boosters.append(booster)
        summaries.append(summary)
    return boosters, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=16)
    parser.add_argument("--case-min", type=int, default=200)
    parser.add_argument("--case-max", type=int, default=299)
    parser.add_argument("--sign", type=float, default=0.05)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    truth = pd.read_feather(args.response)
    truth = truth[truth.case.between(args.case_min, args.case_max)].copy()
    required = {*KEY, "R_blend_truth", f"R_blend_{BASE_TAG}"}
    missing = required - set(truth)
    if missing:
        raise KeyError(f"anchor response lacks {sorted(missing)}")
    if truth.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor keys")
    expected_cases = set(range(args.case_min, args.case_max + 1))
    if set(truth.case.unique()) != expected_cases:
        raise RuntimeError("anchor case window is incomplete")

    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=BASE_TAG, conditions=COND, device="cpu",
    )
    boosters, summaries = load_models(args.model_dir, args.n_bootstrap)
    features = summaries[0]["features"]
    if any(summary["features"] != features for summary in summaries):
        raise RuntimeError("bootstrap feature lists differ")

    parts = []
    replay_max = 0.0
    for case, observed in truth.groupby("case", sort=True):
        case = int(case)
        frame = input_frame(args.base, case, args.sign)
        pairs = predictor.predict_response(frame, frame)
        primary = next(
            column for column in pairs if column.startswith("index") and column.endswith("_p")
        )
        anchors = pd.Index(observed.input_index.to_numpy(np.int64), name="input_index")
        group = pairs.groupby(primary, sort=False)
        aggregate = pd.DataFrame(index=anchors)
        baseline = group.response.sum().reindex(anchors, fill_value=0.0).to_numpy(float)
        stored = observed.set_index("input_index").loc[
            anchors, f"R_blend_{BASE_TAG}"
        ].to_numpy(float)
        error = float(np.max(np.abs(baseline - stored)))
        replay_max = max(replay_max, error)
        if not np.array_equal(baseline, stored):
            raise RuntimeError(f"case {case}: baseline replay differs, max abs={error:.3e}")
        aggregate[f"R_blend_{BASE_TAG}"] = baseline
        matrix = xgb.DMatrix(pairs[features])
        for replicate, (booster, summary) in enumerate(zip(boosters, summaries)):
            standardized = booster.predict(
                matrix, iteration_range=data_utils.get_xgb_iteration_range(booster),
            )
            physical = data_utils.reverse_standardize(
                standardized, summary["response_mean"], summary["response_std"],
            )
            pair_values = pd.Series(physical, index=pairs.index)
            summed = pair_values.groupby(pairs[primary], sort=False).sum()
            aggregate[f"R_rep{replicate:02d}"] = summed.reindex(
                anchors, fill_value=0.0,
            ).to_numpy(float)
        # The stored table already carries the accepted baseline prediction.
        # Its exact replay was asserted above; do not merge a duplicate-named
        # copy and let pandas silently suffix the provenance control.
        aggregate = aggregate.drop(columns=f"R_blend_{BASE_TAG}").reset_index()
        joined = observed.merge(aggregate, on="input_index", validate="one_to_one")
        parts.append(joined)
        shifts = [
            joined[f"R_rep{replicate:02d}"].mean() - joined[f"R_blend_{BASE_TAG}"].mean()
            for replicate in range(args.n_bootstrap + 1)
        ]
        print(
            f"case {case}: anchors={len(joined):,} pairs={len(pairs):,} "
            f"replay={error:.1e} nominal_shift={shifts[0]:+.6f} "
            f"bootstrap_shift_sd={np.std(shifts[1:], ddof=1):.6f}", flush=True,
        )

    table = pd.concat(parts, ignore_index=True)
    table.to_feather(args.table_output)
    case_columns = ["R_blend_truth", f"R_blend_{BASE_TAG}"] + [
        f"R_rep{replicate:02d}" for replicate in range(args.n_bootstrap + 1)
    ]
    by_case = table.groupby("case", sort=True)[case_columns].mean()
    truth_case = by_case.R_blend_truth.to_numpy(float)
    baseline_case = by_case[f"R_blend_{BASE_TAG}"].to_numpy(float)
    baseline_gap = baseline_case - truth_case
    replicate_predictions = np.asarray([
        by_case[f"R_rep{replicate:02d}"].to_numpy(float).mean()
        for replicate in range(args.n_bootstrap + 1)
    ])
    replicate_gaps = np.asarray([
        (by_case[f"R_rep{replicate:02d}"].to_numpy(float) - truth_case).mean()
        for replicate in range(args.n_bootstrap + 1)
    ])
    bootstrap_prediction = replicate_predictions[1:]
    training_se = float(bootstrap_prediction.std(ddof=1))
    anchor_sem = float(baseline_gap.std(ddof=1) / np.sqrt(len(baseline_gap)))

    edges = np.quantile(table[f"R_blend_{BASE_TAG}"].to_numpy(float), np.linspace(0, 1, 6))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_id = np.searchsorted(edges, table[f"R_blend_{BASE_TAG}"], side="right") - 1
    bin_id = np.clip(bin_id, 0, 4)
    table_for_bins = table.assign(prediction_bin=bin_id)
    conditional = {}
    for index in range(5):
        subset = table_for_bins[table_for_bins.prediction_bin == index]
        case = subset.groupby("case", sort=True)[case_columns].mean()
        gaps = np.asarray([
            (case[f"R_rep{replicate:02d}"].to_numpy(float)
             - case.R_blend_truth.to_numpy(float)).mean()
            for replicate in range(args.n_bootstrap + 1)
        ])
        raw = (
            case[f"R_blend_{BASE_TAG}"].to_numpy(float)
            - case.R_blend_truth.to_numpy(float)
        )
        conditional[str(index)] = {
            "edges": [
                None if not np.isfinite(value) else float(value)
                for value in edges[index:index + 2]
            ],
            "n_rows": int(len(subset)),
            "n_cases": int(len(case)),
            "baseline_gap": stat(raw),
            "nominal_reproduction_gap": float(gaps[0]),
            "bootstrap_gap_mean": float(gaps[1:].mean()),
            "training_case_bootstrap_se": float(gaps[1:].std(ddof=1)),
        }

    payload = {
        "design": "whole-case bootstrap of V2.2 cases 40--199; score clean anchors 200--299",
        "n_bootstrap": args.n_bootstrap,
        "n_anchor_cases": int(len(by_case)),
        "n_anchor_rows": int(len(table)),
        "baseline_replay_max_abs": replay_max,
        "baseline_prediction": stat(baseline_case),
        "anchor_truth": stat(truth_case),
        "baseline_prediction_minus_truth": stat(baseline_gap),
        "nominal_reproduction": {
            "prediction_mean": float(replicate_predictions[0]),
            "gap": float(replicate_gaps[0]),
            "prediction_shift_from_stored_v22": float(
                replicate_predictions[0] - baseline_case.mean()
            ),
        },
        "training_case_bootstrap": {
            "prediction_mean": float(bootstrap_prediction.mean()),
            "prediction_se": training_se,
            "prediction_percentile_95": [
                float(value) for value in np.quantile(bootstrap_prediction, [0.025, 0.975])
            ],
            "gap_mean": float(replicate_gaps[1:].mean()),
            "gap_se": float(replicate_gaps[1:].std(ddof=1)),
            "gap_range": [float(replicate_gaps[1:].min()), float(replicate_gaps[1:].max())],
            "fraction_gap_negative": float(np.mean(replicate_gaps[1:] < 0.0)),
        },
        "combined_anchor_and_training_se_quadrature": float(np.hypot(anchor_sem, training_se)),
        "conditional_stored_prediction_quintiles": conditional,
        "training_summaries": summaries,
        "table_output": args.table_output,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_V22_CASE_BOOTSTRAP_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
