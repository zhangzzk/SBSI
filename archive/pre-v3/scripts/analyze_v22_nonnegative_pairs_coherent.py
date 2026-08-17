"""Test coherent V2.2 scene sums after clipping negative pair responses to zero."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.apply_v22_pair_residual_calibration_coherent import (
    discover_pair_paths,
    finite_stat,
    read_references,
)


def summarize_population(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "R_blend_truth", "prediction_raw", "prediction_nonnegative",
        "negative_pair_sum", "positive_pair_sum", "clipping_increment",
        "gap_truth_minus_raw", "gap_truth_minus_nonnegative",
        "n_pairs", "n_negative_pairs", "has_negative_pair",
    ]
    case = frame.groupby("case", sort=True)[columns].mean()
    raw_gap = finite_stat(case.gap_truth_minus_raw.to_numpy(float))
    clipped_gap = finite_stat(case.gap_truth_minus_nonnegative.to_numpy(float))
    return {
        "case_window": [int(frame.case.min()), int(frame.case.max())],
        "n_cases": int(frame.case.nunique()),
        "n_anchors": int(len(frame)),
        "truth": finite_stat(case.R_blend_truth.to_numpy(float)),
        "raw_prediction": finite_stat(case.prediction_raw.to_numpy(float)),
        "nonnegative_pair_prediction": finite_stat(
            case.prediction_nonnegative.to_numpy(float)
        ),
        "negative_pair_sum": finite_stat(case.negative_pair_sum.to_numpy(float)),
        "positive_pair_sum": finite_stat(case.positive_pair_sum.to_numpy(float)),
        "clipping_increment": finite_stat(case.clipping_increment.to_numpy(float)),
        "raw_gap_truth_minus_prediction": raw_gap,
        "nonnegative_gap_truth_minus_prediction": clipped_gap,
        "change_in_absolute_mean_gap": float(
            abs(clipped_gap["mean"]) - abs(raw_gap["mean"])
        ),
        "absolute_gap_ratio_to_raw": float(
            abs(clipped_gap["mean"]) / abs(raw_gap["mean"])
        ) if raw_gap["mean"] != 0.0 else None,
        "mean_pairs_per_anchor": finite_stat(case.n_pairs.to_numpy(float)),
        "mean_negative_pairs_per_anchor": finite_stat(
            case.n_negative_pairs.to_numpy(float)
        ),
        "fraction_anchors_with_negative_pair": finite_stat(
            case.has_negative_pair.to_numpy(float)
        ),
    }


def population_summaries(frame: pd.DataFrame) -> dict[str, Any]:
    windows = {
        "all": (400, 899),
        "c400_599": (400, 599),
        "c600_699": (600, 699),
        "c700_899": (700, 899),
    }
    for start in range(400, 900, 100):
        windows[f"c{start}_{start + 99}"] = (start, start + 99)
    return {
        name: summarize_population(frame.loc[frame.case.between(lo, hi)])
        for name, (lo, hi) in windows.items()
    }


def markdown_report(payload: dict[str, Any]) -> str:
    all_result = payload["populations"]["all"]
    final = payload["populations"]["c700_899"]
    raw = all_result["raw_gap_truth_minus_prediction"]
    clipped = all_result["nonnegative_gap_truth_minus_prediction"]
    raw_final = final["raw_gap_truth_minus_prediction"]
    clipped_final = final["nonnegative_gap_truth_minus_prediction"]
    return "\n".join([
        "# V2.2 coherent prediction with negative pair responses set to zero",
        "",
        "Every exact deployed pair with V2.2 response below zero is replaced by "
        "zero before the scene sum. Coherent anchor measurements are evaluation "
        "only; no threshold or scale is fitted.",
        "",
        f"- Cases: `{payload['case_window'][0]}--{payload['case_window'][1]}`; "
        f"anchors: `{all_result['n_anchors']:,}`; scored pairs: "
        f"`{payload['support']['n_scored_pairs']:,}`.",
        f"- Negative pairs: "
        f"`{100.0 * payload['support']['negative_pair_fraction']:.2f}%`; anchors "
        f"with at least one negative pair: "
        f"`{100.0 * all_result['fraction_anchors_with_negative_pair']['mean']:.2f}%`.",
        f"- Mean measured scene response: "
        f"`{all_result['truth']['mean']:+.6f} +- {all_result['truth']['case_sem']:.6f}`.",
        f"- Mean raw V2.2 prediction: "
        f"`{all_result['raw_prediction']['mean']:+.6f} +- "
        f"{all_result['raw_prediction']['case_sem']:.6f}`.",
        f"- Mean nonnegative-pair prediction: "
        f"`{all_result['nonnegative_pair_prediction']['mean']:+.6f} +- "
        f"{all_result['nonnegative_pair_prediction']['case_sem']:.6f}`.",
        f"- Gross negative/positive pair sums per anchor: "
        f"`{all_result['negative_pair_sum']['mean']:+.6f}` / "
        f"`{all_result['positive_pair_sum']['mean']:+.6f}`.",
        f"- All-case truth-minus-prediction gap: raw "
        f"`{raw['mean']:+.6f} +- {raw['case_sem']:.6f}`; clipped "
        f"`{clipped['mean']:+.6f} +- {clipped['case_sem']:.6f}`.",
        f"- c700--899 gap: raw `{raw_final['mean']:+.6f} +- "
        f"{raw_final['case_sem']:.6f}`; clipped "
        f"`{clipped_final['mean']:+.6f} +- {clipped_final['case_sem']:.6f}`.",
        "",
        "Uncertainties are one SEM across rendered cases. Positive "
        "truth-minus-prediction means underprediction; negative means overprediction.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    outputs = [
        f"{args.output_prefix}.{suffix}" for suffix in ("json", "md")
    ] + [f"{args.output_prefix}_cases.csv"]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    with open(args.source_json, encoding="utf-8") as handle:
        source = json.load(handle)
    case_min, case_max = map(int, source["case_window"])
    if (case_min, case_max) != (400, 899):
        raise RuntimeError("this audit expects the canonical c400--899 source")
    pair_paths, designs = discover_pair_paths(
        [item["path"] for item in source["pair_designs"]], case_min, case_max,
    )
    reference = read_references(
        source["coherent_reference_paths"], case_min, case_max,
    )

    parts = []
    n_scored_pairs = 0
    n_negative_pairs = 0
    maximum_replay_error = 0.0
    for progress, case in enumerate(range(case_min, case_max + 1), start=1):
        local = reference.loc[reference.case == case].copy().set_index("input_index")
        pairs = pd.read_feather(
            pair_paths[case], columns=["case", "anchor_index", "response"],
        )
        if len(pairs) and not pairs.case.eq(case).all():
            raise RuntimeError(f"case {case}: pair file contains another case")
        pairs = pairs.loc[pairs.anchor_index.isin(local.index)].copy()
        response = pairs.response.to_numpy(float)
        if not np.isfinite(response).all():
            raise RuntimeError(f"case {case}: non-finite pair response")
        n_scored_pairs += len(response)
        n_negative_pairs += int((response < 0.0).sum())
        pair_table = pd.DataFrame({
            "input_index": pairs.anchor_index.to_numpy(np.int64),
            "prediction_raw": response,
            "negative_pair_sum": np.minimum(response, 0.0),
            "positive_pair_sum": np.maximum(response, 0.0),
            "n_pairs": np.ones(len(response), dtype=np.int16),
            "n_negative_pairs": (response < 0.0).astype(np.int16),
        })
        aggregate = pair_table.groupby("input_index", sort=False).sum()
        local = local.join(aggregate, how="left")
        aggregate_columns = [
            "prediction_raw", "negative_pair_sum", "positive_pair_sum",
            "n_pairs", "n_negative_pairs",
        ]
        local[aggregate_columns] = local[aggregate_columns].fillna(0.0)
        replay_error = float(np.max(np.abs(
            local.prediction_raw.to_numpy(float)
            - local.R_blend_lsst_r_extnbr_v22.to_numpy(float)
        )))
        maximum_replay_error = max(maximum_replay_error, replay_error)
        tolerance = float(source["coherent_support"]["raw_scene_replay_tolerance"])
        if replay_error > tolerance:
            raise RuntimeError(f"case {case}: raw replay error {replay_error:.3e}")
        identity_error = float(np.max(np.abs(
            local.prediction_raw
            - local.negative_pair_sum
            - local.positive_pair_sum
        )))
        if identity_error > 1.0e-12:
            raise RuntimeError(f"case {case}: signed pair sums fail closure")
        local["prediction_nonnegative"] = local.positive_pair_sum
        local["clipping_increment"] = -local.negative_pair_sum
        local["gap_truth_minus_raw"] = (
            local.R_blend_truth - local.prediction_raw
        )
        local["gap_truth_minus_nonnegative"] = (
            local.R_blend_truth - local.prediction_nonnegative
        )
        local["has_negative_pair"] = (local.n_negative_pairs > 0.0).astype(float)
        parts.append(local.reset_index())
        if progress % 50 == 0 or progress == case_max - case_min + 1:
            print(f"processed {progress}/500 cases", flush=True)

    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != source["coherent_support"]["n_anchors"]:
        raise RuntimeError("anchor count does not reproduce source transfer")
    if n_scored_pairs != source["coherent_support"]["n_scored_pairs"]:
        raise RuntimeError("scored-pair count does not reproduce source transfer")
    populations = population_summaries(frame)
    case_columns = [
        "R_blend_truth", "prediction_raw", "prediction_nonnegative",
        "negative_pair_sum", "positive_pair_sum", "clipping_increment",
        "gap_truth_minus_raw", "gap_truth_minus_nonnegative", "n_pairs",
        "n_negative_pairs", "has_negative_pair",
    ]
    frame.groupby("case", sort=True)[case_columns].mean().reset_index().to_csv(
        f"{args.output_prefix}_cases.csv", index=False,
    )
    payload = {
        "title": "V2.2 coherent scene prediction with negative pair responses clipped to zero",
        "source_json": os.path.abspath(args.source_json),
        "case_window": [case_min, case_max],
        "operation": "replace each deployed V2.2 pair response p by max(p, 0) before summing per anchor",
        "threshold_fitted": False,
        "support": {
            "n_anchors": int(len(frame)),
            "n_scored_pairs": int(n_scored_pairs),
            "n_negative_pairs": int(n_negative_pairs),
            "negative_pair_fraction": float(n_negative_pairs / n_scored_pairs),
            "maximum_raw_scene_replay_error": float(maximum_replay_error),
            "raw_scene_replay_tolerance": tolerance,
        },
        "pair_designs": designs,
        "populations": populations,
        "uncertainty_unit": "rendered case",
        "constgold_opened": False,
    }
    with open(f"{args.output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    Path(f"{args.output_prefix}.md").write_text(
        markdown_report(payload), encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    print("V22_NONNEGATIVE_PAIRS_COHERENT_DONE", flush=True)


if __name__ == "__main__":
    main()
