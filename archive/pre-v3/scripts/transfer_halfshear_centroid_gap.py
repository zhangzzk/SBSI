"""Transfer a frozen half-shear centroid-residual table to coherent anchors.

Half-shear cases 0--19 define 8 prediction x 4 magnitude x 4 size cells and,
inside each cell, four centroid-motion-rate bins.  Their case-balanced mean
V2.2-minus-label residual is frozen.  It is evaluated on held-out half-shear
cases 20--39 and on coherent anchors 200--299.  A model without the centroid
axis is carried in parallel, so only the incremental composition effect is
attributed to centroid motion.

Centroid motion is post-shear and therefore this is a mechanism/transport
diagnostic, not a deployable calibration.  No constgold data are read.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


BASE_FEATURES = ["prediction", "magnitude", "size"]


def edges(values: np.ndarray, count: int) -> np.ndarray:
    boundary = np.unique(np.quantile(np.asarray(values, float), np.linspace(0, 1, count + 1)))
    if len(boundary) != count + 1:
        raise RuntimeError(f"collapsed quantile edges: wanted {count + 1}, got {len(boundary)}")
    boundary[0], boundary[-1] = -np.inf, np.inf
    return boundary


def assign(values: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(boundary, values, side="right") - 1, 0, len(boundary) - 2)


def prepare(path: str, kind: str, halfshear_target: str) -> pd.DataFrame:
    frame = pd.read_feather(path)
    if kind == "halfshear":
        frame = frame.rename(columns={
            "r_input_p": "magnitude", "Re_input_p": "size",
        })
        frame["centroid_rate"] = frame.centroid_shift_arcsec / 0.2
        frame["truth"] = frame.label
        if halfshear_target not in frame:
            raise KeyError(f"half-shear table lacks target {halfshear_target!r}")
        frame["transport_gap"] = frame[halfshear_target].to_numpy(float)
    else:
        frame = frame.rename(columns={
            "r_input_p_plus": "magnitude", "Re_input_p_plus": "size",
            "R_blend_lsst_r_extnbr_v22": "prediction",
        })
        frame["centroid_rate"] = frame.centroid_shift_arcsec / 0.1
        frame["truth"] = frame.R_blend_truth
        frame["transport_gap"] = frame.gap.to_numpy(float)
    required = {
        "case", "prediction", "magnitude", "size", "centroid_rate",
        "gap", "transport_gap", "truth",
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"{kind} table lacks {sorted(missing)}")
    return frame


def map_base_cells(frame: pd.DataFrame, boundaries: dict[str, np.ndarray]) -> np.ndarray:
    indices = [assign(frame[name].to_numpy(float), boundaries[name]) for name in BASE_FEATURES]
    return indices[0] * 16 + indices[1] * 4 + indices[2]


def case_balanced_lookup(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    per_case = frame.groupby(["case", *keys], sort=True).transport_gap.mean()
    return per_case.groupby(level=keys).mean()


def summarize(frame: pd.DataFrame) -> dict:
    case = frame.groupby("case", sort=True)[
        ["transport_gap", "predicted_base_gap", "predicted_centroid_gap"]
    ].mean()
    base_error = case.predicted_base_gap - case.transport_gap
    centroid_error = case.predicted_centroid_gap - case.transport_gap
    correction = case.predicted_centroid_gap - case.predicted_base_gap
    def value(x: pd.Series | np.ndarray) -> dict:
        x = np.asarray(x, float)
        return {
            "mean": float(x.mean()),
            "case_sem": float(x.std(ddof=1) / np.sqrt(len(x))),
        }
    return {
        "n_rows": int(len(frame)), "n_cases": int(len(case)),
        "actual_gap": value(case.transport_gap),
        "base_predicted_gap": value(case.predicted_base_gap),
        "centroid_predicted_gap": value(case.predicted_centroid_gap),
        "incremental_centroid_composition": value(correction),
        "base_prediction_error": value(base_error),
        "centroid_prediction_error": value(centroid_error),
        "base_case_rmse": float(np.sqrt(np.mean(np.square(base_error)))),
        "centroid_case_rmse": float(np.sqrt(np.mean(np.square(centroid_error)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halfshear", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--halfshear-target", default="gap")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    half = prepare(args.halfshear, "halfshear", args.halfshear_target)
    anchor = prepare(args.anchor, "anchor", args.halfshear_target)
    development = half[half.case <= args.development_max].copy()
    validation = half[half.case > args.development_max].copy()
    boundaries = {
        "prediction": edges(development.prediction, 8),
        "magnitude": edges(development.magnitude, 4),
        "size": edges(development["size"], 4),
    }
    for frame in (development, validation, anchor):
        frame["base_cell"] = map_base_cells(frame, boundaries)

    rate_edges: dict[int, np.ndarray] = {
        int(cell): edges(group.centroid_rate, 4)
        for cell, group in development.groupby("base_cell", sort=True)
    }
    if len(rate_edges) != 8 * 4 * 4:
        raise RuntimeError(f"development lacks base cells: found {len(rate_edges)}/128")
    for frame in (development, validation, anchor):
        quartile = np.full(len(frame), -1, dtype=np.int8)
        for cell, positions in frame.groupby("base_cell", sort=False).indices.items():
            if int(cell) in rate_edges:
                quartile[np.asarray(positions, int)] = assign(
                    frame.iloc[np.asarray(positions, int)].centroid_rate, rate_edges[int(cell)],
                )
        frame["centroid_quartile"] = quartile

    base_lookup = case_balanced_lookup(development, ["base_cell"])
    centroid_lookup = case_balanced_lookup(development, ["base_cell", "centroid_quartile"])
    for name, frame in (("development", development), ("validation", validation), ("anchor", anchor)):
        frame["predicted_base_gap"] = frame.base_cell.map(base_lookup)
        keys = pd.MultiIndex.from_frame(frame[["base_cell", "centroid_quartile"]])
        frame["predicted_centroid_gap"] = centroid_lookup.reindex(keys).to_numpy(float)
        if frame[["predicted_base_gap", "predicted_centroid_gap"]].isna().any().any():
            raise RuntimeError(f"{name} has uncovered lookup cells")

    composition = {}
    for name, frame in (("development", development), ("validation", validation), ("anchor", anchor)):
        counts = frame.groupby(["case", "centroid_quartile"], sort=True).size().unstack(fill_value=0)
        fractions = counts.div(counts.sum(axis=1), axis=0)
        composition[name] = {
            str(int(column)): {
                "mean_fraction": float(fractions[column].mean()),
                "case_sem": float(fractions[column].std(ddof=1) / np.sqrt(len(fractions))),
            }
            for column in fractions
        }

    payload = {
        "design": "case-balanced c0--19 half-shear lookup; 8 prediction x 4 mag x 4 size x 4 centroid-rate bins",
        "halfshear_training_target": args.halfshear_target,
        "centroid_rate_definition": {
            "halfshear": "centroid shift arcsec / 0.2 shear step",
            "anchor": "centroid shift arcsec / 0.1 (+/-0.05) shear separation",
        },
        "development": summarize(development),
        "validation": summarize(validation),
        "anchor": summarize(anchor),
        "centroid_quartile_composition": composition,
        "base_boundaries": {
            name: [None if not np.isfinite(value) else float(value) for value in boundary]
            for name, boundary in boundaries.items()
        },
        "post_shear_diagnostic_only": True,
        "correction_proposed_for_deployment": False,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_CENTROID_GAP_TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
