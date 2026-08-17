"""Localize held-out half-shear vector miscalibration on frozen dev bins."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from analyze_halfshear_vector_closure import fit


def edges(values: np.ndarray, count: int) -> np.ndarray:
    result = np.unique(np.quantile(np.asarray(values, float), np.linspace(0, 1, count + 1)))
    if len(result) != count + 1:
        raise RuntimeError("quantile edges collapsed")
    result[0], result[-1] = -np.inf, np.inf
    return result


def weighted_edges(values: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(values)
    x = np.asarray(values, float)[order]
    w = np.asarray(weights, float)[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    result = np.unique(np.interp(np.linspace(0, 1, count + 1), cumulative, x))
    if len(result) != count + 1:
        raise RuntimeError("weighted quantile edges collapsed")
    result[0], result[-1] = -np.inf, np.inf
    return result


def axis_summary(development: pd.DataFrame, validation: pd.DataFrame,
                 column: str, boundary: np.ndarray) -> dict:
    result = {}
    for name, frame in (("development", development), ("validation", validation)):
        index = np.clip(
            np.searchsorted(boundary, frame[column].to_numpy(float), side="right") - 1,
            0, len(boundary) - 2,
        )
        bins = {}
        for bin_index in range(len(boundary) - 1):
            subset = frame.loc[index == bin_index]
            bins[str(bin_index)] = {
                "lo": None if not np.isfinite(boundary[bin_index]) else float(boundary[bin_index]),
                "hi": None if not np.isfinite(boundary[bin_index + 1]) else float(boundary[bin_index + 1]),
                "n_rows": int(len(subset)),
                "model_power_fraction": float(subset.power.sum() / frame.power.sum()),
                "fit": fit(subset),
            }
        result[name] = bins
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    columns = [
        "case", "input_index", "label", "null", "prediction", "n_pairs",
        "cos_shear", "sin_shear", "prediction_cos", "prediction_sin",
    ]
    table = pd.read_feather(args.table, columns=columns)
    direction_power = table.cos_shear**2 + table.sin_shear**2
    table["y1"] = (
        table.label * table.cos_shear - table.null * table.sin_shear
    ) / direction_power
    table["y2"] = (
        table.label * table.sin_shear + table.null * table.cos_shear
    ) / direction_power
    table["power"] = table.prediction_cos**2 + table.prediction_sin**2
    table["model_vector_magnitude"] = np.sqrt(table.power)
    table["dot"] = table.prediction_cos * table.y1 + table.prediction_sin * table.y2
    table["cross"] = -table.prediction_sin * table.y1 + table.prediction_cos * table.y2
    development = table.loc[table.case <= args.development_max].copy()
    validation = table.loc[table.case > args.development_max].copy()
    axes = {}
    boundaries = {}
    specifications = {
        "prediction": ("prediction", edges(development.prediction, args.n_bins)),
        "model_vector_magnitude": (
            "model_vector_magnitude", edges(development.model_vector_magnitude, args.n_bins),
        ),
        "model_vector_magnitude_equal_power": (
            "model_vector_magnitude",
            weighted_edges(
                development.model_vector_magnitude.to_numpy(float),
                development.power.to_numpy(float), args.n_bins,
            ),
        ),
        "n_pairs": (
            "n_pairs", np.asarray([-np.inf, 4.5, 7.5, 10.5, 14.5, np.inf]),
        ),
    }
    for name, (column, boundary) in specifications.items():
        boundaries[name] = [None if not np.isfinite(x) else float(x) for x in boundary]
        axes[name] = axis_summary(development, validation, column, boundary)
    payload = {
        "design": "five equal-occupancy bins frozen on c0--19; vector slopes evaluated separately on c20--39",
        "boundaries": boundaries,
        "axes": axes,
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_VECTOR_STRATIFICATION_DONE", flush=True)


if __name__ == "__main__":
    main()
