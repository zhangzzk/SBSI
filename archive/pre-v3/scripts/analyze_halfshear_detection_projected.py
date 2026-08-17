"""Recheck detection instability in the cross-talk-corrected label coordinate."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


METRICS = [
    "centroid_shift_arcsec", "match_distance_change_pix",
    "flux_fractional_change", "isoarea_fractional_change", "abs_dmag_max",
]


def edges(values: np.ndarray, count: int) -> np.ndarray:
    boundary = np.unique(np.quantile(np.asarray(values, float), np.linspace(0, 1, count + 1)))
    if len(boundary) != count + 1:
        raise RuntimeError("quantile edges collapsed")
    boundary[0], boundary[-1] = -np.inf, np.inf
    return boundary


def assign(values: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(boundary, values, side="right") - 1, 0, len(boundary) - 2)


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    test = stats.ttest_1samp(values, 0.0)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)), "t": float(test.statistic),
        "p_two_sided": float(test.pvalue),
    }


def contrast(frame: pd.DataFrame, metric: str, threshold: pd.Series) -> tuple[np.ndarray, list[int]]:
    work = frame.copy()
    keys = pd.MultiIndex.from_frame(work[["property_cell", "n_pairs"]])
    work["threshold"] = threshold.reindex(keys).to_numpy(float)
    work = work[np.isfinite(work.threshold)].copy()
    work["high"] = work[metric] > work.threshold
    values, used = [], []
    for _, case in work.groupby("case", sort=True):
        differences, weights = [], []
        for _, cell in case.groupby(["property_cell", "n_pairs"], sort=False):
            high = cell.loc[cell.high, "projected_gap"].to_numpy(float)
            low = cell.loc[~cell.high, "projected_gap"].to_numpy(float)
            if len(high) < 3 or len(low) < 3:
                continue
            differences.append(float(high.mean() - low.mean()))
            weights.append(float(min(len(high), len(low))))
        if differences:
            values.append(float(np.average(differences, weights=weights)))
            used.append(len(differences))
    return np.asarray(values), used


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    result, running = {}, 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * pvalues[name]))
        result[name] = running
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    frame = pd.read_feather(args.input)
    development = frame[frame.case <= args.development_max].copy()
    validation = frame[frame.case > args.development_max].copy()
    boundaries = {
        "prediction": edges(development.prediction, 8),
        "r_input_p": edges(development.r_input_p, 4),
        "Re_input_p": edges(development.Re_input_p, 4),
    }
    for part in (development, validation):
        p = assign(part.prediction, boundaries["prediction"])
        m = assign(part.r_input_p, boundaries["r_input_p"])
        s = assign(part.Re_input_p, boundaries["Re_input_p"])
        part["property_cell"] = p * 16 + m * 4 + s
    endpoints, raw_p = {}, {}
    for metric in METRICS:
        threshold = development.groupby(
            ["property_cell", "n_pairs"], sort=True,
        )[metric].median()
        dev, dev_used = contrast(development, metric, threshold)
        val, val_used = contrast(validation, metric, threshold)
        endpoints[metric] = {
            "development": stat(dev), "validation": stat(val),
            "development_median_cells_per_case": float(np.median(dev_used)),
            "validation_median_cells_per_case": float(np.median(val_used)),
            "sign_definition": "high instability minus low instability projected-prediction-minus-label gap",
        }
        raw_p[metric] = endpoints[metric]["validation"]["p_two_sided"]
    adjusted = holm(raw_p)
    for metric, value in adjusted.items():
        endpoints[metric]["validation_p_holm"] = float(value)
    payload = {
        "design": "cross-talk-corrected projected label; c0--19 thresholds and c20--39 validation; 8 prediction x 4 mag x 4 size x exact pair count matching",
        "primary_endpoint": "centroid_shift_arcsec",
        "matched_endpoints": endpoints,
        "supersedes_naive_halfshear_detection_interpretation": True,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_DETECTION_PROJECTED_DONE", flush=True)


if __name__ == "__main__":
    main()
