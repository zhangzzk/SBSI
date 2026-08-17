"""Test whether centroid-motion structure survives blend/property matching.

Development cases 200--249 define a 10 x 4 x 4 grid in frozen V2.2 response,
primary magnitude and primary size, plus the within-cell median centroid shift.
Validation cases 250--299 then compare high- versus low-shift anchor residuals
inside the same cells.  Case-level matched contrasts are the uncertainty units.
This is mechanistic diagnosis, not a fitted response correction.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


CONF = ["R_blend_lsst_r_extnbr_v22", "r_input_p_plus", "Re_input_p_plus"]


def edges(values: np.ndarray, n: int) -> np.ndarray:
    out = np.unique(np.quantile(np.asarray(values, float), np.linspace(0, 1, n + 1)))
    if len(out) != n + 1:
        raise RuntimeError(f"quantile edges collapsed: wanted {n+1}, got {len(out)}")
    out[0] = -np.inf
    out[-1] = np.inf
    return out


def assign(values: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(boundary, values, side="right") - 1, 0, len(boundary) - 2)


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    test = stats.ttest_1samp(values, 0.0)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
        "t": float(test.statistic),
        "p_two_sided": float(test.pvalue),
    }


def matched_case_contrast(frame: pd.DataFrame, metric: str,
                          threshold: pd.Series) -> tuple[np.ndarray, dict]:
    work = frame.copy()
    work["threshold"] = work.cell.map(threshold)
    work = work[np.isfinite(work.threshold)]
    work["high"] = work[metric].to_numpy(float) > work.threshold.to_numpy(float)
    contrasts = []
    cells_used = []
    for case_id, case in work.groupby("case", sort=True):
        differences = []
        weights = []
        used = 0
        for _, cell in case.groupby("cell", sort=False):
            low = cell.loc[~cell.high, "gap"].to_numpy(float)
            high = cell.loc[cell.high, "gap"].to_numpy(float)
            if len(low) < 3 or len(high) < 3:
                continue
            differences.append(float(high.mean() - low.mean()))
            weights.append(float(min(len(low), len(high))))
            used += 1
        if used:
            contrasts.append(float(np.average(differences, weights=weights)))
            cells_used.append(used)
    return np.asarray(contrasts), {
        "median_cells_per_case": float(np.median(cells_used)),
        "min_cells_per_case": int(np.min(cells_used)),
        "max_cells_per_case": int(np.max(cells_used)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--development-max", type=int, default=249)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    frame = pd.read_feather(args.input)
    development = frame[frame.case <= args.development_max].copy()
    validation = frame[frame.case > args.development_max].copy()
    boundaries = {
        CONF[0]: edges(development[CONF[0]], 10),
        CONF[1]: edges(development[CONF[1]], 4),
        CONF[2]: edges(development[CONF[2]], 4),
    }
    for part in (development, validation):
        indices = [assign(part[name].to_numpy(float), boundaries[name]) for name in CONF]
        part["cell"] = indices[0] * 16 + indices[1] * 4 + indices[2]

    metrics = [
        "centroid_shift_arcsec", "match_distance_change_pix",
        "flux_fractional_change", "isoarea_fractional_change", "abs_dmag_max",
    ]
    results = {}
    for metric in metrics:
        threshold = development.groupby("cell", sort=True)[metric].median()
        dev_contrast, dev_cells = matched_case_contrast(development, metric, threshold)
        val_contrast, val_cells = matched_case_contrast(validation, metric, threshold)
        results[metric] = {
            "development": stat(dev_contrast),
            "development_cells": dev_cells,
            "validation": stat(val_contrast),
            "validation_cells": val_cells,
            "sign_definition": "high metric minus low metric V2.2-minus-truth gap",
        }

    # A transparent 2D view of the primary metric within V2.2 prediction quintiles.
    pred_edges = edges(development[CONF[0]], 5)
    validation["prediction_quintile"] = assign(validation[CONF[0]], pred_edges)
    centroid_threshold = development.assign(
        prediction_quintile=assign(development[CONF[0]], pred_edges)
    ).groupby("prediction_quintile")["centroid_shift_arcsec"].median()
    view = {}
    for index, subset in validation.groupby("prediction_quintile", sort=True):
        high = subset.centroid_shift_arcsec > centroid_threshold.loc[index]
        rows = {}
        for label, mask in (("low_shift", ~high), ("high_shift", high)):
            selected = subset.loc[mask]
            case = selected.groupby("case", sort=True)[
                ["gap", "R_blend_truth", "R_blend_lsst_r_extnbr_v22"]
            ].mean()
            rows[label] = {
                "n_rows": int(len(selected)),
                "gap_mean": float(case.gap.mean()),
                "gap_case_sem": float(case.gap.std(ddof=1) / np.sqrt(len(case))),
                "truth_mean": float(case.R_blend_truth.mean()),
                "prediction_mean": float(case.R_blend_lsst_r_extnbr_v22.mean()),
            }
        view[str(int(index))] = rows

    payload = {
        "design": (
            "within-cell high-minus-low detection-instability contrast; cells are frozen "
            "V2.2 prediction decile x primary-mag quartile x primary-size quartile"
        ),
        "development_cases": [int(development.case.min()), args.development_max],
        "validation_cases": [args.development_max + 1, int(validation.case.max())],
        "n_development_rows": int(len(development)),
        "n_validation_rows": int(len(validation)),
        "confounder_edges": {
            name: [None if not np.isfinite(value) else float(value) for value in boundary]
            for name, boundary in boundaries.items()
        },
        "matched_contrasts": results,
        "validation_centroid_shift_by_prediction_quintile": view,
        "correction_fitted": False,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_DETECTION_CONDITIONAL_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
