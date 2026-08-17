"""Localize the V2.2 anchor gap against detection and matching stability.

This is a catalogue-only diagnostic on the already rendered coherent-neighbour
anchors.  It reproduces the exact per-leg bright-neighbour rejection and
crossmatch used by ``retrieve_constant_shear``, then attaches quantities absent
from the pair emulator: centroid motion, match distance, detection crowding,
bright-neighbour-cut margin, flags, and per-leg S/N.

Cases 200--249 define bin edges and stable-subset thresholds.  Cases 250--299
are reported with those choices frozen.  No correction is fitted and constgold
is never opened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import utils  # noqa: E402


KEY = ["case", "input_index"]
TILE = "tile180.0_-0.5"
METRICS = [
    "centroid_shift_arcsec",
    "centroid_offset_max_arcsec",
    "match_distance_max_pix",
    "match_distance_change_pix",
    "abs_dmag_max",
    "snr_min",
    "flux_fractional_change",
    "bright_ratio_max",
    "detection_neighbours_max",
    "isoarea_fractional_change",
]


def sem(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "case_sem": sem(values), "n_cases": int(len(values))}


def paths(base: str, case: int, sign: float, shape_suffix: str) -> tuple[str, str]:
    catalogue = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues",
    )
    shape = os.path.join(
        catalogue, "Shapes",
        f"shape_catalogue_detect_position{shape_suffix}_{TILE}.feather",
    )
    match = os.path.join(catalogue, "CrossMatch", f"{TILE}_rot0_matched.feather")
    return shape, match


def detection_context(shape: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Maximum neighbour flux ratio and detected-neighbour count within 3 arcsec."""
    position = shape[["X_WORLD", "Y_WORLD"]].to_numpy(float)
    first, second, _ = utils.kdt_neighbor_finder(
        position, position, r_min=0.0, r_max=3.0 / 3600.0, k=30,
    )
    flux = shape.FLUX_AUTO.to_numpy(float)
    ratio = flux[second] / flux[first]
    max_ratio = np.zeros(len(shape), dtype=float)
    np.maximum.at(max_ratio, first, ratio)
    count = np.zeros(len(shape), dtype=np.int16)
    np.add.at(count, first, 1)
    return max_ratio, count


def load_leg(base: str, case: int, sign: float, target: pd.DataFrame,
             shape_suffix: str) -> tuple[pd.DataFrame, dict]:
    shape_path, match_path = paths(base, case, sign, shape_suffix)
    shape = pd.read_feather(shape_path)
    match = pd.read_feather(match_path)
    if not isinstance(shape.index, pd.RangeIndex):
        raise RuntimeError(f"case {case} sign {sign}: shape index is not RangeIndex")
    max_ratio, neighbour_count = detection_context(shape)
    rejected = utils.remove_detection_w_bright_neighbour(
        shape.X_WORLD.array, shape.Y_WORLD.array, shape.FLUX_AUTO.array,
        ratio_max=5, r_min=0, r_max=3 / 3600,
    )
    keep = np.ones(len(shape), dtype=bool)
    keep[np.asarray(rejected, dtype=int)] = False
    kept_index = shape.index[keep]

    # Exact filtering and intersection order from blendemu.response._load_constant_match.
    _, match_positions, _ = np.intersect1d(
        match.id_detec.to_numpy(int) - 1, kept_index.to_numpy(int), return_indices=True,
    )
    filtered = match.iloc[match_positions].reset_index(drop=True)
    target_ids = target.input_index.to_numpy(np.int64)
    common, target_positions, _ = np.intersect1d(
        filtered.id_input.to_numpy(np.int64), target_ids, return_indices=True,
    )
    selected_match = filtered.iloc[target_positions].copy()
    detection_index = selected_match.id_detec.to_numpy(int) - 1
    selected_shape = shape.loc[detection_index]
    if len(common) != len(selected_shape):
        raise RuntimeError("matched target and shape lengths differ")

    ownership = filtered.id_detec.value_counts()
    duplicate_input = filtered.id_input.value_counts()
    truth = target.set_index("input_index", verify_integrity=True).loc[common]
    cosdec = np.cos(np.deg2rad(float(np.median(truth.DEC.to_numpy(float)))))
    dx = (selected_shape.X_WORLD.to_numpy(float) - truth.RA.to_numpy(float)) * cosdec * 3600.0
    dy = (selected_shape.Y_WORLD.to_numpy(float) - truth.DEC.to_numpy(float)) * 3600.0
    flux = selected_shape.FLUX_AUTO.to_numpy(float)
    fluxerr = selected_shape.FLUXERR_AUTO.to_numpy(float)
    out = pd.DataFrame({
        "input_index": common,
        "detection_index": detection_index,
        "x_world": selected_shape.X_WORLD.to_numpy(float),
        "y_world": selected_shape.Y_WORLD.to_numpy(float),
        "centroid_offset_arcsec": np.hypot(dx, dy),
        "match_distance_pix": selected_match.distance_pixel_CM.to_numpy(float),
        "dmag": selected_match.dmag_CM.to_numpy(float),
        "flux": flux,
        "snr": flux / fluxerr,
        "flags": selected_shape.FLAGS.to_numpy(int),
        "isoarea": selected_shape.ISOAREA_IMAGE.to_numpy(float),
        "flux_radius": selected_shape.FLUX_RADIUS.to_numpy(float),
        "bright_ratio": max_ratio[detection_index],
        "detection_neighbours": neighbour_count[detection_index],
        "detection_owner_count": selected_match.id_detec.map(ownership).to_numpy(int),
        "input_match_count": selected_match.id_input.map(duplicate_input).to_numpy(int),
    })
    summary = {
        "n_shapes": int(len(shape)),
        "n_rejected_bright_neighbour": int(len(rejected)),
        "n_target": int(len(target)),
        "n_target_retained": int(len(out)),
    }
    return out, summary


def pair_legs(plus: pd.DataFrame, minus: pd.DataFrame, case: int) -> pd.DataFrame:
    paired = plus.merge(
        minus, on="input_index", suffixes=("_plus", "_minus"),
        how="inner", validate="one_to_one",
    )
    dec0 = -0.5
    cosdec = np.cos(np.deg2rad(dec0))
    dx = (paired.x_world_plus.to_numpy(float) - paired.x_world_minus.to_numpy(float)) * cosdec * 3600
    dy = (paired.y_world_plus.to_numpy(float) - paired.y_world_minus.to_numpy(float)) * 3600
    def fractional_change(a: str) -> np.ndarray:
        left = paired[f"{a}_plus"].to_numpy(float)
        right = paired[f"{a}_minus"].to_numpy(float)
        return np.abs(left - right) / np.maximum(0.5 * (np.abs(left) + np.abs(right)), 1e-12)
    out = pd.DataFrame({
        "case": case,
        "input_index": paired.input_index.to_numpy(np.int64),
        "centroid_shift_arcsec": np.hypot(dx, dy),
        "centroid_offset_max_arcsec": np.maximum(
            paired.centroid_offset_arcsec_plus, paired.centroid_offset_arcsec_minus,
        ),
        "match_distance_max_pix": np.maximum(
            paired.match_distance_pix_plus, paired.match_distance_pix_minus,
        ),
        "match_distance_change_pix": np.abs(
            paired.match_distance_pix_plus - paired.match_distance_pix_minus,
        ),
        "abs_dmag_max": np.maximum(np.abs(paired.dmag_plus), np.abs(paired.dmag_minus)),
        "snr_min": np.minimum(paired.snr_plus, paired.snr_minus),
        "flux_fractional_change": fractional_change("flux"),
        "bright_ratio_max": np.maximum(paired.bright_ratio_plus, paired.bright_ratio_minus),
        "bright_margin_min": 5.0 - np.maximum(
            paired.bright_ratio_plus, paired.bright_ratio_minus,
        ),
        "detection_neighbours_max": np.maximum(
            paired.detection_neighbours_plus, paired.detection_neighbours_minus,
        ),
        "isoarea_fractional_change": fractional_change("isoarea"),
        "flux_radius_fractional_change": fractional_change("flux_radius"),
        "flags_or": paired.flags_plus.to_numpy(int) | paired.flags_minus.to_numpy(int),
        "detection_owner_max": np.maximum(
            paired.detection_owner_count_plus, paired.detection_owner_count_minus,
        ),
        "input_match_count_max": np.maximum(
            paired.input_match_count_plus, paired.input_match_count_minus,
        ),
    })
    return out


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * pvalues[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def case_slope(frame: pd.DataFrame, metric: str, center: float, scale: float) -> np.ndarray:
    values = []
    for _, case in frame.groupby("case", sort=True):
        x = np.clip((case[metric].to_numpy(float) - center) / scale, -5.0, 5.0)
        y = case.gap.to_numpy(float)
        variance = float(np.dot(x - x.mean(), x - x.mean()))
        if variance > 0:
            values.append(float(np.dot(x - x.mean(), y - y.mean()) / variance))
    return np.asarray(values, dtype=float)


def subset_summary(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    selected = frame.loc[np.asarray(mask, dtype=bool)]
    by_case = selected.groupby("case", sort=True)[
        ["R_blend_truth", "R_blend_lsst_r_extnbr_v22", "gap"]
    ].mean()
    return {
        "n_rows": int(len(selected)),
        "fraction": float(len(selected) / len(frame)),
        "n_cases": int(len(by_case)),
        "truth": stat(by_case.R_blend_truth.to_numpy(float)),
        "prediction": stat(by_case.R_blend_lsst_r_extnbr_v22.to_numpy(float)),
        "prediction_minus_truth": stat(by_case.gap.to_numpy(float)),
    }


def analyze(frame: pd.DataFrame, development_max: int) -> dict:
    development = frame[frame.case <= development_max].copy()
    validation = frame[frame.case > development_max].copy()
    if development.case.nunique() != 50 or validation.case.nunique() != 50:
        raise RuntimeError("expected 50 development and 50 validation cases")
    result = {"metrics": {}, "subsets": {}}
    raw_p = {}
    thresholds = {}
    for metric in METRICS:
        dev_values = development[metric].to_numpy(float)
        edges = np.quantile(dev_values, np.linspace(0.0, 1.0, 6))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        med = float(np.median(dev_values))
        iqr = float(np.subtract(*np.quantile(dev_values, [0.75, 0.25])))
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        thresholds[metric] = med
        slopes = case_slope(validation, metric, med, iqr)
        test = stats.ttest_1samp(slopes, popmean=0.0)
        raw_p[metric] = float(test.pvalue)
        bins = np.searchsorted(edges, validation[metric].to_numpy(float), side="right") - 1
        bins = np.clip(bins, 0, len(edges) - 2)
        rows = []
        for index in range(len(edges) - 1):
            subset = validation.loc[bins == index]
            by_case = subset.groupby("case", sort=True)["gap"].mean()
            rows.append({
                "bin": index,
                "edges": [
                    None if not np.isfinite(value) else float(value)
                    for value in edges[index:index + 2]
                ],
                "n_rows": int(len(subset)),
                "gap": stat(by_case.to_numpy(float)),
            })
        result["metrics"][metric] = {
            "development_median": med,
            "development_iqr": iqr,
            "validation_case_slope_per_iqr": stat(slopes),
            "validation_slope_t": float(test.statistic),
            "validation_slope_p_raw": float(test.pvalue),
            "validation_bins": rows,
        }
    adjusted = holm_adjust(raw_p)
    for metric, value in adjusted.items():
        result["metrics"][metric]["validation_slope_p_holm"] = value

    # Thresholds are fixed entirely from development distributions.
    for name, subset in (("development", development), ("validation", validation)):
        clean_unique = (
            (subset.flags_or.to_numpy(int) == 0)
            & (subset.detection_owner_max.to_numpy(int) == 1)
            & (subset.input_match_count_max.to_numpy(int) == 1)
        )
        stable_median = clean_unique.copy()
        for metric in (
            "centroid_shift_arcsec", "match_distance_max_pix",
            "flux_fractional_change", "bright_ratio_max",
        ):
            stable_median &= subset[metric].to_numpy(float) <= thresholds[metric]
        result["subsets"][name] = {
            "all": subset_summary(subset, np.ones(len(subset), dtype=bool)),
            "clean_unique": subset_summary(subset, clean_unique),
            "stable_development_medians": subset_summary(subset, stable_median),
        }
    result["stable_thresholds_from_development"] = {
        metric: thresholds[metric] for metric in (
            "centroid_shift_arcsec", "match_distance_max_pix",
            "flux_fractional_change", "bright_ratio_max",
        )
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--case-min", type=int, default=200)
    parser.add_argument("--case-max", type=int, default=299)
    parser.add_argument("--development-max", type=int, default=249)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--shape-suffix", default="_all")
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    response = pd.read_feather(args.response)
    response = response[response.case.between(args.case_min, args.case_max)].copy()
    required = {*KEY, "R_blend_truth", "R_blend_lsst_r_extnbr_v22"}
    if required - set(response):
        raise KeyError(f"response lacks {sorted(required-set(response))}")
    parts = []
    coverage = []
    for case, observed in response.groupby("case", sort=True):
        case = int(case)
        manifest = pd.read_feather(os.path.join(args.base, f"anchors_case{case}.feather"))
        target = manifest[["index", "RA", "DEC"]].rename(columns={"index": "input_index"})
        plus, plus_summary = load_leg(
            args.base, case, args.g, target, args.shape_suffix,
        )
        minus, minus_summary = load_leg(
            args.base, case, -args.g, target, args.shape_suffix,
        )
        paired = pair_legs(plus, minus, case)
        joined = observed.merge(paired, on=KEY, validate="one_to_one")
        if len(joined) != len(observed):
            raise RuntimeError(
                f"case {case}: detection audit covers {len(joined)}/{len(observed)} truth rows"
            )
        joined["gap"] = (
            joined.R_blend_lsst_r_extnbr_v22.to_numpy(float)
            - joined.R_blend_truth.to_numpy(float)
        )
        parts.append(joined)
        coverage.append({
            "case": case,
            "n_manifest": int(len(target)),
            "n_response": int(len(observed)),
            "n_both_leg_after_cut": int(len(paired)),
            "plus": plus_summary,
            "minus": minus_summary,
        })
        print(
            f"case {case}: response={len(observed):,} paired={len(paired):,} "
            f"centroid_shift={paired.centroid_shift_arcsec.median():.4f}arcsec "
            f"bright_ratio={paired.bright_ratio_max.median():.3f}", flush=True,
        )

    table = pd.concat(parts, ignore_index=True)
    if not np.isfinite(table[METRICS + ["gap"]].to_numpy(float)).all():
        raise RuntimeError("non-finite audit metric")
    table.to_feather(args.table_output)
    payload = {
        "design": (
            "catalogue-only detection/matching audit; cases 200--249 set thresholds, "
            "250--299 validate"
        ),
        "case_window": [args.case_min, args.case_max],
        "development_max": args.development_max,
        "n_rows": int(len(table)),
        "coverage": coverage,
        "analysis": analyze(table, args.development_max),
        "table_output": args.table_output,
        "correction_fitted": False,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload["analysis"], indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_DETECTION_BOUNDARY_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
