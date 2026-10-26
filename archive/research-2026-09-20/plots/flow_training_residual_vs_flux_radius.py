#!/usr/bin/env python3
"""Plot staged-flow self-response residual versus fixed-g0 FLUX_RADIUS.

The flow prediction is evaluated from truth/context only.  Measured g=0
``FLUX_RADIUS`` enters only after prediction, as the horizontal diagnostic
coordinate.  The plotted residual is

    measured R_self - flow R_self.

All summaries are equal-weight means over the cases listed in the prediction
manifest, and error bars are one standard error over paired case residuals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.diagnose_flow_self_response_bias import load_leg, measured_case
from scripts.diagnose_flow_self_response_by_radius import add_zero_radius


PIXEL_SCALE_ARCSEC = 0.2
SELECTION_BOUNDARY_ARCSEC = 3.0 * PIXEL_SCALE_ARCSEC
REFERENCE_SLICE_EDGE_ARCSEC = 0.6154


def radius_edges_pixels() -> np.ndarray:
    """Fine boundary bins followed by progressively wider tail bins."""

    fine = np.arange(3.0, 3.5 + 0.5 * 0.025, 0.025)
    middle = np.arange(3.55, 4.5 + 0.5 * 0.05, 0.05)
    upper = np.arange(4.6, 6.0 + 0.5 * 0.1, 0.1)
    tail = np.array([6.25, 6.5, 7.0, 8.0, 10.0, 15.0, np.inf])
    edges = np.concatenate((fine, middle, upper, tail))
    if edges[0] != 3.0 or not np.all(np.diff(edges) > 0):
        raise RuntimeError("radius bin construction is invalid")
    return edges


def case_mean_and_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column means/SEMs over finite case entries, plus case counts."""

    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    means = np.full(values.shape[1], np.nan, dtype=np.float64)
    sems = np.full(values.shape[1], np.nan, dtype=np.float64)
    for index in range(values.shape[1]):
        column = values[valid[:, index], index]
        if len(column):
            means[index] = column.mean()
        if len(column) > 1:
            sems[index] = column.std(ddof=1) / math.sqrt(len(column))
    return means, sems, counts


def build_profile(
    *,
    domain_root: Path,
    predictions_path: Path,
    prediction_manifest_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Accumulate radius-bin response moments without retaining all rows."""

    domain_manifest = json.loads((domain_root / "manifest.json").read_text())
    prediction_manifest = json.loads(prediction_manifest_path.read_text())
    train_cases = set(map(int, domain_manifest["split"]["train_cases"]))
    declared_cases = tuple(map(int, prediction_manifest["cases"]))
    if not declared_cases or not set(declared_cases).issubset(train_cases):
        raise ValueError("the prediction manifest is not confined to training cases")

    predictions = pd.read_feather(predictions_path)
    required = {"case", "input_index", "R_self_model"}
    if required - set(predictions.columns):
        raise ValueError(f"prediction table is missing {sorted(required - set(predictions.columns))}")
    if predictions.duplicated(["case", "input_index"]).any():
        raise ValueError("prediction table contains duplicate identities")
    table_cases = tuple(sorted(map(int, predictions["case"].unique())))
    if table_cases != tuple(sorted(declared_cases)):
        raise ValueError("prediction table cases differ from its manifest")
    if not np.isfinite(predictions["R_self_model"].to_numpy(np.float64)).all():
        raise ValueError("prediction table contains non-finite model responses")

    edges_px = radius_edges_pixels()
    bins = len(edges_px) - 1
    case_measured = np.full((len(declared_cases), bins), np.nan, dtype=np.float64)
    case_model = np.full((len(declared_cases), bins), np.nan, dtype=np.float64)
    object_counts = np.zeros(bins, dtype=np.int64)
    radius_sums = np.zeros(bins, dtype=np.float64)
    accounting: dict[str, dict[str, int]] = {}

    grouped_predictions = {
        int(case): group.drop(columns="case")
        for case, group in predictions.groupby("case", sort=False)
    }
    for row, case in enumerate(declared_cases):
        zero = load_leg(domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        sheared = load_leg(domain_root / "flow" / "g05" / f"case{case:03d}.npz")
        measured, match_counts = measured_case(zero, sheared)
        measured = add_zero_radius(measured, zero)
        predicted = grouped_predictions[case]
        joined = measured.merge(
            predicted,
            on="input_index",
            how="inner",
            validate="one_to_one",
        )
        accounting[str(case)] = {
            **{str(key): int(value) for key, value in match_counts.items()},
            "measured_rows": int(len(measured)),
            "prediction_rows": int(len(predicted)),
            "joined_rows": int(len(joined)),
            "measured_unmatched": int(len(measured) - len(joined)),
            "prediction_unmatched": int(len(predicted) - len(joined)),
        }
        if len(joined) != len(measured) or len(joined) != len(predicted):
            raise RuntimeError(f"measured/predicted identity mismatch in case {case}")

        radius_px = joined["g0_flux_radius_arcsec"].to_numpy(np.float64) / PIXEL_SCALE_ARCSEC
        if np.any(radius_px <= 3.0) or not np.isfinite(radius_px).all():
            raise RuntimeError(f"case {case} violates strict FLUX_RADIUS>3 px")
        assignment = np.searchsorted(edges_px, radius_px, side="right") - 1
        if np.any((assignment < 0) | (assignment >= bins)):
            raise RuntimeError(f"case {case} lies outside radius-bin support")

        measured_values = joined["R_self_measured"].to_numpy(np.float64)
        model_values = joined["R_self_model"].to_numpy(np.float64)
        counts = np.bincount(assignment, minlength=bins)
        measured_sums = np.bincount(assignment, weights=measured_values, minlength=bins)
        model_sums = np.bincount(assignment, weights=model_values, minlength=bins)
        radius_bin_sums = np.bincount(
            assignment,
            weights=radius_px * PIXEL_SCALE_ARCSEC,
            minlength=bins,
        )
        populated = counts > 0
        case_measured[row, populated] = measured_sums[populated] / counts[populated]
        case_model[row, populated] = model_sums[populated] / counts[populated]
        object_counts += counts
        radius_sums += radius_bin_sums

    measured_mean, measured_sem, case_counts = case_mean_and_sem(case_measured)
    model_mean, model_sem, model_case_counts = case_mean_and_sem(case_model)
    residual_mean, residual_sem, residual_case_counts = case_mean_and_sem(
        case_measured - case_model
    )
    if not np.array_equal(case_counts, model_case_counts) or not np.array_equal(
        case_counts, residual_case_counts
    ):
        raise RuntimeError("case coverage differs among response summaries")

    lower_px, upper_px = edges_px[:-1], edges_px[1:]
    mean_radius = np.divide(
        radius_sums,
        object_counts,
        out=np.full(bins, np.nan),
        where=object_counts > 0,
    )
    frame = pd.DataFrame(
        {
            "bin": np.arange(bins),
            "radius_lower_px": lower_px,
            "radius_upper_px": upper_px,
            "radius_lower_arcsec": lower_px * PIXEL_SCALE_ARCSEC,
            "radius_upper_arcsec": upper_px * PIXEL_SCALE_ARCSEC,
            "radius_mean_arcsec": mean_radius,
            "cases": case_counts,
            "objects": object_counts,
            "R_self_measured": measured_mean,
            "R_self_measured_case_sem": measured_sem,
            "R_self_model": model_mean,
            "R_self_model_case_sem": model_sem,
            "residual_measured_minus_model": residual_mean,
            "residual_case_sem": residual_sem,
        }
    )
    frame["residual_z"] = (
        frame["residual_measured_minus_model"] / frame["residual_case_sem"]
    )
    frame = frame.loc[frame["objects"] > 0].reset_index(drop=True)

    metadata = {
        "format_version": 1,
        "purpose": "staged-flow self-response residual versus fixed-g0 measured FLUX_RADIUS on training cases",
        "residual_sign": "R_self_measured - R_self_model",
        "model_conditioning": "truth/context only; measured FLUX_RADIUS is used only for post-prediction diagnostic binning",
        "domain": "detected g0, MAG_AUTO(g0)<25.8, strict FLUX_RADIUS(g0)>3.0 px, no truth cut, no sheared-leg recut",
        "summary_weighting": "equal-weight mean over cases; uncertainty is one SEM over paired case residuals",
        "inputs": {
            "domain_root": str(domain_root.resolve()),
            "predictions": str(predictions_path.resolve()),
            "prediction_manifest": str(prediction_manifest_path.resolve()),
        },
        "cases": list(declared_cases),
        "case_count": len(declared_cases),
        "prediction_draws": int(prediction_manifest["draws"]),
        "prediction_sampling_seed": int(prediction_manifest["sampling_seed"]),
        "row_accounting": {
            "prediction_rows": int(len(predictions)),
            "profile_rows": int(object_counts.sum()),
            "per_case": accounting,
        },
        "binning": {
            "fine_width_px": 0.025,
            "fine_width_arcsec": 0.005,
            "fine_range_px": [3.0, 3.5],
            "edges_px": [None if not np.isfinite(value) else float(value) for value in edges_px],
        },
    }
    return frame, metadata


def _histogram_rate(frame: pd.DataFrame) -> np.ndarray:
    widths = (
        frame["radius_upper_arcsec"].to_numpy(np.float64)
        - frame["radius_lower_arcsec"].to_numpy(np.float64)
    )
    return np.divide(
        frame["objects"].to_numpy(np.float64) * 0.01,
        widths,
        out=np.zeros(len(frame), dtype=np.float64),
        where=np.isfinite(widths) & (widths > 0),
    )


def plot_profile(frame: pd.DataFrame, output_prefix: Path, case_count: int) -> None:
    """Render boundary-zoom and full-central-range residual panels."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    color = "#0072B2"
    boundary_color = "#D55E00"
    histogram_color = "#999999"
    x = frame["radius_mean_arcsec"].to_numpy(np.float64)
    y = frame["residual_measured_minus_model"].to_numpy(np.float64)
    yerr = frame["residual_case_sem"].to_numpy(np.float64)
    lower = frame["radius_lower_arcsec"].to_numpy(np.float64)
    upper = frame["radius_upper_arcsec"].to_numpy(np.float64)
    hist = _histogram_rate(frame)

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.45), constrained_layout=True)
    panels = (
        (axes[0], (0.597, 0.9), "Selection-boundary zoom", False),
        (axes[1], (0.59, 3.0), "Full central radius range", True),
    )
    for panel_index, (ax, limits, title, log_x) in enumerate(panels):
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(yerr)
            & (x >= limits[0])
            & (x <= limits[1])
        )
        finite_width = np.isfinite(upper) & (upper > lower)
        shown_hist = finite_width & (upper > limits[0]) & (lower < limits[1])
        twin = ax.twinx()
        twin.bar(
            x[shown_hist],
            hist[shown_hist],
            width=(upper - lower)[shown_hist],
            color=histogram_color,
            alpha=0.20,
            linewidth=0,
            align="center",
            zorder=0,
        )
        twin.set_ylim(bottom=0)
        twin.tick_params(axis="y", colors="#777777", labelsize=7)
        twin.spines["right"].set_color("#AAAAAA")
        twin.set_ylabel("Objects per 0.01 arcsec", color="#777777", fontsize=8)
        twin.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_alpha(0)

        ax.axhline(0.0, color="#222222", linewidth=0.9, zorder=2)
        ax.axvspan(
            SELECTION_BOUNDARY_ARCSEC,
            REFERENCE_SLICE_EDGE_ARCSEC,
            color="#E69F00",
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.axvline(
            SELECTION_BOUNDARY_ARCSEC,
            color=boundary_color,
            linestyle="--",
            linewidth=1.1,
            label="Strict training-domain boundary (3 px)",
            zorder=3,
        )
        ax.errorbar(
            x[valid],
            y[valid],
            yerr=yerr[valid],
            color=color,
            marker="o",
            markersize=2.7,
            markeredgewidth=0,
            linewidth=1.0,
            elinewidth=0.8,
            capsize=1.6,
            label=r"Measured $R_{\rm self}$ $-$ flow $R_{\rm self}$",
            zorder=4,
        )
        if log_x:
            ax.set_xscale("log")
            ax.set_xticks([0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0])
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(*limits)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Fixed-g0 measured FLUX_RADIUS (arcsec)")
        if panel_index == 0:
            ax.set_ylabel(r"Self-response residual $R_{\rm measured}-R_{\rm flow}$")
            ax.legend(loc="lower right", frameon=False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        twin.spines["top"].set_visible(False)
        ax.text(
            0.02,
            0.98,
            "a" if panel_index == 0 else "b",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=11,
        )

    fig.suptitle(
        f"Staged truth-conditioned flow on {case_count} training cases",
        fontsize=11,
    )
    fig.text(
        0.5,
        -0.02,
        "Error bars: +/-1 SEM across paired case residuals. "
        "Orange shading marks 0.600-0.6154 arcsec.",
        ha="center",
        va="top",
        fontsize=8,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_outputs(frame: pd.DataFrame, metadata: dict, output_prefix: Path) -> None:
    outputs = [
        output_prefix.with_suffix(".csv"),
        output_prefix.with_suffix(".json"),
        output_prefix.with_suffix(".png"),
        output_prefix.with_suffix(".pdf"),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_prefix.with_suffix(".csv"), index=False, float_format="%.10g")
    metadata = {
        **metadata,
        "outputs": {suffix[1:]: str(output_prefix.with_suffix(suffix).resolve()) for suffix in (".csv", ".json", ".png", ".pdf")},
        "rows": frame.to_dict("records"),
    }
    temporary = output_prefix.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_ready(metadata), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output_prefix.with_suffix(".json"))
    plot_profile(frame, output_prefix, int(metadata["case_count"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, metadata = build_profile(
        domain_root=args.domain_root,
        predictions_path=args.predictions,
        prediction_manifest_path=args.prediction_manifest,
    )
    write_outputs(frame, metadata, args.output_prefix)
    first_slice = frame.loc[
        frame["radius_lower_arcsec"] < REFERENCE_SLICE_EDGE_ARCSEC
    ]
    weighted_residual = np.average(
        first_slice["residual_measured_minus_model"],
        weights=first_slice["objects"],
    )
    print(
        f"FLOW_TRAIN_RADIUS_COMPLETE cases={metadata['case_count']} "
        f"objects={metadata['row_accounting']['profile_rows']} "
        f"boundary_slice_approx_residual={weighted_residual:.6f} "
        f"output={args.output_prefix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
