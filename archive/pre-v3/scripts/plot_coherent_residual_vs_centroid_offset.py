#!/usr/bin/env python3
"""Plot the raw V2.2 coherent-anchor residual versus centroid offset.

The response residual is ``R_blend_truth - R_blend_lsst_r_extnbr_v22``.
No bias emulator is loaded or applied.  The centroid coordinate is the larger
truth-to-detection crossmatch distance across the coherent +/- shear legs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats


TILE = "tile180.0_-0.5"
KEY = ["case", "input_index"]
TRUTH = "R_blend_truth"
PREDICTION = "R_blend_lsst_r_extnbr_v22"
RESIDUAL = "residual_truth_minus_v22"
OFFSET = "centroid_offset_max_arcsec"


def finite_stat(values: np.ndarray, hypothesis_test: bool = True) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("expected at least two finite values")
    standard_deviation = float(values.std(ddof=1))
    result: dict[str, Any] = {
        "mean": float(values.mean()),
        "case_sd": standard_deviation,
        "case_sem": standard_deviation / np.sqrt(values.size),
        "n_cases": int(values.size),
    }
    if hypothesis_test:
        test = stats.ttest_1samp(values, popmean=0.0)
        result.update({"t": float(test.statistic), "p": float(test.pvalue)})
    return result


def locate_case_base(case: int, bases: list[Path]) -> Path:
    matches = [
        base for base in bases
        if (base / f"anchors_case{case}.feather").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"case {case}: expected exactly one render base, found {matches}"
        )
    return matches[0]


def load_leg_match_distances(
    base: Path,
    case: int,
    shear_leg: float,
    input_ids: np.ndarray,
) -> pd.DataFrame:
    path = (
        base / f"case{case}_{str(float(shear_leg))}" / "real0" /
        "catalogues" / "CrossMatch" / f"{TILE}_rot0_matched.feather"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    match = pd.read_feather(
        path, columns=["id_input", "id_detec", "distance_pixel_CM"]
    )
    if match.id_input.duplicated().any():
        raise RuntimeError(f"case {case} leg {shear_leg}: duplicate input match")
    indexed = match.set_index("id_input", verify_integrity=True)
    input_ids = np.asarray(input_ids, dtype=np.int64)
    missing = pd.Index(input_ids).difference(indexed.index)
    if len(missing):
        raise RuntimeError(
            f"case {case} leg {shear_leg}: missing {len(missing)} response anchors"
        )
    selected = indexed.loc[input_ids]
    distance = selected.distance_pixel_CM.to_numpy(float)
    if (
        not np.isfinite(distance).all()
        or (distance < 0.0).any()
        or (distance >= 2.5).any()
    ):
        raise RuntimeError(f"case {case} leg {shear_leg}: invalid match distance")
    return pd.DataFrame({
        "input_index": input_ids,
        "distance_pixels": distance,
        "detection_index": selected.id_detec.to_numpy(np.int64) - 1,
    })


def combine_leg_offsets(
    target: pd.DataFrame,
    plus: pd.DataFrame,
    minus: pd.DataFrame,
    pixel_scale: float,
) -> pd.DataFrame:
    """Attach per-leg and maximum radial centroid offsets to target rows."""
    required_target = {*KEY, TRUTH, PREDICTION}
    if missing := required_target.difference(target.columns):
        raise KeyError(f"target lacks columns: {sorted(missing)}")
    for name, leg in (("plus", plus), ("minus", minus)):
        required = {"input_index", "distance_pixels", "detection_index"}
        if missing := required.difference(leg.columns):
            raise KeyError(f"{name} leg lacks columns: {sorted(missing)}")
        if leg.input_index.duplicated().any():
            raise RuntimeError(f"duplicate {name}-leg input key")
    if pixel_scale <= 0.0:
        raise ValueError("pixel scale must be positive")

    joined = target.merge(
        plus.rename(columns={
            "distance_pixels": "centroid_offset_plus_pixels",
            "detection_index": "detection_index_plus",
        }),
        on="input_index",
        how="inner",
        validate="one_to_one",
    ).merge(
        minus.rename(columns={
            "distance_pixels": "centroid_offset_minus_pixels",
            "detection_index": "detection_index_minus",
        }),
        on="input_index",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(target):
        raise RuntimeError("both-leg centroid join lost response anchors")
    joined["centroid_offset_plus_arcsec"] = (
        joined.centroid_offset_plus_pixels * pixel_scale
    )
    joined["centroid_offset_minus_arcsec"] = (
        joined.centroid_offset_minus_pixels * pixel_scale
    )
    joined["centroid_offset_mean_arcsec"] = 0.5 * (
        joined.centroid_offset_plus_arcsec
        + joined.centroid_offset_minus_arcsec
    )
    joined[OFFSET] = np.maximum(
        joined.centroid_offset_plus_arcsec,
        joined.centroid_offset_minus_arcsec,
    )
    joined["centroid_offset_leg_difference_arcsec"] = np.abs(
        joined.centroid_offset_plus_arcsec
        - joined.centroid_offset_minus_arcsec
    )
    joined[RESIDUAL] = joined[TRUTH] - joined[PREDICTION]
    numeric = joined[[
        TRUTH,
        PREDICTION,
        RESIDUAL,
        "centroid_offset_plus_arcsec",
        "centroid_offset_minus_arcsec",
        "centroid_offset_mean_arcsec",
        OFFSET,
    ]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("non-finite response or centroid coordinate")
    return joined


def load_population(
    responses: list[Path],
    bases: list[Path],
    case_min: int,
    case_max: int,
    shear_leg: float,
    pixel_scale: float,
) -> pd.DataFrame:
    pieces = [
        pd.read_feather(path, columns=[*KEY, TRUTH, PREDICTION])
        for path in responses
    ]
    response = pd.concat(pieces, ignore_index=True)
    response = response.loc[response.case.between(case_min, case_max)].copy()
    if response.empty or response.duplicated(KEY).any():
        raise RuntimeError("empty or duplicate coherent-response population")
    expected_cases = set(range(case_min, case_max + 1))
    if set(response.case.unique()) != expected_cases:
        raise RuntimeError("coherent-response population does not cover every requested case")

    output = []
    grouped = response.groupby("case", sort=True)
    for position, (case, target) in enumerate(grouped, start=1):
        case = int(case)
        target = target.copy()
        base = locate_case_base(case, bases)
        input_ids = target.input_index.to_numpy(np.int64)
        plus = load_leg_match_distances(base, case, shear_leg, input_ids)
        minus = load_leg_match_distances(base, case, -shear_leg, input_ids)
        output.append(combine_leg_offsets(target, plus, minus, pixel_scale))
        if position == 1 or position % 10 == 0 or position == len(grouped):
            print(
                f"centroid matches {position}/{len(grouped)}: case={case} "
                f"anchors={len(target):,}",
                flush=True,
            )
    frame = pd.concat(output, ignore_index=True)
    if len(frame) != len(response) or frame.duplicated(KEY).any():
        raise RuntimeError("assembled centroid table fails response-key closure")
    return frame


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if n_bins < 3 or values.size < n_bins or not np.isfinite(values).all():
        raise ValueError("invalid values or bin count")
    finite_edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    if len(np.unique(finite_edges)) != n_bins + 1:
        raise RuntimeError("centroid-offset quantiles are not unique")
    edges = finite_edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def conditional_curve(
    frame: pd.DataFrame,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    edges = quantile_edges(frame[OFFSET].to_numpy(float), n_bins)
    work = frame.copy()
    work["bin"] = np.digitize(
        work[OFFSET].to_numpy(float), edges[1:-1], right=False
    )
    rows = []
    for index in range(n_bins):
        local = work.loc[work.bin.eq(index)]
        by_case = local.groupby("case", sort=True).agg(
            offset_mean=(OFFSET, "mean"),
            truth_mean=(TRUTH, "mean"),
            prediction_mean=(PREDICTION, "mean"),
            residual_mean=(RESIDUAL, "mean"),
        )
        if local.empty or len(by_case) < 2:
            raise RuntimeError(f"centroid bin {index} lacks case support")
        offset_stat = finite_stat(by_case.offset_mean.to_numpy(float), False)
        truth_stat = finite_stat(by_case.truth_mean.to_numpy(float), False)
        prediction_stat = finite_stat(
            by_case.prediction_mean.to_numpy(float), False
        )
        residual_stat = finite_stat(by_case.residual_mean.to_numpy(float))
        rows.append({
            "bin": int(index),
            "lower_arcsec": float(edges[index]),
            "upper_arcsec": float(edges[index + 1]),
            "offset_mean_arcsec": offset_stat["mean"],
            "offset_row_p16_arcsec": float(local[OFFSET].quantile(0.16)),
            "offset_row_p84_arcsec": float(local[OFFSET].quantile(0.84)),
            "n_anchors": int(len(local)),
            "fraction_anchors": float(len(local) / len(work)),
            "n_cases": int(len(by_case)),
            "truth_mean": truth_stat["mean"],
            "truth_case_sem": truth_stat["case_sem"],
            "v22_prediction_mean": prediction_stat["mean"],
            "v22_prediction_case_sem": prediction_stat["case_sem"],
            "residual_truth_minus_v22": residual_stat["mean"],
            "residual_case_sem": residual_stat["case_sem"],
        })
    return pd.DataFrame(rows), work, edges


def paired_bin_contrast(work: pd.DataFrame, low_bin: int, high_bin: int) -> dict:
    grouped = work.loc[work.bin.isin([low_bin, high_bin])].groupby(
        ["case", "bin"], sort=True
    )[RESIDUAL].mean().unstack("bin")
    paired = grouped.dropna(subset=[low_bin, high_bin])
    return finite_stat(
        paired[high_bin].to_numpy(float) - paired[low_bin].to_numpy(float)
    )


def within_case_slopes(frame: pd.DataFrame, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("slope scale must be positive")
    slopes = []
    for _, local in frame.groupby("case", sort=True):
        x = local[OFFSET].to_numpy(float) / scale
        y = local[RESIDUAL].to_numpy(float)
        x = x - x.mean()
        denominator = float(np.dot(x, x))
        if denominator > 0.0:
            slopes.append(float(np.dot(x, y - y.mean()) / denominator))
    return np.asarray(slopes, dtype=float)


def global_summary(frame: pd.DataFrame) -> dict[str, Any]:
    by_case = frame.groupby("case", sort=True)[
        [TRUTH, PREDICTION, RESIDUAL]
    ].mean()
    offset = frame[OFFSET].to_numpy(float)
    iqr = float(np.subtract(*np.quantile(offset, [0.75, 0.25])))
    slopes = within_case_slopes(frame, iqr)
    return {
        "truth": finite_stat(by_case[TRUTH].to_numpy(float)),
        "v22_prediction": finite_stat(by_case[PREDICTION].to_numpy(float)),
        "residual_truth_minus_v22": finite_stat(
            by_case[RESIDUAL].to_numpy(float)
        ),
        "offset_arcsec": {
            "minimum": float(np.min(offset)),
            "median": float(np.median(offset)),
            "p84": float(np.quantile(offset, 0.84)),
            "p95": float(np.quantile(offset, 0.95)),
            "maximum": float(np.max(offset)),
            "iqr": iqr,
        },
        "within_case_residual_slope_per_offset_iqr": finite_stat(slopes),
    }


def plot_figure(
    frame: pd.DataFrame,
    curve: pd.DataFrame,
    edges: np.ndarray,
    summary: dict[str, Any],
    output_prefix: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    figure = plt.figure(figsize=(5.5, 4.6))
    grid = figure.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    axis = figure.add_subplot(grid[0])
    histogram_axis = figure.add_subplot(grid[1], sharex=axis)

    x = curve.offset_mean_arcsec.to_numpy(float)
    y = curve.residual_truth_minus_v22.to_numpy(float)
    yerr = curve.residual_case_sem.to_numpy(float)
    axis.errorbar(
        x,
        y,
        yerr=yerr,
        color="#0072B2",
        marker="o",
        markersize=4.2,
        linewidth=1.35,
        capsize=2.2,
        label=r"$R_{\rm blend}^{\rm coherent}-R_{\rm blend}^{\rm V2.2}$",
    )
    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    global_residual = summary["residual_truth_minus_v22"]
    axis.axhline(
        global_residual["mean"],
        color="#0072B2",
        linestyle=":",
        linewidth=1.0,
        alpha=0.85,
    )
    axis.text(
        0.025,
        0.965,
        (
            f"Global residual = {global_residual['mean']:+.4f} ± "
            f"{global_residual['case_sem']:.4f}\n"
            f"Dotted line: global mean"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    axis.set_ylabel(r"Coherent truth $-$ V2.2 prediction")
    axis.set_title(
        "Held-out c700–899; errors are one SEM across rendered cases",
        pad=4.0,
    )
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelbottom=False)

    maximum = float(frame[OFFSET].max())
    upper_limit = max(0.05, np.ceil(maximum / 0.05) * 0.05)
    histogram_edges = np.linspace(0.0, upper_limit, 41)
    histogram_axis.hist(
        frame[OFFSET].to_numpy(float),
        bins=histogram_edges,
        weights=np.full(len(frame), 100.0 / len(frame)),
        color="0.65",
        edgecolor="white",
        linewidth=0.35,
    )
    for edge in edges[1:-1]:
        histogram_axis.axvline(edge, color="0.30", linewidth=0.35, alpha=0.35)
    histogram_axis.set_xlim(0.0, upper_limit)
    histogram_axis.set_ylabel("Anchors\nper bin (%)", fontsize=7.5)
    histogram_axis.set_xlabel(
        r"Maximum truth–detected centroid offset across $\pm0.02$ legs (arcsec)"
    )
    histogram_axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        r"V2.2 $R_{\rm blend}$ residual versus detection-centroid offset",
        fontsize=11.0,
        y=0.985,
    )
    figure.text(
        0.01, 0.93, "A", fontsize=10.5, fontweight="bold", va="top"
    )
    figure.text(
        0.01, 0.285, "B", fontsize=10.5, fontweight="bold", va="top"
    )
    figure.subplots_adjust(
        left=0.135, right=0.98, bottom=0.12, top=0.89, hspace=0.08
    )
    figure.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    figure.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    global_residual = payload["global"]["residual_truth_minus_v22"]
    contrast = payload["trend"]["highest_minus_lowest_offset_bin"]
    slope = payload["global"]["within_case_residual_slope_per_offset_iqr"]
    lines = [
        "# V2.2 coherent residual versus centroid offset",
        "",
        "This diagnostic uses the original V2.2 `R_blend` emulator only. No bias "
        "emulator is loaded or applied. The plotted residual is "
        "`R_blend_truth - R_blend_V2.2`, so positive values mean V2.2 "
        "underpredicts the coherent response.",
        "",
        f"The figure contains `{payload['population']['n_anchors']:,}` anchors "
        f"from `{payload['population']['n_cases']}` held-out coherent cases. The "
        "centroid coordinate is the maximum truth-to-detection match distance "
        "across the +0.02 and -0.02 renders. The response-free offset quantiles "
        f"define `{payload['binning']['n_bins']}` equal-count bins; error bars are "
        "one SEM across rendered-case conditional means.",
        "",
        f"The case-balanced global residual is `{global_residual['mean']:+.6f} "
        f"+- {global_residual['case_sem']:.6f}`. The highest-minus-lowest offset "
        f"bin contrast is `{contrast['mean']:+.6f} +- "
        f"{contrast['case_sem']:.6f}`. A within-case linear summary gives "
        f"`{slope['mean']:+.6f} +- {slope['case_sem']:.6f}` residual per one "
        "global offset IQR.",
        "",
        "Centroid offset is a post-render measured quantity, not an input to V2.2. "
        "This plot is therefore a localization diagnostic, not a causal test or "
        "a deployable correction.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", action="append", required=True)
    parser.add_argument("--base", action="append", required=True)
    parser.add_argument("--case-min", type=int, default=700)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--shear-leg", type=float, default=0.02)
    parser.add_argument("--pixel-scale", type=float, default=0.2)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if (
        args.case_min > args.case_max
        or args.shear_leg <= 0.0
        or args.pixel_scale <= 0.0
        or args.n_bins < 3
    ):
        raise ValueError("invalid case window, shear leg, pixel scale, or bin count")
    responses = [Path(value) for value in args.response]
    bases = [Path(value) for value in args.base]
    for path in [*responses, *bases]:
        if not path.exists():
            raise FileNotFoundError(path)
    output_prefix = Path(args.output_prefix)
    suffixes = ("csv", "json", "md", "pdf", "png")
    existing = [
        str(Path(f"{output_prefix}.{suffix}")) for suffix in suffixes
        if Path(f"{output_prefix}.{suffix}").exists()
    ]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frame = load_population(
        responses,
        bases,
        args.case_min,
        args.case_max,
        args.shear_leg,
        args.pixel_scale,
    )
    curve, work, edges = conditional_curve(frame, args.n_bins)
    summary = global_summary(frame)
    contrast = paired_bin_contrast(work, 0, args.n_bins - 1)
    spearman = stats.spearmanr(
        curve.offset_mean_arcsec.to_numpy(float),
        curve.residual_truth_minus_v22.to_numpy(float),
    )
    payload = {
        "title": "V2.2 coherent residual versus detection-centroid offset",
        "sources": {
            "response_catalogues": [str(path.resolve()) for path in responses],
            "render_directories": [str(path.resolve()) for path in bases],
            "crossmatch_catalogue": f"CrossMatch/{TILE}_rot0_matched.feather",
        },
        "population": {
            "case_window": [int(args.case_min), int(args.case_max)],
            "n_cases": int(frame.case.nunique()),
            "n_anchors": int(len(frame)),
        },
        "residual": {
            "definition": "R_blend_truth - R_blend_lsst_r_extnbr_v22",
            "sign": "positive means V2.2 underpredicts coherent response",
            "bias_emulator_loaded_or_applied": False,
        },
        "centroid_offset": {
            "definition": (
                "pixel_scale * max(distance_pixel_CM(+0.02), "
                "distance_pixel_CM(-0.02))"
            ),
            "pixel_scale_arcsec": float(args.pixel_scale),
            "response_population_already_retained_in_both_legs": True,
        },
        "binning": {
            "n_bins": int(args.n_bins),
            "method": (
                "equal-count quantiles of held-out centroid offset; response was "
                "not used to define edges"
            ),
            "edges_arcsec": json_clean(edges.tolist()),
        },
        "global": summary,
        "trend": {
            "highest_minus_lowest_offset_bin": contrast,
            "bin_mean_spearman": float(spearman.statistic),
            "bin_mean_spearman_p": float(spearman.pvalue),
        },
        "post_render_diagnostic_only": True,
        "correction_fitted": False,
        "constgold_opened": False,
        "bins": json_clean(curve.to_dict(orient="records")),
    }
    plot_figure(frame, curve, edges, summary, output_prefix)
    curve.to_csv(f"{output_prefix}.csv", index=False)
    with open(f"{output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_markdown(payload, Path(f"{output_prefix}.md"))
    print(
        f"saved coherent residual-centroid diagnostic to {output_prefix}.*",
        flush=True,
    )
    print("COHERENT_RESIDUAL_VS_CENTROID_OFFSET_DONE", flush=True)


if __name__ == "__main__":
    main()
