"""Plot V2.2 pair predictions against their noisy half-shear labels.

The raw pair labels are too noisy for a useful scatter plot.  This script uses
the exact in-sample half-shear population from the secondary-size audit, forms
equal-population bins in the frozen emulator prediction, and plots the
case-balanced mean label with rendered-case uncertainty.  A paired residual
panel shows label minus prediction on the same bins.

Constgold and coherent-anchor products are never read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from sbs_shear.domain import in_domain
from scripts.diag_v22_secondary_size_gap import (
    BLENDEMU_ROOT,
    COND,
    CUT_NAMES,
    MODEL_DIR,
    PAIR_COLUMNS,
    PAIR_FEATURES,
    finite_stat,
    score_pairs,
)
from scripts.apply_v22_pair_residual_calibration_coherent import (
    hinge_residual,
    natural_cubic_spline_residual,
    saturating_hinge_residual,
)


def quantile_edges(prediction: np.ndarray, n_bins: int) -> np.ndarray:
    """Return strictly increasing pooled-prediction quantile edges."""
    prediction = np.asarray(prediction, float)
    if prediction.ndim != 1 or prediction.size == 0:
        raise ValueError("prediction must be a non-empty 1D array")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    edges = np.quantile(prediction, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("prediction has fewer than two distinct quantile bins")
    return edges


def summarize_calibration(
    case: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    edges: np.ndarray,
    case_min: int,
    case_max: int,
) -> dict[str, Any]:
    """Summarize prediction bins with case-balanced means and case SEMs."""
    case = np.asarray(case, np.int64)
    label = np.asarray(label, float)
    prediction = np.asarray(prediction, float)
    edges = np.asarray(edges, float)
    if not (case.shape == label.shape == prediction.shape):
        raise ValueError("case, label, and prediction shapes differ")
    if case.ndim != 1 or case.size == 0:
        raise ValueError("calibration arrays must be non-empty and 1D")
    if not np.isfinite(label).all() or not np.isfinite(prediction).all():
        raise ValueError("calibration values must be finite")
    if np.any((case < case_min) | (case > case_max)):
        raise ValueError("case lies outside the requested window")
    if edges.ndim != 1 or edges.size < 3 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing")

    n_case = case_max - case_min + 1
    n_bin = edges.size - 1
    bin_index = np.searchsorted(edges[1:-1], prediction, side="right")
    case_index = case - case_min
    flat = case_index * n_bin + bin_index
    minlength = n_case * n_bin
    counts = np.bincount(flat, minlength=minlength).reshape(n_case, n_bin)
    label_sum = np.bincount(
        flat, weights=label, minlength=minlength
    ).reshape(n_case, n_bin)
    prediction_sum = np.bincount(
        flat, weights=prediction, minlength=minlength
    ).reshape(n_case, n_bin)

    bins = []
    for index in range(n_bin):
        present = counts[:, index] > 0
        if present.sum() < 2:
            raise RuntimeError(f"prediction bin {index} occurs in fewer than two cases")
        case_label = label_sum[present, index] / counts[present, index]
        case_prediction = prediction_sum[present, index] / counts[present, index]
        case_residual = case_label - case_prediction
        bins.append({
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "n_pairs": int(counts[:, index].sum()),
            "pair_fraction": float(counts[:, index].sum() / counts.sum()),
            "n_cases_with_pairs": int(present.sum()),
            "prediction": finite_stat(case_prediction, test_zero=False),
            "label": finite_stat(case_label, test_zero=False),
            "label_minus_prediction": finite_stat(case_residual),
        })

    total_count = counts.sum(axis=1)
    if np.any(total_count == 0):
        missing = (np.flatnonzero(total_count == 0) + case_min).tolist()
        raise RuntimeError(f"cases without supported pairs: {missing}")
    total_label = label_sum.sum(axis=1) / total_count
    total_prediction = prediction_sum.sum(axis=1) / total_count
    total_residual = total_label - total_prediction
    return {
        "prediction_edges": edges.tolist(),
        "bins": bins,
        "global_pair_label": finite_stat(total_label, test_zero=False),
        "global_pair_prediction": finite_stat(total_prediction, test_zero=False),
        "global_pair_label_minus_prediction": finite_stat(total_residual),
        "n_pairs": int(counts.sum()),
        "n_cases": int(n_case),
        "maximum_bin_fraction_deviation_from_equal": float(
            np.max(np.abs(counts.sum(axis=0) / counts.sum() - 1.0 / n_bin))
        ),
    }


def collect_half_shear_pairs(
    catalogue: str,
    tag: str,
    case_min: int,
    case_max: int,
    shear: float,
    score_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream and score the same supported pairs as the size audit."""
    if BLENDEMU_ROOT not in sys.path:
        sys.path.insert(0, BLENDEMU_ROOT)
    from blendemu.inference import BlendingPredictor

    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=tag, conditions=COND, device="cpu"
    )
    raw_cuts, _, _ = predictor._select("regression")
    if len(raw_cuts) != len(CUT_NAMES):
        raise RuntimeError(f"unexpected regression cuts: {raw_cuts}")
    cuts = dict(zip(CUT_NAMES, raw_cuts))
    case_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    previous_min = -1
    processed_batches = 0
    scanned_rows = 0

    with ipc.open_file(catalogue) as reader:
        missing = set(PAIR_COLUMNS) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                PAIR_COLUMNS
            ).to_pandas()
            if frame.empty:
                continue
            batch_min = int(frame.case.min())
            if batch_min < previous_min:
                raise RuntimeError("response catalogue cases are not ordered")
            previous_min = batch_min
            scanned_rows += len(frame)
            frame = frame.loc[frame.case.between(case_min, case_max)]
            if frame.empty:
                if batch_min > case_max:
                    break
                continue
            processed_batches += 1
            finite_label = np.isfinite(
                frame[["delta_et1", "delta_et2"]].to_numpy(float)
            ).all(axis=1)
            frame = frame.loc[finite_label]
            pair_ok = (
                frame.r_input_p.between(
                    *cuts["r_input_p"], inclusive="neither"
                )
                & frame.Re_input_p.between(
                    *cuts["Re_input_p"], inclusive="neither"
                )
            )
            pair_ok &= in_domain(
                frame.r_input_p.to_numpy(float),
                frame.Re_input_p.to_numpy(float),
            )
            for column in ("r_input_s", "Re_input_s", "distance"):
                pair_ok &= frame[column].between(
                    *cuts[column], inclusive="neither"
                )
            pairs = frame.loc[pair_ok].copy()
            if pairs.empty:
                continue
            if not np.isfinite(pairs[PAIR_FEATURES].to_numpy(float)).all():
                raise RuntimeError("supported pair has a non-finite model feature")
            prediction = score_pairs(predictor, pairs, score_chunk)
            case_parts.append(pairs.case.to_numpy(np.int16, copy=True))
            label_parts.append(
                pairs.delta_et1.to_numpy(np.float64, copy=True) / shear
            )
            prediction_parts.append(prediction.astype(np.float64, copy=False))
            if processed_batches % 100 == 0:
                selected = sum(len(item) for item in case_parts)
                print(
                    f"processed {processed_batches} selected batches; "
                    f"supported pairs={selected:,}",
                    flush=True,
                )

    if not case_parts:
        raise RuntimeError("no supported pairs selected")
    case = np.concatenate(case_parts).astype(np.int64, copy=False)
    label = np.concatenate(label_parts)
    prediction = np.concatenate(prediction_parts)
    metadata = {
        "tag": tag,
        "catalogue": os.path.abspath(catalogue),
        "case_window": [int(case_min), int(case_max)],
        "shear": float(shear),
        "v21_primary_domain": True,
        "regression_cuts": [[float(x) for x in pair] for pair in raw_cuts],
        "processed_batches": int(processed_batches),
        "scanned_rows_through_last_batch": int(scanned_rows),
        "evaluation_status": "in-sample audit on V2.2 half-shear training cases",
        "uncertainty_unit": "rendered simulation case",
    }
    return case, label, prediction, metadata


def validate_reference(payload: dict[str, Any], path: str) -> dict[str, Any]:
    """Require exact population and mean closure to the size audit."""
    with open(path, encoding="utf-8") as handle:
        reference = json.load(handle)
    group = reference["groups"]["all"]
    checks = {
        "case_window": payload["case_window"] == reference["case_window"],
        "n_pairs": payload["n_pairs"] == reference["n_pairs"],
        "pair_label_mean": np.isclose(
            payload["global_pair_label"]["mean"],
            group["pair_label"]["mean"], rtol=0, atol=2e-12,
        ),
        "pair_prediction_mean": np.isclose(
            payload["global_pair_prediction"]["mean"],
            group["pair_prediction"]["mean"], rtol=0, atol=2e-8,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"calibration population fails reference closure: {checks}")
    return {
        "path": os.path.abspath(path),
        "checks": {key: bool(value) for key, value in checks.items()},
    }


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def fit_overlay_curves(
    calibration_payload: dict[str, Any],
    fit_payload: dict[str, Any],
    prediction_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate frozen parametric fits after strict source validation."""
    source = fit_payload.get("calibration_source", {})
    if source.get("n_pairs") != calibration_payload.get("n_pairs"):
        raise RuntimeError("fit overlay pair count does not match calibration")
    if source.get("case_window") != calibration_payload.get("case_window"):
        raise RuntimeError("fit overlay case window does not match calibration")
    fits = fit_payload.get("calibration_fits", {})
    if set(("hinge", "saturating_hinge")) - set(fits):
        raise KeyError("fit overlay lacks hinge or saturating_hinge")
    hinge_parameters = np.asarray(fits["hinge"]["parameters"], float)
    saturating_parameters = np.asarray(
        fits["saturating_hinge"]["parameters"], float
    )
    if hinge_parameters.shape != (3,) or saturating_parameters.shape != (4,):
        raise RuntimeError("fit overlay parameter count is invalid")
    if not (
        np.isfinite(hinge_parameters).all()
        and np.isfinite(saturating_parameters).all()
    ):
        raise RuntimeError("fit overlay parameters are non-finite")
    curves = {
        "hinge": hinge_residual(prediction_grid, hinge_parameters),
        "saturating_hinge": saturating_hinge_residual(
            prediction_grid, saturating_parameters
        ),
    }
    if "natural_cubic_spline" in fits:
        spline_parameters = np.asarray(
            fits["natural_cubic_spline"]["parameters"], float
        )
        if spline_parameters.ndim != 1 or spline_parameters.size % 2:
            raise RuntimeError("fit overlay spline parameters are invalid")
        curves["natural_cubic_spline"] = natural_cubic_spline_residual(
            prediction_grid, spline_parameters
        )
    return curves


def plot_calibration(
    payload: dict[str, Any], stem: str,
    fit_payload: dict[str, Any] | None = None,
) -> None:
    configure_style()
    bins = payload["bins"]
    x = np.asarray([item["prediction"]["mean"] for item in bins])
    y = np.asarray([item["label"]["mean"] for item in bins])
    y_sem = np.asarray([item["label"]["case_sem"] for item in bins])
    residual = np.asarray([
        item["label_minus_prediction"]["mean"] for item in bins
    ])
    residual_sem = np.asarray([
        item["label_minus_prediction"]["case_sem"] for item in bins
    ])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    axes[0].errorbar(
        x, y, yerr=y_sem, color="#0072B2", marker="o", markersize=4.0,
        linewidth=1.15, capsize=2.0, label="Binned half-shear label",
    )
    bound_lo = float(np.min(np.r_[x, y - y_sem]))
    bound_hi = float(np.max(np.r_[x, y + y_sem]))
    padding = 0.06 * max(bound_hi - bound_lo, 1e-3)
    limits = (bound_lo - padding, bound_hi + padding)
    axes[0].plot(
        limits, limits, color="0.35", linestyle="--", linewidth=0.9,
        label="Label = prediction",
    )
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    # Most predictions are very close to zero, while the extreme positive and
    # negative quantiles carry a long response tail.  Matching signed-log axes
    # preserve the identity line and make the central calibration visible.
    axes[0].set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
    axes[0].set_yscale("symlog", linthresh=1.0e-3, linscale=1.0)
    axes[0].set_xlabel(r"V2.2 emulator pair $R_{\rm blend}$")
    axes[0].set_ylabel(r"Mean half-shear pair label")
    axes[0].set_title("Calibration curve")
    axes[0].legend(frameon=False)

    residual_artist = axes[1].errorbar(
        x, residual, yerr=residual_sem, color="#D55E00", marker="o",
        markersize=4.0, linewidth=1.15, capsize=2.0,
        label="Binned residual",
    )
    axes[1].axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    axes[1].set_xlim(limits)
    axes[1].set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)

    fit_artists = []
    fit_labels = []
    if fit_payload is not None:
        negative = -np.geomspace(
            1.0e-7, max(abs(float(x.min())), 1.0e-7), 240,
        )[::-1]
        positive = np.geomspace(
            1.0e-7, max(float(x.max()), 1.0e-7), 240,
        )
        prediction_grid = np.r_[negative, 0.0, positive]
        fitted = fit_overlay_curves(payload, fit_payload, prediction_grid)
        hinge_artist, = axes[1].plot(
            prediction_grid, fitted["hinge"], color="#0072B2",
            linewidth=1.45, label="Hinge fit",
        )
        saturating_artist, = axes[1].plot(
            prediction_grid, fitted["saturating_hinge"], color="0.20",
            linestyle="--", linewidth=1.15, label="Saturating fit",
        )
        fit_artists = [hinge_artist, saturating_artist]
        fit_labels = ["Hinge fit", "Saturating fit"]
        if "natural_cubic_spline" in fitted:
            spline_artist, = axes[1].plot(
                prediction_grid, fitted["natural_cubic_spline"],
                color="#CC79A7", linestyle=":", linewidth=1.35,
                label="Natural cubic spline",
            )
            fit_artists.append(spline_artist)
            fit_labels.append("Natural cubic spline")

    # Put the prediction distribution behind panel B, rising from its lower
    # edge.  The curve bins are exact pooled-prediction quantiles, so density
    # per displayed signed-log width (rather than raw counts) shows the strong
    # concentration near zero.  Its independent axis preserves residual units.
    histogram_edges = np.asarray(payload["prediction_edges"], float)
    histogram_fraction = np.asarray([
        item["pair_fraction"] for item in bins
    ])
    transformed_edges = axes[1].xaxis.get_transform().transform(
        histogram_edges
    )
    transformed_width = np.diff(transformed_edges)
    if np.any(transformed_width <= 0.0):
        raise RuntimeError("prediction histogram edges are not increasing")
    histogram_density = histogram_fraction / transformed_width
    histogram_density /= histogram_density.max()
    histogram_axis = axes[1].twinx()
    histogram_artist = histogram_axis.stairs(
        histogram_density, histogram_edges, baseline=0.0, fill=True,
        color="0.55", alpha=0.24, linewidth=0.7,
        label="Prediction density",
    )
    # A peak of one occupies 56% of the panel height: visibly larger than the
    # earlier top marginal while remaining behind the residual curve.
    histogram_axis.set_ylim(0.0, 1.8)
    histogram_axis.set_yticks([])
    histogram_axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    histogram_axis.set_zorder(axes[1].get_zorder() - 1)
    axes[1].patch.set_alpha(0.0)

    axes[1].set_xlabel(r"V2.2 emulator pair $R_{\rm blend}$")
    axes[1].set_ylabel("Mean label - prediction")
    axes[1].set_title("Paired calibration residual")
    axes[1].legend(
        [residual_artist, *fit_artists, histogram_artist],
        ["Binned residual", *fit_labels, "Prediction density"],
        frameon=False, loc="upper left",
        ncol=(2 if fit_payload is not None else 1),
        columnspacing=0.8, handlelength=1.7,
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(0.040, 0.755, "A", fontweight="bold", fontsize=10, va="top")
    fig.text(0.545, 0.755, "B", fontweight="bold", fontsize=10, va="top")
    fig.suptitle(
        f"V2.2 half-shear training pairs: {len(bins)} prediction-quantile bins\n"
        "Error bars are one SEM across 160 rendered cases",
        fontsize=9.5, y=0.98,
    )
    fig.subplots_adjust(
        left=0.125, right=0.985, bottom=0.18, top=0.74, wspace=0.38
    )
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def flatten_csv(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload["bins"]:
        rows.append({
            "bin": item["bin"],
            "lower": item["lower"],
            "upper": item["upper"],
            "n_pairs": item["n_pairs"],
            "pair_fraction": item["pair_fraction"],
            "n_cases_with_pairs": item["n_cases_with_pairs"],
            "prediction_mean": item["prediction"]["mean"],
            "prediction_case_sem": item["prediction"]["case_sem"],
            "label_mean": item["label"]["mean"],
            "label_case_sem": item["label"]["case_sem"],
            "label_minus_prediction_mean": item[
                "label_minus_prediction"
            ]["mean"],
            "label_minus_prediction_case_sem": item[
                "label_minus_prediction"
            ]["case_sem"],
        })
    return pd.DataFrame(rows)


def markdown_report(
    payload: dict[str, Any], fit_payload: dict[str, Any] | None = None,
) -> str:
    residual = payload["global_pair_label_minus_prediction"]
    largest = max(
        payload["bins"],
        key=lambda item: abs(item["label_minus_prediction"]["mean"]),
    )
    largest_residual = largest["label_minus_prediction"]
    overlay_lines = []
    if fit_payload is not None:
        fits = fit_payload["calibration_fits"]
        overlay_lines = [
            "- Panel B overlays the frozen three-parameter hinge and "
            "four-parameter shared-scale saturating hinge from the coherent-"
            "transfer diagnostic.",
            f"- Fit reduced chi-square: hinge "
            f"`{fits['hinge']['reduced_chi2']:.3f}`; saturating "
            f"`{fits['saturating_hinge']['reduced_chi2']:.3f}`.",
        ]
        if "natural_cubic_spline" in fits:
            spline = fits["natural_cubic_spline"]
            overlay_lines.extend([
                "- Panel B also overlays an exact raw-prediction natural "
                "cubic spline with endpoint clamping.",
                f"- The spline's between-knot residual range is "
                f"`[{spline['between_knot_spline_range'][0]:+.6f}, "
                f"{spline['between_knot_spline_range'][1]:+.6f}]`.",
            ])
    return "\n".join([
        "# V2.2 emulator prediction versus half-shear label",
        "",
        "This is an in-sample calibration curve on the same supported "
        "half-shear training-pair population as the secondary-size audit. "
        "Pairs are split into equal-population quantile bins by the frozen "
        "V2.2 prediction. The plotted label is `delta_et1 / 0.2`.",
        "",
        f"- Cases: `{payload['case_window'][0]}--{payload['case_window'][1]}` "
        f"(`{payload['n_cases']}` rendered cases).",
        f"- Supported pairs: `{payload['n_pairs']:,}`.",
        f"- Prediction bins: `{payload['actual_n_bins']}`.",
        f"- Global pair label: `{payload['global_pair_label']['mean']:+.8f} "
        f"+- {payload['global_pair_label']['case_sem']:.8f}`.",
        f"- Global pair prediction: "
        f"`{payload['global_pair_prediction']['mean']:+.8f} "
        f"+- {payload['global_pair_prediction']['case_sem']:.8f}`.",
        f"- Global label minus prediction: `{residual['mean']:+.8f} "
        f"+- {residual['case_sem']:.8f}`.",
        f"- Largest absolute binned residual: "
        f"`{largest_residual['mean']:+.6f} +- "
        f"{largest_residual['case_sem']:.6f}` in bin `{largest['bin']}`.",
        *overlay_lines,
        "",
        "Vertical errors in panel A are one rendered-case SEM on the mean "
        "label. Panel B uses the paired case SEM of label minus prediction. "
        "The large gray histogram rising from the bottom of panel B is the "
        "relative density of pair predictions per displayed signed-log "
        "interval. "
        "No constgold or coherent-anchor measurement enters the figure.",
        "",
    ])


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--case-min", type=int, default=40)
    ap.add_argument("--case-max", type=int, default=199)
    ap.add_argument("--shear", type=float, default=0.2)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--score-chunk", type=int, default=500_000)
    ap.add_argument(
        "--fit-json", default=None,
        help="optional frozen parametric-fit result to overlay in panel B",
    )
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()
    if args.case_min > args.case_max or args.shear <= 0 or args.n_bins < 2:
        raise ValueError("invalid case window, shear, or bin count")

    outputs = [
        f"{args.output_prefix}.{suffix}"
        for suffix in ("json", "md", "csv", "png", "pdf")
    ]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    case, label, prediction, metadata = collect_half_shear_pairs(
        args.catalogue, args.tag, args.case_min, args.case_max, args.shear,
        args.score_chunk,
    )
    edges = quantile_edges(prediction, args.n_bins)
    payload = summarize_calibration(
        case, label, prediction, edges, args.case_min, args.case_max
    )
    payload.update(metadata)
    payload["requested_n_bins"] = int(args.n_bins)
    payload["actual_n_bins"] = int(len(edges) - 1)
    payload["binning"] = (
        "pooled exact quantiles of the frozen emulator prediction; curve means "
        "and uncertainties are case-balanced"
    )
    payload["reference_closure"] = validate_reference(payload, args.reference)
    payload = json_clean(payload)
    fit_payload = None
    if args.fit_json is not None:
        with open(args.fit_json, encoding="utf-8") as handle:
            fit_payload = json.load(handle)
        # Validate provenance before writing any output.
        fit_overlay_curves(payload, fit_payload, np.asarray([0.0]))

    with open(f"{args.output_prefix}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    Path(f"{args.output_prefix}.md").write_text(
        markdown_report(payload, fit_payload), encoding="utf-8"
    )
    flatten_csv(payload).to_csv(f"{args.output_prefix}.csv", index=False)
    plot_calibration(payload, args.output_prefix, fit_payload)
    print(json.dumps({
        "n_pairs": payload["n_pairs"],
        "global_pair_label": payload["global_pair_label"],
        "global_pair_prediction": payload["global_pair_prediction"],
        "global_pair_label_minus_prediction": payload[
            "global_pair_label_minus_prediction"
        ],
        "actual_n_bins": payload["actual_n_bins"],
    }, indent=2, allow_nan=False), flush=True)
    print("V22_EMULATOR_LABEL_CALIBRATION_DONE", flush=True)


if __name__ == "__main__":
    main()
