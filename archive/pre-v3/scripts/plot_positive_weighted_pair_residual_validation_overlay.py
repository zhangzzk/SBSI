#!/usr/bin/env python3
"""Overlay V2.2 and positive-weighted validation residual curves in one panel.

The half-shear pair population and residual definition match
``plot_positive_weighted_pair_residual_validation.py``.  Cases 20--39 are
external to all model training.  Every model is summarized in 20 quantile bins
of its own frozen prediction; the label is never used for binning.

Constgold and coherent-anchor products are never read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from scripts.plot_positive_weighted_pair_residual_validation import (
    configure_style,
    flatten,
    load_validation_window,
    score_model,
    validate_reference,
)
from scripts.plot_v22_emulator_label_calibration import (
    json_clean,
    quantile_edges,
    summarize_calibration,
)


MODEL_SPECS = (
    {
        "alpha": 0.0,
        "tag": "lsst_r_extnbr_v22",
        "reference_key": "baseline",
        "display": "V2.2",
        "color": "#000000",
        "marker": "o",
        "linestyle": "--",
        "cached": True,
    },
    {
        "alpha": 0.035,
        "tag": "lsst_r_extnbr_v22_rpowposw0035",
        "reference_key": "amp_0.035",
        "display": r"$\alpha=0.035$",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-.",
        "cached": False,
    },
    {
        "alpha": 0.05,
        "tag": "lsst_r_extnbr_v22_rpowposw0050",
        "reference_key": "amp_0.050",
        "display": r"$\alpha=0.05$",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
        "cached": False,
    },
    {
        "alpha": 0.10,
        "tag": "lsst_r_extnbr_v22_rpowposa010",
        "reference_key": "amp_0.100",
        "display": r"$\alpha=0.10$",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": ":",
        "cached": False,
    },
)


def load_cached_baseline(
    cache: Path,
    case_min: int,
    case_max: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the exact frozen V2.2 predictions for a contiguous case window."""
    metadata_path = cache / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    case = np.load(cache / metadata["arrays"]["case"], mmap_mode="r")
    start = int(np.searchsorted(case, case_min, side="left"))
    stop = int(np.searchsorted(case, case_max, side="right"))
    prediction_path = cache / metadata["arrays"]["v22_prediction"]
    prediction = np.asarray(
        np.load(prediction_path, mmap_mode="r")[start:stop], dtype=np.float64
    )
    expected = sum(
        int(metadata["case_counts"][str(value)])
        for value in range(case_min, case_max + 1)
    )
    if prediction.shape != (expected,) or not np.isfinite(prediction).all():
        raise RuntimeError("cached V2.2 prediction window is invalid")
    provenance = {
        "kind": "frozen_prediction_from_half_shear_pair_cache",
        "tag": metadata["source_tag"],
        "prediction_file": str(prediction_path),
        "cache_metadata": str(metadata_path),
        "source_model": metadata["source_model"],
        "source_model_sha256": metadata["source_model_sha256"],
        "source_metadata": metadata["source_metadata"],
        "source_metadata_sha256": metadata["source_metadata_sha256"],
    }
    return prediction, provenance


def summarize_model(
    case: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    spec: dict[str, Any],
    provenance: dict[str, Any],
    reference: dict[str, Any] | None,
    n_bins: int,
    case_min: int,
    case_max: int,
) -> dict[str, Any]:
    """Build one case-balanced prediction-quantile curve with strict checks."""
    edges = quantile_edges(prediction, n_bins)
    summary = summarize_calibration(
        case, label, prediction, edges, case_min, case_max
    )
    summary.update({
        "alpha": float(spec["alpha"]),
        "tag": spec["tag"],
        "display": spec["display"],
        "provenance": provenance,
        "requested_n_bins": int(n_bins),
        "actual_n_bins": int(len(edges) - 1),
        "case_window": [case_min, case_max],
    })
    if reference is not None:
        summary["reference_closure"] = validate_reference(
            summary, reference, spec["reference_key"]
        )
    return summary


def plot_overlay(
    models: list[dict[str, Any]],
    stem: Path,
    suptitle: str | None = None,
) -> None:
    """Render all residual curves in one axis with a V2.2 density backdrop."""
    configure_style()
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    residual_handles = []
    residual_labels = []

    all_x = []
    all_lower = []
    all_upper = []
    for model, spec in zip(models, MODEL_SPECS):
        bins = model["bins"]
        x = np.asarray([item["prediction"]["mean"] for item in bins])
        residual = np.asarray([
            item["label_minus_prediction"]["mean"] for item in bins
        ])
        sem = np.asarray([
            item["label_minus_prediction"]["case_sem"] for item in bins
        ])
        handle = axis.errorbar(
            x,
            residual,
            yerr=sem,
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            markersize=4.0,
            linewidth=1.15,
            capsize=1.7,
            alpha=0.92,
            zorder=3 + int(spec["alpha"] > 0),
        )
        global_residual = model["global_pair_label_minus_prediction"]["mean"]
        residual_handles.append(handle)
        residual_labels.append(
            f"{spec['display']}  (global {global_residual:+.5f})"
        )
        all_x.append(x)
        all_lower.append(residual - sem)
        all_upper.append(residual + sem)

    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9, zorder=2)
    axis.set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
    x = np.concatenate(all_x)
    x_pad = 0.05 * max(float(x.max() - x.min()), 1.0e-3)
    axis.set_xlim(float(x.min() - x_pad), float(x.max() + x_pad))
    lower = float(np.min(np.concatenate(all_lower)))
    upper = float(np.max(np.concatenate(all_upper)))
    y_pad = 0.08 * max(upper - lower, 1.0e-3)
    axis.set_ylim(lower - y_pad, upper + y_pad)
    axis.set_xlabel(r"Emulator pair prediction $p$")
    axis.set_ylabel("Mean pair label − prediction")
    axis.spines[["top", "right"]].set_visible(False)

    baseline = models[0]
    edges = np.asarray(baseline["prediction_edges"], dtype=float)
    fraction = np.asarray(
        [item["pair_fraction"] for item in baseline["bins"]], dtype=float
    )
    transformed_edges = axis.xaxis.get_transform().transform(edges)
    widths = np.diff(transformed_edges)
    if np.any(widths <= 0.0):
        raise RuntimeError("baseline prediction histogram edges are invalid")
    density = fraction / widths
    density /= density.max()
    histogram_axis = axis.twinx()
    histogram_handle = histogram_axis.stairs(
        density,
        edges,
        baseline=0.0,
        fill=True,
        color="0.60",
        alpha=0.20,
        linewidth=0.7,
        label="V2.2 prediction density",
    )
    histogram_axis.set_ylim(0.0, 1.8)
    histogram_axis.set_yticks([])
    histogram_axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    histogram_axis.set_zorder(axis.get_zorder() - 1)
    axis.patch.set_alpha(0.0)

    axis.legend(
        [*residual_handles, histogram_handle],
        [*residual_labels, "V2.2 prediction density"],
        title=r"Model  (global $\langle y-p\rangle$)",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        borderaxespad=0.0,
        handlelength=2.4,
    )
    if suptitle is None:
        suptitle = (
            "Half-shear validation pairs (cases 20–39): residual versus prediction\n"
            "20 equal-population bins per model; error bars are one SEM across 20 rendered cases"
        )
    fig.suptitle(suptitle, fontsize=9.5, y=0.98)
    fig.subplots_adjust(left=0.105, right=0.72, bottom=0.16, top=0.78)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Half-shear validation pair residual curves in one panel",
        "",
        "The frozen V2.2 baseline and positive-response-weighted alpha=0.035, "
        "0.05, and 0.10 models are overlaid on one axis. Every curve uses 20 "
        "equal-population bins of that model's own prediction. Labels do not "
        "enter the binning, and errors are paired SEMs across rendered cases.",
        "",
        f"- Validation pairs: `{payload['n_pairs']:,}`.",
        f"- Rendered cases: `{payload['n_cases']}`.",
        "- Constgold opened: `false`.",
        "- Coherent-anchor truth opened: `false`.",
        "",
        "| model | global label - prediction | lowest-prediction 5% | highest-prediction 5% |",
        "|---|---:|---:|---:|",
    ]
    for model in payload["models"]:
        global_residual = model["global_pair_label_minus_prediction"]
        low = model["bins"][0]["label_minus_prediction"]
        high = model["bins"][-1]["label_minus_prediction"]
        lines.append(
            f"| {model['display']} "
            f"| {global_residual['mean']:+.6f} +- {global_residual['case_sem']:.6f} "
            f"| {low['mean']:+.6f} +- {low['case_sem']:.6f} "
            f"| {high['mean']:+.6f} +- {high['case_sem']:.6f} |"
        )
    lines.extend([
        "",
        "The gray backdrop is only the V2.2 prediction density; overlaying four "
        "nearly identical density histograms would obscure the response curves.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--case-min", type=int, default=20)
    parser.add_argument("--case-max", type=int, default=39)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if (args.case_min, args.case_max) != (20, 39):
        raise ValueError("this validation figure is fixed to cases 20--39")
    if args.n_bins < 2 or args.threads < 1:
        raise ValueError("n-bins and threads must be positive")

    stem = Path(args.output_prefix)
    outputs = [
        stem.with_suffix(f".{suffix}")
        for suffix in ("json", "csv", "md", "png", "pdf")
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    stem.parent.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache)
    case, label, features, cache_metadata = load_validation_window(
        cache, args.case_min, args.case_max
    )
    with Path(args.reference).open(encoding="utf-8") as handle:
        reference = json.load(handle)
    if reference["design"]["reported_validation_cases"] != [20, 39]:
        raise RuntimeError("reference does not use validation cases 20--39")

    models: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        print(f"scoring {spec['display']} ({spec['tag']})", flush=True)
        if spec["cached"]:
            prediction, provenance = load_cached_baseline(
                cache, args.case_min, args.case_max
            )
        else:
            prediction, provenance = score_model(
                spec["tag"], features, args.threads
            )
            if provenance["features"] != cache_metadata["model_features"]:
                raise RuntimeError(
                    f"cached feature order does not match {spec['tag']}"
                )
        models.append(summarize_model(
            case,
            label,
            prediction,
            spec,
            provenance,
            reference,
            args.n_bins,
            args.case_min,
            args.case_max,
        ))
        del prediction

    if len({model["n_pairs"] for model in models}) != 1:
        raise RuntimeError("models do not share the same validation population")
    payload = json_clean({
        "title": "Half-shear validation pair residual curves in one panel",
        "design": {
            "source": "frozen half-shear response-pair cache",
            "cache": str(cache.resolve()),
            "cache_source_catalogue": cache_metadata["source_catalogue"],
            "case_window": [args.case_min, args.case_max],
            "validation_status": "external to all model training cases",
            "training_case_window": cache_metadata["fit_case_window"],
            "binning": "20 pooled prediction quantiles separately for each frozen model",
            "labels_used_in_binning": False,
            "residual_definition": "half-shear pair label minus model prediction",
            "uncertainty_unit": "rendered simulation case",
            "histogram": "V2.2 prediction density only",
            "constgold_opened": False,
            "coherent_anchor_truth_opened": False,
            "reference": str(Path(args.reference).resolve()),
        },
        "n_pairs": models[0]["n_pairs"],
        "n_cases": models[0]["n_cases"],
        "models": models,
    })

    with stem.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    flatten(models).to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    plot_overlay(models, stem)
    print(markdown(payload), flush=True)


if __name__ == "__main__":
    main()
