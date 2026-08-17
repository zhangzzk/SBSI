#!/usr/bin/env python3
"""Plot half-shear validation pair residuals against frozen model predictions.

This is the validation analogue of panel B in
``plot_v22_emulator_label_calibration.py``.  It reads the frozen half-shear
pair cache, evaluates the requested positive-response-weighted emulators on
cases 20--39, and makes one matched panel per model.  Each panel uses its own
20 equal-population prediction bins; uncertainties are paired SEMs across
rendered cases.

Constgold and coherent-anchor products are never read.
"""
from __future__ import annotations

import argparse
import hashlib
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
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU_ROOT = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.plot_v22_emulator_label_calibration import (  # noqa: E402
    json_clean,
    quantile_edges,
    summarize_calibration,
)
from scripts.train_v22_oof_learning import physical_predict  # noqa: E402


DEFAULT_MODELS = (
    (0.05, "lsst_r_extnbr_v22_rpowposw0050"),
    (0.10, "lsst_r_extnbr_v22_rpowposa010"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation_window(
    cache: Path,
    case_min: int,
    case_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one contiguous case window from the frozen pair cache."""
    with (cache / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    arrays = metadata["arrays"]
    case_all = np.load(cache / arrays["case"], mmap_mode="r")
    if case_all.ndim != 1 or len(case_all) != int(metadata["n_rows"]):
        raise RuntimeError("cached case array shape disagrees with metadata")
    start = int(np.searchsorted(case_all, case_min, side="left"))
    stop = int(np.searchsorted(case_all, case_max, side="right"))
    case = np.asarray(case_all[start:stop], dtype=np.int16)
    expected = sum(
        int(metadata["case_counts"][str(value)])
        for value in range(case_min, case_max + 1)
    )
    if len(case) != expected:
        raise RuntimeError(
            f"selected {len(case):,} rows but metadata expects {expected:,}"
        )
    if set(np.unique(case).tolist()) != set(range(case_min, case_max + 1)):
        raise RuntimeError("selected case window is incomplete")
    if start > 0 and int(case_all[start - 1]) >= case_min:
        raise RuntimeError("left case boundary is not contiguous")
    if stop < len(case_all) and int(case_all[stop]) <= case_max:
        raise RuntimeError("right case boundary is not contiguous")

    label_all = np.load(cache / arrays["label"], mmap_mode="r")
    feature_all = np.load(cache / arrays["x_scaled"], mmap_mode="r")
    label = np.asarray(label_all[start:stop], dtype=np.float64)
    features = np.asarray(feature_all[start:stop], dtype=np.float32)
    expected_features = metadata["model_features"]
    if features.shape != (expected, len(expected_features)):
        raise RuntimeError("cached feature matrix has an unexpected shape")
    if label.shape != (expected,):
        raise RuntimeError("cached label vector has an unexpected shape")
    if not np.isfinite(label).all() or not np.isfinite(features).all():
        raise RuntimeError("validation cache contains non-finite values")
    return case, label, features, metadata


def score_model(
    tag: str,
    features: np.ndarray,
    n_threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate one frozen BlendEMU regression model in physical units."""
    metadata_path = BLENDEMU_ROOT / "models" / f"emulator_metadata_{tag}.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    task = metadata["tasks"]["regression"]
    model_path = BLENDEMU_ROOT / "models" / task["model_file"]
    booster = xgb.Booster({"device": "cpu", "nthread": int(n_threads)})
    booster.load_model(model_path)
    booster.set_param({"device": "cpu", "nthread": int(n_threads)})
    standardization = task["standardization"]
    prediction = physical_predict(
        booster,
        features,
        float(standardization["mean"]),
        float(standardization["std"]),
    )
    provenance = {
        "tag": tag,
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "features": task["features"],
        "standardization": standardization,
        "response_power_weight": task["metrics"]["response_power_weight"],
    }
    return prediction, provenance


def validate_reference(
    summary: dict[str, Any],
    reference: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    """Require exact closure to the earlier independent-validation summary."""
    expected = reference["models"][model_key]["validation_c20_39"]
    checks = {
        "case_window": summary["case_window"] == expected["case_window"],
        "n_pairs": summary["n_pairs"] == expected["n_rows"],
        "label_mean": np.isclose(
            summary["global_pair_label"]["mean"],
            expected["pooled_label_mean"],
            rtol=0.0,
            atol=5.0e-7,
        ),
        "prediction_mean": np.isclose(
            summary["global_pair_prediction"]["mean"],
            expected["pooled_prediction_mean"],
            rtol=0.0,
            atol=5.0e-7,
        ),
        "paired_residual": np.isclose(
            summary["global_pair_label_minus_prediction"]["mean"],
            expected["residual_mean"]["mean"],
            rtol=0.0,
            atol=5.0e-12,
        ),
        "paired_residual_sem": np.isclose(
            summary["global_pair_label_minus_prediction"]["case_sem"],
            expected["residual_mean"]["case_sem"],
            rtol=0.0,
            atol=5.0e-12,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"reference closure failed for {model_key}: {checks}")
    return {key: bool(value) for key, value in checks.items()}


def flatten(models: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        for item in model["bins"]:
            rows.append({
                "alpha": model["alpha"],
                "tag": model["tag"],
                "bin": item["bin"],
                "lower": item["lower"],
                "upper": item["upper"],
                "n_pairs": item["n_pairs"],
                "pair_fraction": item["pair_fraction"],
                "prediction_mean": item["prediction"]["mean"],
                "prediction_case_sem": item["prediction"]["case_sem"],
                "label_mean": item["label"]["mean"],
                "label_case_sem": item["label"]["case_sem"],
                "residual_mean": item["label_minus_prediction"]["mean"],
                "residual_case_sem": item["label_minus_prediction"]["case_sem"],
            })
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def plot(models: list[dict[str, Any]], stem: Path) -> None:
    """Render matched panel-B-style residual curves and histograms."""
    configure_style()
    colors = ["#0072B2", "#D55E00"]
    markers = ["o", "s"]
    all_x = np.concatenate([
        np.asarray([item["prediction"]["mean"] for item in model["bins"]])
        for model in models
    ])
    x_lo, x_hi = float(all_x.min()), float(all_x.max())
    x_pad = 0.05 * max(x_hi - x_lo, 1.0e-3)
    limits = (x_lo - x_pad, x_hi + x_pad)
    all_y = np.concatenate([
        np.asarray([
            item["label_minus_prediction"]["mean"]
            for item in model["bins"]
        ])
        for model in models
    ])
    all_sem = np.concatenate([
        np.asarray([
            item["label_minus_prediction"]["case_sem"]
            for item in model["bins"]
        ])
        for model in models
    ])
    y_lo = float(np.min(all_y - all_sem))
    y_hi = float(np.max(all_y + all_sem))
    y_pad = 0.08 * max(y_hi - y_lo, 1.0e-3)

    fig, axes = plt.subplots(1, len(models), figsize=(7.4, 3.5), sharex=True, sharey=True)
    if len(models) == 1:
        axes = np.asarray([axes])
    for index, (axis, model, color, marker) in enumerate(
        zip(axes, models, colors, markers)
    ):
        bins = model["bins"]
        x = np.asarray([item["prediction"]["mean"] for item in bins])
        residual = np.asarray([
            item["label_minus_prediction"]["mean"] for item in bins
        ])
        residual_sem = np.asarray([
            item["label_minus_prediction"]["case_sem"] for item in bins
        ])
        residual_artist = axis.errorbar(
            x,
            residual,
            yerr=residual_sem,
            color=color,
            marker=marker,
            markersize=4.0,
            linewidth=1.15,
            capsize=2.0,
            label="Binned residual",
            zorder=3,
        )
        zero_artist = axis.axhline(
            0.0, color="0.30", linestyle="--", linewidth=0.9, zorder=2
        )
        axis.set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
        axis.set_xlim(limits)
        axis.set_ylim(y_lo - y_pad, y_hi + y_pad)
        axis.set_xlabel(r"Emulator pair prediction $p$")
        axis.set_title(rf"$\alpha={model['alpha']:.2f}$")
        axis.spines[["top", "right"]].set_visible(False)

        edges = np.asarray(model["prediction_edges"], dtype=float)
        fraction = np.asarray(
            [item["pair_fraction"] for item in bins], dtype=float
        )
        transformed_edges = axis.xaxis.get_transform().transform(edges)
        widths = np.diff(transformed_edges)
        if np.any(widths <= 0.0):
            raise RuntimeError("prediction histogram edges are not increasing")
        density = fraction / widths
        density /= density.max()
        histogram_axis = axis.twinx()
        histogram_artist = histogram_axis.stairs(
            density,
            edges,
            baseline=0.0,
            fill=True,
            color="0.55",
            alpha=0.24,
            linewidth=0.7,
            label="Prediction density",
        )
        histogram_axis.set_ylim(0.0, 1.8)
        histogram_axis.set_yticks([])
        histogram_axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
        histogram_axis.set_zorder(axis.get_zorder() - 1)
        axis.patch.set_alpha(0.0)

        global_residual = model["global_pair_label_minus_prediction"]
        axis.text(
            0.98,
            0.96,
            rf"global $\langle y-p\rangle={global_residual['mean']:+.5f}$"
            "\n"
            rf"$\pm{global_residual['case_sem']:.5f}$ (case SEM)",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
        )
        axis.text(
            -0.13,
            1.02,
            chr(ord("A") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="bottom",
        )
        if index == 0:
            axis.set_ylabel("Mean pair label − prediction")
            axis.legend(
                [residual_artist, zero_artist, histogram_artist],
                ["Binned residual", "Zero residual", "Prediction density"],
                frameon=False,
                loc="lower right",
            )

    fig.suptitle(
        "Half-shear validation pairs (cases 20–39): residual versus prediction\n"
        "20 equal-population bins per model; error bars are one SEM across 20 rendered cases",
        fontsize=9.5,
        y=0.98,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.75, wspace=0.16)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Positive-response-weighted pair residuals on half-shear validation",
        "",
        "This remakes the earlier pair-residual panel on independent half-shear "
        "validation cases 20--39. Each model is binned by its own frozen "
        "prediction into 20 equal-population bins. Labels are never used for "
        "binning. Errors are paired SEMs across rendered cases.",
        "",
        f"- Validation pairs: `{payload['n_pairs']:,}`.",
        f"- Rendered cases: `{payload['n_cases']}`.",
        "- Constgold opened: `false`.",
        "- Coherent-anchor truth opened: `false`.",
        "",
        "| alpha | global label | global prediction | label - prediction | largest absolute binned residual |",
        "|---:|---:|---:|---:|---:|",
    ]
    for model in payload["models"]:
        largest = max(
            model["bins"],
            key=lambda item: abs(item["label_minus_prediction"]["mean"]),
        )
        lines.append(
            f"| {model['alpha']:.2f} "
            f"| {model['global_pair_label']['mean']:+.6f} "
            f"| {model['global_pair_prediction']['mean']:+.6f} "
            f"| {model['global_pair_label_minus_prediction']['mean']:+.6f} "
            f"+- {model['global_pair_label_minus_prediction']['case_sem']:.6f} "
            f"| {largest['label_minus_prediction']['mean']:+.6f} "
            f"+- {largest['label_minus_prediction']['case_sem']:.6f} "
            f"(bin {largest['bin']}) |"
        )
    lines.extend([
        "",
        "The gray distribution in each panel is the relative density of that "
        "model's predictions per displayed signed-log interval.",
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
    outputs = [stem.with_suffix(f".{suffix}") for suffix in ("json", "csv", "md", "png", "pdf")]
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
        raise RuntimeError("reference does not use the frozen validation window")

    models: list[dict[str, Any]] = []
    for alpha, tag in DEFAULT_MODELS:
        print(f"scoring alpha={alpha:.2f} ({tag})", flush=True)
        prediction, provenance = score_model(tag, features, args.threads)
        if provenance["features"] != cache_metadata["model_features"]:
            raise RuntimeError(f"cached feature order does not match {tag}")
        edges = quantile_edges(prediction, args.n_bins)
        summary = summarize_calibration(
            case,
            label,
            prediction,
            edges,
            args.case_min,
            args.case_max,
        )
        summary.update({
            "alpha": float(alpha),
            "tag": tag,
            "provenance": provenance,
            "requested_n_bins": int(args.n_bins),
            "actual_n_bins": int(len(edges) - 1),
            "case_window": [args.case_min, args.case_max],
        })
        summary["reference_closure"] = validate_reference(
            summary, reference, f"amp_{alpha:.3f}"
        )
        models.append(summary)
        del prediction

    if len({model["n_pairs"] for model in models}) != 1:
        raise RuntimeError("models were not evaluated on the same pair population")
    payload = json_clean({
        "title": "Pair residual versus prediction for positive-response-weighted models",
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
    plot(models, stem)
    print(markdown(payload), flush=True)


if __name__ == "__main__":
    main()
