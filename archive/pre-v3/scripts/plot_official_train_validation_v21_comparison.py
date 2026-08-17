#!/usr/bin/env python3
"""Compare official fitting and validation residual curves in the V2.1 domain.

Both panels use the same half-shear cases, V2.1 primary cut, four frozen
emulators, residual sign, and case-level uncertainty definition.  The only
population difference is the stored official random 80/20 row split.
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

from scripts.plot_positive_weighted_pair_residual_validation import (
    configure_style,
    flatten,
    sha256,
)
from scripts.plot_positive_weighted_pair_residual_validation_overlay import (
    MODEL_SPECS,
)
from scripts.plot_v22_emulator_label_calibration import json_clean


DOMAIN_KEYS = (
    "primary_re_min",
    "primary_sn_min",
    "sn_sky_var",
    "sn_gain",
    "psf_re",
)


def infer_split(payload: dict[str, Any]) -> str:
    """Read the split from new metadata or infer it from the stored boolean."""
    design = payload["design"]
    if "official_split" in design:
        split_name = design["official_split"]
    else:
        split_name = "training" if design["official_train_value"] else "validation"
    if split_name not in ("training", "validation"):
        raise RuntimeError(f"invalid official split {split_name!r}")
    expected = split_name == "training"
    if bool(design["official_train_value"]) != expected:
        raise RuntimeError("official split name and boolean disagree")
    return split_name


def domain_signature(payload: dict[str, Any]) -> tuple[float, ...]:
    design = payload["design"]
    if design.get("v21_primary_domain") is not True:
        raise RuntimeError("comparison input does not apply the V2.1 primary domain")
    domain = design.get("v21_domain")
    if not isinstance(domain, dict):
        raise RuntimeError("comparison input lacks V2.1 domain metadata")
    return tuple(float(domain[key]) for key in DOMAIN_KEYS)


def validate_inputs(
    training: dict[str, Any],
    validation: dict[str, Any],
    all_rows_reference: dict[str, Any],
) -> dict[str, Any]:
    """Require an exact population partition and common model definitions."""
    if infer_split(training) != "training":
        raise RuntimeError("training input is not the official fitting arm")
    if infer_split(validation) != "validation":
        raise RuntimeError("validation input is not the official validation arm")
    for key in ("cache", "case_window", "random_state", "test_size"):
        if training["design"][key] != validation["design"][key]:
            raise RuntimeError(f"split inputs disagree on {key}")
    if domain_signature(training) != domain_signature(validation):
        raise RuntimeError("split inputs use different V2.1 domain definitions")
    train_domain = training["design"]["v21_domain"]
    validation_domain = validation["design"]["v21_domain"]
    if train_domain.get("truth_precision") != "original Feather float64":
        raise RuntimeError("training input does not use the exact float64 V2.1 mask")
    if (
        train_domain.get("mask_metadata_sha256")
        != validation_domain.get("mask_metadata_sha256")
    ):
        raise RuntimeError("split inputs do not use the same exact V2.1 mask")
    expected_tags = [spec["tag"] for spec in MODEL_SPECS]
    for name, payload in (("training", training), ("validation", validation)):
        tags = [model["tag"] for model in payload["models"]]
        if tags != expected_tags:
            raise RuntimeError(f"{name} model order or tags are wrong")
        if payload["n_cases"] != 160:
            raise RuntimeError(f"{name} input does not contain 160 cases")
        if any(model["n_pairs"] != payload["n_pairs"] for model in payload["models"]):
            raise RuntimeError(f"{name} models do not share one population")

    reference_ok = (
        all_rows_reference.get("case_window") == training["design"]["case_window"]
        and all_rows_reference.get("v21_primary_domain") is True
        and int(all_rows_reference.get("n_pairs", -1))
        == int(training["n_pairs"] + validation["n_pairs"])
    )
    if not reference_ok:
        raise RuntimeError("training + validation do not partition the old V2.1 all-row plot")
    return {
        "same_cache": True,
        "same_case_window": True,
        "same_random_split": True,
        "same_v21_domain": True,
        "same_model_order": True,
        "disjoint_split_values": True,
        "training_plus_validation_equals_old_all_rows": True,
        "old_all_rows": int(all_rows_reference["n_pairs"]),
    }


def brief(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "global": model["global_pair_label_minus_prediction"],
        "lowest_prediction_bin": model["bins"][0]["label_minus_prediction"],
        "highest_prediction_bin": model["bins"][-1]["label_minus_prediction"],
    }


def comparison_payload(
    training: dict[str, Any],
    validation: dict[str, Any],
    checks: dict[str, Any],
    training_path: Path,
    validation_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    models = []
    for train_model, validation_model in zip(training["models"], validation["models"]):
        train_stats = brief(train_model)
        validation_stats = brief(validation_model)
        models.append({
            "tag": train_model["tag"],
            "display": train_model["display"],
            "alpha": train_model["alpha"],
            "training": train_stats,
            "validation": validation_stats,
            "validation_minus_training_point_difference": {
                key: float(validation_stats[key]["mean"] - train_stats[key]["mean"])
                for key in train_stats
            },
        })
    return json_clean({
        "title": "Official fitting versus validation pair calibration in the V2.1 domain",
        "design": {
            "case_window": training["design"]["case_window"],
            "random_state": training["design"]["random_state"],
            "test_size": training["design"]["test_size"],
            "population": (
                "same V2.1 primary domain; split only by the stored official_train mask"
            ),
            "binning": (
                "20 own-prediction quantile bins separately within each split and model"
            ),
            "labels_used_in_binning": False,
            "residual_definition": "half-shear pair label minus model prediction",
            "uncertainty_unit": "rendered simulation case",
            "shared_plot_axes": True,
            "constgold_opened": False,
            "coherent_anchor_truth_opened": False,
        },
        "inputs": {
            "training": str(training_path.resolve()),
            "training_sha256": sha256(training_path),
            "validation": str(validation_path.resolve()),
            "validation_sha256": sha256(validation_path),
            "old_all_rows_reference": str(reference_path.resolve()),
            "old_all_rows_reference_sha256": sha256(reference_path),
        },
        "partition_checks": checks,
        "training_pairs": int(training["n_pairs"]),
        "validation_pairs": int(validation["n_pairs"]),
        "n_cases": int(training["n_cases"]),
        "models": models,
    })


def plot_panel(
    axis: plt.Axes,
    payload: dict[str, Any],
    panel_label: str,
    panel_title: str,
) -> tuple[list[Any], Any]:
    handles = []
    for model, spec in zip(payload["models"], MODEL_SPECS):
        bins = model["bins"]
        x = np.asarray([item["prediction"]["mean"] for item in bins])
        residual = np.asarray([
            item["label_minus_prediction"]["mean"] for item in bins
        ])
        sem = np.asarray([
            item["label_minus_prediction"]["case_sem"] for item in bins
        ])
        handles.append(axis.errorbar(
            x,
            residual,
            yerr=sem,
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            markersize=3.7,
            linewidth=1.1,
            capsize=1.5,
            alpha=0.92,
            zorder=3 + int(spec["alpha"] > 0),
        ))
    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9, zorder=2)
    axis.set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
    axis.set_xlabel(r"Emulator pair prediction $p$")
    axis.set_title(panel_title, pad=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        -0.10,
        1.03,
        panel_label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )

    global_lines = [r"Global $\langle y-p\rangle$"]
    for model, spec in zip(payload["models"], MODEL_SPECS):
        value = model["global_pair_label_minus_prediction"]
        global_lines.append(
            f"{spec['display']}: {value['mean']:+.5f} $\\pm$ {value['case_sem']:.5f}"
        )
    axis.text(
        0.98,
        0.97,
        "\n".join(global_lines),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        zorder=8,
    )

    baseline = payload["models"][0]
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
    histogram = histogram_axis.stairs(
        density,
        edges,
        baseline=0.0,
        fill=True,
        color="0.60",
        alpha=0.20,
        linewidth=0.7,
    )
    histogram_axis.set_ylim(0.0, 1.8)
    histogram_axis.set_yticks([])
    histogram_axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    histogram_axis.set_zorder(axis.get_zorder() - 1)
    axis.patch.set_alpha(0.0)
    return handles, histogram


def plot_comparison(
    training: dict[str, Any],
    validation: dict[str, Any],
    stem: Path,
) -> None:
    configure_style()
    payloads = (training, validation)
    all_x = []
    all_lower = []
    all_upper = []
    for payload in payloads:
        for model in payload["models"]:
            bins = model["bins"]
            x = np.asarray([item["prediction"]["mean"] for item in bins])
            residual = np.asarray([
                item["label_minus_prediction"]["mean"] for item in bins
            ])
            sem = np.asarray([
                item["label_minus_prediction"]["case_sem"] for item in bins
            ])
            all_x.append(x)
            all_lower.append(residual - sem)
            all_upper.append(residual + sem)
    x = np.concatenate(all_x)
    x_pad = 0.05 * max(float(x.max() - x.min()), 1.0e-3)
    lower = float(np.min(np.concatenate(all_lower)))
    upper = float(np.max(np.concatenate(all_upper)))
    y_pad = 0.08 * max(upper - lower, 1.0e-3)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharex=True, sharey=True)
    train_handles, histogram = plot_panel(
        axes[0],
        training,
        "A",
        f"Official fitting rows (80%; {training['n_pairs']:,} pairs)",
    )
    plot_panel(
        axes[1],
        validation,
        "B",
        f"Official validation rows (20%; {validation['n_pairs']:,} pairs)",
    )
    for axis in axes:
        axis.set_xlim(float(x.min() - x_pad), float(x.max() + x_pad))
        axis.set_ylim(lower - y_pad, upper + y_pad)
    axes[0].set_ylabel("Mean pair label − prediction")

    fig.legend(
        [*train_handles, histogram],
        [*[spec["display"] for spec in MODEL_SPECS], "V2.2 prediction density"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.5,
    )
    fig.suptitle(
        "Same V2.1 primary population: official fitting versus validation residual curves\n"
        "20 own-prediction quantile bins per model; error bars are one SEM across 160 cases",
        fontsize=9.5,
        y=0.98,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.73, wspace=0.12)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Official fitting versus validation curves in the V2.1 primary domain",
        "",
        "The two panels differ only by the stored official random row split. "
        "Both use cases 40--199, the identical V2.1 primary cut, the same four "
        "frozen models, and case SEMs. Training and validation together exactly "
        f"partition the `{payload['partition_checks']['old_all_rows']:,}` rows in "
        "the older all-row V2.2 calibration plot.",
        "",
        f"- Fitting pairs: `{payload['training_pairs']:,}`.",
        f"- Validation pairs: `{payload['validation_pairs']:,}`.",
        "- Constgold opened: `false`.",
        "- Coherent-anchor truth opened: `false`.",
        "",
        "| model | fitting global | validation global | validation - fitting | fitting low / high | validation low / high |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in payload["models"]:
        train = model["training"]
        validation = model["validation"]
        delta = model["validation_minus_training_point_difference"]
        lines.append(
            f"| {model['display']} "
            f"| {train['global']['mean']:+.6f} +- {train['global']['case_sem']:.6f} "
            f"| {validation['global']['mean']:+.6f} +- {validation['global']['case_sem']:.6f} "
            f"| {delta['global']:+.6f} "
            f"| {train['lowest_prediction_bin']['mean']:+.6f} / {train['highest_prediction_bin']['mean']:+.6f} "
            f"| {validation['lowest_prediction_bin']['mean']:+.6f} / {validation['highest_prediction_bin']['mean']:+.6f} |"
        )
    lines.extend([
        "",
        "The validation-minus-fitting column is a point difference; no uncertainty "
        "is attached because the saved summaries do not retain the case-level "
        "cross-split covariance. Each displayed curve still has its own case SEM.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-json", required=True)
    parser.add_argument("--validation-json", required=True)
    parser.add_argument("--all-rows-reference", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    training_path = Path(args.training_json)
    validation_path = Path(args.validation_json)
    reference_path = Path(args.all_rows_reference)
    with training_path.open(encoding="utf-8") as handle:
        training = json.load(handle)
    with validation_path.open(encoding="utf-8") as handle:
        validation = json.load(handle)
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    checks = validate_inputs(training, validation, reference)

    stem = Path(args.output_prefix)
    outputs = [
        stem.with_suffix(f".{suffix}")
        for suffix in ("json", "csv", "md", "png", "pdf")
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    stem.parent.mkdir(parents=True, exist_ok=True)

    payload = comparison_payload(
        training,
        validation,
        checks,
        training_path,
        validation_path,
        reference_path,
    )
    with stem.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    tables = []
    for split_name, source in (("training", training), ("validation", validation)):
        table = flatten(source["models"])
        table.insert(0, "official_split", split_name)
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    plot_comparison(training, validation, stem)
    print(markdown(payload), flush=True)


if __name__ == "__main__":
    main()
