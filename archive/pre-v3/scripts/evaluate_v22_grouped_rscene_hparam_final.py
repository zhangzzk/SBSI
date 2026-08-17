#!/usr/bin/env python3
"""Evaluate current and tuned corrections on half-shear c20--39 only."""

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
import pandas as pd
import xgboost as xgb

from scripts.v22_grouped_rscene_common import (
    CONDITIONAL_FEATURES,
    case_offsets,
    conditional_matrix,
    json_clean,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    sha256,
    strict_json,
)
from scripts.v22_grouped_rscene_hparam_common import (
    FINAL_HALF_SHEAR_CASES,
    N_CURVE_BINS,
    physical_prediction,
    prediction_edges,
    rms,
)


MODEL_ORDER = ("V2.2", "Current correction", "Tuned correction")
GAIN_MODEL = "Gain-calibrated"
COLORS = {
    "V2.2": "#0072B2",
    "Current correction": "#D55E00",
    "Tuned correction": "#009E73",
    GAIN_MODEL: "#CC79A7",
}
MARKERS = {
    "V2.2": "o",
    "Current correction": "s",
    "Tuned correction": "^",
    GAIN_MODEL: "D",
}
LINESTYLES = {
    "V2.2": "--",
    "Current correction": ":",
    "Tuned correction": "-",
    GAIN_MODEL: "-.",
}


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("statistic requires at least two finite case values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def load_model_summary(path_string: str) -> dict[str, Any]:
    path = Path(path_string).resolve()
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["feature_names"] != CONDITIONAL_FEATURES:
        raise RuntimeError(f"feature definition drift in {path}")
    if summary["training"]["case_window"] != [40, 199]:
        raise RuntimeError(f"final model was not trained on c40--199: {path}")
    model_path = Path(summary["model"]).resolve()
    if sha256(model_path) != summary["model_sha256"]:
        raise RuntimeError(f"model hash drift in {path}")
    booster = xgb.Booster({
        "device": "cpu",
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    })
    booster.load_model(model_path)
    return {
        "summary_path": str(path),
        "summary_sha256": sha256(path),
        "summary": summary,
        "booster": booster,
    }


def load_strength_selection(
    path_string: str, tuned_model_summary: dict[str, Any]
) -> dict[str, Any]:
    """Load a half-shear-only gain selection and enforce its firewalls."""
    path = Path(path_string).resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    strength = float(payload["selection"]["selected_strength"])
    search = payload["search"]
    if not (
        np.isfinite(strength)
        and float(search["strength_min"]) <= strength <= float(search["strength_max"])
    ):
        raise RuntimeError("selected correction gain is outside its declared search")
    provenance = payload["provenance"]
    if provenance["external_final_cases_20_39_opened"]:
        raise RuntimeError("gain selection unexpectedly opened c20--39")
    if provenance["coherent_anchor_truth_opened"] or provenance["constgold_opened"]:
        raise RuntimeError("gain selection crossed an evaluation firewall")
    if payload["model_name"] != tuned_model_summary["selected_candidate_name"]:
        raise RuntimeError("gain selection and final tuned tree configuration differ")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "strength": strength,
        "payload": payload,
    }


def aggregate_curve(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, bin_index), local in frame.groupby(["model", "bin"], sort=True):
        stat = finite_stat(local.residual.to_numpy(float))
        rows.append({
            "model": model,
            "bin": int(bin_index),
            "coordinate": float(np.average(local.coordinate, weights=local.n_items)),
            "n_items": int(local.n_items.sum()),
            "n_cases": int(local.case.nunique()),
            "residual": stat["mean"],
            "residual_case_sem": stat["case_sem"],
        })
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_figure(
    pair_curve: pd.DataFrame,
    scene_curve: pd.DataFrame,
    summary: dict[str, Any],
    output_prefix: Path,
    model_order: tuple[str, ...] = MODEL_ORDER,
    gain: float | None = None,
) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15))
    for axis, frame, xlabel, title in (
        (
            axes[0],
            pair_curve,
            r"Frozen V2.2 pair prediction $p$",
            "Pair conditional residual",
        ),
        (
            axes[1],
            scene_curve,
            r"Frozen V2.2 scene sum $P_s$",
            "Accumulated scene gap",
        ),
    ):
        for model in model_order:
            local = frame.loc[frame.model == model].sort_values("bin")
            axis.errorbar(
                local.coordinate,
                local.residual,
                yerr=local.residual_case_sem,
                color=COLORS[model],
                marker=MARKERS[model],
                linestyle=LINESTYLES[model],
                linewidth=1.2,
                markersize=3.5,
                capsize=1.7,
                label=model,
            )
        axis.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_xscale("symlog", linthresh=1.0e-3)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Label − prediction")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, title="Model")
    for label, axis in zip("AB", axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes,
                  fontweight="bold", fontsize=11, va="top")
    gain_text = "" if gain is None else f"; gain {gain:.2f} frozen before opening these cases"
    fig.suptitle(
        "Half-shear cases 20–39: not used by this hyperparameter search"
        f"{gain_text}\n"
        "Shared label-free bins; error bars are one SEM across rendered cases",
        fontsize=10.5,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    model_order: tuple[str, ...] = MODEL_ORDER,
    gain: float | None = None,
) -> None:
    lines = [
        "# Final half-shear comparison after correction tuning",
        "",
        "Cases c20--39 were not used in this hyperparameter search. They had "
        "been examined in earlier analyses, so this is search-held-out rather "
        "than a never-seen experiment. No coherent-anchor or ConstGold data are read.",
        "",
    ]
    if gain is not None:
        lines.extend([
            f"The gain-calibrated row multiplies the tuned correction by {gain:.2f}.",
            "",
        ])
    lines.extend([
        "| model | pair residual | pair MSE | MSE d% vs V2.2 | scene gap | P_s>0.1 gap | calibration / V2.2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model in model_order:
        item = summary[model]
        lines.append(
            f"| {model} | {item['pair_residual']['mean']:+.6f} +- "
            f"{item['pair_residual']['case_sem']:.6f} | "
            f"{item['pair_mse']['mean']:.6f} | "
            f"{item['pair_mse_percent_change_from_v22']['mean']:+.5f} | "
            f"{item['scenes']['all']['truth_minus_prediction']['mean']:+.6f} +- "
            f"{item['scenes']['all']['truth_minus_prediction']['case_sem']:.6f} | "
            f"{item['scenes']['tail_gt_0p1']['truth_minus_prediction']['mean']:+.6f} +- "
            f"{item['scenes']['tail_gt_0p1']['truth_minus_prediction']['case_sem']:.6f} | "
            f"{item['calibration_score']:.4f} |"
        )
    lines.extend([
        "",
        "Calibration score equally averages pair-curve and accumulated-scene-curve "
        "RMS, each normalized to frozen V2.2; lower is better.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--current-model-summary", required=True)
    parser.add_argument("--tuned-model-summary", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--strength-selection")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix).resolve()
    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (".json", ".md", ".png", ".pdf", ".case_metrics.csv", ".pair_bins.csv", ".scene_bins.csv")
    ]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing final-evaluation outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    if scene_meta["source_metadata_sha256"] != sha256(source_cache / "metadata.json"):
        raise RuntimeError("scene/source cache provenance differs")
    current = load_model_summary(args.current_model_summary)
    tuned = load_model_summary(args.tuned_model_summary)
    models = {"Current correction": current, "Tuned correction": tuned}
    gain_selection = (
        load_strength_selection(args.strength_selection, tuned["summary"])
        if args.strength_selection
        else None
    )
    model_order = (
        (*MODEL_ORDER, GAIN_MODEL) if gain_selection is not None else MODEL_ORDER
    )

    offsets = case_offsets(source_meta)
    case_all = mmap_array(source_cache, source_meta, "case")
    primary_all = mmap_array(source_cache, source_meta, "input_index")
    label_all = mmap_array(source_cache, source_meta, "label")
    prediction_all = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction_all = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs_all = mmap_array(scene_cache, scene_meta, "scene_n_pairs")
    scene_bin_all = mmap_array(scene_cache, scene_meta, "scene_bin")

    case_min, case_max = FINAL_HALF_SHEAR_CASES
    edge_values = np.concatenate([
        np.asarray(prediction_all[int(offsets[case]):int(offsets[case + 1])], dtype=np.float64)
        for case in range(case_min, case_max + 1)
    ])
    pair_edges = prediction_edges(edge_values)
    del edge_values
    case_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    for case_value in range(case_min, case_max + 1):
        start, stop = int(offsets[case_value]), int(offsets[case_value + 1])
        row_slice = slice(start, stop)
        if not np.all(np.asarray(case_all[row_slice]) == case_value):
            raise RuntimeError(f"case {case_value}: source-cache boundary mismatch")
        primary = np.asarray(primary_all[row_slice], dtype=np.int64)
        change = np.empty(len(primary), dtype=bool)
        change[0] = True
        change[1:] = primary[1:] != primary[:-1]
        starts = np.flatnonzero(change)
        prediction = np.asarray(prediction_all[row_slice], dtype=np.float64)
        label = np.asarray(label_all[row_slice], dtype=np.float64)
        residual0 = label - prediction
        baseline_pair_mse = float(np.mean(np.square(residual0)))
        scene_prediction_rows = np.asarray(
            scene_prediction_all[row_slice], dtype=np.float64
        )
        scene_prediction = scene_prediction_rows[starts]
        baseline_scene_gap = np.add.reduceat(residual0, starts)
        pair_bins = np.searchsorted(pair_edges[1:-1], prediction, side="right")
        scene_bins = np.asarray(scene_bin_all[row_slice], dtype=np.int64)[starts]
        features = conditional_matrix(
            x_scaled,
            x_raw,
            prediction_all,
            scene_prediction_all,
            scene_n_pairs_all,
            row_slice,
        )
        corrections = {"V2.2": np.zeros(len(primary), dtype=np.float64)}
        for model, item in models.items():
            corrections[model] = physical_prediction(
                item["booster"],
                features,
                float(item["summary"]["target_scale"]),
            )
        if gain_selection is not None:
            corrections[GAIN_MODEL] = (
                gain_selection["strength"] * corrections["Tuned correction"]
            )
        for model in model_order:
            correction = corrections[model]
            residual = residual0 - correction
            mse = float(np.mean(np.square(residual)))
            case_records.append({
                "model": model,
                "case": case_value,
                "selection": "pair",
                "n_items": len(primary),
                "residual": float(residual.mean()),
                "correction": float(correction.mean()),
                "mse": mse,
                "mse_percent_change": 100.0 * (mse / baseline_pair_mse - 1.0),
            })
            for bin_index in range(N_CURVE_BINS):
                local = pair_bins == bin_index
                pair_records.append({
                    "model": model,
                    "case": case_value,
                    "bin": bin_index,
                    "coordinate": float(prediction[local].mean()),
                    "n_items": int(local.sum()),
                    "residual": float(residual[local].mean()),
                })
            scene_correction = np.add.reduceat(correction, starts)
            scene_gap = baseline_scene_gap - scene_correction
            for selection, mask in {
                "all": np.ones(len(starts), dtype=bool),
                "tail_gt_0p1": scene_prediction > 0.1,
                "tail_gt_0p2": scene_prediction > 0.2,
            }.items():
                case_records.append({
                    "model": model,
                    "case": case_value,
                    "selection": selection,
                    "n_items": int(mask.sum()),
                    "residual": float(scene_gap[mask].mean()),
                    "correction": float(scene_correction[mask].mean()),
                    "mse": np.nan,
                    "mse_percent_change": np.nan,
                })
            for bin_index in range(N_CURVE_BINS):
                local = scene_bins == bin_index
                scene_records.append({
                    "model": model,
                    "case": case_value,
                    "bin": bin_index,
                    "coordinate": float(scene_prediction[local].mean()),
                    "n_items": int(local.sum()),
                    "residual": float(scene_gap[local].mean()),
                })
        print(
            f"case {case_value}: pairs={len(primary):,} scenes={len(starts):,}",
            flush=True,
        )

    case_frame = pd.DataFrame(case_records)
    pair_curve = aggregate_curve(pd.DataFrame(pair_records))
    scene_curve = aggregate_curve(pd.DataFrame(scene_records))
    baseline_pair_rms = rms(
        pair_curve.loc[pair_curve.model == "V2.2"].sort_values("bin").residual
    )
    baseline_scene_rms = rms(
        scene_curve.loc[scene_curve.model == "V2.2"].sort_values("bin").residual
    )
    summary: dict[str, Any] = {}
    for model in model_order:
        local = case_frame.loc[case_frame.model == model]
        pair = local.loc[local.selection == "pair"]
        pair_rms = rms(
            pair_curve.loc[pair_curve.model == model].sort_values("bin").residual
        )
        scene_rms = rms(
            scene_curve.loc[scene_curve.model == model].sort_values("bin").residual
        )
        summary[model] = {
            "n_cases": int(pair.case.nunique()),
            "n_label_pairs": int(pair.n_items.sum()),
            "pair_residual": finite_stat(pair.residual.to_numpy(float)),
            "pair_mse": finite_stat(pair.mse.to_numpy(float)),
            "pair_mse_percent_change_from_v22": finite_stat(
                pair.mse_percent_change.to_numpy(float)
            ),
            "pair_prediction_curve_rms": pair_rms,
            "scene_prediction_curve_rms": scene_rms,
            "calibration_score": 0.5 * (
                pair_rms / baseline_pair_rms + scene_rms / baseline_scene_rms
            ),
            "scenes": {},
        }
        for selection in ("all", "tail_gt_0p1", "tail_gt_0p2"):
            rows = local.loc[local.selection == selection]
            summary[model]["scenes"][selection] = {
                "n_scenes": int(rows.n_items.sum()),
                "truth_minus_prediction": finite_stat(rows.residual.to_numpy(float)),
                "applied_correction": finite_stat(rows.correction.to_numpy(float)),
            }

    reference_path = Path(args.reference).resolve()
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    reference_metrics = reference["metrics"]["half_shear_final_c20_39"]
    checks = {
        "n_pairs": summary["V2.2"]["n_label_pairs"]
        == reference_metrics["V2.2"]["pairs"]["n_pairs"],
        "v22_pair_residual": np.isclose(
            summary["V2.2"]["pair_residual"]["mean"],
            reference_metrics["V2.2"]["pairs"]["residual"]["mean"],
            rtol=0.0,
            atol=2.0e-12,
        ),
        "current_pair_residual": np.isclose(
            summary["Current correction"]["pair_residual"]["mean"],
            reference_metrics["Grouped residual"]["pairs"]["residual"]["mean"],
            rtol=0.0,
            atol=2.0e-12,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"reference closure failed: {checks}")

    payload = {
        "schema_version": 1,
        "experiment": (
            "search-held-out half-shear evaluation of tuned V2.2 correction"
            if gain_selection is None
            else "search-held-out half-shear evaluation of gain-calibrated tuned V2.2 correction"
        ),
        "metrics": summary,
        "pair_prediction_edges": pair_edges,
        "scene_prediction_edges": scene_meta["scene_bin_definition"]["edges"],
        "audit": checks,
        "uncertainty": "one SEM across rendered cases",
        "provenance": {
            "case_window": list(FINAL_HALF_SHEAR_CASES),
            "status": "not used by this hyperparameter search; examined in earlier analyses",
            "source_cache": str(source_cache),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "scene_cache": str(scene_cache),
            "scene_metadata_sha256": sha256(scene_cache / "metadata.json"),
            "current_model_summary": current["summary_path"],
            "current_model_summary_sha256": current["summary_sha256"],
            "tuned_model_summary": tuned["summary_path"],
            "tuned_model_summary_sha256": tuned["summary_sha256"],
            "strength_selection": (
                None if gain_selection is None else gain_selection["path"]
            ),
            "strength_selection_sha256": (
                None if gain_selection is None else gain_selection["sha256"]
            ),
            "selected_strength": (
                None if gain_selection is None else gain_selection["strength"]
            ),
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
            "labels_used_to_define_bins": False,
        },
    }
    case_frame.to_csv(output_prefix.with_suffix(".case_metrics.csv"), index=False)
    pair_curve.to_csv(output_prefix.with_suffix(".pair_bins.csv"), index=False)
    scene_curve.to_csv(output_prefix.with_suffix(".scene_bins.csv"), index=False)
    strict_json(output_prefix.with_suffix(".json"), payload)
    write_markdown(
        output_prefix.with_suffix(".md"),
        summary,
        model_order=model_order,
        gain=None if gain_selection is None else gain_selection["strength"],
    )
    make_figure(
        pair_curve,
        scene_curve,
        summary,
        output_prefix,
        model_order=model_order,
        gain=None if gain_selection is None else gain_selection["strength"],
    )
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print("V22_RSCENE_HPARAM_FINAL_EVALUATION_DONE", flush=True)


if __name__ == "__main__":
    main()
