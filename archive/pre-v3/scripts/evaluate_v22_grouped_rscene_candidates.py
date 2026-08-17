#!/usr/bin/env python3
"""Select a grouped-residual strength without opening the final test blocks.

Candidate models are tested on official-validation pairs in half-shear cases
160--199 and independently on external development cases 0--19.  Cases 20--39,
coherent-anchor truth, and ConstGold are deliberately not read.
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
import xgboost as xgb

from scripts.v22_grouped_rscene_common import (
    CASE_MAX,
    CONDITIONAL_FEATURES,
    EXTERNAL_DEVELOPMENT_CASE_MAX,
    INTERNAL_VALIDATION_CASE_MIN,
    VALIDATION_SCALE,
    case_offsets,
    conditional_matrix,
    finite_stat,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    physical_correction,
    sha256,
    strict_json,
)


POPULATIONS = {
    "external_development_c0_19": (0, EXTERNAL_DEVELOPMENT_CASE_MAX, False),
    "internal_validation_c160_199": (
        INTERNAL_VALIDATION_CASE_MIN, CASE_MAX, True
    ),
}
SELECTIONS = ("all", "outside_le_0p1", "tail_gt_0p1", "tail_gt_0p2")
PAIR_MSE_TOLERANCE_PERCENT = 0.02


def selection_masks(scene_prediction: np.ndarray) -> dict[str, np.ndarray]:
    prediction = np.asarray(scene_prediction, dtype=np.float64)
    return {
        "all": np.ones(len(prediction), dtype=bool),
        "outside_le_0p1": prediction <= 0.1,
        "tail_gt_0p1": prediction > 0.1,
        "tail_gt_0p2": prediction > 0.2,
    }


def load_candidates(summary_paths: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    strengths: set[float] = set()
    for value in summary_paths:
        path = Path(value).resolve()
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        strength = float(summary["training"]["group_strength"])
        if strength in strengths:
            raise RuntimeError(f"duplicate group strength {strength}")
        strengths.add(strength)
        model_path = Path(summary["model"])
        if summary["feature_names"] != CONDITIONAL_FEATURES:
            raise RuntimeError(f"feature drift in {path}")
        if summary["training"]["case_window"] != [40, 159]:
            raise RuntimeError(f"candidate {path} was not trained only on c40--159")
        if sha256(model_path) != summary["model_sha256"]:
            raise RuntimeError(f"model hash differs from summary: {model_path}")
        booster = xgb.Booster({"device": "cpu", "n_jobs": 1})
        booster.load_model(model_path)
        candidates.append({
            "name": f"lambda_{strength:g}",
            "strength": strength,
            "summary_path": str(path),
            "summary_sha256": sha256(path),
            "model_path": str(model_path),
            "model_sha256": summary["model_sha256"],
            "target_scale": float(summary["target_scale"]),
            "booster": booster,
        })
    return sorted(candidates, key=lambda item: item["strength"])


def stat_columns(records: pd.DataFrame, value: str) -> dict[str, float | int]:
    return finite_stat(records[value].to_numpy(np.float64))


def summarize_case_metrics(case_metrics: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (population, model), local in case_metrics.groupby(
        ["population", "model"], sort=False
    ):
        pair_rows = local.loc[local.selection == "pair"]
        item: dict[str, Any] = {
            "n_cases": int(pair_rows.case.nunique()),
            "n_label_pairs": int(pair_rows.n_items.sum()),
            "pair_residual": stat_columns(pair_rows, "residual"),
            "pair_mse": stat_columns(pair_rows, "mse"),
            "pair_mse_percent_change_from_v22": stat_columns(
                pair_rows, "mse_percent_change"
            ),
            "scenes": {},
        }
        for selection in SELECTIONS:
            scene_rows = local.loc[local.selection == selection]
            item["scenes"][selection] = {
                "n_scenes": int(scene_rows.n_items.sum()),
                "truth_minus_prediction": stat_columns(scene_rows, "residual"),
                "v22_truth_minus_prediction": stat_columns(
                    scene_rows, "baseline_residual"
                ),
                "applied_scene_correction": stat_columns(
                    scene_rows, "correction"
                ),
            }
        output.setdefault(population, {})[model] = item
    return output


def aggregate_bins(records: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    for (population, model, bin_index), local in records.groupby(
        ["population", "model", "bin"], sort=True
    ):
        residual = finite_stat(local.residual.to_numpy(float))
        baseline = finite_stat(local.baseline_residual.to_numpy(float))
        correction = finite_stat(local.correction.to_numpy(float))
        coordinate = np.average(
            local.coordinate.to_numpy(float),
            weights=local.n_items.to_numpy(float),
        )
        rows.append({
            "kind": kind,
            "population": population,
            "model": model,
            "bin": int(bin_index),
            "coordinate": float(coordinate),
            "n_items": int(local.n_items.sum()),
            "n_cases": int(local.case.nunique()),
            "residual": residual["mean"],
            "residual_sem": residual["case_sem"],
            "baseline_residual": baseline["mean"],
            "baseline_residual_sem": baseline["case_sem"],
            "correction": correction["mean"],
            "correction_sem": correction["case_sem"],
        })
    return pd.DataFrame(rows)


def configure_plot() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def make_figure(
    pair_bins: pd.DataFrame,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected: str | None,
    output_prefix: Path,
) -> None:
    configure_plot()
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    candidate_names = [item["name"] for item in candidates]
    color_by_name = {
        name: colors[index % len(colors)]
        for index, name in enumerate(candidate_names)
    }
    for axis, population, title in (
        (axes[0, 0], "internal_validation_c160_199", "Internal half-shear validation"),
        (axes[0, 1], "external_development_c0_19", "External half-shear development"),
    ):
        base = pair_bins.loc[
            (pair_bins.population == population) & (pair_bins.model == "V2.2")
        ]
        axis.errorbar(
            base.coordinate, base.residual, yerr=base.residual_sem,
            color="black", linestyle="--", marker="o", ms=3,
            linewidth=1.4, capsize=2, label="V2.2",
        )
        for name in candidate_names:
            local = pair_bins.loc[
                (pair_bins.population == population) & (pair_bins.model == name)
            ]
            is_selected = name == selected
            axis.plot(
                local.coordinate, local.residual,
                color=color_by_name[name],
                alpha=1.0 if is_selected else 0.38,
                linewidth=2.2 if is_selected else 1.0,
                marker="s" if is_selected else None,
                ms=3, label=name.replace("lambda_", r"$\lambda=$"),
            )
        axis.axhline(0.0, color="0.55", linestyle=":", linewidth=1)
        axis.axvline(0.1, color="0.7", linestyle=":", linewidth=1)
        axis.set_xscale("symlog", linthresh=1.0e-3)
        axis.set_xlabel(r"Frozen V2.2 scene sum $P_s$")
        axis.set_ylabel("Mean pair residual (label − prediction)")
        axis.set_title(title)

    strengths = np.asarray([item["strength"] for item in candidates], float)
    for population, color, marker, label in (
        ("internal_validation_c160_199", "#0072B2", "o", "Internal c160–199"),
        ("external_development_c0_19", "#D55E00", "s", "External c0–19"),
    ):
        tail = np.asarray([
            summary[population][item["name"]]["scenes"]["tail_gt_0p1"]
            ["truth_minus_prediction"]["mean"]
            for item in candidates
        ])
        tail_sem = np.asarray([
            summary[population][item["name"]]["scenes"]["tail_gt_0p1"]
            ["truth_minus_prediction"]["case_sem"]
            for item in candidates
        ])
        axes[1, 0].errorbar(
            strengths, tail, yerr=tail_sem, color=color, marker=marker,
            linewidth=1.6, capsize=3, label=label,
        )
        baseline = summary[population]["V2.2"]["scenes"]["tail_gt_0p1"]
        axes[1, 0].axhline(
            baseline["truth_minus_prediction"]["mean"], color=color,
            linestyle=":", alpha=0.65,
        )
        mse = np.asarray([
            summary[population][item["name"]]
            ["pair_mse_percent_change_from_v22"]["mean"]
            for item in candidates
        ])
        mse_sem = np.asarray([
            summary[population][item["name"]]
            ["pair_mse_percent_change_from_v22"]["case_sem"]
            for item in candidates
        ])
        axes[1, 1].errorbar(
            strengths, mse, yerr=mse_sem, color=color, marker=marker,
            linewidth=1.6, capsize=3, label=label,
        )
    axes[1, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel(r"Grouped-loss strength $\lambda$")
    axes[1, 0].set_ylabel(r"Scene gap for $P_s>0.1$")
    axes[1, 0].set_title("Scene-tail transfer (dotted: V2.2)")
    axes[1, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1, 1].axhline(
        PAIR_MSE_TOLERANCE_PERCENT, color="0.65", linestyle=":", linewidth=1
    )
    axes[1, 1].set_xlabel(r"Grouped-loss strength $\lambda$")
    axes[1, 1].set_ylabel("Pair MSE change from V2.2 (%)")
    axes[1, 1].set_title("Ordinary pair loss")
    axes[0, 1].legend(frameon=False, ncol=2)
    axes[1, 0].legend(frameon=False)
    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes,
                  fontsize=12, fontweight="bold", va="top")
    fig.suptitle(
        "Pair-aware V2.2 residual model with equal scene-bin calibration loss\n"
        "Error bars are one SEM across rendered cases",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    path: Path,
    candidates: list[dict[str, Any]],
    summary: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    lines = [
        "# Grouped scene-residual candidate selection",
        "",
        "Candidate correction models were trained only on official V2.2 "
        "validation rows in half-shear cases 40–159. Selection uses c160–199 "
        "and external development cases c0–19. Cases c20–39 and coherent "
        "anchors remain unopened.",
        "",
        "| model | internal tail gap | external-dev tail gap | internal pair MSE Δ (%) | external-dev pair MSE Δ (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ["V2.2", *[item["name"] for item in candidates]]:
        internal = summary["internal_validation_c160_199"][model]
        external = summary["external_development_c0_19"][model]
        itail = internal["scenes"]["tail_gt_0p1"]["truth_minus_prediction"]
        etail = external["scenes"]["tail_gt_0p1"]["truth_minus_prediction"]
        imse = internal["pair_mse_percent_change_from_v22"]
        emse = external["pair_mse_percent_change_from_v22"]
        lines.append(
            f"| {model} | {itail['mean']:+.6f} ± {itail['case_sem']:.6f} | "
            f"{etail['mean']:+.6f} ± {etail['case_sem']:.6f} | "
            f"{imse['mean']:+.6f} ± {imse['case_sem']:.6f} | "
            f"{emse['mean']:+.6f} ± {emse['case_sem']:.6f} |"
        )
    lines.extend([
        "",
        f"Selection: `{selection['selected_model']}`. "
        f"{selection['reason']}",
        "",
        "Errors are one SEM across rendered cases.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--candidate-summary", action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (
            ".json", ".md", ".png", ".pdf", ".case_metrics.csv",
            ".pair_bins.csv", ".scene_bins.csv",
        )
    ]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    candidates = load_candidates(args.candidate_summary)
    offsets = case_offsets(source_meta)
    case_all = mmap_array(source_cache, source_meta, "case")
    primary_all = mmap_array(source_cache, source_meta, "input_index")
    official_train_all = mmap_array(source_cache, source_meta, "official_train")
    label_all = mmap_array(source_cache, source_meta, "label")
    pair_prediction_all = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction_all = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs_all = mmap_array(scene_cache, scene_meta, "scene_n_pairs")
    scene_bin_all = mmap_array(scene_cache, scene_meta, "scene_bin")

    case_records: list[dict[str, Any]] = []
    pair_bin_records: list[dict[str, Any]] = []
    scene_bin_records: list[dict[str, Any]] = []
    model_names = ["V2.2", *[item["name"] for item in candidates]]
    candidate_by_name = {item["name"]: item for item in candidates}

    for population, (case_min, case_max, sampled_labels) in POPULATIONS.items():
        for case in range(case_min, case_max + 1):
            start, stop = int(offsets[case]), int(offsets[case + 1])
            row_slice = slice(start, stop)
            if not np.all(np.asarray(case_all[row_slice]) == case):
                raise RuntimeError(f"case {case}: source-cache boundary mismatch")
            primary = np.asarray(primary_all[row_slice], dtype=np.int64)
            change = np.empty(len(primary), dtype=bool)
            change[0] = True
            change[1:] = primary[1:] != primary[:-1]
            starts = np.flatnonzero(change)
            counts = np.diff(np.r_[starts, len(primary)])
            if not np.array_equal(
                counts.astype(np.uint8),
                np.asarray(scene_n_pairs_all[row_slice], dtype=np.uint8)[starts],
            ):
                raise RuntimeError(f"case {case}: scene multiplicity mismatch")
            pair_prediction = np.asarray(
                pair_prediction_all[row_slice], dtype=np.float64
            )
            label = np.asarray(label_all[row_slice], dtype=np.float64)
            pair_residual0 = label - pair_prediction
            official_validation = ~np.asarray(
                official_train_all[row_slice], dtype=bool
            )
            pair_label_mask = official_validation if sampled_labels else np.ones(
                len(primary), dtype=bool
            )
            scene_prediction_rows = np.asarray(
                scene_prediction_all[row_slice], dtype=np.float64
            )
            scene_prediction = scene_prediction_rows[starts]
            replay = np.add.reduceat(pair_prediction, starts)
            if np.max(np.abs(replay - scene_prediction)) > 2.0e-6:
                raise RuntimeError(f"case {case}: frozen scene sum does not replay")
            scene_bins = np.asarray(scene_bin_all[row_slice], dtype=np.int64)[starts]
            pair_bins = np.asarray(scene_bin_all[row_slice], dtype=np.int64)
            if sampled_labels:
                sampled = np.where(pair_label_mask, pair_residual0, 0.0)
                baseline_scene_gap = VALIDATION_SCALE * np.add.reduceat(
                    sampled, starts
                )
            else:
                baseline_scene_gap = np.add.reduceat(pair_residual0, starts)
            masks = selection_masks(scene_prediction)

            features = conditional_matrix(
                x_scaled, x_raw, pair_prediction_all, scene_prediction_all,
                scene_n_pairs_all, row_slice,
            )
            corrections: dict[str, np.ndarray] = {
                "V2.2": np.zeros(len(primary), dtype=np.float64)
            }
            for name, candidate in candidate_by_name.items():
                corrections[name] = physical_correction(
                    candidate["booster"], features, candidate["target_scale"]
                )

            for model_name in model_names:
                correction = corrections[model_name]
                residual = pair_residual0 - correction
                selected_residual = residual[pair_label_mask]
                selected_baseline = pair_residual0[pair_label_mask]
                mse = float(np.mean(np.square(selected_residual)))
                baseline_mse = float(np.mean(np.square(selected_baseline)))
                case_records.append({
                    "population": population,
                    "model": model_name,
                    "case": case,
                    "selection": "pair",
                    "n_items": int(pair_label_mask.sum()),
                    "residual": float(selected_residual.mean()),
                    "baseline_residual": float(selected_baseline.mean()),
                    "correction": float(correction[pair_label_mask].mean()),
                    "mse": mse,
                    "baseline_mse": baseline_mse,
                    "mse_percent_change": 100.0 * (mse / baseline_mse - 1.0),
                })
                for bin_index in range(len(scene_meta["scene_bin_definition"]["edges"]) - 1):
                    local = pair_label_mask & (pair_bins == bin_index)
                    if not local.any():
                        continue
                    pair_bin_records.append({
                        "population": population,
                        "model": model_name,
                        "case": case,
                        "bin": bin_index,
                        "coordinate": float(scene_prediction_rows[local].mean()),
                        "n_items": int(local.sum()),
                        "residual": float(residual[local].mean()),
                        "baseline_residual": float(pair_residual0[local].mean()),
                        "correction": float(correction[local].mean()),
                    })

                scene_correction = np.add.reduceat(correction, starts)
                scene_gap = baseline_scene_gap - scene_correction
                for selection, mask in masks.items():
                    if not mask.any():
                        raise RuntimeError(
                            f"case {case}: empty scene selection {selection}"
                        )
                    case_records.append({
                        "population": population,
                        "model": model_name,
                        "case": case,
                        "selection": selection,
                        "n_items": int(mask.sum()),
                        "residual": float(scene_gap[mask].mean()),
                        "baseline_residual": float(
                            baseline_scene_gap[mask].mean()
                        ),
                        "correction": float(scene_correction[mask].mean()),
                        "mse": np.nan,
                        "baseline_mse": np.nan,
                        "mse_percent_change": np.nan,
                    })
                for bin_index in range(len(scene_meta["scene_bin_definition"]["edges"]) - 1):
                    local = scene_bins == bin_index
                    if not local.any():
                        continue
                    scene_bin_records.append({
                        "population": population,
                        "model": model_name,
                        "case": case,
                        "bin": bin_index,
                        "coordinate": float(scene_prediction[local].mean()),
                        "n_items": int(local.sum()),
                        "residual": float(scene_gap[local].mean()),
                        "baseline_residual": float(
                            baseline_scene_gap[local].mean()
                        ),
                        "correction": float(scene_correction[local].mean()),
                    })
            print(
                f"{population}: case {case} pairs={len(primary):,} "
                f"scenes={len(starts):,}", flush=True,
            )

    case_frame = pd.DataFrame(case_records)
    pair_bin_frame = aggregate_bins(pd.DataFrame(pair_bin_records), "pair")
    scene_bin_frame = aggregate_bins(pd.DataFrame(scene_bin_records), "scene")
    summary = summarize_case_metrics(case_frame)

    baseline_internal = abs(
        summary["internal_validation_c160_199"]["V2.2"]["scenes"]
        ["tail_gt_0p1"]["truth_minus_prediction"]["mean"]
    )
    baseline_external = abs(
        summary["external_development_c0_19"]["V2.2"]["scenes"]
        ["tail_gt_0p1"]["truth_minus_prediction"]["mean"]
    )
    selection_rows = []
    for candidate in candidates:
        name = candidate["name"]
        internal = summary["internal_validation_c160_199"][name]
        external = summary["external_development_c0_19"][name]
        internal_tail = abs(
            internal["scenes"]["tail_gt_0p1"]["truth_minus_prediction"]["mean"]
        )
        external_tail = abs(
            external["scenes"]["tail_gt_0p1"]["truth_minus_prediction"]["mean"]
        )
        internal_mse = internal["pair_mse_percent_change_from_v22"]["mean"]
        external_mse = external["pair_mse_percent_change_from_v22"]["mean"]
        eligible = bool(
            internal_tail < baseline_internal
            and external_tail < baseline_external
            and internal_mse <= PAIR_MSE_TOLERANCE_PERCENT
            and external_mse <= PAIR_MSE_TOLERANCE_PERCENT
        )
        selection_rows.append({
            "model": name,
            "strength": candidate["strength"],
            "eligible": eligible,
            "absolute_tail_gap_sum": internal_tail + external_tail,
            "internal_abs_tail_gap": internal_tail,
            "external_abs_tail_gap": external_tail,
            "internal_pair_mse_percent": internal_mse,
            "external_pair_mse_percent": external_mse,
        })
    eligible_rows = [row for row in selection_rows if row["eligible"]]
    if eligible_rows:
        chosen = min(eligible_rows, key=lambda row: row["absolute_tail_gap_sum"])
        selected_model = str(chosen["model"])
        reason = (
            "Lowest summed absolute P_s>0.1 scene gap among candidates that "
            "improve both validation populations and keep each pair-MSE change "
            f"at or below {PAIR_MSE_TOLERANCE_PERCENT:.3f}%."
        )
    else:
        chosen = min(selection_rows, key=lambda row: row["absolute_tail_gap_sum"])
        selected_model = None
        reason = (
            "No candidate passed the predeclared two-population tail-improvement "
            "and pair-MSE guardrail; no final model is selected. The lowest-gap "
            f"descriptive candidate was {chosen['model']}."
        )
    selection_payload = {
        "selected_model": selected_model,
        "descriptive_best_model": chosen["model"],
        "reason": reason,
        "pair_mse_tolerance_percent": PAIR_MSE_TOLERANCE_PERCENT,
        "candidate_rows": selection_rows,
    }

    payload = {
        "schema_version": 1,
        "experiment": "pair-aware residual model with equal R_scene-bin mean loss",
        "candidate_models": [
            {key: value for key, value in item.items() if key != "booster"}
            for item in candidates
        ],
        "metrics": summary,
        "selection": selection_payload,
        "scene_bin_edges": scene_meta["scene_bin_definition"]["edges"],
        "uncertainty": "one SEM across rendered cases",
        "provenance": {
            "source_cache": str(source_cache),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "scene_cache": str(scene_cache),
            "scene_metadata_sha256": sha256(scene_cache / "metadata.json"),
            "candidate_training_cases": [40, 159],
            "internal_validation_cases": [160, 199],
            "external_development_cases": [0, 19],
            "external_final_cases_20_39_opened": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    case_frame.to_csv(output_prefix.with_suffix(".case_metrics.csv"), index=False)
    pair_bin_frame.to_csv(output_prefix.with_suffix(".pair_bins.csv"), index=False)
    scene_bin_frame.to_csv(output_prefix.with_suffix(".scene_bins.csv"), index=False)
    strict_json(output_prefix.with_suffix(".json"), payload)
    write_markdown(
        output_prefix.with_suffix(".md"), candidates, summary, selection_payload
    )
    make_figure(
        pair_bin_frame, summary, candidates, selected_model, output_prefix
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("V22_GROUPED_RSCENE_CANDIDATES_DONE", flush=True)


if __name__ == "__main__":
    main()
