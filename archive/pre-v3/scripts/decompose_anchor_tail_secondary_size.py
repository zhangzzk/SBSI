"""Decompose the panel-E tail by the proposed compact-secondary cut.

The parent all-anchor panel defines a frozen high-scene-response tail.  On the
same final-test coherent anchors this script crosses that tail with the post-hoc
cut on response-dominant secondary size.  It reports conditional gaps after
removal and four additive contributions on the full-anchor denominator, so a
tail-local effect cannot be confused with population-wide closure.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diag_v22_secondary_size_gap import finite_stat, json_clean
from scripts.plot_anchor_bias_scene_tail_curves import frozen_tail_definition


KEY = ["case", "input_index"]
CELLS = ("tail_small", "tail_kept", "outside_small", "outside_kept")


def ratio(values: pd.Series, counts: pd.Series) -> np.ndarray:
    values = values.to_numpy(float)
    counts = counts.to_numpy(float)
    good = counts > 0
    return values[good] / counts[good]


def aggregate(frame: pd.DataFrame, mask: np.ndarray,
              cases: pd.Index) -> pd.DataFrame:
    local = frame.loc[mask, ["case", "residual"]]
    result = local.groupby("case", sort=True).residual.agg(
        ["size", "sum"]
    ).rename(columns={"size": "n", "sum": "residual_sum"})
    return result.reindex(cases, fill_value=0.0)


def summarize_population(table: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(table.n.sum()),
        "n_cases_with_rows": int((table.n > 0).sum()),
        "gap": finite_stat(ratio(table.residual_sum, table.n)),
    }


def paired_population_change(before: pd.DataFrame,
                             after: pd.DataFrame) -> dict[str, Any]:
    good = (before.n > 0) & (after.n > 0)
    delta = (
        after.loc[good, "residual_sum"] / after.loc[good, "n"]
        - before.loc[good, "residual_sum"] / before.loc[good, "n"]
    )
    return finite_stat(delta.to_numpy(float))


def decompose(frame: pd.DataFrame, threshold: float,
              size_cut: float) -> dict[str, Any]:
    """Return tail removal and exact four-cell additive decomposition."""
    required = {
        *KEY, "R_blend_truth", "scene_prediction", "has_deployed_pair",
        "log10_dominant_secondary_size",
    }
    if missing := required - set(frame):
        raise KeyError(f"anchor table lacks {sorted(missing)}")
    if frame.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor key")
    if not np.isfinite(
        frame[["R_blend_truth", "scene_prediction"]].to_numpy(float)
    ).all():
        raise RuntimeError("non-finite anchor response")
    frame = frame.copy()
    frame["residual"] = frame.R_blend_truth - frame.scene_prediction
    paired = frame.has_deployed_pair.to_numpy(bool)
    size = np.power(
        10.0, frame.log10_dominant_secondary_size.to_numpy(float)
    )
    if not np.isfinite(size[paired]).all():
        raise RuntimeError("deployed anchor lacks dominant-secondary size")
    small = paired & (size < size_cut)
    tail = frame.scene_prediction.to_numpy(float) >= threshold
    masks = {
        "tail_small": tail & small,
        "tail_kept": tail & ~small,
        "outside_small": ~tail & small,
        "outside_kept": ~tail & ~small,
    }
    assignment = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.array_equal(assignment, np.ones(len(frame), dtype=np.int8)):
        raise RuntimeError("tail x size cells do not partition anchors")

    cases = pd.Index(sorted(frame.case.unique()), name="case")
    tables = {name: aggregate(frame, mask, cases) for name, mask in masks.items()}
    all_table = aggregate(frame, np.ones(len(frame), dtype=bool), cases)
    tail_table = tables["tail_small"] + tables["tail_kept"]
    tail_after = tables["tail_kept"]
    global_after_tail_small = (
        tables["tail_kept"] + tables["outside_small"] + tables["outside_kept"]
    )
    global_after_all_small = tables["tail_kept"] + tables["outside_kept"]

    all_gap = ratio(all_table.residual_sum, all_table.n)
    tail_gap = ratio(tail_table.residual_sum, tail_table.n)
    cell_result = {}
    for name in CELLS:
        table = tables[name]
        item = summarize_population(table)
        item.update({
            "pooled_fraction_of_all": float(table.n.sum() / all_table.n.sum()),
            "fraction_of_all_by_case": finite_stat(
                (table.n / all_table.n).to_numpy(float), test_zero=False
            ),
            "additive_contribution_to_global_gap": finite_stat(
                (table.residual_sum / all_table.n).to_numpy(float)
            ),
        })
        if name.startswith("tail_"):
            item.update({
                "pooled_fraction_of_tail": float(
                    table.n.sum() / tail_table.n.sum()
                ),
                "fraction_of_tail_by_case": finite_stat(
                    (table.n / tail_table.n).to_numpy(float), test_zero=False
                ),
                "additive_contribution_to_tail_gap": finite_stat(
                    (table.residual_sum / tail_table.n).to_numpy(float)
                ),
            })
        cell_result[name] = item

    cell_global_sum = sum(
        tables[name].residual_sum / all_table.n for name in CELLS
    )
    tail_cell_sum = (
        tables["tail_small"].residual_sum / tail_table.n
        + tables["tail_kept"].residual_sum / tail_table.n
    )
    global_error = cell_global_sum.to_numpy(float) - all_gap
    tail_error = tail_cell_sum.to_numpy(float) - tail_gap
    if not np.allclose(global_error, 0.0, rtol=0, atol=2e-15):
        raise RuntimeError("four cells do not close the global gap")
    if not np.allclose(tail_error, 0.0, rtol=0, atol=2e-15):
        raise RuntimeError("two tail cells do not close the tail gap")

    original_global_mean = float(all_gap.mean())
    original_tail_mean = float(tail_gap.mean())
    return {
        "n_rows": int(len(frame)),
        "n_cases": int(len(cases)),
        "scene_tail_threshold": float(threshold),
        "dominant_secondary_size_cut": float(size_cut),
        "cells": cell_result,
        "tail": {
            "before_removal": summarize_population(tail_table),
            "after_removing_small": summarize_population(tail_after),
            "paired_change_after_minus_before": paired_population_change(
                tail_table, tail_after
            ),
            "fraction_original_gap_remaining": float(
                summarize_population(tail_after)["gap"]["mean"]
                / original_tail_mean
            ),
            "small_share_of_original_tail_gap": float(
                cell_result["tail_small"][
                    "additive_contribution_to_tail_gap"
                ]["mean"] / original_tail_mean
            ),
        },
        "global": {
            "before_removal": summarize_population(all_table),
            "after_removing_tail_small_only": summarize_population(
                global_after_tail_small
            ),
            "paired_change_after_removing_tail_small_only": (
                paired_population_change(all_table, global_after_tail_small)
            ),
            "after_removing_all_small": summarize_population(
                global_after_all_small
            ),
            "paired_change_after_removing_all_small": paired_population_change(
                all_table, global_after_all_small
            ),
            "tail_small_share_of_original_global_gap": float(
                cell_result["tail_small"][
                    "additive_contribution_to_global_gap"
                ]["mean"] / original_global_mean
            ),
            "all_small_share_of_original_global_gap": float(
                (
                    cell_result["tail_small"][
                        "additive_contribution_to_global_gap"
                    ]["mean"]
                    + cell_result["outside_small"][
                        "additive_contribution_to_global_gap"
                    ]["mean"]
                ) / original_global_mean
            ),
        },
        "closure": {
            "maximum_case_global_additive_error": float(
                np.max(np.abs(global_error))
            ),
            "maximum_case_tail_additive_error": float(
                np.max(np.abs(tail_error))
            ),
        },
    }


def markdown(payload: dict[str, Any]) -> str:
    tail = payload["tail"]
    glob = payload["global"]
    lines = [
        "# Panel-E tail x compact-secondary decomposition",
        "",
        "Residual is `R_blend_truth - V2.2`; positive means underprediction. "
        "This uses the same c700--899 coherent anchors as the opened one-"
        "dimensional panels and is post-hoc.",
        "",
        f"- Frozen tail: `scene_prediction >= "
        f"{payload['scene_tail_threshold']:.9g}`.",
        f"- Compact cut: dominant-secondary `Re < "
        f"{payload['dominant_secondary_size_cut']:.3g} arcsec`.",
        "",
        "## Does the cut close the rightmost panel-E point?",
        "",
        f"- Before: `{tail['before_removal']['gap']['mean']:+.6f} +- "
        f"{tail['before_removal']['gap']['case_sem']:.6f}`.",
        f"- After removing compact anchors: "
        f"`{tail['after_removing_small']['gap']['mean']:+.6f} +- "
        f"{tail['after_removing_small']['gap']['case_sem']:.6f}`.",
        f"- Paired change: "
        f"`{tail['paired_change_after_minus_before']['mean']:+.6f} +- "
        f"{tail['paired_change_after_minus_before']['case_sem']:.6f}`.",
        f"- Fraction of tail gap remaining: "
        f"`{tail['fraction_original_gap_remaining']:.1%}`.",
        "",
        "## Four-cell additive accounting on the full denominator",
        "",
        "| cell | fraction of all | conditional gap | case SEM | additive global contribution | share of global gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    global_mean = glob["before_removal"]["gap"]["mean"]
    labels = {
        "tail_small": "tail, Re_s < 0.4",
        "tail_kept": "tail, Re_s >= 0.4",
        "outside_small": "outside tail, Re_s < 0.4",
        "outside_kept": "outside tail, kept",
    }
    for name in CELLS:
        item = payload["cells"][name]
        contribution = item["additive_contribution_to_global_gap"]["mean"]
        lines.append(
            f"| {labels[name]} | {item['pooled_fraction_of_all']:.2%} | "
            f"{item['gap']['mean']:+.6f} | {item['gap']['case_sem']:.6f} | "
            f"{contribution:+.6f} | {contribution / global_mean:.1%} |"
        )
    lines.extend([
        "",
        f"Global before any cut: "
        f"`{glob['before_removal']['gap']['mean']:+.6f} +- "
        f"{glob['before_removal']['gap']['case_sem']:.6f}`.",
        f"After removing only compact anchors in the tail: "
        f"`{glob['after_removing_tail_small_only']['gap']['mean']:+.6f} +- "
        f"{glob['after_removing_tail_small_only']['gap']['case_sem']:.6f}`.",
        f"After removing all compact anchors: "
        f"`{glob['after_removing_all_small']['gap']['mean']:+.6f} +- "
        f"{glob['after_removing_all_small']['gap']['case_sem']:.6f}`.",
        "",
        "The cells add exactly to the full and tail gaps case by case.",
        "",
    ])
    return "\n".join(lines)


def flatten(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name in CELLS:
        item = payload["cells"][name]
        rows.append({
            "cell": name,
            "n_rows": item["n_rows"],
            "pooled_fraction_of_all": item["pooled_fraction_of_all"],
            "conditional_gap": item["gap"]["mean"],
            "conditional_gap_case_sem": item["gap"]["case_sem"],
            "additive_global_contribution": item[
                "additive_contribution_to_global_gap"
            ]["mean"],
            "additive_global_contribution_case_sem": item[
                "additive_contribution_to_global_gap"
            ]["case_sem"],
            "pooled_fraction_of_tail": item.get("pooled_fraction_of_tail"),
            "additive_tail_contribution": item.get(
                "additive_contribution_to_tail_gap", {}
            ).get("mean"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--parent-curves", required=True)
    ap.add_argument("--case-min", type=int, default=700)
    ap.add_argument("--case-max", type=int, default=899)
    ap.add_argument("--size-cut", type=float, default=0.4)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()
    outputs = [f"{args.output_prefix}.{suffix}" for suffix in ("json", "md", "csv")]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    curves = pd.read_csv(args.parent_curves)
    tail_definition = frozen_tail_definition(curves)
    columns = [
        *KEY, "R_blend_truth", "scene_prediction", "has_deployed_pair",
        "log10_dominant_secondary_size",
    ]
    frame = pd.read_feather(args.features, columns=columns)
    frame = frame.loc[frame.case.between(args.case_min, args.case_max)].copy()
    result = decompose(frame, tail_definition["threshold"], args.size_cut)
    result.update({
        "feature_table": os.path.abspath(args.features),
        "parent_curves": os.path.abspath(args.parent_curves),
        "case_window": [args.case_min, args.case_max],
        "tail_definition": tail_definition,
        "target": "R_blend_truth - R_blend_lsst_r_extnbr_v22",
        "positive_target_meaning": "V2.2 underprediction",
        "uncertainty_unit": "rendered simulation case",
        "post_hoc": True,
        "post_hoc_reason": (
            "the 0.4 arcsec cut was proposed after opening this same final-test "
            "tail's one-dimensional secondary-size curve"
        ),
    })
    observed_tail = result["tail"]["before_removal"]
    parent = curves.loc[
        (curves.population == "all")
        & (curves.feature == "scene_prediction")
    ].sort_values("bin").iloc[-1]
    checks = {
        "tail_rows_match_parent": (
            observed_tail["n_rows"] == int(parent.n_rows)
        ),
        "tail_mean_matches_parent": np.isclose(
            observed_tail["gap"]["mean"], float(parent.target_mean),
            rtol=0, atol=2e-12,
        ),
        "tail_sem_matches_parent": np.isclose(
            observed_tail["gap"]["case_sem"], float(parent.target_case_sem),
            rtol=0, atol=2e-12,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"tail does not reproduce parent panel: {checks}")
    result["parent_panel_reproduction"] = {
        key: bool(value) for key, value in checks.items()
    }
    result = json_clean(result)
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_prefix}.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    Path(f"{args.output_prefix}.md").write_text(
        markdown(result), encoding="utf-8"
    )
    flatten(result).to_csv(f"{args.output_prefix}.csv", index=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print("ANCHOR_TAIL_SECONDARY_SIZE_DECOMPOSITION_DONE", flush=True)


if __name__ == "__main__":
    main()
