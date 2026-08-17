"""Decompose the shared V2.2 carrier on exact constgold/half-shear keys.

The full constgold deficit mixes two terms on every matched object::

    total = (R_sim - R_self - R_blend) + (R_self - R_flow)

The first parenthesis is the neighbour/estimand proxy used by the shared-gap
localization; the second is the directly measured self-minus-flow term.  This
script reports both on exactly the same rows and fixed carrier rules, so a
change in population cannot masquerade as cancellation between the terms.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]
RULES = {
    "pilot_frozen_ratio_gt5_positive": (
        "dominant_to_runner_up_abs_response > 5 and dominant_response >= 0"
    ),
    "selected_top_fraction_gt0p7_positive": (
        "top_abs_fraction > 0.7 and dominant_response >= 0"
    ),
    "core_ratio_gt10_positive": (
        "dominant_to_runner_up_abs_response > 10 and dominant_response >= 0"
    ),
    "narrow_ratio_gt20": "dominant_to_runner_up_abs_response > 20",
}


def basic_stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("expected at least two finite case values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": sd / np.sqrt(len(values)),
        "n_cases": int(len(values)),
    }


def summarize(frame: pd.DataFrame, selected: np.ndarray, column: str) -> dict:
    """Case-blocked conditional means and population-weighted contributions."""
    work = frame[["case", column]].copy()
    work["selected"] = np.asarray(selected, bool)
    if work.selected.all() or (~work.selected).all():
        raise ValueError("carrier rule must leave non-empty selected/complement sets")
    by_case = work.groupby("case", sort=True)
    global_case = by_case[column].mean()
    selected_case = work.loc[work.selected].groupby("case", sort=True)[column].mean()
    complement_case = work.loc[~work.selected].groupby("case", sort=True)[column].mean()
    contribution = work.assign(
        selected_value=np.where(work.selected, work[column], 0.0),
        complement_value=np.where(~work.selected, work[column], 0.0),
    ).groupby("case", sort=True).agg(
        selected_fraction=("selected", "mean"),
        selected_contribution=("selected_value", "mean"),
        complement_contribution=("complement_value", "mean"),
    )
    cases = global_case.index
    for values in (selected_case, complement_case, contribution):
        if not values.index.equals(cases):
            raise RuntimeError("a carrier group is absent from at least one case")
    global_mean = float(global_case.mean())
    selected_mean = float(contribution.selected_contribution.mean())
    return {
        "global": basic_stat(global_case.to_numpy(float)),
        "selected_fraction": basic_stat(
            contribution.selected_fraction.to_numpy(float)
        ),
        "selected_conditional": basic_stat(selected_case.to_numpy(float)),
        "complement_conditional": basic_stat(complement_case.to_numpy(float)),
        "selected_contribution": basic_stat(
            contribution.selected_contribution.to_numpy(float)
        ),
        "complement_contribution": basic_stat(
            contribution.complement_contribution.to_numpy(float)
        ),
        "selected_carrier_share": (
            selected_mean / global_mean if global_mean != 0 else None
        ),
    }


def rule_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ratio = frame.dominant_to_runner_up_abs_response.to_numpy(float)
    positive = frame.dominant_response.to_numpy(float) >= 0
    top = frame.top_abs_fraction.to_numpy(float)
    return {
        "pilot_frozen_ratio_gt5_positive": (ratio > 5) & positive,
        "selected_top_fraction_gt0p7_positive": (top > 0.7) & positive,
        "core_ratio_gt10_positive": (ratio > 10) & positive,
        "narrow_ratio_gt20": ratio > 20,
    }


def markdown(payload: dict) -> str:
    lines = [
        "# Exact-key decomposition of the shared V2.2 carrier",
        "",
        "All three terms use the same constgold/half-shear objects. Positive means "
        "V2.2 underpredicts. `Neighbour proxy + self/flow = full total` per row.",
        "",
    ]
    for block_name in ("validation", "development"):
        block = payload["blocks"][block_name]
        lines.extend([
            f"## {block_name.title()} cases {block['case_window'][0]}–{block['case_window'][1]}",
            "",
            "| Fixed rule | Term | Fraction | Selected conditional | Rest conditional | Selected contribution |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for rule_name, result in block["rules"].items():
            for term in ("neighbour_proxy", "self_flow", "total"):
                item = result[term]
                lines.append(
                    f"| `{rule_name}` | {term} | "
                    f"{item['selected_fraction']['mean']:.1%} | "
                    f"{item['selected_conditional']['mean']:+.5f} ± "
                    f"{item['selected_conditional']['case_sem']:.5f} | "
                    f"{item['complement_conditional']['mean']:+.5f} ± "
                    f"{item['complement_conditional']['case_sem']:.5f} | "
                    f"{item['selected_contribution']['mean']:+.5f} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--constgold-gap", required=True)
    ap.add_argument("--constgold-pairs", required=True)
    ap.add_argument("--half-selfresp", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    for output in (args.output_json, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    gap = pd.read_feather(
        args.constgold_gap, columns=[*KEY, "R_sim", "R_blend", "gap"]
    )
    pairs = pd.read_feather(args.constgold_pairs, columns=[
        *KEY, "dominant_response", "top_abs_fraction",
        "dominant_to_runner_up_abs_response",
    ])
    half = pd.read_feather(args.half_selfresp, columns=[*KEY, "r_sim_self"])
    half = half.loc[np.isfinite(half.r_sim_self)].copy()
    for name, frame in (("gap", gap), ("pairs", pairs), ("half", half)):
        if frame.duplicated(KEY).any():
            raise RuntimeError(f"duplicate {name} keys")
    frame = gap.merge(pairs, on=KEY, validate="one_to_one").merge(
        half, on=KEY, validate="one_to_one"
    )
    frame["neighbour_proxy"] = (
        frame.R_sim - frame.r_sim_self - frame.R_blend
    )
    frame["total"] = frame.gap
    frame["self_flow"] = frame.total - frame.neighbour_proxy
    closure = (
        frame.neighbour_proxy + frame.self_flow - frame.total
    ).to_numpy(float)
    maximum_closure = float(np.max(np.abs(closure)))
    if maximum_closure > 1e-12:
        raise RuntimeError(f"per-row decomposition fails: {maximum_closure:.3e}")
    blocks = {}
    for block_name, low, high in (
        ("development", 40, 89), ("validation", 90, 139)
    ):
        local = frame.loc[frame.case.between(low, high)].copy()
        local_masks = rule_masks(local)
        blocks[block_name] = {
            "case_window": [low, high],
            "n_rows": int(len(local)),
            "rules": {
                rule_name: {
                    term: summarize(local, local_masks[rule_name], term)
                    for term in ("neighbour_proxy", "self_flow", "total")
                }
                for rule_name in RULES
            },
        }
    payload = {
        "design": (
            "exact-key constgold/half-shear decomposition; fixed rules only; "
            "case is uncertainty unit; positive means model underprediction"
        ),
        "identity": "total = neighbour_proxy + self_flow",
        "definitions": {
            "neighbour_proxy": "R_sim - r_sim_self - R_blend",
            "self_flow": "r_sim_self - mean_seed_R_flow = total - neighbour_proxy",
            "total": "R_sim - mean_seed_R_flow - R_blend",
        },
        "rules": RULES,
        "n_rows": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "max_abs_row_closure": maximum_closure,
        "blocks": blocks,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown(payload))
    print(markdown(payload), flush=True)
    print("V22_SHARED_CARRIER_DECOMPOSITION_DONE", flush=True)


if __name__ == "__main__":
    main()
