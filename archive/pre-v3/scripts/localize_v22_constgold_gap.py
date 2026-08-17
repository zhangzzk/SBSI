"""Localize the frozen V2.2 constgold response deficit without outcome leakage.

Cases 40--89 define feature quintiles and select one worst bin per feature.
Cases 90--139 validate those frozen bins.  The rendered case is always the
uncertainty unit.  Binning coordinates use only true primary properties and
frozen model/pair predictions; noisy per-object ``R_sim`` is never a feature.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy import stats

from scripts.eval_v2_indomain_m import catalogue_true_props
from scripts.localize_anchorblend_coherent_gap import holm_adjust, quantile_edges, stat


KEY = ["case", "input_index"]
FEATURES = [
    "primary_mag",
    "primary_size",
    "R_flow",
    "R_blend",
    "R_model",
    "blend_fraction_of_model",
    "n_pairs",
    "dominant_abs_response",
    "runner_up_abs_response",
    "R_abs_sum",
    "cancelled_abs_response",
    "signed_to_abs_response",
    "top_abs_fraction",
    "dominant_to_runner_up_abs_response",
]


def key64(case: np.ndarray, primary: np.ndarray) -> np.ndarray:
    case = np.asarray(case, dtype=np.int64)
    primary = np.asarray(primary, dtype=np.int64)
    if len(primary) and (np.min(primary) < 0 or np.max(primary) >= (1 << 40)):
        raise RuntimeError("input_index exceeds packed-key allocation")
    return (case << 40) | primary


def selected_vs_complement(frame: pd.DataFrame, selected: np.ndarray) -> dict:
    """Case-paired conditional deficit contrast for a frozen boolean selection."""
    work = frame[["case", "gap", "R_sim", "R_flow", "R_blend", "R_model"]].copy()
    work["selected"] = np.asarray(selected, dtype=bool)
    rows = []
    for (case, in_bin), local in work.groupby(["case", "selected"], sort=True):
        rows.append({
            "case": int(case), "selected": bool(in_bin), "n": int(len(local)),
            "gap": float(local.gap.mean()), "R_sim": float(local.R_sim.mean()),
            "R_flow": float(local.R_flow.mean()), "R_blend": float(local.R_blend.mean()),
            "R_model": float(local.R_model.mean()),
        })
    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index="case", columns="selected")
    if False not in pivot.columns.levels[1] or True not in pivot.columns.levels[1]:
        raise RuntimeError("a selected-bin contrast is empty in at least one block")
    difference = pivot["gap"][True].to_numpy(float) - pivot["gap"][False].to_numpy(float)
    ttest = stats.ttest_1samp(difference, popmean=0.0)
    try:
        wilcoxon = stats.wilcoxon(difference, alternative="two-sided")
        wilcoxon_stat = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    normality = stats.shapiro(difference)

    def conditional(which: bool) -> dict:
        local = summary.loc[summary.selected == which].sort_values("case")
        m = 100.0 * (local.R_sim.to_numpy(float) / local.R_model.to_numpy(float) - 1.0)
        return {
            "n_rows": int(local.n.sum()),
            "mean_rows_per_case": float(local.n.mean()),
            "gap": stat(local.gap.to_numpy(float)),
            "m_percent": stat(m),
            "R_sim": stat(local.R_sim.to_numpy(float)),
            "R_flow": stat(local.R_flow.to_numpy(float)),
            "R_blend": stat(local.R_blend.to_numpy(float)),
            "R_model": stat(local.R_model.to_numpy(float)),
        }

    all_cases = np.sort(frame.case.unique())
    contribution = frame[["case", "gap"]].copy()
    contribution["selected_gap"] = np.where(selected, contribution.gap, 0.0)
    contribution["selected"] = np.asarray(selected, dtype=float)
    by_case = contribution.groupby("case", sort=True).agg(
        contribution=("selected_gap", "mean"), fraction=("selected", "mean"),
    ).reindex(all_cases, fill_value=0.0)
    return {
        "selected": conditional(True),
        "complement": conditional(False),
        "selected_minus_complement_gap": stat(difference),
        "paired_t": float(ttest.statistic),
        "paired_p_raw": float(ttest.pvalue),
        "paired_wilcoxon_statistic": wilcoxon_stat,
        "paired_wilcoxon_p_raw": wilcoxon_p,
        "contrast_shapiro_w": float(normality.statistic),
        "contrast_shapiro_p": float(normality.pvalue),
        "selected_fraction": stat(by_case.fraction.to_numpy(float)),
        "selected_contribution_to_global_gap": stat(
            by_case.contribution.to_numpy(float)
        ),
    }


def summarize_bins(frame: pd.DataFrame, feature: str, edges: np.ndarray) -> dict:
    coordinate = frame[feature].to_numpy(float)
    bins = np.searchsorted(edges, coordinate, side="right") - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    rows = []
    for index in range(len(edges) - 1):
        selected = bins == index
        local = frame.loc[selected]
        by_case = local.groupby("case", sort=True).agg(
            gap=("gap", "mean"), R_sim=("R_sim", "mean"),
            R_flow=("R_flow", "mean"), R_blend=("R_blend", "mean"),
            R_model=("R_model", "mean"), n=("gap", "size"),
        )
        m = 100.0 * (by_case.R_sim.to_numpy(float) / by_case.R_model.to_numpy(float) - 1.0)
        contribution = frame[["case", "gap"]].copy()
        contribution["selected_gap"] = np.where(selected, contribution.gap, 0.0)
        contribution["selected"] = selected.astype(float)
        by_case_contribution = contribution.groupby("case", sort=True).agg(
            contribution=("selected_gap", "mean"), fraction=("selected", "mean"),
        )
        rows.append({
            "bin": int(index),
            "lo": None if not np.isfinite(edges[index]) else float(edges[index]),
            "hi": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
            "n_rows": int(selected.sum()),
            "conditional_gap": stat(by_case.gap.to_numpy(float)),
            "conditional_m_percent": stat(m),
            "conditional_R_sim": stat(by_case.R_sim.to_numpy(float)),
            "conditional_R_flow": stat(by_case.R_flow.to_numpy(float)),
            "conditional_R_blend": stat(by_case.R_blend.to_numpy(float)),
            "conditional_R_model": stat(by_case.R_model.to_numpy(float)),
            "row_fraction": stat(by_case_contribution.fraction.to_numpy(float)),
            "contribution_to_global_gap": stat(
                by_case_contribution.contribution.to_numpy(float)
            ),
        })
    return {"bins": rows, "assignment": bins}


def categorical_distance_summary(frame: pd.DataFrame) -> dict:
    neighbored = frame.neighbored.to_numpy(bool)
    distance = frame.distance.to_numpy(float)
    masks = {
        "not_flagged": ~neighbored,
        "flagged_0_1_arcsec": neighbored & (distance >= 0.0) & (distance < 1.0),
        "flagged_1_2_arcsec": neighbored & (distance >= 1.0) & (distance < 2.0),
        "flagged_2_3p01_arcsec": neighbored & (distance >= 2.0) & (distance < 3.01),
    }
    output = {}
    for name, mask in masks.items():
        output[name] = selected_vs_complement(frame, mask)
    return output


def markdown_report(payload: dict) -> str:
    development = payload["development_global"]
    validation = payload["validation_global"]
    lines = [
        "# V2.2 constgold response-gap localization",
        "",
        "Cases 40–89 define quintile edges and select the highest-deficit bin for each "
        "feature. Cases 90–139 validate those frozen bins. The rendered case is the "
        "uncertainty unit. Per-object simulation response is excluded from the feature set.",
        "",
        f"Development global `R_sim - R_model`: `{development['gap']['mean']:+.6f} "
        f"+- {development['gap']['case_sem']:.6f}`; validation: "
        f"`{validation['gap']['mean']:+.6f} +- {validation['gap']['case_sem']:.6f}`.",
        "",
        "| Feature | Frozen bin | Validation fraction | Validation gap | Bin−rest | Holm p | Contribution |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in FEATURES:
        item = payload["features"][feature]
        validation_item = item["validation_selected_bin"]
        selected = validation_item["selected"]
        contrast = validation_item["selected_minus_complement_gap"]
        contribution = validation_item["selected_contribution_to_global_gap"]
        fraction = validation_item["selected_fraction"]
        lines.append(
            f"| `{feature}` | {item['development_selected_bin']} | "
            f"{fraction['mean']:.1%} | {selected['gap']['mean']:+.5f} | "
            f"{contrast['mean']:+.5f} ± {contrast['case_sem']:.5f} | "
            f"{validation_item['paired_p_holm']:.3g} | "
            f"{contribution['mean']:+.5f} |"
        )
    lines.extend([
        "", "## Reading this table", "",
        "A high conditional gap identifies a population where V2.2 underpredicts the "
        "absolute response. The contribution additionally accounts for how common that "
        "population is. Correlated features may describe the same objects; this is localization, "
        "not a causal attribution or correction.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--dominance-lookup", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--development-max", type=int, default=89)
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--output-feather", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    for output in (args.output_feather, args.output_json, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(output)

    dumps = sorted(glob.glob(args.dump_glob))
    if len(dumps) != 16:
        raise SystemExit(f"localization requires exactly 16 V2.2 dumps; found {len(dumps)}")
    started = time.time()
    truth = catalogue_true_props(args.catalogue, args.min_case, started)
    reference = pf.read_table(
        dumps[0], columns=["case", "input_index", "r_sim", "R_flow", "R_blend",
                           "neighbored", "distance"],
    ).to_pandas()
    if not truth[KEY].equals(reference[KEY]):
        raise RuntimeError("catalogue/dump key alignment failure")
    magnitude = truth.r_input_p.to_numpy(float)
    size = truth.Re_input_p.to_numpy(float)
    domain = (magnitude < args.mag_max) & (size > args.re_min)

    r_sim = reference.r_sim.to_numpy(float)
    r_blend = reference.R_blend.to_numpy(float)
    r_flow_sum = np.zeros(len(reference), dtype=float)
    for path in dumps:
        table = pf.read_table(
            path, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"],
        )
        if not np.array_equal(table["case"].to_numpy(), reference.case.to_numpy()) \
                or not np.array_equal(
                    table["input_index"].to_numpy(), reference.input_index.to_numpy()
                ):
            raise RuntimeError(f"dump key alignment failure: {path}")
        if not np.array_equal(table["r_sim"].to_numpy(), r_sim):
            raise RuntimeError(f"R_sim differs across seeds: {path}")
        if not np.array_equal(table["R_blend"].to_numpy(), r_blend):
            raise RuntimeError(f"R_blend differs across seeds: {path}")
        r_flow_sum += table["R_flow"].to_numpy(zero_copy_only=False).astype(float)
    r_flow = r_flow_sum / len(dumps)

    dominance = pf.read_table(args.dominance_lookup).to_pandas()
    dominance_key = key64(dominance.case.to_numpy(), dominance.input_index.to_numpy())
    order = np.argsort(dominance_key)
    dominance_key = dominance_key[order]
    if len(dominance_key) != len(np.unique(dominance_key)):
        raise RuntimeError("dominance lookup has duplicate keys")
    truth_key = key64(reference.case.to_numpy(), reference.input_index.to_numpy())
    position = np.searchsorted(dominance_key, truth_key)
    clipped = np.clip(position, 0, len(dominance_key) - 1)
    matched = dominance_key[clipped] == truth_key
    mask = domain & matched
    if float(np.mean(matched[domain])) < 0.999999:
        raise RuntimeError("dominance lookup does not cover V2.2 domain")
    matched_dominance = dominance.iloc[order[clipped[mask]]].reset_index(drop=True)
    if not np.array_equal(
        matched_dominance.case.to_numpy(np.int64), reference.case.to_numpy(np.int64)[mask]
    ) or not np.array_equal(
        matched_dominance.input_index.to_numpy(np.int64),
        reference.input_index.to_numpy(np.int64)[mask],
    ):
        raise RuntimeError("dominance lookup join did not preserve keys")
    if not np.allclose(
        matched_dominance.R_blend.to_numpy(float), r_blend[mask],
        rtol=0.0, atol=1.0e-6,
    ):
        raise RuntimeError("dominance replay R_blend differs from frozen dumps")

    frame = pd.DataFrame({
        "case": reference.case.to_numpy(np.int64)[mask],
        "input_index": reference.input_index.to_numpy(np.int64)[mask],
        "primary_mag": magnitude[mask], "primary_size": size[mask],
        "R_sim": r_sim[mask], "R_flow": r_flow[mask], "R_blend": r_blend[mask],
        "neighbored": reference.neighbored.to_numpy(bool)[mask],
        "distance": reference.distance.to_numpy(float)[mask],
    })
    for column in (
        "n_pairs", "dominant_abs_response", "runner_up_abs_response", "R_abs_sum",
        "top_abs_fraction", "dominant_to_runner_up_abs_response",
    ):
        frame[column] = matched_dominance[column].to_numpy()
    frame["R_model"] = frame.R_flow + frame.R_blend
    frame["gap"] = frame.R_sim - frame.R_model
    frame["blend_fraction_of_model"] = np.divide(
        frame.R_blend, frame.R_model,
        out=np.zeros(len(frame), dtype=float),
        where=np.abs(frame.R_model.to_numpy(float)) > 1.0e-6,
    )
    frame["cancelled_abs_response"] = frame.R_abs_sum - np.abs(frame.R_blend)
    frame["signed_to_abs_response"] = np.divide(
        frame.R_blend, frame.R_abs_sum,
        out=np.zeros(len(frame), dtype=float),
        where=frame.R_abs_sum.to_numpy(float) > 0.0,
    )
    required = [*FEATURES, "gap", "R_sim", "R_flow", "R_blend", "R_model"]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise RuntimeError("non-finite localization feature or response")
    frame.to_feather(args.output_feather)

    development = frame.loc[frame.case <= args.development_max].copy()
    validation = frame.loc[frame.case > args.development_max].copy()
    if development.case.nunique() != 50 or validation.case.nunique() != 50:
        raise RuntimeError("expected two 50-case blocks")

    def global_summary(block: pd.DataFrame) -> dict:
        by_case = block.groupby("case", sort=True).agg(
            gap=("gap", "mean"), R_sim=("R_sim", "mean"),
            R_flow=("R_flow", "mean"), R_blend=("R_blend", "mean"),
            R_model=("R_model", "mean"),
        )
        m = 100.0 * (by_case.R_sim.to_numpy(float) / by_case.R_model.to_numpy(float) - 1.0)
        return {
            "n_rows": int(len(block)), "n_cases": int(block.case.nunique()),
            "gap": stat(by_case.gap.to_numpy(float)), "m_percent": stat(m),
            "R_sim": stat(by_case.R_sim.to_numpy(float)),
            "R_flow": stat(by_case.R_flow.to_numpy(float)),
            "R_blend": stat(by_case.R_blend.to_numpy(float)),
            "R_model": stat(by_case.R_model.to_numpy(float)),
        }

    payload = {
        "design": (
            "c40--89 freeze quintile edges and select one highest-deficit bin per feature; "
            "c90--139 validate; case is uncertainty unit; R_sim excluded from features"
        ),
        "features_fixed_before_response_scan": FEATURES,
        "development_window": [40, args.development_max],
        "validation_window": [args.development_max + 1, 139],
        "domain": {"primary_mag_max": args.mag_max, "primary_size_min": args.re_min},
        "lookup_match_fraction_domain": float(np.mean(matched[domain])),
        "development_global": global_summary(development),
        "validation_global": global_summary(validation),
        "features": {},
        "categorical_nearest_annotated_distance": {
            "development": categorical_distance_summary(development),
            "validation": categorical_distance_summary(validation),
        },
    }
    raw_p = {}
    for feature in FEATURES:
        edges = quantile_edges(development[feature].to_numpy(float))
        dev_bins = summarize_bins(development, feature, edges)
        # Freeze the bin with the largest conditional absolute deficit in development.
        selected_bin = int(np.argmax([
            row["conditional_gap"]["mean"] for row in dev_bins["bins"]
        ]))
        dev_selected = selected_vs_complement(
            development, dev_bins["assignment"] == selected_bin,
        )
        validation_bins = summarize_bins(validation, feature, edges)
        validation_selected = selected_vs_complement(
            validation, validation_bins["assignment"] == selected_bin,
        )
        raw_p[feature] = validation_selected["paired_p_raw"]
        payload["features"][feature] = {
            "edges": [None if not np.isfinite(x) else float(x) for x in edges],
            "development_selected_bin": selected_bin,
            "development_bins": dev_bins["bins"],
            "development_selected_bin_summary": dev_selected,
            "validation_bins": validation_bins["bins"],
            "validation_selected_bin": validation_selected,
        }
    adjusted = holm_adjust(raw_p)
    for feature, value in adjusted.items():
        payload["features"][feature]["validation_selected_bin"]["paired_p_holm"] = value

    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown_report(payload))
        handle.write("\n")
    print(json.dumps({
        "development_global": payload["development_global"],
        "validation_global": payload["validation_global"],
        "validation_selected_bins": {
            feature: {
                "bin": payload["features"][feature]["development_selected_bin"],
                "summary": payload["features"][feature]["validation_selected_bin"],
            } for feature in FEATURES
        },
    }, indent=2, sort_keys=True))
    print("V22_CONSTGOLD_GAP_LOCALIZATION_DONE", flush=True)


if __name__ == "__main__":
    main()
