"""Localize the V2.2 coherent-anchor gap in frozen scene coordinates.

Cases 400--449 define quintile edges.  Cases 450--499 are the validation half.
The rendered case is the uncertainty unit.  This is a descriptive localization:
features are correlated, no correction is fitted, and constgold is never opened.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEY = ["case", "input_index"]
FEATURES = [
    "R_model_sum",
    "R_positive_sum",
    "R_negative_sum",
    "R_abs_sum",
    "R_power",
    "R_max_abs_pair",
    "top_abs_fraction",
    "n_pairs",
    "R_abs_near_0_2",
    "R_abs_mid_2_5",
    "R_abs_far_5_10",
    "primary_mag",
    "primary_size",
]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("stat requires at least two finite one-dimensional values")
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "case_sd": float(values.std(ddof=1)),
        "n_cases": int(len(values)),
    }


def aggregate_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return per-anchor summaries of the frozen deployed V2.2 pair list."""
    required = {"anchor_index", "response", "distance"}
    if missing := required - set(pairs):
        raise KeyError(f"pair table lacks {sorted(missing)}")
    if pairs.duplicated(["anchor_index", "secondary_index"]).any():
        raise RuntimeError("duplicate deployed pair")
    work = pairs[["anchor_index", "response", "distance"]].copy()
    response = work["response"].to_numpy(float)
    distance = work["distance"].to_numpy(float)
    work["positive"] = np.clip(response, 0.0, None)
    work["negative"] = np.clip(response, None, 0.0)
    work["absolute"] = np.abs(response)
    work["power"] = response * response
    work["near"] = np.where(distance < 2.0, np.abs(response), 0.0)
    work["mid"] = np.where((distance >= 2.0) & (distance < 5.0), np.abs(response), 0.0)
    work["far"] = np.where(distance >= 5.0, np.abs(response), 0.0)
    grouped = work.groupby("anchor_index", sort=False)
    out = grouped.agg(
        pair_sum=("response", "sum"),
        R_positive_sum=("positive", "sum"),
        R_negative_sum=("negative", "sum"),
        R_abs_sum=("absolute", "sum"),
        R_power=("power", "sum"),
        R_max_abs_pair=("absolute", "max"),
        n_pairs=("response", "size"),
        R_abs_near_0_2=("near", "sum"),
        R_abs_mid_2_5=("mid", "sum"),
        R_abs_far_5_10=("far", "sum"),
    )
    out["R_power"] = np.sqrt(out["R_power"])
    out["top_abs_fraction"] = np.divide(
        out["R_max_abs_pair"], out["R_abs_sum"],
        out=np.zeros(len(out), dtype=float), where=out["R_abs_sum"].to_numpy() > 0,
    )
    return out


def quantile_edges(values: np.ndarray, nbin: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, nbin + 1)))
    if len(edges) < 4:
        raise ValueError("feature has too few unique quantile edges")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def summarize_bins(frame: pd.DataFrame, feature: str, edges: np.ndarray) -> dict:
    bins = np.searchsorted(edges, frame[feature].to_numpy(float), side="right") - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    work = frame[["case", "gap", "R_model_sum", "R_coherent_truth"]].copy()
    work["bin"] = bins
    all_cases = np.sort(work.case.unique())
    rows = []
    conditional_by_bin: dict[int, pd.Series] = {}
    for index in range(len(edges) - 1):
        selected = work.bin.to_numpy() == index
        local = work.loc[selected]
        conditional = local.groupby("case", sort=True)[
            ["gap", "R_model_sum", "R_coherent_truth"]
        ].mean()
        conditional_by_bin[index] = conditional["gap"]
        contribution_frame = work[["case"]].copy()
        contribution_frame["selected_gap"] = np.where(selected, work.gap, 0.0)
        contribution_frame["selected"] = selected.astype(float)
        contribution = contribution_frame.groupby("case", sort=True).agg(
            contribution=("selected_gap", "mean"), fraction=("selected", "mean"),
        ).reindex(all_cases, fill_value=0.0)
        rows.append({
            "bin": int(index),
            "lo": None if not np.isfinite(edges[index]) else float(edges[index]),
            "hi": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
            "n_rows": int(selected.sum()),
            "row_fraction": stat(contribution.fraction.to_numpy(float)),
            "conditional_gap": stat(conditional.gap.to_numpy(float)),
            "contribution_to_global_gap": stat(contribution.contribution.to_numpy(float)),
            "conditional_model": stat(conditional.R_model_sum.to_numpy(float)),
            "conditional_truth": stat(conditional.R_coherent_truth.to_numpy(float)),
        })
    low = conditional_by_bin[0]
    high = conditional_by_bin[len(edges) - 2]
    paired = high.to_frame("high").join(low.to_frame("low"), how="inner")
    contrast = paired.high.to_numpy(float) - paired.low.to_numpy(float)
    test = stats.ttest_1samp(contrast, popmean=0.0)
    return {
        "bins": rows,
        "high_minus_low_gap": stat(contrast),
        "high_minus_low_t": float(test.statistic),
        "high_minus_low_p_raw": float(test.pvalue),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def markdown_report(payload: dict) -> str:
    global_val = payload["validation_global_gap"]
    lines = [
        "# V2.2 coherent-gap localization, cases 400–499",
        "",
        "Cases 400–449 define all quintile edges; cases 450–499 are the validation half. "
        "Uncertainties use case means. Features are correlated and the tables are descriptive, "
        "not a causal decomposition or a fitted correction.",
        "",
        f"Validation global emulator − coherent gap: `{global_val['mean']:+.6f} "
        f"+- {global_val['case_sem']:.6f}`.",
        "",
        "| Feature | Validation high−low gap | SEM | Holm p | Most negative bin contribution |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature in FEATURES:
        item = payload["features"][feature]["validation"]
        contrast = item["high_minus_low_gap"]
        worst = min(row["contribution_to_global_gap"]["mean"] for row in item["bins"])
        lines.append(
            f"| `{feature}` | {contrast['mean']:+.5f} | {contrast['case_sem']:.5f} | "
            f"{item['high_minus_low_p_holm']:.3g} | {worst:+.5f} |"
        )
    lines.extend(["", "## Validation-bin details", ""])
    for feature in FEATURES:
        lines.extend([
            f"### `{feature}`", "",
            "| Bin | Range | Rows | Conditional gap | Contribution to global gap |",
            "|---:|---|---:|---:|---:|",
        ])
        for row in payload["features"][feature]["validation"]["bins"]:
            lo = "−∞" if row["lo"] is None else f"{row['lo']:.5g}"
            hi = "+∞" if row["hi"] is None else f"{row['hi']:.5g}"
            gap = row["conditional_gap"]
            contribution = row["contribution_to_global_gap"]
            lines.append(
                f"| {row['bin']} | [{lo}, {hi}) | {row['n_rows']:,} | "
                f"{gap['mean']:+.5f} ± {gap['case_sem']:.5f} | "
                f"{contribution['mean']:+.5f} ± {contribution['case_sem']:.5f} |"
            )
        lines.append("")
    lines.extend([
        "## Limitations", "",
        "- Pair-derived coordinates come from the emulator's own deployed pair list.",
        "- One-dimensional bins do not identify which correlated feature is causal.",
        "- Detection and matching coordinates are deliberately excluded because they are post-shear.",
        "- Constgold was not opened.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--case-min", type=int, default=400)
    ap.add_argument("--case-max", type=int, default=499)
    ap.add_argument("--development-max", type=int, default=449)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    for output in (args.output_json, args.output_csv, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(output)
    response = pd.read_feather(args.response, columns=[
        "case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ])
    response = response.loc[response.case.between(args.case_min, args.case_max)].copy()
    response = response.rename(columns={
        "R_blend_truth": "R_coherent_truth",
        "R_blend_lsst_r_extnbr_v22": "R_model_sum",
    })
    if response.duplicated(KEY).any():
        raise RuntimeError("duplicate coherent response key")

    parts = []
    replay_audit = []
    root = Path(args.manifest_dir)
    for case in range(args.case_min, args.case_max + 1):
        anchors = pd.read_feather(root / f"anchors_case{case}.feather")
        pairs = pd.read_feather(root / f"pairs_case{case}.feather")
        features = aggregate_pairs(pairs)
        local = anchors[["index", "r", "Re"]].rename(columns={
            "index": "input_index", "r": "primary_mag", "Re": "primary_size",
        })
        local = local.merge(
            features, left_on="input_index", right_index=True,
            how="left", validate="one_to_one",
        )
        local.insert(0, "case", int(case))
        local = local.merge(
            response.loc[response.case == case], on=KEY, how="inner", validate="one_to_one",
        )
        replay = local.pair_sum - local.R_model_sum
        replay_audit.append({
            "case": int(case), "n_rows": int(len(local)),
            "max_abs": float(np.max(np.abs(replay))), "mean": float(replay.mean()),
        })
        if float(np.max(np.abs(replay))) > 1e-2 or abs(float(replay.mean())) > 1e-5:
            raise RuntimeError(f"case {case}: pair prediction replay is material")
        local["gap"] = local.R_model_sum - local.R_coherent_truth
        if not np.isfinite(local[[*FEATURES, "gap"]].to_numpy(float)).all():
            raise RuntimeError(f"case {case}: non-finite localization row")
        parts.append(local)
    frame = pd.concat(parts, ignore_index=True)
    development = frame.loc[frame.case <= args.development_max].copy()
    validation = frame.loc[frame.case > args.development_max].copy()
    if development.case.nunique() != 50 or validation.case.nunique() != 50:
        raise RuntimeError("expected two 50-case blocks")

    payload = {
        "design": "quintile edges frozen on c400--449; localization validated on c450--499; rendered case is uncertainty unit",
        "case_window": [args.case_min, args.case_max],
        "development_window": [args.case_min, args.development_max],
        "validation_window": [args.development_max + 1, args.case_max],
        "n_rows": int(len(frame)),
        "n_manifest_pairs": int(sum(
            pd.read_feather(root / f"pairs_case{case}.feather", columns=["response"]).shape[0]
            for case in range(args.case_min, args.case_max + 1)
        )),
        "development_global_gap": stat(
            development.groupby("case", sort=True).gap.mean().to_numpy(float)
        ),
        "validation_global_gap": stat(
            validation.groupby("case", sort=True).gap.mean().to_numpy(float)
        ),
        "features": {}, "prediction_replay_audit": replay_audit,
        "constgold_opened": False,
    }
    raw_p = {}
    csv_rows = []
    for feature in FEATURES:
        edges = quantile_edges(development[feature].to_numpy(float))
        dev = summarize_bins(development, feature, edges)
        val = summarize_bins(validation, feature, edges)
        raw_p[feature] = val["high_minus_low_p_raw"]
        payload["features"][feature] = {
            "edges": [None if not np.isfinite(x) else float(x) for x in edges],
            "development": dev, "validation": val,
        }
        for block_name, block in (("development", dev), ("validation", val)):
            for row in block["bins"]:
                csv_rows.append({
                    "feature": feature, "block": block_name, "bin": row["bin"],
                    "lo": row["lo"], "hi": row["hi"], "n_rows": row["n_rows"],
                    "conditional_gap": row["conditional_gap"]["mean"],
                    "conditional_gap_case_sem": row["conditional_gap"]["case_sem"],
                    "contribution": row["contribution_to_global_gap"]["mean"],
                    "contribution_case_sem": row["contribution_to_global_gap"]["case_sem"],
                })
    adjusted = holm_adjust(raw_p)
    for feature, value in adjusted.items():
        payload["features"][feature]["validation"]["high_minus_low_p_holm"] = value
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    pd.DataFrame(csv_rows).to_csv(args.output_csv, index=False)
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown_report(payload))
    print(json.dumps({
        "n_rows": payload["n_rows"], "n_manifest_pairs": payload["n_manifest_pairs"],
        "development_global_gap": payload["development_global_gap"],
        "validation_global_gap": payload["validation_global_gap"],
        "validation_high_minus_low": {
            feature: payload["features"][feature]["validation"]["high_minus_low_gap"]
            for feature in FEATURES
        },
    }, indent=2, sort_keys=True))
    print("ANCHORBLEND_COHERENT_LOCALIZATION_DONE", flush=True)


if __name__ == "__main__":
    main()
