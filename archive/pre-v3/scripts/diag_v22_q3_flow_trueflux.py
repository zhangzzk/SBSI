"""Diagnose the constgold-q3 V2.2 flow residual by true neighbour flux shell.

This is an evaluation-only view of the completed 100-case local-scene
decomposition.  The shell coordinates sum the intrinsic flux of every rendered
source except the anchor, before detection, in disjoint 0--1, 1--3 and 3--10
arcsec ranges.  No response-derived blendness enters the bin definitions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHELLS = [
    ("near", "logflux_abs_near_0_1", r"Near: 0--1$''$"),
    ("mid", "logflux_abs_mid_1_3", r"Mid: 1--3$''$"),
    ("far", "logflux_abs_far_3_10", r"Far: 3--10$''$"),
]
PROXY_COLUMNS = ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"]


def case_stats(frame: pd.DataFrame, column: str) -> dict:
    values = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return {
        "mean": float(values.mean()), "case_sem": float(sem),
        "n_cases": int(len(values)), "case_values": values.tolist(),
    }


def seeded_stats(frame: pd.DataFrame, seed_columns: list[str],
                 subtract: str | None = None) -> dict:
    values = frame[seed_columns].copy()
    if subtract is not None:
        values = values.subtract(frame[subtract].to_numpy(float), axis=0)
    seed_values = values.mean(axis=0).to_numpy(float)
    ensemble = values.mean(axis=1)
    case_values = pd.DataFrame({
        "case": frame["case"].to_numpy(), "value": ensemble.to_numpy(float),
    }).groupby("case", sort=True)["value"].mean().to_numpy(float)
    seed_sem = seed_values.std(ddof=1) / np.sqrt(len(seed_values))
    case_sem = case_values.std(ddof=1) / np.sqrt(len(case_values))
    return {
        "mean": float(seed_values.mean()),
        "seed_sem": float(seed_sem), "case_sem": float(case_sem),
        "quadrature_sem": float(np.hypot(seed_sem, case_sem)),
        "n_cases": int(len(case_values)),
        "seed_values": seed_values.tolist(), "case_values": case_values.tolist(),
    }


def flux_classes(values: np.ndarray) -> tuple[np.ndarray, list[str], list[float], str]:
    """Return a zero class plus positive quartiles, or ordinary quartiles."""
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("flux coordinate contains non-finite values")
    if np.any(values < 0):
        raise ValueError("log1p flux coordinate is negative")
    if np.any(values == 0.0):
        positive = values[values > 0.0]
        if len(positive) < 40:
            raise ValueError("too few positive rows for flux quartiles")
        edges = np.quantile(positive, [0.25, 0.5, 0.75])
        if len(np.unique(edges)) != 3:
            raise ValueError("positive-flux quartile edges are not unique")
        classes = np.zeros(len(values), dtype=np.int8)
        classes[values > 0.0] = 1 + np.digitize(values[values > 0.0], edges, right=True)
        return classes, ["zero", "Q1", "Q2", "Q3", "Q4"], edges.tolist(), \
            "zero_plus_positive_quartiles"
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    if len(np.unique(edges)) != 3:
        raise ValueError("flux quartile edges are not unique")
    return np.digitize(values, edges, right=True).astype(np.int8), \
        ["Q1", "Q2", "Q3", "Q4"], edges.tolist(), "quartiles"


def paired_contrast(low: pd.DataFrame, high: pd.DataFrame, column: str) -> dict:
    lo = low.groupby("case", sort=True)[column].mean()
    hi = high.groupby("case", sort=True)[column].mean()
    paired = hi.to_frame("high").join(lo.to_frame("low"), how="inner")
    values = (paired["high"] - paired["low"]).to_numpy(float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)), "case_values": values.tolist(),
    }


def seeded_contrast(low: pd.DataFrame, high: pd.DataFrame,
                    seed_columns: list[str], subtract: str | None = None) -> dict:
    low_values = low[seed_columns].copy()
    high_values = high[seed_columns].copy()
    if subtract is not None:
        low_values = low_values.subtract(low[subtract].to_numpy(float), axis=0)
        high_values = high_values.subtract(high[subtract].to_numpy(float), axis=0)
    seed_values = high_values.mean(axis=0).to_numpy(float) \
        - low_values.mean(axis=0).to_numpy(float)
    low_case = pd.DataFrame({
        "case": low["case"].to_numpy(), "value": low_values.mean(axis=1).to_numpy(float),
    }).groupby("case", sort=True)["value"].mean()
    high_case = pd.DataFrame({
        "case": high["case"].to_numpy(), "value": high_values.mean(axis=1).to_numpy(float),
    }).groupby("case", sort=True)["value"].mean()
    paired = high_case.to_frame("high").join(low_case.to_frame("low"), how="inner")
    case_values = (paired["high"] - paired["low"]).to_numpy(float)
    seed_sem = seed_values.std(ddof=1) / np.sqrt(len(seed_values))
    case_sem = case_values.std(ddof=1) / np.sqrt(len(case_values))
    return {
        "mean": float(seed_values.mean()),
        "seed_sem": float(seed_sem), "case_sem": float(case_sem),
        "quadrature_sem": float(np.hypot(seed_sem, case_sem)),
        "n_cases": int(len(case_values)),
        "seed_values": seed_values.tolist(), "case_values": case_values.tolist(),
    }


def summarize(frame: pd.DataFrame, flow_columns: list[str]) -> dict:
    result = {
        "n_rows": int(len(frame)), "n_cases": int(frame["case"].nunique()),
        "n_flow_seeds": int(len(flow_columns)),
        "global": {
            "R_self_truth": case_stats(frame, "R_self_truth"),
            "R_flow": seeded_stats(frame, flow_columns),
            "flow_minus_self_truth": seeded_stats(
                frame, flow_columns, subtract="R_self_truth"
            ),
            "R_self_null": case_stats(frame, "R_self_null"),
        },
        "shells": {},
    }
    physical = [column for _, column, _ in SHELLS]
    corr_frame = frame[physical + PROXY_COLUMNS].copy()
    corr_frame["logflux_abs_all_0_3"] = np.log10(
        1.0 + frame["flux_abs_near_0_1"] + frame["flux_abs_mid_1_3"]
    )
    corr = corr_frame.corr(method="spearman")
    result["v22_proxy_spearman"] = {
        physical_column: {
            proxy: float(corr.loc[physical_column, proxy]) for proxy in PROXY_COLUMNS
        }
        for physical_column in [*physical, "logflux_abs_all_0_3"]
    }

    for key, column, title in SHELLS:
        classes, labels, edges, method = flux_classes(frame[column].to_numpy(float))
        shell = {
            "column": column, "title": title, "method": method,
            "internal_edges_log10_1plus_flux": edges, "bins": [],
        }
        subsets = []
        for index, label in enumerate(labels):
            subset = frame.loc[classes == index]
            if subset.empty:
                raise RuntimeError(f"empty {key} flux class {label}")
            subsets.append(subset)
            shell["bins"].append({
                "label": label, "n_rows": int(len(subset)),
                "n_cases": int(subset["case"].nunique()),
                "median_log10_1plus_flux": float(subset[column].median()),
                "R_self_truth": case_stats(subset, "R_self_truth"),
                "R_flow": seeded_stats(subset, flow_columns),
                "flow_minus_self_truth": seeded_stats(
                    subset, flow_columns, subtract="R_self_truth"
                ),
                "R_self_null": case_stats(subset, "R_self_null"),
            })
        shell["last_minus_first"] = {
            "R_self_truth": paired_contrast(subsets[0], subsets[-1], "R_self_truth"),
            "R_flow": seeded_contrast(subsets[0], subsets[-1], flow_columns),
            "flow_minus_self_truth": seeded_contrast(
                subsets[0], subsets[-1], flow_columns, subtract="R_self_truth"
            ),
            "R_self_null": paired_contrast(subsets[0], subsets[-1], "R_self_null"),
        }
        worst = max(
            shell["bins"], key=lambda item: abs(item["flow_minus_self_truth"]["mean"])
        )
        shell["largest_absolute_residual_bin"] = {
            "label": worst["label"], **worst["flow_minus_self_truth"],
        }
        result["shells"][key] = shell
    return result


def make_plot(result: dict, prefix: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.7), sharey="row")
    for panel, (key, _, title) in enumerate(SHELLS):
        shell = result["shells"][key]
        bins = shell["bins"]
        x = np.arange(len(bins))
        labels = [item["label"] for item in bins]
        top = axes[0, panel]
        bottom = axes[1, panel]
        for field, label, color, marker, linestyle in (
            ("R_self_truth", r"true $R_{\rm self}$", "#000000", "o", "-"),
            ("R_flow", r"V2.2 $R_{\rm flow}$", "#0072B2", "s", "--"),
        ):
            stats = [item[field] for item in bins]
            y = np.asarray([item["mean"] for item in stats])
            error = np.asarray([
                item.get("quadrature_sem", item["case_sem"]) for item in stats
            ])
            top.errorbar(
                x, y, yerr=error, color=color, marker=marker, linestyle=linestyle,
                linewidth=1.25, markersize=4.5, capsize=2.5, label=label,
            )
        residual = [item["flow_minus_self_truth"] for item in bins]
        bottom.errorbar(
            x, 100 * np.asarray([item["mean"] for item in residual]),
            yerr=100 * np.asarray([item["quadrature_sem"] for item in residual]),
            color="#D55E00", marker="o", linestyle="-", linewidth=1.25,
            markersize=4.5, capsize=2.5, label=r"$R_{\rm flow}-R_{\rm self,true}$",
        )
        null = [item["R_self_null"] for item in bins]
        bottom.errorbar(
            x, 100 * np.asarray([item["mean"] for item in null]),
            yerr=100 * np.asarray([item["case_sem"] for item in null]),
            color="0.45", marker="D", linestyle=":", linewidth=1.0,
            markersize=3.5, capsize=2, label=r"45$^\circ$ self null",
        )
        bottom.axhline(0.0, color="0.25", linewidth=0.8, linestyle="--", zorder=0)
        top.set_title(title)
        bottom.set_xticks(x, labels)
        bottom.set_xlabel("Absolute neighbour-flux class")
        for ax in (top, bottom):
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Self response")
    axes[1, 0].set_ylabel("Response residual (points)")
    axes[0, 2].legend(frameon=False, loc="best")
    axes[1, 2].legend(frameon=False, loc="best")
    fig.suptitle(
        "V2.2 constgold-q3 flow residual versus true total neighbour flux",
        fontsize=11, y=0.995,
    )
    fig.text(
        0.5, 0.008,
        f"{result['n_rows']:,} common anchors in {result['n_cases']} independent cases; "
        r"shells include every intrinsic input source except the anchor. "
        r"Zero is separate; Q1--Q4 are positive-flux quartiles. "
        r"Errors are case SEM, with 16-seed SEM added in quadrature for flow quantities.",
        ha="center", va="bottom", fontsize=7.5, color="0.30",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.955), h_pad=1.2, w_pad=1.0)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(result: dict, path: Path) -> None:
    rows = []
    for shell, info in result["shells"].items():
        for item in info["bins"]:
            row = {
                "shell": shell, "bin": item["label"], "n_rows": item["n_rows"],
                "n_cases": item["n_cases"],
                "median_log10_1plus_flux": item["median_log10_1plus_flux"],
            }
            for field in ("R_self_truth", "R_flow", "flow_minus_self_truth", "R_self_null"):
                row[field] = item[field]["mean"]
                row[field + "_sem"] = item[field].get(
                    "quadrature_sem", item[field]["case_sem"]
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    outputs = [args.output_prefix.with_suffix(suffix) for suffix in (".json", ".csv", ".pdf", ".png")]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    truth = pd.read_feather(args.truth)
    flow = pd.read_feather(args.flow)
    keys = ["case", "input_index"]
    if truth.duplicated(keys).any() or flow.duplicated(keys).any():
        raise RuntimeError("duplicate truth/flow key")
    frame = truth.merge(flow, on=keys, how="inner", validate="one_to_one")
    if len(frame) != len(truth) or len(frame) != len(flow):
        raise RuntimeError(
            f"truth/flow key mismatch: truth={len(truth)} flow={len(flow)} matched={len(frame)}"
        )
    flow_columns = sorted(
        (column for column in frame if column.startswith("R_flow_s")),
        key=lambda column: int(re.search(r"(\d+)$", column).group(1)),
    )
    if len(flow_columns) != 16:
        raise RuntimeError(f"expected 16 flow seeds, got {flow_columns}")
    needed = [
        "R_self_truth", "R_self_null", "flux_abs_near_0_1", "flux_abs_mid_1_3",
        "logflux_abs_near_0_1", "logflux_abs_mid_1_3", "logflux_abs_far_3_10",
        *PROXY_COLUMNS, *flow_columns,
    ]
    finite = np.isfinite(frame[needed].to_numpy(float)).all(axis=1)
    if not finite.all():
        raise RuntimeError(f"found {(~finite).sum()} non-finite matched rows")
    if frame["case"].nunique() != 100:
        raise RuntimeError(f"expected 100 cases, got {frame['case'].nunique()}")

    result = summarize(frame, flow_columns)
    result["truth_path"] = os.path.abspath(args.truth)
    result["flow_path"] = os.path.abspath(args.flow)
    result["flux_definition"] = (
        "log10(1 + summed intrinsic r-band simulation-count flux of every rendered "
        "source except the anchor in the stated angular shell); no primary-flux division"
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(result, args.output_prefix.with_suffix(".csv"))
    make_plot(result, args.output_prefix)

    global_delta = result["global"]["flow_minus_self_truth"]
    print(
        f"global flow-self = {global_delta['mean']:+.6f} +/- "
        f"{global_delta['quadrature_sem']:.6f}"
    )
    for key, _, _ in SHELLS:
        shell = result["shells"][key]
        worst = shell["largest_absolute_residual_bin"]
        contrast = shell["last_minus_first"]["flow_minus_self_truth"]
        print(
            f"{key}: largest |residual| {worst['label']} = {worst['mean']:+.6f} +/- "
            f"{worst['quadrature_sem']:.6f}; last-first = {contrast['mean']:+.6f} +/- "
            f"{contrast['quadrature_sem']:.6f}"
        )
    print(f"wrote {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
