"""Plot measured per-neighbour R_blend in the one-active anchor experiment."""
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


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray,
) -> np.ndarray:
    """Return deterministic weighted empirical quantiles."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    probabilities = np.asarray(probabilities, float)
    if (
        values.ndim != 1 or weights.shape != values.shape
        or not np.isfinite(values).all() or not np.isfinite(weights).all()
        or (weights <= 0.0).any() or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError("invalid weighted-quantile inputs")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= sorted_weights.sum()
    return np.interp(
        probabilities, cumulative, sorted_values,
        left=sorted_values[0], right=sorted_values[-1],
    )


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("statistic requires at least two finite values")
    return {
        "mean": float(values.mean()),
        "case_sd": float(values.std(ddof=1)),
        "case_sem": float(values.std(ddof=1) / np.sqrt(values.size)),
        "n_cases": int(values.size),
    }


def histogram_by_case(
    frame: pd.DataFrame, edges: np.ndarray, weight_column: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and SEM of within-case normalized histograms."""
    rows = []
    for _, local in frame.groupby("case", sort=True):
        weights = (
            local[weight_column].to_numpy(float)
            if weight_column is not None else np.ones(len(local), dtype=float)
        )
        counts = np.histogram(
            local.R_one_pair.to_numpy(float), bins=edges, weights=weights,
        )[0]
        rows.append(counts / weights.sum())
    values = np.asarray(rows, float)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def summarize(frame: pd.DataFrame, tail_probability: float) -> dict[str, Any]:
    """Summarize selected-pair and HT pair-population distributions."""
    response = frame.R_one_pair.to_numpy(float)
    pair_weight = frame.n_pairs.to_numpy(float)
    case_weighted_means = np.asarray([
        np.average(local.R_one_pair, weights=local.n_pairs)
        for _, local in frame.groupby("case", sort=True)
    ])
    case_selected_means = frame.groupby("case", sort=True).R_one_pair.mean().to_numpy(float)
    probabilities = np.asarray([
        0.0, 0.0005, 0.001, 0.01, 0.05, 0.5,
        0.95, 0.99, 0.999, 0.9995, 1.0,
    ])
    quantiles = weighted_quantile(response, pair_weight, probabilities)
    threshold = float(tail_probability)
    tail_limits = weighted_quantile(
        response, pair_weight, np.asarray([threshold, 1.0 - threshold]),
    )
    symmetric_limit = float(np.max(np.abs(tail_limits)))
    return {
        "n_selected_pairs": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "case_window": [int(frame.case.min()), int(frame.case.max())],
        "selection": (
            "one deployed neighbour selected uniformly per anchor; every row "
            "has a measured two-direction orthogonal response"
        ),
        "selected_pair_case_balanced_mean": finite_stat(case_selected_means),
        "ht_pair_population_case_balanced_mean": finite_stat(case_weighted_means),
        "ht_pair_population_pooled_weighted_mean": float(
            np.average(response, weights=pair_weight)
        ),
        "ht_pair_population_weighted_sd": float(np.sqrt(np.average(
            np.square(response - np.average(response, weights=pair_weight)),
            weights=pair_weight,
        ))),
        "ht_pair_population_weighted_positive_fraction": float(
            np.average(response > 0.0, weights=pair_weight)
        ),
        "ht_pair_population_weighted_quantiles": {
            f"q{probability:.4f}": float(value)
            for probability, value in zip(probabilities, quantiles, strict=True)
        },
        "display_tail_probability_each_side": threshold,
        "display_limits": [-symmetric_limit, symmetric_limit],
        "displayed_ht_weight_fraction": float(np.average(
            (response >= -symmetric_limit) & (response <= symmetric_limit),
            weights=pair_weight,
        )),
    }


def plot_histogram(
    frame: pd.DataFrame, summary: dict[str, Any], bins: int, output_prefix: str,
) -> None:
    lower, upper = summary["display_limits"]
    edges = np.linspace(lower, upper, bins + 1)
    center = 0.5 * (edges[:-1] + edges[1:])
    pair_mean, pair_sem = histogram_by_case(frame, edges, "n_pairs")
    selected_mean, _ = histogram_by_case(frame, edges, None)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axis = plt.subplots(figsize=(7.2, 4.35))
    color = "#0072B2"
    axis.stairs(
        pair_mean, edges, color=color, linewidth=1.25, fill=True, alpha=0.28,
        label=r"Pair population (weight $n_{\rm pair}$)",
    )
    positive_floor = max(float(pair_mean[pair_mean > 0.0].min()) * 0.25, 1.0e-7)
    axis.fill_between(
        center,
        np.maximum(pair_mean - pair_sem, positive_floor),
        pair_mean + pair_sem,
        step="mid", color=color, alpha=0.18, linewidth=0.0,
        label="One case SEM",
    )
    axis.stairs(
        selected_mean, edges, color="#D55E00", linewidth=1.0,
        label="Raw selected pairs (one per anchor)",
    )
    mean = summary["ht_pair_population_case_balanced_mean"]["mean"]
    sem = summary["ht_pair_population_case_balanced_mean"]["case_sem"]
    median = summary["ht_pair_population_weighted_quantiles"]["q0.5000"]
    axis.axvline(0.0, color="0.45", linestyle=":", linewidth=0.9)
    axis.axvline(
        mean, color="#009E73", linestyle="--", linewidth=1.3,
        label=fr"Case-balanced mean $={mean:+.4f}\pm{sem:.4f}$",
    )
    axis.axvline(
        median, color="0.2", linestyle="-.", linewidth=1.0,
        label=fr"Weighted median $={median:+.4f}$",
    )
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(bottom=positive_floor)
    axis.set_xlabel(r"Measured single-active-neighbour $R_{\rm blend}$")
    axis.set_ylabel("Fraction per bin")
    axis.set_title(
        "One-active-neighbour orthogonal test, cases 400–499\n"
        f"{len(frame):,} selected pairs; central 99.8% displayed"
    )
    axis.legend(frameon=False, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", linewidth=0.55)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.84)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_report(summary: dict[str, Any]) -> str:
    mean = summary["ht_pair_population_case_balanced_mean"]
    quantiles = summary["ht_pair_population_weighted_quantiles"]
    return "\n".join([
        "# Single-active-neighbour measured R_blend histogram",
        "",
        f"- Cases: `{summary['case_window'][0]}--{summary['case_window'][1]}`; "
        f"selected measured pairs: `{summary['n_selected_pairs']:,}`.",
        "- Each anchor contributes one uniformly selected deployed neighbour. "
        "The filled histogram uses inverse-selection weight `n_pairs` to estimate "
        "the deployed pair population; the orange outline is the raw one-per-anchor sample.",
        f"- Pair-population case-balanced mean: `{mean['mean']:+.6f} +- "
        f"{mean['case_sem']:.6f}` (one SEM across `{mean['n_cases']}` cases).",
        f"- Weighted median: `{quantiles['q0.5000']:+.6f}`; weighted 1--99% "
        f"interval: `[{quantiles['q0.0100']:+.6f}, {quantiles['q0.9900']:+.6f}]`.",
        f"- Weighted SD: `{summary['ht_pair_population_weighted_sd']:.6f}`; "
        f"positive fraction: "
        f"`{100.0 * summary['ht_pair_population_weighted_positive_fraction']:.2f}%`.",
        "- The figure displays the central 99.8% and uses a logarithmic y-axis; "
        "the numerical summary retains the full tails.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bins", type=int, default=140)
    parser.add_argument("--tail-probability", type=float, default=0.001)
    args = parser.parse_args()
    if args.bins < 20 or not 0.0 < args.tail_probability < 0.1:
        raise ValueError("invalid histogram bin count or tail probability")
    outputs = [
        f"{args.output_prefix}.{suffix}" for suffix in ("png", "pdf", "json", "md")
    ]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_feather(
        args.input,
        columns=["case", "secondary_index", "n_pairs", "selection_probability", "R_one_pair"],
    )
    if frame.empty or frame.secondary_index.isna().any():
        raise RuntimeError("one-active table is empty or has unmeasured selected pairs")
    values = frame[["n_pairs", "selection_probability", "R_one_pair"]].to_numpy(float)
    if not np.isfinite(values).all() or (frame.n_pairs <= 0).any():
        raise RuntimeError("one-active table contains invalid histogram values")
    if not np.allclose(
        frame.n_pairs * frame.selection_probability, 1.0,
        rtol=0.0, atol=1.0e-12,
    ):
        raise RuntimeError("selection probabilities do not equal 1/n_pairs")

    summary = summarize(frame, args.tail_probability)
    summary["input"] = os.path.abspath(args.input)
    summary["histogram_bins"] = int(args.bins)
    with open(f"{args.output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    Path(f"{args.output_prefix}.md").write_text(
        markdown_report(summary), encoding="utf-8",
    )
    plot_histogram(frame, summary, args.bins, args.output_prefix)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print("ONEACTIVE_RBLEND_HIST_DONE", flush=True)


if __name__ == "__main__":
    main()
