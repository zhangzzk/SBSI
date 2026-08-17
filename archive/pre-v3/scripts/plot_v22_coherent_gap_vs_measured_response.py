"""Plot coherent-anchor gaps conditioned on the measured scene response."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np
import pandas as pd

from scripts.apply_v22_pair_residual_calibration_coherent import (
    BIN_LOOKUP_MODEL,
    LINEAR_INTERP_MODEL,
    PRIMARY_MODEL,
    SENSITIVITY_MODEL,
    SPLINE_MODEL,
    apply_to_coherent,
    discover_pair_paths,
    finite_stat,
    read_references,
)


MODEL_STYLES = (
    (PRIMARY_MODEL, "#D55E00", "s", "Hinge-corrected", "-"),
    (SENSITIVITY_MODEL, "0.25", "D", "Saturating sensitivity", "--"),
    (SPLINE_MODEL, "#CC79A7", "^", "Natural cubic spline", ":"),
    (LINEAR_INTERP_MODEL, "#56B4E9", "P", "Direct linear curve", "-"),
    (BIN_LOOKUP_MODEL, "#009E73", "v", "Direct bin lookup", "-."),
)


def measured_response_deciles(
    anchors: pd.DataFrame, fits: dict[str, Any], n_bins: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Bin anchors by measured coherent response with case-level errors."""
    measurement = anchors.R_blend_truth.to_numpy(float)
    edges = np.unique(np.quantile(measurement, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) != n_bins + 1:
        raise RuntimeError("measured-response quantile edges collapsed")
    bin_index = np.searchsorted(edges[1:-1], measurement, side="right")
    rows = []
    for index in range(n_bins):
        local = anchors.loc[bin_index == index]
        columns = ["R_blend_truth", "gap_truth_minus_raw"] + [
            f"gap_truth_minus_{model}" for model in fits
        ]
        case = local.groupby("case", sort=True)[columns].mean()
        if len(case) != anchors.case.nunique():
            raise RuntimeError(
                f"measured-response bin {index} does not contain every case"
            )
        rows.append({
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "n_anchors": int(len(local)),
            "n_cases_with_anchors": int(len(case)),
            "measured_scene_response": finite_stat(
                case.R_blend_truth.to_numpy(float)
            ),
            "raw_gap_truth_minus_prediction": finite_stat(
                case.gap_truth_minus_raw.to_numpy(float)
            ),
            "models": {
                model: finite_stat(
                    case[f"gap_truth_minus_{model}"].to_numpy(float)
                ) for model in fits
            },
        })
    return rows, edges


def response_distribution_summary(anchors: pd.DataFrame) -> dict[str, Any]:
    response = anchors.R_blend_truth.to_numpy(float)
    probabilities = np.asarray([
        0.0, 0.0005, 0.001, 0.01, 0.05, 0.25, 0.5,
        0.75, 0.95, 0.99, 0.999, 0.9995, 1.0,
    ])
    quantiles = np.quantile(response, probabilities)
    case_mean = anchors.groupby("case", sort=True).R_blend_truth.mean().to_numpy(float)
    symmetric_limit = float(max(abs(quantiles[1]), abs(quantiles[-2])))
    return {
        "n_anchors": int(len(anchors)),
        "case_balanced_mean": finite_stat(case_mean),
        "pooled_sd": float(response.std(ddof=1)),
        "pooled_quantiles": {
            f"q{probability:.4f}": float(value)
            for probability, value in zip(probabilities, quantiles, strict=True)
        },
        "display_limits": [-symmetric_limit, symmetric_limit],
        "displayed_fraction": float(np.mean(np.abs(response) <= symmetric_limit)),
    }


def case_balanced_histogram(
    anchors: pd.DataFrame, edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM of within-case normalized response histograms."""
    rows = []
    for _, local in anchors.groupby("case", sort=True):
        counts = np.histogram(local.R_blend_truth.to_numpy(float), bins=edges)[0]
        rows.append(counts / len(local))
    values = np.asarray(rows, float)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def flatten_deciles(deciles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in deciles:
        row = {
            "bin": item["bin"],
            "lower": item["lower"],
            "upper": item["upper"],
            "n_anchors": item["n_anchors"],
            "measured_scene_response_mean": item["measured_scene_response"]["mean"],
            "measured_scene_response_case_sem": item["measured_scene_response"]["case_sem"],
            "raw_gap_mean": item["raw_gap_truth_minus_prediction"]["mean"],
            "raw_gap_case_sem": item["raw_gap_truth_minus_prediction"]["case_sem"],
        }
        for model, value in item["models"].items():
            row[f"{model}_gap_mean"] = value["mean"]
            row[f"{model}_gap_case_sem"] = value["case_sem"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_panel(
    anchors: pd.DataFrame,
    deciles: list[dict[str, Any]],
    distribution: dict[str, Any],
    output_prefix: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    linthresh = 0.1
    axis.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    lower, upper = distribution["display_limits"]
    axis.set_xlim(lower, upper)

    # Uniform visual-width histogram bins on the same signed-log transform.
    transform = axis.xaxis.get_transform()
    transformed_edges = np.linspace(
        transform.transform(lower), transform.transform(upper), 64,
    )
    histogram_edges = transform.inverted().transform(transformed_edges)
    histogram_mean, histogram_sem = case_balanced_histogram(
        anchors, histogram_edges,
    )
    histogram_axis = axis.twinx()
    histogram_axis.set_zorder(0)
    axis.set_zorder(1)
    axis.patch.set_visible(False)
    histogram_axis.stairs(
        histogram_mean, histogram_edges, baseline=0.0, fill=True,
        color="0.72", alpha=0.45, linewidth=0.0,
    )
    center = 0.5 * (histogram_edges[:-1] + histogram_edges[1:])
    histogram_axis.fill_between(
        center,
        np.maximum(histogram_mean - histogram_sem, 0.0),
        histogram_mean + histogram_sem,
        step="mid", color="0.55", alpha=0.22, linewidth=0.0,
    )
    histogram_axis.set_ylim(0.0, max(histogram_mean + histogram_sem) / 0.56)
    histogram_axis.set_yticks([])
    for spine in histogram_axis.spines.values():
        spine.set_visible(False)

    x = np.asarray([item["measured_scene_response"]["mean"] for item in deciles])
    series = [
        (
            "raw_gap_truth_minus_prediction", "#0072B2", "o", "Raw V2.2", "-",
        ),
        *MODEL_STYLES,
    ]
    handles = []
    for field, color, marker, label, linestyle in series:
        if field == "raw_gap_truth_minus_prediction":
            mean = np.asarray([item[field]["mean"] for item in deciles])
            sem = np.asarray([item[field]["case_sem"] for item in deciles])
        else:
            mean = np.asarray([item["models"][field]["mean"] for item in deciles])
            sem = np.asarray([item["models"][field]["case_sem"] for item in deciles])
        handle = axis.errorbar(
            x, mean, yerr=sem, color=color, marker=marker,
            markerfacecolor=(
                "white" if field in (
                    SENSITIVITY_MODEL, SPLINE_MODEL, LINEAR_INTERP_MODEL,
                    BIN_LOOKUP_MODEL,
                ) else color
            ),
            markersize=4.2, linewidth=1.1, linestyle=linestyle,
            capsize=2.0, label=label, zorder=3,
        )
        handles.append(handle)
    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.85, zorder=2)
    axis.set_xlabel(r"Measured coherent anchor-scene response $R_{\rm blend}^{\rm truth}$")
    axis.set_ylabel(r"Coherent truth $-$ prediction")
    axis.set_title(
        "Coherent gap conditioned on measured anchor response\n"
        "Equal-count response deciles; errors are one case SEM"
    )
    histogram_handle = Patch(
        facecolor="0.72", alpha=0.45, edgecolor="none",
        label="Anchor-response density",
    )
    axis.legend(
        [*handles, histogram_handle],
        [handle.get_label() for handle in handles] + [histogram_handle.get_label()],
        frameon=False, loc="upper left", ncol=2,
    )
    axis.text(
        0.99, 0.02,
        "Descriptive: measured truth appears on both axes",
        transform=axis.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="0.35",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", linewidth=0.55)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.83)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_report(payload: dict[str, Any]) -> str:
    distribution = payload["measured_response_distribution"]
    quantiles = distribution["pooled_quantiles"]
    return "\n".join([
        "# Coherent gap versus measured anchor-scene response",
        "",
        f"- Cases: `{payload['case_window'][0]}--{payload['case_window'][1]}`; "
        f"anchors: `{distribution['n_anchors']:,}`.",
        "- The x axis and ten equal-count bins use the measured coherent "
        "anchor response `R_blend_truth`.",
        "- The gray lower-axis histogram is the mean of within-case normalized "
        "response histograms; its band is one SEM across cases.",
        f"- Measured response median: `{quantiles['q0.5000']:+.6f}`; "
        f"5--95% interval: `[{quantiles['q0.0500']:+.6f}, "
        f"{quantiles['q0.9500']:+.6f}]`; 1--99% interval: "
        f"`[{quantiles['q0.0100']:+.6f}, {quantiles['q0.9900']:+.6f}]`.",
        "- This is descriptive rather than a calibration diagnostic: measured "
        "truth enters both x and `truth - prediction`, inducing mathematical coupling.",
        "- The pair-residual corrections are exactly those frozen in the source "
        "transfer result; no curve is refitted in measured-response bins.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()
    if args.n_bins < 4:
        raise ValueError("need at least four measured-response bins")
    outputs = [
        f"{args.output_prefix}.{suffix}" for suffix in ("png", "pdf", "json", "md")
    ] + [f"{args.output_prefix}_deciles.csv"]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    with open(args.source_json, encoding="utf-8") as handle:
        source = json.load(handle)
    case_min, case_max = map(int, source["case_window"])
    fits = source["calibration_fits"]
    expected_models = {
        PRIMARY_MODEL, SENSITIVITY_MODEL, SPLINE_MODEL,
        LINEAR_INTERP_MODEL, BIN_LOOKUP_MODEL,
    }
    if set(fits) != expected_models:
        raise RuntimeError("source calibration-model set is not the expected panel C")
    pair_paths, designs = discover_pair_paths(
        [item["path"] for item in source["pair_designs"]], case_min, case_max,
    )
    reference = read_references(
        source["coherent_reference_paths"], case_min, case_max,
    )
    anchors, support = apply_to_coherent(
        reference, pair_paths, fits,
        source["coherent_support"]["raw_scene_replay_tolerance"],
    )
    for key in ("n_scored_pairs", "n_anchors", "maximum_raw_scene_replay_error"):
        if not np.isclose(support[key], source["coherent_support"][key], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"source replay mismatch for {key}")

    deciles, edges = measured_response_deciles(anchors, fits, args.n_bins)
    distribution = response_distribution_summary(anchors)
    payload = {
        "title": "Coherent gap conditioned on measured anchor-scene response",
        "source_json": os.path.abspath(args.source_json),
        "case_window": [case_min, case_max],
        "n_bins": int(args.n_bins),
        "conditioning": "pooled equal-count quantiles of R_blend_truth",
        "uncertainty_unit": "rendered case",
        "mathematical_coupling_warning": (
            "R_blend_truth appears in both the conditioning coordinate and the "
            "truth-minus-prediction ordinate"
        ),
        "calibration_refit": False,
        "model_order": [item[0] for item in MODEL_STYLES],
        "pair_designs": designs,
        "support": support,
        "measured_response_edges": edges.tolist(),
        "measured_response_deciles": deciles,
        "measured_response_distribution": distribution,
        "constgold_opened": False,
    }
    with open(f"{args.output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    flatten_deciles(deciles).to_csv(
        f"{args.output_prefix}_deciles.csv", index=False,
    )
    Path(f"{args.output_prefix}.md").write_text(
        markdown_report(payload), encoding="utf-8",
    )
    plot_panel(anchors, deciles, distribution, args.output_prefix)
    print(json.dumps({
        "measured_response_distribution": distribution,
        "deciles": deciles,
        "support": support,
    }, indent=2, allow_nan=False), flush=True)
    print("COHERENT_GAP_VS_MEASURED_RESPONSE_DONE", flush=True)


if __name__ == "__main__":
    main()
