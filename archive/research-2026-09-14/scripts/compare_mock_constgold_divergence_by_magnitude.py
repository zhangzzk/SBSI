#!/usr/bin/env python
"""Conditional mock--ConstGold divergence as a function of measured magnitude."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MOCK_COLOR = "#0072B2"
CONSTGOLD_COLOR = "#D55E00"
JS_COLOR = "#009E73"
NULL_COLOR = "#666666"
MAG_EDGES = np.r_[17.0, 19.5, np.arange(20.0, 25.6, 0.5), 25.8]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", type=Path, required=True)
    parser.add_argument("--constgold-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--feature-bins", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--dirichlet-total", type=float, default=0.5)
    return parser.parse_args(argv)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_values(path, pixel_size):
    frame = pd.read_parquet(
        path,
        columns=[
            "measured_ngmix_g1",
            "measured_ngmix_g2",
            "measured_mag_auto",
            "measured_log_flux_radius",
        ],
    )
    values = pd.DataFrame(
        {
            "g1": frame["measured_ngmix_g1"].to_numpy(float),
            "g2": frame["measured_ngmix_g2"].to_numpy(float),
            "mag": frame["measured_mag_auto"].to_numpy(float),
            "log_radius": frame["measured_log_flux_radius"].to_numpy(float)
            + np.log(pixel_size),
        }
    )
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"non-finite measurements in {path}")
    return values


def probabilities(counts, total_pseudocount=0.0):
    counts = np.asarray(counts, dtype=np.float64)
    if total_pseudocount:
        counts = counts + total_pseudocount / counts.size
    return counts / counts.sum()


def kl(p, q):
    active = p > 0
    return float(np.sum(p[active] * np.log(p[active] / q[active])))


def js_from_counts(counts_p, counts_q):
    p = probabilities(counts_p)
    q = probabilities(counts_q)
    middle = 0.5 * (p + q)
    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def fixed_feature_cells(pool, bins):
    coordinates = []
    edge_summary = {}
    for feature in ("g1", "g2", "log_radius"):
        edges = np.quantile(pool[feature], np.linspace(0.0, 1.0, bins + 1))
        if np.unique(edges).size != edges.size:
            raise ValueError(f"cannot form {bins} distinct bins for {feature}")
        edges[0], edges[-1] = -np.inf, np.inf
        coordinates.append(
            np.searchsorted(edges[1:-1], pool[feature].to_numpy(float), side="right")
        )
        edge_summary[feature] = [
            None if not np.isfinite(value) else float(value) for value in edges
        ]
    return (
        np.ravel_multi_index(tuple(coordinates), (bins, bins, bins)),
        edge_summary,
    )


def conditional_metrics(mock_cells, constgold_cells, n_cells, alpha_total, rng, n_perm):
    mock_counts = np.bincount(mock_cells, minlength=n_cells)
    constgold_counts = np.bincount(constgold_cells, minlength=n_cells)
    p = probabilities(mock_counts, alpha_total)
    q = probabilities(constgold_counts, alpha_total)
    forward = kl(p, q)
    reverse = kl(q, p)
    observed_js = js_from_counts(mock_counts, constgold_counts)

    pooled_cells = np.r_[mock_cells, constgold_cells]
    pooled_counts = mock_counts + constgold_counts
    n_mock = len(mock_cells)
    null = np.empty(n_perm, dtype=float)
    for index in range(n_perm):
        selected = rng.permutation(len(pooled_cells))[:n_mock]
        permuted_mock = np.bincount(pooled_cells[selected], minlength=n_cells)
        null[index] = js_from_counts(permuted_mock, pooled_counts - permuted_mock)
    q025, median, q975 = np.quantile(null, (0.025, 0.5, 0.975))
    return {
        "kl_mock_to_constgold_nats": forward,
        "kl_constgold_to_mock_nats": reverse,
        "js_nats": observed_js,
        "js_null_q025_nats": float(q025),
        "js_null_median_nats": float(median),
        "js_null_q975_nats": float(q975),
        "js_excess_over_null_median_nats": float(observed_js - median),
        "permutation_tail_fraction": float(
            (1 + np.count_nonzero(null >= observed_js)) / (n_perm + 1)
        ),
        "mock_occupied_cells": int(np.count_nonzero(mock_counts)),
        "constgold_occupied_cells": int(np.count_nonzero(constgold_counts)),
    }


def analyze(mock, constgold, args):
    pool = pd.concat([mock, constgold], ignore_index=True)
    cells, feature_edges = fixed_feature_cells(pool, args.feature_bins)
    mock_cells = cells[: len(mock)]
    constgold_cells = cells[len(mock) :]
    rng = np.random.default_rng(args.seed)
    rows = []
    for low, high in zip(MAG_EDGES[:-1], MAG_EDGES[1:]):
        mock_mask = (mock["mag"].to_numpy() >= low) & (mock["mag"].to_numpy() < high)
        constgold_mask = (constgold["mag"].to_numpy() >= low) & (
            constgold["mag"].to_numpy() < high
        )
        n_mock = int(mock_mask.sum())
        n_constgold = int(constgold_mask.sum())
        if n_mock == 0 or n_constgold == 0:
            raise ValueError(f"empty sample in magnitude bin [{low}, {high})")
        metrics = conditional_metrics(
            mock_cells[mock_mask],
            constgold_cells[constgold_mask],
            args.feature_bins**3,
            args.dirichlet_total,
            rng,
            args.permutations,
        )
        rows.append(
            {
                "mag_low": float(low),
                "mag_high": float(high),
                "mag_center": float(0.5 * (low + high)),
                "n_mock": n_mock,
                "n_constgold": n_constgold,
                "mock_fraction": n_mock / len(mock),
                "constgold_fraction": n_constgold / len(constgold),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    result["weighted_conditional_kl_mock_to_constgold_nats"] = (
        result["mock_fraction"] * result["kl_mock_to_constgold_nats"]
    )
    result["weighted_conditional_kl_constgold_to_mock_nats"] = (
        result["constgold_fraction"] * result["kl_constgold_to_mock_nats"]
    )

    p_mag = probabilities(result["n_mock"], args.dirichlet_total)
    q_mag = probabilities(result["n_constgold"], args.dirichlet_total)
    decomposition = {
        "magnitude_marginal_kl_mock_to_constgold_nats": kl(p_mag, q_mag),
        "magnitude_marginal_kl_constgold_to_mock_nats": kl(q_mag, p_mag),
        "weighted_conditional_kl_mock_to_constgold_nats": float(
            result["weighted_conditional_kl_mock_to_constgold_nats"].sum()
        ),
        "weighted_conditional_kl_constgold_to_mock_nats": float(
            result["weighted_conditional_kl_constgold_to_mock_nats"].sum()
        ),
    }
    decomposition["decomposed_joint_kl_mock_to_constgold_nats"] = (
        decomposition["magnitude_marginal_kl_mock_to_constgold_nats"]
        + decomposition["weighted_conditional_kl_mock_to_constgold_nats"]
    )
    decomposition["decomposed_joint_kl_constgold_to_mock_nats"] = (
        decomposition["magnitude_marginal_kl_constgold_to_mock_nats"]
        + decomposition["weighted_conditional_kl_constgold_to_mock_nats"]
    )
    bright = result["mag_high"] <= 22.0
    decomposition["bright_m_lt_22_fraction_of_weighted_conditional_forward"] = float(
        result.loc[bright, "weighted_conditional_kl_mock_to_constgold_nats"].sum()
        / decomposition["weighted_conditional_kl_mock_to_constgold_nats"]
    )
    decomposition["bright_m_lt_22_fraction_of_weighted_conditional_reverse"] = float(
        result.loc[bright, "weighted_conditional_kl_constgold_to_mock_nats"].sum()
        / decomposition["weighted_conditional_kl_constgold_to_mock_nats"]
    )
    return result, feature_edges, decomposition


def draw(result, output, feature_bins):
    centers = result["mag_center"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    upper, lower = axes
    upper.fill_between(
        centers,
        result["js_null_q025_nats"],
        result["js_null_q975_nats"],
        color=NULL_COLOR,
        alpha=0.18,
        linewidth=0,
        label="Permutation-null JSD 95% interval",
    )
    upper.plot(
        centers,
        result["js_null_median_nats"],
        color=NULL_COLOR,
        linestyle=":",
        linewidth=1.2,
        label="Permutation-null JSD median",
    )
    upper.plot(
        centers,
        result["kl_mock_to_constgold_nats"],
        color=MOCK_COLOR,
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label=r"$D_{KL}(\mathrm{mock}\Vert\mathrm{ConstGold})$",
    )
    upper.plot(
        centers,
        result["kl_constgold_to_mock_nats"],
        color=CONSTGOLD_COLOR,
        marker="s",
        markersize=3.2,
        linestyle="--",
        linewidth=1.4,
        label=r"$D_{KL}(\mathrm{ConstGold}\Vert\mathrm{mock})$",
    )
    upper.plot(
        centers,
        result["js_nats"],
        color=JS_COLOR,
        marker="^",
        markersize=3.5,
        linewidth=1.5,
        label="Jensen–Shannon",
    )
    upper.set_yscale("log")
    upper.set_ylabel("Conditional divergence (nats)")
    upper.legend(frameon=False, ncol=2, fontsize=8)

    lower.plot(
        centers,
        result["weighted_conditional_kl_mock_to_constgold_nats"],
        color=MOCK_COLOR,
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label="Mock-weighted contribution",
    )
    lower.plot(
        centers,
        result["weighted_conditional_kl_constgold_to_mock_nats"],
        color=CONSTGOLD_COLOR,
        marker="s",
        markersize=3.2,
        linestyle="--",
        linewidth=1.4,
        label="ConstGold-weighted contribution",
    )
    lower.set_yscale("log")
    lower.set_ylabel("Contribution to conditional KL (nats)")
    lower.set_xlabel("Measured MAG_AUTO bin")
    lower.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.axvline(22.0, color="#000000", linestyle="--", linewidth=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(MAG_EDGES[0], MAG_EDGES[-1])
    lower.text(
        22.03,
        0.96,
        r"$m=22$",
        transform=lower.get_xaxis_transform(),
        va="top",
        fontsize=8,
    )
    lower.set_xticks(np.arange(18.0, 26.0, 1.0))
    fig.suptitle(
        "Mock–ConstGold divergence by measured magnitude\n"
        rf"Conditional distribution of $(\hat g_1,\hat g_2,\log R_{{\rm flux}})$; "
        rf"{feature_bins}$^3$ fixed pooled-quantile cells",
        fontsize=11,
        y=0.985,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.88, hspace=0.14)
    fig.savefig(output / "conditional_divergence_by_magnitude.png", dpi=300)
    fig.savefig(output / "conditional_divergence_by_magnitude.pdf")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.feature_bins < 2 or args.permutations < 1 or args.dirichlet_total <= 0:
        raise ValueError("invalid bin, permutation, or pseudocount setting")
    mock = load_values(args.mock, args.pixel_size)
    constgold = load_values(args.constgold_sample, args.pixel_size)
    result, feature_edges, decomposition = analyze(mock, constgold, args)
    args.output.mkdir(parents=True)
    result.to_csv(args.output / "conditional_divergence_by_magnitude.csv", index=False)
    draw(result, args.output, args.feature_bins)

    top_forward = result.nlargest(
        3, "weighted_conditional_kl_mock_to_constgold_nats"
    )[["mag_low", "mag_high", "weighted_conditional_kl_mock_to_constgold_nats"]]
    summary = {
        "comparison": "complete-likelihood mock vs ConstGold plus measurements by measured magnitude",
        "sample_counts": {"mock": int(len(mock)), "constgold": int(len(constgold))},
        "magnitude_edges": MAG_EDGES.tolist(),
        "conditional_features": ["g1", "g2", "natural_log_radius_arcsec"],
        "conditional_histogram": {
            "bins_per_feature": int(args.feature_bins),
            "total_cells": int(args.feature_bins**3),
            "edges": feature_edges,
            "edge_definition": "fixed pooled empirical-quantile edges from both full 100k samples",
        },
        "divergence": {
            "units": "natural-log nats",
            "directed_kl_smoothing": (
                "symmetric Dirichlet pseudocount with fixed total concentration "
                f"{args.dirichlet_total:g} divided across cells"
            ),
            "js_smoothing": "none",
        },
        "finite_sample_reference": {
            "method": "permute labels independently within each magnitude bin at fixed counts",
            "permutations_per_bin": int(args.permutations),
            "seed": int(args.seed),
        },
        "decomposition": decomposition,
        "largest_forward_weighted_bins": top_forward.to_dict(orient="records"),
        "inputs": {
            "mock": str(args.mock.resolve()),
            "mock_sha256": sha256(args.mock),
            "constgold_sample": str(args.constgold_sample.resolve()),
            "constgold_sample_sha256": sha256(args.constgold_sample),
        },
        "unmatched_rows": 0,
        "unmatched_note": "No catalogue join is used; magnitude bins are applied independently.",
        "results": result.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
