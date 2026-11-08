#!/usr/bin/env python
"""Histogram divergence versus bin resolution for bright mock and ConstGold data."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = ("g1", "g2", "mag", "log_radius")
LABELS = {
    "g1": r"Measured $\hat g_1$",
    "g2": r"Measured $\hat g_2$",
    "mag": "Measured MAG_AUTO",
    "log_radius": r"$\log$ measured flux radius",
}
SHORT_LABELS = {
    "g1": r"$\hat g_1$",
    "g2": r"$\hat g_2$",
    "mag": "MAG_AUTO",
    "log_radius": r"$\log R_\mathrm{flux}$",
}
COLORS = {
    "forward": "#0072B2",
    "reverse": "#D55E00",
    "js": "#009E73",
    "null": "#666666",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", type=Path, required=True)
    parser.add_argument("--constgold-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mag-max", type=float, default=22.0)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--permutations", type=int, default=200)
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
            # The input stores log(radius / pixel).  Adding log(pixel size)
            # yields log radius in arcsec without round-trip exponentiation.
            "log_radius": frame["measured_log_flux_radius"].to_numpy(float)
            + np.log(pixel_size),
        }
    )
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"non-finite measurements in {path}")
    return values


def pooled_quantile_edges(values, bins):
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    if np.unique(edges).size != edges.size:
        raise ValueError(f"cannot form {bins} distinct pooled-quantile bins")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def cell_indices(pool, features, bins):
    coordinates = []
    edges = {}
    for feature in features:
        edge = pooled_quantile_edges(pool[feature].to_numpy(float), bins)
        edges[feature] = edge
        coordinates.append(
            np.searchsorted(edge[1:-1], pool[feature].to_numpy(float), side="right")
        )
    cells = np.ravel_multi_index(tuple(coordinates), (bins,) * len(features))
    return cells, edges


def probabilities(counts, total_pseudocount=0.0):
    counts = np.asarray(counts, dtype=np.float64)
    if total_pseudocount:
        counts = counts + total_pseudocount / counts.size
    return counts / counts.sum()


def kl_divergence(p, q):
    active = p > 0.0
    return float(np.sum(p[active] * np.log(p[active] / q[active])))


def js_divergence_from_counts(counts_p, counts_q):
    p = probabilities(counts_p)
    q = probabilities(counts_q)
    middle = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, middle) + 0.5 * kl_divergence(q, middle)


def evaluate_resolution(
    cells,
    n_mock,
    n_cells,
    permutation_mock_indices,
    dirichlet_total,
):
    total_counts = np.bincount(cells, minlength=n_cells)
    mock_counts = np.bincount(cells[:n_mock], minlength=n_cells)
    constgold_counts = total_counts - mock_counts
    mock_probability = probabilities(mock_counts, dirichlet_total)
    constgold_probability = probabilities(constgold_counts, dirichlet_total)
    forward = kl_divergence(mock_probability, constgold_probability)
    reverse = kl_divergence(constgold_probability, mock_probability)
    js = js_divergence_from_counts(mock_counts, constgold_counts)

    null = np.empty(len(permutation_mock_indices), dtype=np.float64)
    for index, selected in enumerate(permutation_mock_indices):
        permuted_mock = np.bincount(cells[selected], minlength=n_cells)
        null[index] = js_divergence_from_counts(
            permuted_mock, total_counts - permuted_mock
        )
    quantiles = np.quantile(null, (0.025, 0.50, 0.975))
    return {
        "kl_mock_to_constgold_nats": forward,
        "kl_constgold_to_mock_nats": reverse,
        "js_nats": js,
        "js_null_q025_nats": float(quantiles[0]),
        "js_null_median_nats": float(quantiles[1]),
        "js_null_q975_nats": float(quantiles[2]),
        "js_excess_over_null_median_nats": float(js - quantiles[1]),
        "permutation_tail_fraction": float(
            (1 + np.count_nonzero(null >= js)) / (len(null) + 1)
        ),
        "mock_occupied_cells": int(np.count_nonzero(mock_counts)),
        "constgold_occupied_cells": int(np.count_nonzero(constgold_counts)),
        "pooled_occupied_cells": int(np.count_nonzero(total_counts)),
    }


def analyze(pool, n_mock, groups, bins_by_dimension, permutations, seed, alpha_total):
    rng = np.random.default_rng(seed)
    n_total = len(pool)
    permutation_mock_indices = np.empty((permutations, n_mock), dtype=np.int32)
    for index in range(permutations):
        permutation_mock_indices[index] = rng.permutation(n_total)[:n_mock]

    rows = []
    edges_summary = {}
    for dimension, feature_groups in groups.items():
        for features in feature_groups:
            group_key = "+".join(features)
            for bins in bins_by_dimension[dimension]:
                cells, edges = cell_indices(pool, features, bins)
                result = evaluate_resolution(
                    cells,
                    n_mock,
                    bins**dimension,
                    permutation_mock_indices,
                    alpha_total,
                )
                rows.append(
                    {
                        "dimension": dimension,
                        "features": group_key,
                        "bins_per_dimension": bins,
                        "total_cells": bins**dimension,
                        **result,
                    }
                )
                if bins == bins_by_dimension[dimension][-1]:
                    edges_summary[group_key] = {
                        feature: [
                            None if not np.isfinite(value) else float(value)
                            for value in edge
                        ]
                        for feature, edge in edges.items()
                    }
    return pd.DataFrame(rows), edges_summary


def plot_panels(results, dimension, feature_groups, output):
    n_columns = 1 if len(feature_groups) == 1 else (2 if dimension == 1 else 3)
    n_rows = int(np.ceil(len(feature_groups) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.2, 3.0 * n_rows + 1.0),
        squeeze=False,
    )
    for panel_index, (ax, features) in enumerate(zip(axes.ravel(), feature_groups)):
        key = "+".join(features)
        subset = results[
            (results["dimension"] == dimension) & (results["features"] == key)
        ].sort_values("bins_per_dimension")
        bins = subset["bins_per_dimension"].to_numpy()
        ax.fill_between(
            bins,
            subset["js_null_q025_nats"],
            subset["js_null_q975_nats"],
            color=COLORS["null"],
            alpha=0.18,
            linewidth=0.0,
            label="Permutation-null 95% interval",
        )
        ax.plot(
            bins,
            subset["js_null_median_nats"],
            color=COLORS["null"],
            linestyle=":",
            linewidth=1.2,
            label="Permutation-null median",
        )
        ax.plot(
            bins,
            subset["kl_mock_to_constgold_nats"],
            color=COLORS["forward"],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=r"$D_{KL}(\mathrm{mock}\Vert\mathrm{ConstGold})$",
        )
        ax.plot(
            bins,
            subset["kl_constgold_to_mock_nats"],
            color=COLORS["reverse"],
            marker="s",
            markersize=3.2,
            linestyle="--",
            linewidth=1.4,
            label=r"$D_{KL}(\mathrm{ConstGold}\Vert\mathrm{mock})$",
        )
        ax.plot(
            bins,
            subset["js_nats"],
            color=COLORS["js"],
            marker="^",
            markersize=3.5,
            linewidth=1.5,
            label="Jensen–Shannon",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(bins)
        ax.set_xticklabels([str(value) for value in bins])
        if len(feature_groups) > 1:
            ax.set_title(
                " × ".join(SHORT_LABELS[feature] for feature in features), fontsize=9
            )
        if panel_index // n_columns == n_rows - 1:
            ax.set_xlabel("Bins per dimension")
        if panel_index % n_columns == 0:
            ax.set_ylabel("Divergence (nats)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.ravel()[len(feature_groups) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        rf"Bright mock versus ConstGold divergence ($m<22$; "
        rf"{dimension}D histograms)",
        fontsize=11,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.09,
        top=0.78,
        hspace=0.42,
        wspace=0.30,
    )
    stem = {1: "marginal_1d", 2: "pairwise_2d", 4: "joint_4d"}[dimension]
    fig.savefig(output / f"divergence_vs_bins_{stem}.png", dpi=300)
    fig.savefig(output / f"divergence_vs_bins_{stem}.pdf")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.permutations < 1:
        raise ValueError("--permutations must be positive")
    if args.dirichlet_total <= 0:
        raise ValueError("--dirichlet-total must be positive")

    mock_parent = load_values(args.mock, args.pixel_size)
    constgold_parent = load_values(args.constgold_sample, args.pixel_size)
    mock = mock_parent.loc[mock_parent["mag"] < args.mag_max].reset_index(drop=True)
    constgold = constgold_parent.loc[
        constgold_parent["mag"] < args.mag_max
    ].reset_index(drop=True)
    if len(mock) == 0 or len(constgold) == 0:
        raise ValueError("magnitude cut emptied one sample")
    pool = pd.concat([mock, constgold], ignore_index=True)

    groups = {
        1: [(feature,) for feature in FEATURES],
        2: list(combinations(FEATURES, 2)),
        4: [FEATURES],
    }
    bins_by_dimension = {
        1: [4, 8, 16, 32, 64, 128, 256],
        2: [4, 8, 16, 32, 64],
        4: [2, 4, 8],
    }
    results, edges = analyze(
        pool,
        len(mock),
        groups,
        bins_by_dimension,
        args.permutations,
        args.seed,
        args.dirichlet_total,
    )
    args.output.mkdir(parents=True)
    results.to_csv(args.output / "divergence_vs_bins.csv", index=False)
    for dimension, feature_groups in groups.items():
        plot_panels(results, dimension, feature_groups, args.output)

    summary = {
        "comparison": "bright complete-likelihood mock vs bright ConstGold plus measurements",
        "selection": f"measured MAG_AUTO<{args.mag_max:g}",
        "sample_counts": {
            "mock_parent": int(len(mock_parent)),
            "constgold_parent": int(len(constgold_parent)),
            "mock_bright": int(len(mock)),
            "constgold_bright": int(len(constgold)),
        },
        "features": list(FEATURES),
        "radius_transform": "natural log of measured flux radius in arcsec",
        "binning": {
            "scheme": "separate pooled empirical-quantile edges at each nested power-of-two resolution",
            "bins_per_dimension": bins_by_dimension,
            "all_values_included": True,
            "outer_edges": "negative and positive infinity",
            "finest_edges": edges,
        },
        "divergence": {
            "units": "natural-log nats",
            "directed_kl_smoothing": (
                "symmetric Dirichlet pseudocount with fixed total concentration "
                f"{args.dirichlet_total:g} divided across all cells"
            ),
            "js_smoothing": "none; Jensen-Shannon divergence remains finite with empty cells",
        },
        "finite_sample_reference": {
            "method": "pool labels and permute at fixed observed sample sizes",
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "common_permutations_across_resolutions": True,
        },
        "inputs": {
            "mock": str(args.mock.resolve()),
            "mock_sha256": sha256(args.mock),
            "constgold_sample": str(args.constgold_sample.resolve()),
            "constgold_sample_sha256": sha256(args.constgold_sample),
        },
        "unmatched_rows": 0,
        "unmatched_note": "No catalogue join is used; both fixed samples are filtered independently.",
        "results": results.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
