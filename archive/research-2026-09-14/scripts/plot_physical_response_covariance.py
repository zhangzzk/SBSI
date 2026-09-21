#!/usr/bin/env python
"""Plot train/validation structure of the physical-response covariance.

The response covariance is cell-specific and mixes dimensionless shape,
pixel-radius, and instrumental-flux units.  Correlation matrices therefore
provide the interpretable unit-invariant view for deciding whether response
blocks may be separated.  Canonical correlations summarize multivariate
coupling between the shape, radius, and flux blocks without fitting a model.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


LABELS = (
    r"$e_1:g_1$",
    r"$e_1:g_2$",
    r"$e_2:g_1$",
    r"$e_2:g_2$",
    r"$r_{\parallel}$",
    r"$r_{\times}$",
    r"$F_{\parallel}$",
    r"$F_{\times}$",
)
BLOCKS = {
    "shape-radius": (np.arange(4), np.arange(4, 6)),
    "shape-flux": (np.arange(4), np.arange(6, 8)),
    "radius-flux": (np.arange(4, 6), np.arange(6, 8)),
}
SPLIT_COLORS = {"Training": "#0072B2", "Validation": "#D55E00"}


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_correlations(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        covariance = np.asarray(payload["covariance"], dtype=np.float64)
        precision = np.asarray(payload["precision"], dtype=np.float64)
    if covariance.ndim != 3 or covariance.shape[1:] != (8, 8):
        raise ValueError(f"{path} does not contain cellwise 8x8 covariance")
    if precision.shape != covariance.shape or not (
        np.isfinite(covariance).all() and np.isfinite(precision).all()
    ):
        raise ValueError(f"{path} has invalid covariance or precision")
    identity_error = np.max(
        np.abs(np.einsum("bij,bjk->bik", precision, covariance) - np.eye(8))
    )
    if identity_error > 1e-7:
        raise ValueError(f"precision/covariance mismatch: {identity_error:g}")
    standard_deviation = np.sqrt(
        np.diagonal(covariance, axis1=1, axis2=2)
    )
    correlation = (
        covariance
        / standard_deviation[:, :, None]
        / standard_deviation[:, None, :]
    )
    if np.max(np.abs(correlation)) > 1 + 1e-10:
        raise ValueError("invalid response correlations")
    return correlation


def inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0):
        raise ValueError("canonical correlation requires positive-definite blocks")
    return (vectors * (1 / np.sqrt(values))) @ vectors.T


def canonical_correlation(
    correlation: np.ndarray, left: np.ndarray, right: np.ndarray
) -> float:
    within_left = correlation[np.ix_(left, left)]
    within_right = correlation[np.ix_(right, right)]
    between = correlation[np.ix_(left, right)]
    whitened = inverse_sqrt(within_left) @ between @ inverse_sqrt(within_right)
    return float(np.linalg.svd(whitened, compute_uv=False)[0])


def summarize(correlation: np.ndarray) -> dict:
    result = {
        "n_cells": int(len(correlation)),
        "median_correlation": np.median(correlation, axis=0).tolist(),
        "q90_absolute_correlation": np.quantile(
            np.abs(correlation), 0.9, axis=0
        ).tolist(),
        "blocks": {},
    }
    for label, (left, right) in BLOCKS.items():
        elementwise = np.abs(correlation[:, left][:, :, right]).reshape(-1)
        canonical = np.array(
            [canonical_correlation(cell, left, right) for cell in correlation]
        )
        result["blocks"][label] = {
            "absolute_elementwise_quantiles_50_90_95_99_max": np.quantile(
                elementwise, [0.5, 0.9, 0.95, 0.99, 1]
            ).tolist(),
            "maximum_canonical_quantiles_50_90_95_99_max": np.quantile(
                canonical, [0.5, 0.9, 0.95, 0.99, 1]
            ).tolist(),
            "maximum_canonical_by_cell": canonical.tolist(),
        }
    return result


def annotate_heatmap(ax, values: np.ndarray, *, absolute: bool) -> None:
    for row in range(8):
        for column in range(8):
            value = values[row, column]
            if absolute:
                dark = value < 0.36 or value > 0.82
            else:
                dark = abs(value) > 0.55
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if dark else "black",
            )


def heatmap(ax, values: np.ndarray, title: str, *, absolute: bool):
    if absolute:
        image = ax.imshow(values, cmap="cividis", vmin=0, vmax=1)
    else:
        image = ax.imshow(values, cmap="PuOr", vmin=-1, vmax=1)
    annotate_heatmap(ax, values, absolute=absolute)
    ax.set_xticks(range(8), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(8), LABELS)
    ax.set_title(title, pad=8)
    ax.tick_params(length=0)
    for boundary in (3.5, 5.5):
        ax.axhline(boundary, color="black", linewidth=1.4)
        ax.axvline(boundary, color="black", linewidth=1.4)
    return image


def plot(train: np.ndarray, validation: np.ndarray, output: Path) -> None:
    summaries = {
        "Training": summarize(train),
        "Validation": summarize(validation),
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(13.2, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1, 1, 0.86))
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]
    signed_images = []
    absolute_images = []
    for column, (name, correlation) in enumerate(
        (("Training", train), ("Validation", validation))
    ):
        signed_images.append(
            heatmap(
                axes[column],
                np.median(correlation, axis=0),
                f"{name}: median correlation",
                absolute=False,
            )
        )
        absolute_images.append(
            heatmap(
                axes[2 + column],
                np.quantile(np.abs(correlation), 0.9, axis=0),
                f"{name}: 90th percentile |correlation|",
                absolute=True,
            )
        )
    figure.colorbar(
        signed_images[0], ax=axes[:2], location="bottom", shrink=0.72,
        pad=0.08, label="Correlation",
    )
    figure.colorbar(
        absolute_images[0], ax=axes[2:], location="bottom", shrink=0.72,
        pad=0.08, label="Absolute correlation",
    )

    coupling_ax = figure.add_subplot(grid[:, 2])
    rng = np.random.default_rng(20260910)
    positions = np.arange(len(BLOCKS))
    for split_index, split in enumerate(("Training", "Validation")):
        offset = -0.13 if split_index == 0 else 0.13
        color = SPLIT_COLORS[split]
        for block_index, block in enumerate(BLOCKS):
            values = np.asarray(
                summaries[split]["blocks"][block]["maximum_canonical_by_cell"]
            )
            jitter = rng.uniform(-0.055, 0.055, size=len(values))
            y = positions[block_index] + offset + jitter
            coupling_ax.scatter(
                values, y, s=8, alpha=0.18, color=color, linewidths=0,
                rasterized=True,
            )
            median, q10, q90 = np.quantile(values, [0.5, 0.1, 0.9])
            coupling_ax.plot([q10, q90], [positions[block_index] + offset] * 2,
                             color=color, linewidth=2.2, solid_capstyle="round")
            coupling_ax.plot(median, positions[block_index] + offset, marker="o",
                             color=color, markeredgecolor="white", markersize=5)
            coupling_ax.text(
                min(0.985, q90 + 0.025), positions[block_index] + offset,
                f"{median:.2f}", color=color, va="center", fontsize=7,
            )
    coupling_ax.set_yticks(positions, [name.replace("-", "–") for name in BLOCKS])
    coupling_ax.set_xlim(0, 1.0)
    coupling_ax.set_ylim(len(BLOCKS) - 0.55, -0.55)
    coupling_ax.set_xlabel(r"Maximum canonical correlation $\rho_{\max}$")
    coupling_ax.set_title("Cellwise multivariate block coupling", pad=8)
    coupling_ax.grid(axis="x", color="0.88", linewidth=0.7)
    coupling_ax.spines[["top", "right", "left"]].set_visible(False)
    coupling_ax.tick_params(axis="y", length=0)
    coupling_ax.legend(
        handles=[
            Line2D([0], [0], color=color, marker="o", label=split)
            for split, color in SPLIT_COLORS.items()
        ],
        frameon=False,
        loc="lower right",
    )
    coupling_ax.text(
        0.02,
        0.02,
        "Points: 180 cells\nLines: 10th–90th percentile\nLabels: median",
        transform=coupling_ax.transAxes,
        va="bottom",
        fontsize=7,
        color="0.3",
    )
    for label, ax in zip("ABCDE", [*axes, coupling_ax]):
        ax.text(
            -0.19,
            1.13,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )
    figure.suptitle(
        "Physical-response covariance structure",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        "Cellwise case-jackknife covariance, normalized to correlation. "
        "Black lines delimit shape (4), radius (2), and flux (2) blocks.",
        ha="center",
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    report = {
        "definition": (
            "Cellwise simulation case-jackknife covariance normalized to "
            "correlation; canonical correlations use marginal response blocks."
        ),
        "labels": list(LABELS),
        "block_order": list(BLOCKS),
        "splits": summaries,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output stem; writes PNG, PDF, and JSON")
    args = parser.parse_args()
    train = load_correlations(args.train_targets)
    validation = load_correlations(args.validation_targets)
    plot(train, validation, args.output)
    metadata = {
        "train_targets": str(args.train_targets.resolve()),
        "train_sha256": digest(args.train_targets),
        "validation_targets": str(args.validation_targets.resolve()),
        "validation_sha256": digest(args.validation_targets),
    }
    metadata_path = args.output.with_name(args.output.name + "_inputs.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
