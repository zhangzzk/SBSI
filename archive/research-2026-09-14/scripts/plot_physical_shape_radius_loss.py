#!/usr/bin/env python3
"""Plot the additive validation objective for shape+radius response training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def response_terms(path: Path) -> dict[str, float]:
    """Decompose the debiased joint quadratic into shape, radius, and cross terms."""
    with np.load(path) as data:
        means = np.asarray(data["group_means"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        precision = np.asarray(data["precision"], dtype=np.float64)
        reported = float(data["loss"])
    residual = means - target[None, :, :]
    center = residual.mean(axis=0)
    deviations = residual - center[None, :, :]
    repeats, _, components = means.shape
    covariance_of_mean = np.einsum(
        "rbi,rbj->bij", deviations, deviations
    ) / (repeats * (repeats - 1))

    def self_term(indices: slice) -> tuple[float, float, float]:
        block_center = center[:, indices]
        block_precision = precision[:, indices, indices]
        block_covariance = covariance_of_mean[:, indices, indices]
        quadratic = 0.5 * np.einsum(
            "bi,bij,bj->b", block_center, block_precision, block_center
        ).mean() / components
        integration = 0.5 * np.einsum(
            "bij,bji->b", block_precision, block_covariance
        ).mean() / components
        return float(quadratic - integration), float(quadratic), float(integration)

    shape, shape_quadratic, shape_integration = self_term(slice(0, 4))
    radius, radius_quadratic, radius_integration = self_term(slice(4, 6))
    cross_precision = precision[:, :4, 4:6]
    cross_quadratic = np.einsum(
        "bi,bij,bj->b", center[:, :4], cross_precision, center[:, 4:6]
    ).mean() / components
    cross_integration = np.einsum(
        "bij,bji->b", cross_precision, covariance_of_mean[:, 4:6, :4]
    ).mean() / components
    cross = float(cross_quadratic - cross_integration)
    reconstructed = shape + radius + cross
    if not np.isclose(reconstructed, reported, rtol=1e-11, atol=1e-11):
        raise ValueError(
            f"response decomposition mismatch for {path}: {reconstructed} != {reported}"
        )
    return {
        "shape": shape,
        "radius": radius,
        "shape_radius_cross": cross,
        "response": reported,
        "shape_mean_quadratic": shape_quadratic,
        "shape_integration_quadratic": shape_integration,
        "radius_mean_quadratic": radius_quadratic,
        "radius_integration_quadratic": radius_integration,
        "shape_radius_cross_mean_quadratic": float(cross_quadratic),
        "shape_radius_cross_integration_quadratic": float(cross_integration),
    }


def load_trajectory(root: Path) -> tuple[list[dict], float, list[float]]:
    history = json.loads((root / "training" / "history.json").read_text())
    records = history["history"]
    rows = []
    for record in records:
        epoch = int(record["epoch"])
        terms = response_terms(root / "training" / f"epoch{epoch:03d}_validation.npz")
        response_weight = (float(record["total"]) - float(record["nll"]["nll"])) / terms[
            "response"
        ]
        if not np.isclose(response_weight, 0.1, rtol=1e-10, atol=1e-12):
            raise ValueError(f"unexpected outer response weight at epoch {epoch}: {response_weight}")
        row = {
            "epoch": epoch,
            "nll": float(record["nll"]["nll"]),
            "nll_old_reference": float(record["nll"]["old_reference_nll"]),
            "total": float(record["total"]),
            **terms,
        }
        row["weighted_shape"] = response_weight * row["shape"]
        row["weighted_radius"] = response_weight * row["radius"]
        row["weighted_shape_radius_cross"] = response_weight * row["shape_radius_cross"]
        row["weighted_response"] = response_weight * row["response"]
        reconstructed = (
            row["nll"]
            + row["weighted_shape"]
            + row["weighted_radius"]
            + row["weighted_shape_radius_cross"]
        )
        if not np.isclose(reconstructed, row["total"], rtol=1e-11, atol=1e-11):
            raise ValueError(f"total objective mismatch at epoch {epoch}")
        rows.append(row)
    initialization = json.loads((root / "training" / "initialization.json").read_text())
    effective_weights = initialization["metadata"]["response_effective_component_weights"]
    return rows, response_weight, effective_weights


def plot(rows: list[dict], output_stem: Path, effective_weights: list[float]) -> None:
    epochs = np.array([row["epoch"] for row in rows])
    panels = [
        ("total", "Total validation objective", "#0072B2"),
        ("nll", "NLL", "#000000"),
        ("weighted_shape", r"Shape response term ($\lambda_e L_e$)", "#56B4E9"),
        ("weighted_radius", r"Radius response term ($\lambda_r L_r$)", "#E69F00"),
        (
            "weighted_shape_radius_cross",
            "Shape–radius covariance term",
            "#009E73",
        ),
        ("weighted_response", "Total weighted response", "#CC79A7"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)
    for label, (axis, (key, title, color)) in zip("ABCDEF", zip(axes.flat, panels)):
        values = np.array([row[key] for row in rows])
        axis.plot(epochs, values, color=color, marker="o", markersize=3.8, linewidth=1.8)
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Loss contribution")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.annotate(
            f"{values[-1]:.4g}",
            (epochs[-1], values[-1]),
            xytext=(-5, 7),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=color,
        )
    for axis in axes[-1]:
        axis.set_xlabel("Epoch (0 = NLL-only parent)")
    for axis in axes.flat:
        axis.set_xticks(epochs[::2])
    fig.suptitle(
        "Shape+radius response training: additive validation loss\n"
        rf"$\lambda_e={effective_weights[0]:.3g}$, "
        rf"$\lambda_r={effective_weights[4]:.5g}$; "
        "flux response excluded",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="physical_response_shape_radius_v1 root")
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_stem = args.output_stem or root / "training" / "validation_loss_terms"
    rows, response_weight, effective_weights = load_trajectory(root)
    plot(rows, output_stem, effective_weights)
    summary = {
        "definition": (
            "Fixed-subset validation objective; response is decomposed exactly into "
            "shape self, radius self, and shape-radius cross-covariance terms after "
            "finite-integration debiasing."
        ),
        "outer_response_weight": response_weight,
        "effective_component_weights": effective_weights,
        "rows": rows,
    }
    output_stem.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output_stem.with_suffix(".png"))
    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".json"))


if __name__ == "__main__":
    main()
