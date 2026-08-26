#!/usr/bin/env python3
"""Plot the persisted learning curve for a mixed-shear flow experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#6B7280"


def _trailing_mean(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    if width < 1 or width > values.size:
        raise ValueError(f"invalid smoothing width {width} for {values.size} epochs")
    smooth = np.convolve(values, np.ones(width) / width, mode="valid")
    epochs = np.arange(width, values.size + 1)
    return epochs, smooth


def _window_summary(values: np.ndarray, first_epoch: int, last_epoch: int) -> dict:
    epochs = np.arange(first_epoch, last_epoch + 1, dtype=float)
    selected = values[first_epoch - 1 : last_epoch]
    return {
        "first_epoch": int(first_epoch),
        "last_epoch": int(last_epoch),
        "mean": float(selected.mean()),
        "standard_deviation": float(selected.std(ddof=1)),
        "linear_slope_per_epoch": float(np.polyfit(epochs, selected, 1)[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-weight", type=float, default=450.0)
    parser.add_argument("--label", default="Experiment E (seed 501)")
    parser.add_argument("--smooth-width", type=int, default=5)
    args = parser.parse_args()

    with np.load(args.curve) as saved:
        required = {"train_nll", "val_nll", "val_resp", "swa_epochs"}
        missing = required.difference(saved.files)
        if missing:
            raise KeyError(f"missing arrays in {args.curve}: {sorted(missing)}")
        train_nll = saved["train_nll"].astype(float)
        val_nll = saved["val_nll"].astype(float)
        val_response = saved["val_resp"].astype(float)
        swa_epochs = saved["swa_epochs"].astype(int)

    if not (train_nll.shape == val_nll.shape == val_response.shape):
        raise ValueError("train_nll, val_nll, and val_resp must have matching shapes")
    epochs = np.arange(1, val_nll.size + 1)
    weighted_response = args.response_weight * val_response
    val_total = val_nll + weighted_response
    best_index = int(np.argmin(val_total))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.45), constrained_layout=True)

    def plot_raw_and_smooth(ax, values, color, linestyle, label):
        ax.plot(epochs, values, color=color, lw=0.8, alpha=0.28, linestyle=linestyle)
        smooth_epochs, smooth = _trailing_mean(values, args.smooth_width)
        ax.plot(
            smooth_epochs,
            smooth,
            color=color,
            lw=2.0,
            linestyle=linestyle,
            label=f"{label} ({args.smooth_width}-epoch mean)",
        )

    ax = axes[0]
    plot_raw_and_smooth(ax, train_nll, BLUE, "-", "Training NLL")
    plot_raw_and_smooth(ax, val_nll, ORANGE, "--", "Validation NLL")
    ax.axhline(0.0, color=GREY, lw=0.7, alpha=0.55)
    ax.set_title("a  Density-fit term")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log likelihood per object")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    plot_raw_and_smooth(ax, val_total, BLUE, "-", "Total validation loss")
    plot_raw_and_smooth(ax, weighted_response, ORANGE, "--", r"$450\,L_{\rm response}$")
    plot_raw_and_smooth(ax, val_nll, GREY, ":", "Validation NLL")
    ax.scatter(
        epochs[best_index],
        val_total[best_index],
        s=36,
        facecolor=GREEN,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label=f"Best total (epoch {epochs[best_index]})",
    )
    if swa_epochs.size:
        ax.axvspan(
            swa_epochs.min() - 0.5,
            swa_epochs.max() + 0.5,
            color=GREEN,
            alpha=0.10,
            lw=0,
            label=f"SWA epochs {swa_epochs.min()}–{swa_epochs.max()}",
        )
    ax.set_title("b  Checkpoint-selection objective")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss per object")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[2]
    tail_start = max(1, val_nll.size - 29)
    tail = epochs >= tail_start
    ax.plot(
        epochs[tail],
        val_total[tail],
        color=BLUE,
        marker="o",
        markersize=2.8,
        lw=1.0,
        alpha=0.45,
        label="Total validation loss",
    )
    ax.plot(
        epochs[tail],
        weighted_response[tail],
        color=ORANGE,
        marker="s",
        markersize=2.5,
        lw=1.0,
        linestyle="--",
        alpha=0.45,
        label=r"$450\,L_{\rm response}$",
    )
    smooth_epochs, smooth_total = _trailing_mean(val_total, args.smooth_width)
    _, smooth_response = _trailing_mean(weighted_response, args.smooth_width)
    smooth_tail = smooth_epochs >= tail_start
    ax.plot(smooth_epochs[smooth_tail], smooth_total[smooth_tail], color=BLUE, lw=2.0)
    ax.plot(
        smooth_epochs[smooth_tail],
        smooth_response[smooth_tail],
        color=ORANGE,
        lw=2.0,
        linestyle="--",
    )
    ax.scatter(
        epochs[best_index],
        val_total[best_index],
        s=36,
        facecolor=GREEN,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    if swa_epochs.size:
        ax.axvspan(
            swa_epochs.min() - 0.5,
            swa_epochs.max() + 0.5,
            color=GREEN,
            alpha=0.10,
            lw=0,
        )
    ax.set_title(f"c  Tail stability (epochs {tail_start}–{epochs[-1]})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss per object")
    ax.set_xlim(tail_start, epochs[-1])
    ax.legend(frameon=False, loc="upper right")

    for index, ax in enumerate(axes):
        if index < 2:
            ax.set_xlim(1, epochs[-1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    fig.suptitle(
        f"{args.label}: mixed-shear flow learning curve",
        fontsize=11,
        fontweight="semibold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    last_ten = max(1, val_nll.size - 9)
    previous_ten_first = max(1, last_ten - 10)
    previous_ten_last = last_ten - 1
    summary = {
        "label": args.label,
        "curve": str(args.curve.resolve()),
        "n_epochs": int(val_nll.size),
        "response_weight": float(args.response_weight),
        "objective_definition": "val_nll + response_weight * val_resp",
        "training_total_available": False,
        "training_total_note": "Per-epoch training response loss was not persisted.",
        "best_total": {
            "epoch": int(epochs[best_index]),
            "value": float(val_total[best_index]),
            "val_nll": float(val_nll[best_index]),
            "weighted_response": float(weighted_response[best_index]),
        },
        "final": {
            "epoch": int(epochs[-1]),
            "train_nll": float(train_nll[-1]),
            "val_nll": float(val_nll[-1]),
            "val_response": float(val_response[-1]),
            "weighted_response": float(weighted_response[-1]),
            "val_total": float(val_total[-1]),
        },
        "swa_epochs": swa_epochs.tolist(),
        "windows": {
            "previous_ten": {
                "val_total": _window_summary(
                    val_total, previous_ten_first, previous_ten_last
                ),
                "val_nll": _window_summary(
                    val_nll, previous_ten_first, previous_ten_last
                ),
                "weighted_response": _window_summary(
                    weighted_response, previous_ten_first, previous_ten_last
                ),
            },
            "last_ten": {
                "val_total": _window_summary(val_total, last_ten, val_nll.size),
                "val_nll": _window_summary(val_nll, last_ten, val_nll.size),
                "weighted_response": _window_summary(
                    weighted_response, last_ten, val_nll.size
                ),
            },
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
