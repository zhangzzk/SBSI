#!/usr/bin/env python3
"""Plot V2.2 constgold bias against absolute intrinsic neighbour flux by shell."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_v22_constgold_localization import Row, global_m, parse_section


SHELLS = [
    ("near 0--1 arcsec", r"Near: $0$–$1''$"),
    ("mid 1--3 arcsec", r"Mid: $1$–$3''$"),
    ("far 3--10 arcsec", r"Far: $3$–$10''$"),
]


def short_labels(rows: list[Row]) -> list[str]:
    labels = []
    for row in rows:
        if row.label == "zero flux":
            labels.append("zero")
        else:
            labels.append(row.label.split()[0].upper())
    return labels


def plot_panel(ax, rows: list[Row], title: str, panel: str) -> None:
    x = np.arange(len(rows))
    y = np.asarray([row.m_pct for row in rows])
    error = np.asarray([row.m_total_sem_pct for row in rows])
    color = "#0072B2"
    ax.axhline(0.0, color="0.30", lw=1.0, zorder=0)
    ax.axhline(global_m(rows), color="0.55", lw=1.0, ls="--", zorder=0)
    ax.errorbar(
        x, y, yerr=error, fmt="o-", color=color, ecolor=color, lw=1.8,
        elinewidth=1.2, capsize=3, ms=5.5, mfc="white", mew=1.5,
    )
    ax.set_xticks(x, short_labels(rows))
    ax.set_xlabel("Absolute neighbour-flux class")
    ax.set_title(title, fontsize=10)
    ax.text(0.01, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def write_csv(path: Path, groups: dict[str, list[Row]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["shell", "bin", "R_sim", "R_flow", "R_blend", "m_pct",
             "m_seed_sem_pct", "m_sim_sem_pct", "m_total_sem_pct", "N"]
        )
        for shell, rows in groups.items():
            for row in rows:
                writer.writerow(
                    [shell, row.label, row.r_sim, row.r_flow, row.r_blend, row.m_pct,
                     row.m_seed_sem_pct, row.m_sim_sem_pct, row.m_total_sem_pct, row.n]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/v22_constgold_true_neighbor_flux"),
    )
    args = parser.parse_args()
    text = args.log.read_text()
    if "Flux mode: absolute" not in text:
        raise RuntimeError("input log is not the absolute-flux diagnostic")

    groups = {
        shell: parse_section(text, f"[by intrinsic neighbour flux: {shell}]")
        for shell, _ in SHELLS
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.55), sharey=True)
    for panel, ax, (shell, title) in zip("ABC", axes, SHELLS):
        plot_panel(ax, groups[shell], title, panel)
    axes[0].set_ylabel(r"Constgold multiplicative bias  $m$  (%)")

    all_y = np.concatenate([[row.m_pct for row in rows] for rows in groups.values()])
    all_e = np.concatenate([[row.m_total_sem_pct for row in rows] for rows in groups.values()])
    low = min(-0.5, float(np.min(all_y - all_e)))
    high = max(1.8, float(np.max(all_y + all_e)))
    pad = 0.12 * (high - low)
    axes[0].set_ylim(low - pad, high + pad)
    axes[-1].annotate(
        "global $m$",
        xy=(0.98, global_m(groups[SHELLS[-1][0]])),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 4), textcoords="offset points", ha="right", va="bottom",
        color="0.40", fontsize=8,
    )

    fig.suptitle("V2.2 constgold residual versus true total neighbour flux", fontsize=11, y=0.995)
    fig.text(
        0.5, 0.005,
        r"Each shell sums every intrinsic input neighbour except the primary; bins use "
        r"$\log_{10}(1+F_{\rm nbr})$ (no division by primary flux). "
        r"$m=R_{\rm sim}/(R_{\rm flow}+R_{\rm blend})-1$; errors = seed SEM ⊕ case-blocked simulation SEM.",
        ha="center", va="bottom", fontsize=7.7, color="0.30",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95), w_pad=1.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    write_csv(args.output.with_suffix(".csv"), groups)
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {args.output.with_suffix('.pdf')}")
    print(f"wrote {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
