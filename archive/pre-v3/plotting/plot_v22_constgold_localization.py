#!/usr/bin/env python3
"""Plot the accepted V2.2 constgold closure residual across three physical axes.

The accepted diagnostics predate this figure and wrote their sufficient statistics to
Slurm logs.  Reading those logs avoids an expensive replay of sixteen 27-million-row
per-object dumps and guarantees that the plotted values are the ones recorded in the
work log.

Every panel shows the same standard constgold quantity

    m = 100 * [R_sim / (R_flow + R_blend) - 1].

The diagnostic logs report uncertainty for the algebraically related flow-demand residual.
This script propagates its flow-seed and case-blocked simulation SEMs into m, then combines
them in quadrature.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    label: str
    r_sim: float
    r_flow: float
    r_blend: float
    needed: float
    residual_pct: float
    seed_sem_pct: float
    sim_sem_pct: float
    n: int

    @property
    def total_sem_pct(self) -> float:
        return float(np.hypot(self.seed_sem_pct, self.sim_sem_pct))

    @property
    def m_pct(self) -> float:
        return 100.0 * (self.r_sim / (self.r_flow + self.r_blend) - 1.0)

    @property
    def m_seed_sem_pct(self) -> float:
        # residual_seed_sem/100 = sigma(R_flow) / needed
        sigma_r_flow = self.seed_sem_pct / 100.0 * self.needed
        total = self.r_flow + self.r_blend
        return 100.0 * self.r_sim / total**2 * sigma_r_flow

    @property
    def m_sim_sem_pct(self) -> float:
        # residual_sim_sem/100 = R_flow * sigma(R_sim) / needed**2
        sigma_r_sim = self.sim_sem_pct / 100.0 * self.needed**2 / self.r_flow
        return 100.0 / (self.r_flow + self.r_blend) * sigma_r_sim

    @property
    def m_total_sem_pct(self) -> float:
        return float(np.hypot(self.m_seed_sem_pct, self.m_sim_sem_pct))


def parse_section(text: str, heading: str) -> list[Row]:
    """Parse one table emitted by diag_rflow_v21/diag_v22_blendness."""
    try:
        tail = text.split(heading, 1)[1]
    except IndexError as exc:
        raise RuntimeError(f"section not found: {heading}") from exc

    rows: list[Row] = []
    for line in tail.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("[") and rows and not stripped[1:2].isdigit():
            break
        if stripped.startswith("direct total-model bias"):
            break
        fields = stripped.split()
        if len(fields) < 10 or not fields[-2].endswith("s"):
            continue
        try:
            r_sim, r_flow, r_blend, needed = map(float, fields[-9:-5])
            residual, seed_sem, sim_sem = map(float, fields[-5:-2])
            n = int(fields[-1].replace(",", ""))
        except ValueError:
            continue
        rows.append(
            Row(
                label=" ".join(fields[:-9]),
                r_sim=r_sim,
                r_flow=r_flow,
                r_blend=r_blend,
                needed=needed,
                residual_pct=residual,
                seed_sem_pct=seed_sem,
                sim_sem_pct=sim_sem,
                n=n,
            )
        )
    if not rows:
        raise RuntimeError(f"no rows parsed from section: {heading}")
    return rows


def global_m(rows: list[Row]) -> float:
    weights = np.asarray([r.n for r in rows], dtype=float)
    r_sim = np.average([r.r_sim for r in rows], weights=weights)
    r_flow = np.average([r.r_flow for r in rows], weights=weights)
    r_blend = np.average([r.r_blend for r in rows], weights=weights)
    return 100.0 * (r_sim / (r_flow + r_blend) - 1.0)


def clean_labels(rows: list[Row], kind: str) -> list[str]:
    labels = [r.label for r in rows]
    if kind == "mag":
        # The final nominal [25.5,26.0) bin is truncated by the V2.2 r<25.8 domain.
        labels[-1] = "25.5–25.8"
        return [s.strip("[])").replace(",", "–") for s in labels]
    if kind == "size":
        return [s.strip("[])").replace(",", "–") for s in labels]
    return ["low\n$R_{blend}<0.02$", "Q1", "Q2", "Q3", "Q4"]


def plot_panel(ax, rows: list[Row], labels: list[str], xlabel: str, panel: str) -> None:
    x = np.arange(len(rows))
    y = np.asarray([r.m_pct for r in rows])
    err = np.asarray([r.m_total_sem_pct for r in rows])
    color = "#0072B2"  # Okabe-Ito blue

    ax.axhline(0.0, color="0.30", lw=1.0, zorder=0)
    ax.axhline(global_m(rows), color="0.55", lw=1.0, ls="--", zorder=0)
    ax.errorbar(
        x,
        y,
        yerr=err,
        fmt="o-",
        color=color,
        ecolor=color,
        lw=1.8,
        elinewidth=1.2,
        capsize=3,
        ms=5.5,
        mfc="white",
        mew=1.5,
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel(xlabel)
    ax.text(0.01, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")


def write_csv(path: Path, groups: dict[str, list[Row]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["axis", "bin", "R_sim", "R_flow", "R_blend", "needed", "flow_residual_pct",
             "flow_seed_sem_pct", "flow_sim_sem_pct", "flow_total_sem_pct", "m_pct",
             "m_seed_sem_pct", "m_sim_sem_pct", "m_total_sem_pct", "N"]
        )
        for axis, rows in groups.items():
            for row in rows:
                writer.writerow(
                    [axis, row.label, row.r_sim, row.r_flow, row.r_blend, row.needed,
                     row.residual_pct, row.seed_sem_pct, row.sim_sem_pct, row.total_sem_pct,
                     row.m_pct, row.m_seed_sem_pct, row.m_sim_sem_pct, row.m_total_sem_pct, row.n]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--properties-log",
        type=Path,
        default=Path("/home/z/Zekang.Zhang/logs/rfv22_15595422.out"),
    )
    parser.add_argument(
        "--blendness-log",
        type=Path,
        default=Path("/home/z/Zekang.Zhang/logs/v22blend_15597424.out"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v22_constgold_localization"),
        help="output stem; writes .png, .pdf, and .csv",
    )
    args = parser.parse_args()

    properties = args.properties_log.read_text()
    blendness = args.blendness_log.read_text()
    size = parse_section(properties, "[by PRIMARY TRUE SIZE, fine, spanning the Re = 0.5 cut]")
    mag = parse_section(properties, "[by PRIMARY TRUE MAG]")
    blend = parse_section(blendness, "[by EMULATOR TOTAL R_blend]")
    groups = {"primary magnitude": mag, "primary size": size, "blendness": blend}

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
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.65), sharey=True)
    plot_panel(axes[0], mag, clean_labels(mag, "mag"), "Primary true magnitude $r$", "A")
    plot_panel(axes[1], size, clean_labels(size, "size"),
               "Primary true size $R_e$ (arcsec)", "B")
    plot_panel(axes[2], blend, clean_labels(blend, "blend"),
               "Emulator total blend response", "C")
    axes[0].set_ylabel(r"Constgold multiplicative bias  $m$  (%)")
    axes[0].set_ylim(-7.0, 8.5)
    axes[2].annotate(
        "global residual",
        xy=(0.98, global_m(blend)),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="0.40",
        fontsize=8,
    )
    fig.suptitle("V2.2 constgold localization on its own domain", fontsize=11, y=0.995)
    fig.text(
        0.5,
        0.005,
        r"$m=R_{\rm sim}/(R_{\rm flow}+R_{\rm blend})-1$; 16 flow seeds; constgold cases 40–139; "
        "errors = seed SEM ⊕ case-blocked simulation SEM.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.30",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), w_pad=1.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    write_csv(args.output.with_suffix(".csv"), groups)
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {args.output.with_suffix('.pdf')}")
    print(f"wrote {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
