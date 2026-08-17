#!/usr/bin/env python
"""Detection response vs blendedness BETA: SIM vs MODEL, faceted by SIZE and by MAG (cont.165).

Two panels. Left: curves per true-size bin (fixed mag band). Right: curves per true-mag bin (pool size).
Solid+filled = constgold SIM ((<e_int>_+ - <e_int>_-)/0.04, with error bars); dashed+open = detection-
classifier MODEL prediction. beta computed for ALL galaxies (isolated -> beta=0, leftmost bin).

Reads scripts/eval_detection_beta_bysize.py (sim) + scripts/eval_model_beta_bysize.py (model).
Login-node; PNG only, Okabe-Ito.
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

COLORS = ["#56B4E9", "#009E73", "#F0E442", "#E69F00", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 12,
        "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 9,
        "xtick.labelsize": 10, "ytick.labelsize": 11, "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def beta_ticklabels(bedges):
    lab = ["0\n(iso)"]
    for i in range(len(bedges) - 1):
        c = np.sqrt(bedges[i] * bedges[i + 1]) if bedges[i] > 0 else bedges[i + 1] / 2
        lab.append(f"{c:.1e}".replace("e-0", "e-"))
    return lab


def plot_facet(ax, zs, zm, name, title, legend_title, prettify):
    labels = [str(s) for s in zs[f"{name}__labels"]]
    bedges = zs[f"{name}__beta_edges"]
    nb = len(bedges)  # x positions: 0=iso, 1..n_beta
    x = np.arange(nb)
    for i, lab in enumerate(labels):
        c = COLORS[i % len(COLORS)]; mk = MARKERS[i % len(MARKERS)]
        db_s = 100 * zs[f"{name}__{lab}__db"]; se_s = 100 * zs[f"{name}__{lab}__se"]
        mk_key = f"{name}__{lab}__db_model"
        db_m = 100 * zm[mk_key] if mk_key in zm.files else np.full(nb, np.nan)
        ax.errorbar(x, db_s, yerr=se_s, color=c, marker=mk, ms=6, lw=1.9, capsize=2.5,
                    label=prettify(lab), zorder=3)
        ax.plot(x, db_m, color=c, marker=mk, ms=6.5, lw=1.6, ls="--", mfc="white", mec=c, zorder=2)
    ax.axhline(0, color="0.5", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(beta_ticklabels(bedges))
    ax.set_xlabel(r"blendedness $\beta$  (0 = isolated,  $\beta$ increases $\rightarrow$)")
    ax.set_title(title)
    ax.add_artist(ax.legend(title=legend_title, loc="lower left"))


def main():
    ap = argparse.ArgumentParser()
    base = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
    ap.add_argument("--sim", default=base + "detection_beta_bysize_v2.npz")
    ap.add_argument("--model", default=base + "model_beta_bysize_v2.npz")
    ap.add_argument("--out", default="/home/z/Zekang.Zhang/SBSI-ablation/figures")
    args = ap.parse_args()
    set_style()
    os.makedirs(args.out, exist_ok=True)
    zs = np.load(args.sim, allow_pickle=True)
    zm = np.load(args.model, allow_pickle=True)
    lo, hi = float(zs["mag_lo"]), float(zs["mag_hi"])

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15.0, 6.3), sharey=True)
    plot_facet(axl, zs, zm, "SIZE",
               f"by true SIZE  (fixed mag {lo:g}–{hi:g})",
               "true size bin",
               lambda s: s.replace("Re", "$R_e$ "))
    plot_facet(axr, zs, zm, "MAG",
               "by true MAGNITUDE  (pooling size)",
               "true r-mag bin",
               lambda s: s.replace("r", "r "))
    axl.set_ylabel(r"detection response  $(\langle e_{\rm int}\rangle_+-\langle e_{\rm int}\rangle_-)/0.04$  [%]")

    sim_h = mlines.Line2D([], [], color="0.3", marker="o", ls="-", label="sim (constgold)")
    mod_h = mlines.Line2D([], [], color="0.3", marker="o", mfc="white", ls="--",
                          label="model (classifier)")
    axr.legend(handles=[sim_h, mod_h], loc="upper right")

    fig.suptitle("Stage-3 detection response vs blendedness: SIM vs MODEL "
                 "(constgold shear-free intrinsic; classifier = cont.160 response-regularized)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_detection_beta_bysize.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)
    print("DETECTION_BETA_BYSIZE_SIMVSMODEL_PLOT_DONE")


if __name__ == "__main__":
    main()
