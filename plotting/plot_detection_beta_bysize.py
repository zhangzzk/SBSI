#!/usr/bin/env python
"""Does binning by SIZE rescue the blendedness-beta severity axis?

Reads scripts/eval_detection_beta_bysize.py output (fixed mag band, size bins x beta bins).
Left panel : det-bias vs beta-severity rank (0 = isolated), one curve per true-size bin.
Right panel: <Re> vs the same rank -- the diagnostic. If size is fully controlled, <Re> is flat
             along a curve and beta behaves; where <Re> still climbs with beta (the wide large-Re
             bin), beta is still sorting on size and the confound persists.

Login-node; PNG only, Okabe-Ito.
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#56B4E9", "#E69F00", "#D55E00"]  # small -> large (light -> dark)
MARKERS = ["o", "s", "^"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11.5,
        "axes.titlesize": 12.5, "axes.labelsize": 11.5, "legend.fontsize": 10,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def main():
    ap = argparse.ArgumentParser()
    base = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
    ap.add_argument("--npz", default=base + "detection_beta_bysize_v1.npz")
    ap.add_argument("--out", default="/home/z/Zekang.Zhang/SBSI-ablation/figures")
    args = ap.parse_args()
    set_style()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)
    labels = [str(s) for s in z["size_labels"]]
    lo, hi = float(z["mag_lo"]), float(z["mag_hi"])

    fig, (axb, axr) = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for i, lab in enumerate(labels):
        beta = z[f"{lab}__beta"]; db = 100 * z[f"{lab}__db"]; se = 100 * z[f"{lab}__se"]
        Re = z[f"{lab}__Re"]
        if beta.size == 0:
            continue
        rank = np.arange(beta.size)  # 0 = isolated, 1.. = increasing beta
        c = COLORS[i % len(COLORS)]; mk = MARKERS[i % len(MARKERS)]
        reff = Re[1:].mean() if Re.size > 1 else Re[0]
        lg = lab.replace("Re", "Re ") + f"  ($\\langle R_e\\rangle{{\\approx}}{reff:.2f}''$)"
        # blended points (rank>=1) solid, isolated (rank 0) open
        axb.errorbar(rank[1:], db[1:], yerr=se[1:], color=c, marker=mk, ms=6, lw=1.8,
                     capsize=2.5, label=lg)
        axb.errorbar(rank[:1], db[:1], yerr=se[:1], color=c, marker=mk, ms=9, mfc="white",
                     mec=c, mew=1.8, lw=0, capsize=2.5, zorder=5)
        axb.plot(rank[:2], db[:2], color=c, lw=1.0, ls=":", zorder=1)
        axr.plot(rank[1:], Re[1:], color=c, marker=mk, ms=6, lw=1.8)
        axr.plot(rank[:1], Re[:1], color=c, marker=mk, ms=9, mfc="white", mec=c, mew=1.8, lw=0)
        axr.plot(rank[:2], Re[:2], color=c, lw=1.0, ls=":")

    for a in (axb, axr):
        a.set_xlabel(r"$\beta$ severity bin   (0 = isolated;  $\beta$ increases $\rightarrow$)")
    axb.axhline(0, color="0.5", lw=0.9)
    axb.set_ylabel(r"detection bias  $(\langle e_{\rm int}\rangle_+ - \langle e_{\rm int}\rangle_-)/0.04$  [%]")
    axb.set_title("det-bias vs $\\beta$, per size bin")
    axb.legend(title=f"true size bin  (mag {lo}–{hi})", loc="lower left")
    axr.set_ylabel(r"$\langle R_e\rangle$ in bin  [arcsec]")
    axr.set_title("size along each curve (the confound check)")
    axr.text(0.5, 0.96, "flat = size controlled;  rising = $\\beta$ still tracks size",
             transform=axr.transAxes, ha="center", va="top", fontsize=9, color="0.4")

    fig.suptitle("Does binning by size rescue $\\beta$?  (constgold, shear-free intrinsic shapes)  —  "
                 "clean in narrow bins, but the wide large-$R_e$ bin still sorts on size",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_detection_beta_bysize.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)
    print("DETECTION_BETA_BYSIZE_PLOT_DONE")


if __name__ == "__main__":
    main()
