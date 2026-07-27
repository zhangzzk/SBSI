#!/usr/bin/env python
"""Detection bias vs neighbour-flux severity (nbr_flux_near), one curve per true-mag bin.

Reads scripts/eval_detection_nbrflux.py --mag-edges output. Four curves (24.5-26.5), each det-bias
= (<e_int>_+ - <e_int>_-)/0.04 (sheared-intrinsic, R_both=1) vs nbr_flux_near; the leftmost point of
each curve (x=0) is ISOLATED (nbr_flux_near==0). Severity axis = the flow's summed 3" neighbour flux,
which -- unlike blendedness beta -- is not entangled with primary size (cont.163).

Login-node; PNG only, Okabe-Ito.
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, dark->light with magnitude
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
MARKERS = ["o", "s", "^", "D"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 12,
        "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10.5,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def main():
    ap = argparse.ArgumentParser()
    base = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
    ap.add_argument("--npz", default=base + "detection_nbrflux_magbins_v1.npz")
    ap.add_argument("--out", default="/home/z/Zekang.Zhang/SBSI-ablation/figures")
    args = ap.parse_args()
    set_style()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)
    bands = [str(b) for b in z["mag_bands"]]

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    for i, bl in enumerate(bands):
        ctr = z[f"{bl}__ctr"]
        db = 100 * z[f"{bl}__db"]
        se = 100 * z[f"{bl}__se"]
        if ctr.size == 0:
            continue
        c = COLORS[i % len(COLORS)]
        mk = MARKERS[i % len(MARKERS)]
        # blended points (ctr>0)
        m = ctr > 0
        ax.errorbar(ctr[m], db[m], yerr=se[m], color=c, marker=mk, ms=6, lw=1.9,
                    capsize=2.5, label=bl.replace("r", "r ="))
        # isolated point (ctr==0) as an open marker, connected with a dotted stub
        iso = ~m
        if iso.any():
            ax.errorbar(ctr[iso], db[iso], yerr=se[iso], color=c, marker=mk, ms=9,
                        mfc="white", mec=c, mew=1.8, lw=0, capsize=2.5, zorder=5)
            xs = np.concatenate([ctr[iso], ctr[m][:1]])
            ys = np.concatenate([db[iso], db[m][:1]])
            ax.plot(xs, ys, color=c, lw=1.0, ls=":", zorder=1)

    ax.axhline(0, color="0.5", lw=0.9)
    ax.axvline(0, color="0.75", lw=0.8, ls="--")
    ax.text(0.01, ax.get_ylim()[1], "  isolated\n  (open)", fontsize=8.5, color="0.4",
            va="top", ha="left")
    ax.set_xlabel(r"blending severity   $\mathrm{nbr\_flux\_near}=\log_{10}(1+F_{\rm nbr\,<3''}/\sigma_{\rm ap})$")
    ax.set_ylabel(r"detection bias   $(\langle e_{\rm int}\rangle_+ - \langle e_{\rm int}\rangle_-)/0.04$   [%]")
    ax.set_title("Stage-3 detection bias vs neighbour-flux severity, by true magnitude\n"
                 "(constgold, shear-free intrinsic shapes; open marker = isolated, $F_{\\rm nbr}=0$)",
                 fontsize=12)
    ax.legend(title="true primary magnitude", loc="lower left")
    fig.tight_layout()
    p = os.path.join(args.out, "fig_detection_nbrflux_magbins.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", p)
    print("DETECTION_NBRFLUX_MAGBINS_PLOT_DONE")


if __name__ == "__main__":
    main()
