#!/usr/bin/env python
"""Detection bias in constgold (Stage-3): the real observed per-leg catalogue vs the both-detected
reference, det-bias = R_full/R_both - 1, from scripts/eval_detection_constgold.py --output.

Left panel  : det-bias vs TRUE r-mag bin (measured shapes), dots+error, for ALL / ISOLATED / BLENDED.
              Shows the faint-end structure and the sign flip past the detection limit; the mag<26
              sample-definition edge is marked, and the beyond-sample regime (mag>26) is shaded.
Right panel : overall det-bias, MEASURED vs SHEARED-INTRINSIC (noise-free) per scope. The gap between
              the two = the fraction of the true detection-shape selection that measurement noise
              dilutes away (full for isolated, partial for blends).

Login-node; PNG only, Okabe-Ito.
"""
from __future__ import annotations
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito
C = {"ALL": "#000000", "ISOLATED": "#0072B2", "BLENDED": "#D55E00"}
MK = {"ALL": "o", "ISOLATED": "s", "BLENDED": "^"}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
                                     "detection_constgold_v1.npz")
    ap.add_argument("--out", default="/home/z/Zekang.Zhang/SBSI-ablation/figures")
    args = ap.parse_args()
    set_style()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)
    scopes = [s for s in ["ALL", "ISOLATED", "BLENDED"]]

    fig, (axb, axo) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    # ---------- Left: det-bias vs true-mag bin ----------
    YCAP = 4.0   # cap so the faint-end +15% spike doesn't squash the -1..-2% structure
    for s in scopes:
        lo, hi = z[f"{s}_bin_lo"], z[f"{s}_bin_hi"]
        xc = 0.5 * (lo + np.minimum(hi, lo + 1.0))   # midpoint (cap open [27,30) width for display)
        db = 100 * z[f"{s}_bin_db"]; dbe = 100 * z[f"{s}_bin_dbe"]
        vis = db <= YCAP
        axb.errorbar(xc[vis], db[vis], yerr=dbe[vis], fmt=MK[s] + "-", color=C[s], ms=6, lw=1.5,
                     capsize=3, label=s.capitalize())
        # off-scale points -> arrow annotation
        for x, y in zip(xc[~vis], db[~vis]):
            axb.annotate(f"{y:+.0f}%", xy=(x, YCAP), xytext=(x, YCAP - 0.9),
                         ha="center", fontsize=8, color=C[s],
                         arrowprops=dict(arrowstyle="->", color=C[s], lw=1.2))
    axb.axhline(0, color="0.6", lw=0.8, ls=":")
    axb.axvline(26.0, color="0.5", lw=1.0, ls="--")
    x0, x1 = axb.get_xlim()
    axb.axvspan(26.0, x1, color="0.9", alpha=0.6, lw=0, zorder=0)
    axb.text(26.05, YCAP * 0.86, "beyond mag<26\nsample def.", fontsize=7.8, color="0.4", va="top")
    axb.set_ylim(top=YCAP)
    axb.set_xlabel("true r-mag bin (midpoint)")
    axb.set_ylabel(r"detection bias  $R_{\rm full}/R_{\rm both}-1$  [%]")
    axb.set_title("Detection bias vs true magnitude")
    axb.legend(loc="lower left")

    # ---------- Right: measured vs sheared-intrinsic per scope ----------
    xpos = np.arange(len(scopes))
    for i, s in enumerate(scopes):
        Rb_m, Rf_m, db_m, se_m = z[f"{s}_meas"]
        Rb_i, Rf_i, db_i, se_i = z[f"{s}_shint"]
        # connector
        axo.plot([i, i], [100 * db_i, 100 * db_m], color="0.7", lw=1.2, zorder=1)
        axo.errorbar(i, 100 * db_m, yerr=100 * se_m, fmt="o", color=C[s], ms=9, capsize=4,
                     zorder=3, label="measured (biases shear)" if i == 0 else None)
        axo.errorbar(i, 100 * db_i, yerr=100 * se_i, fmt="D", mfc="white", mec=C[s], mew=1.8,
                     color=C[s], ms=8, capsize=4, zorder=3,
                     label="sheared-intrinsic (noise-free)" if i == 0 else None)
    axo.axhline(0, color="0.6", lw=0.8, ls=":")
    axo.set_xticks(xpos); axo.set_xticklabels([s.capitalize() for s in scopes])
    axo.set_ylabel(r"detection bias  $R_{\rm full}/R_{\rm both}-1$  [%]")
    axo.set_title("Noise dilution: measured vs true-shape selection")
    axo.legend(loc="lower left")
    axo.set_xlim(-0.5, len(scopes) - 0.5)

    Rb_all, Rf_all, db_all, se_all = z["ALL_meas"]
    fig.suptitle("Constgold DETECTION bias (Stage-3): overall measured "
                 f"{100*db_all:+.2f}% $\\pm$ {100*se_all:.2f}%  "
                 "— blend-dominated, faint-end; single-leg = 0.85%/leg", fontsize=11.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_detection_bias_constgold.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)
    print("DETBIAS_PLOT_DONE")


if __name__ == "__main__":
    main()
