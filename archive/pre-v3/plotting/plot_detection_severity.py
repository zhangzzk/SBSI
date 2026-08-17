#!/usr/bin/env python
"""Detection bias as a function of blending SEVERITY (Stage-3, constgold, truth-only).

Reads the npz from scripts/eval_detection_severity.py. Three panels, one per severity axis:
  * blendedness beta (headline)  * distance_scaled (kernel-free)  * flux_ratio_nbr (kernel-free).
Each panel: det-bias vs severity, one line per true-primary-mag band, isolated (severity=0)
reference shown as an open marker at the left. Sheared-intrinsic shapes (noise-free).

Login-node; PNG only, Okabe-Ito.
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"all": "#000000", "bright r<25.5": "#0072B2", "faint r>=25.5": "#D55E00"}
MK = {"all": "o", "bright r<25.5": "s", "faint r>=25.5": "^"}
AX_LABEL = {
    "beta": r"blendedness  $\beta = f_s K/(f_p+f_s K)$",
    "distance_scaled": r"separation / convolved $R_e$   (distance$_{\rm scaled}$)",
    "flux_ratio_nbr": r"$\log_{10}(f_{\rm nbr}/f_{\rm prim})$   (brighter neighbour $\to$)",
}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def main():
    ap = argparse.ArgumentParser()
    base = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
    ap.add_argument("--npz", default=base + "detection_severity_v1.npz")
    ap.add_argument("--out", default="/home/z/Zekang.Zhang/SBSI-ablation/figures")
    args = ap.parse_args()
    set_style()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)
    axes = [str(a) for a in z["sev_axes"]]
    bands = [str(b) for b in z["mag_band_labels"]]

    fig, axs = plt.subplots(1, len(axes), figsize=(5.0 * len(axes), 4.9), sharey=True)
    if len(axes) == 1:
        axs = [axs]

    for ax_i, sev in enumerate(axes):
        pa = axs[ax_i]
        for bl in bands:
            key = f"{sev}__{bl}"
            if f"{key}__center" not in z.files:
                continue
            x = z[f"{key}__center"]
            db = 100 * z[f"{key}__db"]
            se = 100 * z[f"{key}__se"]
            if x.size == 0:
                continue
            c = COL.get(bl, "0.3"); mk = MK.get(bl, "o")
            pa.errorbar(x, db, yerr=se, color=c, marker=mk, ms=5, lw=1.5, capsize=2.5,
                        label=bl)
            # isolated reference for this band (open marker at left edge)
            iso = z.get(f"iso_{bl}")
            if iso is not None and np.isfinite(iso[2]):
                xr = x.min()
                pa.errorbar([xr], [100 * iso[2]], yerr=[100 * iso[3]], color=c,
                            marker=mk, ms=7, mfc="white", mec=c, lw=0, capsize=2.5, zorder=5)
        pa.axhline(0, color="0.5", lw=0.9)
        pa.set_xlabel(AX_LABEL.get(sev, sev))
        pa.set_title(f"det-bias vs {sev}")
        if sev == "flux_ratio_nbr":
            pa.axvline(0, color="0.7", lw=0.8, ls=":")  # equal-brightness neighbour
    axs[0].set_ylabel(r"detection bias  $R_{\rm full}/R_{\rm both}-1$   [%]")
    axs[0].legend(loc="best", title="true primary mag")

    fig.suptitle("Stage-3: detection bias vs blending severity  "
                 "(constgold, sheared-intrinsic shapes; open markers = isolated reference)",
                 fontsize=11.5, y=1.01)
    fig.text(0.5, -0.02,
             "Blended galaxies binned by severity (equal counts). Response = leg-average of the "
             "detected sample's intrinsic sheared shape; bias = deviation from the both-detected "
             "reference. All binning on TRUE neighbour properties (shear-invariant).",
             ha="center", fontsize=8.2, color="0.35")
    fig.tight_layout()
    p = os.path.join(args.out, "fig_detection_severity.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", p)
    print("DETECTION_SEVERITY_PLOT_DONE")


if __name__ == "__main__":
    main()
