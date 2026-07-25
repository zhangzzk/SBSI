#!/usr/bin/env python
"""Selection bias: sims vs flow predictions (Stage-2 flux/size recovery).

For each moving MEASURED cut applied per leg, the SELECTION SHIFT = R(cut)/R(nocut) - 1 is how much the
cut shifts the leg-average shear response. We compare the TRUTH shift (det_meas half-shear) to the FLOW
shift (pinned lt500 8-seed ensemble): if the flow moves the same way, it recovers the selection bias.
Residual recovery m = R_sim/R_model - 1 is annotated per cut.

Data = ISOLATED (nn_bright>7"), N=508,578, job 15264041 (WORKLOG cont.156). Login-node; PNG only.
"""
from __future__ import annotations
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#0072B2", "#D55E00"
OUT = "/home/z/Zekang.Zhang/SBSI-ablation/figures"

# cut, kind, R_sim, R_model, shift_sim(%), shift_mod(%), m(%)   -- ISOLATED, job 15264041
ROWS = [
    ("mag<24.5", "flux", 1.1958, 1.1870, +13.79, +11.89, +0.75),
    ("mag<25",   "flux", 1.1290, 1.1324,  +7.42,  +6.75, -0.30),
    ("mag<25.5", "flux", 1.0839, 1.0898,  +3.13,  +2.73, -0.55),
    ("mag<26",   "flux", 1.0573, 1.0657,  +0.60,  +0.45, -0.79),
    ("size>2.9", "size", 1.0603, 1.0737,  +0.89,  +1.21, -1.24),
    ("size>3.5", "size", 1.1487, 1.1579,  +9.30,  +9.14, -0.79),
    ("size>4.4", "size", 1.0941, 1.1728,  +4.11, +10.55, -6.71),
]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    labels = [r[0] for r in ROWS]
    shift_sim = np.array([r[4] for r in ROWS])
    shift_mod = np.array([r[5] for r in ROWS])
    m = np.array([r[6] for r in ROWS])
    x = np.arange(len(ROWS)); w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.6))
    b1 = ax.bar(x - w/2, shift_sim, w, color=BLUE, label="sim (half-shear truth)")
    b2 = ax.bar(x + w/2, shift_mod, w, color=ORANGE, label="flow (V2 pinned lt500)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(3.5, color="0.85", lw=1.0, ls="--")            # flux | size divider
    ax.text(1.5, ax.get_ylim()[1]*0.96, "FLUX (mag) cuts", ha="center", fontsize=9.5, color="0.4")
    ax.text(5.0, ax.get_ylim()[1]*0.96, "SIZE cuts", ha="center", fontsize=9.5, color="0.4")

    # annotate residual recovery m per cut
    for xi, (ss, sm_, mi) in enumerate(zip(shift_sim, shift_mod, m)):
        top = max(ss, sm_)
        ax.annotate(f"m={mi:+.1f}%", (xi, top), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color=("#b00000" if abs(mi) > 2 else "0.35"),
                    fontweight=("bold" if abs(mi) > 2 else "normal"))

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("selection shift  $R(\\mathrm{cut})/R(\\mathrm{no\\ cut})-1$  [%]")
    ax.set_title("Selection bias on flux & size axes: sims vs flow  (ISOLATED, N=508k)\n"
                 "m = residual recovery $R_{\\rm sim}/R_{\\rm flow}-1$; nocut floor m=-0.93%",
                 fontsize=11.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_selection_bias_sim_vs_flow.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)
    print("PLOT_SELECTION_BIAS_DONE")


if __name__ == "__main__":
    main()
