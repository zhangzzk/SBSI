#!/usr/bin/env python
"""Step-2 recovery: does the detection classifier reproduce the sim detection response?

Reads the two eval npz from scripts/eval_detection_response.py --output (BCE-only lam=0 and
response-aware lam=300). Shows, per blend bin, the sim target b_sim vs each classifier's induced
b_model; and the un-binned GLOBAL observable (~ R_detect, comparable to the sim's -2.01% and the
Stage-3 step-1 constgold -1.86%). The point: BCE-only gets the WRONG SIGN (+6% per-cell / +1.9%
global), the shear-response regularizer corrects it to match the sim, conditioned on blending.

Login-node; PNG only, Okabe-Ito.
"""
from __future__ import annotations
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY, BLUE, ORANGE = "#000000", "#0072B2", "#D55E00"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def main():
    ap = argparse.ArgumentParser()
    base = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
    ap.add_argument("--bce", default=base + "det_response_eval_lam0.npz")
    ap.add_argument("--resp", default=base + "det_response_eval_lam300.npz")
    ap.add_argument("--constgold", default=base + "detection_constgold_v1.npz",
                    help="Stage-3 step-1 constgold measurement (for the cross-check line)")
    ap.add_argument("--out", default="/home/z/Zekang.Zhang/SBSI-ablation/figures")
    args = ap.parse_args()
    set_style()
    os.makedirs(args.out, exist_ok=True)
    z0 = np.load(args.bce, allow_pickle=True)
    z3 = np.load(args.resp, allow_pickle=True)
    labels = [str(x) for x in z3["blend_labels"]]
    x = np.arange(len(labels))
    bsim = 100 * z3["blend_b_sim"]
    bmod0 = 100 * z0["blend_b_model"]
    bmod3 = 100 * z3["blend_b_model"]

    fig, (axb, axg) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})

    # ---- per-blend recovery ----
    w = 0.26
    axb.bar(x - w, bsim, w, color=GREY, label="sim target $b_{\\rm sim}/g$")
    axb.bar(x, bmod0, w, color=BLUE, label="BCE-only classifier (wrong sign)")
    axb.bar(x + w, bmod3, w, color=ORANGE, label="response-aware classifier")
    axb.axhline(0, color="0.5", lw=0.8)
    axb.set_xticks(x); axb.set_xticklabels(labels)
    axb.set_ylabel(r"detection response  $b/g$  [%]")
    axb.set_title("Per-blend detection response: sim vs classifier")
    axb.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.09), fontsize=9)

    # ---- global observable ----
    gsim = 100 * float(z3["global_b_sim"])
    g0 = 100 * float(z0["global_b_model_obs"])
    g3 = 100 * float(z3["global_b_model_obs"])
    names = ["sim\n$b_{\\rm sim}/g$", "BCE-only\nclassifier", "response-aware\nclassifier"]
    vals = [gsim, g0, g3]
    cols = [GREY, BLUE, ORANGE]
    axg.bar(np.arange(3), vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        axg.text(i, v + (0.15 if v >= 0 else -0.15), f"{v:+.2f}%", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9)
    axg.axhline(0, color="0.5", lw=0.8)
    # constgold step-1 cross-check line (measured -0.88%, sheared-intrinsic -1.86%)
    try:
        zc = np.load(args.constgold, allow_pickle=True)
        cg_int = 100 * float(zc["ALL_shint"][2])   # sheared-intrinsic det-bias (ratio)
        axg.axhline(cg_int, color="0.45", lw=1.1, ls="--")
        axg.text(2.4, cg_int, f"constgold\nstep-1 {cg_int:+.2f}%", fontsize=7.6, color="0.4",
                 ha="right", va="center")
    except Exception:
        pass
    axg.set_xticks(np.arange(3)); axg.set_xticklabels(names)
    axg.set_ylabel(r"global observable  $b/g$  [%]  ($\approx R_{\rm detect}$)")
    axg.set_title("Global detection bias (un-binned)")

    fig.suptitle("Stage-3 step-2: classifier recovers the sim detection response  "
                 "(BCE-only wrong sign $\\to$ response-regularized matches sim, blend-conditioned)",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_detection_response_recovery.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)
    print("DETRESP_RECOVERY_PLOT_DONE")


if __name__ == "__main__":
    main()
