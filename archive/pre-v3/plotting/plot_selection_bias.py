#!/usr/bin/env python
"""Selection bias: sims vs flow predictions (Stage-2 flux/size recovery), dots + error bars.

For each moving MEASURED cut (per leg), the SELECTION SHIFT = R(cut)/R(nocut) - 1 is how much the cut
shifts the leg-average shear response. We compare the TRUTH shift (det_meas half-shear) to the FLOW shift
(pinned lt500 8-seed ensemble). Cuts BRACKET the training limits: flux (mag) 24.5..26.5; size 0.2..0.8"
(measured flux_radius; note the PSF floor -> measured-size selection only bites above ~0.55").

Error bars: sim = analytic SE of the two selected-leg means; flow = per-seed (8 ckpt) spread.
Reads the npz from scripts/eval_selection_response.py --output. Login-node; PNG only, Okabe-Ito.
"""
from __future__ import annotations
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#0072B2", "#D55E00"


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
                                     "selrecover_v2_lt500.npz")
    ap.add_argument("--sim-npz", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
                                         "truth_selerr_v2.npz",
                    help="matched-pair sim R_sim+err (CPU recompute); falls back to --npz if missing")
    ap.add_argument("--out", default="/home/z/Zekang.Zhang/SBSI-ablation/figures")
    ap.add_argument("--scope", type=int, default=0)   # 0 = ISOLATED
    args = ap.parse_args()
    set_style()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)
    s = f"s{args.scope}_"
    px = float(z["pixel_size"]) if "pixel_size" in z.files else 0.2
    kind = z[s + "kind"].astype(str)
    raw = z[s + "raw"].astype(float)
    cut = z[s + "cut"].astype(str)
    Rmod, Rmod_sd = z[s + "R_model"].astype(float), z[s + "R_model_sd"].astype(float)
    Rsim, Rsim_e = z[s + "R_sim"].astype(float), z[s + "R_sim_err"].astype(float)
    # prefer the matched-pair sim error from the CPU recompute (aligned by cut name)
    if os.path.exists(args.sim_npz):
        zs = np.load(args.sim_npz, allow_pickle=True)
        lut = {c: (rs, re) for c, rs, re in zip(zs["cut"].astype(str),
                                                zs["R_sim"].astype(float), zs["R_sim_err"].astype(float))}
        Rsim = np.array([lut.get(c, (Rsim[i], np.nan))[0] for i, c in enumerate(cut)])
        Rsim_e = np.array([lut.get(c, (np.nan, Rsim_e[i]))[1] for i, c in enumerate(cut)])
        print(f"using matched-pair sim errors from {os.path.basename(args.sim_npz)}")

    nc = kind == "nocut"
    Rnc_s, Rnc_m = float(Rsim[nc][0]), float(Rmod[nc][0])
    # selection shift (%) and propagated error (nocut common, its error sub-dominant)
    sh_sim = (Rsim / Rnc_s - 1) * 100; sh_sim_e = Rsim_e / abs(Rnc_s) * 100
    sh_mod = (Rmod / Rnc_m - 1) * 100; sh_mod_e = Rmod_sd / abs(Rnc_m) * 100

    fig, (axf, axs) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # -------- FLUX (mag) panel: keep bright (mag<thr) --------
    mm = kind == "mag"
    xm = raw[mm]
    o = np.argsort(xm)
    axf.errorbar(xm[o], sh_sim[mm][o], yerr=sh_sim_e[mm][o], fmt="o-", color=BLUE, ms=6, lw=1.6,
                 capsize=3, label="sim (half-shear truth)")
    axf.errorbar(xm[o], sh_mod[mm][o], yerr=sh_mod_e[mm][o], fmt="s--", color=ORANGE, ms=6, lw=1.6,
                 capsize=3, label="flow (V2 pinned lt500)")
    axf.axhline(0, color="0.6", lw=0.8, ls=":")
    axf.set_xlabel(r"measured mag cut  (keep $mag<$ x)")
    axf.set_ylabel(r"selection shift  $R(\mathrm{cut})/R(\mathrm{no\,cut})-1$  [%]")
    axf.set_title("Flux axis")
    axf.invert_xaxis()      # brighter (more selective) to the right
    axf.legend(loc="upper left")

    # -------- SIZE panel: keep large (size>thr), x in arcsec --------
    ss = kind == "size"
    xs = raw[ss] * px
    o = np.argsort(xs)
    axs.errorbar(xs[o], sh_sim[ss][o], yerr=sh_sim_e[ss][o], fmt="o-", color=BLUE, ms=6, lw=1.6,
                 capsize=3, label="sim (half-shear truth)")
    axs.errorbar(xs[o], sh_mod[ss][o], yerr=sh_mod_e[ss][o], fmt="s--", color=ORANGE, ms=6, lw=1.6,
                 capsize=3, label="flow (V2 pinned lt500)")
    axs.axhline(0, color="0.6", lw=0.8, ls=":")
    ylo, yhi = axs.get_ylim()
    axs.axvspan(0.20, 0.40, color="0.85", alpha=0.5, lw=0, zorder=0)
    axs.axvline(0.53, color="0.6", lw=0.9, ls="--")
    axs.text(0.30, yhi * 0.55, "requested\n0.2-0.4\"\n(PSF floor:\nno selection)",
             ha="center", va="center", fontsize=7.6, color="0.45")
    axs.text(0.54, yhi * 0.30, "measured\nfloor ~0.53\"", ha="left", va="center",
             fontsize=7.6, color="0.45")
    axs.set_xlabel(r"measured size cut  (keep $R_{\rm flux}>$ x)  [arcsec]")
    axs.set_title("Size axis")   # legend only on the flux panel (identical series)

    fig.suptitle("Selection bias on flux & size axes: sims vs flow  (ISOLATED)\n"
                 "cuts bracket the training limits (mag 26, size 0.3\"); error bars: sim=analytic SE, "
                 "flow=8-seed spread", fontsize=11.5, y=1.03)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_selection_bias_sim_vs_flow.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)

    # console table
    print(f"\n  {'cut':>10} {'keep-implied':>12} {'shift_sim%':>11} {'shift_mod%':>11}")
    for k, x, a, ae, b, be in zip(kind, raw, sh_sim, sh_sim_e, sh_mod, sh_mod_e):
        if k == "nocut":
            continue
        lab = f"mag<{x:g}" if k == "mag" else f"size>{x*px:.2f}\""
        print(f"  {lab:>10} {'':>12} {a:+7.2f}+-{ae:.2f}  {b:+7.2f}+-{be:.2f}")
    print("PLOT_SELECTION_BIAS_DONE")


if __name__ == "__main__":
    main()
