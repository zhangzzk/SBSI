#!/usr/bin/env python
"""Fig 2 & Fig 5 (V2, pinned lt500 4D flow) -- self-response vs the HALF-SHEAR training target.

Truth curve = measured half-shear det_meas self-response R_hs (the flow's actual training objective;
primary-only shear -> pure R_self, neighbour blend averages out). Model curve = V2 flow R_flow
(pinned lt500 8-seed ensemble, mean-head finite diff), same objects, ngmix units, NO bridge, NO R_blend.

  fig5 -> R vs primary true mag & size, 7"-ISOLATED.
  fig2 -> R vs mag / size / NEIGHBOUR FLUX, ALL objects (the blending-severity panel).

Reads the npz from scripts/eval_selfresp_gap.py (--output). Login-node; PNG only, Okabe-Ito.
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
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })


def binned(x, ysim, ymod, nb=11, logx=False, qlo=1.0, qhi=99.0, minn=200):
    good = np.isfinite(x) & np.isfinite(ysim) & np.isfinite(ymod)
    x, ysim, ymod = x[good], ysim[good], ymod[good]
    lo, hi = np.percentile(x, [qlo, qhi])
    xx = np.log10(np.clip(x, 1e-12, None)) if logx else x
    lo_e = np.log10(max(lo, 1e-12)) if logx else lo
    hi_e = np.log10(max(hi, 1e-12)) if logx else hi
    edges = np.linspace(lo_e, hi_e, nb + 1)
    idx = np.digitize(xx, edges) - 1
    cx, sm, slo, shi, mm = [], [], [], [], []
    for b in range(nb):
        sel = idx == b
        if sel.sum() < minn:
            continue
        cen = 10 ** (0.5 * (edges[b] + edges[b + 1])) if logx else 0.5 * (edges[b] + edges[b + 1])
        cx.append(cen); sm.append(np.mean(ysim[sel]))
        slo.append(np.percentile(ysim[sel], 16)); shi.append(np.percentile(ysim[sel], 84))
        mm.append(np.mean(ymod[sel]))
    return map(np.asarray, (cx, sm, slo, shi, mm))


def panel(ax, x, ys, ym, xlabel, logx=False, annot=True):
    cx, sm, slo, shi, mm = binned(x, ys, ym, logx=logx)
    ax.fill_between(cx, slo, shi, color=BLUE, alpha=0.15, lw=0, label="half-shear 16-84%")
    ax.plot(cx, sm, "-o", color=BLUE, ms=5, lw=1.9, label=r"half-shear truth  $R_{\rm hs}$")
    ax.plot(cx, mm, "-s", color=ORANGE, ms=5, lw=1.9, label=r"V2 flow  $R_{\rm flow}$")
    if annot:
        for xx, a, b in zip(cx, sm, mm):
            if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-6:
                ax.annotate(f"{(b/a-1)*100:+.0f}%", (xx, b), textcoords="offset points",
                            xytext=(0, -12), ha="center", fontsize=7.2, color="#8a5a00")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.axhline(0.0, color="0.7", lw=0.8, ls=":", zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
                                     "selfresp_gap_v2_lt500.npz")
    ap.add_argument("--out", default="/home/z/Zekang.Zhang/SBSI-ablation/figures")
    ap.add_argument("--tag", default="V2 pinned lt500 (8-seed)")
    args = ap.parse_args()
    set_style()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)
    Rhs, Rf = z["R_hs"], z["R_flow"]
    mag, size, nbf = z["mag"], z["size"], z["nbr_flux"]
    iso7 = z["iso7"] if "iso7" in z.files else z["iso"]
    good = z["good"].astype(bool)
    gtag = f"g0->g{float(z['delta']):.2f} {str(z['difference'])}"

    # ---------------- fig5: ISOLATED, mag & size ----------------
    iso = good & iso7.astype(bool)
    fig, (axm, axs) = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    panel(axm, mag[iso], Rhs[iso], Rf[iso], r"primary true mag  $r_{\rm input,p}$")
    panel(axs, size[iso], Rhs[iso], Rf[iso], r"primary true size  $R_e$ [arcsec]")
    axm.set_ylabel("self-response  R")
    axm.legend(loc="lower left", fontsize=8.5)
    axm.text(0.98, 0.03, f"7\"-isolated, N={int(iso.sum()):,}\n{args.tag}, {gtag}",
             transform=axm.transAxes, fontsize=7.5, va="bottom", ha="right", color="0.4")
    fig.suptitle("Fig 5 (V2): flow self-response vs half-shear target, 7\"-isolated",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p5 = os.path.join(args.out, "figv2_fig5_selfresp.png")
    fig.savefig(p5, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---------------- fig2: ALL, mag / size / neighbour flux ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    panel(axes[0], mag[good], Rhs[good], Rf[good], r"primary true mag  $r_{\rm input,p}$")
    panel(axes[1], size[good], Rhs[good], Rf[good], r"primary true size  $R_e$ [arcsec]")
    keep = good & np.isfinite(nbf) & (nbf > 1e-3)
    panel(axes[2], nbf[keep], Rhs[keep], Rf[keep], r"neighbour flux  (near shell)", logx=True)
    axes[0].set_ylabel("self-response  R")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].text(0.98, 0.03, f"ALL objects, N={int(good.sum()):,}\n{args.tag}, {gtag}",
                 transform=axes[0].transAxes, fontsize=7.5, va="bottom", ha="right", color="0.4")
    fig.suptitle("Fig 2 (V2): flow vs half-shear self-response across galaxy properties",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p2 = os.path.join(args.out, "figv2_fig2_selfresp.png")
    fig.savefig(p2, dpi=200, bbox_inches="tight"); plt.close(fig)

    print("saved:", p5); print("saved:", p2)
    for lab, sel in [("ISOLATED(7\")", iso), ("ALL", good)]:
        a = float(np.mean(Rhs[sel])); b = float(np.mean(Rf[sel]))
        print(f"  {lab:14s} <R_hs>={a:+.4f}  <R_flow>={b:+.4f}  flow/hs-1={(b/a-1)*100:+.2f}%  N={int(sel.sum()):,}")
    print("PLOT_V2_SELFRESP_DONE")


if __name__ == "__main__":
    main()
