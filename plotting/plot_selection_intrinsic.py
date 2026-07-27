#!/usr/bin/env python
"""Selection bias vs measured mag / measured size cut: constgold sim vs Gold-V2 8-seed flow.

The plotted quantity is the pure moving-boundary selection response

    R_sel = ( <e_int>_plus[pass_plus] - <e_int>_minus[pass_minus] ) / 0.04

on both-detected +/-0.02 constgold pairs, with the cut applied separately in each leg on that
leg's own measured observable. The numerator is the UNSHEARED INTRINSIC ellipticity, which is
identical for the same galaxy in both legs -- so R_sel is zero unless the cut lets different
galaxies through at +g and -g. No cut, and any cut on a true property, give exactly zero.

Right-hand axis is the same number divided by the sample's total measured-shape response
(R_total, measured in the same run) -- i.e. the multiplicative shear bias the cut induces.

Reads the npz from scripts/eval_selection_intrinsic.py --output. Login-node; PNG only.
Colours are Okabe-Ito blue/vermillion (validated: CVD dE 21.9, normal-vision dE 31.2).
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, VERM = "#0072B2", "#D55E00"
INK, MUTED = "#1a1a1a", "#6b6b6b"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#999999", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False,
    })


def panel(ax, x, sim, sim_e, sim_sub, mod, mod_sd, keep, xlabel, more_selective):
    """One cut axis: sim with bootstrap bars, model with 8-seed band.

    `sim_sub` is the sim recomputed on the EXACT 1.5M objects the model ran on. The model's
    8-seed band is only its seed scatter -- it does not include the noise from using a subsample,
    and all 8 seeds share that same subsample so the spread cannot reveal it. Plotting sim_sub
    exposes it: where sim_sub sits on top of sim, subsampling is irrelevant; where it departs
    (the bright-magnitude end, few objects kept) any sim-vs-model gap is partly sampling, not model
    error.
    """
    o = np.argsort(x)
    x, sim, sim_e, sim_sub, mod, mod_sd, keep = (
        a[o] for a in (x, sim, sim_e, sim_sub, mod, mod_sd, keep))

    ax.axhline(0, color="#cccccc", lw=1.0, zorder=0)

    # model: dashed line + 8-seed spread band + open squares
    ax.fill_between(x, mod - mod_sd, mod + mod_sd, color=VERM, alpha=0.18, lw=0, zorder=1)
    ax.plot(x, mod, ls="--", lw=1.8, color=VERM, zorder=3)
    ax.plot(x, mod, ls="none", marker="s", ms=7, mfc="white", mec=VERM, mew=1.8,
            zorder=4, label="Gold-V2 flow (8 seeds)")

    # sim on the model's own subsample -> shows how much of any gap is just subsampling
    ax.plot(x, sim_sub, ls=":", lw=1.4, color=BLUE, alpha=0.75, zorder=2)
    ax.plot(x, sim_sub, ls="none", marker="o", ms=6, mfc="white", mec=BLUE, mew=1.4,
            alpha=0.85, zorder=2, label="sim, model's 1.5M subsample")

    # sim: solid line + filled circles + bootstrap error bars
    ax.errorbar(x, sim, yerr=sim_e, fmt="o-", color=BLUE, ms=7, lw=1.8, capsize=3,
                elinewidth=1.2, zorder=5, label="sim (constgold $\\pm$0.02, all 26.9M)")

    ax.set_xlabel(xlabel)
    ax.margins(x=0.07)

    # kept fraction as one compact corner note (per-point labels collide with the curves)
    ax.annotate(f"kept fraction {keep[0]*100:.0f}% $\\to$ {keep[-1]*100:.0f}%\n{more_selective}",
                xy=(0.5, 1.01), xycoords="axes fraction", ha="center", va="bottom",
                fontsize=8.5, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
                                     "selection_intrinsic_v1.npz")
    ap.add_argument("--out", default=None, help="output directory (default: <repo>/figures)")
    ap.add_argument("--scope", type=int, default=0, help="0 = ALL, 1 = ISOLATED")
    ap.add_argument("--name", default="fig_selection_bias_intrinsic.png")
    args = ap.parse_args()
    set_style()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(here, "figures")
    os.makedirs(out_dir, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    s = f"s{args.scope}_"
    scope = str(z["scope_names"][args.scope])
    kind = z[s + "kind"].astype(str)
    raw = z[s + "raw"].astype(float)
    R_total = float(z[s + "R_total"])
    sim, sim_e = z[s + "R_sim"].astype(float), z[s + "R_sim_err"].astype(float)
    sim_sub = z[s + "R_sim_sub"].astype(float)
    mod, mod_sd = z[s + "R_model"].astype(float), z[s + "R_model_sd"].astype(float)
    keep = z[s + "frac"].astype(float)
    nseed = len(z["ckpts"])

    fig, (axm, axs) = plt.subplots(1, 2, figsize=(13.8, 5.4))

    mm = kind == "mag"
    panel(axm, raw[mm], sim[mm], sim_e[mm], sim_sub[mm], mod[mm], mod_sd[mm], keep[mm],
          r"measured magnitude cut   (keep $m < x$)",
          "$\\longleftarrow$  more selective")
    axm.set_ylabel(r"selection response  $R_{\rm sel}=\Delta\langle e_{\rm int}\rangle\,/\,0.04$")
    axm.legend(loc="best")
    axm.annotate(f"constgold $\\pm$0.02 both-detected pairs, {scope}\n"
                 f"$R_{{\\rm total}}$={R_total:.3f}, {nseed}-seed flow ensemble",
                 xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color=MUTED,
                 va="bottom", ha="left")

    ss = kind == "size"
    panel(axs, raw[ss], sim[ss], sim_e[ss], sim_sub[ss], mod[ss], mod_sd[ss], keep[ss],
          r"measured size cut   (keep $R_{\rm flux} > x$)  [arcsec]",
          "more selective  $\\longrightarrow$")

    # right-hand axis on BOTH panels: same quantity, expressed as the multiplicative bias it causes
    for ax in (axm, axs):
        sec = ax.secondary_yaxis("right",
                                 functions=(lambda v: v / R_total * 100.0,
                                            lambda v: v * R_total / 100.0))
        sec.set_ylabel(r"bias  $R_{\rm sel}/R_{\rm total}$  [%]", color=MUTED)
        sec.tick_params(colors=MUTED)

    fig.tight_layout(w_pad=5.0)
    p = os.path.join(out_dir, args.name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", p)

    print(f"\n  scope={scope}  R_total={R_total:.4f}  seeds={nseed}")
    print(f"  {'cut':>14} {'keep%':>6} {'R_sel sim':>11} {'R_sel mod':>11} "
          f"{'bias sim%':>10} {'bias mod%':>10}")
    names = z[s + "cut"].astype(str)
    for k, x, a, ae, b, bsd, f, nm in zip(kind, raw, sim, sim_e, mod, mod_sd, keep, names):
        lab = nm if k == "null" else (f"mag<{x:g}" if k == "mag" else f"size>{x:g}\"")
        print(f"  {lab:>14} {f*100:>6.1f} {a:>+11.5f} {b:>+11.5f} "
              f"{a/R_total*100:>+10.3f} {b/R_total*100:>+10.3f}")
    print("PLOT_SELECTION_INTRINSIC_DONE")


if __name__ == "__main__":
    main()
