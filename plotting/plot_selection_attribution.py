#!/usr/bin/env python
"""Selection bias in two panels: how much exists in the sims, and what survives the model.

Panel 1  m_sel  = R_sim(cut) / R_sim(nocut) - 1
    The selection bias that REALLY EXISTS in the simulation. Computed with EXACT sheared intrinsic
    shapes on both legs, so no model and no shape noise enter it at all -- this is a property of the
    data, not of anything we fit. Where it is ~0 the cut has no selection bias to test, and a small
    number in panel 2 there means "nothing was being tested", not "the model works".

Panel 2  m_flow = R_flow(cut) / R_sim(cut) - 1
    What is LEFT after the model predicts the selection. Both sides average the same exact shapes
    over the same objects; the only difference is WHICH objects each side selected -- the sim cuts on
    the catalogue's measured size/mag, the model cuts on the flow's sampled measured size/mag. So
    this isolates selection, with the shape response divided out.

WHY THE SPLIT (owner's construction). The end-to-end number mixes shape-response error with
selection-response error and cannot be attributed. Reading panel 2 alone is misleading for the reason
in panel 1's note; the panels are meant to be read together.

NOT A DECOMPOSITION. Selection and shape are correlated -- the cut is on measured size/mag, which
covary with measured shape inside the flow's joint. Using exact shapes deliberately breaks that
correlation, so panel 1 + panel 2 do NOT add up to the end-to-end m; a cross term is set aside by
construction.

Reads the npz written by scripts/eval_selection_attribution.py. Login-node work: reads one small npz
and writes a PNG, no compute.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#999999"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.labelsize": 11, "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def pretty(name, px):
    """'size>3.5' -> 'R > 0.70\"' (thresholds are stored in PIXELS); 'mag<24' -> 'mag < 24'."""
    if name.startswith("size>"):
        return 'R > %.2f"' % (float(name.split(">")[1]) * px)
    if name.startswith("mag<"):
        return "mag < %s" % name.split("<")[1]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/"
                                     "selection_attribution_4seed_n32.npz")
    ap.add_argument("--out", default="figures/fig_selection_bias_sim_vs_flow.png")
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--target", type=float, default=0.3,
                    help="project |m| target in %%, shaded in panel 2 for context")
    args = ap.parse_args()
    set_style()

    z = np.load(args.npz, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    sim = z["intrinsic_sim"].astype(float)
    mod = z["intrinsic_mod"].astype(float)

    # Errors are optional: the first version of the attribution npz did not store them. When absent
    # the figure is drawn WITHOUT error bars and says so, rather than implying a precision it has not
    # measured.
    have_err = "intrinsic_sim_err" in z.files and "intrinsic_mod_sem" in z.files
    sim_err = z["intrinsic_sim_err"].astype(float) if have_err else None
    mod_sem = z["intrinsic_mod_sem"].astype(float) if have_err else None

    nc = names.index("__nocut__")
    # Both panels are exactly 0 at the no-cut row BY CONSTRUCTION (same shapes, same objects), so it
    # carries no information and is dropped from the plot. It is the correctness check, not a datum.
    keep = [i for i, n in enumerate(names) if n != "__nocut__"]
    lab = [names[i] for i in keep]
    m_sel = (sim[keep] / sim[nc] - 1.0) * 100.0
    m_flow = (mod[keep] / sim[keep] - 1.0) * 100.0

    if have_err:
        # m_sel = Rc/Rnc - 1. The two legs SHARE the no-cut denominator and the cut sample is a
        # SUBSET of it, so the errors are strongly correlated and adding them in quadrature would
        # overstate the uncertainty. Use the numerator term only -- the honest cheap bound.
        e_sel = np.abs(sim_err[keep] / sim[nc]) * 100.0
        # m_flow = Rmod/Rsim - 1: model seed-scatter dominates; the sim term is the same exact-shape
        # quantity on both sides of the panel, so it is carried but does not drive the bar.
        e_flow = np.abs(mod_sem[keep] / sim[keep]) * 100.0
    else:
        e_sel = e_flow = None

    is_size = np.array([n.startswith("size>") for n in lab])
    # size cuts first, then magnitude cuts; within each family, ordered by threshold
    order = sorted(range(len(lab)), key=lambda i: (not is_size[i], float(lab[i].split(">" if is_size[i] else "<")[1])))
    lab = [lab[i] for i in order]
    m_sel, m_flow, is_size = m_sel[order], m_flow[order], is_size[order]
    if have_err:
        e_sel, e_flow = e_sel[order], e_flow[order]
    x = np.arange(len(lab))
    col = [BLUE if s else ORANGE for s in is_size]
    xt = [pretty(n, args.pixel_size) for n in lab]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7))
    split = float(np.sum(is_size)) - 0.5    # boundary between the two cut families

    for ax, y, e, ylab in ((ax1, m_sel, e_sel, "selection bias in the sims   $m_{\\rm sel}$  [%]"),
                           (ax2, m_flow, e_flow,
                            "left after model correction   $m_{\\rm flow}$  [%]")):
        ax.axhline(0.0, color=GREY, lw=1.0, ls="--", zorder=1)
        ax.axvline(split, color=GREY, lw=0.8, ls=":", zorder=1)
        if e is not None:
            for m, c in ((is_size, BLUE), (~is_size, ORANGE)):
                ax.errorbar(x[m], y[m], yerr=e[m], fmt="none", ecolor=c, elinewidth=1.4,
                            capsize=3, zorder=2)
        ax.scatter(x[is_size], y[is_size], s=58, color=BLUE, zorder=3, label="size cut")
        ax.scatter(x[~is_size], y[~is_size], s=58, color=ORANGE, marker="s", zorder=3,
                   label="magnitude cut")
        ax.set_xticks(x)
        ax.set_xticklabels(xt, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.6, len(lab) - 0.4)

    # Panel 1: label the two big ones directly -- the story is that only the size axis moves.
    for i, (xi, yi) in enumerate(zip(x, m_sel)):
        if abs(yi) >= 1.0:
            ax1.annotate("%+.1f%%" % yi, (xi, yi), textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=9, color=BLUE)
    ax1.set_ylim(-1.5, max(m_sel.max() * 1.18, 2.0))
    ax1.legend(loc="upper left")

    # Panel 2: the project's |m| target, for scale. Context only -- this is the selection stage, not
    # the end-to-end budget.
    ax2.axhspan(-args.target, args.target, color=GREY, alpha=0.18, zorder=0, lw=0)
    ax2.annotate("|m| < %.1f%% target" % args.target, (len(lab) - 0.5, args.target),
                 ha="right", va="bottom", fontsize=8.5, color="#666666")
    pad = max(0.25, np.abs(m_flow).max() * 0.35)
    ax2.set_ylim(min(m_flow.min(), -args.target) - pad, max(m_flow.max(), args.target) + pad)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("saved %s" % args.out)

    if not have_err:
        print("  NOTE: this npz stores no errors, so the figure has NO error bars. Rerun "
              "scripts/eval_selection_attribution.py (it now saves sim_err and mod_sem).")
    print("\n  %-12s %16s %16s" % ("cut", "m_sel [%]", "m_flow [%]"))
    for i, (n, a, b) in enumerate(zip(xt, m_sel, m_flow)):
        if have_err:
            print("  %-12s %+9.2f +- %-4.2f %+9.2f +- %-4.2f" % (n, a, e_sel[i], b, e_flow[i]))
        else:
            print("  %-12s %+16.2f %+16.2f" % (n, a, b))
    print("\n  no-cut row dropped: both quantities are 0 by construction there "
          "(R_sim=R_model=%+.6f)." % sim[nc])


if __name__ == "__main__":
    main()
