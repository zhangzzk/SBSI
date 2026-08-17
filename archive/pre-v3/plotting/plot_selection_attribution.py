#!/usr/bin/env python
"""Selection bias, 2x2: rows are the two quantities, columns are the two cut axes.

    row 1  m_sel  = R_sim(cut) / R_sim(nocut) - 1
        The selection bias that REALLY EXISTS in the simulation. Computed with EXACT sheared
        intrinsic shapes on both legs, so no model and no shape noise enter it -- a property of the
        data, not of anything we fit. Where it is ~0 the cut has NO selection bias to test.

    row 2  m_flow = R_flow(cut) / R_sim(cut) - 1
        What is LEFT after the model predicts the selection. Both sides average the same exact shapes
        over the same objects; the only difference is WHICH objects each side selected (sim cuts on
        the catalogue's measured size/mag, model on the flow's sampled measured size/mag). So this
        isolates selection with the shape response divided out.

    columns  flux axis (keep mag < x) | size axis (keep R_flux > x)
        Kept from the 2026-07-25 figure this replaces, because the two families have different units
        and cannot honestly share one x-axis.

READ THE ROWS TOGETHER. A small |m_flow| where m_sel ~ 0 means "nothing was being tested", not "the
model works". The magnitude column is largely in that category.

NOT A DECOMPOSITION. Selection and shape are correlated -- the cut is on measured size/mag, which
covary with measured shape inside the flow's joint. Using exact shapes deliberately breaks that
correlation, so the rows do NOT add up to the end-to-end m; a cross term is set aside by
construction.

y IS SHARED WITHIN EACH ROW, on purpose. It is what makes the headline legible: on the intrinsic
(selection-only) measurement the magnitude axis is flat while the size axis climbs to ~+14%. The
original figure's large flux-axis shift was mostly SHAPE response, which this construction removes.

Login-node work: reads one small npz, writes a PNG.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#999999"
PSF_R50 = 0.5268   # Moffat FWHM 0.73", beta 2.224 -> half-light radius, in arcsec


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.labelsize": 10.5, "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/"
                                     "selection_attribution_dense_4seed_n32.npz")
    ap.add_argument("--out", default="figures/fig_selection_bias_sim_vs_flow.png")
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--target", type=float, default=0.3,
                    help="project |m| target in %%, shaded in the bottom row for context")
    # The +15% point at 0.9" compresses the whole top row, which then hides the fact that the
    # magnitude axis is flat (y is shared within a row). Cropping the size axis to <=0.6" keeps the
    # ONSET -- the only part where the two rows are both informative -- and lets the row breathe.
    # This crops the VIEW only; the dropped cuts are still measured and still printed below.
    ap.add_argument("--size-min", type=float, default=0.2, help="lowest size cut to SHOW [arcsec]")
    ap.add_argument("--size-max", type=float, default=0.6, help="highest size cut to SHOW [arcsec]")
    args = ap.parse_args()
    set_style()

    z = np.load(args.npz, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    sim, mod = z["intrinsic_sim"].astype(float), z["intrinsic_mod"].astype(float)
    have_err = "intrinsic_sim_err" in z.files and "intrinsic_mod_sem" in z.files
    sim_err = z["intrinsic_sim_err"].astype(float) if have_err else None
    mod_sem = z["intrinsic_mod_sem"].astype(float) if have_err else None

    nc = names.index("__nocut__")
    # Both quantities are exactly 0 at the no-cut row BY CONSTRUCTION, so it is the correctness
    # check, not a datum. Dropped from the plot.
    fam = {}
    for kind, pref, conv in (("size", "size>", lambda v: v * args.pixel_size),
                             ("mag", "mag<", lambda v: v)):
        idx = [i for i, n in enumerate(names) if n.startswith(pref)]
        thr = np.array([conv(float(names[i][len(pref):])) for i in idx])
        o = np.argsort(thr)
        idx = [idx[i] for i in o]
        thr = thr[o]
        ms = (sim[idx] / sim[nc] - 1.0) * 100.0
        mf = (mod[idx] / sim[idx] - 1.0) * 100.0
        # m_sel: the cut sample is a SUBSET sharing the no-cut denominator, so the two errors are
        # strongly correlated; adding in quadrature would overstate it. Numerator term only.
        es = np.abs(sim_err[idx] / sim[nc]) * 100.0 if have_err else None
        # m_flow: the 4-seed spread of the model dominates. With 4 seeds the sd is itself uncertain
        # by ~40%, so these bars are indicative, not precise.
        ef = np.abs(mod_sem[idx] / sim[idx]) * 100.0 if have_err else None
        fam[kind] = dict(thr=thr, m_sel=ms, m_flow=mf, e_sel=es, e_flow=ef)

    # Crop the size axis for DISPLAY only. `fam_full` is kept so the table printed at the end still
    # covers every measured cut -- the figure is cropped, the measurement is not.
    fam_full = {k: dict(v) for k, v in fam.items()}
    d = fam["size"]
    show = (d["thr"] >= args.size_min - 1e-9) & (d["thr"] <= args.size_max + 1e-9)
    dropped = d["thr"][~show]
    fam["size"] = {k: (v[show] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    if dropped.size:
        print("  size cuts measured but NOT SHOWN (view cropped to %.2f-%.2f\"): %s"
              % (args.size_min, args.size_max, ", ".join('%.2f"' % t for t in dropped)))

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0), sharey="row",
                             gridspec_kw=dict(hspace=0.32, wspace=0.10))
    cols = [("mag", ORANGE, "measured mag cut   (keep $mag < x$)"),
            ("size", BLUE, 'measured size cut   (keep $R_{\\rm flux} > x$)   [arcsec]')]
    rows = [("m_sel", "selection bias in the sims\n$m_{\\rm sel}=R_{\\rm sim}(cut)/R_{\\rm sim}(no\\ cut)-1$  [%]"),
            ("m_flow", "left after model correction\n$m_{\\rm flow}=R_{\\rm flow}(cut)/R_{\\rm sim}(cut)-1$  [%]")]

    for r, (key, ylab) in enumerate(rows):
        for c, (kind, colr, xlab) in enumerate(cols):
            ax = axes[r][c]
            d = fam[kind]
            ax.axhline(0.0, color=GREY, lw=1.0, ls=":", zorder=1)
            if key == "m_flow":
                ax.axhspan(-args.target, args.target, color=GREY, alpha=0.16, zorder=0, lw=0)
            if kind == "size":
                # Below the PSF half-light radius a measured-size cut selects nothing
                # shear-dependent -- the flat part of the top-right panel is this floor.
                ax.axvspan(d["thr"].min() - 0.05, PSF_R50, color=GREY, alpha=0.10, zorder=0, lw=0)
                ax.axvline(PSF_R50, color=GREY, lw=0.9, ls="--", zorder=1)
            e = d["e_sel"] if key == "m_sel" else d["e_flow"]
            ax.errorbar(d["thr"], d[key], yerr=e, color=colr, marker="o", ms=5.5, lw=1.6,
                        capsize=3, elinewidth=1.3, zorder=3)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 1:
                ax.set_xlabel(xlab)
            if kind == "mag":
                ax.invert_xaxis()   # tighter (brighter) cut to the RIGHT, as in the original
    # Annotate the PSF floor once, on the panel where it explains the shape.
    axes[0][1].annotate("PSF $R_{50}$\n%.2f\"" % PSF_R50, (PSF_R50, axes[0][1].get_ylim()[1]),
                        xytext=(-4, -30), textcoords="offset points", ha="right", va="top",
                        fontsize=8.5, color="#666666")
    # AXES coordinates, not data. In data coords this label chased the inverted magnitude x-axis and
    # rendered over the neighbouring panel.
    axes[1][0].text(0.015, 0.03, "shaded: |m| < %.1f%% target" % args.target,
                    transform=axes[1][0].transAxes, ha="left", va="bottom",
                    fontsize=8.5, color="#666666")

    # No tight_layout(): the shared-y row + inverted axis combination is not compatible with it and
    # it warns. hspace/wspace are set explicitly in gridspec_kw instead.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("saved %s" % args.out)

    if not have_err:
        print("  NOTE: npz stores no errors -> no error bars drawn.")
    for kind in ("size", "mag"):
        d = fam_full[kind]          # full table, including cuts cropped out of the figure
        print("\n  --- %s axis ---" % kind)
        print("  %10s %18s %18s" % ("cut", "m_sel [%]", "m_flow [%]"))
        for i, t in enumerate(d["thr"]):
            lo = "%.2f\"" % t if kind == "size" else "%.1f" % t
            if have_err:
                print("  %10s %+10.2f +- %-4.2f %+10.2f +- %-4.2f"
                      % (lo, d["m_sel"][i], d["e_sel"][i], d["m_flow"][i], d["e_flow"][i]))
            else:
                print("  %10s %+18.2f %+18.2f" % (lo, d["m_sel"][i], d["m_flow"][i]))
    print("\n  no-cut row dropped: 0 by construction (R_sim=R_model=%+.6f)." % sim[nc])


if __name__ == "__main__":
    main()
