"""FIGURE 5, SIDE BY SIDE: the fiducial flow and the LABEL-CONSISTENT retrain against the same truth.

WHAT THIS SHOWS. Fiducial figure 5 (`fid_fig5_selfresp_halfshear`) plots the flow ALONE against the
half-shear self-response, so no emulator error can trade against a flow error. This version puts TWO
flows on those same axes:

  fiducial   the V2 dom6x6 flow, supervised by a per-object target fit to the ruler's `R_snc`
  relabelled the SAME architecture, weight and seeds, supervised by a target fit to `r_sim_self`
             -- the very quantity both curves are graded against (WORKLOG 2026-08-03w/x)

Everything else is held fixed: identical rows, identical truth column, the same three seeds
(501/502/503), and the same FORWARD extraction on both dumps. The ONLY difference between the two
model curves is which g=0 measurement the training label was built from.

WHY IT IS WORTH A FIGURE. WORKLOG 2026-08-03u/v/w established that the project has two independent
SNC estimators of the same self-response, differing because their g=0 shape measurements are separate
noise realisations, and that the disagreement grows toward small/faint objects (mean |de| 0.108 at
Re<=0.386 against 0.025 at Re>0.75). A model trained on one and scored against the other therefore
reads low at small size no matter how well it fits. The region numbers say the small-size residual
collapses from -3.74% to -0.08%; this figure shows WHERE along each axis that happens, and -- just as
importantly -- where it does NOT.

READ THE THREE PANELS DIFFERENTLY. The size panel is a TRUE property, and it is where the label fix
is expected to act. The S/N panel is `measured_flux_auto / measured_fluxerr_auto`, a MEASURED
quantity: it selects on the noise realisation, which a true-property-conditioned flow cannot
reproduce by construction, so its faint end is NOT expected to move and its residual there is a
ceiling rather than a defect (03w; under a TRUE-magnitude cut the flow is -0.53 +- 0.57%). The
neighbour-flux panel is diagnostic only -- self-response is the flow's own job and the emulator's
R_blend is deliberately absent from both curves.

WHAT IT DOES NOT SHOW. This is SELF-CONSISTENCY between a label and a score, not evidence that either
g=0 measurement is physically unbiased -- and that choice moves the absolute response scale, so it
propagates into `m`. Nothing here is an `m`: 3 seeds cannot report one under the seed convention
(16 required), and constgold is never read.

Every number printed on the figure is computed in this run from the two dumps. Nothing is transcribed.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.feather as pf  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_fid_flow_figures import BLUE, GREEN, MUTED, VERM, save, set_style  # noqa: E402

FID = "results/halfshear_selfresp.feather"
NEW = ("/project/ls-gruen/users/zekang.zhang/sbsi_caches/pin_realloc/"
       "hs_perobj_dumplbl.feather")
SEEDS = [501, 502, 503]
SMALL_RE = 0.386
FAINT_SN = 13.09


def load(path, seeds):
    df = pf.read_table(path, memory_map=True).to_pandas()
    cols = [f"R_flow_s{s}" for s in seeds]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"REFUSING: {path} lacks {missing}")
    return df, [df[c].to_numpy(float) for c in cols]


def binned_equalcount(x, ysim, ymods, nb, qlo=1.0, qhi=99.0):
    """Bin truth and each seed's model on EQUAL-COUNT (quantile) bins.

    WHY NOT THE FIDUCIAL FIGURE'S LINEAR EDGES. `_binned` in plot_fid_flow_figures spaces edges
    linearly (or in log x) between the 1st and 99th percentiles, which is right for a figure meant to
    be read alongside figure 2. It is the WRONG ruler for this one: `Re` piles up at the small end, so
    the whole region the small-size number is quoted on (Re <= 0.386", 25% of the population) falls
    inside the SINGLE leftmost linear bin. A one-bin region cannot show whether a defect flattened
    across it. Equal-count bins put ~25% of the bins inside that region instead, so the effect the
    caption claims is actually visible on the axis.

    Bin centres are the MEDIAN x in the bin, not the edge midpoint: within a quantile bin at the
    crowded end the objects are not centred, and plotting them at the midpoint would slide the curve.

    Returns (centres, sim_mean, sim_sem, model_means[nbin, nseed], counts).
    """
    good = np.isfinite(x) & np.isfinite(ysim)
    for ym in ymods:
        good &= np.isfinite(ym)
    x, ysim = x[good], ysim[good]
    ymods = [ym[good] for ym in ymods]
    lo, hi = np.percentile(x, [qlo, qhi])
    keep = (x >= lo) & (x <= hi)
    x, ysim = x[keep], ysim[keep]
    ymods = [ym[keep] for ym in ymods]
    edges = np.quantile(x, np.linspace(0.0, 1.0, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    cx, s_m, s_e, m_m, cnt = [], [], [], [], []
    for b in range(nb):
        sel = idx == b
        n = int(sel.sum())
        if n < 30:
            continue
        cx.append(float(np.median(x[sel])))
        s_m.append(float(np.mean(ysim[sel])))
        s_e.append(float(np.std(ysim[sel]) / np.sqrt(n)))
        m_m.append([float(np.mean(ym[sel])) for ym in ymods])
        cnt.append(n)
    return (np.array(cx), np.array(s_m), np.array(s_e), np.array(m_m), np.array(cnt))


def region(y, mod, keep, case, rng, nboot=300):
    """mean(model)/mean(truth) - 1 in a region, bootstrapped over CASES (the unit of independence)."""
    ok = keep & np.isfinite(y) & np.isfinite(mod)
    if ok.sum() < 100:
        return np.nan, np.nan, int(ok.sum())
    cen = 100.0 * (float(np.mean(mod[ok])) / float(np.mean(y[ok])) - 1.0)
    uc = np.unique(case[ok])
    bs = []
    for _ in range(nboot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        m = np.isin(case, pick) & ok
        if m.sum() > 100:
            bs.append(100.0 * (float(np.mean(mod[m])) / float(np.mean(y[m])) - 1.0))
    return cen, float(np.std(bs)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fiducial", default=FID)
    ap.add_argument("--relabelled", default=NEW)
    ap.add_argument("--out", default=None, help="output directory (default: <repo>/figures)")
    ap.add_argument("--nbins", type=int, default=16,
                    help="equal-count bins per panel; 16 puts ~4 bins inside the small-size region")
    ap.add_argument("--nboot", type=int, default=300)
    args = ap.parse_args()

    fdf, fmods = load(args.fiducial, SEEDS)
    ndf, nmods = load(args.relabelled, SEEDS)

    # The two dumps must be the SAME rows in the SAME order against the SAME truth, or the residual
    # difference between the curves would fold in a population change. Checked, not assumed.
    if len(fdf) != len(ndf):
        raise SystemExit(f"REFUSING: row counts differ ({len(fdf):,} vs {len(ndf):,})")
    for k in ("case", "input_index"):
        if not np.array_equal(fdf[k].to_numpy(), ndf[k].to_numpy()):
            raise SystemExit(f"REFUSING: `{k}` differs between the dumps; rows are not aligned")
    ysim = fdf["r_sim_self"].to_numpy(float)
    if not np.allclose(ysim, ndf["r_sim_self"].to_numpy(float), equal_nan=True):
        raise SystemExit("REFUSING: the truth column differs between the two dumps")

    case = fdf["case"].to_numpy(np.int64)
    re = fdf["Re_input_p"].to_numpy(float)
    sn = fdf["SN"].to_numpy(float)
    R_glob = float(np.nanmean(ysim))
    fmean, nmean = np.mean(fmods, axis=0), np.mean(nmods, axis=0)
    print(f"aligned: {len(fdf):,} rows x {len(SEEDS)} seeds   truth <R_self> = {R_glob:+.5f}")
    print(f"<R_flow>  fiducial {np.nanmean(fmean):+.5f}   relabelled {np.nanmean(nmean):+.5f}")

    rng = np.random.default_rng(0)
    stats = {}
    print("\n=== region residuals (model/truth - 1), bootstrapped over cases ===")
    for lab, keep in (("ALL", np.ones(len(ysim), bool)),
                      (f"small size Re <= {SMALL_RE}", re <= SMALL_RE),
                      (f"faint MEASURED S/N <= {FAINT_SN}", sn <= FAINT_SN)):
        fc, fs, n = region(ysim, fmean, keep, case, rng, args.nboot)
        nc, ns, _ = region(ysim, nmean, keep, case, rng, args.nboot)
        d, ds = nc - fc, float(np.hypot(fs, ns))
        stats[lab] = (fc, fs, nc, ns, d, ds, n)
        print(f"  {lab:<30} N={n:>9,}   fiducial {fc:+6.2f} +- {fs:.2f} %   "
              f"relabelled {nc:+6.2f} +- {ns:.2f} %   change {d:+6.2f} +- {ds:.2f} "
              f"({'resolved' if abs(d) > 2 * ds else 'not resolved'})")

    # ---------------- the figure ----------------
    set_style()
    panels = [("SN", "primary flux  (MEASURED S/N)", True, None),
              ("Re_input_p", r"primary size  $R_e$  [arcsec]  (TRUE)", False, None),
              ("nbr_flux_near", "neighbour / blend flux  (near shell)", True, 1e-3)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.4, 1]))
    all_res = []
    for j, (col, xlabel, logx, xmin) in enumerate(panels):
        top, bot = axes[0, j], axes[1, j]
        x = fdf[col].to_numpy(float)
        ys, fms, nms = ysim, fmods, nmods
        if xmin is not None:
            keep = np.isfinite(x) & (x > xmin)
            x, ys = x[keep], ysim[keep]
            fms = [m[keep] for m in fmods]
            nms = [m[keep] for m in nmods]
        # Bin BOTH models in one call so they land on identical edges by construction -- binning them
        # separately would let the finite masks differ and slide one curve against the other.
        cx, sm, se, mm_all, cnt = binned_equalcount(x, ys, fms + nms, nb=args.nbins)
        fmm, nmm = mm_all[:, :len(fms)], mm_all[:, len(fms):]

        print(f"\n  per-bin, {col} (equal-count bins, {args.nbins} requested):")
        print(f"    {'centre':>10} {'N':>9} {'truth':>9} {'ruler%':>9} {'scored%':>9}")
        for k in range(len(cx)):
            fr = 100.0 * (fmm[k].mean() / sm[k] - 1.0)
            nr = 100.0 * (nmm[k].mean() / sm[k] - 1.0)
            print(f"    {cx[k]:10.4f} {cnt[k]:9,} {sm[k]:9.4f} {fr:+9.2f} {nr:+9.2f}")

        if col == "Re_input_p":
            # mark the region the small-size number is quoted on, so the panel and the caption
            # cannot be read as describing different cuts
            x0 = float(np.nanmin(x))
            top.axvspan(x0, SMALL_RE, color="#000000", alpha=0.045, lw=0, zorder=0)
            bot.axvspan(x0, SMALL_RE, color="#000000", alpha=0.045, lw=0, zorder=0)

        top.axhline(R_glob, color="#cccccc", lw=1.0, zorder=0)
        top.fill_between(cx, sm - se, sm + se, color=BLUE, alpha=0.30, lw=0, zorder=1)
        top.plot(cx, sm, "-o", color=BLUE, ms=7, lw=1.8, zorder=5)
        for mm, cl, mk in ((fmm, VERM, "s"), (nmm, GREEN, "^")):
            top.fill_between(cx, mm.min(1), mm.max(1), color=cl, alpha=0.22, lw=0, zorder=2)
            top.plot(cx, mm.mean(1), "--", marker=mk, color=cl, ms=7, lw=1.8, mfc="white",
                     mew=1.6, zorder=4)

        bot.axhline(0.0, color="#cccccc", lw=1.0, zorder=0)
        rse = np.abs(se / sm) * 100.0
        bot.fill_between(cx, -rse, rse, color=BLUE, alpha=0.25, lw=0, zorder=1)
        for mm, cl, mk in ((fmm, VERM, "s"), (nmm, GREEN, "^")):
            res = (mm.mean(1) / sm - 1.0) * 100.0
            bot.fill_between(cx, (mm.min(1) / sm - 1.0) * 100.0, (mm.max(1) / sm - 1.0) * 100.0,
                             color=cl, alpha=0.22, lw=0, zorder=2)
            bot.plot(cx, res, "--", marker=mk, color=cl, ms=6, lw=1.6, mfc="white", mew=1.4,
                     zorder=3)
            all_res.append(res)

        bot.set_xlabel(xlabel)
        if logx:
            top.set_xscale("log")
            bot.set_xscale("log")
            bot.xaxis.set_minor_formatter(NullFormatter())
            bot.xaxis.set_major_formatter(ScalarFormatter())
            bot.ticklabel_format(axis="x", style="plain")
        if j:
            top.tick_params(labelleft=False)
            bot.tick_params(labelleft=False)

    ymin = min(a.get_ylim()[0] for a in axes[0])
    ymax = max(a.get_ylim()[1] for a in axes[0])
    flat = np.concatenate(all_res)
    flat = flat[np.isfinite(flat)]
    lim = max(8.0, float(np.nanpercentile(np.abs(flat), 90)) * 1.6)
    n_clip = int(np.sum(np.abs(flat) > lim))
    for a in axes[0]:
        a.set_ylim(ymin, ymax)
    for a in axes[1]:
        a.set_ylim(-lim, lim)
    axes[0, 0].set_ylabel(r"mean SELF response  $R_{\rm self}$")
    axes[1, 0].set_ylabel(r"model / truth $-$ 1  [%]")
    axes[0, 0].legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ms=7, lw=1.8,
               label=r"half-shear truth  $R_{\rm self}$"),
        Patch(fc=BLUE, alpha=0.30, label="truth s.e. on mean"),
        Line2D([], [], color=VERM, marker="s", ms=7, lw=1.8, ls="--", mfc="white",
               label="flow trained on the RULER label"),
        Line2D([], [], color=GREEN, marker="^", ms=7, lw=1.8, ls="--", mfc="white",
               label="flow trained on the SCORED label"),
        Patch(fc="#999999", alpha=0.22, label=f"{len(SEEDS)}-seed range")], loc="upper left")

    fa = stats[f"small size Re <= {SMALL_RE}"]
    fb = stats[f"faint MEASURED S/N <= {FAINT_SN}"]
    fc_all = stats["ALL"]
    axes[1, 1].annotate(f"shaded: $R_e\\leq{SMALL_RE}$\"\n"
                        f"ruler label {fa[0]:+.2f}$\\pm${fa[1]:.2f}%\n"
                        f"scored label {fa[2]:+.2f}$\\pm${fa[3]:.2f}%",
                        xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8, color=MUTED,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.text(0.5, -0.02,
             "half-shear legs, self-response isolated on $\\hat{g}_p$; flow ONLY, no "
             "$R_{\\rm blend}$.  "
             f"N={len(fdf):,} objects, identical rows, forward extraction, seeds "
             f"{'/'.join(str(s) for s in SEEDS)}, equal-count bins"
             + (f", {n_clip} residual bins clipped" if n_clip else "") + "\n",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.text(0.5, -0.055,
             "the two model curves differ ONLY in which g=0 measurement the training label was built "
             "from -- same architecture, weight, seeds and rows.  "
             f"ALL: {fc_all[0]:+.2f}$\\pm${fc_all[1]:.2f}% $\\to$ {fc_all[2]:+.2f}$\\pm${fc_all[3]:.2f}%"
             f"  |  $R_e\\leq{SMALL_RE}$\": {fa[0]:+.2f}$\\pm${fa[1]:.2f}% $\\to$ "
             f"{fa[2]:+.2f}$\\pm${fa[3]:.2f}% ({fa[4]:+.2f}$\\pm${fa[5]:.2f}, "
             f"{'resolved' if abs(fa[4]) > 2 * fa[5] else 'not resolved'})"
             f"  |  measured S/N$\\leq${FAINT_SN}: {fb[0]:+.2f}$\\pm${fb[1]:.2f}% $\\to$ "
             f"{fb[2]:+.2f}$\\pm${fb[3]:.2f}% ({fb[4]:+.2f}$\\pm${fb[5]:.2f}, "
             f"{'resolved' if abs(fb[4]) > 2 * fb[5] else 'not resolved'})\n"
             "SELF-CONSISTENCY between label and score, NOT evidence that either g=0 measurement is "
             "unbiased.  3 seeds -- no $m$ is quoted and constgold is never read.",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(here, "figures")
    os.makedirs(out_dir, exist_ok=True)
    save(fig, out_dir, "fig5_selfresp_labelfix")
    print("\nPLOT_SELFRESP_LABELFIX_DONE")


if __name__ == "__main__":
    main()
