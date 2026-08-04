"""FIGURE 2, SIDE BY SIDE: the FULL model (flow + tuned emulator) for both flows, on constgold.

WHAT THIS ADDS OVER THE FIG-5 COMPARISON. `plot_selfresp_labelfix.py` puts the two flows on the
half-shear SELF response with NO `R_blend` -- deliberately, so a flow error cannot be masked by an
emulator error. This figure asks the complementary question: once the tuned in-domain emulator's
`R_blend` is added back and the target is the constgold TOTAL response, does the label fix still show
up, or does the emulator absorb it?

  fiducial   V2 dom6x6 flow (tag ablate_s2c_lt500_dom6x6), pin fit to the ruler's `R_snc`
  relabelled same architecture/weight/seeds, pin fit to `r_sim_self` (WORKLOG 2026-08-03w/x)

  model = R_flow(seed) + R_blend(tuned emulator)      -- the fiducial estimator, parameter-free

Both sides use the SAME three seeds (501/502/503), the SAME rows, the SAME `R_sim`, and the SAME
`R_blend` per object. The only difference is which g=0 measurement built the flow's training label.

THREE TRAPS THIS SCRIPT REFUSES RATHER THAN PAPERS OVER.
 1. The dumps carry the OLD emulator's `R_blend`; the fiducial model uses the TUNED in-domain one, so
    the column is replaced by a join. The tuned emulator returns NOTHING outside its training box,
    and zero-filling those rows is the documented failure (spurious +28.9% m, job 15366950). They are
    DROPPED, the retained fraction is printed, and a low match refuses.
 2. Both dump sets must be the same deterministic pass over the same catalogue. Row count and key
    checksums are checked; a mismatch refuses rather than stacking misaligned columns.
 3. **NOTHING HERE IS AN `m`.** The residual plotted is model/truth - 1, which at 3 seeds carries the
    full per-seed offset -- the seed convention requires 16 for any quoted `m`, and AGENTS.md is
    explicit that the offset does NOT cancel in an absolute model-vs-sim ratio. The trustworthy
    quantity in this figure is the CHANGE between the two curves, measured on the same seeds and the
    same rows. Absolute levels are shown as context and must not be quoted.

FIREWALL. constgold is EVALUATION ONLY. Nothing is fit, tuned or selected here; the label change
being visualised was decided on the half-shear ruler (03q/03w), never on a constgold number.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
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
from plot_fid_flow_figures import (  # noqa: E402
    BLEND_LOOKUP, BLUE, CONST_CAT, CROWD, DUMPDIR, GREEN, MIN_MATCH, MUTED, VERM, _read_key_table,
    save, set_style)
from plot_selfresp_labelfix import binned_equalcount, region  # noqa: E402

FID_TAG = "ablate_s2c_lt500_dom6x6"
NEW_TAG = "perobj_DUMPLBL_rw450"
SEEDS = [501, 502, 503]
SMALL_RE = 0.386
FAINT_MAG = 25.0


def dump_paths(tag):
    out = {}
    for p in sorted(glob.glob(str(DUMPDIR / f"{tag}_perobj_s*.feather"))):
        mo = re.search(r"_s(\d+)\.feather$", p)
        if mo:
            out[int(mo.group(1))] = p
    return out


def load_tag(tag, seeds):
    """(ref_frame, R_flow[nseed, nobj]) for one tag, seeds checked for row alignment."""
    paths = dump_paths(tag)
    missing = [s for s in seeds if s not in paths]
    if missing:
        raise SystemExit(f"REFUSING: tag `{tag}` has no constgold dump for seeds {missing}. "
                         f"Found {sorted(paths)}. Run jobs/job_s2c_domain_eval.sh with TAG={tag}.")
    ref = pf.read_table(paths[seeds[0]],
                        columns=["case", "input_index", "r_sim", "R_blend"]).to_pandas()
    ck = (len(ref), int(ref["case"].sum()), int(ref["input_index"].sum()))
    flows = []
    for s in seeds:
        if s != seeds[0]:
            k = pf.read_table(paths[s], columns=["case", "input_index"]).to_pandas()
            if (len(k), int(k["case"].sum()), int(k["input_index"].sum())) != ck:
                raise SystemExit(f"REFUSING: {tag} s{s} rows differ from s{seeds[0]}; the R_flow "
                                 "columns cannot be stacked positionally")
            del k
        flows.append(pf.read_table(paths[s], columns=["R_flow"])
                     .column("R_flow").to_numpy().astype(np.float64))
    print(f"  {tag}: {len(seeds)} seeds x {len(ref):,} rows aligned")
    return ref, np.vstack(flows), ck


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fid-tag", default=FID_TAG)
    ap.add_argument("--new-tag", default=NEW_TAG)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nbins", type=int, default=16,
                    help="equal-count bins per panel (see plot_selfresp_labelfix for why not linear)")
    ap.add_argument("--nboot", type=int, default=300)
    args = ap.parse_args()

    print("loading constgold per-object dumps:")
    fref, fflow, fck = load_tag(args.fid_tag, SEEDS)
    nref, nflow, nck = load_tag(args.new_tag, SEEDS)
    if fck != nck:
        raise SystemExit(f"REFUSING: the two tags were not scored on the same rows "
                         f"({fck} vs {nck}); a residual difference would fold in a population change")
    if not np.allclose(fref["r_sim"].to_numpy(float), nref["r_sim"].to_numpy(float),
                       equal_nan=True):
        raise SystemExit("REFUSING: `r_sim` differs between the two dump sets")

    # ---- tuned emulator R_blend, unmatched rows DROPPED (never zero-filled) -------------------
    lk = pf.read_table(BLEND_LOOKUP, memory_map=True).to_pandas()
    j = fref[["case", "input_index"]].merge(lk, on=["case", "input_index"], how="left",
                                            suffixes=("", "_tuned"))
    rb = j["R_blend"].to_numpy(float)
    keep = np.isfinite(rb)
    frac = float(keep.mean())
    print(f"tuned R_blend matched {100*frac:.2f}% of dump rows "
          f"(<R_blend> old {fref['R_blend'].mean():.4f} -> tuned {np.nanmean(rb):.4f})")
    if frac < MIN_MATCH:
        raise SystemExit(f"REFUSING: tuned emulator covers only {100*frac:.1f}% of rows; zero-filling"
                         " the rest is what produced a spurious +28.9% m (job 15366950).")
    ref = fref.loc[keep].reset_index(drop=True)
    rb = rb[keep]
    fflow, nflow = fflow[:, keep], nflow[:, keep]
    print(f"in-domain rows retained: {len(ref):,}")

    ysim = ref["r_sim"].to_numpy(float)
    fmods = [fflow[i] + rb for i in range(len(SEEDS))]
    nmods = [nflow[i] + rb for i in range(len(SEEDS))]
    case = ref["case"].to_numpy(np.int64)
    print(f"\n<R_sim>={np.nanmean(ysim):.5f}  <R_blend>={np.nanmean(rb):.5f}")
    print(f"<R_flow>  fiducial {np.nanmean(fflow):.5f}   relabelled {np.nanmean(nflow):.5f}")

    # ---- properties for the three panels ------------------------------------------------------
    key = ["case", "input_index"]
    cases = set(ref["case"].unique().tolist())
    props = _read_key_table(CONST_CAT, key + ["Re_input_p", "S/N_plus", "r_input_p"], cases)
    df = ref.merge(props, on=key, how="left")
    del props
    blf = _read_key_table(CROWD, key + ["nbr_flux_near"], cases)
    df = df.merge(blf, on=key, how="left")
    del blf
    if len(df) != len(ref):
        raise SystemExit("REFUSING: the property merge changed the row count")

    re_p = df["Re_input_p"].to_numpy(float)
    mag = df["r_input_p"].to_numpy(float)
    rng = np.random.default_rng(0)
    fmean, nmean = np.mean(fmods, axis=0), np.mean(nmods, axis=0)

    stats = {}
    print("\n=== region residuals, model/truth - 1, bootstrapped over cases ===")
    print("    (NOT an `m`: 3 seeds, and the per-seed offset does not cancel in a model-vs-sim "
          "ratio.\n     The CHANGE column is the quantity this figure is for.)")
    for lab, kp in (("ALL", np.ones(len(df), bool)),
                    (f"small size Re <= {SMALL_RE}", re_p <= SMALL_RE),
                    (f"faint TRUE mag > {FAINT_MAG}", mag > FAINT_MAG)):
        fc, fs, n = region(ysim, fmean, kp, case, rng, args.nboot)
        nc, ns, _ = region(ysim, nmean, kp, case, rng, args.nboot)
        d, ds = nc - fc, float(np.hypot(fs, ns))
        stats[lab] = (fc, fs, nc, ns, d, ds, n)
        print(f"  {lab:<28} N={n:>9,}   fiducial {fc:+6.2f} +- {fs:.2f} %   "
              f"relabelled {nc:+6.2f} +- {ns:.2f} %   change {d:+6.2f} +- {ds:.2f} "
              f"({'resolved' if abs(d) > 2 * ds else 'not resolved'})")

    # ---- the figure ---------------------------------------------------------------------------
    set_style()
    panels = [("S/N_plus", r"primary flux  (S/N$_+$)", True, None),
              ("Re_input_p", r"primary size  $R_e$  [arcsec]  (TRUE)", False, None),
              ("nbr_flux_near", "neighbour / blend flux  (near shell)", True, 1e-3)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.4, 1]))
    R_glob = float(np.nanmean(ysim))
    all_res = []
    for k, (col, xlabel, logx, xmin) in enumerate(panels):
        top, bot = axes[0, k], axes[1, k]
        x = df[col].to_numpy(float)
        ys, fms, nms = ysim, fmods, nmods
        if xmin is not None:
            kp = np.isfinite(x) & (x > xmin)
            x, ys = x[kp], ysim[kp]
            fms = [m[kp] for m in fmods]
            nms = [m[kp] for m in nmods]
        cx, sm, se, mm_all, cnt = binned_equalcount(x, ys, fms + nms, nb=args.nbins)
        fmm, nmm = mm_all[:, :len(fms)], mm_all[:, len(fms):]

        print(f"\n  per-bin, {col} (equal-count bins):")
        print(f"    {'centre':>10} {'N':>9} {'truth':>9} {'ruler%':>9} {'scored%':>9}")
        for b in range(len(cx)):
            print(f"    {cx[b]:10.4f} {cnt[b]:9,} {sm[b]:9.4f} "
                  f"{100.0 * (fmm[b].mean() / sm[b] - 1.0):+9.2f} "
                  f"{100.0 * (nmm[b].mean() / sm[b] - 1.0):+9.2f}")

        if col == "Re_input_p":
            top.axvspan(float(np.nanmin(x)), SMALL_RE, color="#000000", alpha=0.045, lw=0, zorder=0)
            bot.axvspan(float(np.nanmin(x)), SMALL_RE, color="#000000", alpha=0.045, lw=0, zorder=0)

        top.axhline(R_glob, color="#cccccc", lw=1.0, zorder=0)
        top.fill_between(cx, sm - se, sm + se, color=BLUE, alpha=0.30, lw=0, zorder=1)
        top.plot(cx, sm, "-o", color=BLUE, ms=7, lw=1.8, zorder=5)
        bot.axhline(0.0, color="#cccccc", lw=1.0, zorder=0)
        bot.fill_between(cx, -np.abs(se / sm) * 100.0, np.abs(se / sm) * 100.0,
                         color=BLUE, alpha=0.25, lw=0, zorder=1)
        for mm, cl, mk in ((fmm, VERM, "s"), (nmm, GREEN, "^")):
            top.fill_between(cx, mm.min(1), mm.max(1), color=cl, alpha=0.22, lw=0, zorder=2)
            top.plot(cx, mm.mean(1), "--", marker=mk, color=cl, ms=7, lw=1.8, mfc="white",
                     mew=1.6, zorder=4)
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
        if k:
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
    axes[0, 0].set_ylabel("mean shear response  $R$")
    axes[1, 0].set_ylabel("model / truth $-$ 1  [%]")
    axes[0, 0].legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ms=7, lw=1.8, label=r"constgold truth  $R_{\rm sim}$"),
        Patch(fc=BLUE, alpha=0.30, label="truth s.e. on mean"),
        Line2D([], [], color=VERM, marker="s", ms=7, lw=1.8, ls="--", mfc="white",
               label=r"RULER-label flow $+\,R_{\rm blend}$"),
        Line2D([], [], color=GREEN, marker="^", ms=7, lw=1.8, ls="--", mfc="white",
               label=r"SCORED-label flow $+\,R_{\rm blend}$"),
        Patch(fc="#999999", alpha=0.22, label=f"{len(SEEDS)}-seed range")], loc="best")

    sa = stats[f"small size Re <= {SMALL_RE}"]
    axes[1, 1].annotate(f"shaded: $R_e\\leq{SMALL_RE}$\"\n"
                        f"ruler {sa[0]:+.2f}$\\pm${sa[1]:.2f}%\n"
                        f"scored {sa[2]:+.2f}$\\pm${sa[3]:.2f}%",
                        xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8, color=MUTED,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ca = stats["ALL"]
    cb = stats[f"faint TRUE mag > {FAINT_MAG}"]
    fig.text(0.5, -0.02,
             r"model = $R_{\rm flow}$ + $R_{\rm blend}$ (tuned in-domain emulator); constgold, "
             f"in-domain, N={len(df):,} rows, seeds {'/'.join(str(s) for s in SEEDS)}, "
             f"equal-count bins" + (f", {n_clip} residual bins clipped" if n_clip else "") + "\n",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.text(0.5, -0.055,
             "the two model curves differ ONLY in which g=0 measurement built the flow's training "
             f"label.  ALL: {ca[0]:+.2f}$\\pm${ca[1]:.2f}% $\\to$ {ca[2]:+.2f}$\\pm${ca[3]:.2f}%"
             f"  |  $R_e\\leq{SMALL_RE}$\": {sa[0]:+.2f}$\\pm${sa[1]:.2f}% $\\to$ "
             f"{sa[2]:+.2f}$\\pm${sa[3]:.2f}%  |  true mag>{FAINT_MAG}: "
             f"{cb[0]:+.2f}$\\pm${cb[1]:.2f}% $\\to$ {cb[2]:+.2f}$\\pm${cb[3]:.2f}%\n"
             "NOT an $m$: 3 seeds, and the per-seed offset does NOT cancel in a model-vs-sim ratio "
             "(16 seeds required). Read the CHANGE, not the level.",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(here, "figures")
    os.makedirs(out_dir, exist_ok=True)
    save(fig, out_dir, "fig2_labelfix_with_emulator")
    print("\nPLOT_FIG2_LABELFIX_DONE")


if __name__ == "__main__":
    main()
