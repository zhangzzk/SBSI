"""FIGURE 2, REMADE WITH FLOW #2 AS THE BLEND TERM (owner request, 2026-08-03).

  figures/fid_fig2_response_vs_properties_flow2.png

WHAT CHANGES vs `fid_fig2_response_vs_properties.png`. Only the SECOND term of the model. Both
curves share the SAME self-response flow (V2 dom6x6, 16 seeds) and the SAME truth `r_sim`; they
differ only in where `R_blend` comes from:

    fiducial   R_model = R_flow(flow #1) + R_blend(BlendEMU `lsst_r_extnbr_indom_tuned`, native)
    flow #2    R_model = R_flow(flow #1) + R_blend(blend flow, g = 0.2, 16 seeds, all pairs < 7")

The certified `plot_fid_flow_figures.py` is NOT modified -- fig2 there remains the fiducial artifact.
This is an additional figure, and it is a COMPARISON: showing flow #2 alone would invite reading its
residuals as absolute model error, when the informative quantity is the difference against a baseline
scored on the same rows.

WHY THIS FIGURE IS WORTH MAKING. On aggregate `m` the two are nearly tied (BlendEMU -0.126%,
flow #2 -0.206%, WORKLOG 2026-08-03g). But the per-bin ruler says they are NOT equivalent: on the
primary-magnitude conditional the flow/emulator chi2 ratio is 4.30, i.e. flow #2 is much worse
exactly where this figure's LEFT panel looks. A near-tie in the aggregate on top of a 4.3x-worse
conditional is a cancellation, and the left panel is where it should become visible. If the two
residual curves in panel 1 diverge while the aggregate agrees, that IS the cancellation, drawn.

EACH MODEL IS SCORED ON THE PAIR LIST IT WAS TRAINED FOR -- the trap in AGENTS.md "Two traps".
BlendEMU keeps its NATIVE whole-field prediction (r_max 10", its own neighbour cuts); flow #2 keeps
all pairs inside 7". Re-scoring the emulator on the flow's list (or vice versa) handicaps whichever
model relies on the excluded pairs and has already changed a verdict ninefold in this project. The
row POPULATION is intersected so both curves describe the same galaxies; the PAIR LISTS behind each
model's R_blend are deliberately left different, and that is stated on the figure.

SEED BANDS ARE PAIRED i-WITH-i. The fiducial band is flow #1's 16-seed range (BlendEMU is
deterministic). For flow #2 both terms carry seed noise, so seed s of flow #1 is paired with seed s
of flow #2 and the band is the range over those 16 pairings. Seeds are independent and exchangeable
between the two ensembles, so the pairing is arbitrary and the 16 pairings are a valid sample of the
COMBINED model spread -- which is the honest band for a two-noisy-term model. It is wider than the
fiducial band by construction; that extra width is real uncertainty, not a defect of the drawing.

CAVEAT THAT MUST TRAVEL WITH ANY NUMBER READ OFF THIS FIGURE. Flow #2's labels are measured at
g = 0.2, where the blend response carries a curvature contamination of ~+1.46% relative to the g -> 0
limit (`scripts/measure_response_curvature.py`; c = +0.026 +- 0.054, 2-sigma bound 6.08%). That is a
systematic on the flow #2 curve which the BlendEMU curve does not share in the same amount, and it
is NOT corrected here -- correcting it silently is exactly what AGENTS.md "Numerical Integrity"
forbids. It is printed in the run log and noted on the figure.

FIREWALL: constgold is read to SCORE only. Nothing here tunes or selects a model.

PNG only, no gridlines, no title (owner's plotting defaults). Okabe-Ito; the two model sources are a
genuine categorical pair (different identities), so they take two hues, while seeds within a model
are exchangeable replicates and share their model's hue.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.feather as pf  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotting.plot_fid_flow_figures import (  # noqa: E402
    BLUE, VERM, GREEN, MUTED, CONST_CAT, CROWD,
    _binned, _read_key_table, load_dumps, save, set_style,
)

FLOW2_LOOKUP = ("/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow/"
                "blend_lookup_ens16g02_allpairs7_c40-139.feather")
MIN_MATCH = 0.95   # all-pairs-7" covered 11,670,851 / 11,674,408 = 99.97% in the m run
CURV_PCT = 1.46    # g=0.2 blend-label curvature contamination vs the g->0 limit, WORKLOG 2026-08-03d


def load_flow2(ref, seeds):
    """Attach flow #2's ensemble-mean and per-seed R_blend to the fiducial `ref` rows.

    Returns (rb2_mean, rb2_seeds[nseed, nrow], keep_mask). Unmatched rows are DROPPED, never
    zero-filled -- zero-filling an unmatched blend term is the documented failure mode that produced
    a spurious +28.9% m (job 15366950).
    """
    cols = ["case", "input_index", "R_blend_flow"] + [f"R_blend_flow_s{s}" for s in seeds]
    lk = pf.read_table(FLOW2_LOOKUP, columns=cols, memory_map=True).to_pandas()
    missing = [c for c in cols if c not in lk.columns]
    if missing:
        raise SystemExit(f"REFUSING: flow #2 lookup is missing {missing}")
    n_lk = len(lk)
    lk = lk.drop_duplicates(subset=["case", "input_index"])
    if len(lk) != n_lk:
        raise SystemExit(f"REFUSING: flow #2 lookup has duplicate (case,input_index) keys "
                         f"({n_lk:,} rows -> {len(lk):,} unique). A merge would multiply rows.")

    n_before = len(ref)
    j = ref[["case", "input_index"]].merge(lk, on=["case", "input_index"], how="left")
    if len(j) != n_before:
        raise SystemExit(f"REFUSING: merge changed the row count ({n_before:,} -> {len(j):,}).")
    del lk

    rb2 = j["R_blend_flow"].to_numpy(float)
    keep = np.isfinite(rb2)
    frac = float(keep.mean())
    print(f"flow #2 R_blend matched {100*frac:.2f}% of the {n_before:,} fiducial rows "
          f"(<R_blend> flow2 {np.nanmean(rb2):.4f})")
    if frac < MIN_MATCH:
        raise SystemExit(f"REFUSING: flow #2 covers only {100*frac:.1f}% of the fiducial rows "
                         f"(expected ~99.97%). Dropping that many rows would change the population.")
    rb2_seeds = np.vstack([j[f"R_blend_flow_s{s}"].to_numpy(np.float32) for s in seeds])
    return rb2, rb2_seeds, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_style()

    # ---- fiducial arm: flow #1 seeds + NATIVE BlendEMU, exactly as fid_fig2 builds it ----------
    seeds, ref, flows = load_dumps()
    if not seeds:
        raise SystemExit("no per-object dumps found")

    # ---- flow #2 arm: same rows, blend term swapped -------------------------------------------
    rb2, rb2_seeds, keep = load_flow2(ref, seeds)
    ref = ref.loc[keep].reset_index(drop=True)
    flows = flows[:, keep]
    rb2, rb2_seeds = rb2[keep], rb2_seeds[:, keep]
    print(f"common rows for the comparison: {len(ref):,}")

    key = ["case", "input_index"]
    df = ref
    cases = set(df["case"].unique().tolist())
    props = _read_key_table(CONST_CAT, key + ["Re_input_p", "S/N_plus"], cases)
    df = df.merge(props, on=key, how="left")
    del props
    blf = _read_key_table(CROWD, key + ["nbr_flux_near"], cases)
    df = df.merge(blf, on=key, how="left")
    del blf

    ysim = df["r_sim"].to_numpy(float)
    rb_emu = df["R_blend"].to_numpy(float)          # native tuned emulator
    ymod_emu = [flows[i].astype(float) + rb_emu for i in range(len(seeds))]
    ymod_f2 = [flows[i].astype(float) + rb2_seeds[i].astype(float) for i in range(len(seeds))]
    R_sim_glob = float(np.nanmean(ysim))

    # ---- anchor the figure to the already-published aggregate m -------------------------------
    def agg_m(ymods):
        per = [100.0 * (np.nanmean(ysim) / np.nanmean(ym) - 1.0) for ym in ymods]
        return float(np.mean(per)), float(np.std(per, ddof=1) / np.sqrt(len(per)))

    m_emu, s_emu = agg_m(ymod_emu)
    m_f2, s_f2 = agg_m(ymod_f2)
    print(f"\nAGGREGATE ON THESE ROWS (cross-check against WORKLOG 2026-08-03g):")
    print(f"  R_sim                      = {R_sim_glob:.4f}")
    print(f"  <R_flow>                   = {float(np.nanmean(flows)):.4f}")
    print(f"  <R_blend> BlendEMU native  = {float(np.nanmean(rb_emu)):.4f}   "
          f"m = {m_emu:+.3f} +- {s_emu:.3f} %   (published -0.126)")
    print(f"  <R_blend> flow #2 g=0.2    = {float(np.nanmean(rb2)):.4f}   "
          f"m = {m_f2:+.3f} +- {s_f2:.3f} %   (published -0.206)")
    # CURV_PCT is ALREADY a percentage of R_blend, so converting it to points of m is
    # (CURV_PCT/100) * <R_blend> / R_sim * 100 = CURV_PCT * <R_blend> / R_sim. Multiplying by a
    # further 100 double-counts the conversion and inflates 0.23 pt to 23 pt.
    print(f"  flow #2 carries an UNCORRECTED g=0.2 curvature systematic of ~{CURV_PCT:.2f}% on "
          f"R_blend (~{CURV_PCT*float(np.nanmean(rb2))/R_sim_glob:.2f} pt of m; the independent "
          f"estimate in WORKLOG 2026-08-03g is 0.22 +- 0.44 pt).")

    panels = [("S/N_plus", r"primary flux  (S/N$_+$)", True, None),
              ("Re_input_p", r"primary size  $R_e$  [arcsec]", False, None),
              ("nbr_flux_near", "neighbour / blend flux  (near shell)", True, 1e-3)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.4, 1]))
    all_res, all_se = [], []
    for jx, (col, xlabel, logx, xmin) in enumerate(panels):
        top, bot = axes[0, jx], axes[1, jx]
        x = df[col].to_numpy(float)
        ys, ye, yf = ysim, ymod_emu, ymod_f2
        if xmin is not None:
            k = np.isfinite(x) & (x > xmin)
            x, ys = x[k], ysim[k]
            ye = [v[k] for v in ymod_emu]
            yf = [v[k] for v in ymod_f2]
        # one shared binning call per arm; identical edges because x and the quantile range match
        cx, sm, se, mm_e = _binned(x, ys, ye, nb=12, logx=logx)
        _, _, _, mm_f = _binned(x, ys, yf, nb=12, logx=logx)
        me, mf = mm_e.mean(1), mm_f.mean(1)

        top.axhline(R_sim_glob, color="#cccccc", lw=1.0, zorder=0)
        top.fill_between(cx, sm - se, sm + se, color=BLUE, alpha=0.30, lw=0, zorder=1)
        top.plot(cx, sm, "-o", color=BLUE, ms=7, lw=1.8, zorder=5)
        top.fill_between(cx, mm_e.min(1), mm_e.max(1), color=VERM, alpha=0.22, lw=0, zorder=2)
        top.plot(cx, me, "--s", color=VERM, ms=7, lw=1.8, mfc="white", mew=1.6, zorder=4)
        top.fill_between(cx, mm_f.min(1), mm_f.max(1), color=GREEN, alpha=0.22, lw=0, zorder=2)
        top.plot(cx, mf, "-.^", color=GREEN, ms=7, lw=1.8, mfc="white", mew=1.6, zorder=3)

        for mmv, mv, colr, mk, ls in ((mm_e, me, VERM, "s", "--"), (mm_f, mf, GREEN, "^", "-.")):
            res = (mv / sm - 1.0) * 100.0
            all_res.append(res)
            bot.fill_between(cx, (mmv.min(1) / sm - 1) * 100, (mmv.max(1) / sm - 1) * 100,
                             color=colr, alpha=0.22, lw=0, zorder=2)
            bot.plot(cx, res, ls, marker=mk, color=colr, ms=6, lw=1.6, mfc="white", mew=1.4,
                     zorder=3)
        rse = np.abs(se / sm) * 100.0
        all_se.append(rse)
        bot.axhline(0.0, color="#cccccc", lw=1.0, zorder=0)
        bot.fill_between(cx, -rse, rse, color=BLUE, alpha=0.25, lw=0, zorder=1)
        bot.set_xlabel(xlabel)
        if logx:
            top.set_xscale("log")
            bot.set_xscale("log")
            bot.xaxis.set_minor_formatter(NullFormatter())
            bot.xaxis.set_major_formatter(ScalarFormatter())
            bot.ticklabel_format(axis="x", style="plain")
        if jx:
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
    if n_clip:
        axes[1, 1].annotate(f"{n_clip} bin(s) clipped (truth $R\\approx$0)",
                            xy=(0.5, 0.04), xycoords="axes fraction", ha="center",
                            fontsize=7.5, color=MUTED)

    axes[0, 0].set_ylabel("mean shear response  $R$")
    axes[1, 0].set_ylabel("model / truth $-$ 1  [%]")
    axes[0, 0].legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ms=7, lw=1.8, label=r"sim truth  $r_{\rm sim}$"),
        Patch(fc=BLUE, alpha=0.30, label="truth s.e. on mean"),
        Line2D([], [], color=VERM, marker="s", ms=7, lw=1.8, ls="--", mfc="white",
               label=r"$R_{\rm flow}$ + BlendEMU  (fiducial)"),
        Line2D([], [], color=GREEN, marker="^", ms=7, lw=1.8, ls="-.", mfc="white",
               label=r"$R_{\rm flow}$ + flow #2  ($g$=0.2)"),
        Patch(fc=MUTED, alpha=0.22, label=f"{len(seeds)}-seed range")], loc="lower right")
    # NOT loc="best": the response curve leaves the upper-left of panel 1 empty and the lower-left
    # occupied, so "best" lands the legend on top of the population annotation. Both are pinned.
    axes[0, 0].annotate(f"constgold, true neighbours\nN={len(df):,} objects, cases "
                        f"{int(df['case'].min())}-{int(df['case'].max())}",
                        xy=(0.02, 0.98), xycoords="axes fraction", va="top",
                        fontsize=8, color=MUTED,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.text(0.5, -0.02,
             f"same rows, same $R_{{\\rm flow}}$ ({len(seeds)} seeds); only the blend term differs.  "
             f"BlendEMU native 10\" vs flow #2 all pairs <7\" -- each on the pair list it was trained "
             f"for.  aggregate $m$: {m_emu:+.3f}% vs {m_f2:+.3f}%.  flow #2 carries an uncorrected "
             f"$g$=0.2 curvature systematic (~{CURV_PCT:.1f}% on $R_{{\\rm blend}}$).",
             ha="center", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    p = save(fig, args.out_dir, "fid_fig2_response_vs_properties_flow2")

    # ---- print the residual tables so the figure can be read as numbers too -------------------
    # The `truth s.e.` row is the NOISE FLOOR of this comparison. A bin where |BlendEMU - flow #2|
    # is below it is NOT a resolved difference between the two models, however suggestive the rms
    # looks -- the same trap as chi2/dof < 1 (WORKLOG 2026-08-03f). Printed so the rms lines below
    # can never be read without it.
    print("\nper-panel residual (model/truth - 1, %), ensemble mean:")
    for (col, xlabel, _, _), i in zip(panels, range(0, len(all_res), 2)):
        re_, rf_, se_ = all_res[i], all_res[i + 1], all_se[i // 2]
        d = np.abs(rf_ - re_)
        print(f"  {xlabel}")
        print(f"    BlendEMU    : " + " ".join(f"{v:+6.2f}" for v in re_))
        print(f"    flow #2     : " + " ".join(f"{v:+6.2f}" for v in rf_))
        print(f"    |difference|: " + " ".join(f"{v:6.2f}" for v in d))
        print(f"    truth s.e.  : " + " ".join(f"{v:6.2f}" for v in se_))
        print(f"    rms         : BlendEMU {np.sqrt(np.nanmean(re_**2)):.2f}%   "
              f"flow #2 {np.sqrt(np.nanmean(rf_**2)):.2f}%   "
              f"| bins where the two models differ by more than the truth s.e.: "
              f"{int(np.sum(d > se_))}/{len(d)}")
    print("\nFIG2_FLOW2_DONE", p, flush=True)


if __name__ == "__main__":
    main()
