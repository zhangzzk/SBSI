"""FIDUCIAL-model remakes of figures 1-4: V2 dom6x6 flow + TUNED in-domain blend emulator.

  figures/fid_fig1_seed_loss.png              -- validation NLL vs epoch, all 16 dom6x6 seeds
  figures/fid_fig2_response_vs_properties.png -- model vs true response across primary flux,
                                                 primary size and neighbour/blend flux
  figures/fid_fig3_bias_true_neighbours.png   -- per-seed multiplicative bias m
  figures/fid_fig4_selection_near_domain.png  -- selection bias at near-domain MEASURED cuts:
                                                 the shift, the residual bias m at the cut, and the
                                                 selection-induced excess dm over no-cut
  figures/fid_fig5_selfresp_halfshear.png     -- SELF response only: flow alone vs half-shear
                                                 R_self (fig2 without the emulator's freedom)

WHAT MAKES THESE THE FIDUCIAL SET (owner, 2026-07-30): the flow is the V2 dom6x6 ensemble and
R_blend comes from the TUNED in-domain emulator (`lsst_r_extnbr_indom_tuned`), NOT the value stored
in the dumps. The dumps own `R_blend` column is IGNORED and replaced by a join against
`results/blend_lookup_indomtuned_c40-139.feather`.

THE COVERAGE TRAP THIS SCRIPT GUARDS AGAINST. The tuned emulator applies its stored in-domain
training cuts at inference and returns NOTHING outside them. The dumps are the WIDE population, so a
naive join leaves ~57% of rows unmatched; letting those default to R_blend=0 collapses <R_blend> from
0.159 to 0.059 and produces a spurious +28.9% m (job 15366950). Unmatched rows are therefore DROPPED,
not zero-filled, and the retained fraction is printed and checked. Every figure here is consequently
on the IN-DOMAIN population -- which is the population the fiducial model is defined for.

FIGURE 4 IS NEW -- THE OLD ONE WAS DELETED, NOT REGENERATED. V1's `fig4_bias_prob_neighbours`
applied a hardcoded `R_blend_prob = R_blend + 0.0017`, an offset inherited from a 2026-07-09
forward-model residual, because a faithful run was blocked. Regenerating it under a new fiducial
model would have relabelled a stale constant as a current result, so on the owner's instruction it is
gone. Fig4 is now the SELECTION result: how the response shifts under near-domain measured cuts, sim
vs model, plus the residual bias. Every value is read from the table written by
`scripts/eval_selection_constgold_neardomain.py --save-npz`; nothing is transcribed or hardcoded.

Sources
  fig1:   per-seed `*_train_curve.npz` next to each dom6x6 checkpoint (val_nll per epoch + SWA window)
  fig2-3: per-object dom6x6 constgold dumps in `derisk/v2_domain_dumps`, one per seed. R_sim is
          seed-independent and R_blend now comes from the tuned lookup, so the 16-seed spread IS
          the model uncertainty.
  fig4:   `results/constgold_neardomain_table.npz`, written by the near-domain selection job. It is a
          16-SEED run like fig1-3. The seed convention splits by which FLOW OUTPUT drives the number
          (e/shape -> 16, flux/size -> 4), and although fig4's CUTS are on flux and size, the
          quantity it reports is `m` -- a bias on the SHAPE response. So the e-response standard
          binds and 16 seeds are required. An earlier 4-seed version of this table carried +-0.43% on
          the m column, nearly 3x the 16-seed error and too coarse to test the +-0.3% target.

WHAT "NO CUT" MEANS IN FIG 4. The population is already restricted on TRUE properties to the
emulator's in-domain box (mag<26, 0.3<Re<1.5); "no cut" means no further MEASURED cut, not an
unselected catalogue. The two are not the same thing, and the rows behave differently for different
reasons: measured `mag<26` still removes 2.5% (measurement scatter across a boundary the true cut
already applied) and carries a real selection excess, whereas measured `R>0.30"` removes nothing --
not because of the true size cut but because the PSF floors measured flux_radius at R50=0.527", so
no object can land below 0.30" whatever its true size. Keep fractions are printed on the tick labels
precisely so a structurally-empty cut cannot be mistaken for a passing one.

SIGN CONVENTION IS UNIFORM: every m here is sim/model - 1, matching fig3 and AGENTS.md. The previous
fig4 plotted the residual as model/sim - 1 against a no-cut line at sim/model - 1, which mirrored the
two: the `R>0.30"` no-op row sat at +0.247% against a no-cut line at -0.245%, making an exact null
look like a 0.49-pt disagreement.

  m = R_sim / (R_flow^seed + R_blend) - 1, the same parameter-free estimator as V1/V2.

PNG only; Okabe-Ito (validated pair: CVD dE 21.9, normal-vision dE 31.2). Seeds are exchangeable
REPLICATES, not categories, so they share one hue; identity comes from axis/legend text, never colour.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.feather as pf  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from sbs_shear.blend_lookup import join_blend  # noqa: E402
from sbs_shear.paths import CROWD_LOOKUP as CROWD

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED = "#1a1a1a", "#6b6b6b"

ABL = Path("/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation")
DUMPDIR = Path("/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps")
BLEND_LOOKUP = "results/blend_lookup_indomtuned_c40-139.feather"
MIN_MATCH = 0.20     # in-domain lookup vs WIDE dumps: expect ~0.43; refuse far below
CONST_CAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
             "constant_response_catalogue_train.feather")
TABLE_NPZ = "results/constgold_neardomain_table.npz"
SELFRESP = "results/halfshear_selfresp.feather"
CURVE_GLOB = "measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s*_train_curve.npz"
TAG = "ablate_s2c_lt500_dom6x6"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12.5, "axes.labelsize": 11, "legend.fontsize": 9.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#999999", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    })


def save(fig, out_dir, stem):
    p = os.path.join(out_dir, stem + ".png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", p)
    return p


# --------------------------------------------------------------------------
# FIGURE 1 -- validation loss per seed
# --------------------------------------------------------------------------
def figure1(out_dir):
    curves = {}
    for p in sorted(glob.glob(str(ABL / CURVE_GLOB))):
        mo = re.search(r"_s(\d+)_train_curve\.npz$", p)
        if not mo:
            continue
        z = np.load(p, allow_pickle=True)
        if "val_nll" not in z.files:
            continue
        curves[int(mo.group(1))] = dict(val=z["val_nll"].astype(float),
                                        swa=z["swa_epochs"].astype(int)
                                        if "swa_epochs" in z.files else None)
    if not curves:
        print("[fid_fig1] no *_train_curve.npz found -> SKIPPED")
        return None
    seeds = sorted(curves)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    # seeds can stop at different epochs (early stopping) -> pad with NaN, never truncate
    nmax = max(len(c["val"]) for c in curves.values())
    stack = np.full((len(seeds), nmax), np.nan)
    for i, s in enumerate(seeds):
        v = curves[s]["val"]
        stack[i, :len(v)] = v
    ep = np.arange(1, nmax + 1)

    # SWA averaging window (what the released checkpoint actually averages over)
    swa = curves[seeds[0]]["swa"]
    if swa is not None and len(swa):
        ax.axvspan(int(swa.min()), int(swa.max()), color=GREEN, alpha=0.12, lw=0, zorder=0)

    for i in range(len(seeds)):           # replicates: one hue, no per-seed legend entries
        ax.plot(ep, stack[i], color=BLUE, lw=1.0, alpha=0.40, zorder=2)

    # Seeds early-stop at different epochs, so past the shortest run the "ensemble mean" is a mean
    # over a SHRINKING subset -- its meaning changes along x, and a survivor-only tail can drift
    # simply because the seeds that ran longer were the ones still improving. Draw it solid only
    # where every seed is alive, dotted where it is a partial mean, and read the plateau off the
    # all-alive region.
    alive = np.isfinite(stack).sum(axis=0)
    full = alive == len(seeds)
    mean = np.nanmean(stack, axis=0)
    ax.plot(np.where(full, ep, np.nan), np.where(full, mean, np.nan),
            color=BLUE, lw=2.4, zorder=4)
    if not full.all():
        ax.plot(np.where(~full, ep, np.nan), np.where(~full, mean, np.nan),
                color=BLUE, lw=2.4, ls=":", zorder=4)

    last_full = int(np.max(np.flatnonzero(full))) if full.any() else len(ep) - 1
    plateau = float(np.nanmedian(stack[:, max(0, last_full - 11):last_full + 1]))
    ax.axhline(plateau, color=MUTED, lw=1.0, ls=":", zorder=1)

    lo = float(np.nanmin(stack))
    hi = float(np.nanpercentile(stack, 88))
    pad = 0.10 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 3.2 * pad)          # headroom so the legend never sits on a curve
    ax.set_xlim(0, nmax + 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL")

    if swa is not None and len(swa):
        ax.annotate(f"SWA window\nepochs {swa.min()}-{swa.max()}",
                    xy=(float(np.mean(swa)), 0.04), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8.5, color="#0b6b52")
    ax.annotate(f"plateau {plateau:.3f}", xy=(0.03, plateau), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=8.5, color=MUTED)
    handles = [Line2D([], [], color=BLUE, lw=2.4, label=f"ensemble mean (all {len(seeds)} seeds)"),
               Line2D([], [], color=BLUE, lw=1.0, alpha=0.40,
                      label=f"individual seeds ({len(seeds)})")]
    if not full.all():
        handles.insert(1, Line2D([], [], color=BLUE, lw=2.4, ls=":",
                                 label="partial mean (some seeds stopped)"))
    ax.legend(handles=handles, loc="upper right")
    fig.text(0.5, -0.01, f"fiducial: {TAG}  |  seeds: " + ", ".join(f"s{s}" for s in seeds),
             ha="center", va="top", fontsize=8, color=MUTED)
    return save(fig, out_dir, "fid_fig1_seed_loss")


# --------------------------------------------------------------------------
# shared dump loading for FIGURE 2 / 3
# --------------------------------------------------------------------------
def dump_paths():
    out = {}
    for p in sorted(glob.glob(str(DUMPDIR / "ablate_s2c_lt500_dom6x6_perobj_s*.feather"))):
        mo = re.search(r"_s(\d+)\.feather$", p)
        if mo:
            out[int(mo.group(1))] = p
    return out


def load_dumps():
    """(seeds, ref_frame, R_flow[nseed, nobj]).

    Every seed's dump is produced by the same deterministic pass over the same catalogue, so the
    (case, input_index) rows are in identical order. That is CHECKED (length + key checksums), and
    once it holds only the R_flow column needs reading per seed -- a 27M-row reindex per seed is
    both unnecessary and too slow. R_sim / R_blend are seed-independent, read once.
    """
    paths = dump_paths()
    if not paths:
        return [], None, None
    seeds = sorted(paths)
    ref = pf.read_table(paths[seeds[0]],
                        columns=["case", "input_index", "r_sim", "R_blend"]).to_pandas()
    ck = (len(ref), int(ref["case"].sum()), int(ref["input_index"].sum()))
    flows = []
    for s in seeds:
        if s != seeds[0]:
            k = pf.read_table(paths[s], columns=["case", "input_index"]).to_pandas()
            if (len(k), int(k["case"].sum()), int(k["input_index"].sum())) != ck:
                raise RuntimeError(f"dump s{s} row order/content differs from s{seeds[0]}; "
                                   "cannot stack R_flow positionally")
            del k
        flows.append(pf.read_table(paths[s], columns=["R_flow"])
                     .column("R_flow").to_numpy().astype(np.float32))
    print(f"dumps aligned: {len(seeds)} seeds x {len(ref):,} rows")
    flows = np.vstack(flows)

    # ---- swap in the TUNED emulator's R_blend, and DROP rows it does not cover ----------------
    # The dumps carry the old emulator's R_blend; the fiducial model uses the tuned in-domain one.
    # Rows outside the tuned emulator's training box get no prediction. Zero-filling them is the
    # documented failure mode (spurious +28.9% m, job 15366950), so they are dropped instead and
    # both `ref` and the positional `flows` are masked together so they cannot drift apart.
    rb_old_mean = ref["R_blend"].mean()
    rb_new, keep, _frac, _j = join_blend(ref, BLEND_LOOKUP, "R_blend", min_match=MIN_MATCH,
                                         label="tuned emulator lookup")
    print(f"  <R_blend> old {rb_old_mean:.4f} -> new {np.nanmean(rb_new):.4f}")
    ref = ref.loc[keep].reset_index(drop=True)
    ref["R_blend"] = rb_new[keep]
    flows = flows[:, keep]
    print(f"in-domain rows retained: {len(ref):,}")
    return seeds, ref, flows


def _read_key_table(path, cols, cases):
    """Read `cols` for rows whose case is in `cases`, one row per (case, input_index)."""
    case_arr = pa.array(sorted(int(c) for c in cases))
    parts = []
    with ipc.open_file(pa.memory_map(str(path))) as r:
        avail = [c for c in cols if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(avail)
            b = b.filter(pc.is_in(b["case"], value_set=case_arr))
            if b.num_rows:
                parts.append(b.to_pandas())
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=avail)
    return d.groupby(["case", "input_index"], as_index=False).mean(numeric_only=True)


def _binned(x, ysim, ymods, nb=12, logx=False, qlo=1.0, qhi=99.0):
    """Bin truth and EACH seed's model curve on a common grid.

    Returns (centres, sim_mean, sim_sem, model_means[nbin, nseed]). The band shown is the
    STANDARD ERROR ON THE MEAN, not the per-object 16-84% scatter: per-object shear response has
    sigma ~ 0.3 against a mean of ~0.45 (intrinsic-shape noise), so a percentile band is a hundred
    times the difference being judged and would hide it completely.
    """
    good = np.isfinite(x) & np.isfinite(ysim)
    for ym in ymods:
        good &= np.isfinite(ym)
    x, ysim = x[good], ysim[good]
    ymods = [ym[good] for ym in ymods]
    lo, hi = np.percentile(x, [qlo, qhi])
    xx = np.log10(np.clip(x, 1e-12, None)) if logx else x
    lo_e = np.log10(max(lo, 1e-12)) if logx else lo
    hi_e = np.log10(max(hi, 1e-12)) if logx else hi
    edges = np.linspace(lo_e, hi_e, nb + 1)
    idx = np.digitize(xx, edges) - 1
    cx, s_m, s_e, m_m = [], [], [], []
    for b in range(nb):
        sel = idx == b
        n = int(sel.sum())
        if n < 30:
            continue
        cx.append(10 ** (0.5 * (edges[b] + edges[b + 1])) if logx
                  else 0.5 * (edges[b] + edges[b + 1]))
        s_m.append(np.mean(ysim[sel]))
        s_e.append(np.std(ysim[sel]) / np.sqrt(n))
        m_m.append([np.mean(ym[sel]) for ym in ymods])
    return np.array(cx), np.array(s_m), np.array(s_e), np.array(m_m)


# --------------------------------------------------------------------------
# FIGURE 2 -- model vs true response across galaxy properties
# --------------------------------------------------------------------------
def figure2(loaded, out_dir):
    seeds, ref, flows = loaded
    if not seeds:
        print("[fid_fig2] no per-object dumps yet -> SKIPPED")
        return None
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
    rb = df["R_blend"].to_numpy(float)
    ymods = [flows[i].astype(float) + rb for i in range(len(seeds))]
    R_sim_glob = float(np.nanmean(ysim))

    panels = [("S/N_plus", r"primary flux  (S/N$_+$)", True, None),
              ("Re_input_p", r"primary size  $R_e$  [arcsec]", False, None),
              ("nbr_flux_near", "neighbour / blend flux  (near shell)", True, 1e-3)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.4, 1]))
    all_res = []
    for j, (col, xlabel, logx, xmin) in enumerate(panels):
        top, bot = axes[0, j], axes[1, j]
        x = df[col].to_numpy(float)
        ys, yms = ysim, ymods
        if xmin is not None:
            keep = np.isfinite(x) & (x > xmin)
            x, ys, yms = x[keep], ysim[keep], [ym[keep] for ym in ymods]
        cx, sm, se, mm = _binned(x, ys, yms, nb=12, logx=logx)
        mmean = mm.mean(1)

        top.axhline(R_sim_glob, color="#cccccc", lw=1.0, zorder=0)
        top.fill_between(cx, sm - se, sm + se, color=BLUE, alpha=0.30, lw=0, zorder=1)
        top.plot(cx, sm, "-o", color=BLUE, ms=7, lw=1.8, zorder=4)
        top.fill_between(cx, mm.min(1), mm.max(1), color=VERM, alpha=0.25, lw=0, zorder=2)
        top.plot(cx, mmean, "--s", color=VERM, ms=7, lw=1.8, mfc="white", mew=1.6, zorder=3)

        # fractional residual: where does the model over/under-predict the truth?
        res = (mmean / sm - 1.0) * 100.0
        rlo = (mm.min(1) / sm - 1.0) * 100.0
        rhi = (mm.max(1) / sm - 1.0) * 100.0
        rse = np.abs(se / sm) * 100.0
        all_res.append(res)
        bot.axhline(0.0, color="#cccccc", lw=1.0, zorder=0)
        bot.fill_between(cx, -rse, rse, color=BLUE, alpha=0.25, lw=0, zorder=1)
        bot.fill_between(cx, rlo, rhi, color=VERM, alpha=0.25, lw=0, zorder=2)
        bot.plot(cx, res, "--s", color=VERM, ms=6, lw=1.6, mfc="white", mew=1.4, zorder=3)
        bot.set_xlabel(xlabel)
        if logx:
            top.set_xscale("log")
            bot.set_xscale("log")
            # <1 decade of range makes log MINOR labels collide; keep majors only
            bot.xaxis.set_minor_formatter(NullFormatter())
            bot.xaxis.set_major_formatter(ScalarFormatter())
            bot.ticklabel_format(axis="x", style="plain")
        if j:
            top.tick_params(labelleft=False)
            bot.tick_params(labelleft=False)

    ymin = min(a.get_ylim()[0] for a in axes[0])
    ymax = max(a.get_ylim()[1] for a in axes[0])
    # residual limits: robust, so the few bins where the TRUTH response is ~0 (smallest galaxies,
    # below the size domain) cannot blow the ratio up and squash the informative range. Points
    # outside are clipped, and the count is reported.
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
               label=r"fiducial  $R_{\rm flow}+R_{\rm blend}$"),
        Patch(fc=VERM, alpha=0.25, label=f"{len(seeds)}-seed range")], loc="best")
    axes[0, 0].annotate(f"constgold, true neighbours\nN={len(df):,} objects, cases "
                        f"{int(df['case'].min())}-{int(df['case'].max())}",
                        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color=MUTED,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.tight_layout()
    return save(fig, out_dir, "fid_fig2_response_vs_properties")


# --------------------------------------------------------------------------
# FIGURE 3 -- per-seed multiplicative bias
# --------------------------------------------------------------------------
def _bias_figure(loaded, out_dir, stem):
    seeds, ref, flows = loaded
    if not seeds:
        print("[fid_fig3] no per-object dumps yet -> SKIPPED")
        return None, None
    R_sim = float(np.mean(ref["r_sim"].to_numpy(float)))
    R_blend = float(np.mean(ref["R_blend"].to_numpy(float)))
    R_flow = np.array([float(np.mean(flows[i].astype(float))) for i in range(len(seeds))])
    m = (R_sim / (R_flow + R_blend) - 1.0) * 100.0
    m_mean = float(np.mean(m))
    m_std = float(np.std(m, ddof=1)) if len(m) > 1 else float("nan")
    m_sem = m_std / np.sqrt(len(m)) if len(m) > 1 else float("nan")
    x = np.arange(len(seeds))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.axhspan(-1.0, 1.0, color=GREEN, alpha=0.10, lw=0, zorder=0)
    ax.axhspan(m_mean - m_sem, m_mean + m_sem, color="0.45", alpha=0.25, zorder=1)
    ax.axhline(m_mean, color=INK, lw=1.6, zorder=2)
    ax.axhline(0.0, color=MUTED, lw=1.1, ls="--", zorder=2)
    ax.plot(x, m, "o", ms=10, color=BLUE, mec="white", mew=1.0, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_xlim(-0.6, len(seeds) - 0.4)
    ax.set_xlabel("flow seed")
    ax.set_ylabel(r"multiplicative bias  $m$  [%]")
    ax.annotate("Fiducial (dom6x6 + tuned emulator), constgold" "\n"
                rf"$R_{{\rm sim}}={R_sim:.4f}$,  $R_{{\rm blend}}={R_blend:.4f}$" "\n"
                r"$m = R_{\rm sim}/(R_{\rm flow}^{\rm seed}+R_{\rm blend})-1$" "\n"
                rf"ensemble $m = {m_mean:+.2f} \pm {m_sem:.2f}\%$  (std {m_std:.2f}%, N={len(m)})",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top", ha="left", fontsize=9.5,
                color=INK)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ms=10, ls="none", label="per-seed $m$"),
        Line2D([], [], color=INK, lw=1.6, label="ensemble mean"),
        Patch(fc="0.45", alpha=0.35, label=r"mean $\pm$ std/$\sqrt{N}$"),
        Patch(fc=GREEN, alpha=0.18, label=r"$\pm1\%$ band"),
        Line2D([], [], color=MUTED, lw=1.1, ls="--", label=r"$m=0$")],
        loc="lower right", ncol=2)
    # Headroom so the 4-line annotation block never lands on a data point (it did on s501).
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.42 * (yhi - ylo))
    p = save(fig, out_dir, stem)

    print(f"\n  R_sim={R_sim:.4f}  R_blend={R_blend:.4f}")
    for s, rf_, mi in zip(seeds, R_flow, m):
        print(f"    s{s}  R_flow={rf_:.4f}  R_total={rf_+R_blend:.4f}  m={mi:+.3f}%")
    print(f"  ENSEMBLE m = {m_mean:+.3f} +- {m_sem:.3f}%  (std {m_std:.3f}%, N={len(m)})")
    return p, (m_mean, m_sem, len(m))


def figure3(loaded, out_dir):
    return _bias_figure(loaded, out_dir, "fid_fig3_bias_true_neighbours")


def _robust_xlim(ax, vals, y, floor=1.0):
    """Scale to the BULK and label whatever falls outside, rather than to the worst row.

    One known-bad cut (R>0.70") is several times the size of every other row, so scaling to it
    squashes the sub-1% rows -- the ones the +-0.3% target is about -- onto the zero line and makes
    the target band invisible. Out-of-range points are drawn at the edge as arrows WITH their value
    printed, so nothing is hidden: the reader sees both that the row is off-scale and by how much.
    """
    v = np.asarray(vals, float)
    fin = v[np.isfinite(v)]
    if not len(fin):
        return
    lim = max(floor, float(np.percentile(np.abs(fin), 85)) * 1.35)
    ax.set_xlim(-lim, lim)
    for vi, yi in zip(v, y):
        if np.isfinite(vi) and abs(vi) > lim:
            s = np.sign(vi)
            ax.plot(s * lim * 0.965, yi, marker=">" if s > 0 else "<", ms=9,
                    color=INK, clip_on=False, zorder=6)
            ax.annotate(f"{vi:+.2f}%", xy=(s * lim * 0.93, yi), ha="right" if s > 0 else "left",
                        va="center", fontsize=8, color=INK,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=7)


def figure4(out_dir, shape16=None):
    """Selection bias under near-domain measured cuts -- sim vs the fiducial model.

    `shape16` is (m, sem, nseed) computed by figure 3 from the 16-seed dumps in THIS run, passed in
    so the caption's shape reference is a live number rather than a transcribed one. It is the same
    population as this table (both land on the emulator's in-domain box), so the two are directly
    comparable: the table's no-cut m is the 4-seed selection-run counterpart of it.

    Replaces V1's `fig4_bias_prob_neighbours`, which was DELETED rather than regenerated: it applied
    a hardcoded R_blend + 0.0017 inherited from a 2026-07-09 forward-model residual, so remaking it
    under a new fiducial model would have relabelled a stale constant as a current result.

    Everything here is read from the table `eval_selection_constgold_neardomain.py` writes
    (`--save-npz`). No number is transcribed or hardcoded.
    """
    if not os.path.exists(TABLE_NPZ):
        print(f"[fid_fig4] {TABLE_NPZ} not found -> SKIPPED "
              "(run jobs/job_constgold_neardomain.sh first)")
        return None
    z = np.load(TABLE_NPZ, allow_pickle=True)
    if "dm" not in z.files:
        raise SystemExit(
            f"{TABLE_NPZ} predates the sign/error fix (no `dm`). Its `m_flow` column was "
            "model/sim-1 while the no-cut line was sim/model-1, so the two were mirrored, and its "
            "errors came from ensemble means rather than per-seed ratios. Re-run "
            "jobs/job_constgold_neardomain.sh.")
    name = [str(x) for x in z["name"]]
    keep = z["keep"]
    # Column (1) AS DEFINED (denominator = sheared-intrinsic R ~ 1.00), not the /R_meas variant.
    # It is tempting to renormalise it by R_meas (~0.86) so it shares (3)'s denominator, but that is
    # only half the conversion: (1) averages INTRINSIC shapes while (3) averages MEASURED ones, and
    # measured shapes are diluted relative to intrinsic. Putting (1) on (3)'s scale needs that
    # dilution factor as well, and the two corrections work in opposite directions and largely
    # cancel. The factor has not been measured, so neither normalisation can be asserted as correct
    # and the defined one is kept. `pure_sel_meas` is saved alongside as a diagnostic. Read the
    # green ticks as "how much of this shift is the moving boundary", qualitatively -- they are not
    # an exact term-by-term decomposition of (3).
    c1, c3, c4 = z["pure_sel"], z["measured"], z["model_m"]
    e4 = z["model_sem"]
    mcut, emcut, dm, edm, prox = z["m_cut"], z["m_cut_err"], z["dm"], z["dm_err"], z["is_proxy"]
    m_nc, e_nc = float(z["m_nocut"]), float(z["m_nocut_err"])
    y = np.arange(len(name))[::-1]          # first row at the top
    # keep fraction on the tick label: without it a reader cannot tell that the sub-PSF size rows
    # are structurally empty cuts (measured flux_radius is floored at R50=0.527").
    ylab = [f"{n}   ({100*k:.1f}%)" for n, k in zip(name, keep)]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 0.46 * len(name) + 3.0),
                             gridspec_kw={"width_ratios": [1.32, 1.0]})

    # -- left: the shift itself, sim vs model ------------------------------------------------
    ax = axes[0]
    ax.axvline(0.0, color=MUTED, lw=1.1, ls="--", zorder=1)
    ax.errorbar(c4, y + 0.16, xerr=e4, fmt="s", ms=7, color=VERM, mfc="white", mew=1.6,
                lw=1.4, capsize=3, zorder=4, label=r"model  $R_{\rm flow}+R_{\rm blend}$")
    ax.plot(c3, y - 0.16, "o", ms=8, color=BLUE, mec="white", mew=1.0, zorder=4,
            label="sim (measured shapes)")
    ax.plot(c1, y, "|", ms=11, color=GREEN, mew=2.0, zorder=3, label="pure selection")
    ax.set_yticks(y)
    ax.set_yticklabels(ylab, fontsize=9)
    ax.set_ylim(-0.8, len(name) - 0.2)
    ax.set_xlabel(r"shift in response under the cut  [%]")
    ax.legend(loc="lower right", fontsize=9)

    # -- middle: residual bias AT the cut, project sign convention (sim/model - 1) -------------
    ax = axes[1]
    ax.axvspan(-0.3, 0.3, color=GREEN, alpha=0.12, lw=0, zorder=0)
    ax.axvline(0.0, color=MUTED, lw=1.1, ls="--", zorder=1)
    ax.axvline(m_nc, color=INK, lw=1.3, ls=":", zorder=2)
    ax.errorbar(mcut, y, xerr=emcut, fmt="D", ms=6.5, color=INK, mfc="white", mew=1.5,
                lw=1.3, capsize=3, zorder=4)
    for i, p_ in enumerate(prox):           # proxy rows are not a like-for-like cut
        if p_:
            ax.plot(mcut[i], y[i], "*", ms=13, color=VERM, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.8, len(name) - 0.2)
    _robust_xlim(ax, mcut, y)
    ax.set_xlabel(r"residual bias at the cut  $m$  [%]")
    ax.legend(handles=[
        Line2D([], [], color=INK, marker="D", ms=6.5, mfc="white", mew=1.5, ls="none",
               label=r"$R_{\rm sim}/R_{\rm model}-1$"),
        Line2D([], [], color=INK, lw=1.3, ls=":", label=f"no cut ({m_nc:+.2f}%)"),
        Patch(fc=GREEN, alpha=0.18, label=r"$\pm0.3\%$ target"),
        Line2D([], [], color=VERM, marker="*", ms=13, ls="none", label="model cuts a proxy")],
        loc="lower left", fontsize=8.5)

    # The selection-induced excess `dm = m(cut) - m(no cut)` is still computed and stored in the npz
    # (columns `dm` / `dm_err`) and printed by the table script, but is no longer plotted: owner's
    # call, 2026-07-31. It was added when this table ran at 4 seeds, where the absolute `m` column
    # was too noisy (+-0.43%) to test against the +-0.3% target and only the seed-cancelling
    # difference was readable. At 16 seeds the absolute `m` is itself precise enough, so the extra
    # panel no longer earns its space.

    # Caption numbers are read from the npz or passed in from figure 3's own 16-seed computation --
    # never transcribed. A stale pasted constant in a caption is the failure mode that got V1's
    # fig4 deleted.
    # The table path and the per-object dumps compute the same no-cut m by different routes
    # (accumulators over the catalogue vs a per-object dump). When they agree, say so once as a
    # CROSS-CHECK -- printing the identical number twice reads as a copy-paste error and throws away
    # the fact that two independent paths landed on it. When they disagree, print both, because then
    # the disagreement is the interesting thing.
    shape_txt = ""
    if shape16 is not None:
        agree = (abs(shape16[0] - m_nc) < 0.002) and (abs(shape16[1] - e_nc) < 0.002)
        shape_txt = ("  (table and " + str(shape16[2]) + "-seed per-object dumps agree)" if agree
                     else f"; {shape16[2]}-seed dumps give {shape16[0]:+.3f} $\\pm$ {shape16[1]:.3f}%")
    fig.text(0.5, -0.02,
             f"constgold, in-domain (true mag<{float(z['dom_mag_max']):g}, "
             f"$R_e$>{float(z['dom_re_min']):g}), N={int(z['n_rows']):,} -- cuts are on MEASURED "
             f"quantities  |  {int(z['n_seeds'])} flow seeds  |  "
             f"$R_{{\\rm sim}}$={float(z['R_sim_meas']):.4f},  "
             f"$R_{{\\rm flow}}$={float(z['R_flow']):.4f} + "
             f"$R_{{\\rm blend}}$={float(z['R_blend']):.4f}  |  "
             f"no-cut $m$ = {m_nc:+.3f} $\\pm$ {e_nc:.3f}%{shape_txt}",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return save(fig, out_dir, "fid_fig4_selection_near_domain")


def figure5(out_dir):
    """Figure 2's question restricted to the SELF response: flow alone vs half-shear R_self.

    Figure 2 plots `R_flow + R_blend` against the constgold total response, so a flow error and an
    emulator error can trade off against each other and still land on the truth. This figure removes
    that freedom: the model curve is the FLOW ONLY, and the truth is the half-shear self-response,
    isolated by projecting on the primary's own shear direction. Same three axes and same binning as
    figure 2, so the two are read the same way.

    Data: `results/halfshear_selfresp.feather` from `scripts/dump_halfshear_selfresp.py` (forward
    extraction on both sides). Nothing here is hardcoded.
    """
    if not os.path.exists(SELFRESP):
        print(f"[fid_fig5] {SELFRESP} not found -> SKIPPED "
              "(run jobs/job_halfshear_selfresp.sh first)")
        return None
    df = pf.read_table(SELFRESP, memory_map=True).to_pandas()
    scols = sorted([c for c in df.columns if c.startswith("R_flow_s")])
    ysim = df["r_sim_self"].to_numpy(float)
    ymods = [df[c].to_numpy(float) for c in scols]
    R_glob = float(np.nanmean(ysim))
    print(f"[fid_fig5] {len(df):,} rows x {len(scols)} seeds   "
          f"<R_self>sim={R_glob:+.4f}  <R_flow>={np.nanmean([np.nanmean(y) for y in ymods]):+.4f}")

    panels = [("SN", r"primary flux  (S/N)", True, None),
              ("Re_input_p", r"primary size  $R_e$  [arcsec]", False, None),
              ("nbr_flux_near", "neighbour / blend flux  (near shell)", True, 1e-3)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.4, 1]))
    all_res = []
    for j, (col, xlabel, logx, xmin) in enumerate(panels):
        top, bot = axes[0, j], axes[1, j]
        x = df[col].to_numpy(float)
        ys, yms = ysim, ymods
        if xmin is not None:
            keep = np.isfinite(x) & (x > xmin)
            x, ys, yms = x[keep], ysim[keep], [ym[keep] for ym in ymods]
        cx, sm, se, mm = _binned(x, ys, yms, nb=12, logx=logx)
        mmean = mm.mean(1)
        top.axhline(R_glob, color="#cccccc", lw=1.0, zorder=0)
        top.fill_between(cx, sm - se, sm + se, color=BLUE, alpha=0.30, lw=0, zorder=1)
        top.plot(cx, sm, "-o", color=BLUE, ms=7, lw=1.8, zorder=4)
        top.fill_between(cx, mm.min(1), mm.max(1), color=VERM, alpha=0.25, lw=0, zorder=2)
        top.plot(cx, mmean, "--s", color=VERM, ms=7, lw=1.8, mfc="white", mew=1.6, zorder=3)
        res = (mmean / sm - 1.0) * 100.0
        rlo = (mm.min(1) / sm - 1.0) * 100.0
        rhi = (mm.max(1) / sm - 1.0) * 100.0
        rse = np.abs(se / sm) * 100.0
        all_res.append(res)
        bot.axhline(0.0, color="#cccccc", lw=1.0, zorder=0)
        bot.fill_between(cx, -rse, rse, color=BLUE, alpha=0.25, lw=0, zorder=1)
        bot.fill_between(cx, rlo, rhi, color=VERM, alpha=0.25, lw=0, zorder=2)
        bot.plot(cx, res, "--s", color=VERM, ms=6, lw=1.6, mfc="white", mew=1.4, zorder=3)
        bot.set_xlabel(xlabel)
        if logx:
            top.set_xscale("log"); bot.set_xscale("log")
            bot.xaxis.set_minor_formatter(NullFormatter())
            bot.xaxis.set_major_formatter(ScalarFormatter())
            bot.ticklabel_format(axis="x", style="plain")
        if j:
            top.tick_params(labelleft=False); bot.tick_params(labelleft=False)
    ymin = min(a.get_ylim()[0] for a in axes[0])
    ymax = max(a.get_ylim()[1] for a in axes[0])
    flat = np.concatenate(all_res); flat = flat[np.isfinite(flat)]
    lim = max(8.0, float(np.nanpercentile(np.abs(flat), 90)) * 1.6)
    n_clip = int(np.sum(np.abs(flat) > lim))
    for a in axes[0]:
        a.set_ylim(ymin, ymax)
    for a in axes[1]:
        a.set_ylim(-lim, lim)
    axes[0, 0].set_ylabel(r"mean SELF response  $R_{\rm self}$")
    axes[1, 0].set_ylabel(r"model / truth $-$ 1  [%]")
    axes[0, 0].legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ms=7, lw=1.8, label=r"half-shear truth  $R_{\rm self}$"),
        Patch(fc=BLUE, alpha=0.30, label="truth s.e. on mean"),
        Line2D([], [], color=VERM, marker="s", ms=7, lw=1.8, ls="--", mfc="white",
               label=r"flow ONLY  $R_{\rm flow}$  (no $R_{\rm blend}$)"),
        Patch(fc=VERM, alpha=0.25, label=f"{len(scols)}-seed range")], loc="best")
    axes[0, 0].annotate("half-shear legs, self-response isolated on $\\hat{g}_p$\n"
                        f"N={len(df):,} objects" + (f"\n{n_clip} residual bins clipped"
                                                    if n_clip else ""),
                        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color=MUTED,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.tight_layout()
    return save(fig, out_dir, "fid_fig5_selfresp_halfshear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output directory (default: <repo>/figures)")
    ap.add_argument("--only", default=None, choices=["1", "2", "3", "4", "5"])
    args = ap.parse_args()
    set_style()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(here, "figures")
    os.makedirs(out_dir, exist_ok=True)

    if args.only in (None, "1"):
        figure1(out_dir)
    shape16 = None
    if args.only in (None, "2", "3"):
        loaded = load_dumps()
        print(f"per-object dumps found: {loaded[0]}")
        if args.only in (None, "2"):
            figure2(loaded, out_dir)
        if args.only in (None, "3"):
            _, shape16 = figure3(loaded, out_dir)
    if args.only in (None, "4"):
        # fig4/fig5 read saved tables, not the dumps, so they need no `loaded`. `shape16` is the
        # 16-seed m figure 3 just computed; with `--only 4` it is None and the caption omits it
        # rather than falling back to a transcribed value.
        figure4(out_dir, shape16=shape16)
    if args.only in (None, "5"):
        figure5(out_dir)
    print("PLOT_FID_FLOW_FIGURES_DONE")


if __name__ == "__main__":
    main()
