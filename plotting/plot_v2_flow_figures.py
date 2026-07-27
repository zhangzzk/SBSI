#!/usr/bin/env python
"""Gold-V2 (coupling-pinned lt500) 8-seed remakes of figures 1-3.

  figures/figv2_fig1_seed_loss.png             -- validation NLL vs epoch, all 8 seeds
  figures/figv2_fig2_response_vs_properties.png -- model vs true response across primary flux,
                                                   primary size and neighbour/blend flux
  figures/figv2_fig3_bias_true_neighbours.png  -- per-seed multiplicative bias m

These are the V2 counterparts of the V1 fig1-3 (`plot_flow_figures.py`, tag
meas_szfl_noz_lam450_fixresp). They are written under NEW names so the V1 originals survive.

Sources
  fig1: the per-seed `*_train_curve.npz` written next to each checkpoint (val_nll per epoch,
        plus the SWA averaging window) -- no log scraping needed.
  fig2/3: the per-object constgold dumps from `jobs/job_v2_constgold_8seed.sh`
        (case, input_index, r_sim, R_flow, R_blend, ...), one per seed. R_sim and R_blend are
        seed-independent; only R_flow varies, so the 8-seed spread IS the model uncertainty.

  m = R_sim / (R_flow^seed + R_blend) - 1, the same parameter-free estimator as V1.

Login node; PNG only; Okabe-Ito (validated pair: CVD dE 21.9, normal-vision dE 31.2).
Seeds are exchangeable REPLICATES, not categories, so they share one hue rather than being
assigned 8 categorical colours; identity comes from the axis/legend text, never colour alone.
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

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED = "#1a1a1a", "#6b6b6b"

ABL = Path("/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation")
DUMPDIR = Path("/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_constgold_dumps")
CONST_CAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
             "constant_response_catalogue_train.feather")
CROWD = "/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather"
CURVE_GLOB = "measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s*_train_curve.npz"
TAG = "ablate_s2c_coupling_lt500"


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
        print("[figv2_fig1] no *_train_curve.npz found -> SKIPPED")
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
    ax.plot(ep, np.nanmean(stack, axis=0), color=BLUE, lw=2.4, zorder=4)

    plateau = float(np.nanmedian(stack[:, -12:]))
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
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=2.4, label="ensemble mean"),
                       Line2D([], [], color=BLUE, lw=1.0, alpha=0.40,
                              label=f"individual seeds ({len(seeds)})")],
              loc="upper right")
    fig.text(0.5, -0.01, f"Gold-V2 {TAG}  |  seeds: " + ", ".join(f"s{s}" for s in seeds),
             ha="center", va="top", fontsize=8, color=MUTED)
    return save(fig, out_dir, "figv2_fig1_seed_loss")


# --------------------------------------------------------------------------
# shared dump loading for FIGURE 2 / 3
# --------------------------------------------------------------------------
def dump_paths():
    out = {}
    for p in sorted(glob.glob(str(DUMPDIR / "v2_perobj_s*.feather"))):
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
    return seeds, ref, np.vstack(flows)


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
        print("[figv2_fig2] no per-object dumps yet -> SKIPPED")
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
               label=r"Gold-V2  $R_{\rm flow}+R_{\rm blend}$"),
        Patch(fc=VERM, alpha=0.25, label=f"{len(seeds)}-seed range")], loc="best")
    axes[0, 0].annotate(f"constgold, true neighbours\nN={len(df):,} objects, cases "
                        f"{int(df['case'].min())}-{int(df['case'].max())}",
                        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color=MUTED)
    fig.tight_layout()
    return save(fig, out_dir, "figv2_fig2_response_vs_properties")


# --------------------------------------------------------------------------
# FIGURE 3 -- per-seed multiplicative bias
# --------------------------------------------------------------------------
def figure3(loaded, out_dir):
    seeds, ref, flows = loaded
    if not seeds:
        print("[figv2_fig3] no per-object dumps yet -> SKIPPED")
        return None
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
    ax.annotate("Gold-V2, constgold, true neighbours" "\n"
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
    p = save(fig, out_dir, "figv2_fig3_bias_true_neighbours")

    print(f"\n  R_sim={R_sim:.4f}  R_blend={R_blend:.4f}")
    for s, rf_, mi in zip(seeds, R_flow, m):
        print(f"    s{s}  R_flow={rf_:.4f}  R_total={rf_+R_blend:.4f}  m={mi:+.3f}%")
    print(f"  ENSEMBLE m = {m_mean:+.3f} +- {m_sem:.3f}%  (std {m_std:.3f}%, N={len(m)})")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output directory (default: <repo>/figures)")
    ap.add_argument("--only", default=None, choices=["1", "2", "3"])
    args = ap.parse_args()
    set_style()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(here, "figures")
    os.makedirs(out_dir, exist_ok=True)

    if args.only in (None, "1"):
        figure1(out_dir)
    if args.only in (None, "2", "3"):
        loaded = load_dumps()
        print(f"per-object dumps found: {loaded[0]}")
        if args.only in (None, "2"):
            figure2(loaded, out_dir)
        if args.only in (None, "3"):
            figure3(loaded, out_dir)
    print("PLOT_V2_FLOW_FIGURES_DONE")


if __name__ == "__main__":
    main()
