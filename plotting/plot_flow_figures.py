#!/usr/bin/env python
"""Four standalone publication PNGs for the SBSI shear-calibration flow result.

Emits (each its OWN PNG, never a mosaic):
  figures/fig1_seed_loss.png            -- validation NLL vs epoch, one line/seed
  figures/fig2_response_vs_properties.png -- flow-predicted vs true response across
                                            primary flux / primary size / blend flux
  figures/fig3_bias_true_neighbours.png -- per-seed multiplicative bias m (true nbrs)
  figures/fig4_bias_prob_neighbours.png -- per-seed m with probabilistic (forward-
                                            modelled) neighbours

Pure log/feather parsing -> run on the login node (no sbatch).  Re-running picks
up new fixresp seeds automatically: fig1 rescans pilot_train_*.out, fig3/4 rescan
the single-seed CRN pilot_harvest_*.out logs.

Environment:
    eval "$(conda shell.bash hook)"; conda activate sims1
    export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"

HARD RULES honoured: PNG only (no PDF), no gridlines, Agg backend, Okabe-Ito palette.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
LOG_DIR = Path("/home/z/Zekang.Zhang/logs")
SBSI = Path("/home/z/Zekang.Zhang/SBSI")
OUT_DIR = SBSI / "figures"
RES_DIR = SBSI / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAG = "meas_szfl_noz_lam450_fixresp"   # current fixresp convention (seeds 501+)
MIN_SEED = 501

# Seed-independent constants (fixed convention, WORKLOG cont.47)
R_SIM = 0.4534            # simulation truth response (fixed constant catalogue, min-case 40)
R_BLEND_TRUE = 0.1593    # BlendEMU true-neighbour blend response (seed-independent)
G_SHEAR = 0.02
R_FLOW_M0 = 0.2941       # R_flow that gives m=0  (= R_SIM - R_BLEND_TRUE)

# --- Figure-4 probabilistic-neighbour R_blend --------------------------------
# The probabilistic construction draws each detection's neighbours from the
# population prior (probblend_forward.py Level B, "Poisson population draw, no
# true neighbour positions") and emulates R_blend on the drawn field.  The
# forward model reconstructs the truth-fed (true-neighbour) blend response to a
# residual multiplicative bias dm = -0.37% (PROB_BLENDING.md / WORKLOG cont.13,
# OLD 2026-07-09 convention, tag lsst_r_extnbr_ho).  That residual corresponds
# to the forward model slightly OVER-estimating the undetected-neighbour term,
# i.e. an additive dR_blend ~ +0.0017 relative to true neighbours.  We apply
# that documented offset to the current true-neighbour value:
#   R_blend_prob = R_BLEND_TRUE + 0.0017 = 0.1610
# CAVEAT: the forward-model residual is OLD-convention and was NOT re-derived on
# the current constant catalogue -> see report / figure annotation.
PROBBLEND_DM = -0.0037           # forward-model residual multiplicative bias
R_BLEND_PROB = round(R_BLEND_TRUE + 0.0017, 4)   # = 0.1610

# Okabe-Ito colorblind-safe palette (from plot_flow_calibration.py), cycled for
# an arbitrary number of seeds.
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#8C564B",  # brown
    "#117733",  # dark green
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
    "#661100",  # dark red
    "#6699CC",  # slate
]


def seed_color(seed: int, seeds: list[int]) -> str:
    return _PALETTE[seeds.index(seed) % len(_PALETTE)]


# --------------------------------------------------------------------------
# Data source A -- training logs (fig 1)
# --------------------------------------------------------------------------
_HDR_TRAIN = re.compile(r"###\s*PILOT_TRAIN\s+job=(\d+).*?TAG=(\S+)")
_TRAIN_SEED_MARK = re.compile(r"=====\s*TRAIN\s+seed=(\d+)\s*->")
_EPOCH_RE = re.compile(
    r"epoch\s+(\d+):\s*nll=([-\d.eE+]+)/([-\d.eE+]+)\s+<R_model>\(val\)=([-+][\d.eE]+)"
)


def parse_training_logs() -> dict:
    """{seed: {epoch, val_nll, train_nll, R_model, jobid}} for fixresp seeds >= 501.

    Auto-discovers every pilot_train_*.out whose header TAG matches; splits each
    file into per-seed blocks on the `===== TRAIN seed=N ->` markers so still-
    running seeds are included up to their latest epoch.  On duplicate seeds the
    most recently modified log wins.
    """
    out: dict[int, dict] = {}
    best_mtime: dict[int, float] = {}
    for path in sorted(glob.glob(str(LOG_DIR / "pilot_train_*.out"))):
        txt = Path(path).read_text(errors="replace")
        hdr = _HDR_TRAIN.search(txt)
        if not hdr or hdr.group(2) != TAG:
            continue
        jobid = hdr.group(1)
        mtime = os.path.getmtime(path)
        marks = list(_TRAIN_SEED_MARK.finditer(txt))
        for i, mk in enumerate(marks):
            seed = int(mk.group(1))
            if seed < MIN_SEED:
                continue
            end = marks[i + 1].start() if i + 1 < len(marks) else len(txt)
            block = txt[mk.end():end]
            ep, tr, vl, rm = [], [], [], []
            for m in _EPOCH_RE.finditer(block):
                ep.append(int(m.group(1))); tr.append(float(m.group(2)))
                vl.append(float(m.group(3))); rm.append(float(m.group(4)))
            if not ep:
                continue
            if seed in best_mtime and best_mtime[seed] >= mtime:
                continue
            best_mtime[seed] = mtime
            out[seed] = dict(epoch=np.asarray(ep, float),
                             train_nll=np.asarray(tr, float),
                             val_nll=np.asarray(vl, float),
                             R_model=np.asarray(rm, float), jobid=jobid)
    return out


# --------------------------------------------------------------------------
# Data source B -- harvest logs (fig 3 / 4)
# --------------------------------------------------------------------------
_HDR_HARV = re.compile(r"###\s*PILOT_HARVEST\s+job=(\d+)\s+seeds='([^']*)'\s+TAG=(\S+)")
_GLOBAL_RE = re.compile(
    r"\[s(\d+)\]\s*GLOBAL:\s*R_sim=([-\d.]+)\s+R_flow\(self\)=([-\d.]+)\s+"
    r"R_blend\(emulator\)=([-\d.]+)\s+R_total=([-\d.]+)"
)


def parse_harvest_seeds() -> dict:
    """{seed: {R_flow, R_sim_log, jobid}} from the definitive single-seed CRN
    harvests.  Auto-discovers pilot_harvest_*.out with TAG match, keeping only
    single-seed jobs in the current convention (R_sim ~= 0.4534) so the buggy
    pre-CRN serial runs and OLD-convention (R_sim=0.4655) logs are excluded.
    Latest log per seed (by mtime) wins -> picks up seeds 509-516 as they land.
    """
    out: dict[int, dict] = {}
    best_mtime: dict[int, float] = {}
    for path in sorted(glob.glob(str(LOG_DIR / "pilot_harvest_*.out"))):
        txt = Path(path).read_text(errors="replace")
        hdr = _HDR_HARV.search(txt)
        if not hdr or hdr.group(3) != TAG:
            continue
        seed_toks = hdr.group(2).split()
        if len(seed_toks) != 1:          # only single-seed solo CRN harvests
            continue
        mtime = os.path.getmtime(path)
        for m in _GLOBAL_RE.finditer(txt):
            seed = int(m.group(1))
            if seed < MIN_SEED:
                continue
            r_sim_log = float(m.group(2))
            if abs(r_sim_log - R_SIM) > 0.01:   # current convention only
                continue
            if seed in best_mtime and best_mtime[seed] >= mtime:
                continue
            best_mtime[seed] = mtime
            out[seed] = dict(R_flow=float(m.group(3)), R_sim_log=r_sim_log,
                             jobid=hdr.group(1))
    return out


def m_of(rflow: float, rblend: float) -> float:
    """Per-seed multiplicative bias in percent."""
    return (R_SIM / (rflow + rblend) - 1.0) * 100.0


# --------------------------------------------------------------------------
# Style helpers
# --------------------------------------------------------------------------
def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,          # HARD RULE: no gridlines anywhere
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.7,
    })


def save_png(fig, stem: str) -> str:
    p = OUT_DIR / f"{stem}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")   # PNG only
    plt.close(fig)
    return str(p)


# --------------------------------------------------------------------------
# FIGURE 1 -- validation NLL vs epoch, one line per flow seed
# --------------------------------------------------------------------------
def figure1(train: dict) -> str:
    seeds = sorted(train)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    NLL_TOP = 2.45
    for s in seeds:
        d = train[s]
        vl = np.where(d["val_nll"] > NLL_TOP, np.nan, d["val_nll"])
        ax.plot(d["epoch"], vl, color=seed_color(s, seeds), lw=1.7,
                label=f"s{s}")
    ax.axhline(2.17, color="0.35", lw=1.0, ls=":", zorder=0)
    ax.text(0.99, 2.17, "plateau ~2.17  ", color="0.35", fontsize=8.5,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_ylim(2.14, NLL_TOP)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL")
    ax.set_title(f"Flow validation loss per seed ({TAG}, N={len(seeds)} seeds)")
    ax.legend(ncol=2, loc="upper right", fontsize=8.5, title="flow seed",
              title_fontsize=9)
    return save_png(fig, "fig1_seed_loss")


# --------------------------------------------------------------------------
# FIGURE 3 / 4 -- per-seed multiplicative bias
# --------------------------------------------------------------------------
def _bias_figure(harv: dict, rblend: float, stem: str, title: str,
                 subtitle: str, caveat: str | None = None) -> tuple[str, dict]:
    seeds = sorted(harv)
    x = np.arange(len(seeds))
    m = np.array([m_of(harv[s]["R_flow"], rblend) for s in seeds])
    m_mean = float(np.mean(m))
    m_std = float(np.std(m, ddof=1)) if m.size > 1 else float("nan")
    m_sem = m_std / np.sqrt(m.size) if m.size > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    # +/-1% subpercent reference band
    ax.axhspan(-1.0, 1.0, color="#009E73", alpha=0.09, zorder=0)
    # ensemble mean +/- std/sqrt(N) band
    ax.axhspan(m_mean - m_sem, m_mean + m_sem, color="0.45", alpha=0.28,
               zorder=1)
    ax.axhline(m_mean, color="0.15", lw=1.6, zorder=2)
    ax.axhline(0.0, color="k", lw=1.1, ls="--", zorder=2)
    # per-seed points
    for xi, s in zip(x, seeds):
        ax.plot(xi, m[list(seeds).index(s)], "o", ms=9,
                color=seed_color(s, seeds), mec="white", mew=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in seeds], rotation=0)
    ax.set_xlim(-0.6, len(seeds) - 0.4)
    ax.set_ylabel(r"multiplicative bias  $m$  [%]")
    ax.set_xlabel("flow seed")
    ax.set_title(title)
    # annotate ensemble mean
    ax.text(0.02, 0.97,
            f"{subtitle}\n"
            rf"ensemble $m = {m_mean:+.2f} \pm {m_sem:.2f}\%$"
            f"  (std {m_std:.2f}%, N={m.size})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6",
                      alpha=0.95))
    handles = [
        Line2D([], [], color="0.15", lw=1.6, label="ensemble mean"),
        Patch(fc="0.45", alpha=0.4, label=r"mean $\pm$ std/$\sqrt{N}$"),
        Patch(fc="#009E73", alpha=0.18, label=r"$\pm1\%$ subpercent band"),
        Line2D([], [], color="k", lw=1.1, ls="--", label=r"$m=0$"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    if caveat:
        fig.text(0.5, -0.02, caveat, ha="center", va="top", fontsize=7.8,
                 color="#8a5a00",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e5",
                           ec="#E69F00", alpha=0.95))
    stats = dict(seeds=seeds, m=m, mean=m_mean, std=m_std, sem=m_sem,
                 rflow=[harv[s]["R_flow"] for s in seeds])
    return save_png(fig, stem), stats


def figure3(harv: dict):
    return _bias_figure(
        harv, R_BLEND_TRUE, "fig3_bias_true_neighbours",
        "Per-seed multiplicative bias -- true neighbours (constant sims)",
        rf"$R_{{\rm sim}}={R_SIM:.4f}$,  $R_{{\rm blend}}^{{\rm true}}={R_BLEND_TRUE:.4f}$"
        + "\n" + r"$m = R_{\rm sim}/(R_{\rm flow}^{\rm seed}+R_{\rm blend})-1$")


def figure4(harv: dict):
    caveat = (
        "INTERIM -- a faithful current-convention forward-model run is BLOCKED: "
        "probblend_forward.py Level-B is hardcoded to the main set (needs an aggregated "
        "detection_catalogue + g=0 field catalogues);\nthe constant set has neither "
        "(no detection catalogue, only +/-0.02 renders).  "
        r"Shown: $R_{\rm blend}^{\rm prob}=R_{\rm blend}^{\rm true}+0.0017$, the OLD 2026-07-09 "
        r"forward-model residual $\Delta m=-0.37\%$ (PROB_BLENDING cont.13) applied to the current seeds.")
    return _bias_figure(
        harv, R_BLEND_PROB, "fig4_bias_prob_neighbours",
        "Per-seed multiplicative bias -- probabilistic (forward-modelled) neighbours",
        rf"$R_{{\rm sim}}={R_SIM:.4f}$,  $R_{{\rm blend}}^{{\rm prob}}={R_BLEND_PROB:.4f}$"
        + "\n" + r"$m = R_{\rm sim}/(R_{\rm flow}^{\rm seed}+R_{\rm blend}^{\rm prob})-1$",
        caveat=caveat)


# --------------------------------------------------------------------------
# FIGURE 2 -- flow-predicted vs true response across galaxy properties
# --------------------------------------------------------------------------
def _load_dump() -> pd.DataFrame | None:
    """Prefer the fresh CURRENT-convention fixresp dump; fall back to the OLD
    (2026-07-09) constgold dump so the figure still builds while the fresh dump
    job is queued.  Re-running after the fresh dump lands upgrades automatically."""
    cands = sorted(glob.glob(str(RES_DIR / "fig2_perobj_s*_fixresp.feather")),
                   key=os.path.getmtime, reverse=True)
    if cands:
        df = pf.read_feather(cands[0])
        df.attrs["src"] = os.path.basename(cands[0])
        df.attrs["conv"] = "current (fixresp) convention"
        return df
    old = RES_DIR / "constgold_perobj_raw.feather"
    if old.exists():
        df = pf.read_feather(old)
        df.attrs["src"] = old.name
        df.attrs["conv"] = "OLD 2026-07-09 convention (fresh fixresp dump queued)"
        return df
    return None


def _read_key_table(path: str, cols: list[str], cases) -> pd.DataFrame:
    """Read `cols` for rows whose case is in `cases`, collapse to unique
    (case, input_index) by mean.  Case filtering is done at the Arrow level
    (fast) so the multi-GB lookups are cut before pandas ever sees them."""
    import pyarrow.compute as pc
    case_arr = pa.array(sorted(int(c) for c in cases))
    parts = []
    with ipc.open_file(path) as r:
        avail = [c for c in cols if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(avail)
            b = b.filter(pc.is_in(b["case"], value_set=case_arr))
            if b.num_rows:
                parts.append(b.to_pandas())
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=avail)
    return d.groupby(["case", "input_index"], as_index=False).mean(numeric_only=True)


def _read_cat_props(cases) -> pd.DataFrame:
    """Re_input_p (size) and S/N_plus (flux) from the constant catalogue, one row
    per (case, input_index)."""
    cat = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
           "constant_response_catalogue_train.feather")
    return _read_key_table(cat, ["case", "input_index", "Re_input_p", "S/N_plus"], cases)


def _read_blendflux(cases) -> pd.DataFrame:
    """nbr_flux_near (neighbour/blend flux) from the crowd_flux_conc lookup."""
    return _read_key_table(str(RES_DIR / "crowd_flux_conc_c0-199.feather"),
                           ["case", "input_index", "nbr_flux_near"], cases)


def _binned(x, ysim, ymod, nb=12, logx=False, qlo=1.0, qhi=99.0):
    good = np.isfinite(x) & np.isfinite(ysim) & np.isfinite(ymod)
    x, ysim, ymod = x[good], ysim[good], ymod[good]
    lo, hi = np.percentile(x, [qlo, qhi])
    xx = np.log10(np.clip(x, 1e-12, None)) if logx else x
    lo_e = np.log10(max(lo, 1e-12)) if logx else lo
    hi_e = np.log10(max(hi, 1e-12)) if logx else hi
    edges = np.linspace(lo_e, hi_e, nb + 1)
    idx = np.digitize(xx, edges) - 1
    cx, s_mean, s_lo, s_hi, m_mean = [], [], [], [], []
    for b in range(nb):
        sel = idx == b
        if sel.sum() < 30:
            continue
        cen = 10 ** (0.5 * (edges[b] + edges[b + 1])) if logx else 0.5 * (edges[b] + edges[b + 1])
        cx.append(cen)
        s_mean.append(np.mean(ysim[sel]))
        s_lo.append(np.percentile(ysim[sel], 16))
        s_hi.append(np.percentile(ysim[sel], 84))
        m_mean.append(np.mean(ymod[sel]))
    return (np.array(cx), np.array(s_mean), np.array(s_lo), np.array(s_hi),
            np.array(m_mean))


def figure2(status: dict) -> str | None:
    df = _load_dump()
    if df is None:
        print("[fig2] no dump feather yet (results/fig2_perobj_s*_fixresp.feather)"
              " -> SKIPPED; re-run after the dump job finishes.")
        status["fig2"] = "PENDING (dump job not finished)"
        return None
    src = df.attrs.get("src", "")          # attrs are dropped by merge -> grab now
    conv = df.attrs.get("conv", "")
    rflow_scalar = df["R_flow"].nunique() <= 1   # OLD dump stored R_flow as a global scalar
    cases = set(df["case"].unique().tolist())
    cat = _read_cat_props(cases)
    blf = _read_blendflux(cases)
    df = df.merge(cat, on=["case", "input_index"], how="left")
    df = df.merge(blf, on=["case", "input_index"], how="left")

    ysim = df["r_sim"].to_numpy(float)
    ymod = (df["R_flow"] + df["R_blend"]).to_numpy(float)

    panels = [
        ("S/N_plus", "primary flux  (S/N$_+$)", True, None),
        ("Re_input_p", "primary size  $R_e$ [pix]", False, None),
        # near-shell neighbour flux: most objects have ~0 -> bin only rows with
        # a resolved near neighbour so the axis is physically meaningful.
        ("nbr_flux_near", "blend / neighbour flux  (near shell)", True, 1e-3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    for ax, (col, xlabel, logx, xmin) in zip(axes, panels):
        x = df[col].to_numpy(float)
        ys, ym = ysim, ymod
        if xmin is not None:
            keep = np.isfinite(x) & (x > xmin)
            x, ys, ym = x[keep], ysim[keep], ymod[keep]
        cx, sm, slo, shi, mm = _binned(x, ys, ym, nb=12, logx=logx)
        ax.fill_between(cx, slo, shi, color="#0072B2", alpha=0.15, lw=0,
                        label="sim 16-84%")
        ax.plot(cx, sm, "-o", color="#0072B2", ms=5, lw=1.8,
                label=r"true (sim) $r_{\rm sim}$")
        ax.plot(cx, mm, "-s", color="#D55E00", ms=5, lw=1.8,
                label=r"flow model $R_{\rm flow}+R_{\rm blend}$")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.axhline(R_SIM, color="0.5", lw=1.0, ls=":", zorder=0)
    axes[0].set_ylabel("mean response  R")
    axes[0].legend(loc="best", fontsize=8.5)
    axes[0].text(0.02, 0.02, f"dump: {src}\n{conv}\nN={len(df):,} objects",
                 transform=axes[0].transAxes,
                 fontsize=7.5, va="bottom", ha="left", color="0.35")
    fig.suptitle("Flow-predicted vs true response across galaxy properties "
                 "(constant sims, true neighbours)", fontsize=13, y=1.01)
    if rflow_scalar:
        fig.text(0.5, -0.03,
                 "CAVEAT: this OLD dump stored R_flow as a GLOBAL scalar, so the "
                 "flow-model curve varies only through R_blend and cannot trace the "
                 "per-object flux/size trend.\nThe queued fresh fixresp dump "
                 "(per-object R_flow) will let the model track truth across property "
                 "space -- re-run this script once it lands to auto-upgrade Figure 2.",
                 ha="center", va="top", fontsize=8, color="#8a5a00",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fff6e5", ec="#E69F00",
                           alpha=0.95))
    status["fig2_conv"] = conv + (" [R_flow scalar]" if rflow_scalar else "")
    fig.tight_layout()
    status["fig2"] = "OK (OLD dump, R_flow scalar)" if rflow_scalar else "OK"
    return save_png(fig, "fig2_response_vs_properties")


# --------------------------------------------------------------------------
# FIGURE 5 -- flow SELF-response vs truth target, across flux & size
# --------------------------------------------------------------------------
RESP_TARGET_NPZ = RES_DIR / "response_target_crowd_rblend_snc_c0-99_6x3x5.npz"
FIG5_BINS_NPZ = RES_DIR / "fig5_selfresp_bins_s501.npz"


def _marg_target():
    """Rsim(flux) and Rsim(size) from the response target, marginalizing the
    other property axis AND the crowd axis with `counts` weights."""
    z = np.load(RESP_TARGET_NPZ)
    Rsim = z["Rsim"]           # (flux, size, crowd)
    cnt = z["counts"]
    w = np.where(np.isfinite(Rsim), cnt, 0.0)
    Rv = np.where(np.isfinite(Rsim), Rsim, 0.0)
    rflux = (Rv * w).sum(axis=(1, 2)) / np.clip(w.sum(axis=(1, 2)), 1e-9, None)
    rsize = (Rv * w).sum(axis=(0, 2)) / np.clip(w.sum(axis=(0, 2)), 1e-9, None)
    return z["edges_flux"], z["edges_size"], rflux, rsize


def figure5(status: dict) -> str | None:
    if not FIG5_BINS_NPZ.exists():
        print(f"[fig5] no eval npz yet ({FIG5_BINS_NPZ.name}) -> SKIPPED; re-run "
              "after the fig5 self-response GPU job finishes.")
        status["fig5"] = "PENDING (self-response eval job not finished)"
        return None
    m = np.load(FIG5_BINS_NPZ)
    ef, es, rflux_t, rsize_t = _marg_target()
    rflux_m, rsize_m = m["Rmodel_flux"], m["Rmodel_size"]
    cf = 0.5 * (ef[:-1] + ef[1:])     # flux bin centres (r_input_p mag)
    cs = 0.5 * (es[:-1] + es[1:])     # size bin centres (Re)

    fig, (axf, axs) = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    for ax, cx, rt, rm, xlabel in [
        (axf, cf, rflux_t, rflux_m, r"primary flux  (r-band mag $r_{\rm input,p}$)"),
        (axs, cs, rsize_t, rsize_m, r"primary size  $R_e$ [pix]"),
    ]:
        ax.plot(cx, rt, "-o", color="#0072B2", ms=6, lw=1.9,
                label=r"truth target $R_{\rm sim}$")
        ax.plot(cx, rm, "-s", color="#D55E00", ms=6, lw=1.9,
                label=r"flow self-response $R_{\rm model}$")
        for x, a, b in zip(cx, rt, rm):
            if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-6:
                ax.annotate(f"{(b/a-1)*100:+.0f}%", (x, b), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7.5, color="#8a5a00")
        ax.set_xlabel(xlabel)
    axf.set_ylabel("self-response  R")
    axf.legend(loc="best", fontsize=9)
    axf.text(0.02, 0.02, f"s501, N={int(m['n_rows_used']):,}  "
             rf"($\delta={float(m['delta'])}$, central trace/2)",
             transform=axf.transAxes, fontsize=7.5, va="bottom", ha="left", color="0.35")
    fig.suptitle("Flow self-response vs truth target across galaxy properties "
                 "(training objective, g=0 catalogue)", fontsize=13, y=1.01)
    fig.tight_layout()
    status["fig5"] = "OK"
    return save_png(fig, "fig5_self_response_vs_truth")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    set_style()
    status: dict[str, str] = {}

    train = parse_training_logs()
    harv = parse_harvest_seeds()

    p1 = figure1(train)
    p3, s3 = figure3(harv)
    p4, s4 = figure4(harv)
    p2 = figure2(status)
    p5 = figure5(status)

    print("\n" + "=" * 70)
    print("FIGURE 1 -- validation NLL vs epoch")
    print("=" * 70)
    print(f"  seeds (train): {sorted(train)}  -> {p1}")
    for s in sorted(train):
        d = train[s]
        print(f"    s{s}: {d['epoch'].size} epochs, final val NLL="
              f"{d['val_nll'][-1]:.4f}  (job {d['jobid']})")

    def _bias_report(tag, stats, rblend):
        print("\n" + "=" * 70)
        print(f"{tag}  (R_blend={rblend:.4f})")
        print("=" * 70)
        print(f"{'seed':>5} {'R_flow':>9} {'m [%]':>9}")
        for s, rf, mm in zip(stats['seeds'], stats['rflow'], stats['m']):
            print(f"{s:>5} {rf:>9.4f} {mm:>+9.2f}")
        print("-" * 40)
        print(f"  ensemble m = {stats['mean']:+.3f}%  std={stats['std']:.3f}%  "
              f"std/sqrt(N={len(stats['seeds'])})={stats['sem']:.3f}%")

    _bias_report("FIGURE 3 -- per-seed m (TRUE neighbours)", s3, R_BLEND_TRUE)
    _bias_report("FIGURE 4 -- per-seed m (PROBABILISTIC neighbours)", s4, R_BLEND_PROB)
    print(f"  [fig4] R_blend_prob provenance: R_blend_true {R_BLEND_TRUE:.4f} "
          f"+ 0.0017 (forward-model residual dm={PROBBLEND_DM:+.2%}, OLD 07-09 conv)")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    for p in (p1, p2, p3, p4, p5):
        print("  ", p if p else "(pending -- see status)")
    print("  status:", status)


if __name__ == "__main__":
    main()
