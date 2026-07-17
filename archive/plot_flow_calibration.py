#!/usr/bin/env python
"""Publication figures for the SBSI shear-calibration flow result.

Parses the flow-training logs (Data Source A) and harvest logs (Data Source B)
and regenerates two stage figures (Stage 1 = raw flow ingredient; Stage 2 =
constant-sim certified m).  Re-running picks up any harvest seeds that land
later, since parsing is robust to still-running jobs.

STAGE 3 (probabilistic neighbours) -- deferred, see WORKLOG cont.46

Environment:
    eval "$(conda shell.bash hook)"; conda activate sims1
    export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"

Run directly on the login node (pure log-parsing + plotting, negligible compute).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
LOG_DIR = Path("/home/z/Zekang.Zhang/logs")
OUT_DIR = Path("/home/z/Zekang.Zhang/SBSI/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seed-independent physical constants (see prompt / WORKLOG)
R_SIM = 0.4534        # simulation truth response
R_BLEND = 0.1593      # emulator blend response (seed-independent)
G_SHEAR = 0.02        # applied shear
R_FLOW_TARGET = R_SIM - R_BLEND  # 0.2941 -> m=0 target for R_flow

SEEDS = [501, 502, 503, 504, 505, 506, 507, 508]

# Consistent, colorblind-friendly per-seed colors (Okabe-Ito based, 8 distinct)
_PALETTE = [
    "#0072B2",  # blue         501
    "#E69F00",  # orange       502
    "#009E73",  # green        503
    "#D55E00",  # vermillion   504
    "#CC79A7",  # reddish pink 505
    "#56B4E9",  # sky blue     506
    "#F0E442",  # yellow       507
    "#000000",  # black        508
]
SEED_COLOR = {s: _PALETTE[i] for i, s in enumerate(SEEDS)}

# Train-log source map (fixresp, lam450)
CONCAT_TRAIN_LOG = LOG_DIR / "pilot_train_15101239.out"  # seeds 501,502,503
SINGLE_TRAIN_LOGS = {
    504: LOG_DIR / "pilot_train_15118403.out",
    505: LOG_DIR / "pilot_train_15118404.out",
    506: LOG_DIR / "pilot_train_15118405.out",
    507: LOG_DIR / "pilot_train_15118406.out",
    508: LOG_DIR / "pilot_train_15118407.out",
}

# Harvest-log source maps
PRECRN_BASELINE_LOG = LOG_DIR / "pilot_harvest_15118328.out"   # seeds 501,502,503 serial
PRECRN_SOLO_S503_LOG = LOG_DIR / "pilot_harvest_15123285.out"  # s503 solo (pos 1)
POSTCRN_12345 = {
    501: LOG_DIR / "pilot_harvest_15124066.out",
    502: LOG_DIR / "pilot_harvest_15124067.out",
    503: LOG_DIR / "pilot_harvest_15124068.out",
    504: LOG_DIR / "pilot_harvest_15124069.out",
    505: LOG_DIR / "pilot_harvest_15124070.out",
    506: LOG_DIR / "pilot_harvest_15124071.out",
    507: LOG_DIR / "pilot_harvest_15124072.out",
    508: LOG_DIR / "pilot_harvest_15124073.out",
}
POSTCRN_777 = {  # cross-check flow-seed
    501: LOG_DIR / "pilot_harvest_15124074.out",
    503: LOG_DIR / "pilot_harvest_15124075.out",
}

# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
# e.g.  epoch 072: nll=2.17155/2.18118  <R_model>(val)=+0.2963 (target mean 0.1984) ...
_EPOCH_RE = re.compile(
    r"epoch\s+(\d+):\s*nll=([-\d.eE+]+)/([-\d.eE+]+)\s+"
    r"<R_model>\(val\)=([-+][\d.eE]+)"
)
_DONE_RE = re.compile(r"TRAIN_DONE seed=(\d+)")


def _parse_epoch_block(text: str) -> dict:
    """Parse epoch/train_nll/val_nll/R_model arrays from a block of log text."""
    epochs, tr, vl, rm = [], [], [], []
    for m in _EPOCH_RE.finditer(text):
        epochs.append(int(m.group(1)))
        tr.append(float(m.group(2)))
        vl.append(float(m.group(3)))
        rm.append(float(m.group(4)))
    return {
        "epoch": np.asarray(epochs, float),
        "train_nll": np.asarray(tr, float),
        "val_nll": np.asarray(vl, float),
        "R_model": np.asarray(rm, float),
    }


def parse_training_logs() -> dict:
    """Return {seed: {epoch, train_nll, val_nll, R_model}} for all available seeds."""
    out = {}

    # Concatenated 501/502/503: split on TRAIN_DONE seed=<N>; the segment
    # BEFORE each marker belongs to that seed.
    if CONCAT_TRAIN_LOG.exists():
        txt = CONCAT_TRAIN_LOG.read_text(errors="replace")
        pos = 0
        for m in _DONE_RE.finditer(txt):
            seed = int(m.group(1))
            segment = txt[pos:m.start()]
            block = _parse_epoch_block(segment)
            if block["epoch"].size:
                out[seed] = block
            pos = m.end()

    # Single-seed logs 504-508
    for seed, path in SINGLE_TRAIN_LOGS.items():
        if not path.exists():
            continue
        block = _parse_epoch_block(path.read_text(errors="replace"))
        if block["epoch"].size:
            out[seed] = block

    return out


# GLOBAL:  R_sim=0.4534  R_flow(self)=0.2900  R_blend(emulator)=0.1593  R_total=0.4493
_GLOBAL_RE = re.compile(
    r"\[s(\d+)\]\s*GLOBAL:\s*R_sim=([-\d.]+)\s+R_flow\(self\)=([-\d.]+)\s+"
    r"R_blend\(emulator\)=([-\d.]+)\s+R_total=([-\d.]+)"
)
# WITH blend m = ... = +0.93% +/- 0.18%
_M_RE = re.compile(
    r"\[s(\d+)\]\s*WITH blend m\s*=.*?=\s*([-+][\d.]+)%\s*\+/-\s*([\d.]+)%"
)


def parse_harvest_log(path: Path) -> dict:
    """Parse all [sNNN] GLOBAL + WITH-blend-m records in one harvest log.

    Returns {seed: {R_sim, R_flow, R_blend, R_total, m, m_err}}.  Robust to
    logs that have no GLOBAL line yet (still running) -> returns {}.
    """
    if not path.exists():
        return {}
    txt = path.read_text(errors="replace")
    recs: dict[int, dict] = {}
    for m in _GLOBAL_RE.finditer(txt):
        seed = int(m.group(1))
        recs[seed] = {
            "R_sim": float(m.group(2)),
            "R_flow": float(m.group(3)),
            "R_blend": float(m.group(4)),
            "R_total": float(m.group(5)),
        }
    for m in _M_RE.finditer(txt):
        seed = int(m.group(1))
        if seed in recs:
            recs[seed]["m"] = float(m.group(2))
            recs[seed]["m_err"] = float(m.group(3))
    return recs


def parse_precrn_baseline() -> dict:
    """PRE-CRN serial baseline (buggy estimator): seeds 501,502,503."""
    return parse_harvest_log(PRECRN_BASELINE_LOG)


def parse_postcrn(job_map: dict) -> dict:
    """Parse a {seed: logpath} POST-CRN map into {seed: record}, skipping
    seeds whose log has no GLOBAL line yet."""
    out = {}
    for seed, path in job_map.items():
        recs = parse_harvest_log(path)
        # solo harvest -> single [sNNN]; take the matching seed if present
        if seed in recs:
            out[seed] = recs[seed]
        elif len(recs) == 1:
            out[seed] = next(iter(recs.values()))
    return out


# --------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------
def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def _save(fig, stem: str) -> list[str]:
    paths = []
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------
# STAGE 1 -- the raw flow ingredient
# --------------------------------------------------------------------------
def plot_stage1(train: dict, axes=None, standalone=True):
    if axes is None:
        fig, (ax_nll, ax_r) = plt.subplots(1, 2, figsize=(11, 4.4))
    else:
        ax_nll, ax_r = axes
        fig = ax_nll.figure

    seeds = sorted(train)

    # -- 1a: val-NLL vs epoch (+ faint dashed train-NLL) --
    NLL_TOP = 2.45
    for s in seeds:
        d = train[s]
        c = SEED_COLOR[s]
        # clip early train-NLL spikes above the plateau view so they don't
        # draw vertical streaks across the panel
        tr = np.where(d["train_nll"] > NLL_TOP, np.nan, d["train_nll"])
        vl = np.where(d["val_nll"] > NLL_TOP, np.nan, d["val_nll"])
        ax_nll.plot(d["epoch"], tr, color=c, lw=0.9, ls="--", alpha=0.35)
        ax_nll.plot(d["epoch"], vl, color=c, lw=1.6, label=f"s{s}")
    ax_nll.set_ylim(2.14, NLL_TOP)
    ax_nll.axhline(2.17, color="0.35", lw=1.0, ls=":", zorder=0)
    ax_nll.text(0.97, 2.17, "  plateau ~2.17", color="0.35", fontsize=8,
                va="bottom", ha="right", transform=ax_nll.get_yaxis_transform())
    ax_nll.set_xlabel("epoch")
    ax_nll.set_ylabel("NLL")
    ax_nll.set_title("(a) Flow loss converges to the ~2.17 plateau")
    # combined legend: seeds + train/val style
    seed_handles = [Line2D([], [], color=SEED_COLOR[s], lw=1.8, label=f"s{s}")
                    for s in seeds]
    style_handles = [
        Line2D([], [], color="0.4", lw=1.6, ls="-", label="val"),
        Line2D([], [], color="0.4", lw=0.9, ls="--", alpha=0.6, label="train"),
    ]
    ax_nll.legend(handles=seed_handles + style_handles, ncol=2,
                  loc="upper right", fontsize=8)

    # -- 1b: <R_model>(val) vs epoch + per-seed mean lines --
    bounce_stds = []
    for s in seeds:
        d = train[s]
        c = SEED_COLOR[s]
        ax_r.plot(d["epoch"], d["R_model"], color=c, lw=1.1, alpha=0.85,
                  label=f"s{s}")
        mean_r = float(np.mean(d["R_model"]))
        ax_r.axhline(mean_r, color=c, lw=1.2, ls="--", alpha=0.6, zorder=1)
        late = d["R_model"][d["epoch"] >= 60]
        if late.size:
            bounce_stds.append(float(np.std(late)))

    med_bounce = float(np.median(bounce_stds)) if bounce_stds else float("nan")
    ax_r.axhline(R_FLOW_TARGET, color="0.15", lw=1.4, ls="-", zorder=2)
    ax_r.text(0.02, R_FLOW_TARGET, f" m=0 target R_flow={R_FLOW_TARGET:.4f}",
              color="0.15", fontsize=8, va="bottom", ha="left",
              transform=ax_r.get_yaxis_transform())
    ax_r.set_xlabel("epoch")
    ax_r.set_ylabel(r"$\langle R_{\rm model}\rangle$ (val)")
    ax_r.set_title("(b) Single-checkpoint response still bounces")
    ax_r.annotate(
        f"per-epoch bounce (epoch$\\geq$60)\nstd $\\approx${med_bounce:.3f}"
        "\n$\\Rightarrow$ motivates CRN + ensembling",
        xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9))
    ax_r.legend(ncol=2, loc="upper right", fontsize=8)

    if standalone:
        fig.suptitle("Stage 1 -- the raw flow ingredient: converged loss, noisy checkpoint response",
                     fontsize=13, y=1.02)
        fig.tight_layout()
    return fig, med_bounce


# --------------------------------------------------------------------------
# STAGE 2 -- constant sims, true neighbours: certified m
# --------------------------------------------------------------------------
def plot_stage2(precrn, postcrn, s503_solo, axes=None, standalone=True):
    if axes is None:
        fig, (ax_dec, ax_m) = plt.subplots(1, 2, figsize=(11, 4.6))
    else:
        ax_dec, ax_m = axes
        fig = ax_dec.figure

    post_seeds = sorted(postcrn)

    # -- 2a: response decomposition (stacked R_flow + R_blend vs R_sim) --
    x = np.arange(len(post_seeds))
    rflow = np.array([postcrn[s]["R_flow"] for s in post_seeds])
    rbld = np.array([postcrn[s]["R_blend"] for s in post_seeds])
    bar_colors = [SEED_COLOR[s] for s in post_seeds]

    ax_dec.bar(x, rflow, width=0.62, color=bar_colors, alpha=0.9,
               edgecolor="white", linewidth=0.6, label=r"$R_{\rm flow}$ (per seed)")
    ax_dec.bar(x, rbld, width=0.62, bottom=rflow, color="0.7", alpha=0.9,
               edgecolor="white", linewidth=0.6,
               label=r"$R_{\rm blend}=%.4f$" % R_BLEND)
    ax_dec.axhline(R_SIM, color="0.10", lw=1.8, ls="-", zorder=5)
    ax_dec.text(len(x) - 0.5, R_SIM, r" $R_{\rm sim}=%.4f$" % R_SIM,
                color="0.10", fontsize=8.5, va="bottom", ha="right")
    ax_dec.axhline(R_FLOW_TARGET, color="0.25", lw=1.2, ls=":", zorder=4)
    ax_dec.text(-0.45, R_FLOW_TARGET, r" $R_{\rm flow}$ target=%.4f" % R_FLOW_TARGET,
                color="0.25", fontsize=8, va="bottom", ha="left")
    ax_dec.set_xticks(x)
    ax_dec.set_xticklabels([f"s{s}" for s in post_seeds])
    ax_dec.set_ylim(0, 0.52)
    ax_dec.set_ylabel("response R")
    ax_dec.set_title(r"(a) $R_{\rm flow}+R_{\rm blend}$ lands on $R_{\rm sim}$ (POST-CRN)")
    ax_dec.legend(loc="lower center", ncol=1, fontsize=8.5)

    # -- 2b: THE HEADLINE -- per-seed m, PRE-CRN vs POST-CRN --
    pre_seeds = sorted(precrn)
    xp_pre, xp_post = 0.0, 1.0

    # pre-CRN cluster (wild spread)
    for s in pre_seeds:
        r = precrn[s]
        jit = (SEEDS.index(s) - 3.5) * 0.035
        ax_m.errorbar(xp_pre + jit, r["m"], yerr=r["m_err"], fmt="o", ms=8,
                      color=SEED_COLOR[s], mec="white", mew=0.8, capsize=3,
                      elinewidth=1.1, zorder=4, label=f"s{s}")
    # post-CRN cluster (tight)
    post_m = np.array([postcrn[s]["m"] for s in post_seeds])
    for s in post_seeds:
        r = postcrn[s]
        jit = (SEEDS.index(s) - 3.5) * 0.035
        ax_m.errorbar(xp_post + jit, r["m"], yerr=r["m_err"], fmt="s", ms=8,
                      color=SEED_COLOR[s], mec="white", mew=0.8, capsize=3,
                      elinewidth=1.1, zorder=4)

    # ensemble mean band (mean +/- std) and mean's error (std/sqrt N)
    m_mean = float(np.mean(post_m))
    m_std = float(np.std(post_m, ddof=1)) if post_m.size > 1 else float("nan")
    m_sem = m_std / np.sqrt(post_m.size) if post_m.size > 1 else float("nan")
    ax_m.axhspan(m_mean - m_std, m_mean + m_std, xmin=0.42, xmax=0.98,
                 color="0.6", alpha=0.20, zorder=1,
                 label=r"POST-CRN mean $\pm$ std")
    ax_m.axhspan(m_mean - m_sem, m_mean + m_sem, xmin=0.42, xmax=0.98,
                 color="0.35", alpha=0.30, zorder=2,
                 label=r"mean $\pm$ std/$\sqrt{N}$")
    ax_m.hlines(m_mean, xp_post - 0.28, xp_post + 0.28, color="0.15",
                lw=1.8, zorder=3)

    # zero line + subpercent band
    ax_m.axhline(0.0, color="k", lw=1.0, ls="-", zorder=2)
    ax_m.axhspan(-1.0, 1.0, color="#009E73", alpha=0.10, zorder=0,
                 label=r"$\pm1\%$ subpercent band")

    ax_m.set_xticks([xp_pre, xp_post])
    ax_m.set_xticklabels(["PRE-CRN\n(buggy estimator)",
                          f"POST-CRN\n(CRN ensemble, N={post_m.size})"])
    ax_m.set_xlim(-0.5, 1.6)
    ax_m.set_ylabel("multiplicative bias  m  [%]")
    ax_m.set_title("(b) CRN + ensembling certifies subpercent m")

    # RNG-position artifact annotation (same s503 checkpoint, two positions)
    txt = (
        "same s503 checkpoint, buggy estimator:\n"
        r"pos 3: $R_{\rm flow}$=0.268 $\to$ m=+6.04%""\n"
        r"pos 1: $R_{\rm flow}$=0.309 $\to$ m=$-$3.19%""\n"
        "$\\Rightarrow$ RNG serial-position artifact\nthat CRN removes"
    )
    ax_m.annotate(
        txt, xy=(xp_pre, 6.04), xytext=(0.02, 0.62),
        textcoords="axes fraction", fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e5", ec="#E69F00",
                  alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#E69F00", lw=1.1))

    # legend: seed markers + summary handles
    seed_handles = [Line2D([], [], marker="o", ls="", color=SEED_COLOR[s],
                           mec="white", ms=7, label=f"s{s}")
                    for s in sorted(set(pre_seeds) | set(post_seeds))]
    summary_handles = [
        Line2D([], [], marker="o", ls="", color="0.4", ms=7, label="PRE-CRN"),
        Line2D([], [], marker="s", ls="", color="0.4", ms=7, label="POST-CRN"),
        Patch(fc="0.6", alpha=0.30, label=r"mean$\pm$std"),
        Patch(fc="#009E73", alpha=0.18, label=r"$\pm1\%$ band"),
    ]
    leg1 = ax_m.legend(handles=seed_handles, loc="upper right", ncol=2,
                       fontsize=7.5, title="seed", title_fontsize=8)
    ax_m.add_artist(leg1)
    ax_m.legend(handles=summary_handles, loc="lower left", fontsize=7.5)

    # headline text of the ensemble result: placed in the clear space above
    # the POST-CRN cluster (points sit near m in [-1, 1.2]%)
    ax_m.text(
        xp_post, 3.0, f"$m = {m_mean:+.2f} \\pm {m_sem:.2f}\\%$",
        color="0.10", fontsize=11, va="center", ha="center", zorder=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.3", alpha=0.9))

    if standalone:
        fig.suptitle("Stage 2 -- constant sims, true neighbours: certified multiplicative bias",
                     fontsize=13, y=1.02)
        fig.tight_layout()
    return fig, (m_mean, m_std, m_sem)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    set_style()

    # ---- parse ----
    train = parse_training_logs()
    precrn = parse_precrn_baseline()
    s503_solo = parse_harvest_log(PRECRN_SOLO_S503_LOG).get(503, {})
    postcrn = parse_postcrn(POSTCRN_12345)
    postcrn_777 = parse_postcrn(POSTCRN_777)  # cross-check (parsed, not plotted)

    # ---- stage 1 ----
    fig1, med_bounce = plot_stage1(train)
    p1 = _save(fig1, "flow_calibration_stage1")
    plt.close(fig1)

    # ---- stage 2 ----
    fig2, (m_mean, m_std, m_sem) = plot_stage2(precrn, postcrn, s503_solo)
    p2 = _save(fig2, "flow_calibration_stage2")
    plt.close(fig2)

    # ---- combined 2-row ----
    figc, axc = plt.subplots(2, 2, figsize=(11.5, 9))
    plot_stage1(train, axes=(axc[0, 0], axc[0, 1]), standalone=False)
    plot_stage2(precrn, postcrn, s503_solo, axes=(axc[1, 0], axc[1, 1]),
                standalone=False)
    axc[0, 0].annotate("STAGE 1 -- raw flow ingredient", xy=(0, 1.14),
                       xycoords="axes fraction", fontsize=12, fontweight="bold")
    axc[1, 0].annotate("STAGE 2 -- constant sims, certified m", xy=(0, 1.14),
                       xycoords="axes fraction", fontsize=12, fontweight="bold")
    figc.suptitle("SBSI shear-calibration flow: from noisy checkpoint to subpercent m",
                  fontsize=14, y=1.00)
    figc.tight_layout(rect=(0, 0, 1, 0.99))
    pc = _save(figc, "flow_calibration")
    plt.close(figc)

    # ---- report ----
    pending = [s for s, p in POSTCRN_12345.items()
               if s not in postcrn]
    print("\n" + "=" * 68)
    print("POST-CRN ensemble (flow-seed 12345)")
    print("=" * 68)
    print(f"{'seed':>5} {'R_flow':>9} {'m [%]':>9} {'m_err [%]':>10}")
    for s in sorted(postcrn):
        r = postcrn[s]
        print(f"{s:>5} {r['R_flow']:>9.4f} {r['m']:>+9.2f} {r['m_err']:>10.2f}")
    print("-" * 68)
    print(f"ensemble-mean m = {m_mean:+.3f}%   std = {m_std:.3f}%   "
          f"std/sqrt(N={len(postcrn)}) = {m_sem:.3f}%")
    print(f"R_flow target (m=0) = {R_FLOW_TARGET:.4f}   "
          f"mean R_flow = {np.mean([postcrn[s]['R_flow'] for s in postcrn]):.4f}")

    print("\nSeeds available (POST-CRN 12345):", sorted(postcrn))
    print("Seeds still pending (no GLOBAL line yet):", pending or "none")
    print("PRE-CRN baseline seeds:", sorted(precrn))
    print("POST-CRN 777 cross-check seeds:", sorted(postcrn_777))
    if s503_solo:
        print(f"s503 solo (pos1) cross-check: R_flow={s503_solo['R_flow']:.4f} "
              f"m={s503_solo['m']:+.2f}%")
    print(f"Stage-1 per-epoch bounce (median std, epoch>=60): {med_bounce:.4f}")

    print("\nWritten figures:")
    for p in p1 + p2 + pc:
        print("  ", p)


if __name__ == "__main__":
    main()
