"""Train FLOW #2 -- the blending-response flow (`Gold-V3.md`), a firewalled drop-in for BlendEMU.

    loss = nll_weight * NLL(measured shape | pair)  +  response_weight * <(R_model - truth)^2> / Var

DUAL MODE (`--self-response-weight > 0`, added 2026-08-02i): STEP 2 OF MERGING FLOWS #1 AND #2
==============================================================================================
The two flows model the same object -- the primary's measured properties given true properties, on
the g=0 leg -- and differ in what they condition on and which shape the shear map acts on. Flow #2's
context is nearly a superset of flow #1's and ALREADY carries the primary's oriented intrinsic shape,
so the self-response channel exists in this model today and is simply never supervised. Dual mode
supervises it:

    loss += self_response_weight * <w (R_self - self_truth)^2> / Var(self_truth),    w = 1/k

`self_truth` is built by `build_blend_pairset.py` from the SAME measured-shape difference as the
blend label, projected on the primary's shear direction instead of the neighbour's.

WHAT THIS RUN IS FOR, AND WHAT IT IS NOT. It is a FEASIBILITY test of one question: can a single
network hold a ~0.86 self response and a ~0.14 blend response at once, or does the strong channel
crowd out the weak one? That crowding is not hypothetical -- it is exactly what the NLL-only run
measured (the density-derived blend response came out 70-85% too low and nearly flat, because maximum
likelihood spends capacity on the primary's own shape, which correlates ~0.8 with the target, and
under-fits the neighbour's, which correlates ~0.035).

It is NOT a replacement for flow #1, for three reasons that all have to be fixed before it could be:

  1. NO ISOLATED PRIMARIES. Every row here is a (primary, neighbour) pair, so primaries with no
     annotated neighbour inside 7" are absent entirely -- about a quarter of the population. The
     self response learned here is conditioned on being neighboured.
  2. NO UNIQUE R_self PER PRIMARY. `R_self` depends on the whole context, so a primary with k rows
     yields k slightly different predictions. Evaluation averages them, which is the natural
     pair-model estimator, but a scene model that sees all k neighbours at once would not need to.
  3. FEWER INPUTS, FEWER OUTPUTS. Flow #1 conditions on measured mag and size and crowding scalars
     and emits 4 dimensions (shape, flux, size); this emits shape only from true properties. Its
     `R_self` is therefore NOT numerically comparable to the fiducial `R_flow` and must not be
     substituted for it.

Both (1) and (2) are what the scene-level step is for. Read this run as a go/no-go on the merge, not
as a number.

`R_model` is the mean head's shift under a CENTRAL shear of the NEIGHBOUR's intrinsic shape, in the
same trace/2 convention `train_measurement_model_swa_s1_truecond.epoch_response` uses for flow #1, so
the two responses are the same kind of number. `truth` is the per-pair half-shear label built by
`scripts/build_blend_pairset.py`. See `sbs_shear/blend_flow.py` for why the supervision is per pair
rather than on a binned cell grid, and for the statistical floor that sets.

THE SPLIT IS BY CASE, and cases are separate simulated fields, so the held-out set shares no galaxy,
no neighbour and no noise realisation with the training set. Every number reported as held-out is
therefore a genuine generalisation test, not a re-read of the training labels.

WHAT TO READ IN THE OUTPUT. Not the loss -- the label is ~230x noisier than the signal, so both loss
terms are dominated by irreducible measurement noise and barely move. Read the HELD-OUT SEPARATION
TABLE: per-bin <R_model> against <truth> with the label's own sem. The target is the close-pair bins,
where BlendEMU sits at -36% and -39%, WITHOUT degrading 1.5-4" where it is already within a few
percent. A model that fixes the close pairs by breaking the wide ones has not improved anything --
that is exactly the trade the emulator's close-pair weighting failed at.

FIREWALL: half-shear only; constgold is never opened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.blend_flow import (  # noqa: E402
    RAW, SHAPE_S, TARGETS, Standardizer, build_features, build_model, feature_names,
    response_from_contexts, save_blend_flow, shape_column_indices, shifted_shape_columns)

SEP_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]

# Neighbour-magnitude edges for the STRATUM penalty and the 2-D selection metric (added 2026-08-02g).
# WHY THIS AXIS EXISTS AT ALL. Up to now both the response loss and the checkpoint criterion binned by
# SEPARATION only, and a separation bin mixes bright and faint neighbours freely. The measured failure
# (WORKLOG 2026-08-02e) is on the neighbour-BRIGHTNESS axis -- 2.5x over on the pairs fainter than 26,
# 11% under on the ones brighter -- so it was invisible to both: a model can sit at chi2/dof ~ 1 in
# every separation bin while being badly wrong in opposite directions within each one. Adding the
# magnitude axis is not a tweak to the criterion, it is the difference between a criterion that can
# see the defect and one that cannot.
MAG_EDGES = [0.0, 24.0, 25.0, 26.0, 27.0, 28.0, 99.0]

# PRIMARY axes (added 2026-08-02h). `eval_blend_flow.py` bins by the primary's own size and magnitude
# as well as by separation, and that is where the strata-anchored model is still 3-4x worse than the
# emulator (Re_p 4.0 / r_p 2.7 against 1.3 / 0.6) -- a SECOND conditional bias, on an axis the
# sep x neighbour-mag grid never named. Edges are set from the pair set's own quantiles (r_input_p
# p5-p100 = 21.7-26.0, Re_input_p p5-p95 = 0.32-1.65) so the bins carry comparable populations
# instead of leaving most of the data in one bin.
PRIMAG_EDGES = [0.0, 22.0, 23.5, 24.5, 25.0, 25.5, 99.0]
PRISIZE_EDGES = [0.0, 0.35, 0.42, 0.52, 0.70, 1.00, 99.0]

# MARGINAL grids, not one joint grid. A full outer product of the four axes would be ~1700 cells, and
# with the realised anchor S/N already median ~1.7 on 48 cells, splitting the same pairs that far
# would leave almost every anchor consistent with noise -- the penalty would still have a large chi2
# but would be pulling toward nothing. Summing several 2-D grids, each marginalised over the axes it
# does not name, constrains every axis while keeping ~1M pairs per cell.
GRIDS = {
    "sep_x_nbrmag": (("distance", SEP_EDGES), ("r_input_s", MAG_EDGES)),
    "sep_x_primag": (("distance", SEP_EDGES), ("r_input_p", PRIMAG_EDGES)),
    "sep_x_prisize": (("distance", SEP_EDGES), ("Re_input_p", PRISIZE_EDGES)),
    # PRIMARY SIZE x NEIGHBOUR MAGNITUDE (added 2026-08-02l). Primary size is the ONLY axis where
    # BlendEMU still beats flow #2 (chi2/dof 2.2 vs 1.3, replicated at 2.0). Every existing grid
    # crosses its second axis with SEPARATION, so a bias that lives in the size x neighbour-brightness
    # plane -- a large primary next to a faint neighbour behaving unlike a large primary next to a
    # bright one, at the same separation -- is marginalised away in all three. This grid names that
    # plane directly. It is the same reasoning that made the neighbour-magnitude axis worth adding in
    # round 1: a criterion that cannot see a defect cannot fix it.
    "prisize_x_nbrmag": (("Re_input_p", PRISIZE_EDGES), ("r_input_s", MAG_EDGES)),
}
DEFAULT_GRIDS = "sep_x_nbrmag,sep_x_primag,sep_x_prisize"

# SELF-channel grids (added 2026-08-02j). Every axis here is a PRIMARY property, which is not an
# aesthetic choice: the self response is a property of the primary, so all k of a primary's rows fall
# in the SAME cell and the anchor can be formed over primaries exactly. A grid crossing separation or
# neighbour magnitude would split one primary's rows across cells, and the per-primary anchor -- the
# thing that makes the error bar honest -- would stop being well defined.
K_EDGES = [1, 2, 3, 4, 6, 9, 99]
SELF_GRIDS = {
    "primag_x_prisize": (("r_input_p", PRIMAG_EDGES), ("Re_input_p", PRISIZE_EDGES)),
    "primag_x_k": (("r_input_p", PRIMAG_EDGES), ("k", K_EDGES)),
    "prisize_x_k": (("Re_input_p", PRISIZE_EDGES), ("k", K_EDGES)),
}
DEFAULT_SELF_GRIDS = "primag_x_prisize,primag_x_k,prisize_x_k"


def grid_ids(frame, spec):
    """(cell index per pair, n_cells) for a grid spec of (column, edges) pairs."""
    idx = np.zeros(len(frame), dtype=np.int64)
    size = 1
    for col, edges in spec:
        nb = len(edges) - 1
        v = np.asarray(frame[col], dtype=np.float64)
        b = np.clip(np.digitize(v, edges) - 1, 0, nb - 1)
        idx = idx * nb + b
        size *= nb
    return idx, size


def stratum_ids(dist, mag_s, sep_edges=SEP_EDGES, mag_edges=MAG_EDGES):
    """(stratum index per pair, n_strata) on the separation x neighbour-magnitude grid."""
    si = np.clip(np.digitize(dist, sep_edges) - 1, 0, len(sep_edges) - 2)
    mi = np.clip(np.digitize(mag_s, mag_edges) - 1, 0, len(mag_edges) - 2)
    nm = len(mag_edges) - 1
    return (si * nm + mi).astype(np.int64), (len(sep_edges) - 1) * nm


def stratum_targets(sid, truth, mask, n_strat, min_count=2000):
    """Per-stratum label mean and its sem, from `mask` rows only.

    These are the anchors the penalty pulls the model onto. They are ESTIMATES, not external
    constants: computed from the training split in this run and reported in this run's output, which
    is the line AGENTS.md draws (derived-and-reported here = fine; pasted in from elsewhere = not).
    Strata with fewer than `min_count` pairs get weight zero rather than a noisy target -- the label
    scatter is std ~3.9, so a thin stratum's mean is meaningless and would inject pure noise into the
    gradient.
    """
    t = truth[mask].astype(np.float64)
    s = sid[mask]
    cnt = np.bincount(s, minlength=n_strat).astype(np.float64)
    tot = np.bincount(s, weights=t, minlength=n_strat)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tot / cnt
        # TWO-PASS variance. The one-pass `E[x^2] - E[x]^2` cancels catastrophically here: the label
        # mean (~0.02) is ~200x smaller than its scatter (~3.9), so the two terms agree to many
        # digits and their difference is mostly rounding. On a near-constant stratum that can return
        # a tiny positive variance, hence a tiny sem, hence a chi2 weight of ~1/sem^2 that swamps
        # every other stratum. Subtracting the mean first is exact in the degenerate case.
        var = np.bincount(s, weights=(t - mean[s]) ** 2, minlength=n_strat) / cnt
        sem = np.sqrt(var / cnt)
    valid = (cnt >= min_count) & np.isfinite(mean) & (sem > 0)
    mean = np.where(valid, mean, 0.0)
    sem = np.where(valid, sem, 1.0)
    return mean, sem, valid, cnt


def collapse_to_primaries(pid, label, pred, *carry):
    """One row per PRIMARY: its self label, its mean prediction, and the carried features.

    The self response is a property of the primary, not of a pair, so the primary is the sampling
    unit and errors must be formed over primaries. Treating the k rows as k independent measurements
    would shrink the error bar by ~sqrt(k) against labels that are literally identical -- the same
    class of mistake as forming a ratio from ensemble means (AGENTS.md), and it would make a biased
    model look decisively wrong or decisively right on pure bookkeeping.

    A primary's k rows give k slightly different predictions, because `R_self` is read off a context
    that names one neighbour. Averaging them is the natural pair-model estimator; a scene model that
    saw all k at once would not need the average. That is a limitation of this step, not of the idea.
    """
    _, inv = np.unique(pid, return_inverse=True)
    n = inv.max() + 1 if len(inv) else 0
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    lab = np.bincount(inv, weights=label.astype(np.float64), minlength=n) / cnt
    prd = np.bincount(inv, weights=pred.astype(np.float64), minlength=n) / cnt
    got = [np.bincount(inv, weights=np.asarray(c, dtype=np.float64), minlength=n) / cnt
           for c in carry]
    return lab, prd, cnt, got


def bias_table(label, keyname, key, edges, truth, pred, extra=""):
    """Per-bin <truth> vs <model>, errors from the LABEL's own sem over the rows given."""
    print(f"\n[{label}]{extra}")
    print(f"  {keyname:>16}{'N':>12}{'truth':>10}{'sem':>9}{'model':>10}{'mdl/tr-1%':>11}"
          f"{'sigma':>8}")
    chi2, dof = 0.0, 0
    rows = [("ALL", np.ones(len(truth), bool))]
    for i in range(len(edges) - 1):
        rows.append((f"[{edges[i]:g},{edges[i+1]:g})",
                     (key >= edges[i]) & (key < edges[i + 1])))
    for name, m in rows:
        n = int(m.sum())
        if n < 200:
            continue
        a = float(truth[m].mean())
        s = float(truth[m].std(ddof=1) / np.sqrt(n))
        b = float(pred[m].mean())
        rel = (b / a - 1) * 100 if a else np.nan
        z = (b - a) / s if s > 0 else np.nan
        print(f"  {name:>16}{n:>12,}{a:>10.5f}{s:>9.5f}{b:>10.5f}{rel:>+11.2f}{z:>+8.1f}")
        if name != "ALL" and np.isfinite(z):
            chi2 += z * z
            dof += 1
    c2d = chi2 / max(dof, 1)
    print(f"  -> chi2/dof over the {dof} bins = {c2d:.2f}")
    return c2d


def self_stratum_targets(sid, pid, truth, mask, n_strat, min_count=500):
    """Per-cell SELF anchors, formed over PRIMARIES rather than rows.

    Every self grid keys on primary properties only, so a primary's k rows share one cell and
    collapsing is exact. Doing it over rows instead would (a) over-weight crowded primaries -- a 3.1%
    effect measured on this pair set -- and (b) divide the error bar by ~sqrt(k) against labels that
    are literally identical, which would make the chi2 weights wrong by ~2x. Both are silent.
    """
    idx = np.where(mask)[0]
    p = pid[idx]
    first = np.unique(p, return_index=True)[1]      # one row per primary; label and cell are constant
    s = sid[idx][first]
    t = truth[idx][first].astype(np.float64)
    cnt = np.bincount(s, minlength=n_strat).astype(np.float64)
    tot = np.bincount(s, weights=t, minlength=n_strat)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tot / cnt
        var = np.bincount(s, weights=(t - mean[s]) ** 2, minlength=n_strat) / cnt
        sem = np.sqrt(var / cnt)
    valid = (cnt >= min_count) & np.isfinite(mean) & (sem > 0)
    return np.where(valid, mean, 0.0), np.where(valid, sem, 1.0), valid, cnt


def sep_table(label, dist, truth, model_r, edges=SEP_EDGES, emu=None):
    """Per-separation-bin agreement. Errors are the LABEL's sem -- that is the measurement's own
    precision and the floor on what any model can be shown to achieve."""
    print(f"\n[{label}]")
    sep_hdr = 'sep[arcsec]'
    head = (f"  {sep_hdr:>12}{'N':>12}{'truth':>10}{'sem':>9}{'R_model':>10}{'mdl/tr-1%':>11}"
            f"{'+-%':>8}")
    if emu is not None:
        head += f"{'emu':>10}{'emu/tr-1%':>11}"
    print(head)

    def row(name, m):
        n = int(m.sum())
        if n < 200:
            return
        a = float(truth[m].mean())
        s = float(truth[m].std(ddof=1) / np.sqrt(n))
        b = float(model_r[m].mean())
        rel = (b / a - 1) * 100 if a else np.nan
        err = abs(b / a) * (s / abs(a)) * 100 if a else np.nan
        line = f"  {name:>12}{n:>12,}{a:>10.5f}{s:>9.5f}{b:>10.5f}{rel:>+11.2f}{err:>8.2f}"
        if emu is not None:
            e = float(emu[m].mean())
            line += f"{e:>10.5f}{(e/a-1)*100 if a else np.nan:>+11.2f}"
        print(line)

    row("ALL", np.ones(len(truth), bool))
    for i in range(len(edges) - 1):
        row(f"[{edges[i]:.1f},{edges[i+1]:.1f})", (dist >= edges[i]) & (dist < edges[i + 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--val-case-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--mean-hidden", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-flows", type=int, default=6)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--nll-weight", type=float, default=1.0)
    ap.add_argument("--response-weight", type=float, default=10.0)
    ap.add_argument("--strata-weight", type=float, default=0.0,
                    help="weight of the stratum-mean penalty (sep x neighbour-mag grid). 0 keeps the "
                         "pre-2026-08-02g behaviour. This is what targets the CONDITIONAL bias: the "
                         "per-pair MSE is dominated by label noise (std ~3.9 against a signal ~0.02) "
                         "and is nearly indifferent to a systematic offset within a stratum, whereas "
                         "a stratum mean over ~1M pairs has sem ~0.004 and pins it.")
    ap.add_argument("--grids", default=DEFAULT_GRIDS,
                    help="comma-separated grid names to anchor on: " + ", ".join(GRIDS) +
                         ". Each is penalised separately and the chi2/dof averaged, so adding one "
                         "constrains a new axis without thinning the others' anchors.")
    ap.add_argument("--select-on", default="strata", choices=["sep", "strata"],
                    help="held-out criterion for the checkpoint. `sep` is separation bins only -- "
                         "structurally blind to the faint-neighbour bias, which is why the defect "
                         "survived. `strata` is the 2-D grid and is the default from 2026-08-02g.")
    ap.add_argument("--self-response-weight", type=float, default=0.0,
                    help="weight of the SELF-response term (DUAL MODE, 2026-08-02i). 0 = the "
                         "blend-only model, unchanged. > 0 additionally supervises the primary's "
                         "own shear response against `self_truth`, and blinds the residual flow to "
                         "the primary's shape so that response cannot be absorbed by the density "
                         "term. Requires a pair set built with the self label.")
    ap.add_argument("--no-self-weight-by-k", action="store_true",
                    help="do NOT weight the self term by 1/k. The self label is a per-PRIMARY "
                         "quantity repeated across the primary's k rows, so unweighted rows "
                         "over-weight crowded primaries -- and since crowding suppresses the self "
                         "response, that is a bias and not merely an inefficiency. Provided only so "
                         "the size of the effect can be measured.")
    ap.add_argument("--self-strata-weight", type=float, default=0.0,
                    help="weight of the SELF stratum-mean penalty over primary mag / size / crowding "
                         "(2026-08-02j). Targets the measured flatness: at self weight 1000 the model "
                         "spanned 0.94-0.52 in primary magnitude against a truth spanning 1.18-0.28. "
                         "Same remedy that fixed the identical symptom on the blend channel.")
    ap.add_argument("--self-grids", default=DEFAULT_SELF_GRIDS,
                    help="comma-separated SELF grids: " + ", ".join(SELF_GRIDS))
    ap.add_argument("--crowding", action="store_true",
                    help="add the crowding block (nbr_flux_near/far/max, log_k) to the context "
                         "(2026-08-02j). Without it the model has NO input that counts neighbours "
                         "and gets the crowding trend on the self response backwards -- truth falls "
                         "15% from k=1 to k=6-9, the model rises 1%. Anchoring alone cannot fix that: "
                         "it would force the marginal to match with nothing to condition on.")
    ap.add_argument("--no-derived", action="store_true")
    ap.add_argument("--seed", type=int, default=501)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    derived = not args.no_derived
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  seed={args.seed}  derived_features={derived}", flush=True)

    dual = args.self_response_weight > 0
    if args.crowding and args.max_rows:
        raise SystemExit(
            "REFUSING: --crowding with --max-rows. The shell fluxes are SUMS over a primary's "
            "neighbours, so a subsampled frame would drop some of them and report a systematically "
            "under-crowded galaxy -- wrong in a way nothing downstream could detect. Run on the full "
            "pair set, or drop --crowding.")
    if args.self_strata_weight > 0 and not dual:
        raise SystemExit(
            "REFUSING: --self-strata-weight without --self-response-weight. The self channel would be "
            "anchored while never being supervised, which is not a configuration anyone means.")
    t0 = time.time()
    need = list(dict.fromkeys(RAW + TARGETS + ["blend_truth", "blend_null", "case", "pid"]))
    if dual:
        from pyarrow import ipc as _ipc
        have = set(_ipc.open_file(args.pairset).schema.names)
        missing = [c for c in ("self_truth", "k") if c not in have]
        if missing:
            raise SystemExit(
                f"REFUSING: --self-response-weight needs {missing} in the pair set, and "
                f"{args.pairset} does not have them. Rebuild with the current "
                "`scripts/build_blend_pairset.py` (jobs/job_blend_pairset_dual.sh). Falling back to "
                "a blend-only run here would silently produce a model that is NOT what was asked "
                "for, so this stops instead.")
        need += ["self_truth", "k"]
        if "self_null" in have:
            need.append("self_null")
        need = list(dict.fromkeys(need))
    df = pd.read_feather(args.pairset, columns=need)
    print(f"pairset: {len(df):,} pairs, {df['case'].nunique()} cases  ({time.time()-t0:.0f}s)")
    if args.max_rows and args.max_rows < len(df):
        keep = rng.choice(len(df), args.max_rows, replace=False)
        df = df.iloc[np.sort(keep)].reset_index(drop=True)
        print(f"  subsampled to {len(df):,} pairs")

    cases = np.sort(df["case"].unique())
    n_val = max(1, int(round(args.val_case_frac * len(cases))))
    val_cases = set(cases[-n_val:].tolist())
    is_val = df["case"].isin(val_cases).to_numpy()
    print(f"held-out cases: {sorted(val_cases)}  -> {int(is_val.sum()):,} val pairs, "
          f"{int((~is_val).sum()):,} train pairs")

    X = build_features(df, derived=derived, crowding=args.crowding)
    names = feature_names(derived, args.crowding)
    sidx = shape_column_indices(derived, "s", args.crowding)
    std = Standardizer.fit(X[~is_val])
    Xs = std.transform(X)

    # standardize the shifted shape columns with the SAME (mean, scale) as their unshifted twins,
    # so the finite difference is taken in one consistent coordinate system
    def standardize_shift(which, idx):
        out = {}
        for key, (a, b) in shifted_shape_columns(df, args.delta, which).items():
            out[key] = np.column_stack([
                (a - std.mean[idx[0]]) / std.scale[idx[0]],
                (b - std.mean[idx[1]]) / std.scale[idx[1]]]).astype(np.float32)
        return out

    sh_std = standardize_shift("s", sidx)
    pidx = shape_column_indices(derived, "p", args.crowding)
    sh_std_p = standardize_shift("p", pidx) if dual else None

    Y = df[TARGETS].to_numpy(np.float64)
    ymean, yscale = Y[~is_val].mean(0), Y[~is_val].std(0)
    Ys = ((Y - ymean) / yscale).astype(np.float32)
    truth = df["blend_truth"].to_numpy(np.float32)
    nullv = df["blend_null"].to_numpy(np.float32)
    dist = df["distance"].to_numpy(np.float64)
    if dual:
        selft = df["self_truth"].to_numpy(np.float32)
        kcnt = df["k"].to_numpy(np.float64)
        selfw = np.ones_like(kcnt) if args.no_self_weight_by_k else 1.0 / np.maximum(kcnt, 1.0)
        # normalise the weights to mean 1 so `--self-response-weight` means the same thing whether
        # or not the 1/k weighting is on, instead of silently rescaling with <1/k>
        selfw = (selfw / selfw.mean()).astype(np.float32)
    del X, Y

    print(f"\nfeatures ({len(names)}): {', '.join(names)}")
    print(f"neighbour-shape columns (the response channel): {sidx} -> "
          f"{[names[i] for i in sidx]}")
    print(f"target scales (measured ngmix g1,g2 std): {yscale[0]:.4f}, {yscale[1]:.4f}")
    # The REALISED domain, read off the data. Stored in the checkpoint so the lookup builder can
    # flag extrapolation instead of trusting a constant retyped somewhere else.
    dom_mag = float(df["r_input_p"].max())
    dom_re = float(df["Re_input_p"].min())
    dom_nmag = float(df["r_input_s"].max())
    dom_nre = float(df["Re_input_s"].min())
    print(f"realised training domain -- PRIMARY: mag <= {dom_mag:.3f}, Re >= {dom_re:.3f}; "
          f"NEIGHBOUR: mag <= {dom_nmag:.3f}, Re >= {dom_nre:.4f} (full population, as intended)")
    print(f"label: mean {truth.mean():+.5f}  std {truth.std():.4f}")
    print(f"null : mean {nullv.mean():+.5f}  std {nullv.std():.4f}   "
          f"(must be consistent with zero; it is the same projection rotated 45 deg)")

    var_truth = float(truth.var())
    var_self = float(selft.var()) if dual else 1.0
    if dual:
        w64 = selfw.astype(np.float64)
        print(f"\nDUAL MODE: self-response weight {args.self_response_weight}, "
              f"1/k row weighting {'OFF' if args.no_self_weight_by_k else 'ON'}")
        print(f"  primary-shape columns (the self channel): {pidx} -> {[names[i] for i in pidx]}")
        print(f"  self label: row mean {selft.mean():+.5f}  1/k-weighted mean "
              f"{np.average(selft.astype(np.float64), weights=w64):+.5f}  std {selft.std():.4f}")
        if "self_null" in df.columns:
            sn = df["self_null"].to_numpy(np.float64)
            print(f"  self null : mean {sn.mean():+.5f}  std {sn.std():.4f} "
                  "(must be consistent with zero, like the blend null)")
        print(f"  the two responses differ by a factor ~{abs(selft.mean()/max(truth.mean(),1e-9)):.1f}; "
              "the per-pair label scatter is the same for both, so they are NOT interchangeable and")
        print("  carry one weight each.")

    # ---- stratum anchors, one set per GRID, fitted on TRAIN rows only ----
    mag_s = df["r_input_s"].to_numpy(np.float64)
    grid_names = [g.strip() for g in args.grids.split(",") if g.strip()]
    bad = [g for g in grid_names if g not in GRIDS]
    if bad:
        raise SystemExit(f"REFUSING: unknown grid(s) {bad}; available: {list(GRIDS)}")
    grids = []
    print(f"\nanchor grids: {', '.join(grid_names)}")
    for gname in grid_names:
        gsid, gn = grid_ids(df, GRIDS[gname])
        gm, gs, gv, gc = stratum_targets(gsid, truth, ~is_val, gn)
        vm, vs, vv, _ = stratum_targets(gsid, truth, is_val, gn)
        grids.append(dict(name=gname, sid=gsid, n=gn, mean=gm, sem=gs, valid=gv,
                          vmean=vm, vsem=vs, vvalid=vv))
        sig = np.abs(gm[gv]) / gs[gv] if gv.any() else np.array([0.0])
        print(f"  {gname:<16} {int(gv.sum()):>3} of {gn:>3} cells usable, "
              f"anchor S/N median {np.median(sig):.1f} range {sig.min():.1f}-{sig.max():.1f}, "
              f"median cell {int(np.median(gc[gv])) if gv.any() else 0:,} train pairs")
    if args.strata_weight > 0:
        print(f"  penalty weight {args.strata_weight} on the MEAN chi2/dof across these grids")
    else:
        print("  stratum penalty OFF (--strata-weight 0)")

    # ---- SELF anchors, over PRIMARIES, fitted on TRAIN rows only ----
    self_grids = []
    if dual:
        pid_all = df["pid"].to_numpy(np.int64)
        sg_names = [g.strip() for g in args.self_grids.split(",") if g.strip()]
        bad = [g for g in sg_names if g not in SELF_GRIDS]
        if bad:
            raise SystemExit(f"REFUSING: unknown self grid(s) {bad}; available: {list(SELF_GRIDS)}")
        print(f"\nself anchor grids (over PRIMARIES): {', '.join(sg_names)}")
        for gname in sg_names:
            gsid, gn = grid_ids(df, SELF_GRIDS[gname])
            gm, gs, gv, gc = self_stratum_targets(gsid, pid_all, selft, ~is_val, gn)
            vm, vs, vv, _ = self_stratum_targets(gsid, pid_all, selft, is_val, gn)
            self_grids.append(dict(name=gname, sid=gsid, n=gn, mean=gm, sem=gs, valid=gv,
                                   vmean=vm, vsem=vs, vvalid=vv))
            sig = np.abs(gm[gv]) / gs[gv] if gv.any() else np.array([0.0])
            print(f"  {gname:<18} {int(gv.sum()):>3} of {gn:>3} cells usable, "
                  f"anchor S/N median {np.median(sig):.1f} range {sig.min():.1f}-{sig.max():.1f}, "
                  f"median cell {int(np.median(gc[gv])) if gv.any() else 0:,} train PRIMARIES")
        if args.self_strata_weight > 0:
            print(f"  self penalty weight {args.self_strata_weight}")
        else:
            print("  self stratum penalty OFF (--self-strata-weight 0)")
    model, cfg = build_model(Xs.shape[1], mean_hidden=args.mean_hidden, hidden_dim=args.hidden_dim,
                             n_layers=args.n_layers, n_flows=args.n_flows, derived=derived,
                             blind_flow_to_primary_shape=dual, crowding=args.crowding)
    model.to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: {cfg}\n  {npar:,} parameters", flush=True)

    # GPU-resident: ~5 GB for the full ap7 set, well inside an A40. Keeps every epoch off the PCIe
    # bus, which matters because this trains for many epochs on a low-SNR label.
    tX = torch.as_tensor(Xs).to(dev)
    tY = torch.as_tensor(Ys).to(dev)
    tT = torch.as_tensor(truth).to(dev)
    tS = {k: torch.as_tensor(v).to(dev) for k, v in sh_std.items()}
    tSP = {k: torch.as_tensor(v).to(dev) for k, v in sh_std_p.items()} if dual else None
    tST = torch.as_tensor(selft).to(dev) if dual else None
    tSW = torch.as_tensor(selfw).to(dev) if dual else None
    tr_idx = torch.as_tensor(np.where(~is_val)[0]).to(dev)
    va_idx = torch.as_tensor(np.where(is_val)[0]).to(dev)
    del Xs, Ys, sh_std, sh_std_p

    for g in grids + self_grids:
        g["tsid"] = torch.as_tensor(g["sid"]).to(dev)
        g["tmean"] = torch.as_tensor(g["mean"].astype(np.float32)).to(dev)
        g["tsem"] = torch.as_tensor(g["sem"].astype(np.float32)).to(dev)
        g["tvalid"] = torch.as_tensor(g["valid"]).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    i0, i1 = sidx

    def strata_chi2(r, b, glist=None, w=None, min_in_batch=64):
        """Mean over grids of chi2/dof of this batch's per-cell MEAN prediction against the anchors.

        The anchor's sem is the denominator, so a cell contributes only as much as its target is
        actually known -- thin or noisy cells self-attenuate instead of needing a hand-set weight.
        The batch mean of the PREDICTIONS is low-noise (the model's own scatter is far below the
        label's), so a modest per-batch count is enough; the noise that matters lives in the anchor
        and it is fixed before training starts.

        Averaging ACROSS grids rather than concatenating their cells keeps each axis weighted equally
        regardless of how many bins it happens to have.
        """
        terms = []
        for g in (grids if glist is None else glist):
            s = g["tsid"].index_select(0, b)
            # `w` is the per-row weight (1/k for the self channel, so the cell mean is the
            # per-PRIMARY mean in expectation and matches how the anchor was built). The
            # occupancy test stays on the UNWEIGHTED count, which is what "enough rows in this
            # batch" actually means.
            ww = torch.ones_like(r) if w is None else w.index_select(0, b)
            cnt = torch.zeros(g["n"], device=dev, dtype=r.dtype).index_add_(
                0, s, torch.ones_like(r))
            den = cnt if w is None else torch.zeros(
                g["n"], device=dev, dtype=r.dtype).index_add_(0, s, ww)
            tot = torch.zeros(g["n"], device=dev, dtype=r.dtype).index_add_(0, s, ww * r)
            ok = g["tvalid"] & (cnt >= min_in_batch) & (den > 0)
            if not bool(ok.any()):
                continue
            mean = tot[ok] / den[ok]
            terms.append((((mean - g["tmean"][ok]) / g["tsem"][ok]) ** 2).mean())
        if not terms:
            return torch.zeros((), device=dev, dtype=r.dtype)
        return torch.stack(terms).mean()

    p0, p1 = pidx

    def shift_ctx(c0, idx, table, a, bcol):
        """Contexts differing from `c0` only in one channel's two shape columns."""
        out = {}
        for k in ("e1+", "e1-", "e2+", "e2-"):
            c = c0.clone()
            s = table[k].index_select(0, idx)
            c[:, a] = s[:, 0]
            c[:, bcol] = s[:, 1]
            out[k] = c
        return out

    def run(idx, train):
        model.train(train)
        n = idx.numel()
        order = torch.randperm(n, device=dev) if train else torch.arange(n, device=dev)
        tot_nll = tot_res = tot_str = tot_slf = tot_sstr = 0.0
        rs = [] if not train else None
        rsp = [] if not train else None
        for s in range(0, n, args.batch_size):
            b = idx.index_select(0, order[s:s + args.batch_size])
            c0 = tX.index_select(0, b)
            sh = shift_ctx(c0, b, tS, i0, i1)
            with torch.set_grad_enabled(train):
                nll = -model.log_prob(tY.index_select(0, b), c0).mean()
                r = response_from_contexts(model, c0, sh, (yscale[0], yscale[1]), args.delta)
                res = ((r - tT.index_select(0, b)) ** 2).mean() / var_truth
                strat = strata_chi2(r, b) if args.strata_weight > 0 else torch.zeros(
                    (), device=dev, dtype=r.dtype)
                loss = (args.nll_weight * nll + args.response_weight * res
                        + args.strata_weight * strat)
                if dual:
                    # SAME estimator, SAME mean head, PRIMARY's shape columns shifted instead. The
                    # two responses therefore come out in one consistent convention and can be added
                    # -- which is the whole point of merging the flows.
                    shp = shift_ctx(c0, b, tSP, p0, p1)
                    rp = response_from_contexts(model, c0, shp, (yscale[0], yscale[1]), args.delta)
                    wgt = tSW.index_select(0, b)
                    slf = (wgt * (rp - tST.index_select(0, b)) ** 2).mean() / var_self
                    loss = loss + args.self_response_weight * slf
                    if self_grids:
                        sstr = strata_chi2(rp, b, glist=self_grids, w=tSW)
                        if args.self_strata_weight > 0:
                            loss = loss + args.self_strata_weight * sstr
                        tot_sstr += float(sstr.detach()) * b.numel()
                    tot_slf += float(slf.detach()) * b.numel()
                    if rsp is not None:
                        rsp.append(rp.detach().float().cpu().numpy())
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
            w = b.numel()
            tot_nll += float(nll.detach()) * w
            tot_res += float(res.detach()) * w
            tot_str += float(strat.detach()) * w
            if rs is not None:
                rs.append(r.detach().float().cpu().numpy())
        out_r = np.concatenate(rs) if rs is not None else None
        out_p = np.concatenate(rsp) if rsp else None
        return tot_nll / n, tot_res / n, out_r, tot_str / n, tot_slf / n, out_p, tot_sstr / n

    va_np = va_idx.cpu().numpy()
    # Held-out anchors, from the VAL rows -- never the training anchors. Selecting a checkpoint
    # against the same numbers the loss is pulling toward would measure how hard the penalty pulled,
    # not whether the model generalises.
    for g in grids:
        g["sid_val"] = g["sid"][va_np]
    best, best_ep, bad, best_state = np.inf, -1, 0, None
    # Per-primary bookkeeping for the self channel's held-out criterion, built once.
    if dual:
        pid_val = df["pid"].to_numpy(np.int64)[va_np]
        selft_val = selft[va_np]
        primag_val = df["r_input_p"].to_numpy(np.float64)[va_np]
        prisize_val = df["Re_input_p"].to_numpy(np.float64)[va_np]
        # Cell of each held-out PRIMARY, per self grid. Constant within a primary (every self grid
        # keys on primary properties), so the first row of each primary carries it.
        _first = np.unique(pid_val, return_index=True)[1]
        for g in self_grids:
            g["sid_prim_val"] = g["sid"][va_np][_first]
    hdr = (f"\n{'epoch':>6}{'train nll':>12}{'train res':>12}{'train str':>12}{'val nll':>12}"
           f"{'val res':>11}{'sep c2/dof':>12}{'strata c2/dof':>15}")
    if dual:
        # `train sstr` is the SELF stratum term's raw value. Printed even when its weight is 0,
        # because that is the number a later sweep must be sized from -- an earlier sweep was scaled
        # by guesswork and spent all three of its tasks on the flat end of the range.
        hdr += f"{'train sstr':>12}{'self c2/dof':>13}"
    print(hdr + f"{'sec':>7}", flush=True)
    for ep in range(1, args.epochs + 1):
        te = time.time()
        trn, trr, _, trs, _, _, trss = run(tr_idx, True)
        with torch.no_grad():
            vn, vr, vrm, _, _, vrp, _ = run(va_idx, False)
        # The honest quality metric: does the model reproduce the LABEL MEAN per separation bin,
        # measured against the label's own sem? The raw losses are noise-dominated and flat.
        chi2, dof = 0.0, 0
        for i in range(len(SEP_EDGES) - 1):
            m = (dist[va_np] >= SEP_EDGES[i]) & (dist[va_np] < SEP_EDGES[i + 1])
            if m.sum() < 200:
                continue
            a = truth[va_np][m]
            sem = a.std(ddof=1) / np.sqrt(m.sum())
            chi2 += ((vrm[m].mean() - a.mean()) / sem) ** 2
            dof += 1
        c2d = chi2 / max(dof, 1)
        # The 2-D criterion: the same test, but on the separation x neighbour-magnitude grid, so a
        # bias that cancels within a separation bin can no longer hide.
        per_grid = []
        for g in grids:
            gc, gd = 0.0, 0
            for k in range(g["n"]):
                if not g["vvalid"][k]:
                    continue
                m = g["sid_val"] == k
                if m.sum() < 200:
                    continue
                gc += ((vrm[m].mean() - g["vmean"][k]) / g["vsem"][k]) ** 2
                gd += 1
            if gd:
                per_grid.append(gc / gd)
        s2d = float(np.mean(per_grid)) if per_grid else np.nan
        blend_crit = s2d if args.select_on == "strata" else c2d
        # DUAL: the criterion must see BOTH channels, or the checkpoint can be chosen on a blend
        # response that is right while the self response the same network produces is broken. The
        # self side is measured PER PRIMARY over the primary's own magnitude and size bins.
        self2d = np.nan
        if dual and vrp is not None:
            lab, prd, _, _ = collapse_to_primaries(pid_val, selft_val, vrp)
            # Held-out chi2/dof per self grid, against VAL anchors, averaged across grids -- the same
            # construction the blend side uses, so the two halves of the criterion are commensurate.
            # Anchors come from the val split, never the train split: selecting against the numbers
            # the penalty is pulling toward would measure the pull, not generalisation.
            per_self = []
            for g in self_grids:
                sp = g["sid_prim_val"]
                cc, dd = 0.0, 0
                for kk in range(g["n"]):
                    if not g["vvalid"][kk]:
                        continue
                    m = sp == kk
                    if m.sum() < 200:
                        continue
                    cc += ((prd[m].mean() - g["vmean"][kk]) / g["vsem"][kk]) ** 2
                    dd += 1
                if dd:
                    per_self.append(cc / dd)
            self2d = float(np.mean(per_self)) if per_self else np.nan
        # Equal weight to the two channels. Not obviously the right trade -- it is a choice, made
        # explicit here rather than buried in a weight -- but selecting on either alone provably
        # ignores the other, and this run exists to find out whether both can be held at once.
        #
        # A NON-FINITE component never satisfies `crit < best`, so a nan would let the loop run to
        # the end without ever saving, and the closing tables would then describe the final epoch
        # while announcing that they describe the checkpoint. That is silent, and dual mode gives it
        # two chances instead of one. Non-finite parts are dropped with a warning; if none survive
        # the run stops, because "selected on nothing" is not a checkpoint anyone should use.
        parts = [x for x in ([blend_crit] + ([self2d] if dual else [])) if np.isfinite(x)]
        if len(parts) < (2 if dual else 1) and ep == 1:
            print(f"  WARNING: criterion components dropped as non-finite "
                  f"(blend {blend_crit}, self {self2d}). Selecting on what remains.", flush=True)
        if not parts:
            raise SystemExit(
                "REFUSING: every held-out criterion component is non-finite at epoch "
                f"{ep} (blend {blend_crit}, self {self2d}). No checkpoint could be selected, so "
                "this run would produce either nothing or a model chosen at random. Usual cause: "
                "too few held-out rows per anchor cell -- check the 'cells usable' line above.")
        crit = float(np.mean(parts))
        line =(f"{ep:>6}{trn:>12.5f}{trr:>12.6f}{trs:>12.4f}{vn:>12.5f}{vr:>11.6f}"
                f"{c2d:>12.2f}{s2d:>15.2f}")
        if dual:
            line += f"{trss:>12.2f}{self2d:>13.2f}"
        print(line + f"{time.time()-te:>7.0f}", flush=True)
        if crit < best - 1e-4:
            best, best_ep, bad = crit, ep, 0
            meta = dict(feature_names=names, derived=derived, delta=args.delta,
                        shape_indices=sidx, seed=args.seed, epoch=ep, val_chi2_dof=float(c2d),
                        val_chi2_dof_strata=float(s2d), select_on=args.select_on,
                        strata_weight=float(args.strata_weight), grids=grid_names,
                        dual=bool(dual), self_response_weight=float(args.self_response_weight),
                        self_weight_by_k=not args.no_self_weight_by_k,
                        crowding=bool(args.crowding),
                        self_strata_weight=float(args.self_strata_weight),
                        self_grids=[g["name"] for g in self_grids],
                        val_chi2_dof_self=float(self2d),
                        # Which shape columns the residual flow is blind to. In dual mode BOTH
                        # channels are blinded; a reader of the checkpoint should not have to infer
                        # that from the weights.
                        flow_drop_indices=list(cfg.get("flow_drop_indices", [])),
                        val_cases=sorted(int(c) for c in val_cases),
                        pairset=os.path.abspath(args.pairset),
                        target_names=TARGETS, raw_columns=RAW, shape_columns=list(SHAPE_S),
                        # The REALISED training domain on the primary, read off the data rather
                        # than retyped. `build_blend_lookup_flow.py` reads these to flag
                        # out-of-domain rows: a network extrapolates confidently instead of
                        # declining, which is the silent-failure mode AGENTS.md records for the
                        # in-domain emulator (43.4% coverage -> spurious +28.9% m).
                        primary_mag_max=float(dom_mag), primary_re_min=float(dom_re),
                        neighbour_mag_max=float(dom_nmag), neighbour_re_min=float(dom_nre))
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            save_blend_flow(args.output, model, cfg, std, ymean, yscale, meta)
            # Keep the SAVED weights so the final tables describe the checkpoint on disk. Until
            # 2026-08-02h they described whatever state the loop happened to end in, which is a
            # different model -- the control's in-training faint/bright ratios and its
            # `eval_blend_flow` numbers disagreed for exactly this reason.
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop (no held-out improvement for {bad} epochs)")
                break

    print(f"\nbest held-out chi2/dof = {best:.2f} at epoch {best_ep}  -> {args.output}")
    if best_state is None:
        raise SystemExit(
            "REFUSING: no epoch ever improved the held-out criterion, so nothing was written to "
            f"{args.output}. Printing tables now would describe the final epoch's weights while "
            "implying they are the checkpoint -- the exact confusion the 2026-08-02h fix removed. "
            "Investigate the criterion column above before rerunning.")
    model.load_state_dict(best_state)
    print(f"  (reloaded the saved epoch-{best_ep} weights; the tables below describe the")
    print("   CHECKPOINT ON DISK, not the final epoch)")
    with torch.no_grad():
        _, _, vrm, _, _, vrp, _ = run(va_idx, False)
    sep_table("HELD-OUT CASES -- per-pair blend response by separation (saved checkpoint)",
              dist[va_np], truth[va_np], vrm)

    if dual and vrp is not None:
        lab, prd, kc, (pm, ps) = collapse_to_primaries(
            pid_val, selft_val, vrp, primag_val, prisize_val)
        print(f"\n{'='*100}\nSELF RESPONSE -- held-out, ONE ROW PER PRIMARY ({len(lab):,} primaries "
              f"from {len(vrp):,} pairs)\n{'='*100}")
        print("This is the channel flow #1 owns. The question this run asks is whether the SAME")
        print("network can carry it while carrying the blend response, so read the two tables above")
        print("and below together: a model that fixes one by breaking the other has not merged them.")
        a = bias_table("SELF vs primary magnitude", "r_input_p", pm, PRIMAG_EDGES, lab, prd)
        b = bias_table("SELF vs primary size", "Re_input_p", ps, PRISIZE_EDGES, lab, prd)
        c = bias_table("SELF vs neighbour count", "k", kc, [1, 2, 3, 4, 6, 9, 99], lab, prd,
                       extra="   <- crowding suppresses the self response; this is the axis a "
                             "pair model is least equipped for")
        print(f"\n  self chi2/dof: mag {a:.2f}   size {b:.2f}   k {c:.2f}")
        print("  The k table is the one to watch. A pair model reads R_self from a context naming")
        print("  ONE neighbour, so it can only learn crowding through whatever the named neighbour")
        print("  correlates with. If it fails anywhere it should fail here -- and that failure is")
        print("  the argument for the scene-level step, not a reason to abandon the merge.")

    # THE DEFECT THIS RUN EXISTS TO FIX, reported directly on held-out rows. The emulator's own
    # neighbour cut splits the population into the group it scores and the 81% it zeroes; the flow
    # was 2.5x over on the latter and 11% under on the former (WORKLOG 2026-08-02e).
    mg, rg = mag_s[va_np], df["Re_input_s"].to_numpy(np.float64)[va_np]
    kept = (mg >= 18.0) & (mg < 26.0) & (rg >= 0.3) & (rg <= 1.5)
    print("\n[HELD-OUT -- the faint-pair defect, split by the emulator's own neighbour cut]")
    print(f"  {'group':<28}{'pairs':>12}{'<truth>':>11}{'sem':>9}{'<model>':>11}{'ratio':>9}")
    for nm, m in (("KEPT by the emulator", kept), ("DROPPED by the emulator", ~kept),
                  ("all held-out pairs", np.ones(len(mg), bool))):
        if m.sum() < 100:
            continue
        a = truth[va_np][m].astype(np.float64)
        sem = a.std(ddof=1) / np.sqrt(m.sum())
        mm = float(vrm[m].mean())
        print(f"  {nm:<28}{int(m.sum()):>12,}{a.mean():>11.4f}{sem:>9.4f}{mm:>11.4f}"
              f"{mm/max(abs(a.mean()),1e-12):>9.2f}")
    print("  A ratio near 1.00 on BOTH of the first two rows is the objective. Before this change")
    print("  the DROPPED row read ~2.5 and the KEPT row ~0.89 -- opposite signs, which is why the")
    print("  separation-only criterion never flagged it.")
    print("\n  Reference, BlendEMU 'lsst_r_extnbr_indom_tuned' on the SAME quantity (WORKLOG")
    print("  2026-08-02c, whole ap7 sample): -36.00 below 0.5, -38.61 at 0.5-1, -8.95 at 1-1.5,")
    print("  +7.43 at 1.5-2, +2.32 at 2-3, -2.74 at 3-4, -5.93 at 4-5 percent (separations in arcsec).")
    print("  Those are NOT held-out numbers for the emulator (it never saw these labels at all),")
    print("  so the comparison is model-on-held-out vs emulator-on-everything; scoring both on the")
    print("  same held-out rows is `scripts/eval_blend_flow.py`, which is the number to quote.")
    print(f"\nTRAIN_BLEND_FLOW_DONE  ({time.time()-t0:.0f}s)", flush=True)
    print(json.dumps({"best_chi2_dof": float(best), "best_epoch": int(best_ep)}))


if __name__ == "__main__":
    main()
