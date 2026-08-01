#!/usr/bin/env python -B
"""Does spending the node budget on SHAPE RESOLUTION instead of scene diversity fix §5C?

cont.167 settled the diagnosis: the estimator is arithmetically correct, and the failure is
that a finite K-component mixture is not the true marginal.  Three ways of buying a better
bank have now failed -- bigger (K 2000->20000 moves the ratio 17.1->16.9), self- (each
galaxy's OWN true scene in the bank, `closure_v2_lagrangian --self-bank`, WORKLOG cont.161),
and localised (per-galaxy proposal, `diag5c_localprop`, jobs 15418898/15420202).

This tests the one structurally different option left.  Under `primary_only=True` shear,
`S_g` moves EXACTLY TWO of the ~18 scene coordinates: the primary's intrinsic (e1, e2).
`d/dg log p_hat` is therefore a directional derivative along the shape axes alone.  If the
mixture's lumpiness ALONG THOSE AXES is what the derivative is picking up, then covering
them systematically -- at the cost of covering the other 16 more sparsely -- should move the
information equality toward 1.  If it does not, the failure is not about where the nodes sit
in the shear directions, and §5C in this form is finished.

DESIGN.  A budget ladder at FIXED total bank size K, so every rung costs the same:

    arm A (baseline)   K scenes drawn from the catalogue, shapes as they come
    arm B (M, nr, na)  M base scenes, each expanded onto nr*na shape nodes;  K = M*nr*na

The ladder trades M against nr*na.  A monotone trend in the equality ratio answers the
question either way; a flat one says shape coverage is not the lever.

WHY THE NODES KEEP EQUAL WEIGHT.  The shape nodes are STRATIFIED, not a weighted quadrature:
radii at equal-probability quantiles of the empirical intrinsic |e| distribution, angles
uniform over [0, 2pi) in the (e1, e2) plane (which is the correct uniform measure for a
spin-2 quantity under isotropy).  Every stratum then carries prior mass 1/(nr*na), so all
nodes have equal weight and `posterior_weights(..., log_prior=None)` stays correct.  This
deliberately avoids modelling p(e) as a density, which would be a new place to be wrong.

APPROXIMATION, STATED.  Replacing a base scene's shape by grid points assumes the intrinsic
shape is independent of the rest of the scene, i.e. p(e|r) = p(e).  Real size-shape
correlation is therefore broken in arm B.  Arm A has no such approximation, so a WIN for
arm B is meaningful despite it; a LOSS is confounded with it and must be read as inconclusive
on that axis.  Reported explicitly rather than buried.

Usage:
    python scripts/diag5c_shapegrid.py --n-gal 20000 --k 20000 \
        --splits 200x10x10,1000x5x4,2000x5x2 --gamma 0.05
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.lagrangian_score import (  # noqa: E402
    denominator_consistency,
    posterior_weights,
    score_and_information,
    selection_terms,
    shear_estimate_bartlett,
    shear_estimate_louis,
)
from closure_v2_lagrangian import (  # noqa: E402
    CAT,
    CKPT,
    load_rows,
    phi_block,
    rebuild,
    scene_context,
)
from train_joint_forward import intrinsic_shape  # noqa: E402


def parse_splits(text):
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m, nr, na = (int(v) for v in tok.lower().split("x"))
        out.append((m, nr, na))
    return out


def shape_grid(e_abs_pool, n_base, n_rad, n_ang, rng, jitter=True):
    """Stratified (e1, e2) nodes for `n_base` base scenes, index = g * n_base + m.

    Radii: equal-probability strata of the empirical |e| distribution, so each ring holds
    prior mass 1/n_rad.  Angles: equal sectors of [0, 2pi) in the (e1, e2) plane.  One point
    per (ring, sector) cell => every node carries weight 1/(n_rad*n_ang), i.e. equal weight.
    With `jitter` the point is drawn uniformly inside its cell (stratified sampling,
    unbiased); without it, the cell centre (a deterministic grid).
    """
    n_grid = n_rad * n_ang
    total = n_grid * n_base
    j = np.repeat(np.arange(n_grid) // n_ang, n_base)     # ring index
    l = np.repeat(np.arange(n_grid) % n_ang, n_base)      # sector index
    if jitter:
        ur = rng.random(total)
        ua = rng.random(total)
    else:
        ur = np.full(total, 0.5)
        ua = np.full(total, 0.5)
    q = (j + ur) / n_rad                                   # uniform inside the ring's stratum
    radius = np.quantile(e_abs_pool, np.clip(q, 1e-6, 1 - 1e-6))
    theta = 2.0 * np.pi * (l + ua) / n_ang
    return radius * np.cos(theta), radius * np.sin(theta)


def arm_numbers(model, xhat, node_df, node_intr, pre, nbr_std, dev, deltas, chunk, n_boot, seed):
    """Run the §5C estimator on one node bank; return a dict per stencil width."""
    res = {}
    for d in deltas:
        phis, pdets = {}, {}
        for t in (0.0, +d, -d):
            c, pd_ = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
            phis[t] = phi_block(model, xhat, c, torch.log(pd_), chunk)
            pdets[t] = float(pd_.mean())
            del c, pd_
        p0, pp, pm = phis[0.0], phis[+d], phis[-d]
        d1 = (pp - pm) / (2 * d)
        d2 = (pp - 2 * p0 + pm) / d ** 2
        s, info = score_and_information(p0, d1, d2)       # stratified bank: equal weight, no log_prior
        lp = {t: np.log(v) for t, v in pdets.items()}
        s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                       (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
        bart_den, lou_den, ratio = denominator_consistency(s, info, s_sel, i_sel)
        ess = float(np.mean(1.0 / np.sum(posterior_weights(p0) ** 2, axis=1)))

        rng = np.random.default_rng(seed)
        n = len(s)
        rb, gb = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            sb, ib = s[idx], info[idx]
            den = float(np.mean(ib) - i_sel)
            rb.append(float(np.mean((sb - s_sel) ** 2)) / den if den != 0 else np.nan)
            gb.append(shear_estimate_bartlett(sb, s_sel))
        res[d] = dict(
            ghat59=shear_estimate_bartlett(s, s_sel),
            ghat58=shear_estimate_louis(s, info, s_sel, i_sel),
            ratio=ratio, bart_den=bart_den, lou_den=lou_den, ess=ess,
            s_mean=float(np.mean(s)), s_sel=s_sel, i_sel=i_sel,
            ratio_se=float(np.nanstd(rb)), ghat59_se=float(np.std(gb)),
        )
        del phis, p0, pp, pm, d1, d2
    return res


def report(tag, res, gamma):
    for d, r in res.items():
        m = f"{r['ghat59'] / gamma - 1:+8.2%}" if gamma else "      --"
        print(f"  {tag:<22} d={d:<6.3f} ratio={r['ratio']:+8.3f}+-{r['ratio_se']:.3f}  "
              f"ghat(5.9)={r['ghat59']:+9.5f}+-{r['ghat59_se']:.5f} m={m}  "
              f"ghat(5.8)={r['ghat58']:+9.4f}  ESS={r['ess']:7.1f}  "
              f"<s>={r['s_mean']:+9.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--k", type=int, default=20000, help="total bank size, held FIXED across arms")
    ap.add_argument("--splits", default="200x10x10,1000x5x4,2000x5x2",
                    help="comma list of M x n_rad x n_ang; each must satisfy M*nr*na == --k")
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--deltas", default="0.01")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-jitter", action="store_true",
                    help="deterministic cell centres instead of stratified-random points")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    splits = parse_splits(args.splits)
    for (m, nr, na) in splits:
        if m * nr * na != args.k:
            raise SystemExit(f"split {m}x{nr}x{na} = {m * nr * na} != --k {args.k}; "
                             "the whole point is a FIXED budget")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    print(f"device {dev}   checkpoint metadata: primary_only_shear="
          f"{meta.get('primary_only_shear')}", flush=True)

    # galaxies sit AFTER the node pool so no galaxy is ever its own node
    span = args.k + args.n_gal
    rows = load_rows(args.catalogue, span * 4)
    if len(rows) < span:
        raise SystemExit(f"catalogue gave {len(rows):,} rows, need {span:,}")
    pool = rows.iloc[:args.k].reset_index(drop=True)
    gal_df = rows.iloc[args.k:args.k + args.n_gal].reset_index(drop=True)
    print(f"rows: {len(pool):,} node pool + {len(gal_df):,} galaxies (disjoint)", flush=True)

    def intr_of(df):
        d = {}
        d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
        d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
        return d

    gal_intr = intr_of(gal_df)

    # ---- data drawn FROM the model at the true shear, once, shared by every arm --------
    g_true = (0.0, float(args.gamma))
    ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, g_true, pre, nbr_std, dev)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        xhat = model.mean_flow.sample(ctx_g, n_samples=1)
        if xhat.dim() == 3:
            xhat = xhat[:, 0, :]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
    xhat, n_keep = xhat[keep], int(keep.sum())
    del ctx_g, pdet_g
    print(f"detection: kept {n_keep:,}/{len(gal_df):,} = {n_keep / len(gal_df):.1%}", flush=True)
    if n_keep < 100:
        raise SystemExit("too few detected rows")

    deltas = [float(x) for x in args.deltas.split(",")]
    print(f"\ntrue gamma2 = {args.gamma:+.4f}   bank size K = {args.k:,} for EVERY arm")
    print("  ratio is the information equality Var(s-<s>_sel)/(<I>-I_sel); it must be 1.\n")

    if not args.skip_baseline:
        res = arm_numbers(model, xhat, pool, intr_of(pool), pre, nbr_std, dev,
                          deltas, args.chunk, args.n_boot, args.seed)
        report(f"A baseline K={args.k}", res, args.gamma)

    # empirical intrinsic |e| for the strata, from the node pool
    e1p, e2p = intrinsic_shape(pool, "p")
    e_abs = np.hypot(np.asarray(e1p, float), np.asarray(e2p, float))
    e_abs = e_abs[np.isfinite(e_abs)]
    print(f"\n  intrinsic |e| pool: n={len(e_abs):,} median={np.median(e_abs):.4f} "
          f"p90={np.quantile(e_abs, 0.9):.4f}\n", flush=True)

    rng = np.random.default_rng(args.seed + 1)
    for (m, nr, na) in splits:
        base = pool.iloc[:m].reset_index(drop=True)
        n_grid = nr * na
        frame = pd.concat([base] * n_grid, ignore_index=True)   # index = g*m + row
        ge1, ge2 = shape_grid(e_abs, m, nr, na, rng, jitter=not args.no_jitter)
        b1s, b2s = intrinsic_shape(base, "s")
        intr = {"e1p": ge1, "e2p": ge2,
                "e1s": np.tile(np.asarray(b1s, float), n_grid),
                "e2s": np.tile(np.asarray(b2s, float), n_grid)}
        res = arm_numbers(model, xhat, frame, intr, pre, nbr_std, dev,
                          deltas, args.chunk, args.n_boot, args.seed)
        report(f"B {m}base x {nr}x{na}shape", res, args.gamma)
        del frame, intr

    print("\n  Read the TREND across the ladder, not any single row.  Arm B trades scene")
    print("  diversity for shape resolution at fixed cost.  Ratio moving toward 1 as the")
    print("  shape grid grows => the shear-direction lumpiness is the lever.  Flat => it is")
    print("  not, and no proposal that only reorganises WHERE the nodes sit will fix (5.9).")
    print("  Arm B additionally assumes p(e|rest) = p(e); a LOSS is confounded with that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
