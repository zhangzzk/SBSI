#!/usr/bin/env python -B
"""PROBE A, follow-up: is the flat-in-K bank noise carried by the EXTREME TAIL of phi'?

Probe A's first run (job 15461075) refuted the arithmetic it set out to test and replaced
it with a sharper fact:

    Var_w(phi') ~ K^+0.37,  ESS ~ K^+0.88,  so p-q = -0.51 +- 0.02  (NOT 0)
    predicted noise Var_w/ESS UNDER-predicts the directly measured half-bank noise by 6-10x
    directly measured noise is FLAT in K (exponent +0.20 / -0.06 in two banks)
    unweighted |phi'| p50/p99/p99.9 are STATIONARY in K, but the per-galaxy MAX grows
        9,298 -> 35,530 over K = 1,000 -> 20,000 (~K^0.45), pooled max reaches 1e7

so the delta-method/ESS description of this estimator is simply not valid, and what does
grow with K is the extreme order statistic.  That is the signature of a heavy tail with a
tail index near (or below) the variance-existence boundary alpha = 2.

This script tests that reading three ways, all on the same phi array, gamma = 0:

  1. TAIL INDEX.  Hill estimator on |phi'| per galaxy over its K nodes, at several tail
     fractions, plus the pooled version.  alpha < 2 => phi' has no finite second moment
     under the node distribution and `Var_w(phi')` cannot converge, no matter the bank.

  2. CAUSAL CLIP TEST -- the decisive one.  Winsorise |phi'| at a FIXED per-galaxy
     threshold (the per-galaxy 95 / 99 / 99.9 percentile taken once at the SMALLEST K, so
     the truncation does not move with K) and re-measure the half-bank noise ladder.  If
     the noise then falls like 1/K, the tail is what breaks the averaging; if it stays
     flat, the tail is not the mechanism.  Clipping is a DIAGNOSTIC, not a proposed fix:
     the clipped statistic is no longer the score, so it cannot be used to estimate shear.

  3. SHARED OR PER-GALAXY.  For each galaxy, which node attains max |phi'|?  If a handful
     of nodes win for most galaxies the pathology is a property of those SCENES (fixable
     by bank construction / a support constraint); if the argmax is spread over as many
     nodes as there are galaxies it is a property of the (galaxy, node) PAIR.

Usage:
    python scripts/diag5c_probeA2_tail.py --n-gal 20000 --n-nodes 1000,2000,6000,20000
"""

import argparse
import os
import sys

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.lagrangian_score import posterior_weights  # noqa: E402
from closure_v2_lagrangian import (  # noqa: E402
    CAT, CKPT, load_rows, phi_block, rebuild, scene_context,
)
from train_joint_forward import intrinsic_shape  # noqa: E402


def ols_slope(logk, logy):
    x = logk - logk.mean()
    return float(np.dot(x, logy - logy.mean()) / np.dot(x, x))


def hill(x, frac):
    """Hill tail index of a 1-D positive sample, using the top `frac` of order statistics."""
    x = x[np.isfinite(x) & (x > 0)]
    k = max(2, int(frac * x.size))
    top = np.sort(x)[-(k + 1):]
    xk = top[0]
    return float(k / np.sum(np.log(top[1:] / xk)))


def scores_clipped(phi0, d1, K, thr, row_chunk=384):
    """(s_A, s_B, s_full) with |phi'| winsorised at the per-galaxy threshold `thr`.

    `thr` is `(N_gal, 1)` or None.  Half-banks are the first and second K/2 columns.
    """
    n, half = phi0.shape[0], K // 2
    sA = np.empty(n); sB = np.empty(n); sF = np.empty(n)
    for a in range(0, n, row_chunk):
        b = min(a + row_chunk, n)
        dd = d1[a:b, :K]
        if thr is not None:
            t = thr[a:b]
            dd = np.clip(dd, -t, t)
        wA = posterior_weights(phi0[a:b, :half])
        wB = posterior_weights(phi0[a:b, half:2 * half])
        wF = posterior_weights(phi0[a:b, :K])
        sA[a:b] = np.einsum("ij,ij->i", wA, dd[:, :half])
        sB[a:b] = np.einsum("ij,ij->i", wB, dd[:, half:2 * half])
        sF[a:b] = np.einsum("ij,ij->i", wF, dd)
    return sA, sB, sF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="1000,2000,6000,20000")
    ap.add_argument("--clips", default="99.9,99,95")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--row-chunk", type=int, default=384)
    ap.add_argument("--n-hgal", type=int, default=2000, help="galaxies used for Hill/tail work")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    nodes = sorted(int(x) for x in args.n_nodes.split(","))
    clips = [float(x) for x in args.clips.split(",")]
    kmax, d = max(nodes), args.delta
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    print(f"device {dev}  primary_only_shear={meta.get('primary_only_shear')}", flush=True)

    rows = load_rows(args.catalogue, (kmax + args.n_gal) * 4)
    pool = rows.iloc[:kmax].reset_index(drop=True)
    gal_df = rows.iloc[kmax:kmax + args.n_gal].reset_index(drop=True)

    def intr_of(df):
        e1p, e2p = intrinsic_shape(df, "p")
        e1s, e2s = intrinsic_shape(df, "s")
        return dict(e1p=e1p, e2p=e2p, e1s=e1s, e2s=e2s)

    gal_intr, pool_intr = intr_of(gal_df), intr_of(pool)
    ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, (0.0, 0.0), pre, nbr_std, dev)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        xh = model.mean_flow.sample(ctx_g, n_samples=1)
        if xh.dim() == 3:
            xh = xh[:, 0, :]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
    xhat = xh[keep]
    print(f"  gamma=0: kept {int(keep.sum()):,}/{len(gal_df):,}   "
          f"(bank rows 0:{kmax:,}, same layout as job 15461075 bank 0)", flush=True)
    del ctx_g, pdet_g, xh

    ph = {}
    for t in (0.0, +d, -d):
        c, pdn = scene_context(model, pool, pool_intr, (0.0, t), pre, nbr_std, dev)
        ph[t] = phi_block(model, xhat, c, torch.log(pdn), args.chunk)
        del c, pdn
    phi0 = ph[0.0]
    np.subtract(ph[+d], ph[-d], out=ph[+d])
    ph[+d] /= (2.0 * d)
    d1 = ph[+d]
    del ph[-d]
    n = phi0.shape[0]
    print(f"  phi ready {phi0.shape}\n", flush=True)

    # ---- 1. tail index ---------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    hrows = np.sort(rng.choice(n, size=min(args.n_hgal, n), replace=False))
    print("TAIL INDEX (Hill).  alpha < 2 => no finite second moment for phi'.")
    print(f"{'K':>7} {'alpha(top1%)':>13} {'alpha(top0.5%)':>15} {'alpha(top0.1%)':>15} "
          f"{'alpha_pooled(1%)':>17}")
    for K in nodes:
        sub = np.abs(d1[np.ix_(hrows, np.arange(K))])
        per = []
        for fr in (0.01, 0.005, 0.001):
            if int(fr * K) < 5:
                per.append(np.nan)
                continue
            per.append(float(np.mean([hill(sub[j], fr) for j in range(0, sub.shape[0], 4)])))
        pooled = hill(sub.ravel(), 0.01)
        print(f"{K:>7,} {per[0]:>13.3f} {per[1]:>15.3f} {per[2]:>15.3f} {pooled:>17.3f}",
              flush=True)
        del sub

    # ---- 3. is the extreme node SHARED across galaxies? -------------------------------
    print("\nARGMAX NODE STRUCTURE at K = %d (all galaxies)." % kmax)
    am = np.empty(n, dtype=np.int64)
    for a in range(0, n, args.row_chunk):
        b = min(a + args.row_chunk, n)
        am[a:b] = np.argmax(np.abs(d1[a:b, :kmax]), axis=1)
    cnt = np.bincount(am, minlength=kmax)
    order = np.sort(cnt)[::-1]
    cum = np.cumsum(order) / n
    n50 = int(np.searchsorted(cum, 0.5) + 1)
    print(f"  distinct argmax nodes = {int((cnt > 0).sum()):,} of {kmax:,};  "
          f"top node takes {order[0] / n:.2%} of galaxies;  top-10 {order[:10].sum() / n:.2%};  "
          f"{n50:,} nodes cover 50% of galaxies")
    print(f"  (a per-(galaxy,node) pathology would need ~{n:,} distinct nodes; a shared-scene "
          f"pathology, a handful)", flush=True)

    # ---- 2. causal clip test ---------------------------------------------------------
    kmin = min(nodes)
    thr = {}
    base = np.abs(d1[:, :kmin])
    for c in clips:
        thr[c] = np.percentile(base, c, axis=1, keepdims=True)
        print(f"\nclip {c}%: per-galaxy threshold on |phi'| fixed at K={kmin:,}: "
              f"median {float(np.median(thr[c])):.3f}, "
              f"p90 {float(np.percentile(thr[c], 90)):.3f}")
    del base

    logk = np.log(np.array(nodes, dtype=np.float64))
    print("\nCLIP TEST.  sigma^2_half = <(s_A - s_B)^2>/2 (per-galaxy, assumption-light);")
    print("  Var(s) is the full-bank across-galaxy variance = the 5C denominator.")
    print(f"{'clip':>7} " + " ".join(f"{'K=' + format(K, ','):>22}" for K in nodes)
          + f" {'exp r':>8} {'exp Var(s)':>11}")
    for c in [None] + clips:
        t = None if c is None else thr[c]
        s2, vs, cells = [], [], []
        for K in nodes:
            sA, sB, sF = scores_clipped(phi0, d1, K, t, args.row_chunk)
            s2.append(float(np.mean((sA - sB) ** 2)) / 2.0)
            vs.append(float(np.var(sF)))
            cells.append(f"{s2[-1]:10.3f}/{vs[-1]:10.3f}")
        s2, vs = np.array(s2), np.array(vs)
        lab = "none" if c is None else f"{c}%"
        print(f"{lab:>7} " + " ".join(f"{x:>22}" for x in cells)
              + f" {ols_slope(logk, np.log(s2)):>+8.3f} {ols_slope(logk, np.log(vs)):>+11.3f}",
              flush=True)
    print("\n  cells are sigma^2_half / Var(s).  Pure Monte-Carlo would give exponent -1 on")
    print("  sigma^2_half.  If clipping restores it, the extreme tail of phi' is what stops")
    print("  the bank noise averaging down; if not, the tail is not the mechanism.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
