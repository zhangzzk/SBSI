#!/usr/bin/env python -B
"""VERIFY-V: adversarial check on probe C-flow-tail's two soft spots.

Probe C concluded the flow tail is screened off because the deep-deficit bins carry
~0 WEIGHT (a population MEAN) and ~0% of Var(s) via the attribution Cov(s^(b), s)/Var(s).
Two things that attribution cannot see, and that this script measures directly:

  A1  Cov(s^(b), s) can be ~0 while Var(s^(b)) is LARGE, if the bin's partial score
      anti-correlates with the head bins.  Measure sd(s^(b)) and max_i |s_i^(b)| per bin,
      plus the full correlation matrix of the partial scores, and -- decisively -- the
      COUNTERFACTUAL Var(s) with the tail bins physically deleted from the sum.

  A2  the `weight` column is a mean over galaxies.  Var(s) is dominated by ~1% of
      galaxies (kurtosis 167), so a small mean is compatible with a few galaxies putting
      real weight in the tail.  Measure the per-galaxy DISTRIBUTION of bin weight
      (mean / p99 / max), and cross-tabulate the top-1% |s| galaxies separately.

  A3  free sanity theorem: w_k <= exp(deficit_k), because the denominator of the
      self-normalised weight contains exp(0) = 1 from the argmax node.  Verified
      numerically; if it holds, bins beyond -30 are excluded by ARITHMETIC, not by
      measurement, and no sample/seed can change that.

PART B (optional, --with-stencil): probe C claimed the node-uniform stencil
non-convergence on the FULL sample "is entirely the tail", but demonstrated that only on
a 3,000-galaxy subsample whose aggregate rel behaves 4x differently.  Redo the per-bin
stencil table on all 9,918 galaxies at the 0.005 / 0.0025 pair.

Setup is copied verbatim from scripts/diag5c_tailC.py so the numbers are comparable.
"""

import argparse
import os
import sys
import time

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

EDGES = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
LAB = ["[0,-1]", "(-1,-3]", "(-3,-10]", "(-10,-30]", "(-30,-100]", "< -100"]
NB = len(LAB)


def bin_of(deficit):
    return np.searchsorted(EDGES, -deficit, side="left")


def intr_of(df):
    d = {}
    d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
    d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
    return d


def ctxs_at(model, node_df, node_intr, gammas, pre, nbr_std, dev):
    out = {}
    for t in gammas:
        c, pd_ = scene_context(model, node_df, node_intr, (0.0, float(t)), pre, nbr_std, dev)
        out[t] = (c, torch.log(pd_))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--gal-chunk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--with-stencil", action="store_true")
    args = ap.parse_args()

    K = args.n_nodes
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    print(f"device {dev}  primary_only_shear={meta.get('primary_only_shear')}", flush=True)

    rows = load_rows(args.catalogue, (K + args.n_gal) * 4)
    pool = rows.iloc[:K].reset_index(drop=True)
    gal_df = rows.iloc[K:K + args.n_gal].reset_index(drop=True)
    print(f"rows: {len(pool):,} node pool + {len(gal_df):,} galaxies (disjoint)", flush=True)
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
    n = xhat.shape[0]
    print(f"  gamma=0: kept {n:,}/{len(gal_df):,}   [tailC/repro report 9,918 at n-gal 20000]",
          flush=True)
    del ctx_g, pdet_g, xh

    node_df = pool.iloc[:K].reset_index(drop=True)
    node_intr = {k: v[:K] for k, v in pool_intr.items()}
    gc = args.gal_chunk or max(1, int(2.0e7 // K))
    d = args.delta

    # ---------------- PART A ------------------------------------------------------------
    t0 = time.time()
    ctxs = ctxs_at(model, node_df, node_intr, (0.0, +d, -d), pre, nbr_std, dev)
    s_part = np.zeros((n, NB), dtype=np.float64)
    w_part = np.zeros((n, NB), dtype=np.float64)
    absmax_phi1 = 0.0
    viol = 0
    cnt = np.zeros(NB, dtype=np.int64)

    for s0 in range(0, n, gc):
        e0 = min(s0 + gc, n)
        xb = xhat[s0:e0]
        p0 = phi_block(model, xb, ctxs[0.0][0], ctxs[0.0][1], args.chunk)
        pp = phi_block(model, xb, ctxs[+d][0], ctxs[+d][1], args.chunk)
        pm = phi_block(model, xb, ctxs[-d][0], ctxs[-d][1], args.chunk)
        d1 = (pp - pm) / (2.0 * d)
        del pp, pm
        deficit = p0 - p0.max(axis=1, keepdims=True)
        w = posterior_weights(p0)
        del p0
        b = bin_of(deficit)
        cnt += np.bincount(b.ravel(), minlength=NB)
        # A3: theorem w_k <= exp(deficit_k)
        viol += int(np.sum(w > np.exp(deficit) * (1.0 + 1e-9)))
        absmax_phi1 = max(absmax_phi1, float(np.abs(d1).max()))
        sc = w * d1
        for bb in range(NB):
            m_ = (b == bb)
            if not m_.any():
                continue
            s_part[s0:e0, bb] = np.where(m_, sc, 0.0).sum(axis=1)
            w_part[s0:e0, bb] = np.where(m_, w, 0.0).sum(axis=1)
        del d1, w, deficit, b, sc

    for c, _ in ctxs.values():
        del c
    del ctxs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    s_tot = s_part.sum(axis=1)
    var_s = float(np.var(s_tot))
    sc_c = s_part - s_part.mean(axis=0, keepdims=True)
    st_c = s_tot - s_tot.mean()
    cov_bin = (sc_c * st_c[:, None]).mean(axis=0)

    print(f"\n=== PART A: is the tail's 0.00% a CANCELLATION or a real zero?"
          f"   K={K:,}  N_gal={n:,}  delta={d}")
    print(f"    Var(s) = {var_s:.3f}   [tailC job 15461010/15463970 report 34.750]")
    print(f"    A3 theorem  w_k <= exp(deficit_k):  violations = {viol} / {cnt.sum():,} pairs"
          f"   (0 => bins beyond -30 are excluded by ARITHMETIC, not by sampling)")
    print(f"    max |phi'| over all (i,k) = {absmax_phi1:.4g}"
          f"   (log10 histogram in tailC clips at 1e8 -- check for clipping)")
    print(f"\n    {'deficit bin':>12} {'nodes':>8} {'w mean':>10} {'w p99':>10} {'w MAX':>10} "
          f"{'sd(s^b)':>10} {'max|s^b|':>10} {'Cov share':>10} {'sd share':>9}")
    sd_bin = s_part.std(axis=0)
    for bb in range(NB):
        print(f"    {LAB[bb]:>12} {cnt[bb] / cnt.sum():>8.2%} "
              f"{w_part[:, bb].mean():>10.6f} {np.percentile(w_part[:, bb], 99):>10.6f} "
              f"{w_part[:, bb].max():>10.6f} {sd_bin[bb]:>10.4f} "
              f"{np.abs(s_part[:, bb]).max():>10.4f} "
              f"{cov_bin[bb] / var_s:>10.2%} {sd_bin[bb] / np.sqrt(var_s):>9.2%}")
    print(f"    {'TOTAL':>12} {1.0:>8.2%} {w_part.sum(axis=1).mean():>10.6f} "
          f"{'':>10} {'':>10} {np.sqrt(var_s):>10.4f} {np.abs(s_tot).max():>10.4f} "
          f"{cov_bin.sum() / var_s:>10.2%}")

    print(f"\n    COUNTERFACTUAL -- Var of the partial score with only bins 0..b kept "
          f"(the tail physically deleted):")
    for bb in range(NB):
        cum = s_part[:, :bb + 1].sum(axis=1)
        print(f"      keep bins 0..{bb} ({LAB[bb]:>10} and shallower): "
              f"Var = {float(np.var(cum)):>9.4f}   "
              f"ratio to full = {float(np.var(cum)) / var_s:>7.4f}   "
              f"corr with full s = {float(np.corrcoef(cum, s_tot)[0, 1]):>8.5f}")

    print(f"\n    correlation matrix of the per-bin partial scores s^(b) "
          f"(anti-correlation is what a hidden cancellation would look like):")
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(s_part.T)
    print(f"      {'':>12} " + "  ".join(f"{L:>10}" for L in LAB))
    for i in range(NB):
        print(f"      {LAB[i]:>12} " + "  ".join(
            f"{C[i, j]:>10.4f}" if np.isfinite(C[i, j]) else f"{'--':>10}" for j in range(NB)))

    # the top-1% |s| galaxies: do THEY put weight in the tail?
    k1 = max(1, n // 100)
    top = np.argsort(-np.abs(s_tot - s_tot.mean()))[:k1]
    print(f"\n    the top 1% of galaxies by |s-<s>| (n={k1}, they carry ~60% of Var(s)) -- "
          f"where does THEIR weight sit?")
    print(f"      {'deficit bin':>12} {'w mean (top1%)':>16} {'w max (top1%)':>15} "
          f"{'w mean (rest)':>15} {'|s^b| mean (top1%)':>20}")
    for bb in range(NB):
        rest = np.ones(n, dtype=bool)
        rest[top] = False
        print(f"      {LAB[bb]:>12} {w_part[top, bb].mean():>16.6f} "
              f"{w_part[top, bb].max():>15.6f} {w_part[rest, bb].mean():>15.6f} "
              f"{np.abs(s_part[top, bb]).mean():>20.4f}")
    print(f"    [{time.time() - t0:.0f}s]", flush=True)

    # ---------------- PART B ------------------------------------------------------------
    if args.with_stencil:
        t0 = time.time()
        deltas = [0.005, 0.0025]
        gam = [0.0] + [x for dd in deltas for x in (+dd, -dd)]
        ctxs = ctxs_at(model, node_df, node_intr, tuple(gam), pre, nbr_std, dev)
        b_ss = np.zeros(NB); b_bb = np.zeros(NB)
        b_wss = np.zeros(NB); b_wbb = np.zeros(NB)
        bcnt = np.zeros(NB, dtype=np.int64)
        ss = aa = bb_ = 0.0
        wss = 0.0
        npair = 0
        for s0 in range(0, n, gc):
            e0 = min(s0 + gc, n)
            xb = xhat[s0:e0]
            p0 = phi_block(model, xb, ctxs[0.0][0], ctxs[0.0][1], args.chunk)
            w = posterior_weights(p0)
            bi = bin_of(p0 - p0.max(axis=1, keepdims=True))
            del p0
            ds = []
            for dd in deltas:
                pp = phi_block(model, xb, ctxs[+dd][0], ctxs[+dd][1], args.chunk)
                pm = phi_block(model, xb, ctxs[-dd][0], ctxs[-dd][1], args.chunk)
                ds.append((pp - pm) / (2.0 * dd))
                del pp, pm
            diff = ds[0] - ds[1]
            fine = ds[1]
            fl = bi.ravel()
            bcnt += np.bincount(fl, minlength=NB)
            b_ss += np.bincount(fl, weights=(diff ** 2).ravel(), minlength=NB)
            b_bb += np.bincount(fl, weights=(fine ** 2).ravel(), minlength=NB)
            b_wss += np.bincount(fl, weights=(w * diff ** 2).ravel(), minlength=NB)
            b_wbb += np.bincount(fl, weights=(w * fine ** 2).ravel(), minlength=NB)
            ss += float((diff ** 2).sum()); aa += float((ds[0] ** 2).sum())
            bb_ += float((fine ** 2).sum()); wss += float((w * diff ** 2).sum())
            npair += (e0 - s0) * K
            del ds, diff, fine, w, bi
        for c, _ in ctxs.values():
            del c
        del ctxs
        print(f"\n=== PART B: per-deficit-bin stencil convergence on the FULL sample "
              f"(0.0050 vs 0.0025), K={K:,}  N_gal={n:,}")
        print(f"    overall  rms(diff)={np.sqrt(ss / npair):.4f}  rms(fine)={np.sqrt(bb_ / npair):.4f}"
              f"  rel={np.sqrt(ss / bb_):.4f}   WEIGHTED rms(diff)={np.sqrt(wss / n):.4f}")
        print(f"    [tailC on 3,000 galaxies got rel=0.0297, WEIGHTED 0.0538; "
              f"on 9,918 the overall rel was 0.1162]")
        print(f"      {'bin':>12} {'nodes':>8} {'rel':>10} {'wrel':>10} "
              f"{'share of sum(diff^2)':>21} {'share of w*diff^2':>19}")
        for i in range(NB):
            rel = np.sqrt(b_ss[i] / max(b_bb[i], 1e-300))
            wrel = np.sqrt(b_wss[i] / max(b_wbb[i], 1e-300))
            print(f"      {LAB[i]:>12} {bcnt[i] / max(bcnt.sum(), 1):>8.2%} {rel:>10.4f} "
                  f"{wrel:>10.4f} {b_ss[i] / max(ss, 1e-300):>21.2%} "
                  f"{b_wss[i] / max(wss, 1e-300):>19.2%}")
        print(f"    [{time.time() - t0:.0f}s]", flush=True)

    print("\n  READING GUIDE.  PART A: if sd(s^b) and max|s^b| are tiny for the deep bins AND "
          "\n  the counterfactual Var with the tail deleted equals the full Var, probe C's "
          "\n  'the tail carries 0.00%' is a real zero, not a cancellation, and the per-galaxy "
          "\n  weight MAX rules out a few galaxies hiding inside the population mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
