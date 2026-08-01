#!/usr/bin/env python -B
"""PROBE B follow-up: the integrand's OWN per-node distribution, at fixed weights.

Probe B (job 15460992) showed that with the weights held exactly as they are, a bounded
N(0,1) integrand averages down exactly as importance-sampling theory says --
`Var_i(E_w[g]) = mean_i(1/ESS_i) - 1/ESS_pop` to within a few percent at every K -- while
phi' does not.  That localises the failure in the INTEGRAND.  This script measures the
integrand directly, still at fixed weights and fixed galaxies:

  1. `noise_i = sum_k w_ik^2 (phi'_ik - s_i)^2`  -- the EXACT per-galaxy Monte-Carlo
     variance of `s_i = E_w[phi']`, the same formula the N(0,1) probe verified.  Its mean
     over galaxies is the predicted MC contribution to Var(s); compare it with the measured
     Var(s) and with `sigma^2_half / 2`.
  2. `Vw_i = Var_w(phi')` -- the posterior spread of the integrand itself.  The noise is
     `Vw_i / ESS_i` (up to the weight-tail correction), so if `Vw` GROWS with K it can
     cancel the slow improvement in `1/ESS`.  That is the candidate explanation for
     Var(s) being flat/non-monotonic in K.
  3. tail percentiles of `|phi'|` over nodes, per galaxy and pooled: p50 / p99 / p99.9 /
     max.  A tail that grows with K is a bank that keeps finding more extreme nodes.
  4. WINSORISED phi' at per-galaxy percentiles: if clipping the tail restores the 1/ESS
     falloff and lifts the half-bank correlation, the tail is the mechanism.

Usage:
    python scripts/diag5c_probeB2_tail.py --n-gal 20000 --n-nodes 2000,6000,10000,20000
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


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="2000,6000,10000,20000")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--clips", default="1.0,0.1")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    nodes = [int(x) for x in args.n_nodes.split(",")]
    clips = [float(x) for x in args.clips.split(",")]
    kmax = max(nodes)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    print(f"device {dev}  primary_only_shear={meta.get('primary_only_shear')}", flush=True)

    rows = load_rows(args.catalogue, (kmax + args.n_gal) * 4)
    pool = rows.iloc[:kmax].reset_index(drop=True)
    gal_df = rows.iloc[kmax:kmax + args.n_gal].reset_index(drop=True)
    print(f"rows: {len(pool):,} node pool + {len(gal_df):,} galaxies (disjoint)", flush=True)

    def intr_of(df):
        d = {}
        d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
        d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
        return d

    gal_intr, pool_intr = intr_of(gal_df), intr_of(pool)
    d = args.delta

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
    print(f"  gamma=0: kept {int(keep.sum()):,}/{len(gal_df):,}", flush=True)
    del ctx_g, pdet_g

    for K in nodes:
        node_df = pool.iloc[:K].reset_index(drop=True)
        node_intr = {k: v[:K] for k, v in pool_intr.items()}
        half = K // 2
        A, B = np.arange(half), np.arange(half, 2 * half)
        print(f"\n================ K = {K:,} ================", flush=True)

        c0, pd0 = scene_context(model, node_df, node_intr, (0.0, 0.0), pre, nbr_std, dev)
        phi0 = phi_block(model, xhat, c0, torch.log(pd0), args.chunk)
        del c0, pd0
        cp, pdp = scene_context(model, node_df, node_intr, (0.0, +d), pre, nbr_std, dev)
        php = phi_block(model, xhat, cp, torch.log(pdp), args.chunk)
        del cp, pdp
        cm, pdm = scene_context(model, node_df, node_intr, (0.0, -d), pre, nbr_std, dev)
        phm = phi_block(model, xhat, cm, torch.log(pdm), args.chunk)
        del cm, pdm
        np.subtract(php, phm, out=php)
        php /= (2.0 * d)
        d1 = php
        del phm

        # ---- pooled and per-galaxy tail of the integrand -------------------------------
        a = np.abs(d1)
        pooled = np.percentile(a, [50, 99, 99.9, 99.99])
        pg = np.percentile(a, [50, 99.9], axis=1)
        print(f"|phi'| pooled: p50 {pooled[0]:8.3f}  p99 {pooled[1]:9.3f}  "
              f"p99.9 {pooled[2]:10.3f}  p99.99 {pooled[3]:11.3f}  max {a.max():12.3f}")
        print(f"       per-galaxy medians: p50 {np.median(pg[0]):8.3f}  "
              f"p99.9 {np.median(pg[1]):10.3f}  max(med) {np.median(a.max(axis=1)):11.3f}")
        del a, pg

        # ---- half-bank scores, and the PLUG-IN noise estimate at half-bank size ---------
        # Renormalising a column slice of `w` IS the half-bank posterior weight, since
        # w propto exp(phi) row-wise -- so this shares the weights exactly.
        w = posterior_weights(phi0)
        del phi0
        inv_ess = np.sum(w ** 2, axis=1)
        sh, plug_half = {}, {}
        for tag, cols in (("A", A), ("B", B)):
            wh = w[:, cols] / w[:, cols].sum(axis=1, keepdims=True)
            sh[tag] = np.sum(wh * d1[:, cols], axis=1)
            dv = d1[:, cols] - sh[tag][:, None]
            plug_half[tag] = float(np.mean(np.sum(wh ** 2 * dv ** 2, axis=1)))
            del wh, dv
        s = np.sum(w * d1, axis=1)
        dev_ = d1 - s[:, None]
        Vw = np.sum(w * dev_ ** 2, axis=1)                 # posterior spread of phi'
        noise_i = np.sum(w ** 2 * dev_ ** 2, axis=1)       # exact MC variance of s_i
        del dev_

        var_s = float(np.var(s))
        s2h = float(np.var(sh["A"]) - np.cov(sh["A"], sh["B"])[0, 1])
        print(f"\n  Var(s)={var_s:10.4f}   corr(sA,sB)={corr(sh['A'], sh['B']):+.4f}   "
              f"sigma2_half={s2h:9.4f}")
        print(f"  mean(1/ESS_i)={inv_ess.mean():.6f}   mean Var_w(phi')={Vw.mean():12.3f}  "
              f"median {np.median(Vw):12.3f}")
        print(f"  PLUG-IN MC-noise  mean_i sum_k w^2 (phi'-s)^2 = {noise_i.mean():10.4f}"
              f"   -> {noise_i.mean() / var_s:6.1%} of Var(s)")
        print(f"  LIKE-FOR-LIKE at HALF-bank size: plug-in A {plug_half['A']:9.4f}  "
              f"B {plug_half['B']:9.4f}   vs REPLICATE sigma2_half {s2h:9.4f}   "
              f"replicate/plug-in = {s2h / max(0.5 * (plug_half['A'] + plug_half['B']), 1e-30):6.2f}")
        print("  (the same plug-in formula matched the N(0,1) probe to a few percent, so a "
              "ratio far from 1 here is a property of the INTEGRAND, not of the weights)")
        print(f"  corr(noise_i, 1/ESS_i)={corr(noise_i, inv_ess):+.4f}   "
              f"corr(Vw, 1/ESS_i)={corr(Vw, inv_ess):+.4f}")

        # ---- winsorised integrand, same weights ---------------------------------------
        for cp_ in clips:
            lo, hi = np.percentile(d1, [cp_, 100 - cp_], axis=1, keepdims=True)
            dc = np.clip(d1, lo, hi)
            sc = np.sum(w * dc, axis=1)
            # half banks for the clipped integrand.  Renormalising the column slice of `w`
            # IS the half-bank posterior weight, since w propto exp(phi) row-wise.  The clip
            # levels are the full-bank ones, so both halves see the same threshold.
            scA = np.einsum("ij,ij->i", w[:, A], dc[:, A]) / w[:, A].sum(axis=1)
            scB = np.einsum("ij,ij->i", w[:, B], dc[:, B]) / w[:, B].sum(axis=1)
            devc = dc - sc[:, None]
            nz = np.sum(w ** 2 * devc ** 2, axis=1)
            print(f"  clip {cp_:>4}%: Var(s)={float(np.var(sc)):10.4f}  "
                  f"corr(sA,sB)={corr(scA, scB):+.4f}  "
                  f"MC-noise={nz.mean():10.4f} ({nz.mean() / max(float(np.var(sc)), 1e-30):5.1%})"
                  f"  mean Var_w={float(np.mean(np.sum(w * devc ** 2, axis=1))):11.3f}")
            del dc, devc, sc, scA, scB, nz, lo, hi
        del w, d1, s, sh, Vw, noise_i, inv_ess

    print("\nREAD: if `mean Var_w(phi')` grows with K while mean(1/ESS_i) falls, the two "
          "cancel and Var(s) stays flat -- a bank that keeps finding more extreme nodes.  "
          "If winsorising restores a falling Var(s) and a rising corr(sA,sB), the tail of "
          "the integrand is the mechanism.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
