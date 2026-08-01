#!/usr/bin/env python -B
"""ADVERSARIAL CHECK of probe A (jobs 15461075 / 15461552 / 15462371).

Probe A's substantive claim is not the p-q exponent but its replacement:

    "the Var_w/ESS description of this estimator is not merely mis-scaled, it is the WRONG
     MODEL.  It under-predicts the directly measured bank noise by 6-10x at every rung ...
     Variance here does not decompose over disjoint node blocks and ESS is not the
     effective sample size.  Any argument built on Var_w/ESS or on ESS as a bank-quality
     figure of merit ... is unsupported."

Three things were never tested before that was concluded, and all three are cheap:

 1. `Var_w(phi')/ESS` is NOT the delta-method variance of a self-normalised importance
    estimator.  The standard first-order estimator is

        v_i = sum_k w_ik^2 (phi'_ik - s_i)^2                                   (Owen 9.9)

    which equals `Var_w * sum_k w^2` only when the deviations are UNCORRELATED with the
    weights.  Probe A's own numbers say they are strongly correlated (top-1 share of
    sum|w phi'| is 20.36% ranked by TERM and 20.06% ranked by WEIGHT -- the largest term
    IS essentially the largest weight), so the surrogate must under-predict.  Measure `v`.

 2. The batch-means route equal-weights the block estimates: mean_m s^(m).  But the pooled
    estimate is s_K = sum_m W_m s^(m) with W_m = sum_{k in block m} w_k, and the weights
    are heavy, so W_m != 1/M.  Then Var(s_K) ~= Var_block * sum_m W_m^2 while the reported
    batch-means value is Var_block / M.  The ratio is M * sum_m W_m^2, computable exactly.
    If that is ~M the "10x disagreement" is the estimator's definition, not a discovery.

 3. NO error bar was quoted on any measured sigma^2_half, on the ratios, or on the K
    exponent.  `sigma^2_half = Var(sA) - Cov(sA,sB)` is asymmetric: recomputing it from
    the B half of the SAME numbers gives 7.70 instead of 15.38 at K=1000.  Here every
    noise level is measured on EQUAL-SIZE DISJOINT column blocks so the scatter over bank
    REALISATIONS -- the error bar the galaxy bootstrap cannot see -- is measured directly,
    which is also the only honest test of fact (b) ("noise flat in K").

gamma = 0 only, seed 11, delta 0.01, bank rows 0:20000 -- the identical layout to jobs
15459872 / 15461075 (bank 0) / 15461552, so Var(s) and sigma^2_half reproduce as a check.

Usage:
    python scripts/diag5c_verifyR_snis.py --n-gal 20000 --n-nodes 1000,2000,6000,20000
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


def block_stats(phi0, d1, k0, k1, row_chunk=256):
    """Per-galaxy (s, ESS, Var_w, v_snis, wmax) on the node columns [k0, k1)."""
    n = phi0.shape[0]
    s = np.empty(n); ess = np.empty(n); varw = np.empty(n)
    v = np.empty(n); wmax = np.empty(n)
    for a in range(0, n, row_chunk):
        b = min(a + row_chunk, n)
        w = posterior_weights(phi0[a:b, k0:k1])
        dd = d1[a:b, k0:k1]
        si = np.einsum("ij,ij->i", w, dd)
        s[a:b] = si
        ess[a:b] = 1.0 / np.einsum("ij,ij->i", w, w)
        varw[a:b] = np.einsum("ij,ij->i", w, dd * dd) - si ** 2
        r = dd - si[:, None]
        r *= r
        v[a:b] = np.einsum("ij,ij->i", w * w, r)
        wmax[a:b] = w.max(axis=1)
        del w, r
    return dict(s=s, ess=ess, varw=varw, v=v, wmax=wmax)


def boot_mean(x, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    return float(np.std([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="1000,2000,6000,20000")
    ap.add_argument("--sizes", default="500,1000,2500,5000,10000",
                    help="equal-size disjoint block ladder for the error-barred noise curve")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--row-chunk", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-block", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    nodes = sorted(int(x) for x in args.n_nodes.split(","))
    sizes = sorted(int(x) for x in args.sizes.split(","))
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
    n = int(keep.sum())
    print(f"  gamma=0: kept {n:,}/{len(gal_df):,}  (bank rows 0:{kmax:,}, "
          f"layout of jobs 15459872 / 15461075 bank 0)", flush=True)
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
    print(f"  phi ready {phi0.shape}\n", flush=True)

    # =================================================================================
    # 1. THE FORMULA:  Var_w/ESS  vs  the delta-method SNIS variance  vs  MEASURED
    # =================================================================================
    print("=" * 100)
    print("1. IS Var_w/ESS THE RIGHT FORMULA?  Compare BOTH candidate per-galaxy variance")
    print("   estimators, at the SAME bank size as the measurement (the half bank K/2),")
    print("   against the assumption-light measurement <(sA-sB)^2>/2.")
    print("   surrogate = Var_w/ESS (probe A);   delta = sum_k w_k^2 (phi'_k - s)^2 (Owen).")
    print("=" * 100)
    lad = {}
    for K in nodes:
        half = K // 2
        A = block_stats(phi0, d1, 0, half, args.row_chunk)
        B = block_stats(phi0, d1, half, 2 * half, args.row_chunk)
        F = block_stats(phi0, d1, 0, K, args.row_chunk)
        sA, sB = A["s"], B["s"]
        pair = (sA - sB) ** 2 / 2.0
        meas = float(np.mean(pair))
        cov = float(np.cov(sA, sB)[0, 1])
        covA = float(np.var(sA)) - cov
        covB = float(np.var(sB)) - cov
        sur = 0.5 * (A["varw"] / A["ess"] + B["varw"] / B["ess"])
        dm = 0.5 * (A["v"] + B["v"])
        se_meas = boot_mean(pair, args.n_boot, args.seed)
        se_dm = boot_mean(dm, args.n_boot, args.seed + 1)
        lad[K] = dict(meas=meas, sur=float(np.mean(sur)), dm=float(np.mean(dm)),
                      sur_med=float(np.median(sur)), dm_med=float(np.median(dm)),
                      pair_med=float(np.median(pair)),
                      covA=covA, covB=covB, se_meas=se_meas, se_dm=se_dm,
                      var_s=float(np.var(F["s"])), ess=float(np.mean(F["ess"])),
                      wmax=float(np.mean(F["wmax"])), wmax_med=float(np.median(F["wmax"])),
                      v_full=float(np.mean(F["v"])), sur_full=float(np.mean(F["varw"] / F["ess"])))
        print(f"\nbank size K/2 = {half:>6,}   (rung K = {K:,})")
        print(f"   MEASURED  <(sA-sB)^2>/2      mean = {meas:9.3f} +- {se_meas:.3f}"
              f"   median = {float(np.median(pair)):8.4f}")
        print(f"   MEASURED  Var(sA)-Cov        = {covA:9.3f}    "
              f"from the B half instead: {covB:9.3f}   (same quantity, ratio "
              f"{covA / max(covB, 1e-12):.2f})")
        print(f"   surrogate Var_w/ESS          mean = {float(np.mean(sur)):9.3f}"
              f"   median = {float(np.median(sur)):8.4f}"
              f"   ratio to measured mean = {float(np.mean(sur)) / meas:5.3f}")
        print(f"   delta-method sum w^2 (d-s)^2 mean = {float(np.mean(dm)):9.3f} +- {se_dm:.3f}"
              f"   median = {float(np.median(dm)):8.4f}"
              f"   ratio to measured mean = {float(np.mean(dm)) / meas:5.3f}", flush=True)

    logk = np.log(np.array(nodes, dtype=np.float64))
    lk_half = np.log(np.array([K // 2 for K in nodes], dtype=np.float64))
    print(f"\n   EXPONENTS over the half-bank ladder {[K // 2 for K in nodes]}:")
    print(f"     measured <(sA-sB)^2>/2 mean ~ B^"
          f"{ols_slope(lk_half, np.log([lad[K]['meas'] for K in nodes])):+.3f}   "
          f"median ~ B^{ols_slope(lk_half, np.log([lad[K]['pair_med'] for K in nodes])):+.3f}")
    print(f"     surrogate Var_w/ESS    mean ~ B^"
          f"{ols_slope(lk_half, np.log([lad[K]['sur'] for K in nodes])):+.3f}   "
          f"median ~ B^{ols_slope(lk_half, np.log([lad[K]['sur_med'] for K in nodes])):+.3f}")
    print(f"     delta-method           mean ~ B^"
          f"{ols_slope(lk_half, np.log([lad[K]['dm'] for K in nodes])):+.3f}   "
          f"median ~ B^{ols_slope(lk_half, np.log([lad[K]['dm_med'] for K in nodes])):+.3f}")
    print(f"     max posterior weight   mean = "
          f"{[round(lad[K]['wmax'], 4) for K in nodes]}  "
          f"median = {[round(lad[K]['wmax_med'], 4) for K in nodes]}", flush=True)

    # =================================================================================
    # 2. WHY THE BATCH-MEANS ROUTE DISAGREES BY ~10x
    # =================================================================================
    print("\n" + "=" * 100)
    print(f"2. BATCH MEANS.  s_K = sum_m W_m s^(m) with W_m the block's total weight, but")
    print(f"   the batch-means route uses (1/M) sum_m s^(m).  The two estimators differ by")
    print(f"   a variance factor M * sum_m W_m^2 (= 1 only if the blocks carry equal weight).")
    print("=" * 100)
    M = args.n_block
    for K in nodes:
        Bsz = K // M
        Wm = np.empty((n, M))
        for a in range(0, n, args.row_chunk):
            b = min(a + args.row_chunk, n)
            w = posterior_weights(phi0[a:b, :K])
            for m in range(M):
                Wm[a:b, m] = w[:, m * Bsz:(m + 1) * Bsz].sum(axis=1)
            del w
        sw2 = (Wm ** 2).sum(axis=1)
        print(f"   K={K:>6,}  M={M}  sum_m W_m^2: mean={float(np.mean(sw2)):6.3f} "
              f"med={float(np.median(sw2)):6.3f}  (uniform would be {1.0 / M:.3f})   "
              f"=> pooled/batch-means variance factor M*sum W^2: "
              f"mean={float(np.mean(M * sw2)):6.2f} med={float(np.median(M * sw2)):6.2f}",
              flush=True)
        del Wm, sw2

    # =================================================================================
    # 3. NOISE vs BANK SIZE WITH A REALISATION ERROR BAR (equal-size disjoint blocks)
    # =================================================================================
    print("\n" + "=" * 100)
    print("3. NOISE vs BANK SIZE, every level measured on EQUAL-SIZE DISJOINT column blocks")
    print("   of the same 20k pool, so the scatter quoted is over BANK REALISATIONS -- the")
    print("   error bar the galaxy bootstrap cannot see and that probe A never quoted.")
    print("   pair  = mean_i (s_m - s_m')^2/2 over disjoint block PAIRS")
    print("   delta = mean_i sum_k w^2 (d-s)^2 per block")
    print("   Var(s)= across-galaxy variance of the block score (the 5C denominator)")
    print("=" * 100)
    print(f"{'size':>7} {'nblk':>5} {'pair mean+-sd':>24} {'delta mean+-sd':>24} "
          f"{'Var(s) mean+-sd':>24}")
    curve = {}
    for B in sizes:
        nb = kmax // B
        st = [block_stats(phi0, d1, m * B, (m + 1) * B, args.row_chunk) for m in range(nb)]
        pairs = [float(np.mean((st[2 * j]["s"] - st[2 * j + 1]["s"]) ** 2)) / 2.0
                 for j in range(nb // 2)]
        dms = [float(np.mean(x["v"])) for x in st]
        vss = [float(np.var(x["s"])) for x in st]
        curve[B] = dict(pair=np.array(pairs), dm=np.array(dms), vs=np.array(vss))
        pm, ps = np.mean(pairs), (np.std(pairs, ddof=1) / np.sqrt(len(pairs))
                                  if len(pairs) > 1 else np.nan)
        dm_, ds = np.mean(dms), np.std(dms, ddof=1) / np.sqrt(len(dms))
        vm, vs_ = np.mean(vss), np.std(vss, ddof=1) / np.sqrt(len(vss))
        print(f"{B:>7,} {nb:>5} {pm:>13.3f} +- {ps:<8.3f} {dm_:>13.3f} +- {ds:<8.3f} "
              f"{vm:>13.3f} +- {vs_:<8.3f}", flush=True)
        del st
    ls = np.log(np.array(sizes, dtype=np.float64))
    print(f"\n   exponents over bank size: pair ~ B^"
          f"{ols_slope(ls, np.log([curve[B]['pair'].mean() for B in sizes])):+.3f}   "
          f"delta ~ B^{ols_slope(ls, np.log([curve[B]['dm'].mean() for B in sizes])):+.3f}   "
          f"Var(s) ~ B^{ols_slope(ls, np.log([curve[B]['vs'].mean() for B in sizes])):+.3f}")
    print(f"   per-block Var(s) scatter (min..max over disjoint equal-size banks):")
    for B in sizes:
        v = curve[B]["vs"]
        print(f"     size {B:>6,}  n={len(v):>2}  Var(s) in [{v.min():8.3f}, {v.max():8.3f}]"
              f"   max/min = {v.max() / v.min():5.2f}")
    print("\n   pure Monte-Carlo would give exponent -1 on `pair` and `delta`.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
