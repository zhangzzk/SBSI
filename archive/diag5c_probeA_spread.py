#!/usr/bin/env python -B
"""PROBE A: does the WEIGHTED SPREAD of the integrand grow with K fast enough to cancel
the averaging gain?

The §5C score is a self-normalised weighted mean, `s_i = E_w[phi']`.  For such an estimator
the Monte-Carlo variance is approximately

    Var_MC(s_i)  ~=  Var_w(phi')_i / ESS_i ,     ESS_i = 1 / sum_k w_ik^2

so the bank-noise floor has TWO moving parts, not one.  cont.169 left two facts that no
single story explains: half-banks agree only 6-8% per galaxy (reads as pure noise), yet
doubling the bank does not reduce Var(s) at all (reads as a fixed defect).  Both follow at
once if `Var_w(phi')` grows with K at the same rate ESS does: every added node also adds
more extreme `phi'` values, and the extra spread eats the extra averaging.

MEASURED HERE, at gamma = 0 only, on a NESTED bank ladder (K=1000 nodes are the first 1000
of the K=20000 bank, so every rung is a column slice of ONE phi array -- exact, and 1.5x
cheaper than recomputing):

    ESS_i                       mean and median over galaxies
    Var_w(phi')_i               mean and median over galaxies
    predicted noise             mean_i[ Var_w / ESS ], at FULL bank and at HALF bank
    measured half-bank noise    sigma^2_half = Var_gal(s_A) - Cov_gal(s_A, s_B)
    scaling exponents           Var_w ~ K^p, ESS ~ K^q; p - q ~ 0 is the explanation
    |phi'| quantiles            p50 / p99 / p99.9 / max, unweighted over nodes
    top-1 / top-10 MAGNITUDE share of sum_k |w_k phi'_k|

The prediction-vs-measurement line (item 3) is the real test of whether the weighted-mean
variance formula even applies; if it does not, the exponents are describing a formula that
does not hold and must not be read as the mechanism.

Bank 0 uses exactly the rows, galaxies, seed and delta of job 15459872, so its Var(s),
corr(sA,sB) and sigma^2_half at K = 2000/6000/20000 must REPRODUCE that log.  That is the
built-in validity check.  Bank 1 is a disjoint node pool, the same galaxies: it is the only
handle on bank-to-bank scatter, which the galaxy bootstrap cannot see.

Usage:
    python scripts/diag5c_probeA_spread.py --n-gal 20000 --n-nodes 1000,2000,6000,20000
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


# --------------------------------------------------------------------------------------
# per-galaxy weighted statistics, computed in row chunks so nothing (N, K) is duplicated
# --------------------------------------------------------------------------------------

def weighted_stats(phi0, d1, k0, k1, row_chunk=384, want_shares=True):
    """Per-galaxy stats on the node columns [k0, k1).

    Returns arrays of length N_gal: s, ess, varw, and (optionally) the top-1/top-10
    magnitude shares of `sum_k |w_k phi'_k|` under two rankings -- by TERM magnitude
    (the direct question) and by WEIGHT (comparable with `tail_diagnostics`).
    """
    n = phi0.shape[0]
    out = {k: np.empty(n, dtype=np.float64)
           for k in ("s", "ess", "varw", "t1", "t10", "w1", "w10")}
    for a in range(0, n, row_chunk):
        b = min(a + row_chunk, n)
        p = phi0[a:b, k0:k1]
        d = d1[a:b, k0:k1]
        w = posterior_weights(p)
        s = np.einsum("ij,ij->i", w, d)
        out["s"][a:b] = s
        out["ess"][a:b] = 1.0 / np.einsum("ij,ij->i", w, w)
        out["varw"][a:b] = np.einsum("ij,ij->i", w, d * d) - s ** 2
        if want_shares:
            t = np.abs(w * d)
            tot = np.maximum(t.sum(axis=1), 1e-300)
            kk = min(10, t.shape[1])
            part = np.partition(t, -kk, axis=1)[:, -kk:]
            out["t1"][a:b] = part.max(axis=1) / tot
            out["t10"][a:b] = part.sum(axis=1) / tot
            idx = np.argpartition(-w, kk - 1, axis=1)[:, :kk]
            tw = np.take_along_axis(t, idx, axis=1)
            out["w1"][a:b] = tw.max(axis=1) / tot
            out["w10"][a:b] = tw.sum(axis=1) / tot
        else:
            for k in ("t1", "t10", "w1", "w10"):
                out[k][a:b] = np.nan
    return out


def block_score(phi0, d1, k0, k1, row_chunk=384):
    """`s_i` alone, on the node columns [k0, k1)."""
    n = phi0.shape[0]
    s = np.empty(n, dtype=np.float64)
    for a in range(0, n, row_chunk):
        b = min(a + row_chunk, n)
        w = posterior_weights(phi0[a:b, k0:k1])
        s[a:b] = np.einsum("ij,ij->i", w, d1[a:b, k0:k1])
    return s


def batch_means_noise(phi0, d1, K, n_block, row_chunk=384):
    """DIRECT per-galaxy Monte-Carlo variance of `s_i` at bank size K, by batch means.

    Split the K nodes into `n_block` disjoint blocks.  Each block gives an independent
    estimate `s^(m)_i` at bank size K/n_block, and the mean of the blocks is (to
    self-normalisation) the K-node estimate, so

        Var_MC( s_i at K )  ~=  Var_m( s^(m)_i ) / n_block .

    Unlike `Var(s_A) - Cov(s_A, s_B)` this needs NO assumption about how the noise is
    distributed across galaxies -- it is measured per galaxy, from the bank alone.
    """
    B = K // n_block
    sm = np.stack([block_score(phi0, d1, m * B, (m + 1) * B, row_chunk)
                   for m in range(n_block)])                     # (n_block, N_gal)
    return sm.var(axis=0, ddof=1) / n_block


def abs_quantiles(d1, k0, k1, rows, qs=(50.0, 99.0, 99.9), row_chunk=384):
    """Unweighted |phi'| quantiles over nodes: per-galaxy (averaged) and pooled.

    `rows` is a galaxy subsample used for the quantiles; the max is exact over ALL rows.
    """
    sub = np.abs(d1[np.ix_(rows, np.arange(k0, k1))])
    per = np.percentile(sub, qs, axis=1)            # (n_q, n_rows)
    pooled = np.percentile(sub.ravel(), qs)
    per_max = sub.max(axis=1)
    gmax = 0.0
    for a in range(0, d1.shape[0], row_chunk):
        b = min(a + row_chunk, d1.shape[0])
        gmax = max(gmax, float(np.abs(d1[a:b, k0:k1]).max()))
    return dict(per_gal=per.mean(axis=1), per_gal_max=float(per_max.mean()),
                pooled=pooled, pooled_max=gmax)


def ols_slope(logk, logy):
    x = logk - logk.mean()
    return float(np.dot(x, logy - logy.mean()) / np.dot(x, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="1000,2000,6000,20000")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--row-chunk", type=int, default=384)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--n-qgal", type=int, default=1500, help="galaxies used for |phi'| quantiles")
    ap.add_argument("--n-block", type=int, default=10, help="batch-means blocks per bank")
    ap.add_argument("--n-banks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nodes = sorted(int(x) for x in args.n_nodes.split(","))
    kmax = max(nodes)
    d = args.delta
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    print(f"device {dev}  primary_only_shear={meta.get('primary_only_shear')}", flush=True)

    need = (args.n_banks * kmax + args.n_gal) * 4
    rows = load_rows(args.catalogue, need)
    # bank 0 + galaxies laid out EXACTLY as job 15459872: pool = rows[:kmax], gal = next block
    pools = [rows.iloc[i * kmax:(i + 1) * kmax].reset_index(drop=True) for i in range(1)]
    gal_df = rows.iloc[kmax:kmax + args.n_gal].reset_index(drop=True)
    off = kmax + args.n_gal
    for i in range(1, args.n_banks):
        pools.append(rows.iloc[off:off + kmax].reset_index(drop=True))
        off += kmax
    print(f"rows: {args.n_banks} x {kmax:,} node pool + {len(gal_df):,} galaxies (disjoint)",
          flush=True)

    def intr_of(df):
        return {"e1p": intrinsic_shape(df, "p")[0], "e2p": intrinsic_shape(df, "p")[1],
                "e1s": intrinsic_shape(df, "s")[0], "e2s": intrinsic_shape(df, "s")[1]}

    gal_intr = intr_of(gal_df)

    # ---- the data: gamma = 0 only, same draw as the reference job ----------------------
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
    del ctx_g, pdet_g, xh

    rng = np.random.default_rng(args.seed)
    n_keep = xhat.shape[0]
    qrows = np.sort(rng.choice(n_keep, size=min(args.n_qgal, n_keep), replace=False))
    logk = np.log(np.array(nodes, dtype=np.float64))
    results = {}

    for bi, pool in enumerate(pools):
        pool_intr = intr_of(pool)
        print(f"\n{'=' * 94}\nBANK {bi}   K_max = {kmax:,}"
              f"{'  (identical layout to job 15459872)' if bi == 0 else '  (disjoint pool)'}"
              f"\n{'=' * 94}", flush=True)

        # phi at the three stencil points for the FULL K_max bank, once
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
        print(f"  phi block ready: {phi0.shape} float64 "
              f"({2 * phi0.nbytes / 1e9:.2f} GB resident)", flush=True)

        per_k = {}
        for K in nodes:
            half = K // 2
            full = weighted_stats(phi0, d1, 0, K, args.row_chunk)
            hA = weighted_stats(phi0, d1, 0, half, args.row_chunk, want_shares=False)
            hB = weighted_stats(phi0, d1, half, 2 * half, args.row_chunk, want_shares=False)
            q = abs_quantiles(d1, 0, K, qrows, row_chunk=args.row_chunk)
            bm = batch_means_noise(phi0, d1, K, args.n_block, args.row_chunk)
            per_k[K] = dict(full=full, A=hA, B=hB, q=q, bm=bm)

            sA, sB = hA["s"], hB["s"]
            var_a, var_b = float(np.var(sA)), float(np.var(sB))
            cov = float(np.cov(sA, sB)[0, 1])
            sig2_half = var_a - cov
            # assumption-light alternative: PER-GALAXY, needs no across-galaxy independence
            #   E[(s_A - s_B)^2] = 2 Var_MC(s at K/2)   (the reproducible part cancels)
            pair = (sA - sB) ** 2 / 2.0
            pred_half = float(np.mean(hA["varw"] / hA["ess"]))
            pred_full = float(np.mean(full["varw"] / full["ess"]))
            print(f"\nK={K:>6,}   ESS mean={np.mean(full['ess']):8.2f} "
                  f"med={np.median(full['ess']):8.2f}   ESS/K={np.mean(full['ess']) / K:6.3%}")
            print(f"          Var_w(phi')  mean={np.mean(full['varw']):11.2f} "
                  f"med={np.median(full['varw']):11.2f}")
            print(f"          predicted noise  Var_w/ESS   full-bank mean={pred_full:9.3f} "
                  f"med={np.median(full['varw'] / full['ess']):9.3f}")
            print(f"          HALF bank (K/2={half:,}): ESS={np.mean(hA['ess']):8.2f}  "
                  f"Var_w={np.mean(hA['varw']):11.2f}  predicted={pred_half:9.3f}")
            print(f"          MEASURED sigma^2_half = Var(sA)-Cov(sA,sB) = {sig2_half:9.3f}   "
                  f"[Var(sA)={var_a:8.3f} Var(sB)={var_b:8.3f} "
                  f"corr={cov / np.sqrt(var_a * var_b):+.4f}]")
            print(f"          MEASURED sigma^2_half, per-galaxy route <(sA-sB)^2>/2 = "
                  f"{float(np.mean(pair)):9.3f}  med={float(np.median(pair)):9.3f}")
            print(f"          ratio predicted/measured (half bank) = "
                  f"{pred_half / sig2_half:6.3f} (cov route)  "
                  f"{pred_half / float(np.mean(pair)):6.3f} (pair route)")
            pf = full["varw"] / full["ess"]
            print(f"          DIRECT batch-means noise at K ({args.n_block} blocks): "
                  f"mean={float(np.mean(bm)):9.3f} med={float(np.median(bm)):9.3f}   "
                  f"vs predicted full-bank mean={pred_full:9.3f} "
                  f"(ratio {pred_full / float(np.mean(bm)):5.3f}, "
                  f"per-gal corr {float(np.corrcoef(np.log(pf + 1e-300), np.log(bm + 1e-300))[0, 1]):+.3f})")
            print(f"          Var(s) full bank = {float(np.var(full['s'])):9.3f}   "
                  f"mean(s)={float(np.mean(full['s'])):+8.4f}")
            print(f"          |phi'| per-gal quantiles  p50={q['per_gal'][0]:9.3f} "
                  f"p99={q['per_gal'][1]:10.3f} p99.9={q['per_gal'][2]:11.3f} "
                  f"max={q['per_gal_max']:12.3f}")
            print(f"          |phi'| pooled  quantiles  p50={q['pooled'][0]:9.3f} "
                  f"p99={q['pooled'][1]:10.3f} p99.9={q['pooled'][2]:11.3f} "
                  f"max(all rows)={q['pooled_max']:12.3f}")
            print(f"          magnitude share of sum|w phi'|: by TERM  top1="
                  f"{np.mean(full['t1']):6.3%} top10={np.mean(full['t10']):6.3%}   "
                  f"by WEIGHT top1={np.mean(full['w1']):6.3%} top10={np.mean(full['w10']):6.3%}",
                  flush=True)

        del phi0, d1, ph

        # ---- scaling exponents, bootstrapped over galaxies -----------------------------
        vw = np.stack([per_k[K]["full"]["varw"] for K in nodes])      # (n_K, N_gal)
        es = np.stack([per_k[K]["full"]["ess"] for K in nodes])
        p_hat = ols_slope(logk, np.log(vw.mean(axis=1)))
        q_hat = ols_slope(logk, np.log(es.mean(axis=1)))
        n = vw.shape[1]
        brng = np.random.default_rng(args.seed + 1000 * bi)
        bp, bq = [], []
        for _ in range(args.n_boot):
            idx = brng.integers(0, n, n)
            bp.append(ols_slope(logk, np.log(vw[:, idx].mean(axis=1))))
            bq.append(ols_slope(logk, np.log(es[:, idx].mean(axis=1))))
        bp, bq = np.array(bp), np.array(bq)
        pm_hat = ols_slope(logk, np.log(vw.mean(axis=1))) - q_hat
        print(f"\n  SCALING (OLS on log-log over K = {nodes}, bootstrap over galaxies)")
        print(f"    Var_w(phi') ~ K^p   p = {p_hat:+.4f} +- {bp.std():.4f}")
        print(f"    ESS         ~ K^q   q = {q_hat:+.4f} +- {bq.std():.4f}")
        print(f"    p - q               = {pm_hat:+.4f} +- {(bp - bq).std():.4f}"
              f"   (0 => noise flat in K; -1 => noise ~ 1/K)", flush=True)
        # median-based exponents, in case the means are carried by a few galaxies
        vwm = np.array([float(np.median(per_k[K]["full"]["varw"])) for K in nodes])
        esm = np.array([float(np.median(per_k[K]["full"]["ess"])) for K in nodes])
        print(f"    medians:  p_med = {ols_slope(logk, np.log(vwm)):+.4f}   "
              f"q_med = {ols_slope(logk, np.log(esm)):+.4f}   "
              f"p-q = {ols_slope(logk, np.log(vwm)) - ols_slope(logk, np.log(esm)):+.4f}")
        # measured half-bank noise exponent, for reference
        s2h = np.array([float(np.var(per_k[K]["A"]["s"]))
                        - float(np.cov(per_k[K]["A"]["s"], per_k[K]["B"]["s"])[0, 1])
                        for K in nodes])
        s2p = np.array([float(np.mean((per_k[K]["A"]["s"] - per_k[K]["B"]["s"]) ** 2)) / 2.0
                        for K in nodes])
        print(f"    per-galaxy sigma^2_half ~ K^{ols_slope(logk, np.log(s2p)):+.4f}   "
              f"[values {np.array2string(s2p, precision=2)}]")
        print(f"    measured sigma^2_half ~ K^r   r = {ols_slope(logk, np.log(s2h)):+.4f}"
              f"   (pure MC would be -1)")
        predf = np.array([float(np.mean(per_k[K]["full"]["varw"] / per_k[K]["full"]["ess"]))
                          for K in nodes])
        bmm = np.array([float(np.mean(per_k[K]["bm"])) for K in nodes])
        print(f"    predicted Var_w/ESS   ~ K^{ols_slope(logk, np.log(predf)):+.4f}   "
              f"DIRECT batch-means ~ K^{ols_slope(logk, np.log(bmm)):+.4f}   "
              f"Var(s) ~ K^{ols_slope(logk, np.log(np.array([float(np.var(per_k[K]['full']['s'])) for K in nodes]))):+.4f}",
              flush=True)

        results[bi] = dict(
            nodes=np.array(nodes),
            varw_mean=vw.mean(axis=1), varw_med=vwm,
            ess_mean=es.mean(axis=1), ess_med=esm,
            pred_full=np.array([float(np.mean(per_k[K]["full"]["varw"]
                                              / per_k[K]["full"]["ess"])) for K in nodes]),
            pred_half=np.array([float(np.mean(per_k[K]["A"]["varw"]
                                              / per_k[K]["A"]["ess"])) for K in nodes]),
            sig2_half=s2h, sig2_half_pair=s2p,
            bm_mean=np.array([float(np.mean(per_k[K]["bm"])) for K in nodes]),
            bm_med=np.array([float(np.median(per_k[K]["bm"])) for K in nodes]),
            var_s=np.array([float(np.var(per_k[K]["full"]["s"])) for K in nodes]),
            var_sA=np.array([float(np.var(per_k[K]["A"]["s"])) for K in nodes]),
            corr=np.array([float(np.corrcoef(per_k[K]["A"]["s"], per_k[K]["B"]["s"])[0, 1])
                           for K in nodes]),
            q_pergal=np.stack([per_k[K]["q"]["per_gal"] for K in nodes]),
            q_pergal_max=np.array([per_k[K]["q"]["per_gal_max"] for K in nodes]),
            q_pooled=np.stack([per_k[K]["q"]["pooled"] for K in nodes]),
            q_pooled_max=np.array([per_k[K]["q"]["pooled_max"] for K in nodes]),
            share_t1=np.array([float(np.mean(per_k[K]["full"]["t1"])) for K in nodes]),
            share_t10=np.array([float(np.mean(per_k[K]["full"]["t10"])) for K in nodes]),
            share_t1_med=np.array([float(np.median(per_k[K]["full"]["t1"])) for K in nodes]),
            share_t10_med=np.array([float(np.median(per_k[K]["full"]["t10"])) for K in nodes]),
            p=p_hat, p_se=float(bp.std()), q=q_hat, q_se=float(bq.std()),
            pmq=pm_hat, pmq_se=float((bp - bq).std()),
        )
        del per_k, vw, es

    # ---- compact summary table -----------------------------------------------------
    print(f"\n{'=' * 94}\nSUMMARY  (rows: one per bank x K)\n{'=' * 94}")
    print(f"{'bank':>4} {'K':>7} {'ESS':>8} {'Var_w':>11} {'Var_w/ESS':>10} "
          f"{'batchmean':>10} {'pred_half':>10} {'sig2_half':>10} {'ratio':>7} "
          f"{'Var(s)':>9} {'corrAB':>8}")
    for bi, r in results.items():
        for j, K in enumerate(r["nodes"]):
            print(f"{bi:>4} {int(K):>7,} {r['ess_mean'][j]:>8.2f} {r['varw_mean'][j]:>11.2f} "
                  f"{r['pred_full'][j]:>10.3f} {r['bm_mean'][j]:>10.3f} "
                  f"{r['pred_half'][j]:>10.3f} "
                  f"{r['sig2_half'][j]:>10.3f} "
                  f"{r['pred_half'][j] / r['sig2_half'][j]:>7.3f} "
                  f"{r['var_s'][j]:>9.3f} {r['corr'][j]:>+8.4f}")
    for bi, r in results.items():
        print(f"  bank {bi}: p = {r['p']:+.4f}+-{r['p_se']:.4f}  "
              f"q = {r['q']:+.4f}+-{r['q_se']:.4f}  "
              f"p-q = {r['pmq']:+.4f}+-{r['pmq_se']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        flat = {f"b{bi}_{k}": v for bi, r in results.items() for k, v in r.items()}
        np.savez(args.out, **flat)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
