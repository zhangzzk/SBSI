#!/usr/bin/env python -B
"""PROBE C: is the §5C score spread carried by nodes where the FLOW IS IN ITS TAIL?

phi'_ik is the shear-derivative of the flow's log-density for galaxy i at node k.  For a node
that poorly explains galaxy i the flow is queried far from its training support, and
normalizing flows extrapolate badly, so log-density gradients can blow up.  If |phi'| grows
systematically with how badly the node explains the galaxy, then adding nodes adds tail
evaluations -- which would explain why the spread never settles with K.

Two independent measurements, sharing one phi machinery:

PART 1 -- DEFICIT ATTRIBUTION (delta fixed).  Per (galaxy, node) define the log-likelihood
deficit d_ik = phi0_ik - max_k phi0_ik (<= 0; 0 = best explanation of that galaxy).  Bin nodes
by deficit and report per bin: node fraction, TOTAL WEIGHT carried, median/p99 |phi'|, and the
bin's contribution to the two variances that matter:
    Var_w(phi')  -- the WITHIN-galaxy posterior spread of phi' (the Louis variance term),
                    attributed exactly as   sum_{k in bin} w_ik (phi'_ik - s_i)^2 ;
    Var(s)       -- the ACROSS-galaxy variance of the posterior mean s_i, i.e. the actual
                    inflated denominator of m = I/Var(s) - 1.  s_i = sum_b s_i^(b) with
                    s_i^(b) = sum_{k in bin b} w_ik phi'_ik, so
                    Var(s) = sum_b Cov(s^(b), s) is an EXACT additive attribution.
Both are reported because they are different questions and only the second is the puzzle.

PART 2 -- STENCIL CHECK (no tails involved).  Recompute phi' at several central-difference
half-widths and report Var_w(phi') and Var(s) at each, plus the (i,k)-level agreement of
phi' between successive widths.  If the spread GROWS as delta shrinks, the derivative is not
converging and the spread is a finite-difference artefact rather than a property of the flow.
CONFOUND, stated up front: this part does not measure I, so a proportional growth of both
I and Var(s) with 1/delta would leave m unchanged; read the delta trend as a statement about
the derivative's convergence, not directly about m.

Setup mirrors `scripts/diag5c_repro.py` exactly (same catalogue slicing, same seed, same
gamma=0 data draw) so the numbers are comparable to job 15459872 / WORKLOG cont.169.

Usage:
    python scripts/diag5c_tailC.py --n-gal 20000 --n-nodes 2000,20000 --delta 0.01
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


# deficit bins, matching the convention already used in diag5c_chansplit (4b)
EDGES = np.array([1.0, 3.0, 10.0, 30.0, 100.0])            # on u = -deficit >= 0
LAB = ["[0,-1]", "(-1,-3]", "(-3,-10]", "(-10,-30]", "(-30,-100]", "< -100"]
NB = len(LAB)

# log10|phi'| histogram, for quantiles without holding every value
LG_LO, LG_HI, LG_N = -8.0, 8.0, 1600
LG_EDGES = np.linspace(LG_LO, LG_HI, LG_N + 1)


def bin_of(deficit):
    """Bin index in [0, NB) from the (<=0) log-likelihood deficit."""
    return np.searchsorted(EDGES, -deficit, side="left")


def lg_idx(a):
    """Bin index into the log10|.| histogram, clipped to the end bins."""
    lg = np.log10(np.maximum(np.abs(a), 1e-30))
    return np.clip(((lg - LG_LO) / (LG_HI - LG_LO) * LG_N).astype(np.int64), 0, LG_N - 1)


def quant_from_hist(counts, q):
    """Quantile of |phi'| from a log10 histogram; returns NaN for an empty bin."""
    tot = counts.sum()
    if tot <= 0:
        return float("nan")
    c = np.cumsum(counts)
    j = int(np.searchsorted(c, q * tot, side="left"))
    j = min(j, LG_N - 1)
    return float(10.0 ** (0.5 * (LG_EDGES[j] + LG_EDGES[j + 1])))


def intr_of(df):
    d = {}
    d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
    d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
    return d


def node_contexts(model, node_df, node_intr, gammas, pre, nbr_std, dev):
    """ctx and log Pdet for the node bank at each gamma in `gammas` (kept on device)."""
    out = {}
    for t in gammas:
        c, pd_ = scene_context(model, node_df, node_intr, (0.0, float(t)), pre, nbr_std, dev)
        out[t] = (c, torch.log(pd_))
    return out


def part1(model, xhat, node_df, node_intr, pre, nbr_std, dev, delta, chunk, gal_chunk):
    """Deficit attribution at one bank size and one stencil width."""
    K = len(node_df)
    n = xhat.shape[0]
    ctxs = node_contexts(model, node_df, node_intr, (0.0, +delta, -delta),
                         pre, nbr_std, dev)

    cnt = np.zeros(NB, dtype=np.int64)                     # (i,k) pairs per bin
    wsum = np.zeros(NB, dtype=np.float64)                  # total weight per bin
    varw_bin = np.zeros(NB, dtype=np.float64)              # sum_i sum_{k in b} w (d1-s)^2
    hist_u = np.zeros((NB, LG_N), dtype=np.int64)          # |phi'| hist, node-uniform
    hist_w = np.zeros((NB, LG_N), dtype=np.float64)        # |phi'| hist, weight-weighted
    s_part = np.zeros((n, NB), dtype=np.float64)           # per-galaxy per-bin partial score
    ess = np.zeros(n, dtype=np.float64)
    best = np.zeros(n, dtype=np.float64)                   # max_k phi0: how well the BANK
                                                           # explains this galaxy at all
    wcum_lvl = np.zeros((n, 3), dtype=np.float64)          # deficit at 50/90/99% cum weight
    sum_d1 = 0.0
    sum_d1sq = 0.0
    npair = 0

    for s0 in range(0, n, gal_chunk):
        e0 = min(s0 + gal_chunk, n)
        xb = xhat[s0:e0]
        p0 = phi_block(model, xb, ctxs[0.0][0], ctxs[0.0][1], chunk)
        pp = phi_block(model, xb, ctxs[+delta][0], ctxs[+delta][1], chunk)
        pm = phi_block(model, xb, ctxs[-delta][0], ctxs[-delta][1], chunk)
        d1 = (pp - pm) / (2.0 * delta)
        del pp, pm
        mx = p0.max(axis=1, keepdims=True)
        best[s0:e0] = mx[:, 0]
        deficit = p0 - mx
        w = posterior_weights(p0)
        del p0, mx
        s_i = np.sum(w * d1, axis=1)
        ess[s0:e0] = 1.0 / np.sum(w ** 2, axis=1)

        b = bin_of(deficit)
        li = lg_idx(d1)
        flat = (b * LG_N + li).ravel()
        hist_u += np.bincount(flat, minlength=NB * LG_N).reshape(NB, LG_N)
        hist_w += np.bincount(flat, weights=w.ravel(),
                              minlength=NB * LG_N).reshape(NB, LG_N)
        cnt += np.bincount(b.ravel(), minlength=NB)
        wsum += np.bincount(b.ravel(), weights=w.ravel(), minlength=NB)

        contrib = w * (d1 - s_i[:, None]) ** 2
        sc = w * d1
        for bb in range(NB):
            m_ = (b == bb)
            if not m_.any():
                continue
            s_part[s0:e0, bb] = np.where(m_, sc, 0.0).sum(axis=1)
            varw_bin[bb] += float(np.where(m_, contrib, 0.0).sum())
        sum_d1 += float(d1.sum())
        sum_d1sq += float((d1 ** 2).sum())
        npair += d1.size

        # deficit level at which the cumulative weight reaches 50 / 90 / 99 %.
        # deficit = log(w) + log(sum_k exp(deficit)), so sorting w descending sorts the
        # deficit descending too -- no argsort of the full block needed.
        wsort = -np.sort(-w, axis=1)
        logS = np.log(np.sum(np.exp(deficit), axis=1))
        cw = np.cumsum(wsort, axis=1)
        for j, q in enumerate((0.5, 0.9, 0.99)):
            k = np.argmax(cw >= q, axis=1)
            wcum_lvl[s0:e0, j] = (np.log(np.maximum(wsort[np.arange(e0 - s0), k], 1e-300))
                                  + logS)
        del d1, w, deficit, contrib, sc, wsort, cw

    for c, _ in ctxs.values():
        del c
    del ctxs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    s_tot = s_part.sum(axis=1)
    var_s = float(np.var(s_tot))
    # exact additive attribution of Var(s): sum_b Cov(s^(b), s) = Var(s)
    sc_c = s_part - s_part.mean(axis=0, keepdims=True)
    st_c = s_tot - s_tot.mean()
    cov_bin = (sc_c * st_c[:, None]).mean(axis=0)

    return dict(K=K, n=n, cnt=cnt, wsum=wsum, varw_bin=varw_bin / n,
                hist_u=hist_u, hist_w=hist_w, cov_bin=cov_bin,
                var_s=var_s, varw_tot=float(varw_bin.sum() / n),
                s_tot=s_tot, ess=float(ess.mean()), ess_i=ess, best=best,
                wcum=wcum_lvl.mean(axis=0),
                d1_mean=sum_d1 / npair,
                d1_sd=float(np.sqrt(max(sum_d1sq / npair - (sum_d1 / npair) ** 2, 0.0))))


def report1(r, delta):
    print(f"\n=== PART 1: DEFICIT ATTRIBUTION   K={r['K']:,}  N_gal={r['n']:,}  "
          f"delta={delta}  ESS={r['ess']:.1f}")
    print(f"    Var(s) [across galaxies] = {r['var_s']:.3f}     "
          f"<Var_w(phi')> [within galaxy] = {r['varw_tot']:.3f}")
    print(f"    pooled phi' over all (i,k): mean {r['d1_mean']:+.4f}  sd {r['d1_sd']:.4f}")
    print(f"    deficit at cumulative weight 50%/90%/99%: "
          f"{r['wcum'][0]:.2f} / {r['wcum'][1]:.2f} / {r['wcum'][2]:.2f}")
    tot_pairs = r["cnt"].sum()
    vw = r["varw_bin"]
    cb = r["cov_bin"]
    print(f"    {'deficit bin':>12} {'nodes':>8} {'weight':>9} {'med|phi1|':>10} "
          f"{'p99|phi1|':>10} {'wmed|phi1|':>11} {'VarW share':>11} {'VarS share':>11} "
          f"{'VarS abs':>10}")
    for b in range(NB):
        if r["cnt"][b] == 0:
            print(f"    {LAB[b]:>12} {'--':>8}")
            continue
        med = quant_from_hist(r["hist_u"][b], 0.5)
        p99 = quant_from_hist(r["hist_u"][b], 0.99)
        wmed = quant_from_hist(r["hist_w"][b], 0.5)
        print(f"    {LAB[b]:>12} {r['cnt'][b] / tot_pairs:>8.2%} "
              f"{r['wsum'][b] / r['n']:>9.4f} {med:>10.3f} {p99:>10.2f} {wmed:>11.3f} "
              f"{vw[b] / max(vw.sum(), 1e-30):>11.2%} "
              f"{cb[b] / max(abs(r['var_s']), 1e-30):>11.2%} {cb[b]:>10.3f}")
    print(f"    {'TOTAL':>12} {1.0:>8.2%} {r['wsum'].sum() / r['n']:>9.4f} "
          f"{'':>10} {'':>10} {'':>11} {vw.sum() / max(vw.sum(), 1e-30):>11.2%} "
          f"{cb.sum() / max(abs(r['var_s']), 1e-30):>11.2%} {cb.sum():>10.3f}")
    # Is Var(s) itself carried by a handful of GALAXIES?  Free from the arrays in hand, and
    # it bears directly on the K-instability: if a few galaxies dominate the denominator,
    # their bank-sensitivity is the whole of it.
    s = r["s_tot"]
    c2 = (s - s.mean()) ** 2
    o = np.sort(c2)[::-1]
    tot = o.sum()
    n = len(s)
    kurt = float(np.mean(c2 ** 2) / max(np.mean(c2) ** 2, 1e-30))
    k1 = max(1, n // 100)
    k01 = max(1, n // 1000)
    tr = np.abs(s - s.mean()) <= np.sort(np.abs(s - s.mean()))[n - k1 - 1]
    print(f"    per-GALAXY concentration of Var(s): kurtosis {kurt:.1f} (Gaussian = 3)   "
          f"top 1% of galaxies carry {o[:k1].sum() / tot:.1%}, top 0.1% carry "
          f"{o[:k01].sum() / tot:.1%}")
    print(f"      Var(s) = {r['var_s']:.3f}   with the top 1% of |s-<s>| trimmed: "
          f"{float(np.var(s[tr])):.3f}   |s| p50 {np.percentile(np.abs(s), 50):.3f} "
          f"p99 {np.percentile(np.abs(s), 99):.2f} max {np.abs(s).max():.1f}")
    # ARE the dominant galaxies the POORLY-EXPLAINED ones?  This is the galaxy-space
    # analogue of the node-space deficit question above, and it is the natural follow-up
    # if the per-galaxy concentration is high.
    e_i, b_i = r["ess_i"], r["best"]
    q = np.quantile(b_i, np.linspace(0, 1, 6))
    q[0] -= 1e-9
    gb = np.clip(np.searchsorted(q[1:-1], b_i, side="left"), 0, 4)
    print(f"      galaxies binned by how well the BANK explains them (max_k phi0), "
          f"quintiles; share of Var(s) is Cov(s restricted, s)/Var(s):")
    print(f"        {'quintile of max_k phi0':>24} {'range':>18} {'<ESS>':>8} "
          f"{'med|s|':>9} {'p99|s|':>9} {'Var(s) share':>13}")
    sc = s - s.mean()
    for j in range(5):
        m_ = gb == j
        if not m_.any():
            continue
        share = float(np.sum(sc[m_] ** 2) / np.sum(sc ** 2))
        print(f"        {j + 1:>24} {f'{q[j]:.1f}..{q[j+1]:.1f}':>18} "
              f"{e_i[m_].mean():>8.1f} {np.percentile(np.abs(s[m_]), 50):>9.3f} "
              f"{np.percentile(np.abs(s[m_]), 99):>9.2f} {share:>13.2%}")
    print(f"      corr(|s|, max_k phi0) = {float(np.corrcoef(np.abs(s), b_i)[0, 1]):+.4f}   "
          f"corr(|s|, log ESS) = "
          f"{float(np.corrcoef(np.abs(s), np.log(np.maximum(e_i, 1e-30)))[0, 1]):+.4f}",
          flush=True)


def part2(model, xhat, node_df, node_intr, pre, nbr_std, dev, deltas, chunk, gal_chunk):
    """Var_w(phi') and Var(s) as a function of the stencil half-width."""
    K = len(node_df)
    n = xhat.shape[0]
    gam = [0.0]
    for d in deltas:
        gam += [+d, -d]
    ctxs = node_contexts(model, node_df, node_intr, tuple(gam), pre, nbr_std, dev)

    nd = len(deltas)
    s_all = np.zeros((n, nd), dtype=np.float64)
    varw = np.zeros(nd, dtype=np.float64)
    absmean = np.zeros(nd, dtype=np.float64)
    p999 = np.zeros((nd, LG_N), dtype=np.int64)
    # pairwise (i,k)-level comparison between successive deltas
    pair_ss = np.zeros(nd - 1, dtype=np.float64)     # sum (d1_a - d1_b)^2
    pair_wss = np.zeros(nd - 1, dtype=np.float64)    # weighted version
    pair_aa = np.zeros(nd - 1, dtype=np.float64)
    pair_bb = np.zeros(nd - 1, dtype=np.float64)
    pair_ab = np.zeros(nd - 1, dtype=np.float64)
    # the same disagreement resolved by DEFICIT BIN: does the finite difference fail to
    # converge everywhere, or only where the flow is being queried in its tail?
    b_ss = np.zeros((nd - 1, NB), dtype=np.float64)
    b_bb = np.zeros((nd - 1, NB), dtype=np.float64)
    b_wss = np.zeros((nd - 1, NB), dtype=np.float64)
    b_wbb = np.zeros((nd - 1, NB), dtype=np.float64)
    b_cnt = np.zeros(NB, dtype=np.int64)
    npair = 0

    for s0 in range(0, n, gal_chunk):
        e0 = min(s0 + gal_chunk, n)
        xb = xhat[s0:e0]
        p0 = phi_block(model, xb, ctxs[0.0][0], ctxs[0.0][1], chunk)
        w = posterior_weights(p0)
        bidx = bin_of(p0 - p0.max(axis=1, keepdims=True))
        b_cnt += np.bincount(bidx.ravel(), minlength=NB)
        del p0
        d1s = []
        for j, d in enumerate(deltas):
            pp = phi_block(model, xb, ctxs[+d][0], ctxs[+d][1], chunk)
            pm = phi_block(model, xb, ctxs[-d][0], ctxs[-d][1], chunk)
            d1 = (pp - pm) / (2.0 * d)
            del pp, pm
            s_i = np.sum(w * d1, axis=1)
            s_all[s0:e0, j] = s_i
            varw[j] += float(np.sum(w * (d1 - s_i[:, None]) ** 2))
            absmean[j] += float(np.abs(d1).sum())
            p999[j] += np.bincount(lg_idx(d1).ravel(), minlength=LG_N)
            d1s.append(d1)
        for j in range(nd - 1):
            a, b = d1s[j], d1s[j + 1]
            diff = a - b
            pair_ss[j] += float((diff ** 2).sum())
            pair_wss[j] += float((w * diff ** 2).sum())
            pair_aa[j] += float((a ** 2).sum())
            pair_bb[j] += float((b ** 2).sum())
            pair_ab[j] += float((a * b).sum())
            fl = bidx.ravel()
            b_ss[j] += np.bincount(fl, weights=(diff ** 2).ravel(), minlength=NB)
            b_bb[j] += np.bincount(fl, weights=(b ** 2).ravel(), minlength=NB)
            b_wss[j] += np.bincount(fl, weights=(w * diff ** 2).ravel(), minlength=NB)
            b_wbb[j] += np.bincount(fl, weights=(w * b ** 2).ravel(), minlength=NB)
        npair += (e0 - s0) * K
        del d1s, w, bidx

    for c, _ in ctxs.values():
        del c
    del ctxs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n=== PART 2: STENCIL CHECK   K={K:,}  N_gal={n:,}")
    print(f"    {'delta':>9} {'Var_w(phi1)':>12} {'Var(s)':>10} {'<s>':>10} "
          f"{'mean|phi1|':>11} {'p50|phi1|':>10} {'p99|phi1|':>10}")
    for j, d in enumerate(deltas):
        print(f"    {d:>9.4f} {varw[j] / n:>12.3f} {float(np.var(s_all[:, j])):>10.3f} "
              f"{float(np.mean(s_all[:, j])):>10.4f} {absmean[j] / npair:>11.4f} "
              f"{quant_from_hist(p999[j], 0.5):>10.3f} "
              f"{quant_from_hist(p999[j], 0.99):>10.2f}")
    print(f"\n    (i,k)-level agreement between successive stencil widths "
          f"(rms of the DIFFERENCE over the rms of phi'):")
    for j in range(nd - 1):
        rms_d = np.sqrt(pair_ss[j] / npair)
        rms_a = np.sqrt(pair_aa[j] / npair)
        rms_b = np.sqrt(pair_bb[j] / npair)
        corr = pair_ab[j] / np.sqrt(max(pair_aa[j] * pair_bb[j], 1e-30))
        wrms_d = np.sqrt(pair_wss[j] / n)
        print(f"      delta {deltas[j]:.4f} vs {deltas[j+1]:.4f}: "
              f"rms(diff)={rms_d:.4f}  rms(a)={rms_a:.4f}  rms(b)={rms_b:.4f}  "
              f"rel={rms_d / max(rms_b, 1e-30):.4f}  corr={corr:.6f}   "
              f"WEIGHTED rms(diff)={wrms_d:.4f}")
    print("\n    the SAME disagreement, resolved by deficit bin.  `rel` is the node-uniform "
          "rms(diff)/rms(phi'); `wrel` is the posterior-weighted version.")
    hdr = "  ".join(f"{L:>11}" for L in LAB)
    print(f"      {'pair':>18}  {hdr}")
    for j in range(nd - 1):
        rel = np.sqrt(np.divide(b_ss[j], np.maximum(b_bb[j], 1e-300)))
        wrel = np.sqrt(np.divide(b_wss[j], np.maximum(b_wbb[j], 1e-300)))
        tag = f"{deltas[j]:.4f}v{deltas[j+1]:.4f}"
        print(f"      {tag:>18}  " + "  ".join(f"{v:>11.4f}" for v in rel))
        print(f"      {'  (weighted)':>18}  " + "  ".join(f"{v:>11.4f}" for v in wrel))
    print(f"      {'node fraction':>18}  "
          + "  ".join(f"{v:>10.2%}" for v in b_cnt / max(b_cnt.sum(), 1)))
    print(f"\n    across-galaxy corr of s between stencil widths:")
    for j in range(nd - 1):
        c = float(np.corrcoef(s_all[:, j], s_all[:, j + 1])[0, 1])
        print(f"      s(delta={deltas[j]:.4f}) vs s(delta={deltas[j+1]:.4f}): corr={c:.5f}")
    # Richardson on the two finest widths, if the ratio is 2
    if nd >= 2 and abs(deltas[-2] / deltas[-1] - 2.0) < 1e-9:
        sr = (4.0 * s_all[:, -1] - s_all[:, -2]) / 3.0
        print(f"\n    Richardson s from delta={deltas[-2]:.4f}/{deltas[-1]:.4f}: "
              f"Var(s)={float(np.var(sr)):.3f}  <s>={float(np.mean(sr)):+.4f}  "
              f"(vs Var(s)={float(np.var(s_all[:, -1])):.3f} at the finest width alone)")
    print(flush=True)
    return s_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="2000,20000", help="bank sizes for PART 1")
    ap.add_argument("--delta", type=float, default=0.01, help="stencil width for PART 1")
    ap.add_argument("--deltas", default="0.02,0.01,0.005,0.0025",
                    help="stencil widths for PART 2 (descending)")
    ap.add_argument("--stencil-k", type=int, default=20000)
    ap.add_argument("--stencil-ngal", type=int, default=0,
                    help="subsample of DETECTED galaxies for PART 2; 0 = all")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--gal-chunk", type=int, default=0, help="0 = auto from K")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-part2", action="store_true")
    ap.add_argument("--skip-part1", action="store_true")
    args = ap.parse_args()

    nodes = [int(x) for x in args.n_nodes.split(",")]
    deltas = [float(x) for x in args.deltas.split(",")]
    kmax = max(nodes + [args.stencil_k])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    print(f"device {dev}  primary_only_shear={meta.get('primary_only_shear')}", flush=True)

    # --- identical slicing to diag5c_repro.py so the numbers are comparable -------------
    rows = load_rows(args.catalogue, (kmax + args.n_gal) * 4)
    pool = rows.iloc[:kmax].reset_index(drop=True)
    gal_df = rows.iloc[kmax:kmax + args.n_gal].reset_index(drop=True)
    print(f"rows: {len(pool):,} node pool + {len(gal_df):,} galaxies (disjoint)", flush=True)
    gal_intr, pool_intr = intr_of(gal_df), intr_of(pool)

    # --- the gamma = 0 data leg, same seed and same draw order as diag5c_repro ----------
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
    print(f"  gamma=0: kept {int(keep.sum()):,}/{len(gal_df):,}  "
          f"(diag5c_repro job 15459872 reports 9,918)", flush=True)
    del ctx_g, pdet_g, xh

    for K in ([] if args.skip_part1 else nodes):
        t0 = time.time()
        node_df = pool.iloc[:K].reset_index(drop=True)
        node_intr = {k: v[:K] for k, v in pool_intr.items()}
        gc = args.gal_chunk or max(1, int(2.0e7 // K))
        r = part1(model, xhat, node_df, node_intr, pre, nbr_std, dev,
                  args.delta, args.chunk, gc)
        report1(r, args.delta)
        print(f"    [{time.time() - t0:.0f}s]", flush=True)

    if not args.skip_part2:
        t0 = time.time()
        K = args.stencil_k
        node_df = pool.iloc[:K].reset_index(drop=True)
        node_intr = {k: v[:K] for k, v in pool_intr.items()}
        xs = xhat if not args.stencil_ngal else xhat[:args.stencil_ngal]
        gc = args.gal_chunk or max(1, int(2.0e7 // K))
        part2(model, xs, node_df, node_intr, pre, nbr_std, dev, deltas, args.chunk, gc)
        print(f"    [{time.time() - t0:.0f}s]", flush=True)

    print("\n  READING GUIDE.")
    print("  PART 1: if the Var(s) share concentrates in the [0,-1] / (-1,-3] bins the flow")
    print("  tail is EXONERATED -- the spread comes from nodes that explain the galaxy well.")
    print("  If it concentrates at deficits beyond -10 while those bins carry little weight,")
    print("  the tail is implicated: large |phi'| there is being let through by weight that")
    print("  is small but not small enough.")
    print("  PART 2: Var growing as delta shrinks => the central difference has not")
    print("  converged and the spread is partly a stencil artefact.  Flat => the spread is a")
    print("  real property of the flow at this bank.  NOTE this part does not measure I, so")
    print("  a delta trend shared by I and Var(s) would leave m untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
