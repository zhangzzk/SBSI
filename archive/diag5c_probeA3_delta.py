#!/usr/bin/env python -B
"""PROBE A, confound check: is the |phi'| tail a DERIVATIVE or a finite-difference artefact?

Probe A measured (job 15461075) that the unweighted |phi'| bulk is stationary in K while
the extreme grows, with a Hill tail index below 2, and (job 15461552) that the extreme is
supplied by a handful of shared node scenes.  Every one of those numbers came from ONE
stencil width, delta = 0.01.  A pooled max of |phi'| ~ 1e7 means a log-density difference
of ~2e5 nats between gamma = -0.01 and +0.01 for a single node, which is not credible as a
derivative of a smooth density; WORKLOG cont.164 already records that finite differences
did not converge on the V1 flow.

The discriminant is simple.  For a genuine derivative of a smooth curve, phi' is
delta-INDEPENDENT as delta -> 0.  For a jump or kink of size J inside the stencil, the
difference quotient is J / (2 delta) and every extreme scales like 1 / delta.  Same nodes,
same galaxies, same xhat, four stencil widths spanning a factor 8.

Reported per delta: |phi'| quantiles and max, the pooled Hill index, Var_w(phi'), and the
half-bank noise ladder <(s_A - s_B)^2>/2 with its K exponent -- so if the tail IS an
artefact we also see immediately whether removing it (by widening the stencil) restores
the 1/K averaging that job 15461552's clip test could not.

Usage:
    python scripts/diag5c_probeA3_delta.py --n-gal 20000 --deltas 0.02,0.01,0.005,0.0025
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
    x = x[np.isfinite(x) & (x > 0)]
    k = max(2, int(frac * x.size))
    top = np.sort(x)[-(k + 1):]
    return float(k / np.sum(np.log(top[1:] / top[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="1000,2000,6000,20000")
    ap.add_argument("--deltas", default="0.02,0.01,0.005,0.0025")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--row-chunk", type=int, default=384)
    ap.add_argument("--n-hgal", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    nodes = sorted(int(x) for x in args.n_nodes.split(","))
    deltas = [float(x) for x in args.deltas.split(",")]
    kmax = max(nodes)
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
    print(f"  gamma=0: kept {n:,}/{len(gal_df):,}  (same layout as jobs 15461075/15461552)",
          flush=True)
    del ctx_g, pdet_g, xh

    def phi_at(t):
        c, pdn = scene_context(model, pool, pool_intr, (0.0, t), pre, nbr_std, dev)
        out = phi_block(model, xhat, c, torch.log(pdn), args.chunk)
        del c, pdn
        return out

    phi0 = phi_at(0.0)
    rng = np.random.default_rng(args.seed)
    hrows = np.sort(rng.choice(n, size=min(args.n_hgal, n), replace=False))
    logk = np.log(np.array(nodes, dtype=np.float64))
    print(f"  phi0 ready {phi0.shape}\n", flush=True)

    print("A jump J inside the stencil gives phi' = J/(2 delta): every column below then")
    print("scales like 1/delta.  A genuine derivative is delta-INDEPENDENT.\n")
    print(f"{'delta':>8} {'p50':>9} {'p99':>10} {'p99.9':>11} {'max/gal':>12} "
          f"{'max pooled':>13} {'alpha_pool':>10} {'Var_w(K=20k)':>13} {'med':>8}")
    tab = {}
    for d in deltas:
        pp, pm = phi_at(+d), phi_at(-d)
        np.subtract(pp, pm, out=pp)
        pp /= (2.0 * d)
        d1 = pp
        del pm

        sub = np.abs(d1[np.ix_(hrows, np.arange(kmax))])
        q = np.percentile(sub, [50.0, 99.0, 99.9], axis=1).mean(axis=1)
        gmax = float(np.mean(sub.max(axis=1)))
        pooled_max = 0.0
        for a in range(0, n, args.row_chunk):
            b = min(a + args.row_chunk, n)
            pooled_max = max(pooled_max, float(np.abs(d1[a:b]).max()))
        alpha = hill(sub.ravel(), 0.01)
        del sub

        # Var_w and the half-bank noise ladder
        varw = np.empty(n)
        s2, vs = [], []
        for K in nodes:
            half = K // 2
            sA = np.empty(n); sB = np.empty(n); sF = np.empty(n)
            for a in range(0, n, args.row_chunk):
                b = min(a + args.row_chunk, n)
                wF = posterior_weights(phi0[a:b, :K])
                sF[a:b] = np.einsum("ij,ij->i", wF, d1[a:b, :K])
                if K == kmax:
                    varw[a:b] = (np.einsum("ij,ij->i", wF, d1[a:b, :K] ** 2) - sF[a:b] ** 2)
                wA = posterior_weights(phi0[a:b, :half])
                wB = posterior_weights(phi0[a:b, half:2 * half])
                sA[a:b] = np.einsum("ij,ij->i", wA, d1[a:b, :half])
                sB[a:b] = np.einsum("ij,ij->i", wB, d1[a:b, half:2 * half])
            s2.append(float(np.mean((sA - sB) ** 2)) / 2.0)
            vs.append(float(np.var(sF)))
        tab[d] = (np.array(s2), np.array(vs))
        print(f"{d:>8.4f} {q[0]:>9.3f} {q[1]:>10.2f} {q[2]:>11.1f} {gmax:>12.1f} "
              f"{pooled_max:>13.3e} {alpha:>10.3f} {float(np.mean(varw)):>13.2f} "
              f"{float(np.median(varw)):>8.2f}", flush=True)
        del d1, varw

    print(f"\nHALF-BANK NOISE LADDER per delta   (cells sigma^2_half / Var(s))")
    print(f"{'delta':>8} " + " ".join(f"{'K=' + format(K, ','):>22}" for K in nodes)
          + f" {'exp r':>8} {'exp Var(s)':>11}")
    for d in deltas:
        s2, vs = tab[d]
        cells = [f"{s2[j]:10.3f}/{vs[j]:10.3f}" for j in range(len(nodes))]
        print(f"{d:>8.4f} " + " ".join(f"{x:>22}" for x in cells)
              + f" {ols_slope(logk, np.log(s2)):>+8.3f} {ols_slope(logk, np.log(vs)):>+11.3f}")
    print("\n  If phi' is a real derivative the whole table is delta-invariant.  If the tail")
    print("  is a stencil artefact, quantiles/max/Var_w scale like 1/delta while p50 does")
    print("  not, and the noise exponent may move toward -1 at the widest stencil.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
