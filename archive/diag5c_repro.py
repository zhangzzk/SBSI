#!/usr/bin/env python -B
"""Split Var(s) into SIGNAL / REPRODUCIBLE-BUT-SHEAR-BLIND / BANK NOISE, in one run.

cont.168 left the diagnosis mis-aimed.  The 5C attenuation is `m = I / Var(s) - 1` with
I = d<s>/dgamma the genuine information, measured 5.83 against Var(s) = 62.04 -> 9.4%
recovery.  The numerator is sound (0.13 sigma from zero at g=0), so the whole failure is an
inflated denominator -- and the denominator is QUADRATIC in `s`, so mean-zero per-galaxy
error ADDS there instead of cancelling as it does in the linear numerator.  Classic
attenuation / regression dilution.

The question that decides everything: how much of the excess is BANK NOISE (removable by
sampling) versus REPRODUCIBLE (the same for any bank, hence a model-vs-data mismatch)?
A cross-run combination of the cont.167 half-bank correlation with the cont.167 slope
suggested only ~25 of the 62 is noise and ~31 is reproducible-but-shear-blind, which would
mean a perfect bank still leaves m = -84%.  That combination mixed two jobs at different
settings and is NOT certified.  This measures all three terms at ONE setting.

METHOD.  The two half-banks are column slices of the same phi array, so reproducibility is
free: with A and B disjoint node sets, their Monte-Carlo errors are independent GIVEN the
data, so over galaxies

    Cov(s_A, s_B) = Var(t)          # t = the reproducible part, at half-bank size
    Var(s_A) - Cov(s_A, s_B) = sigma^2_half
    sigma^2_full = sigma^2_half / 2                 # noise falls as 1/K -- CHECKED on the ladder
    Var(s_full) - sigma^2_full = Var(t_full)
    blind = Var(t_full) - I                          # reproducible, carries no shear

I comes from the slope of <s> against the TRUE shear, which is denominator-free.  If `blind`
is ~0 the story is pure sampling and a better bank wins; if `blind` dominates, no bank ever
will, and 5C fails for a model reason.

CHANNEL ATTRIBUTION.  log Pdet enters phi as a per-node constant, so its gamma-derivative is
galaxy-INDEPENDENT and `s` splits exactly, at fixed posterior weights:

    s = s_flow + s_det,   d1_det[k] galaxy-independent,   d1_flow = d1_full - d1_det

so Var(s) decomposes into the two channels plus their covariance, and each gets its own
reproducible/noise split.  This says WHERE the blind variance lives without a second run.

Usage:
    python scripts/diag5c_repro.py --n-gal 20000 --n-nodes 2000,6000,20000 --gamma 0.05
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


def s_of(phi0, d1, cols=None):
    """Self-normalised posterior mean of `d1`, optionally on a column subset (a half-bank)."""
    p = phi0 if cols is None else phi0[:, cols]
    d = d1 if cols is None else (d1[:, cols] if d1.ndim == 2 else d1[cols])
    w = posterior_weights(p)
    return np.sum(w * d, axis=1) if d.ndim == 2 else w @ d


def boot(fn, n, n_boot, seed):
    """Bootstrap `fn(idx)` over galaxies; returns (value, se)."""
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="2000,6000,20000")
    ap.add_argument("--gamma", type=float, default=0.05, help="the non-zero leg, for the slope")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    nodes = [int(x) for x in args.n_nodes.split(",")]
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

    # ---- data at both gammas, SAME galaxies and SAME seed: only the shear differs -------
    xhats = {}
    for g in (0.0, float(args.gamma)):
        ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, (0.0, g), pre, nbr_std, dev)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            xh = model.mean_flow.sample(ctx_g, n_samples=1)
            if xh.dim() == 3:
                xh = xh[:, 0, :]
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
        xhats[g] = xh[keep]
        print(f"  gamma={g:+.3f}: kept {int(keep.sum()):,}/{len(gal_df):,}", flush=True)
        del ctx_g, pdet_g

    print(f"\ntrue gamma leg = {args.gamma:+.4f}, delta = {d}\n"
          f"  I  = d<s>/dgamma (denominator-free).  Var(s) at gamma=0.\n"
          f"  blind = reproducible variance that carries NO shear = Var(t_full) - I.\n", flush=True)

    for K in nodes:
        node_df = pool.iloc[:K].reset_index(drop=True)
        node_intr = {k: v[:K] for k, v in pool_intr.items()}
        half = K // 2
        A, B = np.arange(half), np.arange(half, 2 * half)

        # phi for all K nodes at the three stencil points, reused by every variant
        store = {}
        for g, xh in xhats.items():
            ph, lpd = {}, {}
            for t in (0.0, +d, -d):
                c, pdn = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
                lpd[t] = torch.log(pdn)
                ph[t] = phi_block(model, xh, c, lpd[t], args.chunk)
                del c, pdn
            d1 = (ph[+d] - ph[-d]) / (2 * d)
            # log Pdet is a per-NODE constant in phi -> its derivative is galaxy-independent
            d1_det = ((lpd[+d] - lpd[-d]) / (2 * d)).double().cpu().numpy()
            d1_flow = d1 - d1_det[None, :]
            store[g] = dict(phi0=ph[0.0], d1=d1, d1_flow=d1_flow, d1_det=d1_det)
            del ph, lpd

        z, gz = store[0.0], store[float(args.gamma)]
        s0 = s_of(z["phi0"], z["d1"])
        sg = s_of(gz["phi0"], gz["d1"])
        sA = s_of(z["phi0"], z["d1"], A)
        sB = s_of(z["phi0"], z["d1"], B)

        I = (float(np.mean(sg)) - float(np.mean(s0))) / args.gamma
        var_full = float(np.var(s0))
        cov_ab = float(np.cov(sA, sB)[0, 1])
        var_a = float(np.var(sA))
        sig2_half = var_a - cov_ab
        sig2_full = sig2_half / 2.0
        var_t = var_full - sig2_full
        blind = var_t - I

        n = len(s0)

        def _dec(idx):
            a, b, f = sA[idx], sB[idx], s0[idx]
            s2h = float(np.var(a)) - float(np.cov(a, b)[0, 1])
            return float(np.var(f)) - s2h / 2.0
        se_vt = boot(_dec, n, args.n_boot, args.seed)
        se_I = boot(lambda i: (float(np.mean(sg[i])) - float(np.mean(s0[i]))) / args.gamma,
                    n, args.n_boot, args.seed)

        ess = float(np.mean(1.0 / np.sum(posterior_weights(z["phi0"]) ** 2, axis=1)))
        print(f"K={K:>6,}  ESS={ess:6.1f}  corr(sA,sB)={cov_ab / np.sqrt(var_a * np.var(sB)):.4f}")
        print(f"          Var(s)={var_full:8.2f}   =  I {I:7.2f}+-{se_I:.2f}"
              f"  +  blind {blind:8.2f}  +  noise {sig2_full:8.2f}")
        print(f"          Var(t_full)={var_t:8.2f}+-{se_vt:.2f}   "
              f"m now = {I / var_full - 1:+7.2%}   m if noise removed = {I / var_t - 1:+7.2%}")

        # ---- which channel carries the blind variance -------------------------------
        sf0, sd0 = s_of(z["phi0"], z["d1_flow"]), s_of(z["phi0"], z["d1_det"])
        sfg, sdg = s_of(gz["phi0"], gz["d1_flow"]), s_of(gz["phi0"], gz["d1_det"])
        sfA, sfB = s_of(z["phi0"], z["d1_flow"], A), s_of(z["phi0"], z["d1_flow"], B)
        sdA, sdB = s_of(z["phi0"], z["d1_det"], A), s_of(z["phi0"], z["d1_det"], B)
        for nm, x0, xg, xa, xb in (("flow", sf0, sfg, sfA, sfB), ("det ", sd0, sdg, sdA, sdB)):
            v, c, va = float(np.var(x0)), float(np.cov(xa, xb)[0, 1]), float(np.var(xa))
            vt = v - (va - c) / 2.0
            Ic = (float(np.mean(xg)) - float(np.mean(x0))) / args.gamma
            print(f"            {nm}: Var={v:11.5f}  I={Ic:10.5f}  Var(t)={vt:11.5f}  "
                  f"blind={vt - Ic:11.5f}")
        print(f"            cross-channel Cov = {float(np.cov(sf0, sd0)[0, 1]):11.5f}   "
              f"rms(d1_det)={float(np.sqrt(np.mean(z['d1_det'] ** 2))):.5f}   "
              f"sigma2_half={sig2_half:9.3f}  (must fall like 1/K for the split to be clean)",
              flush=True)
        del store, z, gz

    print("\n  blind ~ 0  => pure sampling; a better/bigger bank wins and 5C is salvageable.")
    print("  blind dominant and FLAT in K => a model-vs-data mismatch no bank can fix, and")
    print("  the channel lines say whether it enters through the flow or through detection.")
    print("  Note sigma^2_full = sigma^2_half/2 ASSUMES noise ~ 1/K; the K ladder tests that")
    print("  assumption directly -- if `noise` does not fall like 1/K the split is not clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
