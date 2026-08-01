#!/usr/bin/env python -B
"""PROBE B, third leg: do the SAME weights carry the shear signal at all?

Probe B (job 15460992) used the node's true `e1_p` as the tame integrand, per the probe
spec.  The shear in this harness is applied on AXIS 2 (`g = (0, gamma)`), so `e1_p` is the
component ORTHOGONAL to the shear and its measured near-zero response is a null, not a
statement about the signal.  This script measures the PARALLEL component, `e2_p`, and its
response to a genuine shear in the data, with everything else identical.

    s_i(g_data) = E_w[ e2_p ]   with  w propto exp(phi(0))  on the SAME node bank,
    response    = ( <s>(gamma) - <s>(0) ) / gamma

If the weights are informative about the sheared shape, this must be positive and of order
the shrinkage slope of `E_w[e2p]` on the galaxy's own true `e2p`.  A positive response here
with the same weights that give phi' its tiny `I` says the signal IS in the posterior and
phi' is failing to extract it -- the integrand, not the sampling.

Both legs are evaluated on a COMMON galaxy mask (the gamma=0 detection draw) so the
response is paired galaxy-by-galaxy; the estimator's own per-leg masks are reported too.

Usage:
    python scripts/diag5c_probeB3_response.py --n-gal 20000 --n-nodes 20000 --gamma 0.05
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
    ap.add_argument("--n-nodes", default="6000,20000")
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    nodes = [int(x) for x in args.n_nodes.split(",")]
    kmax = 20000                      # SAME pool/galaxy split as diag5c_repro at its kmax
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

    xhats, masks = {}, {}
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
        masks[g] = keep.cpu().numpy().astype(bool)
        xhats[(g, "own")] = xh[keep]
        xhats[(g, "common")] = xh[torch.as_tensor(masks[0.0], device=dev)]
        print(f"  gamma={g:+.3f}: own mask {int(keep.sum()):,}, "
              f"common mask {int(masks[0.0].sum()):,}", flush=True)
        del ctx_g, pdet_g

    m0 = masks[0.0]
    tgt_gal = {"e1p": np.asarray(gal_intr["e1p"], float)[m0],
               "e2p": np.asarray(gal_intr["e2p"], float)[m0]}

    for K in nodes:
        node_df = pool.iloc[:K].reset_index(drop=True)
        node_intr = {k: v[:K] for k, v in pool_intr.items()}
        gk = {"e1p": np.asarray(pool_intr["e1p"], float)[:K],
              "e2p": np.asarray(pool_intr["e2p"], float)[:K]}
        print(f"\n================ K = {K:,} ================", flush=True)

        c0, pd0 = scene_context(model, node_df, node_intr, (0.0, 0.0), pre, nbr_std, dev)
        lpd = torch.log(pd0)
        res = {}
        for tag in ("own", "common"):
            for g in (0.0, float(args.gamma)):
                p0 = phi_block(model, xhats[(g, tag)], c0, lpd, args.chunk)
                w = posterior_weights(p0)
                del p0
                res[(tag, g)] = {k: w @ v for k, v in gk.items()}
                if tag == "common" and g == 0.0:
                    for k in ("e1p", "e2p"):
                        sv = res[(tag, g)][k]
                        print(f"  {k}: corr(E_w[{k}], own true {k}) = "
                              f"{corr(sv, tgt_gal[k]):+.4f}   shrinkage slope = "
                              f"{float(np.polyfit(tgt_gal[k], sv, 1)[0]):+.4f}   "
                              f"sd {sv.std():.4f} vs prior sd {gk[k].std():.4f}")
                del w
        del c0, pd0, lpd

        for tag in ("common", "own"):
            for k in ("e1p", "e2p"):
                a, b = res[(tag, 0.0)][k], res[(tag, float(args.gamma))][k]
                if tag == "common":
                    dmean = float(np.mean(b - a))
                    se = float(np.std(b - a) / np.sqrt(len(a)))   # paired
                else:
                    dmean = float(np.mean(b)) - float(np.mean(a))
                    se = float(np.sqrt(np.var(b) / len(b) + np.var(a) / len(a)))
                lab = "PARALLEL to shear" if k == "e2p" else "orthogonal   "
                print(f"  [{tag:>6} mask] d<E_w[{k}]>/dgamma = {dmean / args.gamma:+8.4f}"
                      f" +- {se / args.gamma:.4f}   ({lab})")
        del res

    print("\nREAD: a clearly positive d<E_w[e2p]>/dgamma with these weights says the shear "
          "signal IS present in the posterior; then phi' failing to deliver comparable "
          "information is an integrand problem, not a sampling problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
