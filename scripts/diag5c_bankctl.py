#!/usr/bin/env python -B
"""Does fact (b) survive removing the two setup defects?  L0 baseline / L1 dedup / L2 +true_cut.

cont.170 established that the §5C noise does NOT fall with bank size (exponent +0.14 over a
20x range, against -1 for Monte Carlo) but also found TWO uncontrolled defects sitting in
front of that diagnosis, both introduced by `diag5c_repro.py` and inherited by every probe:

  D1 BANK DUPLICATION.  The catalogue is pair-annotated: 8.14 rows share each `input_index`
     and within a group the PRIMARY is byte-identical (within-group sd of Re_input_p,
     axis_ratio_input_p, sersic_n_input_p, measured_x_image all exactly 0) -- only the
     annotated neighbour differs.  A 20,000-ROW bank is 2,456 DISTINCT SCENES.
     Measured ESS_row/ESS_scene = 4.27-4.47, flat in K.
  D2 NO true_cut.  `closure_v2_lagrangian.py` (lines 246-255) filters rows by the
     checkpoint's own `true_cut` before slicing, with the comment that the model was TRAINED
     only on passing rows so the density is unconstrained outside it.  `diag5c_repro.py` and
     `diag5c_shapegrid.py` DO NOT.  Every number in cont.168 and cont.169 therefore evaluates
     the flow out of domain, which is a live alternative origin for the alpha ~ 1 tail.

Neither defect can change a K-SCALING by itself -- both are K-independent -- so neither is a
ready-made explanation of (b).  But nothing about (b) has ever been measured without them,
and a structural verdict on §5C cannot rest on a pathology stacked on two known bugs.

PROTOCOL: the DISJOINT-EQUAL-BLOCK ladder (the only version carrying realisation error bars;
the nested ladder is retired -- its rungs are column prefixes of one another and cannot see
bank-realisation scatter).  For block size B, cut the pool into n = K/B DISJOINT blocks,
score every galaxy on each block independently, then

    noise(B) = mean over block PAIRS and galaxies of (s_m - s_m')^2 / 2

which assumes nothing about how the errors decompose -- unlike `Var(s_A) - Cov(s_A,s_B)`,
which is retired here for being asymmetric (15.38 vs 7.70 on the two halves at K=1,000).

READOUTS per leg: the ladder with realisation sd, its fitted exponent, the pooled Hill index
of |phi'| (alpha < 2 => no finite second moment => Var_w cannot converge), and the SAME
ladder for a tame integrand (the node's own true e1_p) as the healthy-behaviour reference.

OUTCOME MAP (fixed in advance, so this cannot be read after the fact):
  exponent goes clearly negative in L1/L2  -> the pathology was bank construction; §5C is
                                              fixable and the next step is a gamma != 0 leg;
  level drops ~4x but exponent stays ~+0.1 and Hill stays ~1
                                           -> duplication was a LEVEL effect only and (b) is
                                              a real variance-non-existence property of phi';
  nothing changes                          -> both confounds are irrelevant and the diagnosis
                                              must move to the deficit ~ 0 nodes.

Usage:
    python scripts/diag5c_bankctl.py --n-gal 20000 --k 10000 --blocks 250,500,1250,2500,5000
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


def hill(x, frac=0.01):
    """Hill tail index of |x| on its top `frac`.  alpha < 2 => no finite second moment."""
    a = np.sort(np.abs(np.asarray(x, float).ravel()))
    a = a[np.isfinite(a) & (a > 0)]
    k = max(10, int(frac * len(a)))
    top, xmin = a[-k:], a[-k - 1]
    return float(1.0 / np.mean(np.log(top / xmin)))


def block_ladder(phi0, d1, blocks, kmax):
    """Disjoint-equal-block noise ladder.  Returns {B: (noise, noise_sd, var_s, var_sd, n)}."""
    out = {}
    for B in blocks:
        n = kmax // B
        if n < 2:
            continue
        S = np.empty((n, phi0.shape[0]))
        for m in range(n):
            cols = slice(m * B, (m + 1) * B)
            w = posterior_weights(phi0[:, cols])
            # phi' is per (galaxy, node); a tame reference integrand is per node only
            S[m] = np.sum(w * d1[:, cols], axis=1) if d1.ndim == 2 else w @ d1[cols]
        pair, vs = [], [float(np.var(S[m])) for m in range(n)]
        for m in range(n):
            for mp in range(m + 1, n):
                pair.append(float(np.mean((S[m] - S[mp]) ** 2)) / 2.0)
        out[B] = (float(np.mean(pair)), float(np.std(pair)),
                  float(np.mean(vs)), float(np.std(vs)), n)
    return out


def expo(lad):
    """Least-squares exponent of noise vs block size, in logs."""
    if len(lad) < 2:
        return float("nan")
    b = np.log(np.array(sorted(lad)))
    y = np.log(np.array([lad[k][0] for k in sorted(lad)]))
    return float(np.polyfit(b, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--k", type=int, default=10000, help="node pool; blocks are cut from it")
    ap.add_argument("--blocks", default="250,500,1250,2500,5000")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--legs", default="L0,L1,L2")
    args = ap.parse_args()

    blocks = [int(x) for x in args.blocks.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    tc = meta.get("true_cut")
    print(f"device {dev}  primary_only_shear={meta.get('primary_only_shear')}  true_cut={tc}",
          flush=True)

    raw = load_rows(args.catalogue, args.max_rows)
    print(f"raw rows read: {len(raw):,}", flush=True)

    def build(leg):
        r = raw
        if leg == "L2" and tc is not None:
            re_min, mag_max = float(tc[0]), float(tc[1])
            k = ((r["Re_input_p"].to_numpy(float) > re_min)
                 & (r["r_input_p"].to_numpy(float) < mag_max))
            r = r[k].reset_index(drop=True)
            print(f"  [{leg}] true_cut Re>{re_min} & mag<{mag_max}: "
                  f"{int(k.sum()):,}/{len(k):,} = {k.mean():.1%}")
        if leg in ("L1", "L2"):
            before = len(r)
            r = r.drop_duplicates(subset=["input_index"], keep="first").reset_index(drop=True)
            print(f"  [{leg}] dedup on input_index: {before:,} -> {len(r):,} distinct scenes "
                  f"({before / max(len(r), 1):.2f} rows/scene)")
        need = args.k + args.n_gal
        if len(r) < need:
            print(f"  [{leg}] SKIPPED: only {len(r):,} usable rows, need {need:,}")
            return None, None
        return (r.iloc[:args.k].reset_index(drop=True),
                r.iloc[args.k:args.k + args.n_gal].reset_index(drop=True))

    def intr_of(df):
        d = {}
        d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
        d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
        return d

    d = args.delta
    for leg in args.legs.split(","):
        node_df, gal_df = build(leg)
        if node_df is None:
            continue
        gal_intr, node_intr = intr_of(gal_df), intr_of(node_df)

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
        xh, nk = xh[keep], int(keep.sum())
        del ctx_g, pdet_g
        print(f"  [{leg}] detected {nk:,}/{len(gal_df):,}", flush=True)

        ph = {}
        for t in (0.0, +d, -d):
            c, pdn = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
            ph[t] = phi_block(model, xh, c, torch.log(pdn), args.chunk)
            del c, pdn
        phi0 = ph[0.0]
        d1 = (ph[+d] - ph[-d]) / (2 * d)
        del ph

        ess = float(np.mean(1.0 / np.sum(posterior_weights(phi0) ** 2, axis=1)))
        h = hill(d1)
        print(f"\n[{leg}] pool={args.k:,} nodes  ESS(full pool)={ess:.1f}  "
              f"Hill(|phi'|, top 1%)={h:.3f}   (alpha<2 => no finite 2nd moment)")

        lad = block_ladder(phi0, d1, blocks, args.k)
        print(f"      {'block':>7} {'n':>3}  {'noise=<(sm-sm)^2>/2':>21}  {'Var(s) per block':>20}")
        for B in sorted(lad):
            nz, nzs, vs, vss, n = lad[B]
            print(f"      {B:>7,} {n:>3}  {nz:>10.3f} +- {nzs:<8.3f}  {vs:>9.3f} +- {vss:<8.3f}")
        print(f"      EXPONENT of noise vs block size: {expo(lad):+.3f}   "
              f"(Monte Carlo would be -1)")

        # tame reference: the node's own true e1_p, same weights, same blocks
        tame = np.asarray(node_intr["e1p"], float)
        ladt = block_ladder(phi0, tame, blocks, args.k)
        print(f"      TAME (true e1_p) exponent: {expo(ladt):+.3f}   "
              f"noise {min(ladt)}->{max(ladt)}: "
              f"{ladt[min(ladt)][0]:.5f} -> {ladt[max(ladt)][0]:.5f}", flush=True)
        del phi0, d1

    print("\n  Read the EXPONENT column against -1.  L0 reproduces the defective baseline;")
    print("  L1 removes bank duplication; L2 additionally keeps the flow inside its training")
    print("  domain.  The tame line is the healthy-behaviour reference measured on the SAME")
    print("  weights: if it averages down and phi' does not, the defect is the integrand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
