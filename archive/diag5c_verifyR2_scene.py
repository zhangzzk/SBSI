#!/usr/bin/env python -B
"""ADVERSARIAL CHECK of probe A, part 2: is the "6-10x formula failure" just the bank's
own 8.14x row duplication?

Probe A's headline replacement claim is that `Var_w(phi')/ESS` "is not merely mis-scaled,
it is the WRONG MODEL ... ESS is not the effective sample size", on the evidence that it
under-predicts the measured half-bank noise by 6-10x with a ratio stable in K.  Probe A
also lists, as confound 1, that the node pool holds 8.14 catalogue rows per `input_index`
and says that "cannot explain fact (b)" because it is a constant factor -- but it never
connects it to the 6-10x, which is ALSO a constant factor.

The connection is exact.  Rows sharing an `input_index` carry an IDENTICAL primary
(measured here: within-group sd of Re_input_p / axis_ratio_input_p / sersic_n_input_p /
measured_x_image is exactly 0) and differ only in which single neighbour is annotated, so
they are near-duplicate columns of phi.  For duplicated nodes with per-scene total weight
W_g,

    sum_k w_k^2 = (1/r) sum_g W_g^2   =>  ESS_row = r * ESS_scene       (r = copies/scene)
    Var_w(phi')                          is UNCHANGED by duplication
    =>  the surrogate Var_w / ESS_row is exactly r times too SMALL,

while the bank's real Monte-Carlo unit is the SCENE (the pool is a contiguous catalogue
prefix, so `input_index` groups are contiguous runs, mean length 8.14).  r ~ 8 against a
measured under-prediction of 4-10x is the whole effect.

Measured here, on the identical bank/galaxies/seed/delta as jobs 15459872 / 15461075 bank 0
/ 15461552, at gamma = 0:

  * ESS_row vs ESS_scene (= 1/sum_g W_g^2) per galaxy, and their ratio r_eff;
  * the SCENE-level surrogate Var_w / ESS_scene and the scene-level delta-method
    sum_g W_g^2 (D_g - s)^2  with  D_g = sum_{k in g} w_k phi'_k / W_g,
    both against the assumption-light measurement <(s_A - s_B)^2>/2 at the same size;
  * the K-scaling of every one of them.

If the scene-level version lands on the measurement, the formula was never the wrong model
-- the bank is simply not i.i.d. at the row level, which is a fixable bank-construction
fact, not a property of the estimator.

Usage:
    python scripts/diag5c_verifyR2_scene.py --n-gal 20000 --n-nodes 1000,2000,6000,20000
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


def starts_in(ii, k0, k1):
    """Run-start offsets of the contiguous `input_index` groups inside columns [k0, k1)."""
    seg = ii[k0:k1]
    return np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])


def stats(phi0, d1, ii, k0, k1, row_chunk=256):
    """Row-level and SCENE-level weighted statistics on the node columns [k0, k1)."""
    n = phi0.shape[0]
    st = np.flatnonzero(np.r_[True, ii[k0:k1][1:] != ii[k0:k1][:-1]])
    out = {k: np.empty(n) for k in
           ("s", "ess_row", "ess_scn", "varw", "varw_scn", "sur_row", "sur_scn",
            "dm_row", "dm_scn")}
    for a in range(0, n, row_chunk):
        b = min(a + row_chunk, n)
        w = posterior_weights(phi0[a:b, k0:k1])
        dd = d1[a:b, k0:k1]
        s = np.einsum("ij,ij->i", w, dd)
        out["s"][a:b] = s
        out["ess_row"][a:b] = 1.0 / np.einsum("ij,ij->i", w, w)
        vw = np.einsum("ij,ij->i", w, dd * dd) - s ** 2
        out["varw"][a:b] = vw
        rr = dd - s[:, None]
        out["dm_row"][a:b] = np.einsum("ij,ij->i", w * w, rr * rr)
        # scene aggregation: contiguous runs -> reduceat
        W = np.add.reduceat(w, st, axis=1)
        Wd = np.add.reduceat(w * dd, st, axis=1)
        D = Wd / np.maximum(W, 1e-300)
        sw2 = np.einsum("ij,ij->i", W, W)
        out["ess_scn"][a:b] = 1.0 / sw2
        rs = D - s[:, None]
        rs2 = rs * rs
        out["varw_scn"][a:b] = np.einsum("ij,ij->i", W, rs2)
        out["dm_scn"][a:b] = np.einsum("ij,ij->i", W * W, rs2)
        out["sur_row"][a:b] = vw * np.einsum("ij,ij->i", w, w)
        out["sur_scn"][a:b] = vw * sw2
        del w, dd, rr, W, Wd, D, rs, rs2
    out["n_scene"] = len(st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="1000,2000,6000,20000")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--row-chunk", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    nodes = sorted(int(x) for x in args.n_nodes.split(","))
    kmax, d = max(nodes), args.delta
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    print(f"device {dev}  primary_only_shear={meta.get('primary_only_shear')}", flush=True)

    rows = load_rows(args.catalogue, (kmax + args.n_gal) * 4)
    pool = rows.iloc[:kmax].reset_index(drop=True)
    gal_df = rows.iloc[kmax:kmax + args.n_gal].reset_index(drop=True)
    ii = pool["input_index"].to_numpy()
    runs = np.flatnonzero(np.r_[True, ii[1:] != ii[:-1]])
    print(f"  node pool: {kmax:,} rows, {len(runs):,} contiguous input_index groups, "
          f"mean {kmax / len(runs):.2f} rows/group", flush=True)
    for c in ("Re_input_p", "axis_ratio_input_p", "sersic_n_input_p"):
        w_ = pool.groupby("input_index")[c].std().max()
        print(f"    within-group max sd of {c}: {float(w_):.3e}", flush=True)

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
    print(f"  gamma=0: kept {n:,}/{len(gal_df):,}\n", flush=True)
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

    rng = np.random.default_rng(args.seed)
    print("=" * 104)
    print("ROW-level vs SCENE-level weighted statistics, and both against the MEASURED noise.")
    print("  measured  = <(s_A - s_B)^2>/2 at bank size K/2 (halves are scene-disjoint)")
    print("  surrogate = Var_w(phi') * sum w^2   (probe A's Var_w/ESS)")
    print("  delta     = sum w^2 (phi' - s)^2    (the actual delta-method SNIS variance)")
    print("=" * 104)
    tab = {}
    for K in nodes:
        half = K // 2
        A = stats(phi0, d1, ii, 0, half, args.row_chunk)
        B = stats(phi0, d1, ii, half, 2 * half, args.row_chunk)
        F = stats(phi0, d1, ii, 0, K, args.row_chunk)
        pair = (A["s"] - B["s"]) ** 2 / 2.0
        meas = float(np.mean(pair))
        se = float(np.std([pair[rng.integers(0, n, n)].mean() for _ in range(args.n_boot)]))
        avg = lambda k: 0.5 * (A[k] + B[k])                                   # noqa: E731
        reff = A["ess_row"] / A["ess_scn"]
        row = dict(
            meas=meas, se=se,
            sur_row=float(np.mean(avg("sur_row"))), sur_scn=float(np.mean(avg("sur_scn"))),
            dm_row=float(np.mean(avg("dm_row"))), dm_scn=float(np.mean(avg("dm_scn"))),
            ess_row=float(np.mean(F["ess_row"])), ess_scn=float(np.mean(F["ess_scn"])),
            reff=float(np.mean(reff)), reff_med=float(np.median(reff)),
            varw=float(np.mean(F["varw"])), varw_scn=float(np.mean(F["varw_scn"])),
            nscene=F["n_scene"],
        )
        tab[K] = row
        print(f"\nK = {K:>6,}   full bank: {F['n_scene']:,} scenes, "
              f"ESS_row = {row['ess_row']:8.2f}   ESS_scene = {row['ess_scn']:8.2f}   "
              f"r_eff = ESS_row/ESS_scene mean {row['reff']:5.2f} med {row['reff_med']:5.2f}")
        print(f"   bank size K/2 = {half:,}:  MEASURED <(sA-sB)^2>/2 = {meas:9.3f} +- {se:.3f}")
        print(f"      surrogate ROW   = {row['sur_row']:9.3f}  ratio {row['sur_row'] / meas:6.3f}"
              f"   |   surrogate SCENE = {row['sur_scn']:9.3f}  ratio "
              f"{row['sur_scn'] / meas:6.3f}")
        print(f"      delta     ROW   = {row['dm_row']:9.3f}  ratio {row['dm_row'] / meas:6.3f}"
              f"   |   delta     SCENE = {row['dm_scn']:9.3f}  ratio "
              f"{row['dm_scn'] / meas:6.3f}")
        print(f"      Var_w row = {row['varw']:10.2f}   Var_w scene-aggregated = "
              f"{row['varw_scn']:10.2f}   (aggregation removes "
              f"{1 - row['varw_scn'] / row['varw']:5.1%} of the spread)", flush=True)
        del A, B, F

    lk = np.log(np.array([K // 2 for K in nodes], dtype=np.float64))
    lkf = np.log(np.array(nodes, dtype=np.float64))
    print("\n" + "=" * 104)
    print("SCALING")
    print("=" * 104)
    for nm in ("meas", "sur_row", "sur_scn", "dm_row", "dm_scn"):
        print(f"   {nm:>8} ~ B^{ols_slope(lk, np.log([tab[K][nm] for K in nodes])):+.3f}")
    for nm in ("ess_row", "ess_scn", "varw", "varw_scn", "reff"):
        print(f"   {nm:>8} ~ K^{ols_slope(lkf, np.log([tab[K][nm] for K in nodes])):+.3f}"
              f"   values {[round(tab[K][nm], 3) for K in nodes]}")
    print("\n  If the SCENE-level numbers land on the measurement while the ROW-level ones are")
    print("  ~8x low, the 6-10x 'formula failure' is the bank's duplication, not the formula.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
