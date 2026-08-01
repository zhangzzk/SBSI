"""SELF-CONSISTENCY test of the INFERENCE.md 5C Lagrangian score estimator.

THE POINT.  Every closure run so far drew the data from the TRUE marginal: pick a
catalogue scene z_i that is NOT in the node bank, draw xhat ~ p(. | S_g z_i).  The
estimator then scores that datum under the K-component MIXTURE built from the bank,

    p_hat(xhat | g) = (1/K) sum_k p(xhat | S_g z_k) Pdet(S_g z_k) / P(g).

Mixture != marginal, so Bartlett's information identity is not guaranteed and every
failure so far is compatible with a pure bank-coverage defect (hypothesis A).

Here the data is drawn FROM THE ESTIMATOR'S OWN MIXTURE: pick a node index k uniformly
from the SAME bank the estimator will use, draw xhat ~ p(. | S_{g_true} z_k), and apply
detection as Bernoulli(Pdet(S_{g_true} z_k)) with that node's own probability.  The
mixture is then EXACTLY the data-generating density, by construction.  Therefore at
g_true = 0 the identity

    Var(s - <s>_sel)  =  <I> - I_sel

must hold for ANY K, ANY ESS, ANY tail behaviour -- no approximation, no asymptotics, no
coverage assumption -- and ghat must return g_true.  This is the 1-D toy's logic
transplanted onto the real model.

  identity holds under the mixture draw and fails under the marginal draw at the same K
      => hypothesis (A) confirmed: the estimator is right, the finite bank does not cover
         the true marginal, and the gap vs K IS the coverage error.
  identity fails under the mixture draw too
      => hypothesis (A) refuted; the defect is in the estimator or the flow's gamma
         derivatives, in a way the Fisher-identity check and every ablation missed.

The marginal-draw control is run SIDE BY SIDE in the same process, same bank, same K,
same seed, same galaxy count.  Only the data generation differs.  That comparison is the
measurement.

Nothing is copied: rebuild / load_rows / scene_context / phi_block and the whole
lagrangian_score module are imported.  The only new code is the data generator and a
memory-chunked assembly (the (N_gal, K) matrices do not fit at N_gal = 40k, K = 6k).

    python scripts/diag5c_selfmix.py --n-gal 40000 --n-nodes 500,2000,6000 \
        --gammas 0,0.05 --delta 0.01
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy.special import logsumexp

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import (  # noqa: E402
    CAT,
    CKPT,
    load_rows,
    phi_block,
    rebuild,
    scene_context,
)
from sbs_shear.lagrangian_score import (  # noqa: E402
    selection_terms,
)
from train_joint_forward import intrinsic_shape  # noqa: E402


def intr_of(df):
    d = {}
    d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
    d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
    return d


# --------------------------------------------------------------------------------------
# memory-chunked per-galaxy assembly
# --------------------------------------------------------------------------------------

@torch.no_grad()
def per_galaxy_terms(model, xhat, ctxs, logpdets, delta, max_pairs, gen_node=None):
    """`(s_i, I_i, -E_w[phi''], Var_w(phi'), ESS_i, s_direct_i, rank_i)`.

    Identical arithmetic to `score_and_information` / the Fisher check in
    `closure_v2_lagrangian.main`, but the (N_gal, K) blocks are formed one galaxy chunk at
    a time and reduced immediately.  At N_gal = 40k and K = 6k a single float64 block is
    1.9 GB and five of them are held simultaneously in the unchunked version.

    `gen_node` (mixture draw only) is the index of the node that GENERATED each datum; the
    returned `rank` is that node's rank in the posterior weight row (0 = top weight).
    """
    n = int(xhat.shape[0])
    k = int(ctxs[0].shape[0])
    step = max(1, max_pairs // max(k, 1))
    s = np.empty(n)
    mean_dd = np.empty(n)
    var_d = np.empty(n)
    ess = np.empty(n)
    s_dir = np.empty(n)
    i_dir = np.empty(n)
    rank = np.full(n, -1, dtype=np.int64) if gen_node is not None else None
    for st in range(0, n, step):
        en = min(st + step, n)
        xb = xhat[st:en]
        p0 = phi_block(model, xb, ctxs[0], logpdets[0], step, max_pairs)
        pp = phi_block(model, xb, ctxs[1], logpdets[1], step, max_pairs)
        pm = phi_block(model, xb, ctxs[2], logpdets[2], step, max_pairs)
        d1 = (pp - pm) / (2.0 * delta)
        d2 = (pp - 2.0 * p0 + pm) / delta ** 2
        lz0, lzp, lzm = (logsumexp(p0, axis=1), logsumexp(pp, axis=1),
                         logsumexp(pm, axis=1))
        s_dir[st:en] = (lzp - lzm) / (2.0 * delta)
        i_dir[st:en] = -(lzp - 2.0 * lz0 + lzm) / delta ** 2
        del pp, pm
        ll = p0 - p0.max(axis=1, keepdims=True)
        w = np.exp(ll)
        w /= w.sum(axis=1, keepdims=True)
        si = np.sum(w * d1, axis=1)
        s[st:en] = si
        mean_dd[st:en] = np.sum(w * d2, axis=1)
        var_d[st:en] = np.sum(w * d1 ** 2, axis=1) - si ** 2
        ess[st:en] = 1.0 / np.sum(w ** 2, axis=1)
        if gen_node is not None:
            gn = gen_node[st:en]
            own = w[np.arange(en - st), gn]
            rank[st:en] = (w > own[:, None]).sum(axis=1)
        del p0, d1, d2, w, ll
    info = -mean_dd - var_d
    return s, info, -mean_dd, var_d, ess, s_dir, i_dir, rank


# --------------------------------------------------------------------------------------
# bootstrap over galaxies
# --------------------------------------------------------------------------------------

def bootstrap_stats(s, info, s_sel, i_sel, reps, rng):
    """Bootstrap the four conclusion-carrying numbers jointly over galaxies.

    All four are functions of the SAME resample, so `bd - ld` and the ratio get the
    correct (correlated) error rather than a quadrature sum.
    """
    n = s.size
    out = {k: np.empty(reps) for k in ("bd", "ld", "diff", "ratio", "louis", "bart")}
    for r in range(reps):
        idx = rng.integers(0, n, n)
        sb, ib = s[idx], info[idx]
        c = sb - s_sel
        bd = float(np.mean(c ** 2))
        ld = float(np.mean(ib) - i_sel)
        out["bd"][r] = bd
        out["ld"][r] = ld
        out["diff"][r] = bd - ld
        out["ratio"][r] = bd / ld if ld != 0 else np.nan
        out["louis"][r] = (sb.sum() - n * s_sel) / (ib.sum() - n * i_sel)
        out["bart"][r] = c.sum() / np.sum(c ** 2)
    return {k: float(np.std(v)) for k, v in out.items()}


def point_stats(s, info, s_sel, i_sel):
    n = s.size
    c = s - s_sel
    bd = float(np.mean(c ** 2))
    ld = float(np.mean(info) - i_sel)
    return dict(bd=bd, ld=ld, diff=bd - ld, ratio=(bd / ld if ld != 0 else np.nan),
                louis=float((s.sum() - n * s_sel) / (info.sum() - n * i_sel)),
                bart=float(c.sum() / np.sum(c ** 2)))


# --------------------------------------------------------------------------------------

@torch.no_grad()
def sample_flow(model, ctx, seed, batch=20000):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    outs = []
    for st in range(0, int(ctx.shape[0]), batch):
        c = ctx[st:st + batch]
        x = model.mean_flow.sample(c, n_samples=1)
        if x.dim() == 3:
            x = x[:, 0, :]
        outs.append(x)
    return torch.cat(outs, dim=0)


def report(tag, k, g_true, n_draw, n_det, ess, s, info, mdd, vard, s_sel, i_sel,
           s_dir, i_dir, rank, reps, rng):
    p = point_stats(s, info, s_sel, i_sel)
    e = bootstrap_stats(s, info, s_sel, i_sel, reps, rng)
    print(f"\n  --- {tag}  K={k}  g_true={g_true:+.3f} ---")
    print(f"      N_drawn={n_draw:,}  N_detected={n_det:,} ({n_det / n_draw:.1%})  "
          f"<ESS>={ess.mean():.1f}  ESS/K={ess.mean() / k:.3f}")
    print(f"      <s>          = {s.mean():+.6f} +/- {s.std(ddof=1) / np.sqrt(s.size):.6f}"
          f"      <s>_sel = {s_sel:+.6f}      <s>-<s>_sel = {s.mean() - s_sel:+.6f}")
    print(f"      Var(s-<s>_sel) = {p['bd']:.4f} +/- {e['bd']:.4f}")
    print(f"      <I> - I_sel    = {p['ld']:.4f} +/- {e['ld']:.4f}"
          f"        (I_sel = {i_sel:+.5f})")
    print(f"        halves:  -E_w[phi''] = {mdd.mean():+.4f} +/- "
          f"{mdd.std(ddof=1) / np.sqrt(mdd.size):.4f}"
          f"   Var_w(phi') = {vard.mean():+.4f} +/- "
          f"{vard.std(ddof=1) / np.sqrt(vard.size):.4f}")
    print(f"      IDENTITY  Var-<I>  = {p['diff']:+.4f} +/- {e['diff']:.4f}"
          f"      ratio = {p['ratio']:+.3f} +/- {e['ratio']:.3f}   (must be 1)")
    mf = (lambda v, s_: f"   m = {v / g_true - 1:+.2%} +/- {s_ / abs(g_true):.2%}") \
        if g_true else (lambda v, s_: "")
    print(f"      ghat (5.8) Louis    = {p['louis']:+.6f} +/- {e['louis']:.6f}"
          f"{mf(p['louis'], e['louis'])}")
    print(f"      ghat (5.9) Bartlett = {p['bart']:+.6f} +/- {e['bart']:.6f}"
          f"{mf(p['bart'], e['bart'])}")
    cc = np.corrcoef(s, s_dir)[0, 1]
    print(f"      Fisher check: <s>_Ew={s.mean():+.5f} vs <s>_direct={s_dir.mean():+.5f}"
          f"  corr={cc:.6f}   <I>_direct={i_dir.mean():+.4f}")
    if rank is not None:
        print(f"      generating node rank in w: median={np.median(rank):.0f}  "
              f"top1={np.mean(rank == 0):.1%}  top1%={np.mean(rank < max(1, k // 100)):.1%}")
    return p, e


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=40000, help="galaxies DRAWN (pre-detection)")
    ap.add_argument("--n-nodes", default="500,2000,6000")
    ap.add_argument("--gammas", default="0,0.05")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--gal-offset", type=int, default=None,
                    help="marginal-draw galaxies come from this fixed in-domain row offset; "
                         "defaults to max(--n-nodes) so the bank never overlaps them and K "
                         "is the only thing that moves along the ladder")
    ap.add_argument("--max-pairs", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--draws", default="mixture,marginal")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    ks = [int(x) for x in args.n_nodes.split(",")]
    gs = [float(x) for x in args.gammas.split(",")]
    draws = [d.strip() for d in args.draws.split(",")]
    off = args.gal_offset if args.gal_offset is not None else max(ks)
    if off < max(ks):
        raise SystemExit("--gal-offset must be >= max(--n-nodes)")
    print(f"device={args.device}  torch={torch.__version__}")
    print(f"K={ks}  gammas={gs}  delta={args.delta}  n_gal={args.n_gal:,}  "
          f"gal_offset={off:,}  seed={args.seed}  boot={args.boot}")

    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    print(f"checkpoint: {os.path.basename(args.checkpoint)}   metadata: {meta}")

    span = off + args.n_gal
    rows = load_rows(args.catalogue, span * 4)
    tc = meta.get("true_cut")
    if tc is not None:
        re_min, mag_max = float(tc[0]), float(tc[1])
        keep_true = ((rows["Re_input_p"].to_numpy(float) > re_min)
                     & (rows["r_input_p"].to_numpy(float) < mag_max))
        print(f"true_cut Re>{re_min} & mag<{mag_max}: {int(keep_true.sum()):,}/"
              f"{len(rows):,} = {keep_true.mean():.1%} kept")
        rows = rows[keep_true].reset_index(drop=True)
    if len(rows) < span:
        raise SystemExit(f"only {len(rows):,} in-domain rows, need {span:,}")
    gal_df = rows.iloc[off:off + args.n_gal].reset_index(drop=True)
    gal_intr = intr_of(gal_df)
    print(f"rows: bank <= {max(ks):,} from the head; marginal-draw galaxies "
          f"{len(gal_df):,} from offset {off:,}")

    rng_boot = np.random.default_rng(args.seed + 7)
    summary = []
    t0 = time.time()
    for k in ks:
        node_df = rows.iloc[:k].reset_index(drop=True)
        node_intr = intr_of(node_df)
        # estimation contexts: the SAME bank rows at 0, +delta, -delta
        ctxs, pdets = [], []
        for t in (0.0, +args.delta, -args.delta):
            c, pd_ = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
            ctxs.append(c)
            pdets.append(pd_)
        logpdets = [torch.log(p) for p in pdets]
        lp = [float(np.log(float(p.mean().item()))) for p in pdets]
        s_sel, i_sel = selection_terms(
            lp[0], (lp[1] - lp[2]) / (2 * args.delta),
            (lp[1] - 2 * lp[0] + lp[2]) / args.delta ** 2)

        for g in gs:
            # generation contexts
            ctx_bank_g, pdet_bank_g = scene_context(
                model, node_df, node_intr, (0.0, g), pre, nbr_std, dev)
            for draw in draws:
                tic = time.time()
                if draw == "mixture":
                    rng = np.random.default_rng(args.seed)
                    node_idx = rng.integers(0, k, args.n_gal)
                    ni = torch.as_tensor(node_idx, dtype=torch.long, device=dev)
                    ctx_gen = ctx_bank_g[ni]
                    pdet_gen = pdet_bank_g[ni]
                else:
                    ctx_gen, pdet_gen = scene_context(
                        model, gal_df, gal_intr, (0.0, g), pre, nbr_std, dev)
                    node_idx = None
                xh = sample_flow(model, ctx_gen, args.seed)
                gen = torch.Generator(device="cpu").manual_seed(args.seed)
                keep = (torch.rand(int(ctx_gen.shape[0]), generator=gen).to(dev)
                        < pdet_gen)
                xh = xh[keep]
                kmask = keep.cpu().numpy()
                gn = node_idx[kmask] if node_idx is not None else None
                n_det = int(xh.shape[0])
                if n_det < 100:
                    print(f"  SKIP {draw} K={k} g={g}: only {n_det} detected")
                    continue
                s, info, mdd, vard, ess, s_dir, i_dir, rank = per_galaxy_terms(
                    model, xh, ctxs, logpdets, args.delta, args.max_pairs, gn)
                p, e = report(draw.upper(), k, g, args.n_gal, n_det, ess, s, info,
                              mdd, vard, s_sel, i_sel, s_dir, i_dir, rank,
                              args.boot, rng_boot)
                summary.append(dict(draw=draw, k=k, g=g, n=n_det, ess=float(ess.mean()),
                                    s_sel=s_sel, i_sel=i_sel, **p,
                                    **{f"e_{kk}": vv for kk, vv in e.items()}))
                print(f"      [{time.time() - tic:.0f}s cell, "
                      f"{time.time() - t0:.0f}s total]")
                sys.stdout.flush()
                del s, info, mdd, vard, ess, s_dir, i_dir, rank, xh

    print("\n" + "=" * 108)
    print("SUMMARY   (identity: Var(s-<s>_sel) must equal <I>-I_sel; ghat must equal g_true)")
    print(f"{'draw':>9} {'K':>6} {'g_true':>7} {'N_det':>7} {'ESS':>7} "
          f"{'Var(s)':>16} {'<I>-I_sel':>16} {'ratio':>15} {'ghat 5.9':>18}")
    print("-" * 108)
    for r in summary:
        print(f"{r['draw']:>9} {r['k']:>6} {r['g']:>7.3f} {r['n']:>7,} {r['ess']:>7.1f} "
              f"{r['bd']:>9.3f}+/-{r['e_bd']:<6.3f} "
              f"{r['ld']:>9.3f}+/-{r['e_ld']:<6.3f} "
              f"{r['ratio']:>8.3f}+/-{r['e_ratio']:<6.3f} "
              f"{r['bart']:>9.5f}+/-{r['e_bart']:<8.5f}")
    print("=" * 108)
    print("READ: mixture-draw ratio ~ 1 while marginal-draw ratio != 1 at the same K =>")
    print("      bank coverage (hypothesis A).  Mixture-draw ratio != 1 => hypothesis A is")
    print("      refuted and the defect is in the estimator or the flow's gamma derivatives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
