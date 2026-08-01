"""5C det-ablation: isolate the DETECTION channel and the POPULATION terms.

Pdet enters `phi` additively and is the one piece the 1-D toy never exercised, so it is
unvalidated on the real model.  Four configurations of the same closure, sharing the same
sampled data and the same phi arithmetic:

  (a) full  : phi = log p_flow + log Pdet ; data detected ; population terms subtracted.
  (b) meas  : phi = log p_flow            ; data NOT detected (all galaxies kept) ;
              population terms 0.  The clean sub-problem.
  (c) meas+ : phi = log p_flow            ; data detected ; population terms subtracted.
              WRONG by construction -- positive control that the population terms do
              something.
  (d) full0 : phi = log p_flow + log Pdet ; data detected ; population terms 0.
              The other half of the same control.

Because log Pdet is a per-NODE constant it can be added to a single `phi_block` evaluated
with zero detection offset, so all four configurations cost one block per stencil node.

Also:
  * the population terms <s>_sel, I_sel recomputed on a much larger scene bank (detection
    head + context only, no flow) and over a delta sweep;
  * the g_true = 0 null <s> - <s>_sel with its standard error at large N;
  * the distribution of d log Pdet/dgamma across nodes vs d log p_flow/dgamma across pairs.
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO = "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b"
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import (  # noqa: E402
    CAT, CKPT, load_rows, phi_block, rebuild, scene_context,
)
from sbs_shear.lagrangian_score import (  # noqa: E402
    denominator_consistency, posterior_weights, score_and_information,
    selection_terms, shear_estimate_bartlett, shear_estimate_louis,
)
from train_joint_forward import intrinsic_shape  # noqa: E402


def intr_of(df):
    d = {}
    d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
    d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
    return d


@torch.no_grad()
def mean_pdet(model, df, pre, nbr_std, dev, gammas, row_chunk=50_000):
    """`P(g) = mean_k Pdet(S_g z_k)` for several g on the SAME rows (common random
    numbers).  Only the context net and the detection head are touched, so this is cheap
    and can run on a bank far larger than the flow could afford."""
    tot = {g: 0.0 for g in gammas}
    n = len(df)
    for s in range(0, n, row_chunk):
        sub = df.iloc[s:s + row_chunk].reset_index(drop=True)
        si = intr_of(sub)
        for g in gammas:
            _, pd_ = scene_context(model, sub, si, (0.0, g), pre, nbr_std, dev)
            tot[g] += float(pd_.double().sum())
    return {g: tot[g] / n for g in gammas}


def report_config(tag, phi0, d1, d2, s_sel, i_sel, gamma):
    s, info = score_and_information(phi0, d1, d2)
    bd, ld, ratio = denominator_consistency(s, info, s_sel, i_sel)
    louis = shear_estimate_louis(s, info, s_sel, i_sel)
    bart = shear_estimate_bartlett(s, s_sel)
    ess = float(np.mean(1.0 / np.sum(posterior_weights(phi0) ** 2, axis=1)))
    n = s.size
    sem = float(np.std(s, ddof=1) / np.sqrt(n))
    print(f"  {tag:<7} N={n:>6}  <s>={np.mean(s):+9.4f}  <s>_sel={s_sel:+8.5f}  "
          f"<s>-<s>_sel={np.mean(s) - s_sel:+9.4f} +- {sem:.4f}  "
          f"Var(s-sel)={bd:10.3f}  <I>-I_sel={ld:+10.3f}  ratio={ratio:+9.3f}  "
          f"ghat58={louis:+9.5f}  ghat59={bart:+9.5f}  ESS={ess:6.1f}")
    return dict(tag=tag, n=n, s_mean=float(np.mean(s)), sem=sem, s_sel=s_sel,
                bart_den=bd, louis_den=ld, ratio=ratio, g58=louis, g59=bart, ess=ess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=2000)
    ap.add_argument("--n-node", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--deltas", default="0.01")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gal-offset", type=int, default=40000)
    ap.add_argument("--n-pop", type=int, default=0,
                    help="if >0, recompute the population terms on this many scenes")
    ap.add_argument("--pop-deltas", default="0.04,0.02,0.01,0.005,0.0025")
    ap.add_argument("--pop-only", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={dev}  torch={torch.__version__}", flush=True)

    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")

    span = max(args.n_gal + args.gal_offset, args.n_pop, args.n_node)
    rows = load_rows(args.catalogue, span * 4)
    tc = meta.get("true_cut")
    if tc is not None:
        re_min, mag_max = float(tc[0]), float(tc[1])
        keep_true = ((rows["Re_input_p"].to_numpy(float) > re_min)
                     & (rows["r_input_p"].to_numpy(float) < mag_max))
        rows = rows[keep_true].reset_index(drop=True)
    print(f"in-domain rows available: {len(rows):,}", flush=True)

    # ---------------- (2) population terms on a large bank ---------------------------
    if args.n_pop > 0:
        npop = min(args.n_pop, len(rows))
        pop_df = rows.iloc[:npop].reset_index(drop=True)
        print(f"\n=== POPULATION TERMS, bank of {npop:,} scenes ===", flush=True)
        print(f"{'delta':>9} {'P(0)':>12} {'<s>_sel':>12} {'I_sel':>12}")
        for d in [float(x) for x in args.pop_deltas.split(",")]:
            P = mean_pdet(model, pop_df, pre, nbr_std, dev, (0.0, +d, -d))
            lp = {g: np.log(v) for g, v in P.items()}
            s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                           (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
            print(f"{d:>9.4f} {P[0.0]:>12.6f} {s_sel:>12.6f} {i_sel:>12.6f}", flush=True)
        # same on the small (K = n_node) bank, for the record
        small = rows.iloc[:args.n_node].reset_index(drop=True)
        d = float(args.pop_deltas.split(",")[2]) if len(args.pop_deltas.split(",")) > 2 \
            else 0.01
        P = mean_pdet(model, small, pre, nbr_std, dev, (0.0, +d, -d))
        lp = {g: np.log(v) for g, v in P.items()}
        s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                       (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
        print(f"  K={args.n_node} bank at delta={d}: <s>_sel={s_sel:+.6f}  "
              f"I_sel={i_sel:+.6f}  P(0)={P[0.0]:.6f}", flush=True)
        if args.pop_only:
            return 0

    node_df = rows.iloc[:args.n_node].reset_index(drop=True)
    if args.gal_offset < args.n_node:
        raise SystemExit("--gal-offset must be >= --n-node")
    gal_df = rows.iloc[args.gal_offset:args.gal_offset + args.n_gal].reset_index(drop=True)
    if len(gal_df) < args.n_gal:
        raise SystemExit(f"only {len(gal_df):,} galaxy rows available")
    node_intr, gal_intr = intr_of(node_df), intr_of(gal_df)
    print(f"rows: {len(node_df):,} nodes + {len(gal_df):,} galaxies", flush=True)

    # ---------------- data sampled FROM the model ------------------------------------
    g_true = (0.0, float(args.gamma))
    ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, g_true, pre, nbr_std, dev)
    with torch.no_grad():
        xhat = model.mean_flow.sample(ctx_g, n_samples=1)
        if xhat.dim() == 3:
            xhat = xhat[:, 0, :]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
    keep_np = keep.cpu().numpy()
    print(f"detection: kept {int(keep_np.sum()):,}/{len(gal_df):,}  "
          f"(<Pdet>={float(pdet_g.mean()):.4f})", flush=True)

    for d in [float(x) for x in args.deltas.split(",")]:
        print(f"\n=== gamma_true={args.gamma:+.4f}  K={args.n_node}  "
              f"N_gal={args.n_gal}  delta={d} ===", flush=True)
        pm, lpd = {}, {}
        for t in (0.0, +d, -d):
            c, pd_ = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
            zero = torch.zeros(c.shape[0], device=dev)
            pm[t] = phi_block(model, xhat, c, zero, args.chunk)   # log p_flow only
            lpd[t] = torch.log(pd_).double().cpu().numpy()
            del c, pd_
        # population terms on the SAME K-node bank (as the closure script uses)
        Pk = {t: float(np.mean(np.exp(lpd[t]))) for t in lpd}
        lP = {t: np.log(v) for t, v in Pk.items()}
        s_sel, i_sel = selection_terms(lP[0.0], (lP[+d] - lP[-d]) / (2 * d),
                                       (lP[+d] - 2 * lP[0.0] + lP[-d]) / d ** 2)
        print(f"  population (K={args.n_node}): P(0)={Pk[0.0]:.6f}  "
              f"<s>_sel={s_sel:+.6f}  I_sel={i_sel:+.6f}", flush=True)

        # ---- (4) derivative behaviour of the two channels ---------------------------
        dpdet = (lpd[+d] - lpd[-d]) / (2 * d)
        dflow = (pm[+d] - pm[-d]) / (2 * d)
        aa = np.asarray(dpdet).ravel()
        print(f"  d log Pdet/dg over {aa.size} nodes: mean={aa.mean():+.4f} "
              f"sd={aa.std():.4f} min={aa.min():+.3f} "
              f"p1={np.percentile(aa,1):+.3f} p50={np.percentile(aa,50):+.3f} "
              f"p99={np.percentile(aa,99):+.3f} max={aa.max():+.3f}")
        bb = dflow.ravel()
        print(f"  d log p_flow/dg over {bb.size} pairs: mean={bb.mean():+.4f} "
              f"sd={bb.std():.4f} "
              f"|.|: p50={np.percentile(np.abs(bb),50):.3f} "
              f"p99={np.percentile(np.abs(bb),99):.2f} "
              f"p99.9={np.percentile(np.abs(bb),99.9):.2f} "
              f"max={np.abs(bb).max():.2f}", flush=True)

        d2flow = (pm[+d] - 2 * pm[0.0] + pm[-d]) / d ** 2
        d2pdet = (lpd[+d] - 2 * lpd[0.0] + lpd[-d]) / d ** 2

        det_kept = np.flatnonzero(keep_np)
        cfgs = [
            ("full", det_kept, True, True),
            ("meas", None, False, False),
            ("meas+", det_kept, False, True),
            ("full0", det_kept, True, False),
        ]
        print("  --- ablation ---", flush=True)
        for tag, idx, use_det, use_pop in cfgs:
            sl = slice(None) if idx is None else idx
            p0 = pm[0.0][sl]
            a1 = dflow[sl]
            a2 = d2flow[sl]
            if use_det:
                p0 = p0 + lpd[0.0][None, :]
                a1 = a1 + dpdet[None, :]
                a2 = a2 + d2pdet[None, :]
            ss, ii = (s_sel, i_sel) if use_pop else (0.0, 0.0)
            report_config(tag, p0, a1, a2, ss, ii, args.gamma)
    return 0


if __name__ == "__main__":
    sys.exit(main())
