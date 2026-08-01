"""Denominator-free scoreboard for the 5C Lagrangian score: the RESPONSE SLOPE.

Round 1 established that `<I> - I_sel` is a cancellation of two O(100) terms, statistically
consistent with zero at K >= 4000, with a standard error that GROWS with K.  Any diagnostic
with it in the denominator (the "Bartlett ratio", the Louis ghat) is therefore unusable at
the digit level.  The NUMERATOR is sound.  So the honest question is:

    how does the centred mean score  <s> - <s>_sel  move as the TRUE shear moves?

This script measures that curve with the galaxies PINNED, the data-draw seed FIXED, and the
node bank + its three shear stencil evaluations SHARED across every g_true (common random
numbers everywhere they can be had).  The pairing is the point: round 1's two-point slope
estimate 5.6 +/- 2.2 drew the data independently at the two shears, and that is most of its
error bar.

Reported, all with galaxy-bootstrap error bars:
  * <s> - <s>_sel at each g_true, and the null test at g_true = 0;
  * the fitted slope d<s>/dg, its linearity, and slope-restricted-to-small-g;
  * the DENOMINATOR-FREE ratio  slope / Var(s - <s>_sel)  = 1 + m_Bartlett; both pieces are
    measured, neither is a cancellation;
  * the same slope against <I> - I_sel, WITH both error bars -- the exchange-rate deficit;
  * (5.9b) / (M.12): the predicted m_(5.9) - m_(5.8) drift at each gamma vs the measured
    difference, which is the one correction never applied to the real model.

Usage:
    python scripts/diag5c_slope.py --n-gal 20000 --n-node 2000 \
        --gammas 0,0.01,0.02,0.05,0.10,0.20
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
    drift_5_9b, posterior_weights, score_and_information, selection_terms,
    shear_estimate_bartlett, shear_estimate_louis,
)
from train_joint_forward import intrinsic_shape  # noqa: E402


def intr_of(df):
    d = {}
    d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
    d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
    return d


def s_info_for(model, xhat, bank, d, gal_block, chunk):
    """(s_i, I_i) for a whole galaxy set against a PRE-COMPUTED bank stencil.

    The bank contexts at t = 0, +d, -d are built once and reused for every g_true; that is
    the common-random-number structure.  Galaxies are reduced in blocks so the (N, K)
    phi arrays never all exist at once (N=20k, K=10k, float64, x3 = 4.8 GB otherwise).
    """
    n = xhat.shape[0]
    s_all, i_all = np.empty(n), np.empty(n)
    for st in range(0, n, gal_block):
        en = min(st + gal_block, n)
        xb = xhat[st:en]
        p = {t: phi_block(model, xb, bank[t][0], bank[t][1], chunk) for t in (0.0, +d, -d)}
        d1 = (p[+d] - p[-d]) / (2 * d)
        d2 = (p[+d] - 2 * p[0.0] + p[-d]) / d ** 2
        s_all[st:en], i_all[st:en] = score_and_information(p[0.0], d1, d2)
        del p, d1, d2
    return s_all, i_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-node", type=int, default=2000)
    ap.add_argument("--gammas", default="0,0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--gal-block", type=int, default=2500)
    ap.add_argument("--gal-offset", type=int, default=10000,
                    help="galaxies start here, INDEPENDENT of --n-node, so the K ladder "
                         "compares the same galaxies.  Must be >= --n-node.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    gammas = [float(x) for x in args.gammas.split(",")]
    if args.gal_offset < args.n_node:
        raise SystemExit("--gal-offset must be >= --n-node")
    print(f"device={args.device}  torch={torch.__version__}", flush=True)

    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    print(f"checkpoint: {os.path.basename(args.checkpoint)}   meta={meta}")

    span = args.gal_offset + args.n_gal
    rows = load_rows(args.catalogue, span * 4)
    tc = meta.get("true_cut")
    if tc is not None:
        re_min, mag_max = float(tc[0]), float(tc[1])
        keep_true = ((rows["Re_input_p"].to_numpy(float) > re_min)
                     & (rows["r_input_p"].to_numpy(float) < mag_max))
        print(f"true_cut Re>{re_min} & mag<{mag_max}: {int(keep_true.sum()):,}/{len(rows):,}")
        rows = rows[keep_true].reset_index(drop=True)
    if len(rows) < span:
        raise SystemExit(f"only {len(rows):,} in-domain rows, need {span:,}")
    node_df = rows.iloc[:args.n_node].reset_index(drop=True)
    gal_df = rows.iloc[args.gal_offset:args.gal_offset + args.n_gal].reset_index(drop=True)
    node_intr, gal_intr = intr_of(node_df), intr_of(gal_df)
    print(f"bank K={len(node_df):,} (rows 0..{args.n_node})   "
          f"galaxies N={len(gal_df):,} (rows {args.gal_offset}..{span})", flush=True)

    # ---- the SHARED bank stencil: built once, reused at every g_true -------------------
    d = args.delta
    bank, pmean = {}, {}
    for t in (0.0, +d, -d):
        c, pd_ = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
        bank[t] = (c, torch.log(pd_))
        pmean[t] = float(pd_.mean())
    lp = {t: np.log(v) for t, v in pmean.items()}
    s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                   (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
    print(f"population terms (bank-only, shared): <s>_sel={s_sel:+.6f}  I_sel={i_sel:+.6f}",
          flush=True)

    # ---- per-g_true pass --------------------------------------------------------------
    n_gal = len(gal_df)
    S = np.full((len(gammas), n_gal), np.nan)
    I = np.full((len(gammas), n_gal), np.nan)
    KEEP = np.zeros((len(gammas), n_gal), dtype=bool)
    for gi, g in enumerate(gammas):
        ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, (0.0, g), pre, nbr_std, dev)
        torch.manual_seed(args.seed)                      # CRN on the flow noise
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            xhat = model.mean_flow.sample(ctx_g, n_samples=1)
            if xhat.dim() == 3:
                xhat = xhat[:, 0, :]
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        u = torch.rand(n_gal, generator=gen).to(dev)      # CRN on the detection draw
        keep = (u < pdet_g)
        kidx = torch.nonzero(keep).squeeze(-1).cpu().numpy()
        s, info = s_info_for(model, xhat[keep], bank, d, args.gal_block, args.chunk)
        S[gi, kidx], I[gi, kidx], KEEP[gi, kidx] = s, info, True
        print(f"  g_true={g:+.3f}: detected {len(kidx):,}/{n_gal:,} "
              f"(<Pdet>={float(pdet_g.mean()):.4f})   <s>-<s>_sel={s.mean()-s_sel:+.5f}   "
              f"Var(s-<s>_sel)={np.mean((s-s_sel)**2):.4f}   <I>-I_sel={info.mean()-i_sel:+.4f}",
              flush=True)

    # ---- paired galaxy bootstrap ------------------------------------------------------
    rng = np.random.default_rng(20260731)
    B = args.boot
    G = np.asarray(gammas)
    stat_names = ["mean_c", "var_c", "I_c", "ghat58", "ghat59", "drift"]
    boot = {k: np.empty((B, len(gammas))) for k in stat_names}
    boot_slope, boot_slope_lo = np.empty(B), np.empty(B)
    lo_mask = G <= 0.05

    def cell(sv, iv):
        mean_c = sv.mean() - s_sel
        var_c = np.mean((sv - s_sel) ** 2)
        I_c = iv.mean() - i_sel
        return (mean_c, var_c, I_c,
                shear_estimate_louis(sv, iv, s_sel, i_sel),
                shear_estimate_bartlett(sv, s_sel), np.nan)

    def fit_slope(y, mask):
        x = G[mask]
        A = np.vstack([np.ones_like(x), x]).T
        return np.linalg.lstsq(A, y[mask], rcond=None)[0][1]

    for b in range(B):
        idx = rng.integers(0, n_gal, size=n_gal)
        y = np.empty(len(gammas))
        for gi in range(len(gammas)):
            sel = idx[KEEP[gi, idx]]
            sv, iv = S[gi, sel], I[gi, sel]
            vals = cell(sv, iv)
            for nm, v in zip(stat_names, vals):
                boot[nm][b, gi] = v
            boot["drift"][b, gi] = drift_5_9b(sv, iv, s_sel, i_sel, gammas[gi])
            y[gi] = vals[0]
        boot_slope[b] = fit_slope(y, np.ones(len(gammas), bool))
        boot_slope_lo[b] = fit_slope(y, lo_mask)

    pt = {}
    for gi in range(len(gammas)):
        sv, iv = S[gi, KEEP[gi]], I[gi, KEEP[gi]]
        pt[gi] = list(cell(sv, iv))
        pt[gi][5] = drift_5_9b(sv, iv, s_sel, i_sel, gammas[gi])
    y_pt = np.array([pt[gi][0] for gi in range(len(gammas))])
    slope_pt = fit_slope(y_pt, np.ones(len(gammas), bool))
    slope_lo_pt = fit_slope(y_pt, lo_mask)
    se = lambda a: float(np.std(a, ddof=1))

    print("\n" + "=" * 100)
    print("PART 1  --  THE RESPONSE SLOPE   (all errors: paired galaxy bootstrap, "
          f"B={B}, same resample across g_true)")
    print("=" * 100)
    print(f"{'g_true':>8} {'N_det':>7} | {'<s>-<s>_sel':>22} | {'(<s>-<s>_sel)/g':>16} | "
          f"{'Var(s-<s>_sel)':>20} | {'<I>-I_sel':>20}")
    for gi, g in enumerate(gammas):
        m, v, ic = pt[gi][0], pt[gi][1], pt[gi][2]
        sm, sv_, si = (se(boot["mean_c"][:, gi]), se(boot["var_c"][:, gi]),
                       se(boot["I_c"][:, gi]))
        rat = f"{m/g:>8.3f}+-{sm/abs(g):.3f}" if g != 0 else f"{'--':>16}"
        print(f"{g:>8.3f} {int(KEEP[gi].sum()):>7,} | {m:>11.6f} +- {sm:<8.6f} | {rat:>16} | "
              f"{v:>11.4f} +- {sv_:<6.4f} | {ic:>11.4f} +- {si:<6.4f}")

    print(f"\nNULL TEST at g_true=0:  <s>-<s>_sel = {pt[0][0]:+.6f} +- "
          f"{se(boot['mean_c'][:, 0]):.6f}   "
          f"({abs(pt[0][0]) / se(boot['mean_c'][:, 0]):.2f} sigma from zero)")
    print(f"\nFITTED SLOPE d<s>/dg (all g):      {slope_pt:8.3f} +- {se(boot_slope):.3f}")
    print(f"FITTED SLOPE d<s>/dg (g <= 0.05):  {slope_lo_pt:8.3f} +- {se(boot_slope_lo):.3f}"
          f"   [linearity: the two agree iff the response has not saturated]")
    dslope = boot_slope - boot_slope_lo
    print(f"  paired difference (all - small-g): {slope_pt-slope_lo_pt:+.3f} +- {se(dslope):.3f}")

    print("\nDENOMINATOR-FREE SCOREBOARD  (both pieces measured, neither is a cancellation):")
    print(f"  Var(s-<s>_sel) at g=0            = {pt[0][1]:.4f} +- {se(boot['var_c'][:, 0]):.4f}")
    r = boot_slope / boot["var_c"][:, 0]
    rpt = slope_pt / pt[0][1]
    print(f"  slope / Var(s) = 1 + m_Bartlett  = {rpt:.4f} +- {se(r):.4f}"
          f"   -->  m_Bartlett = {rpt-1:+.2%} +- {se(r):.2%}")
    print("\nEXCHANGE RATE against the Louis denominator (the ~11x deficit):")
    rl = boot_slope / boot["I_c"][:, 0]
    rlpt = slope_pt / pt[0][2]
    print(f"  <I>-I_sel at g=0                 = {pt[0][2]:.4f} +- {se(boot['I_c'][:, 0]):.4f}")
    print(f"  slope / (<I>-I_sel) = 1 + m_Louis= {rlpt:.4f} +- {se(rl):.4f}"
          f"   -->  m_Louis = {rlpt-1:+.2%} +- {se(rl):.2%}")
    print(f"  (<I>-I_sel) / slope              = {pt[0][2]/slope_pt:.3f}"
          f"   [1 would be an unbiased exchange rate]")

    print("\n" + "=" * 100)
    print("PART 2  --  THE (5.9b)/(M.12) DRIFT CORRECTION")
    print("=" * 100)
    print("  m_(5.9) - m_(5.8) = -gamma (mu3 - Cov(I_keep, s_keep)) / I_keep + O(gamma^2)")
    print(f"{'g_true':>8} | {'ghat(5.8)':>18} | {'ghat(5.9)':>18} | "
          f"{'m59-m58 measured':>20} | {'(5.9b) predicted':>18}")
    for gi, g in enumerate(gammas):
        g58, g59, dr = pt[gi][3], pt[gi][4], pt[gi][5]
        s58, s59 = se(boot["ghat58"][:, gi]), se(boot["ghat59"][:, gi])
        if g != 0:
            meas = (g59 - g58) / g
            smeas = se((boot["ghat59"][:, gi] - boot["ghat58"][:, gi]) / g)
            mtxt = f"{meas:>10.4%} +- {smeas:.4%}"
        else:
            mtxt = f"{'-- (gamma=0)':>20}"
        print(f"{g:>8.3f} | {g58:>10.6f}+-{s58:<7.6f} | {g59:>10.6f}+-{s59:<7.6f} | "
              f"{mtxt:>20} | {dr:>10.4%} +- {se(boot['drift'][:, gi]):.4%}")
    print("\n  m_(5.8) and m_(5.9) against truth (the away-from-zero discrepancy), and what")
    print("  is LEFT once the (5.9b) drift is removed from the 5.9-vs-5.8 comparison:")
    print(f"{'g_true':>8} | {'m_(5.8)':>18} | {'m_(5.9)':>18} | "
          f"{'m_(5.9) - drift':>18}  [= drift-corrected 5.9, comparable to 5.8]")
    for gi, g in enumerate(gammas):
        if g == 0:
            continue
        m58 = pt[gi][3] / g - 1
        m59 = pt[gi][4] / g - 1
        s58 = se(boot["ghat58"][:, gi] / g)
        s59 = se(boot["ghat59"][:, gi] / g)
        corr = m59 - pt[gi][5]
        scorr = se(boot["ghat59"][:, gi] / g - boot["drift"][:, gi])
        print(f"{g:>8.3f} | {m58:>10.2%} +- {s58:<6.2%} | {m59:>10.2%} +- {s59:<6.2%} | "
              f"{corr:>10.2%} +- {scorr:<6.2%}")
    print("\n  NOTE: the drift is proportional to gamma and is IDENTICALLY ZERO at g_true=0.")
    print("  It therefore cannot touch the g=0 failure; the two must not be conflated.")

    print("\n" + "=" * 100)
    print("PART 3  --  <I> AND ITS ERROR (the retirement case for the Louis form 5.8)")
    print("=" * 100)
    print(f"  K = {args.n_node}:  <I>-I_sel = {pt[0][2]:.4f} +- {se(boot['I_c'][:, 0]):.4f} "
          f"(galaxy bootstrap only; bank MC error is EXTRA and grows with K)")
    print(f"           Var(s-<s>_sel) = {pt[0][1]:.4f} +- {se(boot['var_c'][:, 0]):.4f}  "
          f"(rel. err {se(boot['var_c'][:, 0])/pt[0][1]:.1%})")
    print(f"           ratio Var/<I>  = {pt[0][1]/pt[0][2]:+.3f} "
          f"+- {se(boot['var_c'][:, 0]/boot['I_c'][:, 0]):.3f}")

    out = os.path.join(args.__dict__.get("outdir", "/home/z/Zekang.Zhang/.claude/jobs/"
                                         "be56a7ad/tmp"),
                       f"slope_K{args.n_node}_N{args.n_gal}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, gammas=G, S=S, I=I, KEEP=KEEP, s_sel=s_sel, i_sel=i_sel,
             slope=slope_pt, slope_lo=slope_lo_pt)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
