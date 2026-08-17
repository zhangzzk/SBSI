"""The K -> M limit: `phi` over the ENTIRE discrete population, no sampling at all.

`diag5c_localprop.py` asks whether a per-galaxy localised proposal makes the node bank
converge.  This script asks the prior question that bounds it: what does the estimator
return when the bank IS the whole population?

The population p_0 is the discrete pool of M in-domain catalogue rows (the same p_0 the
global-bank baseline samples).  Evaluating `phi_ik` for every one of the M rows makes the
posterior expectation E_w[.] EXACT -- there is no importance sampling left, no proposal, no
Monte-Carlo error in the bank.  Three things follow, and they are the point of the script:

  1. `s_exact`, `I_exact` and hence `ghat` at K = M.  If the denominator-free response
     ghat(0.05) - ghat(0) is still ~0 with an exact enumeration of the population, then
     bank Monte-Carlo error CANNOT be the explanation and hypothesis (A) is refuted in its
     "not enough nodes / bad proposal" form.  If it recovers, the coverage story is right
     and the localised proposal is the practical fix.

  2. The EFFECTIVE SUPPORT of the posterior inside the pool: ESS over all M rows, and the
     number of rows within a few nats of the best log-weight.  This is a property of the
     POOL, not of any proposal -- it is the ceiling on what any q can achieve.  If a
     typical galaxy has an effective support of a handful of catalogue rows out of M, no
     proposal over this pool can help and the mixture is fundamentally under-supported.

  3. Every sampled arm for FREE, on the SAME galaxies, exactly paired.  A uniform bank of
     size K is K columns drawn i.i.d. from the exact block; a localised bank is K columns
     drawn from q with the log(M q) correction.  So `s_arm - s_exact` is measurable
     per galaxy -- the actual estimation error, not the half-split proxy for it.

Data draw, seeds, true_cut, pool and galaxy row ranges are identical to
`diag5c_localprop.py` with the same `--n-gal`, so the two runs are directly comparable.

    python scripts/diag5c_exactpool.py --n-gal 2000 --n-sub 250 --pool 100000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import CAT, CKPT, load_rows, rebuild, scene_context  # noqa: E402
from diag5c_localprop import (  # noqa: E402
    build_proposal,
    draw_nodes,
    intr_of,
    parse_arm,
    pool_contexts,
)
from sbs_shear.lagrangian_score import (  # noqa: E402
    denominator_consistency,
    posterior_weights,
    score_and_information,
    selection_terms,
    shear_estimate_bartlett,
    shear_estimate_louis,
)


def boot_stat(fn, n, reps, rng):
    return float(np.std([fn(rng.integers(0, n, n)) for _ in range(reps)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=2000,
                    help="MUST match the localprop run: it fixes the seeded xhat draw")
    ap.add_argument("--n-sub", type=int, default=250,
                    help="how many of the detected galaxies get the exact M-node treatment")
    ap.add_argument("--pool", type=int, default=100000)
    ap.add_argument("--gal-offset", type=int, default=None)
    ap.add_argument("--n-nodes", default="500,2000,6000")
    ap.add_argument("--gammas", default="0,0.05")
    ap.add_argument("--arms", default="uniform,a0.2f0.05,a0.2f0.01")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--max-pairs", type=int, default=250_000)
    ap.add_argument("--ctx-batch", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    ks = [int(x) for x in args.n_nodes.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    arms = [parse_arm(a) for a in args.arms.split(",")]
    off = args.gal_offset if args.gal_offset is not None else args.pool
    rng = np.random.default_rng(args.seed + 7)
    d = args.delta

    print(f"device={args.device}  torch={torch.__version__}")
    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")

    rows = load_rows(args.catalogue, int((off + args.n_gal) * 3.0))
    tc = meta.get("true_cut")
    if tc is not None:
        keep_true = ((rows["Re_input_p"].to_numpy(float) > float(tc[0]))
                     & (rows["r_input_p"].to_numpy(float) < float(tc[1])))
        rows = rows[keep_true].reset_index(drop=True)
        print(f"true_cut: {keep_true.mean():.1%} kept")
    pool_df = rows.iloc[:args.pool].reset_index(drop=True)
    gal_df = rows.iloc[off:off + args.n_gal].reset_index(drop=True)
    pool_intr, gal_intr = intr_of(pool_df), intr_of(gal_df)
    m_pool = len(pool_df)
    print(f"pool M = {m_pool:,};  galaxies [{off},{off + args.n_gal}); "
          f"exact treatment for the first {args.n_sub} DETECTED of them")

    ctx_t, pdet_t, logpd_t = {}, {}, {}
    for t in (0.0, +d, -d):
        c, p_ = pool_contexts(model, pool_df, pool_intr, (0.0, t), pre, nbr_std, dev,
                              bs=args.ctx_batch)
        ctx_t[t], pdet_t[t], logpd_t[t] = c, p_, torch.log(p_)
    with torch.no_grad():
        mu_pool = model.mu(ctx_t[0.0]).double()
    mu_scale = mu_pool.std(dim=0).clamp_min(1e-8)
    mu_pool_n = mu_pool / mu_scale

    lp = {t: float(np.log(float(pdet_t[t].double().mean()))) for t in (0.0, +d, -d)}
    s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                   (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
    print(f"population terms on the FULL pool: <s>_sel={s_sel:+.5f}  I_sel={i_sel:+.5f}")

    store, results = {}, {}
    for gamma in gammas:
        print("\n" + "=" * 100)
        print(f"### gamma_true = {gamma:+.4f}")
        print("=" * 100)
        torch.manual_seed(args.seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, (0.0, gamma),
                                      pre, nbr_std, dev)
        with torch.no_grad():
            xhat = model.mean_flow.sample(ctx_g, n_samples=1)
            if xhat.dim() == 3:
                xhat = xhat[:, 0, :]
        gen0 = torch.Generator(device="cpu").manual_seed(args.seed)
        keep = (torch.rand(len(gal_df), generator=gen0).to(dev) < pdet_g)
        kept_idx = torch.nonzero(keep).squeeze(-1).cpu().numpy()
        xhat = xhat[keep][:args.n_sub]
        kept_idx = kept_idx[:args.n_sub]
        n_sub = xhat.shape[0]
        print(f"detected {int(keep.sum()):,}/{len(gal_df):,}; using the first {n_sub}")
        del ctx_g
        xhat_n = xhat.double() / mu_scale

        chunk = max(1, min(n_sub, args.max_pairs // m_pool))
        gen = torch.Generator(device=dev)
        gen.manual_seed(args.seed * 811 + int(round(gamma * 1e4)))
        ex = {kk: np.empty(n_sub) for kk in
              ("s", "I", "ess", "n3", "n10", "wmax", "sdir", "idir", "edd", "vd")}
        arm_s = {(a["name"], k): np.empty(n_sub) for a in arms for k in ks}
        arm_i = {(a["name"], k): np.empty(n_sub) for a in arms for k in ks}
        t0 = time.time()
        for st in range(0, n_sub, chunk):
            en = min(st + chunk, n_sub)
            b = en - st
            phis = {}
            with torch.no_grad():
                for t in (0.0, +d, -d):
                    ll = model.log_prob_obs(
                        xhat[st:en, None, :].expand(b, m_pool, xhat.shape[1])
                        .reshape(-1, xhat.shape[1]),
                        ctx_t[t][None, :, :].expand(b, m_pool, ctx_t[t].shape[1])
                        .reshape(-1, ctx_t[t].shape[1])).view(b, m_pool)
                    phis[t] = (ll + logpd_t[t][None, :]).double().cpu().numpy()
            p0 = phis[0.0]
            dphi = (phis[+d] - phis[-d]) / (2 * d)
            ddphi = (phis[+d] - 2 * p0 + phis[-d]) / d ** 2
            s_, i_ = score_and_information(p0, dphi, ddphi)       # EXACT over the pool
            w = posterior_weights(p0)
            ex["s"][st:en], ex["I"][st:en] = s_, i_
            ex["edd"][st:en] = np.sum(w * ddphi, axis=1)
            ex["vd"][st:en] = np.sum(w * dphi ** 2, axis=1) - s_ ** 2
            ex["ess"][st:en] = 1.0 / np.sum(w ** 2, axis=1)
            ex["wmax"][st:en] = w.max(axis=1)
            top = p0.max(axis=1, keepdims=True)
            ex["n3"][st:en] = np.sum(p0 > top - 3.0, axis=1)
            ex["n10"][st:en] = np.sum(p0 > top - 10.0, axis=1)
            from scipy.special import logsumexp as _lse
            lz = {t: _lse(phis[t], axis=1) for t in (0.0, +d, -d)}
            ex["sdir"][st:en] = (lz[+d] - lz[-d]) / (2 * d)
            ex["idir"][st:en] = -(lz[+d] - 2 * lz[0.0] + lz[-d]) / d ** 2
            # ---- every sampled arm, from the SAME exact block ---------------------------
            for arm in arms:
                q = build_proposal(mu_pool_n, xhat_n[st:en], arm["alpha"], arm["frac"])
                for k in ks:
                    idx, qsel = draw_nodes(q, k, gen)
                    lc = (-(torch.log(qsel) + np.log(m_pool))).cpu().numpy()
                    ii = idx.cpu().numpy()
                    p0s = np.take_along_axis(p0, ii, axis=1) + lc
                    d1s = np.take_along_axis(dphi, ii, axis=1)
                    d2s = np.take_along_axis(ddphi, ii, axis=1)
                    sa, ia = score_and_information(p0s, d1s, d2s)
                    arm_s[(arm["name"], k)][st:en] = sa
                    arm_i[(arm["name"], k)][st:en] = ia
                del q
            del phis, p0, dphi, ddphi, w
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        print(f"exact block done in {time.time() - t0:.0f} s "
              f"({n_sub * m_pool * 3 / 1e6:.0f}M flow evaluations)")

        # ---------------- exact-limit diagnostics ---------------------------------------
        bd, ld, ratio = denominator_consistency(ex["s"], ex["I"], s_sel, i_sel)
        louis = shear_estimate_louis(ex["s"], ex["I"], s_sel, i_sel)
        bart = shear_estimate_bartlett(ex["s"], s_sel)
        e_b = boot_stat(lambda i: shear_estimate_bartlett(ex["s"][i], s_sel),
                        n_sub, args.boot, rng)
        e_l = boot_stat(lambda i: shear_estimate_louis(ex["s"][i], ex["I"][i], s_sel, i_sel),
                        n_sub, args.boot, rng)
        print(f"\n  EXACT over all M={m_pool:,} pool rows (no sampling):")
        print(f"    posterior support INSIDE THE POOL: ESS median "
              f"{np.median(ex['ess']):.1f}  mean {ex['ess'].mean():.1f}  "
              f"[p10 {np.percentile(ex['ess'], 10):.1f}, p90 "
              f"{np.percentile(ex['ess'], 90):.1f}]  = "
              f"{np.median(ex['ess']) / m_pool:.2e} of the pool")
        print(f"    rows within 3 nats of the best: median {np.median(ex['n3']):.0f}; "
              f"within 10 nats: median {np.median(ex['n10']):.0f};  "
              f"max weight median {np.median(ex['wmax']):.3f}")
        print(f"    Fisher check: <s>_Ew={ex['s'].mean():+.5f} vs "
              f"<s>_direct={ex['sdir'].mean():+.5f}  corr="
              f"{np.corrcoef(ex['s'], ex['sdir'])[0, 1]:.6f}")
        print(f"    Var(s)={np.var(ex['s'], ddof=1):.4f}  <I>={ex['I'].mean():+.4f}  "
              f"[-E_w(phi'')={-ex['edd'].mean():+.4f}, Var_w(phi')={ex['vd'].mean():+.4f}]")
        print(f"    information equality ratio = {ratio:+.3f}")
        print(f"    ghat(5.8)={louis:+.6f} +- {e_l:.6f}   ghat(5.9)={bart:+.6f} +- {e_b:.6f}"
              f"   (truth {gamma:+.4f}, N={n_sub})")
        results[(gamma, "exact", m_pool)] = dict(
            louis=louis, bart=bart, e_louis=e_l, e_bart=e_b,
            var_s=float(np.var(ex["s"], ddof=1)), info=float(ex["I"].mean()),
            ess=float(np.median(ex["ess"])), n3=float(np.median(ex["n3"])),
            ratio=ratio)
        store[(gamma, "exact", m_pool)] = (ex["s"].copy(), ex["I"].copy(), kept_idx.copy())

        print("\n  SAMPLED ARMS on the same galaxies, error measured AGAINST the exact s:")
        sd_ex = float(np.std(ex["s"], ddof=1))
        for arm in arms:
            for k in ks:
                sa, ia = arm_s[(arm["name"], k)], arm_i[(arm["name"], k)]
                err = sa - ex["s"]
                lb = shear_estimate_bartlett(sa, s_sel)
                ll_ = shear_estimate_louis(sa, ia, s_sel, i_sel)
                eb = boot_stat(lambda i: shear_estimate_bartlett(sa[i], s_sel),
                               n_sub, args.boot, rng)
                print(f"    {arm['name']:>10} K={k:>5}: rms(s-s_exact)="
                      f"{np.sqrt(np.mean(err ** 2)):9.4f}  "
                      f"(sd of s_exact itself {sd_ex:.4f})  corr(s,s_exact)="
                      f"{np.corrcoef(sa, ex['s'])[0, 1]:+.4f}  "
                      f"Var(s)={np.var(sa, ddof=1):9.4f}  <I>={ia.mean():+9.4f}  "
                      f"ghat(5.9)={lb:+.6f}+-{eb:.6f}  ghat(5.8)={ll_:+.6f}")
                results[(gamma, arm["name"], k)] = dict(
                    bart=lb, louis=ll_, e_bart=eb,
                    rms_err=float(np.sqrt(np.mean(err ** 2))),
                    corr_exact=float(np.corrcoef(sa, ex["s"])[0, 1]),
                    var_s=float(np.var(sa, ddof=1)), info=float(ia.mean()))
                store[(gamma, arm["name"], k)] = (sa.copy(), ia.copy(), kept_idx.copy())
        sys.stdout.flush()

    if len(gammas) >= 2:
        g0, g1 = gammas[0], gammas[-1]
        print("\n" + "=" * 100)
        print(f"### DENOMINATOR-FREE RESPONSE, paired galaxies: ghat({g1}) - ghat({g0}), "
              f"truth {g1 - g0:+.4f}")
        print("=" * 100)
        keys = [("exact", m_pool)] + [(a["name"], k) for a in arms for k in ks]
        for nm, k in keys:
            sa0, ia0, kp0 = store[(g0, nm, k)]
            sa1, ia1, kp1 = store[(g1, nm, k)]
            _, j0, j1 = np.intersect1d(kp0, kp1, return_indices=True)
            n_c = j0.size
            sa0, ia0, sa1, ia1 = sa0[j0], ia0[j0], sa1[j1], ia1[j1]
            db = shear_estimate_bartlett(sa1, s_sel) - shear_estimate_bartlett(sa0, s_sel)
            dl = (shear_estimate_louis(sa1, ia1, s_sel, i_sel)
                  - shear_estimate_louis(sa0, ia0, s_sel, i_sel))
            brng = np.random.default_rng(args.seed + 101)
            vb = [shear_estimate_bartlett(sa1[j], s_sel) - shear_estimate_bartlett(sa0[j], s_sel)
                  for j in (brng.integers(0, n_c, n_c) for _ in range(args.boot))]
            print(f"  {nm:>10} K={k:>7}:  d ghat(5.9) = {db:+.6f} +- {np.std(vb):.6f}"
                  f"   [{db / (g1 - g0):+.1%} of truth]   d ghat(5.8) = {dl:+.6f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({f"{g}|{a}|{k}": v for (g, a, k), v in results.items()}, fh, indent=1)
        print(f"\nwrote {args.out}")
    print("\n### DONE ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
