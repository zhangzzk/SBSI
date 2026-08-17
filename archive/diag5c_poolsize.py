"""Does a BIGGER POPULATION fix it?  Exact enumeration at several pool sizes M.

`diag5c_localprop.py` showed that a per-galaxy localised proposal raises ESS and turns the
half-bank noise exponent negative but does not move the shear response, and
`diag5c_exactpool.py` showed the response is still ~0 when the entire M = 100,000-row pool
is enumerated exactly.  That kills the SAMPLING form of hypothesis (A).  What survives of
(A) is the POPULATION form: the estimator scores a discrete M-component mixture, not the
continuous marginal, and the effective posterior support inside the pool is only ~100-200
rows out of 100,000.  If that is the mechanism, the response must grow with M.

So: enumerate `phi` over the whole pool at M = 10k, 30k, 100k, 300k, with the GALAXIES AND
THE DATA DRAW HELD FIXED (the xhat draw does not involve the pool, so it is literally the
same numbers at every M).  No proposal, no sampling, no Monte-Carlo error anywhere -- the
only thing that changes along the ladder is how many catalogue rows the mixture is built
from.  Every pool is a prefix of the same row range, so the ladder is nested.

  - if ghat(0.05) - ghat(0) grows towards 0.05 with M, the population size is the wall and
    the fix is a bigger catalogue (or a generative proposal that is not a catalogue);
  - if it stays at ~0 while the effective support grows in proportion to M, then the
    mixture size is NOT the mechanism and hypothesis (A) is finished in both its forms.

    python scripts/diag5c_poolsize.py --n-gal 3000 --n-sub 1200 \
        --pools 10000,30000,100000,300000 --gal-offset 300000
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
from diag5c_localprop import intr_of, pool_contexts  # noqa: E402
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
    ap.add_argument("--n-gal", type=int, default=3000)
    ap.add_argument("--n-sub", type=int, default=1200)
    ap.add_argument("--pools", default="10000,30000,100000,300000")
    ap.add_argument("--gal-offset", type=int, default=300000,
                    help="must be >= max(--pools) so no pool overlaps the galaxies")
    ap.add_argument("--gammas", default="0,0.05")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--max-pairs", type=int, default=400_000)
    ap.add_argument("--ctx-batch", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    pools = [int(x) for x in args.pools.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    if args.gal_offset < max(pools):
        raise SystemExit("--gal-offset must be >= max(--pools)")
    rng = np.random.default_rng(args.seed + 7)
    d = args.delta

    print(f"device={args.device}  torch={torch.__version__}")
    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    need = args.gal_offset + args.n_gal
    rows = load_rows(args.catalogue, int(need * 3.0))
    tc = meta.get("true_cut")
    if tc is not None:
        kt = ((rows["Re_input_p"].to_numpy(float) > float(tc[0]))
              & (rows["r_input_p"].to_numpy(float) < float(tc[1])))
        rows = rows[kt].reset_index(drop=True)
        print(f"true_cut: {kt.mean():.1%} kept; {len(rows):,} in-domain rows loaded")
    if len(rows) < need:
        raise SystemExit(f"only {len(rows):,} in-domain rows, need {need:,}")
    gal_df = rows.iloc[args.gal_offset:args.gal_offset + args.n_gal].reset_index(drop=True)
    gal_intr = intr_of(gal_df)

    # ---- data, drawn ONCE: it does not depend on the pool, so it is identical at every M
    data = {}
    for gamma in gammas:
        torch.manual_seed(args.seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, (0.0, gamma),
                                      pre, nbr_std, dev)
        with torch.no_grad():
            xh = model.mean_flow.sample(ctx_g, n_samples=1)
            if xh.dim() == 3:
                xh = xh[:, 0, :]
        gen0 = torch.Generator(device="cpu").manual_seed(args.seed)
        keep = (torch.rand(len(gal_df), generator=gen0).to(dev) < pdet_g)
        kid = torch.nonzero(keep).squeeze(-1).cpu().numpy()[:args.n_sub]
        data[gamma] = (xh[keep][:args.n_sub], kid)
        print(f"gamma={gamma:+.3f}: detected {int(keep.sum()):,}/{len(gal_df):,}, "
              f"using {data[gamma][0].shape[0]}")
        del ctx_g

    results, store = {}, {}
    for m_pool in pools:
        pool_df = rows.iloc[:m_pool].reset_index(drop=True)
        pool_intr = intr_of(pool_df)
        ctx_t, pdet_t, logpd_t = {}, {}, {}
        for t in (0.0, +d, -d):
            c, p_ = pool_contexts(model, pool_df, pool_intr, (0.0, t), pre, nbr_std, dev,
                                  bs=args.ctx_batch)
            ctx_t[t], pdet_t[t], logpd_t[t] = c, p_, torch.log(p_)
        lp = {t: float(np.log(float(pdet_t[t].double().mean()))) for t in (0.0, +d, -d)}
        s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                       (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
        print("\n" + "=" * 100)
        print(f"### POOL M = {m_pool:,}   <s>_sel={s_sel:+.5f}  I_sel={i_sel:+.6f}  "
              f"<Pdet>={float(pdet_t[0.0].mean()):.4f}")
        print("=" * 100)

        for gamma in gammas:
            xhat, kid = data[gamma]
            n_sub = xhat.shape[0]
            chunk = max(1, min(n_sub, args.max_pairs // m_pool))
            S = np.empty(n_sub); I = np.empty(n_sub)
            ESS = np.empty(n_sub); N3 = np.empty(n_sub); WM = np.empty(n_sub)
            EDD = np.empty(n_sub); VD = np.empty(n_sub)
            t0 = time.time()
            for st in range(0, n_sub, chunk):
                en = min(st + chunk, n_sub); b = en - st
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
                d1 = (phis[+d] - phis[-d]) / (2 * d)
                d2 = (phis[+d] - 2 * p0 + phis[-d]) / d ** 2
                s_, i_ = score_and_information(p0, d1, d2)
                w = posterior_weights(p0)
                S[st:en], I[st:en] = s_, i_
                EDD[st:en] = np.sum(w * d2, axis=1)
                VD[st:en] = np.sum(w * d1 ** 2, axis=1) - s_ ** 2
                ESS[st:en] = 1.0 / np.sum(w ** 2, axis=1)
                WM[st:en] = w.max(axis=1)
                N3[st:en] = np.sum(p0 > p0.max(axis=1, keepdims=True) - 3.0, axis=1)
                del phis, p0, d1, d2, w
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            bd, ld, ratio = denominator_consistency(S, I, s_sel, i_sel)
            lo = shear_estimate_louis(S, I, s_sel, i_sel)
            ba = shear_estimate_bartlett(S, s_sel)
            e_ba = boot_stat(lambda i: shear_estimate_bartlett(S[i], s_sel),
                             n_sub, args.boot, rng)
            e_var = boot_stat(lambda i: np.var(S[i], ddof=1), n_sub, args.boot, rng)
            e_inf = boot_stat(lambda i: np.mean(I[i]), n_sub, args.boot, rng)
            print(f"  gamma={gamma:+.4f}  ({time.time() - t0:.0f} s, "
                  f"{n_sub * m_pool * 3 / 1e6:.0f}M evals)")
            print(f"    support in the pool: ESS median {np.median(ESS):.1f} "
                  f"[p10 {np.percentile(ESS, 10):.1f}, p90 {np.percentile(ESS, 90):.1f}] "
                  f"= {np.median(ESS) / m_pool:.2e} of M;  rows within 3 nats median "
                  f"{np.median(N3):.0f};  max weight median {np.median(WM):.4f}")
            print(f"    Var(s)={np.var(S, ddof=1):9.3f} +- {e_var:.3f}   "
                  f"<I>={np.mean(I):+9.3f} +- {e_inf:.3f}   "
                  f"[-E_w(phi'')={-np.mean(EDD):+9.3f}, Var_w(phi')={np.mean(VD):+9.3f}]   "
                  f"info-equality ratio={ratio:+.3f}")
            print(f"    ghat(5.8)={lo:+.6f}   ghat(5.9)={ba:+.6f} +- {e_ba:.6f}   "
                  f"(truth {gamma:+.4f}, N={n_sub})")
            sys.stdout.flush()
            store[(m_pool, gamma)] = (S.copy(), I.copy(), kid.copy(), s_sel, i_sel)
            results[f"M{m_pool}|g{gamma}"] = dict(
                bart=ba, e_bart=e_ba, louis=lo, var_s=float(np.var(S, ddof=1)),
                e_var=e_var, info=float(np.mean(I)), e_info=e_inf,
                ess_med=float(np.median(ESS)), n3_med=float(np.median(N3)),
                ratio=ratio, s_sel=s_sel, i_sel=i_sel)
        for t in list(ctx_t):
            del ctx_t[t], pdet_t[t], logpd_t[t]
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if len(gammas) >= 2:
        g0, g1 = gammas[0], gammas[-1]
        print("\n" + "=" * 100)
        print(f"### RESPONSE vs POPULATION SIZE:  ghat({g1}) - ghat({g0}), truth "
              f"{g1 - g0:+.4f}   (paired galaxies, exact enumeration)")
        print("=" * 100)
        for m_pool in pools:
            s0, i0, k0, ss0, is0 = store[(m_pool, g0)]
            s1, i1, k1, ss1, is1 = store[(m_pool, g1)]
            _, j0, j1 = np.intersect1d(k0, k1, return_indices=True)
            a0, b0, a1, b1 = s0[j0], i0[j0], s1[j1], i1[j1]
            n_c = j0.size
            db = shear_estimate_bartlett(a1, ss1) - shear_estimate_bartlett(a0, ss0)
            dl = (shear_estimate_louis(a1, b1, ss1, is1)
                  - shear_estimate_louis(a0, b0, ss0, is0))
            brng = np.random.default_rng(args.seed + 101)
            vb = [shear_estimate_bartlett(a1[j], ss1) - shear_estimate_bartlett(a0[j], ss0)
                  for j in (brng.integers(0, n_c, n_c) for _ in range(args.boot))]
            print(f"  M={m_pool:>7,}  (N={n_c})   d ghat(5.9) = {db:+.6f} +- "
                  f"{np.std(vb):.6f}   [{db / (g1 - g0):+.1%} of truth]   "
                  f"d ghat(5.8) = {dl:+.6f}")
            results[f"M{m_pool}|resp"] = dict(d_bart=float(db), e_d_bart=float(np.std(vb)),
                                              d_louis=float(dl), n=int(n_c))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {args.out}")
    print("\n### DONE ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
