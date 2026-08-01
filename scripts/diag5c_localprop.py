"""Per-galaxy LOCALISED node proposal for the INFERENCE.md 5C Lagrangian estimator.

WHAT IS BEING TESTED.  Round 1 left two live hypotheses for the ~11x deficit in the
exchange rate between the (sound) numerator and the denominator of (5.8)/(5.9).  This
script tests the first, MIXTURE/BANK COVERAGE: the estimator evaluates the score of the
K-component mixture  p_hat(xhat) = (1/K) sum_k p(xhat | S_g z_k)  built from a GLOBAL prior
sample, and the latent z is the full ~18-dim scene.  If a global sample simply does not
cover any individual galaxy's posterior, then a per-galaxy LOCALISED proposal must fix it,
and the acceptance criterion is sharp: the half-to-half score noise must start falling like
1/K.  It currently does not (round 1: K^+0.26 to +0.33 while ESS/K falls).

THE PROPOSAL, and why it is exactly correct.  The prior p_0 is available only as a SAMPLER
-- the catalogue -- so the population is taken to be the DISCRETE, FINITE pool of M
in-domain catalogue rows, each with prior mass 1/M.  That is the same p_0 the global-bank
baseline draws its bank from (it takes rows[:K] of the same pool).  For galaxy i we draw K
i.i.d. node indices WITH REPLACEMENT from a categorical q(. | xhat_i) over the SAME M rows:

    q_k(i) = alpha / M  +  (1 - alpha) * loc_k(i),      sum_k q_k(i) = 1
    loc_k(i) propto exp( -0.5 * || mu_k - xhat_i ||^2 / h_i^2 )

`mu_k = model.mu(ctx(z_k))` is the model's own predicted mean measurement for pool scene k,
so the locality is measured in the 4-d standardized OBSERVABLE space that xhat lives in --
cheap, and no density estimation anywhere.  `h_i` is set by a quantile: h_i^2 is the
(frac*M)-th smallest squared distance, so `frac` is an interpretable locality fraction.

Self-normalised importance sampling then corrects it exactly, because q_k is a KNOWN,
computable probability mass:

    w_k propto exp(phi_k(0)) * p_0(z_k) / q_k = exp( phi_k(0) - log(M q_k) )

`q_k` does not depend on gamma, so phi'(0) and phi''(0) are untouched and the correction
enters ONLY through the weights -- it is passed as a per-(galaxy, node) log-prior offset
added to phi(0) before `posterior_weights`, which is what the `log_prior` argument of that
function does for a quadrature bank (it only accepts a shared (K,) vector, hence the
addition is done here).

WHY THE DEFENSIVE MIXTURE.  With alpha > 0 every catalogue row keeps q_k >= alpha/M, so the
importance ratio is bounded:  p_0/q = (1/M)/q_k <= 1/alpha.  At alpha = 0.2 no single node
can carry more than 5x its prior share of the correction.  That bound is the whole reason
the estimator stays honest; it is reported with every number.

alpha = 1 reproduces uniform i.i.d. sampling from the pool with log-correction identically
zero, and is run as the CONTROL through the identical code path.

THE HALVES ARE INDEPENDENT BY CONSTRUCTION.  The K draws are i.i.d., so A = first K/2 draws
and B = last K/2 draws are independent SNIS estimates at bank size K/2 given (xhat_i, pool).
This is why per-galaxy sampling is used even for the control: no `input_index` group-split
bookkeeping is needed (the trap `diag5c_splitbank` had to handle for a contiguous bank),
because two independent draws that happen to coincide are still independent draws.

VALIDATION OF q (mandatory -- a wrong q_k silently reintroduces the bias).  Two checks are
printed for every arm:
  (1) sum_k q_k = 1 exactly (assert);
  (2) a LIKELIHOOD-FREE SNIS calibration test: estimate E_p0[Pdet] and E_p0[r_input_p] from
      the drawn nodes using ONLY the correction 1/(M q_k), and compare to the exact pool
      mean.  Any error in q_k shows up here immediately, with no flow involved.

    python scripts/diag5c_localprop.py --n-gal 2000 --pool 100000 \
        --n-nodes 500,2000,6000 --gammas 0,0.05 --arms uniform,a0.2f0.05,a0.2f0.01
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import (  # noqa: E402
    CAT,
    CKPT,
    load_rows,
    rebuild,
    scene_context,
)
from sbs_shear.lagrangian_score import (  # noqa: E402
    denominator_consistency,
    posterior_weights,
    score_and_information,
    selection_terms,
    shear_estimate_bartlett,
    shear_estimate_louis,
)
from train_joint_forward import intrinsic_shape  # noqa: E402


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def intr_of(df):
    d = {}
    d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
    d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
    return d


def parse_arm(tok):
    """`uniform` -> alpha=1; `a0.2f0.05` -> alpha=0.2, locality fraction 0.05."""
    tok = tok.strip()
    if tok == "uniform":
        return dict(name="uniform", alpha=1.0, frac=1.0)
    m = re.fullmatch(r"a([0-9.]+)f([0-9.]+)", tok)
    if not m:
        raise SystemExit(f"cannot parse arm {tok!r}; use 'uniform' or 'a<alpha>f<frac>'")
    return dict(name=tok, alpha=float(m.group(1)), frac=float(m.group(2)))


@torch.no_grad()
def pool_contexts(model, df, intr, gamma_vec, pre, nbr_std, dev, bs=20000):
    """`scene_context` over a large frame, in slices, to bound GPU memory."""
    ctxs, pds = [], []
    for s in range(0, len(df), bs):
        e = min(s + bs, len(df))
        sub = df.iloc[s:e].reset_index(drop=True)
        sub_intr = {k: v[s:e] for k, v in intr.items()}
        c, pd_ = scene_context(model, sub, sub_intr, gamma_vec, pre, nbr_std, dev)
        ctxs.append(c)
        pds.append(pd_)
    return torch.cat(ctxs, 0), torch.cat(pds, 0)


@torch.no_grad()
def build_proposal(mu_pool, xhat_chunk, alpha, frac):
    """`q` for one chunk of galaxies: `(b, M)`, rows summing to 1.

    Distances are in the model's standardized observable space, each axis divided by the
    pool standard deviation of `mu` so no single output dimension dominates the metric.
    """
    m = mu_pool.shape[0]
    if alpha >= 1.0:
        return torch.full((xhat_chunk.shape[0], m), 1.0 / m,
                          dtype=torch.float64, device=mu_pool.device)
    d2 = torch.cdist(xhat_chunk.float(), mu_pool.float()) ** 2       # (b, M), fp32
    kk = int(max(8, min(m - 1, round(frac * m))))
    h2 = torch.kthvalue(d2, kk, dim=1).values.clamp_min(1e-12)       # (b,)
    ll = -0.5 * d2 / h2[:, None]
    ll = ll - ll.max(dim=1, keepdim=True).values
    loc = torch.exp(ll.double())
    loc = loc / loc.sum(dim=1, keepdim=True)
    return alpha / m + (1.0 - alpha) * loc


@torch.no_grad()
def draw_nodes(q, k, gen):
    """K i.i.d. draws per row of `q` by inverse-CDF; returns `(idx, q_sel)`."""
    cdf = torch.cumsum(q, dim=1)
    cdf = cdf / cdf[:, -1:].clone()
    u = torch.rand(q.shape[0], k, generator=gen, device=q.device, dtype=torch.float64)
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous())
    idx = idx.clamp_(max=q.shape[1] - 1)
    return idx, torch.gather(q, 1, idx)


def boot_stat(fn, n, reps, rng):
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(reps)])
    return float(np.std(v))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=2000)
    ap.add_argument("--pool", type=int, default=100000,
                    help="M: the discrete population the bank is drawn from, rows [0,M)")
    ap.add_argument("--gal-offset", type=int, default=None,
                    help="galaxies from this fixed in-domain row offset; default = --pool "
                         "so the galaxy sample is disjoint from the node pool")
    ap.add_argument("--n-nodes", default="500,2000,6000")
    ap.add_argument("--gammas", default="0,0.05")
    ap.add_argument("--arms", default="uniform,a0.2f0.05,a0.2f0.01")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--max-pairs", type=int, default=250_000,
                    help="cap on chunk*K; GPU memory in the phi block scales as this")
    ap.add_argument("--max-qcells", type=int, default=20_000_000,
                    help="cap on chunk*M; the proposal block holds a few float64 (b,M) arrays")
    ap.add_argument("--ctx-batch", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default=None, help="optional .json dump of every number")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    ks = [int(x) for x in args.n_nodes.split(",")]
    if any(k % 2 for k in ks):
        raise SystemExit("--n-nodes must all be even (the A/B halves are K/2 each)")
    gammas = [float(x) for x in args.gammas.split(",")]
    arms = [parse_arm(a) for a in args.arms.split(",")]
    off = args.gal_offset if args.gal_offset is not None else args.pool
    if off < args.pool:
        raise SystemExit("--gal-offset must be >= --pool (galaxies must not be in the pool)")
    rng = np.random.default_rng(args.seed + 7)

    print(f"device={args.device}  torch={torch.__version__}")
    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  metadata: {meta}")

    need = off + args.n_gal
    rows = load_rows(args.catalogue, int(need * 3.0))
    tc = meta.get("true_cut")
    if tc is not None:
        re_min, mag_max = float(tc[0]), float(tc[1])
        keep_true = ((rows["Re_input_p"].to_numpy(float) > re_min)
                     & (rows["r_input_p"].to_numpy(float) < mag_max))
        print(f"true_cut Re>{re_min} & mag<{mag_max}: "
              f"{int(keep_true.sum()):,}/{len(rows):,} = {keep_true.mean():.1%} kept")
        rows = rows[keep_true].reset_index(drop=True)
    if len(rows) < need:
        raise SystemExit(f"only {len(rows):,} in-domain rows, need {need:,}")

    pool_df = rows.iloc[:args.pool].reset_index(drop=True)
    gal_df = rows.iloc[off:off + args.n_gal].reset_index(drop=True)
    pool_intr, gal_intr = intr_of(pool_df), intr_of(gal_df)
    m_pool = len(pool_df)
    print(f"pool  M = {m_pool:,} in-domain rows [0,{args.pool})  "
          f"({int(np.unique(pool_df['input_index'].to_numpy()).size):,} distinct primaries)")
    print(f"galaxies = {len(gal_df):,} rows pinned at [{off},{off + args.n_gal})")
    print("p_0 = uniform over the M pool rows.  The global-bank baseline draws its bank")
    print("from this same population, so the arms differ ONLY in the proposal q.")

    # ---- pool contexts at the three stencil points, once ------------------------------
    d = args.delta
    ctx_t, pdet_t, logpd_t = {}, {}, {}
    for t in (0.0, +d, -d):
        c, p_ = pool_contexts(model, pool_df, pool_intr, (0.0, t), pre, nbr_std, dev,
                              bs=args.ctx_batch)
        ctx_t[t] = c
        pdet_t[t] = p_
        logpd_t[t] = torch.log(p_)
    with torch.no_grad():
        mu_pool = model.mu(ctx_t[0.0]).double()
    mu_scale = mu_pool.std(dim=0).clamp_min(1e-8)
    mu_pool_n = mu_pool / mu_scale
    print(f"pool mu space: dim={mu_pool.shape[1]}  per-axis pool sd="
          f"{np.array2string(mu_scale.cpu().numpy(), precision=3)}")

    # ---- population terms: exact over the whole pool, uniform, galaxy-independent ------
    lp = {t: float(np.log(float(pdet_t[t].double().mean()))) for t in (0.0, +d, -d)}
    s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                   (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
    print(f"population terms on the FULL pool (M={m_pool:,}, uniform): "
          f"<s>_sel={s_sel:+.5f}  I_sel={i_sel:+.5f}   <Pdet>={float(pdet_t[0.0].mean()):.4f}")
    print("  (identical for every arm and every K, so no arm difference can come from them)")

    pool_pdet_np = pdet_t[0.0].double().cpu().numpy()
    pool_rmag_np = pool_df["r_input_p"].to_numpy(float)
    pool_rmag_t = torch.as_tensor(pool_rmag_np, dtype=torch.float64, device=dev)

    results = {}

    for gamma in gammas:
        print("\n" + "=" * 100)
        print(f"### gamma_true = {gamma:+.4f}  (axis 2, primary only)")
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
        xhat = xhat[keep]
        n_keep = int(keep.sum())
        print(f"detection: kept {n_keep:,}/{len(gal_df):,} = {n_keep / len(gal_df):.1%} "
              f"(<Pdet>={float(pdet_g.mean()):.4f});  xhat FIXED across arms and K")
        del ctx_g
        xhat_n = (xhat.double() / mu_scale)

        for arm in arms:
            for k in ks:
                tag = f"{arm['name']}/K={k}"
                t_start = time.time()
                # bounded on BOTH the phi block (chunk*K) and the proposal block (chunk*M)
                chunk = max(1, min(n_keep, args.max_pairs // max(k, 1),
                                   max(1, args.max_qcells // m_pool)))
                gen = torch.Generator(device=dev)
                nm = sum((i + 1) * ord(c) for i, c in enumerate(arm["name"]))
                gen.manual_seed(args.seed * 977 + k * 13 + int(round(gamma * 1e4)) * 7 + nm)
                s_all, i_all = np.empty(n_keep), np.empty(n_keep)
                sA_all, sB_all = np.empty(n_keep), np.empty(n_keep)
                edd_all, vd_all = np.empty(n_keep), np.empty(n_keep)
                ess_all, essA_all = np.empty(n_keep), np.empty(n_keep)
                sdir_all, idir_all = np.empty(n_keep), np.empty(n_keep)
                nuniq, ratio_max, w_max, ess_dist = [], [], [], []
                cal_pdet, cal_rmag = np.empty(n_keep), np.empty(n_keep)
                for st in range(0, n_keep, chunk):
                    en = min(st + chunk, n_keep)
                    b = en - st
                    q = build_proposal(mu_pool_n, xhat_n[st:en], arm["alpha"], arm["frac"])
                    qsum = q.sum(dim=1)
                    assert torch.allclose(qsum, torch.ones_like(qsum), atol=1e-9), \
                        f"q does not normalise: max|sum q - 1| = {float((qsum-1).abs().max()):.2e}"
                    idx, qsel = draw_nodes(q, k, gen)                 # (b,K)
                    logcorr = -(torch.log(qsel) + np.log(m_pool))     # log p_0/q, <= -log alpha
                    ratio_max.append(float(torch.exp(logcorr).max()))
                    flat = idx.reshape(-1)
                    # likelihood-free SNIS calibration of q on two pool functionals
                    u_is = torch.exp(logcorr)
                    cal_pdet[st:en] = ((u_is * pdet_t[0.0].double()[idx]).sum(1)
                                       / u_is.sum(1)).cpu().numpy()
                    cal_rmag[st:en] = ((u_is * pool_rmag_t[idx]).sum(1)
                                       / u_is.sum(1)).cpu().numpy()
                    # these two are O(M) per galaxy; a subsample of a few hundred galaxies
                    # is plenty and keeps the long-K rungs from being dominated by them
                    ii_np = (idx[:min(b, 32)].cpu().numpy() if len(ess_dist) < 256
                             else np.empty((0, k), dtype=np.int64))
                    if ii_np.size:
                        nuniq.append(float(np.mean([len(np.unique(r)) for r in ii_np])))
                    phis = {}
                    with torch.no_grad():
                        for t in (0.0, +d, -d):
                            ll = model.log_prob_obs(
                                xhat[st:en, None, :].expand(b, k, xhat.shape[1]).reshape(-1, xhat.shape[1]),
                                ctx_t[t][flat]).view(b, k)
                            phis[t] = (ll + logpd_t[t][idx]).double().cpu().numpy()
                    p0 = phis[0.0]
                    dphi = (phis[+d] - phis[-d]) / (2 * d)
                    ddphi = (phis[+d] - 2 * p0 + phis[-d]) / d ** 2
                    lc = logcorr.cpu().numpy()
                    p0a = p0 + lc                                    # weights carry p_0/q
                    s_, i_ = score_and_information(p0a, dphi, ddphi)
                    s_all[st:en], i_all[st:en] = s_, i_
                    w = posterior_weights(p0a)
                    edd_all[st:en] = np.sum(w * ddphi, axis=1)
                    vd_all[st:en] = np.sum(w * dphi ** 2, axis=1) - s_ ** 2
                    ess_all[st:en] = 1.0 / np.sum(w ** 2, axis=1)
                    w_max.append(float(np.mean(w.max(axis=1))))
                    # ESS over K columns counts a node drawn twice as two nodes.  Under a
                    # localised proposal WITH REPLACEMENT that inflates it badly, so also
                    # collapse duplicate pool rows before measuring the support.
                    for r_ in range(ii_np.shape[0]):
                        bw = np.bincount(ii_np[r_], weights=w[r_], minlength=m_pool)
                        ess_dist.append(1.0 / np.sum(bw ** 2))
                    h = k // 2
                    sa, _ = score_and_information(p0a[:, :h], dphi[:, :h], ddphi[:, :h])
                    sb, _ = score_and_information(p0a[:, h:], dphi[:, h:], ddphi[:, h:])
                    sA_all[st:en], sB_all[st:en] = sa, sb
                    wa = posterior_weights(p0a[:, :h])
                    essA_all[st:en] = 1.0 / np.sum(wa ** 2, axis=1)
                    from scipy.special import logsumexp as _lse
                    lz = {t: _lse(phis[t] + lc, axis=1) for t in (0.0, +d, -d)}
                    sdir_all[st:en] = (lz[+d] - lz[-d]) / (2 * d)
                    idir_all[st:en] = -(lz[+d] - 2 * lz[0.0] + lz[-d]) / d ** 2
                    del q, idx, qsel, phis, p0, dphi, ddphi, w
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

                # ---------------- diagnostics -------------------------------------------
                louis = shear_estimate_louis(s_all, i_all, s_sel, i_sel)
                bart = shear_estimate_bartlett(s_all, s_sel)
                bd, ld, ratio = denominator_consistency(s_all, i_all, s_sel, i_sel)
                noise = float(np.var(sA_all - sB_all, ddof=1) / 2.0)
                cov_ab = float(np.cov(sA_all, sB_all, ddof=1)[0, 1])
                corr_ab = float(np.corrcoef(sA_all, sB_all)[0, 1])
                e_louis = boot_stat(lambda ii: shear_estimate_louis(
                    s_all[ii], i_all[ii], s_sel, i_sel), n_keep, args.boot, rng)
                e_bart = boot_stat(lambda ii: shear_estimate_bartlett(
                    s_all[ii], s_sel), n_keep, args.boot, rng)
                e_var = boot_stat(lambda ii: np.var(s_all[ii], ddof=1),
                                  n_keep, args.boot, rng)
                e_info = boot_stat(lambda ii: np.mean(i_all[ii]), n_keep, args.boot, rng)
                e_noise = boot_stat(lambda ii: np.var(sA_all[ii] - sB_all[ii], ddof=1) / 2,
                                    n_keep, args.boot, rng)
                e_corr = boot_stat(lambda ii: np.corrcoef(sA_all[ii], sB_all[ii])[0, 1],
                                   n_keep, args.boot, rng)

                print(f"\n--- {tag}  alpha={arm['alpha']} frac={arm['frac']}  "
                      f"gal chunk={chunk}   [{time.time() - t_start:.0f} s]")
                print(f"    q check: sum_k q_k = 1 (asserted);  max p0/q observed "
                      f"{max(ratio_max):.3f}  (BOUND 1/alpha = "
                      f"{1.0 / arm['alpha']:.2f})")
                print(f"    SNIS calibration on pool functionals (LIKELIHOOD-FREE): "
                      f"<Pdet> {np.mean(cal_pdet):.4f} +- {np.std(cal_pdet)/np.sqrt(n_keep):.4f} "
                      f"vs exact {pool_pdet_np.mean():.4f} | "
                      f"<r_mag> {np.mean(cal_rmag):.4f} +- "
                      f"{np.std(cal_rmag)/np.sqrt(n_keep):.4f} vs exact "
                      f"{pool_rmag_np.mean():.4f}")
                print(f"    bank: distinct nodes per galaxy {np.mean(nuniq):.0f}/{k}  "
                      f"mean max weight {np.mean(w_max):.4f}")
                print(f"    ESS full {np.mean(ess_all):.1f}  ESS/K "
                      f"{np.mean(ess_all) / k:.4f}   (half-bank ESS "
                      f"{np.mean(essA_all):.1f}, ESS/(K/2) "
                      f"{np.mean(essA_all) / (k / 2):.4f})")
                print(f"    ESS with DUPLICATE pool rows collapsed: "
                      f"{np.mean(ess_dist):.1f}  (= distinct effective support; the plain "
                      f"ESS counts a row drawn twice as two nodes)")
                print(f"    Fisher check: <s>_Ew={np.mean(s_all):+.5f} vs "
                      f"<s>_direct={np.mean(sdir_all):+.5f}  corr="
                      f"{np.corrcoef(s_all, sdir_all)[0, 1]:.6f}")
                print(f"    Var(s)={np.var(s_all, ddof=1):.4f} +- {e_var:.4f}   "
                      f"<I>={np.mean(i_all):+.4f} +- {e_info:.4f}   "
                      f"[ -E_w(phi'')={-np.mean(edd_all):+.4f} , "
                      f"Var_w(phi')={np.mean(vd_all):+.4f} ]")
                print(f"    information equality: Var(s-<s>_sel)={bd:.4f} vs "
                      f"<I>-I_sel={ld:.4f}  ratio={ratio:+.3f}  (must be 1; see round-1 "
                      f"caveat on <I>)")
                print(f"    HALF-BANK (each K/2={k//2}): corr(A,B)={corr_ab:+.4f} +- "
                      f"{e_corr:.4f}   Cov(A,B) SIGNAL={cov_ab:.4f}   "
                      f"Var(A-B)/2 NOISE={noise:.4f} +- {e_noise:.4f}")
                print(f"    ghat(5.8)={louis:+.6f} +- {e_louis:.6f}   "
                      f"ghat(5.9)={bart:+.6f} +- {e_bart:.6f}   (truth {gamma:+.4f})")
                sys.stdout.flush()

                results[(gamma, arm["name"], k)] = dict(
                    s=s_all.copy(), info=i_all.copy(), sA=sA_all.copy(), sB=sB_all.copy(),
                    kept=kept_idx.copy(), louis=louis, bart=bart, e_louis=e_louis,
                    e_bart=e_bart, var_s=float(np.var(s_all, ddof=1)), e_var=e_var,
                    info_mean=float(np.mean(i_all)), e_info=e_info,
                    edd=float(-np.mean(edd_all)), vard=float(np.mean(vd_all)),
                    noise=noise, e_noise=e_noise, corr=corr_ab, e_corr=e_corr,
                    cov=cov_ab, ess=float(np.mean(ess_all)),
                    ess_frac=float(np.mean(ess_all) / k),
                    ess_distinct=float(np.mean(ess_dist)),
                    n_distinct=float(np.mean(nuniq)),
                    ratio=ratio, cal_pdet=float(np.mean(cal_pdet)),
                    cal_rmag=float(np.mean(cal_rmag)),
                    max_ratio=float(max(ratio_max)),
                    secs=float(time.time() - t_start))
                if args.out:                                  # incremental: survive a kill
                    with open(args.out, "w") as fh:
                        json.dump({f"{g}|{a}|{kk_}": {c: v for c, v in r.items()
                                                      if not isinstance(v, np.ndarray)}
                                   for (g, a, kk_), r in results.items()}, fh, indent=1)

    # ---- noise exponent fit -----------------------------------------------------------
    print("\n" + "=" * 100)
    print("### NOISE SCALING:  Var(s^A - s^B)/2 at half-bank size K/2, fitted as K^p")
    print("### target if the proposal fixes coverage: p = -1.  Round-1 global bank: +0.26..+0.33")
    print("=" * 100)
    for gamma in gammas:
        for arm in arms:
            hs = np.array([k / 2 for k in ks], dtype=float)
            nz = np.array([results[(gamma, arm["name"], k)]["noise"] for k in ks])
            sig = np.array([results[(gamma, arm["name"], k)]["e_noise"] for k in ks])
            slope = np.polyfit(np.log(hs), np.log(nz), 1)[0]
            n_g = results[(gamma, arm["name"], ks[0])]["s"].size
            brng = np.random.default_rng(args.seed + 31)
            sl = []
            for _ in range(args.boot):
                ii = brng.integers(0, n_g, n_g)
                v = [np.var(results[(gamma, arm["name"], k)]["sA"][ii]
                            - results[(gamma, arm["name"], k)]["sB"][ii], ddof=1) / 2
                     for k in ks]
                sl.append(np.polyfit(np.log(hs), np.log(np.array(v)), 1)[0])
            print(f"  gamma={gamma:+.3f}  {arm['name']:>10}:  noise " +
                  "  ".join(f"K/2={int(h)}: {n:.4f}+-{e:.4f}" for h, n, e in zip(hs, nz, sig)) +
                  f"   ->  exponent p = {slope:+.3f} +- {float(np.std(sl)):.3f}")
            print(f"{'':>22}ESS/K " +
                  "  ".join(f"K={k}: {results[(gamma, arm['name'], k)]['ess_frac']:.4f}"
                            for k in ks) +
                  "   corr(A,B) " +
                  "  ".join(f"{results[(gamma, arm['name'], k)]['corr']:+.3f}" for k in ks))

    # ---- the denominator-free response test -------------------------------------------
    if len(gammas) >= 2:
        g0, g1 = gammas[0], gammas[-1]
        print("\n" + "=" * 100)
        print(f"### DENOMINATOR-FREE RESPONSE TEST: ghat({g1:+.3f}) - ghat({g0:+.3f}), "
              f"truth {g1 - g0:+.4f}")
        print("### paired on the galaxies detected at BOTH shears (common random numbers "
              "in the xhat draw)")
        print("=" * 100)
        for arm in arms:
            for k in ks:
                ra, rb = results[(g0, arm["name"], k)], results[(g1, arm["name"], k)]
                common, ia, ib = np.intersect1d(ra["kept"], rb["kept"],
                                                return_indices=True)
                n_c = common.size
                sa, sb = ra["s"][ia], rb["s"][ib]
                ia_, ib_ = ra["info"][ia], rb["info"][ib]
                d_b = (shear_estimate_bartlett(sb, s_sel)
                       - shear_estimate_bartlett(sa, s_sel))
                d_l = (shear_estimate_louis(sb, ib_, s_sel, i_sel)
                       - shear_estimate_louis(sa, ia_, s_sel, i_sel))
                brng = np.random.default_rng(args.seed + 101)
                vb, vl = [], []
                for _ in range(args.boot):
                    jj = brng.integers(0, n_c, n_c)
                    vb.append(shear_estimate_bartlett(sb[jj], s_sel)
                              - shear_estimate_bartlett(sa[jj], s_sel))
                    vl.append(shear_estimate_louis(sb[jj], ib_[jj], s_sel, i_sel)
                              - shear_estimate_louis(sa[jj], ia_[jj], s_sel, i_sel))
                print(f"  {arm['name']:>10}  K={k:>5}  (N_paired={n_c})   "
                      f"d ghat(5.9) = {d_b:+.6f} +- {float(np.std(vb)):.6f}   "
                      f"[{d_b / (g1 - g0):+.1%} of truth]   "
                      f"d ghat(5.8) = {d_l:+.6f} +- {float(np.std(vl)):.6f}")

    if args.out:
        dump = {f"{g}|{a}|{k}": {kk: vv for kk, vv in v.items()
                                 if not isinstance(vv, np.ndarray)}
                for (g, a, k), v in results.items()}
        with open(args.out, "w") as fh:
            json.dump(dump, fh, indent=1)
        print(f"\nwrote {args.out}")
    print("\n### DONE ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
