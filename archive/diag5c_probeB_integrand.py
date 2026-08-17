#!/usr/bin/env python -B
"""PROBE B: are the WEIGHTS broken, or is the INTEGRAND broken?

Everything is held EXACTLY as in `diag5c_repro.py` -- same catalogue slices, same node
bank, same galaxies, same seeded data draw, same posterior weights `w_k propto
exp(phi_k(0))` -- and only the thing being AVERAGED is swapped.  Three integrands, side by
side in one process at every K:

  (i)   phi'          the real integrand (baseline; must reproduce cont.169's Var(s)).
  (ii)  TAME SIGNAL   g_k = the node's own true intrinsic e1_p (a smooth scene coordinate).
        `E_w[g]` is then the posterior mean of the primary's true shape -- a genuinely
        meaningful per-galaxy quantity.  Healthy weights => HIGH half-bank correlation.
        A second tame signal, the node's true primary magnitude `r_input_p`, is carried
        alongside: it is the scene coordinate the measured vector constrains most
        directly, so it separates "the weights carry no information at all" from "the
        weights carry flux information but not shape information".
  (iii) PURE NOISE    g_k ~ N(0,1) drawn once per NODE, independent of everything and
        shared across galaxies.  The true posterior mean is 0 for every galaxy, so
        `Var_i(E_w[g])` is pure weight-induced noise with NO signal component.  With g
        independent of w,  E_g[ Var_i(s) ] = mean_i sum_k w_ik^2  -  sum_k wbar_k^2
                                          = mean_i (1/ESS_i)  -  1/ESS_pop,
        so this measures the averaging power of the weights with the integrand removed as
        a variable.  Several independent draws are used so a single unlucky realisation
        cannot be read as a result.

THE DECISIVE COMPARISON: does Var(s_noise) fall like 1/ESS as K grows?  If it does, the
weights average correctly and the whole pathology lives in phi'.  If it does not, the
weights themselves fail to average and no integrand can be rescued by a bigger bank.

NOTE on which ESS: cont.169 quotes `mean_i ESS_i`, but the noise floor is set by
`mean_i (1/ESS_i)`, which is >= 1/mean_i(ESS_i) by Jensen and can be dominated by a
minority of galaxies with tiny ESS.  Both are printed, together with the ESS distribution
and an ESS-quartile breakdown, because the gap between them is itself a candidate
explanation for a Var(s) that does not fall when the mean ESS rises.

Usage:
    python scripts/diag5c_probeB_integrand.py --n-gal 20000 --n-nodes 2000,6000,20000
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


def boot_stat(fn, n, n_boot, seed):
    """Bootstrap standard error of `fn(idx)` over galaxies."""
    rng = np.random.default_rng(seed)
    return float(np.std([fn(rng.integers(0, n, n)) for _ in range(n_boot)]))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--n-nodes", default="2000,6000,20000")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.05,
                    help="second data leg, used ONLY for the tame-signal response check")
    ap.add_argument("--no-gamma-leg", action="store_true")
    ap.add_argument("--n-draws", type=int, default=20, help="independent N(0,1) probe draws")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    nodes = [int(x) for x in args.n_nodes.split(",")]
    kmax = max(nodes)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pre, nbr_std, tgt_std, meta = rebuild(args.checkpoint, dev)
    print(f"device {dev}  primary_only_shear={meta.get('primary_only_shear')}", flush=True)

    # ---- EXACTLY the diag5c_repro slicing so the phi' column is the cont.169 baseline ----
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

    # node-level scene coordinates used as TAME integrands (fixed, gamma-independent)
    pool_e1p = np.asarray(pool_intr["e1p"], dtype=np.float64)
    pool_mag = pool["r_input_p"].to_numpy(float)
    print(f"tame integrands over the pool: e1p mean {pool_e1p.mean():+.4f} sd "
          f"{pool_e1p.std():.4f} | r_input_p mean {pool_mag.mean():.3f} sd {pool_mag.std():.3f}",
          flush=True)

    # ---- data, same seeded draw as diag5c_repro ----------------------------------------
    legs = [0.0] if args.no_gamma_leg else [0.0, float(args.gamma)]
    gal_mag_all = gal_df["r_input_p"].to_numpy(float)
    xhats, gal_e1p_keep, gal_mag_keep = {}, {}, {}
    for g in legs:
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
        kmask = keep.cpu().numpy().astype(bool)
        gal_e1p_keep[g] = np.asarray(gal_intr["e1p"], dtype=np.float64)[kmask]
        gal_mag_keep[g] = gal_mag_all[kmask]
        print(f"  gamma={g:+.3f}: kept {int(keep.sum()):,}/{len(gal_df):,}", flush=True)
        del ctx_g, pdet_g

    rng = np.random.default_rng(args.seed + 777)
    print(f"\ndelta={d}  n_draws={args.n_draws}  n_boot={args.n_boot}  seed={args.seed}",
          flush=True)

    for K in nodes:
        node_df = pool.iloc[:K].reset_index(drop=True)
        node_intr = {k: v[:K] for k, v in pool_intr.items()}
        half = K // 2
        A, B = np.arange(half), np.arange(half, 2 * half)
        e1p_k = pool_e1p[:K]
        mag_k = pool_mag[:K]
        G = rng.standard_normal((K, args.n_draws))          # the pure-noise probes

        print(f"\n================ K = {K:,} ================", flush=True)

        # ---- optional gamma leg: does E_w[e1p_true] RESPOND to the true shear? ----------
        tame_resp = {}
        if not args.no_gamma_leg:
            gleg = float(args.gamma)
            cg, pdg = scene_context(model, node_df, node_intr, (0.0, 0.0), pre, nbr_std, dev)
            p0g = phi_block(model, xhats[gleg], cg, torch.log(pdg), args.chunk)
            del cg, pdg
            wg = posterior_weights(p0g)
            del p0g
            se1 = wg @ e1p_k
            tame_resp["e1p"] = float(np.mean(se1))
            tame_resp["mag"] = float(np.mean(wg @ mag_k))
            tame_resp["own_corr"] = corr(se1, gal_e1p_keep[gleg])
            del wg, se1

        # ---- phi at the three stencil points on the gamma=0 data -----------------------
        c0, pd0 = scene_context(model, node_df, node_intr, (0.0, 0.0), pre, nbr_std, dev)
        phi0 = phi_block(model, xhats[0.0], c0, torch.log(pd0), args.chunk)
        del c0, pd0
        cp, pdp = scene_context(model, node_df, node_intr, (0.0, +d), pre, nbr_std, dev)
        php = phi_block(model, xhats[0.0], cp, torch.log(pdp), args.chunk)
        del cp, pdp
        cm, pdm = scene_context(model, node_df, node_intr, (0.0, -d), pre, nbr_std, dev)
        phm = phi_block(model, xhats[0.0], cm, torch.log(pdm), args.chunk)
        del cm, pdm
        np.subtract(php, phm, out=php)
        php /= (2.0 * d)
        d1 = php
        del phm

        n = phi0.shape[0]

        # ---- half-bank scores first (they need column slices of phi0) ------------------
        halves = {}
        for tag, cols in (("A", A), ("B", B)):
            wh = posterior_weights(phi0[:, cols])
            halves[tag] = dict(
                phi=np.sum(wh * d1[:, cols], axis=1),
                e1p=wh @ e1p_k[cols],
                mag=wh @ mag_k[cols],
                noise=wh @ G[cols],
                ess=float(np.mean(1.0 / np.sum(wh ** 2, axis=1))),
            )
            del wh

        # ---- full-bank weights ---------------------------------------------------------
        w = posterior_weights(phi0)
        del phi0
        inv_ess = np.sum(w ** 2, axis=1)                 # = 1/ESS_i
        ess_i = 1.0 / inv_ess
        wbar = w.mean(axis=0)
        inv_ess_pop = float(np.sum(wbar ** 2))

        s_phi = np.sum(w * d1, axis=1)
        s_e1p = w @ e1p_k
        s_mag = w @ mag_k
        s_noise = w @ G                                   # (n, n_draws)
        del w, d1

        # ---- ESS anatomy ---------------------------------------------------------------
        q = np.percentile(ess_i, [1, 5, 25, 50, 75, 95, 99])
        print(f"ESS_i: mean {ess_i.mean():8.2f}  median {q[3]:8.2f}  "
              f"p1 {q[0]:6.2f} p5 {q[1]:6.2f} p25 {q[2]:7.2f} p75 {q[4]:8.2f} "
              f"p95 {q[5]:8.2f} p99 {q[6]:8.2f}")
        print(f"       mean(1/ESS_i) = {inv_ess.mean():.6f}   1/mean(ESS_i) = "
              f"{1.0 / ess_i.mean():.6f}   ratio(Jensen) = {inv_ess.mean() * ess_i.mean():.2f}"
              f"   1/ESS_pop = {inv_ess_pop:.3e}")
        print(f"       half-bank ESS: A {halves['A']['ess']:.2f}  B {halves['B']['ess']:.2f}")

        pred_noise = float(inv_ess.mean() - inv_ess_pop)
        print(f"\n  {'integrand':<16}{'Var(s)':>12}{'+- boot':>10}{'corr(sA,sB)':>13}"
              f"{'+- boot':>10}{'sig2_half':>11}{'Var/Var_prior':>15}")

        rowspecs = [
            ("(i)  phi'", s_phi, halves["A"]["phi"], halves["B"]["phi"], None),
            ("(ii) e1p_true", s_e1p, halves["A"]["e1p"], halves["B"]["e1p"],
             float(np.var(e1p_k))),
            ("(ii+) mag_true", s_mag, halves["A"]["mag"], halves["B"]["mag"],
             float(np.var(mag_k))),
        ]
        for name, sf, sa, sb, vprior in rowspecs:
            v = float(np.var(sf))
            c = corr(sa, sb)
            s2h = float(np.var(sa) - np.cov(sa, sb)[0, 1])
            se_v = boot_stat(lambda i, sf=sf: float(np.var(sf[i])), n, args.n_boot, args.seed)
            se_c = boot_stat(lambda i, sa=sa, sb=sb: corr(sa[i], sb[i]),
                             n, args.n_boot, args.seed)
            frac = "--" if vprior is None else f"{v / vprior:14.4f}"
            print(f"  {name:<16}{v:12.4f}{se_v:10.4f}{c:13.4f}{se_c:10.4f}{s2h:11.4f}"
                  f"{frac:>15}")

        # ---- the noise probe, over independent draws ----------------------------------
        vs = np.array([float(np.var(s_noise[:, j])) for j in range(args.n_draws)])
        cs = np.array([corr(halves["A"]["noise"][:, j], halves["B"]["noise"][:, j])
                       for j in range(args.n_draws)])
        s2hs = np.array([float(np.var(halves["A"]["noise"][:, j])
                               - np.cov(halves["A"]["noise"][:, j],
                                        halves["B"]["noise"][:, j])[0, 1])
                         for j in range(args.n_draws)])
        print(f"  {'(iii) N(0,1)':<16}{vs.mean():12.4f}{vs.std():10.4f}{cs.mean():13.4f}"
              f"{cs.std():10.4f}{s2hs.mean():11.4f}{vs.mean():15.4f}")
        print(f"        draws: Var(s) min {vs.min():.4f} max {vs.max():.4f}   "
              f"corr min {cs.min():+.4f} max {cs.max():+.4f}")
        print(f"        PREDICTION mean(1/ESS_i) - 1/ESS_pop = {pred_noise:.6f}   "
              f"measured/predicted = {vs.mean() / pred_noise:.4f}")
        print("        (corr for the noise probe is ~0 BY CONSTRUCTION -- disjoint nodes "
              "carry independent g; it is the null calibration for the corr column.)")

        # ---- does the phi' score behave like the noise probe per galaxy? ---------------
        s2n = (s_noise ** 2).mean(axis=1)                 # per-galaxy noise power
        print(f"\n  per-galaxy: corr(s_noise^2, 1/ESS_i) = {corr(s2n, inv_ess):+.4f}   "
              f"(must be ~1 if the weights behave as theory says)")
        print(f"              corr(phi'^2 , 1/ESS_i) = {corr(s_phi ** 2, inv_ess):+.4f}   "
              f"corr(|s_e1p - <s_e1p>|, 1/ESS_i) = "
              f"{corr(np.abs(s_e1p - s_e1p.mean()), inv_ess):+.4f}")

        # ESS quartile breakdown: where does Var(phi') live?
        order = np.argsort(ess_i)
        qs = np.array_split(order, 4)
        print("  ESS quartile   <ESS>    <1/ESS>    Var(phi')   Var(noise)   corrAB(phi')"
              "   Var(e1p)")
        for qi, idx in enumerate(qs):
            print(f"    Q{qi + 1} (n={len(idx):5,}) {ess_i[idx].mean():9.2f} "
                  f"{inv_ess[idx].mean():10.5f} {float(np.var(s_phi[idx])):11.3f} "
                  f"{float(np.mean([np.var(s_noise[idx, j]) for j in range(args.n_draws)])):12.5f}"
                  f" {corr(halves['A']['phi'][idx], halves['B']['phi'][idx]):13.4f}"
                  f" {float(np.var(s_e1p[idx])):10.5f}")

        # ---- is the posterior mean of the true shape actually tracking the galaxy? -----
        own = gal_e1p_keep[0.0]
        own_mag = gal_mag_keep[0.0]
        print("\n  TAME-SIGNAL REALITY CHECK (gamma=0 data):")
        print(f"    corr(E_w[e1p], own true e1p) = {corr(s_e1p, own):+.4f}   "
              f"slope = {float(np.polyfit(own, s_e1p, 1)[0]):+.4f}   "
              f"sd(E_w[e1p]) = {s_e1p.std():.4f} vs prior sd {e1p_k.std():.4f}")
        print(f"    corr(E_w[mag], own true mag) = {corr(s_mag, own_mag):+.4f}   "
              f"slope = {float(np.polyfit(own_mag, s_mag, 1)[0]):+.4f}   "
              f"sd(E_w[mag]) = {s_mag.std():.4f} vs prior sd {mag_k.std():.4f}")
        if tame_resp:
            base_e1p = float(np.mean(s_e1p))
            print(f"    response: <E_w[e1p]> = {base_e1p:+.6f} at gamma=0 -> "
                  f"{tame_resp['e1p']:+.6f} at gamma={args.gamma}   "
                  f"d/dgamma = {(tame_resp['e1p'] - base_e1p) / args.gamma:+.4f}  "
                  f"(a healthy posterior mean of the true shape should NOT respond much; "
                  f"the SHEARED shape should)")
            print(f"              <E_w[mag]> {float(np.mean(s_mag)):.5f} -> "
                  f"{tame_resp['mag']:.5f}")

        del s_phi, s_e1p, s_mag, s_noise, halves, ess_i, inv_ess

    print("\nREAD: if Var(s_noise) falls ~ mean(1/ESS_i) as K grows, the weights average "
          "correctly and the pathology is in phi'.  If it is flat, the weights themselves "
          "do not average and no integrand can be rescued by a bigger bank.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
