"""Split-bank decomposition of Var(s) for the INFERENCE.md 5C Lagrangian estimator.

The failure signature on the Gold-V2 model is Var(s) / <I> = 7-30 where the Bartlett
identity demands 1.  Two very different diseases produce it:

  (a) s_i is fine and only the SECOND moment is contaminated by node-bank Monte-Carlo
      noise (s_i = signal + eps_i with Var(eps) >> Var(signal));
  (b) s_i genuinely varies that much across galaxies and it is <I> that is wrong.

Splitting the node bank into two INDEPENDENT halves A and B (disjoint catalogue rows),
with the SAME galaxies and the SAME data xhat, separates them without any truth:

    Cov(s^A, s^B)        -> variance of the bank-reproducible part of s   (SIGNAL)
    Var(s^A - s^B) / 2   -> variance of the bank noise at bank size K/2   (NOISE)
    corr(s^A, s^B)       -> reliability of a single-bank s_i

CAVEAT that must be stated with the numbers: s^A and s^B are self-normalised importance
estimates, so each carries a per-galaxy BIAS b_i(K/2) that is COMMON to both halves.
Cov(s^A, s^B) therefore estimates Var_i( E[s^{K/2} | galaxy i] ), i.e. the variance of the
bias-included expected score at bank size K/2, not the variance of the exact score.  It is
the right "reproducible across banks" quantity; it is an upper bound on the true signal
variance only up to that bias term, whose K-dependence the ladder exposes.

Everything is imported from the shared modules; nothing here is copied.

    python scripts/diag5c_splitbank.py --n-gal 2000 --gal-offset 16000 \
        --n-nodes 1000,4000,16000 --gammas 0,0.05
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

from closure_v2_lagrangian import (  # noqa: E402
    CAT,
    CKPT,
    load_rows,
    phi_block,
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


def intr_of(df):
    d = {}
    d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
    d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
    return d


def boot_ci(fn, n, reps, rng):
    """Bootstrap over galaxies; returns the standard deviation of the statistic."""
    vals = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        vals[r] = fn(idx)
    return float(np.std(vals))


def decompose(a, b, full, label, reps, rng):
    """Signal / noise decomposition of a per-galaxy quantity from two half banks."""
    n = a.size
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    noise = float(np.var(a - b, ddof=1) / 2.0)
    var_full = float(np.var(full, ddof=1))
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    e_cov = boot_ci(lambda i: np.cov(a[i], b[i], ddof=1)[0, 1], n, reps, rng)
    e_cor = boot_ci(lambda i: np.corrcoef(a[i], b[i])[0, 1], n, reps, rng)
    return dict(label=label, mean_a=float(a.mean()), mean_b=float(b.mean()),
                mean_full=float(full.mean()), var_a=float(np.var(a, ddof=1)),
                var_b=float(np.var(b, ddof=1)), var_full=var_full,
                cov=cov, cov_err=e_cov, noise=noise, corr=corr, corr_err=e_cor)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=2000)
    ap.add_argument("--gal-offset", type=int, default=16000,
                    help="galaxies come from this FIXED in-domain row offset; must be >= "
                         "max(--n-nodes) so the bank never overlaps the galaxy sample and "
                         "K is the only thing that changes along the ladder")
    ap.add_argument("--n-nodes", default="1000,4000,16000")
    ap.add_argument("--gammas", default="0,0.05")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-pairs", type=int, default=500_000)
    ap.add_argument("--split-mode", default="group",
                    choices=["block", "interleave", "random", "group"],
                    help="how the K nodes are cut into two halves.  The catalogue is "
                         "PAIR-ANNOTATED: ~8 consecutive rows share one `input_index`, i.e. "
                         "one primary truth paired with different neighbours (lag-1 "
                         "autocorrelation of r_input_p = 0.87).  So the halves are only "
                         "INDEPENDENT banks if no primary appears in both.  'group' "
                         "(default) randomly assigns whole `input_index` groups and is the "
                         "only correct one.  'block' = rows [0,K/2) vs [K/2,K): independent, "
                         "but not exchangeable if the file drifts.  'random' = random row "
                         "permutation: splits groups across halves, so it SHARES primaries. "
                         "'interleave' = even/odd rows: shares almost every primary, so its "
                         "corr(A,B) is an artifact and its Var(A-B)/2 is not bank noise.  "
                         "The four together bound the answer.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    ks = [int(x) for x in args.n_nodes.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    if args.gal_offset < max(ks):
        raise SystemExit("--gal-offset must be >= max(--n-nodes)")
    rng = np.random.default_rng(args.seed + 7)

    print(f"device={args.device}  torch={torch.__version__}")
    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  metadata: {meta}")

    need = args.gal_offset + args.n_gal
    rows = load_rows(args.catalogue, need * 6)
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

    gal_df = rows.iloc[args.gal_offset:args.gal_offset + args.n_gal].reset_index(drop=True)
    gal_intr = intr_of(gal_df)
    print(f"galaxies PINNED at rows [{args.gal_offset}, {args.gal_offset + args.n_gal}) "
          f"= {len(gal_df):,} scenes; banks are rows [0, K) for each K")

    # node banks: rows[:K].  The A/B halves are rows[:K/2] and rows[K/2:K], disjoint by
    # construction and drawn from the same p_0 as each other and as the galaxies.
    max_k = max(ks)
    node_df_all = rows.iloc[:max_k].reset_index(drop=True)
    node_intr_all = intr_of(node_df_all)

    for gamma in gammas:
        print("\n" + "=" * 100)
        print(f"### gamma_true = {gamma:+.4f}   (shear on axis 2, primary only)")
        print("=" * 100)
        # ---- data drawn from the model ONCE per gamma; identical for every K -----------
        torch.manual_seed(args.seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, (0.0, gamma),
                                      pre, nbr_std, dev)
        with torch.no_grad():
            xhat = model.mean_flow.sample(ctx_g, n_samples=1)
            if xhat.dim() == 3:
                xhat = xhat[:, 0, :]
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
        xhat = xhat[keep]
        n_keep = int(keep.sum())
        print(f"detection: kept {n_keep:,}/{len(gal_df):,} = {n_keep / len(gal_df):.1%}  "
              f"(<Pdet>={float(pdet_g.mean()):.4f})   xhat is FIXED across the K ladder")

        for k in ks:
            k2 = k // 2
            nd = node_df_all.iloc[:k].reset_index(drop=True)
            ni = {key: v[:k] for key, v in node_intr_all.items()}
            phis, pdets = {}, {}
            for t in (0.0, +args.delta, -args.delta):
                c, pd_ = scene_context(model, nd, ni, (0.0, t), pre, nbr_std, dev)
                phis[t] = phi_block(model, xhat, c, torch.log(pd_), args.chunk,
                                    max_pairs=args.max_pairs)
                pdets[t] = pd_.double().cpu().numpy()
                del c, pd_
            p0, pp, pm = phis[0.0], phis[+args.delta], phis[-args.delta]
            d1 = (pp - pm) / (2 * args.delta)
            d2 = (pp - 2 * p0 + pm) / args.delta ** 2

            srng = np.random.default_rng(args.seed + 1000 + k)
            if args.split_mode == "block":
                ia, ib = np.arange(0, k2), np.arange(k2, 2 * k2)
            elif args.split_mode == "interleave":
                ia, ib = np.arange(0, 2 * k2, 2), np.arange(1, 2 * k2, 2)
            elif args.split_mode == "random":
                perm = srng.permutation(k)
                ia, ib = np.sort(perm[:k2]), np.sort(perm[k2:2 * k2])
            else:                                             # group: whole primaries
                gid = nd["input_index"].to_numpy()
                uq = np.unique(gid)
                side = srng.integers(0, 2, uq.size)
                lut = dict(zip(uq.tolist(), side.tolist()))
                sd = np.array([lut[g] for g in gid.tolist()])
                ia, ib = np.flatnonzero(sd == 0), np.flatnonzero(sd == 1)
                m = min(ia.size, ib.size)                     # matched bank sizes
                ia, ib = ia[:m], ib[:m]
            ifull = np.arange(0, k)
            nu_a = int(np.unique(nd["input_index"].to_numpy()[ia]).size)
            nu_b = int(np.unique(nd["input_index"].to_numpy()[ib]).size)
            nu_ab = int(np.intersect1d(nd["input_index"].to_numpy()[ia],
                                       nd["input_index"].to_numpy()[ib]).size)
            nu_f = int(np.unique(nd["input_index"].to_numpy()).size)

            def block(sl):
                s_, i_ = score_and_information(p0[:, sl], d1[:, sl], d2[:, sl])
                w = posterior_weights(p0[:, sl])
                ess = float(np.mean(1.0 / np.sum(w ** 2, axis=1)))
                lp = {t: float(np.log(v[sl].mean())) for t, v in pdets.items()}
                ss, ii = selection_terms(
                    lp[0.0], (lp[+args.delta] - lp[-args.delta]) / (2 * args.delta),
                    (lp[+args.delta] - 2 * lp[0.0] + lp[-args.delta]) / args.delta ** 2)
                return s_, i_, ess, ss, ii

            sA, iA, essA, ssA, isA = block(ia)
            sB, iB, essB, ssB, isB = block(ib)
            sF, iF, essF, ssF, isF = block(ifull)

            ds = decompose(sA, sB, sF, "s", args.boot, rng)
            di = decompose(iA, iB, iF, "I", args.boot, rng)
            bd, ld, ratio = denominator_consistency(sF, iF, ssF, isF)
            louis = shear_estimate_louis(sF, iF, ssF, isF)
            bart = shear_estimate_bartlett(sF, ssF)

            print(f"\n--- K = {k}  (halves of {ia.size}/{ib.size} rows, "
                  f"split={args.split_mode})   "
                  f"ESS: A {essA:.1f} B {essB:.1f} "
                  f"full {essF:.1f}   <s>_sel={ssF:+.5f} I_sel={isF:+.5f}")
            print(f"    distinct primaries (input_index): full {nu_f} of {k} rows"
                  f" | A {nu_a}  B {nu_b}  SHARED between halves {nu_ab}"
                  f"  (shared must be 0 for A,B to be independent banks)")
            print(f"    ghat(5.8)={louis:+.6f}  ghat(5.9)={bart:+.6f}   "
                  f"Var(s-<s>_sel)={bd:.3f}  <I>-I_sel={ld:.3f}  ratio={ratio:+.3f}")
            for d in (ds, di):
                q = d["label"]
                print(f"    [{q}] mean: A {d['mean_a']:+.4f}  B {d['mean_b']:+.4f}  "
                      f"full {d['mean_full']:+.4f}")
                print(f"    [{q}] Var(full,K={k})   = {d['var_full']:12.4f}"
                      f"      Var(half,K={k2}) = A {d['var_a']:.4f} B {d['var_b']:.4f}")
                print(f"    [{q}] Cov(A,B) SIGNAL  = {d['cov']:12.4f} +- {d['cov_err']:.4f}")
                print(f"    [{q}] Var(A-B)/2 NOISE = {d['noise']:12.4f}   (bank size {k2})")
                print(f"    [{q}] corr(A,B)        = {d['corr']:12.4f} +- {d['corr_err']:.4f}")
            print(f"    RATIOS vs <I^full>={di['mean_full']:+.4f}:  "
                  f"Var(s^full)/<I> = {ds['var_full'] / di['mean_full']:+.3f}   "
                  f"Cov(s^A,s^B)/<I> = {ds['cov'] / di['mean_full']:+.3f}")
            print(f"    signal+noise check: Cov + Var(A-B)/2 = "
                  f"{ds['cov'] + ds['noise']:.4f}  vs Var(s^half) "
                  f"{0.5 * (ds['var_a'] + ds['var_b']):.4f} (must match by identity); "
                  f"Var(s^full,K) = {ds['var_full']:.4f}")
            print(f"    required <I> for Bartlett if s is genuine signal: "
                  f"{ds['cov']:.4f}  (measured <I^full> = {di['mean_full']:+.4f})")
            sys.stdout.flush()
            del p0, pp, pm, d1, d2, phis
    print("\n### DONE ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
