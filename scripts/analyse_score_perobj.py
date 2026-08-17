"""Why the §5B estimator does not need `R_blend`, from the per-object arrays.

Transport and the score route combine per-object responses differently, and on constgold
they disagree by a lot.  Writing `r_i` for the simulation's per-object measured response,
`a_i` for the flow's, `b_i = r_i - a_i` for the part the flow does not model (mostly the
neighbours' shear), and `I_i` for the per-object information:

    transport      m + 1 = <r> / <a>                       ratio of POPULATION means
    score (2.6)    ghat/g = <I_i r_i / a_i> / <I_i>        an INFORMATION-WEIGHTED mean
                                                           of the per-object ratio

The second follows from `ghat = sum_i (s_i^+ - s_i^-)/2 / sum_i I_i` together with the
location-family relation `ds/dehat = I / a`, exact for a flow whose residual density is
blind to the true shape.  The two agree only if `b_i / a_i` is uncorrelated with `I_i`.
This script measures that correlation directly: it bins the per-object `ghat_i` by
blending, by `R_blend`, by magnitude and by information, and reports the transport ratio
and the score ratio side by side in every bin.

Input: the `--perobj-dump` npz written by `eval_score_response.py --mode constgold`.
"""

import argparse
import os

import numpy as np


def weighted(vals, w):
    return float(np.sum(vals * w) / np.sum(w))


def boot_by_case(num, den, cases, n_boot=300, seed=0):
    """Per-case bootstrap of `sum(num)/sum(den)`, the form every ratio here takes."""
    uc = np.unique(cases)
    idx = {c: np.flatnonzero(cases == c) for c in uc}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[c] for c in rng.choice(uc, size=len(uc), replace=True)])
        d = np.sum(den[sel])
        if d:
            out.append(np.sum(num[sel]) / d)
    return float(np.std(out)) if out else float("nan")


def table(name, groups, d, n_boot):
    """One split: per group, the transport ratio and the score ratio."""
    g, cases = float(d["g"]), d["case"]
    s_anti = 0.5 * (d["s_plus"] - d["s_minus"])
    i_anti = 0.5 * (d["i_plus"] + d["i_minus"])
    r_sim, r_blend = d["r_sim"], d["r_blend"]
    print(f"\n--- split by {name} ---")
    print(f"  {'bin':>18} {'N':>10} {'<R_blend>':>10} {'R_sim':>8} "
          f"{'ghat/g':>9} {'+/-':>7} {'m_5B':>9}")
    for label, mask in groups:
        n = int(mask.sum())
        if n < 500:
            continue
        ghat = float(np.sum(s_anti[mask]) / np.sum(i_anti[mask]))
        err = boot_by_case(s_anti[mask], i_anti[mask], cases[mask], n_boot=n_boot)
        print(f"  {label:>18} {n:>10,} {r_blend[mask].mean():>10.4f} "
              f"{r_sim[mask].mean():>8.4f} {ghat / g:>9.4f} {err / g:>7.4f} "
              f"{ghat / g - 1:>+8.2%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    d = np.load(args.npz)
    g = float(d["g"])
    s_anti = 0.5 * (d["s_plus"] - d["s_minus"])
    i_anti = 0.5 * (d["i_plus"] + d["i_minus"])
    r_sim, r_blend = d["r_sim"], d["r_blend"]
    cases = d["case"]
    n = len(r_sim)

    print(f"=== {os.path.basename(args.npz)}: N={n:,}, |g|={g:.4f}, "
          f"{len(np.unique(cases))} cases ===")
    ghat = float(np.sum(s_anti) / np.sum(i_anti))
    err = boot_by_case(s_anti, i_anti, cases, n_boot=args.n_boot)
    print(f"GLOBAL   score  ghat/g = {ghat / g:.4f} +/- {err / g:.4f}   "
          f"(m_5B = {ghat / g - 1:+.2%})")
    print(f"GLOBAL   transport R_sim = {float(d['R_sim']):.4f}, "
          f"R_flow = {float(d['R_flow']):.4f}, R_blend = {float(d['R_blend']):.4f} "
          f"-> R_sim/R_flow = {float(d['R_sim']) / float(d['R_flow']):.4f}, "
          f"m = {float(d['R_sim']) / (float(d['R_flow']) + float(d['R_blend'])) - 1:+.2%}")

    # The mechanism.  a_i is not stored, but the location-family relation gives it back:
    # ds/dehat = I/a, and the antithetic difference is s^+ - s^- = (ds/dehat)(2 g r_i),
    # so  s_anti / (g r_i) = I_i / a_i  and  a_i = g r_i I_i / s_anti.  Averaged over
    # objects that is far too noisy per object (r_i has 10x its own mean as scatter), but
    # the SUMS are exactly what the two estimators contrast:
    print("\n--- the two ways of combining per-object responses ---")
    print(f"  transport  <r>/<a>          = {float(d['R_sim']) / float(d['R_flow']):.4f}"
          f"   (population means)")
    print(f"  score      <I r/a>/<I>      = {ghat / g:.4f}   (information-weighted)")
    print(f"  they differ iff  b_i/a_i = (r_i - a_i)/a_i  correlates with  I_i.")

    # is the information anticorrelated with the blend response?
    q = np.quantile(i_anti, np.linspace(0, 1, 6))
    q[0] -= 1e-9
    q[-1] += 1e-9
    ib = np.clip(np.digitize(i_anti, q) - 1, 0, 4)
    print(f"\n  {'I quintile':>12} {'<I>':>9} {'<R_blend>':>10} {'<R_sim>':>9} "
          f"{'<mag_auto>':>11} {'blended':>8}")
    for k in range(5):
        m = ib == k
        print(f"  {k + 1:>12} {i_anti[m].mean():>9.3f} {r_blend[m].mean():>10.4f} "
              f"{r_sim[m].mean():>9.4f} {d['mag_auto'][m].mean():>11.3f} "
              f"{d['neighbored'][m].mean():>8.3f}")
    c = np.corrcoef(i_anti, r_blend)[0, 1]
    print(f"  corr(I_i, R_blend_i) = {c:+.4f}"
          f"   -> {'blend response sits in LOW-information objects' if c < 0 else 'no anticorrelation'}")

    nb = d["neighbored"].astype(bool)
    table("blending", [("isolated", ~nb), ("blended", nb)], d, args.n_boot)

    qb = np.quantile(r_blend[r_blend > 0.02], [0, 0.25, 0.5, 0.75, 1.0])
    groups = [("R_blend<0.02", r_blend <= 0.02)]
    for k in range(4):
        groups.append((f"R_b {qb[k]:.3f}-{qb[k+1]:.3f}",
                       (r_blend > max(qb[k], 0.02)) & (r_blend <= qb[k + 1])))
    table("R_blend", groups, d, args.n_boot)

    mag = d["mag_auto"]
    edges = [0, 23, 24, 24.5, 25, 25.5, 26, 99]
    table("measured mag", [(f"{edges[k]}-{edges[k+1]}",
                            (mag > edges[k]) & (mag <= edges[k + 1]))
                           for k in range(len(edges) - 1)], d, args.n_boot)

    print("\n  NOTE on the information split below: I_i is a function of ehat_i, so "
          "cutting on it\n  is cutting on the data, which breaks E[s | selected] = I gamma.  "
          "Read those rows as\n  the WEIGHTING mechanism (which objects carry the sum), "
          "not as per-bin biases; quintile 1\n  has <I> < 0, so its ratio is not even "
          "a shear.  The splits on R_blend, blending and\n  magnitude are the honest "
          "ones -- those covariates are shear-independent.")
    table("information (see note)", [(f"I quintile {k + 1}", ib == k) for k in range(5)],
          d, args.n_boot)

    frac = np.cumsum(np.sort(i_anti)[::-1])
    frac = frac / frac[-1]
    n20 = int(0.2 * len(i_anti))
    print(f"\n  the top 20% of objects by information carry {frac[n20]:.1%} of sum(I), "
          f"and their\n  mean R_blend is {r_blend[ib == 4].mean():.4f} against "
          f"{r_blend.mean():.4f} for the catalogue -- the score\n  estimator's effective "
          f"sample is the part of it that is barely blended.")


if __name__ == "__main__":
    main()
