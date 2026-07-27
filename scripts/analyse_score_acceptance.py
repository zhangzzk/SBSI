"""The §5B estimator on the ACCEPTANCE population, not the full catalogue.

`GOALS.md:54` defines the deliverable over a TRUE-property cut on the primary --
`Re_input_p > 0.3`, `r_input_p < 26`, neighbours full population -- and that is the sample
`|m| <= 0.3%` has to hold on.  Every constgold number reported so far is over the full
source-selected catalogue instead, of which only ~43% passes that cut, so the faintest and
smallest bins that dominate the reported structure sit entirely outside the deliverable.

This script re-splits an existing per-object dump on the acceptance cut.  Dumps written
before the `Re_input_p` field was added carry the true magnitude but not the true size, so
the true size is recovered by re-running `load_constgold` with the identical arguments --
deterministic, same rows, same order -- and the alignment is VERIFIED against `r_input_p`
before anything is reported.  A mismatch aborts rather than quietly mis-joining.

Usage:
    python scripts/analyse_score_acceptance.py <dump.npz> [--max-rows N] [--min-case 40]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def boot_by_case(num, den, cases, n_boot=300, seed=0):
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


def recover_true_size(d, args):
    """`Re_input_p` for a dump that predates the field, via a deterministic re-load."""
    if "Re_input_p" in d:
        return d["Re_input_p"]
    print("  dump has no Re_input_p -- re-loading constgold to recover it", flush=True)
    import types
    from scripts.eval_score_response import load_constgold, GOLD_CAT
    a = types.SimpleNamespace(
        catalogue=args.catalogue or GOLD_CAT, min_case=args.min_case,
        max_rows=len(d["r_sim"]), no_source_selection=False,
        crowd_flux_lookup=args.crowd_flux_lookup, meas_prim_lookup=args.meas_prim_lookup,
        blend_lookup=args.blend_lookup)
    df = load_constgold(a)
    if len(df) != len(d["r_sim"]):
        raise SystemExit(f"row-count mismatch: reload {len(df):,} vs dump "
                         f"{len(d['r_sim']):,} -- cannot align")
    same = np.allclose(df["r_input_p"].to_numpy(float), d["r_input_p"], atol=1e-6)
    if not same:
        raise SystemExit("ALIGNMENT FAILED: reloaded r_input_p does not match the dump. "
                         "Refusing to join -- rerun the scoring job to get Re_input_p "
                         "written directly.")
    print("  alignment verified against r_input_p (exact)", flush=True)
    return df["Re_input_p"].to_numpy(float)


def line(label, mask, s_anti, i_anti, cases, g, rb, n_boot):
    n = int(mask.sum())
    if n < 500:
        return
    den = np.sum(i_anti[mask])
    ghat = float(np.sum(s_anti[mask]) / den)
    err = boot_by_case(s_anti[mask], i_anti[mask], cases[mask], n_boot=n_boot)
    print(f"  {label:>34} {n:>9,} {rb[mask].mean():>9.4f} "
          f"{ghat / g:>9.4f} {err / g:>7.4f} {ghat / g - 1:>+8.2%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--mag-cut", type=float, default=26.0)
    ap.add_argument("--size-cut", type=float, default=0.3)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--catalogue", default=None)
    # taken from eval_score_response so the reload cannot drift from the scoring run
    from scripts.eval_score_response import (BLEND_LOOKUP, CROWD_FLUX_LOOKUP,
                                             MEAS_PRIM_LOOKUP)
    ap.add_argument("--crowd-flux-lookup", default=CROWD_FLUX_LOOKUP)
    ap.add_argument("--meas-prim-lookup", default=MEAS_PRIM_LOOKUP)
    ap.add_argument("--blend-lookup", default=BLEND_LOOKUP)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    d = np.load(args.npz)
    g = float(d["g"])
    s_anti = 0.5 * (d["s_plus"] - d["s_minus"])
    i_anti = 0.5 * (d["i_plus"] + d["i_minus"])
    cases, rb = d["case"], d["r_blend"]
    mag_t = d["r_input_p"]
    re_t = recover_true_size(d, args)

    keep = (mag_t < args.mag_cut) & (re_t > args.size_cut)
    print(f"\n=== {os.path.basename(args.npz)}: N={len(mag_t):,} ===")
    print(f"acceptance cut (GOALS.md): true mag < {args.mag_cut}  AND  "
          f"true Re > {args.size_cut}  ->  {keep.mean():.1%} of rows\n")
    print(f"  {'sample':>34} {'N':>9} {'<R_blend>':>9} {'ghat/g':>9} {'+/-':>7} {'m_5B':>9}")
    line("FULL catalogue (as reported)", np.ones_like(keep), s_anti, i_anti, cases, g, rb,
         args.n_boot)
    line("ACCEPTANCE  mag<26 & Re>0.3", keep, s_anti, i_anti, cases, g, rb, args.n_boot)
    line("  rejected (outside acceptance)", ~keep, s_anti, i_anti, cases, g, rb, args.n_boot)

    print(f"\n  --- the two cuts separately ---")
    line(f"true mag < {args.mag_cut}", mag_t < args.mag_cut, s_anti, i_anti, cases, g, rb,
         args.n_boot)
    line(f"true Re  > {args.size_cut}", re_t > args.size_cut, s_anti, i_anti, cases, g, rb,
         args.n_boot)

    print(f"\n  --- inside the acceptance cut, split by R_blend ---")
    sub = np.flatnonzero(keep)
    qb = np.quantile(rb[sub][rb[sub] > 0.02], [0, 0.25, 0.5, 0.75, 1.0])
    m0 = keep & (rb <= 0.02)
    line("R_blend<0.02", m0, s_anti, i_anti, cases, g, rb, args.n_boot)
    for k in range(4):
        m = keep & (rb > max(qb[k], 0.02)) & (rb <= qb[k + 1])
        line(f"R_b {qb[k]:.3f}-{qb[k+1]:.3f}", m, s_anti, i_anti, cases, g, rb, args.n_boot)

    print(f"\n  --- inside the acceptance cut, split by TRUE magnitude ---")
    for lo, hi in ((0, 23), (23, 24), (24, 25), (25, 26)):
        line(f"true mag {lo}-{hi}", keep & (mag_t > lo) & (mag_t <= hi), s_anti, i_anti,
             cases, g, rb, args.n_boot)
    print(f"\n  --- inside the acceptance cut, split by TRUE size ---")
    for lo, hi in ((0.3, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1.5)):
        line(f"true Re {lo}-{hi}", keep & (re_t > lo) & (re_t <= hi), s_anti, i_anti,
             cases, g, rb, args.n_boot)


if __name__ == "__main__":
    main()
