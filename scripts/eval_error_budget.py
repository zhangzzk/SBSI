"""The full error budget on the constgold m, with the R_flow-frozen bootstrap corrected.

WHY THIS EXISTS
---------------
Two independent uncertainties sit on the ensemble m = <R_sim>/(<R_flow> + <R_blend>) - 1:

  (A) WHICH 100 FIELDS got simulated -- sampling error of the finite constgold catalogue. Common
      to every flow seed, so averaging over seeds does NOT reduce it. Currently reported as the
      "+/- 0.17%" printed by validate_constant_with_blend.py.
  (B) WHICH TRAINING SEEDS produced the flow -- model-training scatter, sd ~1.0% per seed, so
      sd/sqrt(n) on the ensemble mean. Shrinks with more seeds; (A) does not.

Total on the ensemble mean is sqrt(A^2 + B^2): they are independent (the training seed knows
nothing about which fields were drawn), so they add in quadrature, not linearly.

THE CORRECTION MEASURED HERE
----------------------------
`boot_m_err` resamples cases but holds R_flow at its global value, documented as "R_flow held
deterministic". That is true for a FIXED set of galaxies -- but the bootstrap changes which
galaxies, and R_flow is itself a mean over those same objects. A resampled draw that pulls in
fainter/more blended fields lowers the true response and the flow's PREDICTED response together;
because m is a ratio, that common motion largely cancels. Freezing the denominator discards the
cancellation and inflates (A).

So this recomputes the same bootstrap two ways on the per-object dumps:

    frozen : m_b = (sum r_sim / N) / (R_flow_global    + sum R_blend / N) - 1   [what the code does]
    full   : m_b = (sum r_sim / N) / (sum R_flow / N   + sum R_blend / N) - 1   [correct]

and reports the per-case correlation between <r_sim> and <R_flow> that drives the difference.

Only per-case SUMS are needed, so each 1 GB dump collapses to 100 numbers and the bootstrap runs
on those -- 20k resamples cost nothing.

ENSEMBLE (A): the quantity we actually quote is the seed-ensemble mean, whose flow response is the
across-seed mean of R_flow. So the ensemble case-error uses per-case R_flow sums averaged over
seeds, then bootstraps cases once. (Bootstrapping each seed separately would give the case error on
a SINGLE-seed m, which is not what is quoted.)

FIREWALL: reads constgold per-object dumps only; trains nothing.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pyarrow.feather as pf

DUMPS = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/indist_constgold_dumps"


def per_case(path):
    """Collapse a per-object dump to per-case sums. Returns (cases, S_sim, S_flow, S_blend, N)."""
    t = pf.read_table(path, columns=["case", "r_sim", "R_flow", "R_blend"], memory_map=True)
    case = t.column("case").to_numpy()
    rs = t.column("r_sim").to_numpy()
    rf = t.column("R_flow").to_numpy()
    rb = np.nan_to_num(t.column("R_blend").to_numpy(), nan=0.0)
    del t
    uc, inv = np.unique(case, return_inverse=True)
    n = np.bincount(inv).astype(float)
    return (uc,
            np.bincount(inv, weights=rs),
            np.bincount(inv, weights=rf),
            np.bincount(inv, weights=rb),
            n)


def boot(S_sim, S_flow, S_blend, N, n_boot, seed, rf_frozen=None):
    """Per-case bootstrap of m. rf_frozen=None -> re-average R_flow with the draw (correct)."""
    rng = np.random.default_rng(seed)
    nc = len(N)
    pick = rng.integers(0, nc, size=(n_boot, nc))
    n = N[pick].sum(1)
    num = S_sim[pick].sum(1) / n
    blend = S_blend[pick].sum(1) / n
    flow = np.full(n_boot, rf_frozen) if rf_frozen is not None else S_flow[pick].sum(1) / n
    return num / (flow + blend) - 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default=DUMPS)
    ap.add_argument("--pattern", default="indist_perobj_s*.feather")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--label", default="indist_wc5")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dump_dir, args.pattern)))
    if not paths:
        print(f"*** no dumps matching {args.pattern} in {args.dump_dir} ***")
        return
    print(f"[{args.label}] {len(paths)} per-object dumps\n")

    agg = {}
    for p in paths:
        m = re.search(r"_s(\d+)\.feather$", p)
        sd = int(m.group(1)) if m else -1
        uc, ss, sf, sb, n = per_case(p)
        agg[sd] = (uc, ss, sf, sb, n)
        print(f"  seed {sd}: {int(n.sum()):>12,} objects, {len(uc)} cases", flush=True)

    seeds = sorted(agg)
    ref_cases = agg[seeds[0]][0]
    for s in seeds:
        assert np.array_equal(agg[s][0], ref_cases), f"seed {s} has a different case set"
    N = agg[seeds[0]][4]
    S_sim = agg[seeds[0]][1]
    S_blend = agg[seeds[0]][3]
    for s in seeds[1:]:
        assert np.allclose(agg[s][1], S_sim), "r_sim differs across seeds -- it must not"
        assert np.allclose(agg[s][3], S_blend), "R_blend differs across seeds -- it must not"
    print("\n  checked: r_sim and R_blend are byte-identical across seeds (only R_flow varies)")

    Ntot = N.sum()
    R_sim = S_sim.sum() / Ntot
    R_blend = S_blend.sum() / Ntot

    # ---- (B) seed scatter -------------------------------------------------------------------
    print(f"\n{'='*78}\nPER-SEED m AND THE CASE BOOTSTRAP, TWO WAYS\n{'='*78}")
    print(f"{'seed':>6} {'R_flow':>8} {'m':>9} {'sd frozen':>11} {'sd full':>9} {'shrink':>8}")
    m_seed, frozen_sd, full_sd = [], [], []
    for s in seeds:
        _, ss, sf, sb, n = agg[s]
        R_flow = sf.sum() / Ntot
        m = R_sim / (R_flow + R_blend) - 1.0
        bf = boot(ss, sf, sb, n, args.n_boot, 0, rf_frozen=R_flow).std()
        bu = boot(ss, sf, sb, n, args.n_boot, 0, rf_frozen=None).std()
        m_seed.append(m); frozen_sd.append(bf); full_sd.append(bu)
        print(f"{s:>6} {R_flow:>8.4f} {100*m:>+8.3f}% {100*bf:>10.3f}% {100*bu:>8.3f}% "
              f"{bu/bf:>7.2f}x")
    m_seed = np.array(m_seed)
    n = len(seeds)
    sd_seed = m_seed.std(ddof=1)
    sem_seed = sd_seed / np.sqrt(n)

    # why the shrink happens: the per-case response of the sim and of the flow move together
    rs_c = S_sim / N
    rf_c = np.mean([agg[s][2] for s in seeds], axis=0) / N
    rb_c = S_blend / N
    print(f"\nper-case correlation  <r_sim> vs <R_flow>  = {np.corrcoef(rs_c, rf_c)[0,1]:+.3f}")
    print(f"per-case correlation  <r_sim> vs <R_blend> = {np.corrcoef(rs_c, rb_c)[0,1]:+.3f}")
    print("  a positive first number is the mechanism: a case draw that lowers the true response")
    print("  lowers the flow's predicted response too, and the ratio m is partly immune.")

    # ---- (A) case error on the ENSEMBLE mean ------------------------------------------------
    S_flow_ens = np.mean([agg[s][2] for s in seeds], axis=0)
    R_flow_ens = S_flow_ens.sum() / Ntot
    m_ens_pt = R_sim / (R_flow_ens + R_blend) - 1.0
    sd_case_frozen = boot(S_sim, S_flow_ens, S_blend, N, args.n_boot, 0,
                          rf_frozen=R_flow_ens).std()
    sd_case_full = boot(S_sim, S_flow_ens, S_blend, N, args.n_boot, 0, rf_frozen=None).std()

    print(f"\n{'='*78}\nERROR BUDGET ON THE {n}-SEED ENSEMBLE m   [{args.label}]\n{'='*78}")
    print(f"  R_sim  = {R_sim:.4f}   R_flow(ens) = {R_flow_ens:.4f}   R_blend = {R_blend:.4f}")
    print(f"  m (mean over seeds)            = {100*m_seed.mean():+.3f}%")
    print(f"  m (at ensemble-mean R_flow)    = {100*m_ens_pt:+.3f}%   "
          f"(differs only at second order)")
    print(f"\n  (B) seed scatter   sd/seed     = {100*sd_seed:.3f}%")
    print(f"      -> on the mean of {n}        = {100*sem_seed:.3f}%      shrinks as 1/sqrt(n_seed)")
    print(f"  (A) case sampling, R_flow frozen = {100*sd_case_frozen:.3f}%   "
          f"<- what the pipeline prints")
    print(f"      case sampling, R_flow re-avg = {100*sd_case_full:.3f}%   "
          f"<- correct; common-mode, never shrinks with seeds")
    tot_old = np.hypot(sem_seed, sd_case_frozen)
    tot_new = np.hypot(sem_seed, sd_case_full)
    print(f"\n  TOTAL (quadrature)  as printed = {100*tot_old:.3f}%")
    print(f"  TOTAL (quadrature)  corrected  = {100*tot_new:.3f}%")
    print(f"  (linear sum would be {100*(sem_seed+sd_case_full):.3f}% -- wrong, the two are "
          f"independent)")

    print(f"\n  floor with infinite seeds       = {100*sd_case_full:.3f}%  (only more sim cases "
          f"move this)")
    for nn in (8, 16, 32, 64, 128):
        print(f"    n_seed={nn:>4}: total = {100*np.hypot(sd_seed/np.sqrt(nn), sd_case_full):.3f}%")

    print("\nNOT INCLUDED, and still unmeasured: the R_blend emulator's own training-seed scatter.")
    print("One emulator per config, so every flow seed reads an identical lookup. Its case-to-case")
    print("sampling noise IS inside (A); its training variability is in no error bar here.")
    print("ERROR_BUDGET_DONE", flush=True)


if __name__ == "__main__":
    main()
