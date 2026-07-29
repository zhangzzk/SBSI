"""The missing cell: the CERTIFIED Gold-v1 flow with the CORRECTED R_blend emulator.

WHY THIS EXISTS
---------------
Three of the four flow x emulator combinations are measured (WORKLOG 2026-07-28n / 2026-07-29):

    Gold-v1 flow + old emulator (_ho)      = +0.245%   <- the certified result
    Gold-V2 flow + old emulator (_ho)      = -0.862% +- 0.230%  (16 seeds)
    Gold-V2 flow + corrected (_indist_wc5) = -1.447% +- 0.269%  (in flight)
    Gold-v1 flow + corrected               = NEVER RUN

The empty cell is the one that decides whether the distance fix should be ADOPTED, because Gold-v1
is the certified pipeline. Everything measured so far says the fix makes m worse, but only on the V2
flow, which carries a ~0.9% offset of its own -- that is not evidence about the certified pipeline.

NO RERUN IS NEEDED. `validate_constant_with_blend.py --dump` writes per-object
(case, input_index, r_sim, R_flow, R_blend) AFTER its selection, and at line 265
`rb_add = rb_i`, i.e. the dumped R_blend column IS the lookup value with no transformation
(the `corr` term at 266-269 is off unless an explicit calibration npz is passed, which the
constgold jobs do not pass). So swapping emulators is a join, not a GPU run:

    m = <r_sim> / ( <R_flow> + <R_blend from the other lookup> ) - 1

r_sim and R_flow are properties of the sim and the flow, untouched by which emulator supplies the
blend term. The 16 Gold-v1 dumps (fig2_perobj_s5*_fixresp.feather) are already on disk.

CONTROL, and the reason to trust the number: recomputing with the dump's OWN R_blend must reproduce
the certified Gold-v1 ensemble m. If it does not, the identity above is being applied wrongly and
the swapped number means nothing. That check is printed first and is not optional.

Errors: per-case bootstrap (resampling the 100 constant cases, re-averaging ALL THREE quantities --
see scripts/eval_error_budget.py for why R_flow must move with the draw) plus the across-seed sem,
added in quadrature. The seed term shrinks with n; the case term is common-mode and does not.

FIREWALL: reads constgold per-object dumps and an emulator lookup. Trains nothing.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.feather as pf

DUMPS = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps"
PATTERN = "fig2_perobj_s*_fixresp.feather"
WC5 = ("/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results/"
       "blend_lookup_wc5_c40-139.feather")
CONSTCAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
            "constant_response_catalogue_train.feather")


def true_prop_cut(mag_max, re_min, cat_path):
    """Keys surviving the DELIVERABLE true-property cut (mag < mag_max, Re > re_min).

    The dumps carry r_input_p but NOT Re_input_p, so the size half of the cut has to come from the
    constgold catalogue. Returns a frame of (case, input_index) to inner-join against.

    The certified m convention is the WIDE population (mag 18-28, Re 0.1-1.5). The acceptance target
    |m| < 0.3% is defined on TRUE-property cuts instead, so m on the wide population is a diagnostic
    and m here is the deliverable. They are different numbers and neither substitutes for the other.
    """
    t = pf.read_table(cat_path, columns=["case", "input_index", "r_input_p", "Re_input_p"],
                      memory_map=True).to_pandas()
    n0 = len(t)
    dup = t.duplicated(["case", "input_index"]).sum()
    keep = t[(t["r_input_p"].to_numpy(float) < mag_max)
             & (t["Re_input_p"].to_numpy(float) > re_min)][["case", "input_index"]]
    print(f"true-property cut: mag < {mag_max}, Re > {re_min}\n"
          f"  catalogue {n0:,} rows ({dup:,} duplicate keys) -> {len(keep):,} pass "
          f"({100*len(keep)/n0:.1f}%)")
    if dup:
        print("  *** duplicate (case,input_index) keys in the catalogue -- the join would fan out; "
              "deduplicating ***")
        keep = keep.drop_duplicates(["case", "input_index"])
    return keep


def case_sums(case, *cols):
    """Per-case sums of each column plus the per-case count, on a shared case index."""
    uc, inv = np.unique(case, return_inverse=True)
    out = [np.bincount(inv, weights=c) for c in cols]
    return uc, out, np.bincount(inv).astype(float)


def boot(S_sim, S_flow, S_blend, N, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    nc = len(N)
    pick = rng.integers(0, nc, size=(n_boot, nc))
    n = N[pick].sum(1)
    return (S_sim[pick].sum(1) / n) / (S_flow[pick].sum(1) / n + S_blend[pick].sum(1) / n) - 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default=DUMPS)
    ap.add_argument("--pattern", default=PATTERN)
    ap.add_argument("--lookup", default=WC5, help="the emulator lookup to swap IN")
    ap.add_argument("--old-lookup", default=None,
                    help="join the BASELINE R_blend from this lookup too, instead of trusting the "
                         "dump's own R_blend column. Required for the V2 dumps: two constgold "
                         "arrays wrote the same dump filename for seeds 510-517 (the SUFFIX fix "
                         "came after they were queued), so their R_blend column is a MIX of the "
                         "two lookups. r_sim and R_flow are emulator-independent, so joining both "
                         "lookups makes the result immune to which array wrote the file.")
    ap.add_argument("--certified-m", type=float, default=None,
                    help="expected control value, e.g. 0.245 for the certified Gold-v1 wide m. "
                         "Omit for dumps with no certified reference (the V2 flow).")
    ap.add_argument("--new-label", default="wc5")
    ap.add_argument("--old-label", default="ho")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--apply-cuts", action="store_true",
                    help="restrict to the DELIVERABLE true-property population instead of the wide "
                         "certified convention")
    ap.add_argument("--cut-catalogue", default=CONSTCAT)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dump_dir, args.pattern)))
    if not paths:
        print(f"*** no dumps matching {args.pattern} in {args.dump_dir} ***")
        return
    print(f"{len(paths)} Gold-v1 dumps; swapping in {os.path.basename(args.lookup)}\n")

    look = pf.read_table(args.lookup, columns=["case", "input_index", "R_blend"]).to_pandas()
    look = look.rename(columns={"R_blend": "R_blend_new"})
    print(f"lookup(new): {len(look):,} rows, {look['case'].nunique()} cases")
    look_old = None
    if args.old_lookup:
        look_old = pf.read_table(args.old_lookup,
                                 columns=["case", "input_index", "R_blend"]).to_pandas()
        look_old = look_old.rename(columns={"R_blend": "R_blend_base"})
        print(f"lookup(old): {len(look_old):,} rows, {look_old['case'].nunique()} cases  "
              f"(joined explicitly -- the dump's own R_blend column is IGNORED)")
    print()

    keep_keys = None
    if args.apply_cuts:
        keep_keys = true_prop_cut(args.true_mag_max, args.true_re_min, args.cut_catalogue)
        print()

    rows = []
    agg = {}
    for p in paths:
        # matches both fig2_perobj_s501_fixresp.feather and indist_perobj_s501.feather
        mo = re.search(r"_s(\d+)(?:_|\.)", os.path.basename(p))
        sd = int(mo.group(1)) if mo else -1
        t = pf.read_table(p, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"],
                          memory_map=True).to_pandas()
        n0 = len(t)
        if keep_keys is not None:
            t = t.merge(keep_keys, on=["case", "input_index"], how="inner")
        t = t.merge(look, on=["case", "input_index"], how="left")
        matched = t["R_blend_new"].notna().mean()
        t["R_blend_new"] = t["R_blend_new"].fillna(0.0)
        if look_old is not None:
            t = t.merge(look_old, on=["case", "input_index"], how="left")
            t["R_blend_base"] = t["R_blend_base"].fillna(0.0)
        else:
            t["R_blend_base"] = t["R_blend"]
        case = t["case"].to_numpy(np.int64)
        rs = t["r_sim"].to_numpy(float)
        rf = t["R_flow"].to_numpy(float)
        rb_old = np.nan_to_num(t["R_blend_base"].to_numpy(float), nan=0.0)
        rb_new = t["R_blend_new"].to_numpy(float)
        del t

        uc, (S_sim, S_flow, S_old, S_new), N = case_sums(case, rs, rf, rb_old, rb_new)
        Nt = N.sum()
        Rs, Rf = S_sim.sum() / Nt, S_flow.sum() / Nt
        Ro, Rn = S_old.sum() / Nt, S_new.sum() / Nt
        m_old = Rs / (Rf + Ro) - 1.0
        m_new = Rs / (Rf + Rn) - 1.0
        rows.append((sd, Rf, Ro, Rn, m_old, m_new))
        agg[sd] = (uc, S_sim, S_flow, S_old, S_new, N)
        print(f"  seed {sd}: {n0:,} rows"
              + (f" -> {int(Nt):,} after cuts ({100*Nt/n0:.1f}%)" if keep_keys is not None else "")
              + f", lookup matched {matched:.1%}  "
              f"R_flow={Rf:.4f}  <R_b>_{args.old_label}={Ro:.4f} -> _{args.new_label}={Rn:.4f}  "
              f"m {100*m_old:+.3f}% -> {100*m_new:+.3f}%", flush=True)

    seeds = [r[0] for r in rows]
    m_old = np.array([r[4] for r in rows])
    m_new = np.array([r[5] for r in rows])
    n = len(seeds)

    print(f"\n{'='*78}\nCONTROL: does the dump's own R_blend reproduce the certified Gold-v1 m?"
          f"\n{'='*78}")
    print(f"  {n}-seed ensemble with the ORIGINAL lookup = {100*m_old.mean():+.3f}% "
          f"(sd {100*m_old.std(ddof=1):.3f}, sem {100*m_old.std(ddof=1)/np.sqrt(n):.3f})")
    if args.certified_m is None:
        print("  no --certified-m given: nothing to control against for this flow. The internal")
        print("  cross-check is the paired shift below, which must match the other flow's.")
    elif keep_keys is None:
        print(f"  certified record                           = {args.certified_m:+.3f}%")
        print("  If these disagree by more than the sem, STOP -- the swap below is not meaningful.")
    else:
        print(f"  certified record = {args.certified_m:+.3f}%, but that is the WIDE convention "
              "(mag 18-28, Re 0.1-1.5).")
        print("  The number above is the post-cut population, so it is NOT expected to match. Run")
        print("  this script without --apply-cuts for the control that does validate the method.")

    # ensemble case-error, R_flow re-averaged with the draw
    ref = agg[seeds[0]][0]
    for s in seeds:
        assert np.array_equal(agg[s][0], ref), f"seed {s} has a different case set"
    N = agg[seeds[0]][5]
    S_sim = agg[seeds[0]][1]
    S_old = agg[seeds[0]][3]
    S_new = agg[seeds[0]][4]
    S_flow_ens = np.mean([agg[s][2] for s in seeds], axis=0)
    sd_case_old = boot(S_sim, S_flow_ens, S_old, N, args.n_boot).std()
    sd_case_new = boot(S_sim, S_flow_ens, S_new, N, args.n_boot).std()

    def tot(mv, sc):
        return float(np.hypot(mv.std(ddof=1) / np.sqrt(n), sc))

    print(f"\n{'='*78}\nTHE MISSING CELL: Gold-v1 (certified) flow, both emulators, {n} seeds"
          f"\n{'='*78}")
    print(f"{'emulator':>14} {'<R_blend>':>10} {'m':>10} {'sem(seed)':>10} {'case':>7} "
          f"{'TOTAL':>8}")
    for lbl, mv, rb, sc in ((args.old_label, m_old, agg[seeds[0]][3].sum() / N.sum(), sd_case_old),
                            (args.new_label, m_new, agg[seeds[0]][4].sum() / N.sum(), sd_case_new)):
        print(f"{lbl:>14} {rb:>10.4f} {100*mv.mean():>+9.3f}% "
              f"{100*mv.std(ddof=1)/np.sqrt(n):>9.3f}% {100*sc:>6.3f}% {100*tot(mv, sc):>7.3f}%")

    d = m_new - m_old
    print(f"\n  paired shift on the SAME {n} seeds = {100*d.mean():+.3f} points "
          f"+- {100*d.std(ddof=1)/np.sqrt(n):.3f}  (sd {100*d.std(ddof=1):.3f})")
    print("  compare the V2-flow paired shift, -0.784 +- 0.015: the shift is a property of the two")
    print("  lookups and should be nearly flow-independent, so a large disagreement is a red flag.")

    print(f"\n{'='*78}\nWHAT THIS DECIDES\n{'='*78}")
    if abs(m_new.mean()) < abs(m_old.mean()):
        print("  The corrected emulator IMPROVES the certified pipeline -> adopt the distance fix.")
    else:
        print("  The corrected emulator makes the CERTIFIED pipeline worse, even though it is")
        print("  verifiably more accurate per pair against the ruler. That is the compensating-error")
        print("  result on the pipeline that matters: the certified +0.245% was partly relying on")
        print("  the old emulator under-counting the blend response. Adopting the fix requires")
        print("  fixing R_flow too, not just swapping the lookup.")
    print("SWAP_LOOKUP_DONE", flush=True)


if __name__ == "__main__":
    main()
