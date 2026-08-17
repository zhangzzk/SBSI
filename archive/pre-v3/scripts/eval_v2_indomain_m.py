"""Gold-V2 8-seed constgold m, inside vs outside the FLOW TRAINING DOMAIN.

The certified constgold convention (validate_constant_with_blend.py, and therefore the certified
Gold-v1 m=+0.245%) applies only `source_select_selection(DEFAULT_SELECTION_CUTS)` -- true mag in
(18,28), true Re in (0.1,1.5). The Stage-2 selection harness instead evaluates inside the flow's
TRAINING DOMAIN, `domain_cut(re_min=0.3, mag_max=26.0)`, because the spin-2 orientation-coupling
target the flow's shear response is pinned to is only defined for true mag < 26.04 and true
Re > 0.30; outside it the flow extrapolates an unconstrained constant.

This reports m = <R_sim> / (<R_flow> + <R_blend>) - 1 under both conventions from the existing
per-object dumps, so the two numbers can be quoted side by side without re-running the GPU pass.

The dumps carry r_input_p but not Re_input_p, so true size is recovered from the catalogue by
replaying the IDENTICAL read + cut the dump was produced with and stacking positionally, guarded by
an exact (case, input_index) equality check rather than assumed.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.preprocessing import (  # noqa: E402
    source_select_selection, DEFAULT_SELECTION_CUTS,
)

CAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
       "constant_response_catalogue_train.feather")
DUMPS = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_constgold_dumps"


def catalogue_true_props(cat, min_case, t0, max_case=None):
    """Replay the dump's own read + cut so true Re/mag line up row-for-row with the dumps."""
    cols = ["case", "input_index", "r_input_p", "Re_input_p", "distance", "neighbored"]
    parts = []
    with ipc.open_file(pa.memory_map(cat)) as r:
        use = [c for c in cols if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            case = b["case"].to_numpy()
            keep = case >= min_case
            if max_case is not None:
                keep &= case < max_case
            b = b[keep]
            if len(b):
                parts.append(b)
    import pandas as pd
    df = pd.concat(parts, ignore_index=True)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    print(f"  catalogue replay: {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)
    return df


def check_emulator_covers_domain(tag, mag, re_, domain_mask):
    """Refuse to report m if the emulator cannot score the whole evaluated domain.

    THE FAILURE THIS CATCHES IS SILENT AND EXPENSIVE. `validate_constant_with_blend.py` joins the
    R_blend lookup and then `.fillna(0.0)`, so a galaxy the emulator never scored is indistinguish-
    able in the dump from a genuinely isolated one -- both read R_blend = 0. AGENTS.md "Two traps"
    #1 is precisely this: the in-domain emulator covers 43.4% of the wide population, which
    collapsed <R_blend> from 0.1593 to 0.0589 and manufactured a spurious +28.9% m.

    Zero-fill cannot be detected after the fact, so the check is STRUCTURAL and runs before any m
    is printed: every row of the evaluated domain must lie inside the emulator's own stored
    inference box. For V2.1 that should hold by construction -- the emulator's box IS the V2.1
    bounding box -- so a failure here means the two have drifted apart, which is exactly the thing
    that must not pass quietly.
    """
    meta_path = f"/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_{tag}.json"
    if not os.path.exists(meta_path):
        raise SystemExit(f"emulator metadata not found: {meta_path}")
    import json
    cuts = json.load(open(meta_path))["tasks"]["regression"]["cuts"]
    mag_lo, mag_hi = float(cuts[1][0]), float(cuts[1][1])     # PRIMARY magnitude
    re_lo, re_hi = float(cuts[3][0]), float(cuts[3][1])       # PRIMARY size
    print(f"  emulator {tag!r} inference box: mag ({mag_lo}, {mag_hi}), Re ({re_lo}, {re_hi})")
    inside = (mag > mag_lo) & (mag < mag_hi) & (re_ > re_lo) & (re_ < re_hi)
    uncovered = int((domain_mask & ~inside).sum())
    n = int(domain_mask.sum())
    frac = uncovered / max(n, 1)
    print(f"  domain rows outside that box: {uncovered:,} / {n:,} ({frac:.4%})")
    if uncovered:
        raise SystemExit(
            f"REFUSING to report m: {uncovered:,} of {n:,} evaluated rows ({frac:.2%}) lie outside "
            f"the emulator's inference box, so their R_blend was zero-filled rather than predicted. "
            f"Fix the box or narrow the domain -- do not average over zeros.")
    print("  OK: the emulator covers the whole evaluated domain (unmatched rows are isolated only)")


def case_blocked_sim_sem(rs, case, mask):
    """Sem of <R_sim> over `mask`, blocked by CASE rather than treating rows as independent.

    WHY THIS TERM EXISTS AT ALL. `m = <R_sim>/(<R_flow>+<R_blend>) - 1` is model-vs-SIM. `R_sim` is
    byte-identical in every seed dump, so it contributes NOTHING to the seed scatter -- yet it is a
    finite-sample mean of a per-object response whose scatter is std ~5 against a mean ~0.86, so it
    carries a sampling error of its own. Quoting the seed sem alone therefore UNDERSTATES the error
    on an absolute `m`, which is exactly the comparison made against the 0.3% objective.

    It is correctly ABSENT from a model-vs-model difference (V2 minus V2.2 on the same rows): the
    same `R_sim` sits on both sides and cancels exactly. This is the same seed/sim split AGENTS.md
    draws for `dm` versus the absolute `m` at a cut, and the reason both are reported separately
    here rather than pre-combined.

    WHY BLOCKED. A naive std/sqrt(N) assumes every galaxy's response is an independent draw. They
    are not: objects sharing a scene share pixels and a noise realisation, so their responses are
    correlated and the naive sem is too SMALL. Blocking on `case` -- each case being a separately
    rendered field -- absorbs that correlation, and matches the convention already used by
    `diag_rflow_v21.py`. Cases are near-equal in size here, so the unweighted mean of per-case means
    is used, as there.
    """
    c = case[mask]
    x = rs[mask]
    uc, inv = np.unique(c, return_inverse=True)
    if len(uc) < 3:
        return float("nan")
    mu = np.bincount(inv, weights=x, minlength=len(uc)) / np.bincount(inv, minlength=len(uc))
    return float(mu.std(ddof=1) / np.sqrt(len(uc)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-glob", default=os.path.join(DUMPS, "v2_perobj_s*.feather"))
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--re-min", type=float, default=0.3)
    ap.add_argument("--mag-max", type=float, default=26.0)
    ap.add_argument("--v21-domain", action="store_true",
                    help="add the V2.1 domain (primary true Re > 0.5\" AND true S/N > 10) as a "
                         "mask, and check that the emulator behind the dump actually covers it.")
    ap.add_argument("--emulator-tag", default="lsst_r_extnbr_v21",
                    help="emulator whose stored inference box is checked against the evaluated mask.")
    ap.add_argument("--check-emulator-coverage", action="store_true",
                    help="verify that the emulator's stored inference box covers the rectangular "
                         "--re-min/--mag-max FLOW TRAINING DOMAIN before reporting m.")
    args = ap.parse_args()
    t0 = time.time()

    dumps = sorted(glob.glob(args.dump_glob))
    if not dumps:
        raise SystemExit(f"no dumps matching {args.dump_glob}")
    print(f"{len(dumps)} seed dumps", flush=True)

    tp = catalogue_true_props(args.catalogue, args.min_case, t0)
    ref = pf.read_table(dumps[0], columns=["case", "input_index"]).to_pandas()
    if len(tp) != len(ref) or not (tp["case"].to_numpy() == ref["case"].to_numpy()).all() \
            or not (tp["input_index"].to_numpy() == ref["input_index"].to_numpy()).all():
        raise RuntimeError("catalogue replay does not align row-for-row with the dump; "
                           "cannot stack positionally")
    print("  row alignment with dump verified exactly", flush=True)

    mag = tp["r_input_p"].to_numpy(float)
    re_ = tp["Re_input_p"].to_numpy(float)
    masks = {
        "certified convention (no domain cut)": np.ones(len(tp), bool),
        f"true mag < {args.mag_max}": mag < args.mag_max,
        f"true Re > {args.re_min}": re_ > args.re_min,
        "FLOW TRAINING DOMAIN (both)": (mag < args.mag_max) & (re_ > args.re_min),
    }
    if args.check_emulator_coverage:
        check_emulator_covers_domain(
            args.emulator_tag, mag, re_, masks["FLOW TRAINING DOMAIN (both)"])
    if args.v21_domain:
        from sbs_shear import domain as sbs_domain
        v21 = sbs_domain.in_domain(mag, re_)
        masks["V2.1 DOMAIN (Re>0.5 & S/N>10)"] = v21
        print(f"\n  {sbs_domain.describe()}")
        check_emulator_covers_domain(args.emulator_tag, mag, re_, v21)
    for k, v in masks.items():
        print(f"  {k:38s} N={int(v.sum()):>12,}  ({v.mean():6.1%})")

    res = {k: [] for k in masks}
    comps = {k: [] for k in masks}     # (R_sim, R_flow, R_blend) per seed, to separate WHICH term moved
    rs_ref = None                      # R_sim must be seed-INDEPENDENT; the error split assumes it
    print(f"\n  {'seed':>6} " + " ".join(f"{k.split('(')[0].strip()[:22]:>24}" for k in masks),
          flush=True)
    for d in dumps:
        t = pf.read_table(d, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"])
        c = t["case"].to_numpy()
        i = t["input_index"].to_numpy()
        if len(c) != len(ref) or not (c == ref["case"].to_numpy()).all() \
                or not (i == ref["input_index"].to_numpy()).all():
            raise RuntimeError(f"{os.path.basename(d)} row order differs from {os.path.basename(dumps[0])}")
        rs = t["r_sim"].to_numpy(zero_copy_only=False).astype(float)
        rf = t["R_flow"].to_numpy(zero_copy_only=False).astype(float)
        rb = t["R_blend"].to_numpy(zero_copy_only=False).astype(float)
        seed = os.path.basename(d).split("_s")[-1].split(".")[0]
        # The seed/sim error split below rests on R_sim being identical across seeds. Check it
        # rather than assume it: if a dump were ever built from a different catalogue read, the
        # sim error would be neither common nor cancelling and the split would be wrong.
        if rs_ref is None:
            rs_ref = rs
        elif not np.array_equal(rs, rs_ref):
            raise RuntimeError(
                f"{os.path.basename(d)} has a different r_sim from {os.path.basename(dumps[0])}; "
                f"R_sim must be seed-independent for the seed/sim error split to hold")
        row = []
        for k, v in masks.items():
            m = rs[v].mean() / (rf[v].mean() + rb[v].mean()) - 1.0
            res[k].append(m * 100)
            comps[k].append((rs[v].mean(), rf[v].mean(), rb[v].mean()))
            row.append(f"{m*100:>+24.3f}")
        print(f"  {seed:>6} " + " ".join(row), flush=True)

    def sd(v):      # a single seed has no seed-scatter estimate; say so rather than print nan
        return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")

    # Two INDEPENDENT error terms; see case_blocked_sim_sem for why they are reported separately.
    case_arr = ref["case"].to_numpy()
    e_seed = {k: sd(res[k]) / np.sqrt(len(res[k])) for k in masks}
    e_sim = {}
    for k, v in masks.items():
        r_tot = np.array(comps[k]).mean(axis=0)[1:].sum()      # <R_flow> + <R_blend>, seed-mean
        e_sim[k] = 100.0 * case_blocked_sim_sem(rs_ref, case_arr, v) / r_tot
    e_tot = {k: float(np.hypot(e_seed[k], e_sim[k])) for k in masks}

    print(f"\n  {'ENSEMBLE':>6} " + " ".join(
        f"{np.mean(res[k]):>+17.3f}+-{e_tot[k]:.3f}" for k in masks))
    print(f"  {'seed sd':>6} " + " ".join(f"{sd(res[k]):>24.3f}" for k in masks))
    print(f"  {'+-seed':>6} " + " ".join(f"{e_seed[k]:>24.3f}" for k in masks))
    print(f"  {'+-sim':>6} " + " ".join(f"{e_sim[k]:>24.3f}" for k in masks))
    if len(dumps) == 1:
        print("\n  NOTE: ONE seed -- no seed-scatter estimate. The 8-seed baseline has per-seed sd "
              "~1.0% (global) / ~0.8% (in-domain), so treat a single-seed m as +-~1% until more "
              "seeds are run.")
    print(f"\n  components (seed-mean):  {'mask':<38} {'R_sim':>9} {'R_flow':>9} {'R_blend':>9} {'R_tot':>9}")
    for k in masks:
        c = np.array(comps[k]).mean(axis=0)
        print(f"  {'':>24} {k:<38} {c[0]:>9.4f} {c[1]:>9.4f} {c[2]:>9.4f} {c[1]+c[2]:>9.4f}")

    for k in masks:
        print(f"\n  {k}:  m = {np.mean(res[k]):+.3f} +- {e_tot[k]:.3f} %  "
              f"(+-{e_seed[k]:.3f} seed, +-{e_sim[k]:.3f} sim; seed sd {sd(res[k]):.3f}, "
              f"N={len(res[k])})")
    print("\n  The quoted error is seed and sim added in quadrature. Use the SEED term alone when "
          "differencing two models on these same rows -- the sim term is common and cancels. Use "
          "the TOTAL when comparing an absolute m against a target.")
    if len(dumps) < 16:
        print(f"  WARNING: {len(dumps)} seeds. AGENTS.md requires 16 for a reported m (an e-response "
              f"quantity); treat this as a pilot. The seed sd itself is poorly determined here: on "
              f"{len(dumps)-1} dof its own 95% range spans roughly a factor of 4.")
    print("\nV2_INDOMAIN_M_DONE", flush=True)


if __name__ == "__main__":
    main()
