"""Re-score `m` with a DIFFERENT blend emulator, without re-running the flow.

WHY THIS EXISTS. A per-object dump stores `r_sim`, `R_flow` and `R_blend` as separate columns, and
only `R_blend` depends on which emulator was used. So swapping emulators is a JOIN on
`(case, input_index)`, not a 16-seed GPU re-score. The expensive path was previously assumed and it
is what made the question look unaffordable.

WHY IT MATTERS FOR V2.1 (2026-08-05s). The fiducial emulator `lsst_r_extnbr_indom_tuned` has a
regression box of primary mag 18-26. The V2.1 domain is `Re > 0.5 AND sn_true > 10`, an S/N CURVE
with NO magnitude ceiling, so it admits primaries past mag 26 that the fiducial emulator drops. On
the ruler, over the FULL V2.1 sample (766,882 primaries), the summed `R_blend` reads:

    truth  0.08636 +- 0.00284
    lsst_r_extnbr_ho            +1.18% +- 3.33     <-- consistent with truth (0.35 sigma)
    lsst_r_extnbr_indom_tuned  -15.56% +- 2.78     <-- 5.6 sigma LOW
    lsst_r_extnbr_v21          -37.05% +- 2.07

`_ho`'s box (mag 18-28, Re 0.1-1.5) covers V2.1 outright. So on V2.1 the fiducial emulator is
convicted on the ruler and `_ho` is not -- which is the promotion argument AGENTS.md requires
(`eval_rblend_gap`, never constgold `m`). This script applies that swap to the reported number.

THE TRAP THIS SCRIPT EXISTS TO AVOID, restated because it caught me once tonight. Inside the
emulator's own inference box all four emulators agree to 0.06%, which is why an earlier note called
this swap futile. That comparison was on the BOX population (138,017 primaries, 18% of V2.1); the
swap matters on the population it is actually scored on. AGENTS.md records the same
population-mismatch trap four separate times. Check which population a number came from before
concluding two numbers describe the same thing.

NEVER ZERO-FILL. AGENTS.md trap #1: an emulator returns nothing outside its box, and averaging those
rows in as `R_blend = 0` silently biases the result. This asserts the match fraction and DROPS
unmatched rows, and refuses outright below `--min-match`. Dropping changes the population, so the
dropped fraction is printed and the same mask is applied to every seed.

SEEDS. This reports `m`, so AGENTS.md requires 16 seeds and the ratio formed INSIDE each seed. Both
are enforced, not assumed.

FIREWALL. constgold is EVALUATION ONLY. The emulator choice here is argued on the ruler, above; no
constgold number selects, tunes, or corrects anything.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import pyarrow.feather as pf

from sbs_shear import domain as sbs_domain
from scripts.eval_v2_indomain_m import catalogue_true_props


def load_lookup(path, t0):
    t = pf.read_table(path, columns=["case", "input_index", "R_blend"])
    print(f"  lookup {path.split('/')[-1]}: {t.num_rows:,} rows ({time.time()-t0:.0f}s)", flush=True)
    return {"case": t["case"].to_numpy(zero_copy_only=False).astype(np.int64),
            "input_index": t["input_index"].to_numpy(zero_copy_only=False).astype(np.int64),
            "R_blend": t["R_blend"].to_numpy(zero_copy_only=False).astype(float)}


def key64(case, idx):
    """One integer per (case, input_index). input_index is far below 2^40 in this catalogue."""
    if idx.max() >= (1 << 40):
        raise RuntimeError("input_index too large for the packed key; widen it")
    return (case.astype(np.int64) << 40) | idx.astype(np.int64)


def report(name, masks, per_seed, ndrop=None):
    print(f"\n{'='*100}\n{name}\n{'='*100}")
    print(f"  {'population':<34}{'N':>12}{'R_sim':>9}{'R_flow':>9}{'R_blend':>9}"
          f"{'m %':>10}{'+-':>8}{'sd':>8}")
    for k in masks:
        v = np.asarray(per_seed[k]["m"])
        print(f"  {k:<34}{per_seed[k]['n']:>12,}{per_seed[k]['rs']:>9.4f}"
              f"{per_seed[k]['rf']:>9.4f}{per_seed[k]['rb']:>9.4f}"
              f"{v.mean():>+10.3f}{v.std(ddof=1)/np.sqrt(len(v)):>8.3f}{v.std(ddof=1):>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--lookup", required=True, help="per-object R_blend from the emulator to swap IN")
    ap.add_argument("--lookup-label", default="swapped")
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--min-match", type=float, default=0.98,
                    help="refuse if the lookup covers less of the domain than this")
    args = ap.parse_args()
    t0 = time.time()

    dumps = sorted(glob.glob(args.dump_glob))
    if len(dumps) < 16:
        raise SystemExit(f"REFUSING: {len(dumps)} dumps found. This reports m, which AGENTS.md "
                         f"requires 16 seeds for. Point --dump-glob at the fiducial dumps.")
    print(f"{len(dumps)} seed dumps", flush=True)

    tp = catalogue_true_props(args.catalogue, args.min_case, t0)
    mag = tp["r_input_p"].to_numpy(float)
    re_ = tp["Re_input_p"].to_numpy(float)
    v21 = sbs_domain.in_domain(mag, re_)
    print(f"catalogue rows {len(tp):,};  V2.1 domain {int(v21.sum()):,} ({v21.mean():.2%})")

    lk = load_lookup(args.lookup, t0)
    kk = key64(lk["case"], lk["input_index"])
    order = np.argsort(kk)
    kk, lkb = kk[order], lk["R_blend"][order]

    # The dump rows are the catalogue rows in order, so the join key comes from the catalogue.
    ck = key64(tp["case"].to_numpy(np.int64), tp["input_index"].to_numpy(np.int64))
    pos = np.searchsorted(kk, ck)
    pos_c = np.clip(pos, 0, len(kk) - 1)
    hit = kk[pos_c] == ck
    rb_swap = np.where(hit, lkb[pos_c], np.nan)

    frac = hit[v21].mean()
    print(f"\nMATCH FRACTION on the V2.1 domain: {frac:.4%} "
          f"({int(hit[v21].sum()):,} of {int(v21.sum()):,})")
    if frac < args.min_match:
        raise SystemExit(f"REFUSING: the swapped lookup covers only {frac:.2%} of the V2.1 domain. "
                         f"Averaging the rest in as R_blend = 0 is exactly the silent bias AGENTS.md "
                         f"trap #1 describes. Use an emulator whose box covers the domain.")

    keep = v21 & hit
    dropped = int(v21.sum() - keep.sum())
    if dropped:
        print(f"  DROPPING {dropped:,} unmatched rows ({dropped/max(int(v21.sum()),1):.3%}) rather "
              f"than zero-filling them. The population below is the KEPT one.")

    masks = {f"V2.1, FIDUCIAL emulator": keep, f"V2.1, {args.lookup_label}": keep}
    per = {k: {"m": []} for k in masks}
    for d in dumps:
        t = pf.read_table(d, columns=["r_sim", "R_flow", "R_blend"])
        if len(t) != len(tp):
            raise RuntimeError(f"{d}: {len(t):,} rows vs catalogue {len(tp):,}; cannot align")
        rs = t["r_sim"].to_numpy(zero_copy_only=False).astype(float)[keep]
        rf = t["R_flow"].to_numpy(zero_copy_only=False).astype(float)[keep]
        rb_fid = t["R_blend"].to_numpy(zero_copy_only=False).astype(float)[keep]
        rb_new = rb_swap[keep]
        a, b = rs.mean(), rf.mean()
        # ratio formed INSIDE the seed, per AGENTS.md
        per["V2.1, FIDUCIAL emulator"]["m"].append(100 * (a / (b + rb_fid.mean()) - 1))
        per[f"V2.1, {args.lookup_label}"]["m"].append(100 * (a / (b + rb_new.mean()) - 1))
        for k, rbm in ((f"V2.1, FIDUCIAL emulator", rb_fid.mean()),
                       (f"V2.1, {args.lookup_label}", rb_new.mean())):
            per[k].update(n=int(keep.sum()), rs=a, rf=b, rb=rbm)

    report(f"V2.1 m: fiducial emulator vs {args.lookup_label}   "
           f"(same dumps, same rows, same R_flow -- ONLY R_blend differs)", masks, per)

    fid = np.asarray(per["V2.1, FIDUCIAL emulator"]["m"])
    new = np.asarray(per[f"V2.1, {args.lookup_label}"]["m"])
    d = new - fid
    print(f"\n  swap moves m by {d.mean():+.3f} pt (+- {d.std(ddof=1)/np.sqrt(len(d)):.3f}).")
    print(f"  R_blend {per['V2.1, FIDUCIAL emulator']['rb']:.4f} -> "
          f"{per[f'V2.1, {args.lookup_label}']['rb']:.4f} "
          f"({100*(per[f'V2.1, {args.lookup_label}']['rb']/per['V2.1, FIDUCIAL emulator']['rb']-1):+.2f}%)")
    print("  R_flow is IDENTICAL in both rows by construction, so the whole move is the emulator.")
    print("  Which emulator is right on THIS population is settled on the ruler, not here.")
    print("\nSWAP_EMULATOR_DONE", flush=True)


if __name__ == "__main__":
    main()
