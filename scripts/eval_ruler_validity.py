"""Is the RULER itself trustworthy at <1", or does the residual -31% live in the truth?

WHY THIS EXISTS
---------------
Correcting the distance definition took the close-pair deficit from -41.5% to -31.1% and closed
1-2" entirely (WORKLOG 2026-07-28n, job 15329219). Something still costs 31% below 1". Before
attributing that to the emulator, the ruler has to be cleared in that bin specifically -- every
number in this investigation is measured against it, and it has only ever been validated GLOBALLY
(the 45-degree null test, 0.6 sigma over all separations).

Two ways the ruler could be biased at <1" and nowhere else:

1. The null test is a global average. A bias confined to the 17% of pairs below 1" would be diluted
   ~6x in it and could sit well inside 0.6 sigma overall while being large where it matters. So run
   the null test PER SEPARATION BIN. This is the honest version of the validation that has been
   quoted all along.

2. The truth is a two-leg difference of the primary's measured shape, and the two legs are matched by
   cross-match. Below 1" the pair may be deblended differently in the two legs -- one detection in
   one, two in the other -- so the "same" object's shape can refer to different light in each leg.
   That inflates the shape difference, and hence the apparent blend response, at exactly these
   separations. The catalogue carries the cross-match diagnostics (`match_distance_pixel_cm`,
   `match_dmag_cm`); if the deficit shrinks on clean matches, the truth is the problem, not the model.

An honest outcome here can go either way: if the null test passes per bin AND the deficit survives on
clean matches, the ruler is cleared and the remaining 31% is the emulator's or its labels'.

FIREWALL: half-shear legs only; constgold is never read.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_rblend_gap import (  # noqa: E402
    BLEND_MODELS, CAT, COND, PAIR_FEATURES, blend_truth, load_legs)

DIST_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
MATCH_COLS = ["match_distance_pixel_cm", "match_dmag_cm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--tag", default="lsst_r_extnbr_indist")
    ap.add_argument("--close-max", type=float, default=1.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base = load_legs(args.gs_leg, args.g0_leg, args.max_case, args.true_re_min, args.true_mag_max,
                     extra_cols=MATCH_COLS)
    truth = blend_truth(base)
    null = blend_truth(base, rotate45=True)
    dist = base["distance"].to_numpy(float)

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    model = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")
    pred = model.predict_on_pairs(base[PAIR_FEATURES].copy(),
                                  task="response")["response"].to_numpy(float)

    ok = np.isfinite(truth) & np.isfinite(null) & np.isfinite(pred)

    # ---------- 1) the null test, PER SEPARATION BIN ----------
    print(f"\n[NULL TEST per separation bin -- 45-deg rotated projection, must be ~0]")
    print(f"  {'distance':>16} {'null':>10} {'sem':>9} {'sigma':>7} {'truth':>10} "
          f"{'null/truth':>11} {'N':>11}")
    for i in range(len(DIST_EDGES) - 1):
        m = ok & (dist >= DIST_EDGES[i]) & (dist < DIST_EDGES[i + 1])
        n = int(m.sum())
        if n < 200:
            continue
        nm = null[m].mean()
        ns = null[m].std(ddof=1) / np.sqrt(n)
        t = truth[m].mean()
        flag = "   <-- FAILS" if abs(nm) > 3 * ns else ""
        print(f"  [{DIST_EDGES[i]:>6.2f},{DIST_EDGES[i+1]:>6.2f}) {nm:>+10.5f} {ns:>9.5f} "
              f"{abs(nm)/ns:>7.1f} {t:>10.4f} {nm/t if t else np.nan:>11.3f} {n:>11,}{flag}")

    # ---------- 2) does the close-pair deficit depend on cross-match quality? ----------
    close = ok & (dist < args.close_max)
    print(f"\n[CLOSE PAIRS (<{args.close_max}\") by CROSS-MATCH QUALITY]  N={int(close.sum()):,}")
    for col in MATCH_COLS:
        if col not in base.columns:
            print(f"  {col}: not in catalogue")
            continue
        v = np.abs(base[col].to_numpy(float))
        good = close & np.isfinite(v)
        if good.sum() < 1000:
            print(f"  {col}: too few finite values ({int(good.sum()):,})")
            continue
        qs = np.quantile(v[good], [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"\n  {col}:  quartiles {np.round(qs, 4).tolist()}")
        print(f"  {'|value| range':>22} {'truth':>9} {'emulator':>9} {'emu/truth-1 %':>14} "
              f"{'null':>10} {'N':>10}")
        for a, b in zip(qs[:-1], qs[1:]):
            m = good & (v >= a) & (v <= b if b == qs[-1] else v < b)
            n = int(m.sum())
            if n < 200:
                continue
            t, p, nl = truth[m].mean(), pred[m].mean(), null[m].mean()
            print(f"  [{a:>9.4f},{b:>9.4f}) {t:>9.4f} {p:>9.4f} "
                  f"{(p/t-1)*100 if t else np.nan:>+14.2f} {nl:>+10.5f} {n:>10,}")

    print("\nREAD THIS AS: a null test that fails in a bin invalidates the truth in THAT bin.")
    print("A deficit that shrinks on clean cross-matches means the truth is inflated by")
    print("deblending inconsistency between legs, not that the emulator is wrong.")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, truth=truth, null=null, pred=pred, distance=dist)
        print(f"\nsaved {args.output}")
    print("RULER_VALIDITY_DONE", flush=True)


if __name__ == "__main__":
    main()
