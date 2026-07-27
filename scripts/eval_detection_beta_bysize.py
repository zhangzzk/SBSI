"""Detection bias vs blendedness BETA, binned by true SIZE -- does size-binning rescue beta?

cont.162 showed beta is entangled with primary size (its kernel divides separation by convolved Re),
so the raw det-bias-vs-beta curve dips below isolated at low beta purely because low beta = small
(high-peak-SB, low-bias) galaxies. Here we hold SIZE fixed (Re bins) inside a fixed magnitude band and
re-draw det-bias vs beta per size bin. If size is the whole confound, each size curve should be clean:
monotone in beta with every blended point MORE biased than that size bin's isolated reference.

Within [--mag-lo, --mag-hi): split into --n-size Re bins (equal-count); in each, isolated ref =
neighbored=False, and beta bins = equal-count quantiles of beta over that (mag,size) blended cell.
det-bias = (<e_int>_+ - <e_int>_-)/0.04 (sheared-intrinsic, R_both=1). FIREWALL: read-only.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from scripts.eval_detection_severity import (  # noqa: E402
    load_leg, build_pair, bias_in_mask, severities, PLUS, MINUS)

BIG = np.int64(1) << 34


def db_mask(etp, etm, both_p, both_m, only_p, only_m, mp, mm, g):
    Rb, Rf, db, se, npall, nmall = bias_in_mask(
        etp, etm, mp, mm, both_p & mp, both_m & mm, only_p & mp, only_m & mm, g)
    return db, se, npall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--mag-lo", type=float, default=25.5)
    ap.add_argument("--mag-hi", type=float, default=26.5)
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--n-beta", type=int, default=6)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    dp = load_leg(PLUS); dm = load_leg(MINUS)
    if args.min_case > 0:
        kp = dp["case"] >= args.min_case; km = dm["case"] >= args.min_case
        dp = {k: v[kp] for k, v in dp.items()}; dm = {k: v[km] for k, v in dm.items()}
    g = float(np.median(np.hypot(dp["g1"], dp["g2"])))

    keyp = dp["case"] * BIG + dp["input_index"]; keym = dm["case"] * BIG + dm["input_index"]
    _, ip, im = np.intersect1d(keyp, keym, assume_unique=True, return_indices=True)
    both_p = np.zeros(keyp.size, bool); both_p[ip] = True
    both_m = np.zeros(keym.size, bool); both_m[im] = True
    only_p = ~both_p; only_m = ~both_m

    sh0 = build_pair(dp, dm, 1.0)
    R_try = (np.nanmean(sh0["meas_p"][both_p]) - np.nanmean(sh0["meas_m"][both_m])) / (2 * g)
    sign = 1.0 if R_try > 0 else -1.0
    sh = build_pair(dp, dm, sign) if sign < 0 else sh0
    etp, etm = sh["shint_p"], sh["shint_m"]

    bp_ = severities(dp)["beta"]; bm_ = severities(dm)["beta"]
    lo, hi = args.mag_lo, args.mag_hi
    magp = (dp["mag"] >= lo) & (dp["mag"] < hi); magm = (dm["mag"] >= lo) & (dm["mag"] < hi)

    # equal-count SIZE edges over both-detected in-band (pooled legs)
    spool = np.concatenate([dp["Re"][magp & both_p], dm["Re"][magm & both_m]])
    sedges = np.quantile(spool, np.linspace(0, 1, args.n_size + 1))
    sedges[0] -= 1e-9; sedges[-1] += 1e-9
    print(f"mag band [{lo},{hi})  g={g:.4f}  size edges (Re) = {np.round(sedges,3)}", flush=True)

    out = {"g": g, "mag_lo": lo, "mag_hi": hi, "size_edges": sedges,
           "size_labels": np.array([f"Re[{sedges[i]:.2f},{sedges[i+1]:.2f})"
                                    for i in range(args.n_size)])}
    for si in range(args.n_size):
        slo, shi = sedges[si], sedges[si + 1]
        szp = (dp["Re"] >= slo) & (dp["Re"] < shi); szm = (dm["Re"] >= slo) & (dm["Re"] < shi)
        lab = out["size_labels"][si]
        print(f"\n==== {lab} ====")
        # isolated ref in this (mag,size) cell
        mip = magp & szp & (~dp["neighbored"]); mim = magm & szm & (~dm["neighbored"])
        dbi, sei, Ni = db_mask(etp, etm, both_p, both_m, only_p, only_m, mip, mim, g)
        print(f"  {'ISOLATED':>20} {'beta=0':>8} <Re>={dp['Re'][mip].mean():.3f} "
              f"det-bias={100*dbi:+.3f}% N={Ni:,}")
        rows = [(0.0, dbi, sei, Ni, dp["Re"][mip].mean())]
        # beta bins within this (mag,size) blended cell
        blp = magp & szp & dp["neighbored"] & np.isfinite(bp_)
        blm = magm & szm & dm["neighbored"] & np.isfinite(bm_)
        pool = np.concatenate([bp_[blp & both_p], bm_[blm & both_m]])
        bedges = np.quantile(pool, np.linspace(0, 1, args.n_beta + 1))
        bedges[0] -= 1e-12; bedges[-1] += 1e-12
        for bi in range(args.n_beta):
            mp = blp & (bp_ >= bedges[bi]) & (bp_ < bedges[bi + 1])
            mm = blm & (bm_ >= bedges[bi]) & (bm_ < bedges[bi + 1])
            if int(mp.sum()) < 2000 or int(mm.sum()) < 2000:
                continue
            db, se, N = db_mask(etp, etm, both_p, both_m, only_p, only_m, mp, mm, g)
            bctr = float(np.nanmedian(np.concatenate([bp_[mp], bm_[mm]])))
            print(f"  beta[{bedges[bi]:.4f},{bedges[bi+1]:.4f}) med={bctr:.4f} "
                  f"<Re>={dp['Re'][mp].mean():.3f} det-bias={100*db:+.3f}% N={N:,}")
            rows.append((bctr, db, se, N, dp["Re"][mp].mean()))
        out[f"{lab}__beta"] = np.array([r[0] for r in rows])
        out[f"{lab}__db"] = np.array([r[1] for r in rows])
        out[f"{lab}__se"] = np.array([r[2] for r in rows])
        out[f"{lab}__N"] = np.array([r[3] for r in rows])
        out[f"{lab}__Re"] = np.array([r[4] for r in rows])

    if args.output:
        np.savez(args.output, **out)
        print(f"\nsaved {args.output}", flush=True)
    print("\nDETECTION_BETA_BYSIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
