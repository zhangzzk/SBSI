"""DEBUG: why is ISOLATED detection response larger (more negative) than the lightly-blended beta bins?

Checks, on constgold (Stage-3, truth-only, unsheared==sheared-intrinsic det-bias with R_both=1):
  A. Is `neighbored` LEG-CONSISTENT? For both-detected pairs, does neighbored_+ == neighbored_-?
     (If not, the isolated/blended split leaks between legs and could inflate isolated.)
  B. COMPOSITION: compare ISOLATED vs LOW-beta BLENDED (neighbored & beta<thr) det-bias in FINE true
     r-mag bins, reporting N, <mag>, <Re> per bin -> exposes whether the gap is a mag/size confound
     (low-beta enriched in bright/small primaries) or survives at fixed primary properties (real effect).
  C. Same but resolved additionally by true size Re (2 size halves), the other detectability driver.

FIREWALL: read-only.
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
    load_leg, build_pair, severities, bias_in_mask, PLUS, MINUS)


def det_bias(etp, etm, both_p, both_m, only_p, only_m, mp, mm, g):
    Rb, Rf, db, se, npall, nmall = bias_in_mask(
        etp, etm, mp, mm, both_p & mp, both_m & mm, only_p & mp, only_m & mm, g)
    return db, se, npall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plus", default=PLUS)
    ap.add_argument("--minus", default=MINUS)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--beta-thr", type=float, default=1e-3,
                    help="low-beta blended = neighbored & beta < thr")
    args = ap.parse_args()

    dp = load_leg(args.plus)
    dm = load_leg(args.minus)
    if args.min_case > 0:
        kp = dp["case"] >= args.min_case
        km = dm["case"] >= args.min_case
        dp = {k: v[kp] for k, v in dp.items()}
        dm = {k: v[km] for k, v in dm.items()}
    g = float(np.median(np.hypot(dp["g1"], dp["g2"])))

    BIG = np.int64(1) << 34
    keyp = dp["case"] * BIG + dp["input_index"]
    keym = dm["case"] * BIG + dm["input_index"]
    common, ip_idx, im_idx = np.intersect1d(keyp, keym, assume_unique=True, return_indices=True)
    both_p = np.zeros(keyp.size, bool); both_p[ip_idx] = True
    both_m = np.zeros(keym.size, bool); both_m[im_idx] = True
    only_p = ~both_p; only_m = ~both_m

    sh0 = build_pair(dp, dm, 1.0)
    R_try = (np.nanmean(sh0["meas_p"][both_p]) - np.nanmean(sh0["meas_m"][both_m])) / (2 * g)
    sign = 1.0 if R_try > 0 else -1.0
    sh = build_pair(dp, dm, sign) if sign < 0 else sh0
    etp, etm = sh["shint_p"], sh["shint_m"]

    sev_p = severities(dp); sev_m = severities(dm)

    # ---------- Check A: neighbored leg-consistency on both-detected ----------
    nb_p_both = dp["neighbored"][ip_idx]
    nb_m_both = dm["neighbored"][im_idx]
    disagree = nb_p_both != nb_m_both
    print("==== A. neighbored leg-consistency (both-detected pairs) ====")
    print(f"  both-detected pairs = {common.size:,}")
    print(f"  neighbored_+ != neighbored_- : {int(disagree.sum()):,} "
          f"({100*disagree.mean():.4f}%)  [expect ~0 if flag is a shear-invariant truth property]")

    # ---------- masks ----------
    iso_p = ~dp["neighbored"]; iso_m = ~dm["neighbored"]
    lob_p = dp["neighbored"] & np.isfinite(sev_p["beta"]) & (sev_p["beta"] < args.beta_thr)
    lob_m = dm["neighbored"] & np.isfinite(sev_m["beta"]) & (sev_m["beta"] < args.beta_thr)
    print(f"\n  ISOLATED: +{int(iso_p.sum()):,} / -{int(iso_m.sum()):,}   "
          f"LOW-beta BLENDED (beta<{args.beta_thr:g}): +{int(lob_p.sum()):,} / -{int(lob_m.sum()):,}")
    print(f"  overall <mag>  iso={dp['mag'][iso_p].mean():.3f}  lowb={dp['mag'][lob_p].mean():.3f}   "
          f"<Re>  iso={dp['Re'][iso_p].mean():.3f}  lowb={dp['Re'][lob_p].mean():.3f}   "
          f"<dist|lowb>={np.nanmedian(dp['dist'][lob_p]):.2f}\"")

    # ---------- Check B: fine true-mag bins ----------
    print("\n==== B. det-bias in FINE true r-mag bins: ISOLATED vs LOW-beta BLENDED ====")
    print(f"  {'mag bin':>12} | {'ISO db':>9} {'ISO<Re>':>7} {'ISO N':>10} | "
          f"{'LOWb db':>9} {'LOWb<Re>':>8} {'LOWb N':>10}")
    edges = np.arange(23.0, 27.51, 0.25)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mip = iso_p & (dp["mag"] >= lo) & (dp["mag"] < hi)
        mim = iso_m & (dm["mag"] >= lo) & (dm["mag"] < hi)
        mlp = lob_p & (dp["mag"] >= lo) & (dp["mag"] < hi)
        mlm = lob_m & (dm["mag"] >= lo) & (dm["mag"] < hi)
        if min(int(mip.sum()), int(mim.sum()), int(mlp.sum()), int(mlm.sum())) < 3000:
            continue
        dbi, _, ni = det_bias(etp, etm, both_p, both_m, only_p, only_m, mip, mim, g)
        dbl, _, nl = det_bias(etp, etm, both_p, both_m, only_p, only_m, mlp, mlm, g)
        rei = dp["Re"][mip].mean(); rel = dp["Re"][mlp].mean()
        print(f"  [{lo:.2f},{hi:.2f}) | {100*dbi:>+8.3f}% {rei:>7.3f} {ni:>10,} | "
              f"{100*dbl:>+8.3f}% {rel:>8.3f} {nl:>10,}")

    # ---------- Check C: fix mag AND size (median Re split within each mag bin) ----------
    print("\n==== C. det-bias at fixed mag x size (Re below/above the per-mag-bin median) ====")
    print(f"  {'mag bin':>12} {'size':>6} | {'ISO db':>9} {'ISO N':>10} | {'LOWb db':>9} {'LOWb N':>10}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        base_p = (dp["mag"] >= lo) & (dp["mag"] < hi)
        base_m = (dm["mag"] >= lo) & (dm["mag"] < hi)
        pool = dp["Re"][base_p]
        if pool.size < 5000:
            continue
        remed = float(np.median(pool))
        for slab, sp_p, sp_m in (("small", dp["Re"] < remed, dm["Re"] < remed),
                                 ("large", dp["Re"] >= remed, dm["Re"] >= remed)):
            mip = iso_p & base_p & sp_p; mim = iso_m & base_m & sp_m
            mlp = lob_p & base_p & sp_p; mlm = lob_m & base_m & sp_m
            if min(int(mip.sum()), int(mim.sum()), int(mlp.sum()), int(mlm.sum())) < 3000:
                continue
            dbi, _, ni = det_bias(etp, etm, both_p, both_m, only_p, only_m, mip, mim, g)
            dbl, _, nl = det_bias(etp, etm, both_p, both_m, only_p, only_m, mlp, mlm, g)
            print(f"  [{lo:.2f},{hi:.2f}) {slab:>6} | {100*dbi:>+8.3f}% {ni:>10,} | "
                  f"{100*dbl:>+8.3f}% {nl:>10,}")
    print("\nDEBUG_ISO_LOWBLEND_DONE", flush=True)


if __name__ == "__main__":
    main()
