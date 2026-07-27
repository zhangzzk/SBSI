"""Detection bias vs the FLOW's neighbour-flux severity axis (nbr_flux_near), on constgold.

Motivation (cont.162): blendedness beta is entangled with primary SIZE (its kernel divides
separation by the primary's convolved Re), so small high-surface-brightness galaxies pile into
low beta and drag the low-severity det-bias toward 0. The flow instead conditions on
  nbr_flux_near = log10(1 + F_near / aperture_rms),  F_near = sum flux of ALL input neighbours <3"
normalized by a FIXED survey aperture_rms -- NO primary-size or primary-flux normalization. So it
should not carry the size confound. Built by scripts/build_crowding_lookup.py, keyed (case,input_index);
lookup /project/.../sbsi_caches/crowd_flux_det_c40-139.feather covers exactly the min-case-40 cases.

Bins ALL galaxies by nbr_flux_near (0 == isolated, >0 in equal-count bins) x true primary mag, averages
the DETECTED sample's sheared-intrinsic shape per leg, det-bias = R_full/R_both-1 (R_both=1). Reports
<Re> and <mag> per bin (to expose/deny a size confound), plus a fixed-(mag,size) control.

FIREWALL: read-only.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from scripts.eval_detection_severity import (  # noqa: E402
    load_leg, build_pair, bias_in_mask, PLUS, MINUS)

LOOKUP = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/crowd_flux_det_c40-139.feather"
BIG = np.int64(1) << 34


def merge_nbrflux(d, col):
    """Map lookup `col` onto leg rows via (case,input_index); NaN if absent."""
    lk = pf.read_table(LOOKUP, columns=["case", "input_index", col]).to_pandas()
    lk_key = lk["case"].to_numpy(np.int64) * BIG + lk["input_index"].to_numpy(np.int64)
    lk_val = lk[col].to_numpy(float)
    order = np.argsort(lk_key, kind="stable")
    lk_key, lk_val = lk_key[order], lk_val[order]
    leg_key = d["case"] * BIG + d["input_index"]
    pos = np.clip(np.searchsorted(lk_key, leg_key), 0, lk_key.size - 1)
    hit = lk_key[pos] == leg_key
    out = np.where(hit, lk_val[pos], np.nan)
    return out, hit.mean()


def db_mask(etp, etm, both_p, both_m, only_p, only_m, mp, mm, g):
    Rb, Rf, db, se, npall, nmall = bias_in_mask(
        etp, etm, mp, mm, both_p & mp, both_m & mm, only_p & mp, only_m & mm, g)
    return db, se, npall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="nbr_flux_near", choices=["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"])
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--nbins", type=int, default=6)
    ap.add_argument("--mag-edges", type=float, nargs="+", default=None,
                    help="consecutive true-r-mag bin edges -> one curve per bin (e.g. 24.5 25 25.5 26 26.5)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    dp = load_leg(PLUS)
    dm = load_leg(MINUS)
    if args.min_case > 0:
        kp = dp["case"] >= args.min_case; km = dm["case"] >= args.min_case
        dp = {k: v[kp] for k, v in dp.items()}; dm = {k: v[km] for k, v in dm.items()}
    g = float(np.median(np.hypot(dp["g1"], dp["g2"])))
    print(f"g={g:.4f}  +leg {dp['case'].size:,}  -leg {dm['case'].size:,}", flush=True)

    nf_p, hp = merge_nbrflux(dp, args.col)
    nf_m, hm = merge_nbrflux(dm, args.col)
    print(f"{args.col} merge hit-rate: +{hp:.4f} / -{hm:.4f}", flush=True)
    print(f"  +leg: frac isolated({args.col}==0)={np.mean(nf_p==0):.3f}  "
          f"vs neighbored=True frac={dp['neighbored'].mean():.3f}", flush=True)

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

    if args.mag_edges:
        e = args.mag_edges
        mag_bands = [(f"r[{e[i]:.1f},{e[i+1]:.1f})", e[i], e[i + 1]) for i in range(len(e) - 1)]
    else:
        mag_bands = [("all", -np.inf, np.inf), ("bright r<25.5", -np.inf, 25.5),
                     ("faint r>=25.5", 25.5, np.inf)]
    # equal-count edges over BLENDED (nf>0) both-detected pooled
    pool = np.concatenate([nf_p[(nf_p > 0) & both_p], nf_m[(nf_m > 0) & both_m]])
    edges = np.quantile(pool, np.linspace(0, 1, args.nbins + 1)); edges[0] -= 1e-9; edges[-1] += 1e-9
    print(f"\n{args.col} blended edges = {np.round(edges,4)}")

    out = {"g": g, "col": args.col, "edges": edges,
           "mag_bands": np.array([b[0] for b in mag_bands])}
    for bl, lo, hi in mag_bands:
        magp = (dp["mag"] >= lo) & (dp["mag"] < hi); magm = (dm["mag"] >= lo) & (dm["mag"] < hi)
        print(f"\n==== mag band: {bl} ====")
        print(f"  {'bin':>22} {'ctr':>6} {'<Re>':>6} {'<mag>':>6} {'det-bias':>9} {'N/leg':>11}")
        # isolated (nf==0)
        mip = (nf_p == 0) & magp; mim = (nf_m == 0) & magm
        db, se, N = db_mask(etp, etm, both_p, both_m, only_p, only_m, mip, mim, g)
        rr, mm_ = dp["Re"][mip].mean(), dp["mag"][mip].mean()
        print(f"  {'ISOLATED (nf=0)':>22} {0.0:>6.2f} {rr:>6.3f} {mm_:>6.2f} {100*db:>+8.3f}% {N:>11,}")
        rows = [("iso", 0.0, rr, mm_, db, se, N)]
        for i in range(args.nbins):
            mp = magp & (nf_p > 0) & np.isfinite(nf_p) & (nf_p >= edges[i]) & (nf_p < edges[i + 1])
            mm2 = magm & (nf_m > 0) & np.isfinite(nf_m) & (nf_m >= edges[i]) & (nf_m < edges[i + 1])
            if int(mp.sum()) < 2000 or int(mm2.sum()) < 2000:
                continue
            db, se, N = db_mask(etp, etm, both_p, both_m, only_p, only_m, mp, mm2, g)
            ctr = float(np.nanmedian(np.concatenate([nf_p[mp], nf_m[mm2]])))
            rr, mm_ = dp["Re"][mp].mean(), dp["mag"][mp].mean()
            print(f"  [{edges[i]:>6.3f},{edges[i+1]:>6.3f}) {ctr:>6.2f} {rr:>6.3f} {mm_:>6.2f} "
                  f"{100*db:>+8.3f}% {N:>11,}")
            rows.append((f"b{i}", ctr, rr, mm_, db, se, N))
        out[f"{bl}__ctr"] = np.array([r[1] for r in rows])
        out[f"{bl}__Re"] = np.array([r[2] for r in rows])
        out[f"{bl}__mag"] = np.array([r[3] for r in rows])
        out[f"{bl}__db"] = np.array([r[4] for r in rows])
        out[f"{bl}__se"] = np.array([r[5] for r in rows])
        out[f"{bl}__N"] = np.array([r[6] for r in rows])

    # fixed (mag,size) control: det-bias vs nf at fixed mag bin, size below/above per-bin median
    print("\n==== fixed mag x size: det-bias vs nbr_flux_near (isolated vs high-nf) ====")
    print(f"  {'mag bin':>12} {'size':>5} | {'ISO(nf=0)':>9} {'N':>9} | {'hi-nf':>9} {'N':>9}")
    hi_edge = edges[-2]  # top bin lower edge
    for lo, hi in [(24.0, 24.5), (25.0, 25.5), (25.5, 26.0), (26.0, 26.5), (26.5, 27.0)]:
        bp = (dp["mag"] >= lo) & (dp["mag"] < hi); bm = (dm["mag"] >= lo) & (dm["mag"] < hi)
        if dp["Re"][bp].size < 5000:
            continue
        remed = float(np.median(dp["Re"][bp]))
        for slab, spp, spm in (("small", dp["Re"] < remed, dm["Re"] < remed),
                               ("large", dp["Re"] >= remed, dm["Re"] >= remed)):
            mip = bp & spp & (nf_p == 0); mim = bm & spm & (nf_m == 0)
            mhp = bp & spp & (nf_p >= hi_edge); mhm = bm & spm & (nf_m >= hi_edge)
            if min(int(mip.sum()), int(mim.sum()), int(mhp.sum()), int(mhm.sum())) < 2000:
                continue
            dbi, _, Ni = db_mask(etp, etm, both_p, both_m, only_p, only_m, mip, mim, g)
            dbh, _, Nh = db_mask(etp, etm, both_p, both_m, only_p, only_m, mhp, mhm, g)
            print(f"  [{lo:.1f},{hi:.1f}) {slab:>5} | {100*dbi:>+8.3f}% {Ni:>9,} | {100*dbh:>+8.3f}% {Nh:>9,}")

    if args.output:
        np.savez(args.output, **out)
        print(f"\nsaved {args.output}", flush=True)
    print("\nDETECTION_NBRFLUX_DONE", flush=True)


if __name__ == "__main__":
    main()
