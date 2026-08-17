"""SIM detection response vs blendedness BETA, faceted by true SIZE and by true MAGNITUDE (constgold,
shear-free intrinsic). For the sim-vs-model plot (cont.165), TWO panels:

  SIZE facet: within a fixed mag band [--mag-lo,--mag-hi), curves = true-Re bins.
  MAG  facet: over [--mag-edges], pooling size, curves = true-r-mag bins.

beta is computed for ALL galaxies (isolated / no neighbour -> beta = 0; the beta==0 bin is just the
lowest beta value, no separate ISOLATED estimator). det-bias per cell = (<e_int>_+ - <e_int>_-)/0.04
(R_both=0 for shear-free shapes -> the whole number is detection selection). Edges are saved so
scripts/eval_model_beta_bysize.py reuses them for the overlay. FIREWALL: read-only.
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


def beta_all(d):
    b = severities(d)["beta"]
    return np.where(np.isfinite(b), b, 0.0)


def db_mask(etp, etm, both_p, both_m, only_p, only_m, mp, mm, g):
    Rb, Rf, db, se, npall, nmall = bias_in_mask(
        etp, etm, mp, mm, both_p & mp, both_m & mm, only_p & mp, only_m & mm, g)
    return db, se, npall


def facet(name, prim_p, prim_m, pedges, pmask_p, pmask_m, plabels, bp_, bm_, bedges,
          dp, dm, etp, etm, both_p, both_m, only_p, only_m, g):
    """Compute det-bias on a (primary x beta) grid. Returns facet-prefixed dict."""
    out = {f"{name}__pedges": pedges, f"{name}__beta_edges": bedges,
           f"{name}__labels": np.array(plabels)}
    print(f"\n#### facet {name}: edges={np.round(pedges,3)}  beta>0 edges={np.round(bedges,4)}")
    for pi, lab in enumerate(plabels):
        pp = pmask_p & (prim_p >= pedges[pi]) & (prim_p < pedges[pi + 1])
        pm = pmask_m & (prim_m >= pedges[pi]) & (prim_m < pedges[pi + 1])
        specs = [(bp_ == 0, bm_ == 0)]
        for b in range(len(bedges) - 1):
            specs.append(((bp_ >= bedges[b]) & (bp_ < bedges[b + 1]),
                          (bm_ >= bedges[b]) & (bm_ < bedges[b + 1])))
        dbs, ses, Ns, bmed, primed = [], [], [], [], []
        for selp, selm in specs:
            mp = pp & selp; mm = pm & selm
            if int(mp.sum()) < 2000 or int(mm.sum()) < 2000:
                dbs.append(np.nan); ses.append(np.nan); Ns.append(0)
                bmed.append(np.nan); primed.append(np.nan); continue
            db, se, N = db_mask(etp, etm, both_p, both_m, only_p, only_m, mp, mm, g)
            dbs.append(db); ses.append(se); Ns.append(N)
            bmed.append(float(np.nanmedian(np.concatenate([bp_[mp], bm_[mm]]))))
            primed.append(float(np.nanmedian(prim_p[mp])))
        out[f"{name}__{lab}__db"] = np.array(dbs)
        out[f"{name}__{lab}__se"] = np.array(ses)
        out[f"{name}__{lab}__N"] = np.array(Ns, float)
        out[f"{name}__{lab}__beta_med"] = np.array(bmed)
        out[f"{name}__{lab}__prim_med"] = np.array(primed)
        print(f"  {lab}: " + "  ".join(f"{100*d:+.2f}%" if np.isfinite(d) else "  --  " for d in dbs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--mag-lo", type=float, default=25.5, help="SIZE facet: fixed mag band low")
    ap.add_argument("--mag-hi", type=float, default=26.5, help="SIZE facet: fixed mag band high")
    ap.add_argument("--n-size", type=int, default=5)
    ap.add_argument("--mag-edges", type=float, nargs="+", default=[24.5, 25.0, 25.5, 26.0, 26.5],
                    help="MAG facet: r-mag bin edges (pooling size)")
    ap.add_argument("--n-beta", type=int, default=5)
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

    bp_ = beta_all(dp); bm_ = beta_all(dm)

    def bpos_edges(mask_p, mask_m, nb):
        pool = np.concatenate([bp_[mask_p & both_p & (bp_ > 0)], bm_[mask_m & both_m & (bm_ > 0)]])
        e = np.quantile(pool, np.linspace(0, 1, nb + 1)); e[-1] += 1e-9
        return e

    out = {"g": g, "mag_lo": args.mag_lo, "mag_hi": args.mag_hi}

    # ---- SIZE facet (fixed mag band) ----
    lo, hi = args.mag_lo, args.mag_hi
    smp = (dp["mag"] >= lo) & (dp["mag"] < hi); smm = (dm["mag"] >= lo) & (dm["mag"] < hi)
    spool = np.concatenate([dp["Re"][smp & both_p], dm["Re"][smm & both_m]])
    sedges = np.quantile(spool, np.linspace(0, 1, args.n_size + 1)); sedges[0] -= 1e-9; sedges[-1] += 1e-9
    slabels = [f"Re[{sedges[i]:.2f},{sedges[i+1]:.2f})" for i in range(args.n_size)]
    sbeta = bpos_edges(smp, smm, args.n_beta)
    out.update(facet("SIZE", dp["Re"], dm["Re"], sedges, smp, smm, slabels, bp_, bm_, sbeta,
                     dp, dm, etp, etm, both_p, both_m, only_p, only_m, g))

    # ---- MAG facet (pool size, mag bins) ----
    medges = np.array(args.mag_edges, float)
    mlo, mhi = medges[0], medges[-1]
    mmp = (dp["mag"] >= mlo) & (dp["mag"] < mhi); mmm = (dm["mag"] >= mlo) & (dm["mag"] < mhi)
    mlabels = [f"r[{medges[i]:.1f},{medges[i+1]:.1f})" for i in range(len(medges) - 1)]
    mbeta = bpos_edges(mmp, mmm, args.n_beta)
    out.update(facet("MAG", dp["mag"], dm["mag"], medges, mmp, mmm, mlabels, bp_, bm_, mbeta,
                     dp, dm, etp, etm, both_p, both_m, only_p, only_m, g))

    if args.output:
        np.savez(args.output, **out)
        print(f"\nsaved {args.output}", flush=True)
    print("\nDETECTION_BETA_BYSIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
