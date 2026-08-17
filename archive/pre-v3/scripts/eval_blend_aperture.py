"""How much blending response sits OUTSIDE the emulator's k=20 / r_max=10" summation aperture?

WHY
---
The in-domain m decomposes (job 15348680) into a pin residual of -3.72% against a target defect of
+3.21%. The target defect is a near-CONSTANT additive deficit of about -0.027 in absolute response
units, flat across a 20x range of `R_blend` -- so it is not a multiplicative mis-calibration of the
emulator. One of the two surviving explanations is that the SUMMED R_blend is simply missing a
roughly constant far-field contribution: `predict_response` sums at most k=20 neighbours inside
r_max=10" (from emulator_metadata_lsst_r_extnbr_ho.json), and anything past that is dropped. A
truncation deficit is additive and does not scale with near-neighbour crowding, which matches the
observed flatness.

This script widens k and r_max and measures how <R_blend> per primary responds. If widening recovers
~0.027, hypothesis 2 is supported and the identity closes on the R_blend side.

HONEST CAVEAT, stated up front: the regression emulator's own training cut is `[0,10]` in distance, so
anything beyond 10" is EXTRAPOLATION, and 28j measured its per-pair bias at +300% in the 7-8.7"
annulus -- i.e. it over-predicts as it approaches the edge. So a widened sum is an UPPER BOUND on the
missing contribution, not a calibrated value. The useful outputs are (a) the k-truncation effect at
FIXED r_max=10", which is entirely inside the trained aperture and therefore trustworthy, and (b) the
r_max scan as a bound.

FIREWALL: reads constgold INPUT catalogues (true positions and properties) and runs the emulator.
No measured shear response, no r_sim, no m. Trains nothing.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from sbs_shear.paths import CONST_SIM_DIR as CBASE

BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", default=[40, 41, 42])
    ap.add_argument("--tag", default="lsst_r_extnbr_ho")
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--variants", default="20:10,40:10,60:10,20:15,40:15,60:20",
                    help="comma list of k:r_max to compare against the certified 20:10")
    args = ap.parse_args()

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")

    base_cuts, base_rmax, base_k = pred._select("regression")
    print(f"emulator {args.tag}: trained regression aperture k={base_k}, r_max={base_rmax}\n"
          f"  cuts={base_cuts}\n"
          f"  (anything beyond r_max={base_rmax} is EXTRAPOLATION -- upper bound only)\n")

    variants = []
    for v in args.variants.split(","):
        k, r = v.split(":")
        variants.append((int(k), float(r)))

    rows = []
    for case in args.cases:
        fp = f"{CBASE}/case{case}_{args.sign}/real0/catalogues/input/gals_info_{TILE}.feather"
        if not os.path.exists(fp):
            print(f"case{case}: MISSING {fp}")
            continue
        t = pf.read_table(fp).to_pandas()
        t = t.rename(columns={c: c.replace("_input", "") for c in t.columns})
        # the deliverable population, on TRUE input properties
        magcol = "r_p" if "r_p" in t.columns else ("r" if "r" in t.columns else None)
        recol = "Re_p" if "Re_p" in t.columns else ("Re" if "Re" in t.columns else None)
        for k, rmax in variants:
            pred._select = (lambda task, _c=base_cuts, _k=k, _r=rmax:
                            (_c, _r, _k))                      # monkeypatch: widen the aperture only
            reg = pred.predict_response(t, t)
            pk = [c for c in reg.columns if c.startswith("index")][0]
            rb = reg.groupby(pk)["response"].sum()
            keep = None
            if magcol and recol:
                idx = t.reset_index().rename(columns={"index": "_i"})
                sel = idx[(idx[magcol].to_numpy(float) < args.true_mag_max)
                          & (idx[recol].to_numpy(float) > args.true_re_min)]
                keep = set(sel["_i"].to_numpy())
            vals = rb.to_numpy(float)
            if keep is not None:
                mask = np.array([i in keep for i in rb.index.to_numpy()])
                vals_dom = vals[mask]
            else:
                vals_dom = vals
            rows.append(dict(case=case, k=k, r_max=rmax, n=len(vals),
                             mean_all=vals.mean(), n_dom=len(vals_dom),
                             mean_dom=vals_dom.mean() if len(vals_dom) else np.nan))
            print(f"  case{case} k={k:>3} r_max={rmax:>4}: {len(vals):,} primaries, "
                  f"<R_blend>_all={vals.mean():.4f}  in-domain n={len(vals_dom):,} "
                  f"<R_blend>_dom={vals_dom.mean() if len(vals_dom) else float('nan'):.4f}", flush=True)

    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("no cases processed")
    g = d.groupby(["k", "r_max"])[["mean_all", "mean_dom"]].mean().reset_index()
    ref = g[(g["k"] == base_k) & (g["r_max"] == base_rmax)]
    r_all = float(ref["mean_all"].iloc[0]) if len(ref) else np.nan
    r_dom = float(ref["mean_dom"].iloc[0]) if len(ref) else np.nan
    print("\n" + "=" * 74)
    print(f"averaged over {d['case'].nunique()} cases; reference = certified k={base_k}, r_max={base_rmax}")
    print("=" * 74)
    print(f"{'k':>4} {'r_max':>6} {'<R_b> all':>10} {'d vs ref':>9} {'<R_b> dom':>10} {'d vs ref':>9} "
          f"{'in-domain':>10}")
    for _, r in g.iterrows():
        print(f"{int(r['k']):>4} {r['r_max']:>6.1f} {r['mean_all']:>10.4f} "
              f"{r['mean_all'] - r_all:>+9.4f} {r['mean_dom']:>10.4f} "
              f"{r['mean_dom'] - r_dom:>+9.4f} "
              f"{'(trained)' if r['r_max'] <= base_rmax else '(extrap)':>10}")
    print(f"\nThe target defect to be explained is +0.0277 in <R_blend> (in-domain, constgold weights).")
    print("Rows at r_max=10 are INSIDE the trained aperture and test k-truncation alone; rows beyond")
    print("are extrapolation and bound the far-field term from above.")


if __name__ == "__main__":
    main()
