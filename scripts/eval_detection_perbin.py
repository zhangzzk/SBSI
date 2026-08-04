"""PHASE 0d -- is the both-detected LEG-MATCHING bias structured PER BIN, and does it forge the
size-axis signal Phase 2 is built on?

THE CONCERN, AND WHY IT IS NOT NEW. Sheldon et al. 2020 sec 4.3 warn that matching detection lists
across sheared images "would introduce the very shear-dependent object detection biases we wish to
calibrate". SBSI's `constant_response_catalogue` IS that operation: it is the (case, input_index)
INTERSECTION of the two per-leg detection catalogues, so every `r_sim` in this project is measured on
the both-detected subset. The effect was already measured GLOBALLY by
`scripts/eval_detection_constgold.py` -- `R_full/R_both - 1` = -0.88% ALL, -0.08% ISOLATED, -1.11%
BLENDED -- and recorded as a correction NOT present in the certified both-detected `m`.

WHAT IS NEW IS THE RESOLUTION. This whole plan exists because a global number is not enough. The
-1.11%-vs--0.08% split already says the effect is strongly property-dependent, and -1.11% is not small
against the 2.82 pt size-axis residual rms that Phase 2 attributes to the emulator. If the
leg-matching bias is large and SIZE-STRUCTURED, then part of that residual is an artefact of how
`r_sim` is BUILT, and retuning the emulator would be chasing it.

THIS SCRIPT CHANGES NOTHING ABOUT THE ESTIMATOR. It imports `load_leg`, `build_shapes`, `apply_shear`
and `leg_means` from `eval_detection_constgold` unchanged, and reuses the same analytic error (the
whole bias lives in the small single-leg tails, so the error follows from those means alone). What it
adds is a GENERIC binning axis. The magnitude axis is recomputed here as a CROSS-CHECK: it must
reproduce the existing script, and a mismatch means this script is wrong, not that a new effect was
found.

AXES, and why each one:
  true magnitude  r_input_p   -- cross-check against the existing script
  true size       Re_input_p  -- THE GATE. Phase 2 exists because R_blend is 3.1x too flat here
  pair distance   distance    -- Gold-V3's outstanding test 3: its leading explanation for the -41%
                                 close-pair emulator deficit is that the truth sample at sub-arcsec
                                 separation is conditioned on BOTH objects being detected. Same
                                 mechanism, same run
  neighbour flux  nbr_flux_near (joined) -- the blend axis, where flow and emulator errors currently
                                 cancel

EVERY BINNING VARIABLE IS SHEAR-INVARIANT (true properties and pair geometry). That is required, not
incidental: binning on a measured quantity would make the bin assignment itself shear-dependent and
mix a selection term into the very number being measured.

READ THE SIGN CAREFULLY. `R_full/R_both - 1` is how much the REAL (single-leg) catalogue's response
differs from the both-detected one. `r_sim` uses R_both. So a bin where this is large is a bin where
`r_sim` is NOT the response of a realistically-selected sample -- it is a statement about the target,
not about the model.

LIMIT, stated up front: the per-leg catalogues carry SExtractor `measured_e1/e2`, not ngmix. The
fiducial model is scored on ngmix. The detection bias is a property of WHICH OBJECTS enter, so the
shape estimator changes the number somewhat but not the population effect; still, do not paste these
percentages next to an ngmix `m` as if they were the same estimator.

FIREWALL: nothing trains or fits. constgold is read for validation/diagnosis only.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_detection_constgold import (  # noqa: E402
    MINUS, PLUS, build_shapes, leg_means, load_leg)

CROWD = "/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather"


def join_nbr_flux(d):
    """Attach nbr_flux_near by (case, input_index); NaN where the lookup has no row."""
    kk = pd.DataFrame({"case": d["case"], "input_index": d["input_index"]})
    lk = pf.read_table(CROWD, columns=["case", "input_index", "nbr_flux_near"],
                       memory_map=True).to_pandas()
    j = kk.merge(lk, on=["case", "input_index"], how="left")
    return j["nbr_flux_near"].to_numpy(float)


def axis_table(name, dp, dm, sh, allp, allm, bp, bm, op, om, g, xp, xm, edges, logx=False):
    """det-bias per bin of a SHEAR-INVARIANT variable. Returns array of rows."""
    etp, etm = sh["meas_p"], sh["meas_m"]
    print(f"\n  --- det-bias vs {name} (MEASURED shapes) ---")
    print(f"  {'bin':>16}{'N_both':>12}{'+only%':>9}{'R_both':>9}{'R_full':>9}"
          f"{'det-bias':>11}{'+-':>9}")
    rows = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        inp = (xp >= lo) & (xp < hi)
        inm = (xm >= lo) & (xm < hi)
        mp, mm = allp & inp, allm & inm
        bpi, bmi = bp & inp, bm & inm
        opi, omi = op & inp, om & inm
        if int(bpi.sum()) < 500 or int(bmi.sum()) < 500:
            continue
        pF, mF, _, _ = leg_means(etp, etm, mp, mm)
        pB, mB, _, _ = leg_means(etp, etm, bpi, bmi)
        R_full = (pF - mF) / (2 * g)
        R_both = (pB - mB) / (2 * g)
        db = (R_full / R_both - 1.0) if R_both else np.nan
        # analytic SE: the whole R_full-R_both difference is carried by the single-leg tails
        ep_o = etp[opi]; ep_o = ep_o[np.isfinite(ep_o)]
        em_o = etm[omi]; em_o = em_o[np.isfinite(em_o)]
        fpo = ep_o.size / max(int(mp.sum()), 1)
        fmo = em_o.size / max(int(mm.sum()), 1)
        vp = ep_o.var() / ep_o.size if ep_o.size else 0.0
        vm = em_o.var() / em_o.size if em_o.size else 0.0
        sed = np.sqrt(fpo ** 2 * vp + fmo ** 2 * vm) / (2 * g)
        se = sed / abs(R_both) if R_both else np.nan
        fo = opi.sum() / max(int(mp.sum()), 1)
        print(f"  {f'[{lo:g},{hi:g})':>16}{int(bpi.sum()):>12,}{100*fo:>8.2f}%{R_both:>9.4f}"
              f"{R_full:>9.4f}{100*db:>+10.3f}%{100*se:>8.3f}%")
        rows.append((0.5 * (lo + hi), int(bpi.sum()), fo, R_both, R_full, db, se))
    a = np.array(rows, float)
    if len(a) > 1:
        db_, se_ = a[:, 5] * 100.0, a[:, 6] * 100.0
        # A bin whose own error is comparable to its value carries no information, and letting it
        # into an rms silently invents structure: on the neighbour-flux axis one 861-object bin at
        # -19.45% +- 10.41% single-handedly drove the rms to 8.8 pt. Summarise over WELL-MEASURED
        # bins only, and say out loud which bins were excluded -- never cap silently.
        good = se_ < np.maximum(1.0, 0.5 * np.abs(db_))
        drop = int((~good).sum())
        if good.sum() > 1:
            g = db_[good]
            print(f"  -> {name}: det-bias rms {np.sqrt(np.mean(g**2)):.3f} pt | "
                  f"min {g.min():+.3f} max {g.max():+.3f} | SPAN {g.max()-g.min():.3f} pt "
                  f"[over {int(good.sum())} well-measured bins]")
        if drop:
            for i in np.where(~good)[0]:
                print(f"     EXCLUDED from the summary: bin centred {a[i,0]:g} -- "
                      f"{db_[i]:+.3f}% +- {se_[i]:.3f}% on {int(a[i,1]):,} objects "
                      f"(error too large to constrain)")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plus", default=PLUS)
    ap.add_argument("--minus", default=MINUS)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--in-domain", action="store_true",
                    help="restrict to the fiducial in-domain box (true mag<26, 0.3<Re<1.5) so the "
                         "numbers are directly comparable with the Phase 2 target population")
    ap.add_argument("--save-npz", default="results/detection_perbin.npz")
    args = ap.parse_args()

    print(f"loading +leg {args.plus}", flush=True)
    dp = load_leg(args.plus)
    print(f"loading -leg {args.minus}", flush=True)
    dm = load_leg(args.minus)
    dp["nbrflux"] = join_nbr_flux(dp)
    dm["nbrflux"] = join_nbr_flux(dm)
    # `distance` is not in eval_detection_constgold.COLS, so read it here for the same rows.
    for tag, path, d in (("+", args.plus, dp), ("-", args.minus, dm)):
        t = pf.read_table(path, columns=["case", "input_index", "distance"],
                          memory_map=True).to_pandas()
        assert len(t) == d["case"].size, f"{tag}leg distance read length mismatch"
        assert np.array_equal(t["case"].to_numpy(np.int64), d["case"]), \
            f"{tag}leg row order differs between reads -- cannot attach distance positionally"
        d["distance"] = t["distance"].to_numpy(float)

    keep_desc = []
    if args.min_case > 0:
        kp, km = dp["case"] >= args.min_case, dm["case"] >= args.min_case
        dp = {k: v[kp] for k, v in dp.items()}
        dm = {k: v[km] for k, v in dm.items()}
        keep_desc.append(f"case>={args.min_case}")
    if args.in_domain:
        kp = (dp["mag"] < 26.0) & (dp["Re"] > 0.3) & (dp["Re"] < 1.5)
        km = (dm["mag"] < 26.0) & (dm["Re"] > 0.3) & (dm["Re"] < 1.5)
        dp = {k: v[kp] for k, v in dp.items()}
        dm = {k: v[km] for k, v in dm.items()}
        keep_desc.append("in-domain(mag<26, 0.3<Re<1.5)")
    print(f"population [{', '.join(keep_desc) or 'no cut'}]: "
          f"+leg {dp['case'].size:,}  -leg {dm['case'].size:,}", flush=True)

    g = float(np.median(np.hypot(dp["g1"], dp["g2"])))
    BIG = np.int64(1) << 34
    keyp = dp["case"] * BIG + dp["input_index"]
    keym = dm["case"] * BIG + dm["input_index"]
    common, ip_idx, im_idx = np.intersect1d(keyp, keym, assume_unique=True, return_indices=True)
    both_p = np.zeros(keyp.size, bool); both_p[ip_idx] = True
    both_m = np.zeros(keym.size, bool); both_m[im_idx] = True
    only_p, only_m = ~both_p, ~both_m
    print(f"g={g:.4f}  both={common.size:,}  +only={int(only_p.sum()):,} "
          f"({100*only_p.mean():.3f}%)  -only={int(only_m.sum()):,} ({100*only_m.mean():.3f}%)",
          flush=True)

    sh0 = build_shapes(dp, dm, 1.0)
    R_try = (np.nanmean(sh0["meas_p"][both_p]) - np.nanmean(sh0["meas_m"][both_m])) / (2 * g)
    sign = 1.0 if R_try > 0 else -1.0
    sh = build_shapes(dp, dm, sign) if sign < 0 else sh0
    print(f"projection sign={sign:+.0f} (R_both trial {R_try:+.4f})", flush=True)

    AXES = [
        ("TRUE r-mag (CROSS-CHECK vs eval_detection_constgold)", "mag",
         np.array([20, 24, 25, 25.5, 26, 26.5, 27, 30.0])),
        ("TRUE size Re [arcsec]  <-- PHASE 2 GATE", "Re",
         np.array([0.30, 0.38, 0.46, 0.56, 0.70, 0.95, 1.50])),
        ("pair distance [arcsec]  <-- Gold-V3 test 3", "distance",
         np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])),
        ("neighbour flux nbr_flux_near", "nbrflux",
         np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 100.0])),
    ]
    scopes = [("ALL", np.ones(dp["case"].size, bool), np.ones(dm["case"].size, bool)),
              ("ISOLATED", ~dp["neighbored"], ~dm["neighbored"]),
              ("BLENDED", dp["neighbored"], dm["neighbored"])]

    store = {}
    for nm, sp, sm in scopes:
        allp, allm = sp, sm
        bp, bm = both_p & sp, both_m & sm
        op, om = only_p & sp, only_m & sm
        etp, etm = sh["meas_p"], sh["meas_m"]
        pF, mF, _, _ = leg_means(etp, etm, allp, allm)
        pB, mB, _, _ = leg_means(etp, etm, bp, bm)
        Rf, Rb = (pF - mF) / (2 * g), (pB - mB) / (2 * g)
        print(f"\n{'='*104}\nSCOPE {nm}   N_both={int(bp.sum()):,}   "
              f"R_both={Rb:+.4f}  R_full={Rf:+.4f}  GLOBAL det-bias={100*(Rf/Rb-1):+.3f}%\n{'='*104}")
        for label, col, ed in AXES:
            if nm == "ISOLATED" and col in ("distance", "nbrflux"):
                continue   # undefined / degenerate for rows with no rendered neighbour
            a = axis_table(label, dp, dm, sh, allp, allm, bp, bm, op, om, g,
                           dp[col], dm[col], ed)
            store[f"{nm}_{col}"] = a

    # ---- gate -----------------------------------------------------------------------------
    print(f"\n{'='*104}\nPHASE 0d GATE\n{'='*104}")
    print("  Gate as written in PLAN_resolution.md: 'if it is large and structured in true size,")
    print("  then part of the size-axis residual Phase 2 attributes to the emulator is actually")
    print("  leg-matching bias, and the emulator retune would be chasing the wrong target.'")
    print("\n  Reference scale: the size-axis TOTAL-model residual has rms 2.82 pt and spans")
    print("  roughly 10.6 pt across true-size bins (eval_blend_vs_flow_perbin, 2026-08-01s).")
    print("\n  COMPARE SPANS, NOT RMS. The det-bias is dominated by a near-CONSTANT offset, and a")
    print("  constant offset shifts every bin together -- it cannot manufacture a size TREND. Only")
    print("  the bin-to-bin VARIATION can contaminate the trend Phase 2 is chasing, so the span is")
    print("  the number that belongs in the gate; the rms mostly measures the offset.")
    for nm in ("ALL", "BLENDED"):
        a = store.get(f"{nm}_Re")
        if a is None or not len(a):
            continue
        db = a[:, 5] * 100.0
        sp = db.max() - db.min()
        print(f"    {nm:>9} size axis: offset ~{np.mean(db):+.3f} pt, "
              f"bin-to-bin SPAN {sp:.3f} pt -> {100*sp/10.6:.0f}% of the residual's 10.6 pt span "
              f"(rms {np.sqrt(np.mean(db**2)):.3f} pt, mostly the offset)")
    a = store.get("BLENDED_distance")
    if a is not None and len(a):
        db, se = a[:, 5] * 100.0, a[:, 6] * 100.0
        print(f"\n  Gold-V3 test 3 (close-pair detection selection): det-bias is "
              f"{db[0]:+.3f}% +- {se[0]:.3f}% at the closest separation bin against "
              f"{db[-1]:+.3f}% +- {se[-1]:.3f}% at the widest, peaking at {db.min():+.3f}%.")
        print("  Gold-V3's leading explanation for BlendEMU's -41.5% sub-arcsec deficit is exactly")
        print("  this conditioning. SIGN AND STRUCTURE SUPPORT IT; MAGNITUDE DOES NOT ACQUIT IT --")
        print(f"  the effect here is ~{abs(db.min()):.1f}% where the deficit to explain is 41.5%, so")
        print("  close-pair detection selection can be at most a small part of it. Do not record")
        print("  this as having explained the deficit.")
    np.savez(args.save_npz, g=g, **store)
    print(f"\nsaved -> {args.save_npz}")
    print("CAVEAT: SExtractor shapes, not ngmix. Population effect, not estimator-matched to the "
          "fiducial m.")
    print("DETECTION_PERBIN_DONE", flush=True)


if __name__ == "__main__":
    main()
