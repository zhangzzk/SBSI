"""28k's unclosed blocker: do the response TARGET's sim and the RULER's sim agree about the response?

WHY THIS IS NOW THE LOAD-BEARING QUESTION
-----------------------------------------
The in-domain m is a cancellation of a -3.72% pin residual against a +3.21% target defect
(WORKLOG 2026-07-30b RESULT 1). Every other explanation for the target defect has now been bounded
and excluded by measurement:

  multiplicative R_blend mis-calibration   EXCLUDED (B is flat over a 20x range of R_blend, and
                                           R_blend is NEGATIVE in the bin where B is largest)
  forward-difference / estimator amplitude ~0.3%, and the chord is EVEN in g so the correction is
                                           smaller still -- far too small for +3.2%
  population / cell transfer               <= 0.02 pp (within-cell conditionals agree to L1=0.02)
  aperture k-truncation                    +0.0002 inside the trained r_max=10"
  coupling-pin competition                 absent (swept at four lam_theta values)

What is left is exactly what WORKLOG 2026-07-28k flagged and never closed:

    "Target and ruler are built from DIFFERENT catalogues (det_meas_crowd_g0.05_val_full vs
     det_meas_ngmix_g0.05_val), so a population or selection difference between them is the obvious
     suspect. Worth checking before any further work on the flow."

with a ~1.8% target-vs-ruler discrepancy already measured in the boundary size bin.

THE TEST, AND WHY IT IS THE SHARPEST AVAILABLE
----------------------------------------------
Both catalogues carry `measured_ngmix_g1/g2` keyed by the same `(case, input_index)`. If they are the
same underlying render measured the same way, the shapes must agree OBJECT BY OBJECT. So instead of
comparing aggregate responses (which confounds population, selection and measurement), join on the key
and compare directly:

  1. how many keys are shared at all, and does either side carry rows the other lacks (SELECTION);
  2. for shared keys, is `measured_ngmix_g*` identical, and if not what is the offset and scatter
     (MEASUREMENT);
  3. the raw response `<e.ghat>/g` from each catalogue on the COMMON object set, in matched
     (mag, size) bins -- the difference here is the quantity that propagates into the pin target.

Step 3 on the common set is the number that matters: it removes population and selection differences
by construction, so anything left is the two pipelines genuinely disagreeing about the same photons.

FIREWALL: reads only half-shear catalogues. No constgold, no r_sim, no m. Trains nothing.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sbs_shear.paths import catalogue

KEY = ["case", "input_index"]
COLS = ["case", "input_index", "r_input_p", "Re_input_p", "detected",
        "measured_ngmix_g1", "measured_ngmix_g2", "applied_g1", "applied_g2"]


def load(path, max_case, cols):
    d = ds.dataset(path, format="feather")
    have = [c for c in cols if c in d.schema.names]
    missing = [c for c in cols if c not in d.schema.names]
    t = d.to_table(columns=have, filter=(ds.field("case") <= max_case)).to_pandas()
    return t, have, missing


def raw_response(df, g1c="applied_g1", g2c="applied_g2"):
    """<e.ghat>/|g| with ghat from the applied shear -- the same projection the target builder uses."""
    g1 = df[g1c].to_numpy(float)
    g2 = df[g2c].to_numpy(float)
    gmag = np.hypot(g1, g2)
    ok = gmag > 0
    gh1 = np.where(ok, g1 / np.maximum(gmag, 1e-12), 0.0)
    gh2 = np.where(ok, g2 / np.maximum(gmag, 1e-12), 0.0)
    proj = df["measured_ngmix_g1"].to_numpy(float) * gh1 + df["measured_ngmix_g2"].to_numpy(float) * gh2
    return proj, gmag, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cat", default=catalogue("det_meas_crowd_g0.05_val_full.feather"))
    ap.add_argument("--ruler-cat", default=catalogue("det_meas_ngmix_g0.05_val.feather"))
    ap.add_argument("--max-case", type=int, default=4,
                    help="a few cases is ample: this is a systematic offset, not a noise question")
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    args = ap.parse_args()

    print(f"cases 0..{args.max_case}\n  TARGET cat: {args.target_cat.split('/')[-1]}"
          f"\n  RULER  cat: {args.ruler_cat.split('/')[-1]}\n")
    A, hA, mA = load(args.target_cat, args.max_case, COLS)
    B, hB, mB = load(args.ruler_cat, args.max_case, COLS)
    print(f"TARGET rows {len(A):,}  (missing cols: {mA})")
    print(f"RULER  rows {len(B):,}  (missing cols: {mB})")

    # all-pairs catalogues repeat a primary once per neighbour; collapse to one row per primary
    for nm, d in (("TARGET", A), ("RULER", B)):
        dup = d.duplicated(KEY).sum()
        print(f"  {nm}: {dup:,} duplicate keys -> {'all-pairs, deduplicating' if dup else 'one row per primary'}")
    A = A.drop_duplicates(KEY)
    B = B.drop_duplicates(KEY)

    # (1) SELECTION
    ka = set(map(tuple, A[KEY].to_numpy()))
    kb = set(map(tuple, B[KEY].to_numpy()))
    both = ka & kb
    print(f"\n--- (1) SELECTION ---")
    print(f"  TARGET-only keys : {len(ka - kb):,}")
    print(f"  RULER-only keys  : {len(kb - ka):,}")
    print(f"  shared keys      : {len(both):,}  "
          f"({100 * len(both) / max(len(ka | kb), 1):.1f}% of the union)")

    M = A.merge(B, on=KEY, how="inner", suffixes=("_t", "_r"))
    print(f"  joined rows      : {len(M):,}")
    if not len(M):
        raise SystemExit("no shared keys -- the two catalogues do not describe the same objects")

    # (2) MEASUREMENT, object by object
    print(f"\n--- (2) MEASUREMENT: same object, same shape? ---")
    for c in ("measured_ngmix_g1", "measured_ngmix_g2"):
        x, y = M[f"{c}_t"].to_numpy(float), M[f"{c}_r"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        d = y[ok] - x[ok]
        ident = float(np.mean(np.abs(d) < 1e-12))
        print(f"  {c}: bit-identical on {100 * ident:.2f}% of shared rows; "
              f"mean diff {d.mean():+.6f}, sd {d.std():.6f}, "
              f"max|diff| {np.abs(d).max():.4g}  (n={ok.sum():,})")
    for c in ("r_input_p", "Re_input_p"):
        if f"{c}_t" in M.columns and f"{c}_r" in M.columns:
            d = M[f"{c}_r"].to_numpy(float) - M[f"{c}_t"].to_numpy(float)
            print(f"  {c}: mean diff {np.nanmean(d):+.6g} (truth columns should be identical)")

    # (3) RESPONSE on the COMMON object set, matched bins
    print(f"\n--- (3) RAW RESPONSE on the SHARED objects, matched (mag,size) bins ---")
    sel = ((M["r_input_p_t"].to_numpy(float) < args.true_mag_max)
           & (M["Re_input_p_t"].to_numpy(float) > args.true_re_min))
    for nm, extra in (("all shared", np.ones(len(M), bool)), ("in-domain", sel)):
        S = M[extra]
        if not len(S):
            continue
        out = {}
        for tag, sfx in (("TARGET", "_t"), ("RULER", "_r")):
            d = pd.DataFrame({
                "measured_ngmix_g1": S[f"measured_ngmix_g1{sfx}"],
                "measured_ngmix_g2": S[f"measured_ngmix_g2{sfx}"],
                "applied_g1": S[f"applied_g1{sfx}"] if f"applied_g1{sfx}" in S else S.get("applied_g1"),
                "applied_g2": S[f"applied_g2{sfx}"] if f"applied_g2{sfx}" in S else S.get("applied_g2"),
            })
            proj, gmag, ok = raw_response(d)
            out[tag] = float(np.nanmean(proj[ok] / gmag[ok]))
        t, r = out["TARGET"], out["RULER"]
        print(f"  {nm:>10}: TARGET R={t:+.4f}   RULER R={r:+.4f}   "
              f"ruler/target-1 = {100 * (r / t - 1):+.2f}%   (n={len(S):,})")

    print("\nHOW TO READ THIS. If (2) shows the shapes are bit-identical and (3) shows ~0%, the two")
    print("catalogues are the same measurement and 28k's discrepancy is a POPULATION/BINNING artefact,")
    print("which the transfer tests have already bounded at <=0.02 pp -- meaning the target defect (B)")
    print("is NOT explained by the catalogue difference and the search must continue elsewhere.")
    print("If (2) shows a systematic shape offset or (3) shows a few percent, the two sims genuinely")
    print("disagree, the pin has been supervised against the wrong number, and rebuilding the target on")
    print("the RULER's catalogue is the indicated fix (firewall-clean: half-shear only).")


if __name__ == "__main__":
    main()
