"""Do constgold and the half-shear legs have the SAME amount of blending?

WHY THIS MATTERS NOW. 2026-08-07f measured, on the half-shear all-pairs leg with no emulator
involved, that the sim's own total response falls well short of constgold's:

    R_self  (ghat_p)                        0.8165 +- 0.0080
    R_blend (ghat_s, summed inside 7")      0.0865 +- 0.0166
    total                                   0.9030            vs constgold R_sim 0.9626  (-6.2%)

and that the blend needed to close it from R_self is 0.1460 -- 69% above what the sim actually
delivers inside 7". Two readings survive, and they call for opposite responses:

  (i)  THE SIMS ARE DIFFERENTLY CROWDED. constgold scenes contain more, or closer, neighbours, so
       they genuinely have more blend response. Nothing is broken; the flow is simply trained on a
       less-blended population than it is judged on, and the fix is a population/target question.
  (ii) THE SIMS DISAGREE ABOUT THE RESPONSE ITSELF at equal crowding, in which case no reweighting
       or emulator change closes V2.2.

They are separated by measuring crowding directly, which is possible because BOTH catalogues are
pair-annotated with a `distance` column: constgold (41.7M pair-rows, `input_index_sec`/`distance`)
and the half-shear all-pairs build `det_meas_ngmix_ap7_*` (25.7M pair-rows). Counting pairs per
object as a function of separation in each gives a like-for-like crowding curve.

READ THE CURVE, NOT ONE NUMBER. Each catalogue was built with its own `k` and `r_max`, and a curve
that flattens is showing you the BUILD LIMIT, not the sky -- the ap7 leg is capped at 7", so its
count cannot rise past that however crowded the field is. Only the range where both curves are still
rising is informative about the actual scenes; beyond that the comparison is about build settings.
This is the same pair-list discipline AGENTS.md insists on, and the reason job 15596566 had to be
thrown away.

WHAT THIS CANNOT SETTLE. Equal pair counts would not prove equal blending -- neighbour brightness and
size matter too -- so neighbour magnitude is reported alongside. And a crowding difference explains a
blend-response difference, not necessarily the 2.4% SELF-response gap, which is a separate finding.
Nothing here is corrected or applied.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

CONSTGOLD = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
             "constant_response_catalogue_train.feather")
HALFSHEAR = ("/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
             "det_meas_ngmix_ap7_g0.02_test.feather")
RADII = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]


def crowding(path, label, mag_max, re_min, min_case, max_case):
    """Pairs per object vs separation, using the catalogue's own pair list."""
    keys, dists, nbr_mag = [], [], []
    with ipc.open_file(pa.memory_map(path)) as reader:
        names = set(reader.schema.names)
        cols = [c for c in ["case", "input_index", "distance", "r_input_p", "Re_input_p",
                            "r_input_s", "neighbored", "detected"] if c in names]
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            c = b["case"].to_numpy()
            b = b[(c >= min_case) & (c <= max_case)]
            if "detected" in b.columns:
                b = b[b["detected"].astype(bool)]
            b = b[(b["r_input_p"].to_numpy() < mag_max) & (b["Re_input_p"].to_numpy() > re_min)]
            if not len(b):
                continue
            keys.append(b["case"].to_numpy(np.int64) * 1_000_003
                        + b["input_index"].to_numpy(np.int64))
            d = b["distance"].to_numpy(float)
            if "neighbored" in b.columns:      # a row with no rendered neighbour is not a pair
                d = np.where(b["neighbored"].to_numpy(bool), d, np.inf)
            dists.append(d)
            nbr_mag.append(b["r_input_s"].to_numpy(float) if "r_input_s" in b.columns
                           else np.full(len(b), np.nan))
    key = np.concatenate(keys)
    dist = np.concatenate(dists)
    smag = np.concatenate(nbr_mag)
    uk, inv = np.unique(key, return_inverse=True)
    nobj = len(uk)

    print(f"\n{label}")
    print(f"  {len(key):,} pair-rows over {nobj:,} objects  "
          f"({len(key)/nobj:.2f} rows per object)")
    print(f"  {'r <':>6}  {'pairs/obj':>10}  {'cumulative':>11}  {'<nbr mag>':>10}")
    prev = 0.0
    for r in RADII:
        sel = np.isfinite(dist) & (dist <= r)
        cum = sel.sum() / nobj
        mm = float(np.nanmean(smag[sel])) if sel.any() else float("nan")
        flag = "  <- flat: build limit reached" if abs(cum - prev) < 1e-9 and prev > 0 else ""
        print(f"  {r:6.1f}  {cum-prev:10.2f}  {cum:11.2f}  {mm:10.2f}{flag}")
        prev = cum
    return {"n_obj": nobj, "cum": {r: float((np.isfinite(dist) & (dist <= r)).sum() / nobj)
                                   for r in RADII}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    args = ap.parse_args()
    t0 = time.time()

    print(f"V2.2 box on the PRIMARY: true mag < {args.mag_max}, Re > {args.re_min}")
    print("neighbours are not cut on; each catalogue contributes its own pair list.")

    cg = crowding(CONSTGOLD, "CONSTGOLD  constant_response_catalogue_train (cases >= 40)",
                  args.mag_max, args.re_min, 40, 10**9)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    hs = crowding(HALFSHEAR, "HALF-SHEAR  det_meas_ngmix_ap7_g0.02_test (all-pairs, 7\" build)",
                  args.mag_max, args.re_min, 0, 10**9)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\nRATIO constgold / half-shear, pairs per object")
    print(f"  {'r <':>6}  {'constgold':>10}  {'half-shear':>11}  {'ratio':>8}")
    for r in RADII:
        a, b = cg["cum"][r], hs["cum"][r]
        print(f"  {r:6.1f}  {a:10.2f}  {b:11.2f}  {a/b if b else float('nan'):8.2f}")

    print("\nHOW TO READ THIS")
    print("  A ratio well above 1 in the range where BOTH curves are still rising means constgold")
    print("  scenes really are more blended -- reading (i) -- and the 6.2% total-response gap is")
    print("  largely a population difference between the training and evaluation sims.")
    print("  A ratio near 1 there means crowding is NOT the explanation and the two sims disagree")
    print("  about the response itself -- reading (ii).")
    print("  Where a curve has gone flat it is reporting its build cap, so the ratio there compares")
    print("  build settings and says nothing about the sky. Ignore those rows.")
    print("\nDIAG_CROWDING_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
