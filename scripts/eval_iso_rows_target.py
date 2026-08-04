"""Can we simply ADD the missing isolated primaries to the response target? Measure, don't assume.

RESULT 10 (WORKLOG 2026-07-30b) found the pin's target catalogue `det_meas_crowd_g0.05_val_full`
(= np7 + crowd columns) is a strict 22.4% subset of the ruler `det_meas_ngmix_g0.05_val`, and is
99.97% neighboured, because the np7 nearest-pair build emits NO ROW for a primary with no neighbour
inside 7". The deliverable population is ~24% isolated, so the pin has never been supervised on an
isolated galaxy.

The proposed fix is the obvious one: give those primaries a row, with zero neighbour flux. Two pieces
of machinery already do most of it --

  * `augment_crowding.py` maps a missing (case, input_index) to **0.0** for nbr_flux_near/far and
    r_blend, so an isolated row automatically lands in the lowest-crowding bin. Nothing to change.
  * the ruler catalogue already carries one row per primary including the isolated ones (it was built
    with the nearest-neighbour attach), so the rows EXIST -- they were dropped by the np7 build.

Before spending any GPU on a rebuilt target + retrain, this script establishes the four facts that
decide whether the fix is sound and what it will do:

  (1) SET      -- how many primaries the target is missing, and are they really the isolated ones
                  (neighbour distance, `neighbored` flag, detected rate)?
  (2) COVERAGE -- does the SNC g=0 lookup carry those primaries? The target is
                  [e(g) - e(0)].ghat / g, so a primary with no g=0 shape CANNOT enter an SNC target,
                  and if coverage is poor the fix needs a lookup rebuild first.
  (3) RESPONSE -- in matched (true mag x true size) cells, what is the SNC response of the missing
                  rows vs the kept rows? This is the size of the target error the omission causes.
  (4) SHIFT    -- reweighting to the deliverable population, how far does the target MOVE if the
                  missing rows are added? Compare against the -0.027 absolute shift that the measured
                  target defect (B) = +3.21% of m requires. That comparison is the whole point: it
                  says in advance whether this fix is the right size to close (B).

FIREWALL: half-shear catalogues + the g=0 lookup only. Reads the constgold catalogue for its
(mag,size) OCCUPANCY WEIGHTS only -- no measured shapes, no r_sim, no m. Trains nothing, selects
nothing.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.feather as pf
from sbs_shear.paths import catalogue

CONSTCAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
            "constant_response_catalogue_train.feather")
KEY = ["case", "input_index"]


def load_keys(path, max_case, cols):
    d = ds.dataset(path, format="feather")
    have = [c for c in cols if c in d.schema.names]
    t = d.to_table(columns=have, filter=(ds.field("case") <= max_case)).to_pandas()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler-cat", default=catalogue("det_meas_ngmix_g0.05_val.feather"))
    ap.add_argument("--np7-cat", default=catalogue("det_meas_ngmix_np7_g0.05_val.feather"))
    ap.add_argument("--g0-lookup", default="/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather")
    ap.add_argument("--max-case", type=int, default=9)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--target",
                    default="/home/z/Zekang.Zhang/SBSI/results/"
                            "response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz",
                    help="only its (mag,size) edges are used, so cells match the pin's own grid")
    args = ap.parse_args()

    # ---------------------------------------------------------------- (1) SET
    cols = ["case", "input_index", "r_input_p", "Re_input_p", "neighbored", "distance", "detected",
            "measured_ngmix_g1", "measured_ngmix_g2", "gamma1_input_p", "gamma2_input_p"]
    R = load_keys(args.ruler_cat, args.max_case, cols)
    print(f"ruler cases 0..{args.max_case}: {len(R):,} raw rows")
    # an all-pairs build repeats a primary once per neighbour; the NEAREST row defines the primary
    if "distance" in R.columns:
        R = R.sort_values("distance", kind="stable")
    R = R.drop_duplicates(KEY, keep="first").reset_index(drop=True)
    print(f"  unique primaries                : {len(R):,}")
    print(f"  detected                        : {R['detected'].astype(bool).mean() * 100:.2f}%")

    N7 = load_keys(args.np7_cat, args.max_case, KEY + ["distance", "detected"])
    N7 = N7.drop_duplicates(KEY).reset_index(drop=True)
    print(f"np7 cases 0..{args.max_case}: {len(N7):,} unique primaries "
          f"(max neighbour distance {N7['distance'].max():.3f}\")" if "distance" in N7.columns else "")

    # MultiIndex.isin returns a plain ndarray, not a Series -- keep it as one throughout
    in7 = np.asarray(pd.MultiIndex.from_frame(R[KEY]).isin(pd.MultiIndex.from_frame(N7[KEY])))
    R["in_np7"] = in7
    det = R["detected"].astype(bool).to_numpy()
    print(f"\n--- (1) SET: what the target is missing ---")
    print(f"  ruler primaries in np7          : {in7.sum():,} ({100 * in7.mean():.2f}%)")
    print(f"  ruler primaries NOT in np7      : {(~in7).sum():,} ({100 * (~in7).mean():.2f}%)")
    for nm, msk in (("in np7 (target has them)", in7), ("MISSING from np7", ~in7)):
        s = R[msk]
        d = s["distance"].to_numpy(float)
        fin = np.isfinite(d)
        print(f"  {nm:>26}: n={len(s):,}  detected={s['detected'].astype(bool).mean() * 100:5.2f}%  "
              f"neighbored={s['neighbored'].astype(bool).mean() * 100:5.2f}%  "
              f"median dist={np.median(d[fin]) if fin.any() else float('nan'):.3f}\"  "
              f"finite dist={100 * fin.mean():.1f}%")

    # the population that actually matters: detected, inside the deliverable domain
    dom = (det
           & (R["r_input_p"].to_numpy(float) < args.true_mag_max)
           & (R["Re_input_p"].to_numpy(float) > args.true_re_min))
    print(f"\n  detected AND in the deliverable domain: {dom.sum():,}")
    print(f"    of those, in np7   : {(dom & in7).sum():,} "
          f"({100 * (dom & in7).sum() / max(dom.sum(), 1):.2f}%)")
    print(f"    of those, MISSING  : {(dom & ~in7).sum():,} "
          f"({100 * (dom & ~in7).sum() / max(dom.sum(), 1):.2f}%)  "
          f"<- rows the pin has never seen")

    # ----------------------------------------------------------- (2) COVERAGE
    g0 = pf.read_table(args.g0_lookup, columns=KEY + ["ngmix0_g1", "ngmix0_g2"]).to_pandas()
    g0 = g0[g0["case"] <= args.max_case]
    print(f"\n--- (2) COVERAGE: does the SNC g=0 lookup carry the missing rows? ---")
    print(f"  g0 lookup rows (cases 0..{args.max_case}): {len(g0):,}")
    M = R.merge(g0, on=KEY, how="left")
    hasg0 = np.isfinite(M["ngmix0_g1"].to_numpy(float))
    for nm, msk in (("in np7", dom & in7), ("MISSING from np7", dom & ~in7)):
        print(f"  {nm:>18}: g0 coverage {100 * hasg0[msk].mean():.2f}%  (n={int(msk.sum()):,})")
    print("  NOTE: an SNC target needs e(0); poor coverage on the missing rows would mean the g0")
    print("        lookup must be rebuilt (it is built from the raw g=0 render, so it can be).")

    # ----------------------------------------------------------- (3) RESPONSE
    g1 = M["gamma1_input_p"].to_numpy(float)
    g2 = M["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2)
    e1 = M["measured_ngmix_g1"].to_numpy(float) - np.nan_to_num(M["ngmix0_g1"].to_numpy(float))
    e2 = M["measured_ngmix_g2"].to_numpy(float) - np.nan_to_num(M["ngmix0_g2"].to_numpy(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        Rsnc = (e1 * g1 / gm + e2 * g2 / gm) / gm
    good = np.isfinite(Rsnc) & hasg0 & (gm > 1e-6)

    z = np.load(args.target, allow_pickle=True)
    ef, es = z["edges_flux"], z["edges_size"]
    nf, ns = len(ef) - 1, len(es) - 1
    ncell = nf * ns
    mag = M["r_input_p"].to_numpy(float)
    size = M["Re_input_p"].to_numpy(float)
    cid = ((np.clip(np.digitize(mag, ef) - 1, 0, nf - 1)) * ns
           + np.clip(np.digitize(size, es) - 1, 0, ns - 1)).astype(np.int64)

    def cellmean(mask):
        mask = mask & good
        n = np.bincount(cid[mask], minlength=ncell).astype(float)
        s = np.bincount(cid[mask], weights=Rsnc[mask], minlength=ncell)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan), n

    kept = dom & in7
    miss = dom & ~in7
    R_kept, n_kept = cellmean(kept)
    R_miss, n_miss = cellmean(miss)
    R_all, n_all = cellmean(dom)

    print(f"\n--- (3) SNC RESPONSE by cell: kept (what the target measures) vs missing ---")
    print(f"{'cell(mag,size)':>15} {'n_kept':>10} {'n_miss':>10} {'R_kept':>8} {'R_miss':>8} "
          f"{'R_all':>8} {'miss-kept':>10}")
    for b in range(ncell):
        if not (n_kept[b] > 200 and n_miss[b] > 200):
            continue
        print(f"{f'({b // ns},{b % ns})':>15} {int(n_kept[b]):>10,} {int(n_miss[b]):>10,} "
              f"{R_kept[b]:>8.4f} {R_miss[b]:>8.4f} {R_all[b]:>8.4f} "
              f"{R_miss[b] - R_kept[b]:>+10.4f}")

    # -------------------------------------------------------------- (4) SHIFT
    fin = np.isfinite(R_kept) & np.isfinite(R_all) & (n_all > 0)
    tr = pf.read_table(CONSTCAT, columns=["case", "input_index", "r_input_p", "Re_input_p"],
                       memory_map=True).to_pandas().drop_duplicates(KEY)
    m2 = tr["r_input_p"].to_numpy(float)
    s2 = tr["Re_input_p"].to_numpy(float)
    k = (m2 < args.true_mag_max) & (s2 > args.true_re_min)
    c2 = ((np.clip(np.digitize(m2[k], ef) - 1, 0, nf - 1)) * ns
          + np.clip(np.digitize(s2[k], es) - 1, 0, ns - 1))
    w = np.bincount(c2, minlength=ncell).astype(float)
    w = np.where(fin, w, 0.0)
    w = w / max(w.sum(), 1)

    Rk = float(np.nansum(w * R_kept))
    Ra = float(np.nansum(w * R_all))
    shift = Ra - Rk
    print(f"\n--- (4) SHIFT: what adding the missing rows does to the target ---")
    print(f"  target as built (kept rows only), deliverable weights : {Rk:.4f}")
    print(f"  target with the missing rows added                    : {Ra:.4f}")
    print(f"  => the target MOVES by {shift:+.4f} in absolute response units")
    print(f"\n  The measured target defect (B) = +3.21% of m needs the target to move by about")
    print(f"  +0.027 (it currently sits ~0.027 BELOW <r_sim> - <R_blend>).")
    frac = 100 * shift / 0.027 if shift else 0.0
    print(f"  This fix supplies {shift:+.4f}, i.e. {frac:+.0f}% of what is needed, sign "
          f"{'MATCHING' if shift > 0 else 'OPPOSITE'}.")
    if shift > 0:
        print("  => Adding the rows pushes the target the RIGHT way. Rebuild the target and retrain.")
    else:
        print("  => Adding the rows pushes the target the WRONG way; the omission is not (B)'s cause.")


if __name__ == "__main__":
    main()
