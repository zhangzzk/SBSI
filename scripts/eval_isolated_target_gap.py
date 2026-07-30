"""Quantify RESULT 10: how wrong is a NEIGHBOURED-ONLY response target for a 24%-isolated population?

RESULT 10 established that the pin's target catalogue (`det_meas_crowd_g0.05_val_full`) is a strict
22.4% subset of the ruler's (`det_meas_ngmix_g0.05_val`) with bit-identical measurements, and that the
subset is **99.97% neighboured** while the deliverable population is **24.1% isolated**. So the pin has
never been supervised on an isolated galaxy.

This measures the consequence directly, on the ONE catalogue that contains both classes. In matched
(true mag x true size) cells, on detected rows inside the deliverable domain:

    R_nbr  = <e.ghat>/|g| for neighboured rows      (what the current target measures)
    R_iso  = <e.ghat>/|g| for isolated rows         (the class the target never sees)
    R_all  = both together                          (what the target SHOULD measure)

and then the population-weighted target error a neighboured-only label carries:

    bias = sum_cell w_cell * (R_nbr,cell - R_all,cell)

with `w_cell` the DELIVERABLE population's cell occupancy. If that lands near the measured
(B) = +3.21% of m (i.e. about -0.027 in absolute response units, target LOW), RESULT 10 is not just a
plausible mechanism but the quantitatively dominant one.

Sign bookkeeping: (B) says the target sits BELOW `r_sim - R_blend`. A neighboured-only target would be
low if neighboured galaxies respond LESS than isolated ones at the same true properties, which is the
physically expected direction (blending dilutes the shear response).

Raw (non-SNC) response is used because the SNC g=0 lookup is keyed to the other catalogue; raw response
is noisier per object but this is a systematic offset over ~10^5-10^6 rows per class, and both classes
are treated identically so the comparison is fair.

FIREWALL: half-shear catalogue only. No constgold, no r_sim, no m. Trains nothing.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.feather as pf

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"
CONSTCAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
            "constant_response_catalogue_train.feather")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler-cat", default=f"{CAT}/det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=9)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--nmag", type=int, default=6)
    ap.add_argument("--nsize", type=int, default=6)
    ap.add_argument("--target",
                    default="/home/z/Zekang.Zhang/SBSI/results/"
                            "response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz",
                    help="only its edges are used, so cells match the pin's own grid")
    args = ap.parse_args()

    need = ["case", "input_index", "r_input_p", "Re_input_p", "neighbored", "distance",
            "detected", "measured_ngmix_g1", "measured_ngmix_g2",
            "gamma1_input_p", "gamma2_input_p"]
    d = ds.dataset(args.ruler_cat, format="feather")
    have = [c for c in need if c in d.schema.names]
    miss = [c for c in need if c not in d.schema.names]
    if miss:
        print(f"MISSING columns in the ruler catalogue: {miss}")
    t = d.to_table(columns=have, filter=(ds.field("case") <= args.max_case)).to_pandas()
    print(f"ruler cases 0..{args.max_case}: {len(t):,} rows")

    t = t.drop_duplicates(["case", "input_index"])
    t = t[t["detected"].astype(bool)]                     # the target builder keeps detected only
    print(f"  after dedupe + detected: {len(t):,}")

    g1 = t["gamma1_input_p"].to_numpy(float)
    g2 = t["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    ok = gmag > 1e-6
    t, g1, g2, gmag = t[ok], g1[ok], g2[ok], gmag[ok]
    gh1, gh2 = g1 / gmag, g2 / gmag
    R = (t["measured_ngmix_g1"].to_numpy(float) * gh1
         + t["measured_ngmix_g2"].to_numpy(float) * gh2) / gmag

    # Drop rows with a non-finite measured shape BEFORE binning. bincount(weights=...) propagates a
    # single NaN to the whole cell, which silently blanked most cells on the first run and left the
    # aggregate computed over an unrepresentative subset.
    good = np.isfinite(R)
    if not good.all():
        print(f"  dropping {int((~good).sum()):,} rows with non-finite measured shape "
              f"({100 * (~good).mean():.2f}%)")
        t, R = t[good], R[good]

    mag = t["r_input_p"].to_numpy(float)
    size = t["Re_input_p"].to_numpy(float)
    nbr = t["neighbored"].astype(bool).to_numpy()
    dom = (mag < args.true_mag_max) & (size > args.true_re_min)
    print(f"  in the deliverable domain: {dom.sum():,}  "
          f"({100 * nbr[dom].mean():.2f}% neighboured, {100 * (~nbr[dom]).mean():.2f}% isolated)")

    z = np.load(args.target, allow_pickle=True)
    ef, es = z["edges_flux"], z["edges_size"]
    nf, ns = len(ef) - 1, len(es) - 1
    fi = np.clip(np.digitize(mag, ef) - 1, 0, nf - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
    cid = (fi * ns + si).astype(np.int64)
    ncell = nf * ns

    def cellmean(mask):
        n = np.bincount(cid[mask], minlength=ncell).astype(float)
        s = np.bincount(cid[mask], weights=R[mask], minlength=ncell)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan), n

    D = dom
    R_nbr, n_nbr = cellmean(D & nbr)
    R_iso, n_iso = cellmean(D & ~nbr)
    R_all, n_all = cellmean(D)

    print("\n--- response by class, in the pin's own (mag,size) cells, deliverable domain ---")
    print(f"{'cell(mag,size)':>15} {'n_nbr':>9} {'n_iso':>8} {'R_nbr':>8} {'R_iso':>8} {'R_all':>8} "
          f"{'iso-nbr':>9} {'nbr-all':>9}")
    for b in range(ncell):
        if not (n_nbr[b] > 100 and n_iso[b] > 100):
            continue
        print(f"{f'({b // ns},{b % ns})':>15} {int(n_nbr[b]):>9,} {int(n_iso[b]):>8,} "
              f"{R_nbr[b]:>8.4f} {R_iso[b]:>8.4f} {R_all[b]:>8.4f} "
              f"{R_iso[b] - R_nbr[b]:>+9.4f} {R_nbr[b] - R_all[b]:>+9.4f}")

    # global, on this catalogue's own occupancy
    fin = np.isfinite(R_nbr) & np.isfinite(R_iso) & np.isfinite(R_all) & (n_all > 0)
    w_self = np.where(fin, n_all, 0.0)
    w_self = w_self / max(w_self.sum(), 1)
    print("\n--- aggregates over occupied cells (this catalogue's weights) ---")
    print(f"  R_neighboured-only (what the target measures) = {np.nansum(w_self * R_nbr):.4f}")
    print(f"  R_all              (what it should measure)   = {np.nansum(w_self * R_all):.4f}")
    print(f"  R_isolated                                    = {np.nansum(w_self * R_iso):.4f}")
    bias_self = float(np.nansum(w_self * (R_nbr - R_all)))
    print(f"  => a neighboured-only target is LOW by {bias_self:+.4f} in absolute response units "
          f"({100 * bias_self / max(np.nansum(w_self * R_all), 1e-9):+.2f}%)")

    # reweight to the DELIVERABLE population's (mag,size) occupancy
    tr = pf.read_table(CONSTCAT, columns=["case", "input_index", "r_input_p", "Re_input_p"],
                       memory_map=True).to_pandas().drop_duplicates(["case", "input_index"])
    m2 = tr["r_input_p"].to_numpy(float)
    s2 = tr["Re_input_p"].to_numpy(float)
    k = (m2 < args.true_mag_max) & (s2 > args.true_re_min)
    c2 = ((np.clip(np.digitize(m2[k], ef) - 1, 0, nf - 1)) * ns
          + np.clip(np.digitize(s2[k], es) - 1, 0, ns - 1))
    w_dev = np.bincount(c2, minlength=ncell).astype(float)
    w_dev = np.where(fin, w_dev, 0.0)
    w_dev = w_dev / max(w_dev.sum(), 1)
    bias_dev = float(np.nansum(w_dev * (R_nbr - R_all)))
    Rall_dev = float(np.nansum(w_dev * R_all))
    print("\n--- reweighted to the DELIVERABLE population's (mag,size) occupancy ---")
    print(f"  R_all = {Rall_dev:.4f}")
    print(f"  a neighboured-only target is LOW by {bias_dev:+.4f} "
          f"({100 * bias_dev / max(Rall_dev, 1e-9):+.2f}%)")
    print("\n=== COMPARE WITH THE MEASURED TARGET DEFECT ===")
    print("  (B) measured on constgold = +3.21% of m, i.e. the target sits about -0.027 BELOW")
    print("  <r_sim> - <R_blend> in absolute response units (flat across crowd bins).")
    print(f"  This mechanism supplies {bias_dev:+.4f}.")
    frac = abs(bias_dev) / 0.027 * 100 if bias_dev else 0.0
    print(f"  => it accounts for roughly {frac:.0f}% of the required -0.027, with the sign "
          f"{'MATCHING' if bias_dev < 0 else 'OPPOSITE (mechanism does not work this way)'}.")


if __name__ == "__main__":
    main()
