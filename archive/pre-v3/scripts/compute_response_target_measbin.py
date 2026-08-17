"""REALISATION-MODULATION target rho(cell, measured sub-bin) for the realisation-aware (RA) head.

WHAT IT IS.  Inside a coarse TRUE-property cell, the half-shear sim's shear response is split by the
galaxy's MEASURED realisation and expressed as a SCALE-FREE ratio

    rho(cell, b) = <R_sim>_b / <R_sim>_cell ,      R_sim(i) = ((e_g - e_0) . ghat) / g

so the target constrains only the SHAPE of the response across measured sub-bins, never its level.
The level is already owned by the per-cell response pin (compute_response_target_blend.py) and the
two terms must not fight: rho is invariant under any rescaling of the cell, by construction.

THE SUB-BIN LABEL COMES FROM THE g=0 LEG.  `b` is a 2-D quantile bin of the g=0 leg's measured
magnitude and log flux-radius, each as a residual about the cell's own median.  Two reasons, and the
second is the decisive one:
  * a sheared-leg label would fold the migration/boundary term into the VALUE target and
    double-count against the model's own re-selection;
  * the flow's training catalogue is g=0 ONLY, so with g=0-leg binning each training row's own
    measured mag/size IS a valid draw of the label axis. With sheared-leg binning there is no
    corresponding quantity in the training data at all.
`--bin-on gsmeas` builds the sheared-leg variant for comparison only; do not train on it.

HARD NESTING CONSTRAINT.  The coarse cells MUST be a strict MERGE of the per-cell response target's
grid (`--fine-target-npz`), never an independent re-quantiling: two response terms binned on
different partitions of the same axis pull against each other and degrade the level R_flow depends
on.  The merge factors are the only freedom; `cell_group_map` is emitted so the trainer can validate
the nesting and raise.

FIREWALL: the half-shear g=0.05 leg + its g=0 SNC lookup. No constgold, nothing trained, no m.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    add_measurement_target_features, raw_columns_for_measurement_targets)
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402


def _selection_cuts(args):
    """DEFAULT_SELECTION_CUTS with the PRIMARY true-property domain optionally narrowed.

    Identical construction to compute_response_target_blend.py so the RA target lives on exactly
    the population the fine target and the trainer do.
    """
    cuts = [list(c) for c in DEFAULT_SELECTION_CUTS]
    if args.primary_mag_max is not None:
        cuts[1][1] = float(args.primary_mag_max)
    if args.primary_re_min is not None:
        cuts[3][0] = float(args.primary_re_min)
    return cuts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True, help="sheared half-shear leg (g=0.05)")
    ap.add_argument("--fine-target-npz", required=True,
                    help="the per-cell response target whose grid the coarse cells MERGE "
                         "(e.g. response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz)")
    ap.add_argument("--snc-lookup", required=True,
                    help="g=0 per-(case,input_index) lookup carrying BOTH the g0 ngmix shape and "
                         "the g0 MEASURED mag/flux-radius (build_g0_lookup.py --extra-cols)")
    ap.add_argument("--snc-cols", nargs=2, default=["ngmix0_g1", "ngmix0_g2"])
    ap.add_argument("--snc-meas-cols", nargs=2,
                    default=["measured_mag_auto_g0", "measured_flux_radius_g0"],
                    help="the g0-leg measured mag and flux radius columns in --snc-lookup")
    ap.add_argument("--target-cols", nargs=2,
                    default=["measured_ngmix_g1", "measured_ngmix_g2"])
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--merge-flux", type=int, default=2)
    ap.add_argument("--merge-size", type=int, default=2)
    ap.add_argument("--merge-crowd", type=int, default=5,
                    help="merge factors applied to the FINE grid's three axes to make the coarse "
                         "cells. Must divide evenly enough to keep every coarse cell populated; "
                         "the nesting is a strict merge whatever the factors are.")
    ap.add_argument("--n-dmag", type=int, default=5)
    ap.add_argument("--n-dlogsize", type=int, default=2)
    ap.add_argument("--bin-on", choices=["g0meas", "gsmeas"], default="g0meas",
                    help="which leg supplies the sub-bin label. TRAIN ONLY ON g0meas.")
    ap.add_argument("--null-rotate", action="store_true",
                    help="project on the spin-2 ORTHOGONAL direction instead. Produces a NULL "
                         "target (rho should be noise) -- a control artifact, never a train target.")
    ap.add_argument("--min-case", type=int, default=None)
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--primary-mag-max", type=float, default=None)
    ap.add_argument("--primary-re-min", type=float, default=None)
    ap.add_argument("--min-count", type=float, default=500.0,
                    help="effective (unique-target) count below which a (cell, sub-bin) is marked "
                         "unusable: its rho is written NaN and its weight zero, so the trainer "
                         "skips it rather than chasing noise.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # ---------------- the fine grid we must MERGE ------------------------------------------------
    ft = np.load(args.fine_target_npz)
    ef, es = np.asarray(ft["edges_flux"]), np.asarray(ft["edges_size"])
    ccol = ft["crowd_col"].item() if "crowd_col" in ft.files else ""
    if not ccol:
        raise SystemExit("--fine-target-npz must be a crowd-binned target (crowd_col set)")
    ec = np.asarray(ft["edges_crowd"])
    nf, ns, nc = np.asarray(ft["Rsim"]).shape
    mgf, mgs, mgc = max(args.merge_flux, 1), max(args.merge_size, 1), max(args.merge_crowd, 1)
    cf, cs, cc = -(-nf // mgf), -(-ns // mgs), -(-nc // mgc)
    n_cells = cf * cs * cc
    fine_ids = np.arange(nf * ns * nc)
    fi = fine_ids // (ns * nc)
    si = (fine_ids // nc) % ns
    ci = fine_ids % nc
    cell_group_map = ((fi // mgf) * cs + (si // mgs)) * cc + (ci // mgc)
    print(f"fine grid {nf}x{ns}x{nc} (crowd_col={ccol}) -> coarse {cf}x{cs}x{cc} = {n_cells} cells "
          f"(merge {mgf}/{mgs}/{mgc})")

    # ---------------- SNC lookup (g0 shape + g0 measured mag/size) --------------------------------
    need_lk = ["case", "input_index", *args.snc_cols, *args.snc_meas_cols]
    lk = pf.read_table(args.snc_lookup, columns=need_lk).to_pandas()
    lkey = lk["case"].to_numpy(np.int64) * 1_000_003 + lk["input_index"].to_numpy(np.int64)
    order = np.argsort(lkey)
    lkey = lkey[order]
    l_e1 = lk[args.snc_cols[0]].to_numpy(float)[order]
    l_e2 = lk[args.snc_cols[1]].to_numpy(float)[order]
    l_mag = lk[args.snc_meas_cols[0]].to_numpy(float)[order]
    l_rad = lk[args.snc_meas_cols[1]].to_numpy(float)[order]
    print(f"SNC lookup {args.snc_lookup}: rows={len(lk):,} cases={lk['case'].nunique()}")

    # ---------------- stream the sheared leg -------------------------------------------------------
    need = set(raw_columns_for_measurement_targets(args.target_cols))
    need |= {"gamma1_input_p", "gamma2_input_p", "detected", "case", "input_index",
             "r_input_p", "Re_input_p", "neighbored", "distance", ccol,
             "measured_mag_auto", "measured_flux_radius"}
    parts = []
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names)
        missing = sorted(c for c in need if c not in avail)
        if missing:
            raise SystemExit(f"catalogue lacks columns {missing}")
        cols = sorted(need)
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            if args.max_case is not None:
                if int(b["case"].min()) > args.max_case:
                    break
                b = b[b["case"] <= args.max_case]
            if args.min_case is not None:
                b = b[b["case"] >= args.min_case]
            if len(b) == 0:
                continue
            b = source_select_selection(b, cuts=_selection_cuts(args))
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            if len(b):
                parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"rows after selection+detected: {len(df):,} over cases "
          f"{int(df['case'].min())}..{int(df['case'].max())}")

    g1 = df["gamma1_input_p"].to_numpy(float)
    g2 = df["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2)
    keep = gm > 1e-6
    df = df[keep].reset_index(drop=True)
    g1, g2, gm = g1[keep], g2[keep], gm[keep]
    gh1, gh2 = g1 / gm, g2 / gm
    if args.null_rotate:                                  # spin-2 orthogonal control direction
        gh1, gh2 = -gh2, gh1
        print("NULL MODE: projecting on the spin-2 ORTHOGONAL direction (control artifact only)")

    # per-(case,target) de-duplication weight (all-pairs safe; 1.0 for nearest-pair catalogues)
    kk = df["case"].to_numpy(np.int64) * 1_000_003 + df["input_index"].to_numpy(np.int64)
    _, inv, npair = np.unique(kk, return_inverse=True, return_counts=True)
    w = (1.0 / npair[inv]).astype(float)

    meas = add_measurement_target_features(df.copy())
    e1 = meas[args.target_cols[0]].to_numpy(float)
    e2 = meas[args.target_cols[1]].to_numpy(float)
    pos = np.clip(np.searchsorted(lkey, kk), 0, len(lkey) - 1)
    match = lkey[pos] == kk
    print(f"SNC match after selection: {match.mean():.2%} ({int(match.sum()):,}/{len(match):,})")
    if match.mean() < 0.5:
        raise SystemExit("SNC match fraction below 50% -- wrong lookup or wrong case range")
    e1 = e1 - np.where(match, l_e1[pos], np.nan)
    e2 = e2 - np.where(match, l_e2[pos], np.nan)
    R = (e1 * gh1 + e2 * gh2) / args.nominal_g

    # ---------------- cells + measured sub-bins ----------------------------------------------------
    flux = df["r_input_p"].to_numpy(float)
    size = df["Re_input_p"].to_numpy(float)
    crowd = df[ccol].to_numpy(float)
    fi_r = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
    si_r = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
    ci_r = np.clip(np.digitize(crowd, ec) - 1, 0, nc - 1)
    fine = (fi_r * ns + si_r) * nc + ci_r
    cell = cell_group_map[fine]

    if args.bin_on == "g0meas":
        b_mag = np.where(match, l_mag[pos], np.nan)
        b_rad = np.where(match, l_rad[pos], np.nan)
    else:
        b_mag = df["measured_mag_auto"].to_numpy(float)
        b_rad = df["measured_flux_radius"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        b_lsz = np.log(np.where(b_rad > 0, b_rad, np.nan))

    fin = np.isfinite(R) & np.isfinite(b_mag) & np.isfinite(b_lsz) & np.isfinite(crowd)
    print(f"rows with a finite response AND a finite {args.bin_on} label: "
          f"{int(fin.sum()):,} ({fin.mean():.2%})")

    med_mag = np.full(n_cells, np.nan)
    med_lsz = np.full(n_cells, np.nan)
    for c in range(n_cells):
        m = fin & (cell == c)
        if m.any():
            med_mag[c] = np.median(b_mag[m])
            med_lsz[c] = np.median(b_lsz[m])
    dmag = b_mag - med_mag[cell]
    dlsz = b_lsz - med_lsz[cell]

    def qedges(v, n):
        e = np.quantile(v[fin & np.isfinite(v)], np.linspace(0.0, 1.0, n + 1))
        e[0] -= 1e-9
        e[-1] += 1e-9
        return e

    edm = qedges(dmag, args.n_dmag)
    eds = qedges(dlsz, args.n_dlogsize)
    mi = np.clip(np.digitize(np.where(fin, dmag, 0.0), edm) - 1, 0, args.n_dmag - 1)
    szi = np.clip(np.digitize(np.where(fin, dlsz, 0.0), eds) - 1, 0, args.n_dlogsize - 1)
    n_mb = args.n_dmag * args.n_dlogsize
    sub = mi * args.n_dlogsize + szi

    # ---------------- rho, counts, sem --------------------------------------------------------------
    flat = (cell * n_mb + sub)[fin]
    ww = w[fin]
    RR = R[fin]
    nbin = n_cells * n_mb
    S = np.bincount(flat, weights=ww * RR, minlength=nbin).reshape(n_cells, n_mb)
    W = np.bincount(flat, weights=ww, minlength=nbin).reshape(n_cells, n_mb)
    Q = np.bincount(flat, weights=ww * RR * RR, minlength=nbin).reshape(n_cells, n_mb)
    W2 = np.bincount(flat, weights=ww * ww, minlength=nbin).reshape(n_cells, n_mb)
    RAW = np.bincount(flat, minlength=nbin).reshape(n_cells, n_mb)

    usable = W >= args.min_count
    with np.errstate(invalid="ignore", divide="ignore"):
        mb = np.where(usable, S / np.where(W > 0, W, np.nan), np.nan)
        var_b = np.maximum(Q / np.where(W > 0, W, np.nan) - (S / np.where(W > 0, W, np.nan)) ** 2, 0.0)
        # variance of a WEIGHTED mean uses the effective sample size (Sum w)^2 / Sum w^2, which
        # equals the row count only when every weight is 1 (nearest-pair). For an all-pairs
        # catalogue the 1/n_pairs weights make n_eff strictly smaller and the plain count would
        # understate the error.
        n_eff = W ** 2 / np.where(W2 > 0, W2, np.nan)
        var_mb = var_b / np.where(n_eff > 0, n_eff, np.nan)
        Wc = W.sum(axis=1)
        mc = S.sum(axis=1) / np.where(Wc > 0, Wc, np.nan)
        f = W / np.where(Wc[:, None] > 0, Wc[:, None], np.nan)
        var_mc = np.nansum(f ** 2 * var_mb, axis=1)
        rho = mb / mc[:, None]
        var_rho = (var_mb / mc[:, None] ** 2
                   + mb ** 2 * var_mc[:, None] / mc[:, None] ** 4
                   - 2.0 * mb * f * var_mb / mc[:, None] ** 3)
        sem = np.sqrt(np.maximum(var_rho, 0.0))
    rho = np.where(usable & np.isfinite(rho), rho, np.nan)
    counts = np.where(usable, W, 0.0)

    print(f"\nrho: {int(np.isfinite(rho).sum())}/{rho.size} usable (cell, sub-bin) pairs "
          f"(min effective count {args.min_count:g})")
    print(f"     range {np.nanmin(rho):.4f}..{np.nanmax(rho):.4f}; "
          f"median sem {np.nanmedian(sem):.4f}")
    okm = np.isfinite(rho)
    cw = np.where(okm, counts, 0.0)
    colw = np.maximum(cw.sum(axis=0), 1e-12)
    agg = np.nansum(np.where(okm, rho, 0.0) * cw, axis=0) / colw
    print("     count-weighted rho per sub-bin (dmag idx, dlogsize idx):")
    for b in range(n_mb):
        print(f"       b{b:02d} (dmag {b//args.n_dlogsize}, dlsz {b%args.n_dlogsize}): "
              f"rho={agg[b]:.4f}  N_eff={cw[:, b].sum():,.0f}  N_raw={RAW[:, b].sum():,}")
    print(f"     flat-response (rho==1) reference loss = "
          f"{float(np.nansum((1.0 - np.where(okm, rho, 1.0)) ** 2 * cw) / max(cw.sum(), 1e-12)):.4e}")

    np.savez(args.output,
             rho=rho, Rsim_ra=mb, Rsim_cell=mc, counts=counts, raw_counts=RAW, sem=sem,
             edges_flux=ef, edges_size=es, edges_crowd=ec, crowd_col=ccol,
             cell_group_map=cell_group_map.astype(np.int64),
             merge_flux=np.int64(mgf), merge_size=np.int64(mgs), merge_crowd=np.int64(mgc),
             coarse_shape=np.array([cf, cs, cc], dtype=np.int64),
             edges_dmag=edm, edges_dlogsize=eds,
             cell_median_mag=med_mag, cell_median_logsize=med_lsz,
             n_dmag=np.int64(args.n_dmag), n_dlogsize=np.int64(args.n_dlogsize),
             bin_on=args.bin_on, nominal_g=float(args.nominal_g),
             null_rotate=bool(args.null_rotate),
             source_catalogue=args.catalogue, fine_target=args.fine_target_npz,
             snc_lookup=args.snc_lookup,
             case_range=np.array([-1 if args.min_case is None else args.min_case,
                                  -1 if args.max_case is None else args.max_case], dtype=np.int64),
             primary_mag_max=(np.nan if args.primary_mag_max is None else args.primary_mag_max),
             primary_re_min=(np.nan if args.primary_re_min is None else args.primary_re_min),
             min_count=float(args.min_count))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
