"""STEP-0 GO/NO-GO for the joint measured+detection flow (#35): does a SHAPE-CORRELATED
(anisotropic) DETECTION selection exist in the regimes where the oracle residual is stuck?

The certified estimator models the intrinsic shape response R_flow but NOT any shear-dependent
SELECTION. For a detection head to break the true-cut TARGET ceiling (oracle ~4% median, worst
8-12% in faint x blend), the DETECTED population's mean INTRINSIC shape must respond to shear:

    dS/dg (cell) = [ <e1_input_rot0_p>_detected(g=0.05) - <e1_input_rot0_p>_detected(g=0.0) ] / g

e1_input_rot0_p is the pre-shear INTRINSIC ellipticity -> it has NO intrinsic shear response, so
any nonzero dS/dg is PURE selection (detection preferentially keeping intrinsic orientations as
shear moves the scene). Rot-pair shape-noise cancellation makes <e1_rot0>_parent ~ 0; detection
can break that cancellation. Compare dS/dg to the oracle residual it would have to explain:
a faint-cell residual m~+7% with (R_flow+R_blend)~0.3 needs dS/dg ~ 0.075*0.3 ~ 0.02.

DECISION: dS/dg ~ 0.02 in the stuck cells  => selection is real -> a detection head CAN help -> BUILD.
          dS/dg ~ 0 (<< the needed 0.02)   => residual is NOT selection -> #35 inherits the ceiling -> STOP.

DIAGNOSTIC ONLY. constgold is never read; the certified m is never touched. Firewall-safe.
Fork of derisk_detection_response.py (kept intact).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402

CATDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"
DEF_G00 = f"{CATDIR}/det_meas_ngmix_g0.0_train.feather"
DEF_G05 = f"{CATDIR}/det_meas_ngmix_g0.05_val.feather"
READ_COLS = ["detected", "r_input_p", "Re_input_p", "distance", "neighbored",
             "gamma1_input_p", "gamma2_input_p", "case",
             "e1_input_rot0_p", "e2_input_rot0_p"]
t0 = time.time()


def stream(path, max_batches=None):
    keep = {c: [] for c in READ_COLS}
    raw = 0
    with ipc.open_file(path) as r:
        names = set(r.schema.names)
        cols = [c for c in READ_COLS if c in names]
        nb = r.num_record_batches if max_batches is None else min(max_batches, r.num_record_batches)
        for bi in range(nb):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            for c in cols:
                keep[c].append(b[c].to_numpy())
    out = {c: (np.concatenate(keep[c]) if keep[c] else np.array([])) for c in cols}
    out["detected"] = out["detected"].astype(bool)
    out["neighbored"] = out["neighbored"].astype(bool)
    # SHEAR-PROJECTED intrinsic ellipticity e_par = (e1*g1 + e2*g2)/|g|  (spin-2 projection onto the
    # per-object shear direction). The shear direction VARIES per case (median g1=g2=0, |g|=0.05), so
    # a raw e1 average cancels the anisotropic-selection signal; the projection preserves it.
    g1 = out["gamma1_input_p"]; g2 = out["gamma2_input_p"]
    gmag = np.hypot(g1, g2)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["e_par"] = np.where(gmag > 1e-6,
                                (out["e1_input_rot0_p"] * g1 + out["e2_input_rot0_p"] * g2) / gmag,
                                np.nan)
    print(f"  {os.path.basename(path)}: raw={raw:,} selected={len(out['detected']):,} "
          f"det_frac={out['detected'].mean():.4f} |g|median={np.nanmedian(gmag):.4f} in {time.time()-t0:.0f}s", flush=True)
    return out


def per_case_shape_sums(e, mask, case, bin_idx, n_bins):
    """Per-(case,bin): sum of e and count, over rows where (mask & finite(e))."""
    w = (mask & np.isfinite(e)).astype(np.float64)
    ez = np.where(np.isfinite(e), e, 0.0)
    ucases, inv = np.unique(case, return_inverse=True)
    nC = len(ucases)
    flat = inv * n_bins + bin_idx
    se = np.bincount(flat, weights=ez * w, minlength=nC * n_bins).reshape(nC, n_bins)
    nd = np.bincount(flat, weights=w, minlength=nC * n_bins).reshape(nC, n_bins)
    return se, nd, ucases


def boot_mean(se, nd, picks):
    with np.errstate(invalid="ignore", divide="ignore"):
        point = se.sum(0) / nd.sum(0)
        vals = np.empty((picks.shape[0], se.shape[1]))
        for b in range(picks.shape[0]):
            idx = picks[b]
            vals[b] = se[idx].sum(0) / nd[idx].sum(0)
    return point, np.nanstd(vals, axis=0)


def sel_response(B, bi, n_bins, g, n_boot, seed, ecol="e_par"):
    """Single-leg (sheared) selection signal: dS/dg = <e_par>_DETECTED / g, with <e_par>_PARENT
    (all rows) as the null control (rot-pairs cancel -> ~0). Case-cluster bootstrap."""
    seD, ndD, ucB = per_case_shape_sums(B[ecol], B["detected"], B["case"], bi, n_bins)
    seP, ndP, _ = per_case_shape_sums(B[ecol], np.ones(len(B["detected"]), bool), B["case"], bi, n_bins)
    rng = np.random.default_rng(seed)
    p = rng.integers(0, len(ucB), size=(n_boot, len(ucB)))
    mD, sD = boot_mean(seD, ndD, p)
    mP, sP = boot_mean(seP, ndP, p)
    dS = mD / g
    sig = sD / g
    return dS, sig, mD, mP, sP / g, ndD.sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g00", default=DEF_G00)
    ap.add_argument("--g05", default=DEF_G05)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--n-mag", type=int, default=10)
    ap.add_argument("--n-flux", type=int, default=4)
    ap.add_argument("--n-size", type=int, default=2)
    ap.add_argument("--n-dist", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out-prefix",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anisosel_ngmix")
    args = ap.parse_args()
    g = args.nominal_g
    lines = []

    def emit(s=""):
        print(s, flush=True); lines.append(s)

    emit("=" * 92)
    emit("STEP-0 ANISOTROPIC (shape-correlated) DETECTION SELECTION  dS/dg = d<e1_input_rot0_p>_detected/dg")
    emit("e1_input_rot0_p is INTRINSIC (pre-shear) -> any nonzero dS/dg is PURE selection.")
    emit("needed to explain a faint-cell oracle residual m~+7% at (R_flow+R_blend)~0.3:  dS/dg ~ 0.02")
    emit("=" * 92)

    emit(f"\n[load] g=0.0  : {args.g00}")
    A = stream(args.g00, args.max_batches)
    emit(f"[load] g=0.05 : {args.g05}")
    B = stream(args.g05, args.max_batches)

    fa, fb = float(A["detected"].mean()), float(B["detected"].mean())
    ga = float(np.nanmedian(np.hypot(A["gamma1_input_p"], A["gamma2_input_p"])))
    gb = float(np.nanmedian(np.hypot(B["gamma1_input_p"], B["gamma2_input_p"])))
    g1b = float(np.nanmedian(B["gamma1_input_p"])); g2b = float(np.nanmedian(B["gamma2_input_p"]))
    emit(f"\n[HARD CHECK] det_frac g0={fa:.4f} g05={fb:.4f} ; median|gamma| g0={ga:.4f} g05={gb:.4f} "
         f"(g05: gamma1={g1b:+.4f} gamma2={g2b:+.4f})")
    assert 0.05 < fa < 0.95 and 0.05 < fb < 0.95, "a leg is det_frac~1 (not a parent) -- STOP"
    assert ga < 1e-3 and gb > 1e-3, "shear legs look swapped/wrong -- STOP"
    emit("    signal = <e_par>_DETECTED / g on the g=0.05 leg (e_par = (e1*g1+e2*g2)/|g|, per-case shear")
    emit("    direction; parent <e_par> ~ 0 by rot-pair cancellation).  needed <e_par>_det ~ 0.001 (=0.02*g).")

    B["e1_input_rot0_p"]  # ensure present
    # --- edges from g=0.05 leg (the signal leg) ---
    mag0 = B["r_input_p"]; size0 = B["Re_input_p"]; dist0 = B["distance"]; nb0 = B["neighbored"]
    ef_mag = np.quantile(mag0, np.linspace(0, 1, args.n_mag + 1)); ef_mag[0] -= 1e-4; ef_mag[-1] += 1e-4
    ef = np.quantile(mag0, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-4; ef[-1] += 1e-4
    es = np.quantile(size0, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-4; es[-1] += 1e-4
    db = dist0[nb0 & np.isfinite(dist0)]
    ed = np.quantile(db, np.linspace(0, 1, args.n_dist + 1)); ed[0] -= 1e-4; ed[-1] += 1e-4

    def mag_bins(D):
        return np.clip(np.digitize(D["r_input_p"], ef_mag) - 1, 0, args.n_mag - 1)

    def grid_bins(D):
        fi = np.clip(np.digitize(D["r_input_p"], ef) - 1, 0, args.n_flux - 1)
        si = np.clip(np.digitize(D["Re_input_p"], es) - 1, 0, args.n_size - 1)
        di = np.where(D["neighbored"], 1 + np.clip(np.digitize(D["distance"], ed) - 1, 0, args.n_dist - 1), 0)
        nblend = args.n_dist + 1
        return (fi * args.n_size + si) * nblend + di, nblend

    # GLOBAL: <e_par>_detected/g (signal) with <e_par>_parent/g control
    zB = np.zeros(len(B["detected"]), int)
    gdS, gsig, gmD, gmP, gpsig, _ = sel_response(B, zB, 1, g, args.n_boot, args.seed)
    emit("\n" + "-" * 92)
    emit(f"GLOBAL  dS/dg(DETECTED) = {gdS[0]:+.5f} +/- {gsig[0]:.5f}   "
         f"(<e_par>_det={gmD[0]:+.6f}, {gdS[0]/gsig[0] if gsig[0] else float('nan'):+.1f} sigma; needed <e_par>~0.001)")
    emit(f"GLOBAL  parent <e_par>/g control = {gmP[0]/g:+.5f} +/- {gpsig[0]:.5f}   (should be ~0)")

    # PER-MAG
    mdS, msig, mmD, mmP, mpsig, mN = sel_response(B, mag_bins(B), args.n_mag, g, args.n_boot, args.seed)
    mag_centers = 0.5 * (ef_mag[:-1] + ef_mag[1:])
    emit("\n" + "-" * 92)
    emit("PER-MAG dS/dg curve (deciles of true r_input_p, g=0.05 leg)   [needed ~0.02; |>2sig| flagged *]")
    emit(f"  {'bin':>3} {'r_mag':>7} {'Ndet':>13} {'<epar>_det':>11} {'<epar>_par':>11} {'dS/dg':>9} {'sig':>8} {'nsig':>6}")
    for k in range(args.n_mag):
        ns = mdS[k] / msig[k] if msig[k] > 0 else np.nan
        flag = "*" if abs(ns) > 2 else " "
        emit(f"  {k:>3} {mag_centers[k]:>7.2f} {int(mN[k]):>13,} {mmD[k]:>+11.6f} {mmP[k]:>+11.6f} "
             f"{mdS[k]:>+9.5f} {msig[k]:>8.5f} {ns:>+5.1f}{flag}")

    # GRID
    giB, nblend = grid_bins(B)
    n_cells = args.n_flux * args.n_size * nblend
    cdS, csig, _, _, _, cN = sel_response(B, giB, n_cells, g, args.n_boot, args.seed)
    cdS = cdS.reshape(args.n_flux, args.n_size, nblend); csig = csig.reshape(args.n_flux, args.n_size, nblend)
    cN = cN.reshape(args.n_flux, args.n_size, nblend)
    emit("\n" + "-" * 92)
    emit(f"(mag x size x blend) dS/dg +/- sig  (blend0=ISOLATED; faint=high mag bin; needed ~0.02)")
    for a in range(args.n_flux):
        for s in range(args.n_size):
            row = []
            for c in range(nblend):
                if cN[a, s, c] > 2000:
                    row.append(f"{cdS[a,s,c]:+.4f}+/-{csig[a,s,c]:.4f}")
                else:
                    row.append("     (thin)    ")
            emit(f"  mag{a} size{s}: " + " | ".join(row))

    emit("\n" + "=" * 92)
    emit("READING: dS/dg >> its sigma AND ~0.02 in the faint/blend cells => shape-correlated selection is")
    emit("   real and large enough to explain the oracle residual => a detection head CAN break the ceiling.")
    emit("   dS/dg ~ 0 (<< 0.02) everywhere => the residual is NOT selection => #35 inherits the ceiling.")

    txt = args.out_prefix + ".txt"; npz = args.out_prefix + ".npz"
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(npz, nominal_g=g, ecol="e_par", global_dS=gdS[0], global_sig=gsig[0], parent_dS=gmP[0]/g,
             mag_centers=mag_centers, mag_dS=mdS, mag_sig=msig, mag_N=mN,
             grid_dS=cdS, grid_sig=csig, grid_N=cN,
             grid_edges_flux=ef, grid_edges_size=es, grid_edges_dist=ed,
             n_flux=args.n_flux, n_size=args.n_size, n_dist=args.n_dist, n_blend=nblend)
    emit(f"\nwrote {txt}\nwrote {npz}\nDIAG_ANISOSEL_DONE")


if __name__ == "__main__":
    main()
