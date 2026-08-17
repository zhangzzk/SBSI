"""DERISK (DIAGNOSTIC ONLY): shear DETECTION-RESPONSE truth target dP(detect)/dgamma measured
as a genuine two-leg FINITE DIFFERENCE on the CURRENT ngmix parents (det_meas_ngmix_g0.0_train,
det_meas_ngmix_g0.05_val), which retain undetected rows (det_frac~0.44).

This supersedes the deprecated-SExtractor shape-contrast label (derisk_btrue_detection.py) with
an assumption-free rate response:

    dP/dg (bin) = [ f_det(g=0.05) - f_det(g=0.0) ] / g ,    f_det = <detected>_parent(bin)

resolved per true-property bin -- a magnitude decile curve AND a (true-mag x true-size x blend)
grid with blend bin 0 = ISOLATED (neighbored==False), bins 1..n_dist = distance quantiles.
BLENDING is resolved on purpose: it dominates the detection response (shear moves blend/deblend
outcomes).  Edges are defined once on the g=0 leg and applied to both legs so bins are comparable.
Case-cluster bootstrap errors per leg, combined in quadrature for the difference.

Purpose (Goal 2): (a) the truth target for the joint flow's detection head; (b) the
finite-difference gradient check AGENTS.md requires before treating dP/dgamma as a science result
-- compare the head's analytic-shift dP/dg to THIS empirical curve, per bin.

IMPORTANT: DIAGNOSTIC. Its output is NEVER wired into the certified parameter-free
m = R_sim/(R_flow+R_blend)-1.
"""
import pathlib as _pathlib, sys as _sys  # noqa: E402  -- make `sbs_shear` importable
_sys.path.insert(0, str(next(p for p in _pathlib.Path(__file__).resolve().parents
                             if (p / 'sbs_shear').is_dir())))
from sbs_shear import paths  # noqa: E402
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
             "gamma1_input_p", "gamma2_input_p", "case"]


def stream(path, max_batches=None):
    """Stream minimal cols, source-select, KEEP the full parent (detected + undetected)."""
    keep = {c: [] for c in READ_COLS}
    raw = 0
    t0 = time.time()
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
            if (bi + 1) % 300 == 0:
                print(f"    [{os.path.basename(path)}] batch {bi+1}/{nb} raw={raw:,} "
                      f"kept={sum(len(x) for x in keep['detected']):,} ({time.time()-t0:.0f}s)", flush=True)
    out = {c: (np.concatenate(keep[c]) if keep[c] else np.array([])) for c in cols}
    out["detected"] = out["detected"].astype(bool)
    out["neighbored"] = out["neighbored"].astype(bool)
    print(f"  {os.path.basename(path)}: raw={raw:,} selected={len(out['detected']):,} "
          f"det_frac={out['detected'].mean():.4f} in {time.time()-t0:.0f}s", flush=True)
    return out


def per_case_counts(det, case, bin_idx, n_bins):
    """Per-(case,bin) parent count and detected count. Returns npar,(nC,nbins) ndet,(nC,nbins), ucases."""
    ucases, inv = np.unique(case, return_inverse=True)
    nC = len(ucases)
    flat = inv * n_bins + bin_idx
    npar = np.bincount(flat, minlength=nC * n_bins).astype(np.float64).reshape(nC, n_bins)
    ndet = np.bincount(flat, weights=det.astype(np.float64), minlength=nC * n_bins).reshape(nC, n_bins)
    return npar, ndet, ucases


def boot_fdet(npar, ndet, picks):
    """f_det(bin) point + case-bootstrap sigma. picks:(n_boot,nC). Returns (point(nbins), sig(nbins))."""
    with np.errstate(invalid="ignore", divide="ignore"):
        point = ndet.sum(0) / npar.sum(0)
        vals = np.empty((picks.shape[0], npar.shape[1]))
        for b in range(picks.shape[0]):
            idx = picks[b]
            vals[b] = ndet[idx].sum(0) / npar[idx].sum(0)
    return point, np.nanstd(vals, axis=0)


def response(A, B, bin_idx_A, bin_idx_B, n_bins, g, n_boot, seed):
    """dP/dg per bin = [f_det_B - f_det_A]/g with quadrature-combined case-bootstrap sigma.
    A = g=0 leg, B = g=0.05 leg (independent samples -> independent case resampling)."""
    nparA, ndetA, ucA = per_case_counts(A["detected"], A["case"], bin_idx_A, n_bins)
    nparB, ndetB, ucB = per_case_counts(B["detected"], B["case"], bin_idx_B, n_bins)
    rng = np.random.default_rng(seed)
    pA = rng.integers(0, len(ucA), size=(n_boot, len(ucA)))
    pB = rng.integers(0, len(ucB), size=(n_boot, len(ucB)))
    f0, s0 = boot_fdet(nparA, ndetA, pA)
    f5, s5 = boot_fdet(nparB, ndetB, pB)
    dP = (f5 - f0) / g
    sig = np.sqrt(s0**2 + s5**2) / g
    return dP, sig, f0, f5, nparA.sum(0) + nparB.sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g00", default=DEF_G00)
    ap.add_argument("--g05", default=DEF_G05)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--n-mag", type=int, default=10)
    ap.add_argument("--n-flux", type=int, default=4)
    ap.add_argument("--n-size", type=int, default=2)
    ap.add_argument("--n-dist", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out-prefix",
                    default=f"{paths.CACHE_DIR}/derisk/detresp_ngmix")
    args = ap.parse_args()
    g = args.nominal_g
    lines = []

    def emit(s=""):
        print(s, flush=True); lines.append(s)

    emit("=" * 84)
    emit("DERISK (DIAGNOSTIC ONLY) -- ngmix two-leg DETECTION RESPONSE dP/dgamma (finite difference)")
    emit("m = R_sim/(R_flow+R_blend)-1 stays parameter-free; this target is NEVER wired in.")
    emit("=" * 84)

    emit(f"\n[load] g=0.0  : {args.g00}")
    A = stream(args.g00, args.max_batches)
    emit(f"[load] g=0.05 : {args.g05}")
    B = stream(args.g05, args.max_batches)

    # HARD CHECK: real parents (undetected retained), correct shear legs
    fa, fb = float(A["detected"].mean()), float(B["detected"].mean())
    ga = float(np.nanmedian(np.hypot(A["gamma1_input_p"], A["gamma2_input_p"])))
    gb = float(np.nanmedian(np.hypot(B["gamma1_input_p"], B["gamma2_input_p"])))
    emit(f"\n[HARD CHECK] det_frac g0={fa:.4f} g05={fb:.4f} ; median|gamma| g0={ga:.4f} g05={gb:.4f}")
    assert 0.05 < fa < 0.95 and 0.05 < fb < 0.95, "a leg is det_frac~1 (not a parent) -- STOP"
    assert ga < 1e-3 and gb > 1e-3, "shear legs look swapped/wrong -- STOP"

    # --- edges from the g=0 leg (applied to both) ---
    mag0 = A["r_input_p"]; size0 = A["Re_input_p"]; dist0 = A["distance"]; nb0 = A["neighbored"]
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
        nb = D["neighbored"]; dist = D["distance"]
        di = np.where(nb, 1 + np.clip(np.digitize(dist, ed) - 1, 0, args.n_dist - 1), 0)
        nblend = args.n_dist + 1
        return (fi * args.n_size + si) * nblend + di, nblend

    # --- GLOBAL ---
    zA = np.zeros(len(A["detected"]), int); zB = np.zeros(len(B["detected"]), int)
    gdP, gsig, gf0, gf5, _ = response(A, B, zA, zB, 1, g, args.n_boot, args.seed)
    emit("\n" + "-" * 84)
    emit(f"GLOBAL dP/dg = {gdP[0]*100:+.3f}% +/- {gsig[0]*100:.3f}%   "
         f"(f_det: g0={gf0[0]:.4f} -> g05={gf5[0]:.4f}, {gdP[0]/gsig[0] if gsig[0] else float('nan'):+.1f} sigma)")

    # --- PER-MAG curve ---
    mdP, msig, mf0, mf5, mN = response(A, B, mag_bins(A), mag_bins(B), args.n_mag, g, args.n_boot, args.seed)
    mag_centers = 0.5 * (ef_mag[:-1] + ef_mag[1:])
    emit("\n" + "-" * 84)
    emit("PER-MAG dP/dg curve (deciles of true r_input_p)   [ |>2sig| flagged * ]")
    emit(f"  {'bin':>3} {'r_mag':>7} {'N':>13} {'f_det(g0)':>10} {'f_det(g05)':>11} {'dP/dg[%]':>10} {'sig[%]':>8} {'nsig':>6}")
    for k in range(args.n_mag):
        ns = mdP[k] / msig[k] if msig[k] > 0 else np.nan
        flag = "*" if abs(ns) > 2 else " "
        emit(f"  {k:>3} {mag_centers[k]:>7.2f} {int(mN[k]):>13,} {mf0[k]:>10.4f} {mf5[k]:>11.4f} "
             f"{mdP[k]*100:>+10.3f} {msig[k]*100:>8.3f} {ns:>+5.1f}{flag}")

    # --- (mag x size x blend) GRID ---
    giA, nblend = grid_bins(A); giB, _ = grid_bins(B)
    n_cells = args.n_flux * args.n_size * nblend
    cdP, csig, cf0, cf5, cN = response(A, B, giA, giB, n_cells, g, args.n_boot, args.seed)
    cdP = cdP.reshape(args.n_flux, args.n_size, nblend)
    csig = csig.reshape(args.n_flux, args.n_size, nblend)
    cN = cN.reshape(args.n_flux, args.n_size, nblend)
    emit("\n" + "-" * 84)
    emit(f"(mag x size x blend) dP/dg [%] +/- sig  (n_flux={args.n_flux} n_size={args.n_size} "
         f"n_blend={nblend}; blend0=ISOLATED)")
    for a in range(args.n_flux):
        for s in range(args.n_size):
            row = []
            for c in range(nblend):
                if cN[a, s, c] > 2000:
                    row.append(f"{cdP[a,s,c]*100:+6.2f}+/-{csig[a,s,c]*100:4.2f}")
                else:
                    row.append("     (thin) ")
            emit(f"  mag{a} size{s}: " + " | ".join(row))
    emit("  blend-bin N-weighted dP/dg:")
    for c in range(nblend):
        tag = "ISOLATED" if c == 0 else f"blend d{c}"
        w = cN[:, :, c]
        bb = np.nansum(cdP[:, :, c] * w) / max(w.sum(), 1)
        emit(f"    {tag:>10}: {bb*100:+.3f}%   (N={int(w.sum()):,})")

    emit("\n" + "=" * 84)
    emit("NOTES: (1) rate response dP/dg is the ISOTROPIC (monopole) detection change; the")
    emit("    anisotropic shape-correlated selection BIAS is a separate projection (future).")
    emit("    (2) g=0 and g=0.05 are independent scene samples -> bins compared at population")
    emit("    level (same true-property binning), not per-object matched.")
    emit("    (3) frozen render seed -> bootstrap captures placement, not independent pixel noise.")

    txt = args.out_prefix + ".txt"; npz = args.out_prefix + ".npz"
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(npz, nominal_g=g, det_frac_g0=fa, det_frac_g05=fb,
             global_dP=gdP[0], global_sig=gsig[0],
             mag_edges=ef_mag, mag_centers=mag_centers, mag_dP=mdP, mag_sig=msig,
             mag_fdet_g0=mf0, mag_fdet_g05=mf5, mag_N=mN,
             grid_edges_flux=ef, grid_edges_size=es, grid_edges_dist=ed,
             grid_dP=cdP, grid_sig=csig, grid_N=cN,
             n_flux=args.n_flux, n_size=args.n_size, n_dist=args.n_dist, n_blend=nblend, n_boot=args.n_boot)
    emit(f"\nwrote {txt}")
    emit(f"wrote {npz}")
    emit("DERISK_DETRESP_DONE")


if __name__ == "__main__":
    main()
