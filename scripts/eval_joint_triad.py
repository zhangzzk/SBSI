#!/usr/bin/env python
"""SIM-side response TRIAD on constant-gold — separate SHAPE / SELECTION / DETECTION bias.

The reframe (GOALS.md) calibrates three response channels; this measures each DIRECTLY from
constgold with an orthogonal design that turns the other two channels off (owner spec 2026-07-22):

  1. SHAPE (coherent)  matched pairs + TRUE cut + MEASURED shape, per-object antithetic difference
       R1 = <(e_+ - e_-)·ghat> / 2g        (constgold shears the WHOLE field -> R1 = R_self + R_blend;
                                            both-detected pairs => detection matched; true cut => no
                                            measured selection). This is the coherent response.
  2. SELECTION         matched-detection + LEG-AVERAGE + MEASURED cut + INTRINSIC e
       R2 = [<e_int·ghat | S/N_+>t> - <e_int·ghat | S/N_->t>] / 2g
       Intrinsic e kills the shape response; both-detected kills detection; a MEASURED (S/N) cut
       moves the boundary with shear -> pure measurement-selection. With NO measured cut R2 -> 0
       (the "Term 2 vanishes for TRUE cuts" check).
  3. DETECTION         LEG-AVERAGE + TRUE cut + INTRINSIC e, detection free to differ
       R3 = [<e_int·ghat>_{+leg,det,truecut} - <e_int·ghat>_{-leg,det,truecut}] / 2g
       True cut kills measurement selection; intrinsic e kills shape; the +g and -g DETECTED
       samples differ (detection is shear-dependent) -> pure detection response.

On the TRUE-cut deliverable the total sim response is R1 + R3 (measurement selection R2 = 0);
the model must match it with R_flow(R_self) + R_blend(emulator) + R_detect. This script produces
the SIM side; the model overlay + assembled m is a follow-on.

FIREWALL NOTE: constgold is the held-out ACCEPTANCE metric. This is an EVALUATION that reads it;
nothing here trains on it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402

CDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
MAG_EDGES = np.array([18.0, 24.0, 25.0, 26.0])          # primary TRUE mag (r_input_p); cut is <26
SIZE_EDGES = np.array([0.30, 0.38, 0.50, 1.50])         # primary TRUE size (Re_input_p); cut is >0.3


def _stream(path, cols, min_case, max_case, true_cut, extra_finite=()):
    """Stream a constgold feather, source-select + apply the TRUE primary cut, return one frame."""
    parts = []
    with ipc.open_file(path) as r:
        avail = set(r.schema.names)
        use = [c for c in cols if c in avail]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            bmin = int(b["case"].min())
            if min_case is not None:
                b = b[b["case"] >= min_case]
            if max_case is not None:
                if bmin > max_case:      # catalogues are written case-ascending -> stop early
                    break
                b = b[b["case"] <= max_case]
            if len(b) == 0:
                continue
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if true_cut is not None:
                re_min, mag_max = true_cut
                b = b[(b["Re_input_p"].to_numpy(float) > re_min)
                      & (b["r_input_p"].to_numpy(float) < mag_max)]
            if len(b):
                parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_int"] = np.asarray(e1i, float)
    df["e2_int"] = np.asarray(e2i, float)
    return df.reset_index(drop=True)


def _ghat(df):
    g = np.hypot(df["applied_g1"].to_numpy(float), df["applied_g2"].to_numpy(float))
    gmed = float(np.median(g))
    gh1 = df["applied_g1"].to_numpy(float) / g
    gh2 = df["applied_g2"].to_numpy(float) / g
    return gmed, gh1, gh2


def _binned(vals, mag, size, case, weight_pos=None, reducer="mean"):
    """Return (global, per-mag, per-size) reductions of `vals`. reducer='mean' -> <vals>;
    reducer='diffsel' expects vals=(proj, mask_plus, mask_minus) and returns the leg-avg diff."""
    out = {}
    mi = np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(MAG_EDGES) - 2)
    si = np.clip(np.digitize(size, SIZE_EDGES) - 1, 0, len(SIZE_EDGES) - 2)
    out["mag_bins"] = [(MAG_EDGES[k], MAG_EDGES[k + 1], mi == k) for k in range(len(MAG_EDGES) - 1)]
    out["size_bins"] = [(SIZE_EDGES[k], SIZE_EDGES[k + 1], si == k) for k in range(len(SIZE_EDGES) - 1)]
    return out


HS_BASE = ["axis_ratio_input_p", "position_angle_input_p", "Re_input_p", "r_input_p",
           "distance", "neighbored", "detected", "case", "input_index",
           "measured_flux_radius", "measured_mag_auto", "measured_flux_auto",
           "measured_fluxerr_auto", "measured_ngmix_g1", "measured_ngmix_g2"]
HS_G05_EXTRA = ["gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s"]


def halfshear_setup2(args):
    """Setup-2 (SELECTION) on the HALF-SHEAR legs, the sanctioned triad estimator applied off
    constgold: g0 (unsheared) vs g05 (primary sheared by g, neighbour FIXED so no R_blend leak),
    matched per (case, input_index). Intrinsic e is shear-INVARIANT, so a MEASURED cut that selects
    a different subset in each leg isolates the moving-boundary selection response:

        R_sel_intr(t) = [<e_int.ghat>_{theta_g05>t} - <e_int.ghat>_{theta_g0>t}] / g

    ghat = the g05 primary shear direction (rotates per case). The measured-shape "moving vs fixed"
    decomposition is the independent cross-check (cont.117/118): R_shape = fixed-cut measured
    response, R_moving = moving-cut response, R_sel_meas = R_moving - R_shape. For a TRUE-property
    cut R_sel=0; it grows only for measured cuts. FIREWALL: half-shear only, constgold untouched.
    """
    g = 0.05
    true_cut = None if args.no_true_cut else (args.true_re_min, args.true_mag_max)
    rng = np.random.default_rng(0)
    t0 = time.time()
    print(f"HALF-SHEAR Setup-2  neighbour={args.hs_neighbour}  true_cut={true_cut}", flush=True)
    G0 = _stream(args.hs_g0, HS_BASE, args.min_case, args.max_case, true_cut)
    G5 = _stream(args.hs_g05, HS_BASE + HS_G05_EXTRA, args.min_case, args.max_case, true_cut)
    G0 = G0[G0["detected"].astype(bool)].copy()
    G5 = G5[G5["detected"].astype(bool)].copy()
    gs = np.hypot(G5["gamma1_input_s"].to_numpy(float), G5["gamma2_input_s"].to_numpy(float))
    if args.hs_neighbour == "fixed":
        G5 = G5[gs < 1e-6].copy()
    elif args.hs_neighbour == "sheared":
        G5 = G5[gs > 1e-6].copy()
    gpv = np.hypot(G5["gamma1_input_p"].to_numpy(float), G5["gamma2_input_p"].to_numpy(float))
    G5 = G5[gpv > 1e-6].copy()
    gpv = np.hypot(G5["gamma1_input_p"].to_numpy(float), G5["gamma2_input_p"].to_numpy(float))
    G5["gh1"] = G5["gamma1_input_p"].to_numpy(float) / gpv
    G5["gh2"] = G5["gamma2_input_p"].to_numpy(float) / gpv
    n5, n0 = len(G5), len(G0)
    G5 = G5.drop_duplicates(["case", "input_index"])
    G0 = G0.drop_duplicates(["case", "input_index"])
    keep0 = ["case", "input_index", "measured_flux_radius", "measured_mag_auto",
             "measured_flux_auto", "measured_fluxerr_auto", "measured_ngmix_g1", "measured_ngmix_g2"]
    m = G5.merge(G0[keep0], on=["case", "input_index"], suffixes=("_g5", "_g0"))
    print(f"  g0={n0:,} g05={n5:,} (neighbour={args.hs_neighbour}) -> matched {len(m):,}  "
          f"g_med={float(np.median(gpv)):.4f}  ({time.time()-t0:.1f}s)", flush=True)
    if len(m) == 0:
        raise SystemExit("no matched half-shear pairs (check --hs-neighbour / case range)")

    gh1 = m["gh1"].to_numpy(float); gh2 = m["gh2"].to_numpy(float)
    eint = m["e1_int"].to_numpy(float) * gh1 + m["e2_int"].to_numpy(float) * gh2
    em5 = m["measured_ngmix_g1_g5"].to_numpy(float) * gh1 + m["measured_ngmix_g2_g5"].to_numpy(float) * gh2
    em0 = m["measured_ngmix_g1_g0"].to_numpy(float) * gh1 + m["measured_ngmix_g2_g0"].to_numpy(float) * gh2
    case = m["case"].to_numpy(int)
    cutvars = {
        "size": (m["measured_flux_radius_g5"].to_numpy(float), m["measured_flux_radius_g0"].to_numpy(float)),
        "mag": (m["measured_mag_auto_g5"].to_numpy(float), m["measured_mag_auto_g0"].to_numpy(float)),
        "sn": ((m["measured_flux_auto_g5"] / m["measured_fluxerr_auto_g5"]).to_numpy(float),
               (m["measured_flux_auto_g0"] / m["measured_fluxerr_auto_g0"]).to_numpy(float)),
    }

    def masks(var, t, direction):
        v5, v0 = cutvars[var]
        return (v5 > t, v0 > t) if direction == ">" else (v5 < t, v0 < t)

    def estim(mask5, mask0, sub=None):
        if sub is not None:
            mask5 = mask5 & sub; mask0 = mask0 & sub
        if int(mask5.sum()) < 2 or int(mask0.sum()) < 2:
            return dict(R_sel_intr=np.nan, R_shape=np.nan, R_moving=np.nan, R_sel_meas=np.nan,
                        m_uncorr=np.nan, m_corr=np.nan, n5=int(mask5.sum()), n0=int(mask0.sum()))
        R_sel_intr = (eint[mask5].mean() - eint[mask0].mean()) / g
        R_shape = (em5[mask0].mean() - em0[mask0].mean()) / g            # fixed cut -> pure shape
        R_moving = (em5[mask5].mean() - em0[mask0].mean()) / g            # moving cut -> shape+sel
        R_sel_meas = R_moving - R_shape
        m_uncorr = R_moving / R_shape - 1.0 if R_shape != 0 else np.nan
        denom = R_shape + R_sel_intr
        m_corr = R_moving / denom - 1.0 if denom != 0 else np.nan
        return dict(R_sel_intr=R_sel_intr, R_shape=R_shape, R_moving=R_moving, R_sel_meas=R_sel_meas,
                    m_uncorr=m_uncorr, m_corr=m_corr, n5=int(mask5.sum()), n0=int(mask0.sum()))

    uniq = np.unique(case)

    def boot(var, t, direction):
        vi, vm = [], []
        m5, m0 = masks(var, t, direction)
        for _ in range(args.n_boot):
            sub = np.isin(case, rng.choice(uniq, size=len(uniq), replace=True))
            e = estim(m5, m0, sub=sub)
            vi.append(e["R_sel_intr"]); vm.append(e["R_sel_meas"])
        return float(np.nanstd(vi)), float(np.nanstd(vm))

    menu = ([("size", t, ">") for t in args.hs_size_cuts]
            + [("mag", t, "<") for t in args.hs_mag_cuts]
            + [("sn", t, ">") for t in args.hs_sn_cuts])
    res = {"g": g, "n_matched": len(m), "neighbour": args.hs_neighbour,
           "true_cut": (-1 if true_cut is None else np.array(true_cut))}
    allm = np.ones(len(m), bool)
    res["baseline_R_sel_intr"] = estim(allm, allm)["R_sel_intr"]

    print(f"\n===== HALF-SHEAR Setup-2 selection response (matched g0/g05, neighbour "
          f"{args.hs_neighbour}) =====", flush=True)
    print(f"matched N={len(m):,}   baseline R_sel(no cut)={res['baseline_R_sel_intr']:+.5f}  (expect ~0)")
    print(f"  {'cut':<12} {'R_shape':>8} {'R_sel_intr':>15} {'R_sel_meas':>15} "
          f"{'m_uncorr':>9} {'m_corr':>8} {'n_g5':>10}")
    for var, t, d in menu:
        e = estim(*masks(var, t, d)); ei, em = boot(var, t, d)
        tag = f"{var}{d}{t:g}"
        for k in ("R_sel_intr", "R_sel_meas", "R_shape", "m_uncorr", "m_corr", "n5"):
            res[f"{k}__{tag}"] = e[k]
        res[f"R_sel_intr_err__{tag}"] = ei; res[f"R_sel_meas_err__{tag}"] = em
        print(f"  {tag:<12} {e['R_shape']:>+8.4f} {e['R_sel_intr']:>+9.4f}±{ei:.4f} "
              f"{e['R_sel_meas']:>+9.4f}±{em:.4f} {100 * e['m_uncorr']:>+8.2f}% "
              f"{100 * e['m_corr']:>+7.2f}% {e['n5']:>10,}")

    # ---------- R_theta: how the MEASURED properties themselves shift with shear (the loss target) ----------
    # Per matched object (same scene, +/- primary shear): d<measured prop>/dg. This is the quantity
    # the flow's mean head carries in cols 2,3 (meas mag, meas size); we measure it here to pin it.
    dmag = (m["measured_mag_auto_g5"].to_numpy(float) - m["measured_mag_auto_g0"].to_numpy(float)) / g
    dsize = (m["measured_flux_radius_g5"].to_numpy(float) - m["measured_flux_radius_g0"].to_numpy(float)) / g
    dshape = (em5 - em0) / g
    v = eint - eint.mean()                                    # spin-2 coupling of the size response to e_int.ghat
    vv = float(np.dot(v, v))
    slope_size = float(np.dot(dsize - dsize.mean(), v) / vv) if vv > 0 else np.nan
    slope_mag = float(np.dot(dmag - dmag.mean(), v) / vv) if vv > 0 else np.nan

    def _mean_err(a):
        vals = [float(np.mean(a[np.isin(case, rng.choice(uniq, size=len(uniq), replace=True))]))
                for _ in range(args.n_boot)]
        return float(np.mean(a)), float(np.nanstd(vals))

    Rshape, Rshape_e = _mean_err(dshape); Rmag, Rmag_e = _mean_err(dmag); Rsize, Rsize_e = _mean_err(dsize)
    res.update(dict(R_shape_global=Rshape, R_mag_global=Rmag, R_size_global=Rsize,
                    R_size_median=float(np.median(dsize)), R_mag_median=float(np.median(dmag)),
                    R_size_slope_eint=slope_size, R_mag_slope_eint=slope_mag))
    print("\n----- MEASURED-PROPERTY RESPONSE R_theta (matched pairs, no cut; d<prop>/dg) -----")
    print(f"  R_shape (meas e.ghat)     = {Rshape:+.4f} ± {Rshape_e:.4f}   (sanity: ~ shape target 0.7)")
    print(f"  R_mag   (d mag_auto/dg)   = {Rmag:+.4f} ± {Rmag_e:.4f}   median {np.median(dmag):+.4f}   (expect ~0: flux ~conserved)")
    print(f"  R_size  (d flux_radius/dg)= {Rsize:+.4f} ± {Rsize_e:.4f}   median {np.median(dsize):+.4f}")
    print(f"  size spin-2 coupling  d(size)/dg vs e_int.ghat: slope = {slope_size:+.4f}  "
          f"(the orientation-driven part that moves a size cut -> R_sel)")
    print(f"  {'bin (true)':<16} {'R_mag':>9} {'R_size':>9} {'R_shape':>9} {'n':>10}")
    magb = m["r_input_p"].to_numpy(float); szb = m["Re_input_p"].to_numpy(float)
    for axis, edges in (("mag", MAG_EDGES), ("size", SIZE_EDGES)):
        arr = magb if axis == "mag" else szb
        for k in range(len(edges) - 1):
            sel = (np.clip(np.digitize(arr, edges) - 1, 0, len(edges) - 2) == k)
            if int(sel.sum()) < 2:
                continue
            tag = f"{axis}[{edges[k]:g},{edges[k + 1]:g})"
            rm, rs, rsh = float(dmag[sel].mean()), float(dsize[sel].mean()), float(dshape[sel].mean())
            res[f"Rmag__{tag}"] = rm; res[f"Rsize__{tag}"] = rs; res[f"Rshape__{tag}"] = rsh
            print(f"  {tag:<16} {rm:>+9.4f} {rs:>+9.4f} {rsh:>+9.4f} {int(sel.sum()):>10,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **{k: v for k, v in res.items()})
    print(f"\nsaved {args.output}   ({time.time() - t0:.1f}s)", flush=True)
    print("HALFSHEAR_SETUP2_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response-cat", default=CDIR + "/constant_response_catalogue_train.feather")
    ap.add_argument("--plus-cat", default=CDIR + "/constant_shear_catalogue_0.02_train.feather")
    ap.add_argument("--minus-cat", default=CDIR + "/constant_shear_catalogue_-0.02_train.feather")
    ap.add_argument("--min-case", type=int, default=None)
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--no-true-cut", action="store_true")
    ap.add_argument("--sn-cuts", type=float, nargs="+", default=[7.0, 10.0, 15.0],
                    help="measured S/N thresholds for the SELECTION setup (setup 2).")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--output", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/joint_triad.npz")
    # --- HALF-SHEAR Setup-2 (selection response measured off the g0/g05 legs, not constgold) ---
    ap.add_argument("--halfshear", action="store_true",
                    help="run Setup-2 selection response on the HALF-SHEAR legs instead of the constgold triad")
    ap.add_argument("--hs-g0", default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
                    "det_meas_crowd_conc_g0.0_train_full.feather")
    ap.add_argument("--hs-g05", default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
                    "det_meas_crowd_g0.05_val_full.feather")
    ap.add_argument("--hs-neighbour", choices=["fixed", "sheared", "both"], default="fixed",
                    help="fixed=neighbour unsheared (clean primary-only R_sel, the flow's R_theta target); "
                         "sheared/both = cross-checks that include neighbour-shear leakage")
    ap.add_argument("--hs-size-cuts", type=float, nargs="+", default=[2.9, 3.5, 4.4],
                    help="measured_flux_radius thresholds (keep LARGE) for the selection setup")
    ap.add_argument("--hs-mag-cuts", type=float, nargs="+", default=[24.0, 24.5, 25.0],
                    help="measured_mag_auto thresholds (keep BRIGHT)")
    ap.add_argument("--hs-sn-cuts", type=float, nargs="+", default=[10.0, 15.0, 20.0],
                    help="measured S/N = flux_auto/fluxerr_auto thresholds (keep HIGH)")
    args = ap.parse_args()

    if args.halfshear:
        return halfshear_setup2(args)

    true_cut = None if args.no_true_cut else (args.true_re_min, args.true_mag_max)
    rng = np.random.default_rng(0)
    t0 = time.time()
    print(f"TRIAD  true_cut={true_cut}  sn_cuts={args.sn_cuts}", flush=True)

    # ---------- load ----------
    rcols = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
             "S/N_plus", "S/N_minus", "applied_g1", "applied_g2",
             "axis_ratio_input_p", "position_angle_input_p", "r_input_p", "Re_input_p",
             "neighbored", "distance", "input_index", "case"]
    scols = ["measured_e1", "measured_e2", "applied_g1", "applied_g2",
             "axis_ratio_input_p", "position_angle_input_p", "r_input_p", "Re_input_p",
             "neighbored", "distance", "input_index", "case"]
    R = _stream(args.response_cat, rcols, args.min_case, args.max_case, true_cut)
    P = _stream(args.plus_cat, scols, args.min_case, args.max_case, true_cut)
    M = _stream(args.minus_cat, scols, args.min_case, args.max_case, true_cut)
    print(f"  loaded response={len(R):,}  plus={len(P):,}  minus={len(M):,}  ({time.time()-t0:.1f}s)", flush=True)

    gR, gh1, gh2 = _ghat(R)
    magR = R["r_input_p"].to_numpy(float); sizeR = R["Re_input_p"].to_numpy(float); caseR = R["case"].to_numpy(int)
    e1int = R["e1_int"].to_numpy(float); e2int = R["e2_int"].to_numpy(float)
    eproj_int = e1int * gh1 + e2int * gh2                                  # intrinsic e projected on ghat

    # ---------- Setup 1: coherent SHAPE response (R_self + R_blend) ----------
    e1p = R["measured_e1_plus"].to_numpy(float); e2p = R["measured_e2_plus"].to_numpy(float)
    e1m = R["measured_e1_minus"].to_numpy(float); e2m = R["measured_e2_minus"].to_numpy(float)
    proj1 = 0.5 * ((e1p - e1m) * gh1 + (e2p - e2m) * gh2)                  # /g below -> R1
    snp = R["S/N_plus"].to_numpy(float); snm = R["S/N_minus"].to_numpy(float)

    # ---------- Setup 3 legs ----------
    gP, gp1, gp2 = _ghat(P); gM, gm1, gm2 = _ghat(M)
    # project intrinsic e onto a COMMON reference axis = the +leg applied direction (unit)
    ref1, ref2 = float(np.median(gp1)), float(np.median(gp2))
    epP = P["e1_int"].to_numpy(float) * ref1 + P["e2_int"].to_numpy(float) * ref2
    epM = M["e1_int"].to_numpy(float) * ref1 + M["e2_int"].to_numpy(float) * ref2
    magP = P["r_input_p"].to_numpy(float); sizeP = P["Re_input_p"].to_numpy(float); caseP = P["case"].to_numpy(int)
    magM = M["r_input_p"].to_numpy(float); sizeM = M["Re_input_p"].to_numpy(float); caseM = M["case"].to_numpy(int)

    def bin_mask(mag, size, axis, k):
        if axis == "mag":
            return (np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(MAG_EDGES) - 2) == k)
        return (np.clip(np.digitize(size, SIZE_EDGES) - 1, 0, len(SIZE_EDGES) - 2) == k)

    def R1_on(sel):
        return float(np.mean(proj1[sel])) / gR if sel.any() else np.nan

    def R2_on(sel, t):
        sp = sel & (snp > t); sm = sel & (snm > t)
        if not sp.any() or not sm.any():
            return np.nan
        return (float(np.mean(eproj_int[sp])) - float(np.mean(eproj_int[sm]))) / (2.0 * gR)

    def R3_on(selP, selM):
        if not selP.any() or not selM.any():
            return np.nan
        return (float(np.mean(epP[selP])) - float(np.mean(epM[selM]))) / (2.0 * gR)

    # ---------- case-level bootstrap ----------
    def boot(fn_R, cases_list, arrays_by_case):
        """fn_R takes a boolean over the concatenated rows; bootstrap resamples unique cases."""
        uniq = np.unique(np.concatenate([np.unique(c) for c in cases_list]))
        vals = []
        for _ in range(args.n_boot):
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            masks = [np.isin(c, draw) for c in cases_list]
            vals.append(fn_R(*masks))
        v = np.asarray(vals, float)
        return float(np.nanstd(v))

    allR = np.ones(len(R), bool)
    allP = np.ones(len(P), bool); allM = np.ones(len(M), bool)

    # globals
    res = {"g": gR, "n_response": len(R), "n_plus": len(P), "n_minus": len(M),
           "true_cut": (-1 if true_cut is None else np.array(true_cut))}
    res["R1_shape_coherent"] = R1_on(allR)
    res["R1_err"] = boot(lambda mR: R1_on(mR), [caseR], [caseR])
    res["R3_detection"] = R3_on(allP, allM)
    res["R3_err"] = boot(lambda mP, mM: R3_on(mP, mM), [caseP, caseM], [caseP, caseM])
    res["R2_nocut"] = R2_on(allR, -1e9)                                    # sanity: measured-sel with no cut -> 0
    for t in args.sn_cuts:
        res[f"R2_sel_SNgt{t:g}"] = R2_on(allR, t)
        res[f"R2_sel_SNgt{t:g}_err"] = boot(lambda mR, tt=t: R2_on(mR, tt), [caseR], [caseR])

    # per-bin (mag, size) for R1 and R3 (the two that build the true-cut deliverable)
    for axis, edges in (("mag", MAG_EDGES), ("size", SIZE_EDGES)):
        for k in range(len(edges) - 1):
            selR = bin_mask(magR, sizeR, axis, k)
            selP = bin_mask(magP, sizeP, axis, k)
            selM = bin_mask(magM, sizeM, axis, k)
            tag = f"{axis}[{edges[k]:g},{edges[k+1]:g})"
            res[f"R1__{tag}"] = R1_on(selR)
            res[f"R3__{tag}"] = R3_on(selP, selM)
            res[f"n__{tag}"] = int(selR.sum())

    # ---------- report ----------
    def fmt(k):
        return f"{res[k]:+.4f}" if isinstance(res.get(k), float) and np.isfinite(res.get(k, np.nan)) else "  nan "
    print("\n===== SIM TRIAD (constgold, true-cut population) =====", flush=True)
    print(f"g={gR:.4f}  N: response={len(R):,} plus={len(P):,} minus={len(M):,}")
    print(f"[1] SHAPE (coherent R_self+R_blend)  R1 = {res['R1_shape_coherent']:+.4f} +- {res['R1_err']:.4f}")
    print(f"[3] DETECTION                        R3 = {res['R3_detection']:+.4f} +- {res['R3_err']:.4f}")
    print(f"[2] SELECTION (measured S/N cut, leg-avg intrinsic-e):")
    print(f"      no cut (sanity, expect ~0)     R2 = {res['R2_nocut']:+.5f}")
    for t in args.sn_cuts:
        print(f"      S/N > {t:<4g}                     R2 = {res[f'R2_sel_SNgt{t:g}']:+.4f} +- {res[f'R2_sel_SNgt{t:g}_err']:.4f}")
    print(f"\n  TRUE-CUT total sim response (R1+R3) = {res['R1_shape_coherent']+res['R3_detection']:+.4f}"
          f"   (measurement selection R2≈0 for true cuts)")
    print("\n  per-bin R1 (coherent shape) and R3 (detection):")
    print(f"  {'bin':<16} {'R1':>9} {'R3':>9} {'n':>10}")
    for axis, edges in (("mag", MAG_EDGES), ("size", SIZE_EDGES)):
        for k in range(len(edges) - 1):
            tag = f"{axis}[{edges[k]:g},{edges[k+1]:g})"
            print(f"  {tag:<16} {fmt('R1__'+tag):>9} {fmt('R3__'+tag):>9} {res['n__'+tag]:>10,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **{k: v for k, v in res.items()})
    print(f"\nsaved {args.output}   ({time.time()-t0:.1f}s)", flush=True)
    print("JOINT_TRIAD_DONE", flush=True)


if __name__ == "__main__":
    main()
