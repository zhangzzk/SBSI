"""ORACLE-#35 FEASIBILITY TEST: decompose the shear response of the DETECTED, selected ensemble on
the g=0.05 leg into a SHAPE part (what the certified pipeline models) and a SELECTION part (what the
joint measured+detection flow #35 would add). Fully self-contained on the det_meas legs; constgold is
never read; the certified m is never touched. Firewall-safe. Fork of diag_anisotropic_selection.py.

For a selection C over the DETECTED population, on the g=0.05 leg (per-case shear dir, |g|=0.05):
    R_total(C) = <g_meas_par>_{det & C} / g      g_meas_par = (mg1*g1 + mg2*g2)/|g|   (MEASURED shape)
    R_sel(C)   = <e_intr_par>_{det & C} / g      e_intr_par = (e1*g1 + e2*g2)/|g|      (INTRINSIC -> pure selection)
    R_shape(C) = R_total - R_sel                 (shape-only response, what R_flow models)
    m_cert(C)  = R_sel(C) / R_shape(C)            (certified pipeline's real-data selection bias under C)

Baseline is 0 at g=0 by isotropy (no preferred direction); the PARENT (all rows) <e_par>/g ~ 0 control
validates it (rot-pair shape-noise cancellation). Case-cluster bootstrap for all errors.

READING: R_sel/R_shape LARGE on the aggressive/measured cuts => selection is the missing term the certified
   pipeline drops => #35 CAN break those cuts (residual = the R_flow non-closure ceiling ~4%).
   R_sel/R_shape ~ 0 => the measured-cut blowups are pure R_flow non-closure => #35 irrelevant => ceiling stands.
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
DEF_G05 = f"{CATDIR}/det_meas_ngmix_g0.05_val.feather"
READ_COLS = ["detected", "case", "r_input_p", "Re_input_p", "distance", "neighbored",
             "gamma1_input_p", "gamma2_input_p",
             "e1_input_rot0_p", "e2_input_rot0_p",
             "measured_ngmix_g1", "measured_ngmix_g2",
             "measured_mag_auto", "measured_flux_radius"]
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
    g1 = out["gamma1_input_p"]; g2 = out["gamma2_input_p"]
    gmag = np.hypot(g1, g2)
    with np.errstate(invalid="ignore", divide="ignore"):
        u1 = np.where(gmag > 1e-6, g1 / gmag, np.nan)
        u2 = np.where(gmag > 1e-6, g2 / gmag, np.nan)
        out["e_intr_par"] = out["e1_input_rot0_p"] * u1 + out["e2_input_rot0_p"] * u2
        out["g_meas_par"] = out["measured_ngmix_g1"] * u1 + out["measured_ngmix_g2"] * u2
    print(f"  {os.path.basename(path)}: raw={raw:,} selected={len(out['detected']):,} "
          f"det_frac={out['detected'].mean():.4f} |g|med={np.nanmedian(gmag):.4f} in {time.time()-t0:.0f}s", flush=True)
    return out


def percase_sums(D, sel, ecol):
    """Per-case sum of D[ecol] over sel, and per-case count over sel. sel already excludes non-finite ecol."""
    e = np.where(np.isfinite(D[ecol]), D[ecol], 0.0)
    w = sel.astype(np.float64)
    uc, inv = np.unique(D["case"], return_inverse=True)
    se = np.bincount(inv, weights=e * w, minlength=len(uc))
    nn = np.bincount(inv, weights=w, minlength=len(uc))
    return se, nn, uc


def eval_cut(D, sel, g, picks):
    """Return (R_total, R_sel, R_shape, m_cert, sems...) for a selection over the DETECTED pop.
    sel_f requires detection & the cut & finite BOTH shapes (common population)."""
    fin = np.isfinite(D["e_intr_par"]) & np.isfinite(D["g_meas_par"])
    s = D["detected"] & sel & fin
    n = int(s.sum())
    if n < 2000:
        return None
    se_i, nn, _ = percase_sums(D, s, "e_intr_par")   # intrinsic -> R_sel
    se_m, _, _ = percase_sums(D, s, "g_meas_par")     # measured  -> R_total
    N = nn.sum()
    Rsel = se_i.sum() / N / g
    Rtot = se_m.sum() / N / g
    Rsh = Rtot - Rsel
    mcert = Rsel / Rsh if abs(Rsh) > 1e-9 else np.nan
    # case-cluster bootstrap
    bs_i = se_i[picks].sum(1); bs_m = se_m[picks].sum(1); bs_n = nn[picks].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bRsel = bs_i / bs_n / g
        bRtot = bs_m / bs_n / g
        bRsh = bRtot - bRsel
        bm = np.where(np.abs(bRsh) > 1e-9, bRsel / bRsh, np.nan)
    return dict(n=n, Rtotal=Rtot, Rsel=Rsel, Rshape=Rsh, mcert=mcert,
                sRsel=float(np.nanstd(bRsel)), sRshape=float(np.nanstd(bRsh)),
                smcert=float(np.nanstd(bm)))


def build_cuts(D):
    """List of (label, kind, mask) over the DETECTED population. mask is the ADDITIONAL cut C."""
    mag = D["r_input_p"]; size = D["Re_input_p"]; nb = D["neighbored"]
    mm = D["measured_mag_auto"]; ms = D["measured_flux_radius"]
    allrows = np.ones(len(mag), bool)
    out = [("GLOBAL", "global", allrows)]
    # --- TRUE-property acceptance cuts (r_input_p = true mag, Re_input_p = true size) ---
    for lo, hi in [(23, 24), (24, 25), (25, 26), (26, 27)]:
        out.append((f"mag{lo}-{hi}", "mag_win", (mag >= lo) & (mag < hi)))
    for x in [24.0, 25.0]:
        out.append((f"mag_lt{x:g}", "mag_cum", mag < x))
    out.append(("mag26-27_iso", "mag_x_blend", (mag >= 26) & (mag < 27) & (~nb)))
    out.append(("mag26-27_blend", "mag_x_blend", (mag >= 26) & (mag < 27) & nb))
    for lo, hi in [(0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]:
        out.append((f"size{lo}-{hi}", "size_win", (size >= lo) & (size < hi)))
    out.append(("size_gt0.3", "size_cum", size >= 0.3))
    out.append(("size_gt1.0", "size_cum", size >= 1.0))
    out.append(("isolated", "env", ~nb))
    out.append(("blended", "env", nb))
    # --- MEASURED-property cuts (the realistic scenario; the ones that blew up on constgold) ---
    good_mm = np.isfinite(mm)
    qm = np.quantile(mm[good_mm], np.linspace(0, 1, 11)); qm[0] -= 1e-6; qm[-1] += 1e-6
    bm = np.digitize(mm, qm[1:-1])
    for k in (0, 4, 8, 9):   # bright / mid / 9th / faintest decile
        out.append((f"meas_mag_q{k+1}/10", "measured", good_mm & (bm == k)))
    good_ms = np.isfinite(ms)
    qs = np.quantile(ms[good_ms], np.linspace(0, 1, 5)); qs[0] -= 1e-6; qs[-1] += 1e-6
    bs = np.digitize(ms, qs[1:-1])
    for k in range(4):        # measured-size quartiles (q1 = smallest blew up -67.9%)
        out.append((f"meas_fluxrad_q{k+1}/4", "measured", good_ms & (bs == k)))
    # measured faint x measured small (the +61% cell)
    out.append(("meas_magFaint_x_sizeSmall", "measured",
                good_mm & good_ms & (bm >= 8) & (bs == 0)))
    out.append(("meas_magBright_x_sizeLarge", "measured",
                good_mm & good_ms & (bm <= 1) & (bs == 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g05", default=DEF_G05)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out-prefix",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selresp_decomp")
    args = ap.parse_args()
    g = args.nominal_g
    lines = []

    def emit(s=""):
        print(s, flush=True); lines.append(s)

    emit("=" * 104)
    emit("ORACLE-#35 SELECTION-RESPONSE DECOMPOSITION  (g=0.05 leg; R_total=meas, R_sel=intrinsic, R_shape=diff)")
    emit("m_cert = R_sel/R_shape = the certified pipeline's real-data bias from the MISSING selection response.")
    emit("=" * 104)
    emit(f"\n[load] g=0.05 : {args.g05}")
    D = stream(args.g05, args.max_batches)

    # parent null control: <e_intr_par>/g over ALL detected rows should be the anisotropic-sel value;
    # over ALL rows (parent) intrinsic should be ~0.
    fin = np.isfinite(D["e_intr_par"])
    par = np.nanmean(D["e_intr_par"][fin]) / g
    parN = np.nanmean(D["e_intr_par"][fin & ~D["detected"]]) / g if (fin & ~D["detected"]).any() else float("nan")
    emit(f"[control] parent <e_intr_par>/g (all)={par:+.4f}  (undetected-only)={parN:+.4f}   (both ~0 => baseline OK)")

    rng = np.random.default_rng(args.seed)
    uc = np.unique(D["case"]); nC = len(uc)
    picks = rng.integers(0, nC, size=(args.n_boot, nC))
    emit(f"[boot] {nC} cases, {args.n_boot} resamples\n")

    cuts = build_cuts(D)
    emit(f"{'label':>28} {'kind':>12} {'Ndet':>12} {'R_total':>9} {'R_sel':>9} {'R_shape':>9} "
         f"{'m_cert%':>9} {'sem%':>7} {'z':>6}")
    emit("-" * 104)
    rows = []
    for label, kind, mask in cuts:
        r = eval_cut(D, mask, g, picks)
        if r is None:
            emit(f"{label:>28} {kind:>12} {'(thin)':>12}")
            continue
        z = abs(r["mcert"]) / r["smcert"] if r["smcert"] > 0 else float("nan")
        emit(f"{label:>28} {kind:>12} {r['n']:>12,} {r['Rtotal']:>+9.4f} {r['Rsel']:>+9.4f} "
             f"{r['Rshape']:>+9.4f} {100*r['mcert']:>+9.2f} {100*r['smcert']:>7.2f} {z:>6.1f}")
        rows.append((label, kind, r))

    emit("\n" + "=" * 104)
    # summarize the SELECTION bias by kind
    emit("SUMMARY: |m_cert| (=|R_sel/R_shape|, the SELECTION bias #35 targets) by kind")
    bykind = {}
    for label, kind, r in rows:
        if np.isfinite(r["mcert"]) and abs(r["Rshape"]) > 0.05:  # well-conditioned only
            bykind.setdefault(kind, []).append(abs(100 * r["mcert"]))
    for kind in sorted(bykind):
        v = bykind[kind]
        emit(f"   {kind:>12}: n={len(v):>2}  median|m_cert|={np.median(v):>6.2f}%  worst={max(v):>6.2f}%")
    allv = [x for v in bykind.values() for x in v]
    if allv:
        emit(f"\n   ALL well-conditioned: median|m_cert|={np.median(allv):.2f}%  worst={max(allv):.2f}%  "
             f"(n={len(allv)})")
    emit("\nREADING: median/worst |m_cert| LARGE (multi-%) on measured/faint/size cuts => selection is a real")
    emit("  missing term #35 supplies (residual then = the ~4% R_flow ceiling). ~0 => #35 cannot help; ceiling is all.")

    txt = args.out_prefix + ".txt"; npz = args.out_prefix + ".npz"
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(npz, nominal_g=g,
             labels=np.array([x[0] for x in rows]), kinds=np.array([x[1] for x in rows]),
             Rtotal=np.array([x[2]["Rtotal"] for x in rows]),
             Rsel=np.array([x[2]["Rsel"] for x in rows]),
             Rshape=np.array([x[2]["Rshape"] for x in rows]),
             mcert=np.array([x[2]["mcert"] for x in rows]),
             smcert=np.array([x[2]["smcert"] for x in rows]),
             n=np.array([x[2]["n"] for x in rows]))
    emit(f"\nwrote {txt}\nwrote {npz}\nSELRESP_DECOMP_DONE")


if __name__ == "__main__":
    main()
