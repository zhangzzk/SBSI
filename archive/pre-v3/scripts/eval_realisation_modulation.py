"""STAGE P0 -- is the realisation dependence VISIBLE ON HALF-SHEAR, i.e. on firewall-clean data?

WHY THIS RUNS FIRST.  Every measured symptom motivating the realisation-aware (RA) head is a
CONSTGOLD number (5.4x over-prediction at measured size < 0.60", +46.2% at measured mag 26.25-26.5).
The firewall forbids training on constgold, so a modulation target must be built from the half-shear
legs.  If the same realisation dependence is NOT present there, the RA design is dead and the honest
output is "blocked" -- not a constgold-derived target.  This script is the decisive measurement.  It
is CPU-only, trains nothing, fits nothing, and opens no constgold file.

WHAT IS MEASURED.  Inside a coarse TRUE-property cell (true flux x true size), split the galaxies
into sub-bins `b` of their MEASURED realisation -- the g=0 leg's measured magnitude and log size,
each taken as a residual about that cell's own median -- and form the SCALE-FREE modulation

    rho(cell, b) = <R>_b / <R>_cell

  rho_sim   from the sim:   R_sim(i) = ((e_g - e_0) . ghat) / g     (measured ngmix, forward legs)
  rho_model from the flow:  R_flow(i), the per-object response already dumped per seed in
                            results/halfshear_selfresp.feather  (a TRUE-property function, so a
                            realisation-blind model must give rho_model ~ flat)

The learnable signal is `rho_sim / rho_model - 1`.  rho is a ratio WITHIN a cell, so the cell's
response level -- which the existing per-cell response pin already controls -- divides out.

CONTROLS, reported next to the signal:
  (4) 45-degree (spin-2 orthogonal) null: repeat with ghat -> (-ghat2, ghat1). Must be ~ 0.
  (5) contamination <e_0 . ghat>/g per bin. ghat is an independent random direction per object, so
      this must vanish for ANY g=0-based binning; a non-zero value would mean the g=0-leg binning
      is picking up the projection itself rather than the response.
  (1) leg noise sharing: std(mag_g - mag_0) and corr(mag_0, mag_g) about the per-cell median. If the
      legs share pixel noise the difference is tiny compared to the single-leg scatter.
  (6) the same rho binned on the SHEARED leg instead. The difference between (2) and (6) is the
      migration/boundary term and is why the target is binned on the g=0 leg: the flow's training
      catalogue is g=0-only, so with g=0-leg binning each training row's own measured mag/size IS a
      valid draw of the label axis; with sheared-leg binning there is no such quantity at all.

Cases are split into a FIT half and a HELD-OUT half so the modulation can be shown to be a property
of the population and not of the fit sample.

FIREWALL: half-shear legs only. No constgold, no emulator, no training, no tuning.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_selection_response as ESR  # noqa: E402
from eval_selection_response import CAT, CROWD, NN  # noqa: E402


# ----------------------------------------------------------------------------------------------
# binning helpers
# ----------------------------------------------------------------------------------------------
def quantile_edges(values, n):
    e = np.quantile(values[np.isfinite(values)], np.linspace(0.0, 1.0, n + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def digitize(values, edges):
    return np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)


def binned_stats(idx, nbin, values, weights):
    """weighted sum, weight, weighted sum of squares, sum of squared weights per flat bin."""
    s = np.bincount(idx, weights=weights * values, minlength=nbin)
    w = np.bincount(idx, weights=weights, minlength=nbin)
    q = np.bincount(idx, weights=weights * values * values, minlength=nbin)
    w2 = np.bincount(idx, weights=weights * weights, minlength=nbin)
    return s, w, q, w2


def rho_and_sem(cell, sub, n_cells, n_mb, values, weights):
    """rho(cell,b) = mean_b / mean_cell, with a delta-method sem.

    Bins are DISJOINT parts of their cell, so cov(mean_b, mean_cell) = f_b var(mean_b) with
    f_b the bin's weight fraction; that term is kept (dropping it overstates the error).
    Returns rho, sem(rho), mean_b, sem(mean_b), effective counts, and the per-cell mean.
    """
    flat = cell * n_mb + sub
    s, w, q, w2 = binned_stats(flat, n_cells * n_mb, values, weights)
    s = s.reshape(n_cells, n_mb)
    w = w.reshape(n_cells, n_mb)
    q = q.reshape(n_cells, n_mb)
    w2 = w2.reshape(n_cells, n_mb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mb = s / np.where(w > 0, w, np.nan)
        var_b = q / np.where(w > 0, w, np.nan) - mb ** 2
        # variance of a WEIGHTED mean uses n_eff = (sum w)^2 / sum w^2 -- equal to the row count
        # only when every weight is 1 (nearest-pair); smaller for an all-pairs catalogue.
        n_eff = w ** 2 / np.where(w2 > 0, w2, np.nan)
        var_mb = np.maximum(var_b, 0.0) / np.where(n_eff > 0, n_eff, np.nan)
        wc = w.sum(axis=1)
        mc = s.sum(axis=1) / np.where(wc > 0, wc, np.nan)
        f = w / np.where(wc[:, None] > 0, wc[:, None], np.nan)
        var_mc = (f ** 2 * var_mb).sum(axis=1)
        rho = mb / mc[:, None]
        var_rho = (var_mb / mc[:, None] ** 2
                   + mb ** 2 * var_mc[:, None] / mc[:, None] ** 4
                   - 2.0 * mb * f * var_mb / mc[:, None] ** 3)
        sem_rho = np.sqrt(np.maximum(var_rho, 0.0))
    return rho, sem_rho, mb, np.sqrt(var_mb), w, mc


def wrms(x, w, mask):
    m = mask & np.isfinite(x) & np.isfinite(w)
    if not m.any() or w[m].sum() <= 0:
        return float("nan")
    return float(np.sqrt((w[m] * x[m] ** 2).sum() / w[m].sum()))


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--selfresp", default=os.path.join(SBSI_ROOT, "results/halfshear_selfresp.feather"),
                    help="per-object dump with r_sim_self and R_flow_s{seed} (dump_halfshear_selfresp.py)")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--fit-max-case", type=int, default=19,
                    help="cases <= this form the FIT half; the rest are HELD OUT")
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--n-flux", type=int, default=6, help="true-magnitude cell bins")
    ap.add_argument("--n-size", type=int, default=2, help="true-size cell bins")
    ap.add_argument("--n-dmag", type=int, default=5, help="measured-mag residual sub-bins")
    ap.add_argument("--n-dlogsize", type=int, default=2, help="measured-log-size residual sub-bins")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    t0 = time.time()
    ru = ESR.build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                        args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"]
    print(f"base N={len(base):,}  g_med={gmed:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- merge the per-object model response (already dumped, 16 seeds) -----------------------
    sr = pf.read_table(args.selfresp).to_pandas()
    seed_cols = sorted(c for c in sr.columns if c.startswith("R_flow_s"))
    if not seed_cols:
        raise SystemExit(f"no R_flow_s* columns in {args.selfresp}")
    key = base[["case", "input_index"]].copy()
    key["__row"] = np.arange(len(base))
    j = key.merge(sr[["case", "input_index", "r_sim_self"] + seed_cols],
                  on=["case", "input_index"], how="left")
    j = j.sort_values("__row")
    matched = np.isfinite(j["r_sim_self"].to_numpy(float))
    print(f"selfresp merge: {matched.mean():.4%} matched, {len(seed_cols)} seeds "
          f"({', '.join(s.replace('R_flow_', '') for s in seed_cols)})", flush=True)
    if matched.mean() < 0.99:
        raise SystemExit("selfresp dump does not cover the base -- rebuild it with the same "
                         "--max-case / --true-* cuts before trusting anything below")
    r_model = np.nanmean(np.stack([j[c].to_numpy(float) for c in seed_cols], axis=1), axis=1)

    # ---- responses -----------------------------------------------------------------------------
    e1_0 = base["measured_ngmix_g1_0"].to_numpy(float)
    e2_0 = base["measured_ngmix_g2_0"].to_numpy(float)
    e1_g = base["measured_ngmix_g1_g"].to_numpy(float)
    e2_g = base["measured_ngmix_g2_g"].to_numpy(float)
    r_sim = ((e1_g - e1_0) * gh1 + (e2_g - e2_0) * gh2) / gmed
    # spin-2 orthogonal direction: (cos2t, sin2t) -> (-sin2t, cos2t)
    r_null = ((e1_g - e1_0) * (-gh2) + (e2_g - e2_0) * gh1) / gmed
    r_cont = (e1_0 * gh1 + e2_0 * gh2) / gmed
    d = r_sim - j["r_sim_self"].to_numpy(float)
    print(f"cross-check vs dumped r_sim_self: max|diff|={np.nanmax(np.abs(d)):.3e}  "
          f"<r_sim>={np.nanmean(r_sim):+.5f}  <r_model>={np.nanmean(r_model):+.5f}", flush=True)

    # ---- per-(case,target) de-duplication weight (all-pairs safety; 1.0 for nearest-pair) -----
    kk = base["case"].to_numpy(np.int64) * 1_000_003 + base["input_index"].to_numpy(np.int64)
    _, inv, npair = np.unique(kk, return_inverse=True, return_counts=True)
    w = (1.0 / npair[inv]).astype(float)

    # ---- cells (TRUE properties) ----------------------------------------------------------------
    tmag = base["r_input_p"].to_numpy(float)
    tsize = base["Re_input_p"].to_numpy(float)
    ef = quantile_edges(tmag, args.n_flux)
    es = quantile_edges(tsize, args.n_size)
    cell = digitize(tmag, ef) * args.n_size + digitize(tsize, es)
    n_cells = args.n_flux * args.n_size

    mag0 = base["measured_mag_auto_0"].to_numpy(float)
    magg = base["measured_mag_auto_g"].to_numpy(float)
    rad0 = base["measured_flux_radius_0"].to_numpy(float)
    radg = base["measured_flux_radius_g"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lsz0 = np.log(np.where(rad0 > 0, rad0, np.nan))
        lszg = np.log(np.where(radg > 0, radg, np.nan))

    good = (np.isfinite(r_sim) & np.isfinite(r_model) & np.isfinite(mag0) & np.isfinite(lsz0)
            & np.isfinite(magg) & np.isfinite(lszg) & matched)
    print(f"rows with everything finite: {good.sum():,} / {len(good):,} ({good.mean():.2%})")

    def cell_median(values):
        med = np.full(n_cells, np.nan)
        for c in range(n_cells):
            m = good & (cell == c)
            if m.any():
                med[c] = np.median(values[m])
        return med

    med_mag = cell_median(mag0)
    med_lsz = cell_median(lsz0)
    dmag = mag0 - med_mag[cell]
    dlsz = lsz0 - med_lsz[cell]
    edm = quantile_edges(dmag[good], args.n_dmag)
    eds = quantile_edges(dlsz[good], args.n_dlogsize)
    sub = digitize(dmag, edm) * args.n_dlogsize + digitize(dlsz, eds)
    n_mb = args.n_dmag * args.n_dlogsize

    # sheared-leg-binned variant (mode 6): same edges/medians, sheared-leg values
    sub_g = (digitize(magg - med_mag[cell], edm) * args.n_dlogsize
             + digitize(lszg - med_lsz[cell], eds))

    # ---- (1) leg noise sharing ------------------------------------------------------------------
    print("\n### (1) LEG NOISE SHARING #############################################")
    for nm, a0, ag, med in (("measured mag", mag0, magg, med_mag),
                            ("measured log size", lsz0, lszg, med_lsz)):
        gd = good
        dd = ag[gd] - a0[gd]
        r0 = a0[gd] - med[cell[gd]]
        rg = ag[gd] - med[cell[gd]]
        cc = float(np.corrcoef(r0, rg)[0, 1])
        print(f"  {nm:>18}: std(leg diff)={np.std(dd):.5f}   single-leg scatter about the cell "
              f"median={np.std(r0):.5f}   ratio={np.std(dd)/max(np.std(r0),1e-12):.4f}   "
              f"corr(leg0, legG)={cc:+.5f}")
    print("  ratio << sqrt(2) and corr ~ 1  => the legs SHARE pixel noise (same seed);")
    print("  ratio ~ sqrt(2) and corr ~ 0   => independent noise. The shear-induced mag shift is")
    print("  b_mag*e*g ~ 4e-4, so anything above that is noise, not signal.")

    # ---- rho tables ------------------------------------------------------------------------------
    splits = {"FIT (case <= %d)" % args.fit_max_case: good & (base["case"].to_numpy() <= args.fit_max_case),
              "HELD OUT (case > %d)" % args.fit_max_case: good & (base["case"].to_numpy() > args.fit_max_case),
              "ALL": good}
    store = {}
    for label, msk in splits.items():
        idx = np.where(msk)[0]
        cc, ss, ww = cell[idx], sub[idx], w[idx]
        rho_s, sem_s, mb_s, _, cnt, mc_s = rho_and_sem(cc, ss, n_cells, n_mb, r_sim[idx], ww)
        rho_m, sem_m, mb_m, _, _, mc_m = rho_and_sem(cc, ss, n_cells, n_mb, r_model[idx], ww)
        # NOTE for the null and the contamination take the sem of the BIN MEAN (4th return), not the
        # sem of rho: their own cell mean mc_n / mc_c is ~0 by construction, so rho and sem(rho)
        # both blow up and are meaningless. They are put on the signal's scale by dividing by mc_s
        # below, exactly as their values are.
        rho_n, _, mb_n, semb_n, _, mc_n = rho_and_sem(cc, ss, n_cells, n_mb, r_null[idx], ww)
        rho_c, _, mb_c, semb_c, _, mc_c = rho_and_sem(cc, ss, n_cells, n_mb, r_cont[idx], ww)
        rho_gs, _, _, _, cnt_g, _ = rho_and_sem(cc, sub_g[idx], n_cells, n_mb, r_sim[idx], ww)

        ok = np.isfinite(rho_s) & np.isfinite(rho_m) & (cnt > 0) & (np.abs(rho_m) > 1e-6)
        sig = np.where(ok, rho_s / np.where(ok, rho_m, 1.0) - 1.0, np.nan)
        # propagate both sems into the ratio (they are independent estimators on the same rows,
        # so this is conservative rather than exact -- stated, not hidden)
        sig_err = np.where(ok, np.abs(rho_s / np.where(ok, rho_m, 1.0))
                           * np.sqrt((sem_s / np.where(np.abs(rho_s) > 0, rho_s, np.nan)) ** 2
                                     + (sem_m / np.where(np.abs(rho_m) > 0, rho_m, np.nan)) ** 2), np.nan)
        # controls expressed on the same (relative-to-cell) scale as rho
        # the control's per-bin DEVIATION from its own cell mean, divided by the SIGNAL's cell
        # response, so it lands on exactly the same scale as (rho_sim - 1).
        null_scale = np.where(ok, (mb_n - mc_n[:, None]) / mc_s[:, None], np.nan)
        cont_scale = np.where(ok, (mb_c - mc_c[:, None]) / mc_s[:, None], np.nan)

        rms_sig = wrms(sig, cnt, ok)
        rms_err = wrms(sig_err, cnt, ok)
        rms_null = wrms(null_scale, cnt, ok)
        rms_cont = wrms(cont_scale, cnt, ok)
        rms_mig = wrms(np.where(ok, rho_gs - rho_s, np.nan), cnt, ok)
        store[label] = dict(rho_sim=rho_s, rho_model=rho_m, sem_sim=sem_s, sem_model=sem_m,
                            counts=cnt, signal=sig, signal_err=sig_err,
                            rho_null=null_scale, rho_cont=cont_scale, rho_sim_gleg=rho_gs,
                            rms=dict(signal=rms_sig, err=rms_err, null=rms_null,
                                     cont=rms_cont, migration=rms_mig))

        print(f"\n### rho TABLE -- {label} ##############################################")
        print(f"  cells={n_cells} ({args.n_flux} true-mag x {args.n_size} true-size), "
              f"sub-bins={n_mb} ({args.n_dmag} dmag x {args.n_dlogsize} dlogsize), "
              f"N_eff={cnt[ok].sum():,.0f}, min cell-bin N_eff={cnt[ok].min():,.0f}")
        print(f"  count-weighted rms:  signal |rho_sim/rho_model - 1| = {rms_sig:.4f}")
        print(f"                       its own propagated sem          = {rms_err:.4f}  "
              f"(signal/sem = {rms_sig/max(rms_err,1e-12):.1f}x, need >5)")
        print(f"                       45-deg null                     = {rms_null:.4f}  "
              f"(signal/null = {rms_sig/max(rms_null,1e-12):.1f}x, need >3)")
        print(f"                       <e_0.ghat>/g contamination      = {rms_cont:.4f}  "
              f"(signal/cont = {rms_sig/max(rms_cont,1e-12):.1f}x, need >3)")
        print(f"                       sheared-leg binning shift       = {rms_mig:.4f}  "
              f"(the migration term; why the target is binned on the g=0 leg)")
        # per-sub-bin aggregate over cells, count weighted
        cw = np.where(ok, cnt, 0.0)
        agg = lambda a: (np.nansum(np.where(ok, a, 0.0) * cw, axis=0) / np.maximum(cw.sum(axis=0), 1e-12))
        # sem of the count-weighted aggregate: sqrt(sum_c w^2 sem^2) / sum_c w  (cells independent)
        agg_sem = lambda e: (np.sqrt(np.nansum(np.where(ok, e, 0.0) ** 2 * cw ** 2, axis=0))
                             / np.maximum(cw.sum(axis=0), 1e-12))
        a_sig, a_sigerr = agg(sig), agg_sem(sig_err)
        a_null, a_nullerr = agg(null_scale), agg_sem(np.where(ok, semb_n / mc_s[:, None], np.nan))
        store[label]["sem_null"] = np.where(ok, semb_n / mc_s[:, None], np.nan)
        store[label]["sem_cont"] = np.where(ok, semb_c / mc_s[:, None], np.nan)
        store[label]["agg_signal"] = a_sig
        store[label]["agg_signal_sem"] = a_sigerr
        store[label]["agg_rho_sim"] = agg(rho_s)
        store[label]["agg_rho_model"] = agg(rho_m)
        store[label]["agg_null"] = a_null
        store[label]["agg_null_sem"] = a_nullerr
        store[label]["agg_cont"] = agg(cont_scale)
        store[label]["agg_neff"] = cw.sum(axis=0)
        print("    sub-bin (dmag idx, dlogsize idx):  rho_sim  rho_model   sim/model-1 +- sem"
              "      null +- sem")
        for b in range(n_mb):
            print(f"      b{b:02d} (dmag {b//args.n_dlogsize}, dlsz {b%args.n_dlogsize}): "
                  f"{agg(rho_s)[b]:8.4f} {agg(rho_m)[b]:9.4f} {a_sig[b]:+10.4f} +-{a_sigerr[b]:7.4f}"
                  f"  ({a_sig[b]/max(a_sigerr[b],1e-12):+7.1f}s)  {a_null[b]:+8.4f} +-{a_nullerr[b]:6.4f}"
                  f"   N_eff={cw[:, b].sum():,.0f}")
        # EXTREME BIN = faintest measured-mag residual bin x smallest measured-log-size bin.
        # dmag = mag0 - cell median, so the LAST dmag index is the faintest; dlsz index 0 is the
        # smallest. Pre-registered requirement: this bin is DEPRESSED by > 3 sigma.
        bx = (args.n_dmag - 1) * args.n_dlogsize + 0
        store[label]["extreme_bin"] = bx
        print(f"    EXTREME (faint+small) bin b{bx:02d}: sim/model-1 = {a_sig[bx]:+.4f} "
              f"+- {a_sigerr[bx]:.4f}  -> {a_sig[bx]/max(a_sigerr[bx],1e-12):+.1f} sigma "
              f"[{'DEPRESSED >3s' if a_sig[bx] < -3*a_sigerr[bx] else 'NOT depressed >3s'}]")

    # ---- verdict ---------------------------------------------------------------------------------
    fkey = "FIT (case <= %d)" % args.fit_max_case
    hkey = "HELD OUT (case > %d)" % args.fit_max_case
    h, a = store[hkey]["rms"], store["ALL"]["rms"]
    print("\n### P0 PRE-REGISTERED CRITERIA (evaluated on ALL; held-out must pass too) ##########")
    verdict = True
    for tag, r in (("ALL", a), ("HELD OUT", h)):
        for name, need in (("signal > 5x its own sem", 5 * r["err"]),
                           ("signal > 3x the 45-deg null", 3 * r["null"]),
                           ("signal > 3x the contamination", 3 * r["cont"])):
            ok_ = r["signal"] > need
            verdict &= bool(ok_)
            print(f"  [{'PASS' if ok_ else 'FAIL'}] {tag:<9} {name:<32}  "
                  f"{r['signal']:.4f} vs {need:.4f}")
    bx = store["ALL"]["extreme_bin"]
    for tag, key in (("ALL", "ALL"), ("HELD OUT", hkey)):
        s = store[key]["agg_signal"][bx]
        e = store[key]["agg_signal_sem"][bx]
        ok_ = s < -3.0 * e
        verdict &= bool(ok_)
        print(f"  [{'PASS' if ok_ else 'FAIL'}] {tag:<9} faint/small bin b{bx:02d} depressed >3s   "
              f"{s:+.4f} +- {e:.4f} ({s/max(e,1e-12):+.1f} sigma)")
    # FIT vs HELD OUT agreement on the extreme bin: their difference must be < 3 sigma of itself.
    sf, ef = store[fkey]["agg_signal"][bx], store[fkey]["agg_signal_sem"][bx]
    sh, eh = store[hkey]["agg_signal"][bx], store[hkey]["agg_signal_sem"][bx]
    dd, de = sf - sh, float(np.hypot(ef, eh))
    ok_ = abs(dd) < 3.0 * de
    verdict &= bool(ok_)
    print(f"  [{'PASS' if ok_ else 'FAIL'}] FIT vs HELD OUT agree on b{bx:02d}          "
          f"{sf:+.4f} vs {sh:+.4f}  diff {dd:+.4f} +- {de:.4f} ({dd/max(de,1e-12):+.1f} sigma)")
    print(f"  rms signal: FIT {store[fkey]['rms']['signal']:.4f}  HELD OUT {h['signal']:.4f}  "
          f"ALL {a['signal']:.4f}")
    print(f"\n  ### P0 VERDICT: {'PASS' if verdict else 'FAIL'} ###")
    print("  If any criterion FAILS: STOP. Report blocked; do NOT fall back to a constgold target.")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        out = dict(edges_flux=ef, edges_size=es, edges_dmag=edm, edges_dlogsize=eds,
                   cell_median_mag=med_mag, cell_median_logsize=med_lsz,
                   n_flux=args.n_flux, n_size=args.n_size,
                   n_dmag=args.n_dmag, n_dlogsize=args.n_dlogsize,
                   gmed=gmed, max_case=args.max_case, fit_max_case=args.fit_max_case,
                   seeds=np.array(seed_cols), g0_leg=args.g0_leg, gS_leg=args.gS_leg)
        for label, dct in store.items():
            tag = label.split(" ")[0].lower()
            for k, v in dct.items():
                if k == "rms":
                    for kk2, vv in v.items():
                        out[f"{tag}_rms_{kk2}"] = vv
                else:
                    out[f"{tag}_{k}"] = v
        np.savez(args.output, **out)
        print(f"\nwrote {args.output}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
