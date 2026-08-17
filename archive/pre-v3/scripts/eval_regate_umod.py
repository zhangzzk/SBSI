"""CORRECTED P0 RE-GATE -- does the shear response depend on the drawn photometry residual u
inside a cell of TRUE properties?

This script executes a pre-registration JSON (schema sbsi.regate.prereg/1) VERBATIM on the FRESH
half-shear cases 40-199.  It implements no statistic, threshold, grid, edge set or anchor that is
not written in that document.  It trains nothing, fits nothing, tunes nothing and opens no
constgold file.

TWO pre-registrations are runnable, selected by --prereg:

  RA-P0-REGATE-2026-08-01e  results/regate_prereg.json    (the PARENT; frozen 2026-08-01T08:38:38Z)
  RA-P0-REGATE-2026-08-01h  results/regate_prereg_h.json  (the DEFAULT; parent + two authorised
                                                           corrections: size-threshold UNITS, and a
                                                           re-derived C2e permutation null)

Everything the two documents share is executed by the same code.  The two places they differ are
dispatched on the document, never on a flag:

  * SIZE THRESHOLDS AND THE UNITS BAND are read from the document (anchor_cuts[*].expr and
    blocking_conditions.units_check).  `measured_flux_radius_0` is SExtractor FLUX_RADIUS in
    PIXELS at 0.2 arcsec/pixel (scripts/build_constgold_measured.py line 21;
    scripts/eval_realisation_response.py PIX = 0.2; scripts/eval_selection_intrinsic.py divides an
    arcsec threshold by pixel_size before comparing).  The parent stated its size anchors in arcsec
    and applied them directly to the pixel column, so the parent still BLOCKS on its own units
    check -- which is the correct audit behaviour and is deliberately preserved.
  * C2e is the LEGACY absolute-bound form when the document's C2e block has no "n_permutations"
    key, and the re-derived permutation-p-value form when it does.  The legacy branch is byte-for-
    byte the parent's code, including its random-stream order, so the parent reproduces exactly.

scripts/eval_realisation_modulation.py (the BURNED gate on cases 0-39) is NOT modified and its
outputs are NOT overwritten.  The known `ef` rebinding bug of that script is not inherited: every
saved edge array here has a distinct name and its length is asserted before the npz is written.

READING DECISIONS forced by ambiguity in the pre-registration are collected in AMBIGUITIES below.
In every case the STRICTER reading is used for the verdict and the alternative is reported.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# SExtractor FLUX_RADIUS is in PIXELS.  This is the repo-wide plate scale, quoted here ONLY to
# print an arcsec equivalent alongside the native-unit value; no statistic multiplies by it.
# Provenance: scripts/build_constgold_measured.py:21 ("FLUX_RADIUS [pixels; x pixel_size -> arcsec]"),
# scripts/eval_realisation_response.py:117 (PIX = 0.2), scripts/eval_selection_intrinsic.py:78.
PIX_ARCSEC = 0.2

AMBIGUITIES = []


def note_ambiguity(where, text):
    AMBIGUITIES.append({"where": where, "reading": text})
    print(f"  [AMBIGUITY] {where}: {text}", flush=True)


# =================================================================================================
# pre-registration readers -- every threshold the gate applies comes from the document, not from a
# literal in this file.
# =================================================================================================
ANCHOR_COLUMNS = {"measured_mag_auto_0", "measured_flux_radius_0"}


def parse_anchor(expr):
    """'measured_flux_radius_0 > 3.0' -> ('measured_flux_radius_0', '>', 3.0).  No eval()."""
    parts = str(expr).split()
    assert len(parts) == 3, f"cannot parse anchor expression {expr!r}"
    col, op, val = parts[0], parts[1], float(parts[2])
    assert col in ANCHOR_COLUMNS, f"unknown anchor column {col!r}"
    assert op in ("<", ">"), f"unknown anchor operator {op!r}"
    return col, op, val


def parse_units_band(text):
    """Pull the admissible median band out of blocking_conditions.units_check."""
    m = re.search(r"must lie in \[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", str(text))
    assert m, f"units_check band not parseable from {text!r}"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert lo < hi, (lo, hi)
    return lo, hi


# =================================================================================================
# small numerical helpers
# =================================================================================================
def quantile_edges(values, n):
    e = np.quantile(values, np.linspace(0.0, 1.0, n + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def digitize_edges(values, edges):
    """edges is the FULL edge array (len n+1); returns 0..n-1."""
    return np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)


def digitize_interior(values, interior):
    """interior is the list of finite interior edges; returns 0..len(interior)."""
    return np.searchsorted(np.asarray(interior, float), values, side="right").astype(np.int64)


def weighted_quantile(x, w, qs):
    o = np.argsort(x, kind="mergesort")
    xs, ws = x[o], w[o]
    cum = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return np.interp(np.asarray(qs, float), cum, xs)


def group_median(values, gidx, ngroup, mask):
    med = np.full(ngroup, np.nan)
    order = np.argsort(gidx[mask], kind="mergesort")
    gg = gidx[mask][order]
    vv = values[mask][order]
    bounds = np.searchsorted(gg, np.arange(ngroup + 1))
    for g in range(ngroup):
        lo, hi = bounds[g], bounds[g + 1]
        if hi > lo:
            med[g] = np.median(vv[lo:hi])
    return med


# =================================================================================================
# accumulation:  per (case, flat-bin) weighted sums.  Everything downstream -- halves, jackknife,
# permutations, chi2, dm, F -- is a contraction of these small arrays, so the 9.5M-row pass happens
# ONCE per binning configuration.
# =================================================================================================
def accumulate(case_local, flat, ncase, nflat, w, series, cuts):
    idx = case_local * nflat + flat
    n = ncase * nflat

    def bc(v):
        return np.bincount(idx, weights=v, minlength=n).reshape(ncase, nflat)

    out = {"W": bc(w), "W2": bc(w * w)}
    for name, v in series.items():
        out["P" + name] = bc(w * v)
        out["P2" + name] = bc(w * v * v)
    cut_out = {}
    for cname, cmask in cuts.items():
        wc = w * cmask
        d = {"W": bc(wc)}
        for name, v in series.items():
            d["P" + name] = bc(wc * v)
        cut_out[cname] = d
    return out, cut_out


def cell_stats(W, W2, P, P2):
    """weighted bin means + delta-method variances, and the cell mean, for one (ncell,nsub) block."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mb = P / np.where(W > 0, W, np.nan)
        var_b = P2 / np.where(W > 0, W, np.nan) - mb ** 2
        neff = W ** 2 / np.where(W2 > 0, W2, np.nan)
        var_mb = np.maximum(var_b, 0.0) / np.where(neff > 0, neff, np.nan)
        Wc = W.sum(axis=1)
        mc = P.sum(axis=1) / np.where(Wc > 0, Wc, np.nan)
        f = W / np.where(Wc[:, None] > 0, Wc[:, None], np.nan)
        var_mc = np.nansum(f ** 2 * var_mb, axis=1)
    return mb, var_mb, neff, mc, var_mc, f


def rho_block(W, W2, P, P2):
    mb, var_mb, neff, mc, var_mc, f = cell_stats(W, W2, P, P2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = mb / mc[:, None]
        var_rho = (var_mb / mc[:, None] ** 2
                   + mb ** 2 * var_mc[:, None] / mc[:, None] ** 4
                   - 2.0 * mb * f * var_mb / mc[:, None] ** 3)
        sem_rho = np.sqrt(np.maximum(var_rho, 0.0))
    return rho, sem_rho, mb, var_mb, neff, mc, var_mc, f


def dev_block(W, W2, P, P2):
    """bin mean MINUS its cell mean, with the sem of that difference (cov term kept)."""
    mb, var_mb, neff, mc, var_mc, f = cell_stats(W, W2, P, P2)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = mb - mc[:, None]
        var_d = var_mb + var_mc[:, None] - 2.0 * f * var_mb
        sem_d = np.sqrt(np.maximum(var_d, 0.0))
    return d, sem_d, mc


# =================================================================================================
# per-sample table: retention, signal, chi2
# =================================================================================================
RET = dict(min_neff=2000.0, max_rel_sem_rho_model=0.02, min_abs_cell_R_model=0.1,
           min_one_plus_s=0.05)


def sample_table(arrs, case_mask, n_cells, n_sub):
    """Everything the pre-registration defines per SAMPLE (ALL / HALF A / HALF B)."""
    sl = lambda k: arrs[k][case_mask].sum(axis=0).reshape(n_cells, n_sub)
    W, W2 = sl("W"), sl("W2")
    rho_s, sem_s, mb_s, _, neff, mc_s, _, _ = rho_block(W, W2, sl("Psim"), sl("P2sim"))
    rho_m, sem_m, mb_m, _, _, mc_m, _, _ = rho_block(W, W2, sl("Pmod"), sl("P2mod"))
    d_n, sem_n, mc_n = dev_block(W, W2, sl("Pnull"), sl("P2null"))
    d_c, sem_c, mc_c = dev_block(W, W2, sl("Pcont"), sl("P2cont"))

    with np.errstate(invalid="ignore", divide="ignore"):
        one_plus_s = rho_s / rho_m
        s = one_plus_s - 1.0
        sig_s = np.abs(one_plus_s) * np.sqrt((sem_s / rho_s) ** 2 + (sem_m / rho_m) ** 2)
        rel_sem_m = np.abs(sem_m / rho_m)

    ok = (np.isfinite(rho_s) & np.isfinite(rho_m) & np.isfinite(s) & np.isfinite(sig_s)
          & (sig_s > 0) & (W > 0)
          & (neff >= RET["min_neff"])
          & (rel_sem_m <= RET["max_rel_sem_rho_model"])
          & (np.abs(mc_m)[:, None] >= RET["min_abs_cell_R_model"])
          & (one_plus_s >= RET["min_one_plus_s"]))

    K = int(ok.sum())
    C = int((ok.any(axis=1)).sum())
    dof = max(K - C, 1)
    z_sig = np.where(ok, s / sig_s, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_null = np.where(ok, d_n / sem_n, np.nan)
        z_cont = np.where(ok, d_c / sem_c, np.nan)
    chi2 = lambda z: float(np.nansum(np.where(ok, z, 0.0) ** 2))
    mx = lambda z: float(np.nanmax(np.abs(np.where(ok, z, np.nan)))) if K else float("nan")
    return dict(W=W, neff=neff, rho_s=rho_s, rho_m=rho_m, sem_s=sem_s, sem_m=sem_m,
                s=s, sig_s=sig_s, ok=ok, K=K, C=C, dof=dof,
                mb_s=mb_s, mb_m=mb_m, mc_s=mc_s, mc_m=mc_m, mc_n=mc_n, mc_c=mc_c,
                d_n=d_n, sem_n=sem_n, d_c=d_c, sem_c=sem_c,
                chi2_sig=chi2(z_sig), chi2_null=chi2(z_null), chi2_cont=chi2(z_cont),
                maxz_sig=mx(z_sig), maxz_null=mx(z_null), maxz_cont=mx(z_cont))


def shat_from(tab):
    """s_hat flat vector: the measured s on retained bins, 0 on dropped bins (pre-registered)."""
    return np.where(tab["ok"], np.nan_to_num(tab["s"], nan=0.0), 0.0).reshape(-1)


# =================================================================================================
# selection excess and recovery fraction
# =================================================================================================
def m_and_dm(psim_all, pmod_all, psim_cut, pmod_cut):
    m_all = psim_all / pmod_all - 1.0
    m_cut = psim_cut / pmod_cut - 1.0
    return 100.0 * m_cut, 100.0 * (m_cut - m_all)


def recovery(arrs, cutarr, case_mask, shat, n_cells, n_sub):
    """F(S) = 1 - dm_hat/dm on the cases selected by case_mask, using the given s_hat."""
    pm_bin = arrs["Pmod"][case_mask].sum(axis=0)
    ps_all = float(arrs["Psim"][case_mask].sum())
    pm_all = float(pm_bin.sum())
    pmh_all = float((pm_bin * (1.0 + shat)).sum())
    pmc_bin = cutarr["Pmod"][case_mask].sum(axis=0)
    ps_cut = float(cutarr["Psim"][case_mask].sum())
    pm_cut = float(pmc_bin.sum())
    pmh_cut = float((pmc_bin * (1.0 + shat)).sum())
    m_cut, dm = m_and_dm(ps_all, pm_all, ps_cut, pm_cut)
    mh_cut, dm_hat = m_and_dm(ps_all, pmh_all, ps_cut, pmh_cut)
    F = 1.0 - dm_hat / dm if dm != 0 else float("nan")
    return dict(F=F, dm=dm, dm_hat=dm_hat, m_cut=m_cut, m_hat_cut=mh_cut)


def recovery_vec(arrs, cutarr, case_mask, SH):
    """recovery()'s F for a STACK of s_hat vectors (SH is (n, nflat)).  Same algebra, vectorised:
    only the two contractions SH @ Pmod_bin depend on the permutation, so 999 permutations cost two
    matrix products instead of 999 passes over the per-case arrays."""
    pm_bin = arrs["Pmod"][case_mask].sum(axis=0)
    ps_all = float(arrs["Psim"][case_mask].sum())
    pm_all = float(pm_bin.sum())
    pmc_bin = cutarr["Pmod"][case_mask].sum(axis=0)
    ps_cut = float(cutarr["Psim"][case_mask].sum())
    pm_cut = float(pmc_bin.sum())
    _, dm = m_and_dm(ps_all, pm_all, ps_cut, pm_cut)
    pmh_all = pm_all + SH @ pm_bin
    pmh_cut = pm_cut + SH @ pmc_bin
    with np.errstate(invalid="ignore", divide="ignore"):
        dm_hat = 100.0 * (ps_cut / pmh_cut - ps_all / pmh_all)
        F = 1.0 - dm_hat / dm if dm != 0 else np.full(len(SH), np.nan)
    return np.asarray(F, float), dm


def permute_within_cells(base, ok, ncell, nsub, rng, variant):
    """One permutation of the b -> s_hat map INSIDE each cell.  variant 'retained' shuffles only the
    retained bins (dropped bins keep their pre-registered s_hat = 0); variant 'all' shuffles the
    whole nsub-vector."""
    sh = base.reshape(ncell, nsub).copy()
    for c in range(ncell):
        if variant == "all":
            sh[c] = rng.permutation(sh[c])
        else:
            i = np.where(ok[c])[0]
            if len(i) > 1:
                sh[c, i] = sh[c, rng.permutation(i)]
    return sh.reshape(-1)


def perm_shat_stacks(tab_A, tab_B, ncell, nsub, seeds, variant):
    """Stacks of permuted s_hat vectors for both halves.  Permutation k of variant v is driven by
    numpy.random.default_rng([seed_k, v_code]) -- an explicit, reproducible, per-(seed, variant)
    stream, so the two variants are independent draws rather than two halves of one stream."""
    vcode = {"retained": 0, "all": 1}[variant]
    bA, bB = shat_from(tab_A), shat_from(tab_B)
    SA = np.empty((len(seeds), ncell * nsub))
    SB = np.empty_like(SA)
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng([int(seed), vcode])
        SA[k] = permute_within_cells(bA, tab_A["ok"], ncell, nsub, rng, variant)
        SB[k] = permute_within_cells(bB, tab_B["ok"], ncell, nsub, rng, variant)
    return SA, SB


def permutation_pvalue(F_obs, F_perm):
    """One-sided permutation p-value with the standard +1 correction:
        p = (1 + #{F_perm >= F_obs}) / (n + 1),
    whose smallest attainable value is 1/(n+1).  Reported alongside the permutation distribution's
    own centre and spread, which is what makes the comparison scale-correct."""
    F_perm = np.asarray(F_perm, float)
    fin = np.isfinite(F_perm)
    n = int(fin.sum())
    ge = int((F_perm[fin] >= F_obs).sum())
    p = (1.0 + ge) / (n + 1.0)
    mu = float(np.mean(F_perm[fin])) if n else float("nan")
    sd = float(np.std(F_perm[fin], ddof=1)) if n > 1 else float("nan")
    z = (F_obs - mu) / sd if (sd and np.isfinite(sd) and sd > 0) else float("nan")
    q = (np.percentile(F_perm[fin], [50, 90, 99]).tolist() if n else [float("nan")] * 3)
    return dict(p=p, n_finite=n, n_ge=ge, p_floor=1.0 / (n + 1.0), mean=mu, sd=sd, z=float(z),
                q50=q[0], q90=q[1], q99=q[2],
                mean_abs=float(np.mean(np.abs(F_perm[fin]))) if n else float("nan"),
                max_abs=float(np.max(np.abs(F_perm[fin]))) if n else float("nan"))


def recovery_jackknife(arrs, cutarr, case_mask, shat):
    """delete-one-case jackknife over the cases of the EVALUATED half (s_hat held fixed)."""
    cidx = np.where(case_mask)[0]
    pm_bin = arrs["Pmod"][cidx]
    ps_all = arrs["Psim"][cidx].sum(axis=1)
    pmc_bin = cutarr["Pmod"][cidx]
    ps_cut = cutarr["Psim"][cidx].sum(axis=1)
    tot = lambda a: a.sum(axis=0)
    PA, PMB, PC, PMC = ps_all.sum(), tot(pm_bin), ps_cut.sum(), tot(pmc_bin)
    reps = []
    for k in range(len(cidx)):
        pa = PA - ps_all[k]
        pmb = PMB - pm_bin[k]
        pc = PC - ps_cut[k]
        pmc = PMC - pmc_bin[k]
        _, dm = m_and_dm(pa, pmb.sum(), pc, pmc.sum())
        _, dmh = m_and_dm(pa, float((pmb * (1 + shat)).sum()), pc, float((pmc * (1 + shat)).sum()))
        reps.append(1.0 - dmh / dm if dm != 0 else np.nan)
    reps = np.asarray(reps, float)
    n = len(reps)
    sem = float(np.sqrt((n - 1) / n * np.nansum((reps - np.nanmean(reps)) ** 2)))
    return sem, reps


def dm_jackknife(arrs, cutarr, case_mask):
    cidx = np.where(case_mask)[0]
    ps_all = arrs["Psim"][cidx].sum(axis=1)
    pm_all = arrs["Pmod"][cidx].sum(axis=1)
    ps_cut = cutarr["Psim"][cidx].sum(axis=1)
    pm_cut = cutarr["Pmod"][cidx].sum(axis=1)
    A, B, Cc, D = ps_all.sum(), pm_all.sum(), ps_cut.sum(), pm_cut.sum()
    reps = np.array([m_and_dm(A - ps_all[k], B - pm_all[k], Cc - ps_cut[k], D - pm_cut[k])[1]
                     for k in range(len(cidx))], float)
    n = len(reps)
    return float(np.sqrt((n - 1) / n * np.nansum((reps - np.nanmean(reps)) ** 2))), reps


def dm_of(arrs, cutarr, case_mask, series="sim"):
    ps_all = float(arrs["P" + series][case_mask].sum())
    pm_all = float(arrs["Pmod"][case_mask].sum())
    ps_cut = float(cutarr["P" + series][case_mask].sum())
    pm_cut = float(cutarr["Pmod"][case_mask].sum())
    return m_and_dm(ps_all, pm_all, ps_cut, pm_cut)


# =================================================================================================
def wcorr(x, y, w):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 3:
        return float("nan")
    x, y, w = x[m], y[m], w[m]
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cxy = np.average((x - mx) * (y - my), weights=w)
    cxx = np.average((x - mx) ** 2, weights=w)
    cyy = np.average((y - my) ** 2, weights=w)
    return float(cxy / np.sqrt(cxx * cyy)) if cxx > 0 and cyy > 0 else float("nan")


def wrms(x, w, mask):
    m = mask & np.isfinite(x) & np.isfinite(w)
    if not m.any() or w[m].sum() <= 0:
        return float("nan")
    return float(np.sqrt((w[m] * x[m] ** 2).sum() / w[m].sum()))


# =================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    HS = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/"
    ap.add_argument("--base-cache", default=HS + "base_c40-199.feather")
    ap.add_argument("--selfresp", default=HS + "halfshear_selfresp_c40-199.feather")
    ap.add_argument("--prereg", default=os.path.join(SBSI_ROOT, "results/regate_prereg_h.json"))
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=199)
    ap.add_argument("--out-json", default=os.path.join(SBSI_ROOT, "results/regate_result_h.json"))
    ap.add_argument("--out-npz",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/ra/regate_umod_h_c40-199.npz")
    ap.add_argument("--overwrite", action="store_true",
                    help="permit writing over an existing --out-json.  Off by default so a re-run "
                         "cannot silently destroy an audit record.")
    args = ap.parse_args()

    if os.path.exists(args.out_json) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing {args.out_json} (pass --overwrite)")

    t0 = time.time()
    pre = json.load(open(args.prereg))
    print(f"### CORRECTED P0 RE-GATE -- executing {pre['id']} (frozen {pre['frozen_at_utc']}) ###")
    print(f"### fresh cases {args.min_case}-{args.max_case}; burned cases 0-39 are NOT read ###\n")

    RESULT = {"prereg_id": pre["id"], "prereg_frozen": pre["frozen_at_utc"],
              "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "blocking": {}, "criteria": {}, "reported": {}, "ambiguities": AMBIGUITIES}
    BLOCK = []

    # ---------------------------------------------------------------------------------- load ----
    meta = json.load(open(os.path.splitext(args.base_cache)[0] + ".json"))
    assert meta["min_case"] == args.min_case and meta["max_case"] == args.max_case, meta
    NEED = ["case", "input_index", "r_input_p", "Re_input_p",
            "measured_ngmix_g1_0", "measured_ngmix_g2_0", "measured_ngmix_g1_g", "measured_ngmix_g2_g",
            "measured_mag_auto_0", "measured_mag_auto_g",
            "measured_flux_radius_0", "measured_flux_radius_g", "_ghat1", "_ghat2"]
    base = pf.read_table(args.base_cache, columns=NEED).to_pandas()
    gmed = float(meta["gmed"])
    print(f"base cache: N={len(base):,}  cases=[{meta['case_lo']},{meta['case_hi']}] "
          f"n_cases={meta['n_cases']}  g_med={gmed:.6f}  ({time.time()-t0:.0f}s)", flush=True)

    cas = base["case"].to_numpy(np.int64)
    if cas.min() < args.min_case or cas.max() > args.max_case:
        BLOCK.append(f"case_range_not_available: base spans [{cas.min()},{cas.max()}]")
    RESULT["blocking"]["case_range"] = [int(cas.min()), int(cas.max())]
    RESULT["blocking"]["n_cases"] = int(len(np.unique(cas)))
    RESULT["blocking"]["N_rows_base"] = int(len(base))

    sr = pf.read_table(args.selfresp).to_pandas()
    seed_cols = sorted(c for c in sr.columns if c.startswith("R_flow_s"))
    RESULT["blocking"]["n_seeds"] = len(seed_cols)
    RESULT["blocking"]["seeds"] = [c.replace("R_flow_s", "") for c in seed_cols]
    if len(seed_cols) < int(pre["data"]["seeds_required"]):
        BLOCK.append(f"seed_count_below_16: {len(seed_cols)}")
    want = set(str(s) for s in pre["data"]["seed_list"])
    if set(RESULT["blocking"]["seeds"]) != want:
        BLOCK.append(f"seed set mismatch: {sorted(set(RESULT['blocking']['seeds']) ^ want)}")

    # keys are expected ELEMENTWISE identical (the dump was scored from this very cache); verify
    # rather than assume, and fall back to an explicit 1:1 join if not.
    same = (len(sr) == len(base)
            and np.array_equal(sr["case"].to_numpy(np.int64), cas)
            and np.array_equal(sr["input_index"].to_numpy(np.int64),
                               base["input_index"].to_numpy(np.int64)))
    if same:
        merge_frac = 1.0
        r_dump = sr["r_sim_self"].to_numpy(float)
        # unweighted mean over the 16 seed columns, accumulated in float64 one column at a time
        # (a 16-column stack of 9.5M rows would cost ~2.4 GB of peak RSS for no benefit)
        r_model = np.zeros(len(sr))
        for c in seed_cols:
            r_model += sr[c].to_numpy(np.float64)
        r_model /= len(seed_cols)
        print("selfresp keys are ELEMENTWISE identical to the base cache -- no join needed", flush=True)
    else:
        sr["__hit"] = 1.0
        key = base[["case", "input_index"]].copy()
        key["__row"] = np.arange(len(base))
        j = key.merge(sr[["case", "input_index", "r_sim_self", "__hit"] + seed_cols],
                      on=["case", "input_index"], how="left", validate="one_to_one")
        j = j.sort_values("__row")
        merge_frac = float(np.nan_to_num(j["__hit"].to_numpy(float)).mean())
        r_dump = j["r_sim_self"].to_numpy(float)
        r_model = np.mean(np.stack([j[c].to_numpy(np.float64) for c in seed_cols], axis=1), axis=1)
        del j, key
    del sr
    RESULT["blocking"]["selfresp_merge_frac"] = merge_frac
    if merge_frac < 0.99:
        BLOCK.append(f"selfresp_merge_below_0.99: {merge_frac:.4%}")

    # ---------------------------------------------------------------------- responses & weights --
    gh1 = base["_ghat1"].to_numpy(float)
    gh2 = base["_ghat2"].to_numpy(float)
    e1_0 = base["measured_ngmix_g1_0"].to_numpy(float)
    e2_0 = base["measured_ngmix_g2_0"].to_numpy(float)
    e1_g = base["measured_ngmix_g1_g"].to_numpy(float)
    e2_g = base["measured_ngmix_g2_g"].to_numpy(float)
    r_sim = ((e1_g - e1_0) * gh1 + (e2_g - e2_0) * gh2) / gmed
    r_null = ((e1_g - e1_0) * (-gh2) + (e2_g - e2_0) * gh1) / gmed
    r_cont = (e1_0 * gh1 + e2_0 * gh2) / gmed
    dchk = np.nanmax(np.abs(r_sim - r_dump))
    RESULT["blocking"]["max_abs_diff_rsim_vs_dump"] = float(dchk)
    print(f"cross-check vs dumped r_sim_self: max|diff| = {dchk:.3e}", flush=True)
    del e1_0, e2_0, e1_g, e2_g

    kk = cas * 1_000_003 + base["input_index"].to_numpy(np.int64)
    _, inv, npair = np.unique(kk, return_inverse=True, return_counts=True)
    w = (1.0 / npair[inv]).astype(float)
    RESULT["reported"]["weight_min"] = float(w.min())
    RESULT["reported"]["weight_mean"] = float(w.mean())
    del kk, inv, npair

    mag0 = base["measured_mag_auto_0"].to_numpy(float)
    magg = base["measured_mag_auto_g"].to_numpy(float)
    rad0 = base["measured_flux_radius_0"].to_numpy(float)
    radg = base["measured_flux_radius_g"].to_numpy(float)
    tmag = base["r_input_p"].to_numpy(float)
    tsize = base["Re_input_p"].to_numpy(float)
    del base

    with np.errstate(invalid="ignore", divide="ignore"):
        lsz0 = np.log10(np.where(rad0 > 0, rad0, np.nan))
        lszg = np.log10(np.where(radg > 0, radg, np.nan))

    # UNITS.  measured_flux_radius_0 is SExtractor FLUX_RADIUS in PIXELS; the band comes from the
    # pre-registration.  The printed label says PIXELS (the parent printed "arcsec" for a pixel
    # column, which is the cosmetic half of the defect this document corrects) and carries the
    # arcsec equivalent alongside so the physical intent stays legible.
    u_lo, u_hi = parse_units_band(pre["blocking_conditions"]["units_check"])
    med_rad = float(np.nanmedian(rad0))
    RESULT["blocking"]["median_measured_flux_radius_0"] = med_rad
    RESULT["blocking"]["units_band"] = [u_lo, u_hi]
    if not (u_lo <= med_rad <= u_hi):
        BLOCK.append(f"units_check: median(measured_flux_radius_0)={med_rad:.4f} "
                     f"outside [{u_lo},{u_hi}]")
    print(f"units check: median(measured_flux_radius_0) = {med_rad:.4f} px "
          f"(= {med_rad*PIX_ARCSEC:.4f} arcsec at {PIX_ARCSEC} arcsec/px); "
          f"must be in [{u_lo}, {u_hi}] px", flush=True)

    good = (np.isfinite(r_sim) & np.isfinite(r_model) & np.isfinite(mag0) & np.isfinite(lsz0)
            & np.isfinite(magg) & np.isfinite(lszg) & np.isfinite(tmag) & np.isfinite(tsize)
            & np.isfinite(r_dump))
    print(f"rows with everything finite: {good.sum():,} / {len(good):,} ({good.mean():.4%})")
    RESULT["blocking"]["N_good"] = int(good.sum())
    RESULT["blocking"]["good_frac"] = float(good.mean())

    # ------------------------------------------------------------------------------- binning ----
    # coarse DECISION cells: 6 true-mag x 2 true-size, quantile edges.
    note_ambiguity("coarse-cell quantile sample",
                   "'np.quantile ... over the finite in-domain fresh rows' -- taken as the rows "
                   "with EVERY quantity finite (the `good` mask). The burned script used the finite "
                   "rows of the binned column alone; both edge sets are reported and compared.")
    n_tm, n_ts = pre["binning"]["coarse_cells_decision"]["n_true_mag"], \
        pre["binning"]["coarse_cells_decision"]["n_true_size"]
    e_tm = quantile_edges(tmag[good], n_tm)
    e_ts = quantile_edges(tsize[good], n_ts)
    e_tm_alt = quantile_edges(tmag[np.isfinite(tmag)], n_tm)
    e_ts_alt = quantile_edges(tsize[np.isfinite(tsize)], n_ts)
    RESULT["reported"]["coarse_edge_shift_mag"] = float(np.max(np.abs(e_tm - e_tm_alt)))
    RESULT["reported"]["coarse_edge_shift_size"] = float(np.max(np.abs(e_ts - e_ts_alt)))
    cell12 = digitize_edges(tmag, e_tm) * n_ts + digitize_edges(tsize, e_ts)
    n_cells12 = n_tm * n_ts

    # refined cells for C3c: 24 x 4
    e_tm24 = quantile_edges(tmag[good], 24)
    e_ts4 = quantile_edges(tsize[good], 4)
    cell96 = digitize_edges(tmag, e_tm24) * 4 + digitize_edges(tsize, e_ts4)
    n_cells96 = 96

    # FINE grid (12 x 6) whose medians define u
    e_tmF = quantile_edges(tmag[good], pre["binning"]["fine_grid_for_u"]["n_true_mag"])
    e_tsF = quantile_edges(tsize[good], pre["binning"]["fine_grid_for_u"]["n_true_size"])
    nfF = pre["binning"]["fine_grid_for_u"]["n_true_size"]
    fine = digitize_edges(tmag, e_tmF) * nfF + digitize_edges(tsize, e_tsF)
    n_fine = pre["binning"]["fine_grid_for_u"]["n_fine_cells"]
    med_mag_f = group_median(mag0, fine, n_fine, good)
    med_lsz_f = group_median(lsz0, fine, n_fine, good)
    u_mag = mag0 - med_mag_f[fine]
    u_lsz = lsz0 - med_lsz_f[fine]
    u_mag_g = magg - med_mag_f[fine]
    u_lsz_g = lszg - med_lsz_f[fine]

    UM_D = [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]
    UL_D = [-0.3, -0.1, 0.1, 0.3]
    UM_F = [-0.5, -0.2, 0.2, 0.5]
    UL_F = [-0.15, 0.15]
    assert len(UM_D) + 1 == pre["binning"]["u_grid_decision"]["n_u_mag"]
    assert len(UL_D) + 1 == pre["binning"]["u_grid_decision"]["n_u_lsz"]
    assert len(UM_F) + 1 == pre["binning"]["u_grid_fallback"]["n_u_mag"]
    assert len(UL_F) + 1 == pre["binning"]["u_grid_fallback"]["n_u_lsz"]

    def usub(um, ul, UM, UL):
        return digitize_interior(um, UM) * (len(UL) + 1) + digitize_interior(ul, UL)

    sub_D = usub(u_mag, u_lsz, UM_D, UL_D)
    sub_F = usub(u_mag, u_lsz, UM_F, UL_F)
    sub_Dg = usub(u_mag_g, u_lsz_g, UM_D, UL_D)
    n_sub_D, n_sub_F = 35, 15

    # PLACEBO (C3d): TRUE-property residuals about the COARSE (12) cell median, quantile edges
    # matched to the decision u-grid's marginal weight fractions.
    med_tm_c = group_median(tmag, cell12, n_cells12, good)
    ltsz = np.log10(np.where(tsize > 0, tsize, np.nan))
    med_ts_c = group_median(ltsz, cell12, n_cells12, good)
    v_mag = tmag - med_tm_c[cell12]
    v_lsz = ltsz - med_ts_c[cell12]
    wg = w[good]
    fm = np.bincount(digitize_interior(u_mag[good], UM_D), weights=wg, minlength=7)
    fl = np.bincount(digitize_interior(u_lsz[good], UL_D), weights=wg, minlength=5)
    cum_m = np.cumsum(fm / fm.sum())[:-1]
    cum_l = np.cumsum(fl / fl.sum())[:-1]
    VM = list(weighted_quantile(v_mag[good], wg, cum_m))
    VL = list(weighted_quantile(v_lsz[good], wg, cum_l))
    sub_P = usub(v_mag, v_lsz, VM, VL)
    RESULT["reported"]["placebo_edges_vmag"] = [float(x) for x in VM]
    RESULT["reported"]["placebo_edges_vlsz"] = [float(x) for x in VL]
    RESULT["reported"]["u_mag_weight_fractions"] = [float(x) for x in fm / fm.sum()]
    RESULT["reported"]["u_lsz_weight_fractions"] = [float(x) for x in fl / fl.sum()]

    # ------------------------------------------------------------------------------- anchors ----
    # Anchors come from the DOCUMENT (anchor_cuts[*].expr), not from literals here, so the two
    # pre-registrations differ only in the JSON they carry.
    COL = {"measured_mag_auto_0": mag0, "measured_flux_radius_0": rad0}
    ANCH = {k: parse_anchor(v["expr"]) for k, v in pre["anchor_cuts"].items()
            if isinstance(v, dict) and "expr" in v}
    assert set(ANCH) == {"A1", "A2", "A3", "A4"}, sorted(ANCH)
    cuts = {}
    for k, (col, op, thr) in sorted(ANCH.items()):
        cuts[k] = (COL[col] < thr) if op == "<" else (COL[col] > thr)
        kf = float((w * good * cuts[k]).sum() / (w * good).sum())
        RESULT["reported"][f"keep_frac_{k}"] = kf
        RESULT["reported"][f"anchor_{k}"] = {"column": col, "op": op, "threshold": thr}
        extra = (f"  [= {thr*PIX_ARCSEC:.3f} arcsec]" if col == "measured_flux_radius_0" else "")
        print(f"  anchor {k}: {col} {op} {thr}{extra}   keep fraction {kf:.6f}")
    cutmask = {k: (v & good).astype(float) for k, v in cuts.items()}
    A2_THR = ANCH["A2"][2]

    # ------------------------------------------------------------------------- accumulation -----
    gm = good
    case_local = (cas - args.min_case).astype(np.int64)
    ncase = args.max_case - args.min_case + 1
    series = {"sim": np.where(gm, r_sim, 0.0), "mod": np.where(gm, r_model, 0.0),
              "null": np.where(gm, r_null, 0.0), "cont": np.where(gm, r_cont, 0.0)}
    wgood = w * gm

    CONF = {}
    for name, cellidx, ncell, subidx, nsub in (
            ("decision", cell12, n_cells12, sub_D, n_sub_D),
            ("fallback12", cell12, n_cells12, sub_F, n_sub_F),
            ("fallback96", cell96, n_cells96, sub_F, n_sub_F),
            ("placebo", cell12, n_cells12, sub_P, n_sub_D),
            ("gleg", cell12, n_cells12, sub_Dg, n_sub_D)):
        flat = cellidx * nsub + subidx
        a, c = accumulate(case_local, flat, ncase, ncell * nsub, wgood, series, cutmask)
        CONF[name] = dict(arrs=a, cuts=c, ncell=ncell, nsub=nsub)
        print(f"  accumulated '{name}': {ncell} cells x {nsub} u-bins  ({time.time()-t0:.0f}s)",
              flush=True)

    all_cases = np.ones(ncase, bool)
    A_mask = np.zeros(ncase, bool)
    B_mask = np.zeros(ncase, bool)
    for c in range(ncase):
        (A_mask if (c + args.min_case) % 2 == 0 else B_mask)[c] = True
    print(f"split: HALF A = even cases ({A_mask.sum()}), HALF B = odd cases ({B_mask.sum()})")

    # ------------------------------------------------------------- coverage / grid selection ----
    def coverage(conf, case_mask):
        t = sample_table(conf["arrs"], case_mask, conf["ncell"], conf["nsub"])
        Wt = t["W"]
        return float(np.where(t["ok"], Wt, 0.0).sum() / Wt.sum()), t

    cov_all_D, tabD_all = coverage(CONF["decision"], all_cases)
    cov_A_D, tabD_A = coverage(CONF["decision"], A_mask)
    cov_B_D, tabD_B = coverage(CONF["decision"], B_mask)
    cov_all_F, tabF_all = coverage(CONF["fallback12"], all_cases)
    print(f"\nretained weight coverage -- decision grid: ALL {cov_all_D:.4f}  "
          f"A {cov_A_D:.4f}  B {cov_B_D:.4f} ;  fallback grid ALL {cov_all_F:.4f}")
    RESULT["blocking"]["coverage_decision_ALL"] = cov_all_D
    RESULT["blocking"]["coverage_decision_A"] = cov_A_D
    RESULT["blocking"]["coverage_decision_B"] = cov_B_D
    RESULT["blocking"]["coverage_fallback_ALL"] = cov_all_F
    note_ambiguity("coverage trigger sample",
                   "'retained weight coverage on the decision u-grid' carries no sample "
                   "qualifier; evaluated on ALL (primary). Half coverages are reported; if a half "
                   "falls below 0.90 while ALL does not, it is flagged, not silently ignored.")

    GRID = "decision"
    if cov_all_D < 0.90:
        GRID = "fallback12"
        if cov_all_F < 0.90:
            BLOCK.append(f"coverage_below_0.90_on_both_u_grids: {cov_all_D:.4f} / {cov_all_F:.4f}")
    RESULT["blocking"]["grid_used"] = GRID
    print(f"DECISION GRID IN FORCE: {GRID}")

    # extreme-|u_mag| retention (blocking condition)
    def extreme_retained(conf, tab, nsub, n_um, n_ul):
        okr = tab["ok"].reshape(-1, nsub)
        idx_ext = [b for b in range(nsub) if (b // n_ul) in (0, n_um - 1)]
        return int(okr[:, idx_ext].sum()), idx_ext

    n_ext_D, _ = extreme_retained(CONF["decision"], tabD_all, n_sub_D, 7, 5)
    n_ext_F, _ = extreme_retained(CONF["fallback12"], tabF_all, n_sub_F, 5, 3)
    RESULT["blocking"]["retained_extreme_umag_bins_decision"] = n_ext_D
    RESULT["blocking"]["retained_extreme_umag_bins_fallback"] = n_ext_F
    note_ambiguity("extreme-u blocking condition on the fallback grid",
                   "the fallback grid has no |u_mag|>1.0 edge; its outermost bins (|u_mag|>0.5) "
                   "CONTAIN that region and are counted as the extreme bins there.")
    if n_ext_D == 0 and n_ext_F == 0:
        BLOCK.append("extreme_u_bins_all_dropped on BOTH u-grids")

    if BLOCK:
        RESULT["verdict"] = "BLOCKED"
        RESULT["blocking"]["reasons"] = BLOCK
        print("\n### BLOCKED ###")
        for b in BLOCK:
            print("  " + b)
        json.dump(RESULT, open(args.out_json, "w"), indent=1, sort_keys=True, default=float)
        return

    # ============================================================================== CRITERIA ====
    conf = CONF[GRID]
    ncell, nsub = conf["ncell"], conf["nsub"]
    tab_all = sample_table(conf["arrs"], all_cases, ncell, nsub)
    tab_A = sample_table(conf["arrs"], A_mask, ncell, nsub)
    tab_B = sample_table(conf["arrs"], B_mask, ncell, nsub)

    crit = {}

    def rec(name, measured, threshold, ok, extra=None):
        crit[name] = dict(measured=measured, threshold=threshold, passed=bool(ok))
        if extra:
            crit[name].update(extra)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} measured {measured}   threshold {threshold}")

    print("\n### C1 EXISTENCE ####################################################")
    print(f"  ALL: retained bins K={tab_all['K']} of {ncell*nsub}, cells C={tab_all['C']}, "
          f"dof={tab_all['dof']}")
    c2n = tab_all["chi2_null"] / tab_all["dof"]
    c2c = tab_all["chi2_cont"] / tab_all["dof"]
    rec("C1a_null_chi2_over_dof", round(c2n, 4), "<= 1.30", c2n <= 1.30)
    rec("C1a_null_maxz", round(tab_all["maxz_null"], 3), "<= 5.0", tab_all["maxz_null"] <= 5.0)
    rec("C1b_cont_chi2_over_dof", round(c2c, 4), "<= 1.30", c2c <= 1.30)
    rec("C1b_cont_maxz", round(tab_all["maxz_cont"], 3), "<= 5.0", tab_all["maxz_cont"] <= 5.0)
    need_sig = 3.0 * max(1.0, c2n)
    for tag, t in (("A", tab_A), ("B", tab_B)):
        v = t["chi2_sig"] / t["dof"]
        rec(f"C1c_signal_chi2_over_dof_HALF_{tag}", round(v, 4), f">= {need_sig:.4f}", v >= need_sig,
            {"K": t["K"], "C": t["C"], "dof": t["dof"], "maxz": round(t["maxz_sig"], 3)})
    print(f"  (ALL chi2_sig/dof = {tab_all['chi2_sig']/tab_all['dof']:.4f}, "
          f"max|z|_sig = {tab_all['maxz_sig']:.2f} -- reported, not a criterion)")

    # ------------------------------------------------------------------------ recovery F --------
    shat_A = shat_from(tab_A)
    shat_B = shat_from(tab_B)

    def F_pair(conf_, shA, shB, anchors=("A1", "A2", "A3", "A4")):
        out = {}
        for a in anchors:
            rAB = recovery(conf_["arrs"], conf_["cuts"][a], B_mask, shA, conf_["ncell"], conf_["nsub"])
            rBA = recovery(conf_["arrs"], conf_["cuts"][a], A_mask, shB, conf_["ncell"], conf_["nsub"])
            out[a] = dict(AtoB=rAB, BtoA=rBA, F_bar=0.5 * (rAB["F"] + rBA["F"]))
        return out

    Fd = F_pair(conf, shat_A, shat_B)
    dm_all, dmsem = {}, {}
    for a in ("A1", "A2", "A3", "A4"):
        mcut, dm = dm_of(conf["arrs"], conf["cuts"][a], all_cases)
        sem, _ = dm_jackknife(conf["arrs"], conf["cuts"][a], all_cases)
        dm_all[a], dmsem[a] = dm, sem
        print(f"  dm({a}) = {dm:+.4f} +- {sem:.4f} pt   m({a}) = {mcut:+.4f}%")

    semF = {}
    for a in ("A1", "A2"):
        sA, _ = recovery_jackknife(conf["arrs"], conf["cuts"][a], B_mask, shat_A)
        sB, _ = recovery_jackknife(conf["arrs"], conf["cuts"][a], A_mask, shat_B)
        semF[a] = dict(AtoB=sA, BtoA=sB,
                       bar_strict=0.5 * (sA + sB), bar_indep=0.5 * float(np.hypot(sA, sB)))
    note_ambiguity("sem_jk(F_bar)",
                   "F_bar averages two directions evaluated on disjoint halves. The STRICTER "
                   "(fully-correlated) sem (sem_A+sem_B)/2 is used for C2d; the independent "
                   "combination 0.5*sqrt(semA^2+semB^2) is reported.")

    print("\n### C2 EFFECT SIZE (BINDING) ########################################")
    for a in ("A1", "A2"):
        f = Fd[a]
        print(f"  {a}: F_AtoB={f['AtoB']['F']:+.4f} (dm {f['AtoB']['dm']:+.4f} -> "
              f"{f['AtoB']['dm_hat']:+.4f})   F_BtoA={f['BtoA']['F']:+.4f} "
              f"(dm {f['BtoA']['dm']:+.4f} -> {f['BtoA']['dm_hat']:+.4f})   F_bar={f['F_bar']:+.4f}")
        rec(f"C2a_dominance_{a}", round(f["F_bar"], 4), ">= 0.50", f["F_bar"] >= 0.50)
    for a in ("A1", "A2"):
        mn = min(Fd[a]["AtoB"]["F"], Fd[a]["BtoA"]["F"])
        rec(f"C2b_both_directions_{a}", round(mn, 4), ">= 0.35", mn >= 0.35)
    rel = Fd["A2"]["F_bar"] * abs(dm_all["A2"])
    rec("C2c_absolute_relevance_A2", round(rel, 4), ">= 0.30 pt", rel >= 0.30)
    for a in ("A1", "A2"):
        v = Fd[a]["F_bar"] - 2.0 * semF[a]["bar_strict"]
        rec(f"C2d_recovery_significant_{a}", round(v, 4), "> 0",
            v > 0, {"sem_jk_strict": semF[a]["bar_strict"], "sem_jk_indep": semF[a]["bar_indep"]})

    # ---------------------------------------------------------------- C2e permutation null ------
    note_ambiguity("C2e permutation support",
                   "'permutation of the b -> s_hat map within each cell' does not say whether "
                   "dropped (s_hat=0) bins take part. BOTH variants are computed -- permuting only "
                   "among RETAINED bins, and permuting the full u-bin vector -- and the STRICTER "
                   "of the two decides.")
    C2E = pre["criteria"]["C2_effect_size_MANDATORY"]["C2e_permutation_null"]
    VARIANTS = ("retained", "all")

    if "n_permutations" in C2E:
        # -------- RE-DERIVED form (RA-P0-REGATE-2026-08-01h): compare F_bar to the permutation
        # DISTRIBUTION, which carries the correct centre AND the correct scale, instead of to an
        # absolute constant that does neither.
        n_perm = int(C2E["n_permutations"])
        s_lo, s_hi = int(C2E["seed_first"]), int(C2E["seed_last"])
        p_thr = float(C2E["p_threshold"])
        assert s_hi - s_lo + 1 == n_perm, (s_lo, s_hi, n_perm)
        seeds = np.arange(s_lo, s_hi + 1)
        p_floor = 1.0 / (n_perm + 1.0)
        assert p_floor < p_thr, f"criterion is floor-limited: 1/(n+1)={p_floor} >= {p_thr}"
        print(f"  permutation null: n={n_perm} (seeds {s_lo}..{s_hi}), p threshold {p_thr}, "
              f"resolvable floor 1/(n+1) = {p_floor:.6f}")
        permstat = {a: {} for a in ("A1", "A2")}
        for variant in VARIANTS:
            SA, SB = perm_shat_stacks(tab_A, tab_B, ncell, nsub, seeds, variant)
            for a in ("A1", "A2"):
                fAB, _ = recovery_vec(conf["arrs"], conf["cuts"][a], B_mask, SA)
                fBA, _ = recovery_vec(conf["arrs"], conf["cuts"][a], A_mask, SB)
                permstat[a][variant] = permutation_pvalue(Fd[a]["F_bar"], 0.5 * (fAB + fBA))
        for a in ("A1", "A2"):
            st = permstat[a]
            p_use = max(st["retained"]["p"], st["all"]["p"])          # stricter variant decides
            worst = "retained" if st["retained"]["p"] >= st["all"]["p"] else "all"
            for variant in VARIANTS:
                s_ = st[variant]
                print(f"    {a} [{variant:<8}] F_bar={Fd[a]['F_bar']:+.4f}  perm mean={s_['mean']:+.4f} "
                      f"sd={s_['sd']:.4f}  q99={s_['q99']:+.4f}  z={s_['z']:+.2f}  "
                      f"#>=F_bar {s_['n_ge']}/{s_['n_finite']}  p={s_['p']:.4f}")
            rec(f"C2e_perm_pvalue_{a}", round(p_use, 6), f"<= {p_thr}", p_use <= p_thr,
                {"deciding_variant": worst, "p_floor": p_floor,
                 "retained_variant": st["retained"], "all_variant": st["all"],
                 "legacy_mean_absF_retained": st["retained"]["mean_abs"],
                 "legacy_mean_absF_all": st["all"]["mean_abs"]})
    else:
        # -------- LEGACY form (RA-P0-REGATE-2026-08-01e), byte-for-byte including stream order, so
        # the parent document reproduces exactly.  Its diagnosed defect is documented in the child.
        perm = {a: {"retained": [], "all": []} for a in ("A1", "A2")}
        for seed in range(1001, 1011):
            rng = np.random.default_rng(seed)
            for variant in ("retained", "all"):
                def permute(tab):
                    sh = shat_from(tab).reshape(ncell, nsub).copy()
                    okr = tab["ok"]
                    for c in range(ncell):
                        if variant == "all":
                            sh[c] = rng.permutation(sh[c])
                        else:
                            i = np.where(okr[c])[0]
                            if len(i) > 1:
                                sh[c, i] = sh[c, rng.permutation(i)]
                    return sh.reshape(-1)
                pA, pB = permute(tab_A), permute(tab_B)
                for a in ("A1", "A2"):
                    rAB = recovery(conf["arrs"], conf["cuts"][a], B_mask, pA, ncell, nsub)
                    rBA = recovery(conf["arrs"], conf["cuts"][a], A_mask, pB, ncell, nsub)
                    perm[a][variant].append(0.5 * (rAB["F"] + rBA["F"]))
        for a in ("A1", "A2"):
            stats = {}
            for variant in ("retained", "all"):
                v = np.abs(np.asarray(perm[a][variant], float))
                stats[variant] = (float(v.mean()), float(v.max()))
            mean_use = max(stats["retained"][0], stats["all"][0])
            max_use = max(stats["retained"][1], stats["all"][1])
            rec(f"C2e_perm_mean_absF_{a}", round(mean_use, 4), "<= 0.10", mean_use <= 0.10,
                {"retained_variant": stats["retained"], "all_variant": stats["all"]})
            rec(f"C2e_perm_max_absF_{a}", round(max_use, 4), "<= 0.20", max_use <= 0.20)
    for a in ("A1", "A2"):
        v = abs(dm_all[a])
        rec(f"C2f_excess_exists_{a}", round(v, 4), f">= {3*dmsem[a]:.4f} (3 sem_jk)",
            v >= 3 * dmsem[a], {"dm": dm_all[a], "sem_jk": dmsem[a]})

    print("\n### C3 NOT TRUE-PROPERTY LEAKAGE ####################################")
    for a in ("A1", "A2"):
        d = abs(Fd[a]["AtoB"]["F"] - Fd[a]["BtoA"]["F"])
        thr = 3.0 * float(np.hypot(semF[a]["AtoB"], semF[a]["BtoA"]))
        rec(f"C3a_halves_agree_{a}", round(d, 4), f"<= {thr:.4f}", d <= thr)

    okAB = tab_A["ok"] & tab_B["ok"]
    cw = np.where(okAB, tab_A["W"] + tab_B["W"], 0.0)
    r_ab = wcorr(tab_A["s"][okAB], tab_B["s"][okAB], cw[okAB])
    rec("C3b_per_bin_corr", round(r_ab, 4), ">= 0.50", r_ab >= 0.50,
        {"n_bins_both_retained": int(okAB.sum())})
    note_ambiguity("C3b weighting",
                   "'count-weighted corr(s_A,s_B) over retained bins' -- bins retained in BOTH "
                   "halves, weighted by the summed effective weight W_A+W_B.")

    # C3c refined cells, both on the FALLBACK u-grid
    tab12F_A = sample_table(CONF["fallback12"]["arrs"], A_mask, n_cells12, n_sub_F)
    tab12F_B = sample_table(CONF["fallback12"]["arrs"], B_mask, n_cells12, n_sub_F)
    tab96F_A = sample_table(CONF["fallback96"]["arrs"], A_mask, n_cells96, n_sub_F)
    tab96F_B = sample_table(CONF["fallback96"]["arrs"], B_mask, n_cells96, n_sub_F)
    F12 = F_pair(CONF["fallback12"], shat_from(tab12F_A), shat_from(tab12F_B), ("A1", "A2"))
    F96 = F_pair(CONF["fallback96"], shat_from(tab96F_A), shat_from(tab96F_B), ("A1", "A2"))
    for a in ("A1", "A2"):
        thr = 0.70 * F12[a]["F_bar"]
        rec(f"C3c_refined_cells_{a}", round(F96[a]["F_bar"], 4),
            f">= {thr:.4f} (0.70 x F_bar[12 cells] = {F12[a]['F_bar']:.4f})",
            F96[a]["F_bar"] >= thr)

    # C3d placebo
    tabP_A = sample_table(CONF["placebo"]["arrs"], A_mask, n_cells12, n_sub_D)
    tabP_B = sample_table(CONF["placebo"]["arrs"], B_mask, n_cells12, n_sub_D)
    FP = F_pair(CONF["placebo"], shat_from(tabP_A), shat_from(tabP_B), ("A1", "A2"))
    for a in ("A1", "A2"):
        fu, fp = Fd[a]["F_bar"], FP[a]["F_bar"]
        rec(f"C3d_placebo_ratio_{a}", round(fu, 4), f">= 2 x placebo ({2*fp:.4f})", fu >= 2 * fp,
            {"F_placebo": fp})
        rec(f"C3d_placebo_gap_{a}", round(fu - fp, 4), ">= 0.25", (fu - fp) >= 0.25)

    print("\n### C4 NULLS DO NOT MANUFACTURE THE EXCESS ##########################")
    for a in ("A1", "A2"):
        for series_, tag in (("null", "C4a_45deg"), ("cont", "C4b_cont")):
            _, dmn = dm_of(conf["arrs"], conf["cuts"][a], all_cases, series=series_)
            ok = (abs(dmn) <= 0.25 * abs(dm_all[a])) and (abs(dmn) <= 0.10)
            rec(f"{tag}_dm_{a}", round(dmn, 5),
                f"<= min(0.25*|dm|={0.25*abs(dm_all[a]):.4f}, 0.10) pt", ok)

    # ============================================================================== REPORTED ====
    print("\n### REPORTED, NOT CRITERIA ##########################################")
    for nm, a0, ag in (("measured mag", mag0, magg), ("measured log10 size", lsz0, lszg)):
        dd = ag[good] - a0[good]
        r0 = a0[good] - (med_mag_f if nm.startswith("measured mag") else med_lsz_f)[fine[good]]
        cc = float(np.corrcoef(r0, ag[good] - (med_mag_f if nm.startswith("measured mag")
                                               else med_lsz_f)[fine[good]])[0, 1])
        print(f"  leg noise {nm:>20}: std(leg diff)={np.std(dd):.5f}  single-leg scatter="
              f"{np.std(r0):.5f}  ratio={np.std(dd)/max(np.std(r0),1e-12):.4f}  corr={cc:+.5f}")
        RESULT["reported"][f"legnoise_{nm.replace(' ','_')}"] = dict(
            std_diff=float(np.std(dd)), std_single=float(np.std(r0)), corr=cc)

    tabG = sample_table(CONF["gleg"]["arrs"], all_cases, n_cells12, n_sub_D)
    mig = wrms(tabG["rho_s"] - tabD_all["rho_s"], tabD_all["W"], tabD_all["ok"] & tabG["ok"])
    RESULT["reported"]["migration_rms"] = mig
    print(f"  migration term (sheared-leg vs g=0-leg binning) rms = {mig:.4f}")

    rms_mod = wrms(tab_all["rho_m"] - 1.0, tab_all["W"], tab_all["ok"])
    rms_sig = wrms(tab_all["s"], tab_all["W"], tab_all["ok"])
    RESULT["reported"]["rms_rho_model_minus_1"] = rms_mod
    RESULT["reported"]["rms_signal"] = rms_sig
    print(f"  count-weighted rms(rho_model - 1) = {rms_mod:.4f}   rms(s) = {rms_sig:.4f}")

    # the "sub-threshold size bin" of the reported block is the COMPLEMENT of anchor A2, so it
    # tracks the document's own A2 threshold instead of repeating a literal.
    for lbl, m_ in ((f"measured_size_lt_{A2_THR:g}px_{A2_THR*PIX_ARCSEC:g}arcsec",
                     (rad0 < A2_THR) & good),
                    ("measured_mag_26.0_26.5", (mag0 >= 26.0) & (mag0 < 26.5) & good)):
        ws_ = w * m_
        wa_ = w * good
        rs = float((ws_ * r_sim).sum() / ws_.sum()) / float((wa_ * r_sim).sum() / wa_.sum())
        rm = float((ws_ * r_model).sum() / ws_.sum()) / float((wa_ * r_model).sum() / wa_.sum())
        RESULT["reported"][lbl] = dict(weight_frac=float(ws_.sum() / wa_.sum()),
                                       rho_sim=rs, rho_model=rm, ratio=rs / rm)
        print(f"  ABSOLUTE bin {lbl}: frac={ws_.sum()/wa_.sum():.5f}  rho_sim={rs:.4f}  "
              f"rho_model={rm:.4f}  sim/model={rs/rm:.4f}")

    for a in ("A3", "A4"):
        RESULT["reported"][f"F_bar_{a}"] = Fd[a]["F_bar"]
        RESULT["reported"][f"dm_{a}"] = dm_all[a]
        print(f"  report-only anchor {a}: F_bar={Fd[a]['F_bar']:+.4f}  dm={dm_all[a]:+.4f} pt")

    okr = tab_all["ok"].reshape(-1, nsub)
    Wr = tab_all["W"].reshape(-1, nsub)
    ret_bins = okr.sum(axis=0).tolist()
    ret_w = (np.where(okr, Wr, 0.0).sum(axis=0) / Wr.sum()).tolist()
    RESULT["reported"]["retained_bins_per_ubin"] = [int(x) for x in ret_bins]
    RESULT["reported"]["retained_weight_per_ubin"] = [float(x) for x in ret_w]
    RESULT["reported"]["weight_abs_umag_gt_1"] = float((w * good * (np.abs(u_mag) > 1.0)).sum()
                                                       / (w * good).sum())
    RESULT["reported"]["weight_abs_ulsz_gt_0.3"] = float((w * good * (np.abs(u_lsz) > 0.30)).sum()
                                                         / (w * good).sum())
    print(f"  weight with |u_mag|>1.0 : {RESULT['reported']['weight_abs_umag_gt_1']:.5f}")
    print(f"  weight with |u_lsz|>0.30: {RESULT['reported']['weight_abs_ulsz_gt_0.3']:.5f}")
    print("  retained bins / retained weight per u-bin index:")
    for b in range(nsub):
        print(f"    b{b:02d} (u_mag {b//(n_sub_D//7 if nsub==35 else 3)}, "
              f"u_lsz {b%(5 if nsub==35 else 3)}): cells retained {ret_bins[b]:3d}/{ncell}  "
              f"weight {ret_w[b]:.5f}")

    RESULT["reported"]["prediction_check"] = {
        "dm_A1_pct": dm_all["A1"], "dm_A1_predicted": "+0.2 to +0.4",
        "dm_A2_pct": dm_all["A2"], "dm_A2_predicted": "+0.8 to +1.5"}

    # ============================================================================== VERDICT =====
    all_pass = all(v["passed"] for v in crit.values())
    RESULT["criteria"] = crit
    RESULT["verdict"] = "PASS" if all_pass else "FAIL"
    RESULT["grid_used"] = GRID
    RESULT["F_table"] = {a: {"F_AtoB": Fd[a]["AtoB"]["F"], "F_BtoA": Fd[a]["BtoA"]["F"],
                             "F_bar": Fd[a]["F_bar"], "dm": dm_all[a], "sem_jk_dm": dmsem[a]}
                         for a in ("A1", "A2", "A3", "A4")}
    RESULT["sensitivity_non_deciding"] = {
        "F_bar_12cells_fallback_ugrid": {a: F12[a]["F_bar"] for a in ("A1", "A2")},
        "F_bar_96cells_fallback_ugrid": {a: F96[a]["F_bar"] for a in ("A1", "A2")},
        "F_bar_placebo": {a: FP[a]["F_bar"] for a in ("A1", "A2")},
        "chi2_sig_over_dof_ALL": tab_all["chi2_sig"] / tab_all["dof"],
        "maxz_sig_ALL": tab_all["maxz_sig"]}
    nfail = [k for k, v in crit.items() if not v["passed"]]
    print("\n#####################################################################")
    print(f"### RE-GATE VERDICT: {RESULT['verdict']}   ({len(crit)-len(nfail)}/{len(crit)} criteria passed)")
    if nfail:
        print("### FAILED CRITERIA: " + ", ".join(nfail))
        print("### The RA build STOPS. No RA target, no warm-start screen, no m quoted.")
    print("#####################################################################")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(RESULT, open(args.out_json, "w"), indent=1, sort_keys=True, default=float)
    print(f"wrote {args.out_json}")

    # npz -- distinct names for every edge array, lengths ASSERTED (the burned-script bug)
    edges = dict(edges_true_mag_coarse=e_tm, edges_true_size_coarse=e_ts,
                 edges_true_mag_fine=e_tmF, edges_true_size_fine=e_tsF,
                 edges_true_mag_refined=e_tm24, edges_true_size_refined=e_ts4,
                 edges_u_mag_decision=np.array(UM_D), edges_u_lsz_decision=np.array(UL_D),
                 edges_u_mag_fallback=np.array(UM_F), edges_u_lsz_fallback=np.array(UL_F),
                 edges_placebo_vmag=np.array(VM), edges_placebo_vlsz=np.array(VL))
    exp = dict(edges_true_mag_coarse=n_tm + 1, edges_true_size_coarse=n_ts + 1,
               edges_true_mag_fine=13, edges_true_size_fine=7,
               edges_true_mag_refined=25, edges_true_size_refined=5,
               edges_u_mag_decision=6, edges_u_lsz_decision=4,
               edges_u_mag_fallback=4, edges_u_lsz_fallback=2,
               edges_placebo_vmag=6, edges_placebo_vlsz=4)
    for k, v in edges.items():
        assert len(np.atleast_1d(v)) == exp[k], f"edge array {k} has length {len(v)} != {exp[k]}"
    out = dict(edges)
    for tag, t in (("all", tab_all), ("halfA", tab_A), ("halfB", tab_B)):
        for k in ("rho_s", "rho_m", "sem_s", "sem_m", "s", "sig_s", "ok", "W", "neff",
                  "mc_s", "mc_m"):
            out[f"{tag}_{k}"] = t[k]
        out[f"{tag}_K"] = t["K"]
        out[f"{tag}_C"] = t["C"]
    out["shat_A"] = shat_A
    out["shat_B"] = shat_B
    out["grid_used"] = GRID
    out["gmed"] = gmed
    out["seeds"] = np.array(seed_cols)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez(args.out_npz, **out)
    print(f"wrote {args.out_npz}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
