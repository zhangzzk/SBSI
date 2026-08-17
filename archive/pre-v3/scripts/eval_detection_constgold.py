"""Quantify the DETECTION bias on constgold (constant +/-0.02 legs) -- truth-only, Stage-3.

Parallel to Stage-2 SELECTION (eval_selection_constgold.py) but the moving boundary is DETECTION
itself, not a measured-property cut:

  * constgold stores two per-leg detection catalogues, `constant_shear_catalogue_+0.02` and
    `_-0.02`, each = ALL objects DETECTED in that leg (SExtractor). Key = (case, input_index),
    unique per leg.
  * the `constant_response_catalogue` is their (case,input_index) INTERSECTION = objects detected
    in BOTH legs. That is the detection-CLEAN reference population (boundary frozen: the same
    galaxies enter the +g and -g means).
  * the REAL observed catalogue in a survey is "everything detected in THIS leg" -- which includes
    the ~0.85%/leg SINGLE-LEG objects: galaxies pushed ACROSS the detection threshold by the shear
    (a tangentially-sheared galaxy elongates -> lower peak SB -> can drop out; the antithetic leg
    keeps it). The +only and -only populations are DIFFERENT galaxies -> their mean shape differs
    -> the leg-average response is biased. That is DETECTION bias.

Estimator (leg-average two-means, g = 0.02):
  R_both = ( <et>_+[both]     - <et>_-[both]     ) / (2 g)   # frozen boundary (reference)
  R_full = ( <et>_+[all +leg] - <et>_-[all -leg] ) / (2 g)   # real observed catalogue
  DETECTION BIAS == R_full / R_both - 1

Three shape kinds (the numerator et):
  * MEASURED   : ngmix/SExtractor measured_e1/e2 projected tangentially -> the realistic magnitude.
  * SHEARED-INTRINSIC : noise-free intrinsic ellipticity (from axis_ratio,PA) reduced-shear-added
    by +/-applied_g -> confirms the bias is real and not a measurement-noise artefact.
  * UNSHEARED-INTRINSIC : the shear-FREE intrinsic shape. For both-detected objects it is identical
    in both legs -> R_both == 0 exactly; R_full != 0 isolates the PURE population asymmetry
    (which galaxies detection adds to +g vs -g), a clean "is the detected sample shape-skewed" probe.

Decomposition used for the error bar (the whole bias lives in the single-leg tails):
  <et>_+[all] - <et>_+[both] = f_+ ( <et>_+[+only] - <et>_+[both] ),  f_+ = N_+only / N_+all
  -> Delta = R_full - R_both depends only on the (small-N) single-leg means -> clean analytic SE.

Magnitude-resolved: detection bias is a faint-end effect, so we also cut on TRUE r-mag (shear-
invariant -> adds NO selection term) both in bins and cumulatively.

FIREWALL: nothing trains or fits. constgold is read only for validation/diagnosis.
"""
from __future__ import annotations

import argparse
import numpy as np
import pyarrow.feather as pf

D = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
PLUS = D + "constant_shear_catalogue_0.02_train.feather"
MINUS = D + "constant_shear_catalogue_-0.02_train.feather"
COLS = ["case", "input_index", "neighbored", "Re_input_p", "r_input_p",
        "axis_ratio_input_p", "position_angle_input_p",
        "applied_g1", "applied_g2",
        "measured_e1", "measured_e2", "S/N"]


def apply_shear(e1, e2, g1, g2):
    """Reduced-shear addition e' = (e + g) / (1 + g* e) for spin-2 ellipticity."""
    e = e1 + 1j * e2
    g = g1 + 1j * g2
    es = (e + g) / (1.0 + np.conj(g) * e)
    return es.real, es.imag


def load_leg(path):
    t = pf.read_table(path, columns=COLS, memory_map=True).to_pandas()
    d = {}
    d["case"] = t.case.to_numpy(np.int64)
    d["input_index"] = t.input_index.to_numpy(np.int64)
    d["neighbored"] = t.neighbored.to_numpy(bool)
    d["Re"] = t.Re_input_p.to_numpy(float)
    d["mag"] = t.r_input_p.to_numpy(float)
    q = t.axis_ratio_input_p.to_numpy(float)
    pa = np.deg2rad(t.position_angle_input_p.to_numpy(float))
    e = (1.0 - q) / (1.0 + q)
    d["e1i"] = e * np.cos(2 * pa)
    d["e2i"] = e * np.sin(2 * pa)
    # each file stores its OWN signed applied_g: the +file (+0.02,..), the -file (-0.02,..).
    d["g1"] = t.applied_g1.to_numpy(float)
    d["g2"] = t.applied_g2.to_numpy(float)
    d["me1"] = t.measured_e1.to_numpy(float)
    d["me2"] = t.measured_e2.to_numpy(float)
    d["sn"] = t["S/N"].to_numpy(float)
    return d


def leg_means(etp, etm, mp, mm):
    """(<et>_+[mp], <et>_-[mm], N_+, N_-) with NaN-safe means."""
    ep = etp[mp]; em = etm[mm]
    ep = ep[np.isfinite(ep)]; em = em[np.isfinite(em)]
    return ep.mean(), em.mean(), ep.size, em.size


def build_shapes(dp, dm, sign):
    """For each leg build the three projected-et flavours.

    The response is the shape component along the FIXED +g shear axis, differenced between legs.
    Since ellipticity and shear are both spin-2, the +g unit direction (cos2b, sin2b) = (g1,g2)/|g|
    for the +leg; the -leg's stored applied_g is ALREADY -(+g), so its +g axis is -(g1,g2)/|g|.
    Projecting BOTH legs onto that same +g axis makes R = (<et>_+ - <et>_-)/2g the true response
    (shear_angle -- which is per-leg and rotates 90 deg -- is deliberately NOT used).
    Sheared-intrinsic uses each file's own signed applied_g directly (no extra sign)."""
    out = {}
    for tag, d, dirsign in (("p", dp, +1.0), ("m", dm, -1.0)):
        gm = np.hypot(d["g1"], d["g2"])
        c2 = dirsign * d["g1"] / gm      # +g-axis unit vector (= cos2b, sin2b)
        s2 = dirsign * d["g2"] / gm
        e1s, e2s = apply_shear(d["e1i"], d["e2i"], d["g1"], d["g2"])
        out[f"meas_{tag}"] = sign * (d["me1"] * c2 + d["me2"] * s2)
        out[f"shint_{tag}"] = sign * (e1s * c2 + e2s * s2)
        out[f"unsh_{tag}"] = sign * (d["e1i"] * c2 + d["e2i"] * s2)
    return out


def report(name, dp, dm, sh, both_p, both_m, only_p, only_m, g, mag_bins, mag_cums, scope_p, scope_m):
    """scope_p/scope_m: boolean masks over the plus/minus leg rows (e.g. neighbored)."""
    kinds = [("MEASURED", "meas"), ("SHEARED-INTRINSIC", "shint"), ("UNSHEARED-INTRINSIC", "unsh")]
    # index sets restricted to the scope
    allp = scope_p
    allm = scope_m
    bp = both_p & scope_p
    bm = both_m & scope_m
    op = only_p & scope_p
    om = only_m & scope_m
    Np_all, Nm_all = int(allp.sum()), int(allm.sum())
    Np_both = int(bp.sum())
    Np_only, Nm_only = int(op.sum()), int(om.sum())
    f_p = Np_only / Np_all if Np_all else 0.0
    f_m = Nm_only / Nm_all if Nm_all else 0.0
    print(f"\n================ {name} ================")
    print(f"  +leg N={Np_all:,}  -leg N={Nm_all:,}  both={Np_both:,}  "
          f"+only={Np_only:,} ({100*f_p:.3f}%)  -only={Nm_only:,} ({100*f_m:.3f}%)")
    res = {"name": name, "f_only": 0.5 * (f_p + f_m)}

    def se_of(mp, mm, bpi, bmi, opi, omi, R_both):
        ep_o = sh["meas_p"][opi]; ep_o = ep_o[np.isfinite(ep_o)]
        em_o = sh["meas_m"][omi]; em_o = em_o[np.isfinite(em_o)]
        fpo = ep_o.size / max(int(mp.sum()), 1)
        fmo = em_o.size / max(int(mm.sum()), 1)
        vp = ep_o.var() / ep_o.size if ep_o.size else 0.0
        vm = em_o.var() / em_o.size if em_o.size else 0.0
        sed = np.sqrt(fpo ** 2 * vp + fmo ** 2 * vm) / (2 * g)
        return sed / abs(R_both) if R_both else np.nan

    for klabel, k in kinds:
        etp, etm = sh[f"{k}_p"], sh[f"{k}_m"]
        pF, mF, _, _ = leg_means(etp, etm, allp, allm)
        pB, mB, _, _ = leg_means(etp, etm, bp, bm)
        R_full = (pF - mF) / (2 * g)
        R_both = (pB - mB) / (2 * g)
        detbias = (R_full / R_both - 1.0) if R_both else np.nan
        ep_o = etp[op]; ep_o = ep_o[np.isfinite(ep_o)]
        em_o = etm[om]; em_o = em_o[np.isfinite(em_o)]
        var_p = ep_o.var() / ep_o.size if ep_o.size else 0.0
        var_m = em_o.var() / em_o.size if em_o.size else 0.0
        se_delta = np.sqrt(f_p ** 2 * var_p + f_m ** 2 * var_m) / (2 * g)
        se_bias = se_delta / abs(R_both) if R_both else np.nan
        res[k] = (R_both, R_full, detbias, se_bias if k != "unsh" else se_delta)
        if k == "unsh":
            print(f"  {klabel:>20}:  R_both={R_both:+.5f}(=0)  R_full={R_full:+.5f}  "
                  f"(pop-asymmetry response; +-{se_delta:.5f})")
        else:
            print(f"  {klabel:>20}:  R_both={R_both:+.4f}  R_full={R_full:+.4f}  "
                  f"det-bias={100*detbias:+.3f}% +- {100*se_bias:.3f}%")

    # magnitude-resolved (MEASURED shapes; true-mag cut = shear-invariant)
    etp, etm = sh["meas_p"], sh["meas_m"]
    print(f"  --- det-bias vs TRUE r-mag bin (MEASURED shapes) ---")
    print(f"  {'mag bin':>14} {'+only%':>7} {'R_both':>8} {'R_full':>8} {'det-bias':>9}")
    bin_lo, bin_hi, bin_db, bin_dbe, bin_fo = [], [], [], [], []
    for lo, hi in mag_bins:
        inp = (dp["mag"] >= lo) & (dp["mag"] < hi); inm = (dm["mag"] >= lo) & (dm["mag"] < hi)
        mp = allp & inp; mm = allm & inm; bpi = bp & inp; bmi = bm & inm
        opi = op & inp; omi = om & inm
        if int(mp.sum()) == 0 or int(mm.sum()) == 0:
            continue
        pF, mF, _, _ = leg_means(etp, etm, mp, mm)
        pB, mB, _, _ = leg_means(etp, etm, bpi, bmi)
        R_full = (pF - mF) / (2 * g); R_both = (pB - mB) / (2 * g)
        db = (R_full / R_both - 1) if R_both else np.nan
        dbe = se_of(mp, mm, bpi, bmi, opi, omi, R_both)
        fo = opi.sum() / mp.sum()
        bin_lo.append(lo); bin_hi.append(hi); bin_db.append(db); bin_dbe.append(dbe); bin_fo.append(fo)
        print(f"  {f'[{lo:g},{hi:g})':>14} {100*fo:>6.2f}% {R_both:>8.4f} {R_full:>8.4f} "
              f"{100*db:>+8.3f}%")
    print(f"  --- det-bias cumulative (keep TRUE mag < X) ---")
    cum_x, cum_db, cum_dbe = [], [], []
    for X in mag_cums:
        inp = dp["mag"] < X; inm = dm["mag"] < X
        mp = allp & inp; mm = allm & inm; bpi = bp & inp; bmi = bm & inm
        opi = op & inp; omi = om & inm
        if int(mp.sum()) == 0 or int(mm.sum()) == 0:
            continue
        pF, mF, _, _ = leg_means(etp, etm, mp, mm)
        pB, mB, _, _ = leg_means(etp, etm, bpi, bmi)
        R_full = (pF - mF) / (2 * g); R_both = (pB - mB) / (2 * g)
        db = (R_full / R_both - 1) if R_both else np.nan
        dbe = se_of(mp, mm, bpi, bmi, opi, omi, R_both)
        cum_x.append(X); cum_db.append(db); cum_dbe.append(dbe)
        print(f"  {'mag<%g' % X:>14} {'':>7} {R_both:>8.4f} {R_full:>8.4f} {100*db:>+8.3f}%")
    res.update(bin_lo=np.array(bin_lo), bin_hi=np.array(bin_hi), bin_db=np.array(bin_db),
               bin_dbe=np.array(bin_dbe), bin_fo=np.array(bin_fo),
               cum_x=np.array(cum_x), cum_db=np.array(cum_db), cum_dbe=np.array(cum_dbe))
    return res


def characterize(name, dp, dm, only_p, only_m, both_p, both_m):
    """Print what the single-leg (threshold) population looks like vs the both-detected reference."""
    print(f"\n  --- single-leg population profile ({name}) ---")
    for tag, d, only, both in (("+only", dp, only_p, both_p), ("-only", dm, only_m, both_m)):
        o, b = only, both
        print(f"  {tag}: <mag>={d['mag'][o].mean():.2f} (both {d['mag'][b].mean():.2f})  "
              f"<Re>={d['Re'][o].mean():.3f} (both {d['Re'][b].mean():.3f})  "
              f"<S/N>={np.nanmedian(d['sn'][o]):.2f} (both {np.nanmedian(d['sn'][b]):.2f})  "
              f"blended={100*d['neighbored'][o].mean():.1f}% (both {100*d['neighbored'][b].mean():.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plus", default=PLUS)
    ap.add_argument("--minus", default=MINUS)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print(f"loading +leg {args.plus}", flush=True)
    dp = load_leg(args.plus)
    print(f"loading -leg {args.minus}", flush=True)
    dm = load_leg(args.minus)

    # min-case filter (match the certified constgold sample definition)
    if args.min_case > 0:
        kp = dp["case"] >= args.min_case
        km = dm["case"] >= args.min_case
        dp = {k: v[kp] for k, v in dp.items()}
        dm = {k: v[km] for k, v in dm.items()}
    print(f"after min-case>={args.min_case}: +leg {dp['case'].size:,}  -leg {dm['case'].size:,}",
          flush=True)

    g = float(np.median(np.hypot(dp["g1"], dp["g2"])))
    print(f"g = {g:.4f}", flush=True)

    # match on (case, input_index)
    BIG = np.int64(1) << 34
    keyp = dp["case"] * BIG + dp["input_index"]
    keym = dm["case"] * BIG + dm["input_index"]
    common, ip_idx, im_idx = np.intersect1d(keyp, keym, assume_unique=True, return_indices=True)
    both_p = np.zeros(keyp.size, bool); both_p[ip_idx] = True
    both_m = np.zeros(keym.size, bool); both_m[im_idx] = True
    only_p = ~both_p
    only_m = ~both_m
    print(f"both-detected pairs={common.size:,}  +only={int(only_p.sum()):,}  "
          f"-only={int(only_m.sum()):,}", flush=True)

    # lock projection sign so R_both(measured) > 0 (matches the stored `response` convention).
    # sign is irrelevant to the det-bias ratio; used only so R prints positive.
    sh0 = build_shapes(dp, dm, 1.0)
    R_try = (np.nanmean(sh0["meas_p"][both_p]) - np.nanmean(sh0["meas_m"][both_m])) / (2 * g)
    sign = 1.0 if R_try > 0 else -1.0
    R_shint = (np.nanmean(sh0["shint_p"][both_p]) - np.nanmean(sh0["shint_m"][both_m])) / (2 * g)
    print(f"projection sign={sign:+.0f} (R_both measured trial={R_try:+.4f}, "
          f"sheared-intrinsic trial={R_shint:+.4f}) "
          f"[expect ~+0.45 measured, ~+1.0 intrinsic]", flush=True)

    sh = build_shapes(dp, dm, sign) if sign < 0 else sh0

    mag_bins = [(20, 24), (24, 25), (25, 25.5), (25.5, 26), (26, 26.5), (26.5, 27), (27, 30)]
    mag_cums = [24, 25, 25.5, 26, 26.5, 27, 30]

    scopes = [("ALL", np.ones(dp["case"].size, bool), np.ones(dm["case"].size, bool)),
              ("ISOLATED", ~dp["neighbored"], ~dm["neighbored"]),
              ("BLENDED", dp["neighbored"], dm["neighbored"])]
    results = {}
    for nm, sp, sm in scopes:
        results[nm] = report(nm, dp, dm, sh, both_p, both_m, only_p, only_m, g,
                             mag_bins, mag_cums, sp, sm)
    characterize("ALL", dp, dm, only_p, only_m, both_p, both_m)

    if args.output:
        out = dict(g=g, Np_all=dp["case"].size, Nm_all=dm["case"].size, Nboth=common.size,
                   Nplus_only=int(only_p.sum()), Nminus_only=int(only_m.sum()),
                   scopes=np.array(list(results.keys())))
        for nm, r in results.items():
            for kk in ("meas", "shint", "unsh"):
                out[f"{nm}_{kk}"] = np.array(r[kk], float)   # (R_both, R_full, detbias, se)
            out[f"{nm}_f_only"] = r["f_only"]
            for kk in ("bin_lo", "bin_hi", "bin_db", "bin_dbe", "bin_fo",
                       "cum_x", "cum_db", "cum_dbe"):
                out[f"{nm}_{kk}"] = r[kk]
        np.savez(args.output, **out)
        print(f"saved {args.output}", flush=True)
    print("\nDETECTION_CONSTGOLD_DONE", flush=True)


if __name__ == "__main__":
    main()
