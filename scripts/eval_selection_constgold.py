"""Confirm the SELECTION bias on constgold (constant +/-0.02 matched pairs) -- truth-only.

The user's Stage-2 framing (constgold edition):
  * different from Stage-1 self-response, use the LEG-AVERAGE response between +0.02 and -0.02
    (constgold stores both legs per row -> the symmetric derivative is exact, matches how m is
    defined and cancels odd-order / additive terms);
  * different from Stage-3 detection, PAIR the two legs first (constgold rows are inherently matched:
    same object at +g and -g), then apply the MEASURED cut SEPARATELY in each leg so the boundary
    MOVES with shear -> this isolates the pure SELECTION term from detection;
  * cut near the training/detection limit (the measured analogue of the true Re>0.3", mag<26 sample
    definition). The certified constgold catalogue stores a per-leg measured BRIGHTNESS observable,
    S/N_plus and S/N_minus (the mag<26 axis); it does NOT carry per-leg measured size, so the size
    cut here can only be the TRUE-Re null.
  * FIRST confirm the bias with INTRINSIC shapes (noise-free): the response numerator is the true
    sheared ellipticity, so the ONLY thing the measured cut can do is move the selection boundary
    -> any response shift is pure selection, uncontaminated by measurement noise. Then repeat with
    the stored MEASURED shapes for the realistic magnitude.

Estimator (leg-average, two-means selected catalogue):
  R(cut) = ( <et>_plus[pass_plus]  -  <et>_minus[pass_minus] ) / (2 g)         g = 0.02
  pass_plus / pass_minus applied to each leg's OWN measured observable -> boundary moves.

Decomposition of a bright (high-S/N) selection into two pieces:
  R_fixed  : pass = (pass_plus AND pass_minus)  -> SAME objects both legs, boundary FIXED.
             R_fixed/R_nocut - 1 = the "subpopulation" effect (bright galaxies simply respond
             differently); NOT a bias -- a calibrated analysis of that subsample would see it.
  R_moving : pass_plus / pass_minus independent -> the real observed selected catalogue.
             SELECTION BIAS  ==  R_moving / R_fixed - 1   (the pure moving-boundary term).

NULL: a cut on a TRUE (non-shearing) property (Re_input_p) has pass_plus==pass_minus by construction
      -> R_moving==R_fixed -> selection bias == 0 exactly. Built-in sanity check.

FIREWALL: nothing trains or fits. constgold is read only for validation/diagnosis.
"""
from __future__ import annotations

import argparse
import numpy as np
import pyarrow.feather as pf

CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")
COLS = ["case", "neighbored", "distance", "Re_input_p", "r_input_p",
        "axis_ratio_input_p", "position_angle_input_p",
        "applied_g1", "applied_g2", "shear_magnitude", "shear_angle",
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "et_plus", "et_minus", "response", "S/N_plus", "S/N_minus"]


def apply_shear(e1, e2, g1, g2):
    """Reduced-shear addition e' = (e + g) / (1 + g* e) for spin-2 ellipticity."""
    e = e1 + 1j * e2
    g = g1 + 1j * g2
    es = (e + g) / (1.0 + np.conj(g) * e)
    return es.real, es.imag


def et_of(e1, e2, sign, c2, s2):
    """Tangential-basis projection onto the (constant) shear direction, with a global sign locked
    to the stored `et` convention. sign*+(e1 cos2phi + e2 sin2phi)."""
    return sign * (e1 * c2 + e2 * s2)


def leg_avg(etp, etm, g, passp, passm):
    """Two-means leg-average response and mean kept-fraction."""
    np_ = int(passp.sum()); nm_ = int(passm.sum())
    if np_ == 0 or nm_ == 0:
        return np.nan, 0.0
    R = (float(np.mean(etp[passp])) - float(np.mean(etm[passm]))) / (2.0 * g)
    return R, 0.5 * (passp.mean() + passm.mean())


def true_sn_leg(mag, Re, e1i, e2i, g1, g2, psf_fwhm, ref_median=None):
    """A DETERMINISTIC 'true' S/N from true flux + shear-sheared size + PSF, per leg.

    flux ~ 10^(-0.4 mag); the galaxy is sheared (area-preserving) then PSF-convolved, and the
    sky-limited optimal S/N ~ flux / sqrt(effective area) with area ~ a_conv * b_conv. The ONLY
    shear dependence is geometric: a galaxy sheared into alignment elongates -> for a fixed
    galaxy area the PSF-convolved area a_c*b_c GROWS (min at round) -> lower true S/N. So this
    is the pure deterministic selection lever, with NO measurement noise. Overall scale is
    arbitrary (only the cut threshold matters) -> rescaled so the median matches ref_median."""
    e1s, e2s = apply_shear(e1i, e2i, g1, g2)
    es = np.hypot(e1s, e2s)
    es = np.clip(es, 0.0, 0.999)
    sigma_g = np.maximum(Re, 1e-3)                       # geometric-mean size (arcsec)
    a_g = sigma_g * np.sqrt((1 + es) / (1 - es))         # a_g * b_g = sigma_g^2 (area preserved)
    b_g = sigma_g * np.sqrt((1 - es) / (1 + es))
    sig_psf = psf_fwhm / 2.3548
    a_c = np.sqrt(a_g ** 2 + sig_psf ** 2)
    b_c = np.sqrt(b_g ** 2 + sig_psf ** 2)
    flux = 10.0 ** (-0.4 * mag)
    sn = flux / np.sqrt(a_c * b_c)
    if ref_median is not None:
        sn = sn * (ref_median / np.nanmedian(sn))
    return sn


def report_scope(name, m, etp_i, etm_i, etp_m, etm_m, sn_sources, Re, g, sn_cuts, re_null):
    """sn_sources: list of (label, snp, snm). For each we print the selection table (both shape
    kinds). The MEASURED S/N carries measurement noise on the selection variable; the TRUE S/N is
    a deterministic (noise-free) function of true flux/size/PSF -> comparing the two isolates the
    noise-driven part of the selection bias from the pure geometric part."""
    idx = np.where(m)[0]
    eip, eim = etp_i[idx], etm_i[idx]
    emp, emm = etp_m[idx], etm_m[idx]
    re = Re[idx]
    fin = np.isfinite(eip) & np.isfinite(eim) & np.isfinite(emp) & np.isfinite(emm)
    allpass = fin.copy()

    R0_i, _ = leg_avg(eip, eim, g, allpass, allpass)
    R0_m, _ = leg_avg(emp, emm, g, allpass, allpass)
    print(f"\n================ {name}  N={idx.size:,} ================")
    print(f"  NO CUT   R_intrinsic={R0_i:+.4f}   R_measured={R0_m:+.4f}")

    for sn_label, snp, snm in sn_sources:
        sp, sm = snp[idx], snm[idx]
        print(f"\n  --- selection cut on {sn_label} ---")
        print(f"  {'S/N cut':>9} {'frac':>6} | "
              f"{'--- INTRINSIC shapes ---':^34} | {'--- MEASURED shapes ---':^34}")
        print(f"  {'':>9} {'':>6} | {'R_fixed':>8} {'R_moving':>8} {'sel-bias':>9} {'subpop':>6} | "
              f"{'R_fixed':>8} {'R_moving':>8} {'sel-bias':>9} {'subpop':>6}")
        for thr in sn_cuts:
            pp = fin & (sp > thr)
            pm = fin & (sm > thr)
            both = pp & pm
            out = []
            for etp, etm, R0 in ((eip, eim, R0_i), (emp, emm, R0_m)):
                Rfix, frac = leg_avg(etp, etm, g, both, both)
                Rmov, _ = leg_avg(etp, etm, g, pp, pm)
                selbias = (Rmov / Rfix - 1.0) if Rfix else np.nan
                subpop = (Rfix / R0 - 1.0) if R0 else np.nan
                out.append((Rfix, Rmov, selbias, subpop))
            fr = 0.5 * (pp.mean() + pm.mean())
            (Rf_i, Rm_i, sb_i, su_i), (Rf_m, Rm_m, sb_m, su_m) = out
            print(f"  {'S/N>%g' % thr:>9} {fr:>6.3f} | "
                  f"{Rf_i:>8.4f} {Rm_i:>8.4f} {sb_i*100:>+8.2f}% {su_i*100:>+5.1f}% | "
                  f"{Rf_m:>8.4f} {Rm_m:>8.4f} {sb_m*100:>+8.2f}% {su_m*100:>+5.1f}%")

    print(f"\n  --- NULL (TRUE-Re cut: boundary cannot move -> sel-bias must be 0) ---")
    for thr in re_null:
        keep = fin & (re > thr)  # identical both legs
        Rf_i, frac = leg_avg(eip, eim, g, keep, keep)
        Rm_i, _ = leg_avg(eip, eim, g, keep, keep)  # same set -> identical
        print(f"  {'Re>%.2f' % thr:>9} {frac:>6.3f} | R_intr={Rf_i:+.4f}  "
              f"sel-bias={100*(Rm_i/Rf_i-1) if Rf_i else np.nan:+.3f}% (0 by construction)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat", default=CG)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--sn-cuts", type=float, nargs="+", default=[5, 7, 10, 15, 20])
    ap.add_argument("--re-null", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    ap.add_argument("--psf-fwhm", type=float, default=0.73, help="arcsec, for the true-S/N model")
    args = ap.parse_args()

    print(f"loading {args.cat}", flush=True)
    t = pf.read_table(args.cat, columns=COLS, memory_map=True).to_pandas()
    t = t[t.case >= args.min_case].reset_index(drop=True)
    g = float(np.median(t.shear_magnitude.to_numpy(float)))
    print(f"N={len(t):,}  cases={t.case.min()}..{t.case.max()}  g={g:.4f}", flush=True)

    # intrinsic ellipticity from (axis ratio, PA); shear +/- g -> intrinsic sheared shape
    q = t.axis_ratio_input_p.to_numpy(float)
    pa = np.deg2rad(t.position_angle_input_p.to_numpy(float))
    e = (1.0 - q) / (1.0 + q)
    e1i = e * np.cos(2 * pa); e2i = e * np.sin(2 * pa)
    ag1 = t.applied_g1.to_numpy(float); ag2 = t.applied_g2.to_numpy(float)
    e1p, e2p = apply_shear(e1i, e2i, ag1, ag2)
    e1m, e2m = apply_shear(e1i, e2i, -ag1, -ag2)

    sa = t.shear_angle.to_numpy(float)
    c2 = np.cos(2 * sa); s2 = np.sin(2 * sa)

    # lock the projection SIGN to the stored `et` convention using the MEASURED shapes
    me1p = t.measured_e1_plus.to_numpy(float); me2p = t.measured_e2_plus.to_numpy(float)
    me1m = t.measured_e1_minus.to_numpy(float); me2m = t.measured_e2_minus.to_numpy(float)
    my_etp = me1p * c2 + me2p * s2
    stored = t.et_plus.to_numpy(float)
    fin = np.isfinite(my_etp) & np.isfinite(stored)
    sign = 1.0 if np.nanmean((my_etp * stored)[fin]) > 0 else -1.0
    print(f"projection sign locked to stored et: sign={sign:+.0f}", flush=True)

    etp_i = et_of(e1p, e2p, sign, c2, s2); etm_i = et_of(e1m, e2m, sign, c2, s2)
    etp_m = et_of(me1p, me2p, sign, c2, s2); etm_m = et_of(me1m, me2m, sign, c2, s2)

    # cross-check: measured leg-avg should reproduce mean(response)
    finm = np.isfinite(etp_m) & np.isfinite(etm_m)
    Rchk = (np.mean(etp_m[finm]) - np.mean(etm_m[finm])) / (2 * g)
    print(f"cross-check measured R (my proj) = {Rchk:+.4f}   vs mean(response)="
          f"{np.nanmean(t.response.to_numpy(float)):+.4f}", flush=True)

    snp = t["S/N_plus"].to_numpy(float); snm = t["S/N_minus"].to_numpy(float)
    Re = t.Re_input_p.to_numpy(float)
    mag = t.r_input_p.to_numpy(float)          # true r-band magnitude (verified 20.8..27.2)
    nb = t.neighbored.to_numpy(bool)

    # deterministic TRUE S/N per leg (true flux + shear-sheared size + PSF), rescaled to the
    # measured-S/N median so cuts are comparable
    ref_med = float(np.nanmedian(snp))
    snp_t = true_sn_leg(mag, Re, e1i, e2i, ag1, ag2, args.psf_fwhm, ref_median=ref_med)
    snm_t = true_sn_leg(mag, Re, e1i, e2i, -ag1, -ag2, args.psf_fwhm, ref_median=ref_med)
    print(f"true S/N rescaled to measured median={ref_med:.2f}; "
          f"corr(true,measured S/N_plus)={np.corrcoef(snp_t, snp)[0,1]:.3f}", flush=True)

    sn_sources = [("MEASURED S/N (noisy)", snp, snm),
                  ("TRUE S/N (deterministic: flux+sheared-size+PSF)", snp_t, snm_t)]

    scopes = [("ALL objects", np.ones(len(t), bool)),
              ("ISOLATED (neighbored=False)", ~nb),
              ("BLENDED (neighbored=True)", nb)]
    for nm, mask in scopes:
        report_scope(nm, mask, etp_i, etm_i, etp_m, etm_m, sn_sources, Re, g,
                     args.sn_cuts, args.re_null)
    print("\nCONSTGOLD_SELECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
