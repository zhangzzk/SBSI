"""Decisive matched-render test of the SBSI sufficiency assumption and the analytic shear map.

The recovery shifts the conditioning truth with the analytic Mobius map S_g; the g=0 flow
learned the measured-shape response to the INTRINSIC shape. Two controlled questions, each a
candidate source of the residual m, both rendered with GalSim + HSM:

(1) MAP MISMATCH. Is GalSim's actual shear  Sersic.shear(e_int).shear(g)  the same image as the
    analytic  Sersic.shear(S_g(e_int))  ?  Compared under an ISOTROPIC PSF (so only the galaxy
    light geometry matters). A non-zero <|Δe|> means the analytic map the scan uses is not the
    sim's shear -> directly fixable by using the exact composition.

(2) PSF-ORIGIN EFFECT. Render ACTUAL sheared galaxies (intrinsic e_int, sheared by g) and measure
    the population shear response R_sim = d<e_meas·ghat>/dg. Compare to the response the analytic
    forward predicts. Do this for ISOTROPIC vs ANISOTROPIC PSF. If implied m ~ 0 for the isotropic
    PSF and grows with PSF ellipticity, the hidden variable absent from the catalogue truth is the
    PSF -> explains why no g=0 reweighting reaches sub-percent (the floor), and what to condition on.

Shapes via galsim.hsm.FindAdaptiveMom (distortion e1,e2). Relative comparisons only, so robust to
the exact shape estimator vs SExtractor.
"""

from __future__ import annotations

import argparse
import numpy as np
import galsim


def mobius(e1, e2, g1, g2):
    """Reduced-shear Mobius composition e' = (e + g)/(1 + g* e), complex spin-2."""
    e = e1 + 1j * e2
    g = g1 + 1j * g2
    ep = (e + g) / (1.0 + np.conjugate(g) * e)
    return ep.real, ep.imag


def measure_e(img):
    """Return HSM REDUCED-shear shape (g1,g2) to match the reduced-ellipticity convention used
    by the catalogue and the analytic Mobius map."""
    try:
        res = galsim.hsm.FindAdaptiveMom(img, guess_centroid=img.true_center)
        return res.observed_shape.g1, res.observed_shape.g2
    except Exception:
        return np.nan, np.nan


def make_psf(fwhm, beta, psf_g1, psf_g2):
    psf = galsim.Moffat(beta=beta, fwhm=fwhm)
    if psf_g1 != 0.0 or psf_g2 != 0.0:
        psf = psf.shear(g1=psf_g1, g2=psf_g2)  # reduced-shear slot, consistent convention
    return psf


def render_shape(eps_final, sersic_n, re, flux, psf, pix, stamp, compose_g=None, eps_int=None):
    """Render a Sersic galaxy and return HSM reduced shape. All shape inputs are REDUCED
    ellipticities applied via the g1/g2 shear slot. If compose_g is given, build the galaxy as
    Sersic.shear(eps_int).shear(g) (the true sim path); else as Sersic.shear(eps_final)."""
    gal = galsim.Sersic(n=sersic_n, half_light_radius=re, flux=flux)
    if compose_g is None:
        gal = gal.shear(g1=eps_final[0], g2=eps_final[1])
    else:
        gal = gal.shear(g1=eps_int[0], g2=eps_int[1]).shear(g1=compose_g[0], g2=compose_g[1])
    obj = galsim.Convolve([gal, psf])
    img = obj.drawImage(nx=stamp, ny=stamp, scale=pix, method="auto")
    return measure_e(img)


def render_shape_noisy(eps_int, compose_g, sersic_n, re, flux, psf, pix, stamp, snr, rng):
    """Render Sersic.shear(eps_int).shear(g), optionally add Gaussian noise to reach `snr`
    (galsim addNoiseSNR), and return HSM reduced g1. snr=None -> noiseless."""
    gal = galsim.Sersic(n=sersic_n, half_light_radius=re, flux=flux)
    gal = gal.shear(g1=eps_int[0], g2=eps_int[1]).shear(g1=compose_g[0], g2=compose_g[1])
    img = galsim.Convolve([gal, psf]).drawImage(nx=stamp, ny=stamp, scale=pix, method="auto")
    if snr is not None:
        seed = int(rng.integers(1, 2**31 - 1))
        noise = galsim.GaussianNoise(galsim.BaseDeviate(seed))
        img.addNoiseSNR(noise, snr, preserve_flux=True)
    try:
        res = galsim.hsm.FindAdaptiveMom(img, guess_centroid=img.true_center)
        return res.observed_shape.g1
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-gal", type=int, default=3000)
    ap.add_argument("--re", type=float, default=0.3)
    ap.add_argument("--flux", type=float, default=1e5)
    ap.add_argument("--pix", type=float, default=0.2)
    ap.add_argument("--stamp", type=int, default=48)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--moffat-beta", type=float, default=2.224)
    ap.add_argument("--e-rms", type=float, default=0.25, help="Rayleigh sigma of |e_int|.")
    ap.add_argument("--shears", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--psf-ell", type=float, nargs="+", default=[0.0, 0.03, 0.05],
                    help="PSF distortion magnitudes to test (applied along +e1).")
    ap.add_argument("--sersic-n", type=float, default=1.0)
    ap.add_argument("--noise-re", type=float, nargs="+", default=[0.2, 0.3, 0.5],
                    help="Sizes (arcsec) for the noise-bias response test.")
    ap.add_argument("--noise-snr", type=float, nargs="+", default=[20, 40, 100],
                    help="SNR levels for the noise-bias response test.")
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Intrinsic REDUCED ellipticities: Rayleigh magnitude, uniform orientation, capped to <0.7.
    # Antithetic pairs (eps_int and -eps_int) cancel intrinsic shape noise in the mean response.
    half = args.n_gal // 2
    emag = np.minimum(rng.rayleigh(args.e_rms, half), 0.7)
    ephi = rng.uniform(0, np.pi, half)
    e1h = emag * np.cos(2 * ephi); e2h = emag * np.sin(2 * ephi)
    e1i = np.concatenate([e1h, -e1h]); e2i = np.concatenate([e2h, -e2h])  # antithetic
    ng = len(e1i)
    print(f"GalSim {galsim.__version__}  n_gal={ng} (antithetic)  Re={args.re}\"  sersic_n={args.sersic_n}  "
          f"PSF Moffat fwhm={args.psf_fwhm}\" beta={args.moffat_beta}")
    print(f"intrinsic <|eps|>={emag.mean():.3f}  (reduced ellipticity)\n")

    # ---- (1) MAP MISMATCH under isotropic PSF: GalSim compose vs analytic Mobius render ----
    psf_iso = make_psf(args.psf_fwhm, args.moffat_beta, 0.0, 0.0)
    g_test = 0.05
    de = []
    for k in range(min(ng, 800)):
        m1, m2 = mobius(e1i[k], e2i[k], g_test, 0.0)
        a1, a2 = render_shape((m1, m2), args.sersic_n, args.re, args.flux, psf_iso, args.pix, args.stamp)
        b1, b2 = render_shape(None, args.sersic_n, args.re, args.flux, psf_iso, args.pix,
                              args.stamp, compose_g=(g_test, 0.0), eps_int=(e1i[k], e2i[k]))
        if np.isfinite(a1) and np.isfinite(b1):
            de.append(np.hypot(a1 - b1, a2 - b2))
    de = np.array(de)
    print("=== (1) analytic-Mobius vs GalSim-compose, ISOTROPIC PSF, g=0.05 (reduced convention) ===")
    print(f"  <|eps_mobius - eps_compose|> = {de.mean():.2e}  (max {de.max():.2e}, N={len(de)})")
    print(f"  -> {'MAP OK: analytic Mobius matches GalSim shear' if de.mean()<1e-3 else 'MAP MISMATCH: analytic map != sim shear (fixable!)'}\n")

    # ---- (2) PSF-ORIGIN: shear response of ACTUAL sheared renders, iso vs anisotropic PSF ----
    # Fit <g1_meas> = c + R*g over shears (incl. g=0): c = additive PSF leakage (a c-bias, handled
    # separately in shear cal), R = multiplicative response.  m comes from R only.
    print("=== (2) shear response of true sheared renders vs PSF anisotropy ===")
    print("    <g1_meas> = c + R*g.  c = additive PSF leakage (NOT m).  R = response.")
    print("    m_R = R/R_ref - 1, R_ref = isotropic-PSF response (what a PSF-blind g=0 forward")
    print("    assumes).  ghat=+e1; intrinsic mean cancels by antithetic pairs.\n")
    shears = sorted(set([0.0] + list(args.shears)))
    print(f"  {'psf_g1':>7s} | " + " ".join(f"g={g:<5g}" for g in shears) + " |   c       R     m_R")
    ref_R = None
    for pe in args.psf_ell:
        psf = make_psf(args.psf_fwhm, args.moffat_beta, pe, 0.0)
        mean_resp = []
        for g in shears:
            g1m = np.empty(ng)
            for k in range(ng):
                g1m[k], _ = render_shape(None, args.sersic_n, args.re, args.flux, psf,
                                         args.pix, args.stamp, compose_g=(g, 0.0),
                                         eps_int=(e1i[k], e2i[k]))
            mean_resp.append(np.nanmean(g1m))
        mean_resp = np.array(mean_resp)
        gs = np.array(shears)
        R, c = np.polyfit(gs, mean_resp, 1)  # slope R, intercept c
        if ref_R is None:
            ref_R = R
        row = " ".join(f"{v:6.4f}" for v in mean_resp)
        print(f"  {pe:7.3f} | {row} | {c:+6.4f} {R:6.3f} {R/ref_R-1:+.4f}")
    print("\n  If m_R grows with psf_g1 and ~0 at psf_g1=0 -> PSF anisotropy changes the multiplicative")
    print("  response (a hidden variable absent from catalogue truth -> the floor; condition on PSF to fix).")
    print("  If m_R~0 throughout, the PSF only adds a c-bias, and the m source is elsewhere.")

    # ---- (3) NOISE BIAS: does pixel noise change the multiplicative response, size-dependently? ----
    # The idealized renders above are NOISELESS. Real measurements have noise bias O(1/SNR^2). The
    # flow is trained on noisy g=0 data, so it should capture noise bias IF it is a function of the
    # FINAL true scene only. We test whether R itself shifts with noise and with size (=SNR proxy):
    # a size-dependent R(noise) that a shape-only forward folds over the population is a candidate m.
    print("\n=== (3) NOISE-BIAS: multiplicative response R vs SNR and size (isotropic PSF) ===")
    print("    Gaussian pixel noise; antithetic pairs + many gals average the response. R from")
    print("    <g1_meas>=c+R*g over shears. m_R(noise) = R_noisy/R_noiseless - 1 at each size.\n")
    print(f"  {'Re_as':>6s} {'SNR':>6s} | " + " ".join(f"g={g:<5g}" for g in shears) + " |   R     m_R")
    for re in args.noise_re:
        # noiseless reference response at this size
        psf = make_psf(args.psf_fwhm, args.moffat_beta, 0.0, 0.0)
        R0 = None
        for snr in [None] + list(args.noise_snr):
            mean_resp = []
            for g in shears:
                g1m = np.empty(ng)
                for k in range(ng):
                    g1m[k] = render_shape_noisy((e1i[k], e2i[k]), (g, 0.0), args.sersic_n, re,
                                                args.flux, psf, args.pix, args.stamp, snr,
                                                rng if snr else None)
                mean_resp.append(np.nanmean(g1m))
            R = float(np.polyfit(np.array(shears), np.array(mean_resp), 1)[0])
            if snr is None:
                R0 = R; lbl = "inf"
                continue
            row = " ".join(f"{v:6.4f}" for v in mean_resp)
            print(f"  {re:6.2f} {snr:6.0f} | {row} | {R:6.3f} {R/R0-1:+.4f}")
    print("\n  m_R != 0 (and size-dependent) under noise -> NOISE BIAS is the residual-m source: the")
    print("  shape-only forward's population-averaged response misses the size-dependent noise bias")
    print("  that shear changes. This is fixable by modelling it (joint size/SNR-aware target), not a")
    print("  fundamental floor. m_R ~ 0 -> noise bias is captured by final-scene conditioning.")


if __name__ == "__main__":
    main()
