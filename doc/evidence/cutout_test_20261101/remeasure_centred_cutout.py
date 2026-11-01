"""Remeasure ConstGold detections with the production cutout and with centred stamps, galaxy by galaxy.

Variants (all with the ngmix jacobian on the subpixel source position, as in production since 2026-07-14):
  prod48  production ``shape.cutout`` 48 px: source at 24+frac, i.e. 0.5-1.5 px towards +x/+y of the stamp centre
  cen49   odd 49 px stamp centred on the nearest pixel: source within +-0.5 px of the centre, zero mean offset
  cen65   odd 65 px stamp, centred the same way (window-size check)
Each galaxy uses one seeded RandomState for all variants (common random numbers), so the per-galaxy difference
cancels optimizer noise, pixel noise and shape noise. The fit code is BlendEMU 46107b8 ``shape.py`` copied
verbatim except for the seeded rng; BlendEMU itself is not modified or imported.
"""
import argparse
import os
from multiprocessing import Pool

import galsim
import ngmix
import numpy as np
import pandas as pd
from astropy.io import fits

TILE = "tile180.0_-0.5"
SCALE = 0.2
VARIANTS = ("prod48", "cen49", "cen65")

_IMG = None
_PSF = None


# --- BlendEMU 46107b8 blendemu/shape.py, verbatim except rng seeding ---
def ngmix_psf_correct(obs, seed):
    rng = np.random.RandomState(seed)
    Image = galsim.Image(obs.image, scale=obs.jacobian.scale)
    try:
        hsm_shape = galsim.hsm.FindAdaptiveMom(Image, strict=False)
    except Exception:
        hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
        hsm_shape = galsim.hsm.FindAdaptiveMom(Image, guess_sig=10.0, hsmparams=hsmparams, strict=False)
    error_msg = hsm_shape.error_message
    if error_msg != '':
        T_guess = 1.0
    else:
        T_guess = 2 * (hsm_shape.moments_sigma * obs.jacobian.scale) ** 2
    prior = _get_prior(rng, scale=obs.jacobian.scale)
    fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=T_guess, prior=prior)
    psf_fitter = ngmix.em.EMFitter()
    psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1)
    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=2)
    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=2)
    boot = ngmix.bootstrap.Bootstrapper(runner=runner, psf_runner=psf_runner)
    res = boot.go(obs)
    return res


def _get_prior(rng, scale):
    g_prior = ngmix.priors.GPriorBA(0.5, rng=rng)
    cen_prior = ngmix.priors.CenPrior(0.0, 0.0, scale, scale, rng=rng)
    T_prior = ngmix.priors.FlatPrior(-10.0, 1.e6, rng=rng)
    F_prior = ngmix.priors.FlatPrior(-1.e4, 1.e9, rng=rng)
    return ngmix.joint_prior.PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)


def make_obs(im, psf_im, scale, gal_cen):
    cen = np.asarray(gal_cen, dtype=float)
    psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0
    jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
    psf_jacobian = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)
    gal_noise = compute_sky(im)
    psf_noise = compute_sky(psf_im)
    if gal_noise == 0.:
        gal_noise = 1.e-6
    if psf_noise == 0.:
        psf_noise = 1.e-6
    wt = im * 0 + 1.0 / gal_noise**2
    psf_wt = psf_im * 0 + 1.0 / psf_noise**2
    psf_obs = ngmix.Observation(psf_im, weight=psf_wt, jacobian=psf_jacobian)
    return ngmix.Observation(im, weight=wt, jacobian=jacobian, psf=psf_obs)


def compute_sky(noisy_im):
    nx, ny = noisy_im.shape
    noise = np.concatenate((noisy_im[0, 0:ny], noisy_im[1:nx, -1], noisy_im[-1, 0:-1], noisy_im[1:-1, 0]))
    return 1.4826 * np.median(np.abs(noise - np.median(noise)))
# --- end of copied code ---


def stamp_and_centre(x, y, variant):
    """Stamp and 0-indexed (row, col) source position for 1-indexed detection position (x, y)."""
    if variant == "prod48":  # shape.cutout + run_shape.py subpixel centre
        n = 48
        r0, c0 = int(y - n / 2) - 1, int(x - n / 2) - 1
        cen = (y - int(y - n / 2), x - int(x - n / 2))
    else:
        n = int(variant[3:])
        h = n // 2
        r0, c0 = int(np.rint(y - 1)) - h, int(np.rint(x - 1)) - h
        cen = ((y - 1) - r0, (x - 1) - c0)
    return _IMG[r0:r0 + n, c0:c0 + n], cen, n


def measure(args):
    i, x, y, seed = args
    out = np.full(2 * len(VARIANTS) + len(VARIANTS), np.nan)
    for v, variant in enumerate(VARIANTS):
        stamp, cen, n = stamp_and_centre(x, y, variant)
        if stamp.shape != (n, n):
            continue
        try:
            res = ngmix_psf_correct(make_obs(stamp, _PSF, SCALE, cen), seed)
            out[2 * v:2 * v + 2] = res['g']
            out[2 * len(VARIANTS) + v] = res['flags']
        except Exception:
            pass
    return i, out


def main():
    global _IMG, _PSF
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N matched detections (smoke test)")
    a = ap.parse_args()
    p = os.path.join(a.case_dir, "real0")
    _IMG = fits.getdata(f"{p}/images/original/{TILE}_bandr_rot0.fits")
    _PSF = fits.getdata(f"{p}/images/original/psf_{TILE}_bandr/psf_ima.fits")
    shp = pd.read_feather(f"{p}/catalogues/Shapes/shape_catalogue_detect_position_all_{TILE}.feather",
                          columns=["X_IMAGE", "Y_IMAGE", "MAG_AUTO", "NGMIX_G1", "NGMIX_G2"])
    mat = pd.read_feather(f"{p}/catalogues/CrossMatch/{TILE}_rot0_matched.feather")
    det = mat["id_detec"].to_numpy() - 1
    df = pd.DataFrame(dict(id_detec=det + 1, id_input=mat["id_input"].to_numpy(),
                           x=shp["X_IMAGE"].to_numpy()[det], y=shp["Y_IMAGE"].to_numpy()[det],
                           mag_auto=shp["MAG_AUTO"].to_numpy()[det],
                           cat_g1=shp["NGMIX_G1"].to_numpy()[det], cat_g2=shp["NGMIX_G2"].to_numpy()[det]))
    if a.limit:
        df = df.iloc[:a.limit].copy()
    jobs = [(i, float(r.x), float(r.y), a.seed_base + int(r.id_detec)) for i, r in enumerate(df.itertuples())]
    res = np.full((len(df), 3 * len(VARIANTS)), np.nan)
    with Pool(a.workers) as pool:
        for i, out in pool.imap_unordered(measure, jobs, chunksize=256):
            res[i] = out
    for v, variant in enumerate(VARIANTS):
        df[f"{variant}_g1"] = res[:, 2 * v]
        df[f"{variant}_g2"] = res[:, 2 * v + 1]
        df[f"{variant}_flags"] = res[:, 2 * len(VARIANTS) + v]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_feather(a.out)
    print(f"{a.case_dir}: {len(df)} matched detections measured, {len(shp) - len(det)} unmatched detections skipped")


if __name__ == "__main__":
    main()
