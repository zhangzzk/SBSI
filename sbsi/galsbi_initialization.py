"""Measurement-centred *initial guesses*, never a prior or posterior proposal.

The rough PSF size subtraction and inverse shear are initialization heuristics.
They do not invert the learned measurement model. All resulting candidates
must be scored with the unchanged native prior and full forward likelihood.
"""
import numpy as np
from scipy.special import logit

from .shear_map import inverse_shear_to_ellipticity


def measurement_seed_coordinates(coordinates, measurement, chain_ids, conditions, parameters, shear, rng):
    """Replace five physical coordinates, retaining bank redshifts and SEDs.

Each of four chain streams has a different Sersic centre. Shape saturation
and positive-radius floors affect initial guesses only; subsequent MH has
the original full support. This routine reads no generating-galaxy fields.
"""
    x = np.asarray(coordinates, float)
    y, chains = np.asarray(measurement, float), np.asarray(chain_ids)
    if (x.ndim != 2 or x.shape[1] != 10 or not len(x) or not np.isfinite(x).all()
            or y.shape != (4,) or not np.isfinite(y).all() or np.any(y[2:] <= 0)
            or chains.shape != (len(x),) or not np.issubdtype(chains.dtype, np.integer)
            or np.any((chains < 0) | (chains > 3))):
        raise ValueError('finite native coordinates, four physical measurements and chain ids0..3 required')
    pixel, zp, fwhm, beta = (float(conditions[k]) for k in
                             ('pixel_size', 'zero_point', 'psf_fwhm', 'moffat_beta'))
    if not np.isfinite([pixel, zp, fwhm, beta]).all() or pixel <= 0 or fwhm < 0 or beta <= 1:
        raise ValueError('physical pixel scale and Moffat PSF required')
    nmin, nmax, amax = (parameters[k] for k in ('sersic_n_min', 'sersic_n_max', 'ell_beta_emax'))
    if not 0 < nmin < nmax or not 0 < amax <= 1:
        raise ValueError('ordered positive Sersic support and distortion bound required')
    magnitude = zp-2.5*np.log10(y[3])
    measured_radius = pixel*y[2]
    psf_radius = .5*fwhm*np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))
    radius = np.sqrt(max(measured_radius**2-psf_radius**2, (.1*measured_radius)**2))
    e1, e2 = inverse_shear_to_ellipticity(y[0], y[1], *shear)
    centre = np.array([e1, e2])
    if not np.isfinite(centre).all():
        raise ValueError('finite inverse-shear initial shape required')
    proposed = x.copy()
    proposed[:, 1] = magnitude+rng.normal(0., .6, len(x))
    proposed[:, 6] = np.log(radius)+rng.normal(0., .5, len(x))
    levels = np.clip(np.array([.75, 1.5, 2.5, 4.]), nmin+.02*(nmax-nmin), nmax-.02*(nmax-nmin))
    proposed[:, 7] = logit((levels[chains]-nmin)/(nmax-nmin))+rng.normal(0., .35, len(x))
    epsilon = centre+rng.normal(0., .2, (len(x), 2))
    amplitude = np.linalg.norm(epsilon, axis=1)
    epsmax = amax/(1+np.sqrt(1-amax**2))
    saturated = amplitude >= .98*epsmax
    epsilon *= np.minimum(1., .98*epsmax/np.maximum(amplitude, np.finfo(float).tiny))[:, None]
    amplitude = np.linalg.norm(epsilon, axis=1)
    distortion = 2*amplitude/(1+amplitude**2)
    proposed[:, 8] = logit(np.maximum(distortion/amax, 1e-12))
    proposed[:, 9] = np.mod(np.arctan2(epsilon[:, 1], epsilon[:, 0]), 2*np.pi)
    return proposed, dict(measured_magnitude=float(magnitude), measured_radius_arcsec=float(measured_radius),
        psf_half_light_radius_arcsec=float(psf_radius), initial_radius_centre_arcsec=float(radius),
        inverse_shear_shape_centre=centre.tolist(), sersic_centres=levels.tolist(),
        proposal_scatter=dict(magnitude=.6, log_radius=.5, sersic_logit=.35, epsilon_component=.2),
        shape_initial_guesses_saturated=int(saturated.sum()),
        caveat='Approximate measurement-centred starts only; no likelihood, prior or sampling support change.')
