"""Continuous native intensity of the implemented GalSBI interpolation sampler.

The measure is counting measure in type and Lebesgue measure in
``z, M_B, c[:4], R_kpc, n, a, psi``, with the fifth SED fraction dependent.
The result is per steradian, with the generator's inclusive intrinsic i-band
cut. It is not divided by parent_rate or an estimated post-cut acceptance.

This evaluates the continuous part of sample()'s numerical inverse-CDF law,
including interpolation slivers beyond the analytic luminosity cap. Finite
inverse tables can leave a failed-draw tail; sample() raises rather than retrying.
Its max(U, float.tiny) also creates a tiny singular clamp atom in the idealized
continuous-uniform law if that point is table-covered. Coverage is exposed below;
neither event is silently renormalized away. A Lebesgue density cannot represent
the clamp atom or the fully discrete floating-point random generator exactly.
No analytic-prior support mask is used, and the production prior is unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import beta, betaprime, norm

from .galsbi_prior import upper_gamma


def numerical_redshift_log_density(prior, name, z):
    """Conditional density of interp(U, cdf, z), including zero-slope gaps."""
    z = np.asarray(z, dtype=float)
    tab = prior.tables[name]
    j = np.clip(np.searchsorted(tab['z'], z, side='right')-1, 0, len(tab['z'])-2)
    slope = np.diff(tab['cdf'])[j]/np.diff(tab['z'])[j]
    valid = np.isfinite(z) & (z > tab['z'][0]) & (z <= tab['z'][-1]) & (slope > 0)
    with np.errstate(divide='ignore'):
        return np.where(valid, np.log(slope), -np.inf)


def _luminosity_parameters(prior, name, z):
    ms, _, alpha = prior.luminosity_parameters(z, name)
    cap = np.minimum(prior.magnitude_cap(z), prior.p['lum_fct_m_max'])
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        logmax = np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))
    return ms, logmax


def numerical_luminosity_log_density(prior, name, z, magnitude):
    """Conditional continuous density in M_B of the actual inverse table.

    Values outside the native redshift domain are rejected before calling the
    user's magnitude-cap function. The point-valued clamp mass is not a density.
    """
    z, magnitude = np.broadcast_arrays(np.asarray(z, float), np.asarray(magnitude, float))
    shape = z.shape
    z, magnitude = z.ravel(), magnitude.ravel()
    result = np.full(len(z), -np.inf)
    valid = np.isfinite(z) & np.isfinite(magnitude) & (z > 0) & (z <= prior.z_max[name])
    rows = np.flatnonzero(valid)
    if len(rows):
        ms, logmax = _luminosity_parameters(prior, name, z[rows])
        y = ms-magnitude[rows]
        tab = prior.tables[name]
        yy, tt = tab['inverse_y'][::-1], tab['log_g'][::-1]
        j = np.clip(np.searchsorted(yy, y, side='right')-1, 0, len(yy)-2)
        slope = np.diff(tt)[j]/np.diff(yy)[j]
        value = tt[j]+slope*(y-yy[j])
        logu = value-logmax
        inside = ((y >= yy[0]) & (y <= yy[-1]) & np.isfinite(logmax)
                  & (logu <= 0) & (logu >= np.log(np.finfo(float).tiny)))
        with np.errstate(divide='ignore'):
            result[rows[inside]] = (logu+np.log(np.abs(slope)))[inside]
    return result.reshape(shape)


def luminosity_table_coverage(prior, name, z):
    """Mass bookkeeping for finite tables under idealized continuous uniforms.

    The continuous mass plus clamp atom plus lower/upper failed-draw masses
    sums to one. The ordinary post-photometry mark intensity is deliberately
    *not* normalized by continuous_mass. z must be finite and in native support.
    """
    z = np.asarray(z, float)
    if np.any(~np.isfinite(z) | (z <= 0) | (z > prior.z_max[name])):
        raise ValueError('coverage requires finite redshifts in native support')
    ms, logmax = _luminosity_parameters(prior, name, z)
    if np.any(~np.isfinite(logmax)):
        raise ValueError('nonfinite conditional luminosity normalization')
    tab, tiny = prior.tables[name], np.finfo(float).tiny
    bottom, top = tab['log_g'][0]-logmax, tab['log_g'][-1]-logmax
    lower = np.exp(np.minimum(0, bottom))
    upper = np.exp(np.minimum(0, top))
    continuous = np.maximum(0, upper-np.maximum(tiny, lower))
    clamp_covered = (bottom <= np.log(tiny)) & (top >= np.log(tiny))
    atom = np.where(clamp_covered, tiny, 0.)
    lower_failure = np.where(bottom > np.log(tiny), lower, 0.)
    upper_failure = np.where(top < np.log(tiny), 1., -np.expm1(np.minimum(0, top)))
    clamp_magnitude = ms-np.interp(logmax+np.log(tiny), tab['log_g'], tab['inverse_y'])
    return dict(continuous_mass=continuous, clamp_atom_mass=atom,
                lower_failed_draw_mass=lower_failure, upper_failed_draw_mass=upper_failure,
                clamp_magnitude=np.where(clamp_covered, clamp_magnitude, np.nan))


def numerical_native_log_intensity(prior, data, *, return_factors=False):
    """Evaluate numerical native intensity, returning -inf for invalid rows.

    Native fields are used as supplied; derived photometry is recomputed and
    extra stored fields are ignored. There is no retry, clipping of native
    values, type marginalization, post-cut normalization, or analytic M cap.
    """
    keys = ('type', 'z', 'M_B', 'R_kpc', 'n', 'a', 'psi')
    d = {key: np.atleast_1d(np.asarray(data[key], dtype=float)) for key in keys}
    nrow = len(d['z'])
    if any(value.shape != (nrow,) for value in d.values()):
        raise ValueError('native fields must be aligned one-dimensional arrays')
    d['c'] = np.asarray(data['c'], dtype=float).reshape(nrow, 5)
    p = prior.p
    valid = (np.isin(d['type'], np.arange(len(prior.types))) & (d['z'] > 0)
             & (d['R_kpc'] > 0) & (d['n'] >= p['sersic_n_min'])
             & (d['n'] <= p['sersic_n_max']) & (d['a'] > 0)
             & (d['a'] < p['ell_beta_emax']) & (d['psi'] >= 0)
             & (d['psi'] < 2*np.pi) & (d['c'] > 0).all(axis=1)
             & np.isclose(d['c'].sum(axis=1), 1, atol=1e-10, rtol=1e-10))
    for value in d.values():
        finite = np.isfinite(value)
        valid &= finite.all(axis=1) if finite.ndim == 2 else finite
    factors = {key: np.zeros(nrow) for key in ('luminosity', 'sed', 'size', 'sersic', 'shape')}
    for index, name in enumerate(prior.types):
        valid[(d['type'] == index) & (d['z'] > prior.z_max[name])] = False
        rows = np.flatnonzero(valid & (d['type'] == index))
        if not len(rows):
            continue
        z, M = d['z'][rows], d['M_B'][rows]
        luminosity = (np.log(prior.tables[name]['rate'])
                      + numerical_redshift_log_density(prior, name, z)
                      + numerical_luminosity_log_density(prior, name, z, M))
        factors['luminosity'][rows] = luminosity
        valid[rows] &= np.isfinite(luminosity)
        rows = rows[np.isfinite(luminosity)]
        if not len(rows):
            continue
        z, M, c = d['z'][rows], d['M_B'][rows], d['c'][rows]
        photo = prior.photometry(z, M, c)
        valid[rows] &= ((photo['m_i'] >= p['gals_mag_min']) & (photo['m_i'] <= p['gals_mag_max'])
                        & np.isfinite(photo['M_i']))
        ac = prior.sed_alpha(z, name)
        factors['sed'][rows] = (gammaln(ac.sum(axis=1))-gammaln(ac).sum(axis=1)
                                + ((ac-1)*np.log(c)).sum(axis=1))
        mean, sigma = prior.size_parameters(z, photo['M_i'], name)
        R = d['R_kpc'][rows]
        factors['size'][rows] = norm.logpdf(np.log(R), mean, sigma)-np.log(R)
        a, b = prior.sersic_parameters(z, name)
        factors['sersic'][rows] = betaprime.logpdf(d['n'][rows], a, b)-prior.log_sersic_mass[name](z)
        a, b = prior.shape_parameters(name)
        factors['shape'][rows] = beta.logpdf(d['a'][rows], a, b, scale=p['ell_beta_emax'])-np.log(2*np.pi)
    total = sum(factors.values())
    total[~valid] = -np.inf
    return (total, factors) if return_factors else total
