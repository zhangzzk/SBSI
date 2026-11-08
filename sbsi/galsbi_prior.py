"""Experimental isolated-galaxy density for the GalSBI Fischbacher+24 model.

The measure is counting measure for ``type`` and Lebesgue measure in
``z, M_B, c[0:4], R_kpc, n, a, psi``; c[4] = 1-sum(c[0:4]).
``a`` is GalSBI's native distortion amplitude, not SBSI's input ellipticity.
There are no neighbours, measurement selection, or fitted density corrections.
GalSBI supplies photometry/cosmology through explicit callables, not imports.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import exp1, gamma, gammaincc, gammaln
from scipy.stats import beta, betaprime, norm


def upper_gamma(shape, x):
    """Upper incomplete gamma, including the negative Schechter shape."""
    x = np.asarray(x, dtype=float)
    if shape > 0:
        return gamma(shape) * gammaincc(shape, x)
    if shape == 0:
        return exp1(x)
    return (upper_gamma(shape + 1, x) - np.exp(shape * np.log(x) - x)) / shape


class GalSBIContinuousPrior:
    """Fixed-parameter, native-coordinate prior with explicit support.

    ``photometry(z, M_B, c)`` returns arrays ``m_r``, ``m_i``, ``M_i``.
    ``volume(z)`` is dV/(dz dOmega) in Mpc^3/sr.
    ``magnitude_cap(z)`` and ``z_max[type]`` reproduce the generator's
    computational population envelope. The reference-band i limits remain
    population limits, never SBSI measured cuts.

    log_intensity is unnormalized per steradian. log_prob requires an explicit
    surface-density normalization; sample below reports its
    Monte Carlo error instead of silently treating it as exact.
    """

    def __init__(self, parameters, photometry, volume, magnitude_cap, z_max,
                 *, grid_size=8193, magnitude_step=0.001):
        self.p = dict(parameters)
        self.photometry, self.volume = photometry, volume
        self.magnitude_cap = magnitude_cap
        self.types = tuple(self.p['galaxy_types'])
        self.z_max = dict(z_max)
        self.tables = {}
        if grid_size < 33 or not 0 < magnitude_step <= .1:
            raise ValueError('invalid continuous sampling grid')
        required = {'lum_fct_parametrization': 'truncated_logexp',
                    'template_coeff_sampler': 'dirichlet_alpha_mode',
                    'logr50_sampling_method': 'sdss_fit',
                    'ellipticity_sampling_method': 'beta_mode_red_blue',
                    'sersic_sampling_method': 'blue_red_betaprime',
                    'reference_band': 'i', 'lum_fct_filter_band': 'B'}
        if any(self.p[k] != v for k, v in required.items()):
            raise ValueError('this adapter supports only the declared F24 parametrization')
        for t in self.types:
            if not np.array_equal(self.p[f'template_coeff_weight_{t}'], np.ones(5)):
                raise ValueError('only unweighted five-template Dirichlet supported')
            z = np.linspace(0, self.z_max[t], grid_size)
            mstar, phistar, alpha = self.luminosity_parameters(z, t)
            cap = np.minimum(self.magnitude_cap(z), self.p['lum_fct_m_max'])
            density = self.volume(z) * phistar * upper_gamma(
                alpha + 1, 10 ** (.4 * (mstar - cap)))
            cdf = np.r_[0., cumulative_trapezoid(density, z)]
            rate = float(cdf[-1])
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError('invalid luminosity intensity')
            y = np.arange(-40., 9., magnitude_step)
            g = upper_gamma(alpha + 1, 10 ** (.4 * y))
            keep = (g > np.finfo(float).tiny) & np.isfinite(g)
            log_g, indices = np.unique(np.log(g[keep])[::-1], return_index=True)
            inverse_y = y[keep][::-1][indices]
            self.tables[t] = dict(z=z, cdf=cdf/rate, rate=rate,
                                  log_g=log_g, inverse_y=inverse_y)
        self.parent_rate = sum(v['rate'] for v in self.tables.values())

    def luminosity_parameters(self, z, t):
        p = self.p
        mstar = (p[f'lum_fct_m_star_{t}_intcpt']
                 + p[f'lum_fct_m_star_{t}_slope']
                 * np.log10(1 + np.minimum(z, p[f'lum_fct_z_const_{t}'])))
        phistar = p[f'lum_fct_phi_star_{t}_amp'] * np.exp(p[f'lum_fct_phi_star_{t}_exp'] * z)
        return mstar, phistar, p[f'lum_fct_alpha_{t}']

    def sed_alpha(self, z, t):
        p = self.p
        f = np.asarray(z)[:, None] / p[f'template_coeff_z1_{t}']
        a0 = np.array([p[f'template_coeff_alpha0_{t}_{i}'] for i in range(5)])
        a1 = np.array([p[f'template_coeff_alpha1_{t}_{i}'] for i in range(5)])
        mode = a0 ** (1-f) * a1 ** f
        mode /= mode.sum(axis=1, keepdims=True)
        s0, s1 = p[f'template_coeff_alpha0_{t}_std'], p[f'template_coeff_alpha1_{t}_std']
        sigma = np.clip(s0 ** (1-f) * s1 ** f, 1e-4, np.sqrt((1/5-1/25)/6)-1e-4)
        return 1 + mode * ((1/5-1/25)/sigma**2 - 6)

    def size_parameters(self, z, M_i, t):
        p = self.p
        m0 = p[f'logr50_sdss_fit_M0_{t}']
        s1, s2 = p[f'logr50_sdss_fit_sigma1_{t}'], p[f'logr50_sdss_fit_sigma2_{t}']
        sigma = s2 + (s1-s2)/(1 + 10 ** (-.8*(M_i-m0)))
        if t == 'red':
            mean = -.4*p['logr50_sdss_fit_a_red']*M_i + p['logr50_sdss_fit_b_red']
        else:
            a, b = p['logr50_sdss_fit_alpha_blue'], p['logr50_sdss_fit_beta_blue']
            mean = (-.4*a*M_i + (b-a)*np.log10(1 + 10 ** (-.4*(M_i-m0)))
                    + p['logr50_sdss_fit_gamma_blue'])
        return mean + p[f'logr50_alpha_{t}']*np.log1p(z), sigma

    def sersic_parameters(self, z, t):
        mode = self.p[f'sersic_betaprime_{t}_mode'] * (1+z) ** self.p[
            f'sersic_betaprime_{t}_mode_alpha']
        b = self.p[f'sersic_betaprime_{t}_size']
        return mode*(b+1)+1, b

    def shape_parameters(self, t):
        total, mode = self.p[f'ell_beta_ab_sum_{t}'], self.p[f'ell_beta_mode_{t}']
        a = total*mode-2*mode+1
        return a, total-a

    def log_intensity(self, data, *, return_factors=False):
        """Evaluate arbitrary native latent galaxies, returning -inf off support."""
        keys = ('type', 'z', 'M_B', 'R_kpc', 'n', 'a', 'psi')
        d = {k: np.atleast_1d(np.asarray(data[k])) for k in keys}
        nrow = len(d['z'])
        d['c'] = np.asarray(data['c'], dtype=float).reshape(nrow, 5)
        if any(v.shape != (nrow,) for k, v in d.items() if k != 'c'):
            raise ValueError('native fields must be aligned one-dimensional arrays')
        p = self.p
        valid = (np.isin(d['type'], np.arange(len(self.types))) & (d['z'] > 0)
                 & (d['R_kpc'] > 0) & (d['n'] >= p['sersic_n_min'])
                 & (d['n'] <= p['sersic_n_max']) & (d['a'] > 0)
                 & (d['a'] < p['ell_beta_emax']) & (d['psi'] >= 0)
                 & (d['psi'] < 2*np.pi) & (d['c'] > 0).all(axis=1)
                 & np.isclose(d['c'].sum(axis=1), 1, atol=1e-10, rtol=1e-10))
        for value in d.values():
            finite = np.isfinite(value)
            valid &= finite.all(axis=1) if finite.ndim == 2 else finite
        factors = {key: np.zeros(nrow) for key in ('luminosity', 'sed', 'size', 'sersic', 'shape')}
        for index, t in enumerate(self.types):
            use = np.flatnonzero(valid & (d['type'] == index) & (d['z'] <= self.z_max[t]))
            valid[(d['type'] == index) & (d['z'] > self.z_max[t])] = False
            if not len(use):
                continue
            z, M, c = d['z'][use], d['M_B'][use], d['c'][use]
            photo = self.photometry(z, M, c)
            inside = ((M <= np.minimum(self.magnitude_cap(z), p['lum_fct_m_max']))
                      & (photo['m_i'] >= p['gals_mag_min']) & (photo['m_i'] <= p['gals_mag_max']))
            valid[use] &= inside
            ms, phi, alpha = self.luminosity_parameters(z, t)
            logx = .4*np.log(10)*(ms-M)
            factors['luminosity'][use] = (np.log(.4*np.log(10)*phi*self.volume(z))
                                           + (alpha+1)*logx - np.exp(logx))
            ac = self.sed_alpha(z, t)
            factors['sed'][use] = (gammaln(ac.sum(axis=1))-gammaln(ac).sum(axis=1)
                                   + ((ac-1)*np.log(c)).sum(axis=1))
            mean, sigma = self.size_parameters(z, photo['M_i'], t)
            R = d['R_kpc'][use]
            factors['size'][use] = norm.logpdf(np.log(R), mean, sigma)-np.log(R)
            a, b = self.sersic_parameters(z, t)
            mass = betaprime.cdf(p['sersic_n_max'], a, b)-betaprime.cdf(p['sersic_n_min'], a, b)
            factors['sersic'][use] = betaprime.logpdf(d['n'][use], a, b)-np.log(mass)
            a, b = self.shape_parameters(t)
            factors['shape'][use] = (beta.logpdf(d['a'][use], a, b, scale=p['ell_beta_emax'])
                                     - np.log(2*np.pi))
        total = sum(factors.values())
        total[~valid] = -np.inf
        return (total, factors) if return_factors else total

    def log_prob(self, data, *, surface_density):
        """Normalize by an explicitly supplied population surface density (per sr)."""
        if not np.isfinite(surface_density) or surface_density <= 0:
            raise ValueError('positive finite surface density required')
        return self.log_intensity(data)-np.log(surface_density)

    def sample(self, n_proposals, seed):
        """Independent continuous proposals followed by the intrinsic i-band limits.

        Return retained latent/derived values and a binomial normalization
        estimate from the fixed proposal count. Retained count is not topped up.
        ``proposal_index`` identifies retained rows in the original proposal
        order, allowing exact Poisson thinning when building neighbour scenes.
        Neither the GalSBI LF grid sampler nor its morphology samplers is used.
        """
        if not isinstance(n_proposals, (int, np.integer)) or n_proposals < 1:
            raise ValueError('n_proposals must be a positive integer')
        rng = np.random.default_rng(seed)
        p = self.p
        weights = np.array([self.tables[t]['rate'] for t in self.types])/self.parent_rate
        kinds = rng.choice(len(self.types), n_proposals, p=weights)
        d = {key: np.zeros(n_proposals) for key in ('z', 'M_B', 'R_kpc', 'n', 'a', 'psi')}
        d['type'], d['c'] = kinds, np.zeros((n_proposals, 5))
        photo_all = {key: np.zeros(n_proposals) for key in ('m_r', 'm_i', 'M_i')}
        cdf_errors = []
        for index, t in enumerate(self.types):
            use = np.flatnonzero(kinds == index)
            n = len(use)
            if not n:
                continue
            tab = self.tables[t]
            z = np.interp(rng.random(n), tab['cdf'], tab['z'])
            ms, _, alpha = self.luminosity_parameters(z, t)
            cap = np.minimum(self.magnitude_cap(z), p['lum_fct_m_max'])
            max_g = upper_gamma(alpha+1, 10 ** (.4*(ms-cap)))
            u = np.maximum(rng.random(n), np.finfo(float).tiny)
            target = np.log(max_g)+np.log(u)
            if np.any(target < tab['log_g'][0]) or np.any(target > tab['log_g'][-1]):
                raise RuntimeError('conditional luminosity inversion outside table support')
            M = ms-np.interp(target, tab['log_g'], tab['inverse_y'])
            cdf_errors.append(float(np.max(np.abs(upper_gamma(alpha+1, 10**(.4*(ms-M)))/max_g-u))))
            c = rng.gamma(self.sed_alpha(z, t))
            c /= c.sum(axis=1, keepdims=True)
            photo = self.photometry(z, M, c)
            mean, sigma = self.size_parameters(z, photo['M_i'], t)
            d['R_kpc'][use] = np.exp(rng.normal(mean, sigma))
            a, b = self.sersic_parameters(z, t)
            low, high = betaprime.cdf(p['sersic_n_min'], a, b), betaprime.cdf(p['sersic_n_max'], a, b)
            d['n'][use] = betaprime.ppf(low+rng.random(n)*(high-low), a, b)
            a, b = self.shape_parameters(t)
            d['a'][use] = rng.beta(a, b, n)*p['ell_beta_emax']
            d['psi'][use] = rng.uniform(0, 2*np.pi, n)
            d['z'][use], d['M_B'][use], d['c'][use] = z, M, c
            for key in photo_all:
                photo_all[key][use] = photo[key]
        selected = (photo_all['m_i'] >= p['gals_mag_min']) & (photo_all['m_i'] <= p['gals_mag_max'])
        fraction = float(selected.mean())
        metadata = dict(n_proposals=n_proposals, n_retained=int(selected.sum()),
                        parent_surface_density_sr=self.parent_rate,
                        surface_density_sr=self.parent_rate*fraction,
                        surface_density_mc_se_sr=self.parent_rate*np.sqrt(fraction*(1-fraction)/n_proposals),
                        maximum_magnitude_inverse_cdf_error=max(cdf_errors))
        retained = {k: v[selected] for k, v in (d | photo_all).items()}
        retained['proposal_index'] = np.flatnonzero(selected)
        return retained, metadata

    def grid_convergence(self):
        """Compare full and alternate-node trapezoidal luminosity normalizers."""
        output = {}
        for t, tab in self.tables.items():
            z = tab['z'][::2]
            ms, phi, alpha = self.luminosity_parameters(z, t)
            cap = np.minimum(self.magnitude_cap(z), self.p['lum_fct_m_max'])
            rate = trapezoid(self.volume(z)*phi*upper_gamma(alpha+1, 10**(.4*(ms-cap))), z)
            output[t] = float((rate-tab['rate'])/tab['rate'])
        return output
