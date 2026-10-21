"""Experimental native target/neighbour exchanges for the mixed-prior sampler.

The target has analytic intensity p, whereas neighbours have numerical
generator intensity q. The target type is temporarily drawn from p(type|x).
Swapping complete native marks then has prior ratio p(new) q(old) /
[p(old) q(new)]. Shared conditional factors cancel, leaving two luminosity
interpolation corrections. Neither an analytic environment nor an uncorrected
swap is substituted for the implemented target measure.
"""
import numpy as np
from scipy.special import logsumexp

from .galsbi_prior import upper_gamma
from .galsbi_scene_prior import GalSBISceneBatch


def numerical_over_analytic(prior, data, *, return_support_audit=False):
    """log(q/p) on analytic support; -inf where either support fails.

q includes the type intensity, the piecewise-linear redshift inverse CDF,
and the piecewise-linear inverse log-gamma luminosity CDF. Other conditional
densities and the intrinsic photometric cut are shared. The unused global
normalization cancels between marks. This function is intentionally restricted
to analytic support, as required by exchanging a mark into the target slot.
The numerical continuous part also requires log(U)>=log(float.tiny), matching
sample()'s lower uniform clamp. Singular clamp atoms and the discrete RNG law
are not represented by this Lebesgue density.

With return_support_audit=True, also return aligned legacy_finite and
below_uniform_clamp masks. A separate frozen-trajectory replay can compare
these masks while returning its original ratios; this diagnostic draws no RNG
and does not provide a production switch to disable the support restriction.
"""
    lp, factors = prior.log_intensity(data, return_factors=True)
    result = np.full(len(lp), -np.inf)
    if return_support_audit:
        legacy_finite = np.zeros(len(lp), dtype=bool)
        below_uniform_clamp = np.zeros(len(lp), dtype=bool)
    for kind, name in enumerate(prior.types):
        ix = np.flatnonzero((data['type'] == kind) & np.isfinite(lp))
        if not len(ix):
            continue
        tab = prior.tables[name]
        z, M = data['z'][ix], data['M_B'][ix]
        jz = np.clip(np.searchsorted(tab['z'], z, side='right')-1, 0, len(tab['z'])-2)
        qz = np.diff(tab['cdf'])[jz]/np.diff(tab['z'])[jz]
        ms, _, alpha = prior.luminosity_parameters(z, name)
        y = ms-M
        yy, tt = tab['inverse_y'][::-1], tab['log_g'][::-1]
        j = np.clip(np.searchsorted(yy, y, side='right')-1, 0, len(yy)-2)
        slope = np.diff(tt)[j]/np.diff(yy)[j]
        value = tt[j]+slope*(y-yy[j])
        cap = np.minimum(prior.magnitude_cap(z), prior.p['lum_fct_m_max'])
        logmax = np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))
        valid = (y >= yy[0]) & (y <= yy[-1]) & (value <= logmax) & (qz > 0)
        with np.errstate(divide='ignore'):
            logq = np.log(tab['rate'])+np.log(qz)+value-logmax+np.log(np.abs(slope))
        result[ix[valid]] = (logq-factors['luminosity'][ix])[valid]
        # Keep the existing finite-path arithmetic exactly. The omitted lower
        # bound formerly let an analytic target enter a zero-density numerical
        # environment, even when its intrinsic photometric cut passed.
        below = valid & (value-logmax < np.log(np.finfo(float).tiny))
        if return_support_audit:
            legacy_finite[ix] = np.isfinite(result[ix])
            below_uniform_clamp[ix[below]] = True
        result[ix[below]] = -np.inf
    if return_support_audit:
        return result, dict(legacy_finite=legacy_finite,
                           below_uniform_clamp=below_uniform_clamp)
    return result


def _relocation_parameters(scale, aperture_arcsec, central_weight):
    if (not np.isfinite(scale) or scale <= 0
            or aperture_arcsec is None or not np.isfinite(aperture_arcsec)
            or aperture_arcsec <= 0 or not np.isfinite(central_weight)
            or not 0 <= central_weight <= 1):
        raise ValueError('positive finite relocation scale/aperture and mixture weight in [0,1] required')
    with np.errstate(over='ignore', under='ignore'):
        cutoff = .5*np.square(np.float64(aperture_arcsec)/scale)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('relocation aperture/scale ratio is outside numerical support')
    return cutoff, -np.expm1(-cutoff)


def relocation_position_log_density(offset_arcsec, *, scale=.1,
                                    aperture_arcsec, central_weight=.9):
    """Normalized Cartesian density of a truncated central Gaussian/disk mixture.

    The central component is an isotropic 2D Gaussian conditioned on the native
    aperture; its radial normalizer is 1-exp(-R**2/(2*scale**2)). The second
    component is uniform over the entire aperture disk. This is a proposal law,
    not a modified spatial population prior.
    """
    _, mass = _relocation_parameters(scale, aperture_arcsec, central_weight)
    offsets = np.asarray(offset_arcsec, float)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not np.isfinite(offsets).all():
        raise ValueError('finite two-component relocation offsets required')
    with np.errstate(over='ignore'):
        radius2 = np.square(offsets/aperture_arcsec).sum(1)
    with np.errstate(divide='ignore', over='ignore'):
        central = (np.log(central_weight)-np.log(2*np.pi)-2*np.log(scale)
                   -np.log(mass)-.5*np.square(offsets/scale).sum(1))
        uniform = np.log1p(-central_weight)-np.log(np.pi)-2*np.log(aperture_arcsec)
    result = np.logaddexp(central, uniform)
    return np.where(radius2 <= 1., result, -np.inf)


def draw_relocation_positions(count, rng, *, scale=.1,
                              aperture_arcsec, central_weight=.9):
    """Draw the exact mixture using its radial inverse CDF, without retries."""
    _, mass = _relocation_parameters(scale, aperture_arcsec, central_weight)
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError('nonnegative integer relocation count required')
    central = rng.random(count) < central_weight
    radial_uniform = rng.random(count)
    phase = 2*np.pi*rng.random(count)
    radius = aperture_arcsec*np.sqrt(radial_uniform)
    radius[central] = scale*np.sqrt(-2*np.log1p(-radial_uniform[central]*mass))
    return radius[:, None]*np.column_stack((np.cos(phase), np.sin(phase)))


def _draw_type_categories(probability, rng):
    """Batch Generator.choice's float64 CDF without changing draw order.

    Rows have already passed the caller's probability checks. NumPy choice
    accumulates in float64 and normalizes by the final cumulative entry. The
    comparison counts entries <= u, i.e. searchsorted(side='right'), including
    zero-probability categories. Custom RNGs keep their original choice calls.
    Ragged neighbour probabilities deliberately keep their existing reductions.
    """
    if (type(rng) is not np.random.Generator
            or probability.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        return np.array([rng.choice(probability.shape[1], p=row) for row in probability])
    cumulative = np.asarray(probability, dtype=np.float64).cumsum(axis=1)
    cumulative /= cumulative[:, -1, None]
    uniforms = rng.random(len(probability))
    return np.count_nonzero(cumulative <= uniforms[:, None], axis=1)


def propose_native_exchange(prior, scene, type_probability, rng, *, radius_scale=1.,
                            relocation_scale=None, aperture_arcsec=None, central_weight=.9):
    """Propose a mark swap and return its native prior/proposal log correction.

Without relocation, neighbour choice depends only on unchanged positions, so
selection probabilities cancel. No-neighbour scenes are inactive. Temporary target type is a Gibbs draw
from the analytic conditional; projecting it out afterwards preserves the
marginal target distribution. The native swap has unit absolute Jacobian.
Callers using transformed/marginal target log densities must replace their prior
difference by the returned native ratio, not add the ratio a second time.

Optional relocation independently redraws the swapped neighbour's position from
a normalized central-Gaussian/uniform-disk mixture. Its reverse/forward density
and changed neighbour-selection probabilities enter the returned correction.
The uniform spatial population prior cancels inside the unchanged aperture.
With relocation_scale=None, the original arithmetic and RNG sequence are kept.
"""
    if not np.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError('positive finite neighbour selection radius required')
    relocation = relocation_scale is not None
    offsets = scene.offset_arcsec
    if relocation:
        _relocation_parameters(relocation_scale, aperture_arcsec, central_weight)
        if (not np.isfinite(offsets).all()
                or np.any(np.square(offsets/aperture_arcsec).sum(1) > 1.)):
            raise ValueError('relocation requires original positions inside the native aperture')
        offsets = offsets.copy()
    probability = np.asarray(type_probability)
    if (probability.shape != (len(scene.counts), len(prior.types))
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or not np.allclose(probability.sum(1), 1., atol=1e-12, rtol=0)):
        raise ValueError('normalized analytic conditional type probabilities required')
    targets = {k: v.copy() for k, v in scene.targets.items()}
    neighbours = {k: v.copy() for k, v in scene.neighbours.items()}
    if targets.keys() != neighbours.keys():
        raise ValueError('target and neighbour native schemas must match')
    active = scene.counts > 0
    rows = np.flatnonzero(active)
    selected = np.empty(len(rows), dtype=int)
    log_selection_old = np.empty(len(rows)) if relocation else None
    for j, row in enumerate(rows):
        start, stop = scene.indptr[row:row+2]
        logw = -.5*np.sum(scene.offset_arcsec[start:stop]**2, axis=1)/radius_scale**2
        w = np.exp(logw-logw.max())
        selected[j] = start+rng.choice(stop-start, p=w/w.sum())
        if relocation:
            log_selection_old[j] = logw[selected[j]-start]-logsumexp(logw)
    logratio = np.zeros(len(active))
    if len(rows):
        old = {k: v[rows].copy() for k, v in targets.items()}
        old['type'] = _draw_type_categories(probability[rows], rng)
        new = {k: v[selected].copy() for k, v in neighbours.items()}
        old_correction = numerical_over_analytic(prior, old)
        new_correction = numerical_over_analytic(prior, new)
        valid = np.isfinite(old_correction) & np.isfinite(new_correction)
        logratio[rows] = -np.inf
        logratio[rows[valid]] = old_correction[valid]-new_correction[valid]
        for key in targets:
            targets[key][rows] = new[key]
            neighbours[key][selected] = old[key]
        if relocation:
            parameters = dict(scale=relocation_scale, aperture_arcsec=aperture_arcsec,
                              central_weight=central_weight)
            proposed_offsets = draw_relocation_positions(len(rows), rng, **parameters)
            position_correction = (relocation_position_log_density(offsets[selected], **parameters)
                -relocation_position_log_density(proposed_offsets, **parameters))
            offsets[selected] = proposed_offsets
            for j, row in enumerate(rows):
                start, stop = scene.indptr[row:row+2]
                logw = -.5*np.sum(offsets[start:stop]**2, axis=1)/radius_scale**2
                position_correction[j] += (logw[selected[j]-start]-logsumexp(logw)
                                          -log_selection_old[j])
            logratio[rows] += position_correction
    return (GalSBISceneBatch(targets, neighbours, scene.indptr, offsets),
            logratio, active)
