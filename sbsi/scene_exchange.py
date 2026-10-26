"""Experimental native target/neighbour exchanges for the mixed-prior sampler.

The target has analytic intensity p, whereas neighbours have numerical
generator intensity q. The target type is temporarily drawn from p(type|x).
Swapping complete native marks then has prior ratio p(new) q(old) /
[p(old) q(new)]. Shared conditional factors cancel, leaving two luminosity
interpolation corrections. Neither an analytic environment nor an uncorrected
swap is substituted for the implemented target measure.
"""
import numpy as np

from .galsbi_prior import upper_gamma
from .galsbi_scene_prior import GalSBISceneBatch


def numerical_over_analytic(prior, data):
    """log(q/p) on analytic support; -inf where either support fails.

q includes the type intensity, the piecewise-linear redshift inverse CDF,
and the piecewise-linear inverse log-gamma luminosity CDF. Other conditional
densities and the intrinsic photometric cut are shared. The unused global
normalization cancels between marks. This function is intentionally restricted
to analytic support, as required by exchanging a mark into the target slot.
"""
    lp, factors = prior.log_intensity(data, return_factors=True)
    result = np.full(len(lp), -np.inf)
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
    return result


def propose_native_exchange(prior, scene, type_probability, rng, *, radius_scale=1.):
    """Propose a complete mark swap and return its native prior log ratio.

Neighbour choice depends only on unchanged positions, so selection probabilities
cancel. No-neighbour scenes are inactive. Temporary target type is a Gibbs draw
from the analytic conditional; projecting it out afterwards preserves the
marginal target distribution. The native swap has unit absolute Jacobian.
Callers using transformed/marginal target log densities must replace their prior
difference by the returned native ratio, not add the ratio a second time.
"""
    if not np.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError('positive finite neighbour selection radius required')
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
    for j, row in enumerate(rows):
        start, stop = scene.indptr[row:row+2]
        logw = -.5*np.sum(scene.offset_arcsec[start:stop]**2, axis=1)/radius_scale**2
        w = np.exp(logw-logw.max())
        selected[j] = start+rng.choice(stop-start, p=w/w.sum())
    logratio = np.zeros(len(active))
    if len(rows):
        old = {k: v[rows].copy() for k, v in targets.items()}
        old['type'] = np.array([rng.choice(len(prior.types), p=probability[row]) for row in rows])
        new = {k: v[selected].copy() for k, v in neighbours.items()}
        old_correction = numerical_over_analytic(prior, old)
        new_correction = numerical_over_analytic(prior, new)
        valid = np.isfinite(old_correction) & np.isfinite(new_correction)
        logratio[rows] = -np.inf
        logratio[rows[valid]] = old_correction[valid]-new_correction[valid]
        for key in targets:
            targets[key][rows] = new[key]
            neighbours[key][selected] = old[key]
    return (GalSBISceneBatch(targets, neighbours, scene.indptr, scene.offset_arcsec),
            logratio, active)
