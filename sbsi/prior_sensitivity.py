"""Paired whole-population sensitivity of fixed saved-state ratio estimates.

Diagnostic only. No draws, density changes, control fitting or sampler gates.
"""
import numpy as np
from scipy.special import logsumexp
from sbsi.particle_diagnostics import combine_populations


def finite_log_density(value):
    x = np.asarray(value, float)
    if np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError('unknown/nonfinite density failure is not a zero mass')
    return x


def type_marginal_ratio(analytic, numerical, correction):
    """Sum full numerical type intensities, including q-only type support.

    All rows must be on the analytic mixture support; a missing analytic type
    does not justify dropping the corresponding numerical type intensity.
    """
    p, q, h = map(finite_log_density, (analytic, numerical, correction))
    if p.ndim != 2 or not p.size or p.shape != q.shape or p.shape != h.shape:
        raise ValueError('aligned nonempty per-state/per-type intensities required')
    lp, lq = logsumexp(p, axis=1), logsumexp(q, axis=1)
    if not np.isfinite(lp).all():
        raise ValueError('saved states must be inside analytic mixture support')
    reconstructed = np.where(np.isfinite(p), p+h, -np.inf)
    shared = np.isfinite(p) & np.isfinite(q)
    supported_p = np.isfinite(p)
    if not np.array_equal(np.isfinite(reconstructed[supported_p]), np.isfinite(q[supported_p])):
        raise ValueError('restricted helper disagrees on shared analytic/numerical support')
    error = np.max(np.abs(q[shared]-p[shared]-h[shared])) if shared.any() else 0.
    if not np.isfinite(error) or error > 1e-9:
        raise ValueError('restricted helper does not reproduce full numerical intensity on shared support')
    restricted = logsumexp(reconstructed, axis=1)
    return dict(log_ratio=lq-lp, analytic_log_intensity=lp, numerical_log_intensity=lq,
        restricted_log_ratio=restricted-lp, q_only_type=np.isfinite(q) & ~np.isfinite(p),
        p_only_type=np.isfinite(p) & ~np.isfinite(q), helper_shared_max_abs_error=float(error),
        full_numerical_zero=~np.isfinite(lq), restricted_numerical_zero=~np.isfinite(restricted))


def population_measure(logz, values, logweights):
    """Use original N in each unnormalized population, even for zero weights."""
    z = finite_log_density(logz)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all() or len(values) != len(z) or len(logweights) != len(z):
        raise ValueError('complete finite original populations required; no dropping failures')
    moments, masses, point_ess, maxima, zero_counts, sizes = [], [], [], [], [], []
    dimensions = None
    for f, lw in zip(values, logweights):
        f, lw = np.asarray(f, float), finite_log_density(lw)
        if (f.ndim != 2 or len(f) < 1 or lw.shape != (len(f),) or not np.isfinite(f).all()
                or (dimensions is not None and f.shape[1] != dimensions)):
            raise ValueError('complete aligned finite observables required')
        dimensions = f.shape[1]
        total = logsumexp(lw)
        normalized = np.exp(lw-total) if np.isfinite(total) else np.zeros(len(lw))
        moments.append(normalized@f)
        masses.append(total-np.log(len(f)))
        point_ess.append(float(1/np.square(normalized).sum()) if normalized.any() else 0.)
        maxima.append(float(normalized.max()))
        zero_counts.append(int(np.isneginf(lw).sum())); sizes.append(len(f))
    masses, moments = np.asarray(masses), np.asarray(moments)
    adjusted = z+masses
    if not np.isfinite(adjusted).any():
        raise ValueError('all numerical mass is zero for this observation; posterior mean undefined')
    summary = combine_populations(adjusted, moments)
    influence = len(z)*summary['weights'][:, None]*(moments-summary['mean'])
    # These are concentration diagnostics only, never independent sample sizes.
    global_max = max(w*m for w,m in zip(summary['weights'], maxima))
    denominator = sum(w*w/ess for w,ess in zip(summary['weights'],point_ess) if ess > 0)
    return dict(summary=summary, influence=influence, per_run_log_mass=masses,
        per_run_log_normalizer=adjusted, per_run_mean=moments,
        per_run_original_particles=sizes, per_run_zero_weights=zero_counts,
        per_run_point_weight_concentration_ess=point_ess, per_run_max_point_weight=maxima,
        global_max_point_weight=float(global_max), global_point_weight_concentration_ess=float(1/denominator))


def paired_sensitivity(logz, values, measure_weights, log_ratios):
    """Paired delta-method contrast, including covariance and both denominators."""
    if len(measure_weights) != len(log_ratios):
        raise ValueError('one ratio per original complete population required')
    base = population_measure(logz, values, measure_weights)
    tilted_weights = []
    for b, r in zip(measure_weights, log_ratios):
        b, r = finite_log_density(b), finite_log_density(r)
        if b.shape != r.shape:
            raise ValueError('prior ratios must align with every original particle')
        tilted_weights.append(b+r)
    tilted = population_measure(logz, values, tilted_weights)
    n = len(logz)
    delta_influence = tilted['influence']-base['influence']
    covariance = delta_influence.T@delta_influence/(n*(n-1))
    cross_covariance = base['influence'].T@tilted['influence']/(n*(n-1))
    difference = tilted['summary']['mean']-base['summary']['mean']
    paired_se = np.sqrt(np.diag(covariance))
    log_mass_ratio = tilted['summary']['log_mean_normalizer']-base['summary']['log_mean_normalizer']
    mass_influence = n*(tilted['summary']['weights']-base['summary']['weights'])
    log_mass_se = float(np.sqrt(mass_influence@mass_influence/(n*(n-1))))
    scaled = np.divide(difference,base['summary']['mcse'],out=np.full_like(difference,np.nan),where=base['summary']['mcse']>0)
    return dict(analytic=base, numerical=tilted, numerical_minus_analytic=difference,
        paired_contrast_mcse=paired_se, paired_contrast_covariance=covariance,
        analytic_numerical_cross_covariance=cross_covariance, paired_contrast_influence=delta_influence,
        shift_over_analytic_mcse=[float(x) if np.isfinite(x) else None for x in scaled],
        intensity_mass_log_ratio=float(log_mass_ratio), intensity_mass_ratio=float(np.exp(log_mass_ratio)),
        intensity_mass_log_ratio_paired_mcse=log_mass_se,
        intensity_mass_ratio_paired_mcse=float(np.exp(log_mass_ratio)*log_mass_se),
        original_independent_populations=n)
