"""Uncertainty from independent complete particle populations, not particles."""
import numpy as np
from scipy.special import logsumexp


def combine_populations(log_normalizer, estimates):
    """Ratio of summed unnormalized estimates with delta-method covariance.

    Rows must be independent complete SMC runs of the same target. If each
    run supplies Zhat and a posterior moment fhat, combine sum(Zhat*fhat) /
    sum(Zhat), not the unweighted mean of fhat. Extinct runs retain Zhat=0 in
    the replicate count. This ratio and its finite-replicate error estimate
    are not unbiased or guaranteed to reveal unrepresented rare events.
    Delta-method uncertainty additionally relies on stable finite replicate
    second moments; a small empirical MCSE does not establish that condition.
    """
    logz, values = np.asarray(log_normalizer, float), np.asarray(estimates, float)
    if (logz.ndim != 1 or len(logz) < 2 or np.isnan(logz).any() or np.isposinf(logz).any()
            or not np.isfinite(logz).any() or values.ndim != 2 or len(values) != len(logz)
            or not values.shape[1] or not np.isfinite(values[np.isfinite(logz)]).all()):
        raise ValueError('at least two independent populations with aligned finite moments required')
    values = np.where(np.isfinite(logz)[:, None], values, 0.)
    scaled = np.exp(logz-np.max(logz))
    weights = scaled/scaled.sum()
    mean = weights@values
    influence = (scaled/scaled.mean())[:, None]*(values-mean)
    n = len(logz)
    covariance = influence.T@influence/(n*(n-1))
    return dict(mean=mean, covariance=covariance, mcse=np.sqrt(np.diag(covariance)),
        weights=weights, run_weight_ess=float(1/np.square(weights).sum()),
        log_mean_normalizer=float(logsumexp(logz)-np.log(n)),
        normalizer_relative_mcse=float(scaled.std(ddof=1)/(np.sqrt(n)*scaled.mean())),
        independent_populations=n, extinct_populations=int(np.isneginf(logz).sum()))


def empirical_second_moment(means, sample_variances, particles):
    """Recover the exact empirical second moment from stored ddof=1 variance.

    This identity needs no independence among particles: var(ddof=1) must be
    multiplied by (N-1)/N before adding the squared empirical mean. The sample
    variance's Bessel correction is not a correction for particle ancestry.
    """
    mean, variance = np.asarray(means, float), np.asarray(sample_variances, float)
    if (isinstance(particles, (bool, np.bool_))
            or not isinstance(particles, (int, np.integer)) or particles < 2
            or mean.shape != variance.shape or not np.isfinite(mean).all()
            or not np.isfinite(variance).all() or np.any(variance < 0)):
        raise ValueError('aligned finite means, nonnegative sample variances and N >= 2 required')
    return variance*((particles-1)/particles)+mean**2
