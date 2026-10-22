"""Direct marginal derivatives and independent-selection Monte Carlo errors.

These mathematical reductions do not establish posterior convergence. In
particular importance-weight ESS is not autocorrelation-adjusted MCMC ESS.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class ShearDerivatives:
    """Score and full log-likelihood Hessian at one explicit common shear."""
    shear: np.ndarray
    score: np.ndarray
    hessian: np.ndarray

    def __post_init__(self):
        for name, shape in [('shear', (2,)), ('score', (2,)), ('hessian', (2, 2))]:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'finite {name} with shape {shape} required')
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if not np.allclose(self.hessian, self.hessian.T, atol=1e-9, rtol=1e-9):
            raise ValueError('symmetric log-likelihood Hessian required')


def posterior_derivatives(shear, score, hessian, *, log_weights=None):
    """Louis identity: E[H log K] + Cov(grad log K) for log integral p K.

Draws must represent the posterior at `shear`, or carry valid importance
weights to it. This function cannot check that premise. The empirical
population covariance (ddof=0) makes the result exactly the derivative of the
corresponding weighted finite mixture; it is not a finite-chain bias correction.
"""
    score, hessian = np.asarray(score, float), np.asarray(hessian, float)
    if (score.ndim != 2 or score.shape[1:] != (2,) or not len(score)
            or hessian.shape != (len(score), 2, 2)
            or not np.isfinite(score).all() or not np.isfinite(hessian).all()):
        raise ValueError('aligned finite rowwise scores and Hessians required')
    if log_weights is None:
        weights = np.full(len(score), 1/len(score))
    else:
        log_weights = np.asarray(log_weights, float)
        if (log_weights.shape != (len(score),) or np.isnan(log_weights).any()
                or np.isposinf(log_weights).any() or not np.isfinite(log_weights).any()):
            raise ValueError('finite or -inf log weights with nonzero total mass required')
        weights = np.exp(log_weights-logsumexp(log_weights))
    mean = weights@score
    centered = score-mean
    second = np.einsum('n,nij->ij', weights, hessian)+(centered.T*weights)@centered
    return ShearDerivatives(shear, mean, second)


def selected_derivatives(numerators, log_selection):
    """Sum object derivatives and subtract N log Z at the SAME shear.

`log_selection` contains derivatives of log Z, including its mixed curvature.
Band-specific normalizers must be combined within their corresponding groups.
No information approximation, one-step correction or convergence claim is
silently substituted for this identity.
"""
    numerators = list(numerators)
    if not numerators or not isinstance(log_selection, ShearDerivatives):
        raise ValueError('object derivatives and explicit log-selection derivatives required')
    if any(not isinstance(item, ShearDerivatives)
           or not np.array_equal(item.shear, log_selection.shear) for item in numerators):
        raise ValueError('all numerator and selection derivatives must share exactly the same shear')
    return ShearDerivatives(log_selection.shear,
        sum(item.score for item in numerators)-len(numerators)*log_selection.score,
        sum(item.hessian for item in numerators)-len(numerators)*log_selection.hessian)


def bounded_newton_proposal(derivatives, *, max_norm):
    """Return a bounded next centre, NOT a fitted root or accepted Newton step.

Caller must reevaluate at the new centre and check likelihood improvement and
score residual relative to numerical error. Indefinite observed information
is rejected rather than replaced with an absolute value or clipped matrix.
"""
    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError('positive finite step bound required')
    information = -derivatives.hessian
    try:
        np.linalg.cholesky(information)
    except np.linalg.LinAlgError as error:
        raise ValueError('observed information is not positive definite') from error
    delta = np.linalg.solve(information, derivatives.score)
    norm = np.linalg.norm(delta)
    if norm > max_norm:
        delta *= max_norm/norm
    return derivatives.shear+delta


def selection_log_contrasts(chunk_mass, contrasts):
    """Log-of-mean contrasts with full covariance from INDEPENDENT chunks.

Rows are independent, identically designed equal-sized population/QMC chunks.
Columns may concatenate all shear nodes and magnitude bands; shared draws
then retain their cross-node and cross-band covariance. Paired random streams
belong in the same row. Individual flow draws within a chunk are not replicas.
Zero-mass chunks are allowed provided each column's overall mean is positive.
"""
    values, contrasts = np.asarray(chunk_mass, float), np.asarray(contrasts, float)
    if (values.ndim != 2 or len(values) < 2 or not values.shape[1]
            or not np.isfinite(values).all() or np.any(values < 0)
            or contrasts.ndim != 2 or contrasts.shape[1] != values.shape[1]
            or not len(contrasts) or not np.isfinite(contrasts).all()):
        raise ValueError('nonnegative independent chunk masses and aligned finite contrasts required')
    mean = values.mean(0)
    if np.any(mean <= 0):
        raise ValueError('positive mean selection mass at every node required')
    influence = (values/mean-1)@contrasts.T
    centered = influence-influence.mean(0)
    covariance = centered.T@centered/(len(values)*(len(values)-1))
    return dict(value=contrasts@np.log(mean), covariance=covariance,
                influence=influence, mean_mass=mean, independent_chunks=len(values))
