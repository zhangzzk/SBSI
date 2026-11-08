"""Fixed analytic phase product-rule control; experimental diagnostic only.

No fitting/recentering and no exact identity for a rounded F32 program.
At the intended continuous endpoint target, conditional phase integration
gives E[a*T + d(a)/d(phase)] = 0. The fixed epsilon was declared before use.
"""
import numpy as np
from scipy.special import logsumexp

from .particle_diagnostics import combine_populations


EPSILON = 0.1
RULE = 'eps=0.1; F=(1+r2)/(r2+eps2); C=(-F*(e2*T+e1),F*(e1*T-e2)); S-C'
MOMENT_COLUMNS = ('raw_g1','raw_g2','constant_g1','constant_g2',
                  'analytic_g1','analytic_g2','analytic_control_g1','analytic_control_g2','T')


def phase_coefficient(ellipticity):
    """Return fixed a=(-F*e2,F*e1) and exact phase derivative a'=(-F*e1,-F*e2)."""
    e = np.asarray(ellipticity, float)
    if e.ndim < 1 or e.shape[-1] != 2 or not np.isfinite(e).all():
        raise ValueError('finite intrinsic ellipticity pairs required')
    radius2 = np.square(e).sum(-1)
    if np.any(radius2 >= 1):
        raise ValueError('intrinsic ellipticity must be inside the unit disk')
    factor = (1+radius2)/(radius2+EPSILON**2)
    a = factor[...,None]*np.stack((-e[...,1],e[...,0]),axis=-1)
    derivative = -factor[...,None]*e
    return a,derivative


def analytic_phase_control(ellipticity, phase_score, *, beta=1.):
    """C=a*T+a' with its essential product-rule term, at beta=1 only."""
    e,t = np.asarray(ellipticity,float),np.asarray(phase_score,float)
    if not np.isscalar(beta) or beta != 1.:
        raise ValueError('endpoint beta=1 control only')
    a,derivative = phase_coefficient(e)
    if t.shape != e.shape[:-1] or not np.isfinite(t).all():
        raise ValueError('finite phase derivatives aligned with ellipticity required')
    result = a*t[...,None]+derivative
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite analytic phase control')
    return result


def summarize_score_comparison(log_normalizers, raw_scores, constant_scores,
                              controls, phase_scores, deltas):
    """Whole independent-population ratios under two explicit numerical measures.

    Every particle is required. F64 weights use Z64hat=Z32hat*sum(exp(delta))/N
    with the original full population N, never a subsample denominator.
    """
    logz = np.asarray(log_normalizers,float)
    fields=(raw_scores,constant_scores,controls,phase_scores,deltas)
    if (logz.ndim != 2 or len(logz)<2 or not np.isfinite(logz).all()
            or any(len(v)!=len(logz) for v in fields)):
        raise ValueError('complete aligned independent populations required')
    raw_moments=[]; precision_moments=[]; precision_logz=[]
    for run,(raw,constant,control,t,delta) in enumerate(zip(*fields)):
        raw,constant,control,t,delta=map(lambda v:np.asarray(v,float),(raw,constant,control,t,delta))
        if (raw.ndim!=3 or raw.shape[0]!=logz.shape[1] or raw.shape[1]<1 or raw.shape[2]!=2
                or constant.shape!=raw.shape or control.shape!=raw.shape
                or t.shape!=raw.shape[:2] or delta.shape!=t.shape
                or not all(np.isfinite(v).all() for v in (raw,constant,control,t,delta))):
            raise ValueError('full finite score/control/delta arrays required')
        values=np.concatenate((raw,constant,raw-control,control,t[...,None]),axis=-1)
        raw_moments.append(values.mean(axis=1))
        logtotal=logsumexp(delta,axis=1)
        weight=np.exp(delta-logtotal[:,None])
        precision_moments.append(np.sum(weight[...,None]*values,axis=1))
        precision_logz.append(logz[run]+logtotal-np.log(raw.shape[1]))
    raw_moments,precision_moments,precision_logz=map(np.asarray,
        (raw_moments,precision_moments,precision_logz))
    rows=[]
    for obj in range(logz.shape[1]):
        raw=combine_populations(logz[:,obj],raw_moments[:,obj])
        precision=combine_populations(precision_logz[:,obj],precision_moments[:,obj])
        rows.append(dict(raw_float32_sampled=dict(raw,
            analytic_control_known_expectation=None,phase_control_known_expectation=None,
            zero_mean_identity_established_for_this_numerical_measure=False),
            precision_reweighted_float64=dict(precision,
                analytic_control_known_expectation=[0.,0.],phase_control_known_expectation=0.,
                scope='Continuous-model periodic identity, numerically evaluated in F64; not exact rounded-program arithmetic.')))
    return rows
