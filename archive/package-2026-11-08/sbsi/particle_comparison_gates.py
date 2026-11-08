"""Fixed n23 development gates; no sampler acceptance or tunable thresholds."""
import math
import numpy as np

from .particle_diagnostics import combine_populations, empirical_second_moment

SIGMA_LIMIT = 3.5


def difference_gate(first, second):
    a,b = (np.asarray(x['mean'],float) for x in (first,second))
    sa,sb = (np.asarray(x['mcse'],float) for x in (first,second))
    if (a.ndim != 1 or not a.size or a.shape != b.shape or sa.shape != a.shape or sb.shape != a.shape
            or not np.isfinite([a,b,sa,sb]).all() or np.any(sa<0) or np.any(sb<0)):
        raise ValueError('finite aligned means and nonnegative independent MCSE required')
    delta=a-b;se=np.hypot(sa,sb);passed=np.abs(delta)<=SIGMA_LIMIT*se
    z=np.divide(delta,se,out=np.zeros_like(delta),where=se>0)
    return dict(difference=delta.tolist(),combined_mcse=se.tolist(),
        signed_standardized_difference=[float(v) if s>0 or d==0 else None for v,s,d in zip(z,se,delta)],
        nonzero_with_zero_estimated_mcse=((se==0)&(delta!=0)).tolist(),
        component_pass=passed.tolist(),pass_all=bool(passed.all()))


def normalizer_gate(a,b):
    logs=np.array([a['log_mean_normalizer'],b['log_mean_normalizer']],float)
    relative=np.array([a['normalizer_relative_mcse'],b['normalizer_relative_mcse']],float)
    if not np.isfinite([logs,relative]).all() or np.any(relative<0):
        raise ValueError('finite normalizers and nonnegative relative MCSE required')
    scaled=np.exp(logs-logs.max())
    result=difference_gate(dict(mean=[scaled[0]],mcse=[scaled[0]*relative[0]]),
                           dict(mean=[scaled[1]],mcse=[scaled[1]*relative[1]]))
    return dict(result,log_ratio=float(logs[0]-logs[1]),common_log_scale=float(logs.max()))


def raw_rows(logz,means,variances,regions,particles,observations):
    """Exactly preserve the existing report's empirical moments and precision gate.

    Explicit extinct runs retain -inf mass in the original R denominator.
    """
    logz,means,variances,regions=map(np.asarray,(logz,means,variances,regions))
    rows=[]
    for i,oid in enumerate(observations):
        alive=np.isfinite(logz[:,i]);count=len(logz)
        if not alive.any():
            rows.append(dict(observation=oid,all_populations_extinct=True,precision_gate=False,
                             independent_populations=count,extinct_populations=count))
            continue
        score=combine_populations(logz[:,i],means[:,i]);region=combine_populations(logz[:,i],regions[:,i])
        second=score['weights']@empirical_second_moment(np.where(alive[:,None],means[:,i],0.),
            np.where(alive[:,None],variances[:,i],0.),particles)
        variance=np.maximum(0.,second-score['mean']**2);mcvar=np.diag(score['covariance'])
        ess=np.divide(variance,mcvar,out=np.full(2,np.inf),where=mcvar>0)
        passed=bool(count>=8 and np.isfinite(ess).all() and np.all(ess>=400)
                    and score['normalizer_relative_mcse']<.1 and score['run_weight_ess']>=.8*count)
        rows.append(dict(observation=oid,all_populations_extinct=False,score=score,region=region,
            posterior_score_variance_estimate=variance,score_effective_count_from_replicates=[float(x) if np.isfinite(x) else None for x in ess],precision_gate=passed))
    return rows


def raw_pair_gate(a,b):
    if a.get('all_populations_extinct') or b.get('all_populations_extinct'):
        return dict(pass_all=False,reason='posterior mean undefined for an entirely extinct arm')
    score=difference_gate(a['score'],b['score']);region=difference_gate(a['region'],b['region'])
    normalizer=normalizer_gate(a['score'],b['score'])
    sd=np.sqrt((np.asarray(a['posterior_score_variance_estimate'])+b['posterior_score_variance_estimate'])/2)
    score_absolute=(np.abs(score['difference'])<=.1*sd).tolist()
    region_absolute=(np.abs(region['difference'])<=.05).tolist()
    return dict(score=score,region=region,normalizer=normalizer,
        score_absolute_pass=score_absolute,region_absolute_pass=region_absolute,
        pass_all=bool(score['pass_all'] and region['pass_all'] and normalizer['pass_all']
                      and all(score_absolute) and all(region_absolute)))


def corrected_phase_gate(a,b):
    """Only corrected g1/g2 gate; preserve raw/T contrasts separately in reports."""
    for value in (a,b):
        mean,se=np.asarray(value['mean'],float),np.asarray(value['mcse'],float)
        if mean.shape!=(5,) or se.shape!=(5,) or not np.isfinite([mean,se]).all() or np.any(se<0):
            raise ValueError('exactly five finite phase moments and nonnegative MCSE required')
    return difference_gate(dict(mean=a['mean'][3:5],mcse=a['mcse'][3:5]),
                           dict(mean=b['mean'][3:5],mcse=b['mcse'][3:5]))
