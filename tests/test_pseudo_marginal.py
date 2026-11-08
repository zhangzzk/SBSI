"""Analytic latent-Gaussian closure with deliberately noisy likelihoods."""
import numpy as np
from scipy.special import logsumexp

from sbsi.pseudo_marginal import metropolis_accept, retained


def test_noisy_likelihood_mh_has_correct_marginal_and_retains_auxiliaries():
    rng = np.random.default_rng(798)
    n, m = 32, 4
    x = rng.normal(size=n)
    sigma, y = .8, .7

    def estimate(value):
        neighbours = rng.normal(size=(n, m))
        logk = -.5*((y-value[:, None]-neighbours)/sigma)**2
        return logsumexp(logk, axis=1)-np.log(m), neighbours

    loglike, env = estimate(x)
    trace, rejected_checked = [], False
    for step in range(4000):
        proposed = x+rng.normal(0, .8, n)
        newlike, newenv = estimate(proposed)
        accepted = metropolis_accept(-.5*x*x+loglike, -.5*proposed*proposed+newlike,
                                     rng.uniform(1e-12, 1, n))
        oldlike, oldenv = loglike.copy(), env.copy()
        x, loglike, env = (retained(old, new, accepted) for old, new in
                            ((x, proposed), (loglike, newlike), (env, newenv)))
        if (~accepted).any() and not rejected_checked:
            np.testing.assert_array_equal(loglike[~accepted], oldlike[~accepted])
            np.testing.assert_array_equal(env[~accepted], oldenv[~accepted])
            rejected_checked = True
        if step >= 1000:
            trace.append(x.copy())
    trace = np.array(trace)
    variance = (1+sigma*sigma)/(2+sigma*sigma)
    mean = y/(2+sigma*sigma)
    for values, expected in ((trace, mean), ((trace-mean)**2, variance)):
        means = values.mean(0)
        assert abs(means.mean()-expected) < 6*means.std(ddof=1)/np.sqrt(n)
    assert rejected_checked
