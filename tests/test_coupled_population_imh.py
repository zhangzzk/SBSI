"""Exact finite-state laws and indexing tests; no scene-sampler integration."""
from fractions import Fraction as F
import json

import numpy as np
import pytest

from sbsi.coupled_population_imh import (
    EstimationCancelled, PopulationSummary, coupled_independence_step,
    coupled_population_estimate, log_acceptance,
)


def summary(weight, *observable):
    return PopulationSummary(np.log(weight) if weight else -np.inf, observable)


WEIGHTS = (F(0), F(1), F(4))
PROBABILITIES = np.array([F(1, 3), F(1, 2), F(1, 6)], dtype=object)
OBSERVABLES = np.array([[F(0), F(0)], [F(2), F(-1)], [F(-3), F(5)]], dtype=object)
STATES = tuple(summary(float(w), *map(float, h)) for w, h in zip(WEIGHTS, OBSERVABLES))
PAIRS = tuple((i, j) for i in range(3) for j in range(3))


def exact_events(i, j):
    """Integrate the implementation over exact rational uniform intervals."""
    for proposal, probability in enumerate(PROBABILITIES):
        ax = min(F(1), WEIGHTS[proposal]/WEIGHTS[i]) if WEIGHTS[i] else F(1)
        ay = min(F(1), WEIGHTS[proposal]/WEIGHTS[j]) if j is not None and WEIGHTS[j] else F(1)
        boundaries = sorted({F(0), ax, ay, F(1)})
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            transition = coupled_independence_step(STATES[i], None if j is None else STATES[j],
                STATES[proposal], float((left+right)/2))
            xi = next(n for n, s in enumerate(STATES) if s is transition.x)
            yi = next(n for n, s in enumerate(STATES) if s is transition.y)
            yield probability*(right-left), xi, yi, transition.common_acceptance


def rational_solve(matrix, rhs):
    """Gauss-Jordan over fractions: no cutoff approximation to the tail sum."""
    n, d = len(matrix), rhs.shape[1]
    rows = [list(matrix[i])+list(rhs[i]) for i in range(n)]
    for col in range(n):
        pivot = next(row for row in range(col, n) if rows[row][col])
        rows[pivot], rows[col] = rows[col], rows[pivot]
        scale = rows[col][col]
        rows[col] = [v/scale for v in rows[col]]
        for row in range(n):
            if row != col:
                scale = rows[row][col]
                rows[row] = [a-scale*b for a, b in zip(rows[row], rows[col])]
    return np.array([row[n:n+d] for row in rows], dtype=object)


def finite_law():
    zero = lambda shape: np.full(shape, F(0), dtype=object)
    kernel, transient, initial = zero((3, 3)), zero((9, 9)), zero(9)
    for i in range(3):
        for mass, x, y, met in exact_events(i, None):
            kernel[i, x] += mass
            if not met:
                initial[PAIRS.index((x, y))] += PROBABILITIES[i]*mass
    for row, (i, j) in enumerate(PAIRS):
        for mass, x, y, met in exact_events(i, j):
            if not met:
                transient[row, PAIRS.index((x, y))] += mass
    delta = np.array([OBSERVABLES[i]-OBSERVABLES[j] for i, j in PAIRS], dtype=object)
    return kernel, transient, initial, delta


def test_exact_tilted_invariant_law_and_detailed_balance_with_zero_mass():
    kernel, _, _, _ = finite_law()
    tilted = PROBABILITIES*np.array(WEIGHTS)/sum(PROBABILITIES*np.array(WEIGHTS))
    assert np.all(kernel.sum(1) == 1)
    np.testing.assert_array_equal(tilted@kernel, tilted)
    flow = tilted[:, None]*kernel
    np.testing.assert_array_equal(flow, flow.T)
    assert tilted[0] == 0 and kernel[1, 0] == 0 and kernel[2, 0] == 0
    np.testing.assert_array_equal(kernel[0], PROBABILITIES)


@pytest.mark.parametrize('k,m', [(0, 0), (0, 3), (1, 1), (2, 5)])
def test_exact_infinite_estimator_expectation_from_rational_absorption_system(k, m):
    kernel, transient, initial, delta = finite_law()
    identity = np.array([[F(int(i == j)) for j in range(9)] for i in range(9)], dtype=object)
    future = rational_solve(identity-transient, delta)
    # E[sum_{n=current}^{tau-1} delta_n] includes the entire infinite tail.
    width = m-k+1
    marginal, unmatched = PROBABILITIES.copy(), initial.copy()
    base, correction = np.array([F(0), F(0)]), np.array([F(0), F(0)])
    for n in range(m+1):
        if n >= k:
            base = base + marginal@OBSERVABLES/F(width)
        marginal = marginal@kernel
    for n in range(1, m+1):
        if n > k:
            correction = correction + F(n-k, width)*(unmatched@delta)
        unmatched = unmatched@transient
    correction = correction + unmatched@future
    target = (PROBABILITIES*np.array(WEIGHTS))@OBSERVABLES/sum(PROBABILITIES*np.array(WEIGHTS))
    np.testing.assert_array_equal(base+correction, target)
    np.testing.assert_array_equal(target, [F(-6, 7), F(17, 7)])
    assert np.any(base != target), 'fixture must expose the finite-chain bias'


class ScriptedUniform:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def scripted(states, uniforms, **kwargs):
    proposals = iter(states)
    return coupled_population_estimate(lambda rng: next(proposals), ScriptedUniform(uniforms), **kwargs)


def test_first_proposal_initializes_lagged_chain_and_can_meet_at_one():
    result = scripted([summary(1, 9), summary(2, -7)], [.5])
    np.testing.assert_array_equal(result.estimate, [9])
    assert (result.meeting_time, result.iterations, result.proposal_calls) == (1, 1, 2)


def test_lag_one_correction_excludes_meeting_transition():
    result = scripted([summary(4, 4), summary(1, 1), summary(2, 2), summary(4, -1)], [.9, .6, .2])
    assert result.meeting_time == 3 and result.proposal_calls == 4
    np.testing.assert_array_equal(result.mcmc_average, [4])
    np.testing.assert_array_equal(result.bias_correction, [5])
    np.testing.assert_array_equal(result.estimate, [9])


def test_average_includes_m_and_first_meeting_time_is_not_overwritten():
    result = scripted([summary(4, 4), summary(1, 1), summary(2, 2), summary(4, -1), summary(4, 7)],
        [.9, .6, .2, .4], k=1, m=4)
    np.testing.assert_array_equal(result.mcmc_average, [3.5])
    np.testing.assert_array_equal(result.bias_correction, [.5])
    np.testing.assert_array_equal(result.estimate, [4.])
    assert (result.meeting_time, result.iterations, result.proposal_calls) == (3, 4, 5)


def test_shared_uniform_acceptance_and_faithfulness():
    x, y, p = summary(4, 2), summary(1, -3), summary(2, 1)
    step = coupled_independence_step(x, y, p, .6)
    assert step.x is x and step.y is p and not step.common_acceptance
    step = coupled_independence_step(x, y, p, .4)
    assert step.x is step.y is p and step.common_acceptance
    for u in (0., .4, .99):
        step = coupled_independence_step(p, p, summary(1, -2), u)
        assert step.x is step.y


def test_equal_summaries_are_not_evidence_of_shared_population_identity():
    x, y, p = summary(4, 1), summary(4, 1), summary(1, 2)
    step = coupled_independence_step(x, y, p, .9)
    assert step.x is x and step.y is y and not step.common_acceptance


def test_zero_mass_is_canonical_retained_and_never_retried():
    zero = summary(0, 123, -17)
    np.testing.assert_array_equal(zero.observable, [0, 0])
    result = scripted([zero, summary(0, 2, 4)], [0.])
    assert result.proposal_calls == 2 and result.meeting_time == 1
    np.testing.assert_array_equal(result.estimate, [0, 0])
    positive = summary(1, 2, 3)
    step = coupled_independence_step(positive, zero, zero, 0.)
    assert step.x is positive and step.y is zero and not step.accepted_x
    step = coupled_independence_step(zero, zero, positive, .99)
    assert step.x is step.y is positive
    assert log_acceptance(-np.inf, -np.inf) == 0
    assert log_acceptance(0, -np.inf) == -np.inf


@pytest.mark.parametrize('value', [np.nan, np.inf, [0], 1j])
def test_unknown_log_normalizers_raise(value):
    with pytest.raises(ValueError):
        PopulationSummary(value, [0])
    with pytest.raises(ValueError):
        log_acceptance(0, value)


@pytest.mark.parametrize('value', [[np.nan], [np.inf], [], [[0]], [1j]])
@pytest.mark.parametrize('logz', [0, -np.inf])
def test_bad_observables_raise_even_when_extinct(value, logz):
    with pytest.raises(ValueError):
        PopulationSummary(logz, value)


@pytest.mark.parametrize('uniform', [-.1, 1, np.nan, np.inf, [.5]])
def test_invalid_uniforms_raise(uniform):
    with pytest.raises(ValueError):
        coupled_independence_step(STATES[0], None, STATES[1], uniform)


@pytest.mark.parametrize('k,m', [(-1, 0), (1, 0), (0., 1), (0, 1.5), (False, 0)])
def test_bad_estimator_indices_raise_before_callback(k, m):
    with pytest.raises(ValueError):
        coupled_population_estimate(lambda rng: pytest.fail('unexpected call'), None, k=k, m=m)


def test_shapes_callback_errors_and_accumulator_overflow_are_not_extinction():
    with pytest.raises(ValueError, match='dimension'):
        scripted([summary(1, 1), summary(1, 1, 2)], [.5])
    with pytest.raises(TypeError):
        coupled_population_estimate(lambda rng: (0., [0.]), None)
    def broken(rng):
        raise RuntimeError('unknown SMC failure')
    with pytest.raises(RuntimeError, match='unknown SMC failure'):
        coupled_population_estimate(broken, None)
    with pytest.raises(FloatingPointError):
        scripted([summary(4, 1e308), summary(1, -1e308)], [.9])


def test_summary_defensively_copies_and_freezes_observable():
    a = np.array([1., 2.])
    value = PopulationSummary(0., a)
    a[:] = 9
    np.testing.assert_array_equal(value.observable, [1, 2])
    with pytest.raises(ValueError):
        value.observable[0] = 5


def test_cancellation_raises_without_partial_estimate_or_redraw():
    calls = []
    def propose(rng):
        calls.append(1)
        return summary(100 if len(calls) == 1 else 1, len(calls))
    with pytest.raises(EstimationCancelled) as caught:
        coupled_population_estimate(propose, ScriptedUniform([.9, .9]), cancelled=lambda: len(calls) == 3)
    assert caught.value.iterations == 2 and caught.value.proposal_calls == 3
    assert not hasattr(caught.value, 'estimate') and len(calls) == 3
    with pytest.raises(EstimationCancelled) as caught:
        coupled_population_estimate(propose, None, cancelled=lambda: True)
    assert caught.value.proposal_calls == 0 and len(calls) == 3


def finite_proposal(rng):
    return STATES[rng.choice(3, p=np.asarray(PROBABILITIES, float))]


def test_deterministic_replay_including_cost_and_correction():
    a = coupled_population_estimate(finite_proposal, np.random.default_rng(2026102201), k=1, m=5)
    b = coupled_population_estimate(finite_proposal, np.random.default_rng(2026102201), k=1, m=5)
    for name in vars(a):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize('k,m,seed', [(0, 0, 2026102202), (2, 5, 2026102203)])
def test_independent_finite_law_replicates(k, m, seed):
    rng = np.random.default_rng(seed)
    results = [coupled_population_estimate(finite_proposal, rng, k=k, m=m) for _ in range(12000)]
    samples = np.array([r.estimate for r in results])
    mean, se = samples.mean(0), samples.std(0, ddof=1)/np.sqrt(len(samples))
    target = np.array([-6/7, 17/7])
    assert np.all(np.abs(mean-target) < 6*se)
    assert all(r.proposal_calls == r.iterations+1 for r in results)
    print(json.dumps(dict(finite_law_only=True, k=k, m=m, seed=seed, replicas=len(samples),
        mean=mean.tolist(), target=target.tolist(), standard_error=se.tolist(),
        max_meeting_time=max(r.meeting_time for r in results))))
