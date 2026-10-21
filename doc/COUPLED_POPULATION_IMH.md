# Coupled whole-population independence MH — isolated n17

## Source and indexing

[Middleton et al. (2019)](https://proceedings.mlr.press/v89/middleton19a/middleton19a.pdf),
Algorithm 3 initializes X_0 from the fixed proposal. At step1 its fresh proposal
becomes Y_0 unconditionally and proposes X_1. Subsequent steps update
(X_{n-1},Y_{n-2}) with one shared proposal and uniform. The first common
acceptance gives tau; coupling remains faithful afterwards. Equation (2) uses

H_{k:m} = sum_{l=k}^m h(X_l)/(m-k+1)
+ sum_{l=k+1}^{tau-1} min(1,(l-k)/(m-k+1)) [h(X_l)-h(Y_{l-1})].

The implementation explicitly simulates through X_m and meeting, and retains
the first tau. Proposition7 requires appropriate observable moments and bounded
particle weights/unbiased resampling; the paper's general coupling conditions
require moment control and a suitable meeting-time tail. None is established
for SBSI here. Whole-population observables use the same algebra as individual
state observables.

## Interface and target identity

`sbsi/coupled_population_imh.py` is independent of the scene sampler. Pass
`coupled_population_estimate(propose, rng, k=0, m=0)` a callback whose every
call produces a fresh `PopulationSummary(log_normalizer, observable)` under
one fixed law Q. The vector observable represents a complete population's
particle-average score (weighted if required by the actual population scheme).
It is not a draw of a single retained particle, and proposals must not be
selected by their scores, weights, success, or a stopping diagnostic.

On this summary space the invariant law is Zhat Q / E_Q[Zhat]. Therefore its
expectation is E_Q[Zhat fhat]/E_Q[Zhat]. Scientific interpretation requires
separately proving both the normalizer and weighted-observable identities for
the actual SMC implementation and endpoint. This module does not establish
them and does not identify F32 and F64 likelihood targets. A finite pool sampled
with replacement has its own empirical Q: exact tests for that pool are not
exact inference for the population-generating law.

The immutable summary copies and validates its finite vector. Acceptance uses
only normalizers. Meeting is common proposal acceptance, not numerical equality
of summaries; this prevents two different populations with identical scores
from being silently treated as the same extended state.

## Zero mass and failure contract

Known extinction is `log_normalizer=-inf`, with a supplied finite placeholder
vector canonicalized to zero. The original proposal probability of extinction
is retained. From positive weight, zero weight is rejected. From zero weight,
all proposals are accepted, including another zero: the explicit 0/0 convention
is1. These transient transitions do not change the positive-weight invariant
law. Tests include zero mass in exact finite-state expectations.
This convention is an explicit extension: the paper's initial particle-filter
description assumes positive weights. Its scientific assumptions must not be
imported automatically into a sampler that can become extinct.

NaN/+inf normalizers, nonfinite vectors, shape changes, unknown callback
exceptions, and accumulator overflow raise. They are never interpreted as
extinction or retried. A proposal law with no positive mass has no normalized
tilted target; the caller must establish 0 < E_Q[Zhat] < infinity. No finite
number of draws can certify that property.

There is no meeting cutoff. An optional cancellation callback raises
`EstimationCancelled` without returning an estimate; it reports completed
iterations/proposal calls only. Cancelled/error replicates cannot be omitted
from an aggregate as if they were finished independent estimates. A production
restart/persistence protocol, real-SMC adapter, and tail/cost study are outside
this change.

## Predeclared CPU checks

The development evaluator tests exact finite-state transition probabilities,
the tilted invariant law, and the infinite expected correction using rational
linear equations. Scripted paths check first-step coupling, lag-one indexing,
the inclusive endpoint m, shared uniforms, and post-meeting faithfulness.
Additional checks cover zeros, deterministic replay, cancellation, numerical
failures, and independent finite-state replicas. Tests run through Slurm with
the checkout's imports pinned. No GPU or scientific efficiency claim is made.

## Validation receipt — 2026-10-22

Focused CPU job 17420741 passed 41/41 cases. The exact finite proposal has weights
(0,1,4), probabilities (1/3,1/2,1/6), and vector observables
((0,0),(2,-1),(-3,5)); its tilted expectation is (-6/7,17/7). Rational absorption
equations give that expectation for windows 0:0, 0:3, 1:1, 2:5 without a tail
cutoff. Two independent 12,000-replicate simulations satisfy their fixed
six-standard-error checks. These are finite-law development tests only.

The complete checkout suite passed 718 tests with 3 skips in CPU job 17420767.
Exact source hashes, commands, XML,
stdout, skip reasons, and simulation summaries are recorded in
[evidence/coupled_population_imh_20261022.json](evidence/coupled_population_imh_20261022.json).
