# Waste-free prototype result — 2026-10-22

## Decision

Do not integrate this configuration into the scene sampler. Correctness gates
pass, but the predeclared matched-cost experiment gives no Gaussian gain and a
clear separated-mixture regression. This bounds one fixed recipe; it does not
establish that all waste-free choices fail.

The frozen [design and own expectation derivation](WASTE_FREE_SMC_PROTOTYPE.md#own-finite-population-derivation)
were written before results. CPU suite17421477 passed793 tests, with4 skips,
including18 new checks. Exact enumeration over a finite two-stage target with
zero initial mass and an extinct population gives E[Zhat]=5/3 and
E[Zhat*fhat]=10/3. No failures were discarded.

## Matched-cost evidence

CPU experiment17421484 completed successfully in30 seconds. Each target/arm
has128 fresh independent populations. Standard SMC used120 endpoints and
7 mutation steps per temperature; waste-free used113 paths of length8, retaining
904 states. Both consumed exactly13,560 target density rows per population,
including initialization, with identical exact caching and16 fixed stages.
Both also used113 actual density batches and241 potential calls per population.

| Target | Waste-free / standard primary variance | 95% population-bootstrap interval | CPU median time ratio |
| --- | ---: | ---: | ---: |
| Correlated Gaussian | 1.0894 | 0.8342–1.3728 | 1.1646 |
| Separated mixture | 2.7551 | 2.0077–3.7195 | 1.1533 |

The primary variance averages the five observable influence variances divided
by their analytic posterior variances. At matched density cost, lower is better.
The predeclared synthetic development score, geometric mean of the inverse
variance ratios, is **0.5772146510222352**; the baseline is1. This is not a
scientific acceptance or real-sampler score.

All four arms pass the fixed5-standard-error checks on mean Zhat and Zhat*fhat;
the largest standardized discrepancy is2.603. Normalizer relative variance
nevertheless worsens from0.01181 to0.02013 for the Gaussian and from0.01080 to
0.02151 for the mixture. Passing these finite checks is not an additional
general convergence theorem.

## Where precision was lost

For the mixture, the x0 mean variance increases3.436-fold and positive-mode
mass variance increases3.408-fold. The x1 mean is essentially unchanged
(ratio0.999), while second-moment variances modestly decrease (0.844 and0.904).
Thus good local moment behavior does not rescue the mode-allocation estimate.
The pooled positive-mode masses are0.7115±0.0076 for standard and
0.6667±0.0140 for waste-free, against analytic truth0.7; these uncertainties
use whole-population ratio influence estimates, not path points.

Median surviving initial-ancestor counts fall from34 to13 for both targets.
This is consistent with dependence and loss of genealogical diversity; the
experiment does not isolate causation. The arms also spend different fractions
of their equal budget on initialization versus mutation. The904 retained states
cannot be counted as904 independent posterior draws.

## Scope and feasibility

The configuration is frozen after this bounded negative result, with no adaptive
tuning, held-out use, GPU job, or real-sampler integration. Extinction/support
contracts and full-state retention still need to be respected in any future
adapter. The prototype supports only NumPy arrays, not native scene payloads.
Actual likelihood calls pad to4096 rows and preserve fixed observation groups;
fewer active parents need not reduce GPU cost. This small CPU experiment cannot
authorize a new batch shape or establish GPU throughput.

The complete source hashes, seeds, exact count checks, XML and scheduler
receipts, estimates and per-observable metrics are linked in
[evidence/waste_free_smc_20261022.json](evidence/waste_free_smc_20261022.json).
External raw per-population arrays remain under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/waste_free_validation/experiment_v1`.
