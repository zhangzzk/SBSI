# Mutation cost attribution

## Frozen replay and limits

GPU job17421460 completed in42 seconds on the pinned V100. The profiler wraps
the immutable source_v5b mutation methods and repeats the original benchmark
with fixed4096 density batches. Original prior/K/payload replay checks pass.
Each profile combines one warmup and five repeated saved-state calls; caching
and profiling change timings. These are cost clues, not a production speedup
or a matched-accuracy comparison. The4096-row case duplicates four observations;
it does not validate larger observation groups.

## Measured cumulative time

All numbers below are seconds accumulated across six calls. Nested columns
overlap and must not be added. An absent likelihood evaluation in the nuisance
block reflects its already-validated invariant-feature cache.

| Block | Rows | Full mutation | Native prior | Likelihood adapter | Scene proposal |
| --- | ---: | ---: | ---: | ---: | ---: |
| physical | 1024 | 0.199 | 0.055 | 0.103 | 0.000 |
| physical | 4096 | 0.342 | 0.119 | 0.106 | 0.000 |
| nuisance | 1024 | 0.120 | 0.108 | 0.000 | 0.000 |
| nuisance | 4096 | 0.390 | 0.365 | 0.000 | 0.000 |
| environment | 1024 | 1.081 | 0.084 | 0.123 | 0.801 |
| environment | 4096 | 3.617 | 0.167 | 0.124 | 3.104 |
| positions | 1024 | 0.297 | 0.084 | 0.127 | 0.000 |
| positions | 4096 | 0.575 | 0.178 | 0.132 | 0.000 |
| exchange | 1024 | 1.027 | 0.089 | 0.126 | 0.694 |
| exchange | 4096 | 3.298 | 0.195 | 0.129 | 2.609 |

## Next experiment

Exchange time is dominated by native proposal construction, including Python
categorical-draw loops. The first implementation hypothesis batches these
draws while retaining their numerical cumulative probabilities, RNG order,
support restriction and full scene payloads. It must reproduce trajectories
before any speed comparison. Neighbour generation also spends substantial
time in numerical morphology sampling; that does not justify replacing its
law with a faster approximation.

[Machine-readable evidence](evidence/mutation_cost_profile_20261022.json)
pins the binary profiles and receipt. Complete text profiles and benchmark
outputs are under `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/mutation_profile_20261022`.
