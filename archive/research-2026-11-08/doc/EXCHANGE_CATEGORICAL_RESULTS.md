# Native exchange type batching: CPU and trained-model replay evidence

## Decision and scope

The fixed n22 candidate passes correctness checks and exceeds its predeclared
CPU proposal-time threshold. Its development score is **1.3609505278392875**:
the geometric mean speedup of ordinary 1024/4096-row proposals without
relocation. This is a microbenchmark score, not scientific sampler efficiency.
The trained-model check also passes bitwise replay, with ordinary full-exchange
block speedups of 1.132–1.169x. Runtime integration remains deferred to preserve
the corrected development/held-out scientific source identity; documentation
and evidence can be integrated independently.

The candidate changes only the temporary target-type categorical draws in
`sbsi/scene_exchange.py`. The corrected 55f1220 baseline and candidate have
identical native support checks. Neighbour selection, ragged reductions,
relocation, and target factors are unchanged. The
[predeclared design](EXCHANGE_CATEGORICAL_BATCHING.md#fixed-candidate-and-numerical-contract)
links the installed NumPy 2.2.6 implementation used to reproduce its arithmetic.

## Validation receipts

All external receipts are under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/exchange_categorical_validation`.
The [portable receipt](evidence/exchange_categorical_20261022.json) includes
source hashes, job results, and every timing sample.

- CPU 17422591: **872 passed, 4 skipped**, one existing NumPy warning, 92.91 s.
  `junit_v1.xml`, `tests-v1-17422591.out`, and `input_v1.json` preserve the run.
- CPU 17422592: all **20 proposal cases** preserve full payload and RNG bits
  across **520 paired-arm calls**, including two warmups and 11 timed repeats.
  `benchmark_v1.json` preserves all samples and before/after source hashes.
- CPU 17422766: **67 passed** in 15.03 s after strengthening full-trajectory
  assertions to byte equality and adding the frozen-manifest guard test.
  `junit_final.xml`, `final_input.json`, and `final-17422766.out` preserve this
  final check and successful saved-input preflight. Runtime code did not
  change after the full-suite run. The earlier test source is retained as
  `test_exchange_categorical_replay_v1.py` beside the receipts.

The tests cover five bit generators, float32/64, zero category probabilities,
right-CDF boundary behavior, empty/single/many/ragged scenes, changing type
probabilities, support-zero marks, relocation, and full mixed-block MH
trajectories with optional informed-environment payloads. There were no failed
jobs in this candidate's validation. Unsupported RNG/dtype combinations use
the original scalar implementation.

## Measured CPU cost

| Ordinary proposal | Baseline median | Candidate median | Speedup |
| --- | ---: | ---: | ---: |
| 1024 rows, no relocation | 54.421 ms | 40.583 ms | 1.341x |
| 4096 rows, no relocation | 209.582 ms | 151.736 ms | 1.381x |
| 1024 rows, relocation | 194.648 ms | 180.208 ms | 1.080x |
| 4096 rows, relocation | 820.841 ms | 766.546 ms | 1.071x |

Counterbalanced paired timings use fixed scenes and matched fresh seeds.
Empty-scene timings are approximately unchanged. Relocation gains are smaller
because its additional neighbour-selection work remains. These results concern
one process, one CPU allocation and NumPy version; GPU evaluation and complete
sampler costs were not measured. No sampling-law or convergence improvement is
claimed.

## Trained-model comparison

`scripts/benchmark_exchange_trained.py` compares the corrected scalar control
against the candidate in the same immutable source tree. It reads completed
`dev_v5_0` as fixed benchmark input and preserves the original four observations,
1024 rows and 4096-row likelihood batches. It verifies saved input hashes,
V3.9 artifact hashes, saved-state reconstruction, and complete candidate/control
state, decision and RNG byte equality. The prepared test uses 18 mixed-block
trajectory steps, followed by paired exchange and relocation timings at beta
0.001 and 1 with one warmup and seven timed repeats. Only the type-draw function
differs between arms; the old saved-run source is not used as the control.

The standalone launcher requires a scheduled job and a fresh output. The
immutable snapshot is retained at
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/exchange_categorical_source_20261022`.
The parent submitted GPU **17422901**, dependent on completed array 17422526,
on one V100 at `th-cl-nv02`. It completed successfully in **45 seconds**. The
[trained-model receipt](evidence/exchange_categorical_trained_20261022.json)
preserves the full output, scheduler record, source/input hashes and original
log hash. The source manifest hash is
`acaa720e64a6017413d19cee8925f27abb0b38383b663c7c671677b5d598cf71`.

Saved kernel and native-prior reconstruction errors were both exactly zero.
All 18 mixed-block trajectory transitions and all paired timing calls preserve
full state, acceptance/active decisions and RNG bytes. The input population,
four-observation grouping, 1024 rows and 4096 density batch remain fixed.

| Full exchange block | Baseline median | Candidate median | Speedup |
| --- | ---: | ---: | ---: |
| beta 0.001, no relocation | 83.618 ms | 71.520 ms | 1.169x |
| beta 1, no relocation | 85.268 ms | 75.316 ms | 1.132x |
| beta 0.001, relocation | 277.643 ms | 279.469 ms | 0.993x |
| beta 1, relocation | 268.655 ms | 254.688 ms | 1.055x |

The ordinary full-block improvement survives the actual trained-model path.
Relocation does not improve uniformly: its low-beta median is slightly slower.
This negative case is retained without further tuning or another run. Seven
timed pairs on one fixed population and one allocation do not establish a
general timing distribution or full-SMC wall-time saving. The complete sampling
trajectory over all temperatures was not benchmarked. Neither convergence nor
normalizer accuracy changed by design. The primary predeclared development
metric remains the CPU proposal score above; these GPU timings are a separate
feasibility check. The isolated runtime candidate remains available for later
integration after the current source-frozen held-out procedure.
