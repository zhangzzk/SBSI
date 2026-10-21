# Native exchange type batching: CPU evidence and pending GPU replay

## Decision and scope

The fixed n22 candidate passes correctness checks and exceeds its predeclared
CPU proposal-time threshold. Its development score is **1.3609505278392875**:
the geometric mean speedup of ordinary 1024/4096-row proposals without
relocation. This is a microbenchmark score, not scientific sampler efficiency.
The parent should integrate documentation and evidence first, holding the
runtime change until the trained-model replay/cost check finishes.

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

## Prepared trained-model comparison

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
snapshot is prepared at
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/exchange_categorical_source_20261022`.
The parent may submit one V100 on `th-cl-nv02` after job array 17422526 completes;
the separately prepared phase job plus this comparison remain within the
parent's combined GPU limit. **No GPU job was submitted by this task.** A failed
replay must retain its receipt and block integration; positive timings alone
are insufficient.
