# N22 saved whole-run-output replay and cost check

## Current status

All four GPU pairs in array 17437046 and CPU report 17437047 complete with exit 0.
Every pair reproduces all saved particle/scene arrays and all 64 scientific stage
records exactly. All eight runs finish without extinction. These are saved-output
checks; the unchanged driver does not save every intermediate state or terminal RNG.

The full measured new-work totals are **852.338 seconds baseline and 841.762
seconds optimized**, a ratio of 1.01256 and 1.24% lower optimized time. These
totals include measured source preparation, full run processes, verification
and report overhead, conditional on the existing frozen fits. They exclude
historical pilot generation, fitting, tests and search costs.

| Pair | Order | Baseline charged seconds | Optimized charged seconds | Baseline / optimized |
| --- | --- | ---: | ---: | ---: |
| 0 | Baseline first | 219.029 | 209.121 | 1.04738 |
| 1 | Optimized first | 208.817 | 213.203 | 0.97943 |
| 2 | Baseline first | 212.375 | 199.815 | 1.06285 |
| 3 | Optimized first | 204.852 | 212.358 | 0.96466 |

Pair charges include shared pair overhead; totals additionally split source
preparation and final-report cost. The mean pair ratio is 1.01358 with descriptive
standard error 0.02438. Both baseline-first pairs favor optimized; both reversed
pairs favor baseline. The design does not identify the cause of this order pattern.
Four timings do not establish a dependable full-run speed benefit. **Leave the
optimization inactive**; preserve the tested candidate without further timing
experiments or integration for now. This decision does not reject its passed
replay checks or change the separate sampler acceptance gates.

The [complete frozen report](evidence/exchange_full_smc_result_20261022.json)
and [parent review](evidence/exchange_full_smc_review_20261022.json) preserve
individual costs, identities, successful scheduler exits and independently
recomputed cost arithmetic. GPU allocation totals 1,681 seconds; CPU reporting
takes eight allocated seconds. Held-out accuracy and inclusive sampler cost
remain pending, and no population shear-calibration claim follows.

## Scheduling history

GPU array 17437046 (`0-3%1`) shared the same four-GPU node as held-out work;
CPU report 17437047 retained its afterany dependency on the benchmark.
Before any benchmark task started, root removed the original dependency on
held-out completion after verifying that both arrays are pinned to th-cl-nv02
and that no other study GPU jobs run elsewhere. The node has exactly four GPUs,
so scheduler capacity enforces the combined ceiling while the benchmark remains
throttled to one worker. Eligibility permits backfill but does not promise an
earlier start. The [scheduling amendment](evidence/exchange_full_smc_scheduling_amendment_20261022.json)
preserves before/after scheduler state and unchanged scientific identities.
Root verified both304-file scientific
copies, all168 external inputs, harness/test/fixture receipts and the declared
single-source difference before submission. The [launch receipt](evidence/exchange_full_smc_launch_20261022.json)
preserves commands, identities and limits. The earlier
[partial parent review](evidence/categorical_parent_review_20261022.json)
pins the first two completed receipts without selecting timing results.

The older held-out array still had higher queue priority and consumed released
slots. Before any benchmark task started, a [temporary priority window](evidence/benchmark_priority_window_20261022.json)
changed only held-out array 17435350's Nice from 0 to 20. Its throttle stayed four;
the one-worker benchmark can run while held-out tasks use remaining capacity.
Running allocations and scientific configuration are unchanged. CPU job 17439409,
dependent afterany on the benchmark array, restores Nice to 0 if the same live
array still has the temporary setting. It preserves any subsequent owner change.
Eight isolated subprocess-fixture checks cover restoration, changed owner/node
settings, absent/terminal jobs, failed updates and unconfirmed restoration.
The first benchmark task 17437046_0 starts at 14:13:39 UTC on th-cl-nv02;
its baseline records meaningful annealing progress without extinction. The
priority receipt records startup evidence. After all four pairs finish, restore
job 17439409 completes successfully at 14:44 UTC. Its receipt records Nice=20
before and Nice=0 after; an independent live scheduler check confirms Nice=0,
throttle four and the original node pin. Held-out task 17435350_12 starts on the
released GPU. The temporary priority window is closed.

Bounded CPU status check 17439000 is scheduled for 18:00 Europe/Berlin
(16:00 UTC), covering both workflows and their final-report existence without
reading science values. Its [receipt](evidence/sampler_workflows_scheduled_status_20261022.json)
pins source/launcher hashes. It writes a status artifact and does not wake this
thread. The earlier held-out-only check 17436101 completed successfully at
14:12 UTC; its result is preserved in the original monitoring receipt.

## Tested integration candidate

An isolated candidate on `codex/categorical-integration-20261022`, commit
`8b3d85468b29507d63c84d2ab56f4e6a7727eefb`, contains the exact benchmarked
optimization. CPU job 17440535 passes 1,084 tests, including all 66 categorical
replay cases, with nine documented optional skips. All 454 frozen source hashes
match before and after. The [candidate record](CATEGORICAL_INTEGRATION_CANDIDATE.md#validation-result)
preserves the complete source inventory, launcher and validation receipt.

Independent parent review confirms the caller's existing probability validation
and contiguous target-type draw order. The replacement follows NumPy 2.2.6's
float64 CDF, normalization and right-sided search; neighbour draws, support
corrections and relocation arithmetic are unchanged. Parent review independently
checks the JUnit counts and receipt hashes. Only these review documents are
copied to the root worktree. The completed four-pair timing result above does
not establish sufficient benefit for integration, so runtime code remains on
the isolated branch. The held-out scientific source remains unchanged.

## Fixed design and decision boundary

Prepare only; no inference is submitted by this change. Four development pairs
use seeds2026103600–2026103603 and ordered observations[58165,0,1,8]. Both arms
use g=0, N256/T64/M64, power2, scale1, density batch4096, the original sixteen
blocks, geometry_gradient_v1.json and pilot_initial_proposal_n13_v2.json.
Each arm has its own Python process, including model/import/prior preparation
and the driver's F64 score and replay diagnostics. Even replicas run baseline
first; odd replicas run optimized first. No warmup population, adaptive choice,
retrial or extra scientific diagnostic is added.

Both frozen arms copy every manifest file from corrected v10's
source_support_fix_20261022. They differ only in sbsi/scene_exchange.py: corrected
scalar baseline c5102579… versus previously validated categorical candidate
738df166…. The current held-out source and root worktree remain untouched.
The development declaration, measured reference, original pilot artifacts,
V3.9 model artifacts, native-prior HDF inputs and installed GalSBI/PyCosmo package
files are hashed. This is not a complete runtime/container image.

## Exact comparison boundary

Each arm must first match the complete audited dev_v10_small_0 protocol,
with only its declared replica seed, checked implementation locations/hash,
and Slurm job ID changed. The original protocol is pinned as the recipe/runtime
template. Python, NumPy, Torch, device, GPU model, node and thread count must
match that template. Fixed settings, observation order, proposal/geometry
metadata, all65 beta values and all other protocol fields must match as well.
Progress requires exactly64 requested and completed stages numbered1–64,
each at its declared beta, ending at1. The result's final stage must equal the
last progress record, including its within-run timestamp. Both arms agreeing
on an incorrect runtime, truncated history or wrong recipe therefore fails.

Compare every particles.npz/scenes.npz array by field name, dtype, shape and
bytes, including signed zero and NaN representation. Validate each archive's
own original SHA256 before comparing decoded arrays. Compare all result,
protocol and 64 progress-stage scientific values exactly, including all audit
counters. The only permitted differences are:

- Result timing_seconds.initialization/sampling/total and final_stage.elapsed_seconds.
- Progress stages[*].elapsed_seconds.
- Protocol runtime.slurm_job_id. GPU, node, versions, threads and scientific
  proposal metadata must match.
- Protocol implementation path roots, normalized only after validation against
  each arm's frozen manifest; only the known scene_exchange.py hash may differ.
- Result protocol_sha256/particles_sha256/scenes_sha256, only after checking
  each original artifact hash. NPZ container bytes may differ while arrays agree.

This is **saved whole-run-output replay**, not an audit of every intermediate
mutation state or terminal RNG state, which the unchanged driver does not save.
The earlier 18-step full-state/RNG audit remains separate evidence. No driver
instrumentation is added to the source-frozen experiment.

## Cost and failures

The outer /usr/bin/time records each pair process, including harness/import,
source verification and array comparison. Each arm has its own subprocess wall
time and unchanged driver timing fields. Shared pair overhead is split equally;
report subprocess cost and source-copy/hash preparation are also charged equally.
The report calls the resulting totals `new_work_seconds_by_arm`: these cover
complete new run processes and measured copy/hash/report overhead, conditional
on the existing frozen geometry and initial proposal. Historical pilot
generation, fitting and search costs are neither recreated nor included.
Time's hundredth-second rounding can yield a tiny negative remainder, clipped
at zero and visible from raw costs. No scientific warmup/report is hidden.
Tiny final declaration/manifest serialization and JSON-only report finalization
are explicitly outside the measured window; report these bookkeeping limits.

Every arm's exit code, log and wall cost is retained. A nonzero first arm does
not cause the second arm to be omitted. Existing output directories prohibit
retries. Missing/failed pairs, changed artifacts or any replay disagreement
block a completed timing report. All four pairs are required; no favorable
subset is reported. Give individual costs, counterbalanced order, totals,
range and descriptive standard error across four ratios. Four pairs on one
V100 do not justify a strong timing-distribution claim. No convergence,
normalizer-accuracy, held-out acceptance or promotion follows from this check.

## Command graph

The GPU array uses inter/th-cl-nv02, one V100, two CPUs and array0–3%1.
Its original declaration recorded afterany:17435348:17435350 as a resource
guard. The pre-start scheduling amendment above supersedes only that allocation
dependency; the original declaration and all scientific inputs remain immutable.
The combined four-GPU ceiling now follows from both arrays' mandatory node pin
and that node's four-GPU capacity. Recheck the limit before submitting study GPU
work on any other node.
The CPU report depends afterany on the new GPU array;17435351 is not a
prerequisite. The prepared declaration and exact commands are recorded in
[evidence](evidence/exchange_full_smc_preparation_20261022.json).
The first prepared snapshot is retained but superseded for submission by
`exchange_full_smc_prepared_v2_20261022`; the v2 receipt records its fresh
declaration, manifests, unchanged scientific sources and updated commands in the
[v2 receipt](evidence/exchange_full_smc_preparation_v2_20261022.json).

## Focused validation

CPU17436600 passed12 initial focused tests. Final CPU17436699 passed13 in
0.29s and completed file-copy/hash-only frozen preparation. It adds an
end-to-end synthetic saved-output comparison, including archive corruption,
audit counters and source-map normalization. Tests cover explicit timing
exceptions, signed-zero/type/shape differences, unexpected source changes,
fixed command settings, both-arm failure preservation, retry rejection and
shared report costs. No full suite or GPU science run is requested here.

The initial13 tests did not enforce agreement with the audited runtime/recipe
or a complete64-stage history. V2 adds explicit failures for each runtime
field, seed/settings, requested count, missing/misnumbered stages, beta/endpoints
and final-stage consistency. A separate scheduled CPU fixture exercises the
real comparators on copies of audited dev_v10_small_0 outputs. It changes only
top-level protocol implementation roots/hash and the result's protocol hash.
All scientific values, original seed, stage history and NPZ files remain
unchanged. A disclosed temporary test-only seed mapping in the imported harness
allows the existing development seed to exercise the comparator; production
has no alternate-seed flag. Original/copy hashes and exact rebinding are saved
in the fixture receipt. This checks the verification path, not new inference.
CPU17436954 passed all35 focused tests in0.22s, prepared v2, and passed this
audited-output integration fixture in the same20-second allocation. No inference
or GPU job was submitted; the prior prepared snapshot remains unchanged.
