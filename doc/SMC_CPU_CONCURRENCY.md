# SMC CPU concurrency — October 22, 2026

## Completed development result

GPU job **17445452** completes all 24 original runs in 3,091 allocated seconds
on one V100 and eight physical cores (16 logical Slurm CPUs). Every setting
preserves the full 64-stage scientific history and saved particle/scene bytes.
Independent scheduled audit **17447670** replays the complete checks and
reproduces the report, allowing only the new audit's elapsed time to differ.

| Workers per GPU | Active physical cores | Forward speedup | Reverse speedup |
| --- | ---: | ---: | ---: |
| 1 | 2 | 1.00 | 1.00 |
| 2 | 4 | 1.959 | 1.904 |
| 4 | 8 | 3.570 | 3.474 |

These are predeclared setting-inclusive ratios: verification, model/process
startup, sampling, saving and saved-output comparison all count. All settings
held the same eight-core reservation. The four-worker settings took
242.263/243.089 seconds for the same four populations, versus
864.905/844.449 seconds serially. Original setting/report process timers sum
to 3,088.90 seconds; preparation adds 6.149 seconds outside the GPU allocation.
Including evenly split new preparation/report cost leaves the four-worker
ratios at **3.544/3.448**. Existing scientific pilot/geometry fits are excluded.

The predeclared ranking selects **four workers** for confirmation on untouched
seeds 2026103800–2026103803. No runtime adoption yet: both fresh timing repeats
must clear the fixed gate with exact replay. This is development throughput
for one four-observation group, not catalogue throughput, four-GPU scaling,
500-CPU capacity or posterior-accuracy acceptance. See
[full timing and independent audit](evidence/smc_cpu_concurrency_development_20261022.json).

The restoration controller 17445497 returned failure because Slurm included an
already-finished array task in its update response. The update did restore the
live array limit: independent current-state inspection confirms all pending
and running records have `ArrayTaskThrottle=4`. The original failed receipt is
preserved; no rerun or additional scheduler mutation was needed.

## Hypothesis and fixed experiment

Independent two-CPU processes sharing one V100 can overlap host-side prior and
scene calculations and improve complete production throughput. The numerical
sampler is unchanged. This is Arbor intervention n24 under implementation
efficiency n21. The mutation-cost profile motivates this experiment; it does
not predict a speedup.

The workload is four complete populations on observations 58165, 0, 1 and 8.
Each population uses 256 particles, 64 stages, 64 moves, the original 16-block
schedule, density batch size 4096, shear (0,0), and existing frozen development
geometry and initial proposal. Seeds are 2026103700 through 2026103703.
Every concurrency setting runs all four seeds.

One allocation reserves a V100 on th-cl-nv02 and eight physical CPU cores.
Concurrency follows **1, 2, 4, 4, 2, 1**. Each worker is pinned to two distinct
physical cores; all settings use the same CPU reservation and GPU. Reverse
order helps expose timing drift. Identical seeds permit exact replay;
timing repetitions are not new independent scientific replicas. No MPS,
model, likelihood-group, thread-count or sampler-setting changes. The
benchmark does not inspect or change ongoing held-out scientific outcomes.

## Development and adoption gates

Require every original run to complete and match the declared recipe and all
64 stages. Compare saved scientific JSON and every particle/scene array's
dtype, shape and bytes; allow only checked provenance and named timing
differences. No retries, replacement seeds or omitted failed settings.
Saved-output equality does not establish sampler accuracy; that remains a
separate held-out gate.

A level above one is eligible when all replay/operational checks pass and
both counterbalanced speedups over concurrency one are at least 1.25. Rank
eligible levels by their worse repetition's speedup; within five percent,
prefer fewer workers. This selects a development candidate only. Adoption
also requires an unchanged-recipe evaluation on untouched seeds 2026103800
through 2026103803 with counterbalanced timing, declared before execution.

Primary timing is complete setting wall time, including child startup, model
loading and saving. Preserve child process times, affinities, GPU identity
and memory, failures, and source/comparison overhead. Full allocated process
costs accompany throughput. Existing pilot fitting and phase diagnostics are
outside this experiment; its rate is not complete catalogue throughput.

## Resource and source boundary

Use the exact 320-file corrected held-out source, manifest
`fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1`.
Preparation verifies active code against the audited development protocol,
copies the source read-only, and pins harness, models, prior packages and
inputs. The categorical optimization remains inactive.

The Slurm allocation requests one GPU, eight CPU cores and
`--hint=nomultithread`; the step uses `--cpu-bind=cores`. Eight chosen CPU IDs
must map to eight distinct physical cores. th-cl-nv02 has 48 physical cores
and four V100s. Keeping this experiment on the same node as the study
preserves its four-GPU total ceiling. Check actual allocation/startup.

500 cores exceed this node's capacity. Remote CPU work would need an
explicit GPU-request service or different GPU hosts. First measure local
worker concurrency and contention. Preserve original likelihood requests
instead of merging them into unvalidated larger numerical batches.

## Implementation and validation

`scripts/benchmark_smc_concurrency.py` manages isolated workers, CPU affinities,
receipts and comparisons. `scripts/prepare_smc_concurrency.py` freezes source
and declaration. `scripts/run_smc_concurrency.sh` executes the complete order
and records process costs. Implementation is committed as 0ee415b; preparation
and launcher are committed as 13a1fca on `codex/smc-cpu-concurrency-20261022`.

All 35 focused tests pass in 2.57 seconds. CPU job 17445433 passes 1,064 tests
with 11 skips in 101.94 seconds (138 allocated seconds, two CPUs). Two additional
skips are real process-affinity tests requiring four physical cores; both
passed in the focused run. All 478 test-snapshot hashes match before/after.
Frozen preparation verifies all 320 sampler files and pinned external inputs.

GPU job 17445452 is submitted on one V100/eight physical cores. The held-out
array is temporarily throttled to one new concurrent task, without interrupting
its already-running tasks. Completion job 17445497 restores its original
throttle of four. Guard job 17445498 restores earlier if the benchmark has not
started after 20 minutes. Both studies remain pinned to the same four-GPU node.
No live scientific settings or sources change.

The [preparation receipt](evidence/smc_cpu_concurrency_preparation_20261022.json)
pins declaration SHA256
`34383474bb5c5bdce06be6a711ae9b088b1ee5e2d5b48a14a1545892951ff185`
and preserves test, source and scheduling evidence. Startup confirms the
CUDA-visible V100 UUID, eight distinct physical cores (CPU IDs 32–39), correct
two-core worker affinity and progress through the first population without
extinction. Slurm reserves both SMT siblings and accounts for 16 logical CPUs;
only eight physical cores are selected for workers. Automatic requeue is
disabled. Complete replay and timing results remain pending; no speedup or
adoption is claimed.


## First concurrent stage

All four serial baseline populations completed normally. The two-worker stage
then started on the same GPU: workers are pinned to physical cores 32–33 and
34–35 and both reached stage 17/64. This establishes actual concurrent CPU
execution and normal startup, not a measured speedup or replay result. Four
workers and reverse-order repetitions remain required.


## Path to many CPU workers

The next scaling boundary is independent CPU scene/proposal work calling a
small pool of persistent GPU likelihood evaluators. CPU workers should retain
per-population RNG, target/prior state, acceptance and resampling; a GPU service
should own resident models and evaluate explicitly identified requests. The
candidate interface is the existing summary/auxiliary-array to log-likelihood
boundary. Preserve each original four-observation group, row order, padding,
dtype and numerical batch size rather than implicitly combining requests.

Start with local processes and measured memory/contention. A remote service
would then need bounded request queues, deterministic request/result matching,
explicit failures, and shared-data placement. Its validation must compare a
single request, a full population, and concurrent populations against the
original saved GPU results before a throughput test. Measure serialization,
network, model loading and queue costs together. Five hundred allocated CPU
cores are useful only if their extra proposal work increases complete accepted
catalogue throughput; they are not a proportional speed multiplier. This is a
follow-up design, not an implemented or measured distributed service.
