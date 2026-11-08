# Sampler concurrency benchmark

## Request and status — October 26, 2026

Owner requested a benchmark of 30, 40 and 46 simultaneous sampling processes
sharing one GPU. Slurm job **17509905** completed all eight phases in
**1 h 15 min 41 s**, with exit code 0, on one **H200 NVL**, with 48 logical
CPUs and 320 GiB host memory. The model-symmetry branch
contains this record; numerical code is the immutable image-study snapshot
`06727be`. No model or production sampler settings were changed.

Study root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_concurrency_20261026_v2`.

The study-local `rule.md` and `plan.json` freeze the question, workloads,
seeds, decision thresholds and hashes before submission. `benchmark.py`
runs the experiment and produces `report.json`; `launch.sh` requests resources.
`BENCHMARK_SHA256SUMS` pins all four inputs. `hardware.json` records the
actual device UUID/model, driver/CUDA versions and allocated CPU IDs.

## Workloads and measurement

Use the original 99,932-observation image-study reference. Bright units retain
609/610 observations and eight tempering levels; faint units retain 1022/1023
observations with no tempering. All processes use four chains and the same
density batches, proposals and shear (zero). The first 30 tasks, observation
lists and seeds are identical across concurrency choices. Bright source units
repeat cyclically to supply 46 processes; these are timing replicas and are
never counted as independent science observations.

Short runs have 1024 steps, 128 warm-up and stride 16. Production remains
20480 steps and 8192 warm-up. Phase orders bracket the comparison:

- Bright: 30, 40, 46, 30.
- Faint: 30, 46, 40, 30.

The primary rate counts galaxy-steps in the common post-warm-up interval
when every process is between steps 128 and 1024. Require at least 30 seconds
of overlap; save progress receipt timestamps and GPU utilization/memory
samples every three seconds. Divide galaxy-step throughput by 20480 only
to report a projected production rate. Report launch-to-completion time
and sampler timing components separately.

## Numerical replay and decision

Every process must finish, preserve its assigned observations/protocol/seeds,
and pass initial/invariant kernel and summary replay plus the co-moving identity
check. Common first-30 tasks must reproduce initial states, final scenes,
profile arrays, frozen scales and morphology factors exactly. Additional
workers pass the individual checks but have no 30-process counterpart.

For each workload, compare with the geometric mean of the two baseline rates.
Recommend increased concurrency only with:

- At least 10% throughput improvement.
- At most 10% drift between baseline repeats.
- Exact common-task replay.
- Peak GPU memory below 90%.

A failed worker or 60-minute phase timeout stops the job. An insufficient common
timing window or excessive baseline drift makes the comparison inconclusive.
No convergence or scientific acceptance is inferred from these short runs.

## Validation and limits

Before submission, shell/Python syntax passed. Lightweight reducer checks
passed for overlap interpolation, insufficient-window rejection, array equality
including NaNs, speed/drift/replay decision gates, and the GPU-UUID interface.
The source manifest and reference hashes are verified again at job startup.
No full repository suite was run: only study-local orchestration and
documentation were added; production numerical code is unchanged.

A different allocated GPU from the old V100 run requires hardware-specific
absolute times. This measures one GPU at zero shear, not four-GPU scaling,
500-CPU scaling, the nonzero bright second step, convergence, or mirror
averaging. A million-observation estimate must retain those distinctions
and the current image-sample magnitude mix.

## Completed results

All 292 worker runs completed. All 180 comparisons of common tasks reproduced
initial states, final scenes, profiles, frozen scales and morphology factors
exactly. Maximum initial and invariant log-kernel replay errors were zero;
maximum invariant summary change was 4.27e-14. The two 30-process baselines
drifted by only 0.063% (bright) and 0.258% (faint).

Rates below are projected galaxies per GPU-hour at 20480 steps, calculated
from the measured common sampling interval. They exclude initialization,
output and diagnostics. Baseline rates are geometric means of the two repeats.

| Processes per GPU | Bright rate | Bright gain | Faint rate | Faint gain | Peak memory across both workloads |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 | 8,852 | baseline | 43,749 | baseline | 22.44 GiB |
| 40 | 9,644 | +8.95% | 48,913 | +11.80% | 29.84 GiB |
| 46 | 10,210 | +15.35% | 52,699 | +20.46% | 34.29 GiB |

Both workloads select **46 processes on the tested H200** under the frozen
decision rule. At equal workload this means about 13.3% less bright sampling
time and 17.0% less faint sampling time. At 46 processes, GPU utilization in
the common interval averaged 67.3% for bright and 82.5% for faint workloads.
All common intervals passed the 30-second gate, but the faint 46-process
interval was only 33.77 seconds: this remains a short throughput measurement.

With the tested allocation, four GPUs would use 184 sampler processes and
192 logical CPUs including controller capacity. Four-GPU scaling and the
use of all 500 CPUs were not measured. The measured 34.29 GiB footprint at
46 processes makes transfer to a 32 GiB V100 unvalidated; GPU architecture
and driver can change per-context memory costs. Absolute H200 rates must
not replace the previous V100-based end-to-end time estimate. A full run
also includes initialization, diagnostics and the bright nonzero-shear
second step. Mirror averaging and convergence remain untested.

Recommendation: use 46 as the candidate concurrency for an H200 production
validation. Retain the existing production settings until the target hardware
and full-run resource behavior are checked. No production defaults changed.

Evidence: study-local `report.json` (SHA256
`76f1ab35e7f15cd2cd277f15c9561933a4107143454950585c628e308c7f436e`),
per-phase `summary.json` / `events.json` / `resources.jsonl`, and
`status_final.json`. Post-run audit independently recomputed baseline rates
and speedups, checked all worker/replay counts and timing gates, and verified
all four `BENCHMARK_SHA256SUMS` entries. Scheduler accounting confirmed
COMPLETED / 0:0. The completion check 17509923 finished; the redundant pending
two-hour check 17509922 was cancelled after reviewing the completed results.

## Attempt history

Job17509655 stopped before any sampler worker started: PyTorch returned a bare
GPU UUID and nvidia-smi required the GPU- prefix. The submitted files are
preserved in the study's attempt_17509655 directory. Added the prefix only
when absent, reran syntax checks, refreshed the orchestration hash manifest
and resubmitted as17509685. The allocated hardware in the first attempt was
an H200 NVL, so absolute timing must not be labelled a V100 measurement.
Workloads, seeds, model code and gates were unchanged.

The first384-step bright baseline completed all30 workers and passed their
replay checks, but its common post-warm-up overlap was only17.2157seconds,
below the frozen30-second minimum. Stopped17509685 after that timing failure;
the40-process phase had begun initialization. All outputs remain in v1.
Before any valid concurrency comparison, increased every timing run to1024
steps, retaining128 warm-up, all workloads/seeds and the original decision
gates. The v2 controller stops immediately on an insufficient timing window.
Resubmitted as17509905 on the same H200 NVL node. All30 first-baseline workers
passed initial replay with maximum error0 and reached sampling (observed step
range52–256 of1024). This established healthy startup before the completed
comparison reported above.
Hardware: H200 NVL, driver580.178.04, PyTorch2.9.1+cu126;48 allocated logical
CPUs. The source sampler, batches, tasks and seeds remain fixed.

One-shot CPU status checks17509922 (about two hours after submission) and
17509923 (after benchmark termination) were scheduled to write compact receipts.
The controller automatically validates each completed phase before advancing.
