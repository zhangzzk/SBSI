# Sampler computation cost — October 22, 2026

## Confirmed local CPU sharing — October 22

One V100 with four independent workers, each pinned to two physical CPU cores,
completes the fixed four-population workload in about 252 seconds, compared with
825–843 seconds serially. Predeclared fresh-seed confirmation and independent
saved-output replay both pass. Complete-setting speedups are 3.27–3.34;
including shared preparation/reporting gives **3.25**. See the
[full scope and receipts](SMC_CPU_CONCURRENCY_CONFIRMATION.md#completed-confirmation-and-independent-audit).

Four GPUs at this layout would require 32 physical CPU cores, but its combined
throughput is not yet measured. The result does not justify multiplying existing
full-pipeline catalogue projections by 3.25: this benchmark covers one fixed
production recipe and excludes pilot fitting and scientific acceptance work.
Using 500 CPUs across nodes requires a separately validated distributed CPU/GPU
execution design. The reusable local pool is the next implementation step.

## Measured recipe and evidence

The current candidate projects to about **0.007 galaxies/s on four V100s**,
or **25 galaxies/hour**, with its current eight-population validation budget,
fresh pilots and per-population score diagnostics. This is a timing projection;
held-out scientific acceptance failed on five of twenty objects. The eventual capacity of four
GPUs with 500 CPU cores is unmeasured.

The complete twenty-object report subsequently measured 11,782.24 candidate
process seconds and 22,896.69 reference process seconds, including the declared
preparation/reporting costs. Its 1.9433 cost ratio is not a matched-accuracy
speedup because the candidate failed acceptance. See
[complete result and cost scope](CORRECTED_HELDOUT_RESULTS.md#scope-and-cost-convention).

The [timing receipt](evidence/sampler_compute_cost_20261022.json) records the
first two complete four-observation held-out groups, all 18 stage identities,
and the calculation. Stage costs reconcile to steps plus verification, and all
referenced step-timing file hashes match. Only operational timing receipts were
read for this estimate; no interim posterior comparisons were inspected.

Each production population uses 256 particles, 64 annealing stages and 64
mutation moves per stage. Eight independent populations give 8,388,608 particle
proposal opportunities per galaxy. Four pilot populations with 32 moves add
2,097,152. Invalid proposals can avoid some expensive evaluations, and proposals
are batched; these counts are neither independent posterior draws nor separate
GPU calls. All work is at one fixed shear centre, g=(0,0).

## Four-GPU rate and observation scaling

The following are mean process costs per four-observation group, measured with
one V100 and two allocated CPUs. GPU allocation time includes CPU work and
waiting; it is not GPU busy time.

| Component | Seconds per group |
| --- | ---: |
| Four pilots, pilot phase diagnostics and fitted geometry/proposal/coefficients | 441.49 |
| Eight production populations | 1,647.24 |
| Production phase diagnostics | 58.56 |
| All warmup and paired-production verification overhead | 209.55 |
| Sum charging all this verification to candidate | 2,356.85 |

The reference-arm sampling and its diagnostics are excluded. Charging all
paired verification to the candidate is conservative attribution; the final
comparison instead splits shared verification equally. Omitting verification
gives 0.00745 gal/s; charging all gives 0.00679 gal/s. This range is an accounting
comparison, not a statistical confidence interval. Final panel reporting,
global preflight, historical search/tests, queueing, mock generation, selection
normalization and additional shear centres are outside this projection.

| Scope | Galaxies/s on four V100s |
| --- | ---: |
| One production population, inference process only | 0.0777 |
| Eight production populations, inference processes only | 0.00971 |
| Eight populations plus pilots, fitting, phase diagnostics and all verification above | 0.00679 |

The one-population row does not satisfy the current eight-population validation
protocol. Eight is our current validation budget, not a universal minimum for
every galaxy or for a final catalogue estimator.

For N observations with the same per-observation recipe, cost is approximately
linear: T(N) = N * 589.21 GPU-seconds / G, with G fully available equivalent
GPUs. Four GPUs give about 147 seconds of total wall time per completed galaxy
in a sufficiently large, balanced workload. This is aggregate throughput;
individual group latency and small-job startup still matter.

| Observations | Projected wall time on four V100s |
| --- | ---: |
| 100 | 4.1 hours |
| 1,000 | 40.9 hours |
| 20,000 | 34.1 days |
| 500,000 | 852 days, about 2.3 years |

These extrapolations assume the two measured groups represent later per-group
costs. They are not catalogue-scale benchmarks. Repeating the entire inference
at additional shear centres adds corresponding cost. Reducing the production
population budget approximately reduces that component linearly; pilots and
other preparation remain. Any reduction needs a new aggregate score-error
assessment, because sampling bias need not cancel across observations.

## What 500 CPUs can change

The current driver sets at most two Torch threads per worker, with one worker
per GPU in the held-out launcher. Thus four workers allocate eight CPU cores.
Assigning 500 cores to those same four processes does not make the existing
code use 500 cores effectively.

The [mutation profile](evidence/mutation_cost_profile_20261022.json) points to
substantial host work: native-prior evaluation accounts for about 90% of the
profiled nuisance mutation, environment refresh about 74% of its mutation, and
exchange proposal construction about 68% of exchange mutation at 1,024 rows.
These are profiled calls on saved final-stage states, not a whole-run resource
decomposition. They support testing more CPU workers sharing each GPU, with
persistent loaded models and independent per-worker random streams. They do
not establish a saturation rate or GPU utilization fraction.

At two cores per worker, 500 cores nominally permit 250 workers, but GPU memory,
GPU service time, transfers and shared storage constrain useful concurrency.
CPU availability must include physical placement/connectivity to the GPUs.
There is no basis for multiplying the present rate by 500/8. A concurrency
sweep on small distinct groups should measure complete recipe time and replay
at fixed numerical likelihood groups before choosing a production layout.
The original four-observation likelihood groups and batch size 4096 remain
fixed; a larger scientific batch is not a demonstrated optimization.

Earlier MH workers benefited from CPU concurrency, as recorded in
[the October 13 resource audit](WORKLOG.md#2026-10-13-afternoon--resource-audit-of-the-last-bright-run-and-speed-benchmark-for-the-bright-second-step).
Those timing gains use a different sampler and sampling budget and cannot be
substituted for the current SMC rate. No extra benchmark was launched for this
estimate, and the running scientific experiment was not changed.


## Measured local CPU concurrency follow-up

The unchanged sampler now has a completed one-V100 development benchmark.
Four two-core workers achieve **3.47–3.57 times** the serial setting-inclusive
throughput, with exact saved-output replay for all 24 original runs. Two workers
achieve 1.90–1.96 times. The primary timing includes process/model startup,
sampling, saving and verification; the same eight physical cores are reserved
throughout. New preparation/report overhead sensitivity remains 3.45–3.54 times
for four workers. Fresh-seed confirmation remains required before adoption.
See [measured comparison](SMC_CPU_CONCURRENCY.md#completed-development-result).

Do not multiply the earlier full-catalogue rate by this factor: pilot fitting,
phase diagnostics, shared costs, different galaxy groups and four-GPU contention
have not been benchmarked under this concurrency layout. The result establishes
that overlapping independent host computations is effective for this measured
production workload; distributed 500-CPU capacity is still unmeasured.
