# SMC CPU concurrency confirmation — October 22, 2026

## Fixed decision and evidence

The completed development experiment selected four independent two-thread
workers sharing one V100. All 24 original populations replayed exactly.
Its complete-setting speedups over one worker were 3.570103 and 3.473819;
two workers reached 1.959108 and 1.903927. Both levels exceeded the
predeclared 1.25 threshold in both repetitions; four workers had the larger
worst repetition and the five-percent preference for fewer workers did not
change that choice.

The development report is pinned by SHA-256
`2d8f7c2f0d106f9da849114b97a9bc16eb0e3f9471988de3bc0ed22b89860c30`.
The independent parent audit, job 17447670, repeated the original full
source/protocol/64-stage/saved-array comparison and verified all seven outer
process-cost receipts. Its receipt is
`smc_concurrency_parent_audit_v1_20261022/result.json` under the experiment
root. Preparation verifies and pins that audit, the original declaration,
report, all six setting receipts, worker identities, outputs and costs.
It independently recomputes the original selection and repeats saved-output
replay before freezing this confirmation. Failed or changed evidence cannot
be excluded or replaced.

This is the predeclared fresh-seed confirmation of the development choice.
The original declaration already named seeds 2026103800–2026103803. This
confirmation does not select another worker count or change sampling.

## Confirmation recipe and gates

Run all four seeds at each concurrency setting in order **1,4,4,1**.
One allocation reserves one V100 on th-cl-nv02 and eight physical CPU cores
throughout. Every worker uses two distinct physical cores; concurrent
workers have disjoint affinities. The CPU reservation, selected cores and
physical GPU remain fixed across all four settings. No MPS change is made.

Each population uses observations 58165, 0, 1, 8; 256 particles; 64 stages;
64 moves; the original 16-block schedule; density batch 4096; shear (0,0);
two Torch threads;and the original frozen development geometry and proposal.
The exact 320-file source has manifest
`fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1`.
The separate prior-reuse candidate is not part of this experiment.

Require all 16 original populations to complete with no extinction and match
both the pinned original protocol and every saved scientific JSON value,
all 64 stage records, and every particle/scene array's dtype, shape and bytes.
Only the original explicitly named timing/job fields and checked source
paths may differ. No retry, replacement seed, omitted run or shorter history
is permitted. An incomplete or failed run prevents a successful report.

Both counterbalanced speedups must be at least 1.25 using complete internal
setting times. Both must also exceed 1.25 using the outer subprocess timers.
A complete report may record a failed efficiency gate; it cannot silently
select another worker count. Reporting keeps the two timing repetitions
separate. The repeated seeds within each timing setting are not additional
independent scientific replicates.

## Cost scope and limitations

Setting timers include source/evidence verification, GPU identity checks,
child interpreter/model startup, production, saving and exact replay checks.
Outer `/usr/bin/time` receipts also capture harness process startup and
final receipt serialization. Each receipt retains wall time, aggregate user
and system CPU time, maximum RSS and exit status. All four setting process
receipts are required before reporting.

Confirmation preparation includes the repeated development proof and frozen
source/harness copies. It excludes final declaration serialization,
top-level manifest writing and permission changes. Confirmation report
process time is measured separately; finalization adds that cost and splits
shared preparation/report costs equally between one and four workers.
Launcher setup, queueing, original development/search costs, existing pilot
fitting and final JSON-only finalization are excluded. These are conditional
production timing results, not full catalogue throughput.

Passing confirms fixed-recipe implementation efficiency on one GPU and
these four observations with fresh seeds. It does not establish posterior
accuracy, catalogue shear calibration, four-GPU scaling or 500-core scaling.
The production driver does not save every intermediate mutation or terminal
RNG state; the replay claim covers all saved science and histories.

## Preparation and execution

Preparation is CPU work and must run through the scheduler:

```bash
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python \
  scripts/prepare_smc_concurrency_confirmation.py \
  --root /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021 \
  --destination /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/smc_concurrency_confirmation_prepared_v1_20261022
```

The destination must not exist. Preparation produces read-only source,
harness, selection proof, declaration and manifests. The declaration uses
schema_version 1 with phase
`predeclared-fresh-seed-concurrency-confirmation-v1`, selected_workers 4,
seeds 2026103800–2026103803 and order [1,4,4,1]. Common recipe/runtime fields
come from the original declaration. All evidence paths and hashes are explicit.
Output is fixed to `smc_concurrency_confirmation_runs_v1_20261022` under
that experiment root and cannot already exist.

Inside a one-GPU/eight-core allocation on th-cl-nv02 requested with
`--hint=nomultithread`:

```bash
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
  bash /absolute/prepared/run_smc_concurrency_confirmation.sh /absolute/prepared
```

The launcher sequentially calls `run-setting --setting-index 0` through 3,
then `report`, then `finalize`. Each command takes `--declaration PATH`;
report also takes `--output PATH`, and finalization takes `--output REPORT
--cost REPORT_PROCESS_COST`. The parent controls submission and the study's
four-GPU limit. This change submits no jobs.

## Implementation validation

The original `benchmark_smc_concurrency.py` is unchanged. The confirmation
entrypoint validates its own fixed phase before projecting only phase fields
for common source/recipe validation. Expected scientific protocols use only
the explicitly predeclared fresh seed. Generic process management, CPU/GPU
identity and exact comparison helpers are reused; development declarations
still reject confirmation seeds and order.

Focused tests cover selection and tie rules, full development evidence,
seven outer costs, independent audit, unchanged development validation,
wrong phase/seeds/order, all 16 saved outputs, both timing gates, failed or
missing workers, changed allocations, shortened histories and byte corruption.
Twenty confirmation tests passed in 10.47 seconds; the combined original and
confirmation suite passed all 55 tests in 12.45 seconds. Full scheduled suite and
GPU confirmation remain pending; no confirmation speedup is claimed.
