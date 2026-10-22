# Corrected heldout runner

## Current launch

While the independent speed benchmark is pending/running, held-out array
17435350 has temporary Nice=20 to allow that one-worker benchmark to share
capacity. Its node pin, four-worker throttle, running allocations and science
remain unchanged. Restore job 17439409 resets Nice=0 after benchmark termination
unless the owner has changed the setting; see the
[scheduling record](evidence/benchmark_priority_window_20261022.json).

The first complete production group (indices 0–7, observations
560126, 270602, 745801, 623219) finishes all eight candidate/reference pairs:
16 populations and 16 phase diagnostics, with no extinction and all recorded
phase numerical checks passing. All eight Slurm tasks exit 0. The
[completion receipt](evidence/heldout_first_group_complete_20261022.json)
verifies 126 metadata/log identities, declared settings/seeds/runtime, fit
identities, counterbalanced step order and receipt-cost arithmetic. This light
review does not decode/re-hash NPZ archives or compare posterior estimates;
the frozen final reporter still must validate every group and every gate.

Corrected preflight17435258 completes exit0 in55 allocated seconds. Declaration
`ad718136d4ac00a37ffcaddf09dcfad92b15e68e4ebbf82ead597ed1b7c1f758`
pins320 source and1826 external files. It preserves exact scientific settings,
proposal laws and seeds from the first declaration; no science preceded repair.
Preparation accounting includes both preflights and failed startup:153 process,
306 CPU-allocation and23 GPU-allocation seconds (conservative allocation bounds).

Warmup17435348 (`0-4%4`) precedes paired production17435350 (`0-39%4`) and CPU
report17435351. The [current launch receipt](evidence/corrected_heldout_v3_launch_20261022.json)
pins all inputs and jobs. Original failed launch evidence remains below.
First group17435348_0 completes all11 steps and frozen-input checks in471
allocated seconds; four pilots and all four phase checks pass, geometry/proposal/
coefficient fits complete. The [startup receipt](evidence/corrected_heldout_first_group_20261022.json)
preserves evidence. All five groups now complete successfully (471/467/467/466/466
allocated seconds), and production17435350_0 has started with no extinction
in its completed stages. The [all-group startup receipt](evidence/corrected_heldout_all_warmups_20261022.json)
verifies all55 step receipts and frozen fit-artifact hashes. Independent
production comparisons and acceptance remain pending. The first paired
production task17435350_0 now completes both populations and both phase checks
without extinction in701.06 measured wrapper seconds. Its [receipt](evidence/corrected_heldout_first_pair_20261022.json)
records workflow validation only; no interim accuracy comparison is used.

## Startup harness repair before any science

The first submitted warmup tasks failed before stage reservation or pilot
creation: preflight incorrectly treated `/proc/cpuinfo` and `/proc/stat` as
immutable inputs. No particle populations, pilot fits or numerical outcomes
exist in that output. Remaining/dependent jobs were cancelled; the original
source, declaration, logs and allocation costs are preserved in the
[startup receipt](evidence/corrected_heldout_startup_failure_20261022.json).

The corrected preflight records virtual filesystem reads and the generated
Matplotlib font inventory separately. Actual scientific caches, including
HDF5 prior/template inputs and the PyCosmo pickle, remain pinned. Same scientific
source/settings/seeds are retained because failure preceded every scientific
computation. The next declaration uses a fresh output and charges both original
preflight and failed launch costs. Initial 982 passing code tests did not cover
this host-input classification failure. New full CPU17435254 and dependent
preflight17435258 use preserved320-file snapshot `source_corrected_heldout_v3_20261022`,
manifest `fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1`.
CPU17435254 passes993 tests in91.63 seconds, one optional skip and one existing
warning, with before/after manifest checks. See the
[repair validation receipt](evidence/corrected_heldout_host_input_repair_20261022.json).
An intermediate v2 snapshot was never executed; review added the font-cache
exclusion before freezing v3.

## Scope and immutable inputs

`scripts/run_corrected_particle_heldout.py` executes the fixed procedure in
[the preparation declaration](CORRECTED_HELDOUT_PREPARATION.md#conditional-fixed-procedure).
It submits no jobs. Every invocation requires Slurm, a frozen declaration and
source, complete external input hashes, and a hashed completed development
receipt with all three raw and corrected-phase comparisons passing. It never
reuses historical heldout pilots, replaces seeds, retries steps, or adapts fits
after production begins. This implementation was validated with synthetic CPU
fixtures only; no heldout scientific computation was performed by its author.

The declaration schema remains `sbsi-heldout-corrected-declaration-v1`. In
addition to the fitter's fields, the runner requires:

- `source_files`: the complete manifest as absolute path/SHA256 pairs.
- `external_files`: actual prior/model/config/reference input hashes, including
  reference `protocol.json`, `result.json`, and `observations.parquet`.
- `development_receipt`: `{path, sha256}` for the completed n23 comparison;
  its three comparison gates and recursively referenced source hashes must pass.
- `preparation_cost_receipt`: `{path, sha256}` with kind
  `scheduler_allocation_wall_upper_bound` and finite nonnegative
  `process_wall_seconds`, `cpu_allocated_seconds`, `gpu_allocated_seconds`.
- Full `pilot_proposal_metadata` and `reference_proposal_metadata`, generated
  from measured inputs before pilots, plus exact runtime versions, original
  V100 node/device and two inference torch threads.
- `blocks`, `power`, `scale` and fixed effort/seed/replica settings for all three
  recipes. Five ordered groups retain four observations and density batch4096.

The declaration must be `OUTPUT/declaration.json`. Both the runner and imported
SBSI package must resolve inside the declared frozen source. All source and
external inputs are verified before and after each successful stage.

## Scheduled stage interface

Use the original environment and the existing study GPU ceiling. Launch from
the frozen source through the same explicit bootstrap used by other workflows:

```bash
python -u -c 'import sys,runpy;sys.path[:0]=[".","scripts"];runpy.run_path(sys.argv.pop(1),run_name="__main__")' \
  scripts/run_corrected_particle_heldout.py warmup \
  --declaration /ABS/OUTPUT/declaration.json --index 0
```

Warmup indices are 0–4. Each runs four fresh Gaussian N256/T64/M32 pilots,
fits geometry and initial proposal using the retained fixed rules, checks all
four complete phase diagnostics, and runs the separate declaration-bound phase
fitter. `group_N/warmup_complete.json` binds all pilots, diagnostics and fits.
A completed corresponding stage receipt is also mandatory for production.

Production indices are 0–39; `group=index//8`, `replica=index%8`. Replace
`warmup` above with `production`. Even replicas run candidate then reference;
odd replicas reverse that order. Each arm's complete phase diagnostic follows
its population. Candidate is fitted N256/T64/M64; reference is broad Student-t5
N1024/T64/M64. Production uses frozen group geometry and phase coefficients.

After all forty production tasks finish, run `report` without `--index` in a
CPU-only Slurm allocation. It validates every run/phase receipt, generates ten
raw reports and ten phase aggregates, and requires all five groups, all eight
populations per arm, unchanged raw gates, and both corrected score components
within3.5 combined whole-population MCSE under both F32 and F64 measures.
Acceptance additionally requires strictly lower all-inclusive candidate cost.

## Failure and cost receipts

Every stage reserves `NAME_started.json` before subprocesses. Each step writes
an exclusive log and timing receipt. Completed or failed stage receipts retain
every child cost and wrapper verification overhead. An exception creates a
stage-specific panel rejection receipt; an interrupted reservation without a
completed stage is also ineligible. There is no retry or overwrite interface.
Phase process exit zero is insufficient: the numerical checks and exact source,
protocol, population count and coefficient identities must pass. Extinction or
unknown failure rejects the panel; no seed disappears from the experiment.

Shared pilots, their phase checks, geometry and coefficient fitting are charged
fully to each recipe; proposal fitting is candidate-only. Production and raw/
phase reporting are arm-specific. Preflight, verification and comparison costs
are split equally. Actual work counts shared steps once. Reports retain all
three cost fields. CPU/GPU allocation seconds mean allocated resource count
times full measured wrapper wall, not busy compute. They exclude queue time,
Slurm allocation time outside the wrapper, and final receipt serialization.
Preflight uses its separately preserved allocation upper bound. Process sums
are not parallel panel elapsed time; development/search cost is not amortized.

The statistical scope is conditional on the frozen independent fits. F32
corrected scores remain qualified diagnostics; the periodic identity applies
to the intended continuous F64 target. Passing is bounded fixed-y evidence,
not proof of global rare-mode coverage or population shear-calibration accuracy.

## Synthetic validation

CPU job17434665 tested the runner and declaration-bound fitter from an immutable
266-file snapshot. Every source hash was checked before and after. Fixtures cover
development gating, external/source/panel changes, exact pilot recipe/provenance,
phase-content failures, independent seeds, counterbalanced order, preserved
failed costs and no retry, all cost components, strict cost inequality and the
raw-plus-both-measure conjunction. The validation receipt records the exact
test count, source hashes and preserved output. Root performs integrated full
suite validation before freezing any actual scientific declaration.

## Integrated source validation

Root CPU17434704 passes982 tests, one optional skip and one existing warning,
in187.79 seconds (205-second allocation). Both manifest checks pass on the
320-file `source_corrected_heldout_20261022` snapshot, manifest SHA256
`b9ccc31f9b701526dace7f1a45da7208141941678060aa23f48af062d7f68b6c`.
The [integration receipt](evidence/corrected_heldout_integration_20261022.json)
pins the full-suite log and launchers. Measurement-only preflight17434715
depends on these checks; a completed allocation receipt and immutable declaration
are still required before scientific warmup.

## Actual launch

Measurement-only preflight17434715 completes in75 allocated seconds using two
CPUs and no GPU. Its internal work is56.35 seconds; the full allocation upper
bound is charged. The declaration pins320 source files and1827 external files,
including the preserved preflight. SHA256:
`6166d8d3c4288e5b1f35f2663779652b56fbf22d3956bfaecefd720a4aed9ef5`.
Warmup17435167 (`0-4%4`) precedes paired production17435171 (`0-39%4`), then
CPU report17435177. Both arrays use one V100 per task on th-cl-nv02, with two
CPUs and fixed batch4096; the afterok graph enforces at most four GPUs in total.
The [launch receipt](evidence/corrected_heldout_launch_20261022.json) preserves
input, source and launcher hashes. No scientific acceptance result exists yet.

## Report integration on audited development

While held-out jobs run, CPU17435839 exercises the actual frozen runner's
comparison and report-validation functions on the three completed v10 arms.
All three pairs reproduce the audited raw and both-measure phase comparisons
(1908 recursive checks, numerical replay tolerance1e-12 only). All three
complete report formats pass and incomplete population counts reject. All32
read artifact hashes are checked against the original audited v10 source map.
The job completes in27 seconds with before/after source checks; the
[receipt](evidence/heldout_report_development_audit_20261022.json) preserves
script/output/log hashes. Temporary diagnostic path rebinding is an explicit
format fixture, not a scientific declaration. No held-out input, new model
evaluation or changed threshold enters this audit. It validates reporting
integration, not additional convergence.

## Bounded long-run status check

CPU-only one-shot 17436101 completes at 14:12 UTC on 2026-10-22, exit 0 in
two allocated seconds. It records 14 completed stages, no failed jobs, and no
final result yet. No posterior moments are read. The
[scheduling and completion receipt](evidence/heldout_scheduled_status_20261022.json)
preserves the original submission, verifies unchanged executable/launcher bytes,
and pins the output hash. A later check of both workflows, 17439000, is scheduled
for 16:00 UTC. These checks write local artifacts and do not wake or message
this thread. Full science/report results remain required before acceptance.
