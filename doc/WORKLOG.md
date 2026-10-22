# Work log

## 2026-10-22 — Preserve the queued pool replay and schedule one bounded follow-up

PoolGPU17449412 remains pending Priority. The required th-cl-nv02 node is only
in inter and all four GPUs are allocated; the scheduler estimates a two-day wait,
which is not a promised start time. No scientific settings, resource binding or
validated source are changed to bypass that queue. A single status/result-presence
check17449573 is scheduled about two hours later; it sends no chat message.
[Queue evidence](evidence/sampler_task_pool_queue_20261022.json) and
[monitor declaration](evidence/efficiency_followup_v2_20261022.json) preserve this
operational state. No repeated long-run polling is needed.

Updated Arbor n25 with the completed independently audited7.06%runtime reduction,
n27 with full validation/queued real replay, and the versioned tree. Goal remains
active: no accepted posterior sampler, combined throughput, four-GPU scaling or
500-core execution claim. Next useful work is numerical-pilot/control preparation
and, after exact pool replay, a combined runtime benchmark; the old held-out
rejection and all failed validation artifacts remain intact.

## 2026-10-22 — Complete and independently audit prior-reuse speed evidence

GPU17448787 completes all four original v2 pairs with exact saved scientific JSON,
64-stage and array replay. Counterbalanced ratios1.0793/1.0725/1.0703/1.0885 all
favor reuse. Inclusive baseline871.965s versus optimized810.386s gives7.06% lower
runtime, conditional on existing fits/native validation. Failedv1 cost248.06s is
preserved separately. No undeclared performance threshold or automatic adoption.

CPU auditor17449393 passes1,052 tests/nine skips, then independently replays all
four pairs and recomputes costs, checking906 evidence pins and end-of-pair GPU
identity.475source hashes matchbefore/after; parent verifies source/JUnit and
report/cost receipts. Added [completed result](PARTICLE_PRIOR_REUSE_CANDIDATE.md#completed-paired-replay-and-independent-audit)
and [versioned evidence](evidence/particle_prior_reuse_completed_result_20261022.json).
The3.25fold concurrency gain is separate; combined throughput is unmeasured.
Scientific acceptance remains failed and the general pool's real replay is queued.

## 2026-10-22 — Validate the reusable CPU pool and launch mandatory real replay

Isolated pool commitac16d107 preserves original sampler/defaults and both frozen
benchmark helpers. Independent review drives fixes for exact physical affinity,
deferred cancellation across threads/child launch/reap, child signal masks,
artifact pins, duplicate source aliases and final report proof/path validation.
Focused tests and fresh full CPU17449328 pass:1,138 tests/nine skips,484 source
hashes before/after, original eight-control preparation. Parent checks812 manifest
entries and JUnit. The old passed v1 preparation remains intact.

GPU17449412 is queued on oneV100/eight physical cores after the CPU gate. Prior
benchmark17448787 completed before submission, preserving its timing isolation.
All eight old development/confirmation tasks must replay exactly; no new seed,
scientific setting, likelihood grouping or runtime optimization is introduced.
Added [pool contract and launch](SAMPLER_TASK_POOL.md#candidate-validation-and-real-replay-launch)
and [preparation evidence](evidence/sampler_task_pool_v2_preparation_20261022.json).
Pool receipts currently rewrite the full ledger; use bounded manifests. GPU
replay/adoption remains pending and scientific convergence remains unaccepted.

Independent prior-reuse auditor6fdcf01 is frozen separately; CPU17449393 starts
only after completed benchmark17448787. It checks the original four pairs and
recomputes descriptive timing/cost without calling the old report/finalizer.
[Audit scope](PRIOR_REUSE_COMPLETED_AUDIT.md#authority-and-completion-barrier)
keeps correctness, speed evidence and scientific acceptance distinct.

## 2026-10-22 — Close inherited prior-configuration identity and pass fresh validation

Commit1d6171a on the isolated numerical-target driver branch adds the specific
UFig common configuration and package initializer pins. This binds seed_ngal,
which determines the maximum-redshift interpolation support; no factory values,
random draws or sampler calculations change. Fresh immutable CPU17449171 passes
1,129 tests/nine skips and installed-native validation. Parent rechecks481 source
entries,173 direct inputs, JUnit and all saved native arrays, which match v2
byte-for-byte. Maximum log-density roundtrip error remains2.2773e-12.

Updated [pipeline candidate](NUMERICAL_TARGET_PIPELINE.md#candidate-location-and-current-validation)
and [versioned evidence](evidence/numerical_target_validation_v3_20261022.json).
Old validation artifacts remain intact. Next scientific work requires a fresh
numerical-target pilot/fit/control declaration; no old analytic fit is relabelled,
and no convergence or shear-calibration claim is made.

## 2026-10-22 — Validate the continuous numerical-target implementation and retain remaining scope

The isolated numerical-target branch now implements the full type sum, coordinate
Jacobian, native exchange correction and explicit driver/proposal/report/control
identity. Analytic defaults remain unchanged. Full CPU17448810 originally failed
two phase-report tests on a missing local root; the failure is preserved. After
repair and added cross-law/input guards, CPU17449051 passes 1,125 tests/nine skips
and the installed-law check (2,182 retained rows, max roundtrip log-density error
2.2773e-12). Parent verifies all481 source entries,170 direct input entries, JUnit
and saved draw hash. Added candidate documentation and versioned validation.

Review then identifies a direct UFig common-configuration input controlling
maximum redshift; the completed test's broader external declaration already pins
it, but the portable target identity needs a narrow follow-up. No numerical
production, posterior convergence or global shear result is claimed. Native
validation excludes singular clamp atoms and does not prove posterior exploration.

Separately, prior-reuse GPU17448787 passes the formerly failing optimized startup
and advances through37 recorded stages; all four original pairs remain required.
[Startup evidence](evidence/particle_prior_reuse_v2_startup_20261022.json) records
that operational check without an interim timing or scientific decision.

## 2026-10-22 — Confirm CPU sharing with fresh seeds and independent replay

GPU17447826 completes all four original settings and 16 populations. Four workers
on one V100/eight physical cores preserve every saved scientific value/history/
array and achieve 3.337/3.269 inclusive speedups; shared preparation/report costs
give 3.250 overall. CPU17448939 independently reruns strict saved-output checks
and recomputes both timing gates and complete-cost arithmetic; all pass. Parent
checks six report/cost hashes and both successful scheduler records. Added the
[confirmation result](SMC_CPU_CONCURRENCY_CONFIRMATION.md#completed-confirmation-and-independent-audit),
versioned receipts and audit script; updated search and compute-cost scope.

This validates the fixed local concurrency layout, not posterior convergence or
four-GPU/500-core scaling. Reusable pool validation remains necessary before its
execution path is adopted. The failed held-out sampler decision is unchanged.

## 2026-10-22 — Repair prior-reuse provenance and validate every production startup

The separate prior-reuse branch preserves its three-file optimization and strict
artifact checks. An explicit derived proposal changes implementation provenance
only; the fit is identical. CPU17448718 passes 1,139 tests and nine skips, all
485 source hashes before/after, eight real production startups and a mismatched
artifact negative control. Parent checks 1,135 manifest entries and 189 external
pins. GPU17448787 is queued for all four original paired replays in a fresh v2
root, with one V100/two physical cores. Failed v1 artifacts and cost remain intact.

Updated [candidate evidence](PARTICLE_PRIOR_REUSE_CANDIDATE.md#startup-failure-and-validated-v2-repair)
and its versioned preparation receipt. Runtime speed and GPU equality remain
unmeasured. Numerical-target development and the reusable worker pool continue
in isolated branches; neither is adopted as a scientifically accepted sampler.

## 2026-10-22 — Reject the completed held-out candidate after independent gate and cost audit

All original held-out work finishes: report17435351 records15/20 objects
passing, five failing, accuracy=false, cost=true and accepted_sampler=false.
Inclusive process costs11,782.2405 versus22,896.6929 seconds give ratio1.9433;
this is not a matched-accuracy speedup. The five failures concern reference
normalizer precision, candidate raw-score precision, two scene-region contrasts
and one raw-score contrast. Both controlled-score measures and all absolute
bounds pass without replacing the failed mandatory gates.

Independent checker27a5323 on codex/smc-completed-panel-audit-20261022 uses the
frozen source's provenance validators but recomputes gates and cost allocation.
CPU17448173 passes1,075 tests, nine skips, one existing warning in85.98s;
then its fresh-process audit validates1,225 source/output pins and all46stages,
235steps,20pilots,80production populations,100phase checks,15fits and20reports.
It reproduces the rejection and each failed requirement. All475archive hashes
match before/after and in the parent check; JUnit and scheduler completion pass.
Original reports were neither rewritten nor rerun.

Added [final results](CORRECTED_HELDOUT_RESULTS.md#direct-failures),
[versioned evidence](evidence/corrected_heldout_final_results_20261022.json), and
[audit contract](COMPLETED_HELDOUT_AUDIT.md#independent-calculations-and-output).
Updated current search/compute status and propagated the transfer failure in
Arbor. This goal turn made progress by completing a scientific decision and
independent audit; no sampler is accepted and the goal remains active.
Further development must use a new untouched acceptance panel. Target remains
p_A q_E K32; no numerical-mock-prior or population-calibration claim.

## 2026-10-22 — Catch prior-reuse startup rejection while CPU concurrency confirmation runs

GPU17446115 fails1:0 after249allocated seconds: the baseline completes, but
the optimized arm's unchanged production validation rejects the frozen initial
proposal's native_target_source_sha256 before optimized sampling. Preserve
all original v1 artifacts and [failure evidence](evidence/particle_prior_reuse_gpu_startup_failure_20261022.json).
No numerical mismatch or speedup is inferred from a startup failure. An
isolated v2 repair is underway: explicitly bind a derived proposal with identical
numerical fit and migrated implementation provenance, retain strict public
validation, and exercise actual production startup on CPU before GPU launch.
The original seeds/order remain fixed; v1 is not overwritten or counted as a
successful benchmark.

Fresh-seed concurrency confirmation17447826 starts on its reserved V100/eight
physical cores; the first serial population completes and the next progresses.
Source/settings remain frozen. The bounded metadata check17447926 remains
scheduled for20:55Berlin; it sends no chat notification. Exact concurrency
confirmation, prior-reuse GPU replay/timing and wider CPU scaling remain open.

## 2026-10-22 — Freeze and queue the predeclared four-worker confirmation

Isolated commit a556047 implements the unchanged development selection's fresh
seed check: 2026103800–2026103803, worker order1/4/4/1, same four observations,
unchanged 320-file sampler and fixed eight physical cores/one V100. All16
original populations must replay; both internal-inclusive and outer-process
ratios must reach1.25 in both repetitions. The original development harness is
unchanged. All55 focused tests pass; full scheduled job17447788 then passes
**1,086 tests, nine skips**, one existing warning, with all483source hashes
unchanged and full frozen preparation preflight.

Parent independently verifies810manifest entries,657external evidence pins,
JUnit and scheduler completion. GPU17447826 is queued on the four-GPU study
node after this success; no unnecessary dependency on other studies. Added
[confirmation protocol/status](SMC_CPU_CONCURRENCY_CONFIRMATION.md#scheduled-validation-and-launch)
and [preparation receipt](evidence/smc_cpu_concurrency_confirmation_preparation_20261022.json).
No fresh-seed performance result or production promotion yet. One-shot metadata
check17447926 replaces17446128 at the same20:55Berlin time and includes the
new confirmation; it does not send chat notifications.

## 2026-10-22 — Audit scientific acceptance separately from numerical mock-prior closure

A read-only audit of the frozen held-out declaration, complete source, runner,
gate functions and tests confirms all-object raw precision/agreement,
both-measure controlled agreement, complete original-run accounting and strict
inclusive process-cost improvement. No partial held-out outcomes were read or
used to tune settings. Recorded [source-bound checklist](evidence/heldout_acceptance_contract_audit_20261022.json)
and [interpretation boundary](CORRECTED_HELDOUT_RUNNER.md#acceptance-audit-and-target-law-boundary).

A pass concerns p_A(target) q_E(environment) K32. The F64 phase reweight changes
likelihood precision only; it does not apply q_N/p_A. Existing numerical-prior
sensitivity on represented development states therefore remains separate from
exact numerical mock-prior inference and absent-mode coverage. Held-out effort
increases N256 to N1024 while T64/M64 remain fixed. Controlled precision and
other supplemental diagnostics are not reclassified as gates. No target, source,
threshold, seed or live scientific job was changed.

## 2026-10-22 — Four CPU workers per GPU deliver exact development replay and 3.47–3.57x throughput

Concurrency job 17445452 completes all 24 original populations and every
64-stage/saved-array comparison. Independent scheduled parent audit 17447670
reruns the frozen source/protocol/worker/result checks and exactly reproduces
the report except for the audit's elapsed time. All seven outer process-cost
receipts pass. Four workers give 3.570/3.474 setting-inclusive speedups; two give
1.959/1.904. Adding shared preparation/report costs leaves four-worker ratios
3.544/3.448. The unchanged predeclared ranking selects four workers for the
fresh-seed confirmation; no runtime promotion or scientific gate is waived.

Added the [audit script](evidence/audit_smc_concurrency_20261022.py),
[complete timing evidence](evidence/smc_cpu_concurrency_development_20261022.json),
and [result/limits](SMC_CPU_CONCURRENCY.md#completed-development-result).
Updated computation-cost notes without extrapolating to complete catalogue or
500-CPU throughput. Previous goal turn made progress through implementation,
full CPU/native validation and queued GPU replay; this turn adds measured
performance evidence and an independent complete replay audit.

Resource controller17445497 returned1 for an already-finished array task;
current scheduler inspection verifies all live held-out records were restored
to throttle4. Preserve the original failed controller receipt and successful
current-state verification. No extra mutation/restart was required. Prior-reuse
GPU17446115 remains pending on scheduler priority; held-out science remains
untouched while the original 40 paired tasks finish.

## 2026-10-22 — Complete GPU benchmark preflight and queue prior-reuse comparison

Isolated source 4385cb8 passes combined CPU job **17446023**: **1,123 passed,
nine skips**, one existing warning, 84.70 test seconds. All 483 source hashes
match before/after; prepared baseline/candidate each contain all 320 pinned
sampler files and differ in exactly three declared implementations. Parent
independently rechecked 1,129 manifest entries, JUnit and scheduler success.

Queued full paired GPU replay/timing **17446115** after successful tests and
current concurrency benchmark 17445452. It uses one V100/two physical cores,
original fixed scientific settings and four fresh alternating-order pairs.
No speedup, GPU replay or scientific acceptance result yet. Added
[preparation evidence](evidence/particle_prior_reuse_gpu_preparation_20261022.json)
and updated [comparison status](PARTICLE_PRIOR_REUSE_CANDIDATE.md#paired-full-gpu-replay-and-timing).
The one-shot read-only status check 17446128 replaces 17445739 and includes
both efficiency experiments; it does not notify chat. Documented the
[remote CPU/GPU service boundary](SMC_CPU_CONCURRENCY.md#path-to-many-cpu-workers)
as future work, separate from the actual local concurrency implementation.

## 2026-10-22 — Prior reuse passes native replay; concurrent CPU workers start

The isolated prior-reuse implementation passes full scheduled validation
17445720: **1,092 tests, nine skips**, one existing warning. All 479 source
hashes match before/after. Actual installed photometry and native-prior outputs
match byte-for-byte on 73,728 saved development states and 15 support probes;
parent checks confirm all 83 native source/input identities, JUnit and successful
scheduler completion. These checks do not establish full sampler replay or
speedup. Recorded [validation](evidence/particle_prior_reuse_validation_20261022.json)
and [native output](evidence/particle_prior_reuse_native_20261022.json).

Prior-reuse branch 4385cb8 now contains complete two-source GPU replay/timing
preparation and launcher: four fresh seed pairs, alternating execution order,
exactly three known source differences and complete saved-result comparisons.
Its 31 focused harness tests pass; full combined CPU/preparation job 17446023
is running. See [remaining comparison](PARTICLE_PRIOR_REUSE_CANDIDATE.md#paired-full-gpu-replay-and-timing).

Concurrency job 17445452 completed all four serial baseline populations and
started two simultaneous two-core workers on one V100. Both progress normally;
full counterbalanced speed/replay results remain pending. Actual eight physical
cores are billed as 16 logical Slurm CPUs. Updated [startup evidence](SMC_CPU_CONCURRENCY.md#first-concurrent-stage)
and Arbor n24/n25 without claiming an accepted improvement. Live held-out
scientific sources remain unchanged.

## 2026-10-22 — Implement and schedule CPU concurrency; isolate redundant-prior work

Following the owner's request for substantially greater CPU use, implemented
[one/two/four workers per V100](SMC_CPU_CONCURRENCY.md#hypothesis-and-fixed-experiment)
in isolated branch `codex/smc-cpu-concurrency-20261022`, harness 0ee415b and
complete experiment 13a1fca. Six fixed settings run all four fresh seeds in
forward/reverse order. Workers have disjoint physical CPU cores; all saved
scientific outputs and 64-stage histories must match the unchanged recipe.
Failures stop and reap children; no dropped populations or retries.

CPU17445433 passes 1,064 tests/11 skips in 101.94 seconds, 138 allocated seconds.
All 35 focused tests passed earlier; two physical-affinity tests skip on this
two-CPU full-suite allocation. All 478 source hashes match before/after, and
frozen preparation verifies the complete 320-file sampler and input pins.
GPU17445452 is queued with one V100/eight physical cores. Temporarily lowered
held-out array17435350's throttle from four to one; its running tasks and
scientific settings remain untouched. Pre-submitted completion restore17445497
and 20-minute startup guard17445498 prevent an indefinite resource restriction.
Both experiments stay on the same physical four-GPU node.

Separately, candidate49918c6 on `codex/smc-prior-reuse-20261022` avoids unchanged
target-prior recalculation for neighbour moves and reuses freshly decoded
photometry within type calculations. It preserves the public native-prior
semantics and RNG; focused tests pass. Full CPU, real photometry and GPU replay
remain pending. The [concurrency preparation evidence](evidence/smc_cpu_concurrency_preparation_20261022.json)
and Arbor n24/n25 record these separate interventions. No throughput result,
scientific acceptance or production adoption is claimed. Documentation JSON,
shell syntax, source hashes and whitespace checks pass.

## 2026-10-22 — Estimate sampler cost and scaling from completed timing receipts

Added [computation-cost evidence](SAMPLER_COMPUTE_COST.md#measured-recipe-and-evidence)
and `doc/evidence/sampler_compute_cost_20261022.json` for the first two complete
held-out four-observation groups. Verified all 18 stage cost identities and
their referenced step-timing hashes. Only operational receipts were used; no
interim held-out posterior comparison or acceptance decision was made.

At the fixed eight-population budget, inference processes cost 1,647.24 seconds
per four observations on average. Fresh pilots/fitting and phase diagnostics
raise this to 2,147.30 seconds; charging all paired verification raises it to
2,356.85 seconds. Four fully available V100s therefore project to roughly
0.007 galaxies/s, about 25/hour. This is an allocation-time projection excluding
the reference arm, final panel reporting, global preflight/search, queueing and
other shear centres. Observation scaling is approximately linear at fixed
settings; the 500-CPU saturation rate has not been measured. Current workers
use two CPUs each, and profiled host costs motivate a later concurrency study.

Validation: receipt arithmetic/status/hash assertions passed, JSON parse and
`git diff --check` passed. No runtime code, scientific settings, live sources,
or jobs changed; no additional computational experiment or test suite needed
for this documentation-only analysis. Held-out scientific acceptance and the
full final cost comparison remain pending.

## 2026-10-22 — Version and validate the complete frozen sampler candidate

An independent source-to-commit audit finds 311 of the 320 held-out source paths
identical at root b7f559a, two older scripts different, and seven older modules,
diagnostic scripts and tests absent from Git. The active particle sampler and
held-out runner already match. Preserved the exact nine frozen files in isolated
branch `codex/frozen-sampler-candidate-20261022`, implementation commit 862db49
and final documentation commit 0301993. Every actual frozen file and all 320
candidate Git blobs pass SHA256 checks. Root's original dirty/untracked files
and all live scientific sources/settings remain untouched; no runtime
optimization or scientific change is introduced.

CPU job 17443057 tests a complete 469-file Git archive plus source receipt with
two CPUs and no GPU: **1,031 passed, nine optional skips**, one existing NumPy
warning, 83.45 test seconds and 108 allocated seconds. All 470 snapshot hashes
match before and after, confirmed again by parent review together with JUnit
and scheduler receipts. There are no failures or retries. Copied only the
[candidate documentation](FROZEN_SAMPLER_CANDIDATE.md#validation), inventories,
launcher and receipts into the root worktree; the complete candidate code stays
on its isolated branch. Updated the search evidence and candidate pointer.
External model/data/package dependencies remain pinned separately. The held-out
accuracy and inclusive-cost gates are still pending; these implementation and
provenance checks do not establish convergence or shear calibration.

## 2026-10-22 — Complete exact categorical replays; full-run speed benefit remains uncertain

All four fixed benchmark pairs in GPU array 17437046 and CPU report 17437047
complete with exit 0. Every saved particle/scene array and all 64 scientific
stage records match exactly in each pair; all eight runs complete without
extinction. The final reporter repeats the frozen source/input/recipe and output
comparisons. Parent review independently checks all scheduler exits, receipt
identities, stage counts, original orders/seeds, and scalar cost arithmetic.
The [complete report](evidence/exchange_full_smc_result_20261022.json) and
[parent review](evidence/exchange_full_smc_review_20261022.json) preserve evidence.

Measured totals are 852.338 seconds baseline versus 841.762 seconds optimized:
1.24% lower optimized time, conditional on existing fits. Pair ratios are
1.04738/0.97943/1.06285/0.96466, mean 1.01358 with descriptive SE 0.02438.
Both baseline-first pairs favor optimized; both reversed pairs favor baseline.
This does not identify the physical cause of the order pattern or establish a
dependable full-run benefit. Leave the tested optimization inactive, preserving
the isolated candidate and all results; no further timing experiment is planned
now. The passed replay checks remain valid. Historical pilot/fitting/test/search
costs are outside this benchmark, and sampler acceptance still requires the
separate held-out scientific and inclusive-cost gates.

Restore job 17439409 completes successfully at 14:44 UTC; its receipt and an
independent live scheduler check confirm held-out Nice=0, throttle four and the
unchanged node pin. Updated the priority receipt, experiment/current-candidate
notes, held-out runner and Arbor evidence. The held-out array resumes with task
12. Runtime code and scientific settings remain unchanged. Documentation JSON,
cost identities, tree validation and whitespace checks pass; no additional
implementation test is needed for these evidence-only changes.

## 2026-10-22 — Review the tested categorical candidate and first two full replays

The isolated exact optimization, implementation commit 413fe75 and reviewed
candidate commit 8b3d854, passes the complete CPU suite in job 17440535:
1,084 passed, nine optional skips, one existing NumPy warning, 93.48 test seconds
and 137 allocated seconds on two CPUs. All 66 categorical replay cases pass;
the complete 454-file snapshot matches before and after. Copied its candidate
note, inventory, launcher and validation receipt into `doc/`; runtime code stays
on the isolated branch. Parent review independently verifies external receipt
hashes and JUnit counts, and checks probability validation, CDF arithmetic and
the unchanged ordering of random draws against NumPy 2.2.6's source.

Full benchmark pairs 0 and 1 complete exact saved-output replay in opposite arm
orders. Both arms finish all 64 stages without extinction; each comparison covers
eight particle arrays and 24 scene arrays. The [parent review receipt](evidence/categorical_parent_review_20261022.json)
pins the harness receipts and scientific metadata. Parent inspection does not
repeat NPZ decoding; the frozen pair harness performs the byte comparison and
the final CPU report will repeat it. No partial timing result is used. Updated
the experiment and search status; all four pairs, the complete cost report and
held-out validation remain required. No live source or scheduler changes.

## 2026-10-22 — Let the independent benchmark share queue capacity and restore priority automatically

After dependency removal, scheduler priority still favored every older held-out
task (544 versus benchmark 536 at inspection), so both available slots continued
to go to that array. Changed only held-out array 17435350's Nice from 0 to 20;
the verified priorities become 525 versus benchmark 537. Existing allocations
continue, throttle four is retained, and the one-worker benchmark may take a
released slot while held-out work uses remaining capacity. Both arrays remain
pinned to the same four-GPU node. No code, seeds, data, runtime, within-pair order,
acceptance gate, declaration or source manifest changed.

Before changing Nice, submitted CPU restore job 17439409 afterany benchmark
17437046. Its hash-pinned script restores Nice=0 only if the same live array
retains Nice=20 and the original node pin, preserving later owner changes.
Local Slurm documentation, AST/Bash syntax, live one-line scheduler parsing, and
eight pure-stdlib subprocess-fixture checks pass, covering normal restoration,
owner/node changes, absent/terminal jobs, failed update and unconfirmed update.
The [priority receipt](evidence/benchmark_priority_window_20261022.json) preserves
before/after states, unchanged scientific hashes, restoration job and test scope.
No scientific outcomes were read. Updated both workflow documents; verify
benchmark startup and eventual priority restoration. Scientific acceptance stays
pending the complete frozen reports.

Benchmark task 17437046_0 starts at 14:13:39 UTC on the declared V100 node
while held-out work continues on another GPU. Its baseline enters annealing
and records progress without extinction; the priority receipt preserves the
startup observation. No complete pair replay or speed result is claimed yet.
The baseline then completes, and the optimized arm also enters annealing
without extinction. Its separate startup observation is appended to the same
receipt. All four complete pairs and the frozen final report remain required.

The original scheduled check 17436101 completes at 14:12 UTC, exit 0 in two
allocated seconds: 14 stages complete, no failed jobs, no final result yet.
Verified its original script/launcher hashes and recorded the output hash in
the [existing receipt](evidence/heldout_scheduled_status_20261022.json), retaining
the original submission state. The later both-workflow check 17439000 remains
scheduled for 16:00 UTC. No posterior moments or comparisons were read.

## 2026-10-22 — First complete held-out production group finishes

All eight Slurm pairs 17435350_0–7 complete with exit 0, covering the first
four held-out observations. A light [completion review](evidence/heldout_first_group_complete_20261022.json)
verifies 126 metadata/log identities, all 32 subprocess receipts, declared
seed/runtime/settings and frozen fit identities, counterbalanced order, and
cost arithmetic. All 16 population runs finish 64 stages without extinction;
all 16 phase diagnostics report passed numerical checks. Summed paired task
allocation is 5,779 seconds; no comparative cost or posterior inference is made.

The first metadata-reader attempt expected an explicit null initial-proposal
field in broad-reference protocols. Using the frozen validator's optional-field
lookup completes the review; production source/data are unchanged. The receipt
records that correction. This review does not decode or re-hash the NPZ archives
and is not the final all-group scientific gate. Updated current workflow and
search status; the untouched remaining groups and final report stay active.

## 2026-10-22 — Make the independent speed benchmark eligible within the four-GPU ceiling

Removed only GPU benchmark array 17437046's dependency on held-out completion.
All four benchmark tasks were pending before the change. Current scheduler state
shows both study arrays pinned to th-cl-nv02, one GPU per task, and no study GPU
jobs elsewhere; the node has exactly four GPUs. This physical capacity preserves
the combined ceiling with held-out throttle four and benchmark throttle one.
The independent benchmark can now use scheduler backfill without waiting for
all forty held-out pairs; availability and priority still govern its start.

The original declaration's allocation dependency is superseded by a separate
[scheduling receipt](evidence/exchange_full_smc_scheduling_amendment_20261022.json),
which preserves before/after scheduler output. Both experiment declarations and
source manifests remain byte-identical; seeds, data, code, runtime, settings,
within-pair order and CPU report dependency are unchanged. No held-out or benchmark
outcome was read. Local Slurm documentation confirms empty `Dependency=` removes
the dependency. Scheduler verification confirms it is cleared and the node pin,
one-worker throttle and dependent report remain intact. Updated the experiment
and search overviews. Recheck combined capacity before any future GPU submission
elsewhere; no speed or acceptance result follows from this scheduling change.

Benchmark eligibility now shows `PENDING/Priority`. Added bounded CPU status
check 17439000 for 18:00 Europe/Berlin (16:00 UTC), covering both GPU workflows
and their reports; the earlier held-out check 17436101 stays unchanged. The
[monitor receipt](evidence/sampler_workflows_scheduled_status_20261022.json)
pins its source and launcher. It reads scheduler state and report-file existence,
handles observation timeout separately from scientific-job failure, and writes
a receipt; it does not wake or message this thread. AST/Bash syntax checks and
the scheduler's pending BeginTime were verified.

## 2026-10-22 — Render the completed development evidence as a reproducible figure

Added `doc/evidence/plot_sampler_development_20261022.py`, a versioned SVG,
a local PNG preview (covered by the existing PNG ignore rule), and a
machine-readable plotted-data receipt; embedded the figure in
[current sampler status](SAMPLER_SEARCH.md#current-status). The four panels
show raw-score precision, maximum pairwise score contrasts, summed inference
allocation, and numerical-prior sensitivity. Every point is computed from the
pinned corrected-v10 comparison and completed prior audit. No interim held-out
outcome is read. The figure preserves the distinction between score expectations
and shear estimates, and between inference allocation and total operational cost.

CPU render 17438391 completed in 12 allocated seconds. Visual inspection found
overlapping legends/annotations; layout-only revision 17438461 completed in
9 seconds. All plotted numbers match exactly across revisions; the initial
snapshot and outputs remain preserved. Final PNG inspected with clear labels;
input/source/output SHA256 checks, AST syntax and static F821/F823 checks pass.
Versioned SVG trailing whitespace was normalized; parsed XML structure, text
and normalized attribute values match the preserved renderer output.
Both CPU jobs total 42 allocated CPU seconds and zero GPU seconds. This adds a
view of existing evidence; acceptance still requires the frozen independent
held-out accuracy and complete-cost report.

## 2026-10-22 — Audit corrected-v10 target-prior sensitivity on all development populations

Added `sbsi/prior_sensitivity.py`, a read-only audit executable and focused
math/provenance tests. The diagnostic sums full numerical native type
intensities, preserves original particle/population denominators, and uses
paired whole-population influences with both ratio denominators. It replays
the authoritative raw and frozen phase summaries before reporting F32 and
F64 numerical-minus-analytic score shifts; no sampler, likelihood or gate
changes are made. See [design and results](CORRECTED_PRIOR_SENSITIVITY.md#result-and-interpretation)
and [receipt](evidence/corrected_prior_sensitivity_20261022.json).

CPU job 17437783 passes the full suite: 1018 passed, 9 optional skips in
73.95 s; static F821/F823 checks pass. CPU audit 17437804 passes all 24
corrected development populations / 73,728 states in 85 allocated seconds.
Baseline replay differs by at most 3.55e−15 and saved prior replay is exact.
Largest raw-or-controlled |score change| / original MCSE is 0.00596526
(F32), 0.00596452 (F64). No numerical-only or analytic-only type support and
no zero-numerical state occur in the saved particles. Numerical-only coordinate
support and missing modes remain unbounded; this is not a global calibration
or convergence result. All original data and frozen coefficients are preserved.

Root integrated the exact tested implementation from isolated commit 6abfe6cb,
verified hashes and AST syntax, and updated the search overview and n14 evidence.
The hypothesis tree validates; no candidate is marked accepted.

The receipt retains two snapshot-packaging test failures and one pre-evaluation
import-scoping failure, followed by corrected complete checks. Total scheduled
cost including failures is 464 allocation-wall seconds / 928 CPU-allocated
seconds / zero GPU seconds. No heldout outcomes were read. The next scientific
conclusion must come from independent sampler/heldout evidence; this result
alone does not authorize a numerical-prior target or calibration correction.

## 2026-10-22 — Prepare corrected-development prior-sensitivity follow-up

The earlier [bridge audit](PARTICLE_BRIDGE_AUDIT.md#new-evidence-on-the-numerical-target-prior-difference)
used old v5 populations with large raw-score uncertainty. Registered a bounded
CPU follow-up under Arbor n14 using all24 corrected v10 development populations,
their original normalizers, full phase diagnostics and frozen coefficients.
The numerical-to-analytic ratio must sum complete native type intensities;
a saved auxiliary type or support-restricted per-type ratio is insufficient.
Each measure's paired population influence functions provide the contrast error,
including covariance and retaining zero-mass populations. Reproduce audited
baseline moments and native-coordinate priors before interpreting any tilt.

The isolated executor works from7eda30d; tests and a frozen design precede the
CPU-only run. No learned likelihood evaluation, new sampler, held-out input or
production-target modification is authorized by this diagnostic. Both priors
are phase invariant, so the continuous F64 control identity persists on the
represented analytic-support domain; the F32 qualification and missing-support/
mode limitations remain. This is a represented-state sensitivity check, not an
empirical calibration correction. No new numerical evidence yet. Updated
hypothesis-tree status only; independent held-out validation remains live.

## 2026-10-22 — Queue isolated full-run categorical replay and cost check

Integrated the independently tested benchmark/preparation/fixture scripts,
launcher,35 focused tests and [experiment design](EXCHANGE_FULL_SMC_REPLAY.md#fixed-design-and-decision-boundary)
from isolated commit254245be. CPU17436954 completes exit0 in20 seconds:
35 tests pass in0.22s, immutable preparation passes, and copied audited
v10 development outputs pass the actual verifier. Initial13 tests had lacked
fixed-runtime/recipe and complete-history checks; v2 now verifies the full
protocol template, all64 stages, every scientific JSON field/audit counter and
all saved NPZ array bytes. Only explicit timing/provenance exceptions apply.
The fixture's source-root rebinding and test-only seed mapping are disclosed;
it is no inference result. Both prepared snapshots remain preserved.

Root verifies both304-file source copies, all168 external inputs and test/
fixture hashes before submitting GPU17437046 (`0–3%1`) afterany both held-out
GPU arrays, then CPU17437047 afterany that array. The [launch receipt](evidence/exchange_full_smc_launch_20261022.json)
pins exact commands. Same four development observations, four paired seeds,
fitted N256/T64/M64 recipe, original geometry/q0 and density4096; both arms
use corrected v10 source, differing only in previously validated categorical
draw batching. Existing held-out source/settings stay unchanged. Every failure
and pair cost is retained; new-work totals include full run subprocesses and
measured copy/hash/report overhead, conditional on existing fits. Historical
pilot/fitting/search costs are excluded explicitly. Four pairs give descriptive
cost evidence, not convergence or promotion. Update n22 evidence without changing
its earlier proposal-only score or declaring a new best.

The first held-out paired task17435350_0 also completes both runs and both phase
checks without extinction; wrapper701.06 seconds. Preserved its [startup receipt](evidence/corrected_heldout_first_pair_20261022.json)
without any interim accuracy contrast. All remaining independent repetitions and
final accuracy/cost gates are still required. New efficiency jobs wait behind
this work; monitor their startup when prerequisites finish. Existing scheduled
status17436101 remains pending. No accepted sampler yet.

## 2026-10-22 — All held-out preparation groups complete; production starts

All five warmup tasks17435348_0–4 complete exit0 in471/467/467/466/466
allocated seconds. Independently checked the declaration identity, all55
successful step-receipt hashes and every frozen fit-artifact hash referenced
by the five complete-group receipts. Production17435350_0 starts; its first
population progresses without extinction. The [startup receipt](evidence/corrected_heldout_all_warmups_20261022.json)
records timestamped scheduler/progress evidence without any interim accuracy
comparison. Updated [runner](CORRECTED_HELDOUT_RUNNER.md#current-launch) and
[search overview](SAMPLER_SEARCH.md#current-status). No scientific source,
proposal, seed, effort or gate changed. Final scientific acceptance and complete
cost comparison remain pending; existing bounded status check17436101 remains
scheduled. Validation: metadata, hash and scheduler checks only.

## 2026-10-22 — Frozen final-report adapter replays audited development

CPU17435839 completes exit0 in27 seconds, exercising the actual frozen held-out
runner on all three previously completed v10 arms. All three pairwise comparisons
match the audited receipt through1908 recursive checks; all complete report
formats pass and incomplete denominators reject. Input hashes match the old
v10 source map; source manifest checks pass before/after. The
[audit receipt](evidence/heldout_report_development_audit_20261022.json) and
preserved executable pin the output/log. Replay tolerance1e-12 applies only to
report arithmetic, not scientific acceptance thresholds. No held-out output or
new model evaluation is used. Running source, settings and gates remain fixed.

Warmup group1 also completes exit0 in467 seconds; group2 is live. To avoid
continuous long-run polling, CPU-only one-shot status17436101 is scheduled for
16:12:03 Europe/Berlin (about two hours later). It records scheduler/stage status
and flags failures without reading moments. This is a local status receipt, not
a thread-wakeup service; see [scheduled check](evidence/heldout_scheduled_status_20261022.json).
The autonomous objective remains active pending independent acceptance.

## 2026-10-22 — First complete held-out preparation group passes

Warmup17435348_0 completes exit0 in471 allocated seconds (466.55 measured wrapper
seconds): four pilots, gradient geometry, initial proposal, all four phase
checks and declared coefficient fit. Before/after input checks pass. The
[startup receipt](evidence/corrected_heldout_first_group_20261022.json) pins stage,
warmup and log hashes. Group1 starts next; remaining groups queue on the matched
node. Production17435350 and report17435351 retain afterok dependencies.
No interim accuracy contrast is used for tuning. No sampler accepted; allow the
longer graph to run and check dependent startup/results on continuation. There
is no available thread-wakeup automation tool; the autonomous goal stays active.

## 2026-10-22 — Update sampler-search overview to completed development and live validation

Replaced stale queued-v10 status in [search overview](SAMPLER_SEARCH.md#current-status)
with the audited development pass, current held-out graph, inclusive cost
requirement and preserved pre-science startup repair. First pilot completes;
no interim held-out comparison is used for tuning. Explained the distinct
roles of corrected support, geometry, normalized initial proposal and mutation
effort, and why physical joint blocks coexist with other invariant moves.
Retained links to negative experiments, unknown historical defect impact,
F32/continuous-measure limits and the unresolved population shear question.
Documentation-only update; no running source, scientific setting or gate changes.

## 2026-10-22 — Relaunch after input-classification repair without changing science

Corrected preflight17435258 passes in55 allocated seconds. Root verifies unchanged
scientific settings, seeds, measured proposal laws and all ordinary external input
hashes before allocating GPUs. Frozen declarationad718136 binds320 source and
1826 external files; prior preflight/failed-startup costs are included, totaling
153 process/306 CPU/23 GPU allocation seconds. No earlier scientific run exists.

Submitted warmup17435348, afterok paired production17435350, then CPU report
17435351; arrays `%4`, same V100 node and batch4096. The
[current launch receipt](evidence/corrected_heldout_v3_launch_20261022.json) retains
source, input and launcher hashes. Startup/scientific results remain pending;
no rejected seed was replaced and no sampler is accepted.

## 2026-10-22 — Repair mutable host-input pinning before first held-out pilot

Warmup17435167 tasks0–2 fail before stage reservation on changed `/proc/cpuinfo`;
remaining/dependent tasks are cancelled. No group directory, stage reservation,
pilot run, fit or posterior output exists. Preserve original declaration/source
and [failure receipt](evidence/corrected_heldout_startup_failure_20261022.json).
Prior preflight75 seconds and failed/cancelled allocation upper bound23 GPU
seconds/46 CPU seconds will be charged to the repaired launch too.

Preflight now separates virtual `/proc`, `/sys`, `/dev` reads and generated
Matplotlib font inventories from immutable scientific inputs. Independent input
review retains scientific HDF5/PyCosmo caches and Astropy configuration. Added
boundary/symlink/cache regression checks. Scientific source/settings/seeds stay
fixed; this repairs orchestration before any outcome, not a failed scientific
gate. Full CPU17435254 passes993 tests, one optional skip and one existing warning
in91.63 seconds, with before/after manifest checks. The
[repair receipt](evidence/corrected_heldout_host_input_repair_20261022.json) pins
the source/log. Afterok preflight17435258 starts on the immutable v3 snapshot. Intermediate v2 was never submitted; original982 passing tests did
not cover this real runtime classification. No sampler accepted.

## 2026-10-22 — Freeze fresh corrected held-out validation and submit dependent stages

Measurement-only preflight17434715 completed exit0 in75 allocated seconds on
the comparison node (two CPUs, no GPU); its full cost is preserved. Frozen
declaration6166d8d3 binds320 source and1827 external inputs, exact measured
proposal laws, untouched twenty-object panel and new pilot/candidate/reference
seed bases2026103200/2026103300/2026103400. Old pilot outcomes are not reused.

Submitted warmup17435167, paired production17435171 afterok warmup, and CPU
report17435177 afterok production. GPU arrays are throttled to four, original
V100 node/four-observation groups/batch4096 unchanged. The
[launch receipt](evidence/corrected_heldout_launch_20261022.json) preserves settings,
cost receipt, source/launcher/input hashes. Monitor startup and dependent stages;
no scientific result or accepted sampler yet. No thresholds/settings changed
after development or in response to held-out outcomes.

## 2026-10-22 — Complete corrected held-out workflow passes integration tests

Integrated isolated runner6b45838 with the tested measurement-only preflight,
declaration-bound phase fitter and completed v10 comparison. CPU17434704
passes982 tests with one optional skip and one existing NumPy warning in187.79
seconds. Both320-file manifest checks pass. The exact source and launcher hashes
are retained in the [integration receipt](evidence/corrected_heldout_integration_20261022.json).
Preflight17434715 is queued afterok on the original comparison node with two
CPUs and no GPU. No held-out pilot or production has launched at this entry.
The four-GPU ceiling, original four-observation groups and fixed settings remain.

## 2026-10-22 — Corrected heldout execution wrapper prepared and CPU-tested

- Added `scripts/run_corrected_particle_heldout.py` with immutable warmup,
  counterbalanced paired production and full-panel reporting stages. The wrapper
  requires passing hashed development evidence, complete source/external inputs,
  exact recipes/runtime/seeds and complete phase diagnostics. It uses the fixed
  comparison helpers unchanged from73bff31. Historical launchers and samplers
  are unchanged. See [runner contract](CORRECTED_HELDOUT_RUNNER.md#scheduled-stage-interface).
- Extinction, diagnostic failure or unknown failure rejects the panel and
  preserves the original seed, logs and costs. No retry/replacement/refit path
  exists. Acceptance is all raw gates AND both-measure corrected-score gates
  AND strictly lower all-inclusive cost. Shared preparation is charged to each
  recipe while actual work is separately counted once; allocation wall is
  distinguished from compute busy time and panel elapsed time.
- CPU17434665:58 tests passed in4.55s, immutable266-file snapshot; before/after
  source checks passed. Tests cover runner plus declaration-bound fitter.
  [Validation receipt](evidence/corrected_heldout_runner_validation_20261022.json)
  preserves exact hashes/logs. No heldout/scientific execution or GPU job was
  submitted. Root integrates and runs the full suite before source freeze and
  a separately validated declaration/preflight can authorize scientific stages.
## 2026-10-22 — Measurement-only preflight validated before fresh held-out launch

Added `scripts/prepare_corrected_particle_heldout.py` and focused tests. It requires
the complete passing development receipt before opening the panel, preserves
scientific source bytes, constructs exact proposal metadata from measurements,
and pins model/prior/package/reference inputs. Explicit HDF paths, actual imports,
read/write audit classification and JSON-safe inherited infinities are covered.
The reference preparation result is also pinned because the driver reads it.

Initial CPU17423593 passes935 tests in91.31 seconds; strengthened17434394 passes
945 tests in110.42 seconds, both with one optional skip and one existing warning.
Before/after source checks pass; [receipts](evidence/heldout_preflight_validation_20261022.json)
retain both snapshots and review limitations. The later integrated runner will
be tested before launch. No fresh held-out pilot or production has run.
Updated [Arbor evidence](evidence/sampler_search_tree_20261022.json): corrected n23
passes development gates; n22 passes bounded GPU replay. Neither is an accepted
full-cost held-out sampler. Runtime optimization stays isolated.

## 2026-10-22 — Corrected three-way development gates pass after audited reporting repair

- Authorized CPU **17434348** completed exit 0 in 31 seconds: **37 focused tests passed in 4.77 seconds**, followed by full source/protocol/artifact/seed/particle replay and comparison over all 24 completed v10 populations. All 12 precision and all three pairwise raw-score/region/Z and controlled F32/F64 gates pass. No sampling rerun, new model forward, held-out outcome access or threshold change.
- Largest controlled contrast is small minus large, object 0 g2, +1.158843 ± 0.600652 (+1.929307 combined MCSE); F64 +1.929200 MCSE. All signed means/errors and costs retained in [completed evidence](V10_PARTICLE_COMPARISON.md#completed-audited-development-comparison). Small/large/broad inference totals 1566.072/3843.086/3717.636 seconds; summed Slurm task times 1667/3944/3820. Shared phase 177 seconds and report 11 seconds; preparation/full cost unknown.
- Repaired immutable source 5285f5b, manifest 4767d2a3eef142ab4f5a387cccc7267d4eaeac59bf7247a2fa2ac6eedf2d0f99. Full result SHA256 09601af0a43c0b5801bc0467b090e588c745713a3c9fd6d86b58f2f9172d1574; [pinned receipt](evidence/v10_comparison_result_20261022.json) includes failed 17423550 and successful log/source hashes. Prior failed snapshot/log remain unchanged.
- Development comparison passes; **sampler remains unaccepted** pending independent assessment. F32 control expectation is unknown; F64 reweighting explicitly changes measure and qualifies the continuous identity. Eight populations cannot exclude shared rare modes. No population shear-calibration conclusion follows from fixed-object score agreement.

## 2026-10-22 — Repair n23 source whitelist before authorized completed evaluation

- Root's evaluation 17423550 stopped in five seconds before reading outcomes: the held-out data guard also rejected a legitimate filename inside the pinned support-fix source manifest. Original source_v10_comparison_20261022_v2, launcher, output log and failure remain preserved.
- Guard now permits only resolved paths inside the declared immutable source whose expected hash matches an entry in the already verified pinned manifest; actual file hashes remain mandatory. Actual held-out data, unlisted files, missing/wrong hash proof and symlink escapes remain forbidden. Added focused regressions; no scientific threshold, sampling setting or aggregate change.
- Root confirmed the entire v10 science graph complete and authorized CPU evaluation after focused tests. No sampling rerun or held-out outcome access. Fresh frozen reporting snapshot/test/evaluation pending.

## 2026-10-22 — N23 comparison helper fully tested and frozen; evaluation withheld

- **Validation:** final immutable source 73bff31 full CPU 17423438: **882 passed, 1 skipped, 1 existing warning in 85.43 seconds**, exit 0, 95-second allocation. Both legacy frozen-source checks enabled. Original focused CPU 17423323 passed 28 tests; six additional guard regressions reject empty/incomplete corrected-score inputs. Two earlier full-suite collection failures (17423354/17423380) are preserved; adding the present frozen scripts directory to the test runner import path fixed collection without source changes.
- **Delivery:** new gate module, guarded comparison script, tests, [prepared command](V10_PARTICLE_COMPARISON.md#prepared-command-and-completed-validation) and pinned preparation evidence. Snapshot source_v10_comparison_20261022_v2 manifest 090b9272823f2a88a72bf327bc61f150a2df52b4fee117ffaddd82f41ccbb5a0; unsubmitted evaluation launcher hash 1f0e9953818cd5ce2d870562f419a1c12f8f9b17983c3922dcf80256c2fb77ce. Exact source/file hashes and validation receipt in doc/evidence/v10_comparison_preparation_20261022.json.
- **Boundary:** no numerical v10 outcome or interim protocol was read; no scientific gate result or evaluation submission. All 24 sampling tasks and both raw/phase jobs must complete before the evaluator opens outcomes. Development gate pass never sets accepted_sampler=true. Preparation/full end-to-end costs remain explicitly unknown; explicit extinction cannot be silently dropped to complete the unsupported phase assessment. Root controls eventual evaluation and held-out preparation.

## 2026-10-22 — Harden n23 gate APIs against incomplete moments

- Root review identified vacuous success for empty difference vectors and raw-only vectors sliced to empty corrected columns. The caller already required the declared five-column schema, but the reusable gate API now also rejects empty vectors and requires exactly five finite phase moments with nonnegative MCSE before corrected-column slicing.
- Added six regressions for empty, missing/extra corrected columns and nonfinite unselected moments. Existing 28 focused tests passed on frozen dafff0e (CPU 17423323); its full-suite job 17423354 is preserved. Amended-source full CPU validation pending; no v10 outcome read, threshold change or evaluation submission.

## 2026-10-22 — Prepare n23 corrected three-way gates without outcome access

- **Scope:** new isolated `codex/v10-comparison-20261022` from 3edb2ee. Read only the v10 declaration/launcher and prior code/artifact schemas; no v10 numerical run/phase outcomes. New gate module and summary executable implement all original raw precision/score/region/Z gates plus all three pairwise corrected components <=3.5 combined MCSE separately in F32/F64. Always leaves accepted_sampler=false.
- **Integrity/cost:** evaluator refuses outcome reads until all 24 array tasks and both report jobs complete successfully. It pins source/artifacts/seeds/protocol, replays complete particle/phase arrays and recomputes whole-population aggregates. Eight-ULP weight replay tolerance only; no altered weights. Explicit extinction blocks unsupported phase aggregation, never drops a run. Costs separate driver, per-arm Slurm elapsed and shared diagnostic wall time; unknown preparation/full cost stays null.
- **Validation:** syntax/diff checks pass; focused synthetic CPU tests pending. [Declared scope/gates](V10_PARTICLE_COMPARISON.md#gates-all-retained). No GPU job, fitting, new model forward, held-out input, sampler change or evaluation submission.

## 2026-10-22 — N22 trained-model replay passes; runtime integration remains deferred

Parent-submitted GPU 17422901 completed in45 s on one V100 at th-cl-nv02,
after development array17422526. The immutable n22 snapshot reproduces saved
kernel/prior values exactly and preserves full state/decision/RNG bytes over
18 mixed-block transitions and all paired timing calls. Ordinary complete
exchange-block median speedups are **1.169x at beta0.001** and **1.132x at
beta1**. Relocation is **0.993x/1.055x**; the slightly slower low-beta case is
preserved without tuning. See [results](EXCHANGE_CATEGORICAL_RESULTS.md#trained-model-comparison)
and [full receipt](evidence/exchange_categorical_trained_20261022.json), including
all source/input hashes, timing samples and scheduler record. This update is
documentation/evidence only; receipt hashes match the frozen265-file manifest.

The primary CPU proposal score remains1.36095x. Seven timing pairs on one saved
population establish a bounded full-block saving, not full-SMC wall time or
convergence. Runtime integration stays deferred to keep current corrected
development and held-out scientific source byte-identical. No new job, source
snapshot edit, or sampler change was made in this follow-up.
## 2026-10-22 — Prepare held-out phase fitting with strict pilot provenance

Added `scripts/fit_declared_particle_phase_control.py`, its focused tests and
[conditional validation design](CORRECTED_HELDOUT_PREPARATION.md#status-and-launch-condition).
The historical n15 fitter remains unchanged. New fitting is bound to a frozen
panel, complete source manifest, exact four fresh pilots, globally disjoint
seeds, declared proposal laws and preproduction timing. Diagnostic hashes must
bind the declared pilot artifacts; mandatory implementation maps and the actual
imported phase module must match the source. Pilot and production outcomes are
not used to choose the fitting rule.

Initial CPU17423221 passes878 tests with1 optional skip in102.93 seconds.
Independent review then finds provenance/source-map/full-proposal validation
gaps; the strengthened code and seven additional rejection checks are frozen
for CPU17423307, which passes885 tests with1 optional skip in90.37 seconds.
Both before/after manifest checks pass; both snapshots are preserved in the
[validation receipt](evidence/declared_phase_fit_validation_20261022.json).
No inference source, running v10 job, old held-out receipt, or scientific gate
was changed. The new full held-out launcher/report remain future work, conditional
on complete v10 evidence. No held-out science was launched or inspected.

## 2026-10-22 — Final reporting and legacy-audit integration validated

CPU17422839 passes854 tests with8 optional skips in97.55 seconds; seven skipped
checks require explicitly located frozen legacy sources. Follow-up17422922
sets that source and passes all26 legacy-harness checks in13.00 seconds,
including those seven. One existing NumPy warning remains. Both jobs verify
the311-file manifest before and after; the
[integration receipt](evidence/reporting_audit_integration_20261022.json)
pins the source and logs. No running inference source was modified.

Corrected array17422526 continues with its first three arms having passed
startup and meaningful stage progress. Raw report17422589, phase17422590 and
optimization replay17422901 retain their afterok dependency on the full array.
The longer comparison is healthy; no interim aggregate is used for tuning.
No candidate is accepted and held-out production remains paused. No suitable
thread-wakeup automation is available in the connected tools; the active goal
and documented job graph retain the outstanding checks.

## 2026-10-22 — Batch categorical draws passes CPU replay; trained-model check queued

Recorded isolated1c0e0c6 documentation/evidence; its20-line runtime change and
benchmark/test harness remain on `codex/exchange-categorical-20261022`.
Full CPU17422591:872 tests pass,4 optional skips; stronger focused17422766:
67 checks pass. Paired CPU17422592:20 cases/520 calls preserve proposed payload
and RNG bits. Ordinary proposal speedups are1.341/1.381 at1024/4096 rows
(geometric mean1.360951); relocation gains1.080/1.071. These are proposal CPU
costs, not full-sampler efficiency or posterior convergence evidence.

Queued trained-model replay17422901 after corrected array17422526, using one
pinned V100 and frozen manifest. It checks full18-step mixed trajectories,
decisions, RNG and paired timings under both corrected controls. The future
phase job and this benchmark use at most2 GPUs after inference completes.
[Results and pending gate](EXCHANGE_CATEGORICAL_RESULTS.md#root-scheduling-receipt)
record commands/limitations. Core integration waits for that result.

## 2026-10-22 — Informed environments show mixed gains at higher cost

Integrated the v8 summary helper/tests and evidence from060cd95, keeping the
root's corrected v9 helper. CPU17422725 passes7 tests and104 input hashes in
7 seconds. Original17422693 failed a bitwise comparison of independently
computed weights differing by1.39e-17; the reporting-only repair permits8
float64 ULP with tests rejecting larger differences. Original failed snapshot
and receipt remain preserved; scientific inputs and thresholds are unchanged.

[Full results](V8_PARTICLE_COMPARISON.md) show1.713 times baseline cost,
worse normalizer precision for every object, and mixed score precision.
Bright phase-controlled MCSE improves4.215/6.670 to1.315/2.081; coarse faint-
target mass remains0–0.000973 with region contrasts<=1.46 MCSE. Observation0
baseline corrected-g1 is0.400543+/-0.249085, descriptively consistent with v6
and v9. Observation1 baseline F64 phase residual is-3.393 MCSE. The old support-
guard difference prevents attributing any effect solely to informed mass.
No held-out outcomes or v10 results entered this summary; no general adoption.

## 2026-10-22 — Prepare complete historical exchange-support replay

Integrated the isolated trajectory-audit harness from fbee440 and its tests.
It instruments original corrections while returning them unchanged, pairs
old/new native marks, and separates proposed from **old accepted** support
violations. Counts become trusted only after matching every persisted field
and complete stage history, including informed-environment payloads. Original
intermediate full states/final RNG were not saved; the receipt states that
limit. [Guide](LEGACY_PARTICLE_TRAJECTORY_AUDIT.md) contains exact commands.

CPU17422670 passes26 tests in13.37 seconds. The initial fixture-dispatch failure
17422503 and its repairs are preserved. No real historical trajectory has
been replayed and no GPU is submitted for this harness. Fresh corrected v10
comparisons take priority; this preparation makes later causal checks possible
without rewriting saved runs or using final particles as proof of history.
Integrated full-suite validation follows the remaining reporting merge.

## 2026-10-22 — Higher effort narrows the score discrepancy; corrected comparison starts

Integrated v9 report helper, six tests and full signed evidence from bdfdb89.
CPU17422294: six tests pass,145 input hashes verified,11 seconds total.
V9 phase17421115: all eight populations pass numerical checks in73 seconds.
Observation0 corrected-g1 v9-v6 is0.163937+/-0.181294 (0.904 MCSE), versus
v7-v6 at4.544 MCSE. V9 costs5.056 times v7, with nonuniform precision gains;
old shared support errors prevent exact-target acceptance. See
[all results](V9_PARTICLE_COMPARISON.md#completed-result-narrower-discrepancy-mixed-precision-and-cost).

Predeclared corrected v10 under the tested304-file snapshot:256 particles,
64 mutations versus1024-particle fitted and broad-proposal references, eight
fresh populations each. Smoke17422524 passes sampler and phase checks;
array17422526 (three GPUs max), report17422589 and phase17422590 retain fixed
four-observation groups and4096 density batches. The declaration predates v8
aggregate inspection. [Job receipt](evidence/v10_jobs_20261022.json) records
the rejected stale completed-job dependency and successful resubmission.
No held-out production is resumed and no candidate is accepted.

Added [cost attribution](MUTATION_COST_PROFILE.md#measured-cumulative-time)
and pinned profiler evidence. Arbor cycle7 now tests batched categorical
exchange draws in an isolated branch; root production code is not changed by
that experiment. All old snapshots and failed hypotheses remain preserved.

## 2026-10-22 — Integrated support correction and controls pass validation

Frozen 304-file `source_support_fix_20261022` combines the unconditional
exchange bound, phase-report interface guards, and standalone analytic phase
control. CPU job17421864 passes **822 tests, 1 optional skip**, with one
existing NumPy warning, in104.98 seconds. Before/after SHA256 manifests agree.
The [integration receipt](evidence/support_fix_integration_20261022.json) pins
all overlaid source paths, manifest and output. Historical inference snapshots
remain unchanged. This validates implementation, not sampler convergence.

Mutation profiler17421460 completed successfully against immutable source_v5b;
all original saved-state/K/prior replay checks passed. Profiles separate five
blocks at1024/4096 rows, with density batches fixed4096. Exchange categorical
loops and numerical neighbour generation incur substantial CPU cost; profiled
times include warmup/caches and instrumentation and are not speedup evidence.
Next: assess completed v9/v8 comparisons with the support caveat, instrument
legacy trajectories, and test probability-preserving overhead reductions.

## 2026-10-22 — Frozen endpoint receipt and unconditional native exchange support fix

- Added [the explicit endpoint audit](PARTICLE_ENDPOINT_COMPATIBILITY_AUDIT.md#frozen-identity-evidence)
  and reproducible CPU scripts/receipts under `doc/evidence/`. Verified all
  286 v7 and 298 v8 frozen manifest files, 21 shared endpoint factor sources,
  actual model/reference artifacts, and available startup protocol settings
  (8 v7, 8 v9, 4 default-v8, 4 informed-v8). No v8/v9 numerical outcomes,
  heldout outcomes, GPU work or frozen snapshot edits were used.
- Found a [production-prior support error](PARTICLE_ENDPOINT_COMPATIBILITY_AUDIT.md#why-the-old-native-exchange-violates-continuous-support):
  the old `scene_exchange.numerical_over_analytic` omitted the numerical
  generator's lower uniform-clamp bound. CPU 17421563 constructed red and blue
  marks passing the actual i14–28 cut with finite old exchange corrections and
  zero continuous environment density. The optional informed guard already
  rejected these; default v7/v9 and v8 baseline did not. Common declared
  endpoints therefore do not certify equal target-preserving kernels.
- Fixed the helper unconditionally, preserving supported-path arithmetic and
  RNG. Added optional aligned support-audit masks for later isolated trajectory
  instrumentation; no production bypass. Updated the informed guard docstring
  and regression to reflect the common check; added 16 focused support tests.
- CPU 17421564: 118 passed, 1 optional skip. Final CPU 17421627: 795 passed,
  3 optional skips in 60.21 s, one existing NumPy deprecation warning;
  before/after input hashes passed. Independent frozen-v7 comparison reproduced
  ordinary full states/decisions/RNG over 24 block transitions, and 268 generated
  production-mark ratios bitwise. Both actual-prior counterexamples now reject
  with unchanged proposed payload/RNG. Two earlier audit-harness import-order
  failures are preserved and explained in the receipt.
- Limits/next steps: no old trajectory was replayed, so occurrence and effect
  on score discrepancies are unknown; small prior tail mass is not an impact
  bound. Clamp atoms/finite-RNG laws remain explicitly qualified. Historical
  external prior/table provenance is incomplete; recommend a shared fail-closed
  endpoint identity rather than changing old protocol/report gates. Root owns
  future corrected-source comparisons and any trajectory replay.

## 2026-10-22 — Reject the matched-cost waste-free SMC recipe

Preserved documentation/evidence from isolated e826f50; its implementation
remains on `codex/waste-free-smc-20261022`. CPU 17421477 passed 793 tests
with 4 skips, including exact normalizer/weighted-moment and extinction checks.
CPU experiment 17421484 ran 128 populations per target/arm at exactly 13,560
density rows each. Waste-free/standard variance was 1.089 on the Gaussian
and 2.755 on the mixture; CPU time increased 15–16%. The recipe is rejected
without further tuning. [Results](WASTE_FREE_SMC_RESULTS.md#matched-cost-evidence)
retain all observables, whole-population bootstrap intervals and limitations.
No real sampler, GPU experiment, or held-out evaluation used this prototype.

For efficient scheduled use, v8 array throttle increased from 1 to 2 within
the existing four-GPU total. Frozen source, seeds, effort and density batches
remain unchanged. Slurm warned about already-finished indices but confirmed
ArrayTaskThrottle=2 for pending work. The timestamped resource addendum
records that concurrency can affect runtime; no aggregate outcome informed
this operational change.

## 2026-10-22 — Fixed analytic phase control has mixed benefits

Added the standalone analytic control, saved-state diagnostic and tests from
77ae6b0; production sampling and score reporting are unchanged. The fixed
regularizer is 0.1, declared before outcomes, with no fitted coefficients.
The periodic product rule includes the essential derivative of its coefficient.
CPU 17421399: 12 tests passed; evaluation 17421436: all 28 populations in
10 seconds, with every particle and original denominator retained.

Estimated MCSE improves in 18/24 evaluation components versus raw and 13/24
versus the constant control. Observation 0 g1 improves, but v7-v6 still differs
by 3.534 combined MCSE. Faint g2 worsens consistently, so blanket adoption
is rejected. [Full signed means and MCSEs](ANALYTIC_PHASE_CONTROL.md#saved-state-result-mixed-variance-benefit-discrepancy-remains)
preserve both numerical measures and negative outcomes. Integrated validation
with the pending support correction remains to run. No accepted sampler yet.

## 2026-10-22 — Independent phase audit confirms the gap and hardens diagnostics

Integrated isolated audit 0883de4: phase diagnostics now reject deterministic
thinning and verify full retained counts. Raw F32 rows no longer claim an exact
zero expectation; F64 rows state the intended continuous-model qualification.
New reports use schema v2. Frozen old reports and queued v8/v9 snapshots remain
unchanged, with an explicit metadata erratum.

CPU 17421220 independently recomputed all 24 old evaluation populations:
maximum mean/covariance differences 1.34e-14/1.14e-13. Observation 0's gap
remains 4.54426 reported MCSE, or 4.54458 after explicit F64 reweighting.
All old inputs used every particle, so the thinning issue did not affect them.
The same job passed 40 focused tests; full suite 17421249 passed 778 tests
with 4 optional skips in 68.07 seconds. No sign, boundary, or weighting error
was found to explain the gap. [Audit](PHASE_CONTROL_INDEPENDENT_AUDIT.md#conclusion-for-the-existing-development-comparison)
records the remaining generic endpoint-provenance limitation.

Started two bounded CPU directions: analytic phase product-rule control with
fixed regularizer 0.1 (n19), and waste-free fixed-schedule SMC (n20). Both
retain frozen hypotheses and require scientific evidence beyond unit tests.
The advisory tree depth/sibling limits extend from 3 to 4 for these concrete
refinements; the 16-cycle budget, four-GPU ceiling and acceptance gates remain.
No production default, likelihood, existing run, or shear estimate changes.

Added a read-only profiler around frozen mutation calls, preserving source,
input and retained-state checks. Syntax and launcher validation pass; V100
job 17421460 depends on phase job 17421115. It attributes host/native/model
cost with fixed 4096-row density batches; profiler timings cannot establish
a speedup. No new posterior population or broader observation batch is used.

## 2026-10-22 — Localize the score discrepancy and validate scene-proposal startup

Added the read-only localization script, nine focused tests, and
[results](PHASE_SCORE_LOCALIZATION.md#completed-cpu-evidence) from isolated
commit 4dd75d5. CPU 17420920: 9 passed; analysis 17420950: exit 0 in 5 seconds.
No single run explains the corrected-score discrepancy; fixed top-5% tails
account for 54%, and 46% remains in the complement. No particles are dropped,
coefficients refit, or held-out outcomes inspected.

Informed-environment smoke 17420872 completed in 30 seconds; both paired
development arms reached startup. Held-out pilot preparation completed,
while its production remains cancelled. Repaired the queued phase diagnostic
launcher to validate the frozen JSON manifest, preserving the original
launcher and cancelled pending job 17420835. New v9 diagnostic 17421115 and
v8 diagnostic 17421116 retain frozen coefficients and measures. Validation:
`bash -n` passes both launchers; source hashes are checked before model imports.
[Search record](SAMPLER_SEARCH.md#score-localization-and-diagnostic-launch-repair)
records the v8 addendum timing and limitations. No candidate is accepted.

## 2026-10-22 — Phase diagnostic reveals disagreement; pause held-out production

Integrated the standalone phase-control diagnostics and preserved both failed
original checks and the separately predeclared finer-step v2 protocol. CPU
checks32/32; V10017420604 completes28/28 saved populations with exact saved
kernel/score replay. Fixed coefficients expose observation0 corrected-g1
v7-v6 disagreement of4.55 combined MCSE. The original raw gate pass remains
recorded, but scientific acceptance is withheld. F64 reweighting is far too
small to explain this difference. [Evidence](SAMPLER_SEARCH.md#phase-control-exposes-a-hidden-score-disagreement).

Cancelled pending held-out production17420233/report17420234 before any task
started; retain fixed warmup17420232 and do not tune on its outcomes. Declared
higher-effort fitted-q0 v9 before launch:17420804 -> raw17420827 and frozen
phase diagnostic17421115 (replaces pending cancelled17420835 after a launcher manifest-format repair). This is still development on the original four
objects, with unchanged proposal/geometry and fresh seeds.

Integrated optional informed-environmentf84ba7a and strict endpoint reporterd98942a.
Author174 tests and independent final density/MH audit pass; reporter isolated
suite495 passed,3 optional skips. Added explicit canonical endpoint metadata
to new protocols. Integrated source_v8 validation/smoke/science/report chain:
17420828 ->17420872 ->17420873 ->17420874. Integrated full suite17420828
passes782 tests,1 skipped in81.91seconds. No existing immutable trial changes.
[Fresh comparison](SAMPLER_SEARCH.md#informed-environment-integration-and-fresh-comparison)
records seeds, target measure, corrections and continuous-density limitations.

Documented the remaining finite-ratio bias boundary and an isolated coupled
whole-population MH prototype3b32f33. Its41 exact/finite-law tests and full
718-test suite pass (3 optional skips); it has no real-SMC adapter or scientific
efficiency result. [Contract](COUPLED_POPULATION_IMH.md#interface-and-target-identity).
No production default, V3.9 artifact or catalogue shear estimate changes.

## 2026-10-22 — First development-pass sampler; freeze held-out evaluation

Eight fitted-proposal v7 populations pass all four precision and independent
higher-effort score/region/normalizer comparisons (report17419615). Bright
normalizer relative MCSE falls from50.34% to2.78%, run-weight ESS7.957/8.
Mean production cost101.44seconds versus477.61 for the reference; this excludes
pilot costs and is a threshold-matched comparison, not equal variance.
[Full evidence](SAMPLER_SEARCH.md#fitted-proposal-development-result) preserves
cost accounting, scores, uncertainty and the remaining finite-mode limitation.

Added `doc/evidence/run_particle_heldout_20261022.py` and
`tests/test_particle_heldout.py`: frozen20-object evaluation, independent
warmup, original four-observation groups, strict all-object gates, immutable
seeds, and complete subprocess/pilot/fit timing. CPU17420215:25 passed in6.73s.
Declaration predates inference; warmup17420232 -> paired production17420233 ->
report17420234, with two V100 inference workers. No candidate promotion yet.
[Protocol](SAMPLER_SEARCH.md#frozen-held-out-validation) records exact settings
and rejection policy. No model identity, production default or shear estimate
changes. Read-only throughput17419751 replays K/summaries exactly; its repeated
states limit generalization of the measured1.23–2.16-times throughput gains.

## 2026-10-21 — Integrated proposal suite passes; fresh GPU trial starts

Source_v7 full suite17419607:660 passed,1 skipped,69.37seconds; smoke17419608
passes and array17419609 reaches meaningful late-stage progress. Frozen fitted
proposal trials retain their declared seeds/settings. No accuracy claim yet.
Integrated unused numerical native intensity helperefd794c with29 new/34 total
CPU tests passing17419657; it does not alter any running target. Continuous
support/tail limitations and prospective informed-environment work are in the
[search log](SAMPLER_SEARCH.md#numerical-native-measure-and-throughput-preparation).

Prepared immutable complete-mutation throughput benchmarka97c6d1, submitted
as17419751 after v6 resource release. It must first reproduce saved/duplicated
likelihoods with fixed4096 batches; repeated-state cache effects limit its cost
interpretation. Read-only periodic-phase zero-mean/variance-reduction work is
also being prepared independently, with no estimator or acceptance-gate change.

## 2026-10-21 — Freeze pilot-informed joint proposals and audit the target measure

Integratede513b995 proposal phase law, frozen pilot fitter/loader, optional
driver flag and tests. Corrected a pre-sampling scale-floor specification
ambiguity; preserved superseded artifact,70 tests pass17419552, and fit17419561
uses only four original v1 pilots. [Launch evidence](SAMPLER_SEARCH.md#fitted-proposal-implementation-and-launch)
records immutable artifact/hash, source_v7, full-suite/smoke/array/report stages.
Added runtime hardware/library metadata. Target and production defaults remain
unchanged; scientific acceptance is still pending.

Integrated read-only bridge audit05ed93b (CPU17419483): no new conditioning or
Jacobian error found; represented-state numerical/analytic target-prior mismatch
has <0.009-MCSE bright score effect. Nuisance-input invariance and a possible
3D marginal-prior identity pass bounded checks. Added the exact first-positive-
temperature zero-weight alternative to the informed-Poisson support discussion.
No collapsed sampler, informed-environment sampler or empirical correction is
introduced. See [audit](PARTICLE_BRIDGE_AUDIT.md#finding).

## 2026-10-21 — Locate the visible normalizer disagreement in late annealing

Preserved read-only stage diagnostic script/JSON fromd79f095. CPU17419480
completes13 frozen input audits; earlier17419416 schema failure is retained.
The [stage-resolved diagnosis](SAMPLER_SEARCH.md#stage-resolved-normalizer-diagnosis)
finds bright v5 logZ SD grows from0.138 at beta0.494 to1.292 at1, despite
32.3% final physical acceptance. Late broad joint proposals accept0.00643%.
The covariance allocation is descriptive, and inherited missing modes remain
possible. This motivates testing the fitted q0 as both initial law and joint
proposal; it does not identify initial redshift as the sole cause. No sampler
or target code changes in this diagnostic.

## 2026-10-21 — Require cross-algorithm normalizer agreement

`report_galsbi_particles.py` now requires mean normalizing masses to agree
within3.5 combined independent standard errors as well as score/region checks.
The stable mass-scale calculation rejects an otherwise invisible factor-two
mass error with identical scores. Frozen proposal/geometry identities are
checked within groups, and production/pilot seed overlap is rejected across
both groups. CPU17419419:44 passed. Immutable new reports17419420/17419428
preserve earlier reports and raw runs. [Audit and next trial](SAMPLER_SEARCH.md#mean-normalizer-comparison-audit-and-v7-declaration)
records the stricter criterion before n13/v7 outcomes, eight fixed fresh seeds,
and full-cost/held-out requirements. No scientific target or sampler defaults
change; all previous candidates still fail their original gates.

## 2026-10-21 — Gradient movement improves; normalizer gate still fails

Eight v5 runs complete at97.85seconds mean total. Three of four precision
checks pass, but bright normalizer MCSE50.3% and run-weight ESS2.884/8 fail.
[Search evidence](SAMPLER_SEARCH.md#gradient-result-and-next-initial-proposal)
records full values and the independently fitted q0 hypothesis. V6 increased
effort/changed-q0 comparison runs as17419248 on explicitly pinned V100 hardware;
partial A40 attempt17419221 remains preserved and excluded.

Integrated `particle_populations.py`, extinction-aware driver/report and tests
fromfa15521: CPU17419141 passes607 tests,1 skip;33 targeted tests include actual
mocked driver paths. Frozen v5/v6 source remains unchanged. Added generic
`poisson_augmentation.py`,16 tests passing17419159 and mathematical documentation;
no native informed-point law or sampler uses it yet. Native exchange support
CPU17419184 finds all saved bright corrections finite and nearzero; relocating
proposals instead pay roughly9.6 log units before the likelihood. Script and
JSON receipt are preserved under `doc/evidence/`.

No sampler is promoted and no shear estimate is recomputed. Next steps are
independent effort/proposal comparisons, then held-out accuracy and full-cost
validation if the development checks pass.

## 2026-10-21 — Preserve failed startup packaging and reject relocating-exchange pilot

V4 audited report fails precision and costs114.64seconds per population;
no bright exchanges accepted. Source_v5a full suite17419085 passes590 tests,
1skip. Array17419086 then failed its pre-Python manifest check because pytest
mutated a tracked cache. New source_v5b excludes ephemeral caches, disables
pytest cacheprovider, and byte-verifies all retained source/fixtures against
the passing snapshot. No science setting, seed, code or target changed and no
population output was retried. Verified completed prerequisite receipts before
replacement array17419140 because old jobs had aged out of Slurm's dependency
registry. Both packaging failures remain preserved and documented in the
[search log](SAMPLER_SEARCH.md#gradient-trial-startup-correction-and-relocation-outcome).

Audited reports17418948 complete for v1–v4. Node n9 explicit extinction path
passes33 isolated targeted tests, with integrated checks pending; not yet in
these real trials. No sampler or shear-calibration claim is accepted.

## 2026-10-21 — Reject global covariance candidate; validate local and scene proposals

Eight v3 populations completed; bright and object0 still fail precision.
Added optional native exchange/position block with exact position and changed
neighbour-choice Hastings corrections;77 targeted tests pass in17418889.
Source_v4/array17418921 isolates this change. Added physical likelihood-gradient
chain rule and frozen local metric,35 targeted tests pass in17418949; GPU
fit17418998 completes with recorded hash. [Search log](SAMPLER_SEARCH.md#covariance-result-and-local-gradient-trial)
contains results, settings, scales, limitations and job identities.

Integrated report audita7d1b83: correct empirical second moments, strict source/
geometry/configuration/receipt checks. Sixteen tests pass; audit changes prior
variance/effective-count diagnostics by<0.4%, not means or MCSE. Separate
re-reports17418948 preserve raw outputs. Added [audit limits](PARTICLE_DIAGNOSTICS_AUDIT.md).

Frozen full suite17419013:589 pass,1 skip,1 failure from11 omitted pinned
V3.9 evidence receipts. New source_v5a copies those unchanged files, with no
code/science change; original failure preserved. Replacement suite17419085
and eight-run gradient array17419086 replace cancelled pending17419016/17419026.
No accepted sampler, target change, or shear-calibration result. Explicit
extinction bookkeeping and informed-environment proposal mathematics continue
in isolated worktrees; neither is in these numerical trials.

## 2026-10-21 — Diagnose late-temperature movement and freeze pilot geometry

Student-t array17418636/report17418639 completed: bright normalizer relative
MCSE0.293, intermediate0.215; no demonstrated accuracy gain. Per-stage
counters expose bright physical acceptance1–3% at beta=1, hidden by aggregate
acceptance. Added optional frozen physical geometry and fitting CLI from
independent pilots, with covariance provenance/seed guards and3 actual bridge
integration tests. Geometry18/18 tests and fit17418727 pass; original failed
stochastic variance test preserved with its analytic uncertainty correction.
Full suite17418779:522 passed,3 skipped,one preexisting warning,83.82seconds.

Source_v3 and geometry hashes were frozen before eight fresh populations
array17418793/report17418795. Gaussian q0, N256/T64/M32, same target; optional
geometry is the experimental change. Independent structural relocation and
local-gradient metrics are being developed separately. The [search log](SAMPLER_SEARCH.md#geometry-and-structural-experiments)
records settings, costs and limitations, including strict failure on extinct
real populations and a small raw-second-moment reporting correction pending.
Production defaults remain unchanged; no sampler is accepted yet.

## 2026-10-21 — First particle populations fail the bright-object precision check

Full suite17418421 passed489 tests,1 skip; smoke17418432 and all four
N256/T64/M32 Gaussian populations17418434 completed. Report17418436 gives
bright score effective counts41.45/11.35 and normalizer relative MCSE0.233,
despite final incremental weight ESS near N. Mean cost99.13 seconds per
four-object population. Four populations are exploratory and not a promotion
sample. Exact sources/results remain under sampler_search_20261021/source_v1
and dev_v1_report.json; [search log](SAMPLER_SEARCH.md#current-status) records
the results and acceptance limits.

Integrated optional Student-t proposal commitb0dbcc8 (27 targeted tests pass,
Gaussian default bitwise replay). Added proposal CLI controls and per-stage
acceptance diagnostics. AST checks pass; source_v2/launcher/manifest frozen
before array17418636 (same effort, four fresh seeds), report17418639. Its
unchanged numerical scales increase t5 covariance by5/3, documented explicitly.
A separate independent-pilot physical geometry is under implementation.
No production-default or posterior-target change; no accuracy/efficiency claim.

## 2026-10-21 — Integrate particle implementation gates and launch frozen small checks

Integrated the isolated Arbor implementations: `annealed_particles.py`,
`galsbi_particle_proposal.py` and audited `galsbi_particle_target.py`, with
27/13/34 mathematical tests respectively. Added `particle_diagnostics.py`
and its ratio-influence/bootstrap checks, the independent-population report
CLI and a measured-only held-out panel preparer. The normalized ratio of
summed Zhat*fhat to summed Zhat retains all populations and cross-component
errors; neither particle weight ESS nor ancestry is treated as posterior ESS.

Bridge tests found/fixed zero-support cache entry, external redshift-domain
handling and undefined auxiliary proposals at beta=0. Gaussian/mixture tests
use 64 independent populations per target. Source/commit/test receipts are
in `doc/evidence/sampler_search_implementation_20261021.json`. Root AST checks
pass. Full integrated suite17418421 gates smoke17418432, development
array17418434 and report17418436. Snapshot source_v1 contains261 hashed
files. Panel17418433 completed before any development particle result.
The four-replica development run is exploratory, not an acceptance gate.

A separate optional Student-t initial proposal is being tested against the
Gaussian tail concern described in [the search log](SAMPLER_SEARCH.md#subsequent-hypothesis-proposal-tails).
It does not alter the frozen first experiment. No posterior or efficiency
improvement has been established yet; monitor each dependent stage at startup.

## 2026-10-21 — Start autonomous sampler search with independent particle construction

Owner authorized autonomous work until a good and efficient sampler is
established. Added [search protocol and acceptance gates](SAMPLER_SEARCH.md#objective-and-acceptance-boundary)
and the Arbor tree under `doc/sampler_search/.arbor/`. Development stays on the
same four objects; promotion requires new objects/seeds and matched-accuracy
cost. Initial tree nodes separate mathematical implementation from science.

Added `sbsi/galsbi_particle_target.py` and `scripts/check_galsbi_particles.py`
for a normalized-proposal bridge, fixed MH kernels, within-observation
resampling, scene-state retention and direct-score diagnostics. Isolated
Arbor executors implement/test the SMC primitives, normalized proposal and
independent bridge-balance checks. Initial syntax checks pass; CPU tests and
integration are pending. No real-posterior or efficiency result is claimed.
All new numerical work is scheduled, with the existing four-GPU ceiling.

## 2026-10-21 — Component fixes validated on small mocks and saved histories

Completed the [full component audit](INFERENCE_COMPONENT_REVIEW.md#component-audit)
and its [predeclared checks](INFERENCE_COMPONENT_REVIEW.md#validation-results).
Added the summary script/receipt under `doc/evidence/` and updated the current
evidence index and sampler design status. The source snapshot contains 250
hashed files; historical catalogues, likelihood acceptance and defaults stay
fixed. Source changes are the full-chunk twin fix, optional full shear
Hessians, mathematical moment/covariance helpers and diagnostic report CLIs.

Job 17416810 replayed four chunks / 7,915 original observations exactly on
A40-16Q. The original-node request 17416799 was cancelled before startup due
to occupancy. Restoring 22 parents adds one selected antithetic twin and two
each for same/independent coupling; no full-catalogue bias extrapolation.
Job 17416798 reused 6,400 selection chunks: point gradients unchanged, bright
g2 selection error down 2.8%, full-population selection error down 0.03%; full
cross-band/axis covariance now retained (largest |correlation| 0.314).

Jobs 17416800_0–2 validate direct scores/full Hessians at zero and nonzero
shear: maximum scaled Hessian-versus-gradient error 1.71e-7, below 1e-4.
Matched-state stencil convergence confirms the bright .002 score distortion.
Both benchmark histories still pass only 1/4 score gates, failing the bright
and intermediate objects. Individual nonconcave numerator terms do not
establish indefinite curvature for the assembled selected objective.

Final full suite: `sbatch --partition=cluster ... jobs/job_tests.sh`, job
17416801, 412 passed / 1 skipped / one existing scalar-conversion deprecation
warning, 85.64 seconds. Summary/provenance job 17416863 completed. All executed
checks succeeded; no study jobs remain. Python AST, source/artifact SHA256,
document-anchor and scoped whitespace checks complete the final audit.

No new shear fit or convergence claim. Next numerical priorities remain an
independent replicated particle reference for relative scene weights,
replacement/validation of the old support-changing score path, and full
selection curvature plus an iterated common-centre solve. Retained catalogue
c2 is not explained by the fixes tested here.

## 2026-10-21 — Review all inference components and implement bounded fixes

Owner requested improvement across the inference pipeline. Added
[component audit and small-check protocol](INFERENCE_COMPONENT_REVIEW.md#component-audit).
Corrected complete-chunk twin parent coverage in `prepare_galsbi_twin.py` with
`sbsi/mock_population.py`. Added full direct shear Hessians to the optional
`SummaryGradients` evaluator and `sbsi/inference_moments.py` for Louis moments,
common-centre assembly, bounded Newton proposals and full selection covariance.
Added two report CLIs and mathematical/coverage tests. Production defaults,
model identities, MH acceptance and frozen catalogues remain unchanged.

Core validation job17416756:412 passed,1 skipped, one existing test deprecation
warning (104.94s). Final report scripts and trained-model/data checks remain
pending. Predeclared checks reuse four mock chunks,6,400 existing selection
chunks and three small saved-chain runs; no full catalogue regeneration or
new shear claim. Unresolved priorities are relative scene-mode weights,
common-centre selection curvature/root solving and full correlated error
propagation. See the protocol for exact datasets, thresholds and limitations.

## 2026-10-21 — Native exchanges improve region access, not general score convergence

Completed the four-object/two-start corrected native-exchange pilot. Full
results and the next independent-reference experiment are recorded in
[SAMPLER_EXCHANGE_PILOT](SAMPLER_EXCHANGE_PILOT.md#decision-and-observed-results)
and `doc/evidence/sampler_exchange_results_20261021.json`. Updated the design
status and current-evidence index to point to measured outcomes rather than
leave the original gradient recommendation current. Added standalone smoke
validation, final-scene replay and direct-score summary scripts under
`doc/evidence/`. No production default or model artifact identity changes.

All eight jobs completed:17414989 tests (401 passed,1 skipped; one test scalar
conversion deprecation warning),17414996 smoke,17415003 validation,
17415005_0–3 benchmark,17415013 final scene replay and score summary. Disabled
exchanges replay all12 saved arrays bitwise. All seven reconstructed final
scenes reproduce priors/responses/likelihoods exactly; physical summaries
agree within2.84e-14. Source snapshot `sampler_exchange_20261021/source_v1`
contains249 hashed files. Python syntax, scoped whitespace and receipt hashes
checked after documentation updates.

Sampling cost rises115.9->124.3s and112.0->119.9s (about7%). Score passes are
1/4->1/4 from prior starts,1/4->0/4 from measurement starts; finite-difference
and direct-score gates agree. Bright prior-start chains accept101/1660 cold
exchanges,13 changing magnitude by>3; three chains spend substantial time
in the bright-target region. One remains faint/crowded. Measurement starts
accept0/1824 retained cold exchanges and stay bright. Pooled bright direct
score Rhat remains1.381/1.605; intermediate objects also fail. The observed
region access is useful but does not establish relative weights or a shear
correction. Mean score differences alone remain inconclusive given the large
conditional MCSE and failed stationarity.

Keep the proposal optional. Next priority is a replicated annealed particle
reference with evaluable initial proposals, exact analytic/numerical density
corrections and region/ancestry diagnostics; it has not been launched. Do not
expand the panel or simply extend chains on these acceptance counts. The
catalogue residual remains positive across magnitude-bin central estimates
and is not localized to this bright stress object.

## 2026-10-21 — Add corrected native target/neighbour exchange pilot

Owner asked to continue general sampling improvement after the failed gradient
candidate. Added `sbsi/scene_exchange.py`, opt-in exchange controls and separate
warmup/retained acceptance counters in `scripts/check_galsbi_local_sampler.py`,
and four measure/support tests in `tests/test_scene_exchange.py`. Complete
native mark exchanges include conditional target-type draws, numerical versus
analytic prior corrections, and transformed-coordinate Jacobian cancellation.
Defaults remain unchanged. The fast unchanged-environment cache is bypassed
when an exchange changes neighbours.

[Protocol and derivation](SAMPLER_EXCHANGE_PILOT.md#exact-target-and-exchange-measure)
predeclare the same four-object/two-start comparison, 15% exchanges and fixed
1 arcsec neighbour selection. Python AST checks passed. Full suite submitted
as17414989; trained-model smoke and benchmark follow only if gates pass.
No convergence improvement or shear correction is claimed yet. Limits and
results will be updated after validation.

## 2026-10-21 — Gradient pilot finishes: no general gain; scene separation and stencil error exposed

The four-object comparison (17413930_0–3) and direct-score summary (17413954)
completed. [Results and decision](SAMPLER_GRADIENT_PILOT.md#decision-and-observed-results)
are recorded with `doc/evidence/sampler_gradient_results_20261021.json`.
All 13 pilot job outcomes and source hashes are retained, including failed
test attempts; no jobs remain from this pilot. Defaults and model identities
remain unchanged.

Baseline/gradient score-gate passes: 1/4 versus 0/4 for prior-draw starts,
1/4 versus 1/4 for measurement starts. Sampling cost rises about 40%
(120.0->169.3s and 118.7->166.6s). All full latent gates fail. Some morphology
diagnostics improve, but this candidate does not earn a broader 20-object
run or production adoption. Preserve it as an opt-in experimental proposal.

Transport histories reveal separated bright scene explanations: prior-draw
chains' mean target magnitudes span about 22–26 with large crowding, while
measurement starts stay near 17 with little crowding. Draw-start ladders
make dozens of trips despite magnitude ESS about 9; measurement ladders make
0–1 trips. Raw score means also differ, but conditional direct-score MCSE is
large, so do not interpret the means alone as statistically established bias.
This is one bright stress case, not localization of the full c2 residual.

Predeclared follow-up 17414397 isolates finite-step error using identical
bright saved states and float64 arithmetic: gradient/measurement g2 is
-6.002975 at d=.002, -2.721982 at .0005, -2.485471 at .000125, versus direct
autodiff -2.469571. Added `doc/evidence/check_gradient_score_spacing_20261021.py`;
no additional sampling or target modification. Direct score noise and failed
posterior convergence remain, and no population shear is inferred.

Validation: unchanged sampler code has the preceding 397-pass/1-skip full
suite and bitwise diagnostic replay. Final receipt hashes, terminal job
states, local document anchors, Python syntax and scoped whitespace checked.
Next algorithm work should test transitions and relative mass between scene
explanations with corrected proposals or an independent annealed particle
reference; the present configuration is not a solved sampling algorithm.

## 2026-10-21 — Implement and validate opt-in gradient morphology proposals

Owner authorized proceeding with the sampler design. Added
`sbsi/gradient_proposals.py` (bounded drift, exact forward/reverse Gaussian
correction and RNG-free replica tracking), `sbsi/summary_gradients.py`
(differentiable summary kernel, context gradients and direct shear scores),
and opt-in flags in `scripts/check_galsbi_local_sampler.py`. Acceptance uses
the unchanged full target and recomputed response at each endpoint. The
proposal uses an approximate deterministic gradient, with prior-boundary
fallbacks counted; defaults retain the original trajectory. Added independent
tests in `tests/test_gradient_proposals.py` and `tests/test_summary_gradients.py`.

Predeclared [the small pilot](SAMPLER_GRADIENT_PILOT.md#frozen-initial-rules)
before trained-model results; source and launchers are frozen under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_gradient_20261021`.
The two `doc/evidence/*gradient_pilot_20261021.py` scripts validate replay and
derivatives, and summarize score precision/transport/cost on four objects.

Validation: targeted job 17413816 passes seven tests, including probability
flux with deliberately wrong drift, curved-target moments and context/shear
finite differences. Initial short toy test failed variance tolerance; its
longer repeat passes without relaxed tolerances. An earlier direct-worktree
repeat printed identical old numerical results despite edited bounds and is
retained as suspected stale shared bytecode. Jobs now use fresh bytecode and
hash-checked snapshots. Snapshot full-suite 17413835 lacked two fixture files;
complete-checkout job 17413893 gives **397 passed, 1 skipped**. Python and
shell syntax checks passed.

GPU smoke 17413836 / validation 17413858: all initial kernel replays exactly
zero, diagnostics off/on bitwise equal in every saved chain field, float64
independent density difference <=1.43e-14, shear derivative/finite-difference
agreement <=2.64e-8 absolute. New proposal sampling cost is about 1.5 times
baseline in the zero-shear smoke; there is no measured efficiency gain yet.
The declared four-object comparison is array 17413930 with summary 17413954.
No production configuration or model identity changed; no catalogue rerun.

## 2026-10-21 — General sampler redesign grounded in earlier failed proposals

Owner asked for a deeper assessment of improving sampling across magnitude
bins. Added [the algorithm design](SAMPLER_IMPROVEMENT_DESIGN.md#decision-and-evidence-boundary)
after reviewing actual target measures, proposal blocks, covariance adaptation,
temperature initialization/exchanges, and the October 16–21 results. Consulted
primary sources on MH/Langevin proposals, parallel tempering, MCMC diagnostics,
SMC and simulation-based calibration; links are retained in the design.

Recommendation: test a Metropolis-corrected, gradient-guided three-coordinate
morphology update within tempering, with full target evaluation and explicit
reverse proposal density. Approximate deterministic proposal gradients need
not change the target; usefulness is untested. Preserve exact type summation,
explicit environments and cheap invariant z/SED moves. Existing joint blocks,
pooled covariance and frozen mixture jumps already failed useful-efficiency
tests, and existing tempering already uses even/odd exchanges. Proposed lineage
diagnostics should measure actual transport, not swap acceptance alone.

The design derives a direct shear-score reference: at fixed scene and observed
data, b is shear independent, so shear autodiff does not need tree-emulator
derivatives. This still requires an independently validated numerical path;
it is not implemented or claimed lower variance. The benchmark spans all five
magnitude bins, with a separate known-failure stress panel, matched hardware,
independent starts and score-error-per-cost decisions. Replicated SMC and
matched-prior/selection calibration are later independent checks. The analytic
target prior versus numerical environment/proposal measure remains explicit.

Validation: source-backed design review; local links/anchors, referenced source
paths and scoped whitespace checked. No numerical tests were needed for this
documentation-only change. No sampler source, settings, model identity or
scientific outputs changed; no jobs launched. Next implementation order is
recorded in [the design](SAMPLER_IMPROVEMENT_DESIGN.md#status-and-immediate-implementation-order).
Sampler improvement remains a justified priority, not attribution of the
entire broadly distributed c2 residual to sampling.

## 2026-10-21 — Complete small posterior-score, tempering and precision pilot

Owner requested autonomous investigation starting with small data. Added
[the result report](SCORE_PILOT_RESULTS.md#outcome), completed the sequential
[pilot declarations](SCORE_PILOT.md#question-and-frozen-initial-design), and
retained `doc/evidence/score_pilot_results_20261021.json` with 60 source
hashes and all 21 Slurm job outcomes. Diagnostic scripts under `doc/evidence/`
cover prior-table density, matched-state scores, two support-complete maps,
independent map invariants, influential-state localization, exact chain-prefix
comparison and covariance-preserving score MCSE. Production code, model
identities, old scientific outputs and the full-catalogue result are unchanged.

Measured findings:

- Four original faint objects, four chains each: both-score Rhat<1.01 and
  bulk ESS>400 passes are 0/4 versus 0/4 at 4k plain MH; 1/4 versus 0/4 in
  the late 8k of 16k plain MH; 3/4 versus 3/4 at 8k eight-temperature sampling
  (prior-draw/measurement starts). Tempered between-run differences are <=1.40
  conditional Monte Carlo errors; each run still has a failed object and all
  runs fail the broader latent-state gate. Extensions replay their 4k prefixes
  exactly. Different GPU models and warm-ups limit causal/time comparisons.
- Near-origin transformed-score instability is localized to a near-circular
  retained state. A tapered valid map reduces one matched-state spacing change
  from -1.680 to -.00104; this is diagnostic only and production faint uses
  fixed-coordinate scores. No reference map is promoted or state deleted.
- First CPU score run 17411708 fails the 1e-6 replay gate and is preserved.
  Matching GPU kernels and historical CPU ratios identifies backend arithmetic
  as a replay issue. Float64 diagnostic 17412375 removes much tiny-step jitter,
  but large finite-step effects remain; these are not float64 posterior draws.
- 20k prior proposals / 10,664 retained: no analytic support failures, relative
  importance ESS .999999852, reweighted intrinsic e2 shift 3.59e-7. This is
  unconditional evidence, not a selected posterior-score bound.

Sampling jobs 17411723/24, 17411792/93, 17412254/55 and diagnostic jobs are
finished; final summaries 17412481/17412706 succeeded. All numerical work
used Slurm and stayed within the four-GPU ceiling. No jobs remain from this
pilot. Exact launchers, prior protocol versions and results are retained in
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/score_pilot_20261021`.

Validation: `PYTHONPYCACHEPREFIX=/tmp/sbsi-score-pilot-pycache python -m
py_compile` on the five diagnostic Python files passed. Single-thread py31
`python -I doc/evidence/test_score_pilot_bounded_map_20261021.py` passed inverse,
Jacobian, semimajor, identity and normalization checks; normalizations were
0.9999999999999947 / 0.9999999999999949. Standard-library receipt validation
passed all 60 hashes, 21 terminal job states (20 completed, one retained replay
failure), and pilot/result links and anchors. Scoped whitespace checks passed.

Limitations/next step: no new shear estimate, twin-tail replay, common-shear
root, independent mock coverage test, or attribution of the full +7.19e-4 c2.
The next bounded test should extend tempering to pass both-start score gates,
then add bright/mid/faintest objects and an explicit precision/spacing budget
before any larger recovery run. Full experiment details and the direct
problems are in [the report](SCORE_PILOT_RESULTS.md#direct-problems-on-the-surface).

## 2026-10-21 — Start bounded posterior-score and prior-density diagnostics

Owner authorized autonomous work starting with small data. Predeclared
[the pilot](SCORE_PILOT.md#question-and-frozen-initial-design) before results;
added `doc/evidence/score_pilot_20261021.py`. Stage A holds saved bright/mid
posterior states fixed and varies shear spacing, state count and score
calculation. Stage B compares the numerical generator's table density with
the analytic target density on 20,000 proposal galaxies. Current production
uses one persistent environment, so a multi-environment bank-size test does
not apply. Faint histories require a fresh small run.

Resource inspection: no active user jobs; login host has no GPU, so numerical
work goes through Slurm. Existing four-GPU follow-up ceiling retained.
No production source, models or old outputs changed. Initial validation:
Python syntax and launcher shell syntax checks; numerical gates are specified
in the pilot, with results and adaptive follow-ups to be recorded below.

## 2026-10-21 — Summarize latest experiments, evidence and visible inference problems

Owner requested the latest branch logs, a written experiment/evidence account,
and the direct problems visible in the results. Fresh `git fetch origin --prune`
and commit-date comparison identify `worktree-galsbi-tempering` at `96eb5ab`,
matching origin. Added [the current evidence report](CURRENT_EXPERIMENT_EVIDENCE.md#scope-and-current-result)
and `doc/evidence/current_experiment_review_20261021.json`. Preserved existing
uncommitted October 18 reviews and their work-log entries.

The report records the successive recovery analyses and October 18–20
localization tests. Retained m is −0.5033% ±0.8875 percentage points;
c2 is +7.1918 ±1.9334 ×10^-4 (3.72σ), still failing the joint rule.
Separates measured bright non-convergence from small tested selection/log-mean
effects, untested production score/prior approximations, and the earlier
support/twin-cutoff defects whose shear effects remain unmeasured. Explicitly
qualifies the measurement oracle: its restricted primary is negative at
2.68σ, not a zero-consistency pass, and its bright diagnostic is not
step-converged. Faint final-state calibration uses a broader cohort than the
final inference sample and is not a full convergence test.

Validation: lightweight standard-library Python JSON/hash checks passed for
34 source identities, 11 unchanged earlier science reports, current population
counts, four saved report reproductions within 1e-9, and oracle decision labels.
Local document links/anchors and scoped patch whitespace checked. No numerical
suite was needed for this documentation-only change. No raw chains, model
evaluations, large catalogue scans or scheduler jobs were run; no scientific
settings or outputs changed. Remaining discriminating checks are documented
under [evidence priorities](CURRENT_EXPERIMENT_EVIDENCE.md#what-the-evidence-now-prioritizes);
this review does not execute them or claim a cause for the full c2.

## 2026-10-20 — 1M pair c2, measurement-only true-parameter test: measurement side cleared for the positive c2 (21–25.8: −2.96 ± 1.11e-4)

Owner: "go on" (step 1 of the proposed plan). Pre-declared in `galsbi_pair1m_20261012/debug/rule_measoracle.md`
(planning choice, result and three exploratory follow-ups appended there, each written before its run).
- **Idea.** The selection half of the true-parameter score is cleared. The measurement half,
  Σ [∂ log K(y_i | z_i) − ∂ log P_sel(z_i)], was far too noisy with one real draw per galaxy (σ c2 33.5e-4).
  - Instead, redraw each galaxy's measurement 128 times from its true summary at the true shear, exactly as
    `mock_panel` does, and check E[1_A ∂ log f] = ∂ P_A per galaxy and band.
  - C8 cancels, so this tests only the flow/transport density against the generator.
  - No chains, prior or score step are involved.
- **New scripts** (data dir `debug/oracle/`, outside git):
  - `measpart.py`: CPU array; truth asserted against `calib/truth_*`; log f = log_kernel − log C8, constant
    difference checked to 9e-16; env MEAS_DRAWS, MEAS_QMC, MEAS_D, MEAS_NCHUNK.
  - `measoracle.py`: analysis on the seloracle loader.
  - `measbright.py`, `measbrightcut.py`, `measdstep.py`: follow-ups.
  - `oracle.sh`: modes `measpart`, `measanalysis`, `measbright`, `measbrightcut`, `measdstep`.
- **Planning** (2 chunks, errors only): QMC redraws project σ 1.65e-4 at 64 draws vs 3.88e-4 iid. 128 QMC redraws
  chosen (projected ≤ 1.17e-4).
- **Runs:**
  - 17386608 + 17386652 + 17386712: 25 × 16 cores, ~110 core-h. The exact-truth assert stopped tasks, before any
    output, on x86-64-v4 nodes (th-cl-262h*, usm-cl-1116cs*) and on usm-cl-826bac02 / usm-cl-seitz3. They were
    rerun on th-cl-hua/rome only.
  - Analysis 17386675.
- **Result.** All 1,990,072 galaxies; step2 reproduced; redraw acceptance equals the selpart acceptance in every
  band (|z| ≤ 0.9).
  - **Primary (mag 21–25.8, total I): c2 −2.96 ± 1.11e-4 (−2.7σ), m +0.51% ± 0.49%.**
  - Bands: 23–24 −0.5 ± 4.6; 24–25 +0.7 ± 3.5; 25–25.8 +0.7 ± 5.7; 21–23 −12.0 ± 5.1; < 21 −228 ± 15 (in their
    own I_B).
- **Follow-ups (exploratory):**
  1. 145 bright galaxies, 2048 redraws, no cuts: the mean score at the truth is 0 to ~1e-5 of its own Fisher
     information, with and without the disk transport (b = 0), at d = 0.001/0.002/0.004.
  2. The same galaxies with each cut region: the identity holds to ~1e-5, and the selpart and redraw code paths give
     the same pass-fraction slopes.
  3. The same redraws with d = 0.001 on 50 chunks:
     - primary Δ +0.18 ± 0.07e-4 → −2.72e-4 at zero step, stable;
     - bright Δ +10,020 ± 822e-4 and 21–23 Δ +17.7 ± 3.5e-4: these bands are not converged in the step.
     Known-truth bright scores are so sharp, compared with the production information, that tiny numerical errors
     of log f dominate. The bright measurement side stays untested at production precision.
- **Reading.**
  - The measurement density does not carry the positive c2 (bound ≤ −0.7e-4 on the positive side). With the
    selection cleared, the true-parameter score at 21–25.8 is, if anything, negative.
  - The +7.2e-4 sits between the true-parameter score and the production score: the posterior side (the scene
    prior vs the generator tables, bank size or step inside the production score), or bright galaxies (untested,
    +2.1 ± 0.5e-4 of c2).
  - Next (cheap): the production score step and bank size on saved states (d = 0.001/0.004), then a sampler run
    with no data against the generator population.

Validation: the code checks above; full-mode asserts (500 files, every galaxy); compile checks. No production code,
models, settings or the 1M result changed; nothing retuned.

## 2026-10-20 — 1M pair c2, selection-only true-parameter test: selection cleared (c2 −0.99 ± 1.14e-4 vs production +7.20 ± 1.93e-4)

Owner: "work autonomously" (the follow-up proposed after the full oracle failed its gate). Pre-declared in
`galsbi_pair1m_20261012/debug/rule_seloracle.md`; Test and Result sections appended there.
- **Idea.** Every mock galaxy's true summary is known.
  - At the true shear (0.02, 0), the selected galaxies' summed ∂ log P_sel(z_i) must equal N ∂ log Z, if the
    generator's selection is what the model's selection describes.
  - P_sel is the C8 probability × the fraction of 256 QMC flow draws passing the cuts, with b per draw and the
    disk transport.
  - This tests H2 galaxy by galaxy, with no chains and no measurement-density term.
- **New scripts:**
  - `oracle/selpart.py`: CPU array, source_snapshot; features asserted against `calib/truth_*`; rows with a zero
    band count are rerun with 4096 draws.
  - `oracle/seloracle.py`: the step2 loader; per-scene errors plus the selection error; draw-count check.
  - `oracle/selnoise.py`: planning.
  - `oracle/oracle.sh`: modes `selpart`, `selanalysis`, `selnoise`.
  - Selection at the truth: `oracle/sel`, 4 replicas, 4 GPUs.
- **Runs:**
  - per-galaxy 17377008 + 17377084: 25 × 16 cores. Tasks 16–22 failed the exact-feature assert on th-cl-dell
    nodes (CPU rounding) and were rerun with those nodes excluded.
  - selection 17376940: 4 × ~1.55 h GPU;
  - analysis 17377695.
  - Check node vs faintleak/sel: replicas 0–2 agree to 1e-8; replica 3 (different GPU model) agrees to 3e-4.
- **Result.** All 1,990,072 galaxies; step2 reproduced to 1e-9; draw check +0.07 ± 0.76e-4.
  - **O c2 = −0.99 ± 1.14e-4 (−0.9σ)**; g1: m −0.13% ± 0.56%.
  - Bands (c2, 1e-4): < 21 −9.99 ± 3.65 (−2.7σ; draw-sensitive); 21–23 +2.41 ± 2.63; 23–24 −4.98 ± 4.19;
    24–25 −0.12 ± 3.85; 25–25.8 −3.51 ± 7.01. None is positive beyond 1σ.
- **Reading (as declared): selection cleared.**
  - H2 can account for at most +1.3e-4 of the +7.2e-4 (O + 2σ; |O| + 2σ = 3.3e-4).
  - Cleared before: the half-chain log-mean bias, the faint g1→g2 leak; shape posteriors ≤ ~1.8e-4.
  - What remains: the measurement density versus the generator (not testable at true parameters with this
    noise), or the posterior side (the sampler's scene prior, the score step).

Validation: the code checks above; the full-mode asserts (500 files, every galaxy present). No production code,
models, settings or the 1M result changed; nothing retuned.

## 2026-10-20 — 1M pair c2, true-parameter (oracle) score test: precision gate fails (projected σ c2 33.5e-4); selection-only version projected at 1.8e-4

Owner: "go with the true-parameter score test". Pre-declared in `galsbi_pair1m_20261012/debug/rule_oracle.md`
(test result and planning estimate appended there).
- Idea: the mock is reproduced bit for bit, so every galaxy's true nine-value summary is known. Score each galaxy at
  its truth and the true shear (0.02, 0): ∂ log K − ∂ log Z. With the generator's K and Z its sum is zero. No chains,
  prior or Newton step are involved, so this splits "likelihood/selection" from "posterior side".
- New scripts:
  - `oracle/oracle_kernel.py`: CPU, source_snapshot; rebuilds z_i with `prepared()` and checks it against
    `calib/truth_*`; b from `response.predict` as `mock_panel`; log K at 5 nodes.
  - `oracle/oracle.py`: the step2 loader from `halfchain.py`; bands as bandsel; posterior-side part P.
  - `oracle/oracle.sh`; `oracle/sel/plan.json` (written, unused).
- Test (chunks 0 and 250; jobs 17376565, 17376642):
  - code checks pass: features exact, 0 nonfinite log K, step2 reproduced to 1e-9;
  - one bug fixed before any number: a parquet row count read with `columns=[]`.
- **Gate fails:** projected σ(c2) = 33.5e-4 (needs ≤ 3.6e-4); bright band 2518e-4, faintest band 14e-4. With the
  true shape known, each score carries the galaxy's full measurement noise, and the antithetic twin does not cancel
  it. As declared, the full run and the GPU selection run were not started; no central value was computed.
- Planning estimate (errors only, `oracle/selnoise.py`, job 17376692, CPU 2 min):
  - The oracle score's selection part, ∂ log P_sel(z_i) − ∂ log Z (C8 probability × cut-pass probability at the
    true summary, 64 QMC draws), is projected at σ(c2) ≈ 1.8e-4 (m ±0.87%). The C8 part alone is 0.05e-4.
  - That is a galaxy-by-galaxy H2 test at ~1/7 the error of the 70–100 GPU-h brute-force generator test.
  - Cost: ~7 CPU-h + the selection at the truth (4 GPUs × ~1.6 h). Needs a new rule; the owner decides.

Validation: the code checks above. No production code, models, settings or the 1M result changed; nothing retuned.

## 2026-10-20 — 1M pair c2, H3 step 1: profile-ratio log-mean bias cleared (Δc2 +0.0065 ± 0.0046e-4); H2 code paths match the generator; independent mocks show the same c2 sign

Pre-declared `galsbi_pair1m_20261012/debug/rule_halfchain.md` (result and follow-ups appended there). Each
galaxy's score is a central difference of log-means of likelihood ratios over (states × 4 chains). The bias of a
log-mean scales as 1/n, so the score from 2 chains against 4 estimates it (jackknife over chain halves;
`halfchain/halfchain.py`, job 17375738, CPU, 4 min).
- Checks: the analysis reproduces the 1M step2 g to 1e-9. The chain halves agree in every band (largest 1.8σ).
- All 1,990,072 galaxies: **Δc2 = +0.65 ± 0.46e-6**, Δg1 = +0.24 ± 0.16e-6 (0.001% of m). Faint is significant
  (+0.16 ± 0.04e-6) but negligible. → this H3 mechanism is cleared (bound 1.6e-6, 1/450 of the excess). Still
  untested in H3: the bank-size bias inside each state's ratio and the finite-difference step d = 0.002.
- Follow-ups (exploratory):
  - H2 by code reading: the generator (`mock_panel`) and the selection (`run_galsbi_bright.selection`) use the
    same sheared shape, detection probability, flow, cuts and shape transport. The QMC draws are randomly shifted
    per galaxy, so they are unbiased.
  - Selection tilt from 16 vs 8 draws: dlogZ/dg2 differs by +0.0005 ± 0.0004, against the ~0.0037 c2 would need.
  - A brute-force generator test would cost ~70–100 GPU-h (flow-draw noise dominates), so it goes to the owner.
  - Earlier independent mocks show the same c2 sign: twin 2×100k +0.00096 ± 0.00056, unpaired 500k
    +0.00028 ± 0.00065. Combined with the 1M (not pre-declared): +0.00071 ± 0.00018. The c2 looks common to the
    method, not a fluke of one mock.

Validation: step2 reproduction and the chain-half check above. No production code, models, settings or the 1M
result changed; nothing retuned.

Next (owner's decision): either the truth-parameter score test (each galaxy's score at its true regenerated
parameters, no chains; a few GPU-h, new script) to split "likelihood/selection" from "chains/score step", or the
brute-force generator test of the selection tilt (~70–100 GPU-h).

## 2026-10-19 — 1M pair faint g1→g2 leak test: no faint leak (Δc2 −0.21 ± 1.79e-4); calibration replay gate passes exactly

Pre-declared `galsbi_pair1m_20261012/debug/rule_faintleak.md` (result appended there). All 5612 faint units of both
sides were re-run at g* = (0.0199, 0.00072), the current 1M estimate, with the production faint command and the same
per-unit seeds (jobs 17349882/17349883, 0 failed). The selection was re-run at g* with the bandsel seeds (job
17349881, 8 replicas × 800 chunks). `faintleak.py` (job 17354580) compares the faint scores at g* with their
straight-line prediction from g = 0.
- Checks: the analysis reproduces the 1M step2 g to 1e-9. Selection node (0, 0) repeats bandsel to ≤ 3e-9 in 6385
  of 6400 chunks, but 15 chunks (replicas 3, 5) differ by up to 3.2e-5 relative. That fails the declared 1e-6, so,
  as declared, the unpaired selection error is used (5.4 vs 5.0e-4 per component; immaterial).
- All faint (1,775,787 galaxies): **Δc2 −0.21 ± 1.79e-4** (score −0.14 ± 1.54, selection −0.07 ± 0.92). Bands:
  23–24 −1.03 ± 1.35, 24–25 +0.18 ± 0.78, ≥ 25 +0.64 ± 0.48. **Δg1 +0.01% ± 0.89% of m**, secant/I 1.000.
- 1M with faint evaluated at g*: m −0.49% ± 0.90%, c2 +0.000698 ± 0.000195 (still FAIL, 3.6σ).
- Reading: no faint leak. Bound |Δc2| + 2σ = 3.8e-4 (at most ~half the +7.2e-4 excess; measured zero). With bright
  checked earlier, the one-step evaluation at g = 0 is cleared as the c2 source for ≥ 91% of galaxies.
- Calibration gate: the GPU replay (job 17349897, A40) rebuilds chunks 0 and 250 of the stored mock bit for bit:
  same scenes (1996, 1951), measured columns and features max |diff| 0.0 (`rule_calib.md` gate section).
- Code fix on the way: the full-mode analysis ran out of memory twice (96G, 360G). `x['L'][i]` on an `NpzFile`
  re-read the whole array for every galaxy, and each row kept it alive. It now reads each array once (peak 1.5 GB);
  no number depends on it.

Validation: the step2 reproduction (1e-9), the node-0 comparison per replica, and the unpaired-error recomputation
(`faintleak_result_unpaired_errors.json`; it reproduces the paired selection errors to 1e-6 first). No production
code, models, settings or the 1M result changed; nothing retuned.

Next (owner's decision): Stage 1b is complete. The one-step evaluation, the faint chains and the shape posteriors
do not explain c2; the bright non-convergence carries ~+1.3–1.8e-4. What is left points at the posterior-to-score
step (profile-ratio score estimate, H3) or the selection term (H2).

## 2026-10-19 — 1M pair faint posterior calibration from final chain states: faint calibrated in e2, shape posteriors do not carry most of c2

Pre-declared `galsbi_pair1m_20261012/debug/rule_faintcalib.md` (result appended there). Faint chains are not kept,
but every faint unit stores each chain's final state (`final_scenes.npz`) and start. A converged chain's final
state is a posterior draw, so the calibration identity of `rule_calib.md` applies to it. `calib/faintcalib.py`
(job 17349935, CPU) compares these states against the regenerated truth.
- Faint, production chains (1,778,506 galaxies): final − truth e2 **−0.73 ± 0.96e-4**; starts −0.30 ± 0.96e-4.
  Bands: 23–24 +3.2 ± 2.2, 24–25 −2.7 ± 1.6, ≥ 25 −0.5 ± 1.6 (1e-4). → faint calibrated; no start offset to
  remember; H1 (faint g2 non-convergence) disfavoured as the c2 source.
- Method check: bright w8 final states +21.3 ± 4.8e-4, the same sign and size as the chain mean (+26.3 raw).
- Approximate conversion to c2 (Gaussian intrinsic prior, score ≈ E_post[e]/σ², step2 information):
  - bright non-convergence +1.8 ± 0.2;
  - mid −1.6 ± 3.1;
  - faint −2.0 ± 2.7;
  - sum ≈ −1.8 ± 4.1e-4, against the 1M c2 of +7.2 ± 1.9e-4.

  The shape posteriors do not carry most of c2. The bright non-convergence plausibly carries ~+1.3–1.8e-4. The
  conversion ignores the measured-shape response path and selection; it is a budget, not a correction.
- Faint leak test (`rule_faintleak.md`): code check on partial outputs passes. The analysis reproduces the 1M
  step2 g to 1e-9, and selection at node (0, 0) reproduces bandsel to 1.9e-9 relative. The full run continues
  (~13 h).

Validation: convention check e(a, ψ) vs truth < 1e-12; `faintcalib.py test` then full, exit 0. No production code,
models, settings or the 1M result changed; nothing retuned.

Next: the faint leak result. What is left of c2 points at the step from posteriors to the score (profile-ratio
estimate, H3) or the selection term (H2), not at the shape posteriors.

## 2026-10-19 — 1M pair Stage 1b: posterior calibration finds unconverged bright chains (blend mode); faint leak test running

Owner: "go with Stage 1b" (`galsbi_pair1m_20261012/debug/reflection_c2_20261018.md` §4). Two pre-declared rules,
written before any of their numbers existed.

**Part 2, posterior calibration** (`debug/rule_calib.md`, result appended there). The mock is model-generated, so
for converged chains the average of (posterior mean − truth) over the observed galaxies is zero for any split made
from the data or the chains. The true latents were regenerated exactly from the mock seeds (`debug/calib/
regen_truth.py`, job 17349896, 2,000,086 rows). Gate: the truth features equal the generator's `prepared` features
exactly (chunks 0 and 250, max |diff| 0.0). Measured mag vs true m_r corr 0.94; posterior vs true e2 corr 0.91.
The GPU replay of the measured columns waits for the GPU cap. `calib/calib.py` ran on the kept chains (job
17349917, ~5 min, 32 CPUs).
- Bright, production w8 chains: posterior e2 − truth (minus the small shear-offset prediction) **+30.0 ± 3.8e-4
  (7.9σ)**. The exact-shear measurement-start chains give +21.7 ± 3.7e-4. e1 is fine (+6.1 ± 3.8e-4).
- Mid 21–23 (w8, D32): e2 is calibrated, −5.6 ± 10.8e-4. That is too weak to test the mid c2 (+6.2 ± 2.9e-4).
- Where (exploratory; split by chain output only):
  - The excess is entirely in the 44% of bright w8 galaxies whose chains visit the blend mode, where the target is
    made fainter and its light is given to neighbours. Their posterior magnitude is **0.42 ± 0.01 mag fainter
    than the truth**, and their e2 excess is +62.8 ± 7.0e-4.
  - These chains start 3.4 mag too faint (the drawn start) and are still relaxing. Magnitude error goes +0.51 →
    +0.34 mag, and e2 error +0.0066 → +0.0052, from the first to the second half of the kept steps.
  - The other bright galaxies are calibrated: e2 −0.0002 ± 0.0006.
  - Measurement starts reach that mode in only 12% of galaxies.
- Reading: the bright posterior e2 excess is chain non-convergence in the blend direction, not a mismatch between
  the sampler's likelihood and the generator. The earlier shear-score checks (w8 vs 4× longer chains) did not show
  it. It does not by itself explain the 1M c2: bright's own c2 is +11.3 ± 6.1e-4, the excess is spread over all
  bands, and faint chains are not kept, so faint is not measured.
- Correction to the rule text: its "pair sum" check is not valid. Being in a pair depends on the twin's data,
  which the chain does not condition on, so the check is not used.

**Part 1, faint g1→g2 leak** (`debug/rule_faintleak.md`; running).
- Run A: every faint unit at g* = (0.01990, 0.00072), with the same per-unit seeds as `faintdraw/`
  (`faintleak/fl.sh`; array 17349882 tasks 0–1, plus 17349883 tasks 2–3 after the selection array and the GPU
  replay).
- Run B: selection at g* with the `bandsel` seeds (`faintleak/sel`, array 17349881, 2 at a time).
- Startup checks:
  - Run A profile nodes sit at g* ± 0.002, and seeds and galaxies equal `faintdraw/` (the only other change is
    the disk term, which must be computed away from g = 0).
  - Run B node (0, 0) reproduces `bandsel` to 7e-9 relative.
- Units take ~750 s (vs 560 s at g = 0), so the run needs ~13–14 h wall within the 4-GPU cap.
- Analysis: `debug/faintleak.py` (CPU).

Validation: truth gate as above; `calib.py test` (40 units/run) then full run, exit 0; both rules pre-declared.
No production code, models, settings or the 1M result changed; nothing retuned.

Next: finish the faint leak test and the GPU replay. The bright drawn-start non-convergence is a separate finding.
Fixing it (longer warm-up, a measurement or mixed start, or moves between the two modes) and checking whether
bright c2 moves is an owner decision.

## 2026-10-18 — 1M pair per-band selection test (Stage 0): c2 excess is spread over all bands, selection re-check agrees

Result of `galsbi_pair1m_20261012/debug/rule_bandsel.md` (array 17347485: 8/8 replicas, 0 failed, 1.5–1.9 h each
on ≤ 4 GPUs; analysis 17347604, 3 min; result appended to the rule; `debug/bandsel_result.txt`, `.json`). Check:
the paired step2 g (0.019899, +0.000719) is reproduced (1,990,072 galaxies).

- Per measured-magnitude band with its own selection term, c2 in 1e-4: < 21 +11.3 ± 6.1, 21–23 +6.2 ± 2.9,
  23–24 +15.4 ± 5.2, 24–25 +2.3 ± 3.7, 25–25.8 +3.1 ± 5.6. Weighted mean **+6.57 ± 1.86 (3.5σ)**; χ² about the
  common mean 5.2/4 (p 0.27) -> **spread (common cause), not one band**. m per band within 1σ of 0.
- The `rule_mirror.md` 23–25 / ≥ 25 sign flip is an edge artefact (the two bands differ by 0.8σ).
- Selection re-check with new seeds: dlogZ/dg2 0.00430 ± 0.00037 vs production 0.00431 ± 0.00038 (0.0σ) -> the
  selection Monte Carlo is fine. Pooling all 16 replicas: m −0.41% ± 0.85%, c2 +0.000720 ± 0.000186.
- Closure, data vs model: band fractions and mean measured e2 all within 3σ (largest: 24–25 fraction +2.4σ) -> no
  mock–model mismatch flag from these checks.
- Limitations: band χ² treats bands as independent although their selection terms share chunks (selection is the
  smaller part of each error); the 23–25 mirror e2 comparison was not printed. Not measured.

Validation: all 8 replicas exit 0 with `result.json`; analysis exit 0, step2 reproduction check passed. No code,
models, settings or the 1M result changed; nothing retuned.

Next (per the rule): the faint long-chain run (`rule_faintlong.md`, ~75 GPU-hours) is not the next step. Next is
Stage 1b (faint g1→g2 leak, posterior calibration) and the selection/likelihood side as a common cause; the source
review's twin chunk cutoff and shape-support gap act on all bands and are unmeasured candidates. Owner decision.

## 2026-10-18 — Mock and inference source audit: support coverage and twin chunk cutoff

Owner requested another review of mock generation and inference on the latest
branch. Verified `worktree-galsbi-tempering` at21ed49e against origin, traced
the frozen4a506ca generation snapshot and the actual production
`original/comoving.py`, `twin/comoving.py`, `pair_step.py` and `debug/step2.py`.
Added [the pipeline review](MOCK_INFERENCE_PIPELINE_REVIEW.md#conclusion-and-scope--october-18-2026)
and linked it from the earlier inference diagnosis.

Two concrete findings, with no measured attribution of c2:

- The effective intrinsic distortion support is a<0.98. The co-moving
  Möbius map does not cover that smaller ellipticity disk; discarding mapped
  off-support states does not restore the missing incoming region. A
  deterministic allowed-node/disallowed-preimage witness passes for both
  the current method and production postprocessor (b=0). A constant-kernel
  uniform-disk example gives ratio0.999481586 instead of1. Actual F24
  per-type shape-prior missing masses in the simplified b=0,K=1 case are
  4.43e-5 red and6.81e-7 blue; these are not posterior or shear-error bounds.
- The frozen twin generator clips every chunk at the last original-selected
  scene, including499 complete chunks. Pair metadata confirms389 complete
  chunks exclude1554 parent positions (maximum21 per chunk). Missing
  selected twins are not counted yet; their selection/score effect requires
  replay. An exhaustive two-parent example demonstrates the induced
  selection bias, without estimating its size in the scientific mock.

Also qualified the frozen-b approximation: an older100k comparison supports
it at the tested precision, but its zero-centre cancellation argument is
insufficient at the current bright correction's nonzero centre. This
supersedes the October15 claim that start memory is the only approximation
that could matter. The model mapping and measured cuts otherwise agree at
source level; replay tests do not constitute integrated distribution closure.

Validation: scheduler CPU audit17348387 completed in38s; extended audit17348743
completed in20s, both exit0. The first launcher17348384 failed before execution
because its temporary script was not on the shared filesystem; the corrected
launchers read the preserved audit from `doc/evidence/`. Added
`mock_pipeline_review_20261018.py` and its JSON receipt there, including16
SHA256 records, actual configuration checks, analytic counterexamples and
pair-table counts. Post-write verification passed all16 source hashes,
16 document links/anchors, JSON assertions, audit syntax and whitespace;
`git diff --check` passed. No F8/C8/emulator evaluation, MCMC or mock regeneration.
No scientific code, models, results or running selection jobs changed.

Next: verify a full-support likelihood-ratio construction and recomputed
response at the actual centre; replay only the omitted twin tails; then
measure a matched selection-normalized score on a controlled subset. The
remaining c2 is still unlocalized. These findings do not establish that either
defect explains it or that the quoted uncertainty covers these systematics.

## 2026-10-18 — Latest-branch review of context, evidence and the unlocalized c2 residual

Owner clarified that the requested review concerns the most up-to-date branch,
not the old `dev` checkout. Verified `worktree-galsbi-tempering` at21ed49e
against the remote head and added
[the current inference diagnosis](INFERENCE_DIAGNOSIS.md#scope-and-conclusion--october-18-2026).
The review covers the model-closure target, successive1M estimates, bright
step correction, faint start/length tests, w8/mid controls and mirror evidence.
It distinguishes matched aggregate results from full convergence and the
experiment's two-sigma rule from sub-percent calibration acceptance.

Read-only small-report audit reproduces current m=-0.50330% ±0.88749 percentage
points and c2=+0.000719179 ±0.000193342 (3.72sigma). The successive versions
reuse one mock. Bright/mid reference agreement does not exclude common bias;
the mid test still indicates g1 sensitivity. Added
`doc/evidence/inference_review_20261018.json` with16 source hashes, extracted
comparisons and branch identity. No raw chains or large arrays were reduced.

Stage0 production result is absent at review time: four array17347485 workers
running, remaining replicas and analysis17347604 pending. Source inspection of
`debug/bandsel.py` finds that its between-band chi-square omits cross-band
covariance, selection covariance drops cross-component terms, and the forward
comparison extrapolates from zero shear without propagating slope uncertainty
while comparing mean e2 directly to the zero-shear mirror run. Documented these
as analysis limitations to resolve before causal readings, not explanations
of the current c2. Band-conditioned likelihoods also omit band-count information;
the review gives the exact score decomposition to check. Original rules,
analysis scripts, outputs and running jobs are unchanged.

Validation: lightweight JSON assertions and SHA256 checks passed16 sources;
all10 review links/anchors resolve; documentation whitespace checks pass.
No inference-code or model changes, numerical suite or new experiment.
Next: qualify Stage0 with joint covariance and matched-shear checks, then choose
a bounded sampler/score-transport diagnostic from its evidence; the full faint
long-run draft stays on hold. See
[next steps](INFERENCE_DIAGNOSIS.md#where-to-look-next).

## 2026-10-18 — 1M pair c2: reflection, and per-band selection test (Stage 0a–0c) launched

Reflection `galsbi_pair1m_20261012/debug/reflection_c2_20261018.md`: the c2 excess (+0.000719 ± 0.000193) is not
localised. Bright and mid chains are converged, yet every band shows some excess, and earlier per-band slopes
(`rule_mirror.md`) put band membership at the g = 0 draw, unlike the data. Earlier claims corrected there: the faint
start effect is only 2.0σ, and the half-split predictor failed for mid. The faint long-chain rule
(`rule_faintlong.md`, ~75 GPU-hours) stays a draft on hold.

Owner: "go" (Stage 0). Pre-declared `debug/rule_bandsel.md`: each measured-magnitude band gets its own c2 with its
own selection term. The production `selection()` (source_snapshot, unchanged) runs with plan
`debug/bandsel/plan.json`: limits [21, 23, 24, 25, 25.8], new seed family 2029107001, 8 replicas × 800 chunks. Band
Z = Z(hi) − Z(lo), so membership follows the measured magnitude at each shear node, as in the data. The same run
re-checks the all-galaxy selection slope with new seeds (0c). It also compares band fractions and mean measured e2,
data vs model (0b).
- Files (data dir, not in git): `debug/bandsel.sh` (1 GPU, 16 cores, replica = array id), `debug/bandsel.py`
  (analysis; `BANDSEL_TEST=1` = code test), `debug/bandsel_analysis.sh`.
- Timing test 17347393 (3 chunks, `bandsel_timing/`, not a result): 9.3 s per chunk on an A40. The array runs at
  5–7 s per chunk (production, sharing GPUs: ~17 s).
- Code test 17347455 on the timing chunks: reproduces the paired step2 g (0.019899, +0.000719) to 1e-9 and runs
  every section; its band numbers mean nothing (3 chunks).
- Jobs: array 17347485 (0–7, at most 4 GPUs), analysis 17347604 (afterok, CPU). Expected ~3–4 h.

Next: apply the rule's readings when `debug/bandsel_result.txt` exists.

## 2026-10-18 — 1M pair mag 21–23 long-chain test: w8 is enough; these chains are not the c2 source

Result of `galsbi_pair1m_20261012/debug/rule_midlong.md` (array 17340158: 2,250/2,250 runs, 0 failed, 11.2 h on
4 GPUs; result appended to the rule; `debug/midlong_result.txt`, `midlong_result.json`). 1,125 units, 22,500
mag 21–23 galaxies, drawn start, same seeds in both arms; scaled ×7.998 to all 179,947 mag 21–23 galaxies.

- w8 − D32: **g2 +0.05 ± 0.07e-4 (+0.7σ), g1 −0.05% ± 0.03% of m (−1.5σ) -> w8 is enough for mag 21–23**.
- D32 − it0 (current step2.py input), scaled: **g2 +0.13 ± 0.67e-4 (+0.2σ) -> not the source of c2** (≤ 1.46e-4
  at 2σ; expected −1 to −3e-4 did not appear). **g1 +1.07% ± 0.31% of m (+3.4σ)**: converged mag 21–23 chains
  would raise the 1M g1 by about 1% of m (scaled, not a measurement). w8 − it0: g2 +0.52 ± 0.77e-4, g1 +0.68% ±
  0.36%.
- Mixing (median rhat of e1 e2 m_r log size Sérsic / bulk ESS / s per galaxy): D32 1.02–1.03, ESS 115–506, 148 s;
  w8 1.06–1.10, ESS 31–165, 50 s.
- Analysis fix: the first analysis job 17340161 ran `debug/mid.py` on one of its 8 cores (~8 MB/s through ~200 GB
  of chain archives, projected ~5.9 h against the 6 h limit; the limit cannot be raised by the user). `mid.py` now
  loads units in a `multiprocessing.Pool(SLURM_CPUS_PER_TASK)` (`load(unit)`, the same scores/mixing code; results
  combined in the original unit order). With owner approval 17340161 was cancelled and rerun as 17343398: 62 min.
  Checked: `py_compile`; job exit 0, `midlong_result.json` written.
- Limitations: a subset (every 8th unit) with no pair partners, so errors are per galaxy rather than paired; the
  scaled numbers assume the subset is representative.

Next (owner decision): the remaining c2 (+0.000719 ± 0.000193) is not in the mag < 21 (w8) or the mag 21–23
chains; what is left is the faint drawn-start band and the F8/C8 frame tilt. A full mag 21–23 w8 rerun would
mainly move m (about +1% of m), not c2.

## 2026-10-18 — 1M pair: mag 21–23 long-chain test launched

Owner: "run the mag 21–23 long-chain test on 4 GPUs". Pre-declared `galsbi_pair1m_20261012/debug/rule_midlong.md`.

- `debug/mid_prep.py` -> `debug/mid_units.txt`: every it0 bright galaxy with measured mag >= 21 after the step2.py
  cuts, 179,947 galaxies in 8,998 units (asserted disjoint from `truth_units.txt` and, with it, all bright
  galaxies). Test subset: every 8th unit, 1,125 units / 22,500 galaxies.
- Two arms, both drawn start, 8 rungs, narrow .5, `--shear 0 0` (as it0), `snapshot_d776abb`, the same seeds per
  unit (2027907001 + k − 1, bank 2032907001 + k − 1): D32 (32768 / 8192, reference) and w8 (14336 / 8192).
  Launcher `debug/midlong/ml.sh` (from `opt1/o1.sh`; arms, steps, `--shear 0 0`, seeds, paths, 320G differ), array
  17340158 (4 GPUs, `inter,cip`, 30 one-core slots each), 2,250 runs, all D32 first.
- `debug/mid.py` (analysis 17340161, afterany; refuses before all runs are done): w8 − D32, and D32 − it0 /
  w8 − it0 against the current `pergal_<side>.npz` scores, scene-paired errors, also scaled ×179,947/22,500;
  mixing per arm. Readings stated in the rule before any number; no 1M m/c2 from the subset; nothing retuned.
- Expected about 13–15 h on 4 GPUs. Checked: `py_compile`, `bash -n`; mid.py refusal path.

## 2026-10-18 — 1M pair with w8 bright chains: m = −0.50% ± 0.89% passes, c2 = +0.00072 ± 0.00019 still fails

Result of `galsbi_pair1m_20261012/debug/rule_brightw8.md` (array 17337490: 1,720/1,720 units, 0 failed, 7.8 h on
4 GPUs; analysis 17337491, 37 min). 34,338 mag < 21 galaxies, drawn start, 14336 steps / 8192 warm-up.

- Repeat check (430 Test 1 units, same seeds, other GPUs): Δg2 +0.07 ± 0.07e-4, Δg1 −0.01% ± 0.04% of m ->
  reproducible.
- w8 − 8192-step drawn start, all bright galaxies: **Δg2 = −1.84 ± 0.25e-4** (expected about −2.3e-4 from Test 1),
  Δg1 = −0.40% ± 0.11% of m. Largest rhat per galaxy median 1.43 -> 1.29.
- 1M pair (w8 bright + drawn-start faint): **m = −0.50% ± 0.89%** (pass), **c2 = +0.000719 ± 0.000193** (3.7σ,
  fail). Before: drawn 8192 bright m −0.10% / c2 +0.000903; measurement-start bright m −0.43% / c2 +0.000702.
  Bright g1 score at g1 +843 ± 1,088 (zero within 1σ).
- Reading: the bright chains' extra +2e-4 is gone; c2 is back near the measurement-start value and still fails.
  Not final (rule): mag 21–23 chains are still the short measurement-start it0 chains. Where the rest of c2 sits
  is not resolved by this run.
- Outputs: `brightw8/` (~110 GB chains), `step2_result_brightw8_faintdraw.json`, `diff_brightw8_vs_*.json`.

Next (owner decision): the mag 21–23 long-chain test with drawn starts and the w8 setting.

## 2026-10-17 (evening) — 1M pair: all bright units with the w8 setting launched

Owner: "run w8 on all bright units on 4 GPUs". Pre-declared `galsbi_pair1m_20261012/debug/rule_brightw8.md`: the
bright drawn-start run of `rule_brightdraw.md` (1,720 units, 34,338 mag < 21 galaxies, same seeds, shear and
`snapshot_d776abb`) with `--steps 14336 --warmup 8192`; all units rerun, including Test 1's 430.

- Launcher `debug/brightw8/bw.sh` (copy of `brightdraw/bd.sh`; only steps/warm-up, paths and memory 200G differ),
  array 17337490 (4 GPUs, `inter,cip`, 30 one-core slots each); analysis `brightw8/analysis.sh` job 17337491
  (afterany): `PERGAL=_faintdraw step2.py debug/brightw8`, and `brightdraw_diff.py` vs the 8192-step drawn-start
  run (all galaxies) and vs Test 1's w8 (430 units, repeat check). Test 1 seeds checked to be the same per unit.
- `brightdraw_diff.py` takes optional `NEW OLD` run names and compares the units OLD has (default unchanged:
  brightdraw vs step2; reproduction check job 17337492). Test 1 analysis rerun 17337125: text identical to the first
  run, `opt1_result.json` written.
- Expected: ~6.7 h (Test 1: 28 min per w8 unit per slot), ~110 GB of chains. Expected readings in the rule file.

## 2026-10-17 (afternoon) — Test 1: a longer warm-up (w8) replaces 4x long bright chains; speed switches not adopted

Result of `galsbi_pair1m_20261012/debug/rule_opt1.md` (array 17331485: 1,290/1,290 runs done, none failed, 4.5 h on
4 GPUs; analysis 17331496 -> `debug/opt1_result.txt`). 430 units, 8,582 mag < 21 galaxies, drawn start; reference
D32 steps 8192–32768. Arm − reference, same galaxies:

- Control D8: g2 +0.63 ± 0.14e-4 (4.6σ), g1 +0.15% ± 0.07% of m -> does not match (as expected; the test can
  tell arms apart).
- **w8** (14336 steps / 8192 warm-up, 8 levels): g2 +0.05 ± 0.09e-4 (0.6σ), g1 +0.03% ± 0.04% -> **matches**.
- w4 (10240 / 4096): g2 +0.38 ± 0.11e-4 (3.3σ) -> does not match. w8l5 (5 levels): g2 +0.27 ± 0.12e-4 (2.3σ) -> does
  not match.
- Mixing (median rhat m_r / log size / Sérsic; slow ESS per s; s per galaxy): w8 1.23 / 1.20 / 1.24, 0.136, 78 s;
  w4 1.31 / 1.25 / 1.22, 0.169, 57 s; w8l5 1.41 / 1.34 / 1.32, 0.117, 67 s.
- Pre-declared decision: **w8 is the candidate production setting** for bright drawn-start chains (~1.75x the
  8192-step cost instead of 4x). The short warm-up was the problem, consistent with the flat D32 windows.
- `opt1.py` stopped after printing the verdict, when writing `opt1_result.json` (numpy bool not JSON
  serialisable); the match flag is now cast to `bool`, no number changed. First text kept as
  `opt1_result_17331496.txt`; rerun job 17337125 writes the JSON.

Result of `debug/rule_speed1.md` (array 17331998, 2 GPUs, 20–23 min; analysis 17331999). Speed vs e4096: 2080 Ti
e1024 x1.06, c1024 x1.12, g1024 x1.15; V100 x1.04, x1.21, x1.29. Exactness: starting log kernels differ from e4096 by
up to 4.5e-4 (e1024) and 4.0e-4 (compiled), above the 1e-4 limit; stored chains identical for a median 0–3 of 2048
steps. Pre-declared decision: **nothing adopted**. Note: even the batch size alone changes the start kernels by
~3–5e-4, so the fixed 4096 batch is what keeps runs comparable.

Next (owner decision): w8 bright drawn-start chains for all 1,720 bright units (~7 h on 4 GPUs), then the mag 21–23
long-chain test.

## 2026-10-17 (late night) — CUDA-graph density mode added; pooled geometry gives no gain

Code: a424768 `--compile-density reduce-overhead` (bare `--compile-density` unchanged): compiles the log-gap F8
density for the fixed padded batch shape and replays recorded CUDA graphs, one launch per call instead of the
~40,000 small linear/activation calls per 1,024 steps seen in the profile. Test extended to both modes and to a
refused bad value; full suite 388 passed, 3 skipped. CPU check (`debug/smoke/compile_check.py`, job 17331982,
256 real starting rows): both compiled modes repeat bitwise; max |compiled − eager| log K 2.6e-5 (`True`) and
4.8e-5 (`reduce-overhead`); eager equals the stored start kernel exactly.

`debug/rule_speed1.md` amended before any speed run: fourth arm `g1024` (batch 1024 + CUDA graphs), all arms on
`snapshot_a424768`, time limit 8 h; adopt the fastest arm that is exact enough and >10% faster than e4096 (compiled
arms also >10% faster than e1024) on both GPUs. Resubmitted as array 17331998 after Test 1, analysis 17331999 (old
17331527/17331528 cancelled unstarted).

Diagnostic (no decision): `debug/smoke/pool.sh` (array 17331971, CPU 16 cores, units k=1 and k=5 = 40 bright
galaxies, 8 levels, 8192 / 2048 warm-up, batch 256, same seeds), pooled vs per-chain morphology geometry. Paired
ESS-per-step ratio pooled/base (geometric mean over 40 galaxies, ± in log): magnitude 0.81 ± 0.11, size 1.02 ± 0.12,
Sérsic 0.95 ± 0.15, e1 0.89 ± 0.12, e2 0.85 ± 0.20. No gain; not pursued. Per-second ratios looked better but the
base k=5 run sat on a slow shared node (33 min vs ~15 min), so wall time is not comparable. Caveat: bulk ESS here is
5–170 with R-hat often >1.3, so each ratio is rough; the 40-galaxy average is the usable number.
Throughput aside: CPU-only (16 cores, batch 256) ~150–245 galaxy-steps/s per job, i.e. ~13 per core; a GPU slot
(one core + shared GPU, Test 1) ~190–220 per core. 500 extra CPU cores would add roughly one GPU's worth.

Next: speed1 after Test 1 (748/1,290 runs done at 12:20, none failed).

## 2026-10-17 (night) — mixture jumps tried and removed; bright likelihood is smooth; speed switches queued

Code (owner: optimise sampling and computing efficiency before the mag 21–23 long-chain test):
- d884b6c: opt-in `--density-batch-size` (default 4096, unchanged) and `--compile-density` (torch.compile of the
  log-gap density; refuses a non-physical-disk flow); pooled morphology allowed with tempering (pooled per level).
  Full suite 388 passed, 3 skipped (job 17331540). CPU smoke of pooled + tempering ran (job 17331530, 77 s).
- 2bb9a79, reverted in this commit: `--mixture-jump-probability` (a frozen per-galaxy, per-level Gaussian-mixture
  independence jump for magnitude / size / Sérsic, fitted at 3/4 of warm-up). Kept in `debug/snapshot_2bb9a79`.

Diagnostics (no decisions; `galsbi_pair1m_20261012/debug/`):
- `opt/jump.py` (offline, 1,080 galaxies): mixture-jump acceptance 26% median, single Gaussian 2% — an upper bound
  using a smoothed stand-in posterior.
- `smoke/jump8.sh` (array 17331691, CPU, 8 bright galaxies of unit k=1, 8 levels, 8192 / 2048 warm-up, same seeds):
  real jump acceptance at the posterior level 0.8–11.5% (median ~2.4%). Paired jump/base ratio of independent
  draws per step (4 chains): magnitude 1.11 (×/÷ 1.55), size 1.11 (×/÷ 1.28), Sérsic 2.1 (×/÷ 1.45). No gain in the
  slow magnitude / size directions, so removed.
- Why the slow directions are slow (long D32 chains, 80 galaxies; scripts kept in the job tmp only):
  other target coordinates, type and neighbour count barely pin magnitude / size / Sérsic (conditional/marginal
  sd 0.89); neighbour light within 1" is ~0.1% of the target flux and mixes ~4x faster than magnitude, so blending
  is not the cause; the three are not a thin sheet (each keeps ~60% of its spread given the other two). On accepted
  tiny steps log K changes ~15–20 per posterior sd.
- `opt/scan.py` + `opt/scan.sh` (job 17331720, CPU): exact log K along each coordinate through 32 chain states,
  everything else fixed. Smooth: cubic-fit residual 1e-4 median (90%: 0.05 / 0.01 / 0.003 for m_r / size / Sérsic).
  Single-coordinate half-width (log posterior within 0.5) 0.05–2.8 posterior sd for m_r, median 0.5–0.7 — the
  narrow direction differs between galaxies and states. Plot `SBSI/plots/galsbi_bright_likelihood_scan.png`.

Queued (pre-declared `debug/rule_speed1.md`): e4096 / e1024 / c1024 exactness and speed, array 17331527 after
Test 1 (17331485, 4 GPUs, 134/1,290 runs at ~45 min, none failed); analysis 17331528.

Next: Test 1 and speed1 results. A gradient-based move would suit a smooth, steep target but needs gradients
through the tree emulator (b) and the numpy prior — not started. More parallel chains per galaxy would fill the
padded GPU batch cheaply, but helps only if Test 1 shows a short warm-up is enough (chain count is fixed at 4 now).

## 2026-10-17 (evening) — where bright sampling time goes; longer warm-up / fewer levels test launched

Owner: before the mag 21–23 long-chain test, optimise the sampling and the computing efficiency. Read-only
diagnostics in `galsbi_pair1m_20261012/debug/opt/` (CPU jobs via `opt/cpu.sh`, profile `opt/profile.sh` job 17331447):

- `opt/window.py` (`window_result.txt`): drawn-start long chains (D32) are flat after 8192 steps — bright g2 per
  6144-step window +0.47 / +0.55 / +0.50 / +0.57e-4 (window 1 − 4: −0.10 ± 0.14e-4) — while the normal run (sampling
  steps 2048–8192) is +0.58 ± 0.16e-4 above window 4. Measurement-start long chains still drift (window 1 − 4
  +0.18 ± 0.13, 2 − 4 +0.19 ± 0.13) and the start gap in window 4 is +0.33 ± 0.14e-4. So the short drawn-start run is
  mostly a too-short warm-up.
- `opt/geometry.py` (`geometry_result.txt`, every 8th D32 unit, 1,080 galaxies): integrated autocorrelation of
  magnitude / size / Sérsic ~743 / 666 / 974 steps (median). Step shape vs posterior: variance ratio along the
  narrowest posterior direction 0.015 (median; ideal ~1.9 for a 3-d block), widest 0.29 — steps are 3–11x too short
  (in sd) because the per-chain covariance is learned from a short window. The valley is bent (median R² of a
  quadratic bend 0.28, 48% above 0.3). Chain means spread by 0.33 posterior sd.
- Profile (A40, one unit, 1024 steps): the GPU is the limit; F8 flow inverse ~43% of a step, transport ~14%,
  C8 ~7%, emulator ~6%, summaries ~8%. Every step evaluates 20 galaxies x 4 chains x 8 levels; level swaps are
  accepted 45–55%, more than needed.

Test 1 (no code change), pre-declared `debug/rule_opt1.md`: all 430 brightlong units (8,582 galaxies), drawn start,
snapshot_d776abb, same seeds as D8/D32; arms `w8` (14336 steps / 8192 warm-up), `w4` (10240 / 4096), `w8l5`
(14336 / 8192, 5 levels = every second rung). Reference D32 steps 8192–32768; control D8. An arm matches if g2 and g1
differences to the reference are both within 2σ; the cheapest matching arm is the candidate (5 levels only if its
slow-quantity ESS per second is not below w8); if D8 also matches nothing is adopted. Launcher `debug/opt1/o1.sh`
(array 17331485, 4 GPUs, 30 slots each, 1,290 runs), analysis `debug/opt1.py` via job 17331496 (afterany; refuses
until all runs are done). Rule text corrected before any result: the long runs keep 384 draws per chain (stride
64), not 1,536.

Next: code track — step shapes pooled over the 4 chains and a longer window, compatible with tempering; a compiled
F8 flow with an exactness check; each with its own pre-declared test.

## 2026-10-17 — bright sampler switches: none speeds up the slow directions; bigger start pool passes by a hair

Result of `galsbi_pair1m_20261012/debug/rule_samplertest.md` (array 17315027, 756 runs, none failed, ~2.6 h on 4 GPUs;
analysis 17315066, `debug/samplertest.py` -> `samplertest_result.txt/.json`). 108 bright units, 2,160 galaxies.

- Between-start agreement A (ln 8-chain rhat vs the current sampler) and speed B (ln independent draws per second):
  joint shape A +0.0084 ± 0.0022, B x0.92; joint environment A +0.0028 ± 0.0021, B x0.84; 4x start pool
  A −0.0131 ± 0.0027, B x1.01. Pre-declared decision: **the bigger start pool helps (only one to pass)**.
- But the gain is tiny (median between-start rhat 1.387 -> 1.365) and no arm moves the slow quantities: median ESS of
  magnitude / size / Sérsic stay at 10–19 / 13–25 / 14–17 at 8192 steps. The sampler already steps jointly in these
  three; the remaining slowness likely needs a different kind of move (untested guess: the valley is curved).
- g2 (reported only, too few galaxies): start gaps all within 1.5σ of 0 (base −0.02 ± 0.08e-4); rerun of the
  current sampler vs the earlier drawn-start runs −0.06 ± 0.04e-4 (same code and seeds, other GPUs).
- Next (owner decision): long chains for the bright set (the start pool can be added for ~4% cost), or new sampler
  code aimed at the magnitude–size–Sérsic valley.

## 2026-10-16 (night) — bright sampler switches tested before longer chains (launched)

Owner: explore a better bright sampling algorithm before longer chains. Read-only diagnosis (1/16 of the brightlong
subset): slowest are Sérsic index, magnitude and size (median bulk ESS 16 / 19 / 26 at 8192 steps), then e1/e2 (64–72);
the drawn start also leaves crowding/b slow. Pre-declared `galsbi_pair1m_20261012/debug/rule_samplertest.md`:
108 bright units (every 4th of the brightlong subset, 2,160 galaxies), the step-2 command (snapshot_d776abb, same
seeds) with one existing switch per arm: `--joint-shape`, `--joint-environment`, `--initial-bank-draws 8192` (drawn
start only), and the unchanged sampler rerun in the same jobs. Pooled morphology and ensemble moves are refused by
the code with tempering, so not tested. Decision: a switch helps if the two starts agree better (8-chain rhat over
e1, e2, magnitude, size, Sérsic; > 2σ) without being slower per independent draw (> 2σ). Scores reported only
(1/16 of the bright set is too small to resolve the start gap).

- Launcher `debug/samplertest/st.sh` (array 17315027, 4 GPUs, 30 one-core slots, arms interleaved per unit),
  analysis `debug/samplertest.py` via job 17315066 (refuses until all 756 runs are done). Chain reading checked
  against the sampler's own rhat on one unit (agree; size differs at 1e-2 because a different size variable is used).
- Startup: first task (2080 Ti) has all 30 slots sampling, every arm included, no failures; ~4 steps/s per slot,
  ~35 min per run; ~4 h with all 4 tasks running (3 still queued at launch).

## 2026-10-16 (evening) — 4x longer bright chains: bright g2 falls from both starts; the start gap is not resolved

Result of `galsbi_pair1m_20261012/debug/rule_brightlong.md` (array 17284374, ~8 h on 4 GPUs; one run tripped the
sampler's 1e-6 invariant audit by 1.42e-6 (GPU rounding) and passed on rerun 17303991; analysis 17293385,
`debug/brightlong.py` -> `brightlong_result.txt/.json`). Quarter subset: 430 units, 8,582 mag < 21 galaxies.

- Bright g2 contribution (1e-4): measurement start 8192 / 32768 steps +0.78 / +0.33; drawn start +1.15 / +0.52.
- Start gap: +0.37 ± 0.15 (short) vs +0.20 ± 0.08 (long); change +0.17 ± 0.15 -> pre-declared **(c)**.
- Not in the rule's cases: longer chains lower g2 from both starts, −0.46 ± 0.08 (5.8σ) and −0.63 ± 0.14 (4.5σ);
  rhat median 1.22 -> 1.08; drawn-start long chains have equal halves (+0.03 ± 0.10), measurement-start long chains
  still drift (−0.21 ± 0.10). g1 moves ≤ 0.15% ± 0.07% of m.
- Reading: 8192-step bright chains overstate g2 whatever the start; chain length matters. Rough scaling to all bright
  galaxies (not a 1M result): about −2e-4 on the 1M c2, leaving roughly +5–6.5e-4.
- Limitations: the rule's power estimate was wrong (subset gap ~2.5σ, not 3.7σ); mag 21–23 chains untested.
- Next (owner decision): long drawn-start chains for all bright galaxies (~4x the step-2 cost, ~16 h on 4 GPUs) and
  a look at the mag 21–23 chains and the faint part for the rest of c2.

## 2026-10-16 — 1M pair: 4x longer bright chains from both starts launched (subset)

Owner: "run the longer bright chains from both starts on a subset". Pre-declared
`galsbi_pair1m_20261012/debug/rule_brightlong.md`: every 4th step-2 unit (430 units, 8,582 mag < 21 galaxies), the
step-2 bright command with `--steps 32768 --warmup 8192` from the measurement start and from the drawn start (same
seeds as the short runs, snapshot_d776abb), comoving stride 64 (same 384 kept draws per chain). Launcher
`debug/brightlong/bl.sh` (array 17284374, 4 GPUs, 30 one-core slots, 320G), analysis `debug/brightlong.py`
(refuses until all 860 runs are done). Decision on g2: do the gaps between starts (2.0e-4 at 8192 steps) close?

- Startup: all 4 jobs running, both starts sampling with the intended settings; ~9 steps/s per slot, so ~1 h per
  unit and ~8 h in all; 45 GB per job at start.

## 2026-10-16 — 1M pair, bright drawn starts: m −0.10% ± 0.88% passes, c2 +0.00090 ± 0.00019 fails; bright g2 depends on the start

Result of `galsbi_pair1m_20261012/debug/rule_brightdraw.md` (GPU array 17208117: 1720/1720 units, 0 failed, ~4 h on
4 GPUs; CPU analysis 17208623 via `debug/brightdraw/analysis.sh`).

- 1M pair with bright and faint drawn starts: **m = −0.10% ± 0.88%** (pass), **c2 = +0.000903 ± 0.000192** (4.7σ,
  fail; with measurement-start bright chains +0.000702 ± 0.000193).
- New bright chains still drift: late − early half D2 = −0.63 ± 0.27e-4 -> (b) start memory remains.
- Same galaxies, drawn − measurement start: **Δg2 = +2.01 ± 0.27e-4** (7σ; expected −0.4 to −0.8e-4), Δg1 =
  +0.33% ± 0.12% of m. So the measurement start does not explain the bright part of c2; the drawn start raises it.
- Reading: the bright g2 score depends on the start by ~2e-4 (both runs' late halves still differ by ~2.1e-4), so
  the 8192-step bright chains are not converged in g2. Drawn starts mix worse (rhat median 1.43 vs 1.22; > 1.1 for
  97% vs 84%); only ~2 useful bank states per chain at the start.
- Limitation: mag 21–23 chains keep their measurement starts (their own D2 −1.21 ± 0.33e-4).
- Next (owner decision): longer bright chains from both starts on a subset, to see whether they meet; the
  1M c2 cannot be attributed until they do.

## 2026-10-15 (night) — 1M pair: bright second step rerun with drawn starts launched

Pre-declared `galsbi_pair1m_20261012/debug/rule_brightdraw.md`: the step-2 bright run (34,338 mag < 21 galaxies,
same seeds, shear and settings) with `--initialization prior-draw`, code `galsbi_faintg05_20261013/snapshot_d776abb`
(library identical to step 2). Launcher `debug/brightdraw/bd.sh`, array 17208117 (4 GPUs, `inter,cip`, 30 one-core
slots each); analysis `step2.py debug/brightdraw` (with `PERGAL=_faintdraw`), `brightdraw_diff.py`, `g2time.py` (BD).

- Startup check (first 25 units, 444 galaxies): no failures; ~11 min sampling per unit. Only ~2 useful bank draws
  per chain at the start (median 1.9). Chains disagree more than in step 2: largest rhat per galaxy median 1.44
  (> 1.1 for 98%) vs 1.22 (83%). The pre-declared stop rule (twice the step-2 fraction) cannot trigger, so the run
  continues; recorded in the rule file before any science number.
- Next: when all 1720 units are done, the three measurements of the rule.

## 2026-10-15 (evening) — 1M c2: bright chains remember their start in g2; faint shows no sign

Owner: "work on this autonomously". Pre-declared `galsbi_pair1m_20261012/debug/rule_g2time.md`: split every chain's
kept draws into early and late halves (no new sampling); `g2time_extract.py` / `g2time.sh` (job 17208073, CPU,
6 tasks, < 15 min) and `g2time.py` -> `g2time_result.txt/.json`. c2 per half = I^-1 sum of raw scores (production I).

- D2 = c2(late) − c2(early), in 1e-4 (the 1M c2 is +7.0 ± 1.9): faint drawn start (current inputs) −1.6 ± 1.9;
  **bright step 2, mag < 21: −0.79 ± 0.22 (3.6σ)**; it0 tempered mag 21–23 (also current inputs): −1.21 ± 0.33
  (3.7σ); old it0 faint −0.7 ± 1.8; old it0 bright < 21 −0.68 ± 0.20.
- Pre-declared decision: **(a) start memory in g2** for the bright chains. The measurement-centred start pushes c2 up.
- Size: if the late halves were converged, the bright chains overstate c2 by about +1.0 ± 0.2e-4 (< 21 and 21–23
  together), a part of +7.0 ± 1.9e-4; more if the late half still remembers. Faint is not resolved (2σ bound 5.4e-4).
- Limitation: the rule named only faint and bright < 21 as deciding; the 21–23 tempered chains are current inputs too
  (noted after the result). D1 is the raw-score change only, not the information-corrected deficit of `timesplit.py`.
- Next: bright chains with drawn starts (`--initialization prior-draw`, works with tempering), checked on a unit
  first because a sharp bright likelihood may leave few useful bank draws.

## 2026-10-15 — Mirror test: F8/C8 are not left-right symmetric; this explains the g2 tilt but not the 1M c2

Owner: "run the mirror test on the bright galaxies first"; "while waiting, check how the sampler handles the e2 tilt".
Pre-declared `galsbi_pair1m_20261012/debug/rule_mirror.md` (one amendment after the 2-chunk smoke run: band membership
by the g = 0 draw; 8000 chunks). `mirror.py` / `mirror.sh` (job 17200347, one GPU, 5 h 41 min) -> `mirror_result.json`.
New scenes from the mock's prior; each scene and its mirror (e2 -> -e2), mirrored base draws.

- The model is not mirror symmetric: mean measured e2 of mirror pairs (exactly 0 if symmetric) is −0.00234 ± 0.00002
  over all galaxies, −0.0040 ± 0.0001 for mag < 21; C8's g2 slope is nonzero in every band; bright per-galaxy kernel
  score asymmetry −2.8 ± 0.1. F8 stores shape-coordinate means (−0.0002, +0.0026).
- Code check passes: the model alone gives dlogZ/dg2 = 0.0048 ± 0.0005 (selection runs 0.0043 ± 0.0004) and the
  mock's tilt (−0.0020 / −0.0028). So the frame-fixed g2 tilt is a property of the trained F8/C8.
- Correction to the c2 split: the bright band is not selection-free (own slope +0.020 ± 0.007), so the bright excess
  is +0.042 ± 0.017 per galaxy (2.4σ), not 3σ. The 1M c2 itself stays: the scores imply 0.0079 per galaxy vs 0.0043–0.0048.
- Sampler survey (read-only): no step assumes mirror symmetry; proposals, swaps and caches exact; comoving_circ keeps b
  frozen (second order). The only approximation that could matter is start memory of the measurement-centred start.
- Limitation: band-wise expected scores ignore band-edge crossing; a model asymmetry alone cannot bias a same-model mock.
- Next (owner decision): posterior intrinsic e2 means over all bright/faint units vs the truth (read-only), or
  the faint chains' start memory in g2.

## 2026-10-15 — Where the 1M c2 comes from: a g2 tilt fixed to the image frame, and the scores over-correct it

Owner: "break down the 1M c2 first". Pre-declared `galsbi_pair1m_20261012/debug/rule_c2split.md`; `c2split.py`
(job 17200258, CPU, no new sampling; reproduces the step2 c2 of both input sets exactly) -> `c2split_result.txt/.json`;
`meane.py` (mean measured shapes, written after the result).

- The paired c2 = +0.00070 is the difference of two bigger pieces: galaxy scores +0.00154 (spread over all bands:
  < 21 +0.00021 ± 0.00005, 21–23 +0.00006 ± 0.00006, 23–24 +0.00041 ± 0.00012, 24–25 +0.00042 ± 0.00009,
  ≥ 25 +0.00045 ± 0.00007) and the selection term −0.00084 ± 0.00007.
- Selection gradient in g2 is clearly nonzero: dlogZ/dg2 = 0.0043 ± 0.0004 (both sides, 7 of 8 replicas), while
  g1's is 0.0006 ± 0.0004. Mean measured e2 of the mock galaxies is −0.0028 ± 0.0004 (original) and −0.0020 ± 0.0004
  (twin, same sign although the twin is turned 90°) -> a g2 tilt fixed to the image frame, with a round PSF.
- Bright galaxies (mag < 21, nearly free of selection) alone point to g2 = +0.0018 ± 0.0006 (own g1 0.0203 ± 0.0008);
  the existing bright run at the truth gives the same (+0.0017 ± 0.0012, unpaired error).
- Pre-declared reading: (a) and (b) both met — scores and selection disagree about the g2 tilt, and the bright
  band shows it by itself. Not one band; nothing corrected.
- Next (owner decision): find where the frame-fixed g2 tilt enters (mirror test of the disk emulator / classifier /
  flow: flip x and compare e2), and whether the mock generator and the likelihood treat it the same way.

## 2026-10-15 — g2 in the existing g = (0.05, 0) truth runs: no sign, but too few galaxies to test the 1M c2

Owner: "check g2 in the existing g05 truth runs first". Pre-declared `.../galsbi_faintg05_20261013/rule_g2.md`;
`g2check.py` -> `g2check_result.txt` / `.json` (no new sampling; reproduces every run's m exactly).

- 23–25, same 82,677 galaxies at the truth: c2 = production −0.00093 ± 0.00096, long −0.00034 ± 0.00088,
  drawn start 4096 −0.00092 ± 0.00110, 2048 −0.00008 ± 0.00122, bigger pool −0.00140 ± 0.00118.
  Paired drawn start − production +0.00012 ± 0.00087; long − production +0.00060 ± 0.00067.
- Reading (pre-declared): **(b) no sign at this precision**. Errors are 3–6x the 1M effect (faint drawn-start g2
  shift +0.00030 ± 0.00015, total c2 +0.00070 ± 0.00019), so this neither confirms nor clears it.
- Next (owner decision): break the 1M c2 down with data already in hand (bright / faint bands / selection term,
  per side), or a dedicated g2 truth test with ~1M faint galaxies.

## 2026-10-15 (night) — 1M pair with drawn faint starts: m = −0.43% ± 0.89% passes, c2 = +0.00070 ± 0.00019 fails

Result of `.../galsbi_pair1m_20261012/debug/rule_faintdraw.md` (`step2_result_faintdraw.json`,
`faintdraw_diff_result.txt`).

- Run: all 5612 faint units (array 17190708, 4 GPUs, 14:30–23:10, none failed); `faintdraw.py` found exactly the
  it0 faint galaxies on both sides (889,208 / 889,298). Estimate (`PERGAL=_faintdraw step2.py`, job 17195372).
- Paired, bright second step + faint drawn starts: **m = −0.43% ± 0.89%** (was −1.68% ± 0.82%),
  **c2 = +0.000702 ± 0.000193** (was +0.000415 ± 0.000178). Pre-declared test: **fails on c2 (3.6σ)**; m passes.
  One step with the new faint values: m = −2.07% ± 0.88%, c2 = +0.000653 ± 0.000194. Per side (no test):
  original −5.28% ± 2.35%, twin +4.40% ± 2.36%.
- Why m moved (diagnostic written after the result, same galaxies, no selection term): the drawn start lowers
  the faint score sum (−5.06% ± 0.72% of m at fixed information) and lowers the faint information estimate more
  (I11 −8.6%: sum s² −4.9%, chain-noise correction +24%). Net +1.2% of m. Each faint band's own one-step g1 moves
  toward 0.02: 23–24 0.01948 → 0.01999, 24–25 0.01938 → 0.01979, 25+ unchanged 0.02007. Consistent with the
  faint-g05 finding (best-fit starts make short faint chains overconfident), but here the effect enters mostly
  through the information's chain-noise correction, which is large for 23–24 (28% of sum s²).
- c2: the faint g2 score sum moved by +0.000301 ± 0.000149 (23–24 +0.000187 ± 0.000130, 24–25
  +0.000129 ± 0.000068). The old c2 already failed narrowly (2.3σ). Cause not investigated.
- Code: `debug/faintdraw.py`, `debug/faintdraw_diff.py`, `step2.py` (`PERGAL`; default output unchanged apart from
  its description text, rerun job 17190959).
- Next (owner decision): look for the source of c2 (g2 offset) — e.g. the same truth-closure test at g = (0, 0.05)
  or a g2 check in the existing faint-g05 runs; the m fix is in hand.

## 2026-10-14 (late afternoon) — 1M pair: faint part rerun with drawn starts launched

Owner: "rerun the 1M pair faint part with drawn starts on 4 GPUs". Pre-declared
`.../galsbi_pair1m_20261012/debug/rule_faintdraw.md`.

- Run: all 2 x 2806 faint units, the it0 faint command of `worker.sh` with the same per-unit seeds plus
  `--initialization prior-draw` (code `snapshot_d776abb`), launcher `debug/faintdraw/fd.sh`, array 17190708
  (4 GPUs, `inter`/`cip`, 30 one-core slots each). Seeds, galaxy lists and settings checked against the it0
  protocols (first, middle and last faint unit of each side).
- Analysis: `debug/faintdraw.py <side>` swaps the faint galaxies' inputs into `pergal_<side>_faintdraw.npz`
  (refuses unless the faint galaxies are exactly those of it0; test mode on 6 finished units: 1901 galaxies found,
  none extra); `PERGAL=_faintdraw step2.py` gives the primary number (bright second step + faint drawn starts).
  `step2.py`'s default output is unchanged (rerun: result identical apart from its description text).
- Expected about 5–8 h; the result goes in a new entry.

## 2026-10-14 (evening) — bigger start pool: 2048-step faint chains pass the rule only by cancellation between bands

Owner: "try a bigger candidate pool first". Study dir `.../galsbi_faintg05_20261013/` (`rule_bank.md`,
`banktest/`, `draw.sh` with `BANK`, `draw.py bank8192`, `bank8192_result.txt` / `.json`).

- Code (d776abb, `scripts/check_galsbi_local_sampler.py`): `--initial-bank-draws N` sets the prior-bank states per
  chain for a fresh start bank (default: the reference 2048). Check job 17189684 (unit original_u0000, 2048 steps):
  default reproduces `draw2048` bitwise (1260 arrays). Pool size vs one-unit cost (one process, A40):

  | per chain | start pick | effective draws median | below 5 | total |
  |---|---|---|---|---|
  | 2048 | 32 s | 7.6 | 37% | 4:34 |
  | 8192 | 126 s | 20.3 | 16% | 6:09 |
  | 16384 | 241 s | 31.7 | 13% | 7:45 |

  Effective draws grow much slower than the pool (heavy-tailed weights); pick time grows in proportion. 8192 was
  chosen (pre-declared): 2048 steps + 8192 pool costs about the same as 4096 steps + 2048 pool, so no speed-up was
  possible at this pool size. In production (30 slots per GPU) the pick took ~324 s per unit vs ~93 s.
- Run `bank8192` (array 17189997, 262 units, none failed; same seeds as draw2048). 23–25, 82,677 galaxies at the
  truth: **m_band = −2.37% ± 2.09%**; D_short +4.00 ± 1.79, D_long −0.13 ± 1.55, D_el −1.58 ± 2.34;
  vs draw2048 (same seeds, small pool) +1.81 ± 1.94. Effective draws median 19.9, 17.6% below 5.
  Pre-declared decision: **(a)**.
- But the sub-bands disagree with the long run in opposite directions: 23–24 +3.01% ± 3.71% (D_long +5.04 ± 2.84),
  24–25 −6.67% ± 2.37% (D_long −4.92 ± 1.42, 3.5σ). The 23–25 agreement is a cancellation. The faintest band at 2048
  steps is low with both pools (draw2048 D_long −5.87 ± 1.44), and the pool size itself changed little
  (+1.8 ± 1.9 vs the small pool). D_short passes 2σ only narrowly (2.2σ; small pool 1.2σ).
- Reading: 2048-step faint chains are not shown to work, with or without the bigger pool; a bigger pool costs as
  much as doubling the steps. The 4096-step drawn start (`rule_draw.md`, (a)) remains the best option, with its
  own 24–25 caveat (D_long −2.69 ± 1.19, 2.3σ).
- Next (owner decision): faint production with `--initialization prior-draw` at 4096 steps and the default pool,
  e.g. rerun the faint part of the 1M pair to see whether its −1.8% ± 0.6% faint shortfall goes away.

## 2026-10-14 (afternoon) — sampler memory fix; likelihood-weighted starts make 4096-step faint chains match 4× longer ones

Owner: "fix the memory issue. and then we should try to improve the sampling from beginning (to see if shorter
chains can deliver good convergence). note: occam's razor." Study dir `.../galsbi_faintg05_20261013/`
(`rule_draw.md`, `draw.sh`, `draw.py`, `draw_result.txt` / `.json`, check job in `memtest/`).

- Code (400ef6a, `scripts/check_galsbi_local_sampler.py`):
  - Memory: with `--no-chain` the per-step state history (trace, counts, types, log-prior, summaries, kernels) is
    now kept only at the profile draws (after warm-up, every `--profile-stride` steps) instead of every step.
    Protocol field `diagnostic_draws` says which. Acceptance/move counters stay full.
  - `--initialization prior-draw`: same per-chain prior bank as `prior-bank`, but the start is drawn with probability
    proportional to the likelihood instead of taking the best bank state (importance resampling, seed stream
    `[seeds, 992]`). `initialization.json` records each chain's effective number of bank draws. Nothing else changed.
- Check (job 17188270, unit original_u0000, 2048 steps, measurement start): old code (60bf987 snapshot) vs new code,
  same arguments: **all 1260 profile arrays bitwise identical**; peak memory 2.90 GB → 2.11 GB (the saving grows
  with chain length; the 16384-step run had reached ~4–5 GB per process). `prior-draw` run: start replay error 0.0.
- Start test (pre-declared `rule_draw.md`; snapshot `snapshot_400ef6a`; arrays 17188290 and 17188292, three GPUs
  at a time, pending tasks moved to `inter,cip`; 262 + 262 units, none failed): same 82,677 faint (23–25, in cuts)
  galaxies at the truth (0.05, 0) as the long-chain test.

  | 23–25 | m_band | vs production 4096 (D_short) | vs long 16384 (D_long) | late − early (D_el) |
  |---|---|---|---|---|
  | production, best start, 4096 | −6.12% ± 1.70% | | | |
  | long, best start, 16384 | −2.06% ± 1.49% | | | |
  | **drawn start, 4096** | **−2.42% ± 1.95%** | +3.96 ± 1.66 | −0.16 ± 1.39 | +0.11 ± 2.12 |
  | drawn start, 2048 | −4.70% ± 2.17% | +2.19 ± 1.80 | −1.94 ± 1.56 | +3.15 ± 2.34 |

  Decisions: **draw4096 (a)**, drawn starts work at 4096 steps (same answer as 4× longer chains, no start memory
  left inside the run); **draw2048 (b)**, no help (still remembers its start, D_el +3.2 ± 2.3).
  Sub-bands (reported alongside, not part of the decision): 23–24 draw4096 −0.15% ± 3.44% (D_long +2.50 ± 2.59);
  24–25 draw4096 −4.29% ± 2.24% (D_long −2.69 ± 1.19, 2.3σ); 24–25 draw2048 −7.85% ± 2.41% (D_long −5.87 ± 1.44).
  Effective bank draws per chain: median 7.8, 35.5% of chains below 5.
- Limitations: the long run and draw4096 are both ~1σ from zero, so this shows "matches 4× longer chains", not a
  sub-percent faint result. With ~8 effective bank draws a drawn start is only an approximate posterior draw; the
  faintest band at 2048 steps is pulled the other way. A bigger start bank was not tried (no retuning).
- Next (owner decision): use `--initialization prior-draw` at 4096 steps for faint production (same cost as now),
  e.g. rerun the faint part of the 1M pair; optionally test a larger start bank before going below 4096 steps.

## 2026-10-14 (early morning) — long-chain test: the faint bias is chain length (long − short +4.1% ± 1.3%)

Result of `.../galsbi_faintg05_20261013/rule_long.md` (`long.py`, `long_result.txt` / `.json`).

- Run: the first array (17185664, 110G) ran out of memory: the sampler keeps every step's state in memory even with
  `--no-chain` (it only skips writing it), so each process grows to ~4–5 GB at 16384 steps, and turning the lists
  into arrays at the end roughly doubles that for a moment. Unclaimed units were marked `held`, the array was
  cancelled, and `long2.sh` (same settings and seeds, 220G, pass number `PASS`, claims/logs in `long/claims$P`,
  `long/logs$P`) reran everything not done: pass 2 (17185687) finished 228 of 262 (34 killed when many slots ended
  together), pass 3 (17185985) the last 34. All 262 units done 04:50; procedure note appended to `rule_long.md`.
  `long.py` needed `sys.modules['closure'] = c` so its worker pool can pickle `closure.one`.
- Same 82,677 galaxies (23–25, in cuts) at the truth (0.05, 0):
  production chains (4096 steps) m_band = −6.12% ± 1.70%; 4× longer chains −2.06% ± 1.49%.
  23–24: −7.56 ± 2.75 → −2.76 ± 2.30; 24–25: −4.83 ± 2.13 → −1.40 ± 2.01.
  Paired difference (same galaxies, short-run information) **dM = +4.13% ± 1.29%** (23–24 +4.88 ± 2.36,
  24–25 +3.49 ± 1.19). Chain noise E/galaxy 0.95 → 0.34.
- Pre-declared decision: **(a), chain length alone** (dM 3.2σ; long run within 2σ of 0).
- Caveat: inside the long run the early half still gives −3.05% and the late half −1.06%, so 16384 steps still
  remember the start a little. The start is `--initialization measurement` (best candidate of each chain's bank).
- Reading for the 1M pair: its faint 23–25 shortfall (−1.8% ± 0.6% of m at g = 0.02) is expected to be mostly the
  same short-chain effect. Not rerun here.
- Next (owner decision): longer or better-started faint chains (4× steps costs ~4× faint sampling time), and/or
  a sampler memory fix (do not keep per-step traces under `--no-chain`) before any long production run.

## 2026-10-14 (night) — faint closure test: the faint score is biased at the truth (−6.6% ± 1.1%); long-chain test launched

Result of `.../galsbi_faintg05_20261013/rule.md` (`closure.py`, `closure_result.txt` / `.json`).

- Run: all 524 units and 8 × 400 selection chunks done, nothing failed. Array 17185329 (task 3 started late and
  computed selection replicas 6–7). Sampling took 1.6 h, selection ran until 00:45.
- 23–25, 165,351 galaxies at the truth (0.05, 0): **m_band = (I⁻¹S)₁/g = −6.61% ± 1.09%** (stat 0.74, chains 0.77,
  selection 0.24); S/(I g) = −0.066; g2 offset −0.58% ± 1.26% of g. I/galaxy 6.24, chain E/galaxy 0.95.
  23–24: −7.95% ± 1.79%; 24–25: −5.34% ± 1.37%. Band selection term dlogZ/dg1 = +0.0073 ± 0.0007.
- Pre-declared decision: **(c)**. It is 6σ from 0 (so not (b)), but 2.9σ beyond the −3.4% of the 1M one-step (so
  not (a)). Plain reading: the faint score is biased (its root is off), more strongly at 0.05 than the 0.02 shortfall.
  The single step is not the cause.
- Addendum diagnostic (written before results): early half of the kept draws −8.18%, late half −4.93%,
  D = +3.52% ± 1.39% (23–24 +4.6 ± 2.5, 24–25 +2.5 ± 1.3). The bias shrinks along the chain, which points to
  chains that start at the most likely state (`--initialization measurement`) and do not move far enough.
- Follow-up `rule_long.md` (written before the run): every second unit (262, ~82,700 galaxies) again at the truth
  with 4× longer chains (16384 steps / 4096 warm-up / stride 16, same 768 kept draws), same starting points (bank
  seed unchanged), new sampler seed 2028407001 + k; `long.sh` (array 17185664, 4 GPUs), `long.py` (dM = long −
  short on the same galaxies; decision (a) dM > 2σ, (b) dM + 2σ < 3%, (c) otherwise).

## 2026-10-13 (late night) — faint 23–25 shortfall: not the single step; closure test at g = 0.05 launched

Owner: "look for the faint bias then, skip the second step", then "dig into this autonomously". Files in
`.../galsbi_pair1m_20261012/debug/` unless stated.

- BFD reading (Bernstein & Armstrong 2014; Bernstein et al. 2016): one Newton step is enough there because the
  error is α g² with α ≈ 0.5–2. Our mag < 21 galaxies break that (α ≈ −80; fixed by the bright second step), but
  for the faint ones the question was whether their shape prior bends the step.
- Exact toy (`rule_toyexact.md` written first; `toy_exact.py`, job 17184710, ~25 min; `toy_exact.json`): GalSBI
  red/blue |e| priors, Möbius shear, Gaussian noise 0.05–0.45, measured |e| < 0.99 cut, rotated twins, all
  integrals by fixed quadrature (no random integration; converged to < 4e-13). The old `toy_faint.py` is invalid at
  noise ≥ 0.1 (its random selection term is off by up to 8σ). Secant slope / I at noise 0.05/0.1/0.2/0.3/0.45:
  red 1.004/1.012/1.021/1.003/0.999 (± 0.003–0.012), blue 0.995–1.001 (± 0.002–0.003). Validity 2 failed in one of
  10 cases (blue 0.05: score at the truth +0.0051 ± 0.0020, just over the 0.005 threshold); reported, not retuned.
- Red share of the GalSBI prior by true r magnitude (`redfrac.py`, 16M draws): 20–20.5 0.46, 23–24 ≈ 0.11,
  24–25 ≈ 0.065, 25–26 ≈ 0.04. Toy prediction for the mock's 23–25 one-step ratio ≈ 1.00; observed
  0.966 ± 0.010. So the faint shortfall is not the single step (in the toy's terms): a biased score, or chance.
- Same pattern in earlier, independent mocks (`oldmocks.py`, truth used, diagnosis only): 2 × 100k rotated-twin
  pair 23–25 one-step ratio 0.961, −2.13% ± 1.82% of its m (1M: −1.82% ± 0.56%).
  Bins 0–21 0.969, 21–23 1.004, 23–24 1.016, 24–25 0.912, 25–26 1.018. Unpaired 500k (job 17185186; `side_cut()`
  accepts its one unsampled row 431089, which is outside the radius cut; reproduces m = −4.09%): 23–25 ratio 0.906,
  −5.11% ± 2.32% of its m; bins 0–21 1.031, 21–23 1.041, 23–24 0.962, 24–25 0.855, 25–26 0.990. Three independent
  mocks, inverse-variance mean of the 23–25 share: −2.0% ± 0.5% of m (about 4σ); 24–25 is the shortest bin in all
  three (0.963, 0.912, 0.855).
- Already ruled out (earlier entries): averaging over 1/2/4 chains, hidden chain noise in I (seed rerun:
  −0.16% ± 0.61%), a pull from the starting point (time split), selection-term noise (0.33% of m, in the error).
- Closure test at a larger shear, `.../galsbi_faintg05_20261013/` (`rule.md` written before the mock existed). At
  0.02 the faint shortfall is 0.004 per galaxy against a noise of 1.2, so testing the score at the truth would need
  ~1M faint galaxies. If the root sits at c × truth, the signal grows with the shear, so at 0.05 about 1/6 suffices.
  - Code (60bf987): `scripts/prepare_galsbi_population.py --shear g1 g2` sets the true shear (default unchanged: the
    parent's). Snapshot `source_snapshot/` = `git archive 60bf987` with the usual manifest.
  - Mock (`prepare.sh`, job 17185272, met-cl-vis01, 9 min): 160,000 original galaxies at (0.05, 0), seed
    2028101301, antithetic twin 160,067 (replay 0.0).
  - `make_units.py`: every 23 ≤ mag < 25 galaxy of both sides, 82,770 + 82,782 in 524 units; plan at the truth,
    selection limits 23/24/25/25.8 so each band has its own Z (8 replicas × 400 chunks, seed base 2028307001).
  - `worker.sh` (array 17185329, 4 GPUs on inter, 30 one-core slots + 2 six-core selection streams per GPU),
    production faint command at `--shear 0.05 0`, seeds 2028207001 + k / 2033207001 + k.
  - `closure.py`: m_band = (I⁻¹ S)₁ / 0.05 for 23–25, errors as `pair_step.py` (scenes, chains, selection).
    Decision fixed in `rule.md`: (a) biased score if m_band < −2σ and within 2σ of −3.4%; (b) score fine if
    |m_band| ≤ 2σ and m_band > −3.4% + 2σ; (c) otherwise report as is.

## 2026-10-13 (night) — speed benchmark for a faint second step

Owner asked how long a faint second step would take on 4 GPUs. `debug/bench/bench_faint.sh` (job 17184055,
met-cl-vis01 / 2080 Ti, 30 one-core slots, step-2 snapshot 27d244f, the first 30 original faint units of 317
galaxies, 1024 steps / 256 warm-up; no science output) and `bench_faint_report.py`:

- Shear 0 (as the first step): 418 s for the 1024-step run; projected full unit (4096 steps) 1,158 s.
- Shear (0.01936, 0.00037): 550 s; projected full unit 1,498 s (1.29× slower: the disk emulator now runs every
  step and set-up doubles, 122 → 245 s). GPU busy 62%.
- All 5,612 faint units (both sides) on 4 such GPUs: ~19.5 h (balanced load, no tail). First-step faint units
  ran ~2× faster per unit on H100/H200/A40 than on the 2080 Ti, so faster cards could give roughly 10–15 h
  (not benchmarked). Mag 21–23 galaxies (tempered bright units) are not included; at the bright step-2 rate they
  would add ~25 h on 4 GPUs.

## 2026-10-13 (evening) — 1M pair, bright-only second Newton step: m = −1.68% ± 0.82%, still fails

Result of `rule_step2.md` (`step2.py`, CPU job 17182222, 24 min; `debug/step2_result.txt` / `.json`).

- Run: all 1,720 units done, 0 failed; the four GPU jobs (17178786_1–3, 17178859) took 4.9–5.0 h each (truth run
  with the old layout: 9.0 h). Chains kept: `debug/step2/` is 71 GB.
- Paired, all 1,990,072 galaxies (34,338 resampled at g1 = (0.01936, 0.00037)): g = (0.019664 ± 0.000163,
  0.000415 ± 0.000178) → **m = −1.68% ± 0.82%, c2 = +0.000415 ± 0.000178. Pre-declared test: FAIL** (|m| 2.05σ,
  |c2| 2.33σ). One step was m = −3.22% ± 0.81%, c2 = +0.000370 ± 0.000178. The step moved m by +1.54%, close to
  the expected +1.4% (bright mag < 21 deficit) and to the code test from samples at the truth (−1.83%).
- Bright galaxies at g1: summed g1 score S_B(g1) = +1,002 ± 1,105 (zero within 1σ; the straight line from g = 0
  predicted −2,531). Secant slope / I_B = 0.825 (original 0.931, twin 0.720). So the bright part has converged:
  a third step would move g1 by about S_B(g1)/I = 0.00009 ± 0.0001, i.e. ≤ 0.5% ± 0.5% of m.
- Not applied (rule): the bright selection-curvature correction would shift g1 by −0.000022 ± 0.000022
  (−0.11% ± 0.11% of m).
- Per side (no test): original m = −5.89% ± 2.25%, twin +2.51% ± 2.26% (one step −6.49% / +0.04%).
- Reading: what remains is about the faint 23–25 share found before (−1.82% ± 0.56%), which keeps its g = 0
  straight-line value here. c2 also sits at 2.3σ (2.1σ after one step). Neither is explained by the bright
  galaxies. Next, if wanted: a second step for the faint galaxies too (the ±0.011 slope test of the faint dig).

## 2026-10-13 (late afternoon) — bright-only second Newton step launched (GPU emulator, 30 one-core slots)

Owner chose the GPU emulator with 30 one-core slots per GPU. Files in `.../galsbi_pair1m_20261012/debug/`.

- Code (27d244f): `scripts/check_galsbi_local_sampler.py` loads the disk-response emulator on `--device` instead of
  always on the CPU (GPU on cuda). GPU and CPU trees agree to float32 rounding (≤ 1.2e-7), not bitwise, so runs
  with this commit do not replay older cuda runs bit for bit. New read-only snapshot `debug/step2/source_snapshot`
  (`git archive 27d244f`; manifest over all files except `SHA256SUMS`, the recipe that reproduces the 1M snapshot's
  manifest exactly); it differs from that snapshot only in docs and the sampler line.
- Rule written first: `rule_step2.md`. The mag < 21 galaxies (`truth_units.txt`, 34,338 galaxies, 1,720 units)
  are resampled at the one-step estimate (0.01936, 0.00037) with the production bright command, seeds
  2027807001 + k and bank 2032807001 + k. Their actual score there replaces their straight-line prediction
  from g = 0; every other galaxy keeps its it0 score: g2 = I⁻¹ Σ t, fixed it0 information, same errors as
  `pair_step.py`, same 2σ test. Expected beforehand: m ≈ −1.8% ± 0.8% if the bright shortfall is one-step
  curvature, so the faint part could still fail.
- Launcher `step2.sh` (truth.sh with one-core slots, cores − 2 = 30 per 32-core job; comoving with 1 thread);
  analysis `step2.py` (refuses until all units are done; with a run dir argument it is a code test only);
  progress `step2_status.py`; `step2_reset.py` releases a dead job's units (moved to `step2/failed_attempts/`).
- Launch: array 17178786 (4 × 1 GPU, inter). Task 0 landed on met-cl-vis02 (Quadro P5000): every slot past the
  ~16th failed with "CUDA-capable device(s) is/are busy or unavailable". Pre-Volta GPUs take at most 16 MPS
  clients, which the truth run's 15 slots fit under. Cancelled after ~3 min, 156 units released, node excluded in
  `step2.sh`, replacement job 17178859 (th-cl-nv02, V100). The others run on met-cl-vis01 (2080 Ti, ×2) and
  th-cl-nv01 (V100).
- First units: 932–989 s per 20-galaxy unit on one core (truth run: ~1,100 s on two cores), i.e. ~2× per GPU;
  ~340 units/h on three GPUs, so about 4–5 h for all 1,720 units.
- Code test of `step2.py` on the truth run (job 17178893, ~45 min CPU, mostly the bootstrap): reproduces the it0
  paired one step to 2e-16 and `truth_result_M.txt`; a step from samples at the truth would give m = −1.83% ± 0.81%
  (code check only). Note: the 1M one-step total error is ±0.81% (±0.74% was stat + chains without selection).
- Next: when all units are done, `step2.py` (CPU job), then report m against the pre-declared test.

## 2026-10-13 (afternoon) — resource audit of the last bright run and speed benchmark for the bright second step

Owner: run the second Newton step on bright galaxies only; first check resource use and speed. Files in
`.../galsbi_pair1m_20261012/debug/bench/`; no science output, no code change.

- Audit of the truth run (job 17159970, 1,720 units of 20 mag < 21 galaxies, 4 GPUs × 32 cores, 15 two-core
  slots each, 9.0 h): slots busy 98–99%, end tail ~15 min, unit time 92% sampling / 5% comoving / 3% start-up,
  CPUs 64–68% busy (ceiling ~94%). Median sampling per unit was the same on P5000, 2080 Ti and V100
  (989–1,117 s), so the GPUs were not the limit. Production step rate on the 2080 Ti: 0.268 s/step for the first
  256 steps, then 0.109 (warm-up) and 0.111 (sampling).
- Profile (`profile_wrap.py`, one unit alone): the disk-response emulator (xgboost `inplace_predict` on the CPU,
  ~16,000 neighbour pairs per call, ~46 ms) is called about once per step and takes ~2/3 of a step's CPU time;
  Python start-up from the project disk costs ~40 s per unit.
- Benchmarks (`bench.sh` job 17178399, `bench2.sh` job 17178664, both on met-cl-vis01 / 2080 Ti, 1 GPU, 1024 steps,
  mag < 21 galaxies at shear 0.02; `bench_report.py` projects a full 8192-step unit). Galaxies/h per GPU vs the
  production layout (1,193; reproduced to <1% by the repeat): 30 one-core slots 1.37×; 30 two-core slots 1.46×;
  40 galaxies per unit 1.17×; emulator on the GPU (`gpu_wrap.py`, bench only) 1.37× with the production layout,
  1.87× with 30 one-core slots on the same 30 cores, 2.19× with 45 one-core slots. GPU-vs-CPU emulator
  predictions differ by at most 1.2e-7 (values ~0.02, max 1.9; 33 checked calls), i.e. float32 rounding, not
  bitwise. GPU memory ~220 MB per slot.
- Projection for the bright-only second step (1,720 units like the truth run, 4 GPUs): ~9 h now; ~6.6 h with
  one-core slots only (exact, launcher change); ~4.8 h with the GPU emulator and 30 one-core slots (sampler change
  `device='cpu'` → the existing `--device`, not bitwise); ~4.1 h with 45 slots (needs ~48 cores per GPU). Only the
  2080 Ti was benchmarked; with the GPU busier, GPU type may now matter.
- Next: owner chooses the variant, then the bright-only second step at (0.01936, 0.00037).

## 2026-10-13 (later) — 1M pair debug: the faint (mag 23–25) shortfall, without a correction step

Owner asked to dig into the faint share (−1.82% ± 0.58% of m) before any second Newton step. Light scripts unless
noted, all in `.../galsbi_pair1m_20261012/debug/`; the published result is unchanged.

- Error bars hold: a leave-one-chunk-out jackknife over the 500 generation chunks gives the same errors as the
  scene pairing (23–25: ±0.56% vs ±0.57%; total ±0.73% vs ±0.74%). The faint shortfall is a real 3σ.
- Finite-sample (averaging) bias ruled out (`chaincount.py` → `chaincount_result.txt`): the same estimator built
  from 1, 2 or 4 chains gives 23–25 deficits −1.85 / −1.92 / −1.82%; 2-vs-4-chain difference −0.09% ± 0.06%,
  1/n extrapolation −1.73%. All other bins also unchanged (bright < 21: −1.40 / −1.37 / −1.39%).
- Starting point (`rule_timesplit.md` written first; `timesplit_extract.py`/`timesplit.sh`, CPU job 17175379,
  3 min; `timesplit.py` → `timesplit_result.txt`): early vs late half of every faint chain's kept draws.
  23–25: early −2.00%, late −1.63%, D = +0.37% ± 0.79% → pre-declared **(b) no sign of a start pull**, but weak
  (2σ bound 1.95%, as large as the shortfall). Per bin: 24–25 D = +0.87% ± 0.37%, 23–24 −0.50% ± 0.69%,
  25–26 +0.08% ± 0.17%; not interpreted (bins not pre-declared, opposite signs).
- Slope-vs-information identity (`faint_local.py` → `faint_local_result.txt`) is unusable for faint: the
  noise-corrected local slope / information falls from 0.96 (25–26) to 0.04 (23.0–23.25) in step with the chain
  noise (E/I 0.02 → 0.47), and disagrees with the one-step ratio there (1.007 ± 0.051). The selection curvature
  d²logZ/dg² is also only known to ±0.37 per galaxy (orig 0.60 ± 0.52, twin 0.14 ± 0.54).
- Selection term: dlogZ/dg1 = 0.00062 ± 0.00054 (orig) / 0.00047 ± 0.00055 (twin), consistent with 0; it moves the
  faint share by −0.25% of m. dlogZ/dg2 is clearly nonzero on both sides (0.0037 ± 0.0005, 0.0049 ± 0.0005), so the
  selection is not round in g2 (PSF or detection); harmless for m (g2 truth 0), not followed up.
- Mag 23.0–23.25 (plain MH, no tempering) has chains stuck in different modes (one of four chains at ±0.15 log
  ratio, chain noise up to ~600), like the bright two-mode trap; that slice is not short (one-step 1.007 ± 0.051).
- 100k blind per-bin slopes (23–24 1.09 ± 0.08, 24–25 0.92 ± 0.04, 25–26 1.00 ± 0.03) predict a faint deficit of
  +0.1% ± 2.4%: too imprecise to separate "slope below the information" from "biased score".
- Faint per-draw ratios hold the intrinsic shapes fixed (`profile_shapes`), so the red prior's spike enters only
  through the posterior, not the ratios themselves.
- Not possible here: a red/blue split (the mock stores measured columns only, no truth type).
- Left: a slope below the information (step problem, fixed by iterating), a real faint score bias, or chance. Only
  scores at a second shear separate them: all faint units at 0.0194 measure the faint slope/I to about ±0.011
  (shape noise cancels), against the one-step ratio 0.967 ± 0.010. This is the faint part of a correction step:
  owner decision.

## 2026-10-13 — 1M pair debug: bright scores are right at the truth; one Newton step is the problem

Valid test of `rule_truth.md` (addendum, group M), study dir `.../galsbi_pair1m_20261012/debug/`:
all 34,338 galaxies with measured mag < 21 of both sides resampled at the true shear (0.02, 0) with the production
bright settings (`truth.sh`, job 17159970, 4 GPUs, 1,720 units, ~9 h, none failed; `truth_result_M.txt`).

- Score sum at the truth S_T = −8 ± 462 (scene-paired; ±1,113 unpaired); original +8 ± 769, twin −16 ± 804.
  The same galaxies at g = 0 give S_0/(I_0 g) = 0.847 and a secant slope (S_0 − S_T)/0.02 = 0.847 ± 0.049 of I_0.
- Pre-declared decision: **(a) curved score** — the one step from g = 0 falls short, the score is zero at the truth.
  So the bright share of the shortfall (−1.39% of m) is the single Newton step, not a biased likelihood or sampler.
  Consistent with the red prior's spike at round shapes (toy, entry below).
- Interim reads during the run were misleading and are not results: an incomplete M over-weights the R/C galaxies
  chosen by measured |e|. `truth.py` now warns when M is incomplete. A shape-blind contiguous range (9,775
  original galaxies) gave +620 ± 573 at the time, consistent with the final result.
- Faint toy (`toy_faint.py`, job 17160179, |e| < 0.99 cut normalized exactly): the red prior also makes one step
  short at noise 0.03–0.3 (−3% to −19%), blue does not (−0.2% to +2%), but the toy's own score at the truth is off
  by up to 0.10 I g at noise 0.3, so it is qualitative only. The faint (23–25) −1.8% ± 0.6% is not yet tested on
  the mock.
- Straight or curved? (`localslope.py`, owner questioned the curvature reading). Relative to I_0: secant 0.847 ±
  0.041; local slope −d²logL/dg² at g = 0 0.74 ± 0.10, at the truth 1.88 ± 0.40 (30% from 5 galaxies);
  outer-product information at the truth minus at 0: +0.17 ± 0.05. So the slope and information do change between
  0 and 0.02 (about 3σ), but the pieces are noisy and not fully consistent with a smooth bend (a linearly changing
  slope would give a secant near 1.3). Stated as: the one-step formula's assumptions fail for these bright
  galaxies; "curved" is the likely reading, not proven.
- Next (owner decision; large): a second Newton iteration of the 1M pair at the one-step estimate
  (0.01936, 0.00037), which the curved-score finding predicts removes at least the bright part. No retuning.

## 2026-10-12 (evening) — debugging the 1M pair's −3.2%: small tests

The owner asked to debug the −3.2% ± 0.74% "carefully, maybe on smaller sets first". All scripts, rule files and
outputs are in `$DATA_DIR/sbsi/catalogue_prior/galsbi_pair1m_20261012/debug/`. Truth is used throughout: this is
a diagnosis after the pre-declared result, which stays as published.

- Where the shortfall sits (`groups1m.py`, reproduces m = −3.22%): by measured magnitude, mag < 21 gives
  −1.39% (±0.35%), 21–23 +0.11%, 23–25 −1.82% ± 0.58%, 25–26 −0.12% ± 0.21%. Splits by measured |e| are shown
  but are not interpretable (see the toy below).
- Cause candidate 1, the GalSBI shape prior: the red population's distortion prior, Beta(α = 1.23) times uniform
  angle, has a 2-D density ∝ |e|^−0.77, an integrable spike at round shapes (blue ∝ |e|^−0.13). Toy
  (`toy_spike.py`, exact likelihood, Möbius shear, antithetic twins): for the red prior one Newton step from g = 0
  gives m = −36% ± 4% (noise 0.003), −12% ± 5% (0.01), −2.4% ± 3% (0.03), while the root of the score sum is
  unbiased (|m| < 0.8%); the blue and smooth priors are fine. So a curved score makes one step fall short.
- Cause candidate 2, hidden Monte-Carlo noise, **ruled out** (`rule_seedrerun.md`, written first): 80 bright + 16
  faint random original units rerun at g = 0 with new seeds (`seedrerun.sh`, job 17159750, 3 GPUs, 25 min;
  6,691 galaxies). Run-to-run score differences match the chain-half noise estimate: ratio 0.94 / 1.27 (g1 / g2,
  mag < 21), 0.99 / 1.06 (21–23), 1.02 / 1.09 (≥ 23). Scaled to the 1M pair: m shift −0.16% ± 0.61% (rule:
  a cause only if |m| > 0.5%). Faint galaxies are the least constrained: up to about −1.4% is not excluded at 2σ.
- Score at the true shear (`rule_truth.md`): 252 bright round galaxies (mag < 21, measured |e| < 0.02) and 252
  bright elongated controls resampled at (0.02, 0) (`truth.sh`, job 17159853, 19 min). Round group: score sum
  −5,740 ± 470, control +79 ± 73. The rule's decision "biased score" is **withdrawn** (addendum): a group chosen
  by measured shape need not sum to zero even for an exact likelihood. `toy_group.py` (job 17159977) confirms it:
  the toy's exact-likelihood |e| < 0.02 group gives S_T/(I_0 g) = −0.8 to −19 (1M: −1.18) and one-step ratios
  0.29–0.77 (1M: 0.36), while the toy's full sample sums to 0.000 ± 0.02 at the truth.
- Valid replacement running: all 34,338 mag < 21 galaxies of both sides at the truth (group M, 1,692 more units,
  job 17159970, 4 GPUs, about 9 h). Pre-declared: zero sum (|S_T| < 2σ, > −1.6e3) means the bright part is a
  curved score (iterate the Newton step); −3.2e3 means a biased score.
- Side finding: the selection term dlogZ/dg1 = 0.00055 ± 0.00036 per galaxy shifts m by −0.48% ± 0.31%. At g = 0
  it should vanish by symmetry for a population with no preferred direction (not verified for the measurement
  model), so it adds noise, not signal. Not changed.
- Next: the M result; the faint (23–25) part needs a different test, since a valid score-at-truth test there would
  resample about half the mock.

## 2026-10-12 — 1M rotated-twin paired mock: m = −3.2% ± 0.8%, the pre-declared test fails

Details: [HYBRID_SAMPLING.md](HYBRID_SAMPLING.md#rotated-twin-paired-1m-mock-m--32--08-the-pre-declared-test-fails-2026-10-12).

- Paired 2 × 1M one step from g = 0 (`pair_step.py comoving_circ`, job 17155372, 2.3 h):
  g1 = 0.01936 ± 0.00016, g2 = +0.00037 ± 0.00018. Against the truth (0.02, 0): m = −3.2% ± 0.8% (−4.0σ),
  c2 +2.06σ. The rule needed both within 2σ, so the test fails. The error matches the rule's prediction (0.00016).
- Single sides: original m = −6.5% ± 2.2%, twin +0.0% ± 2.2% (anti-correlated; difference 1.6σ).
- Inverse-variance mean with the 2 × 100k pair and the 500k mock (not pre-declared): m = −3.1% ± 0.7%,
  χ² = 0.3 for 2 degrees of freedom. The earlier passes were consistent with a −3% bias, only too noisy.
- Not retuned. Candidate causes, not yet tested: the single Newton step (slope/I = 0.98 ± 0.03 on the 100k
  mock allows 3%) and finite-sample bias of the log-mean ratios.
- Plot `SBSI/plots/pair1m_paired.png`. `plot_pair.py`'s save to SBSI/plots from the compute node did not appear;
  the file was copied from the study dir.
- Run: sampling 01:15–13:33, co-moving about 2.5 h per side, 23 units rerun with their own seeds (rule.md addenda;
  entry below).

## 2026-10-12 — 1M rotated-twin paired mock launched (`galsbi_pair1m_20261012`)

The owner asked for the 1M paired mock with `--no-chain`, on about 500 cores "with enough GPUs" (AGENTS.md rule
added). The rule (`rule.md`) was written before the mock existed: the 100k pair's recipe at 10x the galaxies and 10x
the selection chunks; expected sigma(m) about 0.8%; pass if m and c2 are within 2 sigma of zero.

- Study dir `$DATA_DIR/sbsi/catalogue_prior/galsbi_pair1m_20261012/`, code snapshot of 4a506ca (`source_snapshot/`).
- Mock (`prepare.sh`, job 17144346, cip A40): original 1,000,000 selected galaxies at (0.02, 0), seed 2027101201;
  then the antithetic rotated twin in the same job. Original: 110,792 bright (mag < 23) in 5,276 units of 20-21,
  889,208 faint in 2,806 units of 316-317.
- Sides `original/` and `twin/` carry the 500k study's scripts with only the path changed (`copy_scripts.py`);
  `blind_step.py` now reads the plan's `selection_replicas` (default 2), because one replica holds at most 1000
  chunks before its seeds reach the next replica's. Selection: 4 x 800 chunks per side, bases 2027207001 (original)
  and 2027307001 (twin). `pair_step.py` is the twin study's, pointed at the two sides.
- `worker.sh`: one queue over both sides (original units first), MPS, `--no-chain` for faint units, selection
  replicas claimed by workers started with SEL=1; a side's plan is read when a slot reaches it.
- Workers (`launch.sh`, jobs 17146034-44): 4 x 16 cores on met-cl-vis01 (RTX 2080 Ti, the only free inter GPUs),
  4 x 96 cores queued on inter for the H200/H100/A40 nodes (all GPUs busy at launch), 3 x 12 cores on cip A40
  slices. Per-user limits: inter 8 GPUs / 512 cores, cip 3 GPUs / 64 cores.
- Startup: MPS starts on all GPU kinds (including the cip virtual A40s). Bright units run at about 30 s per 1000
  steps on a cip A40 slice and about 140 s on met-cl-vis01, where the GPU is only 20-37% busy, so that node's
  hyperthreaded cores limit it. Selection: 17 s per chunk (about 3.8 h per replica).
- CPU-only supplement (rule.md addendum, before any result): the 96-core inter workers wait for busy GPU nodes, so
  `CPU=1 worker.sh` runs the same queue, seeds and settings with `--device cpu` on `cluster` (array 17146079,
  21 x 20 cores, th-cl-hua28 excluded). Task i takes a new unit only while 20 (i+1) <= 520 - (cores of running
  p1m-w* GPU workers), so the total stays about 500 cores and CPU tasks drain as GPU workers start. First check:
  270 units started on 13 hosts, initial replay max error 0.0 on every host; CPU bright units at about 140-900 s
  per 1000 steps depending on the node (cip A40 about 26, met-cl-vis01 about 80); 40 units done, 0 failed.
- Twin (same job): 1,000,086 selected galaxies, all 500 chunks replay the original with error 0.0; planned with
  `make_pair_plan.py twin` (110,788 bright, 889,298 faint, 8,082 units, selection base 2027307001).
- 05:00: w4 started on an H100 NVL (45 slots, GPU 87-95% busy, about 31 s per 1000 steps); five CPU tasks drained
  as designed. 3,295 of 8,082 original units done; original selection replicas 0-3 done, twin replicas running.
- Two failed units, handled per a rule.md addendum written before any result, with `rerun_unit.sh` (own seeds,
  failed attempt moved to it0/failed_attempts/, claim kept so no worker re-claims it):
  - bright_u2752: no finite starting state, caused by mock row 521886 (measured radius 1807 px, flux 1.1e6), outside
    the radius < 100 px cut, as in the 500k study's unit 2292. Rerun without that row (replay 0.0).
    `pair_step.py` now accepts a missing row only if it is outside the cuts, like `blind_step.py`.
  - bright_u3144: initial replay 7.3e-6 > 1e-6 on CPU, identical on two different CPU machines, so the CPU path for
    that unit, not a node fault. Tolerance unchanged; rerun on one inter GPU (`DEVICE=cuda`), replay 0.0.
  - First `rerun_unit.sh` version stopped at its attempt count under `pipefail` before touching anything; fixed.
- 07:40: 6,089 original units done; w5 started (second H100 worker on usm-cl-nv01). 8 more units failed the initial
  replay (1.07e-6 to 1.21e-5), all on the CPU task on th-cl-hua26; no other node failed it. That task was
  cancelled and replaced by one that avoids th-cl-hua nodes (17146852_0, th-cl-rome01n2); its 10 unfinished
  units were marked failed. `rerun_failed.sh original 17146081` reran all 18 on th-cl-rome nodes: replay 0.0 for
  all 18 (rule.md addendum ~07:50).
- 09:50: original sampling complete (8,082 units, 20 by rerun); selection of both sides complete (4 x 800 chunks
  each). w6 started (kng-cl-nv03). Co-moving post-processing of the 5,276 original bright units: array 17147361,
  32 lists, 4 cores each, named `p1m-wc-o` so the CPU sampling tasks give it their cores (they drained to 0).
  Its replay checks match the 500k study (kernel replay median 4e-4, max 3e-3; 500k: 3e-4, 2e-3).
- 12:00: twin 6,410 of 8,082 units done, all bright units done except twin bright_u4341, which stopped at
  initialisation because of twin row 823385 (measured radius 2791 px, outside the cuts); rerun without it
  (replay 0.0; rule.md addendum ~12:00). Twin co-moving array 17150029 (`p1m-wc-t`) queued after the original's
  and after that rerun.
- 12:20: owner noted 4 GPUs would have been enough; at that point 10 GPUs were in use (4 RTX 2080 Ti, 3 cip A40
  slices, 3 H100-class). The pending eleventh worker (w7) was cancelled; the rest released their GPUs at about
  13:15 when the queue emptied. Original co-moving complete (5,276 units).
- Twin faint_u2601 (H100) failed the invariant-cache audit (cached vs full log-kernel 1.27e-6 > 1e-6); a CPU rerun
  failed the same audit at the same proposal (1.17e-6), so it is this unit's state, not a device. Tolerance
  unchanged; rerun with its own seeds without `--reuse-invariant-work` (the default full-likelihood path) via
  `job_galsbi_tempered_noreuse.sh` and `rerun_unit.sh` `JOB=...`, on one GPU: done (rule.md addendum ~13:20).
- 13:33: twin sampling complete (8,082 units, 2 by rerun). `pair.sh pair_step.py comoving_circ` (job 17155372)
  waits for the twin co-moving array.

Next: twin plan once the twin exists; co-moving post-processing of the bright units; `pair_step.py`.

## 2026-10-11 — Faint chains no longer saved (`--no-chain`); old faint chains deleted

The owner asked to stop keeping the faint units' chains, which nothing reads, and to remove the old ones.
Details: [TEMPERED_MH.md](TEMPERED_MH.md#disk-faint-chains-are-not-saved-2026-10-11).

Code: `scripts/check_galsbi_local_sampler.py` gains `--no-chain` (skip `chain.npz`; protocol `chain_saved`).
Launcher template `optimize_20261011/prod_check/worker.sh` passes it for faint units.

Validation: job 17131270, same GPU, one faint unit: all other outputs bitwise identical; history output
3.7 → 0.15 s. Readers of `chain.npz` in the four finished studies (`comoving.py`, `bcheck.py`) take bright
units only (co-moving lists: 2,680 / 2,094 / 524 bright, 0 faint).

Old faint chains: 3,172 files, 4.12 TB (blind 100k 2.49, 500k mock 1.13, twin 100k 0.23, blind 10k 0.27 TB),
listed in `$DATA_DIR/sbsi/catalogue_prior/faint_chain_cleanup_20261011/inventory.tsv`. Deleted by the owner
on 2026-10-12 from `faint_list.txt` (only `faint_*/chain.npz`; 1,408 / 1,402 / 281 / 81 per study) with
`xargs rm -fv`; `deleted.log` lists all 3,172 removals, none of the listed files remains, and
/project/ls-gruen free space went 13T → 17T (df -h). Bright chains (6,504 files, about 0.38 TB) and every
result, profile and final-scene file stay.

## 2026-10-11 — Sampler speed: b from the pair table only, after warm-up only (exact; sampling 1.16-1.21x)

The owner asked whether the sampler could be made faster still.
Details: [TEMPERED_MH.md](TEMPERED_MH.md#speed-b-straight-from-the-pair-table-after-warm-up-only-2026-10-11).

Code:
- `sbsi/galsbi_summary.py`: new `response_pairs` builds only the response pair table; `scene_model_inputs`
  calls it (same output). The aperture/neighbour check is shared.
- `scripts/check_galsbi_local_sampler.py`: `fill_b` uses `response_pairs` + the disk response directly instead
  of recomputing the eight features; it runs from the last warm-up step on, so warm-up b in `chain.npz` is NaN
  (no reader uses it). `disk_coefficient` protocol text updated.
- `tests/test_galsbi_summary.py`: `response_pairs` equals the pair output of `scene_model_inputs` bit for bit.

Validation (`$DATA_DIR/sbsi/catalogue_prior/optimize_20261011/`):
- 19 crowded units with MPS on kng-cl-nv02 (job 17130336) vs c903adf (job 17130300), comparison 17130356:
  every array and diagnostic bitwise identical except warm-up b, now NaN.
- Tests, job 17130363: 389 passed, 1 skipped (BlendEMU not on PYTHONPATH).

Results: sampling time per unit 53.5 → 46.7 s bright (1.16 ± 0.06x), 48.3 → 40.0 s faint (1.21 ± 0.04x);
expected about 1.13x per production iteration on top of the earlier 2.4x.

Not done: per-pair tree-output reuse (probe 17130318: only 16-21% of pairs repeat); uncompressed chain files
(job 17130343: 41 → 1.9 s per faint unit but 805 → 1046 MB, owner's call); a long-lived process per slot to
skip about 20-30 s of start-up per unit (about 3-5%, risk of carried state).

## 2026-10-11 — Sampler speed: b only for stored states at zero shear (exact; ~2.3x per bright step with MPS)

The owner asked for a deep look at the sampler's efficiency.
Details: [TEMPERED_MH.md](TEMPERED_MH.md#speed-b-only-for-stored-states-at-zero-shear-2026-10-11).

Code:
- `sbsi/galsbi_summary.py`: `scene_model_inputs(..., response=False)` returns the eight features without
  building the neighbour pair table.
- `scripts/check_galsbi_local_sampler.py`: when sampling is persistent at g = (0, 0), the kernel does not depend
  on b, so proposals and measurement-initialization candidates skip the tree emulator. `fill_b` computes b for
  cold rows before each trace is stored. The initial evaluation and the audits still compute b everywhere.
  Protocol key `disk_coefficient` records which mode ran. Non-zero sampling shear is unchanged.
- `tests/test_galsbi_summary.py`: the kernel at (0, 0) is bitwise equal with b set to 0, and differs at
  non-zero shear; `response=False` returns the same features.

Validation (study dir `$DATA_DIR/sbsi/catalogue_prior/optimize_20261011/`):
- Same seeds and node, job 17129212 vs 17129137: bright and faint units bitwise identical in chain,
  initial state, final scenes, local profiles and diagnostics; audits unchanged (replay 0.0, invariant
  2.8e-14 / 0.0). g = (0.02, 0), job 17129232: bitwise identical.
- `compare_all.sh`: all 19 crowding-benchmark units of new code + MPS match old code without MPS bitwise.
- Tests, job 17129230: 389 passed, 1 skipped.

Results (seconds; one process; cProfile):
- Bright: setup 43.1 → 15.3, sampling 45.6 → 22.6 (2.0x), total 91.6 → 40.7.
- Faint: setup 190.2 → 34.2 (5.6x), sampling 44.8 → 34.7 (1.3x), total 270 → 104.
- 19 processes on one A40 (job 17129219), s/step: old 0.098, new 0.062, new + MPS 0.043 (≈2.3x).

Limitations: gain only at zero sampling shear with persistent environments. Faint sampling gains less
(about 40% of rows change per step, so b is still computed often). The xgboost CUDA predictor is about 9x
faster but not bitwise equal to CPU (job 17129157), so it is not used; the old "CUDA replay failure" was this
CPU-vs-GPU comparison. Timings are single runs; per-process spread was not measured here (±4% in 2026-10-03).

Production check (same day, owner's step 1):
- `optimize_20261011/prod_check/`: `prepare.sh` freezes c903adf and pre-claims every unit except 38 that the
  500k A40 worker ran (19 bright, 19 faint), so they keep their seeds. `worker.sh` is the 500k launcher plus MPS
  start/stop. `check.py` does the comparison.
- Job 17129330 (kng-cl-nv02, 40 cores, 19 slots, production chain lengths): 17.5 min, no failures. Job 17129408:
  38 of 38 units bitwise identical to production.
- Per unit total: bright 1016 ± 17 → 398 ± 2 s (2.55 ± 0.04x); faint 1292 ± 12 → 570 ± 7 s (2.27 ± 0.03x);
  bright s/step 0.115 → 0.043. A 500k-sized iteration needs about 2.4x less slot time.

Next: use `prod_check/worker.sh` as the template for the next study.

## 2026-10-07 — Unpaired 500k mock: one step from g = 0 gives m = −4.1% ± 3.2%

The owner asked for a 500k mock without pairing.
Details: [HYBRID_SAMPLING.md](HYBRID_SAMPLING.md#unpaired-500k-mock-m--41--32-2026-10-07).

Code: no repository code changed. Study scripts are in `$DATA_DIR/sbsi/catalogue_prior/galsbi_mock500k_20261006/`
(code snapshot 8cd45c5). `blind_step.py` there now accepts a galaxy with no result only if it is outside the
cuts, and reports it.

Results (rule fixed before the mock was generated):
- g1 = 0.01918 ± 0.00063, g2 = +0.00028 ± 0.00065, so m = −4.1% ± 3.2% (−1.3σ) and c2 at +0.4σ. Pass.
- Error budget for g1: stat 0.00059, chains 0.00018, selection 0.00015; the galaxy bootstrap gives 0.00062.
- 497,482 galaxies used; 2,517 cut, and 1 not sampled (row 431089, measured radius 13,139 px, outside the cuts).

Validation: all 4,062 units sampled. Unit 2292 was rerun without the broken row, with the same seeds. All 2,660
bright units were co-moving processed. The bootstrap matches the error budget.

Limitations: a model-closure test (the mock is drawn from the likelihood model), not an image-simulation test.
One mock and one seed. The stencil Hessian cross-check is noisy (I/H 1.05 here, 0.70 in the 100k runs).

Next: none planned. Pairing gives more precision per CPU hour than size on mocks.

## 2026-10-06 — F8 retrained on measured unsheared |e| < 0.7: ConstGold m worse (−10.4% vs −6.4%)

The owner asked to train the likelihood on data cut at measured |e| < 0.7 (unsheared), per galaxy, from both legs.
Details: [V39_MODEL.md](V39_MODEL.md#f8-retrained-on-measured-unsheared-e--07-2026-10-05).

Code: no repository code changed. All new scripts are in `sbsi_caches/v39_f8_measured07_20261005/`:
`prepare_measured07.py`, `run_stage.py`, `f8_rows.py`, `nll_compare.py`, `summary.py`, `make_experiment.py`,
`job.sh` and `rule.md`. The control (`v39_f8_control_20261005/`) and the cancelled truth-|e| variant
(`v39_f8_truthe07_20261005/`) are kept for the record.

Results:
- ConstGold, |e| < 0.7 with C8 (the pre-declared primary): −10.386% ± 0.338, against F8's −6.384% ± 0.352.
  The paired difference is −4.002 ± 0.038 pp.
- Without the |e| cut: +4.148% ± 0.277 (F8: +0.021% ± 0.265).
- With the true flags in place of C8: −10.227% ± 0.341.
- Training checks: guard loss 0.954 vs 0.006. On the same kept validation galaxies the new flow is 0.040 ± 0.001
  nats per galaxy worse than F8.

Validation:
- `summary.py` reproduces the published +0.020770% ± 0.265152 before reporting.
- The control retraining is byte-identical to F8 at the base stage and within 1.5e-8 after refinement.

Limitations:
- One seed, g1 only, 80 development cases.
- The control's last refinement step ran out of memory on a 16 GB GPU and was not rerun, because it would
  reproduce F8.

Next: none planned. Cutting the training data on one leg's measured shape is not a fix. A per-leg cut that the flow
models explicitly would be a different experiment.

## 2026-10-05 — Measured |e| < 0.7 cut: unbiased on the paired mock, −6.4% on ConstGold

The owner asked to try the |e| < 0.7 cut that real data often uses, first on the 2 × 100k paired mock and then
with V3.9-like on ConstGold, both with C8 and with the true usable flags.
Details: [paired mock](HYBRID_SAMPLING.md#paired-mock-with-a-measured-e--07-cut-m--13--44-2026-10-05),
[ConstGold](V39_MODEL.md#measured-e--07-cut-on-constgold-2026-10-05).

Code: no repository code changed. All new scripts are in the study directories.
- Paired mock (`galsbi_blind100k_20261003/`, `galsbi_twin100k_20261005/`): `make_e07.py`, `selection_e07.sh`,
  `analyse_e07.sh`, `e07_rule.md`. `pair_step.py` takes an optional iteration name; the defaults are unchanged
  (backup `pair_step_before_e07.py.bak`).
- ConstGold (`sbsi_caches/v39_constgold_ecut07_20261005/`): `f8_e07_rows.py`, `c8_probabilities.py`,
  `summary.py`, `job.sh`.

Results:
- Paired mock, one step from g = 0: g = (0.01975 ± 0.00088, 0.00131 ± 0.00092), so m = −1.3% ± 4.4%.
  The slope check gives 0.96 ± 0.06. The cut drops 18,413 originals and 18,266 twins.
- ConstGold, V3.9-like + |e| < 0.7: m = −6.384% ± 0.352 with C8 and −6.248% ± 0.355 with the true flags.
  Without the cut the same code gives +0.021% ± 0.265 and +0.228% ± 0.266. The cut shifts m by
  −6.405 ± 0.261 pp (C8) and −6.476 ± 0.261 pp (true flags).
- The model overpredicts the response with the cut (0.496 against 0.464), while the selected counts agree to 0.1%.
  This points to the F8 flow + disk model near the cut boundary, not to C8.

Validation:
- F8 draws rebuilt on A40 match the saved rows exactly (max difference 0).
- The C8 probabilities match the saved sums.
- `summary.py` reproduces the published +0.020770% ± 0.265152 before reporting.
- The paired-mock selection seeds equal those of the 0.99 run.

Limitations:
- The mock test is self-consistent by construction.
- The ConstGold check covers g1 only, on 80 development cases.

Next, only if the owner wants the |e| < 0.7 cut in production: find where the model's measured-shape
distribution near |e| ≈ 0.7 departs from ConstGold's, and train on half-shear data, never ConstGold.

## 2026-10-05 — Rotated-twin paired mock: one-step m = −2.0% ± 2.5%

The owner noted that m = +9% ± 7% on the blind 100k mock is too loose a test and asked for a rotated-twin
paired mock. Details are in
[the twin section](HYBRID_SAMPLING.md#rotated-twin-paired-mock-m--20--25-2026-10-05).

Code:
- 8cd45c5: `scripts/prepare_galsbi_twin.py`.
  - It rebuilds every chunk of a population mock and checks it matches the stored rows bit for bit.
  - It then draws the same scenes turned by 90° (target intrinsic e → −e), with the same detection random
    numbers.
  - The twin's measurement noise is antithetic: the flow's two shape draws are negated.
  - `--test-chunks` compares the noise couplings.
- The study scripts live in `$DATA_DIR/.../galsbi_twin100k_20261005/`. The pipeline scripts are copies from
  the 100k study (`copy_scripts.py`); new are `make_twin_plan.py`, `launch_it0.sh`, `make_lists.py`,
  `pair_step.py`, `pair_check.py`, `plot_pair.py`.

Result:
- 99,922 twins; 98,630 scenes have both members selected.
- Each catalogue alone gives m = +9.0% ± 7.1% (original) and −12.9% ± 7.0% (twin).
- **Paired:** g = (0.01959 ± 0.00051, 0.00096 ± 0.00056), so m = −2.0% ± 2.5% and g2 is 1.7σ from zero.
- The paired stat error is 0.00037, against 0.00094 if the galaxies were independent.

Validation:
- The original rebuild matches exactly in all 50 chunks.
- `pair_step.py` on the original alone reproduces `blind_step.py` (0.021804, 0.001403) and its errors.
- The bootstrap over scenes (0.00046) agrees with stat plus chains (0.00047).
- Co-moving replay < 0.005 in every bright unit; 805/805 units done.

Limitations:
- One mock pair.
- Chain noise (±0.00029) and selection noise (±0.00020) are now comparable to the shape noise (±0.00037).
- The |e| < 0.99 cut still awaits the owner's decision.

Next: for ±1% on m, longer bright chains and more selection chunks, then more twin pairs.

## 2026-10-05 — Blind 100k run: two fixes make the one-step (BFD) Newton estimate work

The owner said that "in principle we need only one iteration (see BFD)" and asked me to debug autonomously why the 100k blind run needed four.
Details are in [the 100k section](HYBRID_SAMPLING.md#blind-100k-run-and-why-one-newton-step-was-not-enough-2026-10-03-to-10-05).

Code:
- 5847c44: plan key `max_radius_pixels` (selection Z). It removes one 858-px flow-tail galaxy.
- a7b4468: `--comoving-profiles` in `check_galsbi_local_sampler.py`, plus
  `GalSBITargetCoordinates.comove` in `sbsi/galsbi_target_coordinates.py`.
  - The intrinsic shape moves with the shear: ε' = M_g'⁻¹(M_g0(ε)).
  - The circularized radius is kept, and the exact Jacobian is included.
- cc76c1d and 8f16643: plan key `max_measured_ellipticity` in `run_galsbi_bright.py selection()`.
  - Z cuts on the model's measured shape: the flow draw moved by radial_tanh(b g), with b predicted per draw by
    `CatalogueDiskResponse`.
  - b is predicted only for draws with radius > 3 px and flux > 0.
- The study scripts are kept with the study in `$DATA_DIR/.../galsbi_blind100k_20261003/`:
  - new: `comoving.py`, `make_cut.py`, `make_one.py`, `groups.py`, `g2check.py`, `zcheck.py`, `zscan.py`,
    `c0check.py`;
  - now variant-aware: `blind_step.py`, `onestep.py`, `final_report.py`, `plot_blind.py`.

Result:
- The old iteration (fixed-shape ratios, all galaxies) took 4 steps and ended at g1 +2.5σ. The step used
  information about 1/0.68 too large.
- Cause 1: noisy fixed-shape ratios for bright galaxies. Co-moving ratios fix it: bright chain noise falls from
  80% to 1.3% of the bright score variance.
- Cause 2: 492 galaxies with measured |e| ≥ 0.99 have strongly non-linear likelihoods. The |e| < 0.99 cut on
  data and Z removes them.
- One step from g = 0 now gives (0.02180 ± 0.00142, 0.00140 ± 0.00145).
- The test iteration `one`:
  - g1 step +0.00002; measured slope / I = 0.98 ± 0.03.
  - g2 step −0.00152, just outside the pre-declared total error of 0.00141. That is about 2.2σ of the chain
    and selection noise.
- Truth, read once at the end: the one-step g1 is +1.3σ, m = +9.0% ± 7.1%.

Validation:
- `pytest tests/test_galsbi_target_coordinates.py`: 7 passed.
- Co-moving replay: a node at g0 reproduces the stored kernel to < 0.01 in every unit, except the 858-px
  galaxy, which is cut.
- c0, a rerun of it0's bright chains on other GPUs:
  - Half-split chain noise is honest: Σ difference² / predicted = 0.72 co-moving, 1.03 fixed.
  - Recomputing b vs keeping it moves the one-step estimate by 0.0001.
- Selection chunk values are Gaussian (kurtosis 2.8–3.3).
- The g2 scan shows no real change of the selection slope across g2 = 0 (1.4σ).

Limitations:
- One mock.
- The g2 confirmation is noise-limited: selection ±0.0003 and short chains ±0.0004 per iteration.
- The |e| cut is a modelling choice for the owner to confirm (0.5% of galaxies).
- With the cut, selection costs about 19 s per chunk, against 3 s without it.

Next:
- Use the same selection seeds across iterations, so selection noise cancels in the confirmation step.
- More selection chunks.
- Decide on the |e| cut.

## 2026-10-03 — Blind hybrid MH estimate on the 10k mock recovers the shear within 1σ

The owner asked for the following while away for 10 h: more efficient sampling, hybrid sampling on the 10k
mock, and debugging until the shear is recovered, keeping it simple.
Details in [the blind section](HYBRID_SAMPLING.md#blind-hybrid-mh-estimate-on-the-10k-mock-2026-10-03).

Code: none beyond d31ea79 (`--shear`, g1/g2 ratio stencils, `--profile-stride`, 2-D selection nodes). The
study scripts are in `$DATA_DIR/.../galsbi_blind10k_20261003/`.

Run:
- Bright galaxies (mag < 23) ran tempered MH at 32k steps; faint galaxies ran plain MH at 16k steps.
- One selection term at g0 and g0 ± 0.002 in g1 and g2.
- Newton iterations from g = (0, 0) with no truth input: it0 (16805631–34), it1 (16806051–54),
  it2 (16806315–18).
- Each iteration took 80–91 min on 4 GPUs × 44 cores.

Result:
- it2 converged (step inside its errors): g = (0.0240 ± 0.0051, −0.0038 ± 0.0044), against the truth
  (0.02, 0). That is +0.8σ and −0.85σ, or m = +20% ± 25%.
- The decision rule was written before it2 finished (`final_rule.md`).
- it1 and it2 agree at 0.9σ. Bootstrap errors match.

Debugging:
- The stencil-curvature cross-check is unusable at these chain lengths, because of bright-galaxy chain noise.
  The chain-noise-corrected score step is used instead.

Limitations:
- One mock with 10k galaxies, so the test of m is only ±25%.
- One-step Newton estimator.
- Bright chain noise is 35–55% of their score variance.

Next:
- A sharper test of m needs more galaxies, or several mocks.
- Halving the faint steps would save about 13% of the core time.

## 2026-10-03 — Faster exact MH loop; sampler can run at a chosen shear (for a blind estimate)

The owner asked to make MH cheaper per GPU, then to run a blind hybrid estimate on the 10k mock.
Details in [the speed section](TEMPERED_MH.md#speed-exact-prior-memo-and-skipped-sérsic-draws-2026-10-03).

Code:
- a5588b8 (`sbsi/galsbi_prior.py`, `scripts/check_galsbi_continuous_prior.py`,
  `scripts/check_galsbi_local_sampler.py`, `tests/test_galsbi_prior.py`):
  - an exact per-redshift memo (`ExactMemo`) for the Sérsic mass, distances and volume;
  - Sérsic cdf/ppf skipped for window-rejected neighbour proposals, with their uniforms still drawn.
- d31ea79:
  - `check_galsbi_local_sampler.py --shear g1 g2` samples at a chosen shear, instead of the reference
    (truth) shear.
  - Saved likelihood ratios now sit at ±0.002 around that shear, in g1 and in g2; the g1 key names are
    unchanged.
  - `--profile-stride` thins the saved ratios.
  - The non-persistent noise audit uses the sampling shear (the hard-coded 0.02 is gone).
  - `run_galsbi_bright.selection` accepts (g1, g2) nodes.

Validation:
- Bit-identical outputs against the old code: the loop runs (2096 and 304 items) and 12 benchmark shards
  (job 16805404), all with 0 differences.
- Tests (16805402): 386 passed, 3 skipped.
- Speed: 1.23x (plain) and 1.21x (tempered) per GPU on the same node.
- At 6 processes per GPU, the GPU is busy only 12–18%. Production should use all allowed cores at
  2 per process (smoke test 16805601).

Limitations:
- xgboost and scene-input building remain the main per-step cost.
- Disk use at 1M scale is about 2.5 TB for the bright galaxies' chains.

## 2026-10-03 — Correction: the bright score test is a truth-anchored closure check

The owner pointed out that the score test leans on the true shear. Confirmed:
- Every MH chain on the 10k mock sampled at the mock's true shear, read from `reference/protocol.json`
  (`shear` = [0.02, 0]).
- The start-point picker in `scripts/check_galsbi_local_sampler.py` scores candidates at that same shear.
  (Corrected later the same day: an earlier version of this entry said the picker hard-codes 0.02; the
  hard-coded 0.02 is only in `environment_noise_audit`, used without persistent scenes.)
- So both the grid-peak fits and the score test are anchored at the truth.

Doc fix: [the rerun section](HYBRID_SAMPLING.md#tempered-rerun-of-the-mag--22-galaxies-and-a-score-test-at-the-true-shear-2026-10-03)
now calls the 0.026 ± 0.008 result a closure check ("starting from the truth, do the data push away?"),
not an estimate.

Next: a blind version would start from g1 = 0 and iterate (also replacing the audit's hard-coded 0.02). Not launched; it waits for an owner decision.

## 2026-10-03 — Tempered rerun of the bright mag < 22 galaxies and a score test at the true shear

The owner asked to fix the unstable bright-only estimate autonomously. Details in
[the rerun section](HYBRID_SAMPLING.md#tempered-rerun-of-the-mag--22-galaxies-and-a-score-test-at-the-true-shear-2026-10-03).

Code: no repository code changed. The study scripts live in `$DATA_DIR/.../galsbi_mock10k_20261002/`
(`rerun_candidates.py`, `make_rerun_plan.py`, `rerun_job.sh`, `submit_rerun.sh`, `estimate_rerun.py`,
`score_test.py`, `plot_rerun.py`). They run the frozen snapshot of commit e77ce80.

Run:
- The rule was fixed in advance in `rerun_plan.json`: rerun all 456 galaxies with mag < 22, using tempered MH
  (8 rungs, 32k steps).
- Two independent replicas, A (jobs 16803344/46/48/50) and B (16803345/47/49/51), each 39–45 min on 4 GPUs.
- All 48 profile shards passed the fixed-shear replay on their own node.
- Reports: 16803387 (dry run), 16803442 (rerun), 16803446 (score test).

Result (true g1 = 0.02):
- Galaxies with a stuck chain dropped from 246 to 62 out of 456.
- The grid-peak fit still flips: replica A gives 0.015 and replica B gives 0.025 at every cut. The A − B
  difference is 4–5 times its predicted sampling error, because the likelihood more than 0.002 from the
  sampled shear is too noisy for bright galaxies.
- A score test at the true shear, using only the nodes 0.018 and 0.022, gives errors that pass the A-vs-B
  check (|z| ≤ 1.4):
  - mag < 23: g1 = 0.026 ± 0.0065 (stat) ± 0.0049 (sampling), m = +29% ± 41%;
  - mag < 22: g1 = 0.006 ± 0.008 ± 0.007;
  - every cut is within 1.3σ of the truth.
- Figure: `SBSI/plots/galsbi_bright10k_rerun.png`.

Limits:
- This is a closure test, because the chains were run at the true shear. Real data would need sampling at,
  or iteration towards, the estimate.
- The nested cuts are correlated.
- Bright galaxies alone constrain m only to about ±40%.

Next: add the faint galaxies to bring the error down. That means MH for about 8,900 more galaxies, or the
old sampler for them.

## 2026-10-03 — Selection-corrected bright-only g1 on the 10k mock: not stable yet

The owner asked for the bright-only estimate first. Details in
[the estimate section](HYBRID_SAMPLING.md#selection-corrected-bright-only-g1-from-the-same-chains-2026-10-03).

Code:
- `scripts/run_galsbi_bright.py`: `selection()`/`profiles()` take an explicit population reference and run
  folder (commit e77ce80). `tests/test_galsbi_bright.py` has a new test; `pytest tests/test_galsbi_bright.py -q`
  → 6 passed.
- The study driver lives in `$DATA_DIR/.../galsbi_mock10k_20261002/` (`estimate_bright10k.py`). It reuses
  `profiles`, `selection`, `profile_fit`, `selected_log_profile`, `peak_gradient` and `conditional_mcse`
  unchanged.

Run:
- Profiles: jobs 16803116 and 16803122–24. Selection replicas: 16803116–17. Report: 16803125. Diagnostic: 16803138.
- The fixed-shear replay (1e-6) fails across GPU models, so each MH batch was replayed on its own node.
  Jobs 16803117–19 failed for this reason and are superseded.

Result for the true g1 = 0.02:
- Mag < 23 gives 0.0147 at 16k steps and 0.0258 at 32k steps. The quoted errors are ±0.0012 statistical and
  ±0.0004 sampling.
- The likelihood has two peaks (0.015 and 0.025), and which one wins flips with chain length and chain half.
- About ten galaxies at mag 18–21, with ratio ESS 1–25, drive the flip.
- The galaxy bootstrap gives ±0.005–0.007. The selection term changes the fit by less than 0.0001.

Limits:
- No usable calibration number.
- The quoted MCSE misses the peak flip.
- There is no MH replica; chain halves stand in for it.

Next: resample about 100 low-ESS bright galaxies with many more chains (or tempering), then rerun the report.

## 2026-10-02 — Fresh 10k GalSBI mock and MH on its 1,098 bright members

The owner asked for MH on the bright galaxies of a fresh 10k GalSBI mock. Details in
[the 10k section](HYBRID_SAMPLING.md#fresh-10k-galsbi-mock-mh-on-the-bright-members).

Changed:
- `jobs/job_galsbi_shared_gpu.sh` takes an optional `SBSI_OBS_LIST` index file (commit 3518934).
- Without it, behaviour is identical. A dry test confirmed the old range split (0–7 over 3 → 0–1, 2–4, 5–7)
  and the list mode.

Mock:
- `prepare_galsbi_population.py`, same parent and selection as the 1024 panel, seed 2026100601.
- 10,000 observations; job 16803064 took 65 s.
- Bright (measured mag < 23): 1,098.

Run:
- 32k-step untempered MH on 4 GPUs (jobs 16803065–68, 23.5–28 min), analysis job 16803069. All COMPLETED 0:0.
- The analysis asserts that every bright galaxy was covered exactly once.

Result:
- 42% of bright galaxies have a stuck chain.
- Sampling noise is 30–95% of the score scatter per bin.
- The ten noisiest galaxies (mag < 20.8) hold 70% of the noise.
- Mean scores lack the selection term, so they are not a bias measurement.

Limits:
- No replica run.
- The 22–23 stuck rate differs from the 1024 panel by 2.9σ.

Next: targeted reruns (more chains or tempering) of the about 100 noisiest galaxies.

## 2026-10-02 — Rough panel g1 from the 1024 MH run

The owner asked whether these galaxies have a shear estimate. They did not: the run saves only g1
likelihood ratios at 0.018 and 0.022. Two rough estimates, without the selection term and g1 only,
are in [the 1024-panel subsection](HYBRID_SAMPLING.md#rough-panel-shear-from-the-same-run-no-selection-correction).

- Parabola fit: fails, because the bright-galaxy curvature is dominated by Monte Carlo noise.
- Score-based estimate, all 1024 galaxies: -0.006 ± 0.018 (32k run) and +0.018 ± 0.015 (16k run).
- Score-based estimate, fainter than 23: +0.000 ± 0.016 and -0.001 ± 0.016. True g1 = 0.02.
- Shape noise limits the panel to about ±0.013 on g1, i.e. ±65% on m.

Validation: job 16803055 COMPLETED. Scripts `shear_estimate.py` and `shear_from_score.py` live in
`$DATA_DIR/sbsi/catalogue_prior/galsbi_mh1024_20261002/`.

## 2026-10-02 — 1024-panel MH: shear-score noise and stuck chains vs magnitude

The owner proposed MH for bright observations and the old atom method for faint
ones (cut about 20–21), and asked for a 1024-galaxy MH run on 4 GPUs first. No
code changed. Details in
[the 1024-panel section](HYBRID_SAMPLING.md#1024-panel-mh-noise-and-stuck-chains-vs-magnitude).

Run:
- Untempered MH on all 1024 GalSBI panel observations, 32,768 steps.
- 4 GPUs × 6 processes, code frozen at 2758f68.
- Wall time 23–28 min per GPU job (about 6.0 GPU seconds per galaxy).
- Root: `$DATA_DIR/sbsi/catalogue_prior/galsbi_mh1024_20261002/`.

Result:
- Brighter than 22: 5.0% of galaxies, about 88% of the Monte Carlo variance
  of the panel mean.
- Brighter than 23: 12.6% of galaxies, about 98%.
- Stuck chains: 64/129 brighter than 23, 13/163 at 23–24, 1/732 fainter than 24.
- Fainter than 23, Monte Carlo noise is at most 4% of the galaxy-to-galaxy
  spread, and doubling the steps shrinks it by the ideal 1/√2.
- Replica check against the earlier independent 16k run: rms z 1.06 ± 0.02, and
  no per-bin shift.

Validation: jobs 16800304–07 and 16800313, all COMPLETED 0:0. The analysis
asserts that both runs cover galaxies 0–1023 exactly once.

Limits:
- One panel only; the bright bins hold 5–31 galaxies.
- Ten galaxies carry 75% of the Monte Carlo variance.
- GPU types are mixed.

Next: route by step count rather than by method (fewer steps for faint
galaxies, more effort for those brighter than 22–23). The atom/MH prior
mismatch remains open.

## 2026-10-02 — 238-galaxy bias check: tempered vs untempered shear score

The owner asked for diagnosis item 3 on 4 GPUs. No code changed. Details in
[the bias check](TEMPERED_MH.md#238-galaxy-bias-check-tempered-vs-untempered).

Run: all 238 bright galaxies, 32,768 steps.
- Tempered: 8 rungs plus narrow steps, 3 GPUs.
- Untempered: 1 GPU.
- Each GPU ran 6 processes × 8 cores via `jobs/job_galsbi_shared_gpu.sh`, with
  code frozen at 2758f68.
- Root: `$DATA_DIR/sbsi/catalogue_prior/galsbi_bias238_20261002/`.

Result:
- Tempered minus untempered score: weighted -0.24 ± 0.13 (1.9σ), plain
  average -1.80 ± 1.11 (1.6σ).
- Stuck and healthy galaxies give the same weighted shift (-0.19 ± 0.22 and
  -0.26 ± 0.13), so stuck chains do not visibly move the score.
- The plain average is dominated by about 10 galaxies with Monte Carlo errors
  of 26–91 score units. Without them: -0.31 ± 0.57.
- Tempering halves the Monte Carlo error of the plain average (0.48 against
  0.90) at about 5x the GPU time.

Validation: jobs 16800157–60 and 16800163, all COMPLETED 0:0. The comparison
checks that each arm covers galaxies 0–237 exactly once.

Limits:
- One replica per arm, on mixed GPU types.
- The 1.9σ weighted shift is unexplained (chance, or the narrow steps).
- Error bars are about 13% small.

Next: more chains on the about 20 noisiest galaxies.

## 2026-10-02 — Sampler speed: thread fix, faster distance memo, shared GPUs

The owner said 238-galaxy and 1000-galaxy runs are too slow and allowed up to
200 CPU cores (the GPU cap stays at 4). Details in
[Speed](TEMPERED_MH.md#speed-thread-fix-and-shared-gpus).

Changed:
- `scripts/check_galsbi_local_sampler.py`: `torch.set_num_threads(<allotted
  cores>)`. Under `OMP_NUM_THREADS=1` this is what lets xgboost use the cores.
  The distance memo is now a sorted array with exact-match lookup instead of a
  dict loop.
- `jobs/job_galsbi_tempered.sh`: default 16 cores (was 4).
- New `jobs/job_galsbi_shared_gpu.sh`: P processes share one GPU, each with its
  own cores, galaxies and seeds (seed+k, bank seed+k).

Results (238 galaxies × 4 chains, 512 steps):
- One GPU process: 0.154 → 0.070 s per step.
- Four processes sharing one GPU: 0.048 s per step for all 238 galaxies.
- Six processes on a P5000 GPU: 0.044 s per step.
- Identity: every saved array matches the old code (maximum difference 0.0),
  and a shared-GPU process matches its standalone run.
- Projected 32k steps: 1000 galaxies in about 26 min on 4 GPUs × 48 cores
  (before: about 6 h on one GPU).

Validation: Slurm jobs 16800041–45, 16800065–68, 16800079, 16800083,
16800124–25 and 16800133. All COMPLETED except two intentional failures of the
initial-replay check on th-cl-hua28 (16800066, 16800079). Identity was checked
with `compare_arms.py`.

Limits:
- CPU-only jobs have a high per-step floor (0.10 s), so they add little.
- One CPU node (th-cl-hua28) fails the replay check deterministically.
- Shard outputs are separate directories; the analysis must combine them.
- Projections beyond 952 rows per GPU are untested.

Next: launch the 238-galaxy tempered-vs-untempered check (diagnosis item 3)
with the shared-GPU launcher.

## 2026-10-02 — Diagnosis: what limits the bright sampler now

The owner asked what to improve next. This was read-only analysis of the
8-rung and same-seed untempered panel runs (40 galaxies, 2 × 4 chains). Only
docs changed: a new section in
[what to improve](TEMPERED_MH.md#what-to-improve-next-diagnosis-of-the-8-rung-and-untempered-runs)
and the figure `plots/galsbi_improve_diagnosis.png`.

Findings:
- **R-hat fails on spread, not average.** The pass rule fails mostly on the
  spread half of rank R-hat. Score R-hat ≥ 1.01 for 44/80 overall, against
  10/80 for the average half (8 rungs).
- **The average shear score is reliable.** Replica differences: rms 1.14 ±
  0.13 error bars (8 rungs), 1.24 ± 0.14 (untempered). Tempered against
  untempered, galaxy by galaxy: rms 1.07, weighted difference -0.55 ± 0.45
  score units.
- **Noise, not trapping, is the limit.** Per-galaxy Monte Carlo noise is 26–44%
  of the between-galaxy variance. Untempered runs give 1.6 ± 0.6 to 1.8 ± 0.5x
  more precision per GPU second.
- **The GPU is under-used.** 8x the chain-rows cost 3.6x the time.
- **Slowest quantities:** Sérsic, size and magnitude (600–1400 steps per
  independent draw). The score itself takes 29–35 steps.
- **Ruled out:** heavy tails (clipping changes nothing), the estimator choice,
  and a hidden slow part (block-average effective draws match the standard
  count).

Validation: CPU jobs 16799151, 16799358 and 16799375 on `cluster`, all
COMPLETED 0:0. Light summaries were run on the login node from the saved JSON.

Limits:
- The panel was chosen for stuck chains.
- A trap bias up to about 1.4 score units (2σ) is not excluded at this size.
- The GPU packing gain is projected from two timings, not tested.

Next, ranked:
1. Gate on the average part of R-hat plus replica/sampler agreement.
2. Spend GPU on more untempered chains in fuller jobs.
3. Run a 238-galaxy tempered-vs-untempered bias check.
4. Optionally, average each draw over the target's orientation.

## 2026-10-02 — Light-transfer joint move tested on the panel: no gain, code removed

The owner asked for joint target+neighbour sampling. Built
`GalSBIScenePrior.transfer_light` and `--light-transfer-probability`
(commits 4853720, 50eeb5c, b2b72cc).

**The move.** Step the target m_r and let one neighbour within 4" absorb the
flux through a distance weight. Repeating the move with the opposite step
undoes it. Tests:
- exact undo and a Jacobian check;
- toy stationarity within 5σ, failing by more than 10σ without the Jacobian.

With p=0 the output was identical to the unmodified sampler (difference 0.0).

**Panel result, 40 galaxies, 32k steps, against same-seed controls.**
- Plain: 6/40 against 5/40 passes. Log-posterior R-hat 1.149 against 1.137.
  Stuck 17 against 15.
- 8 rungs: 15/40 against 16/40 passes. Log-posterior R-hat 1.042 against 1.037.
  Stuck 9 against 7.

Light-transfer acceptance had a median of about 2%. Accepted target changes
had a median of 0.010 mag. Obs 42 and 155 kept their bright-companion chains
in every arm. No measurable gain, so the code was reverted (sbsi, scripts and
tests now match 60a3646). Tests after the revert: 27 passed (scene prior,
summary, MCMC proposals). Details:
[light transfer](TEMPERED_MH.md#light-transfer-joint-move-no-gain-removed).

Also recorded the 8-rung 32k ladder: 16/40 passes for the same GPU time as
16 rungs at 16k (15/40)
([sparser ladder](TEMPERED_MH.md#sparser-ladder-8-rungs-32k-steps)).

GPU use stayed within 4 inter GPUs throughout: plain jobs followed the ladder
jobs, then tempered jobs, then controls. All jobs exited 0:0.

Next: the bright-companion state (obs 42, 155) and the fainter-target chains
(obs 125, 234) need a move that matches how the measurement model actually
splits light between neighbours. Alternatively, estimate how much mass that
state carries, rather than relying on more proposal tuning.

## 2026-10-02 — Tempering follow-up: more draws help; bright-companion mode found

Owner chose a sparser ladder, longer runs, and a diagnosis of the six
still-stuck galaxies. Doc-only change plus experiment scripts in the data
root: `make_ladder.py`, `stuck_galaxies.py`, `submit_ladder8.sh`, and an
optional draw limit in deep-dive `compare_tempering.py`.

The first-half rerun of panel16k (job 16795404) shows doubling the draws
lowers R-hat for 39/40 galaxies and raises passes from 9/40 to 15/40. Of the
six stuck galaxies, four move between states within the run, and two (42, 155)
stay in a bright-companion mode for the whole run. That mode is also present
in the original sampler: a neighbour as bright as the target about 3" away,
with the target about 0.3 mag fainter. Details:
[follow-up](TEMPERED_MH.md#follow-up-more-draws-and-the-still-stuck-galaxies).

Submitted 8-rung, 32768-step runs on 4 inter GPUs (jobs 16795399–402), plus
dependent comparison 16795418 and trace job 16795419. Results are pending.

## 2026-10-02 — Tempered MH frees the two-mode trap on a 40-galaxy panel

Owner asked to "work it" after the sampling deep dive
(`galsbi_sampling_deepdive_20261002/README.md`). That dive found chains stuck
in either a clean-target mode or a faint-neighbour-shares-light mode, in
85/238 galaxies. Added replica exchange (`--betas`) and a state-independent
narrow-step mixture (`--narrow-step-probability`) to
`scripts/check_galsbi_local_sampler.py`, and
`sbsi.mcmc_proposals.adjacent_swap_permutation` with two tests. One test
recovers the 0.7/0.3 weights of a bimodal toy to within 0.03. Added exact
speed-ups: emulator on all allotted CPU threads, and a PyCosmo distance memo.
Added `jobs/job_galsbi_tempered.sh`. With `--betas 1` the output is bit-identical
to the unmodified sampler before and after the speed-ups (identity2, max
difference 0.0). `tests/test_mcmc_proposals.py`: 10 passed.

Panel results: 16 rungs, 16384 steps, 2 replicas split over 4 inter GPUs, jobs
16795122–5, all 0:0 in 53–57 min.
- Shear-ratio check passes 15/40 (38% ± 8%), against 2/40 for the original at
  32k steps.
- Median log-posterior R-hat 1.04, against 1.25.
- Galaxies with a chain 8+ log units below the best: 6/40, against 32/40.
- Swap acceptance 0.66–0.78.
- Shear score, tempered minus original: -0.01 ± 1.31.

Costs about 7.5x GPU per galaxy, and 25/40 still fail R-hat < 1.01 (mostly
1.01–1.05). Details: [tempered MH](TEMPERED_MH.md#finding--october-2-2026);
figure `plots/galsbi_tempering_panel16k.png`. Next: sparser ladder and longer
runs, or a joint target+environment jump move. No 20k run.

## 2026-10-02 — Deep MH convergence audit isolates geometry and shear-overlap failures

Owner requested a deep investigation. Added `scripts/diagnose_galsbi_convergence.py`,
`probe_galsbi_geometry.py`, `plot_galsbi_convergence.py`, and `probe_galsbi_ridges.py`;
recorded methods, limitations, receipts and next proposal design in
[convergence audit](MH_CONVERGENCE_AUDIT.md#finding--october-2-2026). Linked bright
results and continuous-sampling status. All238 saved observations were audited;
post-hoc geometry panel10, conditional optimized-path panel4. Original bright
runs, immutable source snapshot, frozen models, observations and seeds remain intact.

Exact80-state prior/kernel replay differences0; summary error2.84e-14. Median
morphology acceptance about30% conceals poor movement. In one difficult object,
a straight conditional path drops52.27 log-density units but a curved route
between the same endpoints has no sampled drop. This is geometry evidence,
not posterior samples/global connectivity. Larger Gaussian proposals do not
uniformly improve mobility. Near-shear weights are healthy despite chain
disagreement; atg1=.010,78/128 brightest objects have raw weight ESS<400,
and at.040,89/128. Broad reweighting has a separate overlap problem. No recovered
shear, hybrid acceptance or10-second convergence claim is warranted.

Fixed `scripts/galsbi_bright_math.py::profile_fit` to use the same interpolated
curve for peak and interval crossings; added a narrow off-grid-peak regression
to `tests/test_galsbi_bright.py`. Derived corrected intervals are separate from
original reports and remain scientifically unvalidated. Five focused tests pass;
full scheduled suite16794283:383 passed,1 skipped in101.46s. Ruff passes for the
four diagnostics, helper and test. CPU audit16794252 completed1m49s, GPU geometry
16794276 completed1m54s, profile audit16794287 completed52s, GPU curved paths
16794310 completed1m14s; all0:0. Audit used one GPU at a time and all jobs finished.
Artifacts and executed-script checksums live in `galsbi_convergence_audit_20261002`.
Next: discarded-pilot frozen curved proposals with exact correction and independent
validation, then multiple reference-shear bridges. No20k or automatic longer
sampling run submitted; no failures removed from either cohort.

## 2026-10-02 — Bright MH jobs finish but convergence and shear recovery fail

Owner requested status. Read scheduler accounting and saved reports: both
bright replicas16791908_0/_1 completed0:0 in1h35m10s/1h35m09s, report16791910
in19m13s, and independent throughput16788845 in3h26m. No jobs remain active.
Updated [completed bright results](BRIGHT_MH_TEST.md#completed-result--october-2-2026).

At32k steps, full convergence passes0/128 in both cohorts; both local shear-ratio
checks pass35/128 below20 and10/128 below22. Exploratory conditional fits are
g1=.01410259 and.01077937 against injection.02, but convergence and broader
profile overlap fail, so these are not accepted recovery or bias measurements.
Conditional MH shear MCSE7.66e-4/1.48e-3 dominates measured normalization MCSE
2.75e-5/1.99e-5; selection ladders pass. Noted that the coarse report's mixture
of quadratic peaks and linear interval crossings yields inconsistent intervals
for some replicas. Those intervals are not validated and must be refined before
any future acceptance; original frozen reports remain intact.

Bright sampler cost22.53/22.51 GPU seconds per distinct galaxy at32k; complete
allocated workflows about24.0s including wider profiles/selection. Representative
1024-object16k full-process benchmark costs12.0659s per galaxy, above10s.
Validation: read-only JSON/accounting audit and compact diagnostic count/quantile
calculations. Documentation-only status update; no tests rerun, jobs submitted,
settings changed, or observations removed. The20k run remains paused.

## 2026-10-02 — Bright cohorts and numerical smoke pass; both replicas start

Preparation16791905 completed0:0 in58s:128 observations per measured cut,
238 distinct rows,18 shared. Read-only cohort audit confirms strict<20/<22
membership and no dropped failures. Smoke16791907 passed MH validation,
saved-summary profile replay (maximum error0), and both forward selection
integrals. Both main array tasks16791908_0/_1 are running on inter A40s.
Measurement initialization completed in277.41/275.22s, both initial likelihood
replays have maximum error0, and both reached step128/32768. This is startup
success, not convergence or shear recovery. See
[receipts](BRIGHT_MH_TEST.md#preparation-and-startup-receipts).

Added a bounded CPU status check16792113, scheduled in about2h, to write
`galsbi_bright_mh_20261002/scheduled_check.json`. It reads scheduler state,
progress, failures and any completed report; no chat wakeup tool is available.
The helper compiles and the new protocol's local document anchors validate.
Scientific source snapshot and plan remain unchanged; operational job metadata
is in `jobs.json`. Startup monitoring is complete; long sampling proceeds with
the scheduled check and dependent report, without occupying a chat polling it.

## 2026-10-02 — Launch bright-only continuous MH and selected shear-recovery test

Owner requested tests below measured magnitudes20 and22, including convergence
and shear recovery. Implemented optional cut-specific cohorts in
`scripts/prepare_galsbi_population.py`; added `scripts/galsbi_bright_math.py`,
`scripts/run_galsbi_bright.py`, `scripts/report_galsbi_bright.py`,
`jobs/job_galsbi_bright.sh`, and `tests/test_galsbi_bright.py`. Documented the
protocol in [bright MH test](BRIGHT_MH_TEST.md#owner-request-and-scope--october-2-2026)
and linked the hybrid assessment. No model or inference default changes.

Use128 fresh observations per measured cut, deduplicating shared rows without
altering either cohort; two independent four-chain32768-step runs with8192
warmup. Recovery profiles are conditionalg1 atg2=0, with independent cut-matched
selection populations and explicit covariance/MCSE, overlap, prefix and
selection-ladder checks. Failed objects stay included; unsupported profiles
cannot pass. This is GalSBI model closure, not FS2 recovery or a0.3% claim.

Validation: four focused tests passed; full CPU suite16791497 completed with
382 passed,1 skipped in65.77s. `ruff check` passes (system executable; py31
does not contain the Ruff module); Python compilation and bash syntax pass.
Tests cover strict measured selection and overlap, an analytic selected
Gaussian likelihood, invalid/boundary fits and correlated peak sensitivity.
Frozen225-file source snapshot and immutable plan are under
`catalogue_prior/galsbi_bright_mh_20261002`; job IDs are separate in `jobs.json`.

Submitted preparation16791905, smoke16791907, two-replica array16791908_0–1
(throttle2), and CPU report16791910 with prerequisite dependencies. Preparation
is initially Priority-pending; main jobs cannot start before numerical smoke.
At most two new GPUs plus the existing throughput GPU overlap, within the
four-GPU cap. The20k population run remains paused. Monitor preparation and
smoke through startup/completion; numerical or scientific failures are results
to diagnose, not permission to discard galaxies or force shear recovery.

## 2026-10-02 — Evaluate atom integration with a continuous-MH fallback

Owner requested a deep assessment while the throughput study runs. Added
`doc/HYBRID_SAMPLING.md` with the common-prior derivation, observation-dependent
prior-splice normalization, separate retrieval/discretization/MH errors,
brightness-plus-diagnostic routing, shared-bank/selection error accounting,
cost arithmetic and a bounded proposed comparison. Linked it from
`doc/CONTINUOUS_SAMPLING.md`. Reusing the old algorithm with a new common-prior
bank is distinguished from directly joining FS2 and GalSBI populations.
No inference code, model, routing default or scheduled science settings changed.

Evidence review: actual20k has135 measured-mag<20 and909 mag<22 rows; at the
unverified10 GPU-second target, their two-replica MH components cost0.75 and
5.05 GPU hours, excluding atom integration, bank preparation and normalization.
Light inspection of the1024-row saved GalSBI measurement file finds5 and51
rows respectively. Historical bright-extension results are labelled V3.6 and
three-case diagnostics, not V3.9 validation. Reviewed pop-cosmos v2 inference
and the rank-normalized Rhat reference; both are linked in the assessment.

Scheduler accounting confirms ensemble16788846_0–3 and report16788848 all
completed0:0. At16k steps, control obs1 ratio Rhats1.07745/1.04877 improve to
1.00445/1.00710 with stretch, but stretch obs0 worsens to1.05724/1.05999 and
obs1 bulk ESS remains346/265. No uniform mixing or shear-accuracy acceptance.
Throughput16788845 completed measurement initialization in1191.16s, passed
initial likelihood replay with maximum error0, and reached step346/16384;
full-process timing remains pending. Existing scheduled check16788875 remains;
no additional jobs launched and20k remains paused.

Validation: small saved-JSON/Parquet inspections and explicit count/cost
arithmetic; local Markdown link/anchor checks and `git diff --check` pass.
Documentation-only change, so no numerical tests rerun. Next scientific step
is common-prior bank consistency and paired method validation, including a
representative faint audit; this assessment does not certify either sampler.

## 2026-10-02 — Distinct-observation cache replay passes; main benchmarks released

Paired128-observation job16788841 completed0:0 in6m44s. All retained latent
states, MH decisions, log priors/kernels and all256 local shear profiles agree
exactly between cache-off/on. Maximum summary difference4.26e-14 is within
the unchanged1e-10 tolerance. Initial replay is exact in both arms. Sampling
time is25.097s versus18.525s (26.2% less); full256-step processes are204.806s
and194.07s. Baseline measurement initialization costs156.11s, or1.22s per
galaxy. This short run does not establish the full16384-step throughput target.

Four-arm startup16788842 completed0:0 in1m56s; both independent control and
stretch seed pairs passed numerical receipts. Main control16788846_0 has
passed warmup and made substantial retained-sampling progress. Remaining
ensemble tasks and full1024-observation throughput16788845 are scheduler
Priority-pending, with successful prerequisite dependencies cleared. Queue
waiting has become prolonged; no duplicate jobs or scientific changes made.
CPU ensemble report16788848 remains dependent. One bounded Slurm CPU status
check16788875 is scheduled about2h later to inspect accounting/progress/errors
and write `galsbi_throughput_20261002/scheduled_check.json`. No Codex chat-wakeup
tool is available, so this records cluster status without promising a chat
notification or autonomous repair. Both check files pass syntax validation.
The20k array stays paused. See
[bounded experiment](CONTINUOUS_SAMPLING.md#bounded-throughput-and-hybrid-ensemble-experiment--october-2-2026).

## 2026-10-02 — Launch bounded full-process throughput and hybrid-ensemble comparison

Owner authorized proceeding. Added complementary morphology stretch proposals
to `sbsi/mcmc_proposals.py` and opt-in `--ensemble-morphology` plus stage timings
to `scripts/check_galsbi_local_sampler.py`. The3D stretch move uses frozen
complementary walkers and the exact2log(z) MH correction, retaining full-support
local and original neighbour moves. Models, target density, support, numerical
tolerances and default proposals are unchanged. Interacting walkers are flagged;
`scripts/report_galsbi_sampler_runs.py` rejects independent-chain treatment.
`scripts/report_galsbi_ensemble.py` instead computes expectation diagnostics
after averaging walkers within each independent ensemble. Added mathematical
closure and covariance tests; fixed the scalar-conversion warning in
`scripts/report_galsbi_population.py` without changing its calculation.

Added `scripts/run_galsbi_throughput.py` and `jobs/job_galsbi_throughput.sh`
for full subprocess timing, distinct-observation cache replay, four-arm smoke,
and a matched-seed original/hybrid proposal pilot. Preserved sources and plan
under `catalogue_prior/galsbi_throughput_20261002` (156 files). Preparation
16788816 generated1024 independent selected observations in26s; geometry
error2.22e-16 and full/partial/order likelihood replay errors all zero.
Initial suite16788817:376 passed,1 skipped; final suite16788840 including
new ensemble-report tests:378 passed,1 skipped in55.21s. Ruff and Python/shell
syntax checks pass. No heavy work ran on the login node.

Submitted paired cache16788841 and startup16788842, both observed running
on inter A40s; first control startup completed with valid numerical checks.
Throughput16788845 follows successful cache replay; ensemble16788846_0–3
follows successful startup with throttle2; CPU report16788848 follows any
ensemble outcome. At most three GPUs can overlap, below the four-GPU cap.
Full run16384 steps/4096 warmup/four chains; timing includes initialization,
sampling, output and local profiles. The two-replica proposal pilot uses
the original four-observation stress panel, including the unresolved obs1.
The20k launch remains paused. No measured10-second full-process result or
accuracy acceptance yet; monitor dependent stages through startup. See
[bounded experiment](CONTINUOUS_SAMPLING.md#bounded-throughput-and-hybrid-ensemble-experiment--october-2-2026).

## 2026-10-02 — Assess pop-cosmos paper and adopt initial10 GPU-second target

Owner set10 seconds per galaxy per GPU as the first target and requested
reading arXiv2602.03930. Reviewed its v2 inference section, initialization
and convergence-sensitivity footnotes, and the earlier method paper
arXiv2406.19437v2. Recorded applicability and limits in
[throughput assessment](CONTINUOUS_SAMPLING.md#october-2-ten-gpu-seconds-per-galaxy-and-the-pop-cosmos-comparison).
The new milestone supersedes the interim0.1 core-second target. Retain the
earlier per-replica interpretation and count initialization/output in the
budget. The20k population array remains paused and was never submitted.

Recomputed costs directly from `galsbi_cost_20261001/cuda/result.json`:
at1024 observations,16384 steps give9.24s per galaxy in measured components;
cached-component subtraction estimates7.27s. Neither includes the entire
run or establishes mixing. CPU response and scene inputs account for58.5%
and17.5% of baseline component time, versus2.1% for the neural likelihood.
Affine-invariant target proposals are a possible future controlled comparison,
not an implementation change or permission to replace the frozen models.
Plan receipt updated with the initial speed target. Scheduler check shows no
active/pending user jobs; no new jobs or code changes, so no tests rerun.

## 2026-10-01 — Prepare 20k population diagnostic; pause for owner throughput target

Recent-job check: both64k measurement-start replicas16788678_1/_2 and
report16788679 completed successfully. Combined worst Rhat/minimum bulk ESS
at65536 steps for obs0–3 are1.09042/67.0,1.07064/93.7,1.02013/441.1,
1.03448/289.9; all numerical checks pass, but none passes all exploration
checks. Long array16781571 and report16781574 also completed; at1048576
steps combined obs0/2/3 pass, obs1 fails atRhat1.04247/ESS298.5, including
shear-ratio Rhat1.01513/1.01799. GPU cache replay16787894 passes with identical
latent states and MH decisions. GPU cost16787881 passes the baseline replay;
the alternative CUDA response predictor fails log-kernel replay at512/1024
observations (9.13e-5/4.93e-4), despite small coefficient differences.

Owner requested20k observations and selected16384 steps,4096 warmup, four
chains per observation and two independent replicas. Added
`scripts/prepare_galsbi_population.py`, `scripts/run_galsbi_population_shard.py`,
`scripts/report_galsbi_population.py`, and `jobs/job_galsbi_population.sh`.
The existing mock generator gains optional full measured-selected population
retention; the local sampler accepts a separately validated population
reference with observation checksums and fresh initialization banks. Original
pilot defaults, model artifacts, likelihood, proposal mixture and tolerances
are unchanged. The planned population is newly model-generated with the same
measured cuts, without truth cuts or magnitude-stratum selection.

The reporter compares independent-replica numerator log profiles atg1=.018/.022
and8k/16k prefixes, retaining within-shard and between-shear-point covariance
in conditional MCSEs. It rejects missing shards, coverage mismatches and failed
numerical receipts. It does not infer selected-population shear: the identical
selection normalizer cancels only in replica differences. Two replicas and
prefix agreement cannot rule out shared missed-mode or finite-log bias.
Tests in `tests/test_galsbi_population_report.py` check correlated-noise MCSE,
nonfinite draws, incomplete coverage and full-population measured selection.

Validation: Python compilation, bash-n and Ruff pass. Full suite16788794
completed0:0:374 passed,1 skipped in65.26s, with two NumPy scalar-conversion
deprecation warnings in the new report. Frozen153-file preparation snapshot
and plan are under `catalogue_prior/galsbi_population20k_20261001`.
Preparation job16788795 started and was cancelled after16s when the owner
rejected the36–48h estimate on four GPUs. No main array, smoke or report jobs
were submitted; no completed20k reference or sampling results exist.

Owner subsequently clarified the speed target to approximately10 galaxies
per second per core, replacing the interim10 seconds per galaxy per replica
interpretation. CPU-core versus GPU-worker scope is pending clarification.
Keep population execution paused while reassessing throughput and the method;
do not present tiled component estimates as full-process performance. See
[population throughput target](CONTINUOUS_SAMPLING.md#population-diagnostic-and-throughput-target--october-1-2026).

## 2026-10-01 — Authorize and launch 64k measurement-initialized MH comparison

The 16k initialization comparison completed all four workers in two waves,
about13min total; report16788283 completed. Across two replicas, prior-bank
versus measurement-start worst Rhat by observation was1.4427/1.1643,
1.6090/1.4959,1.1245/1.0703,1.0874/1.0494. Minimum bulk ESS remained below400
for all four observations in both arms. Better starts help but do not establish
convergence. No accepted shear estimate or sampler promotion follows.

Owner authorized two measurement-start replicas at65536 steps, retaining4096
warmup, the same four observations, four chains per observation, MH seeds
2026100141/142 and bank seeds2026100241/242. These replay the original starts
and then extend the trajectory; they are not additional independent replicas
of the earlier same-seed runs. Original frozen150-file source snapshot is
reused unchanged; models, proposal mixture, adaptation and cache-off setting
are unchanged. New external root `catalogue_prior/galsbi_measurement_init_64k_20261001`
contains the plan, launcher, snapshot link and automatic report launcher.
No active library code changed. Both shell launchers pass bash-n.

GPU array16788678 (tasks1/2, throttle2) requests one inter A40 per task and
40min walltime, without unrelated dependencies. No other user GPU jobs were
active at submission, within the combined four-GPU cap. CPU report16788679
on cip-cl-compute3 follows after any array outcome and checks16384/32768/65536
prefixes, including shear likelihood ratios. Expected elapsed is roughly25min
on two GPUs based on the six-minute16k runs; this is an estimate, not a bound.
Both tasks started on inter kng-cl-nv02, passed initial likelihood replay
with exactly zero error, and reached896/65536 steps without startup errors.
Final convergence results remain pending.

## 2026-10-01 — Release initialization comparison alongside old MH runs

Owner authorized running the two inter initialization workers alongside the
two old cip workers, raising this continuous-prior study's combined cap to
four GPUs. Recorded the scope in `AGENTS.md` and both external study plans.
Removed the artificial old-long/cache dependencies from array16788277;
throttle2 remains. Smoke16788261 was independently verified COMPLETED0:0.
Slurm rejected reattaching this already-finished smoke dependency, so cleared
the dependency field after verifying its successful accounting record.
The released array initially showed Priority rather than Dependency.

Cost job16787881 still waits for old task16781571_2 to release one slot;
cache16787894 follows the cost job. With at most two initialization workers
and the remaining old task3, this schedule stays within four GPUs. Report
dependencies, seeds, sources, models and numerical settings are unchanged.
Validation: sacct smoke status and scontrol/squeue dependency/resource audit;
no code changes or tests required. Tasks0/1 started on inter kng-cl-nv02;
both initial likelihood replays have exactly zero error (tolerance1e-6), and
both reached step1920/16384 without errors. Both old cip tasks were still
running, confirming four simultaneous GPUs. The second pilot pair and final
comparison remain pending; startup success is not a convergence result.

## 2026-10-01 — Move queued continuous-sampler GPU stages to inter

Owner requested trying `inter`. Scheduler inspection found four unallocated
A40s on kng-cl-nv02; the queued study stages were dependency-blocked rather
than waiting on GPU capacity. Updated existing jobs16787881 (cost benchmark),
16787894 (paired cache replay) and16788277 (initialization array) in place to
`Partition=inter TresPerNode=gres:gpu:a40:1`. Verified all three partition/GRES
requests and existing dependencies with scontrol; array throttle remains2.
Job ids, seeds, frozen sources, scientific settings and dependent reports are
unchanged. Running cip long jobs are not interrupted, and the combined two-GPU
limit still holds through the preserved dependencies.

Updated both study plan receipts under `galsbi_cost_20261001` and
`galsbi_measurement_init_20261001`. The first TRES spelling with a slash was
rejected after Slurm applied the partition change; corrected immediately to
the colon form and verified the final resource requests. No duplicate jobs
were submitted. All moved jobs are still pending upstream completion; check
their resource receipts and numerical replay when they start. Benchmark
timings on inter full A40s must not be presented as a controlled speedup over
cip virtual8GB A40 allocations. No code or model changes; no tests rerun.

## 2026-10-01 — Measurement-informed versus prior-bank initialization pilot

Owner explicitly requested trying starts near the measured properties. Added
`sbsi/galsbi_initialization.py` and the optional `--initialization measurement`
flag to `scripts/check_galsbi_local_sampler.py`. Candidate initial magnitude
comes from measured flux; radius uses a rough Moffat-PSF quadrature subtraction;
shape uses approximate inverse shear plus broad epsilon scatter. Four streams
have different Sérsic centres. Native redshift/SED and neighbour scenes come
from fresh prior banks. Candidate selection uses target-native log prior plus
the unchanged full likelihood; generating truth is never read. No guess
floor/clamp/scatter restricts subsequent MH support. The default initializer
is unchanged. New `initial_state.npz` and initialization receipts preserve
coordinates, summaries, log densities, neighbour counts, candidate counts and
preparation time. Reporter compatibility includes the initializer setting.

This compares initializer packages: the new arm changes both candidate
centring and ranking from the original highest-likelihood prior-bank control.
It cannot isolate those contributions. Both arms score2048 candidates per
chain and retain the same target, original proposals, warmup/adaptation,
4096-row density calls and numerical tolerances. Invariant-work reuse stays
off in both arms. Four fixed observations; four chains per observation;
MH seeds2026100141/142; bank seeds2026100241/242;16384 total steps with4096
warmup. Task order counterbalances the two arms over two seed pairs/waves.

Added `tests/test_galsbi_initialization.py` for measurement units, inverse
shape conventions, unchanged nuisance coordinates and boundary/support
handling. Full suite16788224:370 passed,1 skipped in67.59s. Ruff, Python
syntax and shell syntax checks pass. CPU startup16788261 exercises all four
arm/seed combinations at64 steps/32 warmup: completed3m47s, peakRSS1.04GB.
All four initial replays are exact; SED logK replay is exact and summary
differences are at most2.84e-14. Short-chain convergence is deliberately not
a startup gate.

Frozen150-file snapshot and plan:
`catalogue_prior/galsbi_measurement_init_20261001`. Launcher
`jobs/job_galsbi_initialization.sh` runs the study; GPU array16788277 has
four tasks with throttle2 and waits for successful startup and completion of
long array16781571 and cost/cache GPU continuation16787894. This preserves
the two-GPU study cap. CPU report16788283 runs after any pilot-array outcome.
`scripts/report_galsbi_initialization.py` generates within-arm prefix reports
and cross-arm likelihood-ratio agreement, initial-state and timing evidence.
It does not assume independent arm MCSEs under matched seeds. Startup has
been monitored through completion; GPU stages remain dependency-pending.

See [initializer comparison](CONTINUOUS_SAMPLING.md#measurement-informed-initialization-comparison--october-1-2026).
Higher initial likelihood is not posterior convergence. Retain Rhat<1.01,
bulkESS>400 and all numerical checks, including shear-relevant ratios; no
automatic promotion, population-scale launch or selected-shear claim.

## 2026-10-01 — Cost audit and validated CPU savings from invariant MH work reuse

Owner requested a deep efficiency investigation after the million-observation
cost estimate. Read the existing long-chain/mixing evidence and profiled the
actual model workload. Added `scripts/benchmark_galsbi_sampler.py` and
`jobs/job_galsbi_benchmark.sh`: controlled tiled-scene component timings,
identical-proposal MH replay, fixed4096 density batches, model/source receipts,
and CPU1/CPU4/CUDA response comparisons in the pending GPU arm. Warmup-only
cProfile does not contaminate reported timing repeats. These probes do not
establish convergence or representative million-observation throughput.

CPU job16787846 completed in2m09s, peakRSS928MB. For4/32/128 observations
(four chains each), weighted baseline component cost is249/276/369ms per
step. CPU neural density stays231ms due to padding, while response prediction
grows4.66/20.76/76.12ms and scene-input mapping6.66/12.24/31.39ms. All batch
and invariant replay checks pass, with no changed MH decisions. GPU timing
cannot be inferred by dividing this CPU measurement by an assumed factor.

Added opt-in `--reuse-invariant-work` to `scripts/check_galsbi_local_sampler.py`:
reuse K on z/SED-only moves (25% of the original mixture), reuse target prior
on neighbour-only moves, and avoid copying unchanged neighbours after target
moves. Full native support/Jacobians/Hastings factors remain active. Guard
requires the established persistent/joint-morphology configuration; invariant
blocks are periodically fully replayed at1e-10 summary/1e-6 logK tolerances.
The flag remains off by default. Reporter protocol compatibility now includes
the flag. Existing long jobs use frozen sources and are unchanged.

Added a full-scene redshift/SED invariance test with nontrivial photometry,
plus `scripts/check_galsbi_cache_replay.py` for paired512-step real-model runs.
Full suite16787870:368 passed,1 skipped in60.77s; Ruff and launcher syntax
checks pass. CPU replay16787880 failed safely against old GPU-derived starts
(2.35e-5 versus1e-6 tolerance). Retried with a fresh bank seed2026100133
evaluated on the same device; no tolerance was relaxed and failed outputs
remain preserved. Retry16787889 passed: exactly identical retained latent
states, MH decisions, log priors/kernels and local profiles; summary error
at most2.84e-14. Whole processes164.41s→132.82s; steps128–512 give249.74ms→
187.80ms, a24.8% loop-time reduction on CPU. The first frozen wrapper's
`sampling_seconds` includes initialization; current source instead derives
loop cost from progress differences. No GPU speedup or mixing improvement
is claimed from this test.

Study root `catalogue_prior/galsbi_cost_20261001` contains frozen sources,
plans, profiles and receipts. GPU benchmark16787881 waits for long task
16781571_2, preserving the two-GPU cap, and compares batch sizes through1024
observations and response devices. GPU paired replay16787894 follows the
benchmark and successful CPU paired replay. Startup was verified for the CPU
jobs; GPU stages remain dependency-pending. No automation wake-up tool is
available; dependencies provide execution, not an automatic agent review.

See [cost audit](CONTINUOUS_SAMPLING.md#cost-and-scalability-audit--october-1-2026)
for measured component costs, limitations and proposed next experiment.
Curved morphology and conditional environment trapping remain sampler issues;
prior pooled covariance/broad joint moves did not solve them. A small frozen
mixture independence proposal with an explicit MH correction is the proposed
next morphology test, not an implemented or accepted replacement. Production
also needs streamed summaries and a shear-level Monte Carlo error budget,
including finite log/ratio bias. No population-scale launch or0.3% claim.

## 2026-10-01 — Long-chain interim check: three observations pass within both completed replicas

Owner requested status. Tasks16781571_0/_1 completed with exit0 in5h10m32s
and5h10m57s. Their1,048,576-step results pass individual exploration gates
for obs0,2,3, but fail obs1 (measured MAG_AUTO21.2551). Seed2026100121
worst Rhat/minimum ESS by observation is1.00292/842.9,1.12420/24.7,
1.00280/855.4,1.00139/2664.3; seed2026100122 gives1.00328/703.2,
1.05566/54.1,1.00107/2429.8,1.00775/732.9. These are separate four-chain
replica checks, not the pending combined-replica result. New initial banks
also differ from the shorter historical runs, so this is not a paired
causal comparison of chain length alone.

Obs1 remains slow in radius/Sérsic/magnitude/crowding. Its two local log-ratio
Rhat values are1.0180/1.0257 and1.0336/1.0397 in the completed replicas;
the shear-relevant functions therefore also fail. Initial replay is exact.
SED invariance remains within the declared tolerance: maximum summary
change4.2633e-14, maximum log-kernel change9.3599e-7 in seed121 and zero in122.
Do not describe every SED evaluation as exactly unchanged.

Tasks2/3 are running normally at about685k/683k of1,048,576 steps, with
exact initial replay. Current throughput implies roughly two more hours,
including final diagnostics. Report16781574 still waits on the array.
No intervention, acceptance, or new job submission; final combined checks
and prefix diagnostics remain pending.

## 2026-10-01 — Owner-authorized million-step original-MH replicas

Owner explicitly requested longer chains after the runtime discussion. Use
the original per-chain3D morphology tuning, corrected SED jump and persistent
Poisson refresh; disable pooled covariance, joint-environment and local-
position variants. Fixed four-observation panel, four chains per observation,
four replicas with MH seeds2026100121–2026100124;1,048,576 total steps and
32,768 discarded warmup per replica. Run two GPUs concurrently, in two waves.
Measured original runtime implies roughly5.2 hours per wave and10.4 hours
overall, excluding queue/preparation/report time. Six-hour limit per GPU job;
24GiB host memory allows the longer unthinned traces. No batching change is
mixed into this numerical convergence experiment.

Added optional `--initial-bank-seed` to the existing runner. Each replica
draws8192 new prior scenes, splits them into four2048-scene streams and
chooses the highest-likelihood scene from each stream for each observation.
Initial bank seeds2026100221–2026100224 differ across replicas; generating
truth and mock observations are never regenerated or used as starting
states. The full-bank likelihood is replayed after selecting/reconstructing
the16 starting scenes, at the unchanged1e-6 tolerance. Save bank seed,
selected indices and likelihoods in `initialization.json`. Default behavior
still reconstructs the original reference bank.

Existing launchers now accept seed base, initial-bank seed base and warmup;
the report launcher accepts replica count and includes longer prefix checks.
No proposal, density or acceptance changes. Ruff/shell syntax pass. Full
suite16781551 submitted on compute3. Startup array16781558 depends on its
success, checks all four new banks with128 steps/64 warmup, and is throttled
to two GPUs. Smoke convergence is not an acceptance condition; execution,
likelihood replay and SED invariance are. Long-run submission follows these
checks. Source snapshot manifest SHA256:
`098c3993f7c6c1601d131b16ffcdbdd7b9d6db97e67e1bc0f5af8526630d4b8e`.
Protocol/output root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_long_mh_20261001`.
Post-run prefix diagnostics use65k/131k/262k/524k/1,048,576 steps; these are
computed after completion, not live convergence stopping. Gates remain
Rhat<1.01 and bulkESS>400 with independent and combined checks. No automatic
promotion, population run, selection normalization or shear-bias claim.

Full suite16781551 passed367 tests/1 skip in71.40s. All four smoke tasks
16781558 completed with exit0: starting likelihood replay error is exactly
zero for every bank; SED log-kernel change is zero and maximum summary
change2.8422e-14. The new-bank path therefore passed its scheduled runtime
validation in addition to the existing suite. GPU array16781571 is running
tasks0/1, with2/3 throttled behind them. Both running replicas passed exact
initial replay and reached768 steps with finite progress. Afterany CPU
report16781574 has32GiB/45min for the four larger traces. Sources and
scientific settings are frozen. No application wakeup tool is available;
Slurm enforces the sequential two-GPU waves and runs the completion report.

## 2026-10-01 — Pooled and joint-environment pilots completed; neither accepted

Owner requested a check. Scheduler accounting confirms pooled array16781077
completed in20m27s–20m31s and joint-environment array16781117 in20m41s–21m01s,
all exit0. Reports16781079/16781121 and comparison16781124 completed; user
queue is empty. Both variants and both seeds pass initial/SED numerical
checks, but every individual run and combined observation fails at least
one exploration gate. No sampler promotion or further job submission.

At matched65536 total steps/32768 warmup, worst combined Rhat for obs0–3:
original1.3009/1.3116/1.0489/1.0264;
pooled1.1175/1.4171/1.0324/1.0438;
pooled+joint1.2918/1.2254/1.0221/1.0211.
Minimum bulkESS respectively20.8/20.3/145.5/422.4,
46.0/17.0/213.9/146.4, and22.0/25.9/323.3/346.2.
Pooled tuning gives mixed results; joint moves improve several Rhat values,
but the faintest ESS remains below the original matched-budget result.
These two-seed development runs do not establish general superiority.

The joint move is often inefficient: faint-observation per-chain acceptance
is about0.1–3.0%. For obs1 one seed2026100101 chain accepts0/1966 joint
attempts. This supports the limitation of broad Gaussian target moves plus
an independently redrawn environment; formal invariance alone does not make
the proposal effective. Bright radius/Sérsic and obs1 magnitude/shape remain
slow. Joint-run profile Rhat atg1=.018/.022 is1.0527/1.0482 for obs0,
1.1026/1.1106 for obs1,1.01014/1.00986 for obs2 and1.00333/1.00334 for obs3.
Passing one ratio diagnostic does not override posterior failures.

Preserved comparison:
`galsbi_joint_env_mh_20261001/report/comparison.md` and `.json`, under the
catalogue-prior output root. No selected-population shear estimate or0.3%
claim. Before another scale-up, any next proposal should address the observed
curved morphology and conditional target/environment dependence, with a
small fixed-budget test and the unchanged acceptance gates.

## 2026-10-01 — Conditional environment trapping and a joint-move pilot

Owner requested continuation. Pooled array16781077 is still running; do not
interpret warmup acceptance as its result. During that run, inspected the
overnight chains using scheduled plot16781089 and tail audit16781093. Scripts,
figures (`geometry.png`, `traces.png`) and JSON are preserved in
`galsbi_mixing_audit_20261001`. Used the Matplotlib skill for these static
diagnostic plots. Plotting thins every32nd draw and trace panels show512-step
means; neither reduction is used for inference or convergence diagnostics.

The observed bright morphology distributions are curved, with directions
changing across Sérsic index; a single larger covariance is not guaranteed
to resolve them. Fainter observations have long excursions toward targets
much fainter than their measured magnitudes. Audited the diagnostic region
true r > measured MAG_AUTO+1 (no samples removed or new population cut).
For obs2, full environment refresh acceptance in this region is2/638 and
1/104 across the two seeds, versus roughly28–29% outside. Obs3 has1/41 and
3/418. Thus the earlier high aggregate acceptance conceals conditional
environment trapping. Tail occupancy ranges0.1–9.3% across obs2 chains and
0.16–11.0% across obs3 chains; these are unconverged occupancies, not inferred
population fractions. Absolute weighted neighbour-flux diagnostics remove
the target-flux denominator algebra: log(expm1(crowding))-0.4log(10)m_r.
Residual magnitude correlations remain substantial (roughly0.3–0.65), so
the apparent relationship is not solely the definition of flux ratios.

Added opt-in `--joint-environment` to the existing pilot, requiring the joint
morphology setup. A fixed20% of morphology attempts use a4-times broader
symmetric target step together with a full independent environment-prior
draw. They count within the existing block budget, use one likelihood
evaluation, and do not tune the scalar scale used by local moves. The
environment proposal cancels its own prior; acceptance keeps the full
target-coordinate prior and likelihood ratio. Whole scenes are retained on
rejection. No analytic neighbour-density approximation, truth cut, or
state-dependent decision to activate the move is introduced.

Save joint-move indicators and separate local/full/joint acceptance. Reporter
compatibility now includes the joint-move protocol. The existing launcher
accepts an optional joint-environment flag with defaults unchanged. Added a
joint target/Poisson-environment closure test against numerical quadrature,
including target/count cross moments and off-support rejection. Ruff and
shell syntax pass; full suite16781100 runs on compute3. Planned second pair
uses the same65k/32k protocol and seeds with pooled tuning plus this move,
after the first pair/report and passing tests, within two concurrent GPUs.
No acceptance or population-sized run follows automatically.

Suite16781100 completed:367 passed,1 skipped in53.58s. Frozen joint-pilot
source manifest SHA256 is
`bdb370a1d9cfcdab6b00816dca40461059495f16d23fa41dd5e598c1079c19d8`.
Queued array16781117 after the passing suite and pooled report16781079;
the two study GPUs cannot overlap the preceding pair. Afterany report16781121
and read-only fixed-budget comparison16781124 follow. Joint output root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_joint_env_mh_20261001`.
Its preserved `compare_reports.py` writes `report/comparison.md` and JSON
from the three source reports at65k/32k. Missing runs are recorded and no
sampler is promoted. The original and pooled pilots remain untouched;
all variant sources and random seeds are frozen separately.

## 2026-10-01 — Audit slow directions and test staged pooled morphology tuning

Owner requested a deeper sampler investigation and improvement. Overnight
receipts show thousands of full environment proposals, accepted roughly
27–75% for seed2026100101; simple failure to replace neighbours is therefore
not an adequate explanation for the remaining slow exploration. This does
not rule out rare unvisited environmental configurations. Started a scheduled
read-only audit of morphology proposal covariance relative to sampled
posterior covariance, and within-chain target/environment correlations.
Audit script and output are preserved under `galsbi_mixing_audit_20261001`.

Prepared opt-in `--pooled-morphology` in the existing local sampler and
`pooled_block_cholesky` in `sbsi/mcmc_proposals.py`. Pool retained warmup
states across the four chains of each observation only, using at most8192
steps, update every1024 steps, and freeze covariance halfway through warmup.
Continue scalar step-size adaptation through the remaining warmup, then
freeze all proposal parameters. The physical block remains magnitude,
log circularized radius and Sérsic logit, with5% diagonal shrinkage. No
likelihood, prior, environment law, block probability or acceptance rule
changes; no shared covariance between observations. This tests the suspected
short-window/per-chain tuning limitation, not a new population model.

Added an independent covariance calculation checking between-chain variation
and observation separation, plus a correlated-Gaussian MH closure test using
displaced warmup chains. Ruff passed. First CPU audit16781055 and suite16781064
were stuck before Python in `autofs_wait` on cip-cl-compute1; cancelled only
these two jobs and resubmitted on compute3 as16781065/16781066.

Predeclared comparison: if tests pass, run65536 steps/32768 warmup on all four
reference observations with seeds2026100101/2026100102 and original starting
scenes, matching the existing overnight65k prefixes. At most two GPUs.
Compare posterior and likelihood-ratio diagnostics for both seeds, not a
favorable shear point estimate. No automatic promotion or population run.

Audit16781065 completed in8s. It compares actual frozen proposal covariance
(including scalar multipliers) with each chain's observed post-warmup
morphology covariance, using eigenvalues after posterior whitening. For
obs1 the smallest ratios are0.00220–0.01777, with several0.002–0.007:
proposal standard deviations can be only5–8% of the sampled posterior width
along a slow combination. For obs0 the smallest ratio reaches0.00247.
This supports inefficient local proposal geometry, but sampled posterior
covariance is itself not a converged reference. Bright radius/Sérsic
correlations reach−0.81; faint magnitude/crowding reaches0.97 (partly the
algebra of target-relative flux). No claim that high environment acceptance
rules out rare tails or that covariance tuning alone solves the problem.

Full suite16781066:366 passed,1 skipped in77.16s. Ruff and launcher shell
syntax pass. Reused the overnight launcher with optional step count and
pooled flag, preserving defaults. Frozen143-file checkout manifest SHA256
`6f65101d430c897e5323dd28d0b113a62db597ca58bbdf17b60da416cf0f9a4e`.
GPU array16781077 runs the declared pair on two cip A40 GPUs; dependent
afterany report16781079 is assigned compute3, avoiding the observed mount
problem. Output root is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_pooled_mh_20261001`.
Separate seeds share covariance only among warmup chains for the same
observation within a run. Each proposal is symmetric after freezing, so no
new Hastings factor is required. The fixed experiment remains unaccepted
pending its individual/combined diagnostics.

## 2026-10-01 — Overnight comparison completed; convergence still fails

Owner requested status. Scheduler accounting confirms both tasks16778955
completed with exit0 in1h18m31s/1h18m35s; dependent report16778957 completed
in2m57s. No jobs remain in the user queue. Read the preserved report JSON
and Markdown under `galsbi_overnight_20261001/report`; both numerical audit
receipts pass and the runs are compatible, but neither individual run nor
the combined panel passes all exploration checks.

At262144 total steps, combined worst Rhat/minimum bulkESS by observation:
0:1.0301/381.9;1:1.0622/165.9;2:1.0112/552.2;3:1.0170/508.6.
Required Rhat<1.01 and bulkESS>400 remain unchanged. Main slow summaries
are circularized radius for obs0, Sérsic/magnitude for obs1, and
crowding/magnitude for obs2–3. The endpoints improve over the65k prefix,
but the faintest observation worsens at131k before improving: short-run
agreement cannot establish convergence.

Combined likelihood-scale ratio Rhat atg1=.018/.022 passes for obs0,2,3
(approximately1.006/1.003/1.003), but fails for obs1 (1.0125/1.0162).
Conditional relative MCSE is0.14–0.17% for the brighter pair and
0.008–0.014% for the fainter pair. These are errors of individual-observation
numerator ratios, not shear bias or guaranteed error bars for unconverged
chains. Largest individual draw fraction is at most5.51e-6 in the final
combined profiles: no observed single-draw dominance, without ruling out
unvisited tails. Bright obs0'sg1=.022 log mean ratio changes+0.00215 at131k
to−0.000944 at262k; its small profile slope remains numerically uncertain.

No acceptance, model change or population-scale launch. Next bounded
diagnostic should examine the slow magnitude/size/Sérsic and environment
directions on this existing panel before increasing observation count.
The missing selection normalization and population-law limitations remain.

## 2026-10-01 — Frozen overnight sampler comparison launched

Completed the optional local-position pilot16778924 in270.97s after
suite16778902 passed362 tests/1 skip. Bright maximum Rhat1.3893 and
numerator-ratio Rhat1.0537; faint maximum1.0634 and ratio1.0261. High local
position acceptance does not establish convergence. Neither this variant nor
the five-coordinate block consistently improves the simpler three-coordinate
joint morphology proposal. No pilot passes the full convergence gates.

The overnight protocol therefore retains the SED jump, three-coordinate
joint morphology, separate shape block, exact type sum and persistent Poisson
environment refresh. No local-position or five-coordinate proposal is used.
Added saved local profile draws and the read-only
`scripts/report_galsbi_sampler_runs.py`, with likelihood-scale MCSE and
largest-draw contributions, individual and combined-seed diagnostics at fixed
prefixes, and explicit incomplete-run handling. Tests cover missing/corrupt
receipts, disagreeing replicas, an IID lognormal MCSE oracle, and the complete
combined-prefix report path. Latest suite16778930:364 passed,1 skipped
in51.77s; Ruff and launcher shell syntax passed.

GPU array16778955 started both tasks on separate cip A40 allocations, after
the passing suite. Each uses four observations, four chains per observation,
262144 steps/32768 discarded warmup, with seeds2026100101/2026100102.
The seeds share the same dispersed reference-prior starting scenes; they do
not independently regenerate initialization. Both initial likelihood replays
have zero error, and both passed8576 steps with ongoing accepted SED-region
transitions. Adaptation is restricted to warmup. Two concurrent GPUs maximum,
four-hour allocation limit, fixed run length and no favorable-result stopping.

Run root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_overnight_20261001`.
`plan.json` records the fixed protocol. A143-file source snapshot is checked
at startup; SHA256 of its manifest is
`245f0e8696512dfbed8997cbb289490dc7e270424b8c561d1143824f0bb9fa6a`.
Launchers are `jobs/job_galsbi_overnight.sh` and
`jobs/job_galsbi_overnight_report.sh`. CPU report16778957 depends on
**afterany:16778955**, so incomplete runs are reported rather than silently
omitted. It writes `report/report.md` and `report/report.json`, comparing total
step prefixes65536/131072/262144. No automatic promotion or population run.

These are posterior-integration stress tests under the fixed F24 target and
implemented Poisson generator. They do not supply the measured-selection
normalizer, an accepted shear estimate, or evidence of0.3% calibration.
Mixing and likelihood-ratio overlap must be assessed separately. Slurm and
the dependent report provide the overnight continuation; no agent wakeup
automation is available. Next: inspect the completed independent-seed report
before deciding whether to change proposals or expand the observation panel.
See [overnight protocol](CONTINUOUS_SAMPLING.md#overnight-comparison--october-1-2026).

## 2026-10-01 — Overnight continuation: joint morphology proposals

Owner explicitly requested autonomous overnight work. Continue small controlled
sampler tests, preserving the posterior, V3.9 artifacts, measured-cut contract
and convergence gates. No population-scale inference until numerical checks
pass. Use at most two study GPUs concurrently for the planned independent
replicates, within existing limits; check the queue before submission.

Added `sbsi/mcmc_proposals.py` and opt-in `--joint-morphology` in the existing
pilot. Magnitude, log angular radius and Sérsic logit now move jointly with
per-chain Gaussian covariance. During discarded warmup only, recompute
covariance every128 steps from at most1024 retained states, including rejects,
shrunk5% toward the original diagonal proposal. Freeze both covariance and
scalar scales after warmup. Remove Sérsic from the separate structure block;
the corrected SED jump, exact type sum and persistent-neighbour law remain.
This is a symmetric proposal after freezing, with unchanged target density.

Added an independent correlated-Gaussian posterior moment test and a stuck-
chain covariance support test. Ruff passed; full CPU suite16778860 submitted.
Initial joint pilot will use the same two observations, four chains each,
seed2026093061 and16384 steps/4096 warmup, output
`galsbi_joint_mh_20261001_v1`. Longer independent-seed diagnostics will be
predeclared only after this implementation test; no shear claim follows from
one favorable run. Native-density/generator approximation and F24 versus FS2
population limitations remain unchanged.

Full suite16778860 passed355 tests/1 skip in53.08s. Joint-morphology GPU
16778863 completed in289.83s script time. SED likelihood replay stays exactly
unchanged. Bright Sérsic Rhat improves1.5730→1.1415, but radius1.2836 and
shape1.2165/1.2797 remain poor; its shear-ratio Rhat is1.0581. Faint maximum
Rhat1.0625 remains in crowding; ratio Rhat1.0094 passes but overall exploration
does not. Do not select this run from its favorable small shear-ratio means.

Extended the same Gaussian block to optionally include distortion logit and
wrapped phase (`--joint-shape`). Total physical-proposal probability stays0.6,
previously split between morphology and shape. Estimate phase covariance in
the circular chart centered on each chain's last warmup phase; Gaussian
increments are wrapped afterward. With fixed covariance the wrapped proposal
is symmetric. Added phase-origin invariance and correlated wrapped-Gaussian
moment tests. Full suite16778895 submitted. The next16k pilot keeps all other
settings, output `galsbi_joint_shape_20261001_v1`.

Also added explicit reference-observation indices and saved local shear-ratio
draws to the existing driver for the forthcoming independent-seed/prefix
checks. Default observations and scientific settings are unchanged. These
diagnostic options were added after the running v1 source snapshot was taken;
that completed run retains its exact executed source under its output folder.

Full five-coordinate physical pilot16778896 completed in278.61s, after
357 tests/1 skip passed (16778895). It does not consistently improve the
bright result: maximum Rhat1.3724, ratio Rhat1.2389; faint maximum1.0646 and
ratio1.0045. Both fail overall. Keep it optional rather than choosing it from
the favorable faint result. All starting/SED numerical replays pass.

Added a read-only independent-run/prefix reporter, including between-seed
Rhat/ESS and likelihood-scale MCSEs of numerator ratios. Missing/incompatible
runs cannot pass, and no sampler is promoted automatically. Tests explicitly
include two individually well-mixed but mutually disagreeing replicas.
Report test16778898 initially failed on a script-only import; the reporter
now has no model imports. Full suite16778899 then passed360 tests/1 skip;
removed its NumPy scalar-conversion deprecation afterward.

Next small test: optional local neighbour-position moves, retaining the
three-coordinate joint morphology block. `GalSBIScenePrior.move_positions`
proposes independent Gaussian steps for all positions and retains each point
whose proposal leaves the aperture. Each bounded uniform-disk kernel is
prior-reversible, as is their product; outer MH uses only the likelihood
ratio. Counts, marks, target and population law stay unchanged. Added
stationary-disk and analytically tilted-disk posterior tests. The new block
gets0.05 probability, taken from the SED jump (now0.10); original refresh
probability0.15 stays fixed. Position scale0.2arcsec, same16k/4k protocol.
Full suite16778902 submitted; output `galsbi_local_positions_20261001_v1`.

## 2026-10-01 — Add a corrected SED independence jump to the continuous sampler

Owner authorized the proposed red/blue SED move. Added `sed_log_proposal`
and `propose_sed` to `sbsi/galsbi_target_coordinates.py`. At fixed redshift,
propose from an equal mixture of the two known Dirichlet laws and preserve
apparent r magnitude, angular radius, Sérsic n, distortion and phase. Decode
new intrinsic luminosity/photometry and evaluate the complete native prior.
The proposal density includes the simplex-to-ALR Jacobian prod(c), and MH
uses log q(old|z)-log q(new|z). Equal proposal weights do not change target
population weights. Off-support proposals are rejected without redrawing.

The existing driver gains opt-in `--sed-jump`, requiring persistent scenes
and exact type summation. Block probabilities become0.3 photometry/size,
0.3 shape,0.1 local structure,0.15 environment and0.15 SED independence.
Every finite-prior SED proposal verifies all nine physical summaries remain
fixed within1e-10 and log likelihood within1e-6; failures stop the pilot.
Output now includes native log prior, SED-region crossings, conditional red
probabilities and separate full/partial environment acceptance. No model,
prior density, inference default or measurement-selection rule changes.

Added independent SciPy proposal-density/Jacobian checks, an analytically
soluble unequal-mixture posterior with a likelihood tilt, and a photometric
replay/support test. Ruff and shell syntax pass. Full Slurm suite16778800
submitted with artifact roots configured. Planned pilot keeps the same two
mock measurements, four chains each, seed2026093061,16384 steps/4096 warmup,
per-chain frozen scales and existing neighbour law. It will depend on the
suite passing. Source/protocol receipts and results are preserved under
`galsbi_sed_jump_20261001_v1`. Full suite16778800 completed:353 passed,1
skipped in52.28s. Dependent GPU16778801 started and exact initial likelihood
replay passed; all eight chains have made SED-region crossings during warmup.
GPU16778801 completed in294.96s script time (5m24s allocation). Across19,808
finite-prior SED proposals, maximum summary change is4.2633e-14 and maximum
log-likelihood change is exactly zero. CPU audit16778804 completed in2s.

The SED-region trapping improves markedly, but **neither observation passes**.
After warmup the four bright chains make24/28/28/86 region crossings, with
mean conditional red probabilities0.9912/0.9793/0.9804/0.9561; previously two
chains stayed red and two blue. Full environment refresh acceptance is now
75.3/86.1/77.7/75.3%, versus a zero-acceptance bright chain previously.
Bright SED ALR rank-Rhat is1.0029–1.0106, bulkESS525–1821. Its largest remaining
Rhat is1.5730 for Sérsic index (bulkESS6.94), and circularized radius is1.3641
(ESS9.13). Theg1=.022 numerator-ratio Rhat improves from1.4675 to1.0489,
but its per-chain log ratios still range−0.01379 to+0.00780 and are not accepted.

Faint maximum Rhat is1.0629 and minimum bulkESS63.07; magnitude/crowding
remain slow. Itsg1=.022 ratio Rhat is1.0106 withESS445.94. Conditional red
probability differs0.0004–0.0744 across faint chains, so crossings alone do
not establish correct component mass. Shape/SED diagnostics improve, but
joint posterior convergence is unresolved. This remains a two-observation
diagnostic, without a selection integral, shear estimate or0.3% claim.

Read-only correlation audit16778808 examines the remaining structural
coupling, using saved traces; it completed in1s with no extra inference run.
Within bright chains, log angular radius and Sérsic logit correlate−0.43 to
−0.80, but current proposals put them in different blocks. Faint magnitude
and close-crowding correlations range0.55–0.98; crowding uses target-relative
flux, so this is partly structural and is not by itself proof of trapping.
The next minimal test is a joint magnitude/size/Sérsic proposal with covariance
adapted during warmup and frozen afterward, retaining the corrected SED jump.
Do not infer that this will resolve environment exploration or authorize a
population-sized run from the favorable SED result. All jobs are complete.
Results and next step are recorded in
[SED-jump pilot](CONTINUOUS_SAMPLING.md#sed-independence-jump--october-1-2026).

## 2026-10-01 — Retain neighbour environments during local MH target moves

Owner authorized the proposed next sampler step. Added a prior-reversible
partial marked-Poisson refresh and complete ragged-scene selection/retention
to `sbsi/galsbi_scene_prior.py`. Extended the existing local pilot with
`--environment-update persistent`: one explicit environment per chain,
unchanged during target moves, with separate refresh moves. A refresh retains
each neighbour with probability0.9 and adds an independent0.1-intensity
population;10% of refresh proposals instead redraw the complete environment.
The fraction choice is independent of state. Random order within each patch
preserves the ordered-iid convention. Acceptance uses the likelihood ratio
for a refresh, and the target-prior times likelihood ratio for local target
moves. Rejected states retain every target and neighbour field.

Bounded protocol: same two independent model-generated bright/faint mock
observations and four chains each;4096 steps including1024 warmup, exact
red/blue target sum, fixed4096 density batching, seed2026093061. Initial scenes
are the best measurement-likelihood candidate in each previous prior stream;
their original environments are retained, and no generating truth is used.
This changes initialization, environment update and run length together; it
is an algorithm refinement, not an isolated causal comparison. Initial full
scene replay must agree with saved reference log kernels within1e-6.
All native coordinates except wrapped phase, neighbour counts and all three
crowding summaries are now monitored in addition to the existing diagnostics.
Rank-Rhat<1.01 and bulkESS>400 remain necessary exploration gates.

Added tests of ragged-state retention, refresh prior stationarity, count
cross-covariance, and an analytically tilted marked-Poisson posterior whose
count and mark laws are known. Also fixed empty-environment field creation:
neighbour fields come from the galaxy generator, not a potentially different
supplied target schema. Ruff and launcher shell syntax pass. Full CPU suite
16778740 passed:348 tests passed,3 skipped in55.48s. Dependent GPU16778741
failed before scientific work because the resource detector's output directory
did not exist; the launcher now creates it. Identical pilot resubmitted as
16778742. Source snapshots,
protocol hashes, final native scenes and resource receipts are preserved.

No production default, model identity, selection integral or shear estimate
changes. The native analytic target prior and numerical environment generator
retain their existing approximation difference.

First pilot16778742 completed in89.68s script time (2m03s allocation).
Starting replay is exactly equal to the reference. Both observations fail:
bright maximum Rhat2.860/minimum bulkESS4.775; faint1.097/34.88.
Bright magnitude/size acceptance ranges0.99–27.92% across chains, despite
using a shared step scale, and the local numerator ratios disagree strongly.
Faint ratios atg1=.022 span0.00414–0.00670 in log ratio; this is neither a
selected likelihood nor a shear estimate. The change removes the independent
likelihood-estimator noise but does not establish posterior exploration.

Added the conventional per-chain warmup-scale option to the existing driver
and launched one bounded follow-up16778765, output
`galsbi_persistent_mh_20261001_v2`:16384 steps/4096 warmup, same eight chains,
observations, seeds, posterior, starting scenes and environment-refresh law.
Scale adaptation freezes after warmup. This tests whether proposal scales
better match the different chains; it also increases length, so is not a
single-factor causal comparison. Ruff passes after this script-only option.

Follow-up16778765 completed in272.29s script time (4m45s allocation), with
exact starting replay. **Neither observation passes**. Bright maximum
Rhat2.2307/minimum bulkESS5.896; faint1.0921/39.20. Bright magnitude/size
acceptance is now18.8–32.1%, but shape acceptance is5.1–33.3% and chains still
disagree on the local shear numerator ratios. Faint ratio bulkESS493/511 is
better, but ratio Rhat1.0137/1.0129 and other diagnostics still fail.

Read-only CPU audit16778767 completed in2s. Its script and
`chain_mode_summary.json` are preserved in the v2 output. Bright chains0/3
have mean conditional red probability>0.99999998; chains1/2 have probabilities
near zero. Summing type analytically did not remove the separated SED regions.
Chain1 samples a much broader intrinsic-magnitude range (10–90% quantiles
19.35–22.39) than chain0 (19.71–19.75). It accepts0/181 full environment
refreshes, despite89.3% acceptance of partial refreshes. The other bright
chains accept67–77% of full refreshes. These are nonconverged chain summaries,
not estimated posterior population weights; do not select the chain with the
largest likelihood and discard the rest.

Conclusion: explicit scenes remove independent likelihood-estimator noise,
but local MH still fails to explore the joint target/environment posterior.
No accepted sampler, selected shear estimate or0.3% claim. No jobs remain.
The next smallest targeted change is an independence proposal for SED from
the known red/blue Dirichlet mixture, at fixed z and likelihood-visible target
properties, with its exact proposal-density/Jacobian correction. This can
jump SED regions without fitting a new prior. Environmental trapping remains
a separate check, potentially requiring local moves of existing neighbours.
See [persistent-scene MH](CONTINUOUS_SAMPLING.md#persistent-scene-mh-pilot--october-1-2026).

## 2026-09-30 — Implement fixed-summary likelihood and test standard MH variants

Owner requested rethinking the sampler around the likelihood's fixed summaries
and then explicitly asked to continue. Added `sbsi/galsbi_summary.py`,
`sbsi/summary_sampling.py`, `scripts/check_galsbi_summary_sampler.py` and
`jobs/job_galsbi_summary_sampler.sh`. For fixed observed radius/flux, the eight
F8/C8 features plus the summed disk response reproduce the full retained
likelihood. The response coefficient is observation-indexed, and stale
auxiliaries are rejected. All neighbours enter crowding; the existing response
uses nearest20 within strict10arcsec, without truth cuts. No model retraining,
emulator simplification, fitted summary density or production default change.

Established the previously unresolved native radius mapping by inspecting the
installed UFig photon-rendering matrix: semi-major=R_native*sqrt(1+a),
circularized=R_native*(1-a^2)^(1/4), with native distortion converted to epsilon.
Independent SVD replay of the actual rendering matrix agrees to2.22e-16.
This explicitly adopts the zero-flexion rendered ellipse, not an empirical
calibration or direct identification of native and circularized radii.

The induced nine-summary prior is available by fresh scene generation and
compression. It is not generally a smooth Lebesgue density: the tree response
can have point masses, as can the no-neighbour event. Ordinary independence
MH with fresh prior proposals avoids evaluating that density because proposal
and prior cancel. Rejections remain in each chain; no generating truth is
inserted and no finite catalogue is resampled indefinitely.

Completed summary pilot16776331, artifact `galsbi_summary_mh_20260930_v2` under
the external catalogue-prior root, script time28.50s.8192 independent model
inputs yielded3188 usable and1961 measured-cut-passing mocks. Four measured
magnitude strata were represented, and each observation used four independent
2048-draw fresh-prior streams. Full-pair versus compressed likelihood,
reversed order and padded partial batches agree exactly. Initial job16776323
stopped before MH because C8 in8192 versus4096-row batches differed by3.98e-6
in log probability; rerun matched classifier batching, retaining the1e-6 gate,
seeds, observations and proposal streams. No threshold was loosened.

All four independence-MH probes **failed exploration**. Prior importanceESS
per2048 draws was1.00–3.08 at measuredmag19.745,1.05–3.76 at21.255,
2.90–11.32 at23.971, and20.45–24.86 at25.118. Bright chains were nearly frozen;
faint acceptance was0.68–2.05%. These are not recovered shear estimates.

Continued with standard pseudo-marginal MH, adding
`sbsi/galsbi_target_coordinates.py`, `sbsi/pseudo_marginal.py`,
`scripts/check_galsbi_local_sampler.py`, and `jobs/job_galsbi_local_sampler.sh`.
Refactored the existing neighbour generator to expose `sample_given_targets`,
preserving its original full-scene RNG sequence. Ten transformed target
coordinates retain the evaluable native prior and all Jacobians; fresh
neighbours supply a nonnegative likelihood average. Rejection retains the
current average and compressed bank. Bank moments use likelihood weights,
and shear numerator ratios use ratios of bank sums, not equal scene weights.
Measurement-selection normalization remains separate and uncomputed here.

First local pilot16776398 completed in51.50s script time: same bright/faint
measurements, four chains each,1024 steps/512 warmup, four environments per
proposal. Local moves improve acceptance but convergence still fails; two
bright chains stick. Independent fixed-target audit16776440 used1024 fresh
environments per chain: those two chains' stored log-likelihood estimates
exceed the audit means by7.59 and8.63; four-environment log-estimate SD is16.86
and17.86. At64 environments SD falls to2.31 and1.76. The1024-draw audit itself
has only2.82 and4.47 effective environments at those targets, so its mean is
not treated as exact. This is demonstrated estimator noise, not evidence of
recovered posterior mass.

An exact additional simplification sums red/blue native prior intensities:
type is not a direct F8/C8/disk input at fixed physical properties, so it need
not be a discrete MH variable. Added `log_mixture_prior` and a test against the
explicit native sum. The bounded refined pilot16776459 uses64 environments,
the exact type sum, and separate proposal-scale adaptation per observation
during warmup only. It keeps the same two measurements and1024 steps. Its
protocol and source snapshot are in `galsbi_local_mh_20260930_v2`. It completed
in330.48s script time, but both observations still fail. Bright rank-Rhat is
1.79–3.11 and bulkESS4.56–55.91 over monitored functions; faint is1.07–1.50
and7.81–51.42. Faint photometry/size and shape acceptance reaches23–44%, but
the chains still disagree. At two bright final targets the independent
1024-environment audit has only5.73/12.17 effective environments, and retained
log-estimates exceed its mean estimates by8.36/3.89. Faint M64 log-estimate SD
is0.13–0.22, showing that target mixing also needs work. This is an algorithm
refinement, not an isolated causal comparison of environment count alone.
Retain exact summary compression, reject both short sampler pilots as inference
replacements. No additional or large jobs remain. Next is proposal design
that can retain useful target/environment configurations; do not interpret
acceptance or a favorable shear proposal as convergence.

Validation: analytic Gaussian posterior/evidence-ratio and point-mass tests,
independent full ten-dimensional Jacobian determinant, noisy latent-Gaussian
pseudo-marginal closure, rejected auxiliary retention, neighbour law with fixed
targets, and likelihood replay. Full Slurm suite16776297:339 passed/1 skipped;
16776388:343/1;16776457 after type marginalization:344/1 in43.53s. Ruff and shell
syntax checks pass. Each main GPU pilot uses one GPU, one CPU,16GB and a20min
cap; the separate noise audit uses8GB/5minutes. No study GPU jobs overlap.
Resource receipts are captured in the main pilot allocations.

Limits: native target density and numerical environment generator retain their
audited grid approximation; fixed F24 population/uniform neighbours differ
from FS2. No representative-population closure, measured-selection integral,
accepted shear estimate or0.3% claim. Design, equations, protocols and results:
[continuous sampling](CONTINUOUS_SAMPLING.md#fixed-summary-prototype--september-30-2026).

## 2026-09-30 — Design continuous-scene posterior sampling before implementation

Owner requested thinking through the next sampling algorithm. Added
[the proposed design](CONTINUOUS_SAMPLING.md#status-and-recommendation--september-30-2026),
after inspecting the native prior, V3.9 disk density/response code, selection
contract and earlier profile/batch failures. This is documentation only;
no sampler, model, inference default or job changed.

Recommend tempered SMC with one blocked Metropolis kernel, posterior scene
reuse for nearby shear likelihood ratios, and an independent CRN forward
selection integral. Derived the partial Poisson refresh proposal: independently
retain1-rho of neighbours, add a fresh rho-intensity population by envelope
thinning, randomize list order. Its prior reversibility cancels prior/proposal
terms, leaving the tempered likelihood ratio; no separately estimated surface
density or custom birth/death factor is needed. Local target/neighbour moves
still require native/transformed-coordinate Jacobians and support checks.

Recorded the full q_U*f_disk posterior target, the population Z(g) selection
factor, reuse of observed-radius/flux response across shear nodes, the remaining
native-to-SBSI mapping, common-support and batching checks, and why simple
gradient methods or independently noisy Newton stencils are poor first choices.
Validation progresses from analytic/replay tests to two observations,32 stress
probes, then representative continuous-model mocks. Proposal budgets are
diagnostic; neither high ESS nor favorable shear is an acceptance guarantee.
The prospective global numerical target is1e-5 absolute shear versus6e-5
corresponding to0.3% atg1=0.02, not a claimed achieved precision.

Validation: checked formulas against the local likelihood contracts and primary
SMC/Poisson/MCMC-diagnostic references; reviewed the Poisson exchangeability
argument and posterior-ratio/selection identities. Documentation links and
whitespace checked. No numerical experiment or speed claim; next step is a
small arbitrary-scene likelihood adapter with pointwise numerical replay.
## 2026-09-30 — Add uniform GalSBI neighbour environments

Owner authorized adding neighbours to the continuous prior, retaining the
previously requested uniform, unclustered approximation. Added
`sbsi/galsbi_scene_prior.py`: independent target-centred 10-arcsec patches,
native galaxy marks, uncapped Poisson neighbour counts, Cartesian sky offsets,
and explicit scene density. Targets and neighbours use the same fixed F24
population/support. Neighbour sampling thins the pre-photometric Poisson
population; no Monte Carlo post-cut normalization enters the sampler.
`GalSBIContinuousPrior.sample` now returns retained `proposal_index` values
to preserve patch membership through this thinning.

The ordered-list density includes the count factorial and per-steradian to
per-arcsec-squared measure conversion. `log_kernel` omits only constants at
fixed population/aperture; `log_prob` accepts an explicit uncertain surface
density. Derived physical Gaussian crowding matches the F8/C8 definition,
using all nonself neighbours within10 arcsec. Native pair marks and offsets
are retained; emulator radius mapping and predictions are not implemented.

Added `scripts/check_galsbi_neighbours.py`, CPU launcher
`jobs/job_galsbi_neighbours.sh`, and `tests/test_galsbi_scene_prior.py` covering
Poisson measure/count changes, rejection thinning, zero-neighbour scenes,
support and existing V3.9 feature replay. The diagnostic compares4,096
continuous patches with disjoint patches made by uniformly placing the
previous independent GalSBI catalogue marks/count, without row reuse. This
reference deliberately does not use its original HEALPix sky positions.

CPU job16775918 passed the full suite:331 passed,3 skipped in59.83s.
Comparison16775919 passed all eight declared gates. After randomizing the
reference list order (GalSBI stores types in blocks), final replay16775946
completed exit0 in48s and again passed all gates; counts, marks, spatial
checks and gate decisions are unchanged. No seeds/thresholds were tuned.
Artifacts: `galsbi_neighbours_20260930_v1` under the external catalogue-prior
root. Resources recorded in the allocation; one CPU/16GB requested for the
comparison despite greater host-wide availability. No GPU or inference job.
Scientific seeds are2026093021 (continuous scenes),2026093022 (reference
positions). Ruff, shell syntax and AST checks pass; final source snapshots
and comparison hashes verified, comparison figure inspected.

The4,096 scenes contain87,622 neighbours (mean21.392 versus21.540 predicted,
-1.62 standard deviations including normalization uncertainty), with count
variance/mean0.99429. The424 disjoint reference patches contain9,134 galaxies;
2,535 square-cell-corner rows are excluded and every retained row is used once.
Minimum spatial raw/adjusted p is0.03464/0.13854; minimum neighbour-mark raw p
is0.08022. Paired crowding means are compatible at allthree scales, and first64
scenes replay production crowding within4.44e-16. Every scene density is finite.
Detailed evidence and usage:
[neighbour report](GALSBI_PRIOR_FEASIBILITY.md#uniform-neighbour-pilot-results--september-30-2026).

Limitations: unclustered independent patches at fixed population parameters
and declared intrinsic support; the reference reassigns uniform positions.
Absolute scene normalization still carries the isolated Monte Carlo error,
while fixed-population density ratios and neighbour sampling avoid that
estimate. No emulator input mapping, posterior sampling, measured-selection
normalization or shear validation. Next is coordinate mapping and a small
inference closure test, not a0.3% claim.

## 2026-09-30 — Implement an isolated GalSBI continuous-prior pilot

Owner authorized an isolated-galaxy density implementation and comparison
against a GalSBI-generated catalogue, without neighbours. Added
`sbsi/galsbi_prior.py` with explicit F24 native-coordinate density factors,
support checks, continuous luminosity/redshift sampling, and an explicitly
uncertain population normalization. Added the diagnostic
`scripts/check_galsbi_continuous_prior.py`, its CPU Slurm launcher
`jobs/job_galsbi_prior_check.sh`, and measure/support tests in
`tests/test_galsbi_prior.py`. No V3.9 model or inference default changes.

The reference uses installed GalSBI0.3.0, F24 posterior row0 (accepted row,
source id8), HSC r2/i2 filters, zero foreground extinction, float64 catalogue
output and intrinsic reference-i limits14–28. It includes no measured cuts,
images, neighbours or shear inference. The analytic adapter is in the native
latent variables; it is not yet a marginal density in SBSI's five features.
The source package differs from the latest upstream checkout, so the actual
installed source and data hashes are recorded by the comparison.

Artifacts are under `galsbi_continuous_prior_20260930_v1` in the external
catalogue-prior root. Preflight CPU job16775773 generated11,669 reference
galaxies successfully and recorded resources. Initial comparison16775826
failed before generation because SBSI was not installed into py31 for script
imports; installing this checkout editable follows ENVIRONMENT.md and avoids
adding PYTHONPATH workarounds. Job16775840 generated the reference, then
exposed the installed photometry wrapper's keyword-only interface; repaired
the callback. Job16775846 completed the calculations but exposed NumPy-bool
JSON serialization; explicitly converted gate values to Python bool. No
scientific settings, seeds or acceptance thresholds were changed after seeing
results. Final comparison16775868 completed exit0 in40s on one inter CPU,
about0.80GB reported peak RSS. No GPU used.

All nine pilot gates passed: 11,669 GalSBI galaxies versus70,507 independently
retained continuous draws from131,072 proposals; magnitude replay max1.42e-14
mag; every density finite; smallest raw two-sample p0.07295 across15 marginal,
five coarse joint and one type-fraction checks. Morphology conditional-CDF
minimum raw p0.04835 (shape), Holm-corrected0.19338. Expected reference count
11,652.83 versus11,669 actual (0.144 standard deviations). Population
normalization is2.91704e9 sr^-1 with0.256% Monte Carlo relative SE, unrelated
to shear accuracy. Luminosity grid change1.81e-9 and magnitude inverse-CDF
error2.35e-8. Results, support and usage are recorded in
[the pilot report](GALSBI_PRIOR_FEASIBILITY.md#isolated-galaxy-pilot-results--september-30-2026).

Validation: full-suite job16775847 gives328 passed,3 skipped in43.31s;
`ruff check sbsi/galsbi_prior.py scripts/check_galsbi_continuous_prior.py
tests/test_galsbi_prior.py` and `bash -n jobs/job_galsbi_prior_check.sh` pass.
Final source snapshots and comparison hashes verified, both plots inspected.
Generation metadata must match comparison arguments; any failed diagnostic
writes evidence and exits unsuccessfully. Tiny samples may omit a galaxy
type; the sampler handles this and rejects nonpositive proposal counts.

Limitations: fixed F24 posterior row and native latent density only; shared
nearest-grid photometry; no five-feature marginal, SBSI radius/shape mapping,
neighbours, measured selection, MCMC or shear calibration. The matched
M_B<=-4 population boundary carries GalSBI's own incompleteness warning;
support completeness and rare bright tails remain unvalidated. The next
inference step needs explicit coordinate mapping and sampler/selection
closure. No sub-percent equivalence or0.3% calibration claim is established.

## 2026-09-30 — Audit GalSBI as a continuous population prior

Owner asked whether GalSBI can evaluate the density of arbitrary true
galaxies, allowing uniform neighbour positions and excluding its clustering
extension. Added [the source-level feasibility audit](GALSBI_PRIOR_FEASIBILITY.md#scope-and-verdict--september-30-2026).
Inspected the public implementation at commit
`df802e1db2f2a4247955720aea227d5b65cdf975` and the non-clustering F24 model,
alongside the primary paper and official documentation. The checkout is
`/tmp/sbsi-galsbi-audit-20260930`; no upstream modules were executed or
dependencies/data installed. No production code, model, prior or job changed.

The package returns ABC population-parameter samples and generates galaxies;
no full galaxy `log_prob` interface was found. Explicit Schechter, Dirichlet,
lognormal, scaled-beta and truncated-beta-prime factors support constructing
a native joint density. A five-dimensional SBSI truth-density evaluator
additionally needs marginalization over redshift, type and SED. The proposed
alternative keeps these nuisance variables in the sampler and uses the
existing generative factors rather than first fitting a Gaussian mixture.

Recorded the uniform marked-Poisson neighbour factorization, shared population
parameters, intensity versus probability normalization, Sérsic truncation,
shape/size/filter conventions, and F24 default intrinsic i-band limits.
Source inspection also finds grid-discretized redshift/magnitude generation
and nearest-grid spectral lookup: continuous-model closure cannot silently
assume the released sampler is exact. GalSBI is a different parent from FS2;
success on GalSBI mocks would not itself establish recovery on ConstGold.

Validation: source API searches and AST inspection of nine upstream Python
modules passed; documentation whitespace and relative links were checked.
This is feasibility analysis only: no complete density normalization,
posterior bundle inspection, generated-catalogue comparison, MCMC benchmark
or calibration result. Next step is a bounded native-density adapter and
generator-consistency audit before an inference pilot.

## 2026-09-30 — Trace the expanded-prior large Newton proposal to cancelling noisy curvature

Owner requested a deep investigation of the full matched-mock proposals
0.18813134 and0.14730972. Added the read-only saved-moment reduction
`scripts/audit_v39_expanded_step.py`, frozen with outputs under
`inference_v39_expanded_step_audit_20260930_v1` in the external catalogue-prior
root. No model, prior, observation, sampler, or production default changed.
CPU job16773458 completed exit0 in19s on one inter CPU with8GB requested;
no GPU jobs were submitted. The resource receipt is retained in the run root.

The audit verifies50 source hashes, original20k row coverage and exclusion of
the3686 mock padding rows, primary/maximum-rung equality, and all six saved
mock and actual aggregate scores/information. Independent2x2 solves and
influence covariance reproduce all six positive mock proposals and their
reported standard errors. Independent differences of the saved selection-mass
nodes reproduce the normalization gradient/Hessian. Exact row-attribution
identities reproduce the512→8k,8k→16k and512→16k changes. Ruff and AST parsing
pass; numerical validation is against the frozen scientific arrays rather
than new tests duplicating the reduction.

At16k the mock weak-direction information is the residual of+5.200966m and
−5.088162m: only1.096% of the sum of absolute contributions remains. This is
scientific/numerical cancellation, not evidence of a floating-point failure
in the2x2 solve (its condition number is about15.3). The134 measured-mag<20
rows account for82.58% of the512→16k g1 movement, while774 mag20–22 rows
supply80.98% of the final g1 score. Thus bright integration errors change the
curvature used to convert other observations' scores into a shear step.
The16k conditional g1 influence error is0.90423; the0.14356 step length is
143.6 times the0.001 derivative stencil. No valid maximum was located.

Replayed the earlier three exhaustive worker files (complete28,012,032-atom
coverage) and both normalizers from raw arrays; receipt
`exact_probe_replay.json` preserves source hashes. For influential mock4460,
the valid new bright atoms supply86.246% of exact centre mass. Sampled16k
g1 information is−233,919.29, versus exhaustive+165,073.89 under the same
expanded prior. The known399k entry error is material to the whole-sample
solve. This demonstrates numerical misrepresentation of the expanded prior,
not failure of bright support alone. Dropping selection normalization only
as an algebraic diagnostic still gives g1=0.12554; it is not the main cause
of the large proposal and is never removed for inference.

Documented the full attribution and limitations under
[expanded-step forensic audit](V39_POPULATION_AUDIT.md#expanded-step-forensic-audit-on-september30),
linked from the original comparison. All numbers remain conditional on the
unconverged old K1024 sampler. The matched inputs are already image selected;
finite-prior resolution, double selection, normalization uncertainty, and
the global likelihood maximum remain unresolved. Next informative work is
an accurately evaluated small-cohort profile, including magnitude20–22
observations, before another large run or further prior expansion.

## 2026-09-29 — Exact group profile completes; A/B peak disagreement persists

At the owner's status check, all four workers16746033 completed in22m35–39s
and reducer16746034 completed in41s. No study GPU jobs remain. All four
workers replayed the source exact centre with maximum error7.8613e-11 at
the unchanged1e-7 tolerance. Added completion_audit.json to
inference_v39_independent_exact_group_20260929_v1: verified ten pinned source
hashes, all four worker hashes, exact49-node coverage and cross-worker centre
agreement, and independently reproduced the replacement/aggregation within
1e-10. All128 fresh-parent rows remain: four exact,124 sampled; normalizers
are unchanged.

The coarse peaks remain A=(-0.00625952,0.01735242) and
B=(0.02374048,-0.00264758). Each prior prefers its own peak over the other's
by3.2310 and5.5163 log-likelihood units, respectively. Seed/draw profile
variation now passes0.05 under both priors (A0.04542; B0.02861), but this
is **not** an oracle accuracy guarantee. Exact replacement changed the B
curve by0.25055 at(-0.00625952,-0.03264758), although the previous saved
variants there differed from their primary by at most0.02088. Common error
can survive seed/draw agreement. Changes at the competing peaks were small:
A+0.000473/-0.000229 and B-0.000040/-0.001878 at own/other peak.

This particular numerical repair does not explain away the prior-realization
sensitivity. Only one group is exhaustive, the grid remains coarse, and the
A-generated bright control retains strong intrinsic overlap across priors.
No accepted shear estimate or0.3% result. No new GPU jobs submitted during
this status check. See
[exact group profile result](V39_POPULATION_AUDIT.md#completed-exact-group-profile).

## 2026-09-29 — Independent profiles complete; target the remaining numerical failure

All v4 jobs completed: preflights16743015[0–1] in13m39–40s, four profile
workers16743017 in3h37–3h52, reducer16743018 in66s. Both exhaustive preflight
errors passed0.001 (maxima0.000390 and0.000361). All87 groups passed full
candidate-density and response replay. No study GPU jobs remained at this
check. Added scripts/audit_v39_independent_profiles.py; CPU audit16746010
verified every group/profile hash and reproduced all four aggregate curves
within1e-10, saved per-observation contributions and coarse-grid figures.
Ruff and syntax checks passed. These are small saved-array reductions, not
a new likelihood calculation or accepted estimator.

The A-generated cohort has the same coarse peak under both priors:
(g1,g2)=(0.02374048,-0.00264758). Near-mode integration differences are
0.00933 (A) and0.00745 (B), and normalizer differences0.01191/0.01790,
all within the fixed0.05 limits. The fresh-parent cohort instead peaks at
(-0.00625952,0.01735242) under A and(0.02374048,-0.00264758) under B.
A fails integration convergence (0.12726); B passes (0.02760). Both
fresh-parent normalizers pass. These are grid points spaced0.01 apart,
not refined shear estimates or0.3% evidence.

Fresh-parent row47 (measured magnitude20.76839) dominates the draw instability:
its maximum8k-to16k change is0.18904. For attribution only, removing its
original numerical group leaves maximum summed integration drift0.04542;
no observations are removed from inference. Other numerically stable rows
also contribute strongly to the A/B likelihood difference. Submitted an
exhaustive full49-node calculation for group66, rows44–47, across both priors:
four one-V100 workers16746033, reducer16746034, frozen in
inference_v39_independent_exact_group_20260929_v1. Each worker replays the
old exact centre, then handles12 other nodes. Only these four rows will be
replaced in the128-row sums; normalization and all other rows stay fixed.
Estimated20–30 minutes based on the completed three-node preflights.
All four exact workers passed startup and reached10.5m of20.23m atoms in
their initial centre scan without errors. Status-only audit16746082 is
scheduled for about two hours later; it is not an agent wakeup or notification.

CPU context-overlap audit16746043 found exact five-intrinsic-feature matches
in B for19/128 A-generated inputs, and no exact eight-feature F8-context
matches. Disjoint scenes share an empirical source catalogue; this is not a
fully independent intrinsic-galaxy holdout. All three measured-bright inputs
have intrinsic matches in B (3,1,2 for rows116,208,245); the common coarse
peak therefore does not cleanly test removing their intrinsic support.
Follow-up ID audit16746067 and saved centre candidates quantify the effect:
the matched B atoms carry68.63%,81.32%,99.973% of the three objects'
posterior mass at the fixed initial centre. Their original generating A atoms
carry66.71%,99.23%,38.38%, respectively. These are centre-only posterior
fractions, not prior weights or inferred shear probabilities; source arrays
and exact centre denominators provide the calculation in
audit/bright_intrinsic_support.json. No extra GPU inference was used.
The complete response neighbour
graphs were not compared. See
[completed independent profiles](V39_POPULATION_AUDIT.md#completed-independent-profiles).

## 2026-09-28 — Padded replay passes; launch independent-profile preflights

V4 preparation16742975 passed all11 targeted tests (0.20s), source receipts,
scene disjointness, prior masses and unchanged cohort checks. Bounded GPU
replay16742976 passed all ten cases exactly: ordered/permuted first, middle,
padded-final blocks, mixed subsets and duplicated draws, for both cohorts.
Maximum coefficient and density differences were0 throughout. The padding
repair therefore removes the reproduced partial-batch discrepancy in these
probes without relaxing any tolerance; the original six candidate differences
still require full-centre replay and are not assumed resolved.

Submitted two one-V100 exhaustive preflights16743015[0–1]. Both must pass
coefficient, candidate-density and full-prior log-profile checks before
CPU gate16743016 releases four persistent one-V100 workers16743017[0–3].
Reducer16743018 retains every numerical variant; no accepted estimate or
population conclusion yet. The unchanged design uses128 scientific rows
per cohort, A/B disjoint scene priors and separate selection normalization.
Runtime remains dominated by20.23m-atom centre scans for87 four-row groups.
The4-GPU ceiling is enforced by stage dependencies and array throttles.
Both preflights reached their full-prior centre scans and passed1m atoms
without error at the startup check. Scheduled status-only audit16743028 for
about two hours later; this records scheduler/progress state and is not an
agent wakeup or user notification. Saved repair_identity_audit.json verifies
all scientific settings,87 group input hashes, priors and normalizer input
hashes are unchanged from v2.
See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Isolate partial density-batch sensitivity before independent profiles

V3 preparation16742917 passed11 tests in0.24s and all input checks. The
bounded GPU replay16742918 reproduced exact coefficients and bitwise density
reordering on the first and middle full16,384-atom blocks. On the final13,365
atoms, coefficients still matched exactly, but5,777 density terms in the
fourth observation differed (maximum0.1183698); the first three rows were
bitwise identical. Saved the complete failed probe arrays. Its last density
batch is shorter, so reordering moves individual atoms into/out of that batch.
This isolates a batch-shape numerical effect; it does not yet explain the
six small candidate differences in v2, nor the historical shear failures.

The frozen v4 driver pads only the final atom block to16,384 using duplicate
atoms and discards those evaluations before summing or ranking. Four original
observation rows and4,096-atom density chunks remain unchanged. Added fixed
mixed-subset and duplicate-atom replays to test changes in the queried atom
set, not only permutations of an identical set. All probes finish and save
diagnostics before their combined gate; tolerances remain unchanged. Two
independent preflight groups can then run on two GPUs before the four-worker
broad comparison. CPU preparation16742975 and GPU replay16742976 are submitted;
no broad scans are submitted. Ruff and shell syntax passed.

## 2026-09-28 — Gate independent profiles on numerical replay; reuse fixed responses

The first full-centre scan in preflight16742809 completed, but candidate
reordering failed the unchanged1e-7 per-atom density replay:6 of65,536 terms,
maximum log-contribution difference3.69084e-6. The broad worker array was
held during a runtime review and all dependent workers/reducer subsequently
cancelled on preflight failure. No A/B likelihood profile has been accepted.
The old scheduled status-only job was cancelled as superseded.

Added experimental scripts/v39_group_response.py, retaining the original
response coefficients for one fixed four-observation group across shear
points. This stores about0.65GB per group and avoids repeating the same
emulator pair sums. It requires exact coefficient replay on reordered
candidate/tail sets and rejects changed observations. No production model,
batch shape, prior, observation, seed, grid or tolerance changed. Added a
focused row/atom/mask/cache-invalidation test; the likelihood driver now saves
replay diagnostics before failing and checks the union of near-modes across
all numerical variants rather than only the primary profile's near-mode.

Frozen v3 preserves v2 scientific settings and outputs. CPU preparation16742917
and bounded reorder diagnostic16742918 are submitted; broad scans are not
submitted pending this diagnostic. Ruff and shell syntax checks passed.
Preparation will run the existing ten tests plus the new memo test. The
reorder diagnostic compares original versus reused coefficients and ordered
versus permuted densities on three fixed atom blocks for each cohort.
See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Start scene-disjoint prior A/B likelihood-profile comparison

Owner clarified that continuation means the independent-population-realization
comparison and likelihood profiles proposed in the reflection. Discarded the
unsubmitted atom-thinning draft; no thinning experiment ran. Added
scripts/run_v39_independent_profiles.py. Use the original F8-ready prior's
scenes20000–20099 as A (10,116,902 atoms) and20100–20199 as B
(10,114,319 atoms). These are disjoint generated scenes from the same empirical
parent source, with the original inverse-inclusion masses separately
normalized. No extra bright atoms, model changes or fitted reweighting.

The first128 existing selected model draws whose generating atom belongs to
A are a conditional sample from A. Preserve their original four-row numerical
groups. The independent-input control is the first128 fresh unselected-parent
mock measurements, selected once. There are87 numerical groups in total;
only128 rows per cohort enter scientific sums. The A cohort has three
measured-mag<20 rows and the fresh-parent prefix has none; no identities were
changed after noting this limitation. Each cohort is evaluated against A and B.

Freeze a7x7 two-component grid, offsets±0.03 in0.01 steps about the original
full20k mock's measured mean, without truth-centred starts. Exact centre
scans supply per-prior top16384 candidates and the existing conditional-tail
proposal; use seeds8701/8702 and4k/8k/16k nested tail draws. Normalize A and B
separately using two independent weighted-prior samples and nested32k/64k
atom and32/64 CRN-QMC flow budgets. All normalizer and proposal variants are
retained. A small preflight compares both priors on the first original group
of each cohort against exhaustive sums at centre and g1±0.01;8k/16k must
agree within0.001 in centred per-object log numerator before broad grids run.
Coarse-grid peaks are diagnostic and require convergence checks and refinement.

Ruff, AST and shell syntax passed; preparation runs ten existing tail/reduction
tests. First attempt16742780 failed on import because the writer alias belongs
to the cohort driver; no science outputs were generated. Preserved v1 and
cancelled dependent preflight automatically. Corrected the alias in a frozen
v2 root. Preparation16742796 completed in33s with ten tests passed and source,
scene-disjointness, mass, cohort and padding checks passed. Preflight16742809
gates four one-V100 workers16742810[0–3], then reducer16742811. No other GPUs
were active. See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Pause runs and reassess the inference failure mechanism

Owner requested reflection on same-prior mock recovery, actual and
matched-input curvature failures, and the unstable expanded-prior result.
No jobs were active and none were submitted. Reviewed generation code,
the four historical comparisons, parent-population audits, the completed
fresh-parent5130 result, exhaustive probes, and the mixture-curvature
derivation in MATH.md. Added a synthesis under
[reflection on the failure mechanism](V39_POPULATION_AUDIT.md#reflection-on-the-failure-mechanism).

The synthesis distinguishes finite-prior approximation, numerical atom
integration, selected-cohort conditioning, and local optimization. Same-prior
mocks contain their generating atom and are internal closure tests; their
iteration stability did not pass draw convergence. Matched-input mocks start
from already selected inputs, so are not ordinary parent-population closure.
The clean fresh-parent5130 run had positive information throughout its draw
ladder, preventing a blanket conclusion that off-prior mocks necessarily
fail. Exact local negative information does not establish a wrong likelihood
maximum; the original actual run never reached BFGS recentering.

Numerical integration errors are established. Finite-prior resolution and
strong mixture nonlinearity remain plausible explanations requiring separate
tests; broad population mismatch is not established. A proposed controlled
same-population comparison would generate from one finite prior and infer
using that prior versus an independently realized prior, with selection once
and converged likelihood profiles. This is a proposed discriminator, not a
newly authorized run or result. No code, model, prior, or acceptance change.

## 2026-09-28 — Exhaustive paired probes complete; negative information survives reweighting

At the owner's check, recovery worker16738603_0 completed in27m43s and
reducer16738604 in24s, both exit0. The scheduled status audit also completed;
no study jobs remain queued or running. The actual group's complete centre
replay has maximum error0 at the unchanged tolerance. All four groups pass
the direct prior-weight/zero-support checks. The reducer verified output
hashes, replayed36 source aggregates, and preserved unprobed rows bitwise.

Saved completion_audit.json in inference_v39_cohort_curvature_20260928_v2:
a light Python/NumPy audit verified pinned source receipts, all four exact
array hashes, and reproduced all32 reported per-object/prior score and
information results from the saved nine-node sums. Preparation had already
passed ten tests. Large centre-array hashes were checked by the scheduled
reducer, not reread on the login node. No implementation or model changes.

Actual row32's exhaustive minimum information eigenvalue is-248854.33 with
expanded weights and-253958.42 with corrected weights. Its repaired sampler
passes all12 comparisons against exhaustive moments (two priors, three
seeds,8k/16k); its baseline passes0/12. The three focal mocks also retain
negative directions with both weights, and their unchanged nonbright sampler
fails all36 comparisons. Sampling errors are real, but do not explain away
these local negative directions.

After replacing the15 scientific probed rows, all36 partial-cohort summaries
remain indefinite. At16k, corrected minimum eigenvalues span-359262 to
-296561 for actual1024 and-64507 to-45578 for mock836. Both uncorrected
expanded arms also remain indefinite. Corrected focal distinct-atom ESS
is4.57–8.11 and one atom contributes30.5–45.5% of each focal numerator.
Concentration makes finite-prior resolution a concern, not an established
cause of population bias. This is a targeted subset at one centre and h=.001;
it neither overturns the old full-mock positive-information result nor
completes the stopped full-cohort comparison. No accepted Newton estimate or
0.3% claim. The next bounded diagnostic would check smaller shear spacings
and local likelihood profiles for these objects; no new run was launched.
See [completed paired probes](V39_POPULATION_AUDIT.md#completed-paired-exact-curvature-probes).

## 2026-09-28 — Three exhaustive mock probes complete; repair actual-centre replay

At the owner's status check,16735501[1–3] completed exit0 in27m27s–27m45s.
Worker0 failed after10m41s: one of65,536 log contributions differed from the
saved actual centre by1.43686243e-7, exceeding the fixed1e-7 replay tolerance.
The dependent reducer never ran; cancelled obsolete16735502. Preserved the
failure and all completed outputs. The scheduled audit16735503 recorded
failure but did not notify an agent; this was discovered at the user check.

Verified completed mock exact-array hashes and saved partial_mock_review.json
in inference_v39_cohort_curvature_20260928_v1. Exhaustive minimum information
eigenvalues for expanded/corrected weights are respectively: mock151
-46145.90/-46150.64; mock734 -28283.90/-28283.91; mock853
-25551.01/-28115.16. Thus these three local negative directions persist
under exhaustive integration and both weightings. Sampling errors remain
substantial (individual16k information-entry errors up to18606), but do not
fully explain the negative curvature. Corrected distinct-atom ESS is6.44,
5.76,8.11. These are per-object, post hoc diagnostics, not a full-cohort result.

Inspection found a replay-input discrepancy: the saved centre-only scan used
the original full-precision centre, while the nine-node stencil rounds it to
14 decimal places (component differences about2e-15). Updated only the replay
path in scripts/probe_v39_cohort_curvature.py to use the original exact float
inputs, retaining the existing rounded inputs for all scientific stencil
sums. Before the full rerun, a focused check over the interval containing
the failure must reproduce rounded-input violations and pass original-input
replay at the unchanged1e-7 tolerance; otherwise it stops. This explanation
is pending that GPU check, not yet established.

New root inference_v39_cohort_curvature_20260928_v2 reuses the three completed
mock groups with their original hash receipts and protocol identities. Only
actual worker0 is rerun. Ruff/AST/shell syntax passed. Preparation16738602
gates one-GPU16738603_0, then reducer16738604; status-only16738605 is scheduled
in two hours. Scientific settings and gates are unchanged. No full-cohort
continuation or model change. See [replay recovery](V39_POPULATION_AUDIT.md#paired-probe-replay-recovery).

V2 preparation completed in27s with ten tests passed. GPU recovery check
confirmed the cause: over the interval containing the old failure, rounded
centre inputs reproduce exactly one1.4368624334792912e-7 violation, while
the original unrounded inputs reproduce every retained value bit for bit
(maximum absolute error0). Direct prior-weight replay also passed all nine
nodes. The actual worker then began its exhaustive scan with no failures.
Only the replay comparison changed; scientific stencil sums and the1e-7
tolerance remain unchanged. Allow about25–30minutes for the remaining
actual scan and reducer; completed mock results are reused.

## 2026-09-28 — Launch four exact curvature probes with paired prior weights

Owner requested continuation after clarifying that the old full-mock positive
information and new subset negatives are different comparisons. Added
scripts/probe_v39_cohort_curvature.py: use the existing likelihood loaders,
normalizations, finite-difference moments and subset reductions for actual
row32 and mock rows151,734,853. These are post hoc stress probes from the
completed subset decomposition, with original four-row density groups.
No new model, prior correction, seed or likelihood centre is introduced.

Sum all28,012,032 atoms at the same nine shear nodes for both expanded and
domain-corrected weights. Reuse each expanded density calculation through
the exact weight ratio; check this against direct corrected densities at
all nodes, including zero-weight atoms. Replay the existing actual group's
full centre array. Retain new mock centre arrays for possible later reuse.
Compare old baseline/bright-repaired moments against the exhaustive reference
at8k/16k for all three seeds, with unchanged numerical tolerances. Separately
replace only scientific rows of the probed groups in the existing subset
totals, exclude padding, verify all untouched moments bitwise and replay
the source aggregate summaries. This is partial diagnostic replacement,
not full-cohort closure or a0.3% calibration test.

Ruff, AST and shell syntax passed. Frozen root:
inference_v39_cohort_curvature_20260928_v1. CPU preparation16735500 runs
the ten existing exact-tail/reduction tests and checks source hashes and
the old actual centre array. Four one-V100 workers16735501[0–3] depend on
preparation; reducer16735502 follows success. Two-hour worker limits and
four-GPU ceiling; no other study GPU jobs were active. Initial expected
runtime30–45minutes from the earlier nine-node pilot, pending actual scan
throughput. Status-only audit16735503 is scheduled in two hours. Startup
monitoring is underway. See [paired exact probes](V39_POPULATION_AUDIT.md#paired-exact-curvature-probes).

Preparation completed exit0 in24s, with ten tests passed and the retained
actual centre hash verified. The groups contain15 scientific observations
and one padding row (mock732), excluded from aggregate replacement. Focal
measured magnitudes are19.4751(actual32),21.04614(mock151),22.18870(mock734),
20.13503(mock853). A lightweight reduction precheck exactly reproduced36
source aggregate score/information/eigenvalue comparisons with the original
padding masks. Active/frozen probe sources agree. Resource receipts confirm
one32GB V100 per worker; startup checks continue through the first scans.

All four workers passed direct expanded-to-corrected weight replay at all
nine nodes, including zero-support checks, and advanced beyond1,064,960
atoms each with no failures. The actual group's retained centre replay also
passes through the scanned prefix. The first65 blocks take about51s per
worker, suggesting roughly22minutes of scanning plus startup/reduction;
allow about25minutes remaining at this check. Leave the normally advancing
jobs and dependent reducer running; final exhaustive and aggregate results
remain pending.

## 2026-09-28 — Clarify positive full-mock result versus negative short diagnostic

Owner recalled that adding bright atoms made the mock information positive.
Rechecked the frozen expanded comparison and current subset results: this
is correct for the original full16,314 mocks at16k draws (minimum eigenvalue
+112804, unstable g1 proposal0.1473); full actual20k remained indefinite
(-995019). Added a comparison-boundary note to
[completed short diagnostic](V39_POPULATION_AUDIT.md#completed-short-cohort-diagnostic).
The new1024/836 test also changes numerical integration, and retains both
expanded and domain-corrected weights. Both repaired subset arms are negative
at16k; this cannot attribute the difference from the earlier full-mock result
to reweighting alone. Full-cohort controlled results remain unavailable
because that run was stopped. No scientific settings or jobs changed.

## 2026-09-28 — Short cohort comparison completes; negative information persists

Workers16735075[0–3] completed exit0 in33m30s–33m47s; reducer16735076
completed exit0 in20s. All eight centre scans and selected-density replay
checks passed. Verified protocol/combined-moment hashes, four worker receipts
and all small repair arrays; large centre arrays were not rehashed during
this completion audit. Confirmed fainter moments remain unchanged. Saved
completion_audit.json in inference_v39_cohort1024_recovery_20260928_v1,
refreshed status_latest.json and cancelled the now-unnecessary16735077 audit.

At16,384 draws, both expanded and domain-corrected bright-repaired arms
have indefinite summed information in all three seeds for actual1024,
actual same836 IDs and mock836. Corrected actual minimum eigenvalues are
[-315697,-359242,-296517]; corrected mock values are[-42441,-38947,-70409].
There is no accepted Newton estimate. Some lower-rung cases and unrepaired
mock seeds are positive; do not generalize the16k result to every rung or
select those positive cases as recovery evidence.

Corrected repaired bright per-object8k→16k changes are at most0.00901 in
score and8.85 in information entries. Along each seed's minimum-information
direction, the actual bright contribution is about-246k, with row32 a large
negative contributor; fainter rows add-50k to-113k. In mocks the bright sum
is positive(+34k to+35k), while fainter rows contribute-74k to-104k, with
rows151,734,853 prominent. Fainter sampling remains unstable. Reweighting
and bright-only repair do not resolve this fixed-centre diagnostic; this
does not establish the full likelihood shape or the original20k outcome.
Next useful bounded diagnostic: exact local likelihood checks for the
dominant bright actual row and fainter mock contributors. No new GPU jobs
or automatic full-cohort continuation were launched. See
[completed short diagnostic](V39_POPULATION_AUDIT.md#completed-short-cohort-diagnostic).

## 2026-09-28 — Replace full-cohort recovery with a short1024-identity diagnostic

Owner asked why the comparison was slow and requested smaller test jobs.
Stopped workers16733516, dependent reducer16733517 and status audit16734855.
Preserved293 completed baseline chunks (157 mock,136 actual), original
inputs/protocols and logs; no bright repairs had completed. The full run
will not resume automatically. Its266 full-prior centre scans dominated
the remaining estimate, about15hours on four GPUs.

Extended scripts/run_v39_cohort_recovery.py to take a fixed prefix, reuse
hash-verified baseline chunks after checking scientific settings against
the source protocol, and reduce the actual subset length. Default full-
cohort behavior remains available; frozen earlier drivers are unchanged.
New root inference_v39_cohort1024_recovery_20260928_v1 takes the first1024
original identity slots in each sample, preserving four-row density groups,
object RNG IDs and padding exclusions. Both priors, three seeds, all draw
rungs, centre, normalization and measured selection are unchanged. Repair
requires only eight groups (four scientific bright rows per sample).
This diagnoses curvature and integration stability; it cannot establish
0.3% bias or settle the original full-cohort failures.

Ruff, AST and shell syntax passed. Preparation16735074 checks source hashes,
runs the existing ten helper/reduction tests and copies the sixteen cached
baseline chunks. It gates four one-V100 workers16735075[0–3], followed by
reducer16735076. The prior GPU preflight is explicitly inherited; every new
repair still requires direct centre-density replay. Worker limits are2hours,
with an initial30–45minute runtime estimate plus queue time. Status-only
audit16735077 is scheduled two hours later. Startup validation is pending.
See [short cohort diagnostic](V39_POPULATION_AUDIT.md#short-original-cohort-diagnostic).

Preparation completed exit0 in25s; all ten tests passed. The prefix contains
1024 actual measurements and836 scientific mocks (188 flagged padding rows),
with four measured-bright rows in each. All sixteen baseline chunks were
copied after hash verification. All four32GB V100 workers started; resource
receipts confirm the same hardware as the source run. AST comparison confirms
repair, evaluate and load_pair are unchanged from its frozen driver, and
the new active/frozen driver hashes agree. All four workers then verified
their cached baseline chunks and reached their first full-prior centre scan
(mock groups8,9,44,145), with no failures. Startup includes about five minutes
of loading, verification and first density compilation; the30–45minute total
remains provisional pending sustained scan throughput. Leave the bounded
run, dependent reducer and status audit running.

## 2026-09-28 — Launch controlled original-cohort prior and bright-repair comparison

Owner requested continuation. Added scripts/run_v39_cohort_recovery.py,
reusing the frozen likelihood loaders, sampler, exact-tail helper and Newton
reduction. The [original-cohort protocol](V39_POPULATION_AUDIT.md#original-cohort-recovery-comparison)
keeps actual20k and matched16,314 scientific mocks at the same centre and
compares two priors × baseline/bright-repaired integration. Three seeds and
every draw rung remain fixed; no optimization or production promotion.

The baseline uses the original expanded-prior proposal for both targets,
preserving common draws/probabilities while target weights and normalization
change. Repair covers all135 actual/134 mock measured-mag<20 rows in266
original groups. Fainter rows remain in the estimates and numerically
approximate; their known integration errors and remaining draw drift must be
reported. This is a bounded partial repair, not a claim of reliable whole-
sample integration. One expanded centre scan serves both prior arms via
the exact prior-weight ratio; selected-density replay verifies that reuse.

Validation before submission: Ruff/AST/bash syntax passed. Preparation16733505
checks input hashes/padding and runs ten existing targeted tests. GPU
preflight16733506 gates four one-V100 workers16733507[0–3] and checks the
historical K1024 replay, corrected nine-node K16,384 sums and prior-weight
identity including zeroed added atoms. Reducer16733508 follows successful
workers; status-only audit16733509 is scheduled about two hours later.
Frozen root inference_v39_original_cohorts_recovery_20260928_v1 pins driver
and source receipts. Initial runtime estimate20–24hours,36hour limits,
four-GPU ceiling intact. Startup is being monitored before yielding.

Preparation16733505 passed all ten tests and source/padding checks in22s.
During startup review, caught a replay-only configuration mismatch: historical
primary arrays use min_ess32/max_fraction.5, whereas the newer parent defaults
are256/.25. Stopped preflight16733506 and cancelled pending16733507/16733508/
16733509 before scientific workers allocated. The corrected replay now calls
the frozen historical evaluator itself; scientific K16,384 settings and all
comparison thresholds are unchanged. Preserve v1 with a superseded receipt;
active root is inference_v39_original_cohorts_recovery_20260928_v2.
Ruff/AST passed again. New preparation16733514 gates preflight16733515,
four workers16733516[0–3], reducer16733517 and two-hour status16733518.

V2 preparation completed exit0 in12s, ten tests passed. GPU preflight passed:
both historical actual1958/mock4460 arrays replay bit for bit; corrected
nine-node K16,384 sums agree within1.05e-13/1.40e-13 in log space. The
weight-ratio identity and zero-weight added-atom checks pass, as do both
target evaluations with a common proposal. Peak allocated GPU memory18.07GB
on a32GB V100. Dependent worker startup is now being checked.

Preflight finished exit0 in5m04s. Four workers16733516[0–3] started on
th-cl-nv01 with one32GB V100 each. Followed startup until every worker
completed its first original four-row group for both priors and all three
seeds, with no failures. Work is balanced as78–79 baseline chunks and66–67
repair groups per worker. Active/frozen source hashes agree. Jobs now
progress normally; leave the dependent reducer and two-hour status audit
in place. Runtime estimate20–24hours awaits sustained-checkpoint refinement.

## 2026-09-28 — Intermediate-magnitude repair passes all96 numerical checks

Owner reported completion. Workers16733295[0–3] completed exit0 in26m16s–26m20s
and reducer16733296 in23s. Refreshed external status, verified four completion
and centre-replay receipts plus all small output hashes; the reducer had
also verified the four retained centre-array hashes. See the
[completed intermediate pilot](V39_POPULATION_AUDIT.md#completed-intermediate-magnitude-pilot).

Repaired integration passes24/24 focal checks and72/72 companion checks,
versus0/24 and24/72 for the baseline. Across all16 rows, repaired maximum
absolute score/information-entry errors are0.08432/1.29420, within unchanged
entrywise thresholds for every seed and both8k/16k. No favorable seed or
focal observation was substituted. All six draw rungs remain saved.

Exhaustive per-observation information is still indefinite on all four focal
rows. This is not evidence that the full-sample matrix must fail, and the
repair correctly preserves negative directions. Three focal posteriors have
distinct-atom ESS2.48–10.25; the fourth has ESS1212.46 and only28.59% minimum
candidate coverage, confirming that its tail cannot be discarded. Finite-prior
resolution and local optimization remain distinct from integration accuracy.

The original actual20k negative information and matched16,314 g1=0.1473
proposal are still open full-sample comparisons. This check changes no code,
models, weights, thresholds or production defaults, and launches no additional
GPU job. Next work returns to those original fixed cohorts with the validated
integration, keeping population correction separate from numerical repair.

## 2026-09-27 — Launch four intermediate-magnitude probes with existing integration methods

Owner authorized continuing while avoiding overengineering. Added only the
diagnostic driver scripts/probe_v39_intermediate.py, reusing the frozen parent
evaluator, exact-tail proposal, corrected-prior loader and moment reduction.
The [fixed four-probe protocol](V39_POPULATION_AUDIT.md#intermediate-magnitude-numerical-pilot)
compares baseline and repaired integration against nine-point exhaustive sums:
two fresh-mock stress probes (one per measured-mag20–22/22–24 bin) and two
fixed-seed actual20k controls. Original four-row groups and random identities
remain unchanged; all twelve incidental companions are also reported.

One full-prior scan per worker accumulates nine-node sums and retains centre
logs/top16,384 for reuse. No estimator, production setting, model, prior or
tolerance changes. Keep all three existing seeds and all draw rungs; apply
the previous gate to every focal row at both8k/16k. This does not replace
the unresolved full actual20k/matched16,314 comparison or establish0.3% bias.

Validation before submission: Ruff, AST and launcher bash syntax checks pass.
Frozen root inference_v39_intermediate_pilot_20260927_v1 pins source and input
receipts. Preparation16733294 runs ten existing targeted tests and input
checks; workers16733295[0–3] depend on its success, then reducer16733296.
Status-only audit16733297 is scheduled in about two hours. Queue was empty
before submission; four persistent one-V100 jobs in inter preserve the cap.
Startup and numerical replay checks are being monitored before yielding.

Preparation subsequently completed exit0 in22s, ten tests passed. Selected
rows are fresh4826/3602 (measured mags20.45964/22.57042) and actual11173/2510
(21.1134/22.3313). Four one32GB-V100 allocations started on th-cl-nv01;
active/frozen driver hashes match and resource receipts confirm one visible
GPU per worker. Both fresh baseline groups subsequently replayed all saved
arrays exactly across all three seeds, and actual baselines completed.
All four exhaustive scans reached1,064,960 of28,012,032 atoms in50.4–50.8s
of scan time, without failures. This is meaningful normal progress; estimated
30–40minutes remain including tail comparisons. Centre replay and final
numerical gates are pending. Leave jobs running with their dependent reducer
and two-hour status audit; no full-sample shear conclusion follows yet.

## 2026-09-27 — Fresh-parent inference completes; remaining draw drift is in nonbright rows

Owner requested a check. Workers16732143[0–3] completed exit0 in2h20m–3h12m23s
and reducer16732144 in57s. All1283 baseline groups and43 bright groups are
complete, with no failures. Refreshed the external status JSON and documented
the [completed first-pass result](V39_POPULATION_AUDIT.md#completed-fresh-parent-first-pass-diagnostic).
Verified aggregate moments/protocol/input hashes, four worker receipts and
43 bright-group receipts. The scheduled reducer had verified every consumed
moment-file hash and unchanged nonbright rows. Small saved-array reductions
independently reproduce the final proposals and attribute draw-count movement
within1e-14; output is completion_audit.json in the frozen run root.

On5,130 scientific mocks (44 measured-bright, two numerical copies excluded),
the three16k bright-repaired g1 proposals are0.016889±0.007046,
0.016785±0.006936 and0.017216±0.007600, versus injected0.02. These are one
fixed-centre Newton step, not converged estimates; seeds share observations.
All36 arm/seed/rung summed information matrices are positive, but no result
meets the small-step flag. Repaired8k→16k movements are+0.000340,+0.000421,
+0.000165: still above the0.000060 scale corresponding to0.3% of injected shear.

Exact additive attribution shows bright rows contribute at most2.80e-7 in
absolute net g1 movement; measured-mag>=20 rows cause the remaining drift.
Bright rows still account for88.8–89.3% of squared g1 sandwich influence,
leaving finite-prior resolution and local quadratic validity unresolved.
Next: numerical probes of intermediate-magnitude rows before a larger run
or recentering. No production code/model/prior changes or new GPU submissions.

## 2026-09-27 — Exact-tail pilot passes at K16k; launch fresh-parent first-pass diagnostic

Owner reported completion. Workers16731880[0–3] finished exit0 in17m35s–17m43s
and reducer16731881 in20s. K16,384 passes all24 predeclared checks across
four focal rows, three seeds and both8k/16k draw rungs. K4096 passes22/24,
K1024 passes6/24; neither passes the combined gate. Verified worker hashes
and centre replay receipts. The hardest actual probe has maximum score error
0.010354 and information-entry error20.6443; the latter is just within the
corresponding21.3091 tolerance. No thresholds or seeds changed, and no general
convergence or0.3% shear result is asserted. See the
[completed pilot and next protocol](V39_POPULATION_AUDIT.md#completed-exact-tail-pilot-and-fresh-parent-first-pass).

Added scripts/run_v39_parent_inference.py and froze root
inference_v39_parent5130_exacttail_20260927_v1. Reuse the verified corrected
prior and normalization at the already frozen actual-sample centre, independent
of these new mock results. All5,130 scientific mocks get a K16,384 Gaussian
baseline; replace only44 measured-mag<20 rows in43 original groups with the
accepted exact-tail integration. Keep three original seeds and every draw rung.
Two flagged numerical copies fill the final group and are excluded from every
estimate. Existing input IDs preserve the per-object RNG streams. This is one
fixed-centre Newton diagnostic, not a converged inference or model acceptance.

Validation: Ruff/AST/bash syntax pass. Initializer16732141 completed exit0 in16s,
including10 targeted proposal/reduction tests, all four large saved-centre
hashes, fresh-mock source hashes, measured-cut checks and no scientific-row
loss. GPU preflight16732142 verifies original four-row group replay and audits
the nearly seed-constant K1024 candidate curvature residual separately from
sampling noise. Canonical K16,384 replay must pass before the four persistent
one-GPU workers16732143[0–3] start; CPU reducer16732144 depends on them.
Initial runtime estimate3–4hours,6hour limits, with the four-GPU ceiling intact.
Status-only audit16732145 is scheduled for about two hours. No production model,
prior, numerical default or acceptance threshold changed.

Preflight16732142 subsequently passed in3m21s and all four workers started on
th-cl-nv01. Fresh four-row groups replay bit for bit. The same-atom direct1024
versus canonical16,384-prefix diagnostic confirms a separate density batch
effect: information11 differences+36.5221 for actual5359 and−41.3445 for
mock3192, from log-integral differences about2e-5. These explain much of the
seed-stable small-K curvature residual; the exact internal numerical operation
is not isolated. Canonical K16,384 replay passes, supporting the new run's
full4096-atom density blocks without rewriting any old result.

Worker startup was monitored through saved inference checkpoints. At3m52s
elapsed all four workers were progressing normally, with40 of1283 baseline
groups complete across the array (all three seeds per group), no failure
receipts, and one32GB V100 visible per worker. The bright work is distributed
as11/12/12/8 groups. The scheduled two-hour status audit remains active;
the3–4hour initial estimate is still appropriate pending bright scan progress.

## 2026-09-27 — Launch exact-centre conditional tail proposal pilot

Owner said to continue numerical repair. Added scripts/v39_exact_tail.py,
scripts/probe_v39_exact_tail.py and tests/test_v39_exact_tail.py. The
[conditional-tail protocol](V39_POPULATION_AUDIT.md#exact-centre-conditional-complement-pilot)
uses the same exact candidate sets and replaces Gaussian whole-prior tail
draws with90% exact centre-likelihood complement plus10% conditional scene
prior. It excludes exact candidates from draws and reports actual CDF-bin
probabilities. This preserves the numerator integral's expectation and full
positive-prior support; log/derivative accuracy still requires the existing
numerical gate. No production runtime or scientific settings changed.

Frozen root inference_v39_exacttail_pilot_20260927_v1 pins script/helper/test
hashes and both reference results. Keep four original groups, K1024/4096/16384,
seeds8701–8703, draw ladder512–16384, h0.001, prior and normalization unchanged.
All scores and information entries must pass at both8k/16k for every seed.
Retain expensive full-prior centre contributions on disk for future reuse;
verify their sum and selected atom densities against the preceding run.

Validation: Ruff and AST passed; CPU16731872 ran the new proposal tests plus
tests/test_catalogue_null.py,37 passed in22.11s, zero regression failures.
Checks include an exact expectation oracle with an atom whose centre
likelihood is zero but displaced likelihood is positive, excluded candidates,
extreme log scales, nested draws and per-object uniform streams. Launcher
bash syntax checks passed. The first array16731873/reducer16731874 were
cancelled while pending to correct a reference-root substitution; no worker
allocated and no scientific result was produced.

Corrected array16731880[0–3] depends on successful tests and uses four one-GPU
V100 workers in inter,45minute limits; CPU reducer16731881 follows it.
Status-only audit16731875 is scheduled in about two hours. Startup checks
confirmed one32GB V100 visible per worker, all four on th-cl-nv01. At4m05s
elapsed all workers had scored1,064,960 of28,012,032 atoms with no failures;
steady throughput was about30s per1.05million atoms after compilation.
Estimated15–20minutes remaining including proposal comparisons. Active and
frozen source hashes match. Next, compare every predeclared check with the
frozen exhaustive sums; no accepted shear result or model promotion is implied.

## 2026-09-27 — Whole-prior ranking completes; mock probes converge but actual curvature still fails

Owner requested status. Array16731606[0–3] completed exit0 in17m24s–17m28s
and reducer16731607 in19s. Verified all worker/reference NPZ hashes and
all four exact-centre replay receipts; refreshed the external status JSON.
Documented the [completed whole-prior pilot](V39_POPULATION_AUDIT.md#completed-whole-prior-ranking-pilot-on-september27).

At K16,384 both mock focal rows pass all six seed/rung checks. Actual1958
passes0/6 and actual5359 passes3/6; their scores pass but curvature does not
consistently meet the unchanged entrywise tolerance. Minimum nine-node
candidate coverage is99.629465%,99.803972%,99.915333%,99.996588% for the
two actual and two mock probes respectively. All three K arms still fail
the combined gate. No seed or result was omitted.

Added external tail_diagnostic.json using small saved-array reductions:
candidate-only versus exhaustive log-numerator finite differences isolate
the remaining tail contribution with normalization cancelling. Omitting the
tail at K16,384 changes actual1958's g1 score by+0.66258 and its11information
entry by+172.77, so coverage alone cannot justify truncation. No runtime code,
models, production prior, gate or defaults changed; no new GPU job submitted
at this status check. Next work is a controlled complement-accuracy repair
on these probes, followed by fresh unselected-parent mock inference after
numerical acceptance. A0.3% shear result remains unestablished.

## 2026-09-27 — Corrected prior verified; pool-limited ranking improves but fails integration gate

Owner requested status. All16730737workers finished exit0 in55m04s–55m07s,
reducer16730742 in26s, independent replay16730778 in1s, and scheduled audit
16730779 in2s. Verified worker NPZ hashes and receipts. All28normalization
shards reproduce old-weight masses exactly; directly reweighted exhaustive
numerators match the earlier component reconstruction within2.85e-14.
Documented the [completed pilot](V39_POPULATION_AUDIT.md#completed-domain-matched-pilot-on-september27).

Every arm fails the unchanged score/information gate on all four focal rows.
Exact ranking still materially improves the1024-atom stratum: minimum
stencil coverage for actual1958 rises from0.000459% to93.4727%, and mock4460
from0.0000239% to98.7186%. Their existing Gaussian ranker misses most local
mass, exposing the integral to noisy complement draws. However, the complete
broader pool itself captures only94.4472% for actual1958 and98.7884% for
mock4460. Saved restricted_pool_errors.json separates deterministic pool
errors from complement noise; increasing K within that pool cannot recover
the omitted mass. No gate, observations, seeds or reported results were changed.

Fresh one-draw generation16730768 also completed exit0 in15s:20,000unselected
input attempts,8,972usable,5,130retained including44measured-bright rows.
No top-up or generating-atom insertion. No inference result is yet accepted
on that cohort.

Continue the already authorized numerical repair with a small
[whole-prior ranking probe](V39_POPULATION_AUDIT.md#exact-whole-prior-ranking-follow-up).
Added scripts/probe_v39_full_rank.py; frozen root
inference_v39_fullrank_pilot_20260927_v1. At the same centre, evaluate all
atoms once and retain the highest exact contributions; compare K1024,4096,
16384 with unchanged full-support complement, seeds and both draw rungs.
Reuse the completed nine-node exhaustive references and normalization.
Centre sums and selected atom densities must replay within log-space1e-7.
Array16731606[0–3] and reducer16731607 stay within four one-GPU V100 workers;
30minute worker limits. Ruff and AST checks pass; existing numerical tests
are unchanged, with runtime replay checks added for this diagnostic. Startup
succeeded on four32GBV100s, one per worker; at6minutes elapsed roughly5.2million
atoms had been scored per worker without failure receipts. Estimated another
15–20minutes at observed throughput. Scheduled status-only audit16731657
for about two hours; it records results without promoting a sampler. No large
inference, model change or0.3% claim follows from the failed pool-limited gate.

## 2026-09-27 — Launch domain-matched bright pilot and prepare unselected-parent mocks

Owner authorized the recommended next steps. Added
scripts/run_v39_domain_matched.py and scripts/run_v39_parent_mock.py; the
[pilot protocol](V39_POPULATION_AUDIT.md#domain-matched-pilot-authorized-on-september27)
and [fresh input cohort](V39_POPULATION_AUDIT.md#fresh-unselected-parent-mock-preparation)
are documented with frozen external roots, seeds, gates and limitations.
No production models, priors, measurements, cuts or defaults changed.

Candidate initialization16730721 passed in28s: zero weight for848,650 added
atoms outside the generator domain,7,201,662 valid bright atoms sharing the
original bright mass, and every nonbright weight unchanged. Four original
groups retain focal actual rows1958/5359 and mock rows4460/3192; the latter
row per sample is selected by fixed seed2026092711, independently of inferred
shear. Normalization reuses cached per-atom C8 probabilities and replays all
64-QMC forward draws/node with identical shard seeds, reducing both old and
new weights to verify the baseline before comparing algorithms.

Four one-GPU V100 workers16730737[0–3] run in inter, CPU reducer16730742
depends on completion, and independent numerator replay16730778 follows it.
Compare Gaussian candidate ranking, exact centre ranking within the existing
131072-atom location pool, and exact ranking within its union with a whole-
catalogue Gaussian131072pool. Each retains K1024, the original tilted
complement, both8k/16krungs and seeds8701–8703. All28,012,032 aligned atoms
are summed at all nine shear stencil points for each four-row group. Exact
pool coverage and all score/information entries are checked against the
predeclared numerical gate; this is not a0.3% scientific acceptance test.

The first allocation16730727 was cancelled during startup because a two-node
nodelist requested both hosts per worker. Corrected launcher to nodes=1 and
one V100; cancelled its reducer16730729. No completed stage was used from
that allocation. Current workers have one GPU each, three on th-cl-nv01 and
one on th-cl-nv02. Their first normalization shards all replayed the old
weighted masses exactly, maximum difference0.0, and progressed to second
shards without failure receipts. Estimated1.5–2hours after startup;4hour
limits. Status-only audit16730779 is scheduled in about two hours and writes
status_latest.json; it does not notify the owner or promote results.

In parallel, CPU16730765 prepared20,000 uniformly sampled unselected inputs
from all100ConstGold cases in1m43s, without truth cuts or input measurement
selection. All identities match;9,982 target-role and10,018 other-role rows,
with four chance overlaps with the former selected20k. Their new forward
generation16730768 waits for successful preparation and the GPU array to
end, preserving the combined four-GPU ceiling. It performs one normal draw
and one measured selection, reporting retained count without top-up. No
generating input is inserted into the inference prior. Generation is useful
regardless of sampler-pilot outcome; subsequent inference remains pending.

Validation: Ruff/AST checks pass. Tests16730722 passed317/skipped3 without
artifact roots; rerun16730766 with configured roots passed319/skipped1,
zero regression or snapshot failures. Parent input construction validates
source hashes and zero unmatched rows; candidate weights and frozen execution
identities passed. No new shear result or resolved sampler failure is claimed
while the normalization/sampling/exhaustive stages run.

## 2026-09-27 — Deep audit separates bright prior domain, cohort conditioning and integration

Owner asked whether population mismatch requires reweighting, then requested
a deeper investigation. Added read-only scripts/audit_v39_population.py and
scripts/profile_v39_population.py; full evidence and interpretation are in
[the population audit](V39_POPULATION_AUDIT.md#question-and-scope), linked from
the matched-input comparison. No production prior, measurement sample, model,
cut, seed or optimizer setting changed.

CPU16730323 scanned69,956,785 unselected input rows across100cases plus both
weighted priors, and audited the saved per-observation curvature. It completed
exit0 in1m35s. The original prior's largest intrinsic-property marginal CDF
discrepancy is0.02382percentage points; no actual input lies outside the
generator morphology domain. The expanded prior assigns10.5418% of its
bright mass to848,650 added rows outside that domain. A histogram-only
counterfactual excludes that mass, preserves the bright total and nonbright
weights, and reduces the bright n CDF discrepancy from9.8729 to0.6158points.
This is generator-support matching, not a truth cut on selected measurements.

CPU16730457 additionally compared27,982,713 unselected target-role contexts
from80cases with F8/C8 neighbour features; exit0 in42s. Original-prior maximum
neighbour-feature marginal CDF difference is0.03609points. These descriptive
checks do not prove full joint equality or0.3% calibration. The matched-input
mock starts from an already selected cohort and applies a second selection;
documented its distinct conditional input distribution and normalization.

The135 measured-bright actual rows contribute-999244.60 along the final
minimum-information direction, versus-995018.53 for all20k. Influential rows
have sampled ESS near1 and large8k-to16k changes. Three GPU jobs16730364[0–2]
sum all28,012,032 atoms for two post-hoc influential probes at the shared
centre and g1±.001; original four-row groups and4096 density chunks retained.
CPU reducer16730370 depends on all workers. Original and domain-matched
numerators are reconstructed from separately accumulated support components;
the latter is explicitly not a selection-normalized inference result.
External root: inference_v39_population_audit_20260927_v1. Source snapshots,
protocols, hashes, coverage and output receipts are preserved.

Validation: system Ruff and AST syntax checks passed for both new scripts;
full tests16730371 passed319, skipped1, no regression/snapshot failures.
No new model training or large inference continuation was launched.

All three exact workers completed exit0 in13m42s–13m46s; reducer16730370
completed in9s. Exhaustive expanded g1 information is-158392.95 for actual
row1958 and+165073.89 for mock row4460, compared with their16k estimates
-476790.52 and-233919.29. The actual16k score also has the wrong sign.
Thus the sampler demonstrably misestimates influential rows under the same
fixed likelihood; population reweighting alone cannot explain the discrepancy.
Exhaustive distinct-atom ESS is5.608 and1.662, with largest atom fractions
39.57% and76.70%, so finite-support resolution remains a separate concern.
Outside-domain atoms carry13.284% and3.431% of each probe's likelihood mass.
The domain-matched numerator g1 information becomes-8834.75 and+170036.63;
this is not a normalized candidate inference or a full-sample curvature test.
CPU16730526 verified exact reconstruction of all old weights in2s.

Recommended next sequence is domain-matched candidate weights with fresh
normalization, broader exhaustive bright integration checks, independent
prior-resolution checks and an unselected-parent small mock closure. All
diagnostic jobs are complete; no scheduled continuation remains. No accepted
shear or0.3% result is claimed. Histograms, post-hoc two-row selection, finite
QMC normalization, h=.001 and floating-point density limitations are recorded.

## 2026-09-27 — Expanded prior completes; mock curvature positive but inference unstable

Owner reported completion. sacct confirms16729459[0–3] all exit0 in
1h39m41s–1h41m24s; status-only audit16729469 also completed. Verified all
eight5000-row worker receipts and SHA256 hashes, complete comparison.json,
and absence of failure receipts. No active jobs remain. Updated external
jobs.json/status_latest.json and documented the
[expanded-prior result](V39_MATCHED_INPUT_MOCK.md#completed-expanded-prior-result-on-september27).

At16384 draws, minimum information eigenvalues change from−1767373.8285
to112803.9799 for matched mocks,−2394269.7775 to−995216.0549 for actual
same IDs, and−2387169.1061 to−995018.5262 for actual20k. Positive mock
information is not a successful recovery: its g1 proposal changes from
0.01905090 at512 draws to0.18813134 at8192 and0.14730972 at16384. The
final two-component step norm is about0.14356, beyond the historical0.05
recentring guard; conditional sandwich errors are about(0.90423,0.27986).
Both actual samples become positive at4096/8192 but return to indefinite
information at16384. The summary flag same_problem_at_shared_center=false
describes only the final mock matrix, not accepted or converged inference.

No accepted shear, robust0.3% result or optimizer continuation follows.
Expanded support and its changed proposal materially affect the result,
but prior support, integration error and local curvature remain unresolved.
No scientific code/settings/output changed and no extra GPU run launched.
Next useful diagnostic remains bright likelihood profiles with stronger
independent integration checks, without choosing a favorable draw prefix.

## 2026-09-27 — Expanded-prior normalization and replay pass; mock evaluation progressing

Owner requested status. All four16729459 workers remain running at56minutes
elapsed, with no failure receipts. All28 normalization shards and assembly
completed. Expanded-prior batching preflight passed exact equality of all
ten moment/diagnostic arrays for60 mock and64 actual rows, including bright
groups, on the same32GB V100 hardware.

Twelve of20 mock evaluation chunks are saved (12000 of20000 evaluation
rows, including explicitly excluded padding); every worker has entered its
fourth1000-row chunk. Actual-measurement evaluation follows each worker's
mock block and has not started yet. No comparison result exists yet. Recent
1000-row chunks take about346–361seconds, suggesting approximately45–55
minutes remaining including actual evaluation and reduction. Updated external
jobs/status snapshots and the
[expanded-prior record](V39_MATCHED_INPUT_MOCK.md#expanded-prior-comparison-requested-september27).
No scientific settings or code changed; no partial shear result inferred.

## 2026-09-27 — Restore extra bright atoms for the fixed matched-input comparison

Owner requested the expanded prior after the matched-input curvature failure.
Added scripts/run_v39_expanded_prior.py and documented the
[controlled expanded-prior comparison](V39_MATCHED_INPUT_MOCK.md#expanded-prior-comparison-requested-september27).
Reuse exact saved16314 mocks and original actual20k, fixed centre, V3.9
models, original four-row groups and sampler seeds/settings. Restore all
7,780,811 extra atoms with existing expanded weights, preserving nonbright
weights and total bright probability. Prior support is28,012,032 atoms.
No generating truth atoms are inserted and no new mock observations drawn.

Preparation validates the cached V3.9 support, identical original atom
features/coordinates/floors, fixed proposal metric and bright-mass allocation.
Old proposal C8 base probabilities stay unchanged; new atoms use their same
evaluation centre. Full selection normalization is recomputed at the fixed
actual centre with64 CRN QMC draws per atom/node and original shard seeds.
Both mock and actual likelihoods are reevaluated under the expanded prior;
final reduction compares with saved original-prior moments and all six draw
counts, rejecting indefinite information without a repair or forced estimate.

Frozen root: inference_v39_constgold20k_matched_expanded_20260927_v1.
CPU initialization16729457 passed its audit in25.8s of Python execution.
Full tests16729474 and initialization gate four persistent V100 workers
16729459[0–3], max4 concurrent GPUs, on inter/th-cl-nv01. Workers normalize
seven shards each, synchronize assembly, pass exact proposal-batching replay
on both datasets including bright groups, then evaluate both5000-row blocks.
Worker0 reduces after all workers finish. Estimated1.5–2hours after GPU
startup;4hour limits. Login resource snapshot retained; heavy work scheduled.
Syntax and system Ruff checks pass (py31 itself has no Ruff module).
No production default, model, optimizer, cut or seed changed; this remains
a selected-input fixed-centre diagnostic, not population closure or a0.3%
acceptance test. Startup/progress verification follows below.

Initial tests16729456 stalled in Python startup on cip-cl-compute4; a
read-only process check showed cl_sync_io_wait and no pytest output. Moved
the identical suite to inter/th-cl-1024us01 as16729474, updated the GPU
afterok dependency, then cancelled16729456. The successful initialization
used this inter CPU node. Scheduled status-only audit16729469 for about two
hours later; it saves a snapshot without agent wakeup or user notification.

Startup verified: replacement tests16729474 pass319 tests with1 optional
skip in27.28s. All four16729459 workers are running and have completed their
first two normalization nodes (zero shear and the fixed actual centre), with
finite positive masses0.01288–0.01293 and no failures. The cached proposal
C8 reference is zero shear; added atoms use that same point. Corrected this
description in the active driver's comment; its frozen running copy remains
unchanged and obtains the correct point directly from the pinned cache.
Saved current jobs/status snapshot; subsequent inference and replay stages
remain automatically gated. Long-run monitoring yields after this progress.

## 2026-09-27 — Completed matched-input mocks reproduce the curvature failure

Owner reported completion. Verified sacct for16729155[0–3] and16729156:
all completed with exit0, four GPU workers in30m30s–30m45s and comparison
in20s. Saved comparison.json/status.json report complete, with no failure
receipts. Of20000 exact input identities,16314 model draws survived C8 and
the new measured cuts; all3686 numerical padding rows are excluded.

At the fixed actual expansion point and16384 draws, observed-information
eigenvalues are(-1767373.8285,2002216.2944) for the mocks,
(-2394269.7775,3423542.5435) for actual measurements of the same IDs,
and(-2387169.1061,3445424.7863) for original actual20k. Read-only JSON
inspection confirms all three matrices are indefinite at all six draw
counts512–16384. No accepted shear estimate or bias result is available.
The failure therefore does not require actual-image measurement mismatch;
finite-prior support, integration error and local curvature remain unresolved.
This selected-input diagnostic is not population closure or a0.3% result.

Updated doc/V39_MATCHED_INPUT_MOCK.md with
[completed evidence](V39_MATCHED_INPUT_MOCK.md#completed-comparison-on-september27)
and limitations, plus external jobs.json/status_latest.json. Cancelled the
redundant pending status-only audit16729189 after checking results directly.
No scientific code, settings, seeds or outputs changed; no additional tests
or GPU runs were needed. A small bright likelihood profile with independent
integration checks remains the proposed next diagnostic.

## 2026-09-27 — Matched-input forward mocks requested for the actual20k cohort

Owner requested mocks from exactly the galaxy inputs associated with the
actual20k measurements, then chose normal forward sampling with the retained
count reported. Added scripts/v39_matched_input.py,
scripts/run_v39_matched_inputs.py and tests/test_v39_matched_input.py.
The preparation joins original(case,input_index) keys, checks catalogue
hashes, reconstructs physical F8/C8 crowding from complete scenes, and builds
the retained disk-emulator pairs without new truth cuts. Matching features
are checked against frozen development contexts where available. Generation
uses one C8 Bernoulli per input and one F8 draw per usable row at(0.02,0),
then disk transport and independent new measured0.6arcsec/25.8 cuts. No
replacement or top-up; both unusable and cut-failing counts are retained.

Inference retains the original weighted20.23m prior and evaluates at the
failed actual run's fixed expansion point, reusing its verified population
normalization. It compares generated observations, actual observations of
identical retained IDs, and original actual20k moments across the complete
draw ladder. Original four-row density groups are preserved using explicitly
excluded numerical padding at nonretained IDs. Importance draws retain
original row IDs. The matched selected input cohort is not population
closure; no shear-bias acceptance or new optimizer is implied.

Frozen active root: inference_v39_constgold20k_matched_mock_20260927_v2.
Preparation16729151, full tests16729152, generation16729153, replay16729154,
four workers16729155[0–3], comparison16729156 are chained by afterok with
invalid-dependency cancellation. Max4 GPUs; no other GPU jobs running at
submission. Initialv1 preparation16729136 and downstream16729138–41 were
cancelled before generation to preserve original density groups following
selection; logs retained. Its15 focused tests and316 full-suite tests passed
(3 optional skips). v2 reruns tests including the padding invariant. Ruff
and syntax checks pass; login resource snapshot retained, heavy preparation
and inference scheduled. See the
[matched-input protocol](V39_MATCHED_INPUT_MOCK.md#controlled-curvature-comparison).

v2 startup:16 focused tests pass; full-suite16729152 passes319 tests with
1 optional skip in26.80s. Preparation16729151 completes all100 cases in
108.8s with zero missing identities and7980 cached target-role feature
replays; reconstructed intrinsic ellipticities match recorded components
to7.50e-16. Generation16729153 completes in10.2s:1402 unusable inputs,
18598 usable draws,2284 measured-cut failures,16314 retained mocks, no
replacement.134 mock-bright rows and134 actual-bright rows share the
retained ID set.3686 numerical padding rows are explicitly excluded.
Replay16729154 passed exact equality of all ten arrays on128 original-group
rows in233.6s of Python execution. All four inference workers16729155[0–3]
then started. Comparison16729156 depends on their success; unlike the
stopped optimizer it records indefinite curvature as a diagnostic outcome.
Scheduled status audit16729189 saves a snapshot about two hours later;
it does not wake an agent or notify the owner.
Production startup verified for all four workers: first4/1000 likelihood
rows emitted in each original5000-row block, no failures. Long-run polling
stops after this startup check; CPU comparison remains automatically chained.

The exact actual20k cohort has9979 structural target-role rows and10021
outside that role; V3.9 development used the target-role parent. This is
verified from original per-case generator counts, not inferred from shape
flags. Retained mock counts are8118 and8196 respectively. Saved
input_role_audit.json and documented this population distinction without
applying a new cut or claiming it explains the failure.

## 2026-09-27 — Actual20k evaluated completely; first solve rejected negative curvature

Owner requested status at09:39 CEST. Slurm and saved receipts show all four
16726749 workers stopped at01:37 CEST after52m15–24s. Every5000-row worker
output and all20 normalization shards completed successfully. The reducer
rejected total observed-information eigenvalues(-2387169.1061,3445424.7863)
at the actual raw-mean centre(0.00983605988,-0.00029219982). The other
workers exited because worker0 stopped. No first estimate or bright update
was produced; all input, likelihood moments and normalization are preserved.
The scheduled CPU audit16726751 correctly recorded the stopped state, but
did not notify/wake an agent, as already documented.

Added scripts/audit_v39_actual_first_pass.py for a read-only saved-moment
audit: receipt/coverage/max-draw checks, curvature by magnitude, nested draw
counts, normalization contribution and influential-row diagnostics. It does
not drop observations or repair the solver. Ruff passes. Initial audit job
16728958 failed on importing SBSI from the external script location; fixed
the launcher to use this run's pinned inference runtime, then submitted
CPU retry16728966. Updated external jobs.json and
[actual-run status](V39_ACTUAL20K_BFGS.md#status-on-september-27-stopped-at-the-first-solve).

Retry16728966 completed in28s. All input/output hashes, row coverage,
finite moments and primary/max-ladder equality checks pass. The first-pass
matrix is indefinite at all six draw counts512–16384. Excluding the
normalization contribution algebraically leaves minimum eigenvalue
-2360166.8824, versus-2387169.1061 including it. The135 measured-mag<20
rows contribute-2239340.4103 along the negative eigenvector (93.8% of net
negative curvature). The774 mag20–22 rows contribute-216479.1020; both
fainter bins contribute small positive totals. Bright zero-view likelihood
relative-error diagnostics: median0.09066,90th percentile0.68966; these
are not shear errors. The same-proposal ladder cannot distinguish true local
non-concavity from correlated numerical integration error. Preserved
first_pass_audit.json; no cuts, solver replacement or GPU restart applied.
Next diagnostic is a safeguarded likelihood profile plus targeted bright
sampling checks, rather than treating this as a converged bias result.

## 2026-09-27 — Actual ConstGold20k input and five-bright-update run submitted

Owner requested the same current inference procedure on actual measurements
instead of model-generated mocks. Reused the first20k of the existing
randomly ordered uniform500k ConstGold plus-leg sample, seed20260914,
cases40–139, g=(0.02,0), after measured radius>0.6arcsec/MAG_AUTO<25.8
and disk usability, without truth-property cuts. Parent input hashes and
matching audit are preserved; this is a new inference on actual measurements,
not reuse of old V3.6 inference features. Pinned V3.9 F8/C8/disk and restored
20.23m prior are unchanged; new7.78m bright extension remains excluded.

Added scripts/v39_image_input.py and tests/test_v39_image_input.py, rejecting
misaligned/duplicate IDs, truth-filtered parents and invalid/cut-boundary
measurements. Prefix order is preserved; measured interior log disk gap is
derived without clipping. Extended scripts/run_v39_bright_bfgs.py to establish
a fresh full20k pass at this sample's raw mean, then five fresh bright-only
BFGS updates from its own first estimate. No mock centre or saved moment
initializes the scientific chain. Initial/fixed-base hashes, full worker
coverage and original four-row density groups are checked.24-row retrieval
and proposal batches require exact numerical replay on actual observations;
that replay's cached reference centre is excluded from the science result.

External root: inference_v39_constgold20k_bright20_bfgs_20260927_v1.
Init16726746 passed38 focused tests in24.26s, verified20k rows with zero new
drops, all100 cases,135 bright rows and536 evaluated rows including401
padding. Full-suite16726747:316 passed,1 skipped in44.53s. Ruff and syntax
checks pass. GPU replay16726748 passed exact equality on all ten arrays for
248 actual observations in5m05s. All four persistent workers16726749[0–3]
then started and emitted their first pass00 normalization masses without
failures. The afterany dependency on completed mock16721029 plus afterok
stage dependencies preserved the four-GPU ceiling. CPU status audit16726751
saves a snapshot about two hours later; it is not an agent wakeup or owner
notification. Estimated duration about
three hours from GPU startup, subject to actual throughput. See the
[actual20k protocol and limits](V39_ACTUAL20K_BFGS.md#numerical-procedure).

The earlier mock five-update run completed: all16721029 workers exit0,
wall1h37m40s. Final g=(0.0205598964,0.0009044658), conditional row
SE=(0.0003832200,0.0003184539). Two consecutive small steps establish
iteration stability, but paired8k/16k changes remain(-0.0001270075,
-0.0000891862), SE=(0.0000858462,0.0000830116), failing draw stability.
Preserved all results and the frozen mock runner. No0.3% acceptance;
actual-case overlap, single positive shear, faint carry and conditional
uncertainty limitations are explicit. Updated doc/V39_BRIGHT20_BFGS.md,
doc/V39_ACTUAL20K_BFGS.md and the inference guide.

## 2026-09-27 — Bright20 BFGS fourth update is small; fifth iteration running

Owner requested status at00:24 CEST. Read all saved iteration receipts,
Slurm accounting and all four worker logs for
inference_v39_mock20k_bright20_bfgs_20260926_v1. Four of five requested bright
updates completed; workers16721029[0–3] remain healthy. Fifth iteration has
12/20 normalization shards complete. Recent fresh iterations take24minutes;
estimated remaining time10–15minutes, not a completion promise.

g1 sequence:0.02076256 ->0.02047071 ->0.02056567 ->0.02055958. Latest
conditional SE0.00037764; g2=0.00090159 ±0.00041802. The latest step
(-6.09e-6,-1.17e-5) satisfies the small-step test once, with two consecutive
passes required. Current relative g1 offset is+2.798% ±1.888pp. Paired
8k-to16k change remains-0.644% ±0.424pp and fails numerical draw stability.
No0.3% acceptance or bright-carry validation; no scientific settings or code
changed. Updated [intermediate results](V39_BRIGHT20_BFGS.md#progress-at-0024-cest-on-september-27)
and refreshed external status. Existing audit16721087 remains scheduled.

## 2026-09-26 — Five measured-mag20 BFGS updates from the first full pass

Owner requested five bright-only iterations at measured magnitude<20, then
explicitly corrected the start to the original first pass rather than the
later BFGS proposal. New continuation starts from Gaussian g=(0.02508452413,
0.00267601819), uses the original pass00 observed-information solver state,
and carries faint numerator moments from its fixed raw-mean evaluation
point(0.01374048162,-0.00264758006). It does not initialize from full-pass01's
BFGS proposal. All20000 observations and the original measurement cuts remain;
only142 measured-bright moments are recomputed. Four-row numerical groups
require560 evaluations with418 padding rows discarded from the replacement.

First bright iteration reuses exactly matching pass01 bright evaluations and
normalization, while every other row is carried from pass00. This avoids a
redundant GPU pass and yields(0.02076255641,0.00075839956). It counts as the
first of five requested updates. No later full-pass faint moments enter the
hybrid. The remaining four iterations refresh full-prior64-QMC normalization
at every centre; carry replaces normalization derivatives on every row.
Keep the original Gaussian-prefilter sampler, K1024/M16384, h0.001, seed8701
and model/prior identities so this isolates the BFGS/brightness change.

Added scripts/run_v39_bright_bfgs.py, scripts/v39_bright_math.py,
jobs/job_v39_bright_bfgs.sh and tests/test_v39_bright_math.py. Existing
historical BFGS algebra is reused, with strict secant/solver checks. New
analytic tests cover nonquadratic normalization and cross-component carry;
CPU init16721013 ran20 focused tests, all passed in15.30s, then verified
source hashes, coverage and matching saved centre and combined the cached
first iteration. Ruff, Bash syntax and whitespace checks pass. Resource
snapshot saved externally; heavy work remains in Slurm.

External root: inference_v39_mock20k_bright20_bfgs_20260926_v1. GPU replay
16721017 must exactly reproduce all10 retained arrays for560 original-group
rows before four persistent workers16721029[0–3] may start. Dependencies
and array throttle preserve4 GPUs total. Five updates are requested; no
early success declaration, truth-based stopping or0.3% acceptance. Carry at
threshold20 and finite-draw/normalizer errors remain unvalidated. Full
[protocol and limits](V39_BRIGHT20_BFGS.md#owner-instruction-and-starting-point).

The earlier ranking probe16720780 completed in9m35s. Both centres exactly
replayed the228 original rows; whole-prior ranking retained identical global
draw IDs and q agreed within5.43e-19. Median top-K overlap was0.885 on128
random rows and0.757 on100 stress rows. It did not consistently reduce weight
relative error or8k-to16k score sensitivity, so it is not substituted into
this continuation. This probe is not a population-bias result.
Replay16721017 started on th-cl-nv01 and emitted4/560 original-group rows.
Scheduled saved-status audit16721087 replaces cancelled old audit16720572;
it runs after two hours but does not wake the agent or notify the owner.
Replay16721017 subsequently passed all ten arrays exactly on560 rows in5m16s;
four workers16721029[0–3] are released from their dependency and awaiting
scheduler allocation.
Production startup followed through: all four V100 workers16721029[0–3]
started on th-cl-nv01 and emitted their first full-shard normalization masses
for bright iteration2, matching the earlier zero-shear masses. No failures.
Yield while the long normalization/inference chain proceeds; audit16721087
is scheduled, and further results must be read from the new root.

## 2026-09-26 — Correct historical recovery; stop unstable Newton chain and audit saved scores

Owner challenged Gaussian g1=0.02508452 ->0.01102622. Reviewed archived
cont.205,342,345,346,369–370 and read the later complete-likelihood100k
chain.json. The earlier recovery omitted material features: cont.370 used
whole-prior Gaussian top-K, normalization-aware measured-m<22 carry and BFGS
score-root updates; cont.205 used a likelihood-increase line search. Current
plain Newton with a131072-location prefilter reproduces neither procedure.
The historical100k final g1 is0.0199264172 ±0.0004137699 for injection0.02;
its models/cuts differ and it is not a0.3% certification.

Cancelled owned workers16715922[0–3] after2h24m52s rather than continue blind
Newton passes; preserved both complete passes and partial third normalization,
recording external intervention.json without changing result receipts. Added
scripts/audit_v39_recenter.py for hash-checked saved-moment decomposition,
normalization-aware carry and reuse of the historical BFGS implementation.
CPU job16720722 completed in24s. It exactly reproduces both saved estimates,
verifies alignment and final-ladder equality, and checks the all-row quadratic
carry invariant. Gaussian secant curvature is2.112 times its initial local
curvature;967 measured-bright rows explain91.92% of absolute projected score
departure from linear carry and86.62% of initial directional information.
The saved normalization term shifts the second estimate by only4.53e-5.

BFGS applied to the same full Gaussian scores proposes g1=0.01916557; using
normalization-aware bright/faint carry proposes0.01918260. These are proposed
centres, not a converged new chain or demonstrated accuracy. The paired
observed-Newton carry/full difference is0.00015457 ±0.00018492, insufficient
to accept the shortcut at0.3%. Distance-only full-score BFGS proposes0.01854994.

Added scripts/probe_v39_historical_ranking.py; one-V100 job16720780 compares
whole-prior top-K with original prefilter on128 random rows plus explicitly
post-hoc stress groups at both saved centres, requiring exact saved replay
of the original four-row groups. No model, prior, density batch or proposal
seed changes. GPU probe is pending validation, not a production-chain restart.
Ruff passes for both new scripts. Updated
[audit and limitations](V39_RECENTER_20K.md#historical-procedure-correction-and-saved-moment-audit)
and doc/INFERENCE.md with the missing historical procedure. Next: read paired
ranking diagnostics, then validate a safeguarded/BFGS bright-carry recovery
separately from sampler convergence; no truth-directed offsets or acceptance.
Probe startup verified on th-cl-nv01,228 total rows: original Gaussian replay
emitted4/228 progress. Existing one-shot audit16720572 will save updated job
and intervention status; it does not wake the agent or notify the owner.

## 2026-09-26 — Original-prior 20k first pass completed; recentering in progress

Owner requested status at 22:10 CEST. Read scheduler accounting, GPU replay
receipt, full pass_00/result.json and all four worker logs for the original-prior
recenter study. Initial normalization and assembly completed; preflight16715921
passed exact replay for both arms on136 rows with identical global draws and
20.41 GB peak allocated memory. Workers16715922[0–3] are running without
failure receipts. First pass completed20k; second pass completed12/20 paired
checkpoints and its fresh full-prior normalization.

First-pass Gaussian16k g1=0.0250845241 ±0.0016664645 conditional SE;
distance16k g1=0.0235238099 ±0.0014268882, for injection0.02. Relative offsets
are +25.42% ±8.33 pp and +17.62% ±7.13 pp. These are intermediate
one-step estimates; recentering has not yet established closure. Both arms
fail the declared paired8k-to16k stability screen at this centre. Updated
[progress and limitations](V39_RECENTER_20K.md#progress-at-2210-cest-on-september-26).
No scientific settings or running code changed; validation is receipt/log
inspection and documentation whitespace checking. Await subsequent complete
passes; do not compare this fresh restored-prior mock as paired with the old500k.

## 2026-09-26 — Launch20k recovery of the iterated pipeline with pre-extension prior

Owner authorized recovery of the successful historical inference setup on20k
mocks, then requested removing extra bright priors. Cancelled expanded-prior
preflight16715903 before production. Asked whether to remove only the new
7,780,811 bright atoms or also the earlier231,221 certainty-stratum additions;
proceeded after the optional clarification interval with removal of the new
extension. Restore the previous20,231,221-atom prior and its original weights,
not renormalized expanded-prior weights. Generate a fresh20k matched-prior
mock, seeds202609281/282/283. Models, physical likelihood, measured0.6arcsec/
25.8 cuts and absence of truth-property cuts are unchanged.

Recovered historical fixedpoint_chain2 receipt binds K1024,131072-candidate
Gaussian reranking, tilted-stratified delta0.1/temperature1, seed8701, h0.001,
full2D steps, float64 proxy scores, retained512–8192 ladder and successive
estimate centres. New study adds16384 as the predeclared primary and a paired
distance-only arm. Both arms use the Gaussian16k chain's centre. Up to4 full
passes; fresh full-prior64-QMC normalization and state-dependent probabilities
at every centre. Exact current density groups of4 replace historical128 to
respect the prior numerical replay finding. No hybrid linear carry is used.

Added scripts/run_v39_recenter.py, scripts/v39_recenter_math.py,
scripts/prepare_v39_reduced_prior.py, two jobs/job_v39_recenter*.sh launchers,
and tests/test_v39_recenter_math.py. Generalized the study driver to an explicit
prior shard count (default28 unchanged) and checked reuse of unsheared reduced
prior coordinates. Runtime/library/model files are reused by pinned hash;
new drivers/protocols are frozen externally. Existing study artifacts remain
immutable. New [protocol and limitations](V39_RECENTER_20K.md#paired-design-and-stopping).

Validation: Ruff, Python syntax and shell syntax checks pass. Full CPU tests
16715901 and16715915 both completed308 passed/1 skipped. New tests cover joint
quadratic recovery from a distant centre, paired covariance cancellation, and
rejection of indefinite information/unsafe steps. GPU preflight is required
before production: exact all-array replay and candidate/draw equality on136
rows, shared global draws across arms, and15% memory headroom.

Submitted dependency chain: restoration16715917[0–3] ->mock16715918
->initial normalization16715919[0–3] ->assembly16715920 ->preflight16715921
->four persistent inference workers16715922[0–3]. No more than4 GPUs at once.
Prior-restoration workers started on th-cl-1024us01; dependent stages pending.
No numerical inference result yet. Monitor first progress and stage transitions;
report iteration, sampling and normalization limits separately, without a0.3%
acceptance claim.

Startup follow-through: all restoration tasks completed successfully in122–125s;
20,231,221 rows and mass1.0000000000000029. Mock16715918 completed all20000
observations (67.8s driver time), with no invalid retained draws and2066
selected true-r>=26 objects. Raw observed mean centre is
(0.013740481623385736,-0.002647580061745958). Initial normalization16715919[0–3]
started on four V100s at th-cl-nv01; all four emitted their first full-shard
node masses without error. Saved-status audit16715934 is scheduled after two
hours; it writes an external snapshot, not a Codex wakeup or notification.

## 2026-09-26 — Recover historical mock closure and correct one-step interpretation

Owner recalled earlier unbiased mock results. Reviewed archived WORKLOG_FULL
cont.174–175,204,259,338,341–342,346,354 and the cross-worktree inference
record. The relevant positive evidence is finite-shear profile recovery and
Newton recentering: the older25000-object full chain changed g1 from0.022960
to0.01906560 then0.01958117 (truth0.02; final conditional SE0.00061195).
Read both saved chain result JSONs and independently checked the saved
five-pass20000-object hybrid chain, which settles at0.01953533 ±0.00071477.
The old8million-object uncut V1 shape-flow result was+0.034% ±0.252%; selected
claims in that entry were subsequently corrected for normalizer uncertainty.
The million-object cont.259 result was explicitly unconverged, and the
complete-likelihood cont.354 closure had only200 objects with69.53pp error.

Current500k V3.9 code and result receipt confirm a single Newton update from
(0.01303596,-0.00261937), not a recentered likelihood fit. The previous status
answer overemphasized atom draws without recovering this established
linearization failure. Draw sensitivity remains real, but it does not isolate
the cause of the endpoint error. Older profiles/chains omitted current
measured selection and disk response; their successful recovery cannot be
transferred as a V3.9 acceptance result. No historical finite-catalogue error
is added to current uncertainties.

Added [historical recovery and current diagnostic implications](INFERENCE.md#historical-mock-recovery-and-the-v39-one-step-limitation).
Next diagnostic should separate recentering, draw-budget, and normalization
effects on fixed existing mock rows, with fresh normalization stencils at new
centres, common random numbers, and stopping independent of injected truth.
Validation: stdlib JSON receipt reads, source inspection, and documentation
whitespace check; no scientific computation, new jobs, or archived restarts.

## 2026-09-26 — Complete500k model-mock inference; draw ladder remains unconverged

All500 partitions/500000 measured-selected observations are complete. Persistent
array16706357[0–3] completed successfully:262000 remaining observations in
6h39m49s wall time (September26 11:42:26–18:22:15 Europe/Berlin). Workers handled
66000/65000/66000/65000 rows; median1000-row checkpoint360.07s. Combiner16706358
completed successfully in52s at18:23:36, verifying original and amended execution
receipts, output hashes, row coverage, and scientific identities. Immediate
completion audit confirms exactly262 distinct optimized partitions and500 total
receipts, unchanged scientific identity and the pinned runtime amendment.
No study jobs remain queued/running. External jobs/status receipts updated.

At the fixed primary M16384: g1=0.020891396494 ±0.000437371819 and
 g2=0.001622446128 ±0.000220339621 (reported robust1SE). Injected shear(0.02,0).
Relative g1 error100*(g1/0.02−1)=+4.456982% ±2.186859 percentage points.
This is fixed-shear model-closure error, not an independently calibrated
multiplicative-slope measurement; see CONVENTIONS.md §7. Measured radius>0.6arcsec
and magnitude<25.8, no truth-property cuts; V3.9-like models and expanded weighted
prior, original fixed seeds, one Newton update and64-draw QMC normalization.

Full draw ladder (relative g1 error% ±1SE pp):
512:−35.380935±0.796048;1024:−31.042642±0.833733;
2048:−25.626496±0.922614;4096:−19.323900±1.089082;
8192:−9.035359±1.476396;16384:+4.456982±2.186859.
All information sums are positive definite, but the largest two rungs differ
by13.492342pp. g2 also remains visibly nonzero. Completion is computational
success, not scientific closure or a robust0.3% result. More observations did
not establish convergence at the retained draw budget. Diagnose finite-draw
score/curvature and the one-step/normalizer approximations before interpreting
another larger observation sample. No new experiment was launched on this
status request. Updated doc/V39_MODEL.md with the completed result and limits.

Source: external500k v2 `result.json`, persistent worker receipts,
`logs/v39_persistent_16706358_4294967294.out`, and sacct for16706357/16706358.
Validation here is receipt/accounting consistency; the already-successful
strict combiner was not rerun and no scientific outputs were modified.

## 2026-09-26 — Resume262k unfinished rows on four persistent V100 workers

Accepted validation16706278:128-row atom search,24-row tilted proposal blocks,
and original4-row likelihood groups. All10 arrays on1032 held-out/stress
observations replay **exactly** (zero maximum difference), as do all candidate
indices, global draw indices and probabilities. Peak allocated26,017,293,824
bytes=24.23GiB, reserved24.55GiB on31.73GiB V100s. End-to-end benchmark including
proposal preparation and uncached response evaluation:2.951926 observations/s.
Four such workers project approximately6.2h for262k remaining rows; use6–7h as
an initial conditional estimate, not a completion guarantee.

Production array16706357[0–3]%4 uses four32GB V100 allocations on inter, one GPU
per persistent job, a shared queue, and atomic1000-row checkpoints. Claims and
checkpoint publication are atomic; same-job Slurm requeue reclaims unpublished
work. Array starts after accepted replay16706278. Full tests16706292 completed
305 passed/1 skipped; Slurm had already aged that job out of dependency tracking,
so its successful accounting/log receipt was verified before submission.
Combiner16706358 waits afterok for all four workers, checks the pinned execution
amendment/validation on every new checkpoint, then invokes the original strict
500-partition coverage/hash/scientific-identity combiner.

Original238 completed partitions, models, mock, prior, all seeds, K1024/M16384,
selection normalizer and all six stencil-prefix moment ladders remain intact.
Original driver/protocol/runtime hashes were rechecked unchanged. No larger
density batch was accepted. External `batch_optimization/amendment.json` SHA
`a6c5b89be71d5377dc0b71defc27147ea54fcd22bfb03514a5a3a10e2a2e6df9`
pins the additional producer and exact-replay receipt. `jobs.json` now points to
the persistent array/combiner and records the original cancelled array separately.
Saved-status job16705939 is rescheduled for two hours after submission; it writes
a scheduler/data snapshot, not a Codex wakeup. Startup verified: all four jobs
are running and independently processing partitions238–241. Each reached176–180
observations without errors; warm density throughput is2.8–2.9 observations/s
per worker. Allowing proposal setup, approximately7h remains from this check.
The scientific run continues; final500k bias/convergence result is still pending.
Updated `doc/V39_MODEL.md` documents the accepted execution and rejected density
batch changes. Long-run polling stops after this successful startup check.

## 2026-09-26 — Proposal-only batching replays exactly; retain four-row densities

Job16706245 tested128-observation candidate searches and32-observation tilted
proposal blocks, with the original4-observation density groups. All10 stored
moment/diagnostic arrays on1032 observations replayed **exactly**, including both
boundary groups; all candidates, draw indices and probabilities also matched.
Observed throughput2.9028 observations/s includes proposal preparation and
uncached response evaluation. GPU peak allocated32,062,547,456 bytes (~29.86GiB)
left insufficient15% headroom, so that deployment was rejected despite exact
numerical replay. Preserved source snapshots and `proposal_validation_16706245`
receipts/arrays remain under the500k root.

The final candidate keeps128-observation search blocks but reduces tilted-CDF
blocks to24. Active helper/validator sources are frozen externally with a24
suffix to preserve the already-running32-block audit. Validation16706278 is in
progress. Current worker requires the exact replay receipt and15% allocated
memory headroom, preserves original1000-row checkpoints, and can reclaim its
own unpublished checkpoint after Slurm requeue. Other workers never steal a
live claim. Four V100 workers will share the262 unfinished partitions238–499.
AGENTS.md records the owner's four-GPU limit for this continuation and the
required four-row density grouping. Latest full-suite16706292:305 passed,
1 skipped; Ruff and shell syntax checks pass. Production switch awaits the
24-block validation receipt; no larger density batch is accepted.

## 2026-09-26 — Reject larger density batches; validate separate proposal batching

Controlled V100 replay16706111 reproduced the original four-row groups exactly,
but density batches8–192 changed float32 likelihood finite differences even
though candidates and full global draws/probabilities replayed exactly. Initial
comparison16706064 mixed V100 with A40 checkpoints;16706065 also regrouped the
extreme rows. These were confounded and are retained as failed diagnostics.
The controlled check uses original V100 checkpoints and intact four-row groups.

Independent1024 ordinary observations234000–235023 plus the8 original-group
rows around86174/112155 were checked in jobs16706174(batch32),16706207(128),
16706206(160),16706212(192). Before those runs, declared a maximum projected
shear shift including2SE of0.003 percentage points (1% of the0.3% bias target).
Using the238k completed information matrix to map paired moment differences,
all four batches gave an estimated g1-relative shift approximately+0.097959pp
with paired SE0.068034pp (upper bound0.234027pp); g2 normalized to0.02 also
failed. This is a finite-sample numerical sensitivity estimate, not a measured
change in the full500k answer. None is accepted. Exact rows, resulting arrays,
hashes and statistics are saved in `batch_optimization/validation_*.json/.npz`.

GPU distance matrices alone require26.71GiB at256 and53.43GiB at512, so
16706180/16706181 exceeded V10032GB memory;1000 probe16706196 was cancelled
once that limit was established. Warm128-row speed comparisons benefited from
reused response-cache entries and must not be advertised as production speedups.
The1032-row validations also warm their first chunk; actual new-checkpoint timing
is required before updating the production ETA.

New `v39_batched_proposal.py` precomputes candidate sets/draws in128-row search
blocks, preserves original global RNG IDs, then serves those exact arrays to
the unchanged four-row likelihood groups. Frees unused search/CDF allocations
between stages. `validate_v39_batched_proposal.py` checks all saved moments and
diagnostics against original V100 checkpoints and independently checks all
candidate/draw identities and probabilities. Job16706245 is in progress; four
persistent production jobs will start only after this stricter replay passes.
The persistent worker retains1000-row atomic checkpoints and a shared queue.
Ruff/shell checks pass; latest full-suite16706226:305 passed,1 skipped.

## 2026-09-26 — Benchmark larger batches and prepare four persistent500k workers

Owner authorized larger observation batches and longer jobs, then explicitly
requested four jobs, one GPU each. New scripts `benchmark_v39_batches.py` and
`run_v39_persistent.py`, plus `jobs/job_v39_persistent.sh`, reuse the immutable
500k v2 driver/runtime and scientific protocol. Persistent workers load once,
claim unfinished1000-row partitions atomically, and publish complete checkpoints
without overwriting completed work. Shared work allocation avoids assigning an
equal observation count to unequal GPUs. Execution amendments and benchmark
hashes accompany every new checkpoint; an amendment-aware combiner also invokes
the original strict coverage/hash/scientific-identity checks.

Benchmarks16706016(inter) and16706022(cip24GB) await original tasks230/234/235.
Both compare128 fixed observations including boundary rows86174/112155 against
original checkpoints and batch4, with exact candidate/draw/probability replay
and all moment/diagnostic arrays (rtol1e-9,atol1e-8). Larger batches must leave
15% GPU memory headroom; warm throughput determines the retained batch among
passing configurations. Resource receipts are recorded on the allocated GPUs.
Validation and production switch remain pending; no claimed speedup yet.

Original queued partitions238–499 and obsolete combiner16702089 were cancelled;
running original tasks continue. No completed partition was removed. Saved
`batch_optimization/migration_snapshot.json` records the262 pending partitions
and four original running tasks at migration. The old array throttle was reduced
to2 while making room for two benchmark GPUs; production will use exactly four
persistent workers. Ruff and shell syntax checks pass. Follow this entry with
measured results and replacement job IDs after replay checks finish.

## 2026-09-26 — Compare current500k runtime with historical full-A40 run

Owner recalled approximately12h on four GPUs. Confirmed historical array
16606146 via Slurm accounting:500000 observations,20 partitions of25000,
first start September20 14:36:22 and last finish September21 01:17:39,
**10h41m17s** elapsed. Four-GPU ceiling, full A40 allocations, object_chunk16,
K1024/M16384 and the same six prefix rungs. Historical final curvature check
failed, but all500k numerical evaluations completed; its runtime is valid
for execution-cost comparison. Source: production_lru_v1/part_00/result.json
and cross-worktree WORKLOG September20 LRU/production entry.

Current16702088 uses500 partitions of1000, object_chunk4, three inter V100s
and three cip virtual A40 workers. Historical prior24m versus current28.012m
adds17% atoms to proposal scans. Models, proposal ranking and precision path
also differ; this is not a controlled kernel benchmark. Completed current
inter tasks median9.53min/1000; cip28.6min/1000. Inter median load138.9s plus
first-four/compile35.2s: about3min of each9.5min task is startup. Historical
25000-row tasks amortized this cost; the current run unnecessarily repeats
it500 times. Smaller object batches are another plausible throughput cost,
but their isolated effect requires a matched replay benchmark. The earlier
24h estimate accurately extrapolated100k setup, not an intrinsic cost of500k.

Snapshot:228/500 partitions complete after9.48h, no failures; all audit gates
passed, including original boundary rows86174/112155. Current cumulative
throughput projects20.8h inference total and11.3h remaining, conditional on
unchanged capacity. Saved runtime_comparison_20260926.json under the500k v2
root. No production jobs, numerical settings or results changed. A future
speed test should compare larger batches on sufficient-memory GPUs and
persistent workers with exact replay before changing the active implementation.

## 2026-09-26 — Preserve disk radial precision and resume fixed-seed500k

Owner authorized continuing the failed500k run. Added `sbsi/disk_precision.py`:
record log(1−|e|²) analytically from unbounded flow coordinates before Cartesian
rounding; propagate it through Möbius transport and use it in inverse density.
This is deterministic precision metadata for the same shape, not a new random
measurement or selection cut. Origin uses cancellation-free series; transport
uses Cartesian log1p only well inside the disk and the analytic gap identity
near the edge. Metadata must agree with Cartesian values to floating-point
tolerance. No clipping, sample replacement, RNG search or parameter changes.

Physical-flow/bundle sampling optionally returns the gap with exactly unchanged
physical draws and random-state consumption. DiskCatalogueLikelihood accepts
the explicit metadata column, uses the same physical-density Jacobian and
original network precision, and retains the old path for ordinary catalogues
without it. New compiled density applies only when this metadata is supplied.
Study driver now records gap metadata, validates encoded shapes, checks
forward/inverse transport and replays the saved failed second batch exactly.
Added `scripts/audit_disk_precision_mock.py` to exercise true-atom conditional
density and the full nine-node M16384 sampler on every rounded-boundary row
plus four ordinary rows before the full inference array can run.

Tests cover120-digit Decimal transport/radial identities from radius0 to80,
forward-Jacobian density at rounded boundaries, old density in the interior,
identical pseudorandom/QMC draws and RNG state, and inconsistent-metadata
rejection. Full-suite16702066:305 passed,1 skipped in30.42s. GPU preflight
16702078 passed old/new and eager/compiled density comparisons in addition to
the existing two-shear checks. Ruff passes. All old experiment snapshots and
the failed catalogue are preserved. Separate external root
`inference_v39_mock500k_distance_20260926_v2` pins the precision amendment,
changed module hashes and unchanged model IDs/seeds. Mock16702083 is submitted;
remaining preparation/inference awaits its validity/replay check.

Mock16702083 completed successfully:500000 observations from2097152 candidate
atoms, same original500k seeds, no replacements. Entire failed second batch
replays identical atoms and physical draws. Both originally invalid retained
Cartesian shapes pass the gap encoding; all encoded observations valid.
53808 retained targets have true r>=26; forward/inverse shape error4.44e-16.
New centre(0.013035957659442396,−0.002619371804842549).

Queued chain: prep16702084[0–27]%6 → assemble16702085 → boundary sampler
audit16702086 → smoke16702087 → inference16702088[0–499]%6 → combine16702089.
All use afterok gates; no GPU stages overlap. Three cip and three inter
preparation workers started. Scheduled saved-status audit16702094 for+2h;
it writes a receipt, not a task notification. Precision audit includes rows
rounded outside either before or after transport, runs original object IDs,
and must produce finite full-stencil moments before releasing inference.
See [precision amendment](V39_MODEL.md#disk-boundary-precision-amendment-for500k).
Verified all six initial workers produced finite positive normalization masses;
four reached the new-centre node. Normal preparation progress established.
Final Ruff/whitespace checks pass. Full sampler audit and500k results pending.

## 2026-09-26 — 500k mock stopped by retained disk-boundary rounding

Preflight16702041 passed unchanged density and selection checks. Mock16702042
failed in1m44s after its first68217 accepted observations: the second batch
contains two measured-selected draws whose squared shape norms round to
1.0000000000000002. Both have finite shapes and positive radius/flux.
Global zero-based selected ordinals86174 and112155 correspond to prior atoms
23945110 and3185269. These are inside the requested500k prefix, unlike the
unused boundary draws in the completed20k/100k runs. No discarded/replaced
objects, clipping or seed changes; no downstream GPU jobs submitted.

The generation guard correctly rejects the non-invertible rounded values.
The output is not a completed500k catalogue and no500k shear result exists.
Saved second-batch draws and audit under the new root's `mock/`, with
`failure_diagnosis.json` recording exact indices, values and validation.
Updated external jobs.json status and the V3.9 study doc. Diagnosis was a
small saved-array inspection; all expensive generation ran on Slurm.
The mathematical radial-tanh map has open-disk support, but float64 output
rounding loses the tiny distance to the boundary; current density rejects
those stored values. Before continuing, generation/likelihood needs a
consistent representation retaining that information, tested for density
and transport agreement. Dropping these observations would alter selection;
changing seeds until500k passes is not acceptable. The unchanged inference
runtime estimate remains about24h once this prerequisite is resolved.

## 2026-09-26 — Start authorized500k distance-only follow-up

Owner requested500k and asked whether it should take approximately10hours.
Measured100k inference elapsed4h41m22s, plus34m56s from mock start to inference
start.79 inter tasks had median9.5min per1000;21 cip tasks median39.6min.
Total GPU time was26.01h. Same-mix scaling predicts23.45h inference plus setup,
so quote about24h subject to contention, not10h. Six workers at inter speed
would need about13.2h inference with unchanged1000-row partitions.

Create separate external root `inference_v39_mock500k_distance_20260926_v1`.
Driver/runtime are byte-identical to the completed100k run. New protocol pins
500000 observations, fresh scene/detection/flow seeds202609271/272/273,
same F8/C8/emulator, expanded weighted prior, measured0.6arcsec/25.8 cuts,
no truth cuts, distance-only K1024/M16384 and every prefix rung. New mean-shape
centre requires new normalization; immutable features and proposal moments
can be reused by verified provenance. This is not a nested sample-size test.
Retained invalid numerical shapes still abort; no seed search or replacement.

Submitted preflight16702041 then mock16702042 to inter V100s. Full inference
will use500 disjoint1000-row partitions, shared six-GPU cap across inter/cip,
and strict complete-population combination. AGENTS.md records the500k scope.
No scientific implementation or estimator changes; completed100k tests and
validation remain applicable, with a new identity/model preflight. Further
submission depends on successful generation; no500k result yet.

## 2026-09-26 — Verify completed100k distance-only study

Immediate owner status check: all28 preparation tasks, assembly16699856,
smoke16699857, all100 inference tasks16699858 and combine16699859 completed
successfully. No pending/running user jobs. The2h limits avoided a repeat
of the20k timeouts; slowest inference task completed in1h04m31s. Strict combine
verified all100 disjoint1000-row partitions and pinned artifact identities.

Primary M16384 result: g1=0.019502411533783973 ±0.0007965901983862846,
relative realization error−2.487942% ±3.982951pp (one robust SE);
g2=0.000339781553263256 ±0.00030211698731576545. At8192 draws the relative
error is−14.030730%; last doubling shifts it by11.542788pp. Unconverged,
not0.3% acceptance. Full ladder and interpretation recorded in
[V3.9 completed100k result](V39_MODEL.md#completed100k-distance-only-result).
This is a fresh realization with a different mean-shape centre, not a paired
20k-versus100k sample-size comparison. No models, numerics or defaults changed;
no additional GPU work launched. Source `result.json` retains full limitations.
Validation: scheduler completion/exit codes,100 completion manifests and strict
combine success; documentation whitespace check. Next diagnostic would compare
exact scores/curvatures and inspect high-influence observations on a fixed subset.

## 2026-09-25 — Launch fresh100k distance-only V3.9 model mock

Owner requested100k using the distance-only arm. New external root
`inference_v39_mock100k_distance_20260925_v1` pins a fresh realization at
injected g=(.02,0), with predeclared scene/detection/flow seeds
202609261/202609262/202609263. Same expanded weighted28,012,032-atom prior,
F8/C8/disk models, measured radius>0.6arcsec and magnitude<25.8, no truth cuts.
This is a new realization, not a nested reuse of the inspected20k sample.
Any numerically invalid retained shape aborts; no dropping, clipping or
seed search. Mean observed shape sets the new centre; normalizer is rebuilt.

Generalized `scripts/run_v39_sampling_study.py` and its launcher to accept
an external study root, protocol-specified arms and observation counts.
Both-arm defaults remain available; prior20k runtime/script snapshots are
untouched. New protocol explicitly pins old feature provenance and coordinate
execution hashes. Reuse verified features and zero-shear proposal moments;
recompute classifier/selection nodes and assembled normalization for the new
centre. Retain K1024/M16384, all512–16384 draw prefixes, full nine-node stencil,
proposal seed8701 and one Newton step.100 disjoint1000-row distance-only tasks
will combine with hash, identity and exact row-coverage checks. Same-draw
invariance is still checked on two probe observations in each task.

`AGENTS.md` records that six GPUs across cip/inter covers this follow-up.
Ruff, Python syntax and Bash syntax pass. Full-suite CPU job16699846 and GPU
preflight16699847 submitted; mock16699848 depends on both. Further stages
will preserve a combined six-GPU cap and longer job limits. Results pending;
more observations do not fix the20k finite-draw convergence failure by itself.

Full-suite16699846 passed294 tests,3 skipped (artifact environment not set),
in42s elapsed; preflight16699847 passed
both tested shears and selection boundaries. Dependency chain:
mock16699848 → prep16699855[0–27]%6 → assemble16699856 → smoke16699857 →
distance-only inference16699858[0–99]%6 → combine16699859. GPU preparation,
smoke and inference use cip/inter with low-memory GPUs excluded and2h limits;
all stages require predecessor success, so total concurrency stays at six.
One-shot saved-status audit16699868 scheduled for +2hours (not a task wakeup).

Mock16699848 completed in1m45s:100000 retained observations,524288 prior
candidates, centre(0.012079337150125146,−0.00029905259816080903).
All retained draws are finite, positive in auxiliary outputs and strictly
inside the disk; no replacement or truth cuts.10827 retained targets have
true r>=26. One unused draw after the100k stopping point rounded to the
boundary; it is recorded in the generation audit. Forward/inverse maximum
shape difference4.44e-16. Preparation16699855 started six tasks concurrently:
three cip A40 workers and three inter V100 workers; all passed coordinate
reuse hash checks and started new normalization. Inference remains gated.
Confirmed every initial worker completed its zero-shear normalization node,
with finite positive mass, and four completed the new-centre node too.
Normal preparation progress established; yield under the long-run policy.

## 2026-09-25 — Complete V3.9 20k comparison; do not adopt distance-only

Both recovery tasks16699600_17/_19 completed in16m52s/16m19s; strict
combine16697082 passed in27s, with20 complete partitions and20000 observations.
At fixed16384 random draws, Gaussian gives g1=0.02215861 ±0.00097257;
distance-only gives0.01864592 ±0.00080721 (robust one-SE). Against injected
0.02 these are +10.79% ±4.86pp and −6.77% ±4.04pp. Paired distance-minus-Gaussian
is −17.56 ±4.41pp. These are realization errors, not established calibration
biases. Distance-only moves from−16.24% at8192 draws to−6.77% at16384, failing
convergence. Gaussian's similar final g1 points do not prove numerical closure.
Neither supports0.3% accuracy or adoption of the simpler exact-set ranking.

Recorded full draw ladder and interpretation in
[V3.9 completed comparison](V39_MODEL.md#completed-20k-comparison).
No scientific code, defaults or model artifacts changed. Result.json preserves
all20-partition hash/identity checks and limitations (one mock/seed, one Newton
step,64-QMC normalizer). Next work would diagnose scores/curvatures using
exact sums on a fixed subset before another large run. Complete result saved;
no further GPU jobs submitted. Cancel obsolete delayed status audit16699607
and save its read-only snapshot immediately.

## 2026-09-25 — Recover two timed-out V3.9 inference partitions

Owner requested immediate status. Eighteen of20 partitions completed both
arms (18000 observations); cip tasks16697081_17 and_19 timed out at one hour
in the distance arm, last logged820/1000 and848/1000 rows. The afterok gate
correctly blocked combine16697082. Inter tasks completed in15–17minutes;
completed cip tasks required29–40minutes, with worse throughput in the last
wave. No combined result exists; partial-population estimates are not used.

Preserved incomplete partition directories under external
`timeouts_16697081/`. Submitted only17/19 as array16699600%2 on inter V100s,
32GiB host memory and a two-hour limit. Both arms rerun from the same pinned
code, observations and seeds; all18 completed partitions remain untouched.
Repointed combine16697082 to afterok:16699600 and verified scheduler acceptance.
External jobs.json records recovery; the read-only status script now includes
retry accounting, and audit16699607 is scheduled for +2hours. This audit
writes a receipt, not a task wakeup. Both retries started on th-cl-nv02
and reached INFERENCE_START after ~161seconds loading. The complete20k
result remains pending.

## 2026-09-25 — Enable cip and inter for V3.9 inference

Owner requested both partitions. Updated pending tasks16697081_[1-19] in
place to `Partition=cip,inter`, generic one-GPU allocation and32GiB host
memory, retaining the single array-wide six-worker cap. Excluded8GiB cip
GPU nodes and the older met visualization GPUs; eligible cip virtual nodes
have at least16GiB GPU memory. Smoke16697080 completed both arms and the
same-draw check with peak RSS10995672K (~10.5GiB), supporting32GiB host
allocation at unchanged object_chunk4. Task0 was already running on an
inter V100 and remains untouched. Assembly16697079 completed successfully.
No seeds, partitions of observations, models, numerical code or dependencies
changed. Hardware can differ between tasks; the paired arms run on the same
GPU within each task. Scheduler accepted and verified all pending updates.
External `jobs.json` records the resource amendment; `AGENTS.md` records the
owner's partition authorization. Three cip A40-16GiB tasks started alongside
the existing inter V100 task. All loaded their inputs; two cip workers have
completed their first four inference rows, and the inter worker has reached
164/1000 rows in its first arm. The remaining cip worker is still compiling
its first batch. No failures observed. Other tasks wait on availability/quota;
results and final validation remain pending. Existing status audit retained.

## 2026-09-25 — Increase V3.9 mock-study inference to six GPUs

Owner explicitly requested more GPUs for the current 20k model-generated
study. Updated live array16697081 with `scontrol update JobId=16697081
ArrayTaskThrottle=6` (previously2), retaining its `inter` V100 allocations
and afterok dependency on smoke16697080. The20 fixed observation partitions,
paired arms, models, seeds, protocol and pinned execution sources are unchanged.
Updated `AGENTS.md` with this scoped authorization and external `jobs.json`
with the scheduler amendment. No other SBSI GPU jobs were active or pending
outside this chain. All28 preparation tasks16697078 completed successfully;
CPU assembly16697079 is running, smoke and inference remain dependency-gated.
Scheduler accepted the throttle update; scientific results remain pending.
The existing one-shot status audit16697094 remains queued.

## 2026-09-25 — Start V3.9 model-generated 20k sampling comparison

Owner requested20k observations and confirmed model-generated V3.9 mocks.
New `sbsi/v39_inference.py` implements physical0.5/1/2arcsec crowding over all
nonself neighbours within10arcsec, and the shared F8/C8 shear-state cache.
`scripts/prepare_v39_study_features.py` rebuilds features for the28,012,032-atom
expanded prior, keeping every supplied bright row, original weights and donor
environments. Emulator pairs remain the separately pinned nearest20 geometry.
No truth cuts. `scripts/run_v39_sampling_study.py` generates20k retained mock
measurements at g=(.02,0), measured radius>0.6arcsec and magnitude<25.8, then
prepares new V3.9 proposal/classifier/selection caches and compares Gaussian
and distance-only exact sets with identical random proposals/draws. Both arms
use K1024/M16384, full nine-node stencil, common normalizer and mean-observed
initial centre. Retain all prefix rungs; primary comparison uses fixed M16384,
not a method-dependent adaptive stopping count. No production defaults change.

External root: `inference_v39_mock20k_distance_20260925_v1`. Protocol and
execution manifests pin sources, models, scripts and isolated runtimes.
Features use the preserved original runtime plus the new module; inference
uses a separate copy with the corrected required base-cache geometry arguments.
This constructor issue was caught by first full-suite job16696807; replacement
16696823 passes296 tests,1 skip. Added tests cover all-neighbour support,
retention of faint secondary neighbours, self removal, scene isolation, shear
invariance of spin0 features and wrong-model rejection. Ruff/compilation/Bash
checks pass. Feature arrays16696810 and16696822 run at two CPU workers each;
GPU stages will preserve the combined two-GPU limit. Numerical preflight and
subsequent mock/cache/inference stages pending; no results or0.3% claim yet.

All28 feature shards completed, total28,012,032 atoms and prior mass
1.0000000000000024. Initial density preflight16696887 correctly rejected one
test draw outside measured selection; corrected the preflight to test retained
draws.16696925 and final pinned-source16696960 pass at shears(.02,0) and
(-.015,.01), with127 selected rows each and maximum standardized log-density
differences3.43e-4 and1.57e-4 (within declared float32 tolerance). Strict cut
boundaries and unchanged spin0 features pass. Explicitly enable the fractional
location metric as well as fractional scatter flooring; they are distinct
runtime settings. Preserve earlier preflight script/execution snapshots.

Queued dependency chain: mock16696965 → preparation16696966[0–27]%2 →
assembly16696967 → paired8-row smoke16696968 → inference16696970[0–19]%2 →
combine16696972. Every downstream stage requires predecessor success; GPU
stages do not overlap and arrays permit at most two GPUs. The mock is running.
Study details are in [V3.9 model closure](V39_MODEL.md#20k-model-closure-sampling-study).

Mock16696965 stopped on an overly broad physical-validity assertion over the
entire generated batch; downstream stages were cancelled before execution.
Diagnostic16697019, then16697024, replayed the identical seeds: one of117897
detected draws rounds to the unit-disk boundary. It is selected object51871
out of68171 in the oversized batch, outside the requested first20000. No
nonfinite or nonpositive auxiliary draws occurred. Move validation to the
predeclared first20000 measured-selected observations and record all batch
rounding counts separately. Selection never uses the validity mask to replace
or redraw an object; any invalid retained draw still aborts. Seeds, population,
cuts and prefix remain unchanged. Partial support/mock files and the failed
execution snapshot are preserved under `failure_16696965`. Diagnostic16697012
was only a launcher failure from a node-local /tmp path; replacement used the
shared project path.

Replacement chain: preflight16697071 → mock16697074 → prepare16697078[0–27]%2
→ assemble16697079 → smoke16697080 → inference16697081[0–19]%2 → combine16697082.
This retains the two-GPU cap and all afterok gates. Results pending.

Replacement preflight16697071 passed. Mock16697074 completed in1m40s with
exactly20000 selected, strictly interior, finite positive observations, no
replacement draws and no truth cuts. Initial mean-shape centre is
(0.015691824930896287,0.0017382505470351646), injected shear(.02,0).
The unused batch boundary draw is recorded explicitly; every retained draw
passes forward/inverse transport checks. New V3.9 preparation16697078 is queued
next. Full inference estimates are not available yet.

Confirmed preparation startup on two V100s: shards0/1 completed proposal
coordinates and their first two selection-normalization nodes, with finite
positive masses (~0.01288/0.01291 per shard). The mock includes2127 selected
targets with true r>=26; forward/inverse maximum shape difference4.44e-16.
Fresh caches and both inference arms remain pending completion. Scheduled
one-shot Slurm status audit16697094 for about two hours later; it writes
`status_latest.json`. No task-wakeup/automation tool is exposed in this
session, so this is a saved scheduler audit, not a promised task notification.
Final Ruff and whitespace checks pass. Yielding after normal GPU progress,
per the long-run monitoring policy; no inference default or acceptance claim.

## 2026-09-25 — Compare simpler atom-selection scores

Owner requested a simpler, more principled sampling score. Added
`scripts/compare_simple_atom_scores.py` and its Slurm launcher. GPU16696646
compares the existing floored Gaussian with nearest-location distance, a
prior/usability-weighted shared-scale kernel, and the Gaussian using raw
flow scatter. All use the existing 131072 location prefilter and fixed
metric; compare K1024/M16384 and K8192/M8192 on the three existing exact-sum
observations, both priors, and all seeds8701–8716. Only deterministic
selection changes: full-support sampling probabilities and draws stay fixed.
Also calculate exact repeated-draw variance from the full finite-prior sum,
since sixteen seeds can miss rare large importance weights. No production
change or shear-bias claim. Ruff, Python compilation and Bash syntax pass.
Results pending under `inference_simple_scores_20260925_v1`.

Startup16696646 exposed a historical artifact-layout difference: baseline
draws are stored in the original atom-map NPZ, not the extension comparison
directory. Corrected the diagnostic reader and resubmitted as16696664 to
`inference_simple_scores_20260925_v2`; the failed source snapshot/log remain
preserved. No model evaluation or scientific settings changed.

GPU16696664 completed in1m46s, peak RSS11.57GiB. All six baseline exact sets,
seed8701 draw lists, full log sums, normalized probabilities and sixteen
reference estimates replay successfully; input/runtime hashes and frozen
metric checks pass. [Report and complete metrics](figures/inference_simple_scores_20260925/README.md#results)
published with source snapshots; linked from the maintained bright-prior doc.
Nearest-location at K1024 reduces expanded numerator RMSE from
84.897%/149.155%/87.967% to2.163%/7.006%/3.421%. Exact theoretical relative
SD remains3.717%/20.124%/8.084%. Raw-scatter Gaussian improves the original
prior more, but is less reliable on expanded support. Shared-kernel weighting
adds little in these cases. At K8192 nearest-location has slightly worse
empirical original-prior RMSE than current ranking; report preserves all arms.
Only the exact-set rule changed. These selected cases justify representative
testing of the simpler retrieval rule, not default promotion or a0.3% claim.
Ruff/Bash/compilation checks pass. No pending jobs remain.

## 2026-09-25 — Trace posterior-important atoms missed by the original sampler

Owner asked which atoms were missed in the annotated original-prior maps and
why. Added `scripts/diagnose_missed_atoms.py`, GPU16696539. Scope is fixed to
the20.23m original prior, observations142230/3563/409188, K1024/M16384 and
seed8701. Replays exact lists/draws and proposal probabilities, ranks important
missed atoms through the actual float32 location prefilter and float64 proxy
reranker, and reports their physical properties, raw/floored flow scatter,
expected draw counts and miss probabilities. No model or numerical settings
change. Ruff passes; job submitted, results pending.

GPU16696539 completed; exact-list/draw/probability/mass replay and independent
re-ranking checks all pass. [Published diagnosis](figures/inference_missed_atoms_20260925/README.md#dominant-missed-atoms):
row3563 atom8865748 has90.0723% posterior mass, location rank13, Gaussian
rank1259, and61.0343% zero-hit probability in16384 draws. Row409188
atom6984342 has69.2913% mass, location rank9, Gaussian rank1257, and58.1699%
zero-hit probability. Both passed the prefilter but fell below K1024 and
were not drawn. Their e1 proposal scatter is broadened47–54fold and flux
scatter19–23fold relative to raw flow scatter. Different Sérsic profiles;
shared characteristic is bright, narrow actual likelihood under a broad
proposal. Other environment realizations of the same intrinsic properties
were sampled, explaining the smaller reached fractions. Original row142230
dominant atom is protected by Gaussian rank234 despite zero random hits.

Saved full per-atom evidence and source under the published report directory.
No settings change. Reserving nearest-location candidates before re-ranking
is a targeted next hypothesis; these examples alone do not establish a
population-wide fix. No pending jobs remain.

## 2026-09-25 — Test 8192 exact plus 8192 sampled atoms

Owner requested the8k+8k split after the bright-prior expansion comparison.
Generalized `scripts/bright_prior_extension.py compare` with explicit exact
and sampled budgets and output directory; the scheduler wrapper forwards
arguments. Preserve the original1024+16384 defaults and source snapshot.
GPU16696513 runs K8192/M8192 on both original and expanded priors, the same
three observations and all16 seeds8701–8716. Model/likelihood, prefilter131072,
metric, floors and prior weights stay fixed. Existing exact centre sums are
reused; no shear fit or default promotion.

Added `scripts/report_inference_budget.py`; dependent CPU16696518 compares
both budgets, independently replays both priors' saved draws, verifies exact
sets are nested, and checks full log sums. Report also measures exact posterior
mass and dominant-atom capture. Ruff and Bash syntax pass. GPU startup is
normal; results pending. External results: `inference_bright_extension_20260925_v1/comparison_8k8k`.

Completed GPU16696513 in1m56s and CPU16696518 in14s. All six independent
log-sum/draw replays and nested-exact-set checks pass; inspected the PNG.
[Published results](figures/inference_8k8k_20260925/README.md#results) show that
K8192/M8192 resolves these three centre-integration cases on the original prior:
relative numerator RMSE is0.00930%,0.04321%,0.00491% over16 seeds, with
99.982%,99.887%,99.978% of exact posterior mass summed deterministically.
These are RMSEs, not maximum errors or shear bias.

For the expanded prior, corresponding RMSE is1.861%,267.712%,87.365%, versus
84.897%,149.155%,87.967% at K1024/M16384. Row142230 recovers its dominant atom
and97.319% exact mass, but rows3563/409188 still miss their dominant atoms
and sum only0.656%/1.634% exactly. Larger K helps but does not resolve ranking
on expanded support; halving the sampled remainder can increase tail variance.
No default change and no full shear fit. Both priors, original results and
all downloaded rows remain preserved. Ruff/Bash checks pass; no jobs remain.

## 2026-09-25 — Download and evaluate the supplied full bright catalogue

Owner supplied FS2_bright and authorized adding all rows to the existing prior.
CPU16696367 downloaded the389.8MB feather and README next to the actual
FS2_25876 source, checked file hashes, and audited7,780,811 unique physical
galaxy IDs. Overlapping151,828 source identities have different assigned
morphologies;848,650 rows are outside the old generator domain. Retain all
new rows and explicitly report the changed prior. Details and frozen protocol:
[bright extension](BRIGHT_PRIOR_EXTENSION.md#weights-neighbours-and-identities).

Added `scripts/bright_prior_extension.py` and scheduler wrapper
`jobs/job_bright_prior_extension.sh`: independent complete neighbour contexts,
canonical target/pair rescaling checks, fixed bright mass, centre-only exact
likelihood comparison on the original three rows, and16 fixed seeds for each
prior. Uses the previously verified external runtime; baseline caches and
default models are untouched. Construction CPU16696388 submitted. Ruff and
Bash syntax checks pass. GPU stages and results pending; no shear-bias or
improvement claim. All raw/download/audit receipts are external under
`inference_bright_extension_20260925_v1`. CPU16696388 completed in2m37s
(peak2.17GiB): all28,012,032 combined atoms represented, new pair rescaling
checks passed. GPU array16696392 started with two V100 workers and an
eight-shard/two-worker throttle. Comparison16696395 depends on its success.

Startup16696392 caught malformed NPY headers: NumPy scalar shape dimensions
were serialized as `np.int64(...)`. Cancelled remaining workers and dependent
comparison. Corrected dimensions to Python integers and added file-reopen
checks; preserved the failed construction separately. Rebuild16696407 passed
in2m19s. Replacement GPU array16696409 has two workers making flow-coordinate
progress; comparison16696411 and CPU report16696424 are dependent stages.
Added `scripts/report_bright_prior_extension.py` for the paired16-seed plot,
RMSE against each exact numerator and posterior-atom ESS. Ruff checks pass.

Owner clarified that the parent distribution is intended to be the same.
Corrected the overly strong interpretation of different individual morphology
assignments: these alone demonstrate no distribution shift. Added descriptive
source-distribution and magnitude-conditioned checks in
`scripts/audit_bright_distribution.py`, CPU16696435. Its matched-generator-domain
audit is distinct from retaining all rows in the primary experiment.
Final CPU16696444 also preserves original float32 boundary comparisons and
matches the saved new-domain mask exactly. Raw magnitude/radius/n/q maximum
CDF differences are0.197/0.685/0.294/0.300pp; median n is2.198459 in both.
Old/new generator exclusions are11.2206%/10.9070%, predominantly n=6.
These descriptive checks support additional realizations of the intended
same parent; they do not prove equality of the complete joint distribution.

All eight GPU shards completed in approximately2m26s–2m38s each. First
comparison16696411 and report16696424 completed; independent log-sum and draw
replay checks passed, but source review caught an implicit proposal metric
refit: `ProposalCoordinateTable.__post_init__` recomputes fractional-coordinate
quartiles even when its public centre/scale fields are supplied. This violated
the prospectively fixed-metric comparison. Preserved those results externally
as `comparison_implicit_metric_refit` / `report_implicit_metric_refit` and marked
the initial reported sampler outcomes provisional. Freeze the derived metric
too and assert exact standardized-coordinate equality on every10000th baseline
atom. Corrected comparison16696478 and report16696481 reuse all valid density
evaluations; no model or target likelihood changed.

Corrected comparison16696478 completed in1m30s; report16696481 completed in9s.
All three independent log-sum/draw-replay checks pass, the original seed8701
baseline is reproduced, and the fixed metric identity assertion passes.
Final [results](BRIGHT_PRIOR_EXTENSION.md#results-and-decision): relative
centre-numerator RMSE over16 seeds changes0.000311→0.848967,
0.927664→1.491547 and1.013688→0.879673. Posterior atom ESS improves modestly
1.15→1.64,1.22→4.73,1.74→2.66, but integration does not reliably improve.
For row142230 an atom carrying77.62% of expanded posterior mass is displaced
from the exact list and missed by seed8701. Outside-generator-domain rows
carry only0.0381%,2.5173%,1.3497% posterior mass; not the main failure.

Published PNG/PDF, summaries, independent validation, distribution audit and
source snapshots under `doc/figures/bright_prior_20260925/`; inspected PNG.
Saved an explicit experimental weighted-union manifest externally as
`combined_prior.json`. Retain all new rows and the unchanged total bright
mass, but keep the baseline default: no representative inference improvement
or shear-bias claim. Next work should improve candidate ranking and establish
numerical convergence before a full fit. Removed only this task's malformed
temporary arrays, retaining failure receipts. Ruff/Bash/whitespace checks
pass; all jobs complete and no new jobs remain.

## 2026-09-25 — Remake sampled-atom maps with the latest tested inference proposal

Owner requested the earlier scatter plots using the current inference
algorithm. Fixed the original three diagnostic observation rows142230/3563/409188
before evaluation. Added `scripts/plot_current_inference_atoms.py` and
`jobs/job_current_inference_atom_maps.sh`. Uses the 20,231,221-atom weighted
bright-enriched prior, median scatter floor, fractional flux metric, K1024,
prefilter131072, M16384, seed8701 and existing initial shear centre from
job16631250. These are V3.6-like model artifacts, not V3.9 inference.

Snapshot commit773295d into the external `inference_atom_maps_20260925_v1/runtime`
directory: every implementation hash matches the saved reference result.
The live proposal-comparison branch only differs in its later truth-cut store
extension; its sampler is unchanged. No live worktree or prepared cache is
modified. The old plotting script chose its exact stratum by top mixture
probability; the production driver uses location-prefilter candidates.
The new plot calls the actual production candidate and draw methods instead.

Each row gets a full-prior centre-node density scan, top64 posterior atoms,
actual reached mass and estimated/exact numerator ratio, plus measured-space
and intrinsic-truth scatter panels. Nonpositive proposal flux means are
reported and omitted from the logarithmic axis. Saved log terms and draw IDs
permit independent checks. A40 job16696306 was cancelled while pending due
to unavailable resources; replacement Slurm16696310 started on th-cl-nv01
(one V100-SXM2-32GB, four CPUs,96GiB,30minutes). Resource-check and Matplotlib
skills applied; scheduler allocations bound usage rather than whole-node
resource-detector recommendations.

Completed [figures and results](figures/inference_atoms_20260925/README.md#results):
full-prior posterior mass reached is99.9247%,9.4684%,30.6568% for
rows142230/3563/409188. Estimated/exact likelihood ratios are0.999554,
0.376566,0.561436. The latter two rows miss dominant atoms; many distinct
sampled atoms do not guarantee useful coverage. These are centre-density
diagnostics, not new shear estimates.

Validation: GPU16696310 completed in9m50s. Rendering16696335 completed in16s;
independent CPU verification16696321 completed in6s, checking output hashes,
full-prior normalization, reached mass, likelihood ratios and sample/exact
membership. Row3563 reproduces the saved production count of14443 distinct
complement atoms. All checks pass. Inspected all three final PNGs; corrected
overlapping title text and painted highest-mass atoms last so minor atoms
cannot obscure them. PNG/PDF, plot-data NPZ, reports, resource/provenance
receipts and both calculation/rendering source snapshots are preserved in
`doc/figures/inference_atoms_20260925/`. Full log terms remain external.
Python syntax/Ruff, launcher Bash syntax and documentation whitespace/link
checks pass. Updated the repository map. No training, default change or new
shear estimate. These are three previously inspected failure cases at one
centre, not independent or population-level validation. No jobs remain and
no further experiments are scheduled.

## 2026-09-25 — Recover later inference tests from other worktrees

Owner requested a cross-branch check after the recap missed later inference
tests. Inspected all eight registered worktrees, branch histories, relevant
working-tree changes, and external result receipts. The newer numerical work
is on `worktree-proposal-coord-compare` (also inherited by `worktree-nocut-retrain`),
not the current `dev` checkout. Added the
[cross-worktree results](INFERENCE.md#later-cross-worktree-inference-tests)
without merging code or changing defaults/models/caches.

That branch declares v1.3-infer: K1024/16384 tilted-stratified inference with
median proposal-scatter floors and fractional flux distance. The saved disk
experiments remain labelled custom, based on v1.2-infer-16k, with original
V3.6 hashes. A weighted 20,231,221-atom bright-enriched prior gave positive
information on two separate 10k windows, jobs16631250/16631862. Pooled
g1=0.03768952 with robust SE0.01724200; injected g1=0.02. Earlier branch
entries confused the Newton step with final shear; the later entry and this
independent re-reduction include the starting centre. The r<27/10,237,320-atom
comparison (16637857/16637858) fails pooled positive definiteness.

Validation: read four result.json receipts, verified all four moment-file
SHA-256 values, and independently reduced the small saved score/information
arrays with single-threaded NumPy. Reproduced both windows, both pooled
information eigenvalues, and robust covariance from per-object influence
residuals. These checks re-read existing outputs; no new likelihood evaluation
or scheduler job. Documentation whitespace/link checks pass. The separate
engineering-review branch adds config binding and tests; sampler-night's
latest committed preflight is August30, not a newer disk inference result.

Limitations: two windows from one catalogue/one prior seed, heavy importance
weights and large robust errors; no robust 0.3% inference claim. V3.9 models
were not used. No branch was merged and no experiment restarted. Next steps:
none scheduled; preserve branch provenance in future pipeline recaps.

## 2026-09-25 — Freeze V3.9-like and close development

Owner requested wrap-up, the V3.9-like name, and thorough repository cleanup.
Pinned the existing F8 flow, primary seed20260913 C8 classifier and unchanged
physical-moment disk emulator in `configs/models_v3_9_like.json`. Artifact
SHA-256 values were verified against the external files. Added the named
preset through a shared disk-model manifest loader in `sbsi/models.py`;
V3.5/V3.6 identities and configured likelihood/estimator remain unchanged.

[The model guide](V39_MODEL.md#development-evidence) records measured-only
per-leg radius >0.6 arcsec/magnitude <25.8 cuts, no truth-property evaluation
cuts, the target-role parent, +0.020770% ±0.265152 percentage-point development
bias, seed replicate, failed broader classifier checks and absence of
independent acceptance. Copied original comparison/protocol/training/selection
receipts into `doc/evidence/v3_9_like/` and pinned their hashes in the manifest.
No training, recalibration, favorable seed selection or emulator reduction.

Moved 37 completed experiment scripts, 38 fixed-run launchers, 15 experimental
test files and five investigation documents out of the active tree, preserving
all bytes under `archive/research-2026-09-25/`. Its hash manifest also records
66 snapshots of pre-existing dirty working-tree files. Retained reusable
preparation/inference/audit code and core regression tests. Renamed model-preset
tests to `tests/test_model_presets.py`; added V3.9 identity/evidence checks and
an optional real-artifact prediction test. Removed unused circularized-feature
training helpers and the historical LRU-patch cache-identity exception: cache
reuse now requires exact scientific and implementation identity. Old prepared
caches must be rebuilt; the disk preparation driver is still V3.6-specific.

Updated README, model guide, API, conventions, inference/environment guides,
AGENTS and [repository map](REPOSITORY.md#september-25-cleanup). Added the
reusable one-CPU Slurm test launcher `jobs/job_tests.sh`. Scientific figures,
large artifacts, datasets, external research checkouts and result files are
preserved. No git reset/clean or commit was performed.

Validation: all four external model/metadata hashes match; all 161 archive
entries pass byte-count and SHA-256 checks; active Python syntax, config JSON,
local imports and maintained-document file links pass. Changed-file Ruff
and `git diff --check` pass. Final full suite, including external V3.9 model
loading and joint prediction: **294 passed, 1 skipped** in Slurm16696170 using
py31 (52.98s); the system Python run16696166 also passed294/skip1 (36.50s).
[Verification receipt](evidence/v3_9_like/cleanup_validation.json) pins source
hashes and preserved test logs.

Initial jobs16696148/16696162 exposed two test portability problems: NumPy
boolean indexing of a Torch tensor and a small random XGBoost fixture that
did not necessarily split on both conditioned outputs. Fixed the test to use
an explicit Torch boolean mask and a balanced factorial fixture; precision,
physical-prediction and exact-pooling assertions remain intact. No production
model calculations were changed for those repairs. The rerun revealed stale
shared bytecode; generated bytecode caches were removed. The reusable test
launcher now uses fresh job-local bytecode and `SBSI_TEST_PYTHON` so an inherited
system `PYTHON` cannot silently override the project interpreter.

Next steps: none scheduled. Development is closed. A future V3.9 inference
integration or independent calibration study requires a new owner request;
model naming alone does not establish robust 0.3% accuracy.

## Earlier work logs

- [Full log before this cleanup](../archive/research-2026-09-25/snapshots/doc/WORKLOG.md)
- [September20 snapshot](../archive/research-2026-09-20/doc/WORKLOG.md)
- [September14 snapshot](../archive/research-2026-09-14/doc/WORKLOG_FULL.md)
