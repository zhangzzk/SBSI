# Frozen legacy particle trajectory audit

## Purpose and boundary

`scripts/audit_legacy_particle_trajectory.py` prepares or executes one full
replay of an existing v7, v9, v8-default or v8-informed development population.
It instruments the **old** native exchange correction and returns the original
ratio array unchanged. The corrected helper supplies support masks only.
No inference/GPU replay was submitted while preparing this harness; no v8/v9
scientific outcome or heldout outcome was inspected. This is a diagnostic,
not a new sampler or a correction to historical estimates.

The harness is deliberately restricted to the audited frozen source manifests,
the original fixed 64-stage schedule, pilot proposal/geometry, development run
names and original runtime. It rejects changed source or configuration, heldout
run names, incomplete/unknown failures, and nonempty audit output directories.
It never edits original runs or snapshots. All old source files and original
artifacts are checked before and after execution.

## What the counters mean

The instrumented helper pairs its old-target and new-neighbour calls within
each `exchange` or `exchange_relocate` mutation. It maps both the finite-prior
subset and the nonempty-neighbour subset back to original observation owners;
owners remain original IDs when extinct observations are skipped. Every beta,
observation and exchange block has a receipt row, including zero-attempt rows.

- `attempted` and `old_accepted` reproduce the legacy driver's actual decisions.
- `proposed_off_support` counts paired proposals with finite legacy corrections
  that the corrected helper would reject for the lower uniform-clamp bound.
- `old_accepted_off_support` counts those proposals **accepted by the old
  implementation**. An informed v8 guard may reject them, which must remain a
  proposed event with zero old acceptance.
- `proposed_target_into_zero` / `old_accepted_target_into_zero` isolate an
  analytic target mark entering a zero-density numerical environment slot.
- `proposed_selected_neighbour_zero` and its acceptance counterpart identify
  an already unsupported selected neighbour. This direction can remove an
  earlier invalid mark; it is not necessarily a fresh entry into zero support.

The pair is counted only when both old mark corrections and the returned
legacy proposal correction are finite. Proposals already rejected by another
old support condition are not attributed to the new restriction. Every stage's
attempt and acceptance counts are reconciled against the full saved progress.
The audit draws no random numbers and makes no additional likelihood calls.

## Required replay agreement

Before counts become trusted, all of the following must pass:

1. Every particle/scene NPZ member has identical field name, dtype, shape and
   raw bytes, and the entire archive bytes match. This includes coordinates,
   priors, kernels, scores, normalizers, genealogy, extinction, all native marks
   and offsets. Informed runs additionally require all raw initial draws and
   complete final environment payloads.
2. Every saved stage agrees, including weights/ESS/normalizers, ancestor counts,
   region summaries, extinction and acceptance. Complete stage numbering and
   beta coverage are required; terminal arrays alone cannot pass.
3. All scientific protocol/result metadata agree. Only explicit elapsed/timing
   fields and scheduler job/node identifiers may differ. The consequent
   protocol digest is verified against each real protocol before comparison;
   model runtime/device/thread settings are not ignored.
4. Original RNG JSON artifacts are compared if present. Unknown/missing files
   or a changed source/diagnostic implementation fail closed.

The original frozen drivers do **not** persist full intermediate states,
cached proposal/type fields or final RNG. Matching terminal arrays cannot
retroactively verify those missing records. The harness instead executes every
old transition with instrumented decisions, checks all saved stage history,
and stores replay-only `unpersisted_final_cache.npz` and
`replayed_final_rng.json` outside the ordinary replay output directory. These
new receipts are explicitly not presented as historical evidence.

On success `audit_receipt.json` says `verified_full_replay` and
`counts_trusted=true`. Any execution error or disagreement leaves a failed
receipt, with counts marked provisional. No generic ignore/allow flag exists.

## Commands and resource expectations

Planning is light and reads only `protocol.json`; it does not require outcomes
or initialize models. For example:

```bash
W=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/endpoint-compatibility-audit-20261022
R=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PY" "$W/scripts/audit_legacy_particle_trajectory.py" \
  --run "$R/dev_v7_0" --output "$R/legacy_support_replay/v7_0" --plan
```

The exact replay command, **inside a root-scheduled Slurm allocation**:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
"$PY" "$W/scripts/audit_legacy_particle_trajectory.py" \
  --run "$R/dev_v7_0" --output "$R/legacy_support_replay/v7_0"
```

Use the original V100 device/runtime and at least two allocated CPU cores so
the driver retains its recorded two torch threads. The CLI checks those
settings. Root schedules any replay under the existing four-GPU study cap;
this harness neither submits jobs nor increases that cap. Replace the run and
fresh output suffix to audit a declared `dev_v9_0..7`,
`dev_v8_baseline_0..7` or `dev_v8_informed_0..7` population. Each run must have
finished and passed the artifact preflight first.

Cost is one complete original inference run, including final scores, plus CPU
support checks and exact artifact verification. v7/v8 execute 2,048 mutation
calls and at most 256 extra vector native-prior/support evaluations (262,144
mark rows for the four-observation N256 configuration). v9 executes 4,096
mutations and at most 512 extra evaluations (2,097,152 mark rows at N1024).
There are zero extra likelihood calls. Wall-clock overhead is unbenchmarked;
this is not a cheap final-state scan. Start with selected runs before deciding
whether a complete batch is justified.

## CPU validation

`tests/test_legacy_exchange_audit.py` tests paired row mapping, original accepted
and rejected events, multiple beta/block groups, extinct-owner subsets, empty
environments, complete artifact and history rejection, and plan-only behavior.
It also loads untouched real v7/v8 modules for an ordinary 24-transition
state/decision/RNG comparison, forced old acceptance/rejection of a known
support-violating mark, and the actual informed v8 rejection guard. These are
synthetic CPU fixtures; no real population trajectory is claimed to be cleared.

The first CPU job 17422503 passed 19 tests but five real-source tests failed
because the fixture MRO selected the real summary builder without observing
conditions. The fixture was repaired to retain its synthetic summary while
using frozen evaluate/mutate functions. Job 17422593 passed 24 tests; adding
the v8 informed guard gave 26 passes in job 17422638. Final CPU job 17422670
passed all 26 tests in 13.37 s. Plan-only commands also succeeded for one real
startup protocol from each of the four groups without reading outcomes.
Final source/validation
receipts are under `doc/evidence/legacy_trajectory_harness_validation_20261022.json`.
