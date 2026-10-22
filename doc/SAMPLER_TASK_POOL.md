# Bounded sampler task pool candidate

## Scope and acceptance boundary

This n27 candidate applies the four-process/two-physical-core layout to an
arbitrary finite list of independent Python sampler invocations. It starts from
`13a1fcaaa52af407a526ac45690ad1325ae14845`; the sampler and original concurrency
benchmark are unchanged. The pool does not merge observations, adjust seeds,
change a likelihood request, choose a recipe or change a scientific artifact.

This is an execution mechanism awaiting scheduled full validation and separate
real saved-output replay. Neither synthetic tests nor a `completed_execution`
receipt establish posterior convergence, scientific acceptance or throughput.
Production use requires the n24 fresh-seed confirmation and the separate real
replay gate. There are no new experiments, scheduled jobs, GPU service, MPS,
network/RPC layer or multi-GPU scheduler in this change.

## Manifest size limit

Receipt updates currently rewrite the full ordered task list after each launch
and completion. For N tasks this produces O(N²) cumulative receipt bytes. The
current real replay is bounded to eight tasks. Use bounded-size manifests for
any later adoption, retaining every original group and seed; do not treat this
implementation as a catalogue-sized queue. A journal or other receipt-persistence
optimization needs a separate implementation and replay gate.

## Immutable declaration

`run_sampler_task_pool.py` accepts one JSON manifest and its independently
recorded SHA256. Duplicate JSON keys, missing/unknown schema fields, nonfinite
JSON, duplicate task IDs, duplicate invocations and overlapping outputs are
rejected. Every source must be a complete regular-file snapshot with a pinned
`SHA256SUMS`; unlisted files, symlinks, missing files and changed bytes fail.
Preparation should make source, artifacts and manifest read-only. The pool
checks every source and declared input before any task and again after execution.

The exact top-level schema is:

```json
{
  "schema": "sbsi-sampler-task-pool-v1",
  "workers": 4,
  "cores_per_worker": 2,
  "output_root": "/absolute/fresh/run",
  "python": "/canonical/absolute/python3.10",
  "environment": {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1"
  },
  "sources": {
    "original": {
      "path": "/absolute/immutable/source",
      "manifest_sha256": "SHA256 of source/SHA256SUMS"
    }
  },
  "harness_files": {"/absolute/harness/file": "SHA256"},
  "input_files": {"/absolute/input/file": "SHA256"},
  "tasks": []
}
```

Use canonical absolute paths (`Path(path).resolve()`), including the interpreter.
The Python executable, recipe, target-law definition and every external artifact
read by the declared sampler belong in `input_files`. The pool verifies the
provided inventory; separate scientific preparation must establish its
completeness. Native/shared-library versions are not inferred or automatically
pinned. Runtime and science equivalence still require the separate replay.

`harness_files` must pin all four files from this candidate:
`run_sampler_task_pool.py`, `run_sampler_task_pool.sh`,
`benchmark_smc_concurrency.py` and `benchmark_exchange_full_smc.py`.
Additional operational environment values may be declared, such as artifact
cache roots. Device visibility and Python import overrides cannot be supplied
by the manifest; device visibility comes from the allocation. Ambient Python
injection variables are removed. Effective environment values are recorded for
explicit operational variables; other values are represented by hashes.

Each task has exactly these fields:

```json
{
  "id": "population_0000",
  "source": "original",
  "cwd": "/absolute/immutable/source",
  "entrypoint": "/absolute/immutable/source/scripts/check_galsbi_particles.py",
  "output": "/absolute/fresh/run/results/population_0000",
  "argv": ["COMPLETE FROZEN COMMAND; see preparation helper below"],
  "identity": {
    "seed": 2026105000,
    "observations": [58165, 0, 1, 8],
    "recipe": {"path": "/absolute/recipe.json", "sha256": "SHA256"},
    "target_law": {"path": "/absolute/target.json", "sha256": "SHA256"}
  },
  "argument_indices": {"seed": 0, "output": 0, "observations": [0, 0, 0, 0]},
  "expected_outputs": ["result.json", "protocol.json", "progress.json", "particles.npz", "scenes.npz"]
}
```

The example's `argv` and argument positions are placeholders, not a runnable
manifest. Construct each full command using the module's preparation helper:

```python
argv = pool.command(python, source, entrypoint, original_arguments)
```

This inserts a fixed, validated Python bootstrap **when preparing the manifest**.
It places the pinned source first on `sys.path`, asserts that `sbsi` comes from
that source, and executes the pinned entrypoint. At execution the complete
`argv` is passed unchanged, with no shell expansion or argument reconstruction.
The source/entrypoint bootstrap is itself checked before launch. Arbitrary
Python sampler entrypoints and numerical arguments can follow it.

The `argument_indices` are zero-based positions in that complete `argv` for
one seed, one output path and the four original observation values. They must
point to six distinct values after the bootstrap and agree exactly with the
identity. Preserve each group's original order. The pool does not concatenate
or repartition groups. The pinned recipe/target artifacts document scientific
identity; the pool does not reinterpret them or validate statistical adequacy.
Tasks may use different predeclared seeds, groups, recipes and source snapshots.

Every output must be a distinct, nonoverlapping directory below
`output_root/results/`. The root is reserved with exclusive creation before
workers start, while individual output directories remain absent for their
drivers to create. Task IDs name separate receipt directories. Reusing an
existing root or output is an error; there is no resume or overwrite mode.

## CLI and resources

Light metadata preflight requires no GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python \
  /absolute/harness/run_sampler_task_pool.py validate \
  --manifest /absolute/tasks.json --manifest-sha256 MANIFEST_SHA256
```

After the separate validation/replay gates, run the frozen launcher inside an
already allocated scheduler step with one `th-cl-nv02` V100 and eight physical
cores, using `--hint=nomultithread` and
`--ntasks=1 --cpus-per-task=8 --cpu-bind=cores`:

```bash
bash /absolute/harness/run_sampler_task_pool.sh \
  /absolute/tasks.json MANIFEST_SHA256 /absolute/fresh_outer_cost.json
```

The launcher submits nothing. It reserves the outer cost file exclusively;
that file must be outside both task outputs and source snapshots. It then times
the complete Python CLI with `/usr/bin/time`, retaining nonzero exit status and
wall/user/system/maxRSS even when startup or a worker fails.

The resource check resolves CUDA-visible ordinal zero to its physical UUID in
a short subprocess, then queries that UUID. It requires one visible V100, the
specified node and an actual coordinator affinity containing exactly eight
distinct physical cores. Wider affinity is rejected; multiple SMT IDs for those
same eight cores are allowed. SMT siblings do not count as separate cores. Each child receives two nonoverlapping cores via
the proven fresh-process affinity wrapper; no threaded `preexec_fn` is used.
The pool worker entrypoint releases inherited TERM/INT masks before invoking
the unchanged affinity helper. The coordinator remains on the allocated CPUs. Four is the maximum number of
active tasks, including multiple waves for longer manifests.

## Failure behavior and receipts

Launches preserve manifest order. Every original task has a record from the
start; completion order may differ. A nonzero exit, missing expected output,
incorrect worker command/affinity, launch error or cancellation stops dispatch.
Parent TERM/INT handlers record cancellation without raising asynchronously;
the scheduler handles it at explicit safe bookkeeping checkpoints. Launch/registration
and reap/bookkeeping are also guarded. Repeated cancellation is ignored during
cleanup and final receipt saving. Live sibling process groups receive TERM, then
KILL if needed, and their leaders are reaped. Tasks left with live descendants are rejected and those descendants
are killed. Pending tasks remain explicitly `unstarted` with the failure reason.
There are no retries, replacement seeds, resubmissions or dropped failures.

`output_root/receipt.json` records:

- Original manifest identity and ordered task definitions/statuses.
- Source and input hashes before/after, resource identities before/after and
  CPU topology/selected cores. Source drift after otherwise successful children
  makes the whole execution fail.
- Each child PID, exact command, cwd, affinity receipt, start/end offsets,
  observed wall duration, `wait4` user/system/maxRSS and context switches.
- Required-output presence plus hashes of **all** produced output files;
  runtime receipt/log hashes and partial-output hashes on failure.
- Internal verification/dispatch/saving costs and explicit execution-only
  success scope. All output files remain in their original directories.

The external `/usr/bin/time` receipt is the complete Python execution cost,
including interpreter startup, resource probes, hashing, child launches,
waiting, saving and final checks. Small launcher metadata checks before that
process, scheduler queueing and allocation setup are outside this cost.
The internal timer excludes Python imports and its final receipt serialization;
use the external receipt for total execution cost. Per-task wall times include
polling detection delay (up to approximately 25 ms under normal scheduling).
Peak RSS values are process/family maxima, not simultaneous summed memory;
GPU snapshots are before/after measurements, not a peak-memory trace.

The pool checks that required files exist and pins their bytes. It does not
classify sampler result JSON or establish exact scientific replay. A task that
exits zero with a scientifically unsuccessful result can therefore be
`completed_execution`; a separate semantic/replay audit must assess it.

## Synthetic validation and remaining gate

Focused tests use tiny Python processes and real Linux physical-core affinity.
They cover nine tasks over multiple waves, unchanged command/seed/group/RNG
payloads, bounded concurrency, SMT/overlap rejection, failure and forced sibling
termination, unstarted records, interruption, launch failure, output collisions,
missing outputs, live descendants, source/input/manifest tampering and final
source drift. Fault injection sends actual TERM/INT immediately after Popen and
reap, during cleanup and final receipt saving, including signals delivered through
an unblocked background thread. Child masks must leave TERM/INT unblocked.
Source aliases cannot bypass duplicate-task rejection, and recipe/target-law
artifacts require canonical paths, valid SHA256 values and explicit input pins. No scientific target, GPU or model is loaded.

The complete scheduled repository suite and a predeclared real sequential versus
pooled saved-output replay remain for the parent workflow. This candidate is
not adopted by any existing launcher or running scientific experiment.

## Fixed eight-task real replay preparation

`benchmark_sampler_task_pool_replay.py` prepares the separate execution replay.
It hard-pins both completed declarations/reports and independent parent audits:
development audit 17447670 and confirmation audit 17448939. The latter rechecked
all 16 saved outputs and the confirmation timing gate. All original report/cost
receipts remain pinned; preparation revalidates both source/recipe declarations
and all eight original serial setting-0 populations.

The replay consists of exactly two fixed groups of four independent tasks:
seeds **2026103700–2026103703**, then **2026103800–2026103803**, each retaining
observations `[58165, 0, 1, 8]`, N256/T64/M64, density4096, zero shear, original
proposal/geometry and the original 320-file source manifest `fd96ec78…64df1`.
The source is copied to a fresh immutable directory. Original driver arguments
are preserved, with only the new output destination and the explicit checked
source bootstrap prepared into the manifest. There is no scientific selection
among tasks and no new fitted artifact.

In a CPU scheduler allocation, using this candidate snapshot:

```bash
PYTHONDONTWRITEBYTECODE=1 /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python \
  scripts/benchmark_sampler_task_pool_replay.py prepare \
  --root /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021 \
  --destination /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/sampler_task_pool_replay_prepared_v2_20261022
```

Preparation saves `tasks.json`, `replay.json`, six pinned harness files, a complete
source copy and top-level `SHA256SUMS`. It validates the copied pool using the
copied paths. The final directory is read-only. No source freezing or scientific
replay work is performed on the login node by this command.

After full scheduled CPU validation, the parent may execute in the specified
one-V100/eight-physical-core allocation:

```bash
bash /absolute/sampler_task_pool_replay_prepared_v2_20261022/run_sampler_task_pool_replay.sh \
  /absolute/sampler_task_pool_replay_prepared_v2_20261022
```

The fresh outputs under the study root are fixed as:

- `sampler_task_pool_replay_runs_v1_20261022/` — all eight task outputs/receipts.
- `sampler_task_pool_replay_process_cost_v1_20261022.json` — complete pool process.
- `sampler_task_pool_replay_audit_cost_v1_20261022.json` — complete replay audit.
- `sampler_task_pool_replay_report_v1_20261022.json` — final audit and charged work.

The launcher submits nothing and cannot overwrite the original attempt.
The audit loads each original phase's pinned validation helper and requires its
original recipe, full 64-stage checks, all scientific JSON and full saved array
bytes. Only existing named timing/job differences and checked implementation
root relocation are normalized. No new scientific/runtime exclusions are added.
It also checks all eight worker records, exact commands, four simultaneous
workers without CPU overlap, unchanged outputs/inputs and complete outer costs.
A mismatch remains a failed replay; it is never converted to a tolerance result.

The pending audit report pins its exact replay declaration, pool receipt and pool
cost receipt. Finalization rechecks all three hashes, the declaration's task-manifest
hash, the exact declared report/audit-cost paths and the original pool/preparation
cost content before marking replay passed. A stale receipt or identically copied
cost supplied at a different path is rejected. Finalization does not repeat the
source or array replay.

The reported new work includes measured preparation, complete pool and audit
processes, and measured finalization proof-validation time. Preparation's
interpreter startup/final freeze bookkeeping, launcher metadata, queueing,
finalizer interpreter startup and final report serialization remain excluded
explicitly. The first prepared v1 snapshot is preserved; use the fresh v2
preparation after this repair. The fixed actual replay output names stay v1,
because no task-pool GPU execution or output directory existed at this repair.
The comparison uses historical serial saved outputs; it does **not** claim a new
paired speedup from timings collected in different allocations. All eight saved
outputs must agree before `generic_pool_replay_passed` can become true. This
still makes no claim about posterior accuracy or catalogue-scale queue efficiency.
