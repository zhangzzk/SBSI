# N22 saved whole-run-output replay and cost check

## Current status

Reviewed v2 is submitted as GPU array17437046 (`0-3%1`) and CPU report17437047.
Its afterany dependencies wait for both held-out GPU arrays; Slurm has already
satisfied completed warmup17435348 and retains production17435350. Both new
jobs are pending their prerequisites. Root verified both304-file scientific
copies, all168 external inputs, harness/test/fixture receipts and the declared
single-source difference before submission. The [launch receipt](evidence/exchange_full_smc_launch_20261022.json)
preserves commands, identities and limits. No full-run timing or replay result
exists yet; no runtime optimization is promoted.

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

After root review only, the GPU array must use inter/th-cl-nv02, one V100,
two CPUs, array0–3%1 and afterany:17435348:17435350. The dependency enforces the
resource ceiling regardless of the held-out experiment's scientific outcome.
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
