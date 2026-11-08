# Particle prior reuse candidate

## Completed paired replay and independent audit

GPU17448787 completes all four original v2 pairs. Every saved scientific JSON
value, all64 stages and particle/scene array bytes replay exactly. Baseline to
optimized time ratios are **1.0793,1.0725,1.0703,1.0885**: each counterbalanced
pair favors the optimization. Inclusive new-work times are **871.965 versus
810.386 seconds**, ratio1.07599, or **7.06% less runtime**. These are conditional
on existing scientific fits and the earlier native replay; the failed startup's
248.06 process seconds remain a separate retained experiment cost.

Independent CPU audit17449393 passes **1,052 tests,nine skips**, then replays all
four pairs and independently recomputes the timing/cost arithmetic. It checks906
source/input/output pins and the end-of-pair GPU identity. All475 audit-source
hashes match before/after; parent checks them, JUnit and the report/cost receipts.
The [completed evidence](evidence/particle_prior_reuse_completed_result_20261022.json)
records exact scope and costs; the [audit contract](PRIOR_REUSE_COMPLETED_AUDIT.md#independent-checks-and-interpretation)
defines what passed.

This supports a modest complete-run efficiency gain for the existing analytic
recipe on one GPU/two physical cores. No timing acceptance threshold was declared,
and no automatic adoption follows. Combined performance with the four-worker
pool is unmeasured; do not multiply their speedups into a measured capacity.
Posterior convergence, numerical-target production and catalogue calibration
remain unaccepted. Earlier preparation/failure entries below are historical.

## Original GPU startup failure and repair

Job17446115 fails1:0 after249allocated seconds. Its baseline population
completes, but the optimized arm's original proposal artifact pins the old
native-target implementation hash. Strict production validation rejects the
new hash before optimized initialization or sampling. This is an artifact
compatibility failure; it supplies no speedup or full GPU replay result.
All v1 output, costs and failure logs remain preserved in
[startup evidence](evidence/particle_prior_reuse_gpu_startup_failure_20261022.json).

The isolated repair prepares a new v2 experiment with an explicitly derived
proposal artifact: numerical fitted-proposal fields remain identical, while
implementation provenance binds both source manifests and the completed
native-prior replay. Public production validation remains strict. A CPU
preflight must exercise the actual production proposal/geometry validation
before any GPU launch. Seeds and alternating arm order remain unchanged.
The repair and v2 GPU result are pending; no optimization is promoted.

## Scope and ownership

Hypothesis n25 removes repeated deterministic calculations from the fixed
particle sampler. The candidate is based on complete frozen source commit
`0301993445398ef0942eac0e2777138d5b405862` on isolated branch
`codex/smc-prior-reuse-20261022`. It does not modify live held-out sources.

- `GalSBIParticleTarget._evaluate_prepared` accepts quantities owned by the
  current transition. Environment and position moves reuse copies of the
  unchanged target's prior, native fields, type probabilities and proposal
  density. Summaries, likelihood and optional environment tilt are still
  evaluated for the proposed environment. Public `evaluate` performs its own
  native evaluation and never trusts native fields supplied with a scene.
- Initialization retains the native evaluation already needed for generating
  neighbours. Independence moves use the proposed state's freshly evaluated
  proposal density in the Hastings correction. Neither consumes additional
  random draws or changes the arithmetic of the acceptance calculation.
- `GalSBITargetCoordinates.log_mixture_prior` supplies photometry immediately
  produced by `decode` to a private native-density path. Only type changes
  between these evaluations. Public `GalSBIContinuousPrior.log_intensity`
  continues recomputing photometry from native variables, including when input
  dictionaries contain stale derived fields. Custom public intensity overrides
  retain their original dispatch. Type-specific support masks, factor order
  and the final log-sum remain unchanged.

No cross-transition physical/redshift cache, new random sampler, larger
likelihood batch, new kernel, model change or target change is introduced.

## Focused validation

The new `tests/test_particle_prior_reuse.py` compares the optimized path with
an uncached reference that calls the public native prior for each type and
re-evaluates all target quantities. It checks byte equality, including finite
and nonfinite values, for every native/particle/payload array; acceptance and
activity masks; random-generator state; and likelihood/reuse audit counters.
Cases include different type redshift bounds, zero native shape/SED support,
off-support initialized targets, informed environments, all main mutation
blocks in sequence, duplicated ancestors, rejected payload retention, fresh
output ownership, stale public photometry, custom intensity overrides, and
nonfinite coordinate rejection. Call-count checks establish only that the
intended repeated calculations were removed.

Targeted command with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`:

```bash
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -m pytest \
  tests/test_particle_prior_reuse.py tests/test_galsbi_particle_target.py \
  tests/test_galsbi_target_coordinates.py tests/test_informed_environment.py \
  tests/test_informed_environment_driver.py -q
```

Result: **104 passed, one skipped in 13.26 seconds**. The skipped existing test
requires `SBSI_ENVIRONMENT_BASELINE_TARGET`, an optional historical source path.
The new uncached comparison tests ran. A final coordinate-conversion adjustment
to match the original validation path was separately checked with the two
nonfinite-coordinate tests. `git diff --check` passes.

## Limits and next checks

This candidate is not adopted. Full CPU and installed-photometry checks pass;
independent baseline/optimized saved GPU replay and timing remain required on
the same native data, models, settings, seeds and original observation groups.
Actual installed photometry must replay exactly: reusing a whole decoded
photometry array replaces evaluations on type-specific subsets, and toy
elementwise photometry does not establish that numerical property for the
external implementation. Support-boundary agreement must be included.

Complete-process matched timing is required before any efficiency claim.
Removed call counts and earlier profiles do not establish a speedup. The
candidate neither certifies sampler accuracy nor changes scientific acceptance.

## Installed-photometry CPU replay

`scripts/check_particle_prior_reuse_native.py` prepares no new draws and uses no
GPU or learned models. It loads the pinned original prior, coordinates and
particle-target classes directly from the frozen baseline source under separate
module names. The original prior factory is bound explicitly to the original
prior and memoization classes; the candidate factory remains byte-identical.
Both sides construct separate priors with the installed photometry and PyCosmo.
The compared reference therefore executes the original `native_prior`,
`log_mixture_prior` and public `log_intensity`, including safe placeholders.

The checker requires the fixed completed development-audit hash and all 24
declared `small`/`large`/`broad` populations with their original seeds and row
counts: 73,728 rows total. It binds every particle NPZ, protocol and result to
that audit before evaluation. It checks the original complete per-population
coordinate arrays, followed by 15 deterministic support probes based on the
first saved coordinate: both types' redshift edges, out-of-range safe
placeholders, shape/SED zero limits, and bright/faint intrinsic support.

Receipt fields preserve dtype, shape, byte hashes, finite/nonfinite counts and
individual mismatch examples. There is no tolerance fallback. Input and source
hashes are checked again after evaluation. Mismatch returns a nonzero status;
exceptions also retain a failure receipt. Costs include both prior
constructions, evaluations, comparisons and hashing; these are validation
costs, not timings from which to claim a sampler speedup.

Run through the scheduler, with this candidate checkout as the import authority:

```bash
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python \
  scripts/check_particle_prior_reuse_native.py \
  --runs-root /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021 \
  --baseline-source /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/source_corrected_heldout_v3_20261022 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/prior_reuse_native_replay_20261022
```

The output directory must be fresh. Five small checker tests pass in 0.12
seconds: byte-level signed-zero/NaN distinctions, dtype/shape and native-schema
failures, complete population identity/hash enforcement, seed rejection and
fixed support-probe coverage. Syntax compilation and `git diff --check` pass.
The parent completed the actual-data replay and full CPU suite in Slurm job
17445720; see the results below.


## Scheduled full CPU and native replay results

Job **17445720** tested immutable source commit
`e513592aad467cd113920d2fb948dca2d8821c4e`: **1,092 passed, nine skipped**, one
existing NumPy warning, in 87.13 test seconds. All 479 snapshot hashes matched
before and after. Four physical CPU cores were requested; Slurm allocated eight
logical CPUs for 156 seconds, with no GPU.

The installed-photometry comparison passed byte-for-byte on **73,728 saved
coordinates plus 15 support probes**. All 25 cases preserve prior values, native
fields and type probabilities, including dtype, shape and nonfinite bit
patterns. Native-check process cost was 34.077 seconds wall and 21.516 CPU
seconds; this includes both implementations and is not a speed measurement.
The parent independently rechecked all 10 source and 73 input identities and
JUnit/scheduler completion. Results are retained under
`sampler_search_20261021/prior_reuse_native_replay_20261022/result.json` and
`smc_prior_reuse_validation_v1_20261022/validation.json` on project storage.

## Paired full GPU replay and timing

`scripts/prepare_prior_reuse_smc.py` freezes two complete 320-file sources that
may differ only in the three declared runtime files. It binds the successful
native replay, audited development protocol, external inputs and executed
harnesses. `scripts/benchmark_prior_reuse_smc.py` compares four complete pairs
with fresh seeds 2026103900–2026103903, alternating baseline-first and
optimized-first. Each arm uses the original four observations, N256/T64/M64,
two physical CPU cores, one V100, original likelihood grouping and saved fits.

All scientific JSON, all 64 stage histories and saved particle/scene bytes
must agree. Every original pair must finish without extinction; failures are
retained without replacement. `/usr/bin/time` receipts include the complete
pair and report processes, with shared verification costs divided equally;
source verification and copying are also counted. Preparation declaration/manifest
serialization and permission changes, existing fits, queueing, launcher setup
and final JSON serialization remain outside this conditional comparison.
The full sampler's unsaved intermediate states and terminal RNG are not
observed by this driver; focused transition tests cover RNG preservation.

The 31 focused harness tests pass in 0.88 seconds. Combined scheduled validation
**17446023** passed **1,123 tests, nine skips**, one existing warning, in 84.70
test seconds. All 483 frozen source hashes match before/after, and the prepared
baseline and candidate pass complete 320-file source/input/protocol preflight.
The parent independently checked 1,129 snapshot/prepared manifest entries,
JUnit and successful scheduler completion (123 seconds, eight logical CPUs).

GPU job **17446115** is queued after this successful test and concurrency job
17445452, on one V100 with two physical CPU cores and requeue disabled. Prepared
declaration SHA256 is
`da9358a05bb3a46714ea92c40030dd04d0161327b9454ecf0fd714c6a08175ef`;
optimized source manifest is
`f95d439d723e25375ca683a7134562f8a5262cd35d04c2b44a0e345bc52039c2`.
All four original paired comparisons remain required. No runtime speedup or
sampler acceptance is claimed. A bounded read-only scheduler/result-presence
check is scheduled as 17446128 about two hours later; it sends no chat message.

## Startup failure and validated v2 repair

Original GPU job 17446115 failed before optimized sampling: its frozen initial
proposal retained the baseline implementation hash. The strict validator rejected
that mismatch. Preserve the failed run and its 248.06 seconds of measured process
cost; it supplies no speedup or numerical comparison.

Commit 519b920 explicitly derives a proposal artifact with identical numerical
fit and migrated implementation provenance. CPU job **17448718** passes
**1,139 tests, nine skips**, then exercises all eight actual production startups
plus the original mismatched-artifact negative control. All 485 source hashes
match before/after. Parent verification checks 1,135 manifest entries and 189
external pins. No public validator or runtime algorithm was relaxed.

GPU job **17448787** is queued after that validation, on one V100 and two physical
CPU cores. It repeats the complete four-pair experiment with the original seeds
and alternating order in a fresh v2 directory. All pairs remain mandatory; no
GPU replay or runtime gain is claimed yet. The [v2 preparation receipt](evidence/particle_prior_reuse_v2_preparation_20261022.json)
binds the declaration, source, migration, startup checks and prior failed cost.
