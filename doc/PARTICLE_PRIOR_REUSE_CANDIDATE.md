# Particle prior reuse candidate

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

GPU job **17446115** failed after completing its first baseline population and
before the optimized arm produced any protocol or sampled artifacts. The
original proposal's native-source fingerprint correctly rejected the changed
coordinate implementation. The failed process used **248.06 seconds**; its
receipt, log, baseline outputs and failed optimized directory remain unchanged.
The completed CPU tests and metadata/source checks had not exercised this
actual production startup boundary. There is no paired speed result.

The original prepared declaration SHA256 is
`da9358a05bb3a46714ea92c40030dd04d0161327b9454ecf0fd714c6a08175ef`;
optimized source manifest is
`f95d439d723e25375ca683a7134562f8a5262cd35d04c2b44a0e345bc52039c2`.
These v1 artifacts are preserved. See the v2 repair below.

## Explicit proposal provenance migration and v2 startup preflight

`prior_reuse_proposal_provenance.py` creates a separate optimized proposal
artifact. It preserves every proposal component, mean, scale, weight, phase
parameter, coordinate name, fit rule, pilot seed and other original pilot
metadata. It changes only the declared native coordinate implementation hash
from `519c9cd4930a2a0a22d9989bad55b8dc436fe16760f8ed6d5dff3d8160047209`
to `2305ce98548f807b6acba8f1d4869a47f619639d80cae5fd0bcf76eda4bfabd8`
and appends an explicit implementation-provenance migration record. That record
binds the original proposal, both complete source manifests, all three known
runtime changes, the successful native replay and hashes of the unchanged
numerical proposal and pilot metadata. No fit or empirical correction is made.
The original artifact is never edited.

The public `FrozenParticleProposal.validate_production` remains unchanged.
It accepts the derived artifact against its new declared implementation and
continues to reject the original artifact, unknown target hashes and reused
pilot seeds. Geometry remains the original byte-identical artifact: its public
validator checks independent seeds and observation order, and has no native
source-hash gate. Both geometry and proposal checks run in the new preflight.

Preparation now executes the actual frozen `check_galsbi_particles.py` startup
in isolated CPU subprocesses for both arms and all four seeds. An explicit
sentinel at the first `torch.set_num_threads` call stops execution after public
geometry/proposal validation and initial protocol saving, before model loading,
CUDA initialization or particle draws. CUDA visibility is empty. The optimized
source must also reject the original proposal in a negative control. The
captured protocol and exact driver arguments are checked against the frozen
recipe; the completed receipt and every command/worker/protocol artifact are
pinned and revalidated before a GPU run. This preflight does not claim to test
GPU numerical results, which still require all four complete paired replays.

The v2 declaration uses `schema_version=2`, retains `initial_proposal` as the
original artifact for comparison, and adds explicit `initial_proposals` by arm,
`proposal_migration`, `startup_preflight` and `previous_failed_startup` records.
The optimized raw production protocol records the derived artifact's own
path, hash and full metadata. Comparison validates that entire artifact and
raw protocol first, then maps only the proven provenance differences to the
original artifact for scientific comparison. No shared comparison exclusions
or public target-identity checks are relaxed.

A new preparation/output directory is required. Run preparation in a CPU
scheduler allocation before requesting a GPU:

```bash
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python \
  scripts/prepare_prior_reuse_smc.py \
  --root /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021 \
  --destination /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/prior_reuse_smc_prepared_v2_20261022
```

The existing frozen launcher interface is unchanged. It now writes to
`prior_reuse_smc_runs_v2_20261022`, rerunning all four original seeds
2026103900–2026103903 with the original alternating order and exact science.
The original optimized arm failed before sampling, so this is a startup repair
without outcome-based seed selection. Every v2 pair must complete and replay;
the successful v1 baseline is not substituted for a v2 population.

Preparation time includes the new CPU startup preflight, its evidence checks
and source/proposal copies. Final declaration/manifest serialization and
permission writes retain their stated exclusion. The failed v1 process cost
is pinned and reported separately and is excluded from successful paired speed
ratios. No failed attempt is removed or classified as a successful comparison.

Focused validation covers byte-identical proposal draws, log densities and RNG
states; strict pilot/native identity; numerical/metadata/evidence tampering;
raw protocol provenance; geometry seed/order checks; preservation of the failed
attempt; and an integration test executing real `driver.main` through the exact
original rejection and derived-artifact pre-model checkpoint. The public
validators and all three runtime candidate files are unchanged by this repair.
Full scheduled tests, actual-data CPU preflight and fresh v2 GPU replay remain
pending. No runtime speedup or posterior-accuracy acceptance is claimed.
