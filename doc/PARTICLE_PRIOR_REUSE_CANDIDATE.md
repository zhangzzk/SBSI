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

This candidate is not adopted. The parent executor must run the full CPU suite
through Slurm and compare independent baseline/optimized saved GPU results on
the same native data, models, settings, seeds and original observation groups.
Actual installed photometry must replay exactly: reusing a whole decoded
photometry array replaces evaluations on type-specific subsets, and toy
elementwise photometry does not establish that numerical property for the
external implementation. Support-boundary agreement must be included.

Complete-process matched timing is required before any efficiency claim.
Removed call counts and earlier profiles do not establish a speedup. The
candidate neither certifies sampler accuracy nor changes scientific acceptance.
