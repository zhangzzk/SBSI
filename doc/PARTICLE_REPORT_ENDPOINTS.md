# Particle report proposal provenance and endpoint compatibility

## Report version and independent groups

`independent-populations-v5-explicit-endpoint` retains the existing source,
observation, likelihood-batch, completion, extinction and independent-seed checks.
Within a replicate group, the informed-environment fingerprint includes its
artifact hash and every configuration/provenance field except the artifact's
filesystem path. Thus a copied identical artifact is compatible, while a changed
mixture weight, position law, proposal, source metadata or validation bridge is
not. Target initial proposals and physical geometry keep their existing checks.

Informed-environment pilot seeds participate in both the group's own production
seed exclusion and both directions of cross-group exclusion. Effective and
source pilot metadata must agree when source metadata is present; omitted pilot
provenance is rejected. This prevents describing populations as independent of
the pilots used to build either group's proposals.

## Comparing distinct annealing bridges

The supported explicit endpoint is exactly
`q_environment(E) * p_analytic(x) * K(y|x,E)`.
Each newly written protocol may declare it as `endpoint_target`. Within one
replicate group that declaration must agree, just as the literal bridge must.

Across groups, different literal bridges are accepted only if both reports
declare that same supported endpoint and have identical hashes for all existing
`TARGET_SOURCES`. Reference/observation hashes, numerical density batch, shear,
selection and report version must still match. The comparison receipt records
both bridges and whether compatibility used their literal equality or the
explicit endpoint. There is no ignore/allow override.

For protocols without an endpoint declaration, the existing literal-bridge
equality remains required. An explicitly conflicting endpoint is rejected even
when the bridge strings agree. A report from the previous version is rejected;
this change does not authorize automatic comparison to historical v5/v6 source
snapshots. Declaring an endpoint also does not bypass source identity checks.

For the reviewed informed proposal, the initial environment is qEnv times R
and the bridge contains R raised to 1−beta, so it disappears at beta=1.
The original-N first-stage weighting and corrected mutation ratios are reviewed
in [the informed-environment audit](INFORMED_ENVIRONMENT_REVIEW.md#mathematical-and-payload-review).
This makes normalizing masses comparable across these normalized proposal paths;
the normalizer-agreement test remains mandatory and can still fail even if
posterior score estimates agree. Endpoint metadata records this reviewed
identity; it is not a numerical proof for arbitrary implementations.

## Scope

The change strengthens reporting provenance and permits one explicitly reviewed
same-endpoint comparison. It changes no sampler, target density, run weights,
uncertainty calculation or scientific acceptance threshold. It supplies no
convergence, efficiency or shear-calibration evidence by itself.

## Validation

Focused CPU Slurm17420642 passed68 report, population-diagnostic and extinction
tests. Final Slurm17420682 ran the isolated checkout's full suite:
**495 passed, 3 skipped in60.88s**, COMPLETED0:0. Final source hashes were verified
before and after execution; the pinned receipt is
`doc/evidence/particle_report_endpoint_validation_20261022.json`.

Regressions cover copied artifacts, changed full environment configuration,
missing/conflicting pilot provenance, self/cross pilot-seed overlap, valid
cross-bridge comparison, missing/conflicting endpoint declarations, altered
target code, old report versions and normalizer disagreement despite equal
score means. Existing missing/failed-run, all-extinct and mixed-extinction tests
also pass. The three skips concern optional model/emulator integration; this
reporting change runs no GPU inference.
