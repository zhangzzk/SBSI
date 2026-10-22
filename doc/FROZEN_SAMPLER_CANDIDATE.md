# Complete frozen sampler candidate

## Status and scope

Branch `codex/frozen-sampler-candidate-20261022` preserves every file in the
320-file source manifest used by the current corrected held-out experiment.
It is a reproducible candidate, not an accepted sampler. The original held-out
accuracy and inclusive-cost gates remain pending. No running source, model,
seed, fit, threshold or scheduler setting changes.

The branch starts from `b7f559a437ad1667bd4efc403d021987114377ce`. An audit found
311 frozen files already identical in that commit, two older scripts with
uncommitted changes, and seven frozen files absent from Git. The isolated branch
copies these nine files exactly from the verified frozen source, preserving the
original worktree's dirty and untracked files.

| Frozen files restored | Role |
| --- | --- |
| `scripts/check_galsbi_local_sampler.py`, `sbsi/gradient_proposals.py`, `tests/test_gradient_proposals.py` | Earlier local-sampler proposals and their tests |
| `scripts/prepare_galsbi_twin.py`, `sbsi/mock_population.py` | Mock parent-coverage accounting |
| `sbsi/inference_moments.py`, `scripts/report_galsbi_direct_scores.py`, `scripts/report_galsbi_selection_covariance.py`, `tests/test_inference_moments.py` | Earlier score, curvature and selection diagnostics |

The active particle sampler and held-out runner were already byte-identical.
The scalar exchange implementation remains unchanged; the inactive categorical
optimization is not included. Post-freeze prior-sensitivity and benchmark tools
already committed in the base remain present but are outside the frozen recipe.
Their names are recorded in the source audit.

## Source and external dependencies

The verified source is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/source_corrected_heldout_v3_20261022`.
Its manifest SHA256 is
`fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1`.
The [320-file inventory](evidence/frozen_sampler_candidate_source_20261022.sha256)
and [source audit](evidence/frozen_sampler_candidate_source_20261022.json) pin
the exact files and all nine restoration identities. Every actual frozen file
was checked before copying, and every candidate path was checked afterward.

This is a complete source match, not a self-contained runtime image. The
experiment declaration still pins the reference data, model/native-prior inputs,
installed packages and other external resources. See the
[held-out launch](CORRECTED_HELDOUT_RUNNER.md#current-launch) and its immutable
source/dependency declaration. The existing scientific run continues to use its
original read-only snapshot.

## Validation

Implementation commit `862db4920fc8ad5997bc737f58332ae189cf31d9` contains all
320 frozen paths exactly, verified directly against committed Git blobs.
CPU job 17443057 runs a complete archive of that commit: 469 tracked files plus
a source receipt, including packaging and example-data fixtures. All 470
snapshot hashes match before and after, and parent review repeats that check.

**1,031 tests pass, nine optional checks skip**, with one existing NumPy
deprecation warning, in 83.45 test seconds and 108 allocated seconds on two CPUs
with no GPU. The skips concern the unavailable BlendEMU catalogue converter
and optional historical frozen-baseline checks; exact names/reasons are in the
[validation receipt](evidence/frozen_sampler_candidate_validation_20261022.json).
The [complete test-source inventory](evidence/frozen_sampler_candidate_test_snapshot_20261022.sha256)
and [launcher](evidence/frozen_sampler_candidate_tests_20261022.sh) preserve the
test command, interpreter, environment and source identity. Scientific promotion
remains conditional on the unchanged held-out report regardless of these tests.
