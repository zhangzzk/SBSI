# Conditional categorical-exchange integration candidate

## Scope and promotion boundary

This isolated candidate starts from root commit
`fefd702476251c5b620a29967197b9cfa21df2bf`. Its implementation commit is
`413fe75bc2d5c7af619617df10827dba5f535684`. It copies the exact previously
validated categorical implementation from `1c0e0c6`, without algorithm changes:

| File | SHA256 | Change |
| --- | --- | --- |
| sbsi/scene_exchange.py | 738df16601bbebbabc15ac7f15e4c2f47255919a7cc0e02e8c0e1df0b5d4508e | Exact optimized implementation |
| tests/test_exchange_categorical_replay.py | 60d1e1d798be575d160f893b534e7816c5de1937809a95f917941767e019cb58 | Existing66-case replay coverage |
| doc/evidence/scene_exchange_baseline_55f1220.py | c5102579e3dfd1cd3c3e9ae5a267cf8ac0df9a6a82d60417fd2772e56aebd5ad | Already tracked; unchanged |

Only temporary target-type categorical draws are batched. The corrected native
support check, neighbour selection, ragged sums, relocation, density corrections
and random draw order retain the validated implementation. No scale, fit,
scientific threshold, model artifact or likelihood batch changes are introduced.

**This is not a merge or promotion.** Promotion remains conditional on all four
saved whole-run-output replays, their complete cost report, the held-out outcome
and the parent's independent review. The first completed benchmark pair alone
does not satisfy that gate. This task reads no held-out scientific outcomes and
does not change the running benchmark, held-out jobs, priority restoration,
frozen scientific sources or harnesses.

## Complete frozen validation source

The source is a complete `git archive` of implementation commit413fe75: all453
tracked files, plus a source receipt. It includes pyproject.toml, all examples,
models, tests, documentation and scripts, notably
`examples/data/example_catalog.feather`, which the packaging/resource tests
require. Root's unrelated untracked/dirty files are neither copied nor edited.
The baseline fixture and imported test helpers (`test_galsbi_prior.py`,
`test_galsbi_particle_target.py`, `test_informed_environment.py`) are already
tracked and byte-identical to the earlier categorical test dependencies.

The read-only snapshot is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/categorical_integration_source_20261022`.
Its454-file manifest SHA256 is
`2d34a06931edde2a6293373acf4271028c5b9f1f5a53f56ff5565cb8a8dcabcd`.
The committed [complete inventory](evidence/categorical_integration_source_20261022.sha256)
and [validation receipt](evidence/categorical_integration_validation_20261022.json)
pin the files, scheduler command and result. CPU17440535 uses two CPUs and no
GPU, checks every source hash before and after the full suite, explicitly
imports SBSI from the frozen snapshot, and keeps bytecode outside the source.

## Validation result

CPU17440535 completed successfully in137 allocated seconds: **1084 passed,
9 skipped**, one existing NumPy deprecation warning, in93.48 test seconds.
All66 categorical replay cases ran and passed. All454 source hashes matched
before and after. The skips are one unavailable BlendEMU catalogue-conversion
cross-check, seven optional frozen legacy-instrumentation checks, and one optional
pre-extension default-target replay; exact names/reasons are in the receipt.
The corrected scalar baseline used by the categorical tests was present and
tested. The test command is `pytest tests -q -p no:cacheprovider`, with the
configured py31 interpreter and existing external artifact roots. Passing this
suite validates this integration candidate's implementation; it does not
establish sampler convergence or full-run speedup.
