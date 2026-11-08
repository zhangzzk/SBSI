# Full common-population paired-seed response repeat

## Purpose

Test whether the measured R11−R22 gap survives a precise repeat with the
original stochastic measurement rule and optimizer seeds paired across the
four shear exposures. The small fixed-image replay establishes seed sensitivity
but has insufficient precision to explain the full-population gap. This is a
focused measurement diagnostic, not a restart of broad model evaluation.

## Fixed scientific settings

Use all5,412,739 original central-cut IDs usable/selected in all four exposures
of the40 existing scenes. Preserve the65-pixel centred stamps, detected positions,
original image/PSF, original weight estimator, prior, default LM convergence and
failure retry rule. No rerendering, flow/classifier/emulator calls or cut changes.
Seed=(2026110700+input_id*17+case*100003) modulo(2^32−1), shared across all four
legs within one worker. This is the same seed-A recipe as the small pilot and
is reused-data evidence, not an independent acceptance sample.

Original per-leg photometry remains fixed. Record new shapes, returned status
flags and exceptions for every original ID. For the main paired contrast,
retain IDs valid in all four refits and recompute the original response on
exactly those same IDs. Report all losses and the original full-common matrix
separately. Do not drop an outlier or choose a seed to reduce the gap. Pool
numerators/masses before response and use paired40-scene jackknife errors.
No success claim until the completed response and paired uncertainty are read.

## Gates and packed execution

Preparation verifies the exact common-ID hashes from the previous catalogue
audit and saves immutable per-case positions and old shapes. Verify all40
prepared data hashes and replay the prior measured matrix before the64-ID
pilot. Require an exact within-job repeat of a successful fit and at least98%
valid pilot fits. The gate does not establish cross-architecture reproducibility.

After the pilot, four persistent CPU allocations each use32 workers and96GB
(up to128CPUs total; no GPUs), processing10 cases apiece in2048-ID chunks.
One chunk owns all four exposures for its IDs, avoiding cross-worker RNG
comparisons. Reuse mapped images and prepared case arrays per worker. Store a
per-chunk checkpoint, IDs, source pins and output checksum. Existing completed
chunks are checked and skipped on restart. Per-job time limit is4hours.
The report depends on successful completion of the entire array and requires
exact one-time ID coverage. This respects the500-CPU study ceiling and keeps
broad array17788286 held. No automatic follow-up timers are enabled.

## Interpretation

If the diagonal difference persists with reduced measurement noise, proceed
to controlled orientation/noise/neighbour tests of the expected response. If
it weakens, quantify the paired change before attributing the old gap to
optimizer sampling. Either result is informative; neither permits empirical
matrix symmetrization or declares the inference calibration repaired.

## Owner CPU expansion, 2026-11-08

The owner explicitly requests more CPUs, up to500. Stop only the two active
recovery workers and preserve1,844 completed chunks. An immutable disjoint
schedule assigns the823 unfinished2048-ID chunks to eight persistent allocations
of60 CPUs each (480 total), leaving room below the500-CPU ceiling for preparation
and reporting. Keep chunks from a case together where practical for image reuse.
Array17824404 replaces17824227; final report17821860 depends on the new array.
No completed output is deleted, and no orphan files need quarantine.

The new BlendEMU scheduling wrapper imports and calls the original pinned
`task` function unchanged. No numerical batch, seed, fitting, selection or model
setting changes. Original pilot gate and per-chunk exact-repeat checks remain
required. Syntax, complete/disjoint schedule coverage and original source pins
pass before release. Fast-I/O x86-64-v3 constraint and exclusion of the two slow
Naples hosts remain. Broad17788286 stays held; automated follow-ups remain off.

### Queue-aware allocation sizes

The scheduler estimated a two-hour wait for seven60-CPU allocations. Cancel
only those queued allocations, preserving the running60-CPU group0. Submit
the same immutable group1–7 tasks as seven32-CPU allocations (17824424), which
all start promptly. Actual concurrency is284 CPUs, with the owner500-CPU limit
retained for this study. No task assignment, seed or measurement code changes.
Final report depends on17824404_0 and17824424. The schedule JSON retains the
original480-CPU request as historical metadata; actual resources are recorded
in each completion receipt and `packing_startup_20261108.json`.
