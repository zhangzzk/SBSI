# V3.9 actual ConstGold20k bright-only BFGS diagnostic

## Status on September 27: stopped at the first solve

Workers16726749[0–3] finished all20,000 observation evaluations and all20
full-prior normalization shards. At01:37 CEST, after52m15s, worker0's
combiner rejected the first-pass observed information matrix: eigenvalues
(-2,387,169.1061,3,445,424.7863). The other workers then stopped at their
barrier. This is the predeclared nonpositive-curvature stop, not a GPU OOM
or missing observation partition. There is no valid initial shear estimate
and none of the five requested bright updates ran.

All four5000-row NPZs, twenty1000-row checkpoints, normalization and input
receipts remain intact. No observations were removed, no solver repair or
truth-based starting point was applied. A read-only saved-moment audit
examines magnitude-bin contributions, the complete draw ladder and
normalization derivatives; its receipts are retained at the run root.

CPU audit16728966 completed successfully in28s. Input/worker hashes,
20k-row coverage, finite moments, and exact equality between the maximum
draw rung and primary arrays all passed. The matrix is indefinite at every
draw rung512,1024,2048,4096,8192,16384. Its minimum eigenvalues at8k/16k
are-2.449385m/-2.387169m. Removing only the normalization term algebraically
still gives-2.360167m, so selection normalization is not the principal source.

Along the16k negative eigenvector,135 measured-mag<20 rows contribute
-2.239340m (93.8% of the net negative curvature);774 mag20–22 rows contribute
-0.216479m. The two fainter bins contribute+0.034884m and+0.033766m.
Influential-row omission is recorded only as a diagnostic; no observation
is actually removed from the inference target.

The bright subset's estimator-relative weight-error diagnostic has median
0.09066 and90th percentile0.68966. These are the sampler's zero-view
likelihood diagnostics, not errors on shear or curvature. Its ESS counts
effective sampled draws, not distinct posterior atoms, and is not directly
comparable across stratification schemes. Reusing the same proposal across
the draw ladder does not establish independent convergence. A likelihood
profile and targeted bright-row sampling checks are needed to distinguish
local non-concavity from numerical integration error before restarting.
No new optimizer or GPU continuation was launched during this status check.
The owner subsequently requested [mocks from the exact matched inputs](V39_MATCHED_INPUT_MOCK.md#owner-request-and-generation).
Their reconstruction also establishes that9979 actual inputs belong to the
simulation target role used in V3.9 development, while10021 lie outside it.
All20000 remain in the requested cohort; this is a population distinction
to investigate, not an additional cut or demonstrated explanation.

## Request and sample

On 2026-09-27 the owner requested the current20k inference procedure on actual
image measurements instead of model-generated mocks. This run uses the first
20,000 rows of the existing randomly ordered uniform500k ConstGold plus-leg
sample (source seed20260914, cases40–139, injected shear(0.02,0)). The adapter
used `rng.choice(..., replace=False)` without sorting the sampled indices;
the prefix therefore remains a uniform selected sample. The prefix rule was
fixed before evaluating the new observations.

Selection is measured radius>0.6arcsec (3pixels) and MAG_AUTO<25.8, plus
finite positive radius/flux and measured shape inside the unit disk. Zero
shape remains usable. There are no cuts on true magnitude, size, shape or
neighbours. Input preparation rejects inconsistent rows rather than dropping
them or drawing replacements. Source case/input/detection IDs are retained.

The parent is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1/input`.
Its V3.6 model hash records the historical adapter's measurement contract;
these are image measurements, not samples generated from that model. The new
inference uses pinned V3.9 F8/C8/original disk artifacts. No old17-input
feature store is relabelled or used as V3.9.

Parent output SHA256:

- measurements.parquet: `f6c2e0bb8d2dbde054256f09c766fea0daab74b6cd50430c9bcd676568c0c53d`
- truth.parquet: `d8ec9e0f5f6eff581907f64a686bf3b8f62badb41f7c6f3b26f5f9c92af80125`

The original matching audit reports322,297 shapes without crossmatches,
zero crossmatches without shapes, zero matched rows without truth, and2,301
invalid/unusable matches across100 cases. It selected18,157,963 rows before
the uniform500k sample. These are parent-level counts, not new20k drops.
The new input manifest binds the parent manifest and both output files,
records case counts, and requires zero additional drops.

## Numerical procedure

The prior and models are exactly those of the
[bright20 mock run](V39_BRIGHT20_BFGS.md#owner-instruction-and-starting-point):
20,231,221 pre-extension atoms, excluding the newly supplied7,780,811 bright
atoms. Historical231,221 enrichment remains in that restored prior.

Pass00 is a fresh full20k likelihood and selection-normalization evaluation
at the actual sample's raw mean measured shape. Its observed-information
step establishes the first full-pass estimate. Passes01–05 perform exactly
five historical BFGS updates, starting at that estimate. Only measured
magnitude<20 observations have their likelihood moments replaced. Every
other observation is carried from the fixed original pass00 numerator;
fresh normalization derivatives replace the old derivatives on all20k rows.
No mock estimate or mock likelihood moment initializes this chain.

Each centre uses full-prior selection normalization with64 CRN QMC draws
per atom, h=.001 and the existing seed8101 shard derivation. Atom proposals
retain the location131072 prefilter, Gaussian top1024, tilted-stratified
delta0.1, seed8701 and draw ladder512–16384. The original four-observation
density groups are preserved, including numerical padding around bright
observations. Padding rows do not replace carried faint moments.

Retrieval/draw batches of24 are allowed only after exact replay on actual
observations against the original four-row implementation. The replay uses
the existing cached reference centre purely as a numerical test; those
scores are excluded from the scientific estimates. Production normalizers
are recomputed at this actual sample's own centres.

For the existing precision-aware density path, input preparation derives
log(1-|e|²) from the recorded strictly interior Cartesian measured shapes.
This does not recover hidden boundary precision or change the measurement.
Boundary, nonfinite and inconsistent encodings are rejected; no clipping.

Nonpositive first-pass information, invalid BFGS secant curvature or a step
norm exceeding0.05 stops the diagnostic for inspection. There is no truth
centering, truth-based stopping, Hessian repair or empirical calibration.

## Execution and validation

Frozen root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_v39_constgold20k_bright20_bfgs_20260927_v1`.

- CPU initialization/input tests:16726746.
- Full repository test suite:16726747.
- Actual-observation numerical replay:16726748, after initialization and
  completion of the earlier four mock workers16721029.
- Four persistent V100 workers:16726749[0–3], after replay and full tests.

The run stays within four concurrent GPUs. A fresh full pass plus five
bright updates is expected to take roughly three hours after GPU startup,
subject to actual throughput. Bright-only updates still refresh the full
prior normalization. Source/runtime/model hashes, selected rows, normalizer
hashes and worker coverage are checked and retained. Per1000-row full-pass
checkpoints make progress and completed work inspectable.

Added `scripts/v39_image_input.py` and its tests for sample/provenance/cut
validation. Extended `scripts/run_v39_bright_bfgs.py` with image input,
fresh full-pass initialization and five fresh updates; the frozen mock
runner is unchanged. Syntax and Ruff checks passed before submission.
Scheduled test/startup outcomes are recorded in the worklog and root receipts.

Initialization16726746 passed38 focused tests in24.26s and validated all20k
rows without drops. All100 source cases are represented (169–244 rows per
case). There are135 measured-mag<20 observations;536 original-group rows
will be evaluated, with401 padding rows excluded from moment replacement.
The initial raw mean is approximately(0.00983606,-0.00029220). Full-suite
job16726747 passed316 tests with one skipped in44.53s. Replay16726748
completed in5m05s, exactly matching all ten retained arrays on248 actual
observations, including bright groups. All four production workers
16726749[0–3] subsequently started on th-cl-nv01 and emitted their first
normalization masses for pass00 without failures. CPU status audit16726751
is scheduled about two hours later; it saves a snapshot and does not wake
an agent or notify the owner.

## Interpretation limits

These cases overlap model development. A single positive shear cannot
separate multiplicative and additive bias. Conditional row errors omit
case correlations, uncertainty of the fitted BFGS matrix and the faint-carry
approximation. Sampler and normalization convergence are not established.
Iteration stability is reported separately from paired8k/16k draw stability.
This diagnostic cannot establish robust0.3% calibration or independent
acceptance by itself; all results are retained regardless of their sign.
