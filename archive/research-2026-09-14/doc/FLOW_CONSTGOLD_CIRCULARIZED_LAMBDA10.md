# ConstGold 50-case evaluation of circularized grid-free lambda 10

**Latest completed combination (2026-09-11):** lambda10 epoch154 plus the
newer ordinary-MSE trial9 emulator gives pure-shape m = +0.2338% +/- 0.2238
percentage points, measured-selection m = -0.7160% +/- 0.2975, and full
detection-plus-selection m = -2.1112% +/- 0.3081. See
[the completed trial9 results](#completed-lambda10-plus-trial9-results--2026-09-11).

The subsequent [detection diagnosis](CONSTGOLD_DETECTION_DIAGNOSIS.md#completed-flow-controls--ten-systematic-cases)
finds both population-weighting errors and a poorly represented single-leg
detection boundary. Correcting detector weights alone does not eliminate the
selected-response residual in the controlled subset.
The initial sections describe the older weighted-emulator run preserved as
a control; [the provenance correction](#newer-trial9-emulator-correction--2026-09-11)
explains why that run did not use the newer emulator.

## User authorization and fixed checkpoint

On 2026-09-11 the owner requested the lambda 10 model on 50 ConstGold cases,
reporting m for pure shape and shape plus selection. This subsequently
authorizes ConstGold evaluation of the completed experimental model; its
earlier training trial did not itself include ConstGold. No retraining,
checkpoint selection using ConstGold, or configured likelihood promotion is
part of this evaluation.

Use the already selected epoch 154 from
`physical_circularized_grid_free_shape_radius_lambda10_v2/paired/epoch154.pt`,
SHA256 `9c5bb028437a4714b454d2bcd202243a2eda6263c36b4c70c46de3d12f725a6f`.
The external output is its sibling directory `constgold50_v1` within the
same experimental root. Snapshot source, checkpoint, input provenance and
commands before submission; validate the frozen source under Slurm.

## Population, response and m

Use cases 40–89 inclusive, the same 50 cases as the previous response
transfer tests. Preserve the complete truth support defined by the registered
Re > 0.37 lookups: strict true 18 < magnitude < 25.8 and 0.37 < semi-major
Re < 1.5. Preserve the full surrounding scene for neighbour features and
the existing physical R_blend shear contribution. Domain verification uses
the original NLL ancestor's source-cut metadata and the matching registered
emulator metadata; the new checkpoint's inherited training population and
condition/physical transforms are verified separately.

ConstGold has +0.02 and -0.02 g1 legs. Both measured and predicted responses
use (mean_plus - mean_minus)/0.04, following
[CONVENTIONS section 6c](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).
Only R11 and R21 are identified. Pool each leg's sums and denominators
across cases before differencing; these are population-weighted responses,
not the 180 equal-cell validation means.

The requested comparisons are:

- Pure shape: fixed usable identities detected in both legs, no measured
  cuts and no classifier weights. The earlier exact sample has 4,338,426
  matched identities; assert reproduction.
- Shape plus measured selection: those same fixed both-detected identities,
  with measured cuts applied independently in each leg. Predictions select
  their own draws. The detector is conditioned out.
- Retain the complete shape + detection + measured-selection result as an
  explicitly named additional diagnostic, using the existing transition
  classifier, impact neighbour ranking and exponent 1.

Measured cuts are |e| < 0.6, MAG_AUTO < 25.8, and flux radius >= 0.75 arcsec
(3.75 pixels). Use 64 Sobol/QMC draws per object, sampling seed 7301,
batch size 2048, and common draws across shear legs. Compute
m = 100*(R11_measured/R11_model - 1) percent, and form that ratio within
each of 10,000 paired case bootstrap replicates, seed 20260902. This is a
response-closure diagnostic; it is not a catalogue-prior inference run.
One trained checkpoint and one integration seed do not quantify retraining
or independent integration uncertainty. These are previously used ConstGold
cases, not an untouched final holdout.

## Circularized inputs and numerical physical tails

The evaluator now supplies `circularized_Re_input_p` when requested by the
checkpoint. It reconstructs q0 from aligned pre-shear truth ellipticity,
matching training, then supplies a0*sqrt(q0) unchanged in both legs. Sheared
ellipticity and all other context fields are preserved. An identity mismatch
fails; no unsheared size is combined with post-shear axis ratio.

The physical flow's audited softplus map occasionally underflows to a
numerical zero radius or flux. The evaluator records those events and
retains their shape samples in uncut averages. For measured selection,
zero radius maps to log radius -infinity and zero flux to magnitude +infinity;
the corresponding finite cuts reject them. No sample is floored, removed
from its parent denominator, or redrawn. Negative/nonfinite physical outputs
remain errors. This evaluates the existing numerical model without changing
its tail distribution or selection boundaries.

Per-case measured and per-model predicted numerators/denominators and physical
tail counts are saved to reproduce m and its uncertainty. Regression tests
cover baseline-size invariance and identity alignment, zero-tail retention
under each cut combination, invalid physical draws and existing physical
versus engineered selection equivalence. The initial focused suite passed
21 tests in 21.54 s. Full validation precedes the GPU evaluation. Wake-ups
remain off.

## Submission and validation

The frozen source contains 304 files/resources. Manifest SHA256 is
`82f0ad1e62deb4d18fe921ade012f87d61a1e178f08aa87908d6e498e4fbd962`.
Validation 16455738 completed with 448 tests passed and 2 skipped in 77.43 s,
then passed the checkpoint/source preflight. Evaluation 16455739 started at
19:06:36 Berlin time on th-cl-nv02 with one V100, 12 CPUs and 96 GB RAM.
The support audit passed all 50 cases and 4,854,422 truth objects. The first
completed cases establish that the circularized condition and physical
sampling path execute successfully. External submission, checkpoint,
domain and startup records preserve those checks. A separate saved-sum
review will independently verify the completed m calculation.

## Completed responses and independent review — 2026-09-11

Evaluation 16455739 completed exit 0:0 at 19:39:51 Berlin time after
33 minutes 15 seconds, with empty stderr and all 50 cases present. The
completion record hashes both the full result and the scored comparisons.
The separate single-thread `constgold50_v1/review_result.py` recomputed
population-weighted responses and all 10,000 paired case-bootstrap ratios
directly from saved per-case numerators and denominators. Point estimates,
standard errors and percentile intervals agree to 1e-10 absolute tolerance;
checkpoint/output hashes and the exact pure-shape identity count pass.
It writes `constgold50_v1/review.json` and prints
`INDEPENDENT_CONSTGOLD_REVIEW_PASSED`.

All uncertainties below are one case-bootstrap standard error. The m
uncertainties are percentage points; response uncertainties are dimensionless.

| Comparison | Measured R11 | Model R11 | m (%) | 95% case interval for m (%) |
| --- | ---: | ---: | ---: | ---: |
| Pure shape | 0.960028 ± 0.002117 | 0.986156 ± 0.000282 | -2.6495 ± 0.2159 | [-3.0605, -2.2193] |
| Shape + measured selection | 0.611183 ± 0.001836 | 0.623114 ± 0.000536 | -1.9147 ± 0.2923 | [-2.4883, -1.3485] |
| Shape + detection + measured selection | 0.605378 ± 0.001904 | 0.624866 ± 0.000512 | -3.1187 ± 0.3028 | [-3.7175, -2.5339] |

Pure shape has 4,338,426 identities in each leg. On that fixed parent,
measured selection retains 2,418,150 / 2,418,461 objects in the plus/minus
legs; model selection has expected counts 2,413,620.875 / 2,413,372.71875.
The complete population has 4,854,422 truth objects; observed detection and
selection retain 2,427,759 / 2,428,038, versus model expected counts
2,440,271.16085 / 2,439,904.79159.

Measured/model R21 are -0.00052793 / 0.00031677 for pure shape,
-0.00202839 / 0.00056346 with measured selection, and
-0.00196370 / 0.00056646 with detection and measured selection. The
corresponding paired residual case standard errors are 0.00224333,
0.00163061 and 0.00173598; none of these R21 residuals exceeds two such
standard errors. R12 and R22 remain unidentified.

The model overpredicts R11 in all three comparisons, leaving a negative
few-percent response-closure m. Improving the training radius response
therefore did not establish ConstGold shape-response calibration. This is
not a controlled ConstGold comparison against another lambda, nor a full
SBSI inference result. Differences from the earlier validation response
proxy involve a different population, pooling, shear extraction and the
physical blend response; they do not by themselves identify a cause.

Across 621,366,016 physical draws, 313 radii and zero fluxes underflowed to
exact zero; no nonfinite/negative draw triggered the rejection checks.
Largest radius/flux were 8,309.95694 pixels / 13,643,901.85184. Those tails
are retained as described above. Case uncertainty is conditional on this
checkpoint and the single QMC seed; independent integration and retraining
uncertainties were not measured. No further jobs, wake-ups or model
promotion are part of this completed evaluation.

## Blend provenance and domain comparison — 2026-09-11

The owner queried the opposite residual sign relative to validation. The
validation R11/R22 are self-response estimates: measured 0.85546743 / 0.86407222
and flow 0.84818248 / 0.84000232. ConstGold pure shape excludes measured
selection and classifier weights but includes the external neighbour term.
Its total response must not be described as the flow's own self-response.
Validation uses random-direction forward 0 to 0.05 shear and 180 equal-weight
cells; ConstGold uses coherent central g1 +/-0.02 and population weighting.
ConstGold alone does not identify its measured self-response separately.

This run reads the five `constgold/rblend_re037_complete_c40-49.feather`
through `c80-89.feather` lookups from the original experiment root. They were
built using `extnbr_reweighted_vector_re037_optuna30_v1/models/`
`regression_model_lsst_r_extnbr_re037.json`, SHA256
`167640dd0b876ed54721493c52f3b9fe9412d005b4973e7948566eb3c4001bde`.
Metadata specifies k=20 neighbours within 10 arcsec. The evaluator applies
`e_draw(g) += R_blend * (e_truth(g) - e_truth(0))` before measured selection.
Consequently a selected result is not corrected by adding the uncut mean
blend response after selection.

The model-independent blend component in the earlier
`response_composition_localization_v1/results.json` uses this exact shift.
Fresh review verifies identical emulator and all five lookup hashes, all
50 matched population counts and measured response vectors to 1e-10. Reusing
only that unchanged deterministic blend component gives:

| Pure-shape component | R11 | R21 |
| --- | ---: | ---: |
| Current flow alone, by subtraction | 0.81854843397 | 0.00026670786 |
| Applied blend contribution | 0.16760781001 | 0.00005005972 |
| Current total model | 0.98615624399 | 0.00031676758 |
| Measured coherent total | 0.96002785416 | -0.00052792969 |

Raw scalar mean R_blend is 0.16759687636 on the 4,338,426 fixed
both-detected identities, versus 0.21825277919 on all 4,854,422 truth-support
objects. The applied response differs slightly from that scalar mean because
the finite-shear intrinsic ellipticity change enters the shift. No older
flow predictions were transferred to the current result.

The completed source-domain audit verifies strict true magnitude
18 < r < 25.8 and semi-major 0.37 < Re < 1.5 arcsec against the flow's
documented source-population ancestor and the emulator metadata. Every
case's lookup identities exactly equal the truth identities passing these
bounds: no missing, extra or duplicate support keys. The support includes
1,667,167 objects with Re < 0.5 arcsec. The selected checkpoint's parent
transforms and non-size standardization were separately checked in preflight;
circularized conditioning preserves these semi-major population cuts.
This establishes primary-cut consistency, not equality of all conditional
feature distributions or of the two experiments' detected populations.

The already completed independent coherent-neighbour anchor test,
`anchorblend_current_re037_recovery_v1/results.json`, uses this same emulator
hash and primary bounds. On 2,613,893 anchors over 500 cases it reports
case-balanced measured/predicted responses 0.12659343878 / 0.15199711180,
with residual +0.02540367302 +/- 0.00288981615 (case SE). It demonstrates
overprediction on that anchor population; domain consistency is not evidence
of unbiased blend response. Its sparse-anchor population and weights differ
from ConstGold, so the residual is not a transferable correction or proof
that blend alone explains the current +0.02612839 total residual.

Saved the compact provenance, component arithmetic and verification record
as `constgold50_v1/blend_domain_review_v1.json`. This review used existing
completed audit artifacts and fresh small-file hashes; it did not rerun the
catalogue scan, change predictions, or submit new work.

## Newer trial9 emulator correction — 2026-09-11

The owner recalled a newer successful retraining. Worklog cont.536 records
the ordinary-MSE trial9 base candidate; cont.537 records its controlled
ConstGold swap and cont.539 its completed result. The earlier review missed
these later entries and called the weighted emulator current. The quoted
anchor excess +0.02540367 +/- 0.00288982 applies to the older weighted
SHA167640...1bde model, not to trial9.

The newer source is
`/project/ls-gruen/users/zekang.zhang/blendemu_runs/extnbr_mse_re037_optuna12_v1/trial_009/base.json`,
SHA256 `9723589880234fa6825ae242d9b978120a6385faf5ccc9942d1993696952e46d`.
It has 218 trees and was chosen by common ordinary development MSE among
retained base/refit artifacts, without using ConstGold. Development
predicted/measured pair-label means are 0.00790723 / 0.00803994 (relative
mean error -1.65%), versus +20.39% for the older weighted model. These
pair-label means are not direct coherent-neighbour response means.

The completed `constgold50_trial9_base_lambda05_epoch060_v1` test fixed the
older capped lambda0.5 epoch60 flow and all 4,338,426 matched identities.
Measured R11 is 0.96002785; old-weighted model total is 0.98119068, while
trial9 total is 0.95282297. Corresponding m values are -2.15685% and
+0.75616%. Trial9 residual -0.00720489 +/- 0.00211789 (paired case SE)
is much smaller but remains 3.40 standard errors from zero. This is evidence
of improvement, not perfect calibration and not a result for lambda10.

Fresh checks confirm source/copy/model-audit SHA equality, completed result
hash, all five new/old lookup hashes and saved exact-key equality checks.
The newer audit passes the same strict primary bounds, 18 < r < 25.8 and
0.37 < semi-major Re < 1.5 arcsec, with 4,854,422 truth-support objects.
Thus the overlooked difference is emulator version, not these primary cuts.

At this correction stage, lambda10 had not been reevaluated with trial9. In
particular selected results cannot be updated by subtracting a constant
blend mean because the shifted draws determine their own selection. Saved
`constgold50_v1/emulator_provenance_correction_v1.json`; original frozen
inputs and numerical results remain intact. No new model evaluation or
promotion was performed during this confirmation request.

## Lambda10 flow with trial9 emulator — 2026-09-11

The owner then requested the new flow with the newer emulator. Submitted
evaluation 16456455 at 20:03:45 Berlin time under
`physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_trial9_v1`.
It combines the same selected physical circularized epoch154 checkpoint
(SHA9c5bb0...a6f) with the verified ordinary-MSE trial9 emulator
(SHA972358...e46d). The previous `constgold50_v1` remains the older-emulator
control and retains all original artifacts.

The evaluator, installed environment, all 304 source hashes, flow checkpoint,
detector, cases40–89, primary bounds, measured cuts, QMC/CRN seed7301,
64 draws, and 10,000 paired case-bootstrap replicates are unchanged. Only the
emulator and keyed blend values change. Reuse the five completed trial9
lookup shards with their verified model metadata and exact old/new support-key
equality; pin their full hashes at worker start and completion. Model and
metadata copies are isolated in the new root. No lookup rebuilding is needed.

New manifest SHA256 is
`416cb795de6b632b521df058d2e0c44ea845b79211041c2ff2063095b9ebc6ce`.
The unchanged worker's checkpoint/source preflight passes, as do Python
compilation and shell syntax. The original frozen evaluator already passed
448 tests with 2 skipped; no scientific implementation changed for this swap.
The V100 allocation uses 12 CPUs, 96 GB RAM and a 90-minute cap. It repeats
the full domain audit and evaluates all three response/selection comparisons.

After scoring, the copied independent saved-sum review verifies each m and
its case uncertainty. A new `compare_emulators.py` computes paired changes
against the completed lambda10 older-emulator run, asserting unchanged
measured responses. It also checks the uncut response change against the
model-independent additive blend change from the earlier fixed-flow swap.
Selection is evaluated afresh from the shifted draws. Results are pending;
neither a prediction from the previous flow nor a job state establishes the
new combination's calibration. Wake-ups remain off, with startup monitored
under the current rule; no training or model promotion is included.

Startup review: job16456455 started at 20:04:07 Berlin on th-cl-nv01.
The fresh full support audit passed all 50 cases and 4,854,422 objects with
the intended trial9 hash. Case40 completed its full flow expectation and
case41 began; stderr is empty. Saved `startup_review.json`. The preceding
identical-size evaluation took 33 minutes, so this run is progressing normally
and is left running after startup monitoring. No wake-up is created because
the owner explicitly stopped them. Final m and comparison reviews remain
pending until the worker completes.

## Completed lambda10 plus trial9 results — 2026-09-11

Job16456455 completed exit0:0 at 20:35:17 Berlin after 31:10, with empty
stderr. All 50 cases, the exact domain audit and the independent saved-sum
bootstrap review completed. `PAIRED_EMULATOR_COMPARISON_PASSED` also verifies
unchanged measured responses and the additive pure-shape blend change.
Fresh result/model hashes agree with the completion records. Output root is
`physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_trial9_v1`;
`review.json`, `paired_emulator_comparison.json`, and `completion_review.json`
preserve the checks and comparisons.

These are population-weighted g1 central responses on cases40–89. All errors
are one standard error from 10,000 paired case-bootstrap replicates;
m uncertainties are percentage points. m = 100*(R11_measured/R11_model - 1).

| Comparison | Measured R11 | Predicted R11 | m (%) | 95% case interval for m (%) |
| --- | ---: | ---: | ---: | ---: |
| Pure shape | 0.960028 +/- 0.002117 | 0.957789 +/- 0.000287 | +0.2338 +/- 0.2238 | [-0.1956, +0.6805] |
| Shape + measured selection | 0.611183 +/- 0.001836 | 0.615590 +/- 0.000525 | -0.7160 +/- 0.2975 | [-1.3000, -0.1373] |
| Shape + detection + measured selection | 0.605378 +/- 0.001904 | 0.618435 +/- 0.000507 | -2.1112 +/- 0.3081 | [-2.7196, -1.5125] |

The fixed pure-shape parent contains 4,338,426 usable both-detected objects;
the same parent with independent measured cuts retains 2,418,150 / 2,418,461
objects in plus/minus legs. New model expected selected counts are
2,413,682.421875 / 2,413,442.828125. Complete support is 4,854,422 truth
objects with observed selected counts 2,427,759 / 2,428,038, versus model
expected counts 2,440,330.894294 / 2,439,962.395577. The primary and measured
cuts, shear, detector and sampling are unchanged from the earlier run.

Holding the new flow fixed, the emulator swap changes pure-shape m from
-2.6495% to +0.2338%, measured-selection m from -1.9147% to -0.7160%, and
complete m from -3.1187% to -2.1112%. Their paired changes are respectively
+2.883320 +/- 0.011290, +1.198746 +/- 0.014613, and
+1.007469 +/- 0.013172 percentage points. In pure shape, the new prediction
is 0.81854843397 flow + 0.13924009122 applied blend = 0.95778852520.
The blend change -0.02836771879 matches the earlier fixed-flow emulator swap
to better than 1e-14; no response offset or retuning was applied.

With trial9 held fixed, the earlier lambda0.5 epoch60 flow gave pure-shape
m = +0.75616%, versus +0.23380% for this new physical circularized lambda10
flow. This compares different flow recipes, not an isolated lambda change.

Pure-shape closure is consistent with zero at the measured case precision
(about one standard error). Measured selection leaves a roughly 2.4-standard-
error residual; full detection plus selection retains a clear roughly 2.1%
negative m. The complete and fixed-parent comparisons have different
populations, so their difference alone does not isolate classifier error.
R21 model values are 0.00031515, 0.00043146 and 0.00044706; measured values
are -0.00052793, -0.00202839 and -0.00196370, respectively. All three R21
residuals are within two paired case standard errors.

All 621,366,016 pre-shift physical draws have identical recorded extrema
and zero-tail counts to the earlier run: 313 zero radii and no zero fluxes,
retained without flooring or redrawing. Only g1 is excited; one flow and one
QMC seed do not estimate training or independent integration uncertainty.
These are previously used diagnostic cases and response-ratio closure,
not full SBSI inference or proof of calibration throughout the domain.
The requested evaluation and review are complete. No model was promoted,
and no wake-up or further job is required.
