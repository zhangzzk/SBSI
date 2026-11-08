# Circularized detector repair trial

**Completed:** all50 ConstGold cases and independent review passed. The detector
representation improves, but complete m remains about-2%. See
[completed comparison](#completed-full-constgold-comparison--2026-09-11).

The subsequent [separate shape/selection/detection tests](CONSTGOLD_SEPARATED_RESPONSE.md#completed-r11-comparisons)
use true sheared ellipticity to isolate measurement-selection and detector
diagnostics. Selection on fixed detected pairs gives m=+1.96824 +/-0.15421%;
this is a different observable from the complete measured-shape result below.

## Aggregate-crowding raw-D ensemble — 2026-09-13

The exact no-|e| residual audit subsequently localized91.5% of the
detector-composition residual to overlapping true-magnitude>=24 and
`nbr_flux_near>=0.5` cells. This is descriptive, not a new cut. It motivated
one minimal architecture change: add the flow's `nbr_flux_near`,
`nbr_flux_far`, and `nbr_flux_max` fields to this trial's valid circularized
raw-`D` detector. At fixed observing conditions, `r_input_p_scaled` differs
from the flow's true magnitude by a constant, and the circularized scaled
radius is a one-to-one transform of the flow radius. The resulting20-feature
detector condition therefore contains recoverable versions of all eight flow
conditions, plus the existing representative-neighbour fields.

External `raw_d_crowd_ensemble_v1` reuses the audited case0--39 preparations
but explicitly reads `d0/d1`, never the invalid target-half `u0/u1` fields.
Three fixed seeds20260913--20260915 use the same3x128 SiLU architecture and
optimizer recipe. They selected epochs80/79/78 at tune BCE
0.127074/0.127001/0.126924. All three checkpoints are included with equal
probability weight; neither the detector test response nor a coherent result
can change those weights.

On the reused but untouched-for-this-model detector test cases30--39, the
ensemble improves raw-`D` BCE from0.15847047 to0.12364730 (22.0%) and Brier
from0.04031612 to0.03252627 (19.3%). Its model-minus-actual rendered
true-shape response mismatch along the applied-shear direction falls only from
0.00100259 to0.00077436 (22.8%). In the endpoint-informed faint/crowded cell,
BCE improves0.189170->0.153194 but marginal probability-minus-frequency moves
from-0.000518 to-0.000932. Thus ordinary predictive accuracy clearly improves,
but there is no basis yet to claim the required69.2% reduction of the coherent
composition error.

Training jobs16468432, development16468433, and independent review16468434
all completed with empty scientific stderr; the only training stderr is a
PyTorch warning from converting the loss tensor for telemetry after the
backward pass. The independent reviewer reconstructs every global/cell score,
count and response from compact sufficient statistics. Result/sufficient/
review SHA256 are `3576752a...b508b`/`a2b3641d...a77d5`/
`82742b3d...c7ee2`. The recorded manifest time string was typed inaccurately;
`timestamp_correction.json` preserves the actual file and submission times
without changing the checkpoint-hashed manifest.

The frozen coherent evaluation subsequently completed all100 cases40--139
(9,709,701 objects) with lambda10 epoch154 and trial9 unchanged. The ensemble
gives no-|e| `m=-0.56808 +/-0.12551%`, versus
`-0.68150 +/-0.12559%` for the current detector and
`-0.12946 +/-0.12532%` for actual U. Its p-versus-U full composition response
error is+0.00444988, only20.64% smaller than the current+0.00560698 and far
short of the frozen69.2% requirement. Truth error improves from-0.00103813 to
+0.00005544, while the dominant residual remains+0.00439444. Individual-seed
m values-0.555/-0.577/-0.573% exclude an unlucky ensemble cancellation.

All case receipts and source/model hashes reproduce; archived actual/U/current
sums agree within4.58e-11. Primary reduction and a separately written reviewer
reconstruct every point, case-bootstrap interval and full=truth+residual
numerator identity (maximum5.61e-15). The first review job16468487 failed only
because its verifier inferred the nonexistent key `truth_path_sha256`; the
explicit receipt mapping repair is recorded in `review_repair.json`, and
replacement jobs16468593--16468594 passed. Result/sufficient/completed/review
SHA256 are `d102e811...a0eaf`/`a0301292...138b2`/
`8a7d9be9...bb2e4e`/`d6271d2f...a5dcd`.

Disposition: reject this ensemble for calibration. It establishes that the
remaining bottleneck is response-relevant conditional residual calibration,
not missing aggregate counts alone. The follow-on coherent-U flow-condition
experiment is documented in `CALIBRATION_STATUS.md`; no configured detector
or likelihood changes here.

## Coherent-U flow-condition follow-on — 2026-09-13

External `coherent_u_flow_condition_ensemble_v1` tested whether the preceding
failure came from the half-shear/raw-`D` training contract rather than the
probability representation. It used actual per-leg usable `U` from fully
targeted coherent +/-0.02 catalogues and exactly the eight lambda10 flow
conditions with its frozen preprocessor. Cases40--99 trained three3x128 SiLU
MLPs; cases100--119 selected each checkpoint by U-BCE alone; cases120--139 were
the response-blind, out-of-training mechanism screen. All three probabilities
were averaged equally. No response loss, probability correction, cut change,
seed choice, or ensemble-weight fit was allowed.

The mechanism fails. On1,938,865 test objects, no-|e| U-oracle/current-
detector/new-ensemble m is respectively
`-0.24327 +/-0.22017%`, `-0.82999 +/-0.22168%`, and
`-0.77791 +/-0.22196%`. The new full p-versus-U composition response error is
`+0.00543363`, versus current `+0.00596611`, only an8.92% reduction against the
prespecified69.2%. The truth component worsens from `-0.00012112` to
`+0.00044937`; the residual improves from `+0.00608723` to `+0.00498426` but
still contributes `-0.49162pp`. All three seed m values
(-0.768/-0.781/-0.784%) agree. Marginal calibration is close
(observed0.896956, predicted0.897135), showing again that a small global
probability gap does not control the response-weighted moment.

Training array16468635, case array16468636, review16468637, and independent
review16468638 all completed with empty stderr. Independent full=truth+
residual response identities are within1.10e-14. Result/sufficient/completed/
review SHA256 are `c5d31b17...3ddc8`/`31b1bf90...2632`/
`71a22887...e64`/`8393791f...a027b`. The independent +/-0.05 cases140--239
remain sealed because the gate failed.

This rejects the claim that valid coherent labels plus this eight-input MLP
solve the finite estimator. The eight variables make the unshifted flow moment
measurable, but the complete evaluated moment also contains the external
`R_blend * (e_g-e_0)` shift. They therefore do not by themselves establish the
tower identity for the complete moment. Empirically, the present marginal-BCE
MLP also does not reach the roughly1e-3 response-weighted precision required.
The historical next test combined coherent U with the already-validated
20-feature circularized representative-neighbour/crowding condition; its
result is recorded below.

## Coherent-U rich-context factorial follow-on — 2026-09-13

External `coherent_u_crowd_ensemble_v1` completed that missing combination.
It retains the exact circularized impact-a1/3-arcsec representative neighbour
and `nbr_flux_near/far/max`, but trains against coherent per-leg U. The split,
three seeds, BCE-only checkpoint rule, equal ensemble, forbidden response
optimization, and69.2% component-aware gate are unchanged from the preceding
experiment. Exact detector views for cases90--139 already existed; cases40--89
were staged once through the frozen scene helper. All100 supports align by
input ID; no row is dropped or substituted.

Rich context materially improves ordinary prediction. Seeds selected
epochs85/75/83 at tune BCE0.125105/0.125065/0.125079. On3,877,730 test-leg
rows the equal ensemble has BCE0.12370163, Brier0.03248269, observed U
fraction0.89695621 and predicted0.89702902. Nevertheless it fails the response
gate. On cases120--139, U/current/rich-ensemble no-|e| m is
`-0.24327 +/-0.22017%`, `-0.82999 +/-0.22168%`, and
`-0.66378 +/-0.22230%`. Full p-versus-U composition error falls from
`+0.00596611` to `+0.00426883`, a28.45% reduction. The residual improves from
`+0.00608723` to `+0.00355431`, but truth changes from `-0.00012112` to
`+0.00071451`, so both the total and component gates fail. Seed m values are
-0.663/-0.655/-0.673%; this is not a checkpoint accident.

Stage preflight16468681 and array16468682, replacement training16468740,
case array16468741, review16468742, and independent review16468743 completed.
The first training submission16468683 stopped before any model update because
the external root lacked its empty `models/` parent; its descendants were
canceled, and `training_setup_repair.json` records the unchanged-worker
resubmission. Successful scientific stderr is empty. Result/sufficient/
completed/review SHA256 are `97fbeebd...32dcb`/`addcc397...dc6f`/
`eea2550e...8bbb`/`b6878275...da3e`.

Disposition: reject the rich fixed-vector detector and keep cases140--239
sealed. Coherent labels add only a modest improvement over the same raw-D rich
context, while representative-neighbour detail gives a larger but still
insufficient improvement over the eight flow variables. The remaining
detector-only hypothesis is therefore specifically information in the full
neighbour set. Any test of it should preserve the fixed-vector path and add a
zero-initialized set correction; the old shared DeepSets measurement trunk's
response failure argues against replacing the primary path.

## Coherent-U full-scene residual follow-on — 2026-09-13

External `coherent_u_full_scene_residual_v1` completed the prespecified final
full-scene detector test. It freezes each matching-seed rich detector and
adds a zero-initialized permutation-invariant correction. Each neighbour
within10arcsec contributes distance, relative magnitude and circularized size,
Sersic index, and the primary/secondary sheared ellipticities projected into
that pair's frame. At most32 neighbours are retained by the same impact-a1
ranking. The set encoder is8->24->24 and its masked sum/sqrt(count), the frozen
40-dimensional primary input, and log count enter a128x2 correction. Epoch0 is
an eligible checkpoint and reproduces the base logits exactly. Only coherent-U
BCE on cases40--99/100--119 trains/selects the model; response is not an
objective.

The complete input fields supply9,709,701 primaries and164,255,670 directed
in-radius edges across cases40--139. All primary IDs match. The cap discards
6,651 edges (0.0040%) affecting3,481 primaries (0.0359%); one primary has no
10-arcsec neighbour. Thus truncation cannot plausibly explain the result.
The first successful30-epoch run was archived before response evaluation when
tune-only curves were still improving at the compute boundary. An explicitly
recorded convergence extension to100 epochs selected epochs100/99/99 at tune
BCE0.1009533/0.1012914/0.1007358. The exact epoch0 BCEs reproduce the frozen
base0.1251052/0.1250650/0.1250790.

On1,938,865 reused test objects, the ensemble has BCE0.09969245 and
Brier0.02641323, improvements of19.41% and18.69% over the rich fixed-vector
ensemble. Observed/predicted usable fractions are0.89695621/0.89685636. Its
no-|e| `m=-0.27936 +/-0.22353%`, compared with U/current/rich
-0.24327/-0.82999/-0.66378%. The full p-versus-U response difference falls
from current+0.00596611 to+0.00036492, a93.88% reduction. Seeds give
-0.27949/-0.28316/-0.27543%, so the total gain is not a seed accident. It
also tracks U closely under no cuts and all cuts: m differs by only-0.019 and
-0.036 percentage points, respectively.

Nevertheless the frozen mechanism gate fails. The no-|e| composition truth
and residual response differences are+0.00108269 and-0.00071777, contributing
-0.10733 and+0.07115pp. Both bootstrap intervals exclude zero and the raw
per-case terms correlate-0.857, so the small net-0.03617pp is a structured
cancellation rather than componentwise closure. The final m95% interval is
`[-0.7091,+0.1648]%`; central-value entry on this repeatedly reused20-case
screen is not0.3% certification. Cases140--239 therefore remain sealed.

Preflight16468787 and staging array16468789 completed; all100 stores pass.
Initial training16468898 failed before model construction because two numeric
manifest keys were omitted despite being present in the frozen textual
architecture; `training_setup_repair.json` records the exact transcription
repair. Job16468905 completed the archived30-epoch run. Tune-only extension
record `tune_only_convergence_extension.json`, converged training16468939,
case array16469024, review16469047, and independent review16469049 all pass;
successful stderr is empty. Result/sufficient/completed/review SHA256 are
`147cd4fc...c0d46`/`c0753131...fd7fa`/`ad975d51...33b67`/
`2b4a4270...b83c5`.

Disposition: reject promotion but retain the representation result. Full-scene
information nearly solves total detector composition, while the component
failure rejects this bounded 20-feature-base plus eight-feature-set model on an
exhausted development endpoint. It does not establish a detector-family limit.
The complete no-|e| moment also contains the separately supplied `R_blend`
shift, omitted from the eight-condition tower argument. The smallest next
test is therefore the eight flow conditions plus that one scalar, selected by
proper score without reopening response. Only if that compact condition fails
fresh acceptance is a joint selected-measurement model justified. A response
offset, threshold change, or post hoc opening of the sealed amplitude set is
not justified.

## Question and scope — 2026-09-11

Owner requested an attempt to solve the detection-inclusive response failure.
The preceding [diagnosis](CONSTGOLD_DETECTION_DIAGNOSIS.md#completed-flow-controls--ten-systematic-cases)
identified detector conditional-weighting errors and an additional
measurement/composition discrepancy. This first repair isolates detector
representation and supervision, keeping lambda10 epoch154 flow and the newer
ordinary-MSE trial9 blend emulator fixed. A detector-only improvement cannot
establish that the measurement-boundary problem is solved.

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1`.

No configured likelihood or inference model is changed. Wake-ups remain disabled.

## Submission and startup

Submitted22:28:44 Berlin: preparation16458344, training array16458345
(0=semi-major BCE,1=circularized BCE,2=circularized paired), dependent
development/ConstGold evaluation16458346. Independent compact-sum final
review16458388 follows evaluation. The exact commands are in submission.json.
Preparation completed6:44, exit0:0, empty stderr. All40 cases pass
identity/feature invariance and provenance checks, with zero unmatched rows.
Train/tune/test pairs:2,427,207 /484,982 /971,259; training has10,763 raw
detection transitions. Both legs of every pair enter marginal BCE.

All three training tasks completed on th-cl-nv02, exit0:0:
semi-major1:15, circularized BCE1:21, circularized paired0:54.
Losses/gradients are finite with no clipped steps in the logged histories.
PyTorch emitted a scalar-conversion warning from telemetry logging; this does
not alter the objective or updates. Frozen workers were left unchanged.
The dependent evaluation16458346 started, passed the development test and
reused case40 and newly sampled case41 original sufficient-statistic reproduction,
with empty evaluation stderr. Full ConstGold results
and independent review remain pending.

The independent final review recomputes full50 responses, m and case errors
from saved sums, verifies the original detector's result, checks checkpoint
selection against tuning histories, and reports the held-out likelihood cost
of paired supervision. Its separate frozen review_hashes.json preserves the
already submitted worker/recipe identities.

## Completed development test — 2026-09-11

All971,259 pairs in ten development test cases30–39; fixed-intrinsic
primary-direction-aligned detection response, forward0->0.05, pooled over
nonzero-primary-shear objects. This is a detection contribution, not total
measured-shape response and not coherent ConstGold R11. Uncertainties below
are10,000-resample case SE, conditional on each single fitted model.

Measured parallel response: -0.00912216 +/-0.00068159.

| Detector | Predicted parallel R | Model minus measured | Held-out BCE |
| --- | ---: | ---: | ---: |
| Old transition detector | -0.00435012 +/-0.00003589 | +0.00477204 +/-0.00065610 | 0.17591481 |
| New semi-major BCE | +0.00823317 +/-0.00006133 | +0.01735533 +/-0.00066598 | 0.15864069 |
| New circularized BCE | -0.00808345 +/-0.00006562 | +0.00103871 +/-0.00066510 | 0.15847047 |
| New circularized paired | -0.00749047 +/-0.00008111 | +0.00163169 +/-0.00066377 | 0.16189368 |

Circularized BCE flips the wrong sign of its matched semi-major control
without sacrificing likelihood: paired test-BCE change-0.00017022
+/-0.00003396. The paired-loss arm costs+0.00342321 +/-0.00012087 relative
to circularized BCE and does not improve this response residual. This
single prescribed paired loss is not evidence that all response supervision
is unhelpful. The old detector comparison also changes training volume,
domain and objective; the two new BCE arms isolate the representation change
within the matched optimization recipe.

Selected epochs/tuning BCE: semi-major74/0.16096696 (stopped89 by patience),
circularized BCE100/0.16083049 (maximum100 reached), paired38/0.16425529
(stopped53 by patience). Circularized BCE convergence is not established.
Its test-response residual is about1.56case SE, with95% case interval
[-0.00017190,+0.00242001]; this is compatible with zero but not a precision
calibration demonstration. R_cross and centered rendered diagnostics remain
in development_test.json. The full measured-shape/detection/selection closure
is still required, and no model is selected from ConstGold outcomes.

## Frozen comparison

Three matched arms use the same 3x128 SiLU MLP, initialization seed20260912,
pair ordering, training population, optimizer settings and checkpoint criterion:

| Arm | Size representation | Objective |
| --- | --- | --- |
| semimajor_bce | Original semi-major | Mean per-leg Bernoulli NLL |
| circularized_bce | Invariant circularized | Mean per-leg Bernoulli NLL |
| circularized_pair | Invariant circularized | Bernoulli NLL + all-pair difference MSE, lambda1 |

For primary and representative neighbour, rho=a0 sqrt(q0) uses pre-shear
truth. The new representation replaces BOTH scaled radii, the primary-size
normalization of separation, and the numerical size term of neighbour impact.
Four distinct feature names prevent accidental interpretation of semi-major
features as circularized features. Rendered ellipticities still change with
shear. Raw q0, applied shear, leg identity and measured outcomes are not
single-leg classifier inputs.

The neighbour identity rule remains original semi-major impact-a1 within
3arcsec on the full scene, common to all arms. Thus this is a controlled
numerical-feature replacement, not a test of a fully invariant neighbour
selection rule or a richer scene representation. All rows outside the
primary domain remain eligible neighbours.

The new paired term is

```text
mean over ALL truth pairs [ ((p1-p0) - (D1-D0)) / 0.05 ]^2.
```

It includes concordant labels, nondetections, and pairs with zero primary
shear. The fixed0.05 denotes the nominal half-sheared scene perturbation;
some neighbours can change when the primary is unsheared. It does not select
only transitions or reinterpret a marginal logit difference as a transition
probability. Linearity gives E[D1-D0|paired covariates]=p1-p0 for correctly
specified conditional marginals, regardless of shared-noise covariance.
Sufficiency of single-leg context remains an assumption: conditioning on
both legs can reveal latent information absent from either single-leg state.
This is probability-change supervision, not a fitted mean R or empirical
calibration correction. The full MSE contains irreducible label variance.

## Development data and optimization

Use original half-shear simulations in
`/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876`,
forward0->0.05, cases0–24 train,25–29 tune,30–39 test. All primaries satisfying
strict18<r<25.8 and0.37<semi-major Re<1.5arcsec are retained. There is no50k
sampling cap, class balancing or both-detected restriction. Labels are raw
unique crossmatched detection, matching the old classifier label; later
measured-shape comparisons also require usable shape measurements.
Unknown/duplicated crossmatch identities are rejected and reported.

Each case stores paired IDs, all labels, original and circularized features,
intrinsic/rendered shapes, native shear vectors, input SHA256s and counts.
Preprocessing is fitted on the training split only. All arms use AdamW
lr0.001, weight decay0.0001, batch16384 pairs, gradient clip5; maximum100
epochs, ReduceLROnPlateau using tuning BCE (factor0.5, patience5, min_lr1e-5),
and stopping after15 epochs without tuning-BCE improvement of1e-6.

**Every checkpoint is selected by minimum tuning marginal BCE only.**
The paired arm does not select on its composite training objective or R.
History records BCE, paired loss, total training loss, gradient clipping,
learning rate and duration. Atomic last-state checkpoints include optimizer,
scheduler and epoch-based ordering for authorized resume with unchanged recipe.
An operational plateau is not evidence of calibrated inference.

After all three arms freeze, report all on the ten development test cases.
The test response uses pooled forward, primary-direction-aligned fixed
intrinsic shapes and a separately centered rendered-shape selection diagnostic.
These differ from coherent central ConstGold responses and are labeled as such.
No test result selects an arm.

## ConstGold evaluation and stopping boundary

The dependent evaluation reports all three frozen arms plus the old detector
on all50 cases40–89 /4,854,422 truth objects. Training and checkpoint
selection never read ConstGold outcomes. These cases were used in the prior
diagnosis, so this is a reused diagnostic evaluation, not a fresh holdout.

Keep original support, full neighbours, flow, trial9 lookup, own measured
cuts, central g1 +/-0.02, per-leg population pooling and64QMC draws seed7301.
Use CRN between shear legs and the exact same conditional moments across
detectors. Reuse frozen per-object moments on ten systematic cases; sample
the remaining40 cases once. Old-detector sufficient statistics for all four
original comparison branches must reproduce before accepting new results.

Report detection-weighted fixed-intrinsic and rendered truth R11/R21,
measured-shape detection-only and complete responses, m, conditional detection
fractions by intrinsic ellipticity, and paired changes versus the old detector.
Bootstrap10,000 case resamples seed20260902 retains covariance. m is
100*(R_measured/R_model-1), a g1 response-closure proxy, not actual SBSI
inference bias; see [response conventions](CONVENTIONS.md#6-response-estimators)
and [m](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention).

The evaluation also extends the observed-detection-mask conditional-flow
control to all50 cases. Two existing QMC seeds on ten cases quantify
sensitivity of detector contrasts, not full-population integration convergence.
Hard-cut absolute response retains finite-draw error. No arm is promoted or
chosen using ConstGold; report improvement or remaining failure honestly.
Further flow/scene repairs depend on the results, not silently changing this
experiment mid-run.

## Implementation and verification

Frozen external workers: core.py, train.py, evaluate.py, test_trial.py and
scheduler wrappers. archived_detection.py retains the feature helper from
gitf3343bb, with imports redirected to installed SBSI; its old transition
helper is never used by the new trial. All SBSI/flow evaluation imports use
the already frozen lambda10 ConstGold environment/source from /tmp.
The live dirty checkout and prior artifacts are preserved.

Small tests establish:
- Expected paired-MSE gradient vanishes at true marginals for a correlated
  Bernoulli joint; discordant-only fitting fails this check.
- Circularized features remain constant under applied shear and fail on
  mismatched baseline IDs.
- Archived training and frozen inference feature builders agree for a
  neighbour/isolate synthetic scene.
- Save/load preserves distinct circularized feature names and rejects a
  semi-major-only frame.
- Central sufficient-statistic pooling matches a direct weighted calculation.

The first local test exposed a mixed bool/float DataFrame comparison dtype;
explicit numerical test/validation arrays corrected that before submission.
The scientific tests pass. Resource inventory: no login GPU,330.96GB
available RAM,6.7TB free storage. All catalogue preparation, training and
evaluation run through Slurm. Manifest, workers, raw input file stats and
source/model hashes are frozen; per-case preparation adds content hashes.

## Completed full ConstGold comparison — 2026-09-11

Evaluation16458346 completed1:05:53, exit0:0; independent review16458388
completed4seconds, exit0:0. Both stderr files are empty. All50 cases40–89
and4,854,422 truth objects completed, with original sufficient-statistic
reproduction for every case. Selected detector hashes, input/source
provenance, checkpoint selection and independent m/bootstrap arithmetic pass.
No active jobs remain in this trial.

**The detector repair does not solve the complete measured-shape bias.**
Circularization improves the isolated detector response and development
likelihood, but full detection-plus-selection m remains about-2%.

All responses below are central g1 +/-0.02 on the original population,
with the same lambda10epoch154 flow, ordinary-MSE trial9 blend emulator,
64QMC draws seed7301, and own measured selection. Measured complete
R11=0.60537833 +/-0.00188801; measured detection-only R11=0.95029592
+/-0.00215464. Errors are case SE, conditional on models/QMC draws.

| Detector | Complete predicted R11 | Complete m percent | Detection-only m percent | Development test BCE |
| --- | ---: | ---: | ---: | ---: |
| Existing transition | 0.61843461 +/-0.00050606 | -2.11118 +/-0.30522 | -1.00528 +/-0.22400 | 0.17591481 |
| New semi-major BCE | 0.62196345 +/-0.00050585 | -2.66658 +/-0.30338 | -1.63814 +/-0.22281 | 0.15864069 |
| New circularized BCE | 0.61816317 +/-0.00050856 | -2.06820 +/-0.30510 | -1.12888 +/-0.22404 | 0.15847047 |
| New circularized paired | 0.61804536 +/-0.00050708 | -2.04953 +/-0.30531 | -1.13991 +/-0.22413 | 0.16189368 |

The circularized BCE complete m change versus the existing detector is
+0.04298 +/-0.00405 percentage points; paired arm+0.06165 +/-0.00363.
These are small resolved changes conditional on the current integration,
not removal of the approximately2% discrepancy. The detection-only
measured-shape response actually increases by0.00120002 +/-0.00002286
for circularized BCE versus the old detector, despite the improved
intrinsic-shape detector response. Weighting functions and conditional
measurement errors differ; isolated detector response is not the full closure.

The same case-bootstrap distribution is used across arms for paired errors.
The new implementation draws multinomial case counts; finite-resampling SE
therefore differs slightly from earlier bootstrap implementations despite the
same nominal seed and replicate count. Original response point estimates
reproduce. Uncertainties do not include fitting variation or full MC error.

### Isolated detector response and conditional fractions

ConstGold fixed-intrinsic R11 from detection alone is measured-0.00694282
+/-0.00025683. This uses fixed intrinsic shapes and must not be confused with
the total measured-shape responses above.

| Detector | Predicted fixed-intrinsic R11 | Residual +/-paired case SE |
| --- | ---: | ---: |
| Existing transition | -0.00490118 | +0.00204165 +/-0.00025623 |
| New semi-major BCE | +0.00876281 | +0.01570563 +/-0.00025688 |
| New circularized BCE | -0.00753210 | -0.00058927 +/-0.00025500 |
| New circularized paired | -0.00717210 | -0.00022927 +/-0.00025802 |

Circularized BCE still differs by about2.3case SE. The paired arm improves
this R11 component but has an intrinsic R21 residual+0.00083481
+/-0.00027749, versus circularized BCE+0.00031079 +/-0.00028408.
One improved component is not a full detector-calibration claim.

The large intrinsic-ellipticity-dependent probability offsets are reduced:
for |e0|<0.2, actual mean detection0.91033903, old0.92647728,
circularized BCE0.91029565. For0.6<=|e0|<0.8, actual0.87323435,
old0.84793398, circularized BCE0.87283163. At |e0|>=0.8 (only6,585objects),
actual0.85694761, circularized BCE0.86505606; not every subgroup is calibrated.
The better probability fractions and isolated response still do not establish
the accuracy of measurement-weighted selected moments.

### Full50 observed-mask conditional-model control

Replacing predicted detection probabilities with actual per-leg usable
detection identities gives:

| Control | Measured R11 | Predicted R11 | m percent |
| --- | ---: | ---: | ---: |
| Actual detection flags, no measured cuts | 0.95029592 | 0.95541617 | -0.53592 +/-0.22716 |
| Actual detection flags, own measured cuts | 0.60537833 | 0.61365453 | -1.34867 +/-0.30744 |

The selected response residual is+0.00827620 +/-0.00188764, about4.4case SE.
Thus even the observed-mask control fails. A correct detector alone is
insufficient to explain away the complete error. The control still includes
the conditional flow, fixed blend correction and model measurement cuts;
it does not uniquely attribute the residual to one of them or prove a
specific missing-variable mechanism. Actual-mask intervention is a diagnostic,
not a deployable calibration correction.

Fixed both-usable comparisons remain unchanged by the detector swap:
pure m+0.23380 +/-0.22218%, own measured selection-0.71599 +/-0.29455%.
Their different parent population must not be substituted for complete m.

### Integration and interpretation limits

Ten-case second-seed selected-response shifts are+0.00046716 for the old
detector,+0.00055774 circularized BCE,+0.00058011 paired. Detector-contrast
shifts versus old are+0.00009058/+0.00011296 respectively. These are
smaller than the full residual but material compared with the tiny detector
improvements; two seeds on ten cases do not establish full50 integration
convergence. Full50 results remain conditional on one64-draw QMC seed.

Circularized BCE reached the100-epoch cap; convergence remains unestablished.
All arms were frozen using development tuning BCE before ConstGold, and all
are reported. This reused ConstGold diagnostic set selects no model.
No empirical correction, model promotion, new training or wake-up follows
from this review. The remaining scientific issue is the conditional
measurement/selection composition and its treatment of detection-boundary
objects; the present tests have not solved or uniquely localized that issue.

Full records: external results.json, final_review.json and completion_review.json.
Results SHA2569e70aa24a0d7542a66fe6717e6b5c88df5ca3c97efae6f7e5633b2cbb60f3273;
final-review SHA2567277d4a57504dec2de11bc5d90b09e93d6bbfbe7c8a341dba38f6ebae6ac7c73.
