# Why response regularization changes joint calibration

## Scope and evidence

The owner asks why one/nine-cut response regularization improves the standalone
flow response but worsens coherent constant-shear calibration, and whether the
improved GalSBI flow recovers the earlier FS2 measured–intrinsic alignment relation.
The [fixed protocol](rule.md#questions-and-fixed-evidence) uses all six fixed
checkpoints, the frozen19-case interim intersection for coherent attribution,
and40 original validation cases for standalone alignment/response. No retraining,
new images, empirical offsets, acceptance claims or checkpoint selection.

Definitions follow [Conventions](../../CONVENTIONS.md#6-response-estimators):
DeltaR=simulation-model; m=100(Rsim/Rmodel-1). Model changes below are new-minus-old.
Alignment error has the opposite convention, A_model-A_simulation. Case
uncertainties condition on fitted models and the existing measurements.

## The coherent shift is mostly in the flow contribution

### Completed matched own-group and neighbour budget

The baseline case1080 and reducers17804354/17805824 completed successfully.
All four predeclared scenes now enter both arms. With actual per-state usability,
generated selection at radius>3pixels/magnitude<25, and global normalization:

| Absolute response error | Unregularized | Weak nine-cut regularization |
|---|---:|---:|
| Own-group shear, including same-group neighbours | -0.008736 +/-0.004575 | +0.001025 +/-0.004583 |
| Opposite-group shear | +0.006989 +/-0.001626 | +0.007538 +/-0.001627 |
| Coherent total | -0.001747 +/-0.005456 | +0.008563 +/-0.005466 |

Each column adds exactly. Paired regularization-induced error changes are
+0.009761 +/-0.000029 in the own-group term and+0.000549 +/-0.000006 in the
opposite term, summing+0.010310 +/-0.000031. The narrow errors on these paired
model contrasts are not the uncertainty on absolute calibration. Four scenes
cannot establish independent acceptance. The point estimates directly expose
the cancellation: the original negative own-group error offset a positive
neighbour-related error; regularization removes most of the former and leaves
the latter. Own-group shear is not a standalone-flow experiment.

Sources: [four-state report](fourstate_report.json),
[matched input comparison](neighbour_input_results.md#paired-changes-in-model-response).

### Four-scene localization: photometry/selection drives neighbour deficit

All four nine-cut mixed-state cases and measured/conditional-input tests now
pass their numerical gates. CPU17805954/17805962 report the same fixed four
scenes for the weak nine-cut checkpoint. The later completed two-arm reducers
17804354/17805824 now provide the original-flow comparison on those same cases.

With actual per-state usability, globally normalized own/opposite/total errors
are+0.001025 +/-0.004583,+0.007538 +/-0.001627,+0.008563 +/-0.005466. They add
exactly and their SEs are correlated. Most of the point residual is opposite-group.

The following conditional-input comparison uses common-all-four usability and
equal within-group A/B normalization, so it is not the preceding global budget:

| Inputs | Neighbour shape error | Error including measured selection |
|---|---:|---:|
| Generated photometry/selection, nine-cut flow shapes | +0.005217 +/-0.001492 | +0.008142 +/-0.001773 |
| Measured photometry/selection, conditional nine-cut flow shapes | -0.001337 +/-0.001546 | +0.001588 +/-0.001839 |
| Measured photometry and measured shapes | -0.001274 +/-0.001458 | +0.001651 +/-0.001762 |

Photometry/selection substitution raises model response+0.006554 +/-0.000175;
the subsequent conditional-to-measured-shape substitution changes it only
-0.000063 +/-0.000115. These are paired model-change SEs, not absolute calibration
uncertainties. Generated and measured selected masses already agree:
ratio0.998324 +/-0.001624. Matching selected counts therefore does not ensure
matching the distribution of measurements and associated neighbour responses.

The missing measured selection response is+0.002925 +/-0.000418. The model
holds radius/flux fixed when only the opposite group is sheared, so this term
is absent by construction. Conditional unregularized-flow shapes at measured
photometry also give shape error-0.001600 +/-0.001546 over all four scenes.
Together these tests support a photometry/selection input-distribution problem
and omitted neighbour-induced selection, rather than a required mean bias in
the standalone emulator or a remaining failure of conditional flow shapes.
They do not yet isolate radius versus flux, selection membership versus
response weighting, or density underfit versus missing neighbour conditioning.

Source: [paired four-scene input comparison](neighbour_input_nine_results.md).
Four development scenes, fixed models; no independent acceptance claim and no
empirical correction. Internal subpopulation cancellation remains acceptable
under the owner's target-cut objective.

The unregularized model has the same input sensitivity: generated-input shape
error+0.004688 +/-0.001488, measured-photometry conditional-shape error-0.001600
+/-0.001546. The paired photometry/selection substitution raises its prediction
+0.006288 +/-0.000176. Thus the neighbour deficit predates regularization.

The subsequent [four-scene factorial](../photometry_selection_factorial_20261106/findings.md#evidence-and-subsequent-localization)
crosses generated/measured photometry and selection rather than replacing them
together. Joint substitution raises prediction0.006556 +/-0.000171, whereas
photometry alone changes0.000655 +/-0.000055 and membership alone-0.000174
+/-0.000126. The interaction is0.006075 +/-0.000032. Scalar emulator coefficient
change0.006547 accounts for nearly all the diagonal difference; finite
shape-transport geometry changes0.000009. The diagonal shift is positive in
every scene and its jackknife SE0.000199 supplements bootstrap0.000171.
Crossed branches intentionally break some measurement associations and cannot
be interpreted as corrected generative predictions or unique causal shares.

CPU17804158 completed0 in20s. Its [decomposition](constant_decomposition.json)
reproduces the19-case interim response sums and checks all additive identities
for every case-bootstrap replicate. With predicted usability and generated
selection, radius>3pixels and magnitude<25, absolute changes in R11 are:

| Continuation | Flow contribution change | Emulator contribution change | Fraction of total change from flow |
|---|---:|---:|---:|
| One cut, weight0.001 | -0.009302 +/-0.000028 | -0.001062 +/-0.000002 | 89.8% |
| Nine cuts, weight0.001 | -0.009282 +/-0.000030 | -0.001121 +/-0.000002 | 89.2% |
| One cut, weight0.01 | -0.011090 +/-0.000048 | -0.001125 +/-0.000003 | 90.8% |
| Nine cuts, weight0.01 | -0.012986 +/-0.000047 | -0.000846 +/-0.000002 | 93.9% |

These small SEs quantify paired changes on identical scenes; they are not
absolute calibration uncertainties. The unchanged emulator still changes its
contribution because it receives different flow draws and selected populations.
They are components of the composed estimator, not separately measured errors.

For the weak nine-cut arm, the simulation R11 is0.906696. Original model flow
0.734660 plus emulator0.174548 gives0.909208. Regularized flow0.725378 plus
emulator0.173427 gives0.898805. This change explains m1 moving from-0.2763%
to+0.8779% on exactly these19 cases.

## Size location and normalization

The weak nine-cut flow shift splits into globally normalized intrinsic-size
contributions: -0.000788 below0.15arcsec, -0.004860 at0.15-0.25,
-0.003339 at0.25-0.4, -0.000343 at0.4-0.7, and+0.000048 above0.7.
The0.15-0.4arcsec region carries88.3% of the flow shift. These intrinsic bins
describe galaxies selected by measured radius>0.6arcsec; no truth-size cut is
applied to the final sample.

A symmetric fraction-times-conditional-mean decomposition gives-0.009658 from
changed conditional means and+0.000376 from changed selected size fractions,
summing to-0.009282. A separate numerator/denominator identity gives-0.010674
from numerator changes and+0.001392 from denominator changes. Thus changed
coarse size proportions or normalization alone do not produce the downward
shift. These identities are descriptive: within-bin changes can still include
selection-shape coupling and do not prove a uniquely causal shape-only error.

## Why improving the standalone score can move the joint result away

The smooth training guard asks for a lower R11: its full160-case measured
target is0.712828, while the fixed model probe initially gives0.728951. The
weak nine-cut endpoint gives0.718312. On the validation probe the measured
target/model move is0.716088 versus0.732637 ->0.721934. These probe values use
smooth selection and measured-reference normalization; they are not the hard-cut
coherent responses in the table above. The learned downward response change
transfers to coherent inputs, where the total model initially already matched
the measured response approximately.

This explains the direction of the change, but alone does not identify which
underlying component was biased before regularization. Target-only forward
shear and coherent target-plus-neighbour shear are different experiments. The
following same-image four-state check tests the neighbour part directly.

### Why the standalone emulator null does not close this budget

The original neighbour-weighted pair residual is-0.002556 +/-0.002293,
consistent with zero for that statistic. It sums projected pair labels over
represented primaries with measured baseline selection. It is not the coherent
R11 error after flow draws, neighbour transport, and per-leg selection. Do not
attribute the joint residual to emulator error by subtracting the standalone
flow statistic. The descriptive true-usability joint-minus-standalone gap is
+0.014789 for baseline,+0.015775 for weak nine-cut,+0.015484 for strong nine-cut;
the underlying experiments have19 coherent scenes versus40 standalone cases.
This approximate persistence supports transfer of the flow change, while leaving
the source of the discrepancy between experiments unresolved. Earlier four-state
results established an opposite-group residual for the older flow; they are not
a numerical attribution for these improved checkpoints.

## Alignment evidence and new check

The completed earlier40-case comparison already shows that the improved
factorized Cartesian GalSBI flow recovers the original problematic relation:
for intrinsic Re0.15-0.25arcsec, output radius2.7-2.8pixels and magnitude<25,
alignment errors are+0.41323 +/-0.00472 for original GalSBI NLL,
+0.03229 +/-0.00463 for FS2 NLL, and-0.00360 +/-0.00460 for improved GalSBI.
Immediately above the cut, radius3.0-3.1, improved Cartesian retains
+0.02604 +/-0.00403; the improved intrinsic-frame model gives-0.00111 +/-0.00417.
These are absolute slope errors, not percentages or m. FS2/GalSBI comparisons
use each family's own measured truth distribution, not identical galaxies.

New array17804090 re-evaluates all six GalSBI endpoints with shared portable
latent banks, full hard-cut response matrices and an exact finite-pair
shape-versus-selection split. Reducer17804172 also compares the preserved
FS2 NLL-only and original GalSBI profiles. This tests whether regularization
damages alignment recovery or instead changes shear sensitivity while the
zero-shear relation stays similar. The completed results follow below.

### Completed full-validation replay

All six17804090 workers completed exit0 in2m37s-3m35s; CPU17804172 completed0
in43s. The [standalone report](standalone_report.json) verifies measured sums
against the original Cartesian evaluation and exact shape/selection response
addition in every bootstrap replicate. All40 validation cases are included.
There are5,817,708 matched usable pairs. The loader drops35,353 zero-leg and
32,816 sheared-leg unmatched usable rows; no matched zero-shear pairs were
excluded. These counts are unchanged across all six model evaluations.

At radius>3pixels and magnitude<25, hard-cut absolute standalone errors are:

| Model | Delta R11 | Delta R22 |
|---|---:|---:|
| Improved Cartesian baseline | -0.015438 +/-0.003378 | +0.000997 +/-0.003818 |
| NLL-only continuation | -0.018125 +/-0.003367 | -0.003603 +/-0.003815 |
| One cut, weight0.001 | -0.006087 +/-0.003365 | +0.005593 +/-0.003819 |
| Nine cuts, weight0.001 | -0.006063 +/-0.003361 | +0.006572 +/-0.003814 |
| One cut, weight0.01 | -0.004204 +/-0.003371 | +0.004860 +/-0.003813 |
| Nine cuts, weight0.01 | -0.002330 +/-0.003351 | +0.006070 +/-0.003819 |

Thus the R11 improvement survives full-case, hard-cut, own-normalization
evaluation; it is not solely a noisy smooth-probe result. Not every response
component improves: the initially small R22 residual increases for every
regularized endpoint. The full standalone model-response change for weak
nine-cut is-0.0093755, almost identical to the coherent flow contribution
change-0.0092820 on the frozen19 scenes. These are different catalogue families,
so their difference is not a paired same-case uncertainty or physical error budget.

The exact finite-pair split assigns-0.0087313 (93.1%) of the weak nine-cut R11
change to the shape-change term, and-0.0006442 to the cut-crossing term, using
the average selection weights and own baseline normalization. Across the four
regularized arms, shape-change fractions are approximately90.9-95.8%. This is
not a causal replacement of one network block: both terms retain the joint
shape/photometry distribution. Measured-reference normalization gives a
weak-nine change-0.0107307, so replacing that normalization by the actual model
one does not remove or reverse the response change.

The paired40-case alignment errors at magnitude<25 are:

| Model | Radius2.7-2.8pixels | Radius3.0-3.1pixels |
|---|---:|---:|
| FS2 original NLL | +0.032291 +/-0.004626 | -0.011131 +/-0.006969 |
| Original GalSBI NLL | +0.413230 +/-0.004720 | +0.096324 +/-0.004180 |
| Improved GalSBI Cartesian | -0.004247 +/-0.004253 | +0.027125 +/-0.004109 |
| Improved GalSBI intrinsic frame | -0.002455 +/-0.004768 | -0.001106 +/-0.004171 |
| Cartesian + one cut, weight0.001 | -0.010899 +/-0.004313 | +0.016964 +/-0.004103 |
| Cartesian + nine cuts, weight0.001 | -0.014892 +/-0.004312 | +0.013163 +/-0.004112 |
| Cartesian + one cut, weight0.01 | -0.008386 +/-0.004333 | +0.014012 +/-0.004170 |
| Cartesian + nine cuts, weight0.01 | -0.017898 +/-0.004379 | +0.009890 +/-0.004139 |

New Cartesian endpoints share portable draws; FS2, old GalSBI and frame entries
use their preserved banks and are reference comparisons. FS2 is an independent
population, so the table is not paired across families. The new random bank
slightly changes the original Cartesian point estimates relative to historical
tables. These alignment errors are slopes, not percentages. At magnitude<25.8,
improved Cartesian errors are-0.008936/+0.009955 below/above; weak-nine errors
are-0.016745/-0.001931. The former large GalSBI alignment failure remains repaired.
Regularization modestly shifts the profile downward, improving the just-above-cut
offset while making the below-cut error somewhat negative. Good zero-shear
alignment does not establish the full shear derivative or neighbour response.

## Same-image neighbour-response test

The four-state images1000/1040/1080/1120 already exist from the earlier
[matched experiment](../constant_split_20261105/findings.md#completed-matched-shear-response-decomposition).
They vary the other galaxy group's shear while holding a target's own shear
fixed. Re-evaluate original improved Cartesian and the weak nine-cut endpoint,
using the same portable draws and unchanged C8/emulator. Own-group includes
same-group neighbours; no doubling of opposite-group results is allowed.

Pilot17804351 gates remaining17804352; reports17804353/17804354. Measured
sums must replay the preserved four-state cache, and coherent ++/-- model
sums must replay the new constant-shear results. The test distinguishes direct
neighbour-response discrepancy from the aggregate standalone-to-joint comparison.
Four-case absolute errors remain uncertain. No new image simulation is run.

### Pilot numerical replay amendment

Original Cartesian pilot17804351_0 completed0 in6m55s. Weak-nine17804351_1
failed the raw-sum comparison by0.00016172, with the cached run on V100 and
new run on A40. Its split-neighbour and measured checks had passed. A40 probe
17804825 completed; V10017804826 and comparison17804827 were cancelled because
the scheduler predicted no V100 allocation for days. Cross-device causation
therefore remains unconfirmed by a paired device probe.

The [amended replay protocol](rule.md#numerical-replay-amendment-after-pilot)
adds independently evaluated all-neighbour coherent predictions on the same
device and keeps their strict raw-sum gate. Historical-cache comparisons now
have explicit small normalized-response bounds and retain the actual differences.
No result is corrected or pasted. Weak-nine retry17804916 gates the remaining
cases and pilot report; this repair is pending verification.

The owner subsequently requested a pause of earlier broad evaluations. Pending
coherent tasks221-399 in17788286 are held; four active tasks finish normally.
Cancel automatic restore17804373. The focused mixed-state diagnostic continues;
held coherent evaluations require an owner request to resume. Existing data and
model identities are preserved.

The same-device retry17804916 passed its strict independent coherent replay
but failed the historical-cache mass fraction gate:1.92e-7 at mag25, with total
R11 difference2.07e-6. Preserve this result; no further tolerance widening.
The subsequent V100 probe completed and confirms identical latent hashes but
different floating-point physical outputs; its original per-draw tolerance did
not pass. The [descriptive audit](v100_verification.json) records mean shape
differences around1e-9 and no target-cut disagreements in the sampled banks,
without treating that as full-catalogue acceptance.

Retry17805334 on V100, matching the cached weak-nine case1000 hardware family.
Remaining tasks now use their corresponding A40/V100/RTX families in17805335,
17805336 and17805337. Broad evaluations remain held and all their active tasks
have finished; up to five focused workers remain below the six-GPU ceiling.
Pilot and full reports retain their gates and scientific definitions.

V100 pilot17805334 passed the numerical replay gates and saved its arrays,
then failed while writing metadata: the replay loop shadowed the device-check
dictionary. Rename that loop variable in evaluate_fourstate.py. CPU17805618
audits the preserved arrays and reconstructs only the missing receipt using
recover_pilot_receipt.py; no predictions are changed. Audit completed exit0
in3s, and the one-case reducer17804353 completed exit0 in2s. Remaining arrays and the
pilot reducer now depend on this audit. The failed GPU job remains recorded
as failed; recovery does not relabel its scheduler status.

### One-case pilot: neighbour discrepancy persists after regularization

The [pilot report](fourstate_pilot_report.json) contains only case1000, so no
case SE or population inference is available. With actual per-state usability,
radius>3px and magnitude<25, exact global-normalization contributions are:

| Flow | Own-group error | Opposite-group error | Coherent total error |
|---|---:|---:|---:|
| Improved Cartesian baseline | -0.014142 | +0.007196 | -0.006946 |
| Nine cuts, weight0.001 | -0.004323 | +0.007748 | +0.003425 |

All entries are simulation minus model; each row adds exactly. Own-group
includes same-group neighbours, not pure flow. Opposite-group varies the other
galaxies' shear while target own shear stays fixed, including selection and
normalization effects. This pilot supports a persistent neighbour-related
discrepancy concealed by a negative own-group contribution before regularization;
it does not isolate emulator weights from missing selection response or establish
the four-scene mean. The original standalone pair-statistic null remains valid.

## Four-scene measured-input neighbour test

All four measured-input jobs and reducer17805770 completed successfully. The
[report](measured_transport_report.json) uses common-all-four usability,
radius>3px, magnitude<25, and the equal average of separately normalized A/B
opposite-group responses. Actual measured shape/photometry anchor the unchanged
emulator's finite transport; these are conditional tests, not deployable draws.

| Quantity | Response or error +/- paired four-case SE |
|---|---:|
| Measured shape response | +0.090175 +/-0.001643 |
| Emulator shape response with measured inputs | +0.091449 +/-0.000392 |
| Shape error, simulation minus model | -0.001274 +/-0.001458 |
| Measured neighbour-induced selection response | +0.002925 +/-0.000418 |
| Total conditional error including selection | +0.001651 +/-0.001762 |

No mean shape bias is detected on these matched measured inputs. Forward versus
reverse anchors differ0.000027 +/-0.000060, and finite transport versus its
local gain approximation differs0.000013 +/-0.000001. The result therefore does
not depend materially on which endpoint anchors the prediction or on the small
finite-transport correction. This does not test the finite0.2 training labels.
Uncut, magnitude-only and radius-only shape errors are also consistent with
zero. The radius0.15–0.25 intrinsic bin contributes+0.002105 of the+0.002925
selection term. Conditional shape residuals do vary by size and cancel; that
does not disqualify the target-cut mean under the owner's stated objective.

The conditional-flow follow-up fixes measured radius/flux and draws only shape
from each flow, with shared latent banks. It separates the photometry/selection
substitution from replacing the conditional shapes with measured shapes. Pilot
17805804 passed loading/reconstruction checks then failed at a group-index dtype
before producing predictions. Force the index tensor to int64 and retry; preserve
the failed log. No scientific settings or result tolerances changed.

### Conditional information available to the flow and emulator

Code inspection of the pinned F8 runtime's fixed_g0_domain.py confirms that the
flow sees five intrinsic target quantities plus three close-crowding flux sums
(Gaussian scales0.5/1/2arcsec). The emulator separately sees each neighbour's
size, magnitude, Sersic index and separation, plus target measured/generated
radius and flux. Therefore matching p(radius,flux|flow context) does not by
itself guarantee the correct joint association of these outputs with the more
detailed neighbour variables used by the emulator. This is a candidate
explanation for an input-substitution response shift even with good marginal
flow validation. It is not established by the current conditional test; density
underfit and insufficient neighbour conditioning remain distinct hypotheses.
An eventual matched-context permutation or held-out richer-context prediction
test would distinguish them. No such permutation result or model improvement
is claimed here.

### Pilot isolates photometry/selection from conditional shape

Conditional-shape pilot17805843 and reducers17805812/17805813 completed exit0.
On case1000, common usability, target cut and equal A/B within-group opposite
normalization, measured shape response is0.090202. Predictions are:

| Inputs | Original flow | Weak nine-cut flow |
|---|---:|---:|
| Generated photometry and generated selection | 0.084603 | 0.084072 |
| Measured photometry/selection, conditional flow shapes | 0.090387 | 0.090128 |
| Measured photometry and measured shapes | 0.090354 | 0.090354 |

The photometry/selection substitution raises the response by0.005784/0.006056.
Replacing the remaining conditional shapes changes it only-0.000033/+0.000226.
Both flows' conditional shape errors are near zero in this scene
(-0.000185/+0.000073). This localizes nearly all of the pilot input-substitution
effect to photometry/selection rather than the conditional shape distribution.
It does not distinguish radius from flux, marginal density error from missing
neighbour conditioning, or selection membership from response weighting.
Three-scene follow-up17805816 is running with the same paired latent convention;
do not report the pilot as a four-scene or independent-acceptance result.
Sources: [baseline conditional pilot](conditional_transport_baseline_pilot_report.json),
[nine-cut conditional pilot](conditional_transport_nine_weak_pilot_report.json).

## Availability of the earlier six-row error table

The [available table](available_error_table.json) covers the improved Cartesian
NLL-only model at radius>3pixels, magnitude<25. Five rows are available; standalone
flow with actual measured selection membership is missing. The conditional
photometry check and measured-reference normalization do not supply that row.

| Test | Flags/selection | Cases | Absolute error +/-case SE |
|---|---|---:|---:|
| Flow alone, R11 | Generated selection | 40 | -0.015366 +/-0.003347 |
| Flow alone, R11 | Actual usability + measured membership | 40 | Not computed |
| Emulator, projected pair response | Actual usability + measured baseline selection | 40 | -0.002556 +/-0.002293 |
| Joint, R11 | Predicted usability + generated selection | 40 | -0.002492 +/-0.002724 |
| Joint, R11 | Actual usability + generated selection | 40 | -0.000607 +/-0.002730 |
| Joint, R11 | Actual usability + measured membership | 40 | +0.027384 +/-0.002781 |

These retain the same40-case original coherent realization for all three joint
rows. Flow uses its own40 held-out training-family cases; emulator its original
40 development cases. The historical quoted flow rows were half-trace, whereas
this table explicitly uses R11 for flow and joint. The projected emulator row
is not a direct coherent R11 error and the standalone rows do not add to the
joint error. Measured-mask substitution changes shape/selection coupling and
is not a conditional-density correction. Regularized measured-mask rows remain
uncomputed; no paused evaluation is resumed to fill them.

## Execution and limits

CPU resource audit found~696GB available host memory and14TB free project disk;
reducers request8GB. Nontrivial work runs through Slurm. The owner paused the
broad constant-shear evaluation: tasks221–399 of17788286 remain user-held,
and the earlier active tasks have finished. Automatic release17804373 was
cancelled; no automatic resumption is scheduled. Focused diagnostics continue.
Hardware-matched pilot17805334 has started on a V100 and logged meaningful
progress; dependent arrays17805335–17805337 permit at most five GPUs combined,
within the study's six-GPU ceiling. Syntax checks pass; completed standalone
workers have exit0. All scientific cases, seeds, checkpoint identities,
original classifier/emulator and final40-case protocol are preserved. Raw
outputs reside under the matching external cache directory.
