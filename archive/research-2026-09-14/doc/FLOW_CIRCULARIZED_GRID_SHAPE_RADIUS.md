# Circularized 6x6x6 grid shape and radius response, lambda1

Related diagnostic: [small circularized-radius response discrepancy and measured-size definition](FLOW_SMALL_CIRCULARIZED_RADIUS.md#size-bin-discrepancy-and-density-fit),
with full validation, finer conditional slices and measured-radius survival.

## Owner request and fixed comparison — 2026-09-12

The owner requested a circularized, grid-based shape+radius response model
with lambda1, replacing semi-major size as a grid axis with circularized
radius and increasing the grid from6x6x5 to6x6x6. This is a new branch from
the same circularized NLL epoch140 checkpoint and committed AdamW state used
by the earlier [grid shape branch](FLOW_CIRCULARIZED_CONDITION.md#nll-continuation-and-shape-only-branch-2026-09-11)
and [grid-free shape+radius branch](FLOW_GRID_FREE_SHAPE_RADIUS.md#user-request-and-controlled-starting-point).
It does not resume the selected lambda10 epoch154 weights.

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_shape_radius_lambda1_6x6x6_v1`.
The immutable `prepare_manifest.json` and `manifest.json` record source and
input hashes, grid definition, parent/optimizer, recipe and launchers.
Scientific library source and interpreter are reused from the frozen
`physical_circularized_grid_free_shape_radius_lambda10_v2/source` and `env`.
No production library or configured likelihood is changed.

## Training-derived 216-cell grid

Use marginal training-pair quantiles, six intervals on each axis. These exact
boundaries are frozen before validation fitting or model optimization:

| Axis | Boundaries, rounded |
|---|---|
| True magnitude | 18.002083,23.189308,24.029625,24.569221,24.983917,25.381142,25.799995 |
| Circularized radius, arcsec | 0.086924,0.296481,0.344920,0.401222,0.483459,0.616280,1.487055 |
| Cached baseline r_blend | -2.300985,-0.008238,0.000815,0.013156,0.049656,0.199282,4.247325 |

Circularized radius is the invariant pre-fold baseline value saved by the
original circularized preparation, exactly identical across both paired legs.
The historical cached emulator-derived r_blend is used only for cell
membership; it is not newly recomputed with trial9 and is not applied as a
blend shift to the self-response predictions. Internal boundary values enter
the bin on their right. Outer bins retain finite validation values outside
training extrema, with counts reported; this dataset has no such overflow.

Every original pair is retained in its original order. Match baseline
catalogue features by(case,input_index), reject duplicates and report/reject
missing cached identities. No measurement or NLL target is reconstructed from
rounded display values. The physical truth population remains
18<true magnitude<25.8 and0.37<semi-major Re<1.5arcsec, with original source
eligibility and complete neighbour context. Changing the grid size coordinate
does not change that support cut.

Preparation16466475 completed successfully with no unmatched pairs or new
exclusions. Training:6,834,259pairs,13,821,950NLL rows,160cases; all216cells
contain all160cases, with1,782–252,999pairs per cell. Validation:
1,708,854pairs,3,455,704NLL rows,40cases; every cell contains all40cases,
with391–63,645pairs per cell. Full/deleted-case shear designs and response
covariances pass the existing conditioning gates. The original frozen
intrinsic scalar frame reproduces exactly in validation.

## Exact six-component response loss

Fit each cell's matched forward0→0.05 WLS responses in the same way on both
sides: four sky-frame ellipticity entries and physical measured flux-radius
parallel/cross entries in the fixed baseline-intrinsic frame. Radius is
exponentiated before averaging flow samples and measured in pixels. See
[CONVENTIONS §6c–6d](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).

Let r_B contain(R11,R12,R21,R22,Rradius_parallel,Rradius_cross), and V_B be
the6x6 marginal case-jackknife covariance of the measured cell response.
The population objective is

    L = NLL + 1/(2*6*216) sum_B (r_model,B-r_data,B)^T V_B^-1 (r_model,B-r_data,B).

Invert the marginal6x6 covariance itself; do not slice the inverse8x8
shape/radius/flux covariance. Shape–radius cross covariance is retained.
All six component weights are1. Flux remains in the full joint NLL and
diagnostics, with no flux response penalty. This covariance-normalized grid
lambda1 is not numerically equivalent to the physical grid-free lambda1.

Sample12cells uniformly per optimizer update and two independent groups
of128pairs per cell with replacement. Each group uses16 antithetic latent
draws and CRN between shear legs. The independent-group cross quadratic
estimates the objective without an extra model-dependent MC variance penalty.
Exact chunked replay retains the NLL gradient. No covariance flooring,
empirical response correction or outlier clipping is introduced.

Joint loss and standalone marginal shape/radius losses are logged separately.
The marginal scores are diagnostic and are not additive pieces of the joint
loss because the joint covariance includes cross terms.

## Optimization, selection and final audit

Restore epoch140 weights and AdamW moments with the original learning rate
1.5625e-6. Keep architecture, eight circularized conditions, physical output
map, seed501 NLL streams, four million NLL rows per epoch, batch8192,
weight decay0 and gradient clip5. Run through at most epoch200.
Reset the objective-dependent ReduceLROnPlateau scheduler: factor0.5,
patience4, absolute threshold1e-4, minimum LR1e-7. Stop after20epochs
without a new best at minimum LR, or at the cap. Neither is proof of convergence.

Select by NLL on the original fixed500k validation rows plus lambda1 times
the joint grid response score. Use a fixed50k stratified validation-pair
subset, refitting targets on those pairs and using the full validation
cell covariance as precision. Subset allocation retains all pairs in cells
smaller than the equal quota and redistributes the unused budget deterministically;
all cells here exceed the quota. Four independent64-draw groups and fixed
CRN are shared across checkpoints. The original parent participates in selection.
Atomic model/optimizer checkpoints support `run.py train --resume` under
unchanged manifest and metadata.

After selection, audit all3,455,704 validation NLL rows and1,708,854pairs with
16independent64-draw groups. Preserve original pair order,1024-pair batches
and response seed3170000000001 to share CRN with earlier models. The same
new per-pair draws produce both216-cell results and the original180-cell
comparison; retain per-pair physical differences for later feature plots.
Review compares NLL-only, grid shape lambda0.1, grid-free shape lambda1 and
grid-free shape+radius lambda10, keeping paired case and MC uncertainty.
Grid averages with different boundaries must not be compared as if they had
the same target or weighting. The original180-cell comparison provides a
common response ruler. Full audits reuse validation data, including selection.

## Validation, execution and limitations

New external tests cover216-cell assignment and tail retention, fixed subset
allocation, independent cross-score normalization and radius-unit invariance,
prepared populations and six-component flux exclusion. The validation job
also runs existing physical-response, population/replay and optimizer
continuation tests against the frozen library. Python AST and Bash syntax
checks pass. Initial validation16466501 stopped during test collection because
its `/tmp` working directory did not expose the frozen `scripts` namespace.
No training ran; pending16466502–16466504 were cancelled. Repair1 changes
only the test launcher's working directory to the frozen source root, keeping
all scientific workers/manifests and failed logs. Slurm had already retired
the completed preparation from dependency lookup, so replacement validation
runs without that obsolete dependency and hash-checks the completed preparation.
Replacement validation16466515 passed all18tests in33.93s, with empty stderr.
Training16466516 started on a V100 and reproduced parent epoch140 NLL exactly
at-2.4749536099890452. Committed epochs141/142 have finite losses and gradients,
with empty stderr. On the fixed validation subset, joint response score changes
18.364025→18.253355→18.263552; fixed500k NLL changes
-2.474954→-2.436736→-2.426523. All489updates per epoch hit gradient clip5;
this is execution progress, not established convergence or calibration.
The unbiased independent-group training score can be negative and is not clamped.
`startup_review.json` records these checks. Full audit16466517 and review16466518
subsequently completed; see the verified final results below. The covariance
one-shot is retired after final review; earlier disabled watches remain disabled.

The response labels condition on usability in both legs; NLL fits individually
usable rows. This potential target mismatch is unchanged. The grid still
uses r_blend, which is absent from the flow inputs, and has no explicit
intrinsic-|e| axis. Grid axes, radius supervision and lambda differ from the
earlier shape-grid recipe, so this is not an isolated grid-resolution test.
Population-mean response improvements would not establish detection/selection
calibration or actual SBSI inference bias. One trained model seed is used.
Frozen workers and failed artifacts must be preserved if a repair is needed.

## Covariance branch final review — 2026-09-12

Training16466516, audit16466517 and review16466518 completed exit0 with empty
stderr (49m52s,23m11s,19s). The selected checkpoint is epoch146, SHA256
`0e9bf0afbdd4833bbffb9aa486903ee99b0c8cc079990c5a1712b55a6323b37a`.
Training stopped at epoch166 after20epochs without a new best. The scheduler
reached1e-7 after the final update; the last trained epoch used1.953125e-7,
so there were **zero training epochs at minimum LR**. All12,714updates hit
gradient clip5. The stopping rule fired as implemented; optimizer convergence
is not established. The subset objective worsened from15.70954 at selection
to16.88635 at the last epoch. Over epochs160–166 the subset radius parallel
mean ranged0.85830–0.86329 and cross-0.02106–-0.01034. These late diagnostics
do not establish response stability. The full audit uses selected epoch146.

Additional independent Slurm review16467236 completed in36s, exit0, empty
stderr. Its frozen worker and results are in `final_review_v1/`. All16saved
physical difference arrays were re-aggregated independently with NumPy
bincount: maximum discrepancy in both216-cell and180-cell group sums was
exactly0. Independent normal-equation solves reproduce responses within
1.14e-13 and case/MC covariances within1.82e-12. Source, metadata, completion
and selected-checkpoint hashes pass. All1,708,854pairs/40cases and all16groups
are present; full NLL covers3,455,704rows. No new flow samples, model fitting
or changes to frozen scientific workers were needed for this review.

For comparison with earlier models, the following uses the **original180-cell
equal-cell average**, not the new216-cell weighting. Errors are case-jackknife
standard errors; model integration errors are reported separately.

| Component | Measured ± case SE | Predicted ± case SE | Paired residual SE | Model MC SE |
|---|---:|---:|---:|---:|
| R11 | 0.855467 ±0.004821 | 0.847558 ±0.000467 | 0.004714 | 0.000013 |
| R12 | -0.004568 ±0.005850 | -0.002545 ±0.000415 | 0.005733 | 0.000010 |
| R21 | 0.007015 ±0.006245 | -0.004952 ±0.000427 | 0.006171 | 0.000021 |
| R22 | 0.864072 ±0.006707 | 0.841546 ±0.000572 | 0.006648 | 0.000019 |
| Radius parallel, pixels/shear | 0.893799 ±0.006957 | 0.890275 ±0.000624 | 0.006857 | 0.000195 |
| Radius cross, pixels/shear | -0.006270 ±0.007526 | 0.000565 ±0.000378 | 0.007504 | 0.000289 |

The radius parallel residual is-0.003524 ±0.006857(case), with0.000195MC SE.
There are178/180positive predicted radius parallel cells versus175/180measured;
cell correlation is0.97680. The shape trace response proxy
100*(tr(R_measured)/tr(R_model)-1) is **+1.80187% ±0.33441%(case)**,
with0.00149percentage-point MC SE. This is matched self-response closure,
without detection/measurement cuts or R_blend, not actual SBSI inference m.

| Model | Full validation NLL | Original180-cell shape score | Shape trace proxy, % |
|---|---:|---:|---:|
| NLL-only epoch320 | -2.485437 | 1.137221 | 1.68741 ±0.33418 |
| Earlier grid shape lambda0.1 epoch169 | -2.467832 | 0.676304 | 1.49517 ±0.33315 |
| Grid-free shape lambda1 epoch200 | -2.481157 | 1.107654 | 1.52672 ±0.33403 |
| Current grid-free shape+radius lambda10 epoch154 | -2.474314 | 1.107612 | 1.85731 ±0.33561 |
| New grid216 shape+radius lambda1 epoch146 | -2.425763 | 0.704291 | 1.80187 ±0.33441 |

The new model's shape score improves36.41% versus the current grid-free
shape+radius model, while its mean shape trace changes little. Its full NLL
is worse by0.048551nat/row versus that model and0.059674versus NLL-only.
The new-minus-current radius parallel response is-0.0068303,
with paired case SE0.0004321 and paired MC SE0.0004448. The current model's
radius residual was already small (+0.0033064), so this mean radius change
does not establish a useful overall accuracy gain. The saved comparison
contains full paired case/MC contrasts against all four earlier models.

Under the **new216-cell equal weighting**, predictions are
(0.837172,-0.002092,-0.004736,0.829378,0.872493,0.000471), versus measured
(0.847696,-0.004491,0.011703,0.862202,0.885311,-0.007443). Its trace proxy is
2.60110% ±0.54418%(case), MC0.00204percentage points. This differs because
the population weights and targets differ, not because a different checkpoint
was evaluated. Radius positive counts are210/216 on both sides, correlation
0.95500. Full216-cell joint/shape/radius scores are0.773353/0.769247/0.591976;
the original180-cell scores are0.730765/0.704291/0.619775. Joint covariance
cross terms mean the marginal scores do not add. These full-population scores
must not be compared directly to the50k-subset selection score18.13209:
the latter uses independently refitted, much noisier subset targets with
full-population covariance weights.

All3,499,732,992audited physical draws were finite, with no material disk
exterior. There were315exact zero-radius values,0zero-flux values and9,288
literal disk-boundary values, retained without correction. Radius maximum
was7,631.70pixels. Rare numerical zeros and large tails already occur in the
older flows (current lambda10:454radius zeros/8flux zeros on the same draws),
so strict positive support is not reproduced at floating-point precision.
Moment agreement does not validate these tails or selection probabilities.

This covariance branch is complete and its tagged monitor is disabled/removed.
The independently authorized no-covariance companion and its monitor remain
unchanged. No ConstGold test, calibrated correction or model promotion follows
from this review.

## No-covariance companion — 2026-09-12

The owner requested a model without covariance after confirming that the first
grid216 branch already uses full within-cell response covariance. The new root is
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_shape_radius_lambda1_6x6x6_nocov_v1`.

Replace every cell's response precision matrix with identity6 in both training
and checkpoint selection. Thus the population objective is

    L = NLL + 1/(2*6*216) sum_B sum_a (R_model,Ba - R_data,Ba)^2.

The six entries are the same four dimensionless shape responses and two
measured-radius responses in pixels per unit shear. The pixel scale remains
0.2arcsec/pixel; no standardization, variance-only weights, occupancy weights,
or fitted scale matching are introduced. Lambda remains1. This comparison
changes both the weighting directions and the effective response-loss strength.
It is not a comparison at matched gradient strength or matched NLL cost.
The loss is unit dependent, as any raw squared error across these different
physical quantities must be. Full covariance and case/MC uncertainties remain
available for reporting, independently of their removal from the objective.

Reuse the exact frozen grid216 prepared arrays, measured targets, pair order,
population, subset and full-validation designs via a read-only-use symlink;
hash verification guards the original files. No preparation rerun or target
refit is needed. Start from the same circularized NLL epoch140 weights and
AdamW state, with all epoch seeds, draws, sampled cells/pairs, NLL batches,
learning-rate and clipping settings unchanged. Keep the same scheduler rule,
maximum epoch200 and stopping rule, now monitoring the unweighted objective.
The four-group fixed-subset score remains an independent-group cross estimate;
negative sampled values are allowed. Standalone unweighted shape and radius
scores obey joint=(4*shape+2*radius)/6.

New immutable copies of run.py, tests.py, review.py, preparation helpers and
launchers are hashed in the companion manifest. The original running branch
is unchanged. New tests check explicit unweighted cross MSE, its component
decomposition and unit dependence, covariance/flux independence, and identity
six-component gradient replay while preserving the NLL gradient. Existing
grid, population and optimizer tests remain in the validation job. AST and
Bash syntax checks passed before submission.

Initial validation16466717 caught that the generic validation summarizer rejects
two-component radius-only scores: one failed test,20passed. No training ran;
pending16466718–16466720 were cancelled. Preserved original workers, manifest
and failed logs. Repair1 changes the identity score helper to the existing
general cross-quadratic implementation supporting two components. The
mathematical loss, units, data, optimizer and seeds are unchanged; repair1 has
its own frozen workers, launchers and manifest.

Replacement validation16466734 passed all21tests in22.08s, empty stderr.
Training16466735 started on a V100 and committed epoch141: fixed500k NLL
-2.47495361→-2.47613518, unweighted subset response score
0.09766024→0.09748971, objective-2.37729337→-2.37864547. All losses/gradients
are finite;487/489updates clipped and stderr is empty. Parent NLL matches the
covariance parent exactly; all four physical-response groups agree within
1.3643e-12 maximum absolute difference (float64 reduction roundoff).
Saved parent_comparison.json and startup_review.json; one epoch establishes
execution progress only. A self-removing one-shot is due approximately
2026-09-12T19:24:43UTC, tag sbsi-grid216-nocov-once. Existing monitors and
unrelated cron entries were preserved.

Full audit16466736 and review16466737 are queued. The review also depends on
covariance review16466518, so both full audits exist before the direct comparison.
Both audits use the same16independent64-draw groups over all
1,708,854validation pairs/40cases and all3,455,704NLL rows. The companion
review reports paired case and integration differences on both common216-cell
and original180-cell definitions, both unweighted and covariance-weighted
diagnostic scores, response residuals and NLL changes. Scientific limitations
from the covariance branch apply unchanged; this is not selection-calibration
validation or actual SBSI inference.

### No-covariance training completed; full audit in progress — 2026-09-12

Bounded scheduled check at19:26UTC: training16466735 completed exit0 in
1h55m27s, selecting epoch182 and stopping at the epoch200 cap. Selected SHA
`07238562dcf9901a81608fa5409828cbe8e115a0e8f7e5c7f2721204debb07e3`.
The stopping criterion was not a demonstrated plateau; convergence remains
unestablished. All29,340 update gradients were finite,28,775 exceeded clip5,
and epochs180–200 trained at minimumLR1e-7. There were18epochs after the
selected checkpoint. Over the last10epochs the fixed500k NLL ranged
-2.47771682 to-2.47763063, while the selected objective was not improved.

At selected epoch182, fixed500k NLL improved from-2.47495361 to-2.47763936,
and the unweighted50k-subset response score from0.09766024 to0.09723244
(0.438% improvement). Standalone shape/radius scores are0.08048155/0.13073422,
versus parent0.08081782/0.13134507. These fixed-subset diagnostics are not
full-validation response scores or final mean-response estimates.

All21 validation tests and source/metadata/selected-checkpoint hashes were
verified. Audit16466736 started normally, with groups0–5 completed, finite
sample checks and empty stderr; review16466737 remains pending. Saved
`training_completion_check.json`. No repair, resubmission or recipe change
was needed. The original one-shot retired itself. A new bounded check in
`one_shot_check2/` is due21:26:52UTC, with the same dedicated
`sbsi-grid216-nocov-once` tag. Its syntax/premature-inert checks passed;
the separately running training-scene selection-table monitor and retired
covariance monitor were preserved. Final16-group review remains outstanding.

### No-covariance final review — 2026-09-12

Training16466735, full audit16466736 and comparison16466737 completed
successfully. Independent CPU review16468046 completed in33s with empty
stderr. Its frozen script and results live in the no-covariance experiment's
`final_review_v1/`; the original scientific workers and outputs are unchanged.
All16 raw group arrays reconstruct both216-cell and original180-cell
sufficient sums with exactly zero difference. Independent NumPy normal solves,
case deletion and integration covariance agree with the saved outputs to
1.14e-13 in responses and1.46e-11 in covariance (including noisy flux entries).
Source/metadata/completion/selected-checkpoint hashes pass. Parent NLL matches
exactly and all four parent physical-response groups agree within1.3643e-12
reduction roundoff. All21 original validation tests passed.

This is the full1,708,854-pair,40-case validation response audit, with
16 independent64-draw groups, original pair order and common random numbers
across flows. Full NLL uses all3,455,704 individual validation rows. Training
used6,834,259 pairs/160 cases and13,821,950 individual NLL rows. Responses
below use the original180 equal-cell definition unless explicitly marked216;
they are not pooled selected-catalogue responses. The response, units and
population conventions are those in
[the fixed recipe](#exact-six-component-response-loss) and
[CONVENTIONS §6c–6d](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).

| Model | Full validation NLL, nat/row (lower better) | Shape trace ratio proxy, percent ± case SE | Radius parallel predicted−measured, pixels/shear ± paired case SE |
|---|---:|---:|---:|
| Covariance grid lambda1, epoch146 | -2.425763 | +1.802 ±0.334 | -0.003524 ±0.006857 |
| No-covariance grid lambda1, epoch182 | -2.481027 | +1.581 ±0.334 | +0.003496 ±0.006844 |
| Previous grid-free shape+radius lambda10, epoch154 | -2.474314 | +1.857 ±0.336 | +0.003306 ±0.006726 |

The proxy is100*[trace(R_measured)/trace(R_model)−1]. Separate integration
SE is0.001494 and0.001323 percentage points for the covariance and
no-covariance proxies, respectively; radius parallel integration SE is
0.0001950 and0.0002919 pixels/shear. Case and integration errors are kept
separate. These are response-calibration proxies, not actual SBSI inference m.

No-covariance improves NLL by0.0552641 nat/row versus covariance and
0.00671261 versus previous grid-free lambda10. It remains0.00441029 worse
than the NLL-only continuation (-2.48543704). These are same-row comparisons;
they do not establish selected-response calibration.

The six no-covariance components are:

| Component | Measured ± case SE | Predicted ± case SE | Predicted−measured ± paired case SE | Separate model integration SE |
|---|---:|---:|---:|---:|
| R11 | 0.855467 ±0.004821 | 0.849853 ±0.000415 | -0.005615 ±0.004724 | 0.00001136 |
| R12 | -0.004568 ±0.005850 | -0.000990 ±0.000386 | +0.003578 ±0.005737 | 0.00001028 |
| R21 | +0.007015 ±0.006245 | -0.000705 ±0.000398 | -0.007720 ±0.006186 | 0.00001843 |
| R22 | 0.864072 ±0.006707 | 0.842924 ±0.000541 | -0.021148 ±0.006623 | 0.00001706 |
| Radius parallel, pixels/shear | 0.893799 ±0.006957 | 0.897294 ±0.000599 | +0.003496 ±0.006844 | 0.00029187 |
| Radius cross, pixels/shear | -0.006270 ±0.007526 | +0.000453 ±0.000412 | +0.006723 ±0.007433 | 0.00028210 |

Relative to covariance, the paired change in the180-cell shape proxy is
-0.220884 percentage points ±0.010450(case), with separate MC SE0.000964.
The predicted radius parallel changes by+0.00701943 ±0.00022460(case),
MC SE0.00023978. Both radius means already agree with measured within case
uncertainty. Across180 cells, all180 predicted parallel radius responses are
positive versus175 measured; cell correlation is0.96522. This does not imply
that every cell or feature slice is calibrated.

On the216-cell definition, measured shape is
[[0.84769597,-0.00449135],[0.01170308,0.86220231]], and no-covariance predicts
[[0.83975050,-0.00053567],[-0.00044474,0.83165730]]. The trace proxy is
+2.302878% ±0.544243%(case), MC0.001883 percentage points. Covariance gives
+2.601096% on this same definition. The paired change is-0.298218 percentage
points ±0.013631(case), MC0.001261. No-covariance radius parallel is0.882895
versus0.885311 measured: residual-0.002416 ±0.007905(case), MC0.000281.
Radius cross is0.000560 versus-0.007443 measured: residual+0.008003
±0.007843(case), MC0.000687. Predicted/measured positive-radius cell counts
are both210/216, with correlation0.94741. Different grid weighting changes
the target averages;180- and216-cell numbers must not be mixed.

Removing covariance does not improve the cell response fit relative to the
covariance branch, even when evaluated with the identity metric:

| Grid and diagnostic score (lower better) | Covariance flow | No-covariance flow |
|---|---:|---:|
| 216 cells, unweighted joint | 0.00798200 | 0.00903972 |
| 216 cells, covariance-weighted joint | 0.773353 | 1.066857 |
| 180 cells, unweighted joint | 0.00430261 | 0.00548719 |
| 180 cells, covariance-weighted joint | 0.730765 | 1.060430 |
| 180 cells, marginal covariance-weighted shape | 0.704291 | 1.092496 |
| 180 cells, marginal covariance-weighted radius | 0.619775 | 0.781613 |

Independent group-pair quadratic reconstruction verifies both joint metrics.
The unweighted marginal identity joint=(4*shape+2*radius)/6 passes for both
models on both grids. Full covariance marginal scores are not addends of the
joint covariance score. Same numerical lambda1 changes both direction and
effective penalty strength, and identity weighting depends on radius units.
This is a tradeoff between density fit and response fit, not evidence that
one weighting is uniformly superior at equal regularization strength.

The selected epoch is182; stopping was the epoch200 cap, with no established
convergence. All29,340 gradients are finite;28,775 (98.07%) exceed clip5.
Epochs180–200 trained at minimumLR1e-7. Late fixed-subset responses move little:
over epochs191–200, R11 is0.841322–0.841723, R22 is0.833163–0.833529 and
radius parallel0.883879–0.884648. Those50k-subset averages have different
targets from the full audit and do not demonstrate calibration or a proven
optimization plateau.

All3,499,732,992 sampled measurements were finite; no material disk exterior
was found. Retained numerical tails include267 zero-radius draws, no zero-flux
draws and4,483 literal disk-boundary values. Maximum radius is6549.1 pixels
and flux2.347e7 in the fixed flux units. These are support/tail limitations;
no samples were corrected or discarded. Flux response remains much noisier
than shape: on180 cells its predicted parallel/cross responses are3.6954/0.8302,
with MC SE0.5799/0.7704, versus measured2.5341/0.3861 with case SE0.5806/0.6427.

The audit uses validation scenes that selected the checkpoint and paired
survivors that differ from the individual-row NLL population. Omitted neighbour
descriptors and the absent intrinsic-ellipticity grid axis remain limitations.
No measured cuts, detector composition, coherent R_blend, ConstGold inference
or model promotion is part of this comparison. The separate covariance-flow
training-scene selection-table evaluation remains a distinct experiment.
Final review retires only `sbsi-grid216-nocov-once`; earlier disabled monitors
and unrelated cron entries are preserved.

## Validation response-versus-feature plots for both new flows — 2026-09-12

The owner requested the earlier feature plots remade for the new models.
Use selected covariance epoch146 and no-covariance epoch182, with the old
grid-free shape+radius lambda10 epoch154 as a reference. The external root is
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid216_validation_feature_plots_v1`.
Frozen manifests/workers and job receipts preserve provenance. CPU array
16467767 (two tasks) completed52s each; independent render/review16467768
completed51s. All exited0 with empty stderr. No new GPU draws, training,
target fit or change to either flow trial was needed.

Reuse the original1,708,854 validation pairs/40cases, all16 independent
64-draw streams and their exact original order, original feature-bin edges,
case assignments and baseline scalar frame. All32 saved per-pair difference
arrays reproduce their corresponding180-cell and216-cell sufficient sums
with maximum discrepancy exactly0. The plots independently refit unit-pair
no-intercept forward0-to0.05 WLS within each marginal feature bin; they do
not average response-grid cells. No measured cuts, detection weights or
R_blend are applied. All ten features and measured targets match the old
figure exactly. Radius responses remain measured-radius pixels/shear.

Seven pages cover R11/R12/R21/R22, radius parallel/cross and half the shape
trace, each against true magnitude, semi-major radius, circularized radius,
intrinsic |e|, Sersic n, intrinsic e1/e2 and near/far/maximum neighbour flux.
Black is measured, orange is covariance grid lambda1, blue is no-covariance
grid lambda1, grey dotted is the previous grid-free lambda10 flow. Upper
panels show predicted/measured curves with case errors; lower panels show
model-minus-measured with paired case-jackknife SE. Separate16-group MC
errors and paired model differences are retained in JSON/NPZ/CSV.
Independent normal-equation solves reproduce plotted values to4.45e-16;
paired case SE, integration SE, all bin counts and targets pass. Both main
PNGs were visually inspected. Completion and display hashes are recorded
in `completed.json` and `final_review.json`.

The covariance model improves the smallest circularized-size bin's half-trace
residual from-0.134144 to-0.107881 ±0.012451(case); no-covariance gives
-0.128194 ±0.012477. It introduces large shape deficits at high neighbour
flux: the highest near-flux bin is-0.128364 ±0.008892 for covariance,
versus-0.005380 ±0.008840 for no-covariance and+0.002811 for the old flow.
Highest far/max bins show covariance residuals-0.082219/-0.103233.
Thus a better aggregate covariance-weighted cell score does not imply
better agreement in every feature slice. These are marginal associations;
the two lambda1 objectives have different effective strengths and these
plots do not establish selection calibration or a unique causal mechanism.

Display/export directory:
`/home/z/Zekang.Zhang/SBSI/figures/flow_grid216_validation_features_v1`.
Main files: `shape_trace_half.png`, `radius_parallel.png`,
`all_response_features.pdf` (seven pages), and `binned_responses.csv`.
The original figure directory and all other experiments/monitors are unchanged.
