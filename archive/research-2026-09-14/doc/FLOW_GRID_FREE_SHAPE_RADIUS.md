# Grid-free shape and radius response, lambda 10

## User request and controlled starting point

On 2026-09-11 the owner requested adding radius response to the grid-free
model, increasing lambda to 10, and retraining. The new external experiment is
`physical_circularized_grid_free_shape_radius_lambda10_v2`, beside the completed
`physical_circularized_grid_free_shape_v2` under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1`.

Restore the same original circularized epoch 140 checkpoint AND committed
AdamW moments as the completed shape-only grid-free trial, with initial
learning rate 1.5625e-6. This is a separate retraining branch from the shared
parent, rather than resuming the completed epoch 200 branch. Architecture,
physical output map, conditions, source cuts, row/pair populations, 160/40-case
split, NLL epoch seeds and response sample/latent streams remain fixed.
Both the radius term and penalty coefficient change, so the comparison does
not isolate either change alone. Wake-ups remain off at the owner's request.

## Exact physical paired loss and radius units

Let h_i=|gamma_i|, let d_i be the matched measured physical change in
(e1,e2,radius), and let f_i be the corresponding difference of the flow's
conditional PHYSICAL means. Radius is exponentiated before averaging; this
does not penalize log radius, mean log radius, or exp(mean log radius).

Use the fixed output scales s=(1,1,1 pixel), with no fitted standardization,
covariance, empirical offset, clipping, or population-dependent radius weight:

    u_i = (f_i - d_i) / (h_i * s)
    L_shape  = E_pair[(u_e1^2 + u_e2^2) / 4]
    L_radius = E_pair[u_radius^2 / 4]
    L_total  = NLL + 10 * (L_shape + L_radius).

The original divisor 4 is retained on adding the third component. Thus the
ellipticity penalty is exactly ten times the prior lambda 1 term; averaging
over three components instead would silently reduce this factor to 20/3.
The radius scale is one physical pixel, matching the established response
ruler in `doc/FLOW_CIRCULARIZED_CONDITION.md#population-and-response-ruler`.
It is a declared weighting convention, not an assertion of optimal scale.
Changing radius units requires changing this scale by the same conversion.

For independent latent replicas A and C on the SAME sampled galaxies, use
mean_i(sum_j u_i,A,j*u_i,C,j/4). Replicas share CRN across the two legs within
each group. The independent-group product removes the extra model-dependent
MC variance penalty; it can be negative and must not be clamped. Target
measurement noise remains in the score. Each pair supervises its actual
directional radius response, with diverse applied directions across pairs;
there is no fitted parallel/cross matrix in training. The full audit retains
the original baseline-intrinsic-frame parallel/cross radius responses.
Flux remains in joint NLL and final diagnostics, with no flux response term.

## Frozen optimization and selection

Train at most 60 further epochs, 141–200, with 4 million NLL rows/epoch,
batch 8192, AdamW weight_decay=0 and gradient clip 5. Every update samples 1536
pairs uniformly with replacement, 16 antithetic draws per latent replica and
256-pair graph chunks. NumPy PCG64 chooses pairs separately from Torch's
latent streams. The previous deterministic seeds and draw budget are kept.

Select by fixed 500k-row validation NLL plus 10 times the total shape+radius
paired loss on the SAME uniform 50k validation-pair subset as the prior
grid-free trial. The callback reports total response loss, true shape-only
loss, radius-only loss, component scores and separate MC diagnostics; radius
does not get mislabeled as shape loss. Four groups of 64 antithetic draws use
fixed CRN across epochs. There is no grid in training or selection.

Reset ReduceLROnPlateau for the changed objective: factor 0.5, patience 4,
absolute threshold 1e-4, minimum LR 1e-7. Stop after 20 epochs without a new best
at minimum LR or at 200. The common parent participates in selection. Atomic
optimizer/model checkpoints support --resume under unchanged manifest and
metadata. An epoch cap or a noisy objective plateau does not prove convergence.

## Full validation and comparisons

After selection, evaluate all 3,455,704 validation NLL rows and 1,708,854 matched
pairs. Use the same 180-cell response ruler only for diagnostics, with 16
independent groups of 64 draws and CRN across completed comparison models.
The dependent CPU comparison includes NLL-only, grid shape lambda 0.1,
grid-free shape lambda 1 and this grid-free shape+radius lambda 10 model. It
retains paired-case and MC covariance of their response differences.

Report all shape entries, radius parallel/cross, full NLL, support/tails,
case and integration uncertainty, and stopping reason. The paired population
requires usable matched legs whereas NLL admits individually usable rows.
The full audit reuses validation data, including the selection subset; one
model seed is trained per recipe. No inference calibration, ConstGold run,
or model promotion is authorized by this retraining trial.

## Implementation and validation

`sbsi/flow_paired_shape.py` adds PairedResponsePopulation with an explicit
optional radius scale; the original PairedShapePopulation remains a strict
shape-only wrapper. Cross scores accept two or three outputs while keeping
divisor 4. Validation reports shape/radius components separately.
`scripts/train_circularized_grid_free.py` builds physical radius differences,
logs component losses and supports the fourth comparison model.
`scripts/continue_circularized_flow.py` selects the explicitly configured
response_loss key and coefficient, preserving legacy shape-only defaults.

Seven new tests in `tests/test_flow_paired_shape_radius.py` cover physical
radius and units, flux exclusion, invalid scales, unchanged shape coefficient,
absence of an MC variance gradient, exact chunked gradients, component
accounting, and checkpoint selection using shape PLUS radius. The focused
new/old paired and continuation suite passed 16 tests in 35.50 s. The frozen
full suite and preflight run under Slurm before training. Submission and
startup records reside in the experiment root; failed artifacts are retained
if an execution repair is needed. No scheduled monitor is created.

The initial v1 source package omitted the example catalogue because its asset
path was outside the earlier code-file hash list. Validation 16454027 caught
this in the existing checkout-resource test (442 passed, 2 skipped, 1 failed);
training 16454028 and comparison 16454029 never started. The v1 snapshot and
logs remain preserved. Replacement v2 adds and hashes the actual example and
configuration resources, without changing the scientific code or recipe,
and repeats the full validation before training.

## Launch and first completed epoch, 2026-09-11

Replacement validation job 16454100 passed 443 tests with 2 skips in 43.96 s,
then passed the frozen-input preflight. Training 16454101 started on a V100
at 15:47:54 Berlin time; comparison 16454102 depends on its successful finish.
The external manifest SHA256 is
`8230dc2c99280591b95f3d1d1fae6389752f9729ed3aa7b5ebea2a2cb89e933f`.

The loaded populations are 13,821,950 training NLL rows / 6,834,259 pairs and
3,455,704 validation NLL rows / 1,708,854 pairs. The restored epoch 140 NLL,
shape loss and exact uniform validation-pair subset match the previous
grid-free trial. Epoch 141 completed 489 updates over 4 million NLL rows in
163.08 s, with finite losses and gradients and empty stderr. All updates were
clipped at 5; pre-clipping norms ranged from 19.19 to 581.96, mean 51.65.

| Fixed-subset validation quantity | Parent epoch 140 | Epoch 141 |
| --- | ---: | ---: |
| NLL, 500k rows | -2.47495361 | -2.47263291 |
| Shape loss, 50k pairs | 6.42431881 | 6.42381668 |
| Radius loss, 50k pairs | 7.87724278 | 7.87722537 |
| Total response loss | 14.30156159 | 14.30104204 |
| NLL + 10 × total response loss | 140.54066225 | 140.53778754 |

Epoch 142 has started. This establishes execution progress only; the small
initial loss change does not establish a calibrated response or convergence.
Full validation and comparison remain pending. The external
`startup_review.json` preserves these checks and the first epoch record.
Wake-ups remain off. Repository documentation was updated after freezing;
the frozen scientific source and recipe remain unchanged.

## Completed full-validation comparison, 2026-09-11

Training/audit 16454101 completed at 17:59:20 Berlin time and comparison
16454102 at 18:00:01, both exit 0:0 with empty stderr. All 16 audit groups
completed. Selected epoch 154, SHA256
`9c5bb028437a4714b454d2bcd202243a2eda6263c36b4c70c46de3d12f725a6f`,
was restored for the full audit. Training stopped at epoch 174 after 20 epochs
without a new best and reaching minimum LR 1e-7. This is the configured
stopping plateau, not evidence of calibrated inference or established
optimization convergence.

All responses below use the same 1,708,854 matched validation pairs in 40
cases, with 180 equal-weight diagnostic cells and 16 independent groups of
64 antithetic latent draws. Both data and model use forward shear legs, with
CRN across legs and models; see
[CONVENTIONS sections 6c–6d](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).
Radius uses the fixed intrinsic parallel/cross frame in physical pixels.
NLL uses all 3,455,704 eligible validation rows. Errors are one-standard-error
case jackknives conditional on each fitted model; they omit retraining and
checkpoint-selection uncertainty.

| Response | Measured | Previous grid-free shape, lambda 1, epoch 200 | Grid-free shape + radius, lambda 10, epoch 154 |
| --- | ---: | ---: | ---: |
| R11 | 0.85547 ± 0.00482 | 0.85061 ± 0.00041 | 0.84818 ± 0.00041 |
| R12 | -0.00457 ± 0.00585 | -0.00090 ± 0.00038 | -0.00119 ± 0.00037 |
| R21 | 0.00702 ± 0.00625 | -0.00149 ± 0.00038 | -0.00050 ± 0.00037 |
| R22 | 0.86407 ± 0.00671 | 0.84307 ± 0.00054 | 0.84000 ± 0.00053 |
| Radius parallel | 0.89380 ± 0.00696 | 0.93255 ± 0.00050 | 0.89710 ± 0.00071 |
| Radius cross | -0.00627 ± 0.00753 | 0.00025 ± 0.00038 | -0.00076 ± 0.00069 |
| Full NLL | — | -2.48115708 | -2.47431414 |
| Common grid shape score | — | 1.10765402 | 1.10761189 |
| Mean-shape trace proxy | — | +1.5267% | +1.8573% |

Radius parallel residual falls from +0.03874864 to +0.00330640: fractional
mismatch 4.3353% to 0.3699%, a 91.47% reduction in absolute mean error. The
new paired residual case SE is 0.00672551, so the remaining mean discrepancy
is 0.49 case standard errors. The paired model change is -0.03544224 with
case SE 0.00047010 and separate MC SE 0.00046843. Radius cell RMS improves
23.31%, from 0.16965481 to 0.13010185, and correlation rises from 0.94569 to
0.97002. Predicted parallel response remains positive in all 180 cells.

The diagonal ellipticity means move farther below their targets, while the
common shape score is effectively unchanged. The new shape residuals are
[-0.00728495, +0.00337884, -0.00751053, -0.02406989], with paired-case SE
[0.00471974, 0.00573587, 0.00620185, 0.00662681]. The trace proxy is
+1.8573% ± 0.3356 percentage points (case), with separate MC SE 0.00100
percentage points. Relative to the old grid-free model it worsens by 0.3306
percentage points, paired-case SE 0.00241 and MC SE 0.000907 percentage
points. This uses tr(R_measured)/tr(R_model)-1 under
[CONVENTIONS section 8](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention)
and does not measure actual SBSI inference bias.

The full NLL cost is 0.00684294 nat/row versus the old grid-free model and
0.01112290 versus continued NLL-only. It remains 0.00648242 better than the
grid shape lambda 0.1 model, whose shape score is substantially better
(0.67630394). Radius and lambda changed together, with different selected
epochs and one trained seed per recipe; these are recipe comparisons.

The selection-subset response loss changes 14.30156159 to 14.29525668 at
epoch 154. Its paired latent-group jackknife change is -0.00630490 ±
0.00377885 (MC only); the late epoch 174 change is -0.00467034 ± 0.00477557.
Late loss flatness therefore does not certify the population optimum. Full
response matrices are audited for the selected model only. Every one of
16,626 updates was clipped at 5; pre-clipping gradient norms had median
43.64, 99th percentile 1899.61 and maximum 288010.09. No nonfinite loss or
gradient occurred, and the frozen optimizer/selection recipe was retained.

All 3,499,732,992 audit draws were finite with no material disk exterior;
8425 numerical boundary events, 454 zero radii and 8 zero fluxes were
retained. Largest radius was 9783.58 pixels and largest flux 1.9552e7 ZP30
units. Flux parallel/cross 2.42889/-1.11872 have paired residual case SE
0.62597/0.84910 and MC SE 0.31144/0.50673; these remain tail-sensitive
diagnostics, not validated flux calibration. Case errors already include
finite MC noise; do not add the separate MC error in quadrature.

External `final_review_v1/review.py` and `final_review_v1/results.json` retain
the four-model comparison, separate response covariances, paired changes,
history, support diagnostics, hashes and successful consistency checks.
The review only reads small saved artifacts. No new training, inference,
promotion or wake-up was started.

## Measured training versus validation responses, 2026-09-11

The owner requested the measured response means in both full splits. These
use the same forward estimator, 180 equal-weight cells, sky shape frame and
fixed intrinsic radius frame as the completed model comparison. They are
measured data responses; the grid-free objective itself samples pairs
uniformly. Errors are 1-sigma delete-one-case jackknives of the cell average,
retaining cross-cell covariance.

| Quantity | Training | Validation |
| --- | ---: | ---: |
| Matched pairs | 6,834,259 | 1,708,854 |
| Simulation cases | 160 | 40 |
| R11 | 0.84823 ± 0.00297 | 0.85547 ± 0.00482 |
| R12 | -0.00140 ± 0.00268 | -0.00457 ± 0.00585 |
| R21 | 0.00090 ± 0.00307 | 0.00702 ± 0.00625 |
| R22 | 0.85203 ± 0.00272 | 0.86407 ± 0.00671 |
| Radius parallel | 0.88322 ± 0.00400 | 0.89380 ± 0.00696 |
| Radius cross | -0.01004 ± 0.00406 | -0.00627 ± 0.00753 |
| (R11 + R22)/2 | 0.85013 ± 0.00192 | 0.85977 ± 0.00295 |

Validation minus training for the diagonal average is 0.00963825 ±
0.00351888, or 2.74 case standard errors. This includes the within-split
covariance between R11 and R22 and assumes independent, disjoint simulation
cases across splits. Individual R11/R22 differences are 1.28/1.66 standard
errors. Radius parallel difference is 0.01058017 ± 0.00802679, or 1.32
standard errors. These are split comparisons; they do not measure the
model's training response or establish the cause of its validation residual.

The eligible individual NLL row counts are 13,821,950 and 3,455,704; response
means use only the matched pairs above. External
`measured_split_comparison_v1/results.json` under this experiment records all
eight measured components including flux, full covariance, differences,
counts, source metadata and hashes of the compact arrays. Saved case-average
replicates permit reproducing the uncertainty calculation. Only target,
jackknife and count members of the frozen target NPZs were read. Means match
the original target-builder summary and the completed validation audit to
1e-12 absolute tolerance; no heavy catalogue/model evaluation was needed.

## Full-validation response versus features remake, 2026-09-12

The owner requested remaking predicted/measured shape and radius response
curves for the current lambda10 epoch154 model. External artifacts live in
`validation_feature_plots_v1/` under the experiment above. Preparation16466057,
16-group GPU array16466058 (at most four concurrent tasks), and dependent
summary16466059 use the existing frozen interpreter and model source.

Use all1,708,854 matched validation pairs in40 cases, with no additional
measured cuts, detector weighting or R_blend. The3,455,704 individual NLL rows
are not the plotted sample. All16 original64-draw groups, their exact seeds,
1024-pair batch boundaries and CRN across forward0→0.05 legs are retained.
Each group must reproduce the original saved180-cell cross statistics before
its per-pair physical differences are accepted. No model refit or selection.

Refit unit-pair, no-intercept WLS separately in up to eight marginal quantile bins
of true magnitude, semi-major radius, circularized radius, intrinsic |e|,
Sérsic index, e1, e2 and three neighbour-flux features. Shape uses the sky
shear design, radius the original fixed intrinsic-frame shear design.
Radius is measured flux radius in pixels, converted before sample averaging;
its response unit remains pixels per unit shear (0.2arcsec/pixel).
The baseline features decode original float32 contexts; circularized radius
uses the raw pre-fold invariant cache. These are marginal associations,
not controlled partial dependences or selected-sample responses.

Report all four shape entries, radius parallel/cross and the diagonal shape
average. Error bars use delete-one-case jackknife; preserve covariance for
the diagonal average and paired model-minus-measurement residuals. Report
separate16-group MC errors without subtracting them from case uncertainty.
These pooled within-bin fits differ from the older equal-cell global means.

Preparation16466057, all16 tasks of16466058, summary16466059 and independent
render/review16466168 completed exit0 with empty stderr. Every group reproduced
the old audit cross statistics (maximum absolute difference1.82e-12), and all
original3,499,732,992 draws and tail counters reproduced. An independent
normal-equation solve matches the reported bin responses to1.34e-15 maximum
absolute error; paired case errors and every bin's population accounting pass.
Near-neighbour flux has seven bins after ties merge; all other features have
eight. No rows are removed for plotting.

The main panels show measured/predicted curves and smaller paired residual
panels with1-SE case bars. Separate MC errors are saved in the tables rather
than folded into or subtracted from those bars. PNGs were visually inspected.
Display copies are in `figures/flow_lambda10_validation_features_v1/`:

- [Shape diagonal average](../figures/flow_lambda10_validation_features_v1/shape_trace_half.png).
- [Radius parallel](../figures/flow_lambda10_validation_features_v1/radius_parallel.png).
- [All seven response panels, PDF](../figures/flow_lambda10_validation_features_v1/all_response_features.pdf).
- [Bin values, counts and case/MC uncertainties, CSV](../figures/flow_lambda10_validation_features_v1/binned_responses.csv).

The smallest semi-major-radius bin (mean0.3900arcsec) has shape diagonal
average0.73814 predicted versus0.81968 measured, residual-0.08154±0.01242.
The smallest circularized-radius bin (mean0.2548arcsec) is0.58335 versus
0.71749, residual-0.13414±0.01246. Those two marginal groupings are correlated,
not independent causal tests. Radius broadly tracks the strong trends with
semi-major size and intrinsic |e|; significant local differences remain,
including+0.05182±0.01471 in the smallest circularized-radius bin.

Across all matched pairs, pooled WLS yields shape diagonal average0.809984
predicted versus0.824185 measured, and radius parallel0.936831 versus0.951228.
These are not the earlier equal-cell means. The curves characterize uncut
matched SELF-response trends; they do not establish selection calibration
or actual inference bias. External `sufficient.npz`, `binned_responses.json`,
`rendered_review.json` and `final_review.json` preserve the scientific sums,
uncertainties, source provenance and completion checks. No monitors were started.
