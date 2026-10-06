# Circularized size conditioning trial

## Question and geometric contract

The 2026-09-10 owner-authorized trial tests whether the negative physical-radius
response of the plain flow is partly caused by combining post-shear ellipticity
with a pre-shear semi-major radius. The available renderer constructs the
profile with circularized radius `a0 * sqrt(q0)` and then applies an
area-preserving shear. The historical renderer version of every archived image
has not been independently pinned; the mechanism remains a hypothesis.

The experimental feature `circularized_Re_input_p` is computed from original
`Re_input_p` and **unsheared** axis ratio, in arcsec. Both matched legs
must have exactly the same value. The sheared catalogue omits the axis-ratio
column; the loader's pre-fold baseline ellipticity copy supplies
`q0 = (1 - abs(e0))/(1 + abs(e0))` for both catalogues. Computing radius from the folded
ellipticity and original semi-major radius is explicitly the wrong operation.
No magnification is included in this contract.

`sbsi/flow_size_condition.py` implements the strict conversion and replaces only
the size column and its training-fitted standardization. The other standardized
condition columns are unchanged. The feature has its own checkpoint name;
loading a checkpoint and supplying only the old size column raises an error.
Callers must supply the invariant feature explicitly. It is not automatically
derived from possibly sheared ellipticity during model evaluation.

## Controlled NLL comparison

`scripts/trial_circularized_flow.py` has scheduler-only preparation and training
phases. It uses the existing physical disk architecture and train-fitted output
maps from NLL-only epoch 123, but **does not load its trained weights**. Two arms
start from identical fresh seed-501 weights:

- `semimajor`: the original eight conditions and original preprocessor;
- `circularized`: the same conditions with semi-major radius replaced by the
  invariant circularized radius and its training-only standardization.

Each arm uses fresh AdamW, initial LR `1e-4`, no weight decay, clip 5, batch 8192,
at most 140 epochs of 4 million rows without replacement per epoch. Both arms
use identical row-sampling seeds, batch permutations, and initial weights on the
same GPU allocation. LR halves after four plateau epochs using validation NLL,
absolute threshold `1e-4`, minimum `1e-6`. Training stops after 20 epochs without
a new best NLL once minimum LR is reached, or at the epoch cap. These conditions
do not establish mathematical convergence. All epochs and optimizer/history are
retained. This first trial has **zero response loss in both arms**.

Selection uses only NLL on the same fixed 500,000 validation rows (seed
900000504). A fresh fit is needed because changing the feature definition
changes the conditional density. The former checkpoint supplies common output
coordinate constants and architecture, not a warm start.

## Population and response ruler

The original source cuts use semi-major radius `Re > 0.37` and true magnitude
less than 25.8. These cuts, the 160/40 case split, all raw targets, NLL row order,
matched identities, actual forward `0 -> 0.05` shear vectors, and original
180-cell magnitude/semi-major-size/blending grid remain fixed. Preparation
reconstructs the original cached contexts, targets and row keys exactly before
replacing the size condition. Float64 raw shape and magnitude columns must also
match exactly. Recomputing natural log radius may differ by at most two float64
ULPs across CPU math implementations; preparation records counts and maximum
errors. Both training and response calculations use the original cached raw
targets without modification. Unmatched legs remain in the existing NLL row
population and are excluded from response pairs with original counts reported;
no new alignment failures are silently dropped.

After NLL selection, each arm receives full held-out NLL evaluation and a full
matched physical response audit: 64 antithetic draws, four independent latent
groups, seed 2370000000001, identical common random numbers across legs and
across arms. Shape responses use sky axes; scalar responses use the fixed
baseline intrinsic frame. Measured radius is in **pixels**, flux in instrumental
zeropoint-30 units. Radius is transformed before averaging. See
[extraction conventions](CONVENTIONS.md#6-response-estimators).

The scalar response frame reproduces the frozen target builder's baseline
ellipticity decoded from its original float32 standardized context in float64.
The circularized input itself uses the original higher-precision pre-fold
catalogue shape. These two sources serve different purposes: preserve the
existing response ruler exactly, while forming the invariant size from truth.

The audit retains eight physical responses, six-component shape/radius marginal
covariance diagnostics, full per-cell model and target values, equal-cell means,
radius-positive cell counts and correlation, paired delete-case uncertainties,
independent MC covariance, and physical support/extrema. Cross-arm uncertainty
uses group-wise CRN differences. Neither these diagnostics nor the response
targets select an epoch or tune the NLL recipe.

## Artifacts and limits

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_condition_v1`.

The frozen source, environment, manifest, data checksums/stat metadata, resource
snapshot, launchers and submission IDs live there. `prepared/results.json`
certifies alignment; `training/{semimajor,circularized}/results.json` holds arm
results; `training/results.json` and `paired_arm_difference.npz` compare them.
The preparation job must pass before training starts. Files are never silently
overwritten to resume a failed trial.

This is one seed and bounded training on the matched detected population. It
does not establish full scene sufficiency, infer shear, use ConstGold, or promote
a configured likelihood. A sign improvement must be read jointly with NLL,
cell-level residuals, case uncertainty and MC diagnostics.

## Completed comparison: 2026-09-10

Training job 16444056 completed both 140-epoch arms and their full audits in
3290 seconds. Selected epochs are 139 (semi-major) and 134 (circularized), using
only the prespecified fixed-subset validation NLL. Both arms had zero response
loss. Full validation covers 3,455,704 rows; responses use the original 180
cells and extraction convention in
[Population and response ruler](#population-and-response-ruler).

| Quantity | Semi-major condition | Circularized condition | Measured target |
| --- | ---: | ---: | ---: |
| Full validation NLL, nat/row | -2.31122631 | -2.47885475 | — |
| Ellipticity R11 | 0.87289122 | 0.84995508 | 0.85546743 |
| Ellipticity R12 | -0.00005691 | 0.00046169 | -0.00456796 |
| Ellipticity R21 | -0.00086160 | -0.00117539 | 0.00701502 |
| Ellipticity R22 | 0.85933764 | 0.84276861 | 0.86407222 |
| Radius parallel, pixels/unit shear | -0.95898247 | 0.92797388 | 0.89379852 |
| Radius cross, pixels/unit shear | -0.00021097 | 0.00058720 | -0.00627032 |
| Positive radius-parallel cells | 5/180 | 180/180 | 175/180 |
| Radius-parallel cell correlation with target | -0.657894 | 0.944513 | — |
| Radius-parallel cell residual RMS | 2.051894 | 0.169871 | — |

NLL improves by 0.16762844 nat/row with circularized conditioning, while the
radius sign reverses and the cell pattern closely follows the target. NLLs
share exactly the same output standardization and reference measure. No
response target was used to select either checkpoint. Post-run review verified
the frozen source/metadata hashes, input size/mtime records, identical initial
state dictionaries and output transforms, selected checkpoint hashes, finite
recorded losses/gradients, and the per-cell summaries against saved arrays.
The review and artifact hashes are in external `review.json`.

This strongly supports the proposed conditioning mismatch as a major source of
the original contradiction. A pre-shear semi-major radius combined with
post-shear ellipticity mixes two geometric states. The invariant circularized
radius retains the pre-shear axis-ratio information needed for the size
coordinate after shear is folded into ellipticity. On sheared rows it cannot
in general be reconstructed from the old supplied conditions alone, so this
is more than an invertible rescaling of those conditions. Here the corrected
size representation permits NLL-only training to learn the positive paired
response with improved likelihood. The old mean-head result therefore does
not establish an unavoidable likelihood cost for the correct response sign;
this comparison does not isolate every reason for that head's earlier cost.

The radius mean is still 0.03417536 above target (3.82%). Its paired-case
jackknife SE is 0.00686855; the separate MC SE is 0.00036286. The remaining
offset is appreciable relative to this conditional-on-fit case uncertainty.
Ellipticity R22 is also 0.02130360 low. Both arms reached the epoch cap, and
99.29%/99.77% of training steps were clipped; convergence and robustness across
initialization seeds remain unestablished. All audited samples were finite and
none materially outside the ellipticity disk, but the circularized arm produced
85 numerical zero-radius draws out of 874,933,248 draws, with a maximum radius
of 3738 pixels and heavy flux tails. These diagnostics preclude treating the
sign correction as full numerical or inference validation. The configured
likelihood remains unchanged.

## NLL continuation and shape-only branch: 2026-09-11

The owner authorized both directions. The new source-frozen experiment is
`physical_circularized_followup_v1` beside the original trial. Both branches
restore the circularized epoch140 weights and its committed AdamW state.
The original epoch134 NLL-best checkpoint remains the NLL incumbent. Neither
branch changes the circularized condition, physical output coordinates,
architecture, population, row identities, or original response ruler.

The first NLL improvement is continued optimization: preserve AdamW moments,
learning rate and ReduceLROnPlateau state and continue the exact epoch-dependent
row-sampling and minibatch seeds. Run epochs141 through at most320, retaining
the original minimum LR1e-6 and20-epoch no-improvement rule. Every epoch uses
4million NLL rows and batch8192, with gradient clip5 and zero weight decay.
Only the original fixed500k validation NLL chooses checkpoints, adjusts LR,
and stops training. This tests the unresolved optimization plateau before
introducing architecture or density-coordinate changes. Satisfying this
operational stopping rule will not by itself certify response convergence.

The shape branch adds lambda0.1 times the four-component ellipticity response
loss to the same NLL updates. It starts from the same weights, AdamW moments,
learning rate and NLL epoch seeds. Its scheduler is reset because the monitored
objective changes: factor0.5, patience4, absolute threshold1e-4, minimum LR1e-7;
stop after20epochs without a new best at minimum LR, or at epoch200. The
response is the unbiased independent-group cross quadratic, normalized by
four components and averaged equally over cells. Each optimizer update draws
12cells and128pairs per cell for each of two independent groups, with16
antithetic latent draws and common random numbers between legs. Exact chunked
gradient replay retains the NLL gradient. Radius and flux remain in the full
joint NLL and all diagnostics, but have no response-loss terms. The precision
is recomputed from the marginal4x4 ellipticity covariance, not sliced from a
larger inverse covariance. This is the first fixed penalty weight, not a tuned
claim of an optimal likelihood/response tradeoff.

Both branches use one fixed50k stratified validation-pair subset, with targets
refitted on those exact pairs and precision from the full validation-cell
shape covariance. Diagnostic draws use64antithetic samples and four independent
groups with common random numbers across checkpoints/branches. Shape evaluates
it every epoch for joint-objective selection; NLL evaluates it at initialization,
every20epochs, at the terminal checkpoint, and at the NLL-selected checkpoint,
without using response values for optimization decisions. Saved arrays retain
all eight response components so radius regressions remain visible.

After selection, both receive full3,455,704-row NLL and full1,708,854-pair audits
with64draws and16independent groups. Final seeds are independent of selection
and shared across branches. A dependent comparison retains paired-case and MC
uncertainty in their differences. The extra groups follow the shape-response
evaluation convention; they are not16independent trained models. Numerical
zero radii, disk rounding, and extrema are reported without clipping or
discarding samples; nonfinite values fail. No inference promotion is included.

`scripts/continue_circularized_flow.py` implements atomic model/optimizer
checkpoints and explicit resumption. The core regression checks continuation
against uninterrupted training, including exact AdamW moments and scheduler
state; a second verifies that a zero response weight leaves NLL updates
identical. Four-component loss tests verify marginalization and zero gradient
through scalar output differences. The already completed trial remains frozen.

### First hourly review: NLL audit complete, shape still training

The09:21UTC wake-up found NLL training at its epoch320 cap, with epoch320
also the NLL-selected checkpoint. The subsequent full audit completed all16
latent groups. Full validation NLL is-2.48543704, improving by0.00658229nat/row
over the original epoch134 selection. Fixed500k validation NLL improved from
the original best-2.47557757 to-2.48204970. The final20-epoch mean still improved
by0.000542nat/row over the preceding20epochs; the plateau criterion was not
reached. Across the continuation,99.22% of optimizer steps were clipped.

Using the frozen matched-forward, equal-cell conventions above, the final
ellipticity response is [[0.84965329,-0.00078738],[-0.00128765,0.84135211]]
against [[0.85546743,-0.00456796],[0.00701502,0.86407222]]. Radius parallel is
0.92915739 against0.89379852, with residual0.03535887, paired-case SE0.00682778
and separate integration SE0.00023658. All180 model cells remain positive;
radius cell correlation is0.94647 and residual RMS0.16708. The trace-response
calibration proxy is+1.6874%, versus the original+1.5842%; this is not an
end-to-end inference bias. Additional NLL optimization has not visibly solved
the remaining response discrepancy. The original and new audits use different
latent seeds and4versus16groups, so small changes are not paired-CRN estimates.

All audited samples were finite, with no material ellipticity-disk exterior.
The3,499,732,992 evaluated draws include249 numerical zero radii and2zero fluxes;
largest sampled radius and flux are5513.36pixels and2.263e7ZP30 units. These
draws remain in the statistics. Larger maxima than the previous4-group audit
do not alone establish worsening tails because the new audit has four times
as many draws.

Shape was observed through epoch173, with provisional joint-objective best
at169. Its fixed50k-subset score fell19.22510to18.65917, while NLL worsened by
0.01039758nat/row relative to the shared epoch140 start. Its full audit and
the dependent comparison are unfinished; subset scores are not comparable to
full-population scores. Both inspected histories contain finite losses and
gradients and both stderr files are empty. The hourly watch remains enabled;
no recipe, checkpoint-selection rule, or running job was changed. Detailed
review: external `hourly_watch/review_20260911_0921.json`.

### Completed follow-up comparison: 2026-09-11

An owner-requested status check found both branches and comparison16450329
completed with exit0:0 and empty stderr. Shape stopped at epoch189 after20
epochs without a new best and at the minimum learning rate1e-7; its selected
checkpoint is epoch169. This meets the declared operational plateau rule,
but does not establish optimizer convergence or calibrated inference. NLL
selected epoch320 at its cap and did not reach its plateau rule.

The full validation audit uses the matched-forward population and equal-cell
response ruler in [Population and response ruler](#population-and-response-ruler).
Both branches use the same16 latent groups with CRN,64 draws per group,
3,455,704 NLL rows and1,708,854 matched pairs.

| Quantity | NLL only, epoch320 | Grid shape, epoch169 | Measured target |
|---|---:|---:|---:|
| Full validation NLL | -2.48543704 | -2.46783172 | — |
| Shape response score | 1.13722085 | 0.67630394 | — |
| e R11 | 0.84965329 | 0.84773912 | 0.85546743 |
| e R12 | -0.00078738 | -0.00095081 | -0.00456796 |
| e R21 | -0.00128765 | -0.00164990 | 0.00701502 |
| e R22 | 0.84135211 | 0.84646921 | 0.86407222 |
| Radius parallel | 0.92915739 | 0.93481166 | 0.89379852 |
| Radius cross | 0.00034499 | -0.00004570 | -0.00627032 |
| Radius parallel cell RMS residual | 0.16708434 | 0.17669852 | — |
| Trace response proxy | +1.6874% | +1.4952% | — |

The covariance-weighted shape score falls40.53%, with NLL cost0.01760532
nat/row against the continued NLL branch. It is not a40.53% reduction in
inference bias: the trace proxy improves by only0.19224 percentage points.
Using ratios recomputed after deleting each validation case, its paired
change has SE0.00966 percentage points; the separate MC SE is0.000803.
The absolute trace proxies have paired-case SE0.33418 and0.33315 percentage
points. These proxies follow `m = R_sim/R_model - 1` in
`doc/CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention` and do not
measure SBSI inference bias. One trained seed was used per branch.

Grid shape leaves radius parallel4.5886% high, versus3.9560% for NLL only.
Its radius residual0.04101313 has paired-case SE0.00685197 and MC SE0.00012380.
The paired branch change is+0.00565426, with case SE0.00018662 and separate
MC SE0.00014924. Both models have positive radius response in180/180 cells;
shape's cell correlation is0.94101, slightly below NLL's0.94647. Shape
regularization therefore improved the shape score while worsening both NLL
and radius accuracy in this trial.

Late checkpoints still fluctuate. On shape's fixed selection subset,
epochs180–189 span R11=0.84493–0.84797, R22=0.84623–0.84790 and radius
parallel0.93401–0.93473. NLL's last saved subset diagnostics at280,300,320
span R11=0.84851–0.84975, R22=0.83866–0.84046 and radius0.92622–0.92701;
there is no clear late response-calibration trend. These subset means have
their own measured targets and must not be compared with the full targets
in the table. The full audit includes the selection subset, so it is not a
new, independent test population. Case uncertainty is conditional on fitted
models and already includes finite integration noise; do not add the two
reported SEs as independent errors.

Shape's3,499,732,992 draws were finite with zero material disk violations,
4800 literal boundary roundings,372 numerical zero radii and no zero flux.
Maximum sampled radius is5779.23 pixels and flux2.545e7 ZP30 units. All were
retained. Flux parallel/cross are4.17383/-0.15131 against2.53406/0.38610,
with residual case SE0.74841/0.79675 and MC SE0.37218/0.61323; these noisy
tail-sensitive diagnostics do not establish a flux improvement.

Verified selected checkpoint hashes and recomputed saved response averages
and paired trace uncertainties using small artifact arrays, without new
training or catalogue scans. Full results and paired covariance remain in
external `physical_circularized_followup_v1/results.json` and
`paired_branch_difference.npz`; review is `review_20260911_final.json`.
The owner stopped wake-ups before this check; the watch remains disabled
and its tagged cron entry absent. The separately authorized grid-free trial
continues without a scheduled monitor. No inference or model promotion was
performed.

### Why the shape score improves more than the mean response

An owner-requested decomposition of the saved full-validation arrays verifies
that the40.53% score reduction primarily reflects better cell-level agreement.
For each cell b, let r_b be its four-component model-minus-measured shape
response and P_b the inverse marginal shape covariance. The score is
mean_b(r_b^T P_b r_b)/8 minus its finite-latent MC correction. The reported
response matrix instead averages the signed cell responses. Opposite cell
errors can cancel in this mean while remaining large in the score.

| Component | Mean residual, NLL | Mean residual, shape | Cell RMS residual, NLL | Cell RMS residual, shape |
|---|---:|---:|---:|---:|
| R11 | -0.005814 | -0.007728 | 0.103908 | 0.081300 |
| R12 | +0.003781 | +0.003617 | 0.066731 | 0.066771 |
| R21 | -0.008303 | -0.008665 | 0.076914 | 0.076374 |
| R22 | -0.022720 | -0.017603 | 0.123023 | 0.098697 |

The covariance-weighted cell score improves in133/180 cells; unweighted
four-component squared error improves in134/180. The ten largest weighted
improvements account for38.76% of the net score reduction. Without covariance
weighting, squared error averaged across cells and components falls26.61%,
from0.00907513 to0.00666056. Its exact decomposition into squared mean residual
plus between-cell residual variance is:

- NLL: 0.00015831 + 0.00891682.
- Shape: 0.00011444 + 0.00654612.

Thus98.18% of the unweighted error reduction comes from the residual-variance
term and1.82% from the squared mean. This decomposition uses observed cell
residuals, including measurement uncertainty; it is not an estimate of
noise-free population heterogeneity. It also is not a decomposition of the
covariance-weighted score, whose per-cell matrices differ. The latter's MC
corrections are only0.000006298 and0.000006278, so changed integration-noise
subtraction does not explain its improvement.

The squared score's40.53% decrease corresponds to22.88% in its square-root
scale, not40.53% less response amplitude error. The equal-cell trace proxy
does improve modestly (+1.6874% to+1.4952%), but R11's mean residual worsens
while R22's improves. Better cell-level agreement and limited global-mean
calibration improvement can therefore coexist. Grid-free paired training
also forms errors before averaging and does not directly target only the
global mean response. No loss, selection rule or running job was changed.
