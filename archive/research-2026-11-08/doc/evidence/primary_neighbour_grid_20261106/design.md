# Primary-versus-neighbour scene replay design

## Status and purpose

The owner suggested drawing detections together with their surrounding scenes,
painting isolated grid cells, and comparing primary-only and coherently sheared
neighbour responses with the constant-shear result. This document makes that
proposal concrete while the existing-catalogue localization continues. No image
campaign has been launched by this design. Image generation/measurement would
be implemented in BlendEMU; SBSI would evaluate the fixed models and estimators.

The gain over the current random A/B split is substantial: each grid cell has
exactly one designated primary, so the primary contrast contains no same-group
neighbours. All other scene galaxies receive the neighbour shear, including
faint or undetected objects. Primary identities, neighbours, photometric band
mapping, PSF, circularized/major-axis size conversion, Sérsic treatment,
positions, pixel scale, detection settings and fitting priors remain fixed.

## Four shear corners and exact closure

Use h=0.02 in g1 initially. For every copied scene render four states
(g_primary,g_neighbours)=(+h,+h),(+h,-h),(-h,+h),(-h,-h). Keep g2=0.
For each state, F is the population weighted selected mean ellipticity, with
independent per-state detection/usability and radius>3px/magnitude<25 cuts.
Use sums of numerators and denominators over cells, not averages of individually
defined per-object response ratios.

    R_primary = [F++ - F-+ + F+- - F--] / (4h)
    R_neighbours = [F++ - F+- + F-+ - F--] / (4h)
    R_coherent = [F++ - F--] / (2h)

Then R_primary+R_neighbours=R_coherent exactly, including selection and
normalization changes. The primary contrast averages the two fixed neighbour
backgrounds +/-h; the neighbour contrast averages the two fixed primary
backgrounds. This is a finite-step decomposition, not a claim that either is
the derivative exactly at zero shear.

The literal axis-only alternative (primary +/-h with neighbours0, and neighbours
+/-h with primary0) also answers a useful question, but its sum need only agree
with the coherent finite difference to the appropriate nonlinear remainder.
If needed, add those four axis states later as a step/interaction check.
The four corners are the minimal exact decomposition with a coherent control.

## Sampling detections versus reproducing a population response

A random sample from the union of +/- detections is useful for a conditional
diagnostic, but it has conditioned on a noisy measured event. Fresh image noise
does not preserve that conditioning automatically. Never silently compare its
average response with the full constant-shear population or model prior.

For full-population closure, use the existing rendered truth-primary catalogue
as the finite sampling frame. Stratify by membership in the union of original
coherent detections, with known positive inclusion probabilities in both strata.
Oversampling detections is allowed; retain inverse-probability weights. Thus the
owner's detection-focused sample remains available as a labelled conditional
subset, while the weighted complete sample targets the original parent pool.
Use source cases1000/1040/1080/1120 initially, keeping all primary IDs and source
case/block identifiers. Report ambiguous or unmatched detection/input matches;
drop/reject those mappings explicitly rather than assigning zero responses.

Freeze the source list and sampling seed before the new shear interventions.
Do not select targets separately from each new leg, require common successful
fits for the main response, or preselect on measured radius/magnitude. Failed
or undetected targets have no numerator contribution; retain their event flags
and the population denominator construction. The fixed-common-U comparison is
an additional conditional diagnostic only.

## Coherent replay control before interpreting a split

For the exact same sampled primary IDs and weights, compare coherently sheared
grid measurements with the corresponding original full-image measurements.
This is a separate check from the algebraic four-corner sum. A passing sum alone
cannot establish that moving scenes to a grid preserves the measurement problem.

Preserve pixel subphases and relative positions. Include all rendered scene
members needed for the primary measurement and detection neighbourhood, not
just the emulator's nearest-K list. Keep the original model's K/radius cap when
predicting, so any truncation in that model remains part of the tested model.
Vary scene guard extent and cell separation in the operational pilot; quantify
their effects on target pixels, detection, photometry and fitted shapes. Original
untruncated Sérsic profiles require a convergence check, not a claim of perfectly
isolated finite-radius scenes. Retain local sky/background and detection context.

Use paired pixel-noise realizations across the four shear states. For direct
original-image replay, retain/reconstruct the original target-patch noise or
explicitly treat a fresh-noise comparison as an ensemble comparison. Copying a
noisy detection and then redrawing noise changes the conditional experiment.
Record precisely which comparison is attempted; do not claim pixel replay for
fresh noise. The coherent control gates any large split-response campaign.

## Fitter randomness and uncertainty

The existing fitter initializes an unseeded RandomState for each measurement.
Earlier identical-pixel repeated fits produced substantial response scatter;
see [fit audit](../constant_split_20261105/fit_report.json). Its projected
full-catalogue SE from independent fitter initializations alone was about0.0074
for one comparison. This is a warning about efficiency, not a sample-size estimate
for the proposed paired replay.

Give each primary/noise replicate an explicit recorded optimizer seed shared
across its four shear states, without changing the fitter, PSF model or prior.
Measure the original-image control with the same seeded implementation. Keep
the old entropy-seeded catalogues as a separate reference; do not assume shapes
will be bit-identical to those old fits. A second independent optimizer seed
on the pilot quantifies fit scatter. Negative-T solutions and failures retain
their original usability rules; do not introduce a new positivity cut.

Use paired model latent draws and the same extraction/selection definitions as
the measurements. Save per-primary weighted moments so response-ratio influence
functions and covariance of primary/neighbour/coherent errors can be estimated.
Resample source cases/spatial blocks, with repeated copies/noise realizations
nested inside their source primary; overlapping scenes are not independent
galaxies. More noise copies reduce conditional measurement noise but cannot
replace independent source scenes.

## Proposed operational pilot and scaling rule

Start with512 distinct primaries per source case, stratified256 detected-union
and256 complementary truth rows (2048 total), with recorded inclusion weights.
Four corners give8192 primary measurements per optimizer/noise replicate.
Include a second fit seed on the same pixels to estimate optimization variance;
additional noise replicates are a separate axis. This is an operational/variance
pilot, not enough by declaration for a0.3% calibration claim.

After replay/guard/estimator checks pass, use the measured variance components
to plan a fixed larger draw count. If the independent-draw component dominates,
N_new approximately equals N_pilot*(SE_pilot/SE_target)^2; retain scene/block and
optimizer variance separately rather than extrapolating all uncertainty as1/N.
Report precision of each component and their correlated sum. Target absolute
response precision first; translate to m using the actual model response.
Publish that budget before scaling instead of selecting a stopping point from
a favourable observed bias. Existing limits are500 CPUs and six GPUs; pack
workers and preserve scientific/pilot prerequisites. Broad17788286 stays held.

## Existing implementation boundary

The constant-scene preparation is in BlendEMU
`scripts/prepare_constant_split.py`, with rendering/detection through
`scripts/run_sim.py` and centred65-pixel fitting through `scripts/run_shape.py`.
The active source renderer is `MultiBand_ImSim_galsbi_resim/modules/Run.py`.
Its source catalogue and noise/configuration must be pinned and reused.
`audit_identical_galaxy_conditions.py` is not a replacement for this pipeline:
it bypasses detection and photometry cuts and used deliberately changed PSFs.
No archived launchers or PSF changes are required for this proposal.
