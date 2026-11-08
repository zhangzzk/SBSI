# Why GalSBI underfits the measured–intrinsic alignment relation

## Question and scope

Follow-up to the [FS2/GalSBI transfer investigation](../fs2_galsbi_transfer_20261105/findings.md#direct-conditional-density-failure-at-the-radius-boundary):
why does the adopted GalSBI flow reproduce the conditional shape–size relation
less accurately than the retained FS2 flow? This stage evaluates frozen
checkpoints and their training records. It does not retrain models or infer
a complete constant-shear bias from alignment alone.

The diagnostic is A = sum(e_measured dot e_intrinsic)/sum(|e_intrinsic|²),
at zero shear, within intrinsic circularized Re[0.15,0.25) arcsec and bins of
each branch's own measured radius. Report both MAG_AUTO<25 and<25.8. Compare
NLL-only/adopted checkpoints on40 training cases and40 validation cases per
family, with two16-draw antithetic banks and paired checkpoint random numbers.
The training40 are the first40 IDs of the original randomized training split.
They are not additional held-out cases. [Pre-execution rule](rule.md#fixed-diagnostic-before-execution).

## Frozen-checkpoint result: density underfitting already on training data

At measured radius2.7–2.8px, the adopted-flow minus simulation alignment is:

| Family / measured magnitude limit | Training40 | Validation40 |
|---|---:|---:|
| FS2 /25 |+0.03245 ±0.00465|+0.03638 ±0.00464|
| FS2 /25.8 |+0.04297 ±0.00339|+0.04474 ±0.00381|
| GalSBI /25 |**+0.37896 ±0.00443**|**+0.38140 ±0.00482**|
| GalSBI /25.8 |+0.14398 ±0.00274|+0.14426 ±0.00276|

The error is already present in the distribution used for training. It is
not primarily a train-to-validation shift of this statistic. The earlier
NLL-only GalSBI checkpoint is worse: +0.41067/+0.41339 at magnitude<25
(training/validation). The response-guard stage improves validation alignment
by0.03199 ±0.00029 but leaves most of the error. FS2's NLL-only checkpoint
already has a small validation error(+0.03294), before guard/refinement.

![Training versus validation at magnitude25](train_validation_mag25.png)

The count and shape statements are distinct. In the GalSBI validation bin,
simulation contains52,985 objects and the adopted flow predicts52,737.4
expected objects, a count difference of only-0.47%. Yet **74.09% of simulated
shapes have negative dot product with intrinsic shape**, versus49.96% of flow
draws. The corresponding FS2 fractions are90.07% and87.52%. The mismatch is
therefore visible in the direction of shape relative to the intrinsic shape,
even where total measured-size-bin counts are good. Negative dot product is
an observable shape statistic, not the same event as negative fitted T.

This GalSBI bin is supported by52,799 training objects in the40-case subset;
the whole intrinsic-size interval has2,723,498 matched training pairs and
2,722,294 validation pairs. It is not an empty training region. Nevertheless,
the particular radius/brightness bin is only about0.9% of all matched objects,
so a substantial local failure can receive a small fraction of global NLL
weight. That observation alone does not explain the family difference: the
corresponding FS2 bin is even less common.

## Within-population error versus different bin composition

For the same GalSBI validation bin at magnitude<25, partition by Sérsic n and
true magnitude into16 cells. An exact symmetric decomposition gives

- Within-cell alignment difference: **+0.38572**.
- Different cell proportions in the measured-radius bin: **-0.00432**.
- Total: **+0.38140**.

All cells contributing to this GalSBI bin have support on both sides.
At magnitude<25.8 the corresponding terms are+0.14646 and-0.00219.
Thus the observed error is not explained by the model putting the wrong
coarse Sérsic/brightness mixture into this radius bin. These cells still
contain variation in other inputs, so this is not an exact fixed-context
conditional-density comparison or a causal partition of shear bias.

### Brightness, crowding and orientation

At radius2.7–2.8px and measured magnitude<25, GalSBI alignment errors by
intrinsic magnitude are:

| True LSST-i magnitude | Alignment error ±case SE | Measured objects |
|---|---:|---:|
| <23 |+1.16669 ±0.03707|3,283|
|23–24 |+0.64927 ±0.01295|11,395|
|24–25 |+0.25089 ±0.00511|36,891|
|>=25 |+0.14018 ±0.02712|1,416|

Alignment is a regression coefficient, not a correlation bounded by1. The
brightest bin's measured/model alignments are-0.854/+0.313. This argues against
describing the failure solely as faint, low-S/N objects. It is strongest
conditionally among bright compact objects that land in this measured-radius
range; such objects are a much smaller part of the global training set.

With nbr_close_100<0.01, the error is+0.44150 ±0.00612 (30,615 measured
objects); the0.01–0.1 bin gives+0.34374 ±0.00701. It is not confined to large
values of the available crowding feature. This is not a proof of isolation
or absence of blends outside that summary.

Errors remain across all four Sérsic groups(+0.446,+0.359,+0.307,+0.215)
and all four intrinsic-ellipticity angle sectors(+0.367,+0.370,+0.390,+0.398).
The rotated-density log-score change in the aggregate bin is only
+0.00574 ±0.00330 for the adopted flow. We find no dominant single orientation
sector responsible for the failure. A galaxy-frame model may still ease
learning, but these results do not establish broken rotation symmetry as
the cause.

## Controlled learning intervention

A predeclared two-arm continuation tests whether increasing input-population
emphasis helps the same architecture. Both arms start from the selected
GalSBI NLL-only checkpoint with its AdamW moments, use the same40 training
cases,20 epochs of four million rows, reset fixed rate1e-4 and clipping5.
One arm samples all inputs, the other only intrinsic Re[0.15,0.25). Both use
all independently usable measurements of both legs within their input support;
neither selects by measured radius, measured magnitude or alignment in training.

This isolates the effect of training emphasis between the two arms under a
shared continuation recipe. A before/after gain also includes extra training,
the rate reset and the40-case subset. It cannot alone identify the original
optimization cause. Endpoints are fixed at20 epochs, with no selection by
the profile. Any modified weights are diagnostic artifacts only.

### Fixed-endpoint result

| GalSBI checkpoint | Validation alignment error at2.7–2.8px, mag<25 | Error at mag<25.8 | Change in full-population validation NLL |
|---|---:|---:|---:|
| Initial NLL-only |+0.41339 ±0.00484|+0.16716 ±0.00278|reference|
| Full-input continuation |+0.37928 ±0.00475|+0.15162 ±0.00278|+0.10396|
| Intrinsic-small continuation |+0.31265 ±0.00480|+0.12011 ±0.00272|+1.30433|

The additional reduction from targeted input sampling is0.06663 ±0.00078
at magnitude<25, paired over validation cases. Thus training emphasis can
change the relation within the same architecture. It removes only part of
the local error and seriously worsens global density quality. Neither arm
is a model improvement for deployment.

The bin counts also change:52,985 measured objects versus58,121.9(full)
and54,390.8(targeted) predicted objects, compared with53,917.5 for the initial
NLL model. Local alignment improvement should therefore not be read as an
isolated correction with all other density properties preserved.

Validation NLL uses the same fixed500k all-population sample in both arms;
the initial value is exactly-3.6941253137412335 in each, and the endpoints
are-3.5901628392083427/-2.3897906557119466. Training has large likelihood spikes,
including a targeted-arm mean NLL of144.2 in epoch10. This reset-rate
experiment therefore cannot establish a model-capacity limit or validate
input restriction as a remedy. It also cannot attribute the starting error
specifically to the original learning-rate schedule.

## Interpretation and next discriminating test

We can now locate the learning failure: **GalSBI's NLL density fit already
underfits the shape-direction distribution conditional on measured size,
on training as well as validation data.** The aggregate radius counts are
close, but the negative-shape fraction and alignment are wrong. The response
penalty did not create the problem and barely repairs it. Coarse population
composition, a single orientation sector and only the most crowded/faint
objects do not explain it.

What is still not established is why the same architecture fits FS2 much
better: optimization/conditioning, capacity for this changed joint density,
and the Cartesian representation remain competing explanations. Neither
similar global NLL nor acceptable marginal histograms certify this local
distribution. The bounded continuation shows that optimization and sampling
matter, but did not provide a stable fit with which to resolve those causes.

The next clean representation test is a paired Cartesian versus
intrinsic-frame fit with identical data, batches, optimizer budget and
validation cases, using a stable learning-rate schedule. Rotate both
shape outputs consistently, retain the full radius/flux density and proper
Jacobian, and handle near-zero intrinsic ellipticity explicitly. Judge the
whole alignment curve and density likelihood before a constant-shear test.
An explicit-frame model is a hypothesis to test, not an already justified
replacement. No such representation change was run in this stage.

## Architecture and preprocessing audit

The two recipes use identical feature lists and architecture settings:
intrinsic e1/e2, Sérsic n, magnitude, circularized Re, and three scalar crowding
features. Standardization adds eight missing-value indicators, giving16
network inputs; there is no feature clipping in the preprocessor. All
diagnostic raw inputs pass the frozen loader's finite-value checks.

The target is a joint distribution of measured g1,g2,FLUX_RADIUS and flux.
Ten alternating affine coupling transforms use256-wide, three-layer SiLU
networks. Radial tanh maps shape outputs into the unit disk; softplus maps
radius/flux to positive support. The family-specific coordinate scales and
means are fitted separately and carried by the checkpoints. Absolute NLL
values from the two families are consequently not directly comparable.

The model sees intrinsic orientation through e1/e2, but does not rotate shapes
into that frame or enforce rotation equivariance. It must learn the angular
dependence through ordinary Cartesian couplings. This is a potential learning
difficulty, not evidence that the architecture cannot represent the relation.
The scalar crowding inputs omit detailed neighbour geometry; a conditional
density can nevertheless learn its marginalized distribution. Such omissions
can make the density more complex without being a proof of impossibility.

## Objective and optimization audit

Both base runs use the same seed611, batch8192, four million rows per epoch,
140 NLL epochs, AdamW, initial rate1e-4, gradient norm clipping at5, and the
same16 global/radius/magnitude/joint response guards afterward. The NLL objective
fits the full four-output distribution; it does constrain shape–size dependence
in principle, but does not explicitly prioritize this alignment profile.
The response guards constrain aggregate selected moments, not this profile
within intrinsic-size, brightness, Sérsic and orientation strata.

The realized NLL schedules differ:

| Rate change | FS2 epoch | GalSBI epoch |
|---|---:|---:|
| 1e-4 to5e-5 |111|27|
| to2.5e-5 |122|45|
| to1.25e-5 |129|54|
| to6.25e-6 |not reached in NLL stage|97|

The scheduler halves the rate after four epochs without an improvement of
1e-4 in validation NLL. Best NLL checkpoints occur at139/137 of140 epochs;
neither endpoint alone establishes convergence of the alignment relation.
Both families clip essentially every late-training update, so clipping is
not a uniquely GalSBI failure. The earlier GalSBI learning-rate reduction is
a concrete optimization difference, but a controlled continuation is needed
to determine whether it causes the profile error. See
[training records](training_audit.json) and [frozen source hashes](source_audit.json).

## Uncertainty and interpretation limits

Case-bootstrap errors condition on frozen models and their existing image
realizations. They exclude retraining and shared-noise uncertainty. The
conditional-density rotation test rotates both measured/intrinsic shape by
90 degrees in ellipticity space; pixelized measurements need not be exactly
rotation invariant. Differences in its score therefore cannot alone prove
an incorrect architectural symmetry.

Alignment error is a diagnostic of the measured-shape distribution. It is
not m, a shear derivative, or an additive joint response error. A change that
improves this profile must still pass held-out density and shear-response
checks before it can be considered a calibration improvement.

## Reproduction and checks

Compact reports, scripts, plots and model/training receipts accompany this
document. Full case moments and diagnostic endpoint weights are under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_learning_20261105/`.
See [job IDs](jobs.json), [validation receipt](validation.json) and
[source hashes](provenance.json). Every non-overlapping intrinsic partition
closes against the total; negative/positive dot-product terms sum to alignment.
The original measured validation profiles replay within3.4e-16. Every frozen
checkpoint is hash-checked, the optimizer state is checked against its starting
weights, and the two pilot initial validation NLL values agree exactly.
No retained preset references either diagnostic endpoint.
