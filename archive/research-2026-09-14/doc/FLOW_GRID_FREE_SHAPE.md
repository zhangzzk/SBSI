# Grid-free paired ellipticity response experiment

## Objective and population

The owner requested a grid-free job on2026-09-11 and explicitly stopped task
wake-ups. `physical_circularized_grid_free_shape_v2` is a separate experiment
beside the frozen circularized comparison and its NLL/grid-shape follow-up.
Existing jobs continue; no recurring or one-shot wake-up is installed.

For each existing matched usable pair, let d_i=e_i,+-e_i,0 be its measured
two-component sky-frame ellipticity change and h_i=|gamma_i| the actual forward
shear amplitude. With mu_theta the physical conditional mean, define

    f_i = mu_theta(c_i,+)[:2] - mu_theta(c_i,0)[:2]
    L_pair = E_uniform_pair [ ||f_i-d_i||^2 / (4 h_i^2) ]
    L_total = NLL + 1.0 * L_pair.

The divisor4 is twice the two output components. Dividing by the actual
nonzero shear amplitude expresses directional response errors, with no fitted
scale or covariance floor. Both legs use their actual catalogue conditions.
This is finite-forward mean-change regression, not an inferred full2x2
per-object response matrix or an exact zero-shear derivative. Response frames,
matching and shared-draw conventions follow
[CONVENTIONS sections6c–6d](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).

Pairs are sampled uniformly with replacement. No bins, cell targets, covariance
precision, or cell occupancy weights enter training or checkpoint selection.
The original matched population and160/40 simulation-case split are retained;
the NLL still uses independently usable individual rows. Thus this does not
remove the pre-existing distinction between paired and row-wise selection.
True-size cuts and the circularized condition stay unchanged.

## Independent latent groups and statistical meaning

For each sampled pair, draw independent latent replicas A and C. Each replica
uses16antithetic draws and identical draws across that pair's two shear legs.
The stochastic loss is

    mean_i [ (f_hat_i,A-d_i) dot (f_hat_i,C-d_i) / (4 h_i^2) ].

The SAME galaxies appear in A and C; only the latent streams differ. Its
expectation is L_pair, without an extra model-dependent Monte Carlo variance
penalty. The product can be negative and is not clamped. Direct differentiable
products are accumulated in chunks before the optimizer step; existing NLL
gradients are retained. No scalar output enters the response product, although
the joint density retains radius and flux and shares network parameters.

This squares residuals before averaging galaxies. It is a stronger conditional
mean target than the grid loss's square of a cell average, and uses different
population weighting and units. Lambda1 is an explicit first-trial choice,
not a calibrated equivalent to the grid branch's lambda0.1. No penalty tuning
is performed. Measured pair noise remains in the score; its irreducible
conditional variance is not a demand that each noisy realization be matched
by the flow mean. Compatibility still depends on conditioning and population.

## Frozen training and selection recipe

Restore the same circularized epoch140 weights and committed AdamW moments as
the NLL and grid-shape follow-up. Initial LR1.5625e-6, zero weight decay,
gradient clip5; reset ReduceLROnPlateau for the new objective, factor0.5,
patience4, absolute threshold1e-4, minimum LR1e-7. Stop after20epochs without a
new best at minimum LR or at epoch200. The exact epoch-dependent NLL row and
minibatch seeds are retained: seed501,4million rows/epoch, batch8192.

Every NLL update samples1536pairs uniformly, with16draws in each latent group
and response graph chunks of256pairs. This gives751104pair occurrences/group
per489-update epoch. Training response seed starts at3250000000001 plus the
existing deterministic epoch/step stream offsets. Pair selection uses NumPy
PCG64; latent draws use Torch generators. Their distinct generator families
prevent pair choices from reusing a previous update's latent stream when seed
integers overlap. The first v1 attempt used Torch for both and was stopped
during startup review; its source, validation, logs and checkpoints are retained.

Checkpoint selection and LR updates use the original fixed500k-row validation
NLL plus the paired loss on a new uniform50k-pair validation subset, sampled
once without replacement using seed2671000501. There is no cell stratification.
Every epoch evaluates four independent groups of64draws with fixed CRN across
checkpoints, seed3260000000001, batches1024. The score averages all distinct
latent-group cross products separately for each pair. Saved per-pair residuals
and latent-group jackknife scores expose integration noise. This subset's
measured noise is fixed across checkpoints; it is not a case-confidence bound.
Epoch140 participates in selection. Atomic model and optimizer checkpoints
support exact resumption under unchanged metadata and recipe.

## Independent final audit and comparison

After selection, audit all3,455,704validation NLL rows and all1,708,854matched
validation pairs. The final response audit reuses the frozen180-cell ruler
ONLY for diagnostics, including shape, radius and flux. It uses64draws and16
independent groups, seed3170000000001, batches1024, with CRN across the earlier
NLL/grid-shape branches and this model. The final seeds differ from selection.
Report paired-case and separate MC uncertainties, support and tail diagnostics;
no draws are clipped or discarded to improve agreement.

A dependent CPU comparison reads the three completed audits and preserves
cross-branch case and MC covariance. Selection objectives differ, so compare
full validation NLL and the common response diagnostics, not raw loss values.
This trial does not promote a model or run ConstGold/SBSI inference. A mean-shape
calibration proxy is not an end-to-end shear-bias measurement.

## Implementation and validation

`sbsi/flow_paired_shape.py` implements the pair population, unbiased cross
product, chunked gradient and validation estimator.
`scripts/train_circularized_grid_free.py` supplies training and comparison;
`scripts/continue_circularized_flow.py` accepts an explicit paired epoch
function while preserving the original NLL/grid branches.
`tests/test_flow_paired_shape.py` checks the absence of a Monte Carlo variance
gradient, noncancellation across pairs, rotation invariance, forward difference
and shear normalization, scalar exclusion, direct/chunked gradient agreement,
validation group products, exact zero-weight NLL equivalence and resumed
optimizer/selection behavior. The focused suite passed9tests in18.34seconds.
Frozen full-suite validation and launch records live in the experiment root.

## Interim shape response: epochs185 and188

An owner-requested check found training16450781 healthy through epoch188,
with epoch185 selected by the unchanged NLL-plus-paired-loss rule. Latest
validation NLL was-2.47770566 versus-2.47495361 at140; paired loss6.42332693
versus6.42431881. No final full-population response audit was available yet.

Slurm diagnostic16453555 reconstructed shape mean differences from the stored
four-group residuals on the existing uniform50k validation pairs:
`Delta_mu = measured_delta + abs(gamma) * stored_residual`. It performed no
new flow evaluations. Both measured and reconstructed model responses were
fitted with the matched forward shear design, on exactly those pairs. The
following means assign equal weight to each of180 cells, with27–1820 pairs
per cell; this is a provisional subset audit, not the final full audit.

| Component | Starting epoch140 | Selected epoch185 | Latest audited epoch188 | Measured subset target |
|---|---:|---:|---:|---:|
| R11 | 0.85168643 | 0.85124295 | 0.85123526 | 0.87151186 |
| R12 | -0.00244254 | -0.00250866 | -0.00254048 | 0.06268981 |
| R21 | 0.00045490 | -0.00035236 | -0.00032115 | -0.01035074 |
| R22 | 0.84630634 | 0.84602805 | 0.84632495 | 0.87913410 |

The mean shape response barely changes. At188, changes from140 are
[-0.00045117,-0.00009794,-0.00077605,+0.00001862], with paired-case SE
[0.00012046,0.00010963,0.00015699,0.00014154] and separate MC SE
[0.00002064,0.00004380,0.00011346,0.00004639]. Small detected changes are not
evidence of materially improved calibration.

The subset's measured target is noisy: epoch188 model-minus-target paired-case
SE is [0.02835,0.04159,0.03441,0.04277]. Its unusual R12 target must not be
substituted for the full-population target. Absolute model MC SE is only
[0.00024268,0.00019185,0.00013057,0.00017917], but this does not measure the
uncertainty in extrapolating the subset mean to the full population. The
subset also participated in checkpoint selection. No inference-bias estimate
or convergence claim is supported by this interim audit.

Pooled pair regression is retained separately in the artifact; at188 its
model matrix is [[0.81466685,-0.00043557],[0.00109088,0.80691924]], against
[[0.84562731,-0.00244041],[-0.00872271,0.84026614]]. These use different
population weighting from the table and are not interchangeable with it.

External `physical_circularized_grid_free_shape_v2/interim_shape_epoch188_v1`
contains the standalone extraction script, submission, resources, results,
case/group arrays and input hashes. The diagnostic completed exit0:0 in25s
with empty stderr. Subset indices, exact paired-loss reconstruction, finite
values and fitted-residual identities passed; payload/raw hashes match the
frozen manifest. Training, selection, final audit recipe and disabled
wake-ups are unchanged.

## Validation history figure through epoch193

External `figures/history_epoch193_v1` contains a single three-panel figure
`grid_free_validation_history.png` (also PDF/SVG), its standalone `plot.py`,
and frozen `data.json`/`data.csv` with source-history SHA256. Panels separately
show total objective, paired-shape response loss and NLL on their actual
scales. The response panel is the loss term, not the response matrix.
All54 recorded epochs140–193 are shown without smoothing; stars mark193,
the incumbent selected by the total objective at snapshot time.

Validation objective changes3.949365to3.945545, response loss6.424319to6.423306,
and NLL-2.474954to-2.477761. The identity `objective = NLL + response_loss`
holds because lambda=1. These fixed-subset curves do not replace the final
full-population response audit or establish convergence.

## Response uncertainties and full audit startup

Owner requested uncertainties for each R and full grid-free validation.
Training reached and selected epoch200, with fixed-subset NLL-2.47777701 and
objective3.94548155. The epoch cap was reached, without the stopping plateau.
Selected checkpoint SHA256 is
`66eb212e37a82e70417cc4f59d619fa81a9e4a07586e70d32e7be8a5630c7a53`.
The original job16450781 then entered its configured full audit; startup
monitoring confirmed the first of16 independent latent groups completed and
stderr remained empty. The audit covers3,455,704 NLL rows and1,708,854 matched
pairs,64 draws per group. The existing comparison16450782 follows it.
No extra job or wake-up was needed.

The following are one-standard-error estimates from the40 delete-one-case
replicates, after averaging responses equally across180 cells. In component
order R11,R12,R21,R22,radius parallel,radius cross:

| Statistic | NLL only, full pairs | Grid shape, full pairs |
|---|---|---|
| Predicted R case SE | 0.000412,0.000383,0.000401,0.000539,0.000517,0.000376 | 0.000469,0.000417,0.000405,0.000580,0.000474,0.000353 |
| Measured R case SE | 0.004821,0.005850,0.006245,0.006707,0.006957,0.007526 | Same measured population |
| Paired residual case SE | 0.004722,0.005736,0.006183,0.006621,0.006828,0.007458 | 0.004715,0.005729,0.006181,0.006630,0.006852,0.007427 |

Grid-free epoch188's interim50k-pair shape case SEs are predicted
[0.003099,0.002529,0.002517,0.003344], measured
[0.028500,0.042269,0.034622,0.042022], and paired residual
[0.028351,0.041589,0.034413,0.042765]. Radius was not reconstructed from those
shape-only saved residuals. These interim errors must not be assigned to the
unfinished full-population audit or to epoch200.

Predicted-case SE describes sensitivity of the fitted model's response
average to which validation cases are included. It does not describe
uncertainty in learned parameters or independently trained models. Finite
MC noise is already present in the case replicates; the separately reported
MC SE is a diagnostic, not an independent term to add again. Measured and
predicted responses share galaxies, so significance of their discrepancy
requires the paired residual covariance rather than quadrature of the two
standalone standard errors. Model selection uncertainty is not included.

External `uncertainty_review_epoch200_v1/results.json` retains all eight
components, the separate predicted/measured/residual case covariances, MC
covariances, sample sizes, artifact hashes and audit startup state. Recomputed
full-audit paired and MC covariances agree with the published summaries.

## Completed full validation comparison: 2026-09-11

Grid-free training/audit16450781 completed at15:22:19 Berlin time, followed
by comparison16450782 at15:22:54. Both exited0:0 with empty stderr and all
completion artifacts. Epoch200 was selected at the cap; the declared
minimum-LR stopping plateau was not reached. The selected checkpoint hash
matches the preceding startup record. This section supersedes the interim
50k-pair audit for comparisons of the selected models.

Every response below uses the same full1,708,854 matched pairs in40 cases,
180 equally averaged cells and16 independent latent groups of64 draws, with
CRN across models. NLL uses all3,455,704 eligible validation rows. Errors are
one-standard-error case jackknives of each response average, conditional on
fitted models; paired residual and MC errors are retained separately.

| Response | Measured | NLL only, epoch320 | Grid shape lambda0.1, epoch169 | Grid-free lambda1, epoch200 |
|---|---:|---:|---:|---:|
| R11 | 0.85547 ± 0.00482 | 0.84965 ± 0.00041 | 0.84774 ± 0.00047 | 0.85061 ± 0.00041 |
| R12 | -0.00457 ± 0.00585 | -0.00079 ± 0.00038 | -0.00095 ± 0.00042 | -0.00090 ± 0.00038 |
| R21 | 0.00702 ± 0.00625 | -0.00129 ± 0.00040 | -0.00165 ± 0.00040 | -0.00149 ± 0.00038 |
| R22 | 0.86407 ± 0.00671 | 0.84135 ± 0.00054 | 0.84647 ± 0.00058 | 0.84307 ± 0.00054 |
| Radius parallel | 0.89380 ± 0.00696 | 0.92916 ± 0.00052 | 0.93481 ± 0.00047 | 0.93255 ± 0.00050 |
| Radius cross | -0.00627 ± 0.00753 | 0.00034 ± 0.00038 | -0.00005 ± 0.00035 | 0.00025 ± 0.00038 |
| Full NLL | — | -2.48543704 | -2.46783172 | -2.48115708 |
| Common grid shape score | — | 1.13722085 | 0.67630394 | 1.10765402 |
| Trace proxy | — | +1.6874% | +1.4952% | +1.5267% |

Grid-free's full shape response score is2.60% lower than continued NLL-only,
at an NLL cost of0.00427996nat/row. Relative to grid shape, its NLL improves
by0.01332536nat/row, while the common grid shape score is63.78% higher. The
grid-free objective used no grid in training/selection; this common score
is an evaluation diagnostic. Lambda values have different units and population
weighting, and NLL-only trained through320 versus200 for grid-free; these
comparisons do not isolate the causal effect of lambda alone.

Grid-free minus NLL-only mean shape changes are
[+0.00095788,-0.00010801,-0.00020296,+0.00171866], with paired-case SE
[0.00001728,0.00001354,0.00002991,0.00001789]. Its trace proxy improves by
0.16070 percentage points, with paired-case SE0.00142 and separate MC
SE0.000611 percentage points. Absolute grid-free trace proxy+1.5267% has
case SE0.3340 and MC SE0.00112 percentage points. These use the simulation/
model-minus-one convention in `doc/CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention`;
they do not measure actual SBSI inference bias.

Grid-free model-minus-measured shape residuals are
[-0.00485626,+0.00367258,-0.00850563,-0.02100145], with paired-case SE
[0.00472443,0.00573815,0.00619380,0.00662109]. Radius parallel residual is
+0.03874864 (4.3353% high), with paired-case SE0.00686214 and MC SE0.00026578.
Radius cross residual+0.00652315 has paired-case SE0.00744238 and MC
SE0.00027256. Radius remains positive in180/180 cells, correlation0.94569
and cell RMS residual0.16965. Relative to NLL-only, parallel rises0.00338977
with paired-case SE0.00007764 and MC SE0.00007056, worsening its agreement.

All3,499,732,992 audit draws were finite with no material disk exterior;
4832 literal boundary roundings and318 numerical zero radii were retained.
There were no zero fluxes. Largest radius was5846.62 pixels and largest flux
2.3734e7 ZP30 units. Flux parallel/cross are5.24239/0.60784 versus measured
2.53406/0.38610, with residual case SE0.72178/0.78890 and MC SE0.54711/0.74647.
These tail-sensitive diagnostics do not establish reliable flux calibration.

The final10 selection-subset epochs are nearly flat, but epoch200 still
sets the best joint objective. Capped optimization, one model seed per
recipe, finite integration and validation reuse prevent a convergence or
inference-calibration claim. Case covariance includes finite MC noise;
standalone model/measurement errors are correlated and must not be added
as if independent to assess their discrepancy.

External root `results.json`, `paired_minus_nll.npz` and
`paired_minus_grid_shape.npz` retain the full comparison. Review
`final_review_v1/results.json` records all separate response uncertainties,
paired changes, late histories, selected hashes and support diagnostics.
Reviewed arrays/histories are finite; common target/design/case arrays,
checkpoint hashes and reconstructed means/covariances agree. Wake-ups remain
disabled and no inference, promotion or further training was performed.
