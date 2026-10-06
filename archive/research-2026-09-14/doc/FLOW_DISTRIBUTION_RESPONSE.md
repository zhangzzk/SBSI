# Paired distribution-response supervision

## Problem and scope

Matched galaxies provide (x, Y0, Y1), with Y the four measured outputs and
x the baseline galaxy properties. The objective is the conditional
distribution change over the measured shear interval, not a selection-cut
response, a histogram of pair velocities, or a fitted multiplicative-bias
correction. Retaining endpoints matters: identical changes at different
starting radii can produce different changes in probability density.

The property-stratified plots show that the approximately linear directional
mean response depends on size, brightness, and R_blend; this is compatible
with a property-dependent b*e approximation for a mean, but does not validate
its distribution response. See [the directional profiles](MEASURED_PAIR_RESPONSE_PLOTS.md#directional-response-split-by-properties).

The implementation includes tested loss primitives, a held-out diagnostic,
and optional training integration. The controlled pilot below has completed. It improves the radius feature-response
score but fails full ConstGold closure; see [the completed pilot](#completed-radius-distribution-pilot).
The full ConstGold |m|<=0.3% objective remains unmet.

## Features of the distribution change

For smooth feature vector phi and each actual paired shear separation h,

    T_i = [phi(Y_i1) - phi(Y_i0)] / h_i
    M_theta(x_i) = [E_theta phi(Y1 | x_i) - E_theta phi(Y0 | x_i)] / h_i

The conditional expectation of T is a feature representation of (p1-p0)/h.
Square loss against T therefore learns the conditional distribution secant
through these features, rather than only the mean of Y. Shared parameters
pool information across x; fixed input-property bins are not necessary.
The feature transform must precede averaging of model draws.

`sbsi/flow_distribution_response.py` contains fixed sine/cosine random Fourier
features for an equal mixture of Gaussian kernels. The diagnostic uses
standardized log radius, bandwidths 0.25, 0.5, 1, 2, 32 frequencies per band,
seed 501, and 256 total sine/cosine features. Both data and every model use
the original new parent's log-radius standardization. No radius threshold
or ConstGold statistic defines the features. The finite approximation does
not identify all possible distribution differences, and bandwidth dependence
must be checked before interpreting a trained result.

## Independent draw groups remove a Monte Carlo penalty

Let A and B be conditionally independent unbiased model estimates of M,
independent of the measured target noise. Each group uses common random
numbers across its two shear legs; the groups use separate random streams.
The primitive returns the per-row score

    A . B - (A + B) . T

Its conditional expectation is

    ||M - E[T | x]||^2 - ||E[T | x]||^2.

The second term is data-only. Omitting T.T removes an unnecessary noisy
constant; negative scores are valid. Reusing A in place of B would add the
model estimator's variance to the objective and could reward reduced
scatter. Independence is a caller obligation; shape checks cannot establish
it. Antithetic draws within a group are permitted, but antithetic partners
must not be split into supposedly independent groups.

The existing likelihood must still anchor the distributions themselves.
Matching finite secants does not identify the infinitesimal derivative at
zero without additional assumptions. A radius-only test does not constrain
joint shape–radius–magnitude dependence. Matching only galaxies measured in
both legs describes that matched population; it does not automatically
describe all galaxies measured in either individual leg. These remain
scientific limits, not implementation checks that have already been passed.

## Diagnostic and validation

`scripts/audit_flow_distribution_response.py` loads the existing 40-case
matched Re>0.5 validation population, samples 200k fixed rows (seed 8301),
and compares original new, old, strong paired, and final dispersion-weight-1
models. It evaluates two independent groups of 16 antithetic draws each at
the matched forward shear contexts. All model log-radius draws are converted
to the same reference coordinates before applying features. It saves the
frequencies, model hashes, matching counts, per-pair scores, and paired
whole-case bootstrap score differences. Scores are diagnostics of feature
response fit, not m or deployment criteria.

Artifact root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/distribution_response_v1/`.
The audit completed under Slurm **16327018** in 1:41, exit 0, empty stderr,
with output `baseline_audit/`.
An isolated `audit_env/` inherits py31 scientific packages and installs SBSI
editable without changing the shared interpreters. The launcher clears
PYTHONPATH and runs the repository's audit module.

`tests/test_flow_distribution_response.py` analytically verifies fixed RNG
behavior, feature normalization/translation-invariant kernel, sensitivity
to a spread change with identical means, absence of sampling-variance and
label-noise gradient penalties, actual per-row shear normalization, endpoint
coupling invariance of the mean feature secant, and invalid inputs. Initial
full suite **316 passed, 2 skipped** (16327012, 45.87 s); a test-only tensor
conversion warning was then removed. Full suite **16327019** passed 316/2
in 36.83 s. Training integration adds unequal pair-batch aggregation and
optimizer-update coverage; the final full suite **16327036** passed
**318 tests, 2 skipped**, in 37.53 s with no warnings.

## Completed baseline audit

All four models use the same 200k held-out pairs and feature coordinates.
Scores omit the common target-only constant; lower is better, including
negative values. They are not multiplicative bias or an absolute error norm.

| Model | Distribution-response score | Case SE | Naive single-group score |
|---|---:|---:|---:|
| Original new plain 100/100 | 0.856385 | 0.006508 | 1.466109 |
| Old mean-head 500/500 | -0.226956 | 0.003545 | -0.075388 |
| Strong paired transport | 0.409719 | 0.004749 | 0.779579 |
| Final dispersion weight 1 | 0.351368 | 0.004605 | 0.663527 |

The old-minus-new score is -1.083342 +/- 0.007333 (paired case SE), while
the strong paired and dispersion models improve over the new parent by
-0.446666 +/- 0.004142 and -0.505018 +/- 0.004243. This cut-free diagnostic
therefore distinguishes the observed old/new failure and its partial repair.
The larger naive scores demonstrate a material contribution from model
Monte Carlo variance that the independent-group score excludes. Target
squared feature secants average 20.33466, showing the noisy constant avoided
in the reported score. This audit motivates a pilot; it does not establish
that optimizing this finite feature set will close ConstGold.

## Controlled training pilot

`--distribution-response-weight` defaults to zero. A positive value requires
plain affine complete-flow response training, exactly two folded shear
catalogues, and a case-disjoint split. It reuses existing matched-pair loaders
and source/destination shear checks. Unmatched rows remain in the NLL loader;
zero-shear destination pairs are dropped/reported. It uses the configured
paired shear (0.05 in this pilot), checked against the original per-row norms
before shear folding. Features select standardized output 3 only. Training
uses independent random streams; validation uses deterministic independent
group seeds, with shared latent draws within each group's two shear legs.

The distribution score enters optimization, separately pair-count-weighted
train/validation diagnostics, history, full-objective checkpoint selection,
and final metadata. The feature frequencies themselves are saved in metadata.
No new network parameters or inference-time operations are introduced.

The first pilot fixes weight **1**, 16 draws per independent group, the
original new 100/100 parent, frozen input/output transforms, 17.28M uncapped
selected rows, seed 501, 160/40 case split, five epochs, batch size 4096,
LR 1e-5, NLL weight 1, and existing 100/100 mean-response penalties on the
6x6x5 grids. Paired-transport and dispersion penalties are off. Weight 1
is a first controlled scale for the normalized feature score, not selected
by a ConstGold m result. The candidate minimizes the complete held-out
training objective; SWA is retained separately. The finite feature set,
both-measured population, reused validation set, and single training seed
remain limitations.

External `trial_definition.json`, `jobs.json`, and launchers record:

| Stage | Slurm job |
|---|---:|
| Train and derivative/density audits | 16327062 |
| ConstGold domain audit | 16327063 |
| Full 100-case ConstGold | 16327064 |
| Matched paired audit | 16327065 |
| Exact-radius boundary diagnostic | 16327066 |
| Pooled m and whole-case bootstrap | 16327067 |
| Independent distribution-response audit | 16327068 |

The primary artifact is
`measurement_flow_dist1_r100_t100_full_re037_s501_swabase.pt` in the external
root. All wrappers pass shell/inline-Python syntax checks. Isolated
`train_env/` inherits sims1 packages; its initially too-old setuptools 58.1
produced UNKNOWN metadata, so setuptools was upgraded to 68.2.2 **only in
that task venv**, then SBSI 3.0.0 was installed correctly editable. Import
from /tmp resolves to this workspace. The shared live-job environments are
unchanged. Final ConstGold uses the established cuts, 64 CRN draws, response
convention, and original ratio/bootstrap scorer.

## Independent feature-bank and higher-draw check

Audit **16327111** completed in 2:33, exit 0, empty stderr. It uses the same
200k rows and model hashes, but a different Fourier bank (seed **1501**) and
**64 draws per independent group**. The script now exposes `--feature-seed`,
default 501; training settings are unchanged. Output is
`holdout_features_d64_audit/` in the external root.

| Model | Independent-bank score | Case SE |
|---|---:|---:|
| Original new plain 100/100 | 0.929329 | 0.006056 |
| Old mean-head 500/500 | -0.206279 | 0.002940 |
| Strong paired transport | 0.448229 | 0.004549 |
| Final dispersion weight 1 | 0.397812 | 0.004444 |

Old-minus-new is -1.135608 +/- 0.007149. The ordering and large separation
persist. This is a joint robustness check, not an isolated measurement of
draw convergence: both the feature bank and draw count changed, and absolute
scores from different banks need not agree. The feature bank is independent
of the training features, while the validation galaxies are reused.

Candidate check **16327123** completed after training 16327062, using this
same independent bank and 64 draws/group against the original parent. It is
diagnostic and does not select a training epoch. The final full test suite
after the audit-option change is **318 passed, 2 skipped**, no warnings,
37.27 s (**16327112**). No new training variant was launched for this check.

## Completed stronger-dispersion distribution audit

**16327459** completed in 1:16, exit 0, empty stderr. It uses the unchanged
independent bank (seed 1501), 64 draws per independent group, fixed 200k
held-out matched galaxies, and reference log-radius coordinates. Original
parent score and case SE reproduce the earlier independent-bank audit
**exactly**. Dispersion-10 score is **-0.05654097 +/- 0.00340198**, versus
parent **0.92932938 +/- 0.00605589**. Paired difference **-0.98587035 +/-
0.00675585**, 95% interval **[-0.99901915,-0.97268931]**. The previously
evaluated old flow remains better at -0.20627871, and dispersion-1 scored
+0.39781176 on this protocol. Output: `disp10_holdout_features_d64_audit/`.

Thus stronger suppression repairs radius-distribution response as well as
the near-boundary diagnostic, yet [full m remains +2.61%](FLOW_RESPONSE_DISPERSION.md#completed-weight-10-constgold-result).
The strong paired model has a worse radius-distribution score (+0.44822947)
but a smaller net full m (+1.08%); cancellation among response errors means
these orderings need not agree. This does not show that radius fidelity is
irrelevant or that this finite feature score certifies its exact derivative.
No new weight or model is chosen using this ConstGold ordering. The current
direct distribution-response pilot and its predefined audits remain active.

## Joint shape–radius audit

The optional audit argument `--target-index` can be repeated to include
multiple measured output coordinates. The default remains index 3 (log
radius). A fixed joint audit uses indices **0,1,3**: measured e1, e2, and
log flux radius, all converted from each model's output standardization to
the original parent's physical reference coordinates. Data and model use
the same selected coordinates and transform. The 256 fixed Gaussian Fourier
features are now drawn in three dimensions for this audit, retaining the
same bandwidths, feature seed 1501, 64 draws per independent group, and
200k matched held-out rows. It contains no measured selection threshold.

This probes association changes invisible to the radius marginal. Joint
and marginal feature scores have different kernels and target constants;
compare models within an audit, never their raw scores across coordinates.
The finite joint feature set still does not certify all distribution
changes. The pilot's training loss remains radius-only and unchanged.

Tests verify physical measurement coordinates in both legs, conversion
between different model standardizations, and a constructed association
change with exactly unchanged marginal distributions. Full suite
**325 passed, 2 skipped**, 48.47 s, **16327483**. Five-model joint baseline
**16327488** covers parent, old, strong paired, dispersion-1 and dispersion-10.
Default radius-only rerun **16327489** plus exact per-pair regression
verification **16327493** checks compatibility with the saved baseline.
Candidate joint audit **16327490** waits for the existing pilot. These
checks do not select a training epoch or change the configured likelihood.

## Completed joint baseline and unchanged marginal verification

Joint baseline **16327488** completed in 2:52, exit 0, empty stderr. All
models share reference coordinates, measured targets, 200k row identities,
three-dimensional feature bank (seed 1501), two independent groups of 64
antithetic draws, and the actual forward shear norms. Lower scores are
better; target-squared is omitted and averages 93.414589 in this bank.

| Model | Joint shape–radius score | Case SE | Difference from new parent | Paired SE |
|---|---:|---:|---:|---:|
| Original new plain 100/100 | -9.519654 | 0.043407 | 0 | 0 |
| Old mean-head 500/500 | -5.120929 | 0.033254 | +4.398725 | 0.030179 |
| Strong paired transport | -8.994996 | 0.040848 | +0.524657 | 0.008765 |
| Dispersion weight 1 | -8.593362 | 0.039636 | +0.926292 | 0.012918 |
| Dispersion weight 10 | -5.825494 | 0.038018 | +3.694159 | 0.033514 |

The old flow does **not** dominate every distribution-response diagnostic:
the new parent is better under this joint kernel despite its poorer radius
marginal and full ConstGold m. Stronger dispersion suppression improves the
radius marginal while worsening the joint score. Thus broadly suppressing
all output-distribution changes is not supported as uniformly physically
correct. These are aggregate finite-feature losses: they do not isolate
correlation error from marginal shape error, identify exact distribution
derivatives, or certify a hard-cut response. In particular, the ranking
alone does not prove that the new model's shape–radius association is correct.

The direct radius-distribution pilot is unchanged. Its already queued joint
check will show whether radius repair retains the parent's stronger joint
fit. No new training variant, response scaling, or ConstGold-selected weight
is introduced on the basis of these scores.

Default radius regression **16327489** completed in 1:01; verifier
**16327493** completed in 2 s, exit 0. **All 200,000 per-pair scores are
bitwise identical** to the pre-extension audit, along with row identities,
frequencies, model summary and uncertainties. `default_coordinate_regression/verification.json`
records maximum difference 0. This checks numerical compatibility of the
pending radius-only candidate audits after the optional joint extension.

## Completed radius-distribution pilot

Scheduled bounded follow-up on 2026-09-06: training 16327062, domain 16327063,
full 100-case evaluation 16327064_0, paired 16327065, boundary 16327066,
m scorer 16327067, and distribution audits 16327068/16327123/16327490
all completed with exit 0. No live job was restarted or waited on.
Candidate SHA-256 is
`e9ed21c8de65fc3ec9f70b0b04c3ed9385dbc43a690005127c4c9bab629bc980`.

Full ConstGold uses cases 40–139, true primary 18<mag<25.8 and
0.37<Re<1.5, measured |e|<0.6, MAG_AUTO<25.8, flux radius>=0.75 arcsec;
64 QMC common-random draws, seed 7301, antithetic shear +/-0.02.
Each leg independently detects and selects individual draws, then pools
its own numerator and denominator. See [conventions](CONVENTIONS.md).
The complete-population result is **R11_sim=0.6063833218070074**,
**R11_model=0.5275359820823432**, hence
**m=100*(R11_sim/R11_model-1)=+14.946343% +/-0.252100 percentage points**.
The 10,000 whole-case bootstrap (seed 20260902) gives 95% interval
**[14.457173%, 15.442400%]**. The 0.3% objective is unmet; best previously
verified full result remains strong paired **+1.075825% +/-0.219835 pp**.
No model promotion, empirical correction, or ConstGold-based weight tuning.

On the independent feature bank (1501), same 200k matched Re>0.5 validation
rows, and 64 draws per independent group:

| Diagnostic (lower is better) | Parent | Radius-distribution pilot | Paired change +/- case SE |
|---|---:|---:|---:|
| Log-radius feature response | 0.929329 | -0.194411 | -1.123740 +/-0.006274 |
| Joint e1/e2/log-radius feature response | -9.519654 | -7.987260 | +1.532394 +/-0.014454 |

The old model's radius score is -0.206279; the pilot approaches that score
without recovering full closure. Joint response regresses, while matched
balanced NLL is 0.325441 (parent 0.119628). These diagnostics use a different
population and stencil from ConstGold; finite feature agreement is not a
certificate of exact distribution response or selection closure.

The ConstGold ordered response error is +0.038341 before measured cuts and
-0.117188 in the additional selection shift, summing to -0.078847 complete
response error. The fixed matched-selected shape error is +0.009748.
These are response differences, not m or isolated causal component errors.
The matched observed-zero-leg radius diagnostic also remains partial:
near-boundary directional slope 0.251819 versus measured 0.498731;
radius-only selection response 0.028240 versus 0.055408. Its identity and
measured-reference match is exact for all 1,120,160 rows. It is not a
complete-population closure result.

Verification: actual checkpoint and all.json SHA-256 agree with m/domain/audit
metadata; 100 case identities, support cuts and sampling conventions checked;
pooled complete responses independently reconstructed from per-case weighted
sums using stdlib math.fsum, reproducing m. The completed scorer independently
reproduced the original response residual bootstrap SE. Parent scores exactly
reproduce previous audit results. Record:
`scheduled_check/completed_validation.json`. Full-output SHA-256:
`14ffff38fbcad3bbcdcd3d4f4b82b0bd316764a2b867263abd0dd548175d7359`.

Next bounded diagnostic: **16329884**, `audit_shape_marginal_holdout.sh`,
evaluates e1/e2 alone for parent, old, strong paired and the pilot on the
same independent bank protocol. This checks whether shape response itself
regressed before changing training. Scores from different kernels cannot be
subtracted to isolate an association error. No new training is launched.
The existing two-hour watcher is enabled for this job; yield until its
scheduled one-shot check. Single training seed and reused validation remain
limitations.

## Completed shape marginal audit

Owner-requested immediate bounded status check: **16329884** completed exit 0,
empty stderr. Same 200k held-out matched Re>0.5 pairs, 40 cases, independent
feature seed1501, 64 draws/group; features now use measured e1/e2 only.

| Model | Shape distribution-response score | Change from parent +/- paired case SE |
|---|---:|---:|
| New parent | -14.505576 | 0 |
| Old mean-head | -10.055954 | +4.449622 +/-0.035112 |
| Strong paired | -14.170868 | +0.334708 +/-0.008181 |
| Radius distribution pilot | -14.076301 | +0.429275 +/-0.009728 |

Lower is better. The radius pilot therefore worsens the shape marginal itself;
its joint-score regression cannot be attributed solely to changed shape-radius
association. The old model also fits this shape-response diagnostic worse than
the new parent despite better historical selection closure. No subtraction
between shape/radius/joint scores identifies a separate association error.
The finite-feature matched-pair metric is not complete ConstGold closure.

Actual model hashes, population/case equality with joint audit, feature/draw
settings, arithmetic differences and empty stderr verified; saved
`shape_marginal_d64_audit/verification.json`. No inference/bootstrap rerun.
Full m results remain unchanged. A next training design should constrain joint
output response while retaining the successful radius marginal constraint;
this audit does not yet establish that such a change reaches 0.3%.
