# Current lambda10 flow: selection localization

## Question and fixed diagnostic recipe — 2026-09-12

The owner asks to investigate selection deeply after the
[separated diagnostics](CONSTGOLD_SEPARATED_RESPONSE.md#completed-r11-comparisons).
Those results establish a selection-probability response error. They do not
uniquely attribute the negative complete measured-shape closure bias: the
true-shape selection error has the opposite sign. This audit tests individual
cuts, their interactions, and the selected measured-minus-true shape moment.

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1/selection_localization_v1`.

Keep lambda10 shape+radius flow epoch154, ordinary-MSE trial9 blend lookup,
and the circularized-BCE detector fixed. Same50 ConstGold cases40–89,
4,854,422 truth objects, strict18<true magnitude<25.8 and
0.37<true semi-major Re<1.5arcsec, complete neighbouring scenes. The fixed
both-usable parent B contains4,338,426 pairs. Other controls use actual
independent usable masks U, or the frozen circularized detector p.

All central responses pool numerators and denominators across cases before
taking the per-leg ratio, then difference g1+0.02 and-0.02 with divisor0.04.
There is no response grid or equal-cell averaging in this audit. Follow
[response conventions](CONVENTIONS.md#6-response-estimators) and the
[response-closure m definition](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention).
These are not SBSI posterior-inference calibration measurements.

## Cut factorial and threshold curves

All eight combinations of |e_measured|<0.6, MAG_AUTO<25.8, and
FLUX_RADIUS>=3.75pixels (0.75arcsec) are evaluated from the same draws.
Cut bits1/2/4 denote shape/magnitude/radius. The model applies its own cuts;
it never inherits observed selection for the primary diagnostic.

Supplementary threshold curves keep the other two cuts fixed:

- Shape thresholds0.5,0.55,0.6,0.65,0.7.
- Magnitude thresholds25.3,25.5,25.8,26.0,26.3.
- Radius thresholds0.6,0.675,0.75,0.825,0.9arcsec.

Report each cut's effect with all other cuts present, and its order-averaged
Shapley allocation across all eight combinations. The latter is an exact
accounting convention; interactions prevent unique physical attribution.
No threshold is optimized or adopted from this diagnostic.

Preset subgroups use true magnitude[18,22,24,25,25.8], true semi-major
Re[0.37,0.5,0.7,1,1.5], and intrinsic |e|[0,0.2,0.4,0.6,0.8,1].
These report within-subgroup responses and are not additive global contributions.

## Completed compact decomposition of the existing run

A light read of the saved50-case sufficient statistics already establishes
the exact all-cut decomposition on fixed B. No new flow evaluation is used
for this table; errors are paired case bootstrap SE from the original
64draw seed7301 run, excluding full integration uncertainty.

| Selected response term R11 | Measured | Model | Model minus measured |
| --- | ---: | ---: | ---: |
| True ellipticity | 0.85009292 | 0.83368399 | -0.01640893 +/-0.00128398 |
| Measured minus true ellipticity | -0.23891038 | -0.21809386 | +0.02081652 +/-0.00163816 |
| Full measured ellipticity | 0.61118255 | 0.61559013 | +0.00440758 +/-0.00181376 |

Both large terms cancel partly; their covariance is retained in the total
uncertainty. The selected measurement-residual discrepancy is about12.7
case SE. It does not imply a12.7SE error in the unselected mean shape.
Matching selected counts or correcting true-shape selection alone therefore
does not guarantee improvement in complete measured-shape calibration.
The two sides have their own selections, so the second term mixes population
composition and conditional measurement error; it is not an independently
isolated causal measurement bias.

These matched-parent numbers differ from the independent-detection result:
the latter also contains the detection-boundary population. The matched
all-cut response-closure m is-0.71599 +/-0.29455%, not the complete-2.07%.
Machine-readable result: external compact_matched_decomposition.json,
including source hashes and R21. Arithmetic identity checked for every
bootstrap resample. This is an accounting result, not yet cut localization.

## Exact selected-moment decomposition

For each actual/model selection and its own selected denominator:

    R_selected_measured = R_selected_true + R_selected_(measured - true).

Consequently the model-minus-data full response residual is exactly the sum
of the true-shape selection residual and the selected measurement-residual
response. This identity uses the SAME weights and denominators in its two
terms. It is distinct from summing the three m values of the previous audit.

With model q=P(pass|x,g,D), mu=E[y|x,g,D], u=E[y I(pass)|x,g,D], further
controls compute q*mu, u-q*mu and model mu under actual cuts. These use
finite-MC products and must not be described as independently identified
data conditional covariances. Data contain one measured pair per object.
The core true-shape plus measurement-residual identity avoids this product.

Midpoint-frozen selection weights remove direct true-shape responsivity
when reporting the moving-weight selection contribution. Its response can
be zero or negative, so no multiplicative m is assigned to that increment.

## Uncertainty, provenance and launch

Use16 independent QMC groups, seeds7301–7316, each64 draws/object/leg;
CRN across shear legs and every cut. Average sufficient statistics across
groups before forming ratios. Case errors use10,000 common multinomial
bootstrap resamples, seed20260902. Separate integration errors use
delete-one-latent-group jackknife. Case errors already contain finite-MC
variation; do not add the two uncertainties blindly in quadrature.

Preparation re-reads matched measurement catalogues, reports unmatched and
invalid rows, and requires exact reproduction of prior raw detection,
usable detection, selected flags and measured shapes. Per-case frozen flow
conditions preserve the previous crowding and circularized size convention.
Seed7301 must reproduce previously saved mu/q/u before a case passes.
Workers and small inputs are hashed; large source inputs retain frozen
stat records, and prepared arrays get SHA256 receipts. Failed artifacts
are preserved. No training, model selection, correction, promotion or
scheduled wake-up is enabled.

Jobs: preparation16460158; case array16460159, at most4 concurrent V100
jobs; dependent summary16460160. The preparation uses CPU12/96GB, each
GPU task CPU8/64GB, and review CPU4/16GB. Resource scan found no login GPU,
345.88GB available RAM and6.7TB free; all catalogue/sampling work uses Slurm.

Initial validation: Python syntax and three shell syntax checks passed;
synthetic cut-boundary, intersection and exact response-identity tests passed.
Independent pooled-ratio, additive-cut and three-way interaction allocation
checks also passed. Preparation reproduced raw flags on its first five cases.
Task16460159_0 was released from the full-preparation dependency after its
case040 receipt completed; it passed prior seed7301 moment reproduction and
the first full cut/threshold evaluation, with empty stderr. Other tasks retain
the preparation dependency; final review still requires the complete array.
Execution status and completed scientific results are recorded below when
available. A successful Slurm exit alone is not a scientific result.

## Hourly wake-up authorized — 2026-09-12

Owner explicitly requested hourly wake-ups after the interim17-case status.
External hourly_watch/watch.json tracks16460158,16460159 and16460160,
returning to this existing task. First check03:16Europe/Berlin onSeptember12,
then hourly at minute16. Local cron tag: sbsi-selection-localization-hourly.
The built-in automation tool was unavailable; the existing bounded local
watcher is copied and hashed. The scientific manifest remains frozen;
submission.json records the current monitoring setting.

Saved prompt requires silence for unchanged/non-actionable states and
notifications for meaningful progress, completion, failure or required user
action. Actual execution repairs may continue within the frozen diagnostic.
After independently reviewing final outputs, disable this watch and remove
only its tagged cron entry. No earlier disabled watches are re-enabled.
Four watcher tests, due-time guard and exact crontab readback pass. Initial
scheduled delivery has not yet been observed.

## Completed full50 review — 2026-09-12

All50 cases and16 groups completed successfully. Preparation16460158 took33:13;
each array16460159 task took about6minutes; summary16460160 took57seconds.
Independent review16460408 completed86seconds, exit0:0. All stderr files are
empty. Its separate pooling implementation reproduces all23 configurations
and four response components for all three parents, case SE and MC SE.
All prepared/output/source hashes, prior seed7301 reproduction, cut nesting,
default-threshold identities, exact numerator identities, row counts and
16-group receipts pass. No additional model sampling or fitting in review.

The conclusion is a substantial ellipticity-selection discrepancy and a
shape–radius interaction. The magnitude cut at25.8 has little effect on the
full response error with the other cuts present. This does not establish
perfect flux calibration or attribute all independent-detection bias to cuts.

### Full eight-cut matched comparison

Errors below are model minus data R11 on fixed B, with paired case SE.
All50/4,338,426 matched pairs and16x64draws per leg. These replace the interim17-case estimates.

| Cuts | Measured R11 | Model R11 | Full error ± case SE | True-shape selection error ± case SE |
| --- | ---: | ---: | ---: | ---: |
| None | 0.960028 | 0.957787 | -0.002241 ± 0.002129 | +0.000000 ± 0.000000 |
| Shape | 0.570603 | 0.571892 | +0.001289 ± 0.001851 | -0.014298 ± 0.000989 |
| Magnitude | 0.966904 | 0.964432 | -0.002472 ± 0.002195 | +0.000835 ± 0.000187 |
| Shape + magnitude | 0.573264 | 0.572562 | -0.000702 ± 0.001878 | -0.013792 ± 0.001051 |
| Radius | 1.013241 | 1.010297 | -0.002944 ± 0.001828 | +0.001804 ± 0.000565 |
| Shape + radius | 0.611139 | 0.615670 | +0.004530 ± 0.001822 | -0.016588 ± 0.001278 |
| Magnitude + radius | 1.014062 | 1.011548 | -0.002513 ± 0.001774 | +0.002450 ± 0.000603 |
| All | 0.611183 | 0.615639 | +0.004456 ± 0.001805 | -0.016304 ± 0.001269 |

Shape-only true-e selection error is-0.01429849 +/-0.00098895, about14.5 case SE.
Its measured-minus-true error is+0.01558796 +/-0.00180689, hiding much of it
in the net measured response. Magnitude-only and radius-only true-e errors
are small but statistically detectable; neither should be called zero.

### Cut contributions and interactions

The following is E(all cuts)-E(other two), where E=R_model-R_data.
Each cut therefore has a different comparison population; these rows do not add.

| Added cut | Change in full response error | Paired case SE | Paired MC SE |
| --- | ---: | ---: | ---: |
| shape | +0.00696963 | 0.00224442 | 0.00002823 |
| magnitude | -0.00007407 | 0.00020871 | 0.00000245 |
| radius | +0.00515824 | 0.00149117 | 0.00003079 |

With the magnitude cut fixed, the shape–radius interaction
E(SMR)-E(SM)-E(MR)+E(M) is+0.00519984 +/-0.00179338, MCSE0.00002441.
Its true-shape part is-0.00412673 +/-0.00072829 and measurement-residual
part+0.00932658 +/-0.00165486. Thus the interaction itself also cancels partly.
Without magnitude the full interaction is+0.00394407 +/-0.00179302.

Order-averaged allocations of the matched full error change from no cuts
are shape+0.00504067 +/-0.00213069, magnitude-0.00036185 +/-0.00023056,
and radius+0.00201817 +/-0.00152608. Their sum is+0.00669699.
These Shapley values are an accounting choice; their case errors are correlated.

### Cancellation persists after reducing integration uncertainty

For all cuts on B, the exact error decomposition is:
- Selected true e: -0.01630425 +/-0.00126855 case, MCSE0.00003362.
- Selected measured-minus-true e: +0.02076041 +/-0.00164854 case, MCSE0.00003020.
- Full measured e: +0.00445616 +/-0.00180512 case, MCSE0.00002868.

The components share their respective selected denominators. Midpoint-frozen
weights leave a moving-selection residual-0.01664783 +/-0.00127166.
The effect is not explained by direct intrinsic-shape responsivity.

### Threshold and subgroup findings

Relaxing measured |e| from0.5 to0.7 changes the true-shape selection error
from-0.02205825 to-0.00782538; it remains negative. The full error is
nonmonotonic because the measurement-residual component changes too.
Changing minimum measured radius from0.6 to0.9arcsec makes the true-shape
error more negative (-0.01371236 to-0.02233089), while the opposing component
grows. At0.6arcsec the full error is-0.00009401 +/-0.00187071 although the
true-shape error remains-0.01371236: a nearly zero total is not calibration.
Relative to the default0.75arcsec cut, this full-error change is
-0.00455017 +/-0.00146834 with MCSE0.00002997. No cut change is adopted.

Tightening magnitude to25.3 reduces the full error by0.00235577 +/-0.00065743
(MCSE0.00001148), while its true-shape selection error stays about-0.0162.
This changes the selected population; the weak effect of removing the default
25.8 cut must not be generalized to every magnitude threshold.

Selection discrepancies grow with intrinsic |e|: true-e errors for bins
[0,0.2),[0.2,0.4),[0.4,0.6),[0.6,0.8) are respectively-0.00066712,
-0.00948014,-0.03210986,-0.09296055 with caseSE0.00071923,0.00137504,
0.00377768,0.01334729. The last bin has about94,000 selected objects per leg.
The[0.8,1) tail has only1716/1708 selected objects and SE0.1344; avoid strong
inferences from it. Full measured errors also vary with true size and brightness:
true magnitude18–22 gives-0.00601093 +/-0.00294203;25–25.8 gives
+0.01041039 +/-0.00361757. Subgroups are diagnostic within-bin ratios, not
additive global contributions; exploratory comparisons have no multiplicity correction.

### Selection does not explain the whole independent-detection bias

| Parent/control | No measured cuts: m percent ± case SE | All cuts: m percent ± case SE |
| --- | ---: | ---: |
| Fixed matched detections | +0.23396 ± 0.22227 | -0.72383 ± 0.29317 |
| Actual independent usable detections | -0.53844 ± 0.22730 | -1.36104 ± 0.30462 |
| Circularized predicted detections | -1.12894 ± 0.22421 | -2.06772 ± 0.30370 |

Full composed predictedR11=0.61816017 versus measured0.60537833.
All-cut MC errors on m are0.00462/0.00461/0.00471 percentage points for B/U/p.
The actual-detection oracle remains biased, and the model already has a
negative closure m before measured cuts when detections are independent.

An exact telescoping path from matched uncut to the complete calculation
has response-error terms-0.00224083 (matched uncut),+0.00669699 (add cuts
on B),+0.00389699 (include one-leg detections),+0.00442869 (replace actual
detection weights). They sum+0.01278184. Dividing each by the same final
model response with a minus sign gives+0.36250,-1.08337,-0.63042,-0.71643
percentage points, summing-2.06772%. This depends on the chosen path and
is not a unique physical separation or a sum of the earlier three diagnostic m values.

### Statistical interpretation and limits

The evidence localizes a failure in selection-weighted moments of the joint
measured ellipticity/radius distribution, including a strong error in
ellipticity-cut probabilities. A correct unselected mean response and a good
mean radius response do not determine probability flow across either cut
or the ellipticities carried across it. The tests do not uniquely distinguish
conditional scatter, correlation, missing context, blending response or
detection-conditioned transfer. B itself is a survivor condition; it is not
an unbiased substitute for the full detection population. The distinct U/p
controls expose remaining boundary effects. No flux-response penalty or
other training change follows automatically from this audit.

All9,941,856,256 model draws were finite. Retained5107 zero radii and23 zero
fluxes; maxima9564.78pixels and2.7998e7 instrumental flux. These rare tails
remain a separate physical-distribution concern. Matching totals across100
legs: zero crossmatch rows without shape,324788 shape detections without a
crossmatch excluded,22,716,686 crossmatched objects outside the fixed truth
support excluded. Raw/usable flags reproduce the previous catalogue audit.
All figures use paired case SE; separate integration errors are not added
blindly in quadrature. Only g1 is probed;50 reused diagnostic cases and one
fitted flow/blend/detector recipe cannot establish general inference calibration.

### Artifacts and review receipts

External final_review.json stores independent paired case/MC contrasts,
interactions, exact path, counts, tails and hashes; review_sufficient.npz
retains compact per-case/per-group sums. plot_review.py produces
selection_localization.png and selection_localization.pdf; PNG visually reviewed.
Results SHA6277b285f4b597b381389147190505d6011630b08d82475d32840972009f77da.
Independent-review SHA438b49d3e43614451cdc24df7d09cda125829392abffb70db44d1d86f7f047d9.
Reviewer SHA6add263ca18a1d0d0d36df9d18ed0c5f901cc1727197d38c1b5057c87d079109.
No model, selection threshold or configured inference likelihood was changed.

Final cleanup: independent review completed and hourly watch disabled;
removed only sbsi-selection-localization-hourly from cron, with exact readback
and archived prior settings. Earlier disabled watches remain disabled.
completion_review.json records successful checks and all final artifact hashes.
There are no remaining stages in this diagnostic.

## Origin of the remaining bias without the ellipticity cut — 2026-09-12

No e cut, retaining magnitude<25.8 and measured radius>=0.75arcsec. The
full model has m=-0.93410 +/-0.17621%, while actual independent detection
flags give-0.41862 +/-0.17552% and matched detections+0.24848 +/-0.17542%.

Compact no_e_accounting.py/json computes an exact telescoping path, with
paired case and separate MC errors. All percentage-point contributions below
use the SAME final model response1.01407133 as denominator, so they add;
they are not the separate diagnostic m ratios just quoted.

| Step | Added response error ± case SE | Contribution to final m, percentage points ± case SE |
| --- | ---: | ---: |
| Fixed matched population | -0.00251347 ± 0.00177433 | +0.24786 ± 0.17498 |
| Include detections present in only one leg | +0.00673664 ± 0.00039878 | -0.66432 ± 0.03932 |
| Replace actual detection masks with probabilities | +0.00524924 ± 0.00015151 | -0.51764 ± 0.01495 |

Opening the population changes measuredR11 by-0.00946265, but changes
oracle-weighted predictedR11 by only-0.00272601. Selected counts increase
3,345,124/3,345,234 to3,357,815/3,357,879:12,691/12,645 single-leg-only
selected detections, about0.38% per leg. Their asymmetry matters because the
response divides the leg-mean difference by0.04. The second step also changes
normalization; it is not a direct sum of their shapes under a fixed denominator.

This supports a problem with detection-boundary populations and the joint
detection/measurement prediction. Actual masks leave a residual, so correcting
the detector alone is insufficient evidence of a solution. Conversely the
probability-weight replacement term is not exclusively a detector derivative
error: it also changes which conditional flow/blend predictions are weighted.
Good detector calibration with true ellipticity can coexist with inaccurate
measured-shape-weighted moments.

Noise, neighbour geometry or other unresolved scene information could couple
detection and measured shape in ways current conditional moments miss.
These are candidate mechanisms, not separately identified causes. The flow's
NLL does include individual detected rows; paired response regularization
does not directly constrain the unmatched boundary. A single-leg conditional
model need not reproduce every subgroup defined by both-leg detection flags,
but its complete aggregate must be calibrated. No such subgroup target,
model change or new monitor was introduced here.

## Deeper interpretation of the remaining bias — 2026-09-12

Owner requests conceptual analysis of the remaining discrepancy. New light
`remaining_bias_moments.py/json` reduces the existing compact50-case/16-group
sums; no raw catalogue scan, new flow draws, training or scheduled jobs.
It uses the full p prediction against actual U and the existing paired10,000
case bootstrap seed20260902. All entries below are model-minus-data response
errors. The measured-minus-true residual includes measurement, noise, blending
and estimator effects; it is not a unique estimate of noise bias.

| Measured cuts | Error in selected true-e R11 | Error in selected (measured-true)-e R11 | Full R11 error |
|---|---:|---:|---:|
| None | -0.00062704 ± 0.00025624 | +0.01147784 ± 0.00217198 | +0.01085080 ± 0.00215518 |
| Magnitude + radius, no e cut | -0.00054877 ± 0.00068785 | +0.01002118 ± 0.00174285 | +0.00947241 ± 0.00178740 |
| All cuts | -0.01665640 ± 0.00135433 | +0.02943824 ± 0.00169152 | +0.01278184 ± 0.00187807 |

For magnitude+radius, actual/model true-e responses are1.06533907/1.06479030;
actual/model residual responses are-0.06074014/-0.05071897. Thus the model
underpredicts the magnitude of the negative measurement-residual response.
With one common final model-response denominator, contributions to m are
+0.054115 ± 0.067828 and -0.988212 ± 0.171797 percentage points, summing
-0.934097%. Separate integration SE are0.001102/0.000948pp. Allcut
contributions are+2.694511 ± 0.219487 and -4.762234 ± 0.273916pp, summing
-2.067723%; the larger terms partly cancel. Exact identities pass every
bootstrap/group-deletion replicate. Components are correlated accounting
terms, not independently isolated causal biases.

### Conditional feature sets need not compose correctly

The selected flow checkpoint's actual condition preprocessor contains
`e1_input_p`, `e2_input_p`, `sersic_n_input_p`, `r_input_p`,
`circularized_Re_input_p`, `nbr_flux_near`, `nbr_flux_far`, `nbr_flux_max`.
The frozen detector `circularized_preprocessor.json` has17 features covering
primary/representative-neighbour scaled magnitude and circularized radius,
both Sersic indices, separation, neighbour impact/presence, sky-frame shapes,
and separation-frame shapes. It omits the three aggregate crowding-flux
features. Conversely the flow lacks the detector's detailed neighbour
geometry and shapes. These condition sets are not nested.

The valid identity `P(D|X) P(Y|D,X)` assumes both factors refer to the same
conditioning information, or that omitted information can be integrated
consistently. Our earlier observation that the factorization itself is valid
must not be read as proof that this particular pair of compressed models
composes correctly. Separate networks are not intrinsically a problem.

An exact illustration at fixed shear g, with no measured cuts: let F be flow
features, G detector features, and suppose both conditionals are perfect on
their own features:

\[
\mu_F=E[Y\mid D=1,F],\qquad p_G=E[D\mid G].
\]

Then `E[DY]=E[D mu_F]`, but the composed numerator is `E[p_G mu_F]`.
Consequently

\[
E[p_G\mu_F]-E[DY]=-E[\operatorname{Cov}(D,\mu_F\mid G)].
\]

Total detection counts remain exact, `E[p_G]=E[D]`. If true ellipticity is
already specified by G, its covariance term vanishes, while the measured
conditional mean can still vary with crowding features missing from G.
This gives a concrete route to correct true-shape weighting but wrong
measured-shape weighting, even with optimal separate BCE and conditional NLL.
The shear variation of this covariance matters for R. With measured cuts,
the same issue applies to `u_F=E[Y S|D,F]` and `q_F=E[S|D,F]`, affecting
numerator and denominator. Conditional sufficiency or compatible integration
could remove the issue; feature mismatch alone does not prove nonzero bias.
This result assumes calibration in the population at that g; transferring
between mixed-shear training and coherent ConstGold can add further error.

The script checks a finite toy example with an unobserved-to-detector
crowding state A=+/-1, probability `p0+k*g*A`, and exact flow mean `g+b*A`.
The detector correctly predicts marginal p0. The composed response is1,
whereas the actual detected mean has response`1+b*k/p0`. Both individual
models can be correct for their respective inputs. Arbitrary toy constants
are unrelated to SBSI measurements; this is a mathematical possibility,
not a fitted explanation of the observed sign or magnitude.

### Paired-survivor supervision can target a different conditional mean

Source `sbsi/flow_paired_shape.py:44` compares differences of flow conditional
means with measured differences from usable pairs. NLL uses individual usable
rows. In general,

\[
E[Y_+-Y_-\mid D_+=D_-=1,X_+,X_-]
\ne E[Y_+\mid D_+=1,X_+]-E[Y_-\mid D_-=1,X_-].
\]

The other leg's detection can reveal information about the common noise or
scene state. Even an exact detected-conditional density can therefore incur
a systematic paired-response penalty. Stronger regularization can move its
mean away from the NLL optimum. This is a possible incompatibility of target
quantities, not just missing training weight on rare objects.

An analytic scalar toy establishes the possibility without hidden galaxy
state: `Y_g=g+epsilon`, common Gaussian epsilon across legs, and detection
`epsilon>t-k*g` with k>0. Every both-detected pair has response exactly1.
The detected conditional mean is `g+E[epsilon|epsilon>t-k*g]`; its response
is below1 because raising g admits lower-noise realizations. Forcing its
conditional mean difference toward the paired value overpredicts the full
detected response and produces negative closure m. The script checks this
finite-difference counterexample. The real model's error is not proven to
come from this mechanism, and its NLL is not restricted to matched pairs.

### Mean offsets and response constraints

The generative mock comparison also found significant even shape offsets;
see [mock comparison](CONSTGOLD_GENERATIVE_MOCK_CLOSURE.md#completed-comparison--2026-09-12).
A condition-dependent mean error b_i that is constant across a given pair
is invisible to that pair's mean-difference penalty. If normalized detection/
selection weights are alpha_i(g), its finite response contribution is
`sum_i (alpha_i,+-alpha_i,-)*b_i / 0.04`. Thus static conditional errors can
become response errors when the contributing population changes. A single
global additive constant cancels when it does not change selection; the
observed midpoint offset alone is not proof of the multiplicative mechanism.

More generally the finite identity for each model is
`Delta(sum alpha*mu)=sum mean(alpha)*Delta(mu)+sum Delta(alpha)*mean(mu)`.
Good matched differences do not fix the second term. Under measurement cuts,
use conditional selected means and normalized `p*q` weights. Matching mean
radius or flux responses likewise does not determine shape-weighted boundary
probabilities or correlations in the joint distribution.

The evidence supports an error in the detected population's measured-residual
response. Incompatible compressed conditionals, paired-survivor supervision,
incomplete optimization, and changes in omitted scene information remain
distinct candidate explanations. A calibrated mean blend shift does not
certify coherent-neighbour changes in conditional scatter or selection.
Nor can a different cross-leg mock coupling fix a mean response while holding
the single-leg marginals fixed. Explicit mocks verified numerical integration
but shared the same conditional inputs and blend construction.

Shear-dependent detection is a known source of percent-level calibration
effects in blended image simulations
([Sheldon et al. 2020](https://arxiv.org/abs/1911.02505)); that establishes
plausibility, not the cause of the present model discrepancy. The two
conditional-model arguments above are explicit derivations, not conclusions
attributed to that paper. No model or loss changes follow from this analysis.
