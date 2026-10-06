# V3.6-related model comparison — 2026-09-24

## Decision and scope

**The September24 close-crowding flow gives +0.286% in the normalized
target-role development statistic, without truth-property cuts.
Robust 0.3% calibration is not established.**
Its case-bootstrap standard error is 0.266 percentage points, and substantial
opposing magnitude-region errors remain. The previous +0.93% headline used a
different, inconsistent reducer. This report supersedes the coverage-first
priority in the earlier [recovery plan](V36_RECOVERY_PLAN.md#priority-and-evidence)
for the newer models.

The owner fixes measured `FLUX_RADIUS >0.60 arcsec` (3 pixels) and
`MAG_AUTO <25.8`, without truth-property cuts. The completed saved-data
comparison uses cases40–119 of the **generated target-role** parent, with no
truth-magnitude or g0-usability restriction. This is about half of each rendered
scene's identities. A fresh two-case check of both generated roles together
is running separately; it must not be confused with the completed80-case
comparison. The legacy retained parent is truth-limited and is reported only
as a distinct control.

Shapes are ngmix reduced-shear ellipticities; the stencil is coherent g1
+/-0.02. Each leg's selected mean is normalized by its own selected count.
Bias is `100*(R_data/R_model - 1)` percent. The production-like comparison uses
the smooth classifier's independent per-leg probabilities. Measured per-leg
usability is a diagnostic control; intersection usability is not acceptance.
Definitions: [response](CONVENTIONS.md#6-response-estimators),
[bias](CONVENTIONS.md#7-multiplicative-bias), and
[model selection](CONVENTIONS.md#8-seeds-and-model-selection).

## Completed comparison of saved predictions

Recomputed sufficient sums with one reducer and 10000 paired case-bootstrap
replicates, seed20260924. Measured per-case sums are bit-identical across all
six rows below. Every row uses64 draws, latent seed7301 and the original
physical disk-response emulator `194c1d11b67d...` with Möbius composition.
The mag26.5 flow uses the old smooth classifier `ad975edf7020...`; all five
later flows use the unbounded smooth classifier `13cd6a81ecff...`.

| Flow/candidate | Flow SHA256 prefix | Predicted-U m (%) | Measured per-leg U m (%) |
| --- | --- | ---: | ---: |
| True-r<26.5 retrain, evaluated without truth cuts | `9babb174b9aa` | +2.950 ±0.272 | +2.626 |
| Unbounded, 8 inputs | `da73a8db1d36` | +2.393 ±0.271 | +2.494 |
| Flux/size refinement, epoch08 | `509ce7868e7a` | +3.012 ±0.273 | +3.112 |
| 8-input magnitude guard, epoch02 | `66f5b951c67c` | +4.697 ±0.278 | +4.800 |
| Close-crowding, 11 inputs | `1433b7217fb7` | **+0.286 ±0.266** | **+0.371** |
| Close-crowding magnitude guard, epoch03 | `c91e4539b99e` | +2.181 ±0.270 | +2.272 |

The retained narrow-parent V3.6-like control replays -0.199979%; the mag26.5
flow on that same narrow parent gives +1.129283%. Neither is the no-truth-cut
headline. The unbounded single-output versus row-dump replay differs by only
+0.000333 ±0.000849pp, with identical measured sums; both versions are kept.

The close-crowding change relative to the unbounded row-dump baseline is
**-2.107048 ±0.014785pp**, paired95% interval[-2.136369,-2.078430]. This is
clear development evidence of improvement, not two independent noisy
measurements. It also improves measured-radius transfer:

| Diagnostic radius region, magnitude cut unchanged | Unbounded m (%) | Close-crowding m (%) | Close-crowding guard m (%) |
| --- | ---: | ---: | ---: |
| Radius >0.60 arcsec | +2.393 | +0.286 | +2.181 |
| Radius >=0.75 arcsec | +3.795 | +0.421 | +1.297 |
| 0.60<radius<0.75 arcsec | -3.760 ±1.260 | -0.102 ±1.303 | +7.654 ±1.405 |

The shell is noisy, so near-zero shell bias is not precision acceptance.
No threshold was moved or selected to obtain these numbers.

![Candidate comparison and residual contributions](figures/v36_model_comparison_20260924.png)

## Why the previous headline differed

Inspection of the newer run-checkout `scripts/joint_response_decomposition.py`
found two differences from the declared estimator:

1. Model weights use `usable_plus & usable_minus` on both legs, whereas the
   simulated selected sample uses each leg's own usability.
2. `m_total` divides unnormalized shape-numerator slopes, without dividing each
   model/data leg by its own selected count.

The independently written row reducer reproduces the old scalar exactly, then
separates these effects for the close-crowding flow:

| Calculation | m (%) |
| --- | ---: |
| Historical unnormalized, mixed-usability calculation | +0.927667 |
| Normalize both legs but retain the inconsistent model intersection | +0.121774 |
| Normalize and give model the measured per-leg U | +0.371344 |
| Normalize and use the trained classifier per leg | **+0.285858** |

This is an estimator correction, not an empirical rescaling. The repaired
statistic was already present in the original evaluator's saved sufficient
sums. The later diagnostic reducer produced a misleading headline. The same
issue affects its shape/selection budgets, so their numerical attributions
must be recomputed before assigning a physical cause. The cached fixed-g0
R_blend lookup and physical disk candidate are also distinct: in these six
evaluations, the explicit disk candidate overrides the additive lookup shift.
A test of the lookup on its own training data does not validate the deployed
disk candidate's selected response.

## Exact remaining cancellation

Replayed240 row files across unbounded, close-crowding and guard variants.
Each case's measured and model count/numerator sums reproduce the original
evaluator, for predicted U and measured per-leg U. File hashes, case membership
and one-leg-only usable counts are retained. No truth-bin filtering enters the
global response.

For each bin k, define `H_k=(A_k,+-A_k,-)/(2h)` and
`B_k=(D_k,+-D_k,-)/(2h)`. The exact contribution to the global normalized
response is

`R_k = (Dbar*H_k - Abar*B_k)/(D_+*D_-)`.

Global Abar and Dbar are the means of total leg sums. Summing R_k recovers
the separately normalized finite difference exactly, without assuming equal
counts. Taking data-minus-model contributions and dividing by total model R11
gives this additive error allocation:

| Diagnostic true magnitude | Unbounded contribution (pp) | Close-crowding (pp) | Close-crowding guard (pp) |
| --- | ---: | ---: | ---: |
| r<24 | -1.047 | **-1.199 ±0.139** | -0.163 |
| 24<=r<25.5 | +0.778 | -0.067 ±0.184 | +0.689 |
| 25.5<=r<27.5 | +2.528 | **+1.459 ±0.153** | +1.562 |
| r>=27.5 | +0.135 | +0.093 ±0.013 | +0.092 |
| Sum | +2.393 | **+0.286** | +2.181 |
| Sum of absolute contributions | 4.488 | **2.818** | 2.506 |

These are contributions to the global bias, not conditional-bin biases or
causal self/neighbour components. Bin errors are correlated; their errors
must not be combined as independent. The sum uses the paired bootstrap.
Measured-usability controls retain close-crowding contributions approximately
-1.204, -0.040, +1.532, +0.084pp. Therefore classifier repair alone is not the
identified route to removing this cancellation.

The guard partly removes the bright excess but worsens the intermediate
region and leaves the dominant faint deficit. Its larger total does not mean
every physical change was worse; conversely its smaller sum of absolute errors
does not make it calibrated. Both aggregate and resolved errors matter.

## Second review: reproducibility, reuse and mixture sensitivity

The owner's follow-up requested a thorough review of the +0.286% claim.
`scripts/review_v36_closecrowd.py` uses the previously replayed row sums but
imports none of the earlier reducers. It directly differences each leg's
`sum(e1)/count`, forms the data/model ratio, and checks it against the saved
comparison. This independently verifies the reduction, not fresh image
measurement or forward prediction. CPU16689895 completed the initial review;
16689907 adds hashed population and checkpoint-selection receipts. The final
output is `closecrowd_review_v2.json` in the evidence directory below.

The estimate reproduces **+0.285857927%**. Delete-one-case jackknife SE is
**0.266597pp**. A separate 20000-replicate paired bootstrap, seed20260925,
gives SE0.263548pp and95% interval[-0.237071,+0.796948]%; the small difference
from the reported0.265609pp is bootstrap Monte Carlo variation, not a revised
preferred uncertainty. The paired improvement versus the unbounded flow
reproduces -2.107048pp with SE0.014780pp. Deleting any single case gives
m between+0.231241 and+0.373206%, with maximum change0.087348pp (case56).
No case is removed from the headline. Thus no single scene explains the
improvement, while merely crossing0.3% is not a stable pass/fail conclusion.

### Every evaluation scene has been used in model development

The classifier's frozen protocol trains on cases40–99 and chooses its
checkpoint by minimum mean-case tuning BCE on cases100–119. Its labels come
from the half-shear measurements, not the coherent +/-0.02 measurements used
here. Nevertheless, **all80 scene identities are reused**. The flow's
half-shear domain split likewise contains62 of these scenes in training and18
in validation; no evaluation scene is outside both sets. This is more specific
than calling the result simply a development measurement.

The following subsets were inspected for this review and are not alternative
headline samples or independent acceptance tests:

| Subset | Cases | m (%) | Jackknife SE (pp) |
| --- | ---: | ---: | ---: |
| First40 cases,40–79 | 40 | +0.0145 | 0.3891 |
| Last40 cases,80–119 | 40 | +0.5567 | 0.3644 |
| Classifier training scenes | 60 | +0.2736 | 0.2976 |
| Classifier tuning scenes | 20 | +0.3227 | 0.5975 |
| Flow training scenes | 62 | +0.0709 | 0.3022 |
| Flow validation scenes | 18 | +1.0281 | 0.5452 |

The flow-validation result deserves follow-up, but its uncertainty and reuse
for flow selection prevent a conclusion of either proven overfitting or
validated failure at1%. The split contrasts are noisy; the 80-case result
does not establish how a fully untouched population behaves. Splits across
different rows of this table overlap and are not independent experiments.

The close-crowding parent manifest has `truth_analysis_cut=null` and reports
**zero truth-cut removals**: all27,982,713 generated target-role parents enter.
It retains20,141,603 g0-unusable parent instances. Three g0 diagnostic keys
outside that parent are reported and dropped during the g0 diagnostic join;
g0 membership is not used in the acceptance statistic. This does not change
the remaining scope limitation: the generated target role is only about half
the rendered identities.

The close-crowding baseline was selected at common-descent step1; its full
validation NLL increase is0.0004431nats/row, within its0.001 gate. Its completion
receipt explicitly says `production_accepted=false`. In contrast, the separate
close-crowding guard epoch03 missed its predeclared0.005 NLL gate, increasing
NLL by approximately0.0055. The newer work log openly records that exception.
It must not be described as a candidate that passed every original rule, even
though it was chosen on half-shear diagnostics rather than constant-shear m.

### Modest changes in target weights expose the cancellation

An exploratory sensitivity check multiplies one previously defined magnitude
bin by0.9 or1.1, identically on data/model and both shear legs. All other weights
stay1; **all rows and the original measured cuts remain**. Both data and model
leg normalizations are recomputed exactly. These are diagnostic target weights,
never applied to the headline model, acceptance catalogue or inference.

| Target-weight perturbation | m (%) | Paired change (pp) |
| --- | ---: | ---: |
| r<24 weight0.9 | +0.4125 | +0.1266 ±0.0129 |
| r<24 weight1.1 | +0.1681 | -0.1177 ±0.0121 |
| 25.5<=r<27.5 weight0.9 | +0.1754 | -0.1105 ±0.0131 |
| 25.5<=r<27.5 weight1.1 | +0.3924 | +0.1065 ±0.0125 |

The uncertainties describe paired shifts, not the absolute m uncertainty.
This demonstrates sensitivity to which targets contribute, using the existing
scenes. It does **not** simulate a changed luminosity function, number density
or neighbour field: changing a physical population also changes other targets'
blending and requires new images. It therefore supports the cancellation
concern without claiming a new-domain calibration result. No favorable weight
is adopted. The scene-resampling uncertainty also excludes fresh latent draws,
training variation and changes to the finite source catalogue.

## Revised route toward robust 0.3%

### Keep the demonstrated improvement

Use the already frozen close-crowding checkpoint as the baseline for new
development, retaining its physical-distance crowding inputs and the full-range
classifier. No further checkpoint is selected by proximity to zero in these
constant-shear results. Do not repeat the completed coverage retraining merely
because the old pinned V3.6-like had narrow support. The original disk emulator
still has narrow training support; its separately trained mag26.5 replacement
needs its own controlled evaluation.

### Repair the training statistic before adding more guard steps

The current `PerLegGuardResponsePopulation.contributions` in the newer training
checkout divides by measured `baseline_mass` and subtracts measured
`baseline_mean`. These remain constants during optimization. Magnitude strata
do not remove that mismatch with the model's own normalized selected mean.
This is a surrogate objective, not automatically an invalid training loss.
In the derivative limit, write model mass and shape numerator as Dm and Am,
and model/data selected means as mum and mud. Its model statistic is
`Gm=(Am' - mud*Dm')/Dd`, whereas the selected model response is
`Rm=(Am' - mum*Dm')/Dm`. Consequently
`Gm=(Dm/Dd)*Rm + (mum-mud)*Dm'/Dd`.
They agree when baseline mass and selected mean agree. This explains what must
be audited before strengthening the guard; it does not prove that this
mismatch causes the entire observed residual. The identity is local in shear,
not an assertion that the finite0.05 random-direction fit equals a derivative.
The guard also uses a softened forward0.05 half-shear target, whereas acceptance
uses hard cuts and coherent antithetic0.02. Its additional paired flux/size
objective uses both-leg usable rows and floored log outputs. Matching those
conditional mean changes cannot by itself constrain the selected joint tail.

First compare model and simulation **on designated training scenes** using
own-normalized A/D, per-leg mass, and joint shape/size/flux probabilities near
the actual cut boundaries. Verify hard-cut integration and soft-bandwidth
convergence. Check global moment gradients rather than averages of minibatch
ratios. Preserve proper full-population density and usability losses, with
declared tolerances and common random numbers. The experiment should test
whether these corrected objectives reduce the bright and faint contributions
together; it should not set their labels to the constant-shear residuals above.

### Identify and model the missing neighbour effect

The physical disk transport changes shape while leaving radius/flux unchanged.
That architecture cannot directly transport the target across measured flux/
radius cuts when only its neighbours are sheared. Newer half-shear diagnostics
suggest this matters for faint targets, but the old inconsistent decomposition
does not establish its exact percentage contribution.

Use matched signed image perturbations with target-only, neighbours-only and
coherent shear, both axes, at weak amplitudes. Measure per-leg usability,
selected mass, shape numerator and their **joint** response on the same parent;
do not intersect valid rows in the science statistic. Also check the disk
candidate's native response on that selected population, rather than substituting
a successful test of the older fixed-g0 lookup.

If neighbour-only flux/size/usability response is resolved, train a joint
response extension from those simulation labels. For example, a density-
consistent transport of shape and log radius/flux with a corresponding usable-
event response can represent this missing effect. Preserve a valid forward
density and its Jacobian. A detached extra selection weight or scalar m offset
would not supply that consistency. BlendEMU owns new images, measurements and
emulator training; SBSI owns conditional models, composition and validation.

This is a targeted hypothesis to test, not a claim that the entire +1.46pp
faint residual has already been causally assigned. Correct bright-only errors
without addressing faint residuals and the current cancellation disappears,
potentially increasing total m before the model is actually improved.

### Separate numerical stability from independent acceptance

The close-crowding80-case95% interval is approximately[-0.246,+0.800]%, so
the current result is a near-target point estimate, not a bound at0.3%.
New full-parent checks below test role/population transfer, not independence.
Additional 64/128/256-draw and independent-latent checks are still needed;
the old retained model's seed repeat does not validate the newer flow.
Both shear axes, off-diagonal response and matched-amplitude checks must be
included before claiming robust calibration.

Freeze the full candidate and estimator, then use new scene-disjoint
acceptance data with all components respecting the same split. Plan precision
from the actual case-level ratio variance and its sensitivity, without choosing
the stopping point when m happens to approach zero. An SE near0.1pp would be
materially more informative than the present0.266pp. The practical approximate
target and the stronger claim of a confidence interval entirely inside±0.3%
must remain distinct. No bin-by-bin0.3% requirement is imposed; inspect weighted
contributions and predeclared domain perturbations for dependence on cancellation.

## Fresh full-rendered-parent and emulator checks

At this report's preparation, array16689736 evaluates the unbounded and
close-crowding flows on **all rendered identities**, cases40/41, FP64 neural
arithmetic,64 antithetic CRN draws, old disk emulator and unbounded classifier.
Both pass measured replay, original8-input context replay and cached pair
identity/feature replay (2102 pairs on128 parents for case40). The close-crowding
worker also reproduces all349784 cached11-input parent contexts' three added
features. Both have entered forward prediction. Results are pending.

The query uses k=20 including self (at most19 nonself neighbours), matching
the saved80-case candidate. Earlier full-role pilots used k=21 (at most20),
so their results must not be treated as otherwise identical population-only
comparisons. Pending array16689655 was cancelled to use available24GB GPU
allocations. Initial16689690 stopped because transformed model context width
was mistaken for raw feature count; the check now uses preprocessor feature
names. Partial16689706 was cancelled after detecting the neighbour-query
difference. No partial result enters any reported bias.

After16689736 completes, dependent16689747_2 swaps only the disk emulator to
the trained mag26.5 artifact `d2dc49113ae0...`, retaining the close-crowding flow,
classifier, cases and CRN. CPU reducers16689756/57 follow successful completion.
At most two GPUs can run in this chain. This is an emulator comparison, not
an automatic promotion. No new model training has been launched.

## Reproducibility and validation

New scripts: `compare_v36_saved_models.py`, `audit_v36_saved_rows.py`,
`plot_v36_model_comparison.py`, `review_v36_closecrowd.py`;
extended `audit_v36_domain_response.py` for
explicit candidate identities, close-crowding reconstruction and pair replay.
New/extended tests check per-leg normalization, spurious raw-count bias, exact
bin allocation, and physical-distance neighbour features.

Completed CPU jobs16689642/91 reconstruct model comparisons;16689678_0–2
replay240 row files and original evaluator sums;16689719 renders the plot.
Full suite16689712: **322 passed,2 skipped in29.32s**, before the final pair-
replay/optional-emulator extension. Final workers run9 focused tests before
evaluation. Final focused job16689797 passes14 tests in9.61s after those
extensions. Nine flow/classifier checkpoint byte hashes match the recorded
identities. Ruff, documentation links and shell syntax checks pass. No archived
code is imported. Scheduled one-shot16689796 writes a read-only job/summary
status receipt about two hours after submission; no conversational wakeup tool
is available in this session.

Evidence root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/`.

- `all_saved_models_v2.json`: all source hashes, model identities, case sums'
  comparisons, primary/high-radius/shell results and paired shifts.
- `rows_{unbounded_rows,closecrowd,closecrowd_guard_ep03}.json`: hashed row
  sources, original-evaluator replay, exact magnitude contributions and
  reproduction of the historical unnormalized statistic.
- `closecrowd_review_v2.json`: independent algebra replay, jackknife and second
  bootstrap, all scene-subset results, population/checkpoint receipts and
  exploratory mixture-weight sensitivity; v1 is preserved separately.
- `candidates/*.json`: frozen full-parent candidate identities.
- `new_*_16689736/`, `new_closecrowd_emulator265_16689747/`: running/queued
  fresh workers; completed summaries will be adjacent `*_summary.json` files.
- `doc/figures/v36_model_comparison_20260924.{png,pdf}`: standalone figure.

All completed results remain development evidence. The saved comparison does
not establish independent two-axis, full-rendered-parent0.3% performance or
validate the separate likelihood inference machinery.
