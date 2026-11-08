# V3.6-like recovery plan — 2026-09-24

**Superseded for current candidates by the
[completed model comparison](V36_MODEL_COMPARISON.md#decision-and-scope).**
The close-crowding model's correctly normalized saved development result is
+0.286 +/-0.266%, rather than the later reducer's +0.93%. Magnitude cancellation
remains. See the [revised recovery route](V36_MODEL_COMPARISON.md#revised-route-toward-robust-03)
before using the older recommendations below.

## Model-identity correction — newer work found

**The plan below was based on the September 20 pinned V3.6-like artifacts,
not the September 22–24 retrains. Its coverage-first priority is incomplete
for the newer models and must not trigger duplicate retraining.** The owner's
identity check exposed newer history in the separate checkout
`/home/z/Zekang.Zhang/SBSI/.claude/worktrees/nocut-retrain/doc/WORKLOG.md`,
which was missing from the earlier root-worklog review.

Verified distinct artifact identities from completion records:

| Component | Retained September 20 | Newer model |
| --- | --- | --- |
| Flow | `5434cfafb94e...`, 8 inputs, true r<26 | September 23 unbounded `da73a8db1d36...`; September 24 close-crowding `1433b7217fb7...`, 11 inputs |
| Smooth classifier | `ad975edf7020...`, true r<26 | `13cd6a81ecff...`, protocol explicitly no truth magnitude cut |

New flow roots are `flow_joint_unbounded_20260922_v1` and
`flow_joint_closecrowd_20260924_v1`, each under the SBSI cache with
`joint_probability_refinement_full/{completed,protocol}.json` and `selected.pt`.
The classifier is under `scene_classifier_unbounded_20260922_v1/`.
Their flow manifests retain a technical r<30 predicate; the newer worklog
records that the generated population ends below29, so it excludes no generated
objects. The close-crowding domain adds `nbr_close_050`, `nbr_close_100` and
`nbr_close_200`. The new classifier's protocol explicitly removes the truth cut.

The newer worklog reports modeled-smooth m=+2.393 +/-0.270% after the unbounded
flow/classifier retrain (80 cases). The close-crowding decomposition JSON reports
+0.927667 +/-0.260796%, and its guard-epoch3 variant +2.739784 +/-0.265277%.
Those latter decomposition results are **not automatically the same estimator
as the modeled-smooth result**: inspection of the run-checkout
`scripts/joint_response_decomposition.py` finds model weights using both-leg
usability, while simulation cuts use each leg's usability. These read-only
findings have not been reconciled against saved row sums in this task. Do not
claim their differences as matched improvements of the requested estimator.

Emulator identity also requires evaluation-branch specificity. The newer
close-crowding launcher supplies the original September18 physical disk model
as `--response-candidate`, as well as separate fixed-g0 `--rblend` tables.
The newer worklog corrects the latter emulator's population to measured cuts
only. A separate mag26.5 disk emulator also exists. Thus the retained model's
three-component true-r<26 statement cannot describe every newer branch.

The September24 two-scene audit and error allocation below remain evidence
about the **old pinned set only**. Next work must start with the newer model
identities, evaluator branches and latest corrected logs, then reconcile the
reported statistics before revising the recovery experiment.

## Fixed scientific target

The owner requests approximately **0.3% multiplicative bias on constant-shear
simulations after measured radius >0.60 arcsec and MAG_AUTO <25.8, without
truth-property cuts**. Include all rendered parents of both generated roles.
Usability and selection are evaluated independently in each shear leg; do not
intersect successful or selected objects across legs. Truth strata below are
diagnostics, never production cuts or correction weights.

Use the [shape conventions](CONVENTIONS.md#4-shapes-and-shear),
[response extraction](CONVENTIONS.md#6-response-estimators) and
[bias definition](CONVENTIONS.md#7-multiplicative-bias).
For the direct forward audit, report each diagonal
`m_ii = 100*(R_data,ii/R_model,ii - 1)` percent, both cross-responses and
zero-shear mean shapes. The primary stencil is coherent +/-0.02 on both axes.
This plan concerns forward response calibration. Successful forward calibration
does not establish convergence or accuracy of the separate likelihood estimator.

## Priority and evidence

**First fix training coverage, then test a correctly normalized response
objective.** Keep the current architecture as the initial control. The available
evidence identifies actionable errors but does not establish an attainable
0.3% floor or identify a unique responsible component.

The retained -0.200% result is development evidence on the narrower true-r<26,
secondary-role parent. Its measured-radius subregions have opposing residuals:
approximately -6.47% at 0.60<radius<0.75 and +1.32% above 0.75. Actual-usability
and latent-seed controls retain that pattern. See
[the controlled transfer audit](V36_DOMAIN_RECOVERY.md#controlled-measured-radius-transfer).
This makes a single low aggregate bias insufficient evidence of robustness.

The full-parent two-scene pilot uses the requested measured cuts. An exact
algebraic decomposition of its pooled g1 response error gives:

| Contribution to total m11 (percentage points) | Predicted usability | Measured usability control |
| --- | ---: | ---: |
| Within bright-population response | -0.1062 | +0.0544 |
| Within faint-population response | +0.6966 | +0.6512 |
| Wrong selected bright/faint proportions | +0.6727 | +0.2836 |
| Finite-step selection migration | +0.0009 | +0.0012 |
| Total | **+1.2640** | **+0.9904** |

Here bright/faint means true r<26 / r>=26 solely for diagnosis. Faint objects
are 10.894% of the measured selected sample, versus 13.594% predicted by the
model, or 12.040% when given measured usability. Their conditional response
bias is about +7.49%. The approximately +0.7pp response and composition terms
are comparable priorities. Perfect usability labels alone leave substantial
error in both. Near-correct total selected counts would also be insufficient
if bright and faint errors compensated each other.

This allocation is an identity, not an intervention that isolates causation.
Within-faint error includes conditional flow, transport and selection effects.
The pilot consists of two reused development scenes and has no reliable
population calibration interval. Its matched retained-to-full-parent shift is
+0.3321pp; comparing its +1.2640% directly with the old 80-scene -0.200% would
confound population change with case variation.

For each population write `R=(1-f) R_b + f R_f + T`, where f is the mean of
the two selected faint fractions and T is the finite-step migration term.
The data-minus-model identity uses the average data/model f to weight
`delta R_b` and `delta R_f`, plus
`delta f * (average R_f - average R_b)` and `delta T`.
It closes to <1.3e-16 in this pilot. The symmetric allocation avoids choosing
one population as an arbitrary expansion reference.

Local sensitivities illustrate the required scale, not fitted corrections:
near measured f=0.109, R_b=0.685, R_f=0.540 and R=0.669, a 0.004 absolute
selected-fraction error contributes about 0.087pp; a 1% relative error in
faint response contributes about 0.088pp. Thus percent-level faint response
accuracy and selected fractions accurate to a few tenths of a percentage
point are useful planning scales. They neither guarantee the final target
nor require every low-weight bin to have |m|<0.3%.

## Training coverage and labels

The retained artifacts have **all three** target-training populations restricted
to true r<26. This corrects the earlier report's claim of a broader classifier.

| Component | Verified retained training support | Consequence |
| --- | --- | --- |
| Conditional flow | Harmonized generated secondary-role true-r<26 parent; independent usable rows per leg | Faint selected tails and added roles need explicit coverage checks |
| Smooth 17-input classifier | Strict true-r<26 generated target-role parent, including U=0; no measured cut | Faint usability probabilities are extrapolations |
| Physical-moment disk emulator | True-r<26 structural primary; usable g0 and both response-label shapes valid; labels at h=0.2 | Faint response, label-amplitude transfer and validity conditioning need validation |

The flow refinement protocol pins domain SHA256 `1ab650f03ae2...`. The
classifier protocol hash `b1ba4c56a04a...` is pinned by its completion receipt,
which names the retained checkpoint `ad975edf7020...`; both prepared-manifest
hashes match. Emulator metadata hash `e9c28e921fa0...` and model identity match
the retained configuration, as does its prepared-manifest hash. This is a
metadata-lineage audit, not a new verification of every training record.

Full-domain labels cannot be obtained by merely deleting a filter in the flow
loader: the harmonized measurement receipt itself enumerates the narrow
secondary-role parent. Audit reusable image products and regenerate or
remeasure missing labels in **BlendEMU**, preserving the harmonized centering,
optimizer RNG and shape convention. Do not mix historical incompatible
measurement conventions. Keep all neighbours in each simulated image.

Classifier labels must include every rendered target, including nondetections
and failed fits. Flow labels use each leg's usable measurements before science
cuts. Response labels need both self and neighbour signed perturbations on
matched populations, both axes, and a coherent-shear cross-check. The emulator
may require paired-valid shapes for a physical label, but that intersection
must never silently define the science parent: quantify and model transport
to the independently usable population.

Existing h=0.2 emulator labels are not automatically weak-shear derivatives.
Compare matched labels at smaller amplitudes with common random numbers and
report curvature, validity changes and variance. For the coherent science
response, compare h=0.01,0.02,0.04 on the same development scenes. Do not select
the amplitude yielding the smallest calibration bias.

## Objective that matches the reported response

Let C(y) denote the fixed hard measured cuts, q the conditional distribution of
usable measurements, p_U the usability model, and w the declared parent
integration weights. At each shear leg compute

`D_g = sum_x w_x p_U(x,g) E_q[C(y)]`,

`A_g = sum_x w_x p_U(x,g) E_q[C(y) e(y)]`,

`mu_g = A_g / D_g`, and `R_h = (mu_+ - mu_-)/(2h)`.

The conditional model includes the physical neighbour transport. Train
probabilities and conditional densities on their full declared populations,
using proper Bernoulli and conditional-density losses. If response supervision
is needed, constrain this own-normalized coherent response using designated
training simulations, alongside selected mass and density constraints.
Separate A, D and response diagnostics identify whether a candidate merely
changes normalization to improve a ratio.

The old measured-reference guard omits parameter dependence relevant to the
reported statistic. In derivative notation its correction is
`R=[H-(A/D) outer B]/D`, differentiating A and D as well as H and B.
The old read-only stable-gradient audit used a predecessor checkpoint and made
no update; repeat it on the new full-domain candidate. See
[the historical normalization finding](V36_DOMAIN_RECOVERY.md#an-unfinished-objective-correction-in-the-historical-logs).

Three implementation checks are essential:

1. **Selection boundaries:** naive pathwise differentiation through a hard
   indicator misses probability crossing the cut. Use a verified integration
   or score estimator, or a soft training guard with bandwidth-convergence
   checks against the actual hard-cut response. A soft-loss improvement alone
   is not acceptance.
2. **Global normalization:** averaging minibatch ratios differs from the
   pooled A/D. Accumulate global moments and propagate their derivatives
   consistently, for example with two passes and current-parameter replay.
   Check finite differences with common random numbers, independent integration
   banks, and invariance to a common rescaling of all integration weights.
3. **Mass and density:** response-only descent previously conflicted with
   selected probability. Reject a step that improves response by materially
   degrading probability or density according to predeclared tolerances.
   Keep physical support; do not clip invalid tails or silently renormalize
   away failed draws.

## Ordered experiments

| Stage | Concrete experiment | Decision it resolves |
| --- | --- | --- |
| 1. Label and support audit | Small full-parent development batch; harmonized measurements, label receipt reconciliation, both roles, faint support and both axes | Whether a consistent full-domain training set can be assembled and which labels must be regenerated |
| 2. Coverage control | Retrain the existing component architectures on that domain, with proper density/BCE/native-moment objectives and declared response-blind selection | How much extrapolation alone explains |
| 3. Objective comparison | From the same coverage-trained starting point, compare frozen baseline, measured-reference guard control, and corrected own-normalized guard; identical training labels and matched random streams | Whether objective correction produces real response improvement while preserving mass and density |
| 4. Physical diagnosis if needed | Matched signed self/neighbour probes and coherent responses, partitioned by measured radius, magnitude and crowding | Whether remaining error needs flow changes, emulator labels or a richer physical response model |
| 5. Freeze and validate | New disjoint scenes, both axes, original hard cuts, converged integration and declared statistical precision | Whether approximately 0.3% is supported outside development data |

Stages 2 and 3 separate coverage from optimization instead of changing all
ingredients at once. Repeated training seeds assess stability; none is chosen
because its response bias happens to be closest to zero. Checkpoints use their
declared response-blind scores under
[model-selection conventions](CONVENTIONS.md#8-seeds-and-model-selection).
Training response labels can guide physical optimization; independent
acceptance results cannot choose seeds, checkpoints or corrections.

Do not enlarge the architecture first. If full-domain density and mass are
accurate but matched physical probes still fail, investigate the scalar disk
velocity and its feature sufficiency: neighbour geometry, tensor response,
and neighbour-induced auxiliary-observable changes may matter. These are
hypotheses, not established causes. Any extension must preserve a consistent
forward density and be trained on simulation labels, not on a pasted aggregate
bias. BlendEMU owns measurement and emulator training; SBSI owns classifier,
conditional flow, model composition, validation and inference.

## Independent validation and precision

Split all components by **scene identity**, with every shear leg, rotation and
noise reuse in the same split. Cases already used for any model or diagnostic
are development data. The cancelled 22000–22039 chain, including its 34
completed scenes, remains stopped and is not silently repurposed as a completed
independent validation.

Use a new development variance study of roughly 16–20 scenes to size the final
sample, keeping paired model comparisons within scenes. Resample scenes, not
millions of correlated object rows. Estimate variance for the actual pooled
response ratio and both axes. Sample size must reflect the desired precision,
not whether an early running mean happens to land near zero. Freeze the final
case list and analysis before opening acceptance responses.
The 16–20 scenes are a variance-estimation starting point, not a powered
acceptance sample. Examine sample-size sensitivity to that variance estimate
before committing the final allocation.

A useful prospective precision target is approximately 0.1 percentage point
standard error per diagonal. A point estimate near 0.3% with a large uncertainty
does not demonstrate 0.3% accuracy. A stronger claim that both |m_ii| are below
0.3% requires intervals inside that band, with joint coverage accounted for.
For a single approximately normal marginal interval this means roughly
`|m_hat| + 1.96*SE < 0.3%`; the distinction must be stated in reporting.
The two-scene pilot cannot set a reliable final scene count, and observed-only
response variance omits data/model covariance.

On matched development scenes, compare 64/128/256 draws and at least two latent
streams using common random numbers across shear legs. A proposed numerical
error allocation is <=0.03pp (one tenth of the science target); demonstrate
convergence rather than adopting it as an assumed error bar. Report scene,
integration and training-repeat variation separately. Common image noise and
optimizer random numbers can reduce difference variance while retaining each
leg's marginal distribution; they must not enforce common usability. Rotation
partners, if used, belong to the same scene cluster.

Keep 0.60/25.8 as the primary selection throughout. Nearby measured thresholds
and diagnostic partitions probe fragility; they do not replace the primary
criterion or impose 0.3% on every bin. Report signed and absolute weighted
error contributions so a new aggregate success can be distinguished from
another cancellation. Predeclare any additional population perturbations;
robustness only extends to the domains actually tested.

## Artifacts, execution and stopping conditions

New code: `scripts/decompose_v36_population_error.py` and three focused tests.
CPU job16689396 completed successfully in25s: three tests passed, then all
source hashes and pooled/per-case decomposition identities passed. Ruff passes.
This diagnostic changes no model. No new training or GPU job was launched.

Receipts under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/`:

- `population_error_allocation.json`: exact pilot allocation, source hashes,
  per-case values and explicit limitations.
- `training_coverage_receipt.json`: retained protocol/manifest lineage and
  all three training-population restrictions.
- `forward_summary.json` and `observed_summary.json`: original pilot and
  80-scene measured audit; see
  [completed pilot](V36_DOMAIN_RECOVERY.md#completed-uncut-pilot).

Heavy label preparation, training and validation must use the scheduler.
Preserve the combined two-GPU limit, with dependencies and at most one training
and one diagnostic worker when other SBSI work permits. Monitor stage startup;
do not revive archived launchers implicitly. Stop and diagnose inconsistent
labels, unstable gradients, failed density/mass constraints or unconverged
integration before scaling up. If a frozen candidate fails independent
acceptance, retain that result; further tuning consumes that holdout and
requires a new independent test.

The present result is a prioritized, testable route to recovery. No recovered
0.3% model or end-to-end inference result has been produced.
