# Prior attempts to retain when simplifying the training recipe

## Scope and constraints after owner correction

Owner requests review of previous attempts before proposing another loss.
The goal is fewer training stages with reliable performance over useful cuts,
not network compression or a fortunate calibration at one cut. Constant-shear
simulations remain evaluation-only; do not use them as training targets. The
architecture-compression jobs are cancelled and no new training is launched.

## Measured-input localization must qualify the loss summaries

The owner points out that the terse primary-penalty summary omitted the key
intervention: supplying measured size/flux and membership makes the neighbour
prediction agree much better, and regularized full calibration is substantially
closer. A generated-input coherent shift is not sufficient reason to reject
the primary-response regularizer or endorse NLL-only cancellation.

In the four-scene weak-nine factorial, the restricted neighbour shape-response
residual changes from+0.005217±0.001492 (generated/generated) to
−0.001339±0.001549 (measured/measured). Neither single replacement closes it.
The total prediction change0.006556 is almost entirely the scalar emulator
coefficient reweighting0.006547. At identical radius/flux within cells, the
contrast remains: generated selection weights choose different primary/neighbour
configurations, with lower average neighbour response. This localizes a joint
conditional-photometry/scene-selection mismatch, not merely wrong marginal
histograms or demonstrated failure of the emulator response function.
[Factorial](../photometry_selection_factorial_20261106/findings.md#evidence-and-subsequent-localization),
[fixed-cell test](../photometry_fixed_cell_20261106/findings.md#four-scene-fixed-photometry-result).

The full measured-input comparison also showed the response-regularized flow
much closer than NLL-only, though old-reference axis2 residuals remained.
With new paired measurements, the latest measured-input nine-cut RMS is0.25525%
versus0.63457% generated; central m1/m2=−0.13281±0.12329% /−0.41639±0.16471%.
Thus the working priority is conditional size/flux and its selection association
while retaining the useful response constraint. This is strong localization,
not proof of zero response error at all cuts: radius dependence remains, and
usability/photometry/membership were jointly substituted in the full test.
[Full original comparison](../joint_measured_photometry_20261106/findings.md#full-forty-scene-result),
[updated comparison](../paired_measured_photometry_20261108/findings.md#completed-result-and-interpretation).

The simple NLL-only model's good total bias may therefore involve cancellation.
Its RMS alone does not justify dropping regularization. Simplifying the training
sequence must preserve component evidence and conditional associations, not
choose a recipe solely by its aggregate constant-shear result.

## Completed attempts

| Attempt | Observed outcome | What not to infer |
|---|---|---|
| One-/nine-cut primary-response penalties | Improved the training/validation response proxy; earlier coherent tests often worsened. Factor swaps located much of the induced change in conditional shapes. | A better primary matched-pair score guarantees full selected calibration. |
| Added static neighbour summary | Improved held-out predictive likelihood/selection diagnostics; coherent gains were small or mixed. | A static neighbour score supplies neighbour-shear-varying information. |
| Selected-neighbour moment penalty | Validation static association error improved about72%; original four-scene coherent calibration worsened. | Matching passing mass and a static weighted moment is enough. |
| Same moment penalty with shape weights frozen | Static score improved; original nine-cut coherent RMS still worsened on both axes. | Fixed conditional-shape parameters preserve the selected shape response. |
| Per-row conditional cut-probability loss from neighbour-trained start | With new paired-seed reference, full40 RMS0.75735→0.63457%, paired−0.12278±0.00515pp. | This proves necessity of earlier stages or robustness to arbitrary cuts. |
| Matched four-way probability-loss/input experiment from original nine-cut start | All probability scores improved, but the original coherent evaluation did not improve. Extra-loss central shifts were+0.0697/+0.0854pp. | The same loss works irrespective of starting model. |
| Earlier photometric derivative supervision | Sometimes dramatically improved derivative scores while worsening selected calibration or disagreeing with boundary migration. | A global size/flux slope penalty is a new, demonstrated remedy. |

Sources: [primary regularization](../alignment_regularized_20261106/findings.md#interim-constant-shear-comparison-19-common-cases),
[factor swaps](../neighbour_factor_swap_20261107/findings.md),
[neighbour input](../neighbour_conditioned_flow_20261106/findings.md#four-scene-matched-comparison),
[moment penalty](../neighbour_moment_guard_20261106/findings.md#joint-constant-shear-evaluation),
[frozen shapes](../frozen_shape_neighbour_20261107/findings.md#completed-frozen-shape-comparison-on-both-axes),
[updated probability comparison](../selected_seed_calibration_20261108/findings.md#frozen-training-comparison-and-radius-dependence),
[four-way comparison](../photometry_learning_information_20261107/findings.md#relation-to-the-earlier-comparisons),
[derivative history](../photometric_derivative_history_20261107/findings.md#which-experiments-actually-ran).

Historical coherent results above retain their original measurement references;
they are not all re-evaluated with the new paired-seed fits. Do not mix absolute
biases across those references, checkpoints, four-/forty-scene cohorts, or
predicted/actual usability modes. In particular, the latest stage audit changes
some comparisons; it does not make the original historical claims current.

## Mechanisms already investigated

The exact probability-loss localization found selected-mass growth of roughly
0.35–0.38%; numerator growth was insufficient, so normalized response decreased.
Membership and output-value terms were positive, normalization larger and
negative. This was not an estimator bug. Frozen shape weights do not freeze
photometry-conditioned shapes or the selected population.
[Exact accounting](../photometry_response_localization_20261107/findings.md#main-finding).

The existing training response uses soft cuts and a zero-to-forward primary
shear regression on both-usable pairs, with measured baseline moments in an
influence function. It omits classifier transitions and disk transport.
All-parent detection/usable labels do exist separately; claiming they are absent
would be wrong. Dedicated neighbour-only/full coherent targets are absent from
that prepared response contract. The original review already warned that
selected-shape supervision from these pairs is a primary-only constraint.
[Target coverage](../response_strategy_review_20261107/findings.md#training-target-coverage-audit).

Physical replay showed neighbour-only detection/cut crossings while scalar
flow/classifier inputs stay invariant. Its uncertainty was too large to assign
percent-level calibration error to that missing channel. The primary-pair audit
also found substantial global photometric slope error alongside nearly correct
pooled intrinsic-shape selection response at the central cut; more derivative
penalty is not automatically beneficial.
[Primary/selection audit](../primary_selection_transport_20261107/findings.md#primary-shear-a-slope-error-does-not-imply-a-target-cut-shape-error).

## Current evidence and implication

Paired-seed remeasurement addressed fitter randomness; the latest model still
has cut-dependent residuals. Measured size/flux/membership substitution reduces
RMS0.63457→0.25525% but leaves radius dependence, so photometry/classification
alone is not a complete attribution. Original NLL-only factorized flow has
RMS0.39999% but two cut/component tradeoffs and no demonstrated general-cut
robustness. Neither observation establishes a sufficient training recipe.

Do not propose another generic cut-probability, static-moment, frozen-shape or
photometric-slope loss as if it were untried. Conditional selected-shape moment
supervision was discussed in prior reviews; the precise per-row objective is
not established as a successful single-stage recipe in this evidence. Any
future test must identify its specific difference from these attempts and
what available training labels support it. A clean one-stage versus staged
comparison with matched inputs/objective/budget has not been demonstrated here.
This review adds no numerical results, jobs, or model changes.
