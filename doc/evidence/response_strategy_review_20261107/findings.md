# Review of the shear-calibration investigation

## Decision after reviewing the complete chain

The current probability loss is a useful predictive improvement and a partial
calibration improvement. It has not solved the remaining component-dependent
response error. Do not immediately add another selected-shape loss: first close
the missing measured-input comparison for the exact current checkpoints.
That focused diagnostic has now [completed](../current_measured_photometry_20261107/findings.md#completed-result-and-decision), retaining a central component-2 bias of−0.9555±0.2411%. See its
[fixed protocol](../current_measured_photometry_20261107/rule.md#question-and-fixed-models).

Earlier summaries emphasizing improvement in one component and worsening in
the other were incomplete as an overall assessment. The joint RMS across both
components and all nine target cuts decreases, although neither every-component
improvement nor independent calibration acceptance is established. The owner
permits internal cancellations; individual bins are not new rejection gates.

## New calculation: the common shift improves, the component difference remains

On all40 scenes at measured radius >3 pixels and magnitude <25, with predicted
usability and generated selection, define the average bias as (m1+m2)/2 and
half the component difference as (m1−m2)/2. These are descriptive arithmetic
contrasts, not physical causes or corrections.

| Quantity | Control | Extra probability loss | Paired change ± scene jackknife SE |
|---|---:|---:|---:|
| Average central bias (%) | +0.35137 | +0.23729 | −0.11408 ±0.00262 pp |
| Half component difference (pp) | +0.57190 | +0.57081 | −0.00109 ±0.00190 pp |
| Joint nine-cut RMS bias (%) | 0.68902 | 0.60523 | −0.08378 ±0.02554 pp |

Absolute SEs for the new average and half difference are0.19769 and0.19932pp.
On the additional36 scenes alone, joint RMS decreases0.69768→0.62503%, paired
change−0.07265±0.02935pp. Its half difference changes−0.00184±0.00195pp.
The original4 had a half difference of opposite sign (−0.18806pp control),
illustrating why those four should not set the broader scientific conclusion.
These cuts are correlated and are not independent repetitions.

Actual usability leaves the all40 component difference similar: half difference
0.58472→0.58264pp. This substitution does not eliminate the pattern. Symmetric
and antisymmetric off-diagonal residuals have scene errors larger than their
point estimates; there is no comparable resolved off-diagonal result in this
readout. Full matrices, all14 cuts, all three cohort sizes and flag modes are
preserved in [report.json](report.json) and the [nine-cut table](all_cuts.md).

Conventions: m=100(Rsim/Rmodel−1), antithetic±0.02, each leg normalized by its own
selected mass. The same whole-scene weights are used across axes, arms and
cuts, retaining their covariance. [Bias definition](../../CONVENTIONS.md#7-multiplicative-bias),
[response convention](../../CONVENTIONS.md#6-response-estimators),
[fixed audit protocol](rule.md#questions-and-fixed-reporting).

## What the earlier measured-input result actually tells us

The original NLL/nine-weak pair is distinct from the current neighbour-trained
pair. Under measured photometry and membership, on all40 scenes:

| Original model | Average central bias (%) | Half component difference (pp) |
|---|---:|---:|
| NLL baseline | −1.30472 ±0.19425 | +0.10225 ±0.19719 |
| Nine-weak response continuation | −0.36785 ±0.19599 | +0.36521 ±0.19927 |
| Paired continuation change | +0.93686 ±0.00199 | +0.26296 ±0.00218 |

This confirms that the original response-training intervention changes the
two components differently even under common measured photometry/membership.
It improves joint nine-cut RMS in that intervention (1.26622→0.42645%), so the
different component shifts alone are not grounds to reject it. Neither result
certifies the later model's conditional shape or transport as correct.

The earlier statement that measured inputs substantially reduce the discrepancy
needs its scope: the restricted neighbour diagnostic improved, while full
coherent bias shifted downward by approximately1.0–1.15pp and did not always
move closer to zero. [Original full measured-input tables](../joint_measured_photometry_20261106/full_results.md).
The current checkpoint pair has never had that exact40-scene measured-input
comparison. Repeating it now avoids borrowing an attribution from a different
model family.

## Evidence map: what is established and what is still open

| Evidence | Supported inference | Inference not established |
|---|---|---|
| Direct photometry predictors using existing inputs reduce the highlighted radius/flux residuals | Practical conditional learning can improve | All relevant information is already present, or a flow trained on these scores fixes shear |
| Added truth neighbour score improves held-out predictive losses | Some additional scene information is useful | That scalar is sufficient for coherent shear response |
| Static-neighbour moment loss improves its score but worsens the matched coherent test | That proxy can reward an unfavorable response change | Photometry is irrelevant, or the emulator alone causes the bias |
| Conditional probability loss improves prediction and modestly reduces joint RMS | A useful partial effect that replicates | Both components/cuts are calibrated, or more weight will solve the remainder |
| Frozen shape weights still give changed selected response | Photometric inputs, selection and normalization matter | The fixed conditional shape factor is accurate |
| Original measured-input continuation changes half component difference by+0.26296pp | The original response-training effect is component dependent | The current model has the same remaining physical error |
| Neighbour-only image changes cause real detection/cut crossings, while current scalar inputs stay fixed | A response channel is missing from those input paths | Its contribution to percent-level bias has been measured precisely |
| Small physical replay has large sampling and fit-repeat errors | Existing absolute primary/neighbour attribution is inconclusive | More loss terms or a larger naive repeat will automatically identify the cause |

Evidence: [predictive information](../flow_photometry_information_20261107/findings.md#main-conclusion),
[frozen moment trial](../frozen_shape_neighbour_20261107/findings.md#completed-frozen-shape-comparison-on-both-axes),
[probability replication](../conditional_cut_replication_20261107/findings.md#completed-result),
[selection/normalization accounting](../photometry_response_localization_20261107/findings.md#main-finding),
[physical input contract](../primary_selection_transport_20261107/findings.md#which-shear-the-model-can-see),
[physical replay uncertainty](../primary_neighbour_replay_20261107/findings.md#why-two-optimizer-seeds-disagree).

## Training-target coverage audit

The prepared training cache has160 scenes,46,833,127 usable NLL rows and
23,279,984 matched usable response pairs. Its disjoint half-shear validation
has40 scenes and5,817,708 pairs. The response intersection excludes141,107
zero-leg and132,052 forward-leg training rows lacking usable counterparts;
validation exclusions are35,353 and32,816. These are reported eligibility
exclusions, not all necessarily failures of catalogue matching.

The executed response code uses soft cuts, measured zero-leg mean/mass held
fixed in an influence function, and a regression on zero-to-forward primary
shear pairs. It does not evaluate the classifier or disk transport in that
loss. This is a legitimate primary matched-population target, but it differs
from coherent antithetic selected N(+)/D(+)−N(−)/D(−).

All-parent per-leg usable labels do exist in the domain; the problem is not
that the data have no detection labels. They are not included in the current
flow response loss. Dedicated neighbour-only and coherent antithetic targets
are also absent from that current prepared response contract. This audit does
not claim to have inventoried every possible external training dataset.
[Counts, source hashes and coverage](target_coverage.json).

Consequently, adding conditional selected-shape supervision on the existing
pairs could test a primary-response hypothesis. It cannot by itself claim to
supervise the full classifier/selection/neighbour response. Before such a loss,
its reference distribution and remaining emulator contribution must be explicit
to avoid double counting. The known missing neighbour input channel also cannot
be repaired merely by optimizing a function of invariant inputs more strongly.

## Concrete continuation and decision branches

Run the [current-checkpoint measured-input diagnostic](../current_measured_photometry_20261107/rule.md#question-and-fixed-models)
on both axes/all40 scenes with the existing seeds and nine cuts. Both endpoints
have the same conditional shape factor, so their measured-input predictions
must be identical. Check all nonphotometry tensors and conditions, exact paired
conditional draws, and an original-model full-field replay before expansion.

If the component difference remains with measured size/flux/membership, focus
next on conditional shape/transport response and the mismatch between primary
training and coherent evaluation; additional photometry accuracy alone would
not be a complete remedy under that counterfactual. If it largely disappears,
focus on the conditional association of photometry, shape and selection. Mixed
results require both pathways. These branches guide diagnostics, not automatic
acceptance, parameter offsets or attribution of all error to one module.

Do not promote a checkpoint, change the scientific estimator, use these40
reused scenes as training labels, or impose subgroup no-cancellation rules.
A deployable improvement still needs declared objectives, matched training
controls, both axes/nine cuts and eventually fresh acceptance scenes.

## Validation and execution

CPU17821064 completed0:0 in11s. The joint-axis audit verifies240 saved output
files and reproduces1,120 previous scalar values/jackknife errors, passes the
unequal-count and zero-error fixtures, and checks simulation sums across study
families. Measured-input files lacked original data digests; their current
hashes are captured and their old scalar summaries replayed. No model draws or
training were performed by this audit. Training-domain/guard source hashes
match the original preparation receipt. Resource metadata is saved.

The focused measured-input follow-up uses one pilot then six persistent workers,
with at most six GPUs and24 CPUs, and a dependent CPU report. It completed successfully; the [direct zero-shear shape check](../conditional_shape_check_20261107/findings.md#completed-result) also completed. Broad17788286 remains held and automated follow-ups remain off.
