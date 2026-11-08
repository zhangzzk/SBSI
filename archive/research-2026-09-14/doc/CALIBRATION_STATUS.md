# Shear-calibration status and route to 0.3%

## Bottom line — 2026-09-14

SBSI has **not** demonstrated 0.3% image-level shear calibration. The newest
and most independent available test rejects the frozen minimalist candidate.
On existing whole-scene cases140--239 at `g1=+/-0.05`, with
`MAG_AUTO<25.8`, convolved SExtractor `FLUX_RADIUS>=0.75arcsec`, and no
measured-|e| cut, measured/model R11 is `1.00937065/1.01458544`. Therefore
`m=R_sim/R_model-1=-0.51398%`, with case-bootstrap SE `0.06495pp`, case95%
interval `[-0.64034,-0.38437]%`, and the fully expanded interval
`[-0.64736,-0.37735]%`. The central value and interval both fail the0.3%
gate.

This failure is not a lucky cancellation or a numerical-integration problem.
The truth-population and measurement-residual contributions are
`-0.28341pp` and `-0.23057pp`, respectively, so both reinforce the total.
The four-seed QMC SE is only `0.000541pp`; its Student-t95% half-width is
`0.001720pp`. The physical zero-acceptance support bound is `0.005303pp`, a
narrow miss of its stricter `0.005pp` numerical gate, but reducing that bound
cannot rescue the out-of-band central response. A matched-detection,
separately-selected diagnostic has the opposite signed error,
`m=+0.34934%` with95% interval `[+0.22255,+0.47597]%`, which specifically
shows that the conditional selected-shape model is not solved.

No images were simulated and no measurements were rerun for this result. All
100 existing cases completed with exit0 and empty stderr. Between legs there
are8,646,110 matched detected identities and6,528,018 matched selected
identities. CrossMatch rows without Shapes are0/0; Shapes rows without
CrossMatch are326,784/322,452; invalid-shape drops are439/472; and measured-cut
drops are1,984,466/1,984,987. Every per-case detected-to-usable and
usable-to-selected identity closes exactly.

The proposed two-output selected-shape follow-on is now rejected on model-
contract grounds. Measured magnitude and radius define the per-leg hard
selection, so they must remain endogenous joint outputs whose values move with
shear; a direct Bernoulli `p(S|x)` is not a substitute for the flow-derived
selection normalization required by inference. Response array16491606 and
reviewer16491607 were cancelled after53m39s, before either response artifact or
a decision existed, and their partial logs are retained. The next branch must
keep the same nine inputs and the joint four-output
`(e1,e2,MAG_AUTO,log(FLUX_RADIUS))` target. No new catalogue or image production
is justified until a response-blind candidate inside that contract exists.

The all-cut route is not close enough to justify incremental tuning. Its
large true-shape and measurement-residual errors cancel: model-minus-data
response is `-0.016285 + 0.028277 = +0.011993`. The measured-|e| boundary
also moves different flow branches by 1--2.3 percentage points. A result near
zero after that cancellation would be fragile.

A frozen three-seed raw-`D` detector ensemble with the three missing aggregate
crowding fields improves detector-test BCE/Brier by22.0%/19.3%, but fails its
subsequent coherent mechanism screen. On cases40--139 it moves no-|e| m only
from-0.682 to-0.568%, and removes20.6% of the p-versus-U composition error
instead of the prespecified69.2%. The important split is that it nearly removes
the composition truth error, but leaves a-0.434pp residual-moment contribution.
All three seeds agree, so the remaining failure is structural rather than a
checkpoint accident.

A second frozen experiment removed the remaining label/distribution ambiguity:
it learned actual usable `U` on coherent scenes using exactly the flow's eight
conditions. It also fails. On cases120--139 its ensemble gives
`m=-0.778 +/-0.222%`, versus `-0.830 +/-0.222%` for the current detector and
`-0.243 +/-0.220%` for the U oracle. It removes only8.9% of the current
p-versus-U composition error, worsens the truth component, and leaves a
`-0.492pp` residual contribution. This rejects that eight-input marginal-BCE
MLP. It does not establish sufficiency of its conditioning for the complete
moment: the evaluated shape also contains the separately supplied
`R_blend * (e_g-e_0)` shift, and `R_blend` is not one of those eight inputs.

The final fixed-vector factorial combination—coherent U with the existing
20-feature circularized representative-neighbour/crowding condition—also
fails. On the same cases120--139 it gives `m=-0.664 +/-0.222%` and removes
28.4% of the current composition error. Its residual response error improves
to `+0.003554`, but the truth error worsens to `+0.000715`; the full
`+0.004269` remains far above what this block's U-oracle central value can
tolerate. Rich context lowers U-BCE sharply to0.12370, but still does not
deliver the needed moment calibration.

The subsequent full-scene residual is the first detector candidate to reach
the numerical target, but it is not a valid success. A zero-initialized
permutation-invariant branch over the impact-ranked 10-arcsec neighbour set
lowers test U-BCE from0.12370 to0.09969 and gives no-|e|
`m=-0.279 +/-0.224%` on the same reused cases120--139. It removes93.88% of
the current detector's total composition error and tracks the U oracle within
0.019--0.037 percentage points for all three cut variants. However, its tiny
no-|e| net composition term is a cancellation between a significant
`-0.107pp` truth contribution and `+0.071pp` residual contribution. It
therefore fails the frozen component gate; its95% interval
`[-0.709,+0.165]%` is not inside the target band, and cases140--239 remain
sealed. This rejects one bounded full-scene architecture on an exhausted
development endpoint; it does not prove a detector-family limit. It does show
that scene information is predictive, but not that all eight retained scene
features or the large residual architecture are necessary.

The minimalist follow-on adds only the omitted `R_blend` scalar to the eight
flow conditions and removes their eight always-zero missing indicators. With
the same small2x64 MLP in both arms, mean case-balanced tune BCE changes from
about0.18676 to0.17585 across all three seeds. The paired change is
`-0.010912 +/-0.000111`, with case-bootstrap95% interval
`[-0.011121,-0.010693]`; all seed contrasts are negative. This passes the
frozen proper-score screen using only cases40--119 and adds64 weights. It has
also lowered mean tune Brier from0.05428 to0.05154, while the small signed
probability gap does not consistently improve (three-seed mean about
0.000306 ->0.000337).

Before spending the fresh catalogue budget, the frozen eight- and nine-input
ensembles were inserted into the already verified cases120--139 cached flow
moments. This reran no image simulation, shape measurement, or flow. The
eight-input control gives `m=-0.786 +/-0.222%`,95%
`[-1.214,-0.345]%`; adding `R_blend` improves the net to
`m=-0.372 +/-0.223%`,95% `[-0.800,+0.068]%`. The latter is not a pass and is
formed from `-0.549pp` truth-population and `+0.177pp` residual contributions,
so it fails the frozen anti-cancellation component bound. The exact U-oracle
reproduction is `-0.243 +/-0.220%`. These are reused g1 development results,
not promotion evidence, but they make a760-case fresh test of the unchanged
detector-only candidate an unjustified first step. The one scalar remains
useful. The compact shared-condition test has now answered its response-blind
proper-score gate: the same flow plus `R_blend` improves full held-out NLL by
`0.015496` relative to its function-matched eight-input control, with paired
case-bootstrap95% contrast `[-0.015803,-0.015158]` and improvement in all20
cases100--119. The checkpoint was frozen at epoch16 without reading response.
Its existing-catalogue response screen is now complete and rejects the shared
candidate: measured/model R11 are1.005962/1.012883, giving
`m=-0.683%`, case-bootstrap95% `[-1.118,-0.245]%`. The
truth-population/measurement-residual contributions are
`-0.661/-0.022pp`; only the aggregate residual and numerical-QMC gates pass.
No new images or measurements were generated. The conditional
cases140--239 chain saw the failed gate and exited before building lookups or
opening that response endpoint.

The small aggregate residual is partly cancellation, not a clean flow
validation. With both legs restricted to the same actually selected
identities, the predicted-minus-measured R11 error is `+0.014612`, with
case-bootstrap95% interval `[+0.010746,+0.018267]`. The next experiment is
therefore deliberately narrow: retain exactly the same nine raw inputs, the
frozen flow and the0.75arcsec sample, and test a direct joint selected-event
probability head using only cases40--119 and response-blind proper score.
Neither this failure nor that diagnostic licenses a larger feature vector.

A pre-response provenance audit caught and corrected one acceptance-wrapper
error: its first block attempt pointed to the later v22 `R_blend` regression,
whose primary-radius support starts at0.5arcsec, instead of the trial9 model
used to construct the frozen detector input and the declared0.37arcsec
population. On historical case40, all97,254 ordered trial9 values reproduce
the legacy detector-training reconstruction to maximum absolute difference
`4.44e-16`. Job16470471 was therefore cancelled before `R_blend` construction
and before any response was computed; its29GB partial output is retained as a
failed-provenance artifact. The corrected candidate and job now pin trial9,
and the block guard requires the `R_blend` key set to equal the exact
`18<r<25.8`, `0.37<Re<1.5` truth support case by case. This changes
provenance, not the model or population.

A second pre-response audit corrected the numerical-assurance calculation.
The prior740-case design inferred four-seed QMC error from16 seeds; directly
re-pooling the frozen seeds7301--7304 gives a worse historical50-case SEM of
0.006291pp. The corrected minimum is751 cases, rounded to760 cases1001--1760
in38 blocks. This changes only sample size and the later response-task map;
the nine inputs, model artifacts, population, QMC seeds and0.3% gate remain
frozen. No fresh response has been inspected.

A third static audit traced the actual renderer, measurement executable and
random seeds. The corrected block0 wrapper still used a mutable
`MultiBand_ImSim` checkout and a generic SExtractor module; the latter resolved
to the x86-64-v2 SExtractor2.29.0 binary built on2026-08-27. That build
postdates the candidate's July source catalogues, so job16470674 was cancelled
after1h23m15s, before `R_blend` or response, and its29GB partial output is
quarantined. An initial2.28 correction based on a stale configuration note was
also stopped after6m44s, before shape measurement: retained producer logs for
training cases40,100 and119 directly record SExtractor2.29.0 on2026-07-09.
Production now uses the surviving non-v2 RoyBranch2.29.0 binary built on
2026-05-12, explicitly loaded and hashed, plus a read-only byte-equivalent
renderer snapshot and pinned Python/package versions. The historical logs do
not retain an executable hash, so identifying this pre-training build as the
compatible binary is a date-grounded inference, not exact binary recovery.
The render code is clean at commit `1266a745`; among the old checkout's dirty
paths only the active `kids_sims.sex` mattered and its exact bytes are in the
snapshot. The seed trace also establishes that cases independently resample
galaxy identities and positions but reuse one18181x18181 Gaussian pixel-noise
field, while all shear legs share both scene and noise as required.
Consequently the case bootstrap is conditional on this very large frozen noise
field; it is not evidence across independent image-noise seeds. Adding a
second full noise-seed campaign is not justified before seeing the
prespecified fresh response.

The extraction-config identity has the same bounded provenance limitation.
The active `kids_sims.sex` file's SHA is frozen and its mtime/ctime are both
2025-06-16, before the July2026 source catalogues. The retained case40 log
independently reports threshold/RMS=`0.504284/0.336189=1.5000015`, matching the
dirty overlay's1.5sigma threshold rather than the committed1.2 default. The
producer did not retain the complete config bytes or their hash, so the other
overlay parameters are date-grounded rather than independently byte-recovered.

The current detector feature boundary is already the smallest defensible one for this
factorization. Eight always-zero missing indicators were removed. The retained
eight flow conditions are exactly the variables used by the unchanged
conditional measurement model—primary e1/e2, Sersic index, magnitude,
circularized radius, and three compact neighbour-flux summaries—and the ninth
input is the separately applied `R_blend` term whose omission has a decisive
held-out proper-score cost. There is no representative-neighbour vector,
redshift, profile expansion or scene network. This is not proof that every
input is causally indispensable, but deleting one now would break the exact
conditioning interface and begin another model-selection loop. The equal
three-seed average is retained to avoid choosing a lucky checkpoint; it adds
no features and is cheap relative to flow evaluation. The cached response
screen now rules out spending directly on fresh acceptance, but it does not
justify a larger feature vector. Sharing the ninth scalar with the
measurement flow improved NLL and the aggregate residual term yet failed the
truth-population gate. The admitted follow-up changes only the event being
predicted—from usable measurement `U` to the exact deployable joint event
`S = U & (MAG_AUTO<25.8) & (FLUX_RADIUS>=0.75arcsec)`—while retaining the
same nine inputs and2x64 MLP.

The fresh decision now also closes a small anti-cancellation gap found before
any response was available. In addition to the expanded95% total-m interval
and QMC gate, each endpoint requires the central truth-population and
measurement-residual m contributions to lie strictly inside+/-0.3pp. This
rejects a nominal total pass formed by cancelling an individually out-of-band
component. It deliberately does not treat the algebraically dependent
decomposition as two additional equivalence intervals or add a tunable
cancellation-fraction threshold.

## Current coherent ruler

All values below combine cases 40--139, central coherent g1 `+/-0.02`, the
mean of 16 independent 64-draw QMC groups, lambda10 epoch154 flow, ordinary-
MSE trial9 `R_blend`, and the circularized raw-detection classifier. The parent
rows are distinct controls and must not be mixed.

| Parent/weighting | No measured cuts | Magnitude + radius, no \|e\| | All cuts |
|---|---:|---:|---:|
| Fixed both-usable `B` | +0.318 +/-0.152% | +0.499 +/-0.121% | -0.576 +/-0.212% |
| Actual per-leg usable `U` oracle | -0.422 +/-0.157% | **-0.129 +/-0.125%** | -1.187 +/-0.216% |
| Learned raw-`D` probability `p` | -1.044 +/-0.155% | **-0.682 +/-0.126%** | -1.939 +/-0.216% |
| Frozen raw-`D` crowd ensemble | -0.820 +/-0.155% | **-0.568 +/-0.126%** | -1.601 +/-0.217% |

For the no-|e| `p` endpoint, measured/model R11 is
`1.00745477/1.01436770`. The 0.3% band requires model R11 in
`[1.00444145,1.01048623]`; the present model is 0.00388147 above the nearest
edge. Its full response error is 2.28 times the allowed positive error, so at
least 56% of that signed error must be removed before the central value enters
the band. Under all cuts the response error is about 6.6 times the allowed
positive error.

The no-|e| learned-`p` response error is almost entirely conditional
measurement residual: the true-shape term is `-0.000066`, while the residual
term is `+0.006979`. Under the actual-`U` oracle the corresponding residual
error is only `+0.000334`. Marginal detection accuracy or the isolated
true-shape selection derivative is therefore insufficient; the probability
model must weight the same scene strata in which the flow's measurement
residual changes.

## Where the no-|e| error lives

An exact additive audit of all 9,709,701 objects in cases 40--139 now separates
the current learned-`p` result into two steps, using each step's own global
normalization. Contributions below use the final learned-`p` response as the
common denominator, so they add exactly to `m=-0.682%`.

| Step | Full contribution to m | True-e part | Measurement-residual part |
|---|---:|---:|---:|
| Flow with actual per-leg `U` | -0.129pp | -0.096pp | -0.033pp |
| Replace `U` with learned `p` | **-0.553pp** | +0.102pp | **-0.655pp** |
| Total | **-0.682pp** | +0.007pp | **-0.688pp** |

The detector-composition residual is not a cancellation across the tested
one-dimensional bins: every true-size, Sersic, and aggregate-crowding bin has
the same negative sign. In the predeclared magnitude-by-near-crowding grid,
the cells with true magnitude at least 24 and `nbr_flux_near>=0.5` contain
45.4% of rows but contribute `-0.599pp` of the `-0.655pp` composition-residual
term. Including their true-e contribution, the same cells contribute
`-0.530pp`, or 95.8%, of the full `-0.553pp` composition step. Representative
neighbours at 1--3arcsec contribute 94.1% of the residual term; neighbours
with log primary-to-secondary flux ratio below 1 contribute 93.4%.

These are overlapping, post-result summaries of fixed exhaustive partitions;
they are not independent effects, a causal decomposition, or permission to
tune a population cut. They do show that the error is structured rather than
case noise. The detector already sees one impact-ranked neighbour, so the
finding is not simply "add separation." Its material information mismatch
with the flow is the flow's three aggregate 0--7arcsec crowding summaries.

Adding `nbr_flux_near/far/max` to the valid raw-`D` detector is therefore the
smallest justified next architecture test. With those fields, the detector's
20-feature condition contains recoverable versions of all eight flow
conditions (primary e1/e2, Sersic n, magnitude, circularized radius, and the
three crowd summaries), plus its detailed representative-neighbour fields.
That nesting applies to the unshifted flow moment, not automatically to the
complete evaluated moment. For the retained no-|e| sample the latter also
contains `R_blend`; exact tower composition therefore requires conditioning on
`G=(F,R_blend)` or a demonstrated vanishing conditional covariance. The
earlier crowd-augmented usable-event arm does not answer this question because
its `U` label was the invalid randomized-half target.

The completed test now shows where that finite-model argument breaks. For the
crowd ensemble, the p-versus-U full/truth/residual response differences are
`+0.004450/+0.000055/+0.004394`, contributing
`-0.439/-0.005/-0.434pp` to its final m. The old detector's corresponding
contributions are `-0.553/+0.102/-0.655pp`. Thus the added fields repair the
true-shape weighting and about one third of the residual error, but the
conditional measurement residual is still misweighted. The ensemble selected
count is already close to the U oracle; a global probability offset or
temperature cannot repair this conditional error.

The external `no_e_residual_strata_v1` audit fixes all bins before reduction,
reproduces every archived per-case sufficient sum, and independently rebuilds
all 13 partitions, 116 bins, point estimates, and 10,000 case-bootstrap
intervals. It uses the reused development endpoint and cannot validate or
promote a replacement.

## What the recent branches actually establish

| Branch | Valid conclusion | Disposition |
|---|---|---|
| Lambda10 epoch154 + trial9 blend | Best current coherent baseline; oracle `U` nearly closes only without measured \|e\| | Retain as baseline, not a release claim |
| Circularized detector | Repairs much of the isolated intrinsic-shape detection response, but complete m stays near -2% with all cuts | Retain current detector; representation improvement alone was insufficient |
| Raw-`D` + aggregate-crowding ensemble | BCE/Brier improve22.0%/19.3%, but coherent composition improves only20.6%; no-|e| m remains-0.568%, with-0.434pp residual error | Rejected by its prespecified69.2% mechanism gate |
| Coherent-U, flow-condition detector | Valid U labels and coherent +/-0.02 deployment distribution; exactly the unshifted flow's eight-variable condition but not external `R_blend` | Rejected: only8.9% composition improvement; no-|e| m=-0.778%; truth worsens and residual remains-0.492pp |
| Minimal coherent-U flow+`R_blend` detector | Same small2x64 model as an eight-input control; nine inputs,4,865 parameters; paired tune-BCE change -0.010912 with95% interval[-0.011121,-0.010693] | Proper-score screen passed; freeze for newly simulated acceptance, with no response claim |
| Coherent-U, rich fixed-vector detector | Valid U labels plus circularized impact-ranked neighbour and aggregate crowding; U-BCE0.12370 | Rejected: 28.4% composition improvement, no-|e| m=-0.664%; truth component worsens |
| Coherent-U, additive full-scene detector | Frozen rich base plus zero-initialized top-32/10-arcsec DeepSets residual; U-BCE0.09969; total composition tracks U within0.037pp | Reject promotion: no-|e| m=-0.279% is a truth/residual cancellation, its95% interval is[-0.709,+0.165]%, and the component gate fails |
| Direct selected-event nine-input ensemble | Exact `p(S|x)` improves held-out BCE on all20 cases; existing cases140--239 give m=-0.51398%, expanded95% interval[-0.64736,-0.37735]%, truth/residual=-0.28341/-0.23057pp | Rejected for direct promotion; endpoint is consumed |
| Two-output selected-shape disk flow | Same nine inputs and direct selector; trained only on existing selected rows by held-out NLL | Rejected on model-contract grounds; magnitude/radius must remain joint outputs because they define selection; response jobs cancelled before completion |
| Covariance grid216 lambda1 flow | Training-scene `U/no-e=-0.025%` does not generalize: coherent `U/no-e=+1.153 +/-0.127%`, `p/no-e=+1.017 +/-0.128%`, with same-sign truth/residual errors | Rejected; retain lambda10 epoch154 |
| Identity/no-covariance grid216 | NLL recovers to -2.48103, but shape trace remains +1.581% and cell scores are worse than covariance weighting | Do not promote; no selection result |
| Direct usable-`U` detector | Target was invalid: `_secondaries` bookkeeping labeled the randomized first half as failures | Physical interpretation retracted |
| Two-stage `U|D` hurdle | BCE converged to ln(2), correctly exposing the random half assignment | Cancelled; negative label-contract control only |

The usable-label correction matters. In an exact case-0 audit, all 156,859
crossmatched first-half objects have the deliberate `-1` sentinel because they
were not targeted. Of 157,116 targeted second-half crossmatches, 157,103 are
usable: 99.9917%. ConstGold's actual-`U` oracle remains valid because its shape
catalogues measure all targets. The target error invalidates the two new
detector interpretations, not the older coherent baseline or the properly
restricted flow-training self-response table.

## Honest improvement plan

1. **Freeze the candidate science sample without measured |e|.** Keep the
   existing magnitude and radius thresholds unchanged. This is a development
   decision motivated by an unstable boundary; it must be validated once on
   fresh simulations and must not be retuned there.

2. **Retire the covariance-flow branch.** Its100-case coherent screen changes
   no-|e| `U` from-0.129% to+1.153% and learned `p` from-0.682% to+1.017%.
   The p-model response0.997311 lies0.007131 beyond the nearest allowed response
   boundary, versus0.003881 for lambda10. Truth and residual errors reinforce
   rather than cancel. The apparent training-scene oracle closure was in-sample
   and did not generalize; no more covariance-flow tuning is justified on this
   endpoint.

3. **Retire the raw-D crowd ensemble.** Its coherent20.6% improvement is real
   but far short of69.2%, and the final-0.568% is not a cancellation-free entry
   into the target band. Do not choose a seed, ensemble weight, temperature, or
   probability offset using cases40--139.

4. **Retire the eight-input coherent-U MLP, but keep its control result.** The
   frozen `coherent_u_flow_condition_ensemble_v1` experiment used valid
   per-leg usable `U`, cases40--99/100--119/120--139 for train/BCE tune/test,
   and no response-selected checkpoint. On1,938,865 test objects, U/current/
   candidate no-|e| m is-0.243/-0.830/-0.778%. The candidate's full composition
   response error is+0.00543363 versus current+0.00596611: an8.9% reduction,
   not69.2%. Its truth/residual errors are+0.000449/+0.004984, so even the
   component gate fails. Its condition omitted the external `R_blend` term, so
   do not describe this result as a test of exact nesting for the complete
   moment. Cases140--239 stay sealed.

5. **End fixed-vector detector work.** The frozen
   `coherent_u_crowd_ensemble_v1` completed the missing factorial combination.
   Its U-BCE0.12370 is far better than the eight-feature model's0.18560, but
   full composition error only falls from+0.005966 to+0.004269 (28.4%), truth
   worsens from-0.000121 to+0.000715, and m remains-0.664%. All three seeds are
   between-0.655 and-0.673%. No further representative-neighbour feature,
   temperature, offset, or fixed-vector architecture should be selected on
   cases40--139.

6. **Retire the tested full-scene architecture and its response endpoint.**
   Across9,709,701 staged primaries, the frozen
   `coherent_u_full_scene_residual_v1` experiment has
   zero unmatched IDs; only6,651/164,255,670 edges and3,481/9,709,701
   primaries are affected by the top-32 cap. On reused cases120--139 it lowers
   rich-model U-BCE0.12370 ->0.09969 and moves no-|e| m
   -0.664 ->-0.279%, but its p-versus-U truth/residual contributions are
   -0.107/+0.071pp. The total93.88% improvement is therefore not the
   componentwise identity that the factorized estimator needs. This rejects
   this 20-feature base plus eight-feature set residual; it does not rule out
   every detector-only model. Do not tune another model on this response
   endpoint, and do not open cases140--239 after the failed gate.

7. **The one-omitted-scalar detector screen passed, but its reused response
   screen did not.** For the
   no-|e| moment, use only the eight frozen flow conditions plus `R_blend`.
   The completed comparison used that nine-input model and an identical
   eight-input
   control using cases40--99 for training and100--119 for tune coherent-U BCE;
   make the paired comparison at the simulation-case level. Remove the eight
   always-zero missing indicators and use a small common architecture. Do not
   read cases120 onward or any response statistic. Continue only if all three
   seeds improve and the paired case-bootstrap95% interval for the BCE change
   is below zero. The observed paired change is
   `-0.010912 +/-0.000111`,95% interval[-0.011121,-0.010693], with all three
   seed contrasts negative. The later cached cases120--139 response is
   `m=-0.372 +/-0.223%`, with truth/residual contributions
   `-0.549/+0.177pp`; this fails the anti-cancellation component gate. Keep the
   detector frozen as a useful minimal component, but do not send the
   unchanged detector-only candidate to fresh simulation.

8. **The same-scalar measurement-flow proper-score gate passes, but its
   cached response screen rejects the candidate.** The epoch154 flow has eight real context columns followed by eight
   missing-value indicators that are exactly zero in its existing train and
   validation arrays. Compare a function-preserving eight-input control with
   the same flow in which those dead columns are removed and one
   zero-initialized `R_blend` column is appended. Use existing measurement
   catalogues only: cases40--99 for NLL continuation and100--119 for NLL
   checkpoint selection plus a paired case-bootstrap proper-score gate. Cases
   120--139 and every response statistic remain firewalled until the checkpoint
   is frozen. The nine-input arm improves pooled held-out NLL by `0.015496`,
   its paired case-bootstrap95% contrast is
   `[-0.015803,-0.015158]`, and all20 validation cases improve. Epoch16 is now
   frozen. Job16476074 then used only existing cases120--139 and generated no
   images. It gives `m=-0.683%`,95% `[-1.118,-0.245]%`, with
   truth/residual contributions `-0.661/-0.022pp` and QMC SE `0.00204pp`.
   This fails the central and truth-population gates. The conditional
   cases140--239 jobs exit before reading that endpoint. Do not default to the
   old20-feature vector, neighbour encoder, or DeepSets trunk.

8a. **Reject the direct selector plus unchanged U-flow, and retire the minimal
   selected-shape flow.** The properly frozen direct selected-event candidate
   was evaluated on existing cases140--239 and gives `m=-0.51398%`, expanded
   95% interval `[-0.64736,-0.37735]%`, with same-sign truth/residual
   contributions `-0.28341/-0.23057pp`. More draws cannot repair that central
   failure. A response-blind experiment then fit a two-output disk flow for
   `e1,e2` on the selected rows and froze `refine013.pt` by held-out NLL. This
   experiment is retired before response completion: collapsing measured
   magnitude and radius into `p(S|x)` removes the endogenous cut variables
   needed by the inference selection normalization. Jobs16491606/16491607
   were cancelled after53m39s and produced no response or decision artifact.
   Preserve the result as a negative control; do not promote, resume, or send
   it to fresh simulation.

The failure routing was frozen before that response became available.  A QMC
gate failure changes only the numerical integration budget.  At the independent
cases140--239 endpoint, a confidence-interval failure with central `m` and both
components inside band is an imprecision result: keep the model fixed and add
independent cases rather than changing features.  An out-of-band
truth-population contribution opens only a same-nine-input detector/selection
calibration branch with the flow fixed; candidate choice must use response-blind
proper scores.  An out-of-band measurement-residual contribution opens only a
same-nine-input flow-refit branch with the detector fixed; candidate choice must
use held-out NLL.  If both fail, test those two minimal branches separately.
Do not move to a joint representation or add raw inputs unless both
response-blind branches earn that escalation and subsequently fail on a new
holdout.  The truth/residual decomposition routes diagnostics, but is not by
itself a causal attribution: the truth-population term includes the complete
detection-and-selection weighting.

The observed failure takes the prespecified same-nine-input selection branch.
The aggregate residual gate passes, but the common-selected-identity R11 error
is `+0.014612` with95% interval `[+0.010746,+0.018267]`, so the flow is not
declared solved. First test the exact joint selected event with proper score;
if that cannot beat the frozen composed probability on cases100--119, stop
without opening another response endpoint. If it does, freeze the direct head
before using any cases120 onward.

That response-blind test is complete and passes. The frozen composed
probability has mean case BCE0.366176; the three direct heads give
0.361191/0.361214/0.361070 and their equal-probability ensemble gives
0.360738. The ensemble delta is `-0.005438`, case-bootstrap95%
`[-0.005568,-0.005315]`, with all20 cases improving; its upper endpoint
expanded by numerical QMC uncertainty is still `-0.005299`. Only existing
cases40--99 for training and100--119 for tune proper score were used. No case
at or above120 and no response statistic was read. This qualified one frozen
response test on the then-sealed cases140--239 catalogue; it was not a shear
calibration claim.

The now-retired diagnostic response estimator was frozen in direct form:
`sum_x p_S(x) E_flow[e|U,C,x] / sum_x p_S(x)`. The selected-event head supplies
the complete event probability and the unchanged flow supplies only the
conditional selected-shape mean; the cut probability is not multiplied a
second time. The cases100--119 support-only audit chose256 draws as the smallest
passing budget. The subsequent existing cases140--239 result is
`m=-0.51398%`, expanded95% interval `[-0.64736,-0.37735]%`, with
truth/residual contributions `-0.28341/-0.23057pp`; it rejects this candidate.
The observed four-seed QMC SE is `0.000541pp` and Student-t95% half-width is
`0.001720pp`. The worst-case support contribution is `0.005303pp`, narrowly
above its numerical bound but immaterial to the central scientific rejection.
This direct factorization is not a replacement for the four-output inference
likelihood: the deployed measured magnitude/radius cut must be derived from
their joint distribution with measured shape.

The measured-size rule should not be confused with the common ngmix resolution
rule. `FLUX_RADIUS` is a circular half-light radius; ngmix `T` is a moment trace
with squared-angle units. For this Moffat PSF, `R50_PSF=0.526782arcsec`, so the
current0.75arcsec threshold is1.424 PSF half-light radii, not
`0.5*T_PSF`. Neither galaxy `T` nor PSF `T` is retained in the shape
catalogues, and `T` is not a target of the four-output measurement flow. An
exact `T/T_PSF>=0.5` cut therefore cannot be claimed from these artifacts. A
half-FWHM `FLUX_RADIUS>=0.365arcsec` proxy would be a different, explicitly
declared sample and must not be selected after reading its shear response.  The
owner clarified before the pending result was visible that the0.75arcsec cut
should remain.  This is a convolved measurement: SExtractor measures
`FLUX_RADIUS` on the PSF-convolved detection image.  It is not a
PSF-deconvolved ngmix `T`, and no0.365arcsec response comparison will be run. A
response-blind count on cases120--139 shows that it retains99.996% of usable,
`MAG_AUTO<25.8` rows in each leg; even the0.526782arcsec PSF-half-light-radius
threshold retains99.918%. The current0.75arcsec threshold retains78.59%.
Consequently either radius proxy is effectively no radius cut for these
catalogues, not an observational implementation of the ngmix resolution rule.

The existing cases140--239 endpoint has now been used exactly once for the
proper-score-qualified direct selected-event candidate. It is independent in
galaxy realization and shear amplitude from the development response, but it
rejects that candidate and is now consumed for model selection. It cannot be
reused as fresh evidence for the selected-shape follow-on. There is
no matching catalogue-population g2 or g=0 set: the ten-case fixed-g2
cases400--409 material targets a special anchor/non-anchor mechanism and is
not an R22 acceptance sample.  This is explicit in the retained case400 input:
each pilot leg has 174,892 rows, of which 171,183 carry `g2=+/-0.05` while 3,709
remain at `g2=0`; the ordinary case140 endpoint has 699,568 rows all carrying
the same nonzero shear (`gamma1=0.05`, `gamma2=0`).  Thus the pilot changes both
the scene population and the shear assignment, rather than rotating the target
whole-scene experiment into g2.  The half-shear store does have g=0 products for
cases0--199, but they belong to the forward-training family and a different
legacy shape-file contract.  They can be considered for a labelled additive
diagnostic after overlap/version checks, not substituted for the untouched
fully targeted g=0 acceptance leg.

9. **Treat genuine measurement failure only on targeted rows.** Quantify
   `F=usable|D,targeted` separately. If it stays at the audited 99.99% level,
   document it as a bounded approximation or make the measurement stage return
   a defined output for every detection. Build a separate hurdle only if fresh
   targeted-row data show that `F` is material.

10. **Qualify the minimal selection branch, then exhaust the existing
   independent g1 catalogue.** Keep cases140--239 sealed while the exact
   selected-event head is selected only by cases40--119 proper score. If it
   passes its frozen gate, run that unchanged candidate with the retained
   0.75arcsec cut on existing whole-scene cases140--239 at `g1=+/-0.05`.
   Do not render this already available endpoint again.  Only if both existing
   g1 amplitudes survive should new production supply the missing g2 and g=0
   evidence.  Final acceptance still requires g1 and g2, a g=0
   additive check, at least two nonzero shear amplitudes, at least three model
   seeds or a frozen ensemble, both response closure and full inference, and
   all declared unmatched/invalid rows. The conservative pass rule is that
   the full 95% interval for each multiplicative component lies within
   `[-0.3,+0.3]%`; central-value entry on a reused diagnostic is only a screen.
   At the present `0.126pp` case SE, 100 cases can meet that rule only if the
   central value lands within about `+/-0.053pp` of zero. If the true central
   value stays near the `U`-oracle `-0.129%`, 210 cases merely put the
   *expected* nearer confidence bound at about `-0.3%`; it does not make
   acceptance likely. Under the planning model
   `SE_N=0.126 sqrt(100/N)pp`, the exact normal-approximation probability that
   a marginal two-sided95% interval is wholly inside the band is only0.50 at
   N=210 and0.56 at N=240. The required independent-case counts are:

   | Assumed true \|m\| | 80% endpoint pass | 90% | 95% |
   |---:|---:|---:|---:|
   | 0.000pp | 186 | 230 | 272 |
   | 0.050pp | 210 | 272 | 332 |
   | 0.100pp | 312 | 418 | 516 |
   | 0.129pp | 427 | 571 | 706 |
   | 0.150pp | 554 | 742 | 917 |
   | 0.200pp | 1,247 | 1,669 | 2,064 |

   Four endpoint gates need stronger marginal assurance. Case uncertainty
   alone needs706 cases for at least95% marginal pass probability and hence an
   80% four-endpoint union-bound floor. A pre-response compute review also
   removed unnecessary QMC replication: on independent historical50-case
   blocks, the no-|e| 16-seed integration SEM was0.00132 and0.00200pp. Exact
   re-pooling of the already frozen four64-draw seeds7301--7304 gives0.00212
   and0.00629pp SEM at50 cases; the earlier0.00401pp estimate inferred from16
   seeds was optimistic. Pooling both historical blocks lowers the same
   frozen-four value to0.00304pp at100 cases, supporting conservative
   `1/sqrt(N)` planning without permitting seed selection. Fresh acceptance
   requires QMC SEM<=0.005pp and expands its case-bootstrap interval by the
   four-seed Student-t95% half-width before the0.3% gate. Anchoring to the worse
   actual four-seed SEM needs751 cases for95% marginal assurance; round to760,
   giving0.95245 marginal probability and at least0.80979 joint probability by
   the union bound. This is a conditional design calculation, not evidence
   that the candidate is near the oracle. If its true bias is near the current
   learned value, more cases only reject it more precisely. The smoke consumed
   case1000, so use untouched cases1001--1760; bounded directory, top-level
   range-name, and recent registered-metadata audits found no case1760 use.
   Each case is fully targeted at
   `(g1,g2)=(0,0),(+/-0.02,0),(+/-0.04,0),(0,+/-0.02),(0,+/-0.04)` with common
   scene/noise bookkeeping across legs. Cluster inference by case and require
   the gate independently for both axes and both amplitudes. Model seeds do
   not reduce this simulation-case floor. Process20-case blocks endpoint by
   endpoint and retain catalogues/sufficient statistics, not all raw images:
   keeping all nine raw legs would need about8.9TB. The earlier launch audit
   found6.6TB free, but a live capacity recheck at2026-09-13T14:44+02 found
   only1.23TB available on the100%-reported shared filesystem. Block0 remains
   the sole capacity pilot. Set the blocks1--37 array concurrency only after
   its validated pruning, using its measured peak/retained footprint and a
   fresh free-space check; do not inspect response to make that operational
   choice. The historical validated constant-shear implementation is BlendEMU
   commit `46107b86`; the later current branch no longer carries that mode, so its source
   identity must be frozen before launch. Response is the computational gate;
   run the much more expensive full-inference stage only if all response gates
   pass, without changing the candidate or thresholds.

   The exact candidate hashes, legs, QMC streams, population, gates, storage
   rule, and forbidden changes are frozen in
   `configs/calibration_acceptance_minimal_rblend.json`.

   That file and its response worker are now a historical protocol for the
   rejected four-output Flow-E plus coherent-U detector arm and must not be
   launched. The direct-selector plus two-output selected-shape branch is also
   retired and opens no successor protocol. Any future candidate must retain
   the four joint measured outputs and earn a response-blind gate before the
   unchanged simulation legs, no-|e| sample, 0.75arcsec convolved-radius cut,
   QMC streams, and acceptance gates can be re-pinned.

   Execution status: the capped case1000 nine-leg plumbing smoke passed, so it
   is excluded from acceptance. The production range is cases1001--1760.
   The first block0 attempt, job16470471, was cancelled before response after
   the wrong v22 `R_blend` provenance was caught; its partial files are
   preserved. A second block0 attempt, job16470674, was cancelled before
   `R_blend` or response because its default SExtractor2.29 binary was an
   August build that postdates the July training catalogues; its partial output
   is also preserved. Job16471078, briefly pinned to2.28 from a stale note, was
   cancelled during image simulation after retained training logs proved the
   actual2.29 version. Full-density block0 (cases1001--1020) restarted as Slurm
   job16471110 with the exact trial9 Re>0.37 artifact, immutable renderer/config
   snapshot, hashed pre-training SExtractor2.29.0 build, and corrected38-block
   guard. It passed preflight, but was cancelled during g0 shape measurement
   after the existing-catalogue sequencing review; it produced no response.
   Blocks1--37 are not launched. The later four-endpoint response worker is prepared but
   has no default array and refuses to start until all38 simulation receipts
   pass; it uses only four frozen64-draw QMC groups and one pooled bootstrap.
   A hash-pinned decision reducer will collapse the four endpoint artifacts
   into one verified pass-to-inference or reject-candidate record. No
   production response or0.3% claim exists yet. Existing cases120--139 cached
   moments now reject the frozen shared-condition candidate at
   `m=-0.683%`; its conditional cases140--239 chain exited before opening that
   response endpoint. The existing `g1=+/-0.05` catalogues were then consumed
   once by the response-blind-qualified direct selected-event head and reject
   it at `m=-0.51398%`. The selected-shape follow-on was cancelled before its
   response completed because it violates the four-output selection contract.
   Fresh production stays stopped; no current candidate is qualified. Full inference is deliberately
   not presented as ready: the generic image adapter assumes one flat
   simulation root, and the generic inference summary is not the required
   38-block case-clustered multi-axis reducer. Design and resource-size that
   stage only if all four response gates pass.

## What not to do

- Do not fit or paste a response offset, rescale `R_blend`, or tune thresholds
  until this reused endpoint reads zero.
- Do not select checkpoints by ConstGold m or by an accidental cancellation.
- Do not substitute fixed both-detected objects for the deployable population.
- Do not infer shear calibration from NLL/BCE alone; both have already improved
  while response closure remained wrong.
- Do not promote the full-scene detector because its reused no-|e| central
  value is barely inside0.3%; it failed the frozen component gate and its
  confidence interval does not certify the band.
- Do not spend more compute on the invalid half-assignment `U` label.

This document is the current decision layer. Detailed provenance stays in the
linked experiment documents and the newest entries of `WORKLOG.md`.
