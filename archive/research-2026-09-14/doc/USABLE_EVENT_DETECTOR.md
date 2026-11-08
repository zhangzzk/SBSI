# Usable-measurement event detector

## Interpretation correction — 2026-09-13

**Do not interpret the direct-`U` or hurdle results below as tests of a
physical usable-measurement model.** Their development labels were constructed
from `shape_catalogue_detect_position_secondaries_...` while the candidate
truth support contained both input halves. `--targets secondaries` measures
only the randomized second half; the first half receives `NGMIX_G1/G2=-1` by
design. The apparent loss of roughly half the raw detections was therefore
target-assignment bookkeeping, not measurement failure.

The exact case-0 zero-shear audit has 699,568 truth rows with half boundary
349,784. Among crossmatches, all 156,859 first-half objects have the sentinel
and zero are usable; among 157,116 targeted second-half objects, 157,103 are
usable and only 13 have the sentinel. Thus physical usability conditional on
being targeted is 99.9917% in this audit, whereas the invalid prepared support
reported raw/constructed-`U` counts 86,822/43,455. This reproduces the
historical 2026-07-01 finding that the apparent 50% failure was the deliberate
half split and that selection should be `detected` for these catalogues.

The completed model artifacts and tables are preserved as label-contract
controls, but their prior physical conclusion is retracted. They neither
reject nor motivate a direct physical-`U` detector. ConstGold's oracle `U`
comparisons remain valid: that preparation reads
`shape_catalogue_detect_position_all_...`, which measures the full target
population. The frozen audit and correction are recorded in
`usable_event_detector_v1/interpretation_correction.json`.

## Frozen question and comparison — 2026-09-12

The current circularized classifier predicts raw unique crossmatch detection
`D`. The conditional measurement flow, catalogue numerator and observed sample
instead require a usable measurement event `U`: a unique crossmatch with a
joined shape row, finite NGMIX_G1/G2, MAG_AUTO and positive finite FLUX_RADIUS,
valid NGMIX flags and a non-placeholder measured shape. This experiment asks
whether replacing `D` by `U`, and then sharing the flow's three aggregate
crowding inputs, repairs the detector-composition part of the complete response.

The frozen external root is
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/usable_event_detector_v1`.
Its manifest, workers, launchers and hashes were fixed before training. Nothing
in the configured likelihood or model store is changed.

Two arms use cases 0--24 for training, 25--29 for minimum usable-event BCE
checkpoint selection, and 30--39 for development reporting:

| Arm | Label | Conditions |
|---|---|---|
| `u_circularized` | usable `U` | the same 17 circularized inputs as the current raw-`D` detector |
| `u_circularized_crowd` | usable `U` | the same 17 inputs plus `nbr_flux_near`, `nbr_flux_far`, `nbr_flux_max` |

Both use the current 3x128 SiLU architecture, seed 20260912, marginal per-leg
BCE, AdamW settings, scheduler, clipping and tune-BCE selection. The first arm
isolates the event label. The second adds only the three flow crowd summaries.
No paired response loss, response-selected checkpoint, empirical correction,
threshold search or class balancing is used.

An independent case-0 check reproduced the previously prepared masks exactly
after identity alignment. Raw/constructed-`U` counts were 86,822/43,455 at
zero shear and 86,823/43,452 at the half-shear leg. As corrected above, this is
a material *label-contract error*, not a material physical event difference.
Population normalization can cancel the almost constant half assignment, so
the coherent response table below remains numerically reproducible despite
the invalid target.

## Evaluation and interpretation boundary

After tune-BCE checkpoint freezing, every arm and the unchanged raw-`D`
reference are evaluated on the development test cases and coherent ConstGold
cases 40--139. Coherent evaluation reuses the independently reviewed mean of
16 QMC flow groups, the same lambda10 flow, trial-9 R_blend, truth support,
measured cuts and central g1 +/-0.02 ruler. It reports no cuts, magnitude+radius
without measured ellipticity, and all cuts, including selected true-shape and
measurement-residual accounting.

Before either new arm is interpreted, the evaluator must reproduce the archived
actual-`U` and current-detector sufficient arrays from both 50-case blocks.
ConstGold never selects a checkpoint. This is a one-seed mechanism screen, not
an acceptance test: the fixed flow was trained on part of the broader case
universe; only g1 and one amplitude are tested; no g2, zero-shear additive
closure, model-seed replication or final posterior inference is included.

Success would establish that the event/context repair is worth integrating
into the joint hurdle likelihood. It would not establish 0.3% calibration.
Failure of both arms would rule out the minimal tabular repair and motivate a
full-scene representation, rather than more detector response-weight sweeps.

## Execution status

The first preparation, 16468027, stopped safely at case 11 because the
historical crowd lookup lacks one required ID. Its partial outputs and all
dependent-job records are preserved under `failed_attempt_1`; downstream jobs
16468028--16468031 were cancelled before running. Two bounded repair attempts
then exposed API-contract errors in the frozen scene utility, before any model
training: 16468040 called `object_view` on `ScenePrior` rather than its sheared
view, and 16468054 exposed that the frozen crowding implementation cannot use
`primary_indices` because it indexes subset-sized arrays by full-scene rows.
Both exact failures are preserved under `failed_attempt_2` and
`failed_attempt_3`.

The final repair leaves frozen SBSI untouched. Only for cases with an absent
lookup ID, it builds the complete zero-shear object view and then key-selects
the missing and 100 validation rows. Preparation 16468056 completed in 2m39s.
It reconstructed one row in each of development cases 11 and 26; maximum
absolute disagreement with 100 historical rows was respectively
`4.44e-16` and `2.22e-16`. Across all 40 cases it retained 3,883,448 truth
rows and reports raw-minus-usable totals 1,741,936/1,741,902 in the two legs.
No row was dropped or imputed. The original 17-feature preprocessor identity
assertions passed.

Training array 16468063 completed both arms by the frozen tune-BCE rule:
`u_circularized` selected epoch 47 at BCE 0.6544005 and
`u_circularized_crowd` selected epoch 70 at BCE 0.6500675. Development job
16468064 completed on cases 30--39. Against usable-event labels, pooled BCE is
1.781286 for the unchanged raw-`D` reference, 0.653838 for the label-only arm,
and 0.649461 with crowding. The corresponding normalized true-shape response
residuals (predicted minus actual) are 0.000992 +/- 0.000731,
0.005145 +/- 0.000755, and 0.005432 +/- 0.000750 by case SE. Thus predicting
the right event greatly repairs probability level and ordinary scoring, but
does not repair its shear derivative on this development diagnostic; both new
arms are worse there.

Coherent array 16468065 completed all100 cases and review16468066 passed in22s,
both with empty stderr. The reviewer reproduces the two archived actual-U and
current-detector sufficient arrays to maximum absolute errors3.59e-11 and
4.57e-11. All preparation, model and result hashes pass an additional direct
check. The now-unneeded one-shot was retired and only its tagged cron line was
removed.

| Detector weighting | No measured cuts | Magnitude + radius | All cuts |
|---|---:|---:|---:|
| Current raw-`D` | -1.044 +/-0.153% | -0.682 +/-0.126% | -1.939 +/-0.212% |
| Direct `U`, base17 | -1.302 +/-0.152% | -0.957 +/-0.125% | -2.093 +/-0.212% |
| Direct `U`, base17+crowd | -1.088 +/-0.152% | -0.820 +/-0.126% | -1.710 +/-0.213% |

The label-only arm changes m relative to current by
-0.258/-0.276/-0.153percentage points for none/no-e/all. Adding aggregate
crowding changes them by-0.043/-0.139/+0.230pp. Paired case SE of these
within-case changes is at most0.00283pp, so their directions are resolved;
none approaches the0.3% target. All candidate95% intervals remain wholly
outside the target band.

The component results explain why better event semantics and BCE did not
translate to closure. With no measured ellipticity cut, current model-minus-
data response splits into true/residual pieces-0.000066/+0.006979. Direct U
changes these to+0.003199/+0.006540, and U+crowd to+0.003962/+0.004371:
measurement-residual weighting improves, but a new true-shape response error
more than offsets it. At all cuts, direct U reduces the magnitude of both
large opposing component errors yet worsens their total because it removes
some existing cancellation. This is not a reason to preserve the invalid raw-D
event; it is evidence that event relabeling in the same small tabular network
does not supply the missing joint scene information.

Corrected conclusion: this experiment does **not** test a physical direct-`U`
repair and cannot reject one. It instead confirms that a model trained on the
randomized target-half indicator is scientifically meaningless even when its
BCE and probability level look better. For these half-shear catalogues the
valid selection target remains raw detection `D`; rare genuine fit failures
must be assessed only among objects actually passed to the shape fitter. The
remaining coherent bias should therefore be attacked through the learned
detection derivative and conditional measurement distribution, especially
without the unstable measured-|e| boundary, rather than by another model of
this bookkeeping split. No empirical response offset is permitted.
