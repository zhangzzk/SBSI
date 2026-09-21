# V3.6-like retained joint model

## Decision and model identity

On 2026-09-20 the owner judged the existing results sufficient to stop this
investigation, named the retained model set **V3.6-like**, and requested
repository cleanup. The original flow, classifier, and emulator remain the
retained models; training replicas are evidence, not deployment ensemble members.

| Component | Retained artifact | SHA-256 prefix |
| --- | --- | --- |
| Single joint flow | `flow_shape_augmentation_20260918_v1/joint_probability_refinement_full/selected.pt`, original step2 | `5434cfafb94e` |
| Single usable-event classifier | `scene_classifier_geometry_20260918_v1/smooth_classifier_full/selected.pt`, seed20260913, epoch99 | `ad975edf7020` |
| Physical-moment disk emulator | `disk_response_moment_20260918_v1/model/model.json`, original 218-tree model | `194c1d11b67d` |

Full hashes, paths, population, results, and the original manifest hash are
pinned in [the V3.6-like manifest](../configs/models_v3_6_like.json). The source
manifest and external artifacts retain their original bytes and names.
V3.5-like is a separate historical model; its meaning is unchanged.

## Goal and retained population

The practical goal was approximately 0.3% joint calibration, reproducible
across the tested sampling and training streams with a single flow. It did
not require every noisy estimate or confidence interval to fit within ±0.3%.

The parent is the generated secondary-role true-r<26 population with rendered
identities. Detection and measurement usability are predicted per leg.
Primary measured cuts are strict `FLUX_RADIUS > 0.6 arcsec` (3 pixels) and
`MAG_AUTO < 25.8`; each leg is selected independently. The joint statistic is
`m = 100*(R_measured/(R_flow + R_blend) - 1)` percent, with coherent antithetic
extraction, 64 common-random-number draws and latent seed7301. Definitions:
[reporting checklist](CONVENTIONS.md#9-reporting-checklist).

## Retained evidence

| Development endpoint | Primary m (%) | Case-bootstrap SE (pp) |
| --- | ---: | ---: |
| Main80, g1, h=0.02 | -0.199979 | 0.263649 |
| Existing40, g1, h=0.05 | +0.402474 | 0.188664 |
| Four additional scenes, g1, h=0.02 | +0.347168 | 0.817220 |
| Same four scenes, g2, h=0.02 | -0.443430 | 0.794924 |

Large-sample paired shifts from one-factor repeats are about -0.01pp for
latent draws, +0.065pp for classifier training, +0.0015–0.0019pp for flow
refinement, and +0.034–0.035pp for emulator training. These quantify the tested
perturbations; independent full flow pretraining and interactions among seeds
were not tested. No favorable seed was selected or averaged into the model.

The primary normalized selected-density score improves by roughly
0.033 nat/object versus the historical classifier. Absolute shape variances
remain underpredicted by about 1.3–1.7% (e1) and 2.4–2.6% (e2); cross-response
residuals remain secondary caveats. Main80 and existing40 differ in both scene
membership and shear amplitude. Existing endpoints informed development.
The retained error bars do not include training or finite-draw uncertainty.

Detailed results and derivations are in the
[archived investigation](../archive/research-2026-09-20/doc/JOINT_CALIBRATION.md#completed-emulator-training-repeat)
and the external `emulator_training_repeat_report/evidence.json` under the
scene-classifier cache. No scientific result was recomputed during cleanup.

## Independent validation stopped

The planned cases22000–22039 had both primary axes at ±0.05 and auxiliary
±0.02 on the first four cases. At cancellation, 34 scenes had completed and
six generation tasks were running: case22000 and cases22035–22039.
Generation array16593294 and all remaining recovery/preparation/binding/
evaluation/report jobs16602287,16594999,16595000,16595003,16595004 and
snapshot16602111 were cancelled on 2026-09-20 at about11:10 CEST.
The scheduler queue was subsequently empty.

Completed products, partial products, render audits, receipts, frozen
protocols, and logs remain in the external `joint_independent_validation_20260919_v1`
and `independent_joint_validation_20260919_v1` directories. No independent
bias result was produced. The 34 completed scenes are not substituted for
the predeclared 40-scene validation. The investigation is closed by owner
decision, with this evidence boundary recorded. Do not restart its jobs
without a new request.

## Runtime and inference boundary

The flow, disk transport, disk-response predictor, classifier loader, and
state-dependent crowding feature functions remain in `sbsi/`. See
[model loading](API.md#v36-like-model-loading).

The configured inference likelihood still composes fixed additive response.
V3.6-like requires per-draw disk velocity with Möbius transport and a
17-input classifier whose geometry follows the current shear state. Merely
replacing checkpoint paths in an older likelihood configuration would not
reproduce it. Naming the model does not silently change inference defaults.

The separately authorized uncut500k inference now has a dedicated disk driver;
it does not restart the stopped model-development investigation. Its production
partitions completed but failed the combined positive-information check. See
[disk inference status](INFERENCE.md#v36-like-disk-driver).

## Research history

Use the [archive inventory and restoration rules](REPOSITORY.md#research-archive)
to locate the original scripts, tests, launchers, and documents. Historical
scientific source hashes refer to the original relative paths; the archive
manifest records their new locations and unchanged hashes.
