# Constant-shear error localization, 2026-11-04

Owner requested localization of the discrepancy between standalone checks and the joint constant-shear response. No training, model changes, offsets, or restart of the stopped single-cut run.

## Existing evidence

Re-reduce the 40-case GalSBI-dev constant-grid outputs (1000–1009, 1040–1049, 1080–1089, 1120–1129). Report response-unit shape/common-selection and selection-remainder discrepancies, with paired case errors, and the true-magnitude decomposition. The common-draw split depends on the chosen CRN coupling; it is diagnostic, not a unique physical decomposition.

## Geometry and population diagnostic

On emulator development cases 0–39 at neighbour shear 0.2, rebuild original labels, additionally retaining detections removed by the original >5x bright-neighbour/3arcsec exclusion. Keep all other label validity and measured-baseline cuts radius>3 pixels/MAG_AUTO<25, truth magnitude<30. Label generation remains the existing BlendEMU implementation; only the exclusion is bypassed locally and the rejected identities are recorded.

For the same neighbour identities compare the existing GalSBI-dev emulator with (a) its training distance from measured primary position to true neighbour position and (b) true-primary-to-true-neighbour distance used in constant-shear evaluation. All other features and measured baseline inputs remain fixed. Separate originally included/excluded primaries and true-magnitude bins. Verify originally included pair counts and label/prediction sums against saved emulator sums before interpretation. Unmatched or invalid rows are counted and excluded by the original preparation contract.

Report simulation-minus-model per represented primary, and paired geometry-induced changes using 10000 whole-case resamples, seed20260925. This population shears only secondary neighbours, so it is not numerically interchangeable with all-neighbour coherent constant shear. A large geometry/exclusion effect localizes a deployment-contract gap; a small effect rules out that particular gap on this family, not all composition effects. No claim of a validated correction from this diagnostic.

## Measured-input constant-shear sensitivity (declared before its outputs)

Use cases 1000,1001,1040,1041,1080,1081,1120,1121, within the existing 40-case comparison. Evaluate the frozen emulator on measured radius/flux/shape and true usability at the target cut, comparing the production true-centred nearest-19-neighbour set (k=20 query includes self), measured-primary distances on identical pairs, and up to40 nonself neighbours within10arcsec. Compare exact finite Mobius shifts with the infinitesimal disk derivative. This is a measured-input substitution diagnostic, not a new joint-model calibration: its selected population differs from the generated-draw population. Quantify that combined substitution against the saved model emulator contribution; do not claim it isolates radius/flux from shape/population by itself. Record all unmatched catalogue rows. CPU only; no flow resampling.

## Matched-draw substitution replay (declared after measured-input diagnostic)

Cases1000,1040,1080,1120; frozen original model, 64 draws, same full parent blocks of1024 and original per-case/block seed. At most two GPUs. Evaluate original true-flag emulator response, then replace only the emulator's radius/flux inputs by measured values while keeping generated shapes and selection masks; then also replace the selection masks by measured masks. Finally use measured shapes. The first substitution must reproduce the original summed true-flag response and masses within numerical tolerance before differences are interpreted. Every leg uses actual usability and unchanged pair geometry; unusable rows contribute zero. Flow sampling keeps the complete original blocks even when emulator prediction is restricted to usable rows. These are diagnostic interventions, not deployable corrections or retrained models.
