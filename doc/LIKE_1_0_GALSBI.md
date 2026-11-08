# like-1.0-galsbi

## Decision and identity

On 2026-11-08 the owner selected **like-1.0-galsbi** as the current GalSBI
fiducial. It is the existing `control_nine` model, retained without retraining
or modifying weights. The complete artifact set is pinned in
[`configs/models_like_1_0_galsbi.json`](../configs/models_like_1_0_galsbi.json).

- Flow: `neighbour_conditioned_flow_20261106/control/nine.pt`, SHA256
  `9bc10a6548e792cdcbb29b226e20e4e53dff74d88c1a29a972e246cb07227417`.
- Classifier: GalSBI C8 seed20260913, SHA256
  `ad5c05d2dcd9af9d711b18b532cd681ac5de6df6350bab288cdec94a3019a3c6`.
- Emulator: GalSBI physical-moment disk response, SHA256
  `cd32781db7edd2cab576a2ef7478d722df2984788bc83a7fe12caa6adf19fb43`,
  with output-conditioned photometry and Möbius transport.

The name does not denote the older fully joint Cartesian flow, the original
`nine_weak` continuation, the `neighbour_nine` arm, or the latest weighted
photometry continuation. Historical V3.9/V3.6 identities remain intact.

## Architecture and training

The factorization is p(radius,flux | x) times
p(shape1,shape2 | radius,flux,x), in Cartesian shape coordinates.
The original eight raw galaxy/crowding inputs are preprocessed with missing
indicators. This control checkpoint retains an appended constant-zero column
from the matched neighbour-input experiment: `use_score=false`. It does not
use the emulator-derived neighbour score as an informative input.

Training was initial140-epoch NLL, another40 NLL epochs, then64 updates with
NLL plus the weak nine-cut primary-response penalty, weight0.001. Both flow
factors were trainable. The model was not retrained on the new paired-seed
measurements, and constant-shear simulations were not training targets.
This retained history is not evidence that all stages are necessary or that
a one-run recipe has been demonstrated.

## Paired-seed development evidence

The saved model predictions were compared with the new paired-seed measured
responses on cases1000,1040,1080,1120. Predicted usability and generated
selection are used on the model side; each measured/generated leg has its own
selected normalization. Central cut: radius>3 pixels (0.6arcsec), magnitude<25.

| Statistic | Value | Whole-scene jackknife SE |
|---|---:|---:|
| m1 | +0.38554% | 0.39755 percentage points |
| m2 | +0.04296% | 0.66616 percentage points |
| Both-component, nine-cut RMS | 0.53431% | 0.25735 percentage points |

Here m_i=100*(R_measured,ii/R_model,ii−1), and the nine cuts cross
radius>2.8/3.0/3.2pixels with magnitude<24.9/25.0/25.1.
See [matched stage results](evidence/training_stage_audit_20261108/results.md#matched-four-scene-stage-comparison)
and [response conventions](CONVENTIONS.md#6-response-estimators).
The manifest pins the numerical report and records its exact values.

Only four reused development scenes have this paired-seed comparison; no
independent acceptance or robust0.3% claim is established. There is no completed
40-scene paired-seed result for this exact checkpoint. Conditional photometry
and selection errors have not been established as solved. Good total response
can contain cancellation. The new name records the owner's working reference,
not a stronger validation result.

## Loading and inference boundary

With no external-root overrides, the included artifacts load directly.
`SBSI_CACHE_DIR` and `BLENDEMU_RUNS_DIR` optionally select external artifact
roots. To inspect and verify the fiducial:

```bash
python -m sbsi show-model like-1.0-galsbi
python -m sbsi validate-model like-1.0-galsbi
```

`sbsi.models.get_model("like-1.0-galsbi")` returns the pinned model paths;
`sbsi.models.CURRENT_FIDUCIAL` identifies this same model set. The existing
classifier/disk-emulator loaders accept these companion paths.

The flow checkpoint format is `sbsi_experimental_neighbour_factorized_v1`.
It now loads through `sbsi.measurement_model.load_measurement_model`, backed
by the permanent `sbsi/flows/factorized.py` implementation. The original
checkpoint bytes and affine arithmetic are preserved; the disabled score
column is supplied as zero without requiring a placeholder input column.
Both this model and `galsbi-multistage` are vendored under `models/` with their
shared classifier/emulator. External root overrides remain available.

See [current inference integration](CURRENT_INFERENCE.md#models-and-compatibility)
for the separate inference-reference and selection-normalization requirements.

No likelihood, estimator, observing conditions, existing cache identity or
historical checkpoint is changed by naming the fiducial. Its GalSBI observing
conditions are pinned in the manifest; do not substitute FS2 defaults.
The broad evaluation remains held and automated follow-ups remain disabled.

## Retained multistage comparison

`galsbi-multistage` retains the weighted photometry continuation, pinned in
[`configs/models_galsbi_multistage.json`](../configs/models_galsbi_multistage.json).
Its flow SHA256 is
`43673e5db42e90d976d45dc7de1e54682adb0fd18bea74c6602e8dd4630c8cbd`;
classifier and emulator are shared with the fiducial. The training sequence
was initial NLL, neighbour-input NLL, weak primary-response continuation,
then a photometry-only weighted cut-probability objective with shape weights
fixed. Stage necessity and a simpler equivalent recipe have not been demonstrated.

On 40 paired-seed development scenes it gives central
m1=+0.55576 ±0.13143%, m2=+0.21011 ±0.16898%, and both-component nine-cut
RMS=0.63457 ±0.11008%. The initial NLL model has lower RMS on that comparison;
this retained model is not a universal winner. The fiducial has only a
four-scene comparison, so the different-cohort totals cannot rank them.

Its extra input is the pinned disk emulator's summed coefficient at reference
radius3.5 pixels and magnitude24.5, not the coefficient at observed photometry.
Direct density/sampling with that input is supported. Current inference
rejects it until an expanded summary adapter is validated. Conditional
photometry errors and independent acceptance remain unresolved.
