# Current GalSBI inference and model consolidation

## Location and provenance

As of2026-11-08 the latest inference code is consolidated into `dev` from
`codex/model-symmetry` (986a821) and its shared sampler ancestry with
`sampler-recipe` (4b211ca). The older `worktree-galsbi-tempering` contains
additional experimental candidates; those uncommitted candidates are not
silently promoted. Source worktrees remain intact.

The latest **executed** sampler was a frozen source copy under
`galsbi_constgold1m_signsym_20261028/recovery_v1/source_snapshot`, not simply
the branch's earlier local-sampler script. That production copy evaluates the
full implemented likelihood for every proposal: analytically invariant nuisance
moves had crossed float32 boundaries and invalidated cached likelihood reuse.
The integrated `scripts/run_galsbi_inference.py` retains this numerical body,
proposal order, random streams, replay tolerances and adaptation settings;
its changes provide named-model selection, reference-identity checks and source
provenance. The [source receipt](evidence/dev_consolidation_20261108/production_source.json)
pins the original bytes. Historical scripts remain for replay.

## Entry points

| Component | Entry point |
|---|---|
| Current continuous-scene sampler | `scripts/run_galsbi_inference.py` |
| Population/reference preparation | `scripts/prepare_galsbi_population.py`, `scripts/prepare_constgold_inference_input.py` |
| Selection normalization and bright follow-up | `scripts/run_galsbi_bright.py`, `scripts/report_galsbi_bright.py` |
| Completed production orchestration/recovery | `scripts/extend_sign_symmetric_inference.py`, `scripts/recover_sign_symmetric_inference.py` |
| Independent minus-leg preparation and paired reduction | `scripts/run_sign_symmetric_minus.py`, `scripts/paired_sign_symmetric_estimate.py` |
| Core likelihood/prior/proposal implementation | `sbsi/galsbi_summary.py`, `sbsi/galsbi_scene_prior.py`, `sbsi/galsbi_target_coordinates.py`, `sbsi/mcmc_proposals.py` |

The algorithm samples continuous target coordinates and neighbour scenes,
sums over target type, uses frozen post-warmup proposals and optional
likelihood tempering, and reads nearby shear likelihood ratios with co-moving
coordinates. Independent forward population selection normalization remains
necessary. The old finite-atom `run_inference.py` is a separate retained path.

Use `python scripts/run_galsbi_inference.py --help` (or
`python -m scripts.run_galsbi_inference --help` from the checkout) to inspect
arguments. The scientific runtime also needs the existing GalSBI/PyCosmo,
ArviZ and XGBoost environment; these optional research dependencies are not
bundled model artifacts.
`--model` defaults to `like-1.0-galsbi`; reference metadata must bind exactly
the selected flow/classifier/emulator/metadata hashes. Reusing old normalization
or model-dependent preparations by relabelling them is rejected. Production
setup/recovery scripts preserve their original study-specific interfaces;
this consolidation does not turn them into validated generic launchers for
new model sets. No inference, normalization or new simulation was launched.

## Models and compatibility

| Model | Role | Central paired-seed m1 / m2 | Evidence |
|---|---|---|---|
| `like-1.0-galsbi` | Current fiducial; no informative extra neighbour score | +0.38554 ±0.39755% / +0.04296 ±0.66616% | Four development scenes; RMS0.53431% |
| `galsbi-multistage` | Retained weighted multistage comparison | +0.55576 ±0.13143% / +0.21011 ±0.16898% | Forty development scenes; RMS0.63457% |

These are forward response calibration results, **not inference results**.
Different scene sets must not be compared as a matched ranking. The original
NLL model sometimes has lower total calibration RMS; “multistage comparison”
does not claim a universally best model.

Both checkpoint files, the shared GalSBI C8 classifier, and disk emulator and
metadata are included under `models/`, at the relative paths recorded in the
manifests. Original hashes are preserved. `SBSI_CACHE_DIR` and
`BLENDEMU_RUNS_DIR` can still override their roots. Use
`load_measurement_model(get_model(name).flow_checkpoints[0])` for the permanent
loader. It preserves the physical disk density, positive radius/flux, target
standardization, and conditional factor order.

The fiducial's extra control column is exactly zero. Its eight informative
raw inputs fit the existing nine-summary likelihood contract. End-to-end
inference/normalization still requires fresh, matching preparation and numerical
validation; naming and loader replay do not establish inference calibration.

The multistage model additionally needs `neighbour_reference_coefficient`,
the pinned emulator's summed coefficient at radius3.5pixels/magnitude24.5.
This is not the response evaluated at each observation's photometry. The
existing nine-summary pipeline lacks that second coefficient; its current
entry point explicitly rejects this model until an expanded summary adapter
is validated. Direct density/sampling with explicit correct inputs is supported.

## Scientific limits and preserved work

The latest historical paired **inference** result used the sign-symmetric FS2
flow, not either newly retained GalSBI model. It gave m1=-5.5391 ±0.5200% and
retained substantial chain-convergence flags. See the preserved
[paired inference report](evidence/dev_consolidation_20261108/PAIRED_MEASUREMENT_INFERENCE.md#purpose-and-fixed-endpoints).
Thus merging the latest code makes it available; it does not establish an
accepted calibration pipeline or transfer old results to the new models.

A recovery Git snapshot before integration is retained at
`refs/codex-snapshots/pre-galsbi-integration-20261108`; its tree includes the
previously uncommitted dev code/evidence. The integration inventory records
incoming paths and versions. Source branches/worktrees and their uncommitted
work are preserved. Held broad evaluations and disabled automated follow-ups
remain unchanged. Branch/worktree removal is a later cleanup step.
