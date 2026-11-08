# infer-1.0-galsbi: core inference pipeline

## Name and reference mock

The owner named the core GalSBI-prior inference pipeline **infer-1.0-galsbi**
on2026-11-08. The identity is recorded in
[`configs/infer_1_0_galsbi.json`](../configs/infer_1_0_galsbi.json).
This names the continuous-scene prior, posterior integration, selection
normalization and score/information estimator. Model names remain separate:
`like-1.0-galsbi` is the current fiducial model set.

The historical model-generated mock is the reference evidence:1,990,072
selected observations, original/rotated pairing, injected shear(0.02,0),
and recovered shear(0.01990248 ±0.00014818,0.00033189 ±0.00016430).
Its fractional g1 error is-0.48759 ±0.74091%; the g2 offset is2.02sigma.
The first component agrees with truth within uncertainty, while the declared
joint2sigma gate narrowly fails. This is not independent calibration acceptance.
See the [audited mock comparison](../archive/research-2026-11-08/doc/MOCK_MEASUREMENT_REVIEW.md#outcomes-and-pairing).

That endpoint used the retained V3.9 model and a mixture of magnitude-dependent
chain settings and shear-score evaluations. It is not an end-to-end validation
of the later uniform, full-proposal-likelihood sampler below or of the new
fiducial model. Exact model hashes, sampler settings, source and seeds remain
part of each run's identity. The naming manifest is descriptive, not a runnable
replacement for the old finite-catalogue inference configuration.

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
pins the original bytes. Historical study scripts are preserved in the
[archive](REPOSITORY.md#restoring-an-experiment) for deliberate restoration.

## Entry points

| Component | Entry point |
|---|---|
| Current continuous-scene sampler | `scripts/run_galsbi_inference.py` |
| Population/reference preparation | `scripts/prepare_galsbi_population.py`, `scripts/prepare_constgold_inference_input.py` |
| Selection normalization and bright follow-up | `scripts/run_galsbi_bright.py`, `scripts/report_galsbi_bright.py` |
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
or model-dependent preparations by relabelling them is rejected. The preparation and bright-follow-up helpers retain their
historical reference assumptions. Archived production setup/recovery scripts
are study-specific; none is a validated generic launcher for the new models. No inference, normalization or new simulation was launched.

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
[paired inference report](../archive/research-2026-11-08/doc/PAIRED_MEASUREMENT_INFERENCE.md#purpose-and-fixed-endpoints).
Thus merging the latest code makes it available; it does not establish an
accepted calibration pipeline or transfer old results to the new models.

A recovery Git snapshot before integration is retained at
`refs/codex-snapshots/pre-galsbi-integration-20261108`; its tree includes the
previously uncommitted dev code/evidence. The integration inventory records
incoming paths and versions. Source branches/worktrees and their uncommitted
work are preserved. Held broad evaluations and disabled automated follow-ups
remain unchanged. Branch/worktree removal is a later cleanup step.
