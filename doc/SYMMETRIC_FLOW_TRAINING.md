# Sign-symmetric F8 retraining

## Request and model boundary — October 26, 2026

Owner requested retraining F8 with symmetry built into the flow, under both
independent ellipticity sign flips. C8 stays unchanged. This is a new candidate,
not a replacement or relabelling of V3.9. The scalar response emulator is fixed.
F8 conditional symmetry alone does not guarantee symmetry of C8 times F8 or of
the complete inference procedure. No calibration result is implied.

Current study: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/f8_signsym_20261026_v2`.
The failed first attempt and diagnostics remain in `f8_signsym_20261026_v1`.
Branch: `codex/model-symmetry`.

## Completion results — October 27, 2026

Subsequent [80-case stage validation](SYMMETRIC_FLOW_DIAGNOSIS.md#completed-follow-ups--october-27-2026)
retains `base/flow/guard/selected.pt` as a development candidate: its forward
bias is +0.027957% ±0.264014 percentage points, while the final refined
checkpoint described below gives +1.587790%. No production promotion occurred.

Base17516711 and refinement17516727 completed with exit0, in19m25s and39m32s
respectively:58m57s combined on one H200, excluding the earlier failed attempt
and diagnostics. Label recovery17516726 also completed. All dependent stages
finished; the redundant pending two-hour check17516740 was cancelled after
reviewing the final receipts.

Base fitting selected NLL epoch137, then guard phase epoch12. Selected base
full-validation NLL was -4.689806082965132. Harmonized refinement selected
step6 and passed its full-density gate; its full-validation NLL changed from
-4.703562082533013 to -4.702567884474872. These standardized losses are not a
direct comparison with the original asymmetric model, or across domains.
The final joint-probability refinement rejected all six attempted step sizes
and selected step0, retaining the incoming harmonized model. Its receipt has
`eligible_for_joint_validation=false` and `production_accepted=false`.

Final candidate, relative to the study directory:
`joint/joint_probability_refinement_full/selected.pt`.
Independently verified SHA256:
`05dc7a986610851208f10d3a336a7805c5b2a643e3e16f9ed3ca1f8a0d33fd57`.
`trained_candidate.json` pins the unchanged C8 and scalar emulator.

Both selected-checkpoint symmetry audits passed. On1024 real case000 contexts
including explicit zero/near-zero probes, all four sign actions had exactly
zero measured generation, density, log-determinant, log-gap and score errors.
Final inverse error was1.16706e-4 and inverse-logdet error1.71661e-5, both below
the predeclared2e-4 threshold. See `base_symmetry.json` and `final_symmetry.json`.
Frozen orchestration/source/domain-manifest/retained-artifact verification
passed again after completion; the final checkpoint hash matches its receipt.

This establishes the candidate flow's structural symmetry, not removal of c2
in image-simulation inference. C8 remains unchanged and complete-likelihood
symmetry is not established. Next scientific validation would require
candidate-specific normalization/caches and the original/reflected image
cohort; no such run or production promotion was made during this status review.

## Architecture and preprocessing

`sbsi/flow_sign_symmetric.py` implements
`flow_type=sign_symmetric_physical_disk_affine`. Preserve four joint physical
outputs, eight physical conditioning features (plus eight missing indicators),
ten alternating affine coupling layers, 256 hidden units and three hidden layers.

For unchanged layer coordinates u_A and conditioning c, the conditioner sees
even auxiliary coordinates/features, squares of psi(c_j) and psi(u_A,j),
and products psi(c_j) psi(u_A,j), where psi(v)=v/hypot(1,v), for j=1,2.
This smooth odd bijection bounds conditioner features without clipping the
flow coordinates or changing physical densities. All are invariant under independent
simultaneous sign flips of conditioning and latent/output shapes. Every shape
shift is c_j times a learned invariant coefficient; other shifts and all log
scales are invariant. A transformed coordinate never enters its own conditioner.
The affine determinant and inverse remain analytic, with one network call per
layer and no branch averaging. Shape distributions may cross zero.

Conditioning shape means/fills, standardized target shape means and unbounded
disk-coordinate means are exactly zero; their scales are RMS about zero.
These are the statistics of conceptual reflection augmentation, computed
without duplicating rows. Remaining preprocessing is unchanged. Missing flags
are invariant. Loading/saving rejects incompatible shape preprocessing. The
existing radial disk and positive radius/flux maps remain; physical-coordinate
buffers remain float64. Both zero-component boundaries are included in tests.

## Frozen training recipe

### Current recipe: stop after base response training

On October 27 the owner requested removing the later refinement after the
[completed comparison](SYMMETRIC_FLOW_DIAGNOSIS.md#completed-follow-ups--october-27-2026).
New studies created by `scripts/setup_sign_symmetric_flow.py` now run:

1. The original NLL fit and base response-guard training, with unchanged data,
   seeds, epochs and checkpoint-selection rules.
2. The real-context symmetry/inverse audit on `base/flow/guard/selected.pt`.
3. Write `trained_candidate.json` pointing to that exact checkpoint, including
   its hash, audit hash, unchanged component identities and
   `production_accepted=false`.

`scripts/run_sign_symmetric_flow.py` implements this base-only path. The
launcher accepts only `train` (also its default); labels, harmonized refinement,
joint-probability refinement and their dependencies are removed. Setup links
only the base domain and freezes the runner, audit and launcher. The usual
isolated environment, resource detector, input verifier and launch-input hashes
must still be prepared before submission. Candidate publication requires a
passing audit whose checkpoint path and hash match the selected base artifact.

The already trained V2 base-guard checkpoint can be reused; removing the later
stage does not require retraining it. Frozen historical study runners, manifests,
failed-refinement evidence and V3.9 identities are preserved. Historical
refinement diagnostics are separate from this training path. The base response
guard remains: removing that too would instead choose the NLL-only model,
which has different measured behavior.

Validation: seven small orchestration tests pass, covering rejection of removed
stages and candidate publication with matching, stale, failed and wrong-path
audits. Setup against the real frozen reference passes in a disposable directory;
Python and shell syntax checks pass. No new training was launched for this change.

### Historical full-chain recipe

The following describes the completed V1/V2/asinh experiments, not the current
training path.

Reference: `sbsi_caches/v36_f8_20260924_v1` and its hashed
`sbsi_flow_f8_20260924` source. Restore its training modules deliberately into
the study's isolated `source_snapshot`, overlay current compatible flow density
paths plus the new architecture, and install in a task-local environment.
No active library imports from the archive. Setup and launch scripts are kept
in `scripts/setup_sign_symmetric_flow.py` and `jobs/job_sign_symmetric_flow.sh`.

Reuse the original base and harmonized projected domains without edits:
cases 0–199, the original 160/40 case split (seed501), fresh model seed611,
truth magnitude <30, independent valid measurements in each leg, no added
measured ellipticity cut. Base data have 50,246,394 training rows across both
legs. Unmatched response pairs remain excluded and counted by the retained
loader. No ConstGold training or checkpoint selection.

Preserve the full chain and its existing checkpoint-selection rules:

1. Fresh NLL fit, at most 140 epochs, 4 million rows per epoch, batch8192,
   learning rate1e-4; then at most60 epochs of the original guard-response fit.
2. Regenerate original per-leg target banks and require numerical replay
   against the retained targets. This CPU stage can run alongside base fitting.
3. Recompute model-specific control-variate references and run the original
   harmonized refinement, gradient audits and common-descent refinement.
   Do not reuse the asymmetric model's response-reference integrals.
4. Audit the selected base and final checkpoints for generation/density/score
   equivariance and inverse consistency on real contexts plus axis probes.

One GPU at a time, with explicit base/labels/refinement dependencies. Runtime
resource detection records the actual allocation. Frozen input hashes cover
the original/new source, runner, launcher, domain manifests and retained model
artifacts. Projected data remain original read-only references; manifest
provenance is checked, without claiming a new full byte audit of every data file.

## Validation and acceptance

Architecture tests use nonzero learned shifts and scales, all four sign actions,
physical densities/Jacobians, inverse replay, scores, an optimizer update,
Möbius transport, preprocessing, checkpoint round trips and invalid contracts.
Run the complete active suite on Slurm after library changes. A scheduled GPU
preflight exercises a nonidentity model on1024 real contexts before training.

The study is training and validation of a candidate, not automatic production
adoption. Record likelihood and response checks even if unfavorable. Symmetry
must survive checkpoint reload and all refinement stages. Later image-measurement
inference must use candidate-specific normalization and cache identities;
C8 asymmetry remains an explicit limitation. Final results are recorded in
[completion results](#completion-results--october-27-2026).

## Current V2 status

The following records the October26 startup; it is superseded by the
[completion results](#completion-results--october-27-2026).

Full suite17516696: **663 passed,4 skipped**, exit0; the external retained
V3.9 flow/C8/emulator prediction smoke also passed with artifact roots set.
V2 source and frozen inputs verified, all eight architecture tests and GPU
preflight passed. Job17516711 is training on one H200; its first six epochs
completed with finite updates and selected checkpoints. Validation NLL had
an early spike at epoch4, then recovered by epoch6; retain the complete history,
not only its selected minimum. These standardized NLL values are not a direct
comparison to the old model's differently standardized targets.

Original label regeneration17516601 completed its calculations but failed the
old absolute2e-12 replay threshold for `shape_normal`: max difference2.51e-10,
relative1.44e-14. Target responses differ by at most8.00e-15, masses5.56e-17 and
means2.99e-19. Do not report that original label stage as passed.

Amendment: recovery17516726 used `scripts/reuse_frozen_flow_labels.py` to verify
the identical domain, label-source hash and case split, record all regenerated
differences, and copy the exact original F8 target banks plus their receipts.
It passed, and all copied file hashes equal the retained originals. No target
numbers or acceptance thresholds were adjusted. This supersedes the planned
reuse of regenerated V1 banks; failed/cancelled job receipts remain preserved.

Refinement17516727 waits for successful17516711 and17516726, including the
base checkpoint's symmetry audit. It recomputes all model-dependent reference
integrals. `jobs.json` and `checks.json` record the live dependencies and one-shot
two-hour/base-end/refinement-start/final CPU status checks. Training is ongoing;
no final flow or production adoption is claimed.
At the final startup check, NLL epoch32/140 had completed with finite updates;
validation NLL was improving and selected checkpoints continued to save.

## Launch and startup evidence

- Full active suite, Slurm17516571: 662 passed, 4 skipped; exit0 in2m13s.
  An initial targeted test exposed a test-fixture precision mistake: `.float()`
  cast physical buffers. Fixed the fixture to cast only the neural base, as the
  physical coordinate contract requires float64. All seven new tests then pass.
- CPU target-bank generation: job17516601, running through the original cases.
- Base training: job17516602, H200 NVL,8 allocated logical CPUs,96GiB host RAM.
- Refinement: job17516603, after successful17516602 and17516601, one H200.
  Invalid prerequisites cancel refinement; maximum concurrent GPUs is one.
- Source/launcher/domain/retained-artifact verification passed. GPU preflight
  on1024 real contexts (including zero/near-zero probes) had exactly zero error
  for all four reflections in generation, density, log determinant, log gap
  and conditioning scores. Inverse error7.16e-7; inverse-logdet error3.54e-8.
- The frozen trainer's density/backpropagation preflight passed and base fitting
  began loading the full original domain. A preflight model is a randomized
  test artifact and must never be mistaken for a trained checkpoint.

## First-attempt optimizer failure

Job17516602 loaded all50,246,394 training and12,563,706 validation rows, but
stopped in its first NLL epoch on a non-finite gradient norm. No epoch checkpoint
was selected. Its dependent refinement17516603 was automatically cancelled.
All logs, source and protocol remain preserved. Symmetry tests and the real-data
preflight passed; these do not by themselves establish optimizer stability.

A bounded replay17516633 passed64 batches before its intentional diagnostic
stop; the subsequent failed-batch comparison17516641 had no batch to load.
Replay17516645 extends the capture to the full first epoch, with a dependent
comparison17516647 of raw quadratic versus smoothly bounded invariant features.
Replay17516645 captured the failure at zero-based batch337: gradient entries
were finite, but the float32 total norm overflowed (float64 norm4.16e25).
On exactly that checkpoint and batch, comparison17516647 found raw-quadratic
NLL2.78e22 versus3.63e5 with smoothly bounded invariants; the latter had finite
norm2.05e8 and passed the unchanged gradient-clipping check. This is a numerical
diagnostic, not an accuracy comparison or a training result.

V2 starts from scratch with the bounded invariant features above. Data, seed,
optimizer, response targets and all checkpoint-selection rules remain fixed.
Add a regression test of finite training gradients for extreme internal shape
coordinates. Reuse only the independently regenerated and replay-checked V1
response-label banks; recompute every candidate-dependent response reference.
No non-finite check, training cut, seed, response target or selection gate was
relaxed. CPU response-label regeneration17516601 continues independently.
