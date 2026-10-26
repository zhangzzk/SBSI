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
C8 asymmetry remains an explicit limitation. Final training results are pending.

## Current V2 status

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
