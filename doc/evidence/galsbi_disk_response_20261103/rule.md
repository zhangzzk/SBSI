# Disk-response emulator retrained on the GalSBI re-simulation (2026-11-03)

Owner request 2026-11-03: "go ahead, retrain the emulator on the new sims".
Fixed before any training output exists.

## What is retrained
The output-conditioned disk-response moment emulator, same format as
`disk_response_moment_20260918_v1` (V3.9-like / GalSBI-F8C8-dev emulator), so SBSI
loads it unchanged (`sbsi.output_conditioned_response.DiskResponseMoment`).

## Data
- Training family only: `$DATA_DIR/sbsi/galsbi_resim/training/sims`, cases 0-199,
  legs 0.0 / 0.2 (`response_catalogue_train.feather` + per-case g=0 CrossMatch/Shapes).
- Never the constant-shear family (cases 1000+), not for fitting, stopping or choosing.
- Fit cases 40-199; development cases 0-39 (report only; nothing is selected on them).

## Population (unchanged)
Structural primary target; true r<26; usable g0 shape and positive radius; both
response-label shapes valid; no measured science cut. Every drop is counted per case in
the manifests. "r" is the GalSBI render magnitude (LSST-i).

## Only change from FS2
Feature-scaling conditions = the GalSBI sims' LSST-i values from SBSI
`configs/galsbi_resim_training.yaml`: pixel_rms 0.6159, psf_fwhm 0.71, moffat_beta 2.458,
pixel_size 0.2, zero_mag 30 (FS2: 0.312 / 0.73 / 2.224). SBSI callers using this model must
pass the same conditions.

## Recipe (unchanged, frozen)
Trial9 xgboost parameters, seed 20260816, 218 trees, moment objective
E[(s*b^2/2-z*b)/label_std^2], gain u^T[(1+|e0|^2)I-2e0e0^T]u, CPU hist. No tuning, no
early stopping, no checkpoint choice.

## Code
BlendEMU branch `galsbi-resim`: `scripts/prepare_output_conditioned_response.py` (stage
pairs), `scripts/prepare_disk_response_moment.py` (gains), `scripts/train_disk_response_moment.py`
(train), launcher `jobs/job_disk_response_galsbi.example.sh`. Copied from the FS2 run's untracked
scripts; only the scaling conditions became arguments (FS2 defaults) and the frozen PARAMS were
inlined.

## Checks reported
Per-case drop counts; fit label mean/std; development (cases 0-39) pair sums of label vs
prediction, uncut and with mag<25.8 & radius cuts, compared with the FS2 model's development
numbers for scale only. No pass/fail is declared from development; evaluation on the
constant-shear family is a separate, owner-requested step.
