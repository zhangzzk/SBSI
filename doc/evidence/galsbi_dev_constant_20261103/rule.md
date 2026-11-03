# GalSBI-dev on the GalSBI constant-shear family (2026-11-03)

Owner request 2026-11-03: "go ahead, evaluate GalSBI-dev on the constant-shear sims".
Fixed before any model output exists. Code: SBSI `scripts/evaluate_galsbi_constant.py`,
`jobs/job_galsbi_constant_eval.sh` (branch sampler-recipe).

## Model
Preset `GalSBI-dev` (`configs/models_galsbi_dev.json`): flow abac069f…, C8 ad5c05d2…, disk emulator
cd32781d…; hashes checked before use. Observing conditions from the preset (LSST-i: pixel_rms 0.6159,
psf_fwhm 0.71, moffat_beta 2.458, pixel 0.2″, zero point 30).

## Data
GalSBI constant family `$DATA_DIR/sbsi/galsbi_resim/constant_g002`, all 140 cases 1000–1139, legs
g = (+0.02, 0) and (−0.02, 0), whole scene sheared, noise shared by the legs. Test-only: nothing here
trains, tunes or selects any model.

## Estimator (same as V3.9-like, doc/V39_MODEL.md#development-evidence)
- Parent = every rendered galaxy of a case (gals_info rows); no truth cut. The renderer's edge drops
  (22 in this family) are absent and reported. FS2 used a target-role half; these scenes have no
  roles, so all galaxies are parents.
- Inputs: intrinsic e1, e2 sheared ±0.02 (Möbius), rendered Sérsic n, LSST-i magnitude, Re·√q, close
  crowding over all rendered objects (same as the training builder).
- 64 antithetic flow draws, seed 7301 + 10000019·case + 1000003·block, blocks of 1024; Möbius disk
  transport by the summed emulator pair response; pairs = up to k = 20 nearest rendered objects within
  10″ (self excluded), no cuts (BlendEMU rule; checked by rebuilding the saved FS2 case-40 table).
- Cuts per leg, same on draws and measurements: main = FLUX_RADIUS > 3 px (0.6″) and MAG_AUTO < 25.8;
  also uncut and ≥ 0.75″. Measured legs: CrossMatch ⋈ Shapes `_all`, usable = finite, |e| < 1, positive
  radius and flux.
- Weights: C8 probability (headline) and, as a diagnostic, the simulation's own usable flag.
- R = (mean e1₊ − mean e1₋)/0.04 with each leg's own selected normalization; m = 100 (R_sim/R_model − 1).
  Also R21 (sim − model), flow-only (no emulator) and true-magnitude bins (< 24, 24–25.5, 25.5–27.5,
  ≥ 27.5) as diagnostics.
- Errors: case bootstrap, 10,000 multinomial(140) resamples, seed 20260925.

## Reading
Headline: C8, main cut. m within 2σ of 0 = consistent with no bias at this precision; |m| > 2σ = the
models do not reproduce the measured response. No result is used to change any model.
