# F8/C8 retraining on the GalSBI re-simulation (fixed 2026-11-02, before any training)

Owner request 2026-11-02: "go ahead, restore the F8/C8 builders and retrain".

## What changes

Only the training data. The new data are the GalSBI re-simulation training family
(`$DATA_DIR/sbsi/galsbi_resim/training/sims`, cases 0-199, shear legs 0.0 and 0.05; LSST-i, centred 65 px
stamps, per-case noise shared by the legs). The domain is built by SBSI `scripts/prepare_galsbi_f8c8_domain.py`
(branch `sampler-recipe`) into `harmonized/domain`; `base/domain` is a symlink to it (one measurement, so the
FS2 base/harmonized distinction does not exist here).

## What stays fixed

- F8: frozen checkout `/project/ls-gruen/users/zekang.zhang/sbsi_flow_f8_20260924` (hashes in experiment.json,
  checked at every stage), frozen venv, stages base -> labels -> refine exactly as the 2026-10-05 control
  (`../v39_f8_control_20261005/run_stage.py`), training seed 611, split seed 501 / validation 0.2 over cases
  0-199 (same case split as FS2), eight inputs, four targets, truth bound r < 30 (removes nothing).
- C8: frozen checkout `/project/ls-gruen/users/zekang.zhang/sbsi_classifier_f8_inputs_20260925` (227 pinned
  hashes in c8/protocol.json), unchanged trainer `scripts.train_smooth_classifier --feature-arm C8`, train cases
  40-99, tune cases 40-119 minus train (100-119), selection by minimum mean-case tune BCE. Seeds 20260913
  (primary, as V3.9) and 20260914 (replicate).
- No constant-shear data (cases 1000+) and no FS2 ConstGold data in training, early stopping or selection.
- No empirical offsets. Rows the renderer omitted are dropped and listed in the domain manifest.

## Resources

At most four GPUs at once (owner, 2026-10-12 follow-ups). F8 base and refine: one GPU each, in sequence.
C8: one GPU per seed.

## Not part of this rule

Evaluation on the constant-shear family, a new model preset, and emulator retraining are separate steps.
