# Matched training controls on constant-shear scenes

## Question and predeclared scope

Does factorizing photometry then conditional shape improve joint constant-shear
calibration relative to full four-dimensional affine flows under the same
FS2 learning-rate schedule,140epoch budget, GalSBI data and NLL selection?
The earlier comparison with the retained regularized model mixes multiple
training changes. Evaluate the frozen joint-Cartesian and joint-frame controls
on all40 already-declared scenes, both magnitude cuts and the tighter-radius
control. No checkpoint selection by constant-shear outcome, retraining or images.

## Fixed settings and interpretation

Use the validated factorized constant-shear evaluator, unchanged except checkpoint
loading/output paths. Same original C8 and emulator,64 antithetic CRN draws,
1024 blocks, per-leg normalization, measured-replay checks and three flag modes.
The existing retained replay gate remains applicable. Whole-case paired10000
bootstraps; absolute response error sim−model; m=100(Rsim11/Rmodel11−1).
Compare all arms including completed factorized controls in one paired reducer.
Full-dimensional and factorized models have slightly different parameter counts
(1390160 vs1377320), and their latent transports differ; this tests architecture
and capacity allocation, not a uniquely isolated physical cause. One training
seed and development scenes only. Intrinsic-frame output coordinates are the
second matched pair. True selection on independent joint outputs remains an
intervention that removes output correlation, not corrected predictive closure.

## Resources and stages

80tasks, initially four simultaneous one-GPU workers across cip/inter alongside
two conditional-photometry workers. After both conditional jobs completed,
raise the control throttle to six, within the owner-authorized six-GPU ceiling.
First four scenes give a predeclared diagnostic pilot; all40 remain required
regardless of pilot. Frozen checkpoints and hashes are in manifest.json.
Follow startup; full reducer depends on all80 tasks. A bounded status job about
two hours later records completion/failure; it does not wake the chat.
