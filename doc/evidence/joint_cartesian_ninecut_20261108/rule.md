# Frozen older joint Cartesian: two axes and nine cuts

## Authorized comparison

Owner explicitly requests evaluation of the older joint Cartesian flow on both
shear components and all nine target cuts, against the new paired-seed shape
measurements. Freeze checkpoint d3c671a23e21 (likelihood-only FS2-rate140epoch
GalSBI joint Cartesian), original C8 and physical-moment disk emulator. No
training, new images, shape refits or production-preset changes.

Use the same40 parent scenes, radius>2.8/3.0/3.2pixels crossed with magnitude
<24.9/25.0/25.1, independent per-leg cuts and normalization, and±0.02 shear.
The model uses predicted usability and generated photometry/shape selection.
The measured side uses the completed new per-leg usable shapes and measured
photometry. No truth-property cuts or common-four membership. Report full R,
m_i=100*(Rmeasured_ii/Rmodel_ii−1), and paired model comparisons. Reuse the
validated measured sums with their source receipt and exact case/cut checks.
[Conventions](../../CONVENTIONS.md#6-response-estimators).

## Random draws and pilot gates

Use64 PCG64 antithetic draws,1024-parent blocks, seed
7301+10000019*case+1000003*block: same portable stream as the latest flow.
Generate complete parent blocks; keep common draws across±legs. Record two
independent32-draw antithetic half-banks for integration sensitivity.

The older saved joint-flow predictions used CUDA-generated normal draws,
which vary with GPU type. Do NOT require the new portable bank to equal that
old stochastic estimate. On an A40-8Q pilot, independently replay the legacy
stream at case1000/axis1 with the same old counts/central cut. Require selected
masses within2e-12 relative/2e-7 absolute roundoff and normalized flow/transport
response within1e-6. Record the actual portable-minus-legacy difference.
Pilot then evaluates case1000/axis2, checking zero orthogonal truth shear,
verified image-receipt presence, aligned axis2 classifier cache and all outputs.
No tolerance changes based on a failed result. Both pilot tasks gate workers.

Pool exact emulator tree cells across shear legs (and legacy/portable streams
in the first pilot). First block must reproduce separate evaluations within
2e-14 relative/absolute. Retain original K20/radius10arcsec complete-neighbour
pair rules and Möbius transport. Require finite physical outputs, disk support,
finite transformed draws, source/model identities and half-bank sum identity.
Older central measured sums replay against the old reference on every scene.
Restart only complete outputs with matching hashes and source identities.

## Reporting and resources

Report all40 scenes, both axes, all nine cuts, normal generated-model matrices,
flow/transport accounting, m1/m2 and paired differences from the frozen latest
weighted model. Whole-scene delete-one jackknife and10,000 paired bootstrap
resamples(seed202611061020) preserve correlations across cuts/axes/models.
Report two-axis nine-cut RMS, draw-half sensitivity and radius-cut contrasts.
Use an independent scalar saved-sum audit. These development scenes do not
supply untouched acceptance evidence. Neither model is chosen by a partial
cohort or a single favorable component.

One24GiB pilot GPU gates six persistent one-GPU workers with13 case/axis tasks each;
models/emulator remain loaded. Four CPUs and32GiB per worker. Maximum six GPUs
across cip/inter, within partition limits, and24 CPUs (owner ceiling500).
Broad17788286 remains held. A dependent CPU report waits for pilot and all
workers. Automated follow-up timers remain disabled. Monitor pilot and first
meaningful worker progress, then yield while a long run proceeds.
