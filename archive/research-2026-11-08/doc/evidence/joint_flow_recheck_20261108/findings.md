# Older joint-flow baseline on the new measured response

## Superseded by the complete comparison

The subsequent [two-axis, nine-cut evaluation](../joint_cartesian_ninecut_20261108/findings.md#completed-result-and-interpretation)
finds central m2=+2.73865 ±0.17291% for the older joint Cartesian checkpoint.
Its first-component result below remains valid, but does not establish a good
two-component baseline. Retain the latest model for the requested target.

## Central-cut recheck

User asks whether a simpler older joint flow is worth retaining. Reuse saved
predicted-usability/generated-selection model sums on the same40 scenes;
replace only the measured reference with completed paired-seed measurements.
Radius>3pixels, magnitude<25, first shear component only.
m=100*(Rmeasured11/Rmodel11−1), whole-scene jackknife errors in percentage points.

| Flow | New m1 (%) |
|---|---:|
| Older joint Cartesian, likelihood only | +0.65144 ±0.13284 |
| Older joint intrinsic-frame, likelihood only | −0.95383 ±0.12902 |
| Latest photometry-first weighted flow | +0.55576 ±0.13143 |

Joint Cartesian minus latest is+0.09568±0.01004pp. It is a credible simpler
baseline at this cut. No updated R22/m2 or complete target-nine-cut result is
available for these older joint candidates from this recheck. The historical
Cartesian m1 wasapproximately+0.9040% using the original unpaired measurement; do not mix
that reference with the new results.

## Architecture and training

The joint Cartesian flow uses ten four-dimensional affine couplings, generating
shape1,shape2,radius,flux together with their correlations. Physical unit-disk
shape and positive-photometry maps remain. It does not require the explicit
photometry/conditional-shape split or added emulator-derived neighbour score.
The frozen candidate was trained by likelihood alone on160 GalSBI scenes with
40 validation scenes,140epochs×4million rows, batch8192. After epoch20 it follows
the realized FS2 learning-rate schedule. Validation likelihood selects the
checkpoint; constant-shear bias does not select the epoch.

Checkpoint `alignment_trajectory_20261105/galsbi_fs2rate/selected.pt`, SHA
`d3c671a23e2104b345f8215afe2fa31c24b0aa7ae2f367db21264c38ffa044cd`.
[Training contract](../alignment_trajectory_20261105/rule.md#stage-2-causal-learning-rate-test-on-galsbi).
The retained older FS2 V3.9 F8 joint flow is a separate population/model; its
historical near-zero calibration must not be transferred to GalSBI.

## Verification and scope

`reduce.py` checks80 saved-output receipts/source/checkpoint identities for the
two older candidates, both original measured sums per scene, old pooled model
response/bias and all80 historical per-case biases. It reproduces the latest
m1 and records current cache hashes (the older receipts lack data hashes).
Only small saved arrays are read; no prediction, training or catalogue scan.
[Response convention](../../CONVENTIONS.md#6-response-estimators).
No independent acceptance, production change or new jobs. Next useful comparison
is both shear components and all nine cuts for the joint Cartesian candidate.
