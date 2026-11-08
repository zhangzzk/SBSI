# Older joint Cartesian flow: completed comparison

## Completed result and interpretation

The older joint checkpoint is not a suitable replacement for the latest model
on the requested two-component, nine-cut target. Its earlier first-component
central-cut result hid a much larger second-component error. On the same 40
scenes and new paired-seed measurements, radius>3 pixels/magnitude<25 gives:

| Model | m1 (%) | m2 (%) | Both-component nine-cut RMS (%) |
|---|---:|---:|---:|
| Older joint Cartesian | +0.65528 ±0.13286 | +2.73865 ±0.17291 | 1.80345 ±0.10883 |
| Latest weighted factorized | +0.55576 ±0.13143 | +0.21011 ±0.16898 | 0.63457 ±0.11008 |

Errors are whole-scene jackknife standard errors. The paired older-minus-latest
central differences are +0.09952 ±0.01009 and +2.52853 ±0.01312 percentage
points. Paired nine-cut RMS difference is +1.16888 ±0.06961 percentage points
(bootstrap 95% interval [1.01516,1.28512]). Across the radius thresholds, older
m2 rises from roughly 0.84–1.00% at 2.8 pixels to 2.74–2.80% at 3.0 and
3.01–3.03% at 3.2. [All nine cuts and matrices](results.md#both-components-and-all-nine-cuts).

Central measured R22 is 0.902380, versus 0.878326 for the older joint and
0.900488 for the latest model. The older joint's flow contribution has diagonal
(0.724602,0.702836), while disk transport contributes (0.174374,0.175490).
This describes where the predicted diagonal difference occurs; it does not
independently identify the physical source of the calibration error. Architecture
and training differ between the two flows, so this comparison does not prove
that a joint architecture cannot work.

Retain the latest model as the baseline. This completes the requested frozen
checkpoint comparison; no training, measurement rerun or further evaluation is
launched. These reused development scenes are not independent acceptance data.

## Verification and conventions

Pilot 17825569, six packed workers 17825577/17825578 and report 17825579 all
completed with exit code 0:0. All 80 case/axis receipts, source/model identities,
measured sums, half-bank identities and 168 independent scalar point/jackknife
checks passed. Completed report SHA256 is
`bf4406b6c04d457dc14b7d5eb9981c7363372526deb1fa4e86a963c13cf0f802`.

The evaluator checks axis-specific truth shear and classifier metadata; explicit
shear frames and disk transport use the requested axis. The report assembles
columns in shear-axis order and reads the diagonal. Source inspection at
completion found no axis swap. Portable integration changes historical central
m1 by only +0.00384 ±0.00715 percentage points. The central m2 half-bank
difference is −0.00056 ±0.01031 percentage points, much smaller than its bias.

Use independent per-leg generated/measured selection, predicted usability,
64 portable antithetic draws and common case/block seeds. Define
m_i=100*(Rmeasured_ii/Rmodel_ii−1); sum scenes before separately normalizing
plus/minus legs and divide their difference by 0.04. Matrix rows are shape
components and columns are shear components. Catalogue matching reports zero
cross rows without shapes and 228683 shape rows outside the cross catalogue;
unmatched rows are excluded. [Conventions](../../CONVENTIONS.md#6-response-estimators),
[protocol](rule.md#authorized-comparison).

Held broad array 17788286 and dependent 17788351 remain untouched. No automatic
follow-up timer is enabled. Completed job status is recorded in `jobs.json`.
