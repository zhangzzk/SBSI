# GalSBI photometry-conditional shape: completed comparison

## What changed

Both factorized flows recover the original below-cut alignment relation.
The intrinsic-frame version also corrects the previous excess immediately
above radius3. Held-out likelihood improves with slightly fewer parameters.
This establishes a useful GalSBI density-model improvement, but residual
standalone responses remain and joint constant-shear calibration is unproven.

Training17764099_0/1, full40-case evaluation17764100_0/1, paired report17764101
and bounded status17764102 all completed exit0. Cartesian selected epoch139,
frame epoch138; exact checkpoint reload and measured replay pass. See
[predeclared architecture](rule.md#predeclared-architecture),
[completed receipt](completed_receipt.json) and external
`sbsi_caches/alignment_factorized_20261105/comparison.json`.

## Alignment and likelihood

Held-out40 cases, intrinsic circularizedRe0.15–0.25arcsec. Each branch uses
its own radius/magnitude cuts. A=sum(g_measured dot g_intrinsic)/sum(|g_intrinsic|²),
error=A_flow−A_sim. Entries are absolute alignment errors ±whole-case1sigma,
not percentages or m. Models fixed; SEs exclude retraining uncertainty.

| Model | Radius2.7–2.8,mag25 | Radius3.0–3.1,mag25 | Common500k NLL |
|---|---:|---:|---:|
| Original GalSBI NLL | +0.41323 ±0.00472 | +0.09632 ±0.00419 | −3.69413 |
| Joint Cartesian,FS2-rate | +0.25333 ±0.00480 | +0.03438 ±0.00408 | −3.83296 |
| Joint intrinsic frame,FS2-rate | +0.09174 ±0.00485 | +0.11492 ±0.00408 | −3.88700 |
| Conditional shape,Cartesian | −0.00360 ±0.00460 | +0.02604 ±0.00403 | −3.90471 |
| Conditional shape,intrinsic frame | −0.00246 ±0.00480 | −0.00111 ±0.00417 | −3.93722 |

Atmag25.8, conditional-frame alignment errors are−0.00813 ±0.00282 below
and−0.00983 ±0.00267 above the cut. These are small compared with the original
discrepancy, but not all consistent with zero under fixed-model case SEs.
Generated counts remain imperfect: in the two mag25 bins, measured counts
52985/114448 versus conditional-frame53123/112572 (+0.26%/−1.64%). Atmag25.8,
129214/189762 measured versus130713/191237 generated (+1.16%/+0.78%).
Do not describe the complete joint distribution as exact.

![Factorized model comparison](comparison.png)

## Full response remains a separate test

DeltaR=simulation−model, forward matched-leg standalone flow response with
independent per-leg output selection and each branch's own normalization.

| Conditional model/cut | DeltaR11 | DeltaR22 |
|---|---:|---:|
| Cartesian,r>3,mag25 | −0.01537 ±0.00335 | +0.00099 ±0.00385 |
| Intrinsic frame,r>3,mag25 | −0.00935 ±0.00336 | −0.00063 ±0.00385 |
| Cartesian,r>3,mag25.8 | −0.00802 ±0.00301 | +0.00759 ±0.00313 |
| Intrinsic frame,r>3,mag25.8 | +0.00084 ±0.00299 | +0.00909 ±0.00311 |
| Cartesian,r>=3.75,mag25.8 | −0.00455 ±0.00476 | +0.01305 ±0.00352 |
| Intrinsic frame,r>=3.75,mag25.8 | −0.00233 ±0.00475 | +0.00887 ±0.00348 |

For the intrinsic-frame candidate, off-diagonalR12 residuals are−0.00536
atmag25 and−0.00721 atmag25.8; R21 is−0.00144/+0.00093. Main mag25 R11
still differs from zero; mag25.8/tighter-cut R22 also remains positive.
This is not a blanket improvement of every cut/component over every control.
The full profile/likelihood improvement now justifies a joint diagnostic test
of the original bias, not production acceptance or checkpoint selection by m.

## Interpretation

The original four-dimensional couplings interleave shape and photometry updates.
The new exact triangular density gives every shape coupling direct access to
both photometry coordinates. It uses p(radius,flux|input) times
p(shape|radius,flux,input), with generated photometry at prediction time.
No measured-output leakage, empirical correction or physical-support change.
Both new representations share initialization; models have1,377,320 parameters
versus1,390,160 for the original architecture. Total10 couplings are split5+5.

These interventions support a learning/representation explanation for much of
the GalSBI alignment failure, with populations and measured catalogues fixed.
They do not uniquely isolate factorization from layer allocation, prove that
GalSBI differs from FS2 only in this respect, or map alignment directly to m.
Only one training seed has been tested. No retained artifacts changed.

## Next: joint constant-shear comparison

The [constant-shear protocol](../alignment_factorized_constant_20261105/rule.md#purpose-and-fixed-choices)
freezes both NLL-selected candidates and original emulator/C8, evaluates40
existing scenes with paired64-draw streams, and compares predicted usability,
actual usability and actual usability plus measured final selection. Retained
case1000 replay gates the array; full40 results are predeclared independently
of the first-four-case interim report. No additional training or simulation.

## Verification and conventions

Nontrivial triangular inverse/autograd determinant, photometry independence
from shape latent, identical paired initialization, physical support, exact
reload and full density/Jacobian(max8.49e-7) pass. Full evaluation measured
replay and paired response reduction identities pass. Training/evaluation jobs
all exit0. Plot generated on CPU by17768598. These are development cases;
fixed-model case SEs omit training variability and uncertainty from rerunning
the measurement process. See [selection](../../CONVENTIONS.md#3-population-and-selection-order),
[shapes](../../CONVENTIONS.md#4-shapes-and-shear),
[responses](../../CONVENTIONS.md#6-response-estimators).
