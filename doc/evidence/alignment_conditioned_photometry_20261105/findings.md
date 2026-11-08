# Conditional shapes with measured photometry

## Result and interpretation

Replacing generated photometry by exact measured per-leg radius/flux does not
eliminate the standalone shape-response residual. Both40-case evaluations
17771721_0/1 completed in4m04s; paired reducer17771722 completed in51s, all exit0.
[Protocol](rule.md#predeclared-scope), [completed receipt](completed_receipt.json).
Same frozen factorized checkpoints, data, cuts and antithetic shape latent streams;
no new training, simulation or empirical correction.

This is a standalone flow validation-domain test. It evaluates the learned
p(shape|measured radius,measured flux,context), with exact measured selection.
It is different from attaching a measured membership flag to an unconditional
shape draw, and it is not the joint constant-shear estimator.

## Absolute response errors

DeltaR=simulation−model;1sigma whole-case bootstrap errors conditional on models.
All radius cuts are pixels at0.2arcsec/pixel. Original denotes ordinary generated
photometry and selection from the same factorized checkpoint.

| Flow | Selection | Original DeltaR11 | Measured-photometry DeltaR11 | Measured-photometry DeltaR22 |
|---|---|---:|---:|---:|
| Cartesian | r>3,mag<25 | −0.01537±0.00335 | −0.01361±0.00347 | +0.00321±0.00386 |
| Cartesian | r>3,mag<25.8 | −0.00802±0.00301 | −0.00965±0.00299 | +0.00596±0.00315 |
| Cartesian | r>=3.75,mag<25.8 | −0.00455±0.00476 | −0.00695±0.00464 | +0.00925±0.00361 |
| Frame | r>3,mag<25 | −0.00935±0.00336 | −0.00582±0.00342 | +0.00345±0.00389 |
| Frame | r>3,mag<25.8 | +0.00084±0.00299 | −0.00274±0.00292 | +0.00585±0.00318 |
| Frame | r>=3.75,mag<25.8 | −0.00233±0.00475 | −0.00282±0.00462 | +0.00625±0.00359 |

At r>3/mag25, the paired change in DeltaR11 is+0.001757±0.000432 Cartesian,
+0.003522±0.000427 frame. These are changes in the diagnostic, not an additive
physical-error decomposition. At mag25.8 the shifts reverse sign. Conditional
photometry is therefore neither a uniform correction nor the sole remaining
source of standalone error. The different diagonal residuals also warrant
checking orientation-dependent response; scalar alignment alone cannot test
all four response-matrix entries.

## Diagonal response imbalance

Reducer17771808 adds paired-bootstrap matrix combinations without new sampling.
At r>3/mag25 after measured-photometry conditioning,
(DeltaR11−DeltaR22)/2 is−0.008410±0.002400 Cartesian and−0.004637±0.002405 frame.
The intervention changes this difference by only−0.000233±0.000268 and
−0.000281±0.000262 respectively, while changing the mean diagonal by+0.001990
and+0.003803. Thus the photometry intervention mainly shifts the common diagonal
component; it does not remove the difference between the two shear components.
This is an error relative to each simulation response, not an assumption that
the physical R11 andR22 must be equal. The evidence supports checking angular
shape-response learning, but does not yet establish its architectural cause.
The current constant-shear scenes test g1 only, so their near-zero m cannot
certify both diagonal responses.

## Intrinsic size and alignment

For the conditional frame at r>3/mag25, intrinsic circularizedRe0.25–0.4arcsec
contributes−0.005422±0.002048 to total DeltaR11=−0.005823±0.003419. Re0.15–0.25
contributes+0.000584±0.002563. The previously problematic smallest-bin alignment
can therefore be learned while a response residual remains in the next size bin.
For Cartesian, Re0.15–0.25 and0.25–0.4 contribute−0.005052 and−0.005531.
These contributions sum exactly with the other bins; they are not independently
normalized subgroup responses.

At Re0.15–0.25/mag25, measured-radius3.0–3.1 alignment error is+0.02042±0.00386
Cartesian,−0.00345±0.00377 frame. Supplying measured photometry preserves the
advantage of the frame in this relation. Nevertheless, the [constant-shear
comparison](../alignment_factorized_constant_20261105/findings.md#joint-multiplicative-bias)
favors Cartesian at radius>3. This discrepancy makes residual cancellation a
live concern; no calibrated replacement is accepted.

## Verification and limits

Conditioning on a joint sample's own radius/flux with the same shape latent
reconstructs that sample exactly for both arms. Photometry latents do not affect
conditional shapes; identical legs give exact CRNs; observed radius/flux and
per-draw selection are exact. Everycase passes physical support, measured-sum
replay, identical selection masses and floating-point-tolerant cross sums.
Two16-draw banks have R11 differences below0.00062 on the three listed conditional
cuts. Bootstrap errors do not include training-seed variation or repeated image
measurement uncertainty. All40 cases are existing development validation data.

Next: [matched architecture constant-shear controls](../alignment_matched_constant_20261105/rule.md#question-and-predeclared-scope)
are running. They distinguish architecture/allocation changes from other training
changes before choosing the next improvement. Conditional standalone findings
cannot by themselves assign the tighter-cut joint residual to the emulator.
