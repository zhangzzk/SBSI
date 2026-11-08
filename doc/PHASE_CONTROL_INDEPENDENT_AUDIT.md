# Independent phase-control audit — 2026-10-22

## Conclusion for the existing development comparison

The reviewed beta=1 identity, sign, score/control pairing, precision importance
weights and whole-population covariance are consistent. Independent sums over
every saved particle reproduce the observation0 v7−v6v100 corrected-g1 gap as
**0.6652878792 ±0.146401785**, or **4.54426 reported MCSEs**. Explicit F64
reweighting gives **0.6652983152 ±0.146393717**, or **4.54458 MCSEs**.
No algebraic or phase-boundary error found in this audit explains that gap.

It is evidence that the sampling estimates are not stable at their reported
precision. It does not identify which estimate is closer, prove a violation of
the sampler's invariant distribution, or constitute a calibrated Gaussian
significance test. Each group has only eight independent populations, and this
particular component was examined among several development comparisons.
Finite-population ratio bias, rare unvisited mass and inaccurate few-run error
estimates remain possible. These are shear-score expectations at fixed y and
g, not shear estimates or a measurement of catalogue calibration c.

## Identity, sign and phase regularity

For fixed nonphase coordinates and environment, the analytic prior is uniform
in psi and the endpoint density is proportional to K(psi). Thus, conditional on
an absolutely continuous periodic K with integrable derivative,
integral K times dlogK/dpsi equals K(2pi)−K(0)=0. Integrability after averaging
the other variables is still required. No boundary-vanishing assumption is used.

`phase_control.py:36–53` computes
T=−e2*dlogK/de1+e1*dlogK/de2 and rejects beta other than1. This agrees with
the counterclockwise rotation at lines56–66. The phase is the native ellipticity
phase, so no extra factor2 is missing. The uniform prior, native support and
triangular coordinate Jacobian are phase independent; the type mixture does
not change this fact (`galsbi_prior.py:165–215`,
`galsbi_target_coordinates.py:30–70`). A nonuniform initial phase proposal is
irrelevant at beta=1 and would matter at intermediate temperatures.

`galsbi_summary.py:21–42,73–92` shows that the rendering circularity threshold,
semiaxes, response-neighbour selection and response features are independent of
target phase. The remaining F8/C8 features are likewise phase independent
apart from e1/e2 (lines52–70). Holding response b fixed therefore omits no phase
derivative. `summary_gradients.py:14–36,69–84` differentiates both the F8
density and C8 usability through the intrinsic shear map.

The exact retained checkpoint hashes were inspected on CPU, without forwarding
the models. Both use **SiLU**, no log-transformed input features and zero
dropout; F8 is the physical disk affine flow. In particular, neither phase
coordinate enters a logarithm/missing-value boundary.
`selection_model.py:111–137` is affine preprocessing along this finite phase
circle; the existing missing indicators stay constant. The models run in eval
mode (`measurement_model.py:328–336`, `selection_model.py:167–177`).
Affine couplings and their tanh scale bounds are smooth in context
(`measurement_model.py:200–218`). The disk-gap and origin-series branches in
`disk_precision.py:39–89` depend on the fixed observation, shear and b, not
on target phase. They therefore create no phase crossing for this control.

This establishes the applicability of the intended continuous model identity,
not an exact identity for floating-point rounding. The saved v2 derivative,
wrap and replay checks remain local numerical evidence; no new global error
bound or derivative threshold was introduced.

## Score pairing, frozen coefficient and uncertainty

`check_particle_phase_control.py:162–192` evaluates context and shear
derivatives at the same summary, observation and shear. Saved F64 shear scores
are replayed against the original artifact. Arrays retain identical particle
ordering for S, T and Delta. `FrozenPhaseControl.apply` subtracts raw uncentred
T (`phase_control.py:112–129`); the independent recomputation confirms the
stored corrected arrays equal S−cT exactly where present.

The actual coefficient artifact uses only v1 seeds2026102130–2026102133.
Its observation0 g1 coefficient is3.035284934020041. Production v5, v6v100
and v7 seeds are disjoint from those pilots and from one another. The same
v1 pilots also supplied geometry/proposals; comparisons are conditional on
these frozen artifacts. Equal-run covariance fitting without Z weights can be
suboptimal, but an arbitrary fixed finite coefficient preserves the continuous
endpoint expectation. It is not an empirical offset. No coefficient was refit
in this audit.

For run r and particle j, the independent calculation uses raw weights
Zhat_r/N_r and F64 weights Zhat_r*exp(Delta_rj)/N_r. The latter imply exactly
Z64hat_r=Z32hat_r*mean(exp(Delta_rj)); multiplying a normalized F64 moment by
the old Z32 alone would be wrong. The implementation includes the adjustment
at `phase_control.py:204–215`.

For each measure, define B_r=sum_j weights_rj and A_r=sum_j weights_rj*f_rj.
The reported mean is sum A_r/sum B_r. With R complete independent populations,
influence_r=(A_r−B_r*mean)/mean(B), and covariance=sum influence_r*influence_r^T
/[R(R−1)]. This reproduces `particle_diagnostics.py:23–32` including the
S/T cross covariance. Particles and resampled ancestors are not counted as
independent replicates. A small T residual cannot test phase-independent mode
coverage; whole modes can satisfy the conditional phase identity separately.

## Findings fixed without rewriting evidence

1. **Misleading raw-F32 expectation metadata.** The former summary attached
   `control_known_expectation=0` to both numerical measures, despite the explicit
   lack of an exact rounded-F32 identity. New raw-F32 rows use null plus
   `unestablished_for_rounded_float32_target`; F64 rows retain zero with
   `intended_continuous_float64_periodic_identity`. Both retain zero as an
   explicitly labelled comparison reference and unchanged descriptive T/MCSE.
   New population reports use schema `sbsi-independent-phase-diagnostics-v2`.
   Original reports and queued jobs using immutable old snapshots retain their
   original metadata; this paragraph is their erratum, not a replacement result.

2. **Deterministic thinning was not a valid generic estimator contract.**
   Original `validate_saved_run` accepted stride>1, while fitting/reporting
   used the complete run's Z with the thinned mean. Stratified-resampled output
   slots need not be exchangeable. For weights(.75,.25) and N=2, the first
   stratum always selects the first atom; retaining slot0 alone gives expected
   indicator0 for the second atom instead of.25. New diagnostics and estimator
   loaders reject any stride other than1 and verify the retained array count.
   All24 evaluated populations here used stride1 and all original particles:
   N256 in v5/v7, N1024 in v6v100. Therefore this bug did not affect the result.

No sampler, score, normalizer, precision threshold or existing acceptance gate
was changed. Failed coarse-step evidence and successful fine-step evidence
remain separate and unchanged.

## Remaining provenance limitation and fail-closed follow-up

`check_particle_phase_control.py:27–28,124–137` identifies only coordinates,
summary, gradient code and the model manifest. `report_particle_phase_control.py:
35–44` compares that reduced identity. Changes in the analytic prior,
environment law or inference target can therefore escape this generic check
while changing the nonphase estimand. Correct phase derivatives alone do not
establish that two posterior targets agree.

For these specific24 evaluation populations, the audit independently verified
identical recorded target-module/native hashes and identical analytic-prior and
environment-source files in their preserved source snapshots. Checkpoint hashes
also match V3.9. The old protocols did not themselves declare the two prior
hashes; the new receipt records that distinction. This is bounded evidence for
the current comparison, not a generic compatibility rule or a hash of every
external prior configuration/runtime dependency.

A future diagnostic should record a separate endpoint identity covering the
declared endpoint, analytic prior/configuration, environment law, target module
and existing model/native identities. Group aggregation should reject absent
or differing endpoint provenance. Distinct bridges can use the explicit
endpoint-and-source rule already adopted by the particle report, with a
separate reviewed compatibility receipt for specific historical snapshots.
Missing historical identities must not be silently inferred as equal. This
audit does not clear v8/v9 or other cross-version comparisons.

## Independent evidence

CPU Slurm17421220, cluster, COMPLETED0:0 in26s, independently reconstructed
all24 evaluation populations (49,152 saved states) and inspected checkpoint
metadata. Maximum mean discrepancy from the preserved reports was1.34e−14;
maximum covariance discrepancy was1.14e−13. All arrays, diagnostics, original
source receipts and coefficients matched their hashes. The same job passed
40 focused phase, gradient and population-diagnostic tests in10.59s, including
the new metadata and thinning regressions.

Reproducer: `doc/evidence/audit_phase_control_20261022.py`. Receipt:
`doc/evidence/phase_control_independent_audit_20261022.json`. No GPU, model
forward, new posterior sample, scientific fit or heldout outcome was used.

Final CPU Slurm17421249 completed0:0 in78s: **778 passed,4 skipped** in68.07s,
with one existing NumPy deprecation warning in a scene-exchange test. Source
hashes were verified before and after the full suite. Validation receipt:
`doc/evidence/phase_control_audit_validation_20261022.json`.

## 2026-10-22 endpoint-kernel qualification

The later [endpoint audit](PARTICLE_ENDPOINT_COMPATIBILITY_AUDIT.md#why-the-old-native-exchange-violates-continuous-support)
confirms matching declared endpoint factors but finds a reachable continuous
support error in the old default native exchange. This does not alter the
phase arithmetic or invalidate the reproduced discrepancy. It does mean that
sampling error cannot yet be attributed exclusively to non-equilibration:
the old kernel itself could violate the intended target. No old trajectory
was replayed and no v8/v9 outcome was inspected in that audit.
