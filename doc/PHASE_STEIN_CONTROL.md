# Periodic phase control variate

## Hypothesis and scope

Arbor n15 tests whether an independent-pilot frozen periodic-phase control
can reduce shear-score variance without changing the sampling distribution.
The implementation is an estimator/diagnostic prototype. It does not modify
the particle driver, sampling law, normalizers, raw-score gates, or production
reports. Four original v1 runs are the only authorized coefficient pilots;
v5, v6v100, and v7 are evaluation groups and cannot enter the fit.

Gradient-based zero-mean controls are established variance-reduction methods;
see [Mira, Solgi and Imparato](https://arxiv.org/abs/1012.2983) and
[Oates, Girolami and Chopin](https://arxiv.org/abs/1410.2392). The particular
one-dimensional periodic identity below is derived directly for this model;
the papers' broader guarantees are not asserted for the current sampler.

## Periodic identity and code applicability

Write the target as `pi(u,psi|y) proportional to p(u) qEnv(E) K(y|u,psi,E)`
at fixed shear, where u contains every nonphase target coordinate and E the
complete environment. For almost every fixed (u,E), assume K is absolutely
continuous in psi, integrable and periodic on one full circle, with matching
endpoint values. Then

`E[T | u,E,y] = integral_0^(2pi) dK/dpsi dpsi / integral_0^(2pi) K dpsi = 0`,

where `T = d log K/dpsi`. Integration over (u,E) preserves zero expectation
when the derivative is integrable. No normalization constant is required.
The interval is periodic: this does not assume K vanishes at either boundary.
For a nonuniform phase prior, its log derivative must also be included.

The actual retained mapping satisfies the phase conditions:

- `galsbi_prior.py::log_intensity` contributes uniform `-log(2*pi)` phase
  density. Native support cuts do not depend on phase within the circle.
- `galsbi_target_coordinates.py::decode` has a phase-independent triangular
  Jacobian. The type mixture preserves phase independence.
- `galsbi_summary.py::native_shapes` gives intrinsic
  `e1=rho*cos(psi), e2=rho*sin(psi)`. At the rendering circularity threshold,
  rho=0 and the phase derivative is zero.
- `galsbi_scene_prior.py::v39_crowding` depends on flux ratios and offsets.
  `galsbi_summary.py::response_pairs` depends on semimajor radii, magnitudes,
  Sersic indices and distance. None depends on target phase. Thus response b
  is phase-independent even at nonzero shear in this retained model.
- `SummaryGradients(..., wrt='context')` differentiates unsheared intrinsic
  e1/e2 through the context shear map and includes both F8 density and C8
  usability. Therefore the full phase derivative is exactly the chain rule
  `T = -e2 * gradient_e1 + e1 * gradient_e2` for the intended continuous
  model. No response derivative is missing for this direction.

Apply this T only at beta=1. At an intermediate bridge with informed q0,
the phase score is `(1-beta)*dlog(q0)/dpsi + beta*T`; T alone need not have
zero expectation. The generic helper rejects intermediate beta.

## Frozen coefficient and estimator

The declared fit uses the four independent v1 populations, equal total
weight per run and equal weights per particle within a run. For each
observation and shear component, set
`c = pooled_empirical_covariance(S,T) / pooled_empirical_variance(T)`;
set c=0 when the empirical variance is exactly zero. Centre pilot values
when computing covariance, but subtract **raw uncentred T** in production:
`S_corrected = S_raw - c*T`. No intercept or empirical production-mean
correction is applied. Record pilot seeds, source receipts, exact fit-array
hashes, source hash, coordinate/derivative identity, and coefficients. Reject
every pilot seed during application, and reject changed target identities.

For any fixed finite c independent of production, the continuous posterior
expectation of the corrected score equals the raw score expectation. The
pooled regression is a proposal for a useful coefficient, not an uncertainty
estimate or an optimality claim for correlated SMC particles. A run-level
ratio-optimal coefficient could differ. Keep raw S, T and corrected S for
every evaluated particle; no scientific coefficients have yet been fit.

An exact zero residual is possible for the analytic translation model
`K=exp(kappa*cos(psi-mu(g)))`, since its shear score is a fixed multiple of T.
That cancellation also occurs for states that never equilibrated. Small
corrected variance alone therefore cannot establish posterior exploration.
Raw-score and normalizer gates remain required.

## Float32 target versus float64 diagnostic

The current particle histories target the recorded float32 evaluation;
SummaryGradients uses converted float64 networks. The identity above is for
the intended continuous model, not an exact identity for the rounded
float32 program. Autodiff through casts is not the derivative of its
piecewise-constant rounding map. Float64 is itself a numerical approximation.

The saved-state diagnostic requires unchanged input hashes and beta=1, then
checks sampled float32 replay (absolute tolerance1e-6), independent float64
density replay (1e-10), stored float64 score replay (1e-6), phase-wrap replay
(1e-10), and phase finite differences at1e-4,1e-5,1e-6 (scaled error1e-4,
scaled by max(1,abs(T))). These predeclared local checks are necessary, not
global equivalence certificates. Failed replay blocks fitting/application;
failed outputs remain preserved. The sampled float32/float64 density gap is
reported against1e-5 as a diagnostic flag, not a replacement acceptance gate.

The F64 replay is a diagnostic-only no-grad forward path: NumPy applies
intrinsic shear, then F64 tensor preprocessing and the retained physical
transport/density evaluate the likelihood. It does not call the gradient
evaluator. The legacy dataframe classifier and flow wrappers hard-code F32
inputs, so they cannot replay converted F64 networks. The original F32
`SummaryLikelihood` replay stays unchanged. This distinction was established
by a preserved startup failure, not by changing precision thresholds.

Store every pointwise `Delta = logK_F64 - saved_logK_F32`. Report E[T] and
independent-population ratio MCSE both under the original sampled measure
and under an explicit diagnostic change to the float64 measure. For run r,

`Z64hat_r = Z32hat_r * mean_r(exp(Delta))`,

`T64hat_r = sum_r(exp(Delta)*T) / sum_r(exp(Delta))`.

Combine `sum_r(Z64hat_r*T64hat_r)/sum_r(Z64hat_r)` using the same whole-run
delta-method influence calculation as the original ratio estimator. Logsumexp
prevents overflow. Apply identical weights to raw/corrected score diagnostics.
This reweighting changes only the diagnostic numerical measure; it does not
alter production targets, stored scores, or normalizers. Report all whole-run
weights and their concentration. Importance failure or shared missed modes
can still invalidate apparent precision.

A finite-state maximum of abs(Delta) does not bound all posterior states.
If a uniform log-density discrepancy epsilon were independently established,
one could bound the induced control mean using a density-ratio bound such as
`abs(E32[T64]) <= (exp(2*epsilon)-1) * E64[abs(T64)]`; the diagnostic does
not establish that global premise. A large frozen c can amplify a small
nonzero numerical control mean. Preserve E[T], MCSE, raw and reweighted
results rather than silently declaring the control exactly zero in F32.

## Prepared experiment and execution boundary

`jobs/job_particle_phase_control_n15.sh` defines the frozen experiment.
It requests one V100, checks the GPU model, reads exactly4 v1 plus8 v5,
8 v6v100 and8 v7 completed runs (53,248 saved states), fits only v1, and
evaluates all three fresh groups with that one frozen coefficient artifact.
It keeps every particle, raw/corrected score, T and Delta. Missing runs,
unknown failures, and extinctions are rejected rather than silently dropped.
Extinction-aware estimator integration is outside this prototype.

The root coordinator reviewed tests and scripts and authorized one V100 on
th-cl-nv02, inter, two CPUs and 30 minutes after both prerequisite studies
completed. The first source snapshot contained 293 files, referenced V39
receipts and no caches, with import assertions and a hash manifest. Required
environment variables are
`SBSI_PHASE_SOURCE`, `SBSI_PHASE_ROOT`, `SBSI_PHASE_OUTPUT`, `SBSI_CACHE_DIR`
and `BLENDEMU_RUNS_DIR`. The launcher reads existing models and saved states;
it performs no new posterior sampling. It refuses existing output directories.

Initial job 17420174 failed after 26 seconds at the legacy dataframe F32/F64
dtype mismatch before any scientific diagnostic output or coefficient fit.
The original `source_phase_n15_20261022`, experiment receipt, output directory
and log are preserved under the sampler-search data root. Repair is limited
to the diagnostic forward adapter described above; target files, pilot
selection, input populations and numerical thresholds remain unchanged.

## Correctness evidence

CPU Slurm job 17420019 passed **30 tests in 18.64 seconds** across
`tests/test_phase_control.py` and `tests/test_summary_gradients.py`. Earlier
job 17419937 passed the first 28 checks. The final additions exercised the
complete saved-state diagnostic with analytic model doubles, preserving raw
and pointwise precision arrays, applying independent coefficients, and
blocking correction while retaining evidence when a phase derivative fails.
The initial trained-model run failed before diagnostics or coefficient fitting.
The F64 adapter repair added two real untrained-network replay tests; CPU
Slurm 17420296 passed all 32 tests in 17.16 seconds.

## Real diagnostic outcome: original phase gate failed

Repair job **17420413** ran on the authorized V100 and completed all four v1
pilot diagnostics in a 49-second job. The strict loader then rejected all
four, so **no coefficients were fitted and none of the 24 evaluation
populations were processed**. Raw score/normalizer gates remain unchanged.
All JSON/NPZ, source manifests, logs and receipts are retained at
`sampler_search_20261021/phase_n15_20261022_repair1*`; the consolidated outcome
is `phase_n15_20261022_failure_receipt.json`.

| Pilot | Max scaled phase error h=1e-4 | h=1e-5 | h=1e-6 | Max abs F64−F32 logK |
|---|---:|---:|---:|---:|
| v1_0 | 7.519e-4 | 7.515e-6 | 1.144e-7 | 4.463e-4 |
| v1_1 | 2.811e-4 | 2.811e-6 | 9.994e-8 | 3.266e-4 |
| v1_2 | 1.550e-2 | 1.550e-4 | 1.584e-6 | 5.838e-4 |
| v1_3 | 1.268e-3 | 1.268e-5 | 1.696e-7 | 9.232e-4 |

Every pilot has exact saved F32-kernel replay and exact stored F64-score
replay. Independent F64 density replay differs by at most 1.78e-15, and
periodic wrap by at most 3.20e-13. The failing checks are phase finite
differences against the fixed 1e-4 scaled-error limit at every declared step.
The approximately 100-fold error reduction per tenfold step reduction is
consistent with second-order central-difference truncation at sharp phase
profiles. It does not make the original all-step check pass. No thresholds
were changed and no failed diagnostic was dropped or used for fitting.
Separately, the sampled F32/F64 logK gaps exceed the 1e-5 diagnostic flag;
this flag was never an acceptance gate or a global-equivalence certificate.

A new, separately declared numerical-check protocol could investigate
truncation explicitly, but is not part of this failed experiment. There is
no real-data variance-reduction result or all-group T/MCSE summary from n15.

## Separately declared v2 fine-step protocol

After preserving the original failed experiment, the coordinator authorized
**a new numerical diagnostic protocol** before execution. It replaces the
three phase central-difference steps by **[2e-6, 1e-6, 5e-7]** and retains the
**same 1e-4 scaled-error threshold at every step**. All F32/F64 density,
saved-score and periodic-wrap replay tolerances are unchanged. Receipts and
target diagnostic identity explicitly record `phase-fine-difference-v2`.
This is a finer derivative check, not a changed phase identity, a claim that
the original gate passed, or a change to any sampling/score/normalizer gate.

The motivation is the preserved approximately O(h²) central-difference
truncation evidence in all four pilots. Fine steps test the derivative with
smaller truncation while retaining three independent step comparisons. They
can still fail because of rounding or a wrong derivative; tests deliberately
alter the gradient magnitude and reverse the chain-rule sign to verify that
such errors remain rejected at the same threshold. F64 independent forward
checks exercise the exact new steps at zero and nonzero shear.

The experiment remains 4 v1 pilots and 8 each v5/v6v100/v7. Freeze coefficients
from v1 only by the original covariance/variance recipe, then apply unchanged
to all evaluation groups. Keep raw and precision-reweighted diagnostics and
qualify the F32/F64 difference. No production or heldout CV integration is
authorized. Use a fresh source/receipt/output on one V100 with two CPUs and a
30-minute limit, alongside at most two heldout GPUs. Old coarse-step files
remain immutable and available separately.

CPU Slurm 17420583 validated v2: 32 focused tests passed in 10.88 seconds.

## V2 real result: score noise reduction and remaining disagreement

Slurm **17420604 completed successfully in 3m43s** on one V100, with two
CPUs and no new sampling. All **28 populations / 53,248 saved states** passed
the separately declared v2 numerical checks. Four v1 pilots alone produced
the frozen coefficients, then all eight populations in each evaluation group
used that same artifact. Original coarse-step failures remain preserved.

Coefficient artifact: `phase_n15_20261022_v2/frozen_v1_coefficients.json`,
SHA256 `a31062ffe9c970d8c3ae9214f5f9b7a67781829cd14d37f8ee2461fb155412c7`.
Full means, independent-population MCSEs, raw/F64-reweighted comparisons and
hashes: `phase_n15_20261022_v2_summary.json` under the sampler-search data root,
SHA256 `ca5a07670fae46109614d788f9dc750b897916a2c6199a54a343578cd2b7159f`.

The table reports **shear-score MCSE**, not shear-estimate uncertainty. Each
evaluation group has eight independent populations; within-population
particle variability is not treated as independent replicate evidence.
The coefficients are fixed from different v1 runs.

| Group | Observation | g1 MCSE raw → corrected | g2 MCSE raw → corrected | T mean ± MCSE | T/MCSE raw → F64-reweighted |
|---|---:|---:|---:|---:|---:|
| v5 | 58165 | 4.8511 → 1.4819 | 7.2778 → 2.4932 | 4.2204 ± 2.5364 | 1.6639 → 1.6641 |
| v5 | 0 | 1.4158 → 0.18701 | 0.67347 → 0.64694 | -0.45751 ± 0.46387 | -0.9863 → -0.98629 |
| v5 | 1 | 0.37969 → 0.25041 | 0.3956 → 0.34594 | -0.11283 ± 0.16581 | -0.68049 → -0.68049 |
| v5 | 8 | 0.095123 → 0.092607 | 0.040085 → 0.037664 | 0.028271 ± 0.010462 | 2.7023 → 2.7023 |
| v6v100 | 58165 | 1.4846 → 1.1212 | 2.8027 → 1.7952 | 0.013845 ± 0.71049 | 0.019486 → 0.019539 |
| v6v100 | 0 | 0.56655 → 0.095263 | 0.41244 → 0.41684 | 0.02595 ± 0.18272 | 0.14202 → 0.14223 |
| v6v100 | 1 | 0.10821 → 0.10453 | 0.14272 → 0.1282 | -0.026121 ± 0.05352 | -0.48806 → -0.48806 |
| v6v100 | 8 | 0.035424 → 0.036406 | 0.026269 → 0.023142 | 0.017356 ± 0.0096674 | 1.7953 → 1.7953 |
| v7 | 58165 | 4.6116 → 2.9045 | 4.9461 → 4.7625 | 0.26747 ± 1.3599 | 0.19668 → 0.19688 |
| v7 | 0 | 1.1675 → 0.11117 | 1.2185 → 1.1772 | -0.68086 ± 0.41398 | -1.6446 → -1.6445 |
| v7 | 1 | 0.24184 → 0.19046 | 0.54728 → 0.3526 | 0.22631 ± 0.20644 | 1.0963 → 1.0963 |
| v7 | 8 | 0.068893 → 0.066715 | 0.03667 → 0.037775 | 0.018147 ± 0.014424 | 1.2581 → 1.2582 |

Benefits vary: the bright v5 score MCSE decreases by factors about3.27 and
2.92; bright v7 decreases by about1.59 and1.04. Observation0 g1 improves
substantially in every group. Three evaluated components increase slightly
(v6 observation0 g2, v6 observation8 g1, v7 observation8 g2). This frozen
one-control estimator is useful for some components, not a uniform sampler
efficiency improvement.

The more precise control exposes a disagreement. Observation0 corrected
g1 score means are v5 **0.478684 ±0.187010**, v6v100 **0.404688 ±0.095263**,
and v7 **1.069976 ±0.111169**. The v7−v6 difference is about**4.55 combined
reported MCSEs** (quadrature of the two independent-group errors); the raw
difference was below1.2 combined MCSEs. This descriptive comparison uses
finite eight-population errors and is not an added acceptance gate. It
shows why score variance reduction cannot be equated with convergence.

Global saved F32-kernel and stored F64-score replays were exact. Independent
F64 forward replay maximum error was1.78e-15; periodic wrap5.45e-13. Maximum
scaled phase errors at [2e-6,1e-6,5e-7] were [8.10e-6,2.08e-6,8.74e-7], all
below the unchanged1e-4 limit. The global sampled F64−F32 logK maximum was
9.23e-4. Evaluation-group maxima were6.21e-4 (v5),7.04e-4 (v6v100), and
6.26e-4 (v7), so the1e-5 diagnostic gap flag fails even though the separate
derivative/replay checks pass.

Explicit F64 precision reweighting changes evaluated score means by at
most0.000790 of their corresponding raw reported MCSE. It has negligible
effect on the observed T residuals in these states and cannot explain the
observation0 cross-group discrepancy. This finite-state evidence is not a
global target-equivalence bound and does not establish an exact F32 Stein
identity. The largest evaluated absolute T residual is2.70 MCSEs (v5,
observation8); none of these single-control checks can rule out rare modes
or modes with the same phase expectation.

**Scope remains diagnostic.** No CV is integrated into production sampling,
heldout estimation, or existing reports/gates. No raw score or normalizer
gate is relaxed. Lower score MCSE and small T residuals cannot establish
normalizer accuracy, full posterior coverage, or calibrated shear.
