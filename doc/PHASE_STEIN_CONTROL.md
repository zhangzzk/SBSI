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

`jobs/job_particle_phase_control_n15.sh` is prepared but **not submitted**.
It requests one V100, checks the GPU model, reads exactly4 v1 plus8 v5,
8 v6v100 and8 v7 completed runs (53,248 saved states), fits only v1, and
evaluates all three fresh groups with that one frozen coefficient artifact.
It keeps every particle, raw/corrected score, T and Delta. Missing runs,
unknown failures, and extinctions are rejected rather than silently dropped.
Extinction-aware estimator integration is outside this prototype.

The root coordinator must review tests and scripts, freeze source without
caches, and set scheduler dependencies/resource limits after jobs17419615
and17419751 before any submission. Required environment variables are
`SBSI_PHASE_SOURCE`, `SBSI_PHASE_ROOT`, `SBSI_PHASE_OUTPUT`, `SBSI_CACHE_DIR`
and `BLENDEMU_RUNS_DIR`. The launcher reads existing models and saved states;
it performs no new posterior sampling. It refuses existing output directories.

## Correctness evidence

CPU Slurm job 17420019 passed **30 tests in 18.64 seconds** across
`tests/test_phase_control.py` and `tests/test_summary_gradients.py`. Earlier
job 17419937 passed the first 28 checks. The final additions exercised the
complete saved-state diagnostic with analytic model doubles, preserving raw
and pointwise precision arrays, applying independent coefficients, and
blocking correction while retaining evidence when a phase derivative fails.
No trained-model evaluation or real-data coefficient fit was performed.
