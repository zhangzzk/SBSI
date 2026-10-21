# Particle bridge and nuisance audit — 2026-10-21

## Finding

Under the continuous sampling-law and numerical-replay assumptions below, the
current bridge and fixed mutation kernels target the declared **analytic
target prior × numerical Poisson environment × likelihood** posterior. That is
not literally the mock generator's conditional latent law: mock targets also
come from the numerical prior sampler. On all eight saved v5 populations, this
target-prior difference has a much smaller score effect than the current Monte
Carlo uncertainty. No production density, reported calibration, or sampler
setting was changed by this audit.

The nuisance cache is supported by the actual likelihood input contract. An
independent algebraic reduction also shows that integrating out target redshift
and SED can be organized through a **three-dimensional function per galaxy
type**. Its numerical evaluation remains future work, not an exact replacement
already available to the sampler.

## Which posterior is being sampled?

Write x for the ten transformed target coordinates, E for the ordered native
environment, y for the observed four-vector, p_A for the type-summed analytic
target intensity including its coordinate Jacobian, and q_N for the numerical
scene generator. The implementation uses

\[
\gamma_\beta(x,E)=q_{N,\mathrm{env}}(E)
q_0(x)^{1-\beta}[p_A(x)K(y\mid x,E;g)]^\beta.
\]

At beta=1 the normalized conditional posterior is proportional to
p_A(x) q_N,env(E) K. The corresponding mock posterior replaces p_A by the
numerically sampled target law. The target intensity's omitted surface-density
constant is independent of x, E and fixed g; it cancels from conditional moments
and shear derivatives. Absolute reported normalizers retain that intensity
convention. The flow's standardized-output density differs from physical-output
density by another constant; `catalogue_disk_likelihood.py:97–115` includes the
transport Jacobian and documents that convention.

The following code distinctions matter:

| Component | Source evidence and consequence |
|---|---|
| Mock joint law | `scripts/check_galsbi_summary_sampler.py:48–75` draws complete numerical scenes, applies C8 Bernoulli retention, samples F8, and transports measured shapes. The transport leaves measured radius/flux unchanged, so its use after their cuts is consistent. |
| Target-independent environment | `galsbi_scene_prior.py:121–129,180–203` uses supplied targets only to determine row count; its count/mark/position laws do not depend on target properties. This is correct for these independent unclustered patches, not a clustered/shared sky catalogue. |
| Ordered environment | `galsbi_prior.py:230–278` assigns each type's samples into original random proposal slots, then returns ascending retained proposal indices. It does **not** concatenate type groups. `galsbi_scene_prior.py:184–198` preserves that order; refresh explicitly randomizes within owners at lines151–161. The ordered law includes 1/N!, as in lines206–217. |
| Usability and measured selection | `galsbi_summary.py:182–188` adds log C8 exactly once. Conditional on a fixed accepted y, deterministic measured cuts contribute a constant indicator and no extra latent-dependent selection divisor. A population shear fit still needs the separate selection normalizer. |
| Observed auxiliaries | `galsbi_summary.py:114–129` binds the response coefficient to the observed radius/flux. It does not discard their density: `catalogue_disk_likelihood.py:184–205` scores the full four-vector. |
| Type mixture | `galsbi_target_coordinates.py:61–82` sums raw type intensities before adding the common Jacobian. It preserves relative type abundance. The returned placeholder type never enters F8, C8, response, or environment sampling. |
| Prior mismatch | `galsbi_prior.py:94–110,242–250` uses a trapezoidal redshift rate and piecewise-linear inverse redshift/log-gamma luminosity CDFs. `log_intensity:192–206` evaluates the analytic formulas. Shared morphology/SED conditional laws do not remove those interpolation differences. |

These are source statements about the current fixed F24 support and model
configuration. They do not establish fidelity to an external galaxy population.
There is a finer exactness qualification: `galsbi_prior.py:246–250` clamps the
luminosity uniform at float.tiny and raises on finite inverse-table overflow.
A continuous interpolation density does not describe the clamp atom or failed
draws, nor literally the finite NumPy RNG law. The standalone numerical-density
work must quantify continuous coverage and these tail events rather than silently
renormalizing them away. No such event was diagnosed in the saved populations;
this is not evidence that this tail explains their score or normalizer errors.

## Bridge weights, mutations, and Jacobians

The beta=0 distribution q0(x)q_N,env(E) is normalized. Its incremental log
potential is log p_A + log K − log q0. The environment factor cancels from the
increment, rather than being omitted from the model. `galsbi_particle_target.py:
19–45,142–150`, `annealed_particles.py:57–103`, and
`scripts/check_galsbi_particles.py:130–141` implement weighting, observation-local
resampling and fixed mutation in that order. This supports the usual unbiased
normalizing-mass estimate under the stated support/invariant-kernel assumptions;
neither log Z nor finite-population normalized score estimates become unbiased.

The coordinate Jacobian in `galsbi_target_coordinates.py:30–54` is triangular:
z, product of the five SED fractions, native radius R, and the two bounded-logit
derivatives. The apparent-to-absolute magnitude derivative is one. There is no
extra distortion-radius factor: the native prior uses da dpsi, not Cartesian
ellipticity area. Phase lives on a circle; its uniform proposal and wrapped moves
match the native 1/(2 pi) law.

The physical/nuisance random walks are symmetric in these transformed
coordinates; SED and independence moves include explicit proposal ratios.
Environment refresh and rejected-at-aperture position moves are reversible under
q_N,env (`galsbi_scene_prior.py:131–177`). Their reduced bridge ratio is therefore
sufficient. Native exchange needs more care: the numerical/analytic mark ratio,
auxiliary p_A(type|x), coordinate Jacobians and, for relocation, position/index
proposal ratios must all be retained. The caller adds the native correction
minus delta log p_A (`galsbi_particle_target.py:187–224`). Combined with the
bridge ratio this is correct at intermediate beta as well as beta=1; drawing the
auxiliary type from p_A(type|x) does not require it to be the tempered conditional.
This is an augmented proposal with its reverse law included, not an uncorrected
tempered Gibbs step.

No new missing Jacobian, incorrect target conditioning or missing normalizer
factor was found here. Float32 likelihood evaluation and the saved float64 score
diagnostic remain different numerical evaluators of the same mathematical model
(`scripts/check_galsbi_particles.py:100–105,160–166`). This audit uses the saved
score as an observable; it does not certify the derivative of a rounded F32
program or remove previously recorded precision limitations.

## New evidence on the numerical target-prior difference

CPU Slurm job **17419483**, partition cluster, exit **0:0**, elapsed **37 s**,
loaded all eight predeclared v5 runs, seeds2026102170–77, and all8192 saved target
particles. It loaded no learned model and reran no sampler. For each particle it
computed the type-marginalized intensity ratio

\[
r(x)=\frac{\sum_t p_A(x,t)\exp[\log(q_N/p_A)(x,t)]}
{\sum_t p_A(x,t)}.
\]

Scores were compared using sum_run Zhat times the particle sums, retaining every
run and its original normalizing mass. The counterfactual changes only the target
prior; the environment already has the numerical mock law. Any unknown global
target-normalization constant cancels from the score ratio.

| Observation | Range of log r on saved states | Score change (g1,g2) | Original score MCSE (g1,g2) |
|---|---:|---:|---:|
| 58165, bright | [−0.05368,+0.06605] | (−0.04143,−0.05589) | (4.8511,7.2778) |
| 0 | [−0.002240,+0.002301] | (−0.000233,+0.001057) | (1.4158,0.6735) |
| 1 | [−0.002950,+0.002207] | (−0.0000165,−0.0000581) | (0.3797,0.3956) |
| 8 | [−0.001781,+0.001343] | (+0.0000105,+0.0000071) | (0.0951,0.0401) |

Every type-marginal numerical density on these states was finite. Maximum prior
replay error was2.05e−12 and maximum conditional type-probability change was
4.28e−7. The bright score changes are below0.009 of their existing MCSE. Its
run-weighted intensity-mass ratio was1.00003171; it is not a separately normalized
population-evidence ratio.

This makes the interpolation mismatch a demonstrated small contribution **on the
represented states**. It does not bound unvisited modes, all observations, or a
calibration c. The score changes are a diagnostic and must not be used as an
empirical calibration correction.

## Nuisance caching and a possible exact reduction

At fixed v=(m_r, log circular radius, n_logit, a_logit, phase) and E, changing
u=(log z, four SED ALRs) adjusts M_B and R_kpc but leaves all likelihood inputs
unchanged. `galsbi_summary.py:21–42,52–95` uses only target angular shape, radius,
magnitude and Sérsic index plus environment geometry/brightness. Neither z, SED
nor type enters directly. The full response-pair list obeys the same rule.

On128 independent native scenes, a z/SED perturbation changed both the F8/C8
inputs and response-pair inputs by at most1.42e−14; pair ownership/order was
unchanged. Same-seed environment draws before/after the perturbation were
bitwise identical. These are stronger input checks than testing only K at g=0,
where the response contribution vanishes. Production cache replay remains a
numerical tolerance check, not bitwise identity everywhere; first-stage checks
and final-state replay do not cover every proposed state.

For a future marginal prior, define a_r=a for a>1e−7 and zero otherwise, and
eta=log Rc−(1/4)log(1−a_r²). Native radius is
R=exp(eta)1000 D_A(z)/(180/pi·3600). Transforming du back to dz dc cancels the
z and product(c) Jacobians exactly. Define, separately for each type,

\[
H_t(m,\eta,n)=\int dz\,dc\;
I_{\rm support}\,\Phi_t(z,M_B(m,z,c))\,
f_{\rm SED,t}(c\mid z)\,
\underbrace{R f_{R,t}(R\mid M_i,z)}_{f_{\log R,t}(\log R\mid M_i,z)}
f_{n,t}(n\mid z).
\]

Then the exact transformed physical prior is

\[
p_{A,\mathrm{marg}}(v)=\frac{J_nJ_a}{2\pi}
\sum_t f_{a,t}(a)H_t(m,\eta,n).
\]

The CPU audit independently assembled this integrand from luminosity, Dirichlet,
log-radius, truncated Sérsic and shape factors. It reproduced the full prior
including Jacobian to1.82e−12. Varying a while holding eta fixed left the
shape-free core invariant to5.69e−14. Thus only three external coordinates per
type are needed for H, although its inner integral still has five dimensions.

There is no obvious analytic closed form: color-dependent intrinsic cuts,
photometry, luminosity, size and redshift-dependent Sérsic laws couple z and SED.
Interpolating a numerical H without correction would change the target. An
unbiased nonnegative estimator with properly retained auxiliary randomness could
support a pseudo-marginal construction, but variance and cost remain untested.
The collapsed bridge would need its own normalized physical q0; simply deleting
nuisance terms from the current bridge is incorrect, since marginalization does
not commute with powers. Native exchange would also need a compatible
conditional nuisance reconstruction. Existing nuisance moves already reuse K,
so the potential benefit is mixing and prior cost, not an automatic reduction in
physical likelihood calls. No efficiency claim follows from this identity.

## Addendum: off-support informed Poisson draws need not be truncated

The earlier Poisson proposal discussion overstates the need to evaluate the full
uncancelled mixture density off the environment support. There is an exact
alternative for the existing execution order. Let h be normalized on an extended
typed native-mark/position space and draw

\[
Q(E)=(1-\alpha)q_{N,\rm env}(E)
+\alpha[q_{N,\rm env}\oplus h](E),\qquad 0<\alpha<1,
\]

where the extra point is inserted uniformly among the resulting ordered slots.
On valid numerical-environment scenes,

\[
R(E)=Q(E)/q_{N,\rm env}(E)
=(1-\alpha)+\alpha\sum_{j=1}^{N}h(z_j)/\lambda_N(z_j).
\]

An extra point outside q_N support has zero final target mass. Such a draw can
remain explicitly marked invalid at beta=0, receive zero weight at the first
strictly positive beta, and be removed only by ordinary resampling. No likelihood
or mutation may run on it first. Its original draw remains in the N denominator;
there is no retry, conditioning on acceptance, or unknown truncation normalizer.
Consequently its finite off-support Q density need not be computed along this
path. Safe payload placeholders are bookkeeping only and may never turn it into
a positive-weight scene.

For the power bridge with this initial law, valid scenes carry R(E)^(1−beta).
The incremental potential must subtract log R, and **all beta<1 mutations** must
include (1−beta) delta log R. Initialization-only correction would target the
wrong bridge. If h depends on x, use R(x,E) everywhere. The valid-scene numerical
intensity must include type density, native Jacobians, aperture density and the
generator's actual interpolation support. The existing
`scene_exchange.numerical_over_analytic:17–48` is intentionally restricted to
analytic support for exchanging into the analytic target slot; it is not a full
numerical environment-density evaluator. This remains an implementation gate for
informed augmentation, including numerical-support edge slivers. The ordered-iid
native generator is compatible with uniform insertion; sorting marks would
require a different measure. This addendum establishes a permissible design,
not a production implementation or efficiency result.

## Reproducibility and remaining problems

The read-only script is `doc/evidence/audit_bridge_nuisance_20261021.py`; its
hashes, all input receipts, and complete numbers are preserved in
`doc/evidence/bridge_nuisance_audit_20261021.json`. The external original is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/smc_bridge_audit_20261021/result.json`.
The source audit refers to the parent worktree's files as of2026-10-21; numerical
check dependency hashes are in the receipt. The audit had no production edits.

The direct unresolved issues are finite-population mass/score stability and
unrepresented modes, the bounded F32/F64 distinction, and implementing any future
prior marginalization or informed environment law without silently changing its
measure. The tested analytic/numerical target-prior difference and nuisance cache
are not leading demonstrated explanations for the current bright-population
normalizer instability. That statement does not establish convergence or a
well-determined population shear estimate.
