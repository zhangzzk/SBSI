# Continuous-scene sampling design

## Status and recommendation — September 30, 2026

The October2 [convergence audit](MH_CONVERGENCE_AUDIT.md#finding--october-2-2026)
finds curved morphology/shape paths, insufficient local exploration despite
30% acceptance, and a separate wider-shear overlap failure. It recommends a
frozen curved proposal with exact MH correction, followed by overlapping
reference-shear bridges. No sampler acceptance or10-second accuracy claim follows.

The October2 owner-proposed division between atom integration for easier
observations and continuous MH for difficult ones is evaluated in
[hybrid sampling](HYBRID_SAMPLING.md#assessment--october-2-2026).
It requires a common scene prior; joining the old FS2 atom population to the
current GalSBI continuous target is not a pure numerical acceleration.

This began as a design; small sampler prototypes and failed exploration tests
are now recorded below. No sampler has passed the inference acceptance gates.
The owner requested careful design before building the algorithm.
The [continuous scene prior](GALSBI_PRIOR_FEASIBILITY.md#uniform-neighbour-implementation--september-30-2026)
can generate independent scenes and evaluate a native joint density up to a
fixed-population constant. Its catalogue checks do not validate posterior
exploration, the native-to-SBSI mapping, or the selected shear likelihood.

Updated after the owner's fixed-summary observation: **start with exact
likelihood compression and an ordinary prior-proposal independence-MH
reference**, before choosing a local proposal or adding tempering. The
[implemented pilot](#fixed-summary-prototype--september-30-2026) samples fresh
continuous-prior scenes and retains nine numbers per observation/scene.
Variable neighbour count is not an intrinsic requirement of the likelihood.
SMC with blocked scene Metropolis remains a possible escalation, described
below, not a prerequisite or demonstrated improvement over simple chains.

Reuse posterior draws for nearby shear likelihood ratios; an independent
forward-population calculation still supplies measured selection. Tempering
could help sharp bright-object likelihoods and separated modes, but does not
by itself solve a poor mutation proposal. Gradient-free methods avoid relying
on derivatives through nearest-table photometry and the retained tree response.
Continuous support removes dependence on a fixed list of allowed galaxies;
it does not guarantee posterior mass is found or make the chosen population
correct for ConstGold, whose parent is FS2 rather than this fixed F24 model.

## Fixed-summary prototype — September 30, 2026

### Exact compression, not a fitted nine-dimensional prior

At a fixed observed auxiliary pair a=(measured radius, measured flux), define

```text
T_a(scene) = (e1, e2, n, m_r, circularized_Re,
              crowding_050, crowding_100, crowding_200, b_a).
b_a        = sum of retained pair-emulator coefficients at a.
```

The five target properties are pre-shear. All nonself neighbours within10
arcsec enter the three crowding summaries. The retained inference response
contract uses the nearest20 within a strict10-arcsec aperture, without truth
cuts; this existing response truncation is not a cap on the generative prior
or the crowding calculation. At fixed a, the response sum does not depend on
trial shear. The nine summaries are sufficient for the **implemented**
likelihood, not a claim of sufficient information about real images.

The induced prior P_a is the distribution obtained by drawing a scene and
applying T_a. Consequently I_i(g)=E_{P_a}[K_i(T,g)] exactly. No Jacobian or
learned density is needed to generate this distribution. The subscript a is
an index of the deterministic compression; it is not conditioning the scene
prior on observed photometry. The flow density still accounts for those
observed values exactly once.

P_a need not have an ordinary smooth nine-dimensional density: a finite tree
ensemble produces discrete response values and their finite sums, with a
possible mass at b=0. The no-neighbour event also has exactly zero crowding.
Smoothing these with a KDE/flow would be a new approximation. Moreover, one
response number cannot be reused at arbitrary radius/flux observations. The
prototype explicitly rejects such accidental reuse.

### Standard independence MH without evaluating the summary prior

For each chain, generate fresh independent T'~P_a and use

```text
acceptance = min(1, K_i(T',g0) / K_i(T,g0)).
```

The target is K_i(T,g0) dP_a(T) / I_i(g0); proposal and prior cancel, including
point masses. This is ordinary independence Metropolis, not a new method.
It targets the prior defined by the implemented numerical generator; the
previously documented CDF/photometry approximations remain. There is no
analytic-density versus interpolated-proposal ratio to approximate here.

Precomputing a fresh iid proposal stream is harmless: each draw is consumed
once, chains retain rejections, and no generating galaxy is inserted. This
must not be confused with indefinitely resampling a fixed finite catalogue.
The current sampler uses one initial prior draw per chain, without choosing
the highest-likelihood start. It records every state and marks burn-in.

This is a correctness/performance reference, **not an expected efficient
bright-object sampler**. Compressing the state does not increase prior mass
near a narrow likelihood peak. If exploration fails, longer prior-proposal
runs are not the automatic next step. A local move directly in nine-summary
space requires an evaluable joint law or a demonstrably prior-reversible
proposal. Proposals based on explicit latent variables remain another option.
An arbitrary symmetric random walk accepted using only the likelihood would
silently replace the prior and is invalid.

### Implemented boundary and independent pilot

- `sbsi/galsbi_summary.py`: native-to-rendered-ellipse mapping; retained
  all-neighbour crowding and nearest20 response features; nine-summary batch;
  thin likelihood wrapper using the unchanged catalogue density. Density calls
  have fixed4096-row padding, including partial batches. C8 remains a factor.
- `sbsi/summary_sampling.py`: ordinary independence-MH bookkeeping, including
  zero-likelihood starts, mixed measures and repeated states after rejection.
- `tests/test_galsbi_summary.py`, `tests/test_summary_sampling.py`: full-pair
  likelihood replay at several shears, stale-auxiliary rejection, geometry,
  nearest20 versus all-neighbour separation, an analytic normal posterior and
  evidence ratio, and a point-mass prior with an analytic posterior.
- `scripts/check_galsbi_summary_sampler.py`,
  `jobs/job_galsbi_summary_sampler.sh`: independent GalSBI model-mock panel,
  four ordinary chains, standard ArviZ rank-Rhat/bulk-ESS diagnostics, old/new
  density replay, order/partial-batch replay, and local numerator ratios.

The radius mapping was previously unresolved. The installed UFig
`rendering_util.compute_ellip_matrices` together with `draw_photon_gal` gives
semi-axes R_native*sqrt(1+a) and R_native*sqrt(1-a). Thus SBSI circularized
radius is R_native*(1-a^2)^(1/4), not R_native. Distortion converts to
epsilon=a/(1+sqrt(1-a^2)); phase is unchanged. The installed renderer's
1e-7 circularity threshold is reproduced. The pilot independently checks
semi-axes by SVD of the actual rendering matrix (including its off-diagonal
sign). This is an explicit zero-flexion rendered-ellipse interpretation; it
does not change any existing FS2 catalogue or model artifact.

Protocol seed2026093041 fixes8192 independent mock inputs, ordinary C8 draws
and fresh measured cuts. Choose the first retained measurement in each
measured-magnitude bin <20,20–22,22–24,24–25.8 (report empty bins). Separately
generate8192 fresh proposal scenes with seed2026093051, divided into four
independent2048-draw streams. Half is burn-in. Share those proposal scenes
across observations as common random numbers, recalculating b for each.
Inject and sample atg=(.02,0); this tests latent integration, not recovery of
a shear estimate. Generating rows are saved only for diagnostics.

Code gates: renderer replay<1e-12; full-pair/compressed, reversed-order and
partial-batch log-kernel replay<1e-6. Exploratory convergence checks, declared
before launch: rank-Rhat<1.01, bulkESS>400 for nonconstant monitored quantities,
and prior-importanceESS>100 per stream. These are screens, not sufficient
proof of missing-mode absence or0.3% shear precision. Ratios atg1=.018,.022
are numerator ratios only; no population-selection factor or shear result is
claimed. One GPU/one CPU/16GB/20minutes; no automatic scale-up.

## Exact statistical target

Let s denote a native target plus its complete uniform neighbour patch, p(s)
the normalized prior at fixed population parameters, and y_i an observed
four-component measurement that passes the requested cuts. Define

```text
K_i(s,g) = q_U(s,g) * f_disk(y_i | s,g,U)
I_i(g)   = integral p(s) K_i(s,g) ds
Z(g)     = integral p(s) q_U(s,g) P_flow(W | s,g,U) ds
W        = measured radius > 0.6 arcsec and measured magnitude < 25.8

log L(g) = sum_i log I_i(g) - n log Z(g)
pi_i(s | y_i,g0) proportional to p(s) K_i(s,g0).
```

Here f_disk includes the retained Möbius inverse and Jacobian, not just the
base flow. For a fixed observation its measured radius/flux condition the
disk-response coefficient. Do not draw a fresh noisy forward measurement to
estimate K inside Metropolis: K has a deterministic density evaluator.

The selected posterior for a known, retained y_i has no extra latent-specific
division by P(W|s,g). Selection is already implied by this y_i. Equivalently,
the selected latent prior and truncated measurement density have cancelling
selection factors. Z(g) belongs in the catalogue likelihood, once per row.
Using only f_disk drops C8, and inserting P(W|s,g) a second time changes the
target. There are no new truth-property cuts; the declared finite GalSBI
population support remains explicit and needs later completeness checks.

The unknown native population constant in log_kernel cancels from fixed-
population Metropolis ratios. Initial scenes can be drawn by the existing
exact-in-principle rejection/Poisson-thinning construction, subject to the
audited numerical grids. The measured-selection Z(g) does not cancel and is
different from the surface-density normalizer in the prior pilot.

## Arbitrary-scene likelihood contract

The new summary adapter addresses this boundary using temporary row indices
into fresh inputs, not a persistent finite feature store. Required contracts:

- GalSBI distortion to SBSI ellipticity, native radius to the required
  circularized/major-axis conventions, photometric band and output units.
- F8/C8 physical crowding from all guarded neighbours; the emulator's own
  pair-feature construction and pair selection from its retained contract.
- Existing target shear map and output-conditioned disk composition, preserving
  the original model artifacts, physical support and shape log-gap encoding.
- For each unchanged scene/observation, reuse the exact response coefficient
  across shear nodes. Recompute after a latent change affecting its inputs.
- Pointwise density replay against existing code at matching feature/pair
  inputs, including reordered, duplicated and padded batches. Earlier
  batch-shape effects reached0.118 log units; a new sampler cannot fix those.

Do not smooth trees, interpolate a new photometric model or replace the
emulator as part of this numerical experiment. The prior mapping must be
settled before describing the prototype as ready for the retained likelihood.

## Optional explicit-scene Metropolis kernel

### Target and individual-neighbour moves

Use local symmetric proposals in explicitly transformed coordinates, with
the appropriate Jacobians included in the prior density. Blocks should
roughly follow the likelihood: brightness/size, shape, and redshift/SED/Sérsic
variables, plus a discrete type move. Joint proposals may be needed for
correlated variables; do not independently draw a new full galaxy at every
local step. Periodic angles need periodic proposals, and support violations
are rejected rather than clipped into the domain.

Apparent true r magnitude is a useful brightness coordinate. At fixed z and
normalized SED, m_r=M_B+Delta_r(z,c), so dM_B/dm_r=1. Coupled z/SED proposals
can hold m_r fixed and recompute M_B with GalSBI photometry; this follows the
observational constraint without changing the prior. Any additional log,
logit or simplex transformations still need their own Jacobians.

For neighbours, an occasional local update to one uniformly chosen neighbour
helps move an important close companion without destroying the rest of the
scene. Keep the complete latent marks in the first implementation. Analytic
marginalization of unused orientations is possible later, but must be proved
against every likelihood feature rather than assumed.

### A simple count-changing move: partial Poisson refresh

For a state-independent refresh fraction rho in (0,1]:

1. Retain each existing neighbour independently with probability1-rho.
2. Generate an independent marked Poisson patch with intensity rho times
   the original population intensity. Use the existing pre-photometric
   envelope rate multiplied by rho, followed by the ordinary intrinsic cuts.
3. Combine survivors and new neighbours, randomly permuting the resulting
   list to preserve the ordered-iid representation.
4. Rebuild changed likelihood features and perform a Metropolis accept/reject.

This proposal is reversible under the homogeneous marked-Poisson prior. To
see why, write the old/new pair as U+V and U+W, where U has intensity
(1-rho)lambda and independent V,W both have intensity rho*lambda. Swapping
V and W leaves the pair law unchanged. Independent random ordering lifts
that argument to the stored ordered lists.
The thinning-plus-independent-immigration construction is also described in
[Last, Peccati and Schulte, section3](https://arxiv.org/html/1401.7568v1#S3);
the exchangeability argument above establishes the reversibility used here.

At tempering exponent beta, its acceptance is therefore

```text
min(1, exp(beta * [log K_i(s_new,g0) - log K_i(s_old,g0)])).
```

The count factorial is not being dropped: it is accounted for by reversibility
of the proposal with the full Poisson prior. This avoids a separate hand-written
birth/death acceptance ratio and requires no estimated post-cut surface
density. A fixed mixture of small and full refresh fractions changes both
local structure and neighbour count. rho=1 supplies a global environment
refresh; small rho preserves useful close neighbours. State-dependent choices
based on brightness/influence require a fresh balance calculation.

The cancellation assumes the proposal draws the same prior whose density is
being evaluated. Our current redshift/magnitude sampler uses interpolated
inverse CDFs: its small audited CDF error is not a proof of pointwise density
agreement at arbitrarily narrow likelihoods. Include grid refinement and
sampling-versus-density consistency in this gate. If that approximation is
material, evaluate/correct the implemented proposal density or improve the
sampler; do not silently call the practical refresh exactly prior-reversible.
The same consideration applies to the initial SMC prior sample.

Before adding data, unit-test that this kernel preserves the analytic count,
mark and spatial law. Test a soluble likelihood that depends on neighbour
count, so prior-only stationarity cannot hide a wrong posterior acceptance.

## Optional tempered particle initialization and posterior exploration

Use the sequence

```text
pi_beta(s) proportional to p(s) K_i(s,g0)^beta, 0 <= beta <= 1.
```

Start with independent continuous prior scenes. Increase beta gradually,
weighting by K_i^(delta beta), resampling when weights concentrate, then
applying the blocked Metropolis kernel. An ESS-based temperature schedule is
a practical initial choice. Freeze proposal parameters within each mutation
stage; adaptation and stopping rules must be recorded. This is the standard
[SMC sampler construction](https://www.cs.ubc.ca/~arnaud/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf),
applied here to the union of variable-count scene spaces.

The particles can move to new continuous galaxies and environments. Resampling
alone cannot discover a missed mode: retain independent repeats, track ancestry
and actual particle movement, and compare longer mutation budgets. Equal
weights after resampling are not evidence of independent posterior draws.
Very bright objects may still have an expensive prior-to-posterior transition;
measure this before promising a speed improvement or adding a proposal fit.

Never seed a production chain with a mock's generating truth. It may be used
only as an explicitly labelled diagnostic to expose a mode ordinary starts
miss. Adaptive schedules and finite particle counts do not justify a claim
that log evidence or curvature estimates are unbiased.

## Reuse the posterior for shear profiles

At nearby g with common support, the exact identity is

```text
I_i(g)/I_i(g0) = E_{pi_i(s | y_i,g0)} [K_i(s,g)/K_i(s,g0)].
```

Compute that expectation using the same weighted scenes at every profile
node. Then

```text
log L(g)-log L(g0)
 = sum_i log E_i[K_i(s,g)/K_i(s,g0)] - n log[Z(g)/Z(g0)].
```

This avoids needing separately estimated absolute evidences at every nearby
shear and reduces differencing noise. Its finite-sample log remains biased;
control it through repeats, particle/mutation ladders and an independent
reference calculation. It is not a new exact finite-draw estimator.

Monitor the reweighting distribution itself. If overlap is poor, bridge or
recenter with the same particle/Metropolis machinery rather than extrapolating
large ratios. Compare forward and reverse bridges and more than one reference
shear on the probes. Check common support, including numerical zeros in C8;
do not add arbitrary probability floors to manufacture overlap.

Locate a stable two-component likelihood maximum before using curvature for
uncertainty or Newton steps. Negative curvature away from a maximum is not
automatically a sampler failure. Finite-difference scores and Hessians may be
reported only after direct profiles and stencil-size checks agree.

## Independent measured-selection calculation

Generate a separate population of scenes and flow base-noise draws. Reuse
both across g nodes, but regenerate them independently for uncertainty checks.
Estimate q_U times the cut indicator averaged over flow draws, integrating
the Bernoulli usability factor as a probability rather than drawing another
Bernoulli variate. The retained disk map changes shape only, so these radius/
magnitude cuts need no emulator evaluation in Z(g).

Increase population scenes and conditional flow draws separately. Selection
error is shared by every observation and multiplied by n in the log likelihood;
it must have its own convergence budget. The old catalogue normalizer is not
valid for the new GalSBI population, and posterior samples around observed
galaxies cannot replace the forward-population sample. Do not differentiate
hard sampled cut indicators as if their zero pathwise derivative were the
derivative of the selection probability.

## Small experiments and acceptance order

1. **Analytic tests and deterministic replay.** Validate coordinate Jacobians,
   count changes, constant-likelihood stationarity, a soluble count-dependent
   posterior, and native-scene/legacy-feature likelihood agreement. Include
   a reduced continuous toy problem with known integral/profile; matching
   multiple noisy runs alone previously missed shared numerical errors.
2. **Two-observation runtime probe.** One bright and one faint selected mock,
   hidden truth, several independent starts. Benchmark native proposals,
   likelihood calls, pair rebuilding and ESS for shear-ratio observables.
   Try512 and2048 particles as a diagnostic ladder, not an acceptance guarantee.
3. **32-observation stress panel.** Eight observations in each measured-magnitude
   band <20,20–22,22–24,24–25.8, chosen by fixed seed without looking at shear
   results. Use four independent complete sampler runs, and increase particles
   and mutation sweeps independently. Include crowded objects and monitor
   neighbour count, crowding, response coefficient, likelihood and shear ratios.
   Stratification is for numerical diagnosis; it is not a representative
   calibration sample and cannot use an unmodified all-magnitude selection law
   to claim a population shear result.
4. **Representative model closure.** Draw new independent continuous scenes,
   apply C8 and forward F8/disk generation, then measured cuts once. Start
   with hundreds of selected observations, then20k only after convergence and
   runtime are understood. Inference never receives their generating scenes.
   This tests new-prior/model closure, not recovery on FS2 actual measurements.

Compare replicate shear profiles, reference-shear choices, temperature paths,
and particle/sweep ladders. Monitor importance-ratio ESS and ancestor diversity;
for independent frozen-kernel diagnostic chains, use rank-normalized R-hat and
functional ESS as in
[Vehtari et al.](https://arxiv.org/abs/1903.08008).
Do not apply ordinary independent-chain R-hat to resampled SMC particles, or
treat good scalar diagnostics as proof that an unseen mode is absent.

The eventual numerical budget must be in shear units. At injected g1=0.02,
0.3% corresponds to an absolute shift6e-5. A proposed numerical target is
at most1e-5 in the recovered shear, including selection integration, leaving
most of the budget for population/model error. Small diagnostic panels cannot
establish that bias requirement; replicated, adequately powered calibration
is a later test. Do not accept a numerically unstable result merely because
its point estimate lies near the injection.

## Implementation and resource boundary

The first implementation contains exact summary compression and ordinary
independence MH. The failed prior-proposal pilot motivates the bounded local
pseudo-marginal experiment below; explicit-scene mutation and SMC remain
unimplemented alternatives. Use existing prior and forward-selection machinery
where its contracts match.
Cache response terms only for unchanged latent/observed inputs. CPU handles
native photometry and ragged scene manipulation; batch neural likelihood
calls on GPU after pointwise batch replay. Benchmark larger batches explicitly.

The original design was documentation only. Subsequent owner-authorized
implementation and short jobs are recorded below. Existing resource limits
and startup monitoring rules apply; no long or population-scale run follows
automatically from a short diagnostic.

## Summary independence-MH results — September 30, 2026

Artifact directory: `galsbi_summary_mh_20260930_v2` under the external
catalogue-prior root. GPU job16776331 completed the numerical script in28.50s.
Of8192 independent input scenes,3188 were usable and1961 passed measured cuts.
All four predeclared magnitude strata were populated. Geometry replay against
the actual renderer had maximum error2.22e-16; full-pair/summary, reversed-order
and padded-partial density replay was exactly equal after matching classifier
batches. No extra compressed-density approximation was introduced.

The first job16776323 deliberately stopped before MH at a3.98e-6 log-kernel
replay discrepancy. It came entirely from C8 evaluated in8192 versus4096
rows, not summary compression. The rerun kept the1e-6 threshold and matched
classifier batching to the wrapper; the original batch discrepancy is saved.
Scientific seeds, observations, proposals and acceptance gates were unchanged.

| Measured magnitude | Prior importance ESS per2048-draw stream | Acceptance after burn-in | Result |
| --- | --- | --- | --- |
|19.745|1.00–3.08|0–0.098%|failed exploration|
|21.255|1.05–3.76|0–0.587%|failed exploration|
|23.971|2.90–11.32|0.196–0.684%|failed exploration|
|25.118|20.45–24.86|0.684–2.053%|failed exploration|

These ESS values describe normalized importance weights of independent prior
proposals, not effective independent chain samples. Rank-Rhat also failed.
This establishes that likelihood compression alone does not make proposals
from the broad prior effective. It does not reject the continuous prior or
standard MCMC, and it is not a shear estimate.

## Local pseudo-marginal follow-up — September 30, 2026

The next simple established method keeps a fixed target vector and integrates
the environment, without fitting a new density or moving individual neighbours.
Let theta be the native target coordinates, eta an independent uniform
environment, and K the nine-summary likelihood above. Define

```text
L(theta,g) = E_eta[K(T_a(theta,eta),g)]
Lhat(theta,g) = (1/M) sum_j K(T_a(theta,eta_j),g), eta_j iid from p_env.

accept = min(1, p_theta(theta') Lhat(theta',g0)
                / [p_theta(theta) Lhat(theta,g0)])  # symmetric proposal
```

This is [pseudo-marginal MH, Andrieu and Roberts2009](https://arxiv.org/abs/0903.5480).
The nonnegative average is unbiased on the likelihood scale for the declared
environment generator. Rejecting a proposal retains the current target,
estimate, and complete compressed environment bank. Recomputing the current
estimate at every step would instead be an uncontrolled noisy-MH approximation.
The extended target is proportional to p_theta(theta) product_j p_env(eta_j)
Lhat; integrating environments recovers p_theta(theta)L(theta).

The ordinary MH state uses ten continuous target coordinates (log z, apparent
r magnitude, four SED additive-log-ratios, log angular circularized radius,
bounded Sérsic/distortion logits, phase) and binary galaxy type. This is
fixed-dimensional, but not a pure nine-coordinate random walk. The extra
target latents retain the known native prior rather than learn its difficult
five-property marginal. Jacobians include z, R_kpc, the simplex product of
SED fractions and bounded logits; dM_B/dm_r=1. The full ten-dimensional
Jacobian has an independent finite-difference determinant test.

Under uniform positions at fixed population parameters, the environment law
is independent of target properties. `sample_given_targets` therefore draws
only the neighbours for fixed targets. The response/crowding still depend on
both target and neighbours and are recomputed. No neighbour-count cap or
Gaussian approximation is added. Target density is the analytic native prior;
the environment is its numerical generator, so the previously audited
grid approximation is explicit rather than described as exact GalSBI.

For posterior scene moments, each retained bank is internally weighted by
K_j/sum K. For shear ratios, the extended-target identity is

```text
I(g)/I(g0) = E_extended_posterior[sum_j K_j(g) / sum_j K_j(g0)].
```

Do not average its M environments as equally weighted posterior draws. Each
retained bank and target is one correlated MCMC state; rejects remain in all
moments. The measurement-selection normalization Z(g) is still separate.

Bounded pilot: the same bright/faint observations from the summary run, four
chains each, M=4 environments per proposal,1024 steps with512 warmup. Start
from the highest-likelihood candidate in each independent reference proposal
stream, using only measurements, never generating truth. Proposal blocks:
photometry/size, shape, redshift/SED/Sérsic, symmetric type flip, and an
environment-only refresh, with probabilities.3/.3/.2/.05/.15. Three proposal
scale factors adapt during warmup only and are frozen afterward. Seed2026093061.
Monitor rank-Rhat/ESS for target summaries, redshift, type, weighted environment
summaries and nearby shear ratios; a constant type is unresolved, not passing.
No shear estimate or automatic increase of sample size is authorized by a
favorable point estimate. Full tests after this addition:343 passed,1 skipped
in51.41s (CPU job16776388); the local GPU pilot is recorded in the work log.

The first M=4 pilot completed in51.50s of script time (job16776398). It improves
acceptance, particularly for the faint measurement, but fails between-chain
agreement. A separate1024-environment probe at each final target (16776440)
shows why two bright chains stick: their stored likelihood estimates exceed
the probe's mean estimates by exp(7.59) and exp(8.63), approximately2000 and5600.
The independent probe itself has only2.82 and4.47 effective environments at
those targets; these factors are diagnostic, not exact error measurements.
The standard deviations of log averages fall from16.86/17.86 atM=4 to2.31/1.76
atM=64. This identifies severe environmental integration noise for those
target states, without claiming that all bright targets have that noise.

The refined local pilot keeps the same observations and chain length, uses
M=64, and computes logsumexp of the two native type intensities exactly.
`log_mixture_prior` marginalizes galaxy type because photometry, geometry,
F8/C8, response and the uniform environment law are type independent at fixed
native continuous coordinates. Relative red/blue population abundances remain
in their intensities; they are not assigned equal prior mass. The target
proposal therefore has ten continuous coordinates and no discrete type move.
Conditional type probabilities are monitored instead. Scale adaptation is
separate for bright and faint observations and freezes after warmup. This is
an implementation refinement, not a controlled claim that only M changed.
An independent fixed-target noise probe follows without altering the chains.
Full tests after exact type marginalization:344 passed,1 skipped in43.53s
(CPU job16776457). The complete result/protocol/source snapshot lives at
`galsbi_local_mh_20260930_v2` under the external catalogue-prior root.

Refined GPU job16776459 completed successfully in330.48s script time
(about5.5minutes), but **both observations fail exploration checks**:

| Diagnostic | Bright, MAG_AUTO19.745 | Faint, MAG_AUTO25.118 |
| --- | --- | --- |
| Rank-Rhat over monitored summaries, type probability, z and ratios |1.79–3.11|1.07–1.50|
| BulkESS over those quantities |4.56–55.91|7.81–51.42|
| Photometry/size proposal acceptance per chain |5.9–27.4%|23.0–42.2%|
| Shape proposal acceptance per chain |1.9–46.6%|33.5–42.2%|
| Environment-only proposal acceptance per chain |3.3–100%|81.1–92.2%|

The two problematic bright targets still have only5.73 and12.17 effective
environments among1024 independent noise-audit draws. Their retained
log-estimates exceed the audit means by8.36 and3.89; at the two other bright
targets the noise is small. Faint log-average SD atM=64 is0.13–0.22, yet its
target/latent mixing remains insufficient. Thus more environment draws alone
does not fix all the observed problems. The refined run changed several
algorithm choices and is not a causal estimate of any single improvement.

**Retain the likelihood compression; do not adopt either sampler for shear
inference.** The bounded experiment is complete. No20k/500k follow-up was
launched and no selected shear likelihood or estimate was produced.

The next design issue is proposal efficiency, separately for target moves and
rare environments. One principled option is to retain native environments as
auxiliary state during local target moves, recompute their summaries, and
refresh only part of the environment with a prior-reversible proposal. That
reduces the independent-estimator noise but requires retaining more than the
nine summaries internally; the physical likelihood still sees only nine.
An evaluable compressed prior is another route, with its additional density
approximation and mixed-measure validation. This evidence does not justify
pretending a symmetric nine-coordinate random walk has a known prior, or
removing inconvenient neighbours, type populations, or observed galaxies.

## Persistent-scene MH pilot — October 1, 2026

The owner authorized the next step above. The existing local driver now has
an explicit-scene mode, using one environment per chain. Its stationary law is

```text
pi(theta, eta | y,g0) proportional to p_target(theta) p_env(eta) K(y|theta,eta,g0).
```

Here `p_env` is explicitly the implemented numerical GalSBI neighbour
generator. The target retains the analytic native density, exact red/blue sum
and ten-coordinate Jacobian. This is the same marginal target as the previous
pseudo-marginal pilot; the finite chains and initial auxiliary states differ.
The known native sampling/density approximation is still present.

Symmetric target proposals hold the complete neighbour scene fixed and use
the target-prior times likelihood ratio. Environment proposals hold the target
fixed and use the [partial Poisson refresh](#a-simple-count-changing-move-partial-poisson-refresh),
so only the likelihood ratio remains. There is no noisy likelihood average
to regenerate during target moves. A refresh independently retains90% of
neighbours and adds a0.1-intensity fresh patch; a state-independent10% mixture
of refresh proposals instead redraws the whole patch. All accepted and
rejected chain states are recorded. Useful environments can persist, but
environment and target modes may still trap chains; determinism is not proof
of exploration.

Predeclared small pilot: same measuredmag19.745/25.118 observations, four
chains each,4096 steps with1024 warmup, seed2026093061. Starting scenes retain
the original neighbours from each reference stream's highest-likelihood
candidate, selected without generating truth. Initial log kernels must
reproduce the saved reference within1e-6. F8/C8 calls remain padded to4096.
All SED coordinates, Sérsic/distortion logits, neighbour counts and three
crowding summaries join the previous Rhat/ESS diagnostics; wrapped phase is
represented by the two ellipticity components. Necessary gates stay at
rank-Rhat<1.01 and bulkESS>400. No selection integral or shear estimate is
produced by this two-observation diagnostic.

Implementation tests cover ragged scenes and rejection retention, stationary
Poisson counts/marks/positions, the expected old/new count covariance, and a
likelihood with analytically known tilted-Poisson counts and mark density.
Source snapshots, starting replay, protocol and final native scenes are saved
with the output under `galsbi_persistent_mh_20261001_v1`.

The first pilot16778742 completed in89.68s script time, with exactly matching
starting log kernels. **Both observations fail**: maximum Rhat is2.860/1.097
and minimum bulkESS4.775/34.88 for bright/faint. Bright magnitude/size acceptance
ranges0.99–27.92% across chains under a common proposal scale. Its shear
numerator ratios also disagree strongly. Environment acceptance near90% does
not establish that target or environment posterior modes are explored.

One bounded follow-up uses separate warmup scales for each chain, with the
same posterior and refresh law,16384 steps/4096 warmup. This changes both
adaptation and length; any change in diagnostics is not attributable to one
alone. Output `galsbi_persistent_mh_20261001_v2`, job16778765. It completed in
272.29s script time, with exact initial replay and neither observation passing:

| Final diagnostic | Bright19.745 | Faint25.118 |
| --- | --- | --- |
| Maximum rank-Rhat |2.2307|1.0921|
| Minimum bulkESS |5.896|39.20|
| Rank-Rhat of log numerator ratio atg1=.022 |1.4675|1.0129|
| BulkESS of that ratio |72.61|511.12|
| Magnitude/size acceptance across chains |18.8–32.1%|29.6–40.2%|

The separate read-only audit16778767 exposes the remaining geometry. Bright
chains0/3 remain in red SED regions (mean conditional red probability above
0.99999998), chains1/2 in blue SED regions (near zero). Exact type summation
removes a discrete variable, but leaves the separated components in the
continuous SED density. Chain1 explores a broad intrinsic-magnitude range,
with10–90% quantiles19.35–22.39, versus19.71–19.75 for chain0. It also accepts
zero of181 full environment proposals, while accepting89.3% of partial
refreshes. Good overall acceptance therefore hides two exploration problems.
These chains do not establish the relative masses of their occupied regions;
choosing the chain with the highest likelihood would be invalid.

The next minimal proposal to test is a red/blue SED-mixture independence move.
At fixed z, m_r, angular size, n, distortion and phase, draw template fractions
from the known mixture of GalSBI Dirichlet laws. Convert them to the existing
additive-log-ratio coordinates and recompute M_B/photometry and native prior
support. In those coordinates the proposal density is

```text
q_x(x_SED | z) = [0.5 Dir(c; alpha_red(z)) + 0.5 Dir(c; alpha_blue(z))] prod_j c_j.
```

Accept using the full target-density ratio times q_x(old|z)/q_x(new|z).
The equal mixture is only a proposal; native red/blue abundance and intrinsic
cuts remain in the target. This directly connects the observed SED regions
without estimating an additional population density. Verify that the
likelihood-visible physical summaries remain fixed to numerical precision.
The subsequent [SED-jump pilot](#sed-independence-jump--october-1-2026)
implements this proposal. It does not by itself solve trapping in rare
environments, which may need local moves of existing neighbours. No shear
estimate or measured-selection normalization has been produced.

## SED independence jump — October 1, 2026

The owner authorized this next step. `GalSBITargetCoordinates.propose_sed`
draws a component uniformly, samples its Dirichlet fractions by normalized
gamma draws, and returns the ALR-coordinate proposal with the full mixture
log q(old|z)-log q(new|z). The density sums both components, not just the one
used to generate the proposal. `log_mixture_prior` retains native population
abundance, all morphology/luminosity factors, Jacobians and intrinsic limits.
Rejection preserves the old state; off-support proposals are not retried.

The opt-in driver block replaces0.15 of the former0.25 local structure
probability with this independence jump. Photometry/size, shape and
environment proposal probabilities are unchanged. For every valid-prior SED
proposal, the pilot recomputes and checks that all nine physical summaries
agree within1e-10 and log likelihood within1e-6. It stops on a failed check,
instead of assuming exact cancellation in finite-precision code. It also
records the log prior, region transitions, conditional red-probability means
and separate partial/full environment acceptance.

The tests independently compare the proposal density with SciPy Dirichlet
densities and the ALR Jacobian; recover the moments of a known unequal
mixture multiplied by a c0 likelihood; and check native luminosity changes,
physical-feature invariance and intrinsic-cut rejection after a colour move.

Bounded protocol: same bright19.745/faint25.118 independent model mocks,
four chains each,16384 steps/4096 warmup, per-chain scale adaptation frozen
after warmup, seed2026093061 and persistent neighbours with the previous
refresh law. Starting scenes remain reference-prior candidates selected by
likelihood, without generating truth. No population weights, model artifacts
or measured cuts are adjusted. Rank-Rhat<1.01 and bulkESS>400 remain necessary
gates. Output `galsbi_sed_jump_20261001_v1`.

Full test suite16778800:353 passed,1 skipped in52.28s. GPU16778801 completed
in294.96s script time. Starting likelihood replay is exact. Across19,808
finite-prior SED proposals the largest summary difference is4.2633e-14 and
the largest log-kernel difference is zero. The read-only chain audit16778804
is saved alongside the protocol, source, chain arrays and final native scenes.

| Diagnostic | Bright19.745 | Faint25.118 |
| --- | --- | --- |
| Maximum rank-Rhat |1.5730|1.0629|
| Minimum bulkESS |6.94|63.07|
| SED ALR rank-Rhat range |1.0029–1.0106|1.0056–1.0085|
| Post-warmup region crossings per chain |24,28,28,86|36,2,40,36|
| Mean conditional red probability per chain |0.9912,0.9793,0.9804,0.9561|0.05485,0.00041,0.07438,0.02091|
| Rank-Rhat of log numerator ratio atg1=.022 |1.0489|1.0106|
| BulkESS of that ratio |1660.64|445.94|

The bright chains now occupy similar SED components and accept75–86% of
full environment proposals, including the chain that previously accepted
none. Nevertheless, the component weights and physical posterior are not
validated. Bright Sérsic index is now the worst diagnostic, with Rhat1.5730
andESS6.94; angular radius has Rhat1.3641 andESS9.13. Faint magnitude/crowding
remain slow, and faint chains disagree on rare red-region mass. The change
allocates fewer attempts to the old joint structure block, which also moves
Sérsic index and redshift; this affects any efficiency comparison.

**Retain the corrected SED proposal as a tested building block, but do not
adopt this sampler for inference.** Atg1=.022 the bright per-chain log numerator
ratios still range−0.01379 to+0.00780. These are numerator ratios for one
observation, not shear estimates or selected-population likelihoods. Passing
some marginal ESS checks, or crossing components, does not establish the
required joint exploration.

Read-only audit16778808 finds within-chain correlations−0.43 to−0.80 between
bright log angular radius and Sérsic logit. Those quantities currently move
in different blocks, and the new SED block also reduces the frequency of
Sérsic moves. A minimal next test is a joint magnitude/size/Sérsic Gaussian
proposal, with covariance estimated during warmup and frozen afterward.
This changes the proposal only; retain the full posterior and SED correction.
The [overnight follow-up](#overnight-comparison--october-1-2026) implements
and tests this proposal. Faint magnitude/crowding show
correlations0.55–0.98, partly inherent because crowding uses target-relative
flux; those correlations alone do not prove an environmental trapping mechanism.
At the end of this SED-only pilot all its jobs were complete; no
population-sized follow-up or production change was accepted.

## Overnight comparison — October 1, 2026

Owner authorized autonomous overnight work. Three small proposal pilots keep
the same posterior, two observations, four chains each, seed2026093061 and
16384 steps/4096 warmup. All fail the full convergence gates:

| Proposal | Bright maximum Rhat | Bright ratio Rhat | Faint maximum Rhat | Faint ratio Rhat |
| --- | --- | --- | --- | --- |
| Joint magnitude/radius/Sérsic |1.2836|1.0581|1.0625|1.0094|
| Joint morphology plus distortion/phase |1.3724|1.2389|1.0646|1.0045|
| Joint morphology plus local neighbour positions |1.3893|1.0537|1.0634|1.0261|

Ratio diagnostics here are for the log numerator ratio atg1=.022. Maximum
Rhat includes native and physical-summary diagnostics. These short runs
cannot rank small efficiency differences reliably. Neither added proposal
consistently improves the simpler joint morphology variant, which is retained
for the longer comparison without claiming acceptance.

`sbsi/mcmc_proposals.py` estimates a per-chain covariance from at most1024
retained warmup states, including rejections, every128 steps. Shrink5% toward
the original diagonal scales maintains proposal support. Covariance and
scalar multipliers freeze after warmup. Independent correlated-Gaussian
tests check stationary moments. The optional five-coordinate block wraps
phase; optional position moves reject individual out-of-aperture steps and
use a prior-reversible uniform-disk kernel. Both remain disabled overnight.

The frozen overnight protocol uses all four reference observations
(MAG_AUTO approximately19.745,21.255,23.971,25.118), four chains each and
two MH seeds2026100101/2026100102. Each run has262144 total steps and32768
warmup. Independent seeds share the four dispersed reference-prior starting
scenes per observation; generating truth is not used for initialization.
SED mixture jumps, exact red/blue summation and persistent Poisson refresh
are unchanged. The source snapshot and protocol are saved under
`galsbi_overnight_20261001` in the catalogue-prior output root.

GPU array16778955 runs at most two concurrent GPUs, each with a four-hour
limit. Both initial likelihood replays are exact and both have passed startup.
Dependent CPU job16778957 runs after either success or failure and writes a
read-only report at total-step prefixes65536,131072,262144. The report checks
individual runs and combined seeds, native latents, all nine summaries,
red probability, neighbour counts and the local shear-ratio functions.
It also reports likelihood-scale Monte Carlo error and largest individual
draw contributions: convergence of log ratios alone does not establish an
accurate mean likelihood ratio. Missing/incompatible runs cannot pass.

Rank-Rhat<1.01 and bulkESS>400 remain necessary gates, not proof of global
exploration. Fixed run length avoids stopping on an attractive estimate.
There is no automatic sampler promotion or population-sized follow-up.
Latest full suite16778930 passed364 tests with1 skip. These experiments still
omit the measured-selection normalization and cannot produce a selected
shear estimate or a0.3% accuracy claim.

## Proposal geometry audit and staged pooling — October 1, 2026

The overnight pair completed in about79 minutes. All numerical checks passed,
but worst combined Rhat at262k steps was1.0301/1.0622/1.0112/1.0170 for
observations0–3. See the saved `galsbi_overnight_20261001/report/report.json`
for individual, combined and prefix diagnostics. The selected-population
inference boundary is unchanged.

A read-only scheduled covariance audit identifies proposal steps that are
too local relative to the explored morphology distribution. In the hardest
observation, several eigenvalues of posterior-whitened proposal covariance
are0.002–0.007: proposed standard deviations along these directions are
only5–8% of the sampled posterior width. The original tuning uses each
chain's last1024 warmup states, so slow chains can learn only their local
motion. The sampled covariance is not a converged ground truth; this is a
testable efficiency hypothesis. Thousands of accepted full environment
refreshes make simple inability to replace neighbours an incomplete account.

The opt-in `--pooled-morphology` experiment pools discarded warmup states
across four chains within each observation, uses an8192-step window, and
updates covariance every1024 steps. Covariance freezes halfway through
warmup; the second half tunes only scalar scales. All adaptation stops before
retained sampling. The three-dimensional magnitude/log-radius/Sérsic block
and5% shrinkage remain unchanged, as do posterior density, likelihood and
environment law. Frozen symmetric Gaussian proposals require no additional
Hastings correction. Independent covariance and Gaussian-target tests pass;
full suite16781066 has366 passed and1 skipped.

GPU array16781077 compares65536 steps/32768 warmup on the same four
observations and seeds2026100101/2026100102 against the saved65k overnight
prefix. Sources and protocol are frozen under `galsbi_pooled_mh_20261001`.
CPU report16781079 follows either success or failure. This is an efficiency
test, with no automatic acceptance or population-sized launch.

## Coupled target and environment proposals — October 1, 2026

The aggregate environment acceptance hides slow conditional exploration.
In overnight obs2, when true r exceeds measured MAG_AUTO by1 magnitude,
only3 of742 full environment attempts were accepted across both seeds,
versus roughly28% outside that diagnostic region. Obs3 accepts4 of459.
The region is a diagnostic partition, never a prior or sampling cut. Chain
occupancies disagree strongly, so their values cannot be taken as posterior
mass estimates. Curved bright morphology and these faint-target excursions
are distinct issues; pooled covariance need not fix both.

An opt-in joint proposal occasionally changes morphology and the entire
environment together. Write the posterior as pi(x,e) proportional to
p(x)r(e)K(x,e), with target coordinates x and target-independent numerical
environment generator r. Propose x' from a symmetric frozen Gaussian q and
e' independently from r. Then

```text
pi(x',e') q(x|x') r(e) / [pi(x,e) q(x'|x) r(e')]
    = p(x') K(x',e') / [p(x) K(x,e)].
```

This cancellation uses the implemented environment law; no approximation to
its density is evaluated. A state-independent20% of morphology attempts use
a4-times larger target increment and full refresh. Other blocks and the
iteration budget remain unchanged. Joint attempts do not tune the local
scalar scale; all adaptation still stops after discarded warmup. The entire
target/environment state is accepted or rejected. A coupled uniform-target/
Poisson-count test checks stationary moments against quadrature. This move
is experimental and does not change the scientific prior or likelihood.

Both65k-step development pairs completed with numerical replay checks
passing. Neither passes exploration gates. At matched budgets, worst combined
Rhat for obs0–3 is1.1175/1.4171/1.0324/1.0438 with pooled tuning, and
1.2918/1.2254/1.0221/1.0211 after adding joint environment moves. Original
tuning at the same endpoint gives1.3009/1.3116/1.0489/1.0264. Minimum ESS
is also insufficient in each joint-run observation. Joint moves have only
about0.1–3% acceptance for the faint observations; one obs1 chain accepts
zero of1966 attempts. These results do not establish a generally better
sampler. No method is promoted; the full fixed-budget comparison is saved in
`galsbi_joint_env_mh_20261001/report/comparison.md` under the output root.

## Longer original-MH chains — October 1, 2026

Owner authorized longer chains. This experiment returns to the original
per-chain three-coordinate morphology covariance, SED mixture jump and
persistent Poisson environment refresh. Pooled covariance and joint target/
environment moves remain disabled. Four replicas use MH seeds2026100121–124,
four chains per observation, the same four observations,1,048,576 total steps
and32,768 discarded warmup. A two-GPU array throttle gives two successive
waves; measured timings imply roughly ten hours for all four replicas.

Initialization is independently regenerated: bank seeds2026100221–224 each
draw8192 prior scenes, partitioned into four2048-scene streams. Each chain
starts at the highest-likelihood scene in its stream for its observation.
No generating truth is used, and the observed panel is unchanged. The new
optional initial-bank flag records seed, selected rows and kernels, and
checks full-bank versus reconstructed-start likelihoods within1e-6 before
sampling. Default initialization still replays the historical reference bank.

All four banks receive short startup checks before the long jobs. Sources and
protocol are frozen in `galsbi_long_mh_20261001`. The final report compares
individual and combined replicas and prefixes65k/131k/262k/524k/1,048,576.
Prefix diagnostics are computed after completion from saved unthinned
states, not used for favorable-result stopping. Rhat<1.01 and bulkESS>400
remain necessary gates. The experiment tests exploration and integration,
without measured-selection normalization or a shear-bias estimate.

## Cost and scalability audit — October 1, 2026

The million-observation cost estimate extrapolated a diagnostic configuration,
not a production requirement: four observations per GPU, four chains per
observation, one million iterations, four independent replicas, and every
state retained. The completed jobs take about5h11m each with about10GB peak
host RAM. Repeating this configuration is prohibitively expensive. Filling
4096 padded model rows with useful states offers room for batching, but a
256-fold end-to-end gain was only an optimistic illustration, not a forecast.

### Measured components and exact-work reuse

`scripts/benchmark_galsbi_sampler.py` uses the sixteen saved final scenes
from long seed2026100121, tiles them across batch sizes, and makes independent
local proposals. It measures the original morphology/shape/structure/SED/
partial/full-environment mixture, using eight timed repetitions per block
after a warmup. Profiling is restricted to the discarded warmup repetition,
so the reported timings do not include cProfile overhead. These are component
probes, not independent new observations or an end-to-end convergence test.
All likelihood evaluations keep the4096-row contract and measured shape
log-gap encoding. No truth selection, population change or retraining occurs.

CPU job16787846 completed in2m09s on one allocated core, peakRSS928MB.
Median component timings, weighted by original proposal probabilities:

| Observations (four chains each) | Full components/step | F8/C8 likelihood | Response prediction | Scene inputs | Target prior |
| --- | ---: | ---: | ---: | ---: | ---: |
|4|249ms|231ms|4.66ms|6.66ms|4.72ms|
|32|276ms|231ms|20.76ms|12.24ms|8.02ms|
|128|369ms|231ms|76.12ms|31.39ms|18.85ms|

The remaining time includes environment generation and scene bookkeeping.
These CPU likelihood times must not be substituted for GPU throughput.
The table identifies growing CPU work underneath the fixed padded neural
evaluation. The original response model runs single-threaded on the CPU;
its support for CUDA and additional CPU threads should be benchmarked with
numerical replay before choosing a larger observation batch.

The opt-in `--reuse-invariant-work` flag in
`scripts/check_galsbi_local_sampler.py` implements three bounded changes:

- Structure moves in log-z/SED and SED independence moves preserve r magnitude,
  angular circularized radius, Sérsic n, distortion, phase and neighbours.
  They therefore reuse the full likelihood; the complete native prior,
  Jacobian, intrinsic support and SED Hastings correction are still evaluated.
  Together these are25% of the original proposal mixture.
- Neighbour-only updates reuse the unchanged target prior, while evaluating
  the new likelihood and retaining the same Poisson refresh law.
- Target-only moves retain existing neighbour arrays instead of concatenating
  and copying an unchanged neighbour catalogue after every MH decision.

The flag requires the existing joint-morphology setup so that Sérsic n is
outside the invariant structure block. Full reconstruction audits the first
and every1024th occurrence of each invariant block, with the existing
1e-10 summary/1e-6 log-kernel tolerances. Failure aborts; the flag remains off
by default. SED-region-switch bookkeeping is retained even when K is cached.
The report compatibility key includes this flag. No chain is thinned or
stopped early by this change.

All CPU component probes passed invariant and batch replay with no changed
MH decisions. The component subtraction suggests about24% less work on CPU;
this is not an end-to-end speed measurement. Unit validation includes a
nontrivial colour/redshift prior and complete neighbour-response summaries.
Suite16787870:368 passed,1 skipped in60.77s.

### Paired replay and GPU benchmark

Sources and plans are preserved under
`catalogue_prior/galsbi_cost_20261001`. CPU component profiling uses its first
snapshot; GPU profiling uses `source_snapshot_optimized`. A512-step paired
sampler replay uses identical proposals, seeds and independently regenerated
starting banks, comparing retained states, MH decisions and local profiles.
Its CPU-only first attempt16787880 correctly failed the historical GPU-start
replay (2.35e-5 difference, versus1e-6 tolerance). Retry16787889 uses bank
seed2026100133 evaluated on the same device in both arms; this changes the
initialization, not the replay tolerance. The failed output is preserved.

CPU paired replay16787889 completed in5m00s and passed. Both512-step arms
retain exactly identical target coordinates, acceptances, types, neighbour
counts, log priors, log kernels and local likelihood-ratio samples. Summary
differences are at most2.84e-14. Cache audits replay32 valid proposals with
zero log-kernel error, skipping130 likelihood calls and86 target-prior calls.
Whole-process timings are164.41s versus132.82s. Differencing progress records
at128 and512 steps removes initialization:249.74ms/step versus187.80ms/step,
about24.8% less loop time. This is one short paired CPU run, not a confidence
interval or a GPU speed measurement. The first frozen wrapper's
`sampling_seconds` field includes initialization; use the progress differences
or whole-process numbers instead. The current wrapper fixes that label.

GPU benchmark16787881 follows completion of long task16781571_2 and successful
CPU profiling, preserving the two-GPU study limit. It covers4/32/128/512/1024
observations and compares the unchanged response artifact on CPU1, CPU4 and
CUDA. Acceptance requires likelihood replay, not only similar response means.
Its Torch peak-allocation counter excludes XGBoost-managed GPU memory.
GPU paired replay16787894 follows the benchmark and successful CPU paired
replay. These bounded jobs do not automatically launch larger inference.

### What must improve beyond step cost

The completed long replicas still fail for MAG_AUTO21.2551, including the
shear-dependent likelihood ratios. Millions of stored steps are not millions
of independent samples. Earlier audits show curved magnitude/radius/Sérsic
structure and, in other observations, low acceptance of full environment
refreshes conditional on faint-target states. A pooled Gaussian covariance
and broad joint target/environment move have already failed to establish a
general improvement; repeat runs do not erase that evidence.

After measuring the execution improvements, the next small proposal test
should address the curved morphology distribution: a frozen small Gaussian
mixture fitted on discarded pilot states, used for occasional independence
MH proposals with the explicit q(old)/q(new) correction. Retain ordinary
local/environment moves and the exact target. This is a proposal design,
not a learned replacement prior; it is not yet implemented or validated.
The proposal may still reject strongly if morphology depends on the current
environment, and it cannot establish unvisited posterior mass by itself.

Judge alternatives by agreement across independent starts and Monte Carlo
error in shear-relevant likelihood ratios per unit time, while retaining
latent diagnostics as evidence of exploration. Population-scale integration
needs an error budget propagated to shear: independent zero-mean integration
noise can average across galaxies, whereas systematic missed modes do not.
Finite-sample bias from taking logs or ratios also needs separate control;
it does not disappear merely by adding observations. Shared-randomness
covariances must be included when propagating errors between shear points.
There is no demonstrated justification for requiring the same million-step,
sixteen-chain stress test of every galaxy, or for arbitrarily shortening it.
Full histories also need streaming diagnostics before large-batch production.
No million-observation runtime, selected shear estimate or0.3% accuracy is
validated by this cost audit.

## Measurement-informed initialization comparison — October 1, 2026

Owner requested a direct test of starting near measured properties. The
comparison changes initialization only; it uses the same four observations,
V3.9 artifacts, GalSBI prior, fixed shear and original proposal/adaptation
settings. Invariant-work reuse remains disabled in both arms so the GPU
validation of that separate optimization cannot confound this test.

Two matched seed pairs use MH seeds2026100141/142 and fresh prior-bank
seeds2026100241/242. Each arm runs16384 total steps with4096 discarded warmup
and four chains per observation. The four GPU tasks are ordered control141,
measurement141, measurement142, control142 to counterbalance arm order over
two waves. The same seeds pair the arms; the two seed pairs are independent.
They use two GPUs at most, after the existing long-chain/cost jobs finish.

### Initial states and scope of the comparison

The control keeps the existing highest-likelihood scene in each independent
2048-scene stream of an8192-scene prior bank. The measurement arm uses the same
native redshift/SED draws and neighbour scenes, replacing physical target
coordinates with dispersed guesses around the observation:

- Magnitude centre is zero point minus2.5log10(observed flux), with0.6mag
  Gaussian scatter.
- Radius centre uses measured radius in pixels times pixel scale, with the
  Moffat half-light radius subtracted in quadrature and a positive initial
  floor at10% of measured radius. Log-radius scatter is0.5. Quadrature is an
  initial guess, not an exact Sérsic/PSF/measurement inversion.
- Initial shape uses inverse reduced shear of the measured shape plus0.2
  scatter in each epsilon component, followed by the exact epsilon-to-native
  distortion conversion. Saturation at98% of the supported epsilon maximum
  applies to starting guesses only. It is not a shape cut or a correction for
  measured-shape shrinkage/blending.
- The four streams centre Sérsic n on0.75,1.5,2.5,4, restricted to the interior
  of native support, with0.35 scatter in the bounded-logit coordinate.

`sbsi/galsbi_initialization.py` supplies these guesses. The unchanged native
prior and full forward likelihood then score every valid candidate, and the
measurement arm chooses the largest target log prior plus logK in each stream.
It does not read generating-galaxy properties. The full native support check
still rejects invalid candidates. Initial candidate restrictions never become
posterior restrictions: subsequent MH has the original full support.

This is an **initializer-package** comparison: measured centring and candidate
ranking both differ from the existing control. An improvement would not by
itself distinguish those contributions. Both arms evaluate2048 candidates
per chain; preparation time is included separately in the report. This is a
bounded candidate search, not a new fitted inverse model or MAP solver.

The optional `--initialization measurement` flag requires a fresh bank and the
existing persistent-scene/type-marginalized sampler. Defaults preserve the
control. Initial coordinates, physical summaries, kernel, prior and neighbour
counts are now saved in `initial_state.npz`; initialization records include
conversion centres, candidate counts and preparation time. Exact padded
full-bank versus selected-start replay remains a1e-6 gate.

### Validation and interpretation

Unit tests cover flux/pixel units, inverse-shear and epsilon/distortion
conventions, unchanged nuisance coordinates, finite boundary/unresolved
guesses, and input validation. Full suite16788224:370 passed,1 skipped in
67.59s. A CPU startup job exercises both arms and both banks for64 steps,
with32 warmup, before the GPU array is allowed to start; these short runs
are not required to pass convergence gates.

Startup job16788261 completed all four arm/seed combinations in3m47s,
peakRSS1.04GB. Initial likelihood replay is exactly zero in every case;
SED log-kernel replay is also zero, with maximum summary difference2.84e-14.
The GPU pilot array is16788277 (four tasks, throttle2), dependent on this
successful startup and completion of long array16781571 and GPU cost/cache
continuation16787894. CPU comparison report16788283 runs after the pilot
array finishes, including failures. GPU results remain pending.

At the owner's request, pending GPU stages16787881,16787894 and16788277
were moved in place to `inter`, requesting `gpu:a40:1`. Four unallocated A40s
were observed on kng-cl-nv02 at the check. Existing dependencies and the
two-GPU combined cap remain; the moves do not bypass upstream completion.
Actual resource receipts must identify the inter hardware when interpreting
timings against earlier cip virtual8GB-A40 measurements.

The owner subsequently authorized the two inter initialization workers to
run alongside the two old cip workers: the combined cap is now four GPUs.
After verifying smoke16788261 completed successfully, removed the old-long
and cache dependencies from initialization array16788277 (still throttle2).
The cost benchmark retains its dependency on old task16781571_2, taking that
task's freed slot, and the cache replay follows the benchmark. Report
dependencies and all scientific settings are unchanged.

Frozen sources, plan and output live under
`catalogue_prior/galsbi_measurement_init_20261001`.
`jobs/job_galsbi_initialization.sh` launches the smoke/pilot stages;
`scripts/report_galsbi_initialization.py` runs the existing within-arm reports
at8192/16384-step prefixes and compares local likelihood-ratio samples across
all sixteen chains. It preserves initial-state and timing evidence. Common
seeds may correlate arm errors, so the report does not interpret a difference
divided by independent MCSEs as a significance test.

The gates remain Rhat<1.01 and bulkESS>400, including shear-relevant ratios,
plus unchanged numerical replay. Faster arrival at a high likelihood is not
convergence. Agreement only within the measurement arm is also insufficient
if independent starts or the control disagree. No initializer is automatically
promoted, and the experiment produces no selected-population shear estimate.

## Population diagnostic and throughput target — October 1, 2026

The owner requested20k observations with16384 steps,4096 warmup, four chains
per observation and two independent replicas, then paused execution after
reviewing the estimated36–48h walltime on four GPUs. On October2 the owner
set an initial target of10 GPU-seconds per galaxy, superseding the interim
10-galaxies-per-second-per-core target for the next milestone. Use the earlier
per-replica interpretation, with four chains and initialization/output costs
included; two independent replicas cost twice as much. No20k sampling array was submitted.
Preparation16788795 was cancelled after16s; suite16788794 passed374 tests
with1 skip. The external root is `catalogue_prior/galsbi_population20k_20261001`.

The prepared workflow generates independent model observations in generation
order, applies C8 and the existing measured cuts, and retains the first20k
selected observations without magnitude strata or truth cuts. Sampling uses
fresh measurement-informed starts and independent seed pairs per shard.
The reporter compares aggregate numerator likelihood ratios across replicas
and8k/16k prefixes, including covariance within a shard and between shear
points. Identical selection normalization cancels from replica differences;
no selected shear estimate or accuracy acceptance is implied.

The measured full-component GPU workload at1024 observations costs0.5777s
per step, or9.24s per observation per16384-step replica before initialization,
storage and reporting. Invariant-work subtraction estimates7.27s, not a
measured full-process result. A0.1s-per-observation target requires a substantial
change in throughput and/or integration method. Fewer samples must be justified
by aggregate numerical error, not assumed cancellation. Keep this population
launch paused while evaluating that requirement.

### October 2: ten GPU-seconds per galaxy and the pop-cosmos comparison

Read [Halder et al., section3.3](https://arxiv.org/html/2602.03930v2#S3.SS3)
and the sampling/performance discussion of
[Thorp et al.](https://arxiv.org/html/2406.19437v2).
The paper supports the plausibility of this throughput scale for galaxy
posterior sampling; its hardware, target distribution and validation are
different from SBSI and do not establish our shear-integration accuracy.

Our own1024-observation component benchmark gives9.24 GPU-seconds per galaxy
for16384 four-chain steps, before initialization, storage and reporting;
7.27 seconds with invariant-work subtraction remains a component estimate.
Baseline step cost is58.5% CPU response prediction,17.5% scene inputs,
11.6% target prior and2.1% F8/C8 likelihood. Thus the immediate runtime
work should address response/scene costs and initialization, then measure
the entire run on distinct representative observations. Moving the response
model to CUDA is not validated at these batch sizes. Preserve its identity
and the original replay tolerances.

Batched affine-invariant target-coordinate proposals are worth a separate
bounded comparison with the original MH proposals. Explicit neighbour
refreshes still need their own valid transitions; the tree response and
variable neighbour scenes prevent a direct adoption of an everywhere
differentiable MAP pipeline. Ensemble walkers are coupled and cannot simply
be counted as independently initialized chains. Evaluate proposals by
independent-ensemble agreement and shear-ratio MC error per elapsed second,
including the difficult observation from the million-step experiment.
This is a recommendation, not an implemented sampler or accepted result.

At10 seconds per galaxy per replica,20k galaxies and two replicas require
111.1 GPU-hours, or27.8h on four equally fast GPUs before queueing and any
population-level preparation/selection-normalization cost. Keep the large
array paused until a representative full-process benchmark meets the budget
with acceptable aggregate integration error. No jobs were submitted for
the paper review.

## Bounded throughput and hybrid-ensemble experiment — October 2, 2026

Owner authorized proceeding with the ten-GPU-second milestone. The20k array
remains paused. A separate frozen156-file study under
`catalogue_prior/galsbi_throughput_20261002` prepares1024 distinct, newly
model-generated measured-selected observations (seed2026100401). Preparation
16788816 completed in26s; geometry replay error2.22e-16, full likelihood,
partial-batch and reordered-batch log-kernel replay errors all exactly zero.
No magnitude strata, truth cuts or generating-truth initial states are used.

The new launcher `jobs/job_galsbi_throughput.sh` and driver
`scripts/run_galsbi_throughput.py` provide a paired256-step cache-off/on replay
on128 distinct observations, followed on success by one full1024-observation
replica with16384 steps/4096 warmup and four chains per observation. The
measured process cost includes interpreter/model setup, measurement-informed
initialization, retained-history output, local shear profiles and diagnostics.
Population preparation is recorded separately. A short run or component
extrapolation cannot satisfy the10-second target.

### Complementary morphology stretch proposal

The opt-in `--ensemble-morphology` arm replaces half the original0.3
morphology-block probability with a0.15 stretch proposal on coordinates
`[m_r, log_circularized_radius, sersic_logit]` (indices1/6/7). Within each observation, choose one of
two fixed halves of the four walkers at random; choose each active walker's
anchor from the frozen complementary half. Draw z with density proportional
to1/sqrt(z) on[1/2,2], propose x'=anchor+z(x-anchor), and include2log(z) in
the MH ratio. All other target coordinates and each walker's environment
are held fixed. Prior support remains enforced by the original density.
Local full-support morphology, shape, SED and environment moves remain;
stretch proposals do not tune the local proposal scale. This is a hybrid
three-coordinate experiment, not the paper's512-walker full-SPS sampler.

`sbsi/mcmc_proposals.py` tests cover affine equivariance, unchanged nuisance
coordinates, and Gaussian closure with walker-specific conditional targets.
The sampler records stage timings and flags coupled walkers. Existing
independent-chain reporting rejects this arm; its old per-walker diagnostics
are descriptive only and cannot confer an exploration pass.

`scripts/report_galsbi_ensemble.py` averages likelihood ratios over the four
interacting walkers at each iteration before computing Rhat/ESS/MCSE from
the two independent ensemble time series. Tests ensure duplicating perfectly
correlated walkers does not artificially shrink MCSE and that displaced
replicas fail agreement. Agreement of two ensemble means cannot establish
all latent-mode coverage. The report uses8k/16k prefixes of the original four
stress observations, including difficult obs1. MH seeds2026102101/102 and
bank seeds2026102201/202 are matched across arms; equal steps do not imply
equal active target proposals because a stretch event moves only half the
walkers. Both arms use validated invariant-work reuse and CPU response
prediction, with4096-row neural density calls and unchanged replay tolerances.

Full suite16788840:378 passed,1 skipped in55.21s; Python and shell syntax
and Ruff checks pass. Paired cache16788841 and four-arm startup16788842
precede throughput16788845 and ensemble array16788846 (throttle2), respectively.
CPU report16788848 follows the ensemble array after any outcome and rejects
missing results. At most three GPU allocations overlap in the submitted
dependency graph, within the four-GPU ceiling. Startup results are pending;
no throughput or accuracy claim, selected shear estimate or sampler promotion.

Startup update:16788841 and16788842 both completed successfully. On128
distinct observations, cache reuse preserves all retained states, MH decisions,
log priors/kernels and local profile draws exactly; summary differences are
at most4.26e-14. Sampling time falls25.097s→18.525s (26.2%), while complete
256-step runs cost204.806s→194.07s. Initialization alone costs1.22s per galaxy.
All four control/stretch startup runs pass numerical checks. Full control
16788846_0 is progressing; remaining main jobs are scheduler-priority queued.
A bounded2h Slurm status check16788875 writes a receipt without sending a chat
notification. No full-process10-second result or sampling-accuracy acceptance
has yet been established.
