# Optional continuous numerical target sampler

## Candidate location and current validation

Implementation is isolated on `codex/numerical-target-driver-20261022`, commit
`1d6171a`; this working branch retains the
existing runtime. CPU17449171 passes 1,129 tests and nine skips, followed by an
installed-prior check on all 2,182 returned draws from 4,096 proposals at fixed
seed2026104200. Decoded numerical density/type sums agree exactly; the maximum
native-coordinate roundtrip log-density difference is 2.2773e-12. All 481 source
hashes match before/after; parent verification checks these, 173 direct input
pins, JUnit and saved draw hashes. See the [validation receipt](evidence/numerical_target_validation_v3_20261022.json).

Review also identified the inherited `ufig.config.common.seed_ngal`, which
controls maximum redshift construction. V3 now binds that source and its package
initializers. Every saved installed-check array is byte-identical to v2. The
failed v1 and passed v2 checks remain preserved; no numerical-target production
or posterior-accuracy claim follows from these tests.




## Candidate scope and API

Arbor n26 is a new scientific candidate based on complete frozen source0301993.
It does not modify or rerun the completed corrected-heldout study. Its numerical
law is optional; existing constructors retain the analytic law.

```python
coordinates = GalSBITargetCoordinates(galaxy, distance, target_prior='numerical')
```

`target_prior` accepts only `'analytic'` and `'numerical'` and is exposed through
a read-only property. Both `log_prior` and `log_mixture_prior` follow it.
`GalSBIParticleTarget` obtains its exchange law from these coordinates; callers
of `propose_native_exchange` can explicitly pass the same `target_prior` keyword.
Conditional type probabilities supplied to an exchange must follow that law.
The default analytic path retains its original density arithmetic and random
draw sequence, including its existing support restriction.

This change covers library routing only. Drivers, protocol identities, fitted
proposal compatibility and phase/report provenance require separate integration
before any scientific run. No likelihood, model, physical coordinate transform,
proposal batch size or empirical correction changes here.

## Target measure and full support

Let u denote native continuous coordinates and t the discrete type. The existing
`numerical_native_log_intensity` supplies q_N(t,u), in counting measure on type
and Lebesgue measure on z, M_B, four independent SED fractions, R_kpc, n, a and
phase. It retains absolute type rates and the implemented inverse-CDF slopes.

For transformed coordinates x, u=u(x), the target contribution is

    p_N(x) = |du/dx| * sum_t q_N(t,u(x))
    Pr(t | x) = q_N(t,u(x)) / sum_s q_N(s,u(x)).

The full type intensities are evaluated directly. In particular, the analytic
support-restricted q_N/p_A helper is not used to build p_N. Numerical-only
interpolation slivers therefore remain available, including points supported
by only one type. Zero inverse-CDF slope, the lower-uniform cutoff and all
other continuous numerical support boundaries remain in force. All-zero type
mass yields -infinity target density and zero conditional probabilities.

`native_prior` retains safe finite placeholders for out-of-domain redshift
draws, while their original coordinates/proposal densities and zero target
weights remain unchanged. There is no redraw or clipping to obtain support.

## Exchange detailed balance at every temperature

Write q_E(E) for the unchanged numerical Poisson environment, q0(x) for the
normalized initial proposal and K for the declared likelihood kernel. The
new bridge is

    pi_beta(x,E) proportional to q_E(E) q0(x)^(1-beta) [p_N(x) K(x,E)]^beta.

An exchange draws a temporary target type from Pr(t|x), chooses a neighbour,
and swaps complete native marks. Both native slots have intensity q_N, so
their native prior factors cancel wherever both marks have positive density.
The returned native correction is zero for ordinary supported exchanges.
Unsupported marks return -infinity and are never retried. A target outside
p_N support receives the existing null exchange at beta=0; other kernels can
enter support.

The cancellation does not remove the transformed-coordinate bridge correction.
With Delta denoting proposed minus retained values, particle mutation uses

    log A = beta Delta log K
            + (1-beta) (Delta log q0 - Delta log p_N) + C_position.

At beta=1 the target/environment native prior terms cancel. At intermediate
beta, the second term is necessary. The temporary type is drawn from the
untempered conditional Pr(t|x), not a tempered substitute.

Without relocation, C_position=0 because the unchanged positions give the same
neighbour-selection probability. With relocation, it contains the reverse/
forward relocation density and the changed neighbour-selection probability:

    C_position = log r(old_position) - log r(new_position)
                 + log s(selected | new_positions) - log s(selected | old_positions).

The original uniform spatial population density cancels within the unchanged
aperture and fixed neighbour count. The existing transformed/marginal Hastings
step subtracts Delta log p_N from the returned native correction; it must not
be removed or multiplied by beta. Auxiliary type probabilities and the swap's
coordinate Jacobian complete the forward/reverse probability-flux identity.

## Validation and limits

Focused tests independently differentiate the generator's redshift/luminosity
inverse maps, combine SciPy conditional densities with absolute unequal type
rates, and verify the transformed type sum/probabilities. Complete MH flux tests
cover beta=0,0.001,0.4,1, both auxiliary types, exchange and relocation, and
targets outside all analytic support. Reverse moves, changed neighbour-selection
weights, relocation densities, zero table slopes, lower-uniform boundaries,
safe placeholders and no-redraw behavior are checked. Analytic regressions
check saved-array bytes, legacy exchange arithmetic and RNG state.

This is the continuous density of the implemented interpolation sampler. It
does not represent singular clamp atoms or the literal discrete finite-RNG
law. Finite inverse-table failures remain errors; no omitted mass is silently
renormalized. Density-law and detailed-balance tests do not establish posterior
mixing, missing-mode coverage, efficient integration or population shear
calibration. Installed-prior checks, combined driver validation and fresh
development experiments remain necessary, followed by an untouched acceptance
panel after any tuning.
