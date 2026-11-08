# Optional Student-t proposal and the annealing tail question

## Status and interface

This is a candidate for a later particle comparison, not a demonstrated
sampling improvement. The Gaussian proposal remains the default and preserves
its original samples, densities and random-number consumption bitwise.

`MeasurementParticleProposal.from_measurement(...,
component_family='student_t', student_df=5.)` selects a normalized mixture of
nine-dimensional Student-t components, with the same measurement-based centres
and independent uniform phase as the Gaussian version. The constructor accepts
the same options. A proposed driver interface is `--proposal-family student_t
--proposal-df 5`; this branch does not modify the experiment driver.

One chi-squared draw supplies a shared radial scale for all nine coordinates
in a particle. The supplied coordinate scales define the diagonal *shape*
matrix, not marginal standard deviations. At five degrees of freedom the
covariance is 5/3 times that matrix. Thus a comparison at unchanged scales
changes both the tails and the covariance. A separate comparison multiplying
the Student-t scales by sqrt(3/5) would match Gaussian component covariances.
The normalized multivariate density follows the [SciPy Student-t
definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).

## What the native SED factor establishes

Write the four SED ALR coordinates as `y_j = log(c_j/c_4)`. The Dirichlet
density has factors `c_j**(alpha_j-1)`; the ALR Jacobian contributes
`prod(c_j)`, giving

\[
  p_{\rm ALR}(y\mid z,t)\ \propto\ \prod_{j=0}^4 c_j(y)^{\alpha_j}.
\]

This is the factor implemented in `GalSBITargetCoordinates.sed_log_proposal`
and the corresponding native-prior Jacobian. The starting density is the
[Dirichlet definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html).
At fixed redshift, type and other ALR coordinates, `y_j -> -infinity` gives
`log p_ALR = alpha_j*y_j + O(1)`; `y_j -> +infinity` gives
`log p_ALR = -(sum(alpha)-alpha_j)*y_j + O(1)`. These tails are exponential,
whereas any finite Gaussian mixture with fixed positive scales has quadratic
log tails along this direction.

This establishes a tail mismatch for the **SED factor**, not yet the full
posterior. In the intended target coordinates, changing SED at fixed physical
coordinates leaves the likelihood inputs unchanged. However, the native
absolute magnitude, colour, luminosity/radius factors and population-envelope
cuts still change through photometry. They could remove a candidate tail or
alter its mass. A posterior tail statement requires a surviving region of
positive width in the other coordinates and positive probability in the
environment, together with bounds on those remaining factors. These bounds
have not been established for the retained model.

## Consequence for the implemented geometric bridge

Let `f_E(x) = p(x) K(y|x,E)` and let `q_E` be the unchanged environment law.
The particle driver uses

\[
 \pi_\beta(x,E)\ \propto\ q_E(E)\,q_0(x)^{1-\beta}f_E(x)^\beta.
\]

For an ideal exact draw from this bridge, the next incremental weight is
`w_delta = (f_E/q0)**delta`. Its second moment is proportional to

\[
 \int q_E(E)\,q_0(x)^{1-\beta-2\delta}
               f_E(x)^{\beta+2\delta}\,dx\,dE.
\]

This formula follows directly by multiplying the bridge density by the
squared weight. If the full target has a surviving exponential tail and
`q0` has a Gaussian tail on that region, the last jump from `beta < 1` to one
has `beta+2*delta = 2-beta > 1`. Its negative power of `q0` overwhelms the
exponential target decay, so that ideal incremental second moment diverges.
Smaller final steps reduce its coefficient without removing this conditional
asymptotic argument. Full support alone does not ensure finite variance.

The antecedent is unverified for this full posterior. This is **not** an
observation of infinite variance in the actual runs, nor a proof that their
finite errors are dominated by tails. Resampling, finite arithmetic, mutation
mixing and normalizer products require additional analysis. Stage weight ESS
alone also cannot establish that a population has visited all relevant tails.

A Student-t proposal has polynomial tails, which removes this specific
Gaussian-versus-exponential obstruction when the full target satisfies the
assumed exponential bounds. It does not prove efficient sampling, accurate
score expectations, adequate scene-mode coverage, or stable global moments.
Heavier tails can spend more computation on impossible or irrelevant scenes.

## Validation and next comparison

The CPU tests compare Student-t mixture densities with SciPy, integrate the
actual mixture over a conditional slice, test the shared-scale radial F law
and marginal tail probabilities, exercise very large finite coordinates, and
replay the disabled Gaussian branch exactly. No real-model GPU experiment is
part of this branch.

The later comparison must retain the same final target, observations,
particle count, independent replicate seeds and fixed production schedule,
and compare score agreement, normalizer variability, ancestor concentration,
runtime and failed/extinct replicates. Any separate schedule-tuning pilot must
be frozen before fresh production repetitions. A successful analytic density
test is not the development score for this sampling-efficiency hypothesis.
