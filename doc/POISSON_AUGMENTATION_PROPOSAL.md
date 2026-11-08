# One-point environment augmentation: exact tilt and remaining requirements

## Status

Arbor n11 supplies a mathematical derivation and generic CPU prototype. It
does not change the SBSI sampler or construct a real-model informed mark law.
The motivation is to seed rare bright-neighbour explanations that changing
only the target proposal may fail to represent. Whether this improves score
accuracy or runtime remains untested.

The new functions in `sbsi/poisson_augmentation.py` are
`point_log_augmentation_ratio(log_h, log_intensity)` and
`poisson_augmentation_log_tilt(log_h_over_intensity, indptr,
mixture_probability)`. They operate on arbitrary point densities and ordered
CSR scenes. The caller must establish that h is normalized and is supported
where the original point intensity is positive. No estimator of h's unknown
normalizer, clipping, rejected-point refill, or real-model approximation is
hidden in these helpers.

## Ordered density and the insertion factorial

Let `u` contain native marks, discrete type, and Cartesian offsets in arcsec.
All densities below use the same product of native-coordinate Lebesgue
measure, type counting measure, and Cartesian area. Write `lambda(u)` for the
environment intensity and `Lambda = integral(lambda)` for its finite total.
An ordered environment `E=(u_1,...,u_N)` has density

\[
 q_E(E)=\frac{e^{-\Lambda}}{N!}\prod_{j=1}^{N}\lambda(u_j).
\]

The factorial is present because the sequence conditional on N is iid ordered.
The augmented mechanism draws the original process, draws one independent
point from normalized h, and inserts it uniformly among the new N slots. For
N>=1 its density is

\[
 q_A(E)=\sum_{j=1}^{N}\frac1N h(u_j)
       \frac{e^{-\Lambda}}{(N-1)!}\prod_{k\ne j}\lambda(u_k)
       =\frac{e^{-\Lambda}}{N!}\sum_j h(u_j)\prod_{k\ne j}\lambda(u_k).
\]

Consequently, on positive q_E support,

\[
 \frac{q_A(E)}{q_E(E)}=\sum_j\frac{h(u_j)}{\lambda(u_j)},\qquad
 R(E)\equiv\frac{(1-a)q_E(E)+a q_A(E)}{q_E(E)}
   =(1-a)+a\sum_j\frac{h(u_j)}{\lambda(u_j)}.
\]

There is no remaining N or 1/N factor: the count-factorial ratio and uniform
insertion cancel. At N=0 the sum is zero and R=1-a. Taking `0<a<1` retains
every original environment, including the empty one. Summing over all possible
inserted slots also makes the density independent of arbitrary neighbour
ordering. Deterministically appending the new point and then pretending that
its distinguished last slot was iid ordered would not have this law.

If h is normalized and h is absolutely continuous with respect to lambda,
Campbell's formula gives `E_qE[sum(h/lambda)] = integral(h) = 1`, hence
`E_qE[R]=1`. This expectation identity is the specialization of
[Last and Penrose, Proposition 2.7](https://stoch.math.kit.edu/img/Last/lastpenrose2017.pdf#page=25).
Equivalently, direct summation of the ordered densities yields
`exp(-Lambda) * sum_{N>=1} Lambda**(N-1)/(N-1)! = 1`.
The unknown Poisson normalizing factor cancels from the ratio; the **absolute
point intensity lambda**, including its rate, is still required.

## Support cases and singular mass

If h(u)=0 but lambda(u)>0, that point contributes zero; other points may still
contribute. The helper accepts its log ratio as negative infinity. A scene
where all contributions vanish still has R=1-a.

If h has probability b outside positive lambda support, q_A has singular mass
b relative to q_E. The finite expression above remains the density ratio of
the absolutely continuous part, but its expectation becomes

\[
 E_{q_E}[R]=1-a b,
\]

not one. A point with lambda=0,h>0 produces an infinite formal ratio, while
lambda=h=0 is an undefined 0/0 at a point outside both laws. The supplied
helpers reject these cases. They cannot certify global support from ratios
evaluated only on existing points.

For a general implementation that evaluates densities or mutates singular
states, use the full mixture density

\[
 q_{\rm mix}(E)=\frac{e^{-\Lambda}}{N!}
 \left[(1-a)\prod_j\lambda(u_j)
       +a\sum_j h(u_j)\prod_{k\ne j}\lambda(u_k)\right],
\]

evaluated without cancelling zero base factors. With exactly one zero-lambda
point its augmented term can be positive; with two such points a single-point
augmentation cannot produce the scene. The final posterior still assigns zero
mass to these off-environment-support scenes. They must remain among the
original initial draws and receive zero weight at positive beta. Replacing
them, discarding them, or treating an invalid addition as an unchanged base
scene changes the normalized proposal unless the induced probabilities are
derived explicitly. No such singular-support machinery is implemented here. A narrower exact
alternative is available when the first operation is weighting at strictly
positive beta: keep every singular initial draw with zero target weight, and
never score or mutate it before ordinary resampling. The original population
denominator must remain unchanged. Only valid-state R is then evaluated; all
intermediate-temperature mutations must still include its tilt. See the
[audited first-stage construction](PARTICLE_BRIDGE_AUDIT.md#addendum-off-support-informed-poisson-draws-need-not-be-truncated).

## What a real SBSI informed point law must supply

The retained environment is generated by numerical GalSBI inverse CDFs and
Poisson thinning. Its point intensity inside the aperture is
`lambda_env(mark,offset) = q_native(mark)/ARCSEC2_PER_SR`; outside the aperture
it is zero. The area factor cancels between the Poisson count rate and uniform
Cartesian position density. The analytic native intensity used by
`GalSBIScenePrior.log_kernel` is not automatically this numerical intensity.

On shared analytic/numerical support, the existing
`scene_exchange.numerical_over_analytic` provides the numerical correction to
the native intensity, including type rates and both inverse-CDF slopes. Its
table rate supplies the absolute pre-cut intensity; post-cut surface-density
estimates are unnecessary for this reconstruction. However, that helper is
explicitly restricted to analytic support. Evaluating R for all generated
marks requires either a faithful numerical-intensity evaluator on the needed
support or a proved h support restriction allowing zero contributions elsewhere.
An arbitrary intensity constant cannot be dropped from h/lambda.

A normalized transformed-coordinate proposal `q_X(x)`, an explicit normalized
type law `pi_t(x)`, and normalized Cartesian position law `s(v)` would give

\[
 h(\operatorname{native}(x),t,v)
 =q_X(x)\,\pi_t(x)\,s(v)/|J_{\rm native\leftarrow x}|.
\]

The type probability and native-coordinate Jacobian must both be present.
The unnormalized analytic target intensity is not a normalized h. Nor is an
original-mark rejection sampler's density known merely because it can draw
marks: normalizing the post-cut mark law requires its accepted surface density.
The fixed pre-cut parent rate cannot be substituted for that unknown quantity.

The existing broad measurement-based q_X has mass outside native population
support. Calling its draws "bright guesses" does not remove that mass. Using
it directly for h therefore needs explicit singular-support accounting, either
the general density above or the audited zero-weight-first-stage construction,
with every off-support draw retained. The finite-tilt route instead needs an
explicit, normalized mark proposal whose support is certified inside the
numerical environment support. Bounded or conditional proposals are possible,
but every truncation normalizer and conditional feasibility restriction must
be evaluated, rather than introduced by retrying until a mark is valid.

Positions can use an exactly normalized aperture-supported law, such as an
inverse-CDF draw from a truncated central Gaussian with its known radial mass.
Its evaluated density must be Cartesian: a radial density cannot be used as
an area density without its polar Jacobian. Mixture weight a and h may depend
on the fixed observation or a frozen independent pilot. Adapting them to the
unrecorded base environment would require a different derivation.

## Bridge and mutation accounting

For the absolutely continuous construction, the full normalized initial law
is `q_target(x) q_E(E) R(E)`. With `f=p_analytic*K`, the bridge is

\[
 \gamma_\beta(x,E)=q_E(E)
       [q_{\rm target}(x) R(E)]^{1-\beta}f(x,E)^\beta.
\]

The incremental potential becomes
`log p + log K - log q_target - log R`. Every proposed state must have the
appropriate R evaluated; the additional reduced bridge log-density difference
is `(1-beta) * delta(log R)`. At beta=1 the added term vanishes and the intended
posterior is unchanged.

Environment refreshes and position moves retain their original q_E-reversible
proposal corrections, but must include this bridge term. Native exchanges
change neighbour marks and possibly their positions, so they also change R.
Their existing numerical/analytic and auxiliary-type corrections remain.
A target-only independence move draws from q_target and has Hastings correction
`log q_target(old)-log q_target(new)`; it must not use a full joint initial
density as though the environment had also been redrawn. Keep target proposal
and environment tilt as separate state fields. If h is later conditioned on
x, R also depends on x and its change must be included in target moves.

## Validation and bounded feasibility

Slurm 17419159 ran
`python -m pytest tests/test_poisson_augmentation.py -q`:
**16 passed in 3.25 s**. The tests include direct enumeration of ordered
insertion histories, normalized spatial/mark integrals, the shifted-Poisson
count identity, empty scenes, zeros in h, order invariance, stable log ratios
of magnitude 10000, invalid support and invalid CSR/probability inputs.

A Cartesian unit-disk toy with a continuous mark and two discrete types uses
160,000 original environments to check `E[R]=1`. Another 180,000 environments
drawn from the augmented mixture recover normalization, mean neighbour count,
mark and radius moments, type counts, a rare central-mark region, and empty
scene probability under weights 1/R. All checked Monte Carlo discrepancies
are within five estimated standard errors. A separate analytic test shows
the missing mass `a*b` when h extends beyond base support.

These results validate the generic tilt and its accounting. The unresolved
work is a known normalized SBSI h, faithful numerical intensity evaluation
over its support, and all bridge/proposal updates. No real-model convergence,
independent score agreement, efficiency gain or accepted sampler follows yet.
