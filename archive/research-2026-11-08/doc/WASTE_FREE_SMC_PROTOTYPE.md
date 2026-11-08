# Waste-free fixed-schedule SMC — n20 prototype

## Fixed scope and source

The prototype is isolated from the real scene driver and frozen snapshots.
[Dau and Chopin, sections 2.4–2.5](https://arxiv.org/html/2011.02328v2#S2.SS4)
describe resampling M=N/P ancestors and retaining each ancestor plus P-1
invariant-kernel iterates. Their path-space construction supports normalizer
estimation. This prototype uses the existing SBSI ordering: incremental
weighting, resampling, then mutation at the new temperature. It does not assume
the paper's asymptotic or practical efficiency claims apply to SBSI.

## Own finite-population derivation

Let gamma_t=q0 exp(beta_t L), Z_t=gamma_t(1), and suppose the previous empirical
measure satisfies E[Zhat_{t-1} muhat_{t-1}(g)]=gamma_{t-1}(g) for integrable g.
The fixed incremental weight is G=exp((beta_t-beta_{t-1})L). Write
c=muhat_{t-1}(G), W_i=G_i/sum G_i, and Zhat_t=Zhat_{t-1}c. Conditional on the
previous population, unbiased resampling gives an expected fraction W_i of M
ancestors. For one fixed pi_t-invariant K, retaining a path of P states gives
conditional expected final mean sum_i W_i A f(x_i), where
A=(I+K+...+K^(P-1))/P. Consequently

E[Zhat_t muhat_t(f)]
=E[Zhat_{t-1} muhat_{t-1}(G A f)]
=gamma_{t-1}(G A f)=gamma_t(A f)=gamma_t(f).

This proves both the normalizer identity (f=1) and the weighted-observable
identity by induction. It does not make fhat or log Zhat unbiased. Dependence
between retained path points does not enter this linear expectation argument.
Uncertainty must still use independent complete populations.

If predetermined kernels K_1,...,K_(P-1) each preserve pi_t, replace K^p by the
prefix K_1...K_p. Every prefix preserves pi_t, so the same average operator A
does too. This is our direct derivation for varying fixed block kernels, not an
asymptotic theorem imported from the paper. The implemented API restricts one
fixed composite kernel callback per temperature; it can compose predetermined
invariant blocks, but does not choose them from production particles.

The first empirical measure is the full N iid draws from normalized q0.
Initial zero-target samples retain their original denominator. If all weights
vanish, c=0 and both unnormalized estimates are zero; no resampling/mutation or
refill occurs. Integrability and target support are prerequisites. Unknown
numerical failures are errors, not extinction.

## Predeclared synthetic experiment — before results

Two 2D targets: a Gaussian with mean(1,-0.5), covariance
[[1,0.92],[0.92,1.5]], and normalizing constant2.5; and a mixture with weights
(0.3,0.7), means(-4,0)/(4,0), common variances(0.65^2,0.9^2), and constant1.7.
Normalized q0 is centered Gaussian with standard deviations(3,3) and(5,2),
respectively. The fixed bridge schedule has16 stages, beta_j=(j/16)^2.
Both use the same fixed symmetric random walk: covariance0.7^2 times target
covariance for the Gaussian, and diagonal standard deviations(0.52,0.72) for
the mixture. No production adaptation or pilot fitting occurs.

Standard generic SMC uses N=120 and7 mutations per stage. Waste-free uses
M=113, P=8, N=904 and retains7 transitions per path. Exact memoization of the
synthetic log potential avoids reevaluating unchanged states and is applied
identically to both algorithms. Each fresh proposal needs one target density
row. Including initialization, each arm therefore uses exactly13,560 target
density rows per complete population:120*(1+16*7)=113*(8+16*7).
The assertion on actual counts must pass; Python calls and wall time are also
reported, so density-row savings cannot be mistaken for measured GPU speed.

Run128 independent populations per target/arm. Seeds are2026103000+i for
Gaussian standard,2026104000+i for Gaussian waste-free,2026105000+i for mixture
standard, and2026106000+i for mixture waste-free, i=0..127. Observables are
(x0,x1,x0^2,x1^2,1[x0>0]), with analytic moments and variances. Compare pooled
sum Zhat fhat/sum Zhat, and test mean Zhat and all Zhat fhat against their exact
integrals using a fixed5 standard-error diagnostic (not a guarantee).

The primary empirical metric is mean over observables of the between-population
ratio influence variance divided by the analytic posterior observable variance.
At equal density-row cost, waste-free/standard below0.8 is a useful candidate
gain, above1.25 is a regression, otherwise inconclusive. Report a95% percentile
interval from1,000 independent whole-population bootstrap draws, seed2026107000
for Gaussian and2026107001 for mixture. No path point is a replicate. Preserve
per-observable regressions, normalizer variance, wall time, and all failures.
The scientific dev_score is the geometric mean of standard/waste-free primary
metrics across the two targets, explicitly a small synthetic development result.

## Real-scene compatibility boundary

Only NumPy particle arrays are supported here. A scene integration must retain
every native payload, nuisance value, log density, observable cache, original
ancestor, and optional environment tilt at each path position, including
rejections; copying target coordinates alone is invalid. Extinction must retain
the original population denominator and cannot trigger replacement. The M-path
batch shape changes density evaluation batches: existing fixed four-observation
groups and numerical replay constraints still apply. Extra retained states
increase memory and diagnostic work. Fixed block composites may have different
costs; counted density rows here do not establish real-scene wall-time savings.
The current SummaryLikelihood pads all density evaluations to4096 rows. Smaller
M therefore does not proportionally shrink GPU work. Existing V100 measurements
of1024 versus4096 active rows show only1.23–2.16 throughput ratios despite the
shared4096 density shape; the full physical/environment/exchange paths have
different costs. These are compatibility observations supplied by the parent,
not a new benchmark or permission to change batching. Fewer independently
resampled parents may also lose modes even when the retained path count is large.
