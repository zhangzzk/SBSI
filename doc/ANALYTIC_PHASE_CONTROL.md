# N19 analytic state-dependent phase control

## Predeclared candidate and scope

Before inspecting any new outcome, fix epsilon = **0.1**, r² = e1²+e2²,
F = (1+r²)/(r²+0.01), a = (-F e2, F e1), a_prime = (-F e1, -F e2).
The two-vector control is C = a T + a_prime, namely
C1 = -F(e2 T + e1), C2 = F(e1 T - e2). The candidate score is **S - C**.
No coefficient fit, empirical recentering, regularizer search or sign choice
is permitted after results. This is Arbor n19 under n15, not a sampler or
heldout change. The external declaration is
`analytic_phase_n19_20261022_declaration.json` in the sampler-search data root.

Use only the four v1 pilots and eight each of v5/v6v100/v7, their successful
phase-v2 arrays and saved intrinsic summaries. Keep every particle, stride 1,
original Zhat and full N denominators. Compare original score, the existing
frozen constant control, and this fixed analytic control. The constant-control
comparison on its own v1 pilot data is explicitly in-sample/descriptive.
Raw F32-sampled and exp(Delta logK)-reweighted F64 measures stay separate;
no exact zero expectation is asserted for the rounded F32 sampling program.
All original score/normalizer gates remain unchanged. CPU Slurm only: no
new model forward pass, GPU, sampling, heldout input or higher-effort-arm input.

## Derivation to verify against actual code

At fixed radius and other phase-invariant latent variables, phase has a
uniform native prior, and T = derivative_psi log K. Since a and K are
periodic, integral (a T + derivative_psi a) K dpsi = integral derivative_psi
(a K) dpsi = 0. F depends only on radius, so derivative_psi a = (-F e1,-F e2).
The epsilon regularizer makes a and a_prime continuous and zero at e=0;
no division by radius or phase convention at the origin is required.

The actual reduced-shear map is e'=(e+g)/(1+conjugate(g)e), with real spin-2
components. Its g=0 Jacobian has rows
[1-e1²+e2², -2e1e2] and [-2e1e2, 1+e1²-e2²]. Dotting each shear column with
(-e2,e1)/r² gives the unregularized phase tangent
(1+r²)(-e2,e1)/r². The chosen coefficient shrinks it by r²/(r²+0.01).
This motivation is specific to the intrinsic shear map at g=0; the full
score also has observed-shape response/transport terms. The zero-expectation
periodic product-rule identity does not require claiming that C removes the
entire shear score, or that the same tangent expression holds at nonzero g.

The planned independent tests check this Jacobian and its radial/tangential
decomposition against finite differences of `apply_shear_to_ellipticity`,
nonuniform periodic integration, product-rule signs, origin/wrap behaviour,
and an intentionally missing a_prime term. Evaluation follows only after
those tests, with frozen code/source/input hashes and no outcome selection.

## Correctness evidence before saved-state evaluation

CPU Slurm 17421399 on cluster passed **12 tests in 4.17 seconds**. They
integrate both control components to zero for nonuniform periodic targets
at three radii and two phase centres, prove the missing-product-term variant
has nonzero mean, check the exact g=0 shear map/Jacobian and radial/tangential
score decomposition, and cover origin, phase wrap, component signs, fixed
regularization, invalid-domain rejection and unequal-N/Z precision weighting.
Raw-measure metadata explicitly leaves the known expectation unspecified;
only the separately qualified F64 continuous-model diagnostic references zero.
Cip test job 17421396 was cancelled during a queue-switch/startup race before
any test output, then rerun unchanged on cluster; no outcome selection.

Code confirms native phase is uniform and the retained summaries change only
intrinsic e1/e2 under phase rotation. Crowding uses fluxes/distances; retained
response pairs use magnitudes, semimajor radii, Sersic indices and separation,
not orientation. Therefore fixed-radius a and a_prime introduce no omitted
phase derivative through response or other native latent factors. The saved
comparison protocol uses g=(0,0), so the tangent motivation applies directly.

## Saved-state result: mixed variance benefit, discrepancy remains

Cluster CPU job **17421436 completed in 10 seconds**, exit 0. It processed
all **28 populations / 53,248 saved states** without a new likelihood/model
forward. Source commit `cdb6fbe5de67a9334ac500250306e46fb7d6a1db`; manifest
SHA256 `89359c8d11bffb3a99318dcb4623708bde242ca63a0be4fe744d5725cbcd1cfd`.
The pre-execution receipt `analytic_phase_n19_20261022_experiment.json` pins
all 141 allowed input files and the earlier fixed-formula declaration.

Outputs under `analytic_phase_n19_20261022/` preserve all raw scores, constant
CV scores, analytic scores, a, a_prime, C, T and Delta logK in 28 NPZ files.
The result JSON carries output hashes, full moment covariance/MCSE matrices
and both numerical measures. Result SHA256:
`1562f25c3aea45394359d3bd2a5eb3a82f20f7125725341cd4d86a3a0ada525f`.
`comparisons.csv` is a direct tabulation of every signed mean/MCSE under both
measures; its SHA256 is `152b2a5f740974980ab47f0556a69493e427dac0fdb152d00cdec2ae53484cfd`.

### Per-object MCSE comparison

These are score MCSEs from complete independent populations. V1 has four
runs and its constant CV comparison is in-sample/descriptive; the three
evaluation groups each have eight runs. Tables show the original F32-sampled
measure; explicit F64-reweighted values remain in JSON/CSV.

| Group | Observation | g1 MCSE raw / constant / analytic | g2 MCSE raw / constant / analytic |
|---|---:|---:|---:|
| v1* | 58165 | 32.3221 / 42.8762 / 44.4162 | 80.7993 / 68.3082 / 73.1043 |
| v1* | 0 | 3.75839 / 0.105303 / 0.292954 | 1.51406 / 1.48836 / 1.64073 |
| v1* | 1 | 0.65585 / 0.377362 / 0.327399 | 0.317853 / 0.395744 / 0.432225 |
| v1* | 8 | 0.0644447 / 0.0597086 / 0.0820641 | 0.0281543 / 0.0240243 / 0.123629 |
| v5 | 58165 | 4.85112 / 1.48194 / 1.73873 | 7.27781 / 2.49317 / 3.83152 |
| v5 | 0 | 1.41585 / 0.18701 / 0.15078 | 0.673466 / 0.646936 / 0.612784 |
| v5 | 1 | 0.379691 / 0.25041 / 0.243608 | 0.395603 / 0.345935 / 0.322019 |
| v5 | 8 | 0.0951232 / 0.0926066 / 0.0987982 | 0.0400846 / 0.0376644 / 0.0694672 |
| v6v100 | 58165 | 1.48456 / 1.12123 / 1.15427 | 2.80268 / 1.79517 / 2.10922 |
| v6v100 | 0 | 0.566552 / 0.0952629 / 0.0785458 | 0.412437 / 0.416842 / 0.444922 |
| v6v100 | 1 | 0.108209 / 0.104532 / 0.0811203 | 0.142721 / 0.128204 / 0.124325 |
| v6v100 | 8 | 0.0354237 / 0.0364059 / 0.0324717 | 0.0262694 / 0.0231416 / 0.0410568 |
| v7 | 58165 | 4.61161 / 2.90446 / 2.8521 | 4.94611 / 4.76248 / 4.6296 |
| v7 | 0 | 1.16755 / 0.111169 / 0.0693418 | 1.21847 / 1.17721 / 1.11439 |
| v7 | 1 | 0.241836 / 0.190462 / 0.162431 | 0.547281 / 0.352599 / 0.360493 |
| v7 | 8 | 0.0688931 / 0.0667148 / 0.0832694 | 0.0366705 / 0.0377753 / 0.0878737 |

### Signed score means

| Group | Observation | g1 mean raw / constant / analytic | g2 mean raw / constant / analytic |
|---|---:|---:|---:|
| v1* | 58165 | 16.64569 / 30.31706 / 31.51105 | 65.8439 / 51.37522 / 56.29304 |
| v1* | 0 | -2.356125 / 0.4750123 / 0.05440267 | -3.84756 / -3.680766 / -3.605331 |
| v1* | 1 | -3.711229 / -3.416661 / -3.407045 | -4.277356 / -4.456863 / -4.450406 |
| v1* | 8 | -1.731523 / -1.764465 / -1.856783 | 1.147967 / 1.102321 / 1.190962 |
| v5 | 58165 | -3.777199 / 5.172865 / 6.076542 | 17.71548 / 8.243451 / 11.46214 |
| v5 | 0 | -0.9099996 / 0.4786842 / 0.3930113 | -6.176905 / -6.095092 / -5.848983 |
| v5 | 1 | -3.18869 / -2.986423 / -2.929468 | -3.647668 / -3.770927 / -3.794436 |
| v5 | 8 | -1.839443 / -1.811128 / -1.813012 | 1.090917 / 1.130151 / 1.216339 |
| v6v100 | 58165 | 7.254081 / 7.283441 / 7.283135 | 11.70315 / 11.67207 / 11.63607 |
| v6v100 | 0 | 0.4834533 / 0.4046878 / 0.4717932 | -5.330297 / -5.334937 / -5.342597 |
| v6v100 | 1 | -2.968628 / -2.921802 / -2.957487 | -3.651751 / -3.680286 / -3.662249 |
| v6v100 | 8 | -1.980711 / -1.963328 / -1.945256 | 1.226141 / 1.250228 / 1.270094 |
| v7 | 58165 | 1.736343 / 2.303558 / 2.446897 | 4.95466 / 4.354365 / 4.482165 |
| v7 | 0 | -0.996627 / 1.069976 / 0.8421148 | -8.172525 / -8.050772 / -8.016561 |
| v7 | 1 | -2.44774 / -2.853435 / -3.071054 | -4.326681 / -4.079456 / -3.952662 |
| v7 | 8 | -1.854643 / -1.836467 / -1.87403 | 1.232366 / 1.257551 / 1.159522 |

### What the fixed candidate improves and worsens

Across the 24 evaluation group/object/component combinations, estimated
analytic MCSE is lower than raw in **18/24**, and lower than the existing
constant CV in **13/24**. These counts are descriptive development evidence,
not significance tests, an optimized selector or an adoption gate. The
formula and epsilon stayed fixed and every negative result is retained.

Observation 0 g1 improves beyond the constant control in all three groups:
v5 0.18701 to 0.15078, v6 0.095263 to 0.078546, v7 0.111169 to 0.069342.
Analytic means are **0.393011**, **0.471793**, **0.842115**, respectively.
Thus v7−v6 remains **0.370322 ± 0.104775**, or **3.534 combined reported
MCSEs**. The constant-control contrast was 0.665288 ± 0.146402 (4.544 MCSEs).
The analytic candidate does not eliminate the disagreement.

The bright object improves versus raw scores in the evaluation groups but
is worse than the constant CV in both components for v5 and v6; its v7
improvements versus the constant CV are small. The faint observation 8 g2
worsens consistently: raw MCSE [0.04008, 0.02627, 0.03667] becomes analytic
[0.06947, 0.04106, 0.08787] for v5/v6/v7. V7 faint g2 is 2.40 times noisier
than raw and 2.33 times noisier than the constant CV. This has a direct
mathematical possibility: for a nearly phase-flat likelihood, T is small
but a_prime can still fluctuate; the product-rule control is unbiased for
the continuous conditional phase target but need not reduce variance. This
observation is an explanation of a possible mechanism, not a causal fit to
these saved populations. No epsilon/phase/brightness adaptation was tried.

### Numerical measure and correctness limits

The maximum shift of any evaluated raw/constant/analytic score mean under
explicit F64 reweighting is **0.000790 of its corresponding raw reported
MCSE**. The largest F64 analytic-control mean/MCSE magnitude is **2.374**
(v5, observation 0, g2); this is descriptive finite-eight-population evidence.
Raw F32 metadata leaves the known control expectation unspecified. The
F64 zero reference expresses the intended continuous periodic identity,
not an exact identity for rounded arithmetic or a global precision bound.

No production score, sampler, heldout result or acceptance gate changed.
A separate endpoint-support audit is pending; these old saved-population
comparisons cannot prove universal target invariance of their samplers.
This node establishes the analytic product-rule implementation and mixed
development variance effects. Full coverage, unbiased finite-particle ratios
and calibrated shear remain unestablished. No blanket replacement of the
constant control or raw score is supported by this experiment.
