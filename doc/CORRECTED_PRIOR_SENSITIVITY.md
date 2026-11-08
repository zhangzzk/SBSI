# Corrected-v10 numerical target-prior sensitivity

## Result and interpretation

On all 24 corrected v10 development populations, replacing the analytic target
intensity by the full numerical native intensity on the represented coordinate
support changes every raw or frozen phase-controlled score by less than
0.006 of its original whole-population Monte Carlo standard error (MCSE).
The maximum is 0.00596526 under the sampled F32 measure and 0.00596452 under
the F64 importance-reweighted measure, both large / observation 58165 /
controlled g2. This is a small conditional sensitivity, not a bound on
numerical-only coordinate support, missed modes, global additive bias, or
shear calibration. The sampler and its acceptance gates are unchanged.

## Predeclared diagnostic

The pre-evaluation design remains frozen in `source_v5/doc/CORRECTED_PRIOR_SENSITIVITY.md`,
identified by the [validation receipt](evidence/corrected_prior_sensitivity_20261022.json).
The fixed populations are `dev_v10_{small,large,broad}_{0..7}`, observations
58165, 0, 1, 8. Small has 256 particles per observation; large and broad have
1024. There are 73,728 saved target states, eight independent populations per
arm, and no excluded population. The authoritative input is
`v10_corrected_comparison_20261022_v3.json`, SHA256
`09601af0a43c0b5801bc0467b090e588c745713a3c9fd6d86b58f2f9172d1574`.
Its complete source/input map and original `source_support_fix_20261022`
manifest were checked before and after the audit. No heldout data, new draws,
likelihood model evaluation, control fits, or production edits were used.

For each saved coordinate x, evaluate both native types independently and use
`log r = logsumexp_t(log q_N(t,x)) - logsumexp_t(log p_A(t,x))`.
The common coordinate Jacobian cancels in this ratio but must reproduce the
saved analytic mixture prior. Native-to-coordinate replay is also required.
Numerical intensities are evaluated without an analytic-support mask:
numerical-only **type** mass at represented x is included. The restricted
exchange helper is cross-checked on shared support. Numerical-only
**coordinate** support absent from the saved analytic posterior cannot be
reconstructed by this diagnostic.

## Observables, measures and paired uncertainty

All particles from the audited phase diagnostics and the original n15 frozen
coefficients are retained. Columns are T, raw g1/g2, controlled g1/g2. Under
the recorded F32 measure the base point weight is 1; under the F64 measure it
is `exp(log K64 - saved log K32)`. The prior tilt multiplies either weight
by r. The raw scores are the retained F64 derivative observable evaluated
under each indicated measure. Both the saved-score raw report and the full
phase report are reproduced before interpreting any sensitivity.

For population j with original size N_j, preserve `Z_j mean(w)` and
`Z_j mean(w f)` with denominator N_j. Combine the eight complete populations
as a ratio of sums. An individual all-zero numerical population keeps zero
mass and remains in R = 8. An all-zero observation is explicitly undefined;
unknown numerical failures never become zero weights or dropped populations.

Let W_j^A and W_j^N be the two estimators' normalized population weights and
m_j^A, m_j^N their within-population moments. The paired contrast influence is
`I_j = R W_j^N(m_j^N-mu_N) - R W_j^A(m_j^A-mu_A)`.
The reported covariance is `sum_j I_j I_j^T / [R(R-1)]`. It includes the
same-population covariance and each estimator's denominator. The full output
retains both influence arrays, cross covariance, per-population masses and
moments, and paired contrast MCSE. Errors are conditional on the frozen
pilot coefficients; independent-particle errors do not apply.

## Score sensitivity by arm and observation

Entries are the maximum absolute g1/g2 change divided by the corresponding
analytic-estimator MCSE within the indicated raw or controlled pair.
They are descriptive sensitivity ratios, not new acceptance thresholds.

| Arm | Observation | F32 raw | F32 controlled | F64 raw | F64 controlled |
|---|---:|---:|---:|---:|---:|
| small | 58165 | 0.002446726 | 0.0003416035 | 0.002445923 | 0.0003410487 |
| small | 0 | 0.001963064 | 0.002120582 | 0.001962344 | 0.002119727 |
| small | 1 | 0.000645917 | 0.000509409 | 0.0006459179 | 0.0005094292 |
| small | 8 | 0.0002411657 | 0.0001346691 | 0.0002411663 | 0.0001346718 |
| large | 58165 | 0.003610568 | 0.005965262 | 0.003609601 | 0.005964518 |
| large | 0 | 0.001302452 | 0.001189632 | 0.001302512 | 0.001189691 |
| large | 1 | 0.0002985935 | 0.0002530174 | 0.0002985995 | 0.0002530187 |
| large | 8 | 0.0003477928 | 0.0005075385 | 0.0003477906 | 0.0005075396 |
| broad | 58165 | 0.003305201 | 0.004731464 | 0.00330545 | 0.004732832 |
| broad | 0 | 0.0002295594 | 0.0004881019 | 0.0002295729 | 0.0004880358 |
| broad | 1 | 0.0005521011 | 0.000544461 | 0.0005520959 | 0.0005444514 |
| broad | 8 | 0.0002889311 | 0.0003036311 | 0.0002889372 | 0.0003036339 |

The largest scaled change, large / 58165 controlled g2, is
−0.0129611 ± 0.0113991 paired MCSE under F64; its original analytic MCSE is
2.17303. Under F32 it is −0.0129624 ± 0.0113984. For observation 0, the large
F64 raw-g2 change is −0.000422512 ± 0.000187952, only 0.00130251 of its original
MCSE. Thus small sensitivity is not a claim that every contrast is exactly
zero or statistically indistinguishable from zero at the paired precision.
These values are score changes, not shear estimates or additive calibrations.

## Density support, mass and concentration

Across all 73,728 states, log r ranges from −0.0644180058241 to
0.0306258605579, corresponding to r in [0.937612989915, 1.03109965666].
No numerical-zero state, numerical-only type, or analytic-only type occurred
in these saved states. The restricted helper therefore happens to reproduce
the full type sum here: the maximum full-minus-restricted log-r difference
is 1.07e−14. The implementation still includes numerical-only type support;
its meaningful unit fixture verifies a ratio of 5.5 where the restricted
helper would produce 2.0.

The numerical/analytic **intensity-mass** ratios range from 0.999997250361 to
1.00018257870 under F32 and 0.999997250354 to 1.00018258092 under F64.
They are not normalized model-evidence ratios. The largest ratio is small /
58165: 1.00018258092 ± 0.00018787972 paired MCSE under F64. Each mass ratio
uses the original population normalizers and particle denominators.

Under F64, numerical population-weight concentration ESS ranges from
7.67424 to 7.99449 out of eight populations. Point-weight concentration ESS
ranges from 2023.43 to 2046.59 out of 2048 points for small, from 8172.07 to
8186.21 out of 8192 for large, and from 7858.07 to 8179.72 for broad. The
largest single normalized point weight is 0.000605622 (small). These describe
weight concentration, not effective independent particle counts or mixing.
All per-run masses, means, zeros, concentration diagnostics and pointwise
type intensities are preserved in the output artifacts.

## Replay and provenance checks

The maximum discrepancy against the audited raw and phase baseline summaries
is 3.55e−15. Native non-phase coordinate replay differs by at most 7.94e−13;
periodic phase replay is exact. Every saved analytic prior, including its
coordinate Jacobian, replays exactly. The full numerical intensity and
restricted helper agree on shared type support within 1.70e−12 in log density.
All original 466 audited source/input identities and 554 native package,
photometry, ABC-posterior, cosmology-cache and compiled-wrapper inputs are
recorded. Constructed prior tables are separately fingerprinted. All inputs
and diagnostic source files passed the final unchanged check.

Final source snapshot:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/corrected_prior_sensitivity_20261022/source_v5`
(manifest SHA256 `7fd05890419f38e9ca4329c0e5afcd77f9b55e63827670332ee9c80e44d4a2df`).
The [validation receipt](evidence/corrected_prior_sensitivity_20261022.json)
records exact implementation bytes, job outputs, skipped tests, all preserved
failed attempts and output hashes. Full result:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/corrected_prior_sensitivity_20261022/audit_v2/result.json`
(SHA256 `9d59a54255a2c5f8c2248f663aa430a2e5e2ccf9f30f89c125ec9e4f7ec86598`).
The adjacent `prior_ratios.npz` holds all pointwise ratios and both type
intensities. Original population, normalizer and phase artifacts were retained.

## Validation, failures and cost

The focused mathematical/native-density suite passed 40 tests in job
17437469. Final full-suite job 17437783 passed 1018 tests, with nine skips
and one existing deprecation warning, in 73.95 s. The skips are the optional
BlendEMU converter and eight optional frozen historical replay fixtures;
they are enumerated in the receipt. Static `ruff` F821/F823 checks passed.
Final CPU audit 17437804 completed successfully in 85 allocated seconds
(65.9903 s process time), using two CPUs and no GPU. The launcher checked
its frozen manifest both before and after execution.

Two earlier full-suite attempts failed only because snapshot packaging
omitted test source fixtures: job 17437595 lacked the historical wrapper,
and job 17437644 lacked `jobs/job_inference.sh`. Job 17437696 then passed
1018 tests. Audit attempt 17437709 failed before prior evaluation because a
redundant function-local `import importlib` shadowed its earlier use. Removing
that redundant import was the sole diagnostic-code repair; the data, recipe
and gates stayed fixed. The failed audit receipt remains in `audit_v1`.

All seven scheduled jobs, including failures, used 464 allocated wall seconds
and 928 CPU-allocated seconds in total, with zero GPU-allocated seconds.
These are scheduler allocation-wall upper bounds, not measured compute busy
time. No further numerical run is needed for this diagnostic.

## Limits and next implication

The continuous numerical density excludes singular clamp atoms and finite-table
failed draws; it does not represent the literal finite-RNG law everywhere.
The prior ratio is phase-independent, so the intended continuous F64
zero-mean phase identity also holds after this tilt on analytic coordinate
support. F32 controls retain their numerical qualification.

This result weakens the explanation that the analytic-versus-numerical target
prior mismatch dominates the corrected v10 development score uncertainty,
including its lower-noise controlled estimates, **on the represented states**.
It cannot validate unseen coordinates, missed posterior modes, other objects,
posterior calibration or a global c correction. The current sampler still
targets the analytic target prior with its numerical environment law. No
posterior target or empirical correction is promoted by this sensitivity check.
