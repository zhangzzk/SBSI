# Mock versus measured-image inference review — October 29, 2026

## Conclusion and scope

These runs share the inference framework, but they are not an experiment in
which only the observations changed. The mock result used the retained F8,
rotated pairs, mixed chain settings and mixed shear-score evaluations. The
measurement result used symmetric F8, an unpaired image catalogue and the
uniform co-moving recipe, with a later cache repair. The full mock truth test
of that recipe was deferred. Neither run established zero additive bias.

The compared endpoints are:

- Mock: `galsbi_pair1m_20261012/debug/step2_result_brightw8_fl32.json`, the
  October 25 aggregate after the bright-w8 and faint-long follow-ups.
- Measurement: `galsbi_constgold1m_signsym_20261028/recovery_v1/c2/estimate.json`,
  the completed October 29 million-observation symmetric-flow run.

Both live under `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior`.
This review checks frozen protocols, source, saved results and summary arrays.
It does not regenerate observations, rerun chains, repeat bootstraps, or
independently prove posterior convergence. The earlier 100k measured run and
initial October 12 mock are provenance, not substitutes for these endpoints.

## Outcomes and pairing

All errors below are one sigma; m uses `100*(g1/0.02-1)`, assuming c1=0.
Only one injected shear `(0.02,0)` is available, so m1 and c1 are not separately
identified, and m2 is not measured. See
[bias convention](CONVENTIONS.md#7-multiplicative-bias) and
[inference-specific definitions](SYMMETRIC_FLOW_INFERENCE.md#completed-million-observation-result--october-29-2026).

| Endpoint | Selected observations | m1 (%) ± percentage points | c2 | c2 / sigma |
|---|---:|---:|---:|---:|
| Mock, paired | 1,990,072 | −0.4876 ±0.7409 | 0.00033189 ±0.00016430 | 2.02 |
| Mock, original side | 995,015 | −5.4154 ±2.2179 | 0.00005990 ±0.00046334 | 0.13 |
| Mock, rotated twin | 995,057 | +4.4247 ±2.2241 | 0.00059662 ±0.00046048 | 1.30 |
| Measured images, unpaired | 999,285 | −2.8639 ±2.2373 | 0.00118408 ±0.00044399 | 2.67 |

The mock pair contains 1,010,119 distinct generating scene IDs. The twin
rotates scenes by 90 degrees and couples draws antithetically, with its own
selection. Pairing cancels much of the shape noise. Its two sides are correlated
and are not independent replications. Their individual m results illustrate
why an unpaired million-galaxy result can fluctuate by several percentage
points even when the paired result is near zero. Per-side bright corrections
also use the shared sampled shear; they are diagnostic outputs, not separate
complete reruns of the measured pipeline.

The mock's declared gate required both |m| and |c2| below 2 sigma. Its final
c2=2.020 sigma narrowly fails that gate. Describing it as an entirely unbiased
mock test was too strong. The measurement m is 1.28 sigma below zero; its c2
is the more concerning residual. Different models, populations, pairing and
error prescriptions prevent treating the difference as a controlled estimate
of image-model mismatch.

## What is unchanged

Eight core source files match byte-for-byte between `snapshot_d776abb`, used
by the mock follow-ups, and the recovered measurement snapshot:
`galsbi_prior.py`, `galsbi_scene_prior.py`, `galsbi_target_coordinates.py`,
`mcmc_proposals.py`, `pseudo_marginal.py`, `catalogue_disk_likelihood.py`,
`catalogue_likelihood.py`, and `catalogue_disk_response.py` under `sbsi/`.
Hashes are in the [audit receipt](evidence/audit_mock_measurement_20261029.json).

Both retain the F24 row0 inference prior, native i14–28 support, uniform
neighbours, four chains, density batches of 4096, disk Möbius transport,
the same measured quantities and observing conditions, and a ±0.002 shear
stencil. The model-generating numerical target prior and inference's analytic
native target density retain their documented approximation distinction;
this is not a proof of exact prior-law closure.

The C8 classifier hash is
`231bbbdf1f885ab22ccfca2525ae46b6bdfd068474c42ebac24973feb861cb6a`;
the disk model hash is
`194c1d11b67d6fd52039060270663e8e1cd610b478b9e9a66e34271701761fe7`
in both configurations. Shared point-estimate algebra is
`I = sum(s_i s_i^T - E_i)` followed by `g = I^-1 sum(t_i)`, with a second
evaluation for mag<21 and the original C1 information retained.

## Material differences

### Data and flow

Mock observations were generated through the retained V3.9 forward model on
GalSBI scenes. Measurements are from rendered ConstGold plus-leg images,
cases 40–139, with actual detection, crossmatching and shape measurement.
The inference prior is unchanged, but agreement between that prior and the
image population is not established by the model-generated mock test.

Mock F8 hash:
`bd51dbbbeeb72763f6982a958cb8f3a0b2d3906d4f78935d9d959fa85a17797e`.
Measured-run symmetric F8 hash:
`21f772cfb87339292f9ef1df350d52fa44ed8cd9b3715068623051acb3737591`.
This is a retrained joint measurement density, not merely a new sampler flag.
The flow loader changes support the new architecture. `galsbi_summary.py`
also gained optional compilation, but the production constructor does not
enable it. No whole-likelihood symmetry guarantee follows from F8 symmetry
alone: C8 and the response model remain unchanged.

Both use measured radius>3 pixels, magnitude<25.8, radius<100 pixels and
|e|<0.99, without truth-property cuts. The measurement source selected
18,157,963 eligible rows; sampling produced 1M raw rows, then 715 failed the
ellipticity cut. No newly unmatched rows or duplicate identities were
introduced by the extension. Upstream source accounting records 322,297
shape rows without crossmatches, zero crossmatches without shapes, zero
matched rows without truth, and 2301 invalid/unusable matched rows. Those
pre-sample exclusions are distinct from omitting a selected inference row.

### Chain lengths, starts and shear-score evaluation

| Component | Mock endpoint actually used | Measurement endpoint |
|---|---|---|
| C1 mag<23 | Original tempered 8192/2048-step chains, measurement-centred starts | Tempered 20480/8192, likelihood-weighted prior-draw starts |
| C1 mag 23–24 | 32768/8192, prior-draw; 320-row exception below | 20480/8192, prior-draw |
| C1 mag 24–25 | 16384/4096, prior-draw | 20480/8192, prior-draw |
| C1 mag 25–25.8 | 4096/1024, prior-draw | 20480/8192, prior-draw |
| C2 mag<21 | Tempered 14336/8192, prior-draw, at (0.01936,0.00037) | Tempered 20480/8192, prior-draw, at (0.01886,0.00122) |
| Shear scores | Bright: offline `comoving_circ`; faint: fixed-intrinsic-shape ratios | In-run co-moving ratios for every galaxy |

Counts denote total/warmup steps. The table describes the actual aggregate,
not the longer settings proposed for a future full run. The co-moving ratio
changes latent coordinates with shear and includes the prior ratio and
Jacobian; it is not just moving the same calculation into the loop. The
mathematical change of variables targets the same integral, but finite-chain
estimates and numerical implementations can differ.

The October 25 simplification passed short bit-identical replay tests.
Stage B also reproduced 549/549 old chains and passed the bright recipe
comparison. However, the all-band scaled change in score evaluation was
only constrained to Δm=−6.09 ±4.33 percentage points and
Δc2=(6.97 ±9.14)e−4, dominated by the heavily extrapolated faintest subset.
These are noisy comparison estimates, not measured biases against truth.
They do not clear a sub-percent effect. The full Stage C truth test was
explicitly deferred in `debug/rule_one.md`, amendment 2. It was not completed
with either the uniform retained-flow recipe or the new symmetric flow.

### Selection normalization and uncertainty

The mock pooled eight replicas, 6400 chunks of 8192 scenes: 52,428,800 scenes,
with 16 conditional flow draws per scene. The measurement run recomputed
normalization for the symmetric flow using two replicas, 1222 chunks:
10,010,624 scenes, also with 16 draws. This is appropriate to a changed
joint flow; the old retained-F8 normalization was not silently reused.

The shear selection-error component changes from
`(6.8622e-5,7.1535e-5)` on the mock pair to
`(1.5458e-4,1.5225e-4)` on measurements. It is larger but does not dominate
the measured errors. A smaller budget is not, by itself, evidence of bias.

The mock uses covariance and bootstrap over generating scenes, preserving
the original/twin dependence; its reported error is the total covariance
error. The measurement code uses the unpaired information-based statistical
term and quotes the maximum of total, galaxy-bootstrap and case-bootstrap
errors. Its total covariance wins for both components. Thus even the error
calculation is not identical, although both propagate chain and selection
components.

### Cache failures and convergence

The cache issue predates the symmetric flow. The original October 12 mock
had a twin unit failing the invariant replay check, followed by a full-
likelihood rerun without reuse. Later long-chain unit `L_original_u0394`
failed twice (maximum log-kernel difference 1.056e−4). Its 320 galaxies were
excluded from the long-chain comparison but **retained in the final mock
aggregate with their earlier 4096-step faintdraw values**. This review checked
the stored arrays: all 320 survived the cuts, and their L/h1/h2/ndraw entries
equal the earlier arrays exactly. Calling them simply “excluded” obscures
that retained-short-chain fallback.

The current measurement C1 contains 1043 cached units plus two fully
evaluated repairs covering 1660 galaxies. All 311 C2 units disable nuisance
likelihood reuse. The strict retained-state replay checks passed; none of
the selected galaxies was omitted. This fixes that numerical shortcut,
but does not establish convergence or validate every old cached trajectory.

Log-Z R-hat>=1.1 remains on 8.26% of measured C1 galaxies and 30.44% of
bright C2 galaxies. Chain-half variance measures variability; it cannot
guarantee absence of shared nonconvergence bias. The mock also retained
short bright C1 and faintest-band settings. Neither result is a fully
converged gold-standard control.

## Independent checks performed in this review

CPU audit job **17599211** completed 0:0 in 14 seconds using
[the saved audit script](evidence/audit_mock_measurement_20261029.py).

- Recomputed the mock paired C1 information and estimate from both saved
  per-galaxy summary arrays and all 6400 selection chunks, recovering
  1,990,072 observations. Applying its saved bright sums reproduced final
  g to maximum absolute error 3.02e−16.
- Recomputed measured final g and its total error from `c2/pergal.npz`;
  final g agreed to 2.17e−19, total errors within 1e−12, and all 999,285 IDs
  occurred exactly once.
- Verified the old 320-row fallback and eight identical core source files.
- Recorded hashes of the result files, reduction code, deferred-test rule,
  subset-comparison report, cache-repair source and recovery rule.

The [receipt](evidence/audit_mock_measurement_20261029.json) is reproducible
evidence for output consistency, not a new scientific acceptance run.

## Recommended next controlled check

The missing control is model-generated data from the **current symmetric
forward model**, analysed with the **exact current full-likelihood sampler,
co-moving score evaluation, cuts and estimator**, with the generator/inference
prior-law distinction declared explicitly. Reusing old asymmetric-model
mock observations would introduce model mismatch rather than test closure.
Use a predeclared paired setup for precision, then a prespecified longer-chain
comparison on the bright and faint problem populations. If current-model
closure passes, the image residual can be investigated as population or
measurement-model mismatch with much less ambiguity.

This review did not launch that study or change any scientific settings.
