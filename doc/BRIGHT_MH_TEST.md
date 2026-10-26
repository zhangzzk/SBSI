# Bright continuous-prior MH and shear recovery

## Completed result — October 2, 2026

Both replicas16791908_0/_1 completed0:0 in1h35m10s/1h35m09s; CPU report16791910
completed0:0 in19m13s. All numerical replay checks passed, but neither cohort
passed convergence or recovery. No jobs remain active at this status check.

| Measured cut | Full convergence checks passing at32k | Both local shear-ratio checks passing | Exploratory conditional g1 |
| --- | ---: | ---: | ---: |
| <20 | 0/128 | 35/128 | 0.01410259 |
| <22 | 0/128 | 10/128 | 0.01077937 |

Injection isg1=.02,g2=0. These fits are unvalidated, not measurements of shear
bias. Local ratio checks useg1=.018,.022 and all eight independent chains;
they demonstrate that the failures are not confined to irrelevant latent
coordinates. Median ratio Rhats are1.0268/1.0232 for<20 and1.0762/1.0718 for<22.
Checks over the broader fitted-profile/interval nodes pass only19/128 and6/128.
The latter depend on the exploratory fit and its approximate interval; they
are diagnostic counts, not separate population acceptance claims.

Independent replica fits are(.0158213,.0123065) for<20 and
(.0133733,.0115206) for<22. Conditional propagated MH shear MCSE is approximately
7.66e-4 and1.48e-3, versus selection MCSE2.75e-5 and1.99e-5. Selection ladders
pass their declared screen, so the measured numerical problem is dominated by
posterior sampling/reweighting rather than this normalization diagnostic.
These MCSEs themselves assume adequate exploration and can miss shared bias.

The saved report combines a local quadratic peak with piecewise-linear interval
crossings on a relatively coarse grid. Some individual-replica intervals are
missing or even exclude the interpolated peak; do not interpret those intervals
or the reported curvature SEs as validated uncertainty. Recovery is already
rejected by the convergence and overlap gates. Any future accepted recovery
needs consistent profile interpolation/refinement as well as better exploration.
Original report receipts remain unchanged. The subsequent
[deep convergence audit](MH_CONVERGENCE_AUDIT.md#finding--october-2-2026) fixes
the interpolation helper, saves corrected intervals separately, and diagnoses
curved proposal geometry plus inadequate wider-shear importance overlap.
Neither scientific failure is resolved by the reporting fix.

The32k sampler processes took5363.13/5357.87s, or22.53/22.51 GPU seconds per
distinct galaxy per replica; the full allocated workflows, including wider
profiles and selection, took about24.0 GPU seconds per galaxy. Both cuts share
18 observations, so do not add their counts when computing throughput.
The separate representative1024-object16k benchmark16788845 also completed:
12355.51s, or12.0659 GPU seconds per galaxy per replica, above the10-second
target. Its throughput does not establish sampling accuracy.

Conclusion: the current32k continuous-MH method is not yet a validated bright
fallback for the proposed hybrid. No automatic longer run or20k launch follows.

## Owner request and scope — October 2, 2026

The owner requested MH tests on bright observations, measured magnitude<20
and<22, assessing convergence and shear recovery. This implements a bounded
GalSBI/F8/C8/original-disk model-closure test. It does not use the FS2 atom prior,
change the frozen models, or resume the paused20k population experiment.

First recovery is conditional in g1, with g2 fixed to its injected value0.
Injection is(g1,g2)=(0.02,0). The test reports likelihood intervals and numerical
uncertainty;128 galaxies per cut cannot establish sub-percent calibration bias.
Native-prior versus numerical-generator approximation limits remain those in
[continuous sampling](CONTINUOUS_SAMPLING.md#a-simple-count-changing-move-partial-poisson-refresh).

## Observations and independent replicas

Generate fresh scenes and measurements with seed2026100501. Preserve independent
C8 Bernoulli usability, forward F8/disk measurements, measured radius>3pixels
(0.6arcsec), and the base measured magnitude<25.8 selection. From the retained
generation sequence take the first128 observations with magnitude<20, and
separately the first128 with magnitude<22. No truth cuts or stratified weighting.
Deduplicate the union for sampling, retaining membership in both cut cohorts.
The two cohort fits are correlated because they can share observations.

Preparation retains the existing full-pair/summary, partial-batch, order and
geometry replay gates. Save measurement and cohort checksums. Generating scenes
are not given to inference or used for initialization.

Each distinct observation receives four independent chains in each of two
independent runs:32768 steps,8192 warmup, measurement-informed initialization,
fresh prior-bank seeds2026103201/2026103202 and MH seeds2026103101/2026103102.
Use the original joint-morphology/SED-jump/persistent-environment sampler and
the validated invariant-work cache. No interacting ensemble stretch move.
Models, CPU response and4096-row density convention are unchanged.

Compare prefixes16384 and32768 after the same warmup. Original local profiles
retain all draws atg1=.018,.022. Full recovery profiles use every16th retained
state, preserving chain/time axes and autocorrelation-aware MCSE. These thinned
states are not labelled independent. Fit gridg1=-.06 through.10 in.005 steps,
with additional.018 and.022 nodes, alwaysg2=0.

## Selection and uncertainty

Each magnitude cut has its own selected likelihood:

```text
log L_c(g)-log L_c(g0)
 = sum_{i in cohort c} log E_posterior_g0[K_i(g)/K_i(g0)]
   - N_c log[Z_c(g)/Z_c(g0)]

Z_c(g) = E_prior[q_U(s,g) P_flow(radius>3 pixels, magnitude<c | s,g,U)].
```

The disk response changes only shape, so these radius/flux selection probabilities
use the base flow. They do not require response-emulator predictions. C8 is
integrated as a probability rather than an additional Bernoulli draw.

For each independent run, use131072 independently generated prior scenes in
16 chunks of8192, with16 QMC conditional flow draws per scene. Start selection
seeds2026103301 and2026113301, adding10 per chunk. Scenes and flow noise are
independent of observations and MH; retain common random numbers across all
shear nodes. Compare8 versus16 nested conditional draws and half versus full
scene populations. Independent chunks and complete replicas quantify selection
uncertainty, including covariance across shear nodes.

The report fits a local quadratic around an interior grid maximum and obtains
68%/95% connected likelihood intervals by grid interpolation. Boundary maxima,
missing intervals and unsupported reweighting cannot pass recovery. The fit is
exploratory whenever numerical diagnostics fail. It is not clipped toward the
injection or repaired using a known truth value.

Numerical shear MCSE propagates the three-node peak gradient through summed
posterior-ratio influences and independently sampled selection chunks. Retain
covariance between observations within a run and between shear nodes. Conditional
statistical curvature SE and numerical MCSE are reported separately. Local
quadratic and interval interpolation remain approximations, especially for a
small or non-Gaussian sample; they are not a full two-component inference.

## Diagnostic gates and interpretation

- Report per-object latent/local-ratio diagnostics across all eight independent
  chains, using rank-Rhat<1.01 and bulkESS>400 for nonconstant monitored quantities.
- Inspect likelihood-ratio convergence and raw importance overlap at nodes near
  the fitted peak and its95% interval, as well as.018/.022. Raw weightESS>400
  does not replace autocorrelation-aware functional ESS or independent replicas.
- Keep every selected object in the fit. A failing chain causes a failed or
  unresolved diagnostic, not removal of that observation.
- Require numerical shear MCSE below10% of conditional statistical SE, selection
  ladder shifts within twice numerical MCSE, and16k/32k prefix stability on the
  same scale before accepting this small recovery check.
- Record whether injectedg1=.02 is inside the exploratory95% interval. Statistical
  compatibility alone cannot pass a numerically failed test and does not establish
  an unbiased method across populations. Disagreement is also possible from
  ordinary finite-sample scatter; interpret alongside the interval and diagnostics.

These screens cannot prove absence of modes missed by every chain. If the
posterior bridge lacks overlap across the statistically relevant shear range,
the result calls for additional reference-shear sampling; no recovery is claimed
from extrapolating a failed bridge.

## Implementation, validation and jobs

`scripts/prepare_galsbi_population.py` gains optional measured-cut cohorts;
its original preparation path remains the default. New helpers and scripts:
`scripts/galsbi_bright_math.py`, `scripts/run_galsbi_bright.py`,
`scripts/report_galsbi_bright.py`, `jobs/job_galsbi_bright.sh`, and
`tests/test_galsbi_bright.py`.

Four focused tests pass: strict cuts, first-occurrence/overlap accounting,
known Gaussian shear recovery with a shear-dependent selected normalization,
boundary/zero-normalizer rejection, and covariance-preserving peak sensitivity.
Full scheduled suite16791497:382 passed,1 skipped in65.77s. Ruff and Python/bash
syntax checks pass. All heavy computation runs under Slurm.

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_bright_mh_20261002`.
`plan.json` is immutable. `source_snapshot` pins225 files with SHA256 checks;
`jobs.json` records the scheduler graph separately. Each allocated job records
CPU/GPU/memory/disk resources. Login-node resource metadata is labelled separately.

| Stage | Job | Dependency / resources |
| --- | --- | --- |
| Full suite |16791497| CPU; completed successfully |
| Prepare bright cohorts |16791905| after successful suite; one A40 |
| Four-observation smoke |16791907| after preparation; one A40 |
| Two complete replicas |16791908_0–1| after smoke; array throttle2, one A40 each |
| Combined diagnostics/recovery |16791910| after either replica outcome; CPU |

The smoke exercises MH, saved-state likelihood replay, short shear profiles and
forward selection before releasing the long jobs. Each replica then runs MH,
full shear profiles, and its independent selection calculation sequentially.
At most two new study GPUs overlap with the existing throughput GPU, within
the owner's four-GPU combined limit. No new sampler or recovery result is
accepted merely because scheduler jobs complete.

## Preparation and startup receipts

Preparation16791905 completed0:0 in58s. The two128-object cohorts contain
238 distinct observations and18 shared rows. Measured magnitude ranges are
15.1649–19.9879 for the<20 cohort and17.9625–21.9785 for<22. Every row satisfies
its stated strict measured cut; no replacements or truth cuts were used.

Smoke16791907 passed its four-observation256-step MH checks, full saved-state
shear replay (maximum log-kernel difference0), and forward selection for both
cuts. Both main replicas16791908_0/_1 have started on inter A40s. Measurement
initialization completed in277.41/275.22s, initial likelihood replay has maximum
error0 in both runs, and both reached step128/32768. Final results are not yet
available; long sampling is now progressing normally.
CPU status check16792113 is scheduled in about two hours and writes
`scheduled_check.json`; it records cluster state but does not send a chat
notification or automatically repair failures. CPU report16791910 remains
dependent on the two replicas. The existing throughput job is still running.
