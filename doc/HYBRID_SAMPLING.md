# Hybrid atom integration and continuous MH

## Assessment — October 2, 2026

The subsequent owner-authorized [bright MH test](BRIGHT_MH_TEST.md#owner-request-and-scope--october-2-2026)
tests the proposed continuous fallback on measured-mag<20 and<22 cohorts.

The owner's proposal is a strong candidate for reducing population inference
cost: use cached atom integration for observations it resolves, and continuous
posterior sampling for narrow or poorly covered observations. Brightness is a
useful initial routing variable. It is not yet an adequate acceptance rule.

The essential condition is a common underlying scene prior and likelihood.
The existing FS2 atom catalogue and the current fixed-F24 GalSBI continuous
prior are different population models. Directly joining their results would
confound a numerical improvement with a population change. The recommended
first experiment uses a GalSBI atom bank approximating the same scene law as
the continuous sampler. Reuse the old integration algorithm and cache design;
do not assume the old atom contents represent the new prior.

This is an evaluation and proposed experiment, not an implemented hybrid or
authorization receipt for another production run. Existing benchmarks continue;
the paused20k population run remains paused. F8, C8 and the original disk
response are unchanged. No shear-accuracy claim is made.

## Evidence for concentrating work on bright observations

The [actual ConstGold20k audit](V39_ACTUAL20K_BFGS.md#status-on-september-27-stopped-at-the-first-solve)
found an indefinite first-pass information matrix at every draw rung through
16384. Along its negative eigenvector:

| Measured magnitude | Observations | Fraction of20k | Curvature contribution |
| --- | ---: | ---: | ---: |
| <20 | 135 | 0.675% | -2.239340 million |
| 20–22 | 774 | 3.870% | -0.216479 million |
| 22–24 | See original audit | — | +0.034884 million |
| 24–25.8 | See original audit | — | +0.033766 million |

The135 brightest rows contribute93.8% of the net negative curvature. Removing
the selection-normalizer term algebraically leaves the matrix indefinite.
This establishes concentration of the observed problem, not its cause:
local non-concavity, finite-prior support, proposal failure and model mismatch
remain distinct possibilities. Positive faint-bin curvature does not prove
accurate faint-bin scores. No rows were removed from inference.

The [historical bright-extension study](BRIGHT_PRIOR_EXTENSION.md#results-and-decision)
separates two numerical problems. It used V3.6 models and three inspected
observations, so its numerical values are not V3.9 population results:

- Posterior atom ESS was only1.15–1.74 under the original prior. A narrow
  posterior can depend on one or two allowed scenes even with perfect summation.
- Adding7.78million bright atoms did not reliably improve integration: candidate
  ranking missed dominant atoms. Under the original prior, an8192-exact plus
  8192-sampled estimator instead captured>99.8% of the posterior mass in all
  three cases, with numerator RMSE0.00930%,0.04321%,0.00491% across16 seeds.

Thus more proposal draws can fix integration of a finite sum without fixing
its approximation to a continuous population. Conversely, sparse posterior
support alone does not prove an error if the declared target really is finite.
Treat the atom bank as quadrature only when a parent scene law is specified.

Brightness has a plausible mechanism: higher signal-to-noise can constrain a
smaller volume of latent scenes, leaving fewer bank atoms inside the likelihood.
Size, morphology, crowding and unusual shapes also matter. This is a working
explanation to test, not proof that all faint observations are safe.

## One target with two numerical methods

Use the [continuous-scene target](CONTINUOUS_SAMPLING.md#exact-statistical-target):

```text
I_i(g) = integral K(y_i,s,g) p(ds)
Z(g)   = integral p(ds) integral_W K(y,s,g) dy
log L(g) = sum_i log I_i(g) - N log Z(g).
```

K includes C8 usability, F8 density and the retained disk transformation and
Jacobian. W is the existing measured radius/magnitude selection. The same
prior includes target properties, environment marks, neighbour counts and
positions, with matching units, support and shape conventions.

For well-resolved observations, use a weighted bank from p and the existing
exact-stratum/defensive-complement algorithm to approximate I_i. For difficult
observations, draw from p(ds)K(y_i,s,g0)/I_i(g0) and estimate

```text
R_i(g) = I_i(g)/I_i(g0)
       = E_posterior_at_g0 [K(y_i,s,g)/K(y_i,s,g0)]

log L(g)-log L(g0)
       = sum_i log R_i(g) - N log[Z(g)/Z(g0)].
```

This identity requires posterior exploration and adequate support/overlap
across the shear range. Average likelihood ratios before taking logs. Absolute
MCMC evidence is unnecessary for a shear profile. At large centre changes,
check reweighting overlap and refresh the reference; do not assume local draws
or a carried Taylor expansion remain accurate indefinitely.

Let r_i be a fixed observed-data routing decision. Replacing an integral by
either consistent numerical estimator does not introduce a new selection cut.
Keep every selected observation and one common Z. Sum likelihood contributions
or score/information at a common centre; do not average separately fitted
bright/faint shear estimates. A measured-bright observation must still integrate
over the full latent prior, including intrinsically fainter scenes that could
produce its measurement. Routing is not a truth-magnitude truncation.

Fix routes across the shear stencil. Prefer a separate pilot stream for any
noisy routing diagnostic, followed by fresh production estimates; selecting
between noisy estimates using those same estimates can introduce selection
bias. This does not remove finite-bank or finite-log bias, which require
independent checks.

## Why directly joining the current priors is a different model

The old bank is an FS2-based weighted empirical scene population. The new MH
target uses a fixed GalSBI F24 population and explicit uniform Poisson neighbour
environments. [Prior compatibility](GALSBI_PRIOR_FEASIBILITY.md#compatibility-and-a-bounded-next-step)
has not established their equivalence. Matching a magnitude histogram would
not match conditional sizes, shapes, Sérsic indices or environment correlations.

For distinct priors A and C, a measured-brightness switch r(y) would give

```text
H(y,g) = r(y) I_C(y,g) + [1-r(y)] I_A(y,g)
Z_H(g) = integral_W H(y,g) dy.
```

H/Z_H could be defined as a new observed-data model, but neither original
normalizer generally equals Z_H. Normalizing both original models separately
and then selecting pieces by y does not generally yield a normalized density.
Such a splice requires a new statistical specification and validation; it is
not simply accelerating one prior integral. Smooth routing weights do not
resolve this population mismatch.

Practical options:

1. **Common continuous population:** generate new weighted atoms from the
   chosen continuous scene law; use atom quadrature and continuous MH under
   that law. This is the preferred GalSBI model-closure experiment.
2. **Retain the exact FS2 finite target:** improve retrieval, exact strata or
   discrete atom-index MCMC. Continuous off-catalogue moves cannot preserve
   a purely discrete target. This addresses integration but cannot fill gaps
   between its allowed scenes.
3. **Define a continuous FS2 population model:** possible future modelling
   work, with its own validation. Jittering catalogue coordinates or fitting
   a density silently is not an exact implementation of the old prior.

Old atoms may supply initialization locations or centres of an explicitly
evaluable continuous proposal. They cannot be turned into an exact continuous
prior by a naive density ratio: finite point masses and a continuous target
have incompatible support measures. Proposal corrections require the actual
sampling law and full target support.

Even a new GalSBI bank needs a consistency check: the numerical scene generator
uses interpolated CDFs, while local MH uses analytic native prior factors.
[Existing approximation limits](CONTINUOUS_SAMPLING.md#a-simple-count-changing-move-partial-poisson-refresh)
must be checked through grid refinement or proposal-density correction. Sharing
the GalSBI name alone does not prove the two numerical targets agree.

## Routing and error accounting

A practical candidate policy is: route measured-bright objects directly to MH;
apply atom integration to the remainder; escalate any unresolved remainder.
Compare thresholds20 and22 rather than treating either as already validated.
The magnitude21.255 stress case demonstrates why a threshold20 alone is weak.

For the atom route, assess independent sampling-seed changes in shear ratios,
scores and curvature, candidate-list expansion, largest posterior atom mass,
distinct-atom ESS, and independent-bank/size sensitivity. Draw ESS can be large
while repeatedly selecting the same few atoms. No single ESS threshold proves
coverage, and two proposal seeds on one bank cannot measure bank discretization.
Audit faint objects throughout magnitude, size and crowding space, including
a random sample of objects whose cheap diagnostics look good.

For MH, assess independent complete replicas, starts, step ladders and the
actual shear-ratio functionals. Do not treat interacting ensemble walkers as
independent chains. Rank-based Rhat and functional ESS are useful exploration
screens ([Vehtari et al.](https://arxiv.org/abs/1903.08008)); Rhat<1.01 is not a
direct bound on aggregate shear error. Failed observations remain flagged for
more work, not silently dropped from the selected population.

Numerical error should be propagated through the combined score. If J is the
total observed information at a stable solution, the first-order perturbation is

```text
delta_g approximately J^{-1} [sum_i delta_score_i - N delta_grad_log_Z].
```

Measure covariance across shear nodes and observations. Reusing one atom bank
creates shared numerical error; the selection estimate is shared as well.
Both require independent population/bank replicas. An unbiased positive
numerator estimate also generally gives a biased log estimate:
E[log(Ihat)] approximately log(I)-Var(Ihat)/(2 I^2) for small relative error.
The expansion is unreliable for severe weight tails. Neither coherent bias
nor common-bank error is guaranteed to cancel across20k observations.

Use the existing proposed numerical budget of absolute shear error<=1e-5
for the combined estimator, including normalization, as a design target rather
than an achieved result. It is below6e-5, corresponding to0.3% of injected
g1=0.02. Independent finite panels cannot establish population calibration bias.

## Runtime consequences

Let f be the routed fraction, t_A atom cost per faint observation and t_M
the cost of a converged MH observation, in GPU seconds per replica. Then

```text
T = N[(1-f)t_A + f t_M] + T_bank/cache + T_selection + T_other.
```

If every observation first receives atom evaluation, replace (1-f)t_A by t_A.
Account separately for CPU work, queueing, output and shared setup. Multiply
replicated observation work by two for the owner's two-replica comparison;
bank and selection replication have their own budgets.

The table uses actual ConstGold20k counts and the **unverified target**
t_M=10 GPU seconds per galaxy per replica:

| Direct MH route | Galaxies | Fraction | MH GPU time, two replicas |
| --- | ---: | ---: | ---: |
| Measured magnitude<20 | 135 | 0.675% | 0.75 GPU hours |
| Measured magnitude<22 | 909 | 4.545% | 5.05 GPU hours |
| All observations | 20000 | 100% | 111.11 GPU hours |

These are only MH components, not end-to-end wall times. Even at100 seconds
per difficult galaxy, the two-replica MH components would be7.5 and50.5 GPU
hours respectively. This is why a small difficult subset could afford longer
chains. Conversely,10 seconds averaged over representative1024-galaxy batches
need not hold for a smaller, exclusively difficult batch. Batch efficiency and
mixing can both worsen. Benchmark the actual routed workload.

The fresh1024-observation GalSBI throughput panel contains5 measured-mag<20
and51 mag<22 objects (0.488% and4.980%). These are descriptive counts from
the saved measurements, not estimates proving equality to FS2. The historical
finite-prior model mock had142/20k below20, also a different population.

The [pop-cosmos paper](https://arxiv.org/html/2602.03930v2#S3.SS3) demonstrates
8.2 GPU seconds per galaxy on a GH200 with batches of1000 and512 walkers. It
supports the feasibility of batched posterior inference, but its SED target,
hardware and validation metrics do not establish SBSI throughput or shear
precision. The present1024-object full-process timing is still pending.

## Proposed bounded comparison and decision criteria

1. Pin a common scene law, all likelihood artifacts and W. Resolve or bound
   numerical generator/native-density differences. Generate independent atom
   banks with explicit weights and complete neighbour information.
2. Use the existing1024 independent GalSBI measurements for a representative
   comparison after their MH run completes. That MH run is a candidate
   comparator, not ground truth. Use an additional predeclared32-object panel
   (eight per existing magnitude band) to examine bright tails, crowding and
   both sides of proposed thresholds. Keep stratified diagnostic results
   separate from population calibration estimates.
3. Compare atom, MH, threshold20 hybrid and threshold22 hybrid on identical
   observations and shear nodes. Use both independent atom banks and independent
   integration seeds, plus longer/multiple-start MH references on difficult
   cases. Examine common-bank errors, not just sampler-seed differences.
4. Freeze a brightness-plus-diagnostic routing rule on the pilot; test it on
   fresh data. Record route counts, escapes among apparently safe faint rows,
   residual score/information differences and induced shear shifts. Require
   overlap checks around both routing boundaries; do not tune to injected shear.
5. Include an independently replicated common-prior selection calculation,
   combine all rows, and benchmark actual routed batch sizes. Advance to20k
   only after the combined numerical error and complete resource cost are
   understood. Model closure remains distinct from FS2/image-level calibration.

No fixed bank size, magnitude threshold or per-object ESS value is accepted
by this assessment. The useful outcome is a cost-versus-shear-error comparison,
with unresolved rows visible and no empirical corrections to force agreement.

## Current MH comparison relevant to the fallback

Array16788846_0–3 and report16788848 completed successfully. At16384 steps,
after averaging four walkers within each of two independent replicas:

| Observation | Measured magnitude | Control ratio Rhat at g1=.018/.022 | Stretch ratio Rhat |
| --- | ---: | --- | --- |
| 0 | 19.745 | 1.00396 / 1.00471 | 1.05724 / 1.05999 |
| 1 | 21.255 | 1.07745 / 1.04877 | 1.00445 / 1.00710 |
| 2 | 23.971 | 1.00486 / 1.00457 | 1.03128 / 1.03029 |
| 3 | 25.118 | 1.01682 / 1.01661 | 1.00422 / 1.00424 |

Stretch improves the difficult obs1 but worsens obs0 and obs2; obs1 stretch
bulk ESS remains346/265. There is no uniformly accepted fallback and no
population shear estimate. These results support continued algorithm checks,
not selecting whichever proposal gives a preferred shear.

Receipts: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_throughput_20261002/ensemble_report.json`
and sibling `ensemble_report.md`; magnitude counts use
`reference/observations.parquet` and `30-2.5*log10(measured_flux_from_mag_auto)`.
The distinct1024-object throughput job16788845 remains active. The ensemble
proposal experiment and the atom/continuous hybrid proposed here are separate
ideas despite both using the word hybrid.

## 1024-panel MH: noise and stuck chains vs magnitude

October 2, 2026. The owner asked whether to sample bright observations with
MH and faint ones with the old atom method (threshold about 20 or 21). As a
first step, untempered MH ran on all 1024 GalSBI panel observations, binned by
measured magnitude, to see where MH has trouble. No code changed.

Setup:
- Code frozen at 2758f68; 32,768 steps (8,192 warmup), 4 chains per galaxy.
- 4 GPUs, each running 6 processes × 8 cores (`jobs/job_galsbi_shared_gpu.sh`).
- Same reference and sampler flags as the earlier 16,384-step `throughput1024`
  run, but with new seeds, so that run serves as an independent replica.
- Definitions follow the 238-galaxy check:
  - score s = (r.022 − r.018)/.004 per draw;
  - per-galaxy Monte Carlo (MC) error = sd(s)/√ESS_bulk;
  - "stuck" means rank R-hat of the log posterior ≥ 1.1.

Cost: 1480 s of sampling plus 46 s of setup for the slowest job, so 1024
galaxies took about 25 minutes on 4 GPUs. That is about 6.0 GPU seconds per
galaxy at 32k steps (mixed A40/other nodes).

| Measured mag | n | Mean score | Median MC error | Between-galaxy spread | MC share of score variance | Share of MC variance of panel mean | Stuck | Replica rms z |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| < 20 | 5 | −6.8 ± 5.1 | 5.2 | 8.6 | 0.43 | 0.09 | 4/5 | 0.78 ± 0.25 |
| 20–21 | 15 | −1.1 ± 2.9 | 3.6 | 5.0 | 0.80 | 0.49 | 6/15 | 0.96 ± 0.18 |
| 21–22 | 31 | +0.5 ± 1.0 | 3.3 | 2.0 | 0.88 | 0.29 | 14/31 | 1.11 ± 0.14 |
| 22–23 | 78 | −0.41 ± 0.44 | 1.18 | 3.4 | 0.24 | 0.094 | 40/78 | 1.26 ± 0.10 |
| 23–24 | 163 | −0.31 ± 0.23 | 0.44 | 2.9 | 0.038 | 0.018 | 13/163 | 1.14 ± 0.06 |
| 24–25 | 358 | −0.12 ± 0.12 | 0.14 | 2.2 | 0.008 | 0.005 | 1/358 | 1.07 ± 0.04 |
| 25–25.8 | 374 | +0.04 ± 0.08 | 0.067 | 1.6 | 0.003 | 0.001 | 0/374 | 0.97 ± 0.04 |

Whole panel: mean score −0.14 ± 0.09 (galaxy-to-galaxy spread), of which the
MC part is ±0.053 at 32k steps and ±0.058 at 16k. 78/1024 galaxies are stuck.
Replica rms z is 1.06 ± 0.02.

Findings:
- **The trouble is all bright.**
  - Galaxies brighter than 22 are 5.0% of the panel but carry about 88% of the
    MC variance of the panel mean.
  - Galaxies brighter than 23 are 12.6% of the panel and carry about 98%.
  - Ten galaxies (all brighter than 22.3) carry 75%, so these shares are
    themselves uncertain.
- **Faint galaxies are easy for MH.**
  - Fainter than 23, MC noise is at most 4% of the galaxy-to-galaxy score
    variance, and stuck chains are rare (13/163 at 23–24, 1/732 fainter).
  - Doubling the steps cut their MC error by the ideal 1/√2: median ratios
    0.71–0.76.
  - For brighter galaxies the median ratio was only 0.80–0.83, and the panel
    mean barely improved (0.058 → 0.053).
- **The error bars are roughly honest at every magnitude**: replica rms z is
  0.8–1.3 per bin, and no bin shows a 32k − 16k shift. The largest is
  22–23: −0.04 ± 0.06.
- **Implications for the hybrid proposal:**
  - Faint galaxies need no rescue from atoms: MH already handles them with
    small, correctly sized errors, and stays in the GalSBI prior used for
    bright ones.
  - The open problem is the about 50–130 bright galaxies. A cut at 20–21
    would cover only 20 panel galaxies and about 60% of the MC variance.
  - A simpler split is by step count, not by method: fewer steps for faint
    galaxies, many more chains/steps (or tempering) for those brighter than 22–23.
  - This is an estimate from one panel. The bright bins hold only 5–31
    galaxies.

Projection (extrapolated, not run): all-MH 20k at 32k steps ≈ 33 GPU-hours ≈
8 h on 4 GPUs. The faint 87% would probably cost about half that at 16k steps.

Receipts: `$DATA_DIR/sbsi/catalogue_prior/galsbi_mh1024_20261002/`
(`submit_mh1024.sh`, `analyse_mh1024.py` → `mh1024_analysis.json`,
`step_gain.py`, `plot_mh1024.py`). Jobs 16800304–07 (sampling) and 16800313
(analysis), all COMPLETED. Figure: `SBSI/plots/galsbi_mh1024_noise_vs_mag.png`.

### Rough panel shear from the same run (no selection correction)

The run saves no shear estimate. Its protocol says so explicitly, and it only saves g1 likelihood ratios at
0.018 and 0.022 around the true shear (0.02, 0). Two rough g1 estimates were made from them.
Neither has the selection term, and neither covers g2.

- **Parabola through L(0.018), L(0.02), L(0.022)** (`shear_estimate.py`): this fails.
  - For galaxies brighter than about 23, the curvature is swamped by Monte Carlo noise and comes out negative.
  - For example, it averages −240 per galaxy below 22.
- **Score with information from its spread** (`shear_from_score.py`): ĝ1 = 0.02 + Σs/Σ(s² − e²).
  - s is each galaxy's mean score and e its MC error. The error bar is a bootstrap over galaxies.

| Galaxies | 32k run | 16k run | MC part of the error |
| --- | --- | --- | --- |
| All 1024 | −0.006 ± 0.018 | +0.018 ± 0.015 | ±0.009–0.010 |
| Fainter than 23 (895) | +0.000 ± 0.016 | −0.001 ± 0.016 | ±0.002–0.003 |

Notes on the table:
- True g1 = 0.02.
- The two runs use the same galaxies, so they differ only by Monte Carlo noise. That noise comes almost
  entirely from the bright galaxies: the all-galaxy difference is 0.023 ± 0.013.
- The faint-only estimates agree with each other and sit 1.25σ below the truth.
- The information is about 5–6 per galaxy, so 1024 galaxies give a shape-noise error of about ±0.013 on g1,
  i.e. ±65% on m. This panel cannot measure calibration; it only checks the sampler.

## Fresh 10k GalSBI mock: MH on the bright members

October 2, 2026. The owner asked for MH on the bright galaxies of a fresh 10k mock. The old August 10k sets
(`infer_v1_contrastive_stencil_v1/*10k*`) were not used: they were made with the FS2 model at zero shear, while
MH uses the current GalSBI model.

Mock:
- Made with `scripts/prepare_galsbi_population.py`, using the same parent (`galsbi_summary_mh_20260930_v2`)
  and selection as the 1024 panel: C8, measured radius > 0.6", mag < 25.8, no truth cuts, shear (0.02, 0).
- New seed 2026100601. Five chunks of 8,192 inputs gave 10,000 observations in 65 s on one GPU (job 16803064).
- Bright = measured mag < 23: 1,098 galaxies (11.0%; the 1024 panel had 12.6 ± 1.0%).
  - mag < 20: 75
  - 20–21: 106
  - 21–22: 275
  - 22–23: 642

Run:
- Untempered MH, 32,768 steps (8,192 warmup), 4 chains.
- 4 GPUs × 6 processes, code at commit 3518934. That commit adds the `SBSI_OBS_LIST` index-list option to
  `jobs/job_galsbi_shared_gpu.sh`; the sampler itself is unchanged from 2758f68.
- Jobs 16803065–68 took 23.5–28 minutes, about 5.6 GPU-s per galaxy. Analysis job 16803069. All COMPLETED 0:0.

| Measured mag | n | Mean score | Median MC error | rms MC error | Between-galaxy spread | MC share of score variance | Share of bright MC variance | Stuck |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| < 20 | 75 | +3.4 ± 4.3 | 7.2 | 25.5 | 27.6 | 0.46 | 0.61 | 29/75 (39 ± 6%) |
| 20–21 | 106 | +2.7 ± 1.4 | 3.9 | 14.2 | 3.2 | 0.95 | 0.27 | 57/106 (54 ± 5%) |
| 21–22 | 275 | −0.60 ± 0.40 | 2.6 | 4.7 | 4.6 | 0.51 | 0.077 | 160/275 (58 ± 3%) |
| 22–23 | 642 | +0.54 ± 0.17 | 1.13 | 2.3 | 3.6 | 0.30 | 0.044 | 213/642 (33 ± 2%) |

Findings:
- **The 1024-panel picture holds with ten times more bright galaxies.**
  - 459/1098 (42%) have a stuck chain.
  - At 32k steps, sampling noise is 30–95% of the score scatter between bright galaxies.
  - The MC-share values agree with the 1024 panel within its small-n noise.
  - One tension: the 22–23 stuck rate, 33 ± 2% here against 51 ± 6% there (2.9σ).
- **The noise is concentrated in very few galaxies.**
  - The ten noisiest galaxies (all brighter than 20.8, nine of them stuck) hold 70% of the bright MC variance.
  - The single worst galaxy, obs 9223 at mag 18.4, has an MC error of 165.
  - The score distribution is heavy-tailed at the bright end: the between-galaxy spread is 28 below mag 20.
- **The mean scores are not a bias measurement.**
  - The 20–21 bin is +2.7 ± 1.4 (1.9σ) and the 22–23 bin is +0.54 ± 0.17 (3.2σ).
  - Without the selection term, the expected score at the true shear need not be zero.
  - The rough bright-only g1 (0.02 + Σs/Σ(s² − e²)) is unstable: 0.030 ± 0.39 for all mag < 23, with the
    bootstrap error blown up by the heavy tail, and 0.061 ± 0.013 for 22–23. Neither is a calibration
    statement.
- **Implication:** plain MH at this length is not enough for mag < 21. The cheap fix is targeted: rerun the
  roughly 100 noisiest galaxies with more chains and steps (or tempering), rather than everything. This run has
  no independent replica, so its error bars are not checked; the 1024-panel replica check (rms z 0.8–1.3 per
  bin) is the best evidence that they are honest.

Receipts: `$DATA_DIR/sbsi/catalogue_prior/galsbi_mock10k_20261002/`:
- `prepare.sh`, `make_snapshot.sh`, `source_commit.txt`;
- `select_bright.py` → `bright_ids.txt`, `bright_selection.json`;
- `submit_bright.sh`;
- `analyse_bright10k.py` → `bright10k_analysis.json`;
- `plot_bright10k.py`.

Figure: `SBSI/plots/galsbi_bright10k_mh.png`.
