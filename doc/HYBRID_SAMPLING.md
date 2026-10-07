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

### Selection-corrected bright-only g1 from the same chains (2026-10-03)

Question: with the selection correction, do the 1,098 bright (measured mag < 23) galaxies of the fresh 10k mock
give back the true g1 = 0.02? The answer is **no usable answer yet**. The combined likelihood has two bumps,
and the chains do not settle which one wins.

Method:
- Same likelihood and fit as the earlier bright study (`doc/BRIGHT_MH_TEST.md`, "Selection and uncertainty"):
  per-galaxy log E_post[K(g)/K(0.02)] on the 35-node g1 grid, minus N_c log[Z_c(g)/Z_c(0.02)].
- Z_c uses 2 × 16 × 8192 forward prior scenes with C8 × (radius > 3 px) × (mag < cut), seed 2026104301.
- Cuts at measured mag 20, 21, 22 and 23 are nested subsets of the same chains.
- Code frozen at commit e77ce80 (`estimate_snapshot/`). Driver `estimate_bright10k.py`.
- There is only one MH run per galaxy, so chains 1–2 and chains 3–4 are fitted separately in place of a replica.

Fixed-shear replay depends on the GPU model:
- The profile stage recomputes each saved draw's likelihood at g1 = 0.02 and requires it to match the saved
  value to 1e-6.
- It failed on the first galaxy whenever it ran on a different GPU model from the one that sampled it. For
  example, chains made on V100 nodes failed when replayed on P5000 or RTX2080Ti nodes.
- The tolerance was kept. Each MH batch was replayed on the node that sampled it (`resubmit_profiles.sh`);
  every batch then passed.
- Jobs: 16803116 and 16803122–24 (profiles), 16803116–17 (selection), 16803125 (report), 16803138 (diagnostic).
  Jobs 16803117–19 failed on the wrong-node replay and are superseded.

| Cut | n | g1, 16k steps | g1, 32k steps | Quoted errors at 32k (stat, sampling) | Chains 1–2 / 3–4 | Galaxy bootstrap 16–50–84% | Bootstrap sd |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| < 20 | 75 | 0.0242 | 0.0250 | ±0.0013, ±0.0003 | 0.0253 / 0.0151 | 0.014 / 0.025 / 0.026 | 0.0057 |
| < 21 | 181 | 0.0140 | 0.0255 | ±0.0013, ±0.0004 | 0.0264 / 0.0141 | 0.013 / 0.025 / 0.026 | 0.0055 |
| < 22 | 456 | 0.0136 | 0.0253 | ±0.0012, ±0.0003 | 0.0260 / 0.0138 | 0.011 / 0.025 / 0.026 | 0.0070 |
| < 23 | 1098 | 0.0147 | 0.0258 | ±0.0012, ±0.0004 | 0.0265 / 0.0240 | 0.025 / 0.026 / 0.027 | 0.0048 |

Findings:
- **The quoted errors are far too small.**
  - Doubling the chain length moves the estimate by 0.011, about 25 times the quoted sampling error.
  - Chain halves differ by up to 0.012.
  - The combined log likelihood has two near-equal peaks, at g1 = 0.015 and 0.025, with a dip of 3–7 units at
    the true 0.020 (figure, left).
  - The quadratic fit describes only whichever peak is higher, so its curvature error and influence-function
    sampling error ignore the flip.
- **A few galaxies cause it.**
  - Ten galaxies at mag 18–21 (e.g. obs 5367, 7319, 964, 7156) carry 83–95% of the slope shift between 16k and
    32k steps, and 66–81% of the shift between chain halves.
  - Several have likelihood-ratio effective sample sizes of 1–25 out of 6144 draws.
  - At 32k steps, 108/1098 galaxies have ESS < 400 at the peak nodes.
  - The spikes sit exactly on grid nodes. That is the pattern of a stuck chain in a poor state: there, K(g)/K(0.02)
    can be huge for one g, and that single draw dominates the log-mean. It is not real shear information.
- **The selection correction is negligible for the bright galaxies.** The fit with and without the N log Z term
  differs by less than 0.0001 at every cut.
- **Honest range:**
  - The galaxy bootstrap sd is ±0.005–0.007 (m ≈ ±25–35%). Its distribution is two-humped, at about 0.014 and
    0.025.
  - The truth lies inside the 16–84% range for the cuts at 20, 21 and 22. At 23 it lies outside that range
    (0.025–0.027) but inside the bootstrap mean ± sd (0.024 ± 0.005).
  - The bootstrap resamples galaxies only, not chain noise, so even this range is optimistic.
  - So this neither confirms nor rules out the 30–45% low result of the earlier bright study.
- Dropping the ten galaxies with the largest score MC error does not stabilise the answer (mag < 23: 0.025 at
  32k, 0.014 at 16k).

Next: a stable bright-only number needs the roughly 100 galaxies with low ratio ESS resampled with many more
independent chains (or tempering, which cut stuck chains from 32/40 to 6/40). Then rerun this report unchanged.

Receipts (`$DATA_DIR/sbsi/catalogue_prior/galsbi_mock10k_20261002/`):
- `plan.json`, `estimate_snapshot/` with `estimate_snapshot.commit`;
- `estimate_bright10k.py` → `estimate_bright10k.json`;
- `estimate_gpu.sh`, `submit_estimate.sh`, `resubmit_profiles.sh`;
- `selection0/`, `selection1/`, and `bright_g*_k*/shear_profiles.npz`;
- `diagnose_estimate.py` → `diagnose_estimate.json`;
- `plot_estimate.py`.

Figure: `SBSI/plots/galsbi_bright10k_estimate.png`.

### Tempered rerun of the mag < 22 galaxies and a score test at the true shear (2026-10-03)

The owner asked to fix the noisy bright-only estimate autonomously.

**Rerun.**
- Rule, fixed in advance in `rerun_plan.json`: rerun every bright galaxy with measured mag < 22 (456 galaxies).
  The choice is by magnitude only.
  - The basis is `rerun_candidates.json`, the smallest likelihood-ratio ESS over g1 in [0.010, 0.030].
  - That ESS is below 400 for 54/75 (< 20), 47/106 (20–21) and 24/275 (21–22), and for 0/642 at 22–23 (median 5037).
- Sampler: the tempered settings of the 238-galaxy check (8 rungs plus narrow steps), 32k steps (8k warmup),
  4 chains.
- Two independent replicas, A and B (`submit_rerun.sh`, seeds in the plan):
  - jobs 16803344/46/48/50 (A) and 16803345/47/49/51 (B), each 39–45 min on 4 GPUs, about 23 GPU-s per galaxy;
  - each job ran its profile stage on its own node (`rerun_job.sh`), and all 48 shards passed the 1e-6 replay.
- Galaxies with a stuck chain (log-posterior R-hat ≥ 1.1 in either replica): 62/456, against 246/456 in the
  original run.

**The grid-peak fit is still unusable** (`estimate_rerun.py`, job 16803442):
- Replica A gives 0.015 at every cut; replica B gives 0.024–0.026.
- The A − B difference is 4–5 times its predicted sampling error.
- With per-node sampling errors, the summed likelihood is nearly flat between 0.015 and 0.025, within about
  2.5 log units. The nodes 0.005 away from the sampled shear carry errors of ±0.7–0.9; the nodes 0.002 away
  carry ±0.2–0.3. The peak hops between nodes.
- Its curvature claims about 10–25 times more information than the score variance (below). So the likelihood
  more than about 0.002 from the sampled shear is badly estimated for bright galaxies, both in level and in shape.
- Stuck chains were not the root cause. The cause is estimating the likelihood far from where we sampled.

**Score test at the true shear** (`score_test.py`, job 16803446). It uses only the reliable nodes 0.018 and 0.022:
- Per-galaxy selected score s_i = [L_i(.022) − L_i(.018)]/.004 − d log Z_c/dg.
- Per-galaxy sampling variance e_i² from independent halves: A vs B for rerun galaxies, chains 1–2 vs 3–4
  otherwise.
- Information I = Σ(s_i² − e_i²). One-step offset Δg = Σ s_i / I.
- Errors: statistical 1/√I, sampling √(Σe²)/I.
- Replica check: the A − B score difference is −1.0σ to −1.4σ of its prediction, so these errors pass.

**This is a closure check, not a blind estimate.** Every MH chain in this mock, including the original run
and both reruns, sampled the scenes at the mock's true shear. The sampler reads it from `reference/protocol.json`
(`shear` = [0.02, 0]), and the start-point picker scores its candidates at that same shear. All likelihoods are
reweighted from those samples, so they are reliable only near the truth. The grid-peak fits above are
truth-anchored in the same way.
The table answers one question: starting from the truth, do the data push the shear away? They do not, at
these errors. A blind estimate must choose the sampling shear without the truth, for example by starting at
g1 = 0 and iterating.

| Cut | n | g1 (score test) | ± stat | ± sampling | Total | Pull vs 0.02 | m |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| < 20 | 75 | 0.011 | 0.010 | 0.005 | 0.012 | −0.8σ | −45% ± 58% |
| < 21 | 181 | 0.010 | 0.010 | 0.007 | 0.012 | −0.8σ | −49% ± 60% |
| < 22 | 456 | 0.006 | 0.008 | 0.007 | 0.011 | −1.3σ | −68% ± 54% |
| < 23 | 1098 | 0.026 | 0.007 | 0.005 | 0.008 | +0.7σ | +29% ± 41% |

The cuts are nested, so the rows are correlated. Replica A alone gives 0.021 ± 0.010 at mag < 23, and replica B
alone gives 0.030 ± 0.008. The original untempered chains give 0.029, but their half-split errors are not
trustworthy, because stuck chains inflate the score spread.

Reading:
- **The closure check finds no bias.** Every cut is within 1.3σ of the truth. This does not test a blind
  inference.
- **The bright galaxies alone cannot test m better than about ±40%.** Shape noise gives ±0.0065 at
  mag < 23 and sampling gives ±0.005. Earlier peak-fit errors of ±0.0012 were five to seven times too small.
- **The "30–45% low" result of the earlier bright study is not reproduced, and not excluded.** That study used
  the same grid-peak fit, which these checks show is unreliable for bright galaxies.

Limits:
- The sampling shear is the truth (see above). A blind version must start without it, for example at g1 = 0,
  and iterate. The start-point picker already uses the sampling shear. (A hard-coded 0.02 sits only in
  `environment_noise_audit`, which runs only without persistent scenes; a blind run should replace it too.)
- For the 22–23 galaxies, the sampling error comes from untempered chain halves. Their ratio ESS is healthy.
- One-step estimate only; generator and native-density approximations remain.

Next: the error is now set by the number of galaxies, not sampling. To test m below ±40%, the estimate needs
the faint galaxies too: about 8,900 more MH runs, or the old sampler for them.

Receipts (`$DATA_DIR/sbsi/catalogue_prior/galsbi_mock10k_20261002/`):
- `rerun_candidates.py` → `rerun_candidates.json` (job 16803343);
- `make_rerun_plan.py` → `rerun_plan.json`, `rerun_ids.txt`;
- `rerun_job.sh`, `submit_rerun.sh`, `tempA_g*_k*/`, `tempB_g*_k*/`;
- `estimate_rerun.py` → `estimate_rerun.json`, with a dry run on replica A only in `estimate_rerun_dry.json`
  (job 16803387);
- `score_test.py` → `score_test.json`;
- `plot_rerun.py`.

Figure: `SBSI/plots/galsbi_bright10k_rerun.png`.

## Blind hybrid MH estimate on the 10k mock (2026-10-03)

The owner asked for a blind estimate, more efficient sampling, the hybrid on the 10k mock, and debugging
until the shear comes out right, keeping things simple ("Occam's razor").

**Design: one sampler, one prior, one selection term.**
- Bright galaxies (measured mag < 23, 1098) run tempered MH: 8 rungs, narrow steps 0.5, 32k steps, 8k warmup.
- Faint galaxies (the other 8902) run plain MH: 16k steps, 4k warmup, with saved ratios thinned by 4.
- The "hybrid" is only a split by brightness. No atom bank is used; a GalSBI atom bank does not exist and
  would need the same scene law.

**Blind iteration.** No input uses the mock truth.
- Start at g = (0, 0).
- Each iteration samples every galaxy at the current shear g0 (`--shear`). On the same GPU, it saves the
  per-draw likelihood ratios at g0 ± 0.002 in g1 and in g2.
- One selection integral Z(g) at the same five shears: 2 replicas × 16 chunks × 8192 scenes, with the cuts
  C8 × radius > 3 px × mag < 25.8.
- Per galaxy, the score s_i is the central difference of log E[K(g)/K(g0)] minus d log Z/dg.
- Chain noise E_i comes from the chain halves (chains 1–2 vs 3–4).
- Information I = Σ(s_i s_iᵀ − E_i). Newton step Δg = I⁻¹ Σ s_i.
- Errors:
  - statistical: I⁻¹;
  - chain: I⁻¹ΣE_i I⁻¹;
  - selection: the scatter over chunks;
  - plus a bootstrap over galaxies.
- The rule for the final answer was written before iteration 2 finished (`final_rule.md`): use an
  iteration's estimate only if its sampling shear lies within one total error of it.

| iteration | sampling shear | Newton estimate g1 | g2 | ± stat | ± chains | ± selection | total | bootstrap |
|---|---|---|---|---|---|---|---|---|
| it0 | (0, 0) | 0.0215 | −0.0016 | 0.0039, 0.0040 | 0.0019, 0.0024 | 0.0007, 0.0009 | 0.0044, 0.0048 | 0.0044, 0.0048 |
| it1 | (0.0215, −0.0016) | 0.0272 | −0.0038 | 0.0040, 0.0042 | 0.0021, 0.0022 | 0.0007, 0.0008 | 0.0046, 0.0048 | 0.0046, 0.0047 |
| it2 | (0.0272, −0.0038) | **0.0240** | **−0.0038** | 0.0042, 0.0038 | 0.0028, 0.0022 | 0.0010, 0.0007 | **0.0051, 0.0044** | 0.0053, 0.0045 |

**Result.** it2's step (−0.0033, +0.0000) is inside its errors, so it2 has converged.
- **Primary: g = (0.0240 ± 0.0051, −0.0038 ± 0.0044).**
  - Against the mock truth (0.02, 0), used only now: +0.8σ in g1 and −0.85σ in g2.
  - In terms of m: +20% ± 25%.
- it1's g1 step was 1.25 errors long, so by the rule it is not used.
- it1 and it2 agree within their independent errors: g1 differs by 0.0033 against 0.0037 (0.9σ).
- Their mean, (0.0256 ± 0.0045, −0.0038 ± 0.0043), is +1.2σ and −0.9σ from the truth.
- The bootstrap over galaxies matches the error budget in every iteration.
- No single galaxy dominates: the top 10 carry 12–31% of the score variance.

**Debugging notes.**
- **The stencil-curvature check is unusable at these chain lengths.**
  - It gave a nonsense step in it1 (−5.4). Single bright galaxies show curvatures of ±10,000–25,000 from
    chain noise alone, when the expected value is about 6.6 per galaxy.
  - The chain-half difference in a bright galaxy's log ratio is 0.05–0.1, and the curvature multiplies it by
    1/0.002².
  - The score-based step corrects for chain noise and is the one used.
- **The it0 → it1 → it2 path is consistent with noise.**
  - The first step from (0, 0) was five errors long. A one-step estimate from that far away is only a
    starting move.
  - it1 and it2, both close to the answer, agree.
- **The chain noise is the bright galaxies'.** It makes up 35–55% of their score variance, against 4.5% for the
  faint galaxies.
- **Stuck chains** (log-posterior R-hat ≥ 1.1): 22–29 bright and 456–475 faint galaxies per iteration. The
  faint ones barely matter, given their 4.5% chain-noise share.

**Cost.** Each iteration took 80–91 min on 4 GPUs × 44 cores (176 cores): 2080 Ti, P5000 and 2× V100. That is
about 5.4 GPU-hours per iteration.
- Layout per GPU: 13 bright + 7 faint processes at 2 cores each, with the selection on the spare cores
  (it0 used 15 + 5, and the faint processes then took longest).
- Bright: 0.13 s per step at about 690 rows.
- Faint: 0.21 s per step at about 1270 rows, plus 451 s of setup and 630 s of output.
- The speed limit is CPU cores; the GPU was busy 45–67% of the time.
- A projection, not a run: halving the faint steps to 8k would raise the faint chain share to about 9% and the
  total chain error by about 4%, while saving about 13% of the core time.

**Limits:**
- One mock realisation, with 10k galaxies. The test of m is ±25%, set mostly by shape noise; it is not a
  0.3% test.
- Two iterations near the answer; the estimator is the one-step Newton estimate.
- The generator and native-density approximations of the continuous prior remain.
- Disk: about 96 GB of chains and ratio files per iteration.

Receipts (`$DATA_DIR/sbsi/catalogue_prior/galsbi_blind10k_20261003/`):
- `source_snapshot/` (d31ea79), `bright_ids.txt`, `faint_ids.txt`;
- `make_plan.py`, `iter_job.sh`, `blind_step.py`, `outliers.py`, `timing.py`, `plot_blind.py`, `final_rule.md`;
- `smoke/` (job 16805601);
- `it0/` (jobs 16805631–34), `it1/` (16806051–54), `it2/` (16806315–18), each with `plan.json` and
  `blind_step.json`.

Figure: `SBSI/plots/blind10k_iterations.png`.

## Blind 100k run and why one Newton step was not enough (2026-10-03 to 10-05)

**Run.** A fresh 100k GalSBI mock (`galsbi_blind100k_20261003`, seed 2026100701), with the same blind Newton
iteration as the 10k study.
- 4 GPU workers pull units from a shared queue. The selection term uses 2 replicas × 160 chunks.
- it0 and it1 run short chains: bright 8192 steps (2048 warm-up), faint 4096 (1024). it2 and it3 run full
  chains: 32768 and 16384.
- The rule (`final_rule.md`, written before sampling): accept an iteration when its own step is inside its
  total error.
- An amendment, made before any estimate: a measured-size cut of radius < 100 px. It removes one 858-px
  flow-tail galaxy whose ratios moved by 6000 per 0.002 in g.

| iteration | sampling shear g1 | next g1 | next g2 | total error | step inside error? |
|---|---|---|---|---|---|
| it0 | 0 | 0.01388 | −0.0007 | 0.00152, 0.00202 | no |
| it1 | 0.01388 | 0.01927 | 0.00228 | 0.00158, 0.00192 | no |
| it2 | 0.01927 | 0.02266 | −0.00019 | 0.00149, 0.00155 | no |
| it3 | 0.02266 | **0.02364** | **0.00114** | 0.00147, 0.0015 | yes |

**Pre-declared answer:** g = (0.02364 ± 0.00147, 0.00114 ± 0.0015).
- The truth (0.02, 0), read afterwards, puts it at +2.5σ and +0.8σ.
- The score's measured slope between iterations was only 0.68 ± 0.10 of the information I used for the step.
  So each step covered about two thirds of the distance, and the error bars were too small by that factor.

### Why the step fell short (no truth used)
BFD's one-step argument: if every galaxy's likelihood is close to linear in g over the step, then the
information at g = 0 equals the slope of the total score, and one Newton step lands on the root. Two things
broke it.

1. **Fixed-shape likelihood ratios are very noisy for bright galaxies.**
   - The sampler re-evaluates each saved draw at g0 ± 0.002, holding the intrinsic shape fixed. A bright
     galaxy's measured shape pins its sheared shape, so at the neighbouring shear a fixed intrinsic shape puts
     the draw in the tail.
   - Bright chain noise was 35–80% of the bright score variance.
   - Fix (a7b4468: `--comoving-profiles` in the sampler, `comoving.py` to post-process saved chains): move the
     intrinsic shape with the shear, ε' = M_g'⁻¹(M_g0(ε)), keep the circularized radius, and include the exact
     Jacobian.
   - Bright chain noise drops to about 1% of the bright score variance.
2. **492 galaxies with measured |e| ≥ 0.99 claim information that their score does not show.**
   - 368 of them are bright and 124 faint. They are flow-tail draws; the mock contains them because it is
     generated from the same model.

| group (co-moving ratios) | n | information at g = 0 | at it2 | measured slope −ΔS/Δg1 |
|---|---|---|---|---|
| faint, \|e\| < 0.99 | 88,872 | 400k | 388k | 400k ± 14k |
| bright, \|e\| < 0.99 | 10,635 | 167k | 179k | 154k ± 3k |
| bright, \|e\| ≥ 0.99 | 368 | 233k | 13k | 43k ± 14k |

("Information" here is the outer product minus chain noise.) The last group adds about 40% to I at g = 0
and almost nothing at it2.
- A rerun of it0's bright chains on other GPUs (c0, same seeds) shows its chain noise is not underestimated:
  Σ(run difference)² / Σ(predicted) = 0.02–0.6.
- So these galaxies' likelihoods are far from linear over 0.02 in g, unless both runs got stuck in the same
  mode (their chains coincide for the first 500–1600 steps).

**Fix: a measured-shape cut, |e| < 0.99, on the data and in Z** (`max_measured_ellipticity`, cc76c1d and 8f16643).
- Observed galaxies are cut on their measured ngmix shape.
- Z(g) applies the same cut to the model's measured shape: the flow draw moved by radial_tanh(b g).
- b is predicted per selection draw from its measured radius and flux and the scene's neighbour pairs,
  exactly as in the mock generator and the likelihood.
- Cost: selection takes about 19 s per chunk with the cut (3 s without), about 50 min per replica on 6 cores.
- Each iteration's cut Z uses that iteration's own seeds, so cut and uncut results differ only by the cut.

### One step from g = 0
All rows are from it0 (short chains); the stat error is about ±0.0013 throughout.

| variant | next g1 | next g2 | ± chains (g1) | total (g1, g2) | galaxy bootstrap (g1, g2) |
|---|---|---|---|---|---|
| original (fixed ratios, all galaxies) | 0.01388 | −0.0007 | 0.0011 | 0.00152, 0.00202 | 0.0035, 0.0022 |
| co-moving, no cut | 0.01659 | 0.00117 | | 0.0012 (g1) | 0.0030 (g1) |
| fixed ratios + cut | 0.01940 | −0.00054 | 0.00159 | 0.0021, 0.0021 | 0.0027, 0.0022 |
| **co-moving + cut** | **0.02180** | **0.00140** | 0.00041 | **0.00142, 0.00145** | 0.00136, 0.00143 |

With both fixes, the galaxy bootstrap matches the error budget, and bright chain noise is 1.3% of the bright
score variance.

**Converged values for comparison** (same cut and Z, co-moving):
- The Newton step at it2 gives (0.02214, −0.00047) ± (0.00137, 0.00139).
- The straight line through the it0 and it2 g1 scores crosses zero at g1 = 0.02229.
- The one step from g = 0 is 0.0005 short in g1 (0.4 stat σ). The remaining non-linearity: I at g = 0 is
  566k, against a measured slope of 554k (2%).

**The g2 difference between the two steps (0.0019) is selection Monte Carlo noise.**
- d log Z/dg2 per galaxy at it2 is 0.0077 ± 0.0016. The other iterations give 0.0021–0.0025.
- That one fluctuation moves it2's g2 by −0.0009. The rest (about 1.7σ) is chain noise.
- The chunk values are Gaussian (kurtosis 2.8–3.3), so the selection error bars hold.
- The small positive mean of dlogZ/dg2 is allowed: the scene layout is not mirror-symmetric.

### Checks of the co-moving numbers
- **Replay.** Re-evaluating the saved draws on CPU matches the sampler's fixed-shape ratios to < 0.01 in
  every unit except obs 99194 (the 858-px galaxy, already cut).
- **b recomputation.** c0 was sampled with in-sampler co-moving ratios, with b recomputed from the retained
  scene. Against `comoving_circ` (b kept) on the same chains, the one-step estimate moves by (+0.00011, +0.00005)
  against chain noise of 0.00012, so the post-processing is a faithful stand-in.
- **Chain noise.** it0 against the c0 rerun, both co-moving: the one-step estimate moves by (−0.00004,
  +0.00001) ± 0.00012. Σ(difference)² / Σ(half-split prediction) = 0.72, so the half-split error is not too
  small. It is not a fully independent rerun (same seeds).

### Confirmation iteration `one`
The test iteration was declared in `final_rule.md` (amendment 2) before it was sampled.
- It sampled at the one-step shear (0.0218, 0.0014) with short chains like it0 (jobs 16818772–75, about 2 h on
  4 GPUs).
- It used co-moving post-processing and the same cut, with new selection seeds.
- The rule: one iteration suffices if `one`'s own step is inside its total error in both components.

| at `one` | step | next | ± stat | ± chains | ± selection | total | bootstrap |
|---|---|---|---|---|---|---|---|
| g1 | **+0.00002** | 0.02182 | 0.00131 | 0.00040 | 0.00028 | 0.00140 | 0.00136 |
| g2 | **−0.00152** | −0.00012 | 0.00132 | 0.00040 | 0.00029 | 0.00141 | 0.00140 |

- **g1: one step is enough.** The measured score slope between g = 0 and `one` (Δg1 = 0.0218) is
  0.98 ± 0.03 of I, against 0.68 ± 0.10 before the fixes (`final_report.py it0_cut:comoving_circ one:comoving_circ`).
- **g2 fails the pre-declared test narrowly:** a step of 0.00152 against a total error of 0.00141.
  - Both estimates use the same galaxies, so shape noise cancels in the step. Its own noise is chains plus
    selection at both iterations, about ±0.0007, so the step is about 2.2σ.
  - The largest single piece is the selection Monte Carlo. d log Z/dg2 per galaxy was 0.0032 ± 0.0016 at
    it0_cut and 0.0080 ± 0.0016 at `one`, which by itself moves g2 by about −0.0009.
  - A dedicated scan on common scenes (`zscan/`, g1 = 0.02 and g2 from −0.004 to +0.004; `zscan.py`) shows no
    real change of that slope across g2 = 0: +0.0033 ± 0.0023 per galaxy, 1.4σ. The interval-to-interval
    scatter matches the quoted errors.

**Against the truth** (0.02, 0), read once, after the test:
- The one-step estimate from g = 0 is (0.02180 ± 0.00142, 0.00140 ± 0.00145): +1.3σ and +1.0σ, m = +9.0% ± 7.1%.
- `one`'s next shear is (0.02182 ± 0.00140, −0.00012 ± 0.00141): +1.3σ and −0.1σ, m = +9.1% ± 7.0%.
- The old pre-declared it3 answer was +2.5σ, m = +18.2% ± 7.4%.

**Conclusions:**
- With co-moving ratios and the |e| < 0.99 cut, the BFD one-step estimate works on this mock. The remaining
  g2 step is Monte Carlo noise.
- Where the remaining noise comes from:
  - selection: 320 chunks give ±0.0003 in g per iteration;
  - short chains: ±0.0004.
  - Both are small next to shape noise (±0.0013), but they set how cleanly a second iteration can confirm the first.
- Cheap improvements, not run here:
  - more selection chunks;
  - the same selection seeds in every iteration, so the selection noise cancels in the step.
- The measured-shape cut is a modelling choice the owner should confirm. It is applied identically to the
  data and to Z and uses only measured quantities, but it removes 0.5% of galaxies.
- Cost of the cut: about 19 s per selection chunk, against 3 s without it.

Figure: `SBSI/plots/blind100k_onestep.png` (it0–it3, then the one-step estimate and `one`).

Receipts (`$DATA_DIR/sbsi/catalogue_prior/galsbi_blind100k_20261003/`):
- `final_rule.md`, including amendment 2 (the one-step test), written before `one` was sampled.
- `blind_step.py`, `onestep.py`, `groups.py`, `g2check.py`, `zcheck.py`, `zscan.py`, `c0check.py`, `comoving.py`,
  `final_report.py`, `plot_blind.py`.
- `make_cut.py` (it0_cut and it2_cut), `make_one.py`.
- `source_cut/` (8f16643), `cut_selection.sh`.
- Logs in `<iteration>/logs/` and `logs/`.

Figures: `SBSI/plots/blind100k_iterations.png`, `SBSI/plots/blind100k_onestep.png`.

### Rotated-twin paired mock: m = −2.0% ± 2.5% (2026-10-05)
The owner noted that m = +9% ± 7% is too loose a test for a mock and asked for a rotated-twin paired mock.
Almost all of the ±0.0014 is the shape and pixel noise of these particular 100k galaxies; chain and selection
noise are only ±0.0004 and ±0.0003.

**The twin** (`scripts/prepare_galsbi_twin.py`, commit 8cd45c5):
- Every scene of the original mock is drawn again with the same seed and turned by 90° about its target.
- The model sees a scene only through quantities that do not change under rotation (magnitudes, sizes,
  Sérsic index, neighbour distances, crowding), plus the target's intrinsic shape. So the rotation is exactly
  e → −e for the target, and nothing else changes.
- Each twin goes through the same steps as the original: shear, the detection draw (same random numbers),
  the measurement draw, the measured cuts and the disk-response shift. Its selection is its own; pairs are
  not forced to be complete, and no truth cuts are applied.
- The original's 50 chunks were rebuilt and reproduce the stored mock bit for bit (difference 0.0), on the
  node and GPU model that made it (met-cl-vis01).
- Size: 99,922 twins. 98,630 scenes have both members selected; 1,370 have only the original and 1,292 only
  the twin (`reference/pairs.parquet`).

**Measurement-noise coupling.** The noise of the measurement flow comes from four standard-normal draws:
two drive the shape, two drive size and flux. The twin can reuse them, negate the shape pair ("antithetic"),
or draw fresh ones; each choice gives a correctly distributed twin. A 2-chunk test (`coupling_test/`), using
the scatter of the pair-mean measured shape relative to two independent galaxies (1 = no gain):

| coupling | e1 | e2 | both members selected |
|---|---|---|---|
| antithetic | **0.12** | **0.22** | 98.5% |
| same draws | 1.42 | 1.46 | 98.9% |
| fresh draws | 0.73 | 0.78 | 87% |

Antithetic was used. The choice was made on this variance proxy alone, without the truth.

**Analysis**, the same recipe as the original's one-step result (`it0_cut:comoving_circ`):
- Sampling at g = 0 with the same chains (bright 8192 steps, faint 4096), co-moving bright ratios, and the
  measured radius < 100 px and |e| < 0.99 cuts.
- 2 × 160 new selection chunks (seed 2026807001).
- Run on 4 GPUs (340 cores): all 805 units done with no failures, in about 2 h. Co-moving replay < 0.005.
- `pair_step.py` (fixed before the twin was sampled):
  - one Newton step from g = 0 over both catalogues;
  - Z from the 640 pooled selection chunks;
  - errors with the scene as the unit (an original and its twin are deliberately correlated); bootstrap
    over scenes.
  - Check: on the original alone it reproduces `blind_step.py` exactly.

| one step from g = 0 | g1 | g2 | m |
|---|---|---|---|
| original 100k | 0.02180 ± 0.00142 | +0.00140 ± 0.00145 | +9.0% ± 7.1% |
| rotated twin 100k | 0.01741 ± 0.00141 | +0.00055 ± 0.00144 | −12.9% ± 7.0% |
| **paired** | **0.01959 ± 0.00051** | **+0.00096 ± 0.00056** | **−2.0% ± 2.5%** |

- Paired error budget, g1: stat 0.00037, chains 0.00029, selection 0.00020. The bootstrap over scenes gives
  0.00046, consistent with stat plus chains (0.00047).
- If the 200k galaxies were independent, the stat error would be 0.00094: the twin removes about 85% of the
  shape-noise variance.
- Dropped by the cuts: 493 originals and 464 twins.
- The truth was read only in the last printed block of `pair_step.py`.

**Conclusions:**
- The +9% of the original was mostly the shape noise of that one draw. The twin lands at −13%, and the pair
  at −2.0% ± 2.5%, consistent with no multiplicative bias.
- g2 is +0.00096 ± 0.00056 (1.7σ), also consistent with zero.
- Chain and selection noise are now as large as the shape noise. The next factor in precision needs longer
  bright chains and more selection chunks, not more galaxies.

Figure: `SBSI/plots/twin100k_paired.png`.

Receipts (`$DATA_DIR/sbsi/catalogue_prior/galsbi_twin100k_20261005/`):
- `prepare_twin.sh`, `source_twin/` (8cd45c5), `reference/`, `coupling_test/`.
- `make_twin_plan.py`, `launch_it0.sh`, `make_lists.py`.
- The pipeline scripts copied from the 100k study by `copy_scripts.py` (code from its `source_cut`, 8f16643).
- `pair_step.py`, `pair_check.py`, `pair_step_comoving_circ.json`, `plot_pair.py`.
- Logs in `logs/` and `it0/logs/`.

### Paired mock with a measured |e| < 0.7 cut: m = −1.3% ± 4.4% (2026-10-05)
The owner asked for the |e| < 0.7 cut that real surveys often use, applied to the same 2 × 100k paired mock.
The rule was written down before any result existed (`e07_rule.md`).
- The cut acts on the measured shape. It is applied to the data and also inside the selection integral Z
  (plan key `max_measured_ellipticity: 0.7`), so the model knows which galaxies a shear would push across it.
- The chains are reused unchanged: `make_e07.py` writes `it0_e07/` and `one_e07/` with plan copies and
  symlinks to the existing runs.
- New selection chunks (`selection_e07.sh`, code from `source_cut`, 8f16643) reuse the seeds of the 0.99 run,
  so the two results differ only by the cut.
- `pair_step.py` takes an optional iteration name (`pair_step.py comoving_circ it0_e07`); the defaults are
  unchanged.
- The cut drops 18,413 originals and 18,266 twins; 75,602 scenes keep both members.

| one step from g = 0, \|e\| < 0.7 | g1 | g2 | m |
|---|---|---|---|
| original 100k | 0.02352 ± 0.00185 | +0.00291 ± 0.00192 | +17.6% ± 9.2% |
| rotated twin 100k | 0.01601 ± 0.00188 | −0.00019 ± 0.00185 | −19.9% ± 9.4% |
| **paired** | **0.01975 ± 0.00088** | **+0.00131 ± 0.00092** | **−1.3% ± 4.4%** |

- Paired error budget, g1: stat 0.00053, chains 0.00035, selection 0.00061. The bootstrap over scenes gives 0.00062.
  Selection noise is now the largest term; the cut makes Z noisier.
- The paired error is 1.7× larger than with |e| < 0.99. The cut removes about 18% of galaxies, and the pairs
  cancel less of the shape noise.
- Slope check on the original (`final_report.py it0_e07:comoving_circ one_e07:comoving_circ`):
  0.96 ± 0.06, so one step from g = 0 is still enough.
- Conclusion: on a mock drawn from the model itself, the |e| < 0.7 cut causes no bias detectable at the ±4%
  level. This is a self-consistency test. It cannot show whether the model reproduces how real images put
  galaxies across the cut; [the ConstGold check](V39_MODEL.md#measured-e--07-cut-on-constgold-2026-10-05)
  tests that, and the model fails it.

Receipts, both study directories: `make_e07.py`, `selection_e07.sh`, `analyse_e07.sh`, `e07_rule.md`,
`it0_e07/`, `one_e07/`, `pair_step_comoving_circ_it0_e07.json`, and
`final_report_it0_e07-comoving_circ_one_e07-comoving_circ.json`.

### Unpaired 500k mock: m = −4.1% ± 3.2% (2026-10-07)
The owner asked for a 500k mock without a rotated twin. The rule was fixed before the mock was generated
(`rule.md`): the recipe of the original's one-step result, applied once, with the truth read only at the end.
- Mock: same generator and settings as the 100k mock, new seed 2026100801, 500,000 selected galaxies at true
  shear (0.02, 0). Code 8cd45c5 (= 8f16643 + the twin script).
- Sampling at g = 0 with the same chains: bright (measured mag < 23) tempered MH, 8192 steps; faint plain MH,
  4096 steps. Co-moving bright ratios (`comoving_circ`), cuts measured radius < 100 px and |e| < 0.99 on the data
  and in Z.
- Selection: 2 × 640 chunks (4× the 100k run), new seeds (base 2026907001).
- Pass rule: m and c2 both within 2σ of zero.

**Run.** 4,062 units (2,660 bright of 21 galaxies, 1,402 faint of 317) on 4 GPUs.
- The large H200/H100 nodes had no free GPU, so it ran on 132 cores (3 V100 + 1 A40) in about 25 h.
- Per unit: bright 20 min, faint 24 min. Selection: 17 s per chunk, both replicas done in 3 h.
- Co-moving post-processing: about 2 h.
  - 64 tasks, at most 32 at a time.
  - Each task spends about 40 min on setup, then about 1 min per unit.
  - One task landed on a crowded node; helper tasks finished its last 20 units.

**One failure, outside the cuts.** Bright unit 2292 stopped at initialisation ("measurement candidate stream has
no finite posterior starting state").
- Its row 431089 has a broken measurement: radius 13,139 px, flux 7.7 × 10⁶, |e| = 1.0.
- The other three rows with radius ≥ 100 px (858, 424 and 104 px) sampled normally.
- The unit was rerun with its own seeds without that row (`rerun_u2292.sh`).
- `blind_step.py` now accepts a galaxy with no result only if it is outside the cuts, and lists it as
  `unsampled_outside_cuts`. The row was always excluded from the data and from Z, so the result's definition
  does not change (`rule.md` addendum, written before any result).
- Worker slots read `units.txt` from the top, so a freed unit can be re-claimed by a slot that has not yet
  passed its line. That happened once here, and the unit failed again the same way. Free a unit only after every
  running slot is past it.

| one step from g = 0 | g1 | g2 | m |
|---|---|---|---|
| **unpaired 500k** | **0.01918 ± 0.00063** | **+0.00028 ± 0.00065** | **−4.1% ± 3.2%** |
| original + twin, paired (for comparison) | 0.01959 ± 0.00051 | +0.00096 ± 0.00056 | −2.0% ± 2.5% |

- **Error budget, g1:**
  - stat 0.00059, chains 0.00018, selection 0.00015, total 0.00063;
  - the galaxy bootstrap gives 0.00062;
  - the rule predicted 0.00064.
- **Galaxies:** 497,482 used; 2,517 dropped by the cuts, plus the one unsampled row.
- **Against the truth:**
  - m is −1.3σ and c2 is +0.4σ, so **the pre-declared test passes**.
  - Combined with the paired 2 × 100k result as an inverse-variance mean of two independent mocks, not
    pre-declared: m = −2.8% ± 2.0% (1.4σ).
- **Checks:**
  - Bright chain noise is 1.4% of the bright score variance.
  - The faint chains flagged as slow-mixing (R-hat ≥ 1.1, 51%) match the 100k runs.
  - The 10 largest galaxies carry 2.2% of the g1 score variance.
- **The stencil Hessian is a noisy cross-check, not an alternative estimate.**
  - Here I/H is 1.05 in g1 and 0.96 in g2, and the Hessian step would give m = +0.4%.
  - In the 100k runs the same ratio was 0.70–0.72 in g1, in the opposite direction.
  - The measured slope between iterations, 0.98 ± 0.03 of I on the 100k mock, is the check that supports the
    one-step estimate.

**Conclusions:**
- At 5× the galaxies, the one-step estimate from g = 0 is consistent with no bias at the ±3% level.
- The unpaired 500k is slightly less precise than the paired 2 × 100k (±3.2% against ±2.5%), at about 2.5×
  the compute.
- Chains and selection are now small: together ±0.00023, against ±0.00059 shape noise. Pairing remains the
  cheaper way to gain precision on mocks.

Figure: `SBSI/plots/mock500k_onestep.png`.

Receipts (`$DATA_DIR/sbsi/catalogue_prior/galsbi_mock500k_20261006/`):
- `rule.md` and its addendum; `prepare.sh`, `reference/`.
- `copy_scripts.py`, which copies the twin study's pipeline with only the root path changed.
- `make_500k_plan.py`, `launch_it0.sh`, `rerun_u2292.sh`, `units_rerun_u2292.txt`.
- `it0/blind_step_comoving_circ.json`, `plot_m.py`.
- Logs in `logs/` and `it0/logs/`; failed attempts in `it0/failed_attempts/`.
