# Work log

## 2026-10-02 — Fresh 10k GalSBI mock and MH on its 1,098 bright members

The owner asked for MH on the bright galaxies of a fresh 10k GalSBI mock. Details in
[the 10k section](HYBRID_SAMPLING.md#fresh-10k-galsbi-mock-mh-on-the-bright-members).

Changed:
- `jobs/job_galsbi_shared_gpu.sh` takes an optional `SBSI_OBS_LIST` index file (commit 3518934).
- Without it, behaviour is identical. A dry test confirmed the old range split (0–7 over 3 → 0–1, 2–4, 5–7)
  and the list mode.

Mock:
- `prepare_galsbi_population.py`, same parent and selection as the 1024 panel, seed 2026100601.
- 10,000 observations; job 16803064 took 65 s.
- Bright (measured mag < 23): 1,098.

Run:
- 32k-step untempered MH on 4 GPUs (jobs 16803065–68, 23.5–28 min), analysis job 16803069. All COMPLETED 0:0.
- The analysis asserts that every bright galaxy was covered exactly once.

Result:
- 42% of bright galaxies have a stuck chain.
- Sampling noise is 30–95% of the score scatter per bin.
- The ten noisiest galaxies (mag < 20.8) hold 70% of the noise.
- Mean scores lack the selection term, so they are not a bias measurement.

Limits:
- No replica run.
- The 22–23 stuck rate differs from the 1024 panel by 2.9σ.

Next: targeted reruns (more chains or tempering) of the about 100 noisiest galaxies.

## 2026-10-02 — Rough panel g1 from the 1024 MH run

The owner asked whether these galaxies have a shear estimate. They did not: the run saves only g1
likelihood ratios at 0.018 and 0.022. Two rough estimates, without the selection term and g1 only,
are in [the 1024-panel subsection](HYBRID_SAMPLING.md#rough-panel-shear-from-the-same-run-no-selection-correction).

- Parabola fit: fails, because the bright-galaxy curvature is dominated by Monte Carlo noise.
- Score-based estimate, all 1024 galaxies: -0.006 ± 0.018 (32k run) and +0.018 ± 0.015 (16k run).
- Score-based estimate, fainter than 23: +0.000 ± 0.016 and -0.001 ± 0.016. True g1 = 0.02.
- Shape noise limits the panel to about ±0.013 on g1, i.e. ±65% on m.

Validation: job 16803055 COMPLETED. Scripts `shear_estimate.py` and `shear_from_score.py` live in
`$DATA_DIR/sbsi/catalogue_prior/galsbi_mh1024_20261002/`.

## 2026-10-02 — 1024-panel MH: shear-score noise and stuck chains vs magnitude

The owner proposed MH for bright observations and the old atom method for faint
ones (cut about 20–21), and asked for a 1024-galaxy MH run on 4 GPUs first. No
code changed. Details in
[the 1024-panel section](HYBRID_SAMPLING.md#1024-panel-mh-noise-and-stuck-chains-vs-magnitude).

Run:
- Untempered MH on all 1024 GalSBI panel observations, 32,768 steps.
- 4 GPUs × 6 processes, code frozen at 2758f68.
- Wall time 23–28 min per GPU job (about 6.0 GPU seconds per galaxy).
- Root: `$DATA_DIR/sbsi/catalogue_prior/galsbi_mh1024_20261002/`.

Result:
- Brighter than 22: 5.0% of galaxies, about 88% of the Monte Carlo variance
  of the panel mean.
- Brighter than 23: 12.6% of galaxies, about 98%.
- Stuck chains: 64/129 brighter than 23, 13/163 at 23–24, 1/732 fainter than 24.
- Fainter than 23, Monte Carlo noise is at most 4% of the galaxy-to-galaxy
  spread, and doubling the steps shrinks it by the ideal 1/√2.
- Replica check against the earlier independent 16k run: rms z 1.06 ± 0.02, and
  no per-bin shift.

Validation: jobs 16800304–07 and 16800313, all COMPLETED 0:0. The analysis
asserts that both runs cover galaxies 0–1023 exactly once.

Limits:
- One panel only; the bright bins hold 5–31 galaxies.
- Ten galaxies carry 75% of the Monte Carlo variance.
- GPU types are mixed.

Next: route by step count rather than by method (fewer steps for faint
galaxies, more effort for those brighter than 22–23). The atom/MH prior
mismatch remains open.

## 2026-10-02 — 238-galaxy bias check: tempered vs untempered shear score

The owner asked for diagnosis item 3 on 4 GPUs. No code changed. Details in
[the bias check](TEMPERED_MH.md#238-galaxy-bias-check-tempered-vs-untempered).

Run: all 238 bright galaxies, 32,768 steps.
- Tempered: 8 rungs plus narrow steps, 3 GPUs.
- Untempered: 1 GPU.
- Each GPU ran 6 processes × 8 cores via `jobs/job_galsbi_shared_gpu.sh`, with
  code frozen at 2758f68.
- Root: `$DATA_DIR/sbsi/catalogue_prior/galsbi_bias238_20261002/`.

Result:
- Tempered minus untempered score: weighted -0.24 ± 0.13 (1.9σ), plain
  average -1.80 ± 1.11 (1.6σ).
- Stuck and healthy galaxies give the same weighted shift (-0.19 ± 0.22 and
  -0.26 ± 0.13), so stuck chains do not visibly move the score.
- The plain average is dominated by about 10 galaxies with Monte Carlo errors
  of 26–91 score units. Without them: -0.31 ± 0.57.
- Tempering halves the Monte Carlo error of the plain average (0.48 against
  0.90) at about 5x the GPU time.

Validation: jobs 16800157–60 and 16800163, all COMPLETED 0:0. The comparison
checks that each arm covers galaxies 0–237 exactly once.

Limits:
- One replica per arm, on mixed GPU types.
- The 1.9σ weighted shift is unexplained (chance, or the narrow steps).
- Error bars are about 13% small.

Next: more chains on the about 20 noisiest galaxies.

## 2026-10-02 — Sampler speed: thread fix, faster distance memo, shared GPUs

The owner said 238-galaxy and 1000-galaxy runs are too slow and allowed up to
200 CPU cores (the GPU cap stays at 4). Details in
[Speed](TEMPERED_MH.md#speed-thread-fix-and-shared-gpus).

Changed:
- `scripts/check_galsbi_local_sampler.py`: `torch.set_num_threads(<allotted
  cores>)`. Under `OMP_NUM_THREADS=1` this is what lets xgboost use the cores.
  The distance memo is now a sorted array with exact-match lookup instead of a
  dict loop.
- `jobs/job_galsbi_tempered.sh`: default 16 cores (was 4).
- New `jobs/job_galsbi_shared_gpu.sh`: P processes share one GPU, each with its
  own cores, galaxies and seeds (seed+k, bank seed+k).

Results (238 galaxies × 4 chains, 512 steps):
- One GPU process: 0.154 → 0.070 s per step.
- Four processes sharing one GPU: 0.048 s per step for all 238 galaxies.
- Six processes on a P5000 GPU: 0.044 s per step.
- Identity: every saved array matches the old code (maximum difference 0.0),
  and a shared-GPU process matches its standalone run.
- Projected 32k steps: 1000 galaxies in about 26 min on 4 GPUs × 48 cores
  (before: about 6 h on one GPU).

Validation: Slurm jobs 16800041–45, 16800065–68, 16800079, 16800083,
16800124–25 and 16800133. All COMPLETED except two intentional failures of the
initial-replay check on th-cl-hua28 (16800066, 16800079). Identity was checked
with `compare_arms.py`.

Limits:
- CPU-only jobs have a high per-step floor (0.10 s), so they add little.
- One CPU node (th-cl-hua28) fails the replay check deterministically.
- Shard outputs are separate directories; the analysis must combine them.
- Projections beyond 952 rows per GPU are untested.

Next: launch the 238-galaxy tempered-vs-untempered check (diagnosis item 3)
with the shared-GPU launcher.

## 2026-10-02 — Diagnosis: what limits the bright sampler now

The owner asked what to improve next. This was read-only analysis of the
8-rung and same-seed untempered panel runs (40 galaxies, 2 × 4 chains). Only
docs changed: a new section in
[what to improve](TEMPERED_MH.md#what-to-improve-next-diagnosis-of-the-8-rung-and-untempered-runs)
and the figure `plots/galsbi_improve_diagnosis.png`.

Findings:
- **R-hat fails on spread, not average.** The pass rule fails mostly on the
  spread half of rank R-hat. Score R-hat ≥ 1.01 for 44/80 overall, against
  10/80 for the average half (8 rungs).
- **The average shear score is reliable.** Replica differences: rms 1.14 ±
  0.13 error bars (8 rungs), 1.24 ± 0.14 (untempered). Tempered against
  untempered, galaxy by galaxy: rms 1.07, weighted difference -0.55 ± 0.45
  score units.
- **Noise, not trapping, is the limit.** Per-galaxy Monte Carlo noise is 26–44%
  of the between-galaxy variance. Untempered runs give 1.6 ± 0.6 to 1.8 ± 0.5x
  more precision per GPU second.
- **The GPU is under-used.** 8x the chain-rows cost 3.6x the time.
- **Slowest quantities:** Sérsic, size and magnitude (600–1400 steps per
  independent draw). The score itself takes 29–35 steps.
- **Ruled out:** heavy tails (clipping changes nothing), the estimator choice,
  and a hidden slow part (block-average effective draws match the standard
  count).

Validation: CPU jobs 16799151, 16799358 and 16799375 on `cluster`, all
COMPLETED 0:0. Light summaries were run on the login node from the saved JSON.

Limits:
- The panel was chosen for stuck chains.
- A trap bias up to about 1.4 score units (2σ) is not excluded at this size.
- The GPU packing gain is projected from two timings, not tested.

Next, ranked:
1. Gate on the average part of R-hat plus replica/sampler agreement.
2. Spend GPU on more untempered chains in fuller jobs.
3. Run a 238-galaxy tempered-vs-untempered bias check.
4. Optionally, average each draw over the target's orientation.

## 2026-10-02 — Light-transfer joint move tested on the panel: no gain, code removed

The owner asked for joint target+neighbour sampling. Built
`GalSBIScenePrior.transfer_light` and `--light-transfer-probability`
(commits 4853720, 50eeb5c, b2b72cc).

**The move.** Step the target m_r and let one neighbour within 4" absorb the
flux through a distance weight. Repeating the move with the opposite step
undoes it. Tests:
- exact undo and a Jacobian check;
- toy stationarity within 5σ, failing by more than 10σ without the Jacobian.

With p=0 the output was identical to the unmodified sampler (difference 0.0).

**Panel result, 40 galaxies, 32k steps, against same-seed controls.**
- Plain: 6/40 against 5/40 passes. Log-posterior R-hat 1.149 against 1.137.
  Stuck 17 against 15.
- 8 rungs: 15/40 against 16/40 passes. Log-posterior R-hat 1.042 against 1.037.
  Stuck 9 against 7.

Light-transfer acceptance had a median of about 2%. Accepted target changes
had a median of 0.010 mag. Obs 42 and 155 kept their bright-companion chains
in every arm. No measurable gain, so the code was reverted (sbsi, scripts and
tests now match 60a3646). Tests after the revert: 27 passed (scene prior,
summary, MCMC proposals). Details:
[light transfer](TEMPERED_MH.md#light-transfer-joint-move-no-gain-removed).

Also recorded the 8-rung 32k ladder: 16/40 passes for the same GPU time as
16 rungs at 16k (15/40)
([sparser ladder](TEMPERED_MH.md#sparser-ladder-8-rungs-32k-steps)).

GPU use stayed within 4 inter GPUs throughout: plain jobs followed the ladder
jobs, then tempered jobs, then controls. All jobs exited 0:0.

Next: the bright-companion state (obs 42, 155) and the fainter-target chains
(obs 125, 234) need a move that matches how the measurement model actually
splits light between neighbours. Alternatively, estimate how much mass that
state carries, rather than relying on more proposal tuning.

## 2026-10-02 — Tempering follow-up: more draws help; bright-companion mode found

Owner chose a sparser ladder, longer runs, and a diagnosis of the six
still-stuck galaxies. Doc-only change plus experiment scripts in the data
root: `make_ladder.py`, `stuck_galaxies.py`, `submit_ladder8.sh`, and an
optional draw limit in deep-dive `compare_tempering.py`.

The first-half rerun of panel16k (job 16795404) shows doubling the draws
lowers R-hat for 39/40 galaxies and raises passes from 9/40 to 15/40. Of the
six stuck galaxies, four move between states within the run, and two (42, 155)
stay in a bright-companion mode for the whole run. That mode is also present
in the original sampler: a neighbour as bright as the target about 3" away,
with the target about 0.3 mag fainter. Details:
[follow-up](TEMPERED_MH.md#follow-up-more-draws-and-the-still-stuck-galaxies).

Submitted 8-rung, 32768-step runs on 4 inter GPUs (jobs 16795399–402), plus
dependent comparison 16795418 and trace job 16795419. Results are pending.

## 2026-10-02 — Tempered MH frees the two-mode trap on a 40-galaxy panel

Owner asked to "work it" after the sampling deep dive
(`galsbi_sampling_deepdive_20261002/README.md`). That dive found chains stuck
in either a clean-target mode or a faint-neighbour-shares-light mode, in
85/238 galaxies. Added replica exchange (`--betas`) and a state-independent
narrow-step mixture (`--narrow-step-probability`) to
`scripts/check_galsbi_local_sampler.py`, and
`sbsi.mcmc_proposals.adjacent_swap_permutation` with two tests. One test
recovers the 0.7/0.3 weights of a bimodal toy to within 0.03. Added exact
speed-ups: emulator on all allotted CPU threads, and a PyCosmo distance memo.
Added `jobs/job_galsbi_tempered.sh`. With `--betas 1` the output is bit-identical
to the unmodified sampler before and after the speed-ups (identity2, max
difference 0.0). `tests/test_mcmc_proposals.py`: 10 passed.

Panel results: 16 rungs, 16384 steps, 2 replicas split over 4 inter GPUs, jobs
16795122–5, all 0:0 in 53–57 min.
- Shear-ratio check passes 15/40 (38% ± 8%), against 2/40 for the original at
  32k steps.
- Median log-posterior R-hat 1.04, against 1.25.
- Galaxies with a chain 8+ log units below the best: 6/40, against 32/40.
- Swap acceptance 0.66–0.78.
- Shear score, tempered minus original: -0.01 ± 1.31.

Costs about 7.5x GPU per galaxy, and 25/40 still fail R-hat < 1.01 (mostly
1.01–1.05). Details: [tempered MH](TEMPERED_MH.md#finding--october-2-2026);
figure `plots/galsbi_tempering_panel16k.png`. Next: sparser ladder and longer
runs, or a joint target+environment jump move. No 20k run.

## 2026-10-02 — Deep MH convergence audit isolates geometry and shear-overlap failures

Owner requested a deep investigation. Added `scripts/diagnose_galsbi_convergence.py`,
`probe_galsbi_geometry.py`, `plot_galsbi_convergence.py`, and `probe_galsbi_ridges.py`;
recorded methods, limitations, receipts and next proposal design in
[convergence audit](MH_CONVERGENCE_AUDIT.md#finding--october-2-2026). Linked bright
results and continuous-sampling status. All238 saved observations were audited;
post-hoc geometry panel10, conditional optimized-path panel4. Original bright
runs, immutable source snapshot, frozen models, observations and seeds remain intact.

Exact80-state prior/kernel replay differences0; summary error2.84e-14. Median
morphology acceptance about30% conceals poor movement. In one difficult object,
a straight conditional path drops52.27 log-density units but a curved route
between the same endpoints has no sampled drop. This is geometry evidence,
not posterior samples/global connectivity. Larger Gaussian proposals do not
uniformly improve mobility. Near-shear weights are healthy despite chain
disagreement; atg1=.010,78/128 brightest objects have raw weight ESS<400,
and at.040,89/128. Broad reweighting has a separate overlap problem. No recovered
shear, hybrid acceptance or10-second convergence claim is warranted.

Fixed `scripts/galsbi_bright_math.py::profile_fit` to use the same interpolated
curve for peak and interval crossings; added a narrow off-grid-peak regression
to `tests/test_galsbi_bright.py`. Derived corrected intervals are separate from
original reports and remain scientifically unvalidated. Five focused tests pass;
full scheduled suite16794283:383 passed,1 skipped in101.46s. Ruff passes for the
four diagnostics, helper and test. CPU audit16794252 completed1m49s, GPU geometry
16794276 completed1m54s, profile audit16794287 completed52s, GPU curved paths
16794310 completed1m14s; all0:0. Audit used one GPU at a time and all jobs finished.
Artifacts and executed-script checksums live in `galsbi_convergence_audit_20261002`.
Next: discarded-pilot frozen curved proposals with exact correction and independent
validation, then multiple reference-shear bridges. No20k or automatic longer
sampling run submitted; no failures removed from either cohort.

## 2026-10-02 — Bright MH jobs finish but convergence and shear recovery fail

Owner requested status. Read scheduler accounting and saved reports: both
bright replicas16791908_0/_1 completed0:0 in1h35m10s/1h35m09s, report16791910
in19m13s, and independent throughput16788845 in3h26m. No jobs remain active.
Updated [completed bright results](BRIGHT_MH_TEST.md#completed-result--october-2-2026).

At32k steps, full convergence passes0/128 in both cohorts; both local shear-ratio
checks pass35/128 below20 and10/128 below22. Exploratory conditional fits are
g1=.01410259 and.01077937 against injection.02, but convergence and broader
profile overlap fail, so these are not accepted recovery or bias measurements.
Conditional MH shear MCSE7.66e-4/1.48e-3 dominates measured normalization MCSE
2.75e-5/1.99e-5; selection ladders pass. Noted that the coarse report's mixture
of quadratic peaks and linear interval crossings yields inconsistent intervals
for some replicas. Those intervals are not validated and must be refined before
any future acceptance; original frozen reports remain intact.

Bright sampler cost22.53/22.51 GPU seconds per distinct galaxy at32k; complete
allocated workflows about24.0s including wider profiles/selection. Representative
1024-object16k full-process benchmark costs12.0659s per galaxy, above10s.
Validation: read-only JSON/accounting audit and compact diagnostic count/quantile
calculations. Documentation-only status update; no tests rerun, jobs submitted,
settings changed, or observations removed. The20k run remains paused.

## 2026-10-02 — Bright cohorts and numerical smoke pass; both replicas start

Preparation16791905 completed0:0 in58s:128 observations per measured cut,
238 distinct rows,18 shared. Read-only cohort audit confirms strict<20/<22
membership and no dropped failures. Smoke16791907 passed MH validation,
saved-summary profile replay (maximum error0), and both forward selection
integrals. Both main array tasks16791908_0/_1 are running on inter A40s.
Measurement initialization completed in277.41/275.22s, both initial likelihood
replays have maximum error0, and both reached step128/32768. This is startup
success, not convergence or shear recovery. See
[receipts](BRIGHT_MH_TEST.md#preparation-and-startup-receipts).

Added a bounded CPU status check16792113, scheduled in about2h, to write
`galsbi_bright_mh_20261002/scheduled_check.json`. It reads scheduler state,
progress, failures and any completed report; no chat wakeup tool is available.
The helper compiles and the new protocol's local document anchors validate.
Scientific source snapshot and plan remain unchanged; operational job metadata
is in `jobs.json`. Startup monitoring is complete; long sampling proceeds with
the scheduled check and dependent report, without occupying a chat polling it.

## 2026-10-02 — Launch bright-only continuous MH and selected shear-recovery test

Owner requested tests below measured magnitudes20 and22, including convergence
and shear recovery. Implemented optional cut-specific cohorts in
`scripts/prepare_galsbi_population.py`; added `scripts/galsbi_bright_math.py`,
`scripts/run_galsbi_bright.py`, `scripts/report_galsbi_bright.py`,
`jobs/job_galsbi_bright.sh`, and `tests/test_galsbi_bright.py`. Documented the
protocol in [bright MH test](BRIGHT_MH_TEST.md#owner-request-and-scope--october-2-2026)
and linked the hybrid assessment. No model or inference default changes.

Use128 fresh observations per measured cut, deduplicating shared rows without
altering either cohort; two independent four-chain32768-step runs with8192
warmup. Recovery profiles are conditionalg1 atg2=0, with independent cut-matched
selection populations and explicit covariance/MCSE, overlap, prefix and
selection-ladder checks. Failed objects stay included; unsupported profiles
cannot pass. This is GalSBI model closure, not FS2 recovery or a0.3% claim.

Validation: four focused tests passed; full CPU suite16791497 completed with
382 passed,1 skipped in65.77s. `ruff check` passes (system executable; py31
does not contain the Ruff module); Python compilation and bash syntax pass.
Tests cover strict measured selection and overlap, an analytic selected
Gaussian likelihood, invalid/boundary fits and correlated peak sensitivity.
Frozen225-file source snapshot and immutable plan are under
`catalogue_prior/galsbi_bright_mh_20261002`; job IDs are separate in `jobs.json`.

Submitted preparation16791905, smoke16791907, two-replica array16791908_0–1
(throttle2), and CPU report16791910 with prerequisite dependencies. Preparation
is initially Priority-pending; main jobs cannot start before numerical smoke.
At most two new GPUs plus the existing throughput GPU overlap, within the
four-GPU cap. The20k population run remains paused. Monitor preparation and
smoke through startup/completion; numerical or scientific failures are results
to diagnose, not permission to discard galaxies or force shear recovery.

## 2026-10-02 — Evaluate atom integration with a continuous-MH fallback

Owner requested a deep assessment while the throughput study runs. Added
`doc/HYBRID_SAMPLING.md` with the common-prior derivation, observation-dependent
prior-splice normalization, separate retrieval/discretization/MH errors,
brightness-plus-diagnostic routing, shared-bank/selection error accounting,
cost arithmetic and a bounded proposed comparison. Linked it from
`doc/CONTINUOUS_SAMPLING.md`. Reusing the old algorithm with a new common-prior
bank is distinguished from directly joining FS2 and GalSBI populations.
No inference code, model, routing default or scheduled science settings changed.

Evidence review: actual20k has135 measured-mag<20 and909 mag<22 rows; at the
unverified10 GPU-second target, their two-replica MH components cost0.75 and
5.05 GPU hours, excluding atom integration, bank preparation and normalization.
Light inspection of the1024-row saved GalSBI measurement file finds5 and51
rows respectively. Historical bright-extension results are labelled V3.6 and
three-case diagnostics, not V3.9 validation. Reviewed pop-cosmos v2 inference
and the rank-normalized Rhat reference; both are linked in the assessment.

Scheduler accounting confirms ensemble16788846_0–3 and report16788848 all
completed0:0. At16k steps, control obs1 ratio Rhats1.07745/1.04877 improve to
1.00445/1.00710 with stretch, but stretch obs0 worsens to1.05724/1.05999 and
obs1 bulk ESS remains346/265. No uniform mixing or shear-accuracy acceptance.
Throughput16788845 completed measurement initialization in1191.16s, passed
initial likelihood replay with maximum error0, and reached step346/16384;
full-process timing remains pending. Existing scheduled check16788875 remains;
no additional jobs launched and20k remains paused.

Validation: small saved-JSON/Parquet inspections and explicit count/cost
arithmetic; local Markdown link/anchor checks and `git diff --check` pass.
Documentation-only change, so no numerical tests rerun. Next scientific step
is common-prior bank consistency and paired method validation, including a
representative faint audit; this assessment does not certify either sampler.

## 2026-10-02 — Distinct-observation cache replay passes; main benchmarks released

Paired128-observation job16788841 completed0:0 in6m44s. All retained latent
states, MH decisions, log priors/kernels and all256 local shear profiles agree
exactly between cache-off/on. Maximum summary difference4.26e-14 is within
the unchanged1e-10 tolerance. Initial replay is exact in both arms. Sampling
time is25.097s versus18.525s (26.2% less); full256-step processes are204.806s
and194.07s. Baseline measurement initialization costs156.11s, or1.22s per
galaxy. This short run does not establish the full16384-step throughput target.

Four-arm startup16788842 completed0:0 in1m56s; both independent control and
stretch seed pairs passed numerical receipts. Main control16788846_0 has
passed warmup and made substantial retained-sampling progress. Remaining
ensemble tasks and full1024-observation throughput16788845 are scheduler
Priority-pending, with successful prerequisite dependencies cleared. Queue
waiting has become prolonged; no duplicate jobs or scientific changes made.
CPU ensemble report16788848 remains dependent. One bounded Slurm CPU status
check16788875 is scheduled about2h later to inspect accounting/progress/errors
and write `galsbi_throughput_20261002/scheduled_check.json`. No Codex chat-wakeup
tool is available, so this records cluster status without promising a chat
notification or autonomous repair. Both check files pass syntax validation.
The20k array stays paused. See
[bounded experiment](CONTINUOUS_SAMPLING.md#bounded-throughput-and-hybrid-ensemble-experiment--october-2-2026).

## 2026-10-02 — Launch bounded full-process throughput and hybrid-ensemble comparison

Owner authorized proceeding. Added complementary morphology stretch proposals
to `sbsi/mcmc_proposals.py` and opt-in `--ensemble-morphology` plus stage timings
to `scripts/check_galsbi_local_sampler.py`. The3D stretch move uses frozen
complementary walkers and the exact2log(z) MH correction, retaining full-support
local and original neighbour moves. Models, target density, support, numerical
tolerances and default proposals are unchanged. Interacting walkers are flagged;
`scripts/report_galsbi_sampler_runs.py` rejects independent-chain treatment.
`scripts/report_galsbi_ensemble.py` instead computes expectation diagnostics
after averaging walkers within each independent ensemble. Added mathematical
closure and covariance tests; fixed the scalar-conversion warning in
`scripts/report_galsbi_population.py` without changing its calculation.

Added `scripts/run_galsbi_throughput.py` and `jobs/job_galsbi_throughput.sh`
for full subprocess timing, distinct-observation cache replay, four-arm smoke,
and a matched-seed original/hybrid proposal pilot. Preserved sources and plan
under `catalogue_prior/galsbi_throughput_20261002` (156 files). Preparation
16788816 generated1024 independent selected observations in26s; geometry
error2.22e-16 and full/partial/order likelihood replay errors all zero.
Initial suite16788817:376 passed,1 skipped; final suite16788840 including
new ensemble-report tests:378 passed,1 skipped in55.21s. Ruff and Python/shell
syntax checks pass. No heavy work ran on the login node.

Submitted paired cache16788841 and startup16788842, both observed running
on inter A40s; first control startup completed with valid numerical checks.
Throughput16788845 follows successful cache replay; ensemble16788846_0–3
follows successful startup with throttle2; CPU report16788848 follows any
ensemble outcome. At most three GPUs can overlap, below the four-GPU cap.
Full run16384 steps/4096 warmup/four chains; timing includes initialization,
sampling, output and local profiles. The two-replica proposal pilot uses
the original four-observation stress panel, including the unresolved obs1.
The20k launch remains paused. No measured10-second full-process result or
accuracy acceptance yet; monitor dependent stages through startup. See
[bounded experiment](CONTINUOUS_SAMPLING.md#bounded-throughput-and-hybrid-ensemble-experiment--october-2-2026).

## 2026-10-02 — Assess pop-cosmos paper and adopt initial10 GPU-second target

Owner set10 seconds per galaxy per GPU as the first target and requested
reading arXiv2602.03930. Reviewed its v2 inference section, initialization
and convergence-sensitivity footnotes, and the earlier method paper
arXiv2406.19437v2. Recorded applicability and limits in
[throughput assessment](CONTINUOUS_SAMPLING.md#october-2-ten-gpu-seconds-per-galaxy-and-the-pop-cosmos-comparison).
The new milestone supersedes the interim0.1 core-second target. Retain the
earlier per-replica interpretation and count initialization/output in the
budget. The20k population array remains paused and was never submitted.

Recomputed costs directly from `galsbi_cost_20261001/cuda/result.json`:
at1024 observations,16384 steps give9.24s per galaxy in measured components;
cached-component subtraction estimates7.27s. Neither includes the entire
run or establishes mixing. CPU response and scene inputs account for58.5%
and17.5% of baseline component time, versus2.1% for the neural likelihood.
Affine-invariant target proposals are a possible future controlled comparison,
not an implementation change or permission to replace the frozen models.
Plan receipt updated with the initial speed target. Scheduler check shows no
active/pending user jobs; no new jobs or code changes, so no tests rerun.

## 2026-10-01 — Prepare 20k population diagnostic; pause for owner throughput target

Recent-job check: both64k measurement-start replicas16788678_1/_2 and
report16788679 completed successfully. Combined worst Rhat/minimum bulk ESS
at65536 steps for obs0–3 are1.09042/67.0,1.07064/93.7,1.02013/441.1,
1.03448/289.9; all numerical checks pass, but none passes all exploration
checks. Long array16781571 and report16781574 also completed; at1048576
steps combined obs0/2/3 pass, obs1 fails atRhat1.04247/ESS298.5, including
shear-ratio Rhat1.01513/1.01799. GPU cache replay16787894 passes with identical
latent states and MH decisions. GPU cost16787881 passes the baseline replay;
the alternative CUDA response predictor fails log-kernel replay at512/1024
observations (9.13e-5/4.93e-4), despite small coefficient differences.

Owner requested20k observations and selected16384 steps,4096 warmup, four
chains per observation and two independent replicas. Added
`scripts/prepare_galsbi_population.py`, `scripts/run_galsbi_population_shard.py`,
`scripts/report_galsbi_population.py`, and `jobs/job_galsbi_population.sh`.
The existing mock generator gains optional full measured-selected population
retention; the local sampler accepts a separately validated population
reference with observation checksums and fresh initialization banks. Original
pilot defaults, model artifacts, likelihood, proposal mixture and tolerances
are unchanged. The planned population is newly model-generated with the same
measured cuts, without truth cuts or magnitude-stratum selection.

The reporter compares independent-replica numerator log profiles atg1=.018/.022
and8k/16k prefixes, retaining within-shard and between-shear-point covariance
in conditional MCSEs. It rejects missing shards, coverage mismatches and failed
numerical receipts. It does not infer selected-population shear: the identical
selection normalizer cancels only in replica differences. Two replicas and
prefix agreement cannot rule out shared missed-mode or finite-log bias.
Tests in `tests/test_galsbi_population_report.py` check correlated-noise MCSE,
nonfinite draws, incomplete coverage and full-population measured selection.

Validation: Python compilation, bash-n and Ruff pass. Full suite16788794
completed0:0:374 passed,1 skipped in65.26s, with two NumPy scalar-conversion
deprecation warnings in the new report. Frozen153-file preparation snapshot
and plan are under `catalogue_prior/galsbi_population20k_20261001`.
Preparation job16788795 started and was cancelled after16s when the owner
rejected the36–48h estimate on four GPUs. No main array, smoke or report jobs
were submitted; no completed20k reference or sampling results exist.

Owner subsequently clarified the speed target to approximately10 galaxies
per second per core, replacing the interim10 seconds per galaxy per replica
interpretation. CPU-core versus GPU-worker scope is pending clarification.
Keep population execution paused while reassessing throughput and the method;
do not present tiled component estimates as full-process performance. See
[population throughput target](CONTINUOUS_SAMPLING.md#population-diagnostic-and-throughput-target--october-1-2026).

## 2026-10-01 — Authorize and launch 64k measurement-initialized MH comparison

The 16k initialization comparison completed all four workers in two waves,
about13min total; report16788283 completed. Across two replicas, prior-bank
versus measurement-start worst Rhat by observation was1.4427/1.1643,
1.6090/1.4959,1.1245/1.0703,1.0874/1.0494. Minimum bulk ESS remained below400
for all four observations in both arms. Better starts help but do not establish
convergence. No accepted shear estimate or sampler promotion follows.

Owner authorized two measurement-start replicas at65536 steps, retaining4096
warmup, the same four observations, four chains per observation, MH seeds
2026100141/142 and bank seeds2026100241/242. These replay the original starts
and then extend the trajectory; they are not additional independent replicas
of the earlier same-seed runs. Original frozen150-file source snapshot is
reused unchanged; models, proposal mixture, adaptation and cache-off setting
are unchanged. New external root `catalogue_prior/galsbi_measurement_init_64k_20261001`
contains the plan, launcher, snapshot link and automatic report launcher.
No active library code changed. Both shell launchers pass bash-n.

GPU array16788678 (tasks1/2, throttle2) requests one inter A40 per task and
40min walltime, without unrelated dependencies. No other user GPU jobs were
active at submission, within the combined four-GPU cap. CPU report16788679
on cip-cl-compute3 follows after any array outcome and checks16384/32768/65536
prefixes, including shear likelihood ratios. Expected elapsed is roughly25min
on two GPUs based on the six-minute16k runs; this is an estimate, not a bound.
Both tasks started on inter kng-cl-nv02, passed initial likelihood replay
with exactly zero error, and reached896/65536 steps without startup errors.
Final convergence results remain pending.

## 2026-10-01 — Release initialization comparison alongside old MH runs

Owner authorized running the two inter initialization workers alongside the
two old cip workers, raising this continuous-prior study's combined cap to
four GPUs. Recorded the scope in `AGENTS.md` and both external study plans.
Removed the artificial old-long/cache dependencies from array16788277;
throttle2 remains. Smoke16788261 was independently verified COMPLETED0:0.
Slurm rejected reattaching this already-finished smoke dependency, so cleared
the dependency field after verifying its successful accounting record.
The released array initially showed Priority rather than Dependency.

Cost job16787881 still waits for old task16781571_2 to release one slot;
cache16787894 follows the cost job. With at most two initialization workers
and the remaining old task3, this schedule stays within four GPUs. Report
dependencies, seeds, sources, models and numerical settings are unchanged.
Validation: sacct smoke status and scontrol/squeue dependency/resource audit;
no code changes or tests required. Tasks0/1 started on inter kng-cl-nv02;
both initial likelihood replays have exactly zero error (tolerance1e-6), and
both reached step1920/16384 without errors. Both old cip tasks were still
running, confirming four simultaneous GPUs. The second pilot pair and final
comparison remain pending; startup success is not a convergence result.

## 2026-10-01 — Move queued continuous-sampler GPU stages to inter

Owner requested trying `inter`. Scheduler inspection found four unallocated
A40s on kng-cl-nv02; the queued study stages were dependency-blocked rather
than waiting on GPU capacity. Updated existing jobs16787881 (cost benchmark),
16787894 (paired cache replay) and16788277 (initialization array) in place to
`Partition=inter TresPerNode=gres:gpu:a40:1`. Verified all three partition/GRES
requests and existing dependencies with scontrol; array throttle remains2.
Job ids, seeds, frozen sources, scientific settings and dependent reports are
unchanged. Running cip long jobs are not interrupted, and the combined two-GPU
limit still holds through the preserved dependencies.

Updated both study plan receipts under `galsbi_cost_20261001` and
`galsbi_measurement_init_20261001`. The first TRES spelling with a slash was
rejected after Slurm applied the partition change; corrected immediately to
the colon form and verified the final resource requests. No duplicate jobs
were submitted. All moved jobs are still pending upstream completion; check
their resource receipts and numerical replay when they start. Benchmark
timings on inter full A40s must not be presented as a controlled speedup over
cip virtual8GB A40 allocations. No code or model changes; no tests rerun.

## 2026-10-01 — Measurement-informed versus prior-bank initialization pilot

Owner explicitly requested trying starts near the measured properties. Added
`sbsi/galsbi_initialization.py` and the optional `--initialization measurement`
flag to `scripts/check_galsbi_local_sampler.py`. Candidate initial magnitude
comes from measured flux; radius uses a rough Moffat-PSF quadrature subtraction;
shape uses approximate inverse shear plus broad epsilon scatter. Four streams
have different Sérsic centres. Native redshift/SED and neighbour scenes come
from fresh prior banks. Candidate selection uses target-native log prior plus
the unchanged full likelihood; generating truth is never read. No guess
floor/clamp/scatter restricts subsequent MH support. The default initializer
is unchanged. New `initial_state.npz` and initialization receipts preserve
coordinates, summaries, log densities, neighbour counts, candidate counts and
preparation time. Reporter compatibility includes the initializer setting.

This compares initializer packages: the new arm changes both candidate
centring and ranking from the original highest-likelihood prior-bank control.
It cannot isolate those contributions. Both arms score2048 candidates per
chain and retain the same target, original proposals, warmup/adaptation,
4096-row density calls and numerical tolerances. Invariant-work reuse stays
off in both arms. Four fixed observations; four chains per observation;
MH seeds2026100141/142; bank seeds2026100241/242;16384 total steps with4096
warmup. Task order counterbalances the two arms over two seed pairs/waves.

Added `tests/test_galsbi_initialization.py` for measurement units, inverse
shape conventions, unchanged nuisance coordinates and boundary/support
handling. Full suite16788224:370 passed,1 skipped in67.59s. Ruff, Python
syntax and shell syntax checks pass. CPU startup16788261 exercises all four
arm/seed combinations at64 steps/32 warmup: completed3m47s, peakRSS1.04GB.
All four initial replays are exact; SED logK replay is exact and summary
differences are at most2.84e-14. Short-chain convergence is deliberately not
a startup gate.

Frozen150-file snapshot and plan:
`catalogue_prior/galsbi_measurement_init_20261001`. Launcher
`jobs/job_galsbi_initialization.sh` runs the study; GPU array16788277 has
four tasks with throttle2 and waits for successful startup and completion of
long array16781571 and cost/cache GPU continuation16787894. This preserves
the two-GPU study cap. CPU report16788283 runs after any pilot-array outcome.
`scripts/report_galsbi_initialization.py` generates within-arm prefix reports
and cross-arm likelihood-ratio agreement, initial-state and timing evidence.
It does not assume independent arm MCSEs under matched seeds. Startup has
been monitored through completion; GPU stages remain dependency-pending.

See [initializer comparison](CONTINUOUS_SAMPLING.md#measurement-informed-initialization-comparison--october-1-2026).
Higher initial likelihood is not posterior convergence. Retain Rhat<1.01,
bulkESS>400 and all numerical checks, including shear-relevant ratios; no
automatic promotion, population-scale launch or selected-shear claim.

## 2026-10-01 — Cost audit and validated CPU savings from invariant MH work reuse

Owner requested a deep efficiency investigation after the million-observation
cost estimate. Read the existing long-chain/mixing evidence and profiled the
actual model workload. Added `scripts/benchmark_galsbi_sampler.py` and
`jobs/job_galsbi_benchmark.sh`: controlled tiled-scene component timings,
identical-proposal MH replay, fixed4096 density batches, model/source receipts,
and CPU1/CPU4/CUDA response comparisons in the pending GPU arm. Warmup-only
cProfile does not contaminate reported timing repeats. These probes do not
establish convergence or representative million-observation throughput.

CPU job16787846 completed in2m09s, peakRSS928MB. For4/32/128 observations
(four chains each), weighted baseline component cost is249/276/369ms per
step. CPU neural density stays231ms due to padding, while response prediction
grows4.66/20.76/76.12ms and scene-input mapping6.66/12.24/31.39ms. All batch
and invariant replay checks pass, with no changed MH decisions. GPU timing
cannot be inferred by dividing this CPU measurement by an assumed factor.

Added opt-in `--reuse-invariant-work` to `scripts/check_galsbi_local_sampler.py`:
reuse K on z/SED-only moves (25% of the original mixture), reuse target prior
on neighbour-only moves, and avoid copying unchanged neighbours after target
moves. Full native support/Jacobians/Hastings factors remain active. Guard
requires the established persistent/joint-morphology configuration; invariant
blocks are periodically fully replayed at1e-10 summary/1e-6 logK tolerances.
The flag remains off by default. Reporter protocol compatibility now includes
the flag. Existing long jobs use frozen sources and are unchanged.

Added a full-scene redshift/SED invariance test with nontrivial photometry,
plus `scripts/check_galsbi_cache_replay.py` for paired512-step real-model runs.
Full suite16787870:368 passed,1 skipped in60.77s; Ruff and launcher syntax
checks pass. CPU replay16787880 failed safely against old GPU-derived starts
(2.35e-5 versus1e-6 tolerance). Retried with a fresh bank seed2026100133
evaluated on the same device; no tolerance was relaxed and failed outputs
remain preserved. Retry16787889 passed: exactly identical retained latent
states, MH decisions, log priors/kernels and local profiles; summary error
at most2.84e-14. Whole processes164.41s→132.82s; steps128–512 give249.74ms→
187.80ms, a24.8% loop-time reduction on CPU. The first frozen wrapper's
`sampling_seconds` includes initialization; current source instead derives
loop cost from progress differences. No GPU speedup or mixing improvement
is claimed from this test.

Study root `catalogue_prior/galsbi_cost_20261001` contains frozen sources,
plans, profiles and receipts. GPU benchmark16787881 waits for long task
16781571_2, preserving the two-GPU cap, and compares batch sizes through1024
observations and response devices. GPU paired replay16787894 follows the
benchmark and successful CPU paired replay. Startup was verified for the CPU
jobs; GPU stages remain dependency-pending. No automation wake-up tool is
available; dependencies provide execution, not an automatic agent review.

See [cost audit](CONTINUOUS_SAMPLING.md#cost-and-scalability-audit--october-1-2026)
for measured component costs, limitations and proposed next experiment.
Curved morphology and conditional environment trapping remain sampler issues;
prior pooled covariance/broad joint moves did not solve them. A small frozen
mixture independence proposal with an explicit MH correction is the proposed
next morphology test, not an implemented or accepted replacement. Production
also needs streamed summaries and a shear-level Monte Carlo error budget,
including finite log/ratio bias. No population-scale launch or0.3% claim.

## 2026-10-01 — Long-chain interim check: three observations pass within both completed replicas

Owner requested status. Tasks16781571_0/_1 completed with exit0 in5h10m32s
and5h10m57s. Their1,048,576-step results pass individual exploration gates
for obs0,2,3, but fail obs1 (measured MAG_AUTO21.2551). Seed2026100121
worst Rhat/minimum ESS by observation is1.00292/842.9,1.12420/24.7,
1.00280/855.4,1.00139/2664.3; seed2026100122 gives1.00328/703.2,
1.05566/54.1,1.00107/2429.8,1.00775/732.9. These are separate four-chain
replica checks, not the pending combined-replica result. New initial banks
also differ from the shorter historical runs, so this is not a paired
causal comparison of chain length alone.

Obs1 remains slow in radius/Sérsic/magnitude/crowding. Its two local log-ratio
Rhat values are1.0180/1.0257 and1.0336/1.0397 in the completed replicas;
the shear-relevant functions therefore also fail. Initial replay is exact.
SED invariance remains within the declared tolerance: maximum summary
change4.2633e-14, maximum log-kernel change9.3599e-7 in seed121 and zero in122.
Do not describe every SED evaluation as exactly unchanged.

Tasks2/3 are running normally at about685k/683k of1,048,576 steps, with
exact initial replay. Current throughput implies roughly two more hours,
including final diagnostics. Report16781574 still waits on the array.
No intervention, acceptance, or new job submission; final combined checks
and prefix diagnostics remain pending.

## 2026-10-01 — Owner-authorized million-step original-MH replicas

Owner explicitly requested longer chains after the runtime discussion. Use
the original per-chain3D morphology tuning, corrected SED jump and persistent
Poisson refresh; disable pooled covariance, joint-environment and local-
position variants. Fixed four-observation panel, four chains per observation,
four replicas with MH seeds2026100121–2026100124;1,048,576 total steps and
32,768 discarded warmup per replica. Run two GPUs concurrently, in two waves.
Measured original runtime implies roughly5.2 hours per wave and10.4 hours
overall, excluding queue/preparation/report time. Six-hour limit per GPU job;
24GiB host memory allows the longer unthinned traces. No batching change is
mixed into this numerical convergence experiment.

Added optional `--initial-bank-seed` to the existing runner. Each replica
draws8192 new prior scenes, splits them into four2048-scene streams and
chooses the highest-likelihood scene from each stream for each observation.
Initial bank seeds2026100221–2026100224 differ across replicas; generating
truth and mock observations are never regenerated or used as starting
states. The full-bank likelihood is replayed after selecting/reconstructing
the16 starting scenes, at the unchanged1e-6 tolerance. Save bank seed,
selected indices and likelihoods in `initialization.json`. Default behavior
still reconstructs the original reference bank.

Existing launchers now accept seed base, initial-bank seed base and warmup;
the report launcher accepts replica count and includes longer prefix checks.
No proposal, density or acceptance changes. Ruff/shell syntax pass. Full
suite16781551 submitted on compute3. Startup array16781558 depends on its
success, checks all four new banks with128 steps/64 warmup, and is throttled
to two GPUs. Smoke convergence is not an acceptance condition; execution,
likelihood replay and SED invariance are. Long-run submission follows these
checks. Source snapshot manifest SHA256:
`098c3993f7c6c1601d131b16ffcdbdd7b9d6db97e67e1bc0f5af8526630d4b8e`.
Protocol/output root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_long_mh_20261001`.
Post-run prefix diagnostics use65k/131k/262k/524k/1,048,576 steps; these are
computed after completion, not live convergence stopping. Gates remain
Rhat<1.01 and bulkESS>400 with independent and combined checks. No automatic
promotion, population run, selection normalization or shear-bias claim.

Full suite16781551 passed367 tests/1 skip in71.40s. All four smoke tasks
16781558 completed with exit0: starting likelihood replay error is exactly
zero for every bank; SED log-kernel change is zero and maximum summary
change2.8422e-14. The new-bank path therefore passed its scheduled runtime
validation in addition to the existing suite. GPU array16781571 is running
tasks0/1, with2/3 throttled behind them. Both running replicas passed exact
initial replay and reached768 steps with finite progress. Afterany CPU
report16781574 has32GiB/45min for the four larger traces. Sources and
scientific settings are frozen. No application wakeup tool is available;
Slurm enforces the sequential two-GPU waves and runs the completion report.

## 2026-10-01 — Pooled and joint-environment pilots completed; neither accepted

Owner requested a check. Scheduler accounting confirms pooled array16781077
completed in20m27s–20m31s and joint-environment array16781117 in20m41s–21m01s,
all exit0. Reports16781079/16781121 and comparison16781124 completed; user
queue is empty. Both variants and both seeds pass initial/SED numerical
checks, but every individual run and combined observation fails at least
one exploration gate. No sampler promotion or further job submission.

At matched65536 total steps/32768 warmup, worst combined Rhat for obs0–3:
original1.3009/1.3116/1.0489/1.0264;
pooled1.1175/1.4171/1.0324/1.0438;
pooled+joint1.2918/1.2254/1.0221/1.0211.
Minimum bulkESS respectively20.8/20.3/145.5/422.4,
46.0/17.0/213.9/146.4, and22.0/25.9/323.3/346.2.
Pooled tuning gives mixed results; joint moves improve several Rhat values,
but the faintest ESS remains below the original matched-budget result.
These two-seed development runs do not establish general superiority.

The joint move is often inefficient: faint-observation per-chain acceptance
is about0.1–3.0%. For obs1 one seed2026100101 chain accepts0/1966 joint
attempts. This supports the limitation of broad Gaussian target moves plus
an independently redrawn environment; formal invariance alone does not make
the proposal effective. Bright radius/Sérsic and obs1 magnitude/shape remain
slow. Joint-run profile Rhat atg1=.018/.022 is1.0527/1.0482 for obs0,
1.1026/1.1106 for obs1,1.01014/1.00986 for obs2 and1.00333/1.00334 for obs3.
Passing one ratio diagnostic does not override posterior failures.

Preserved comparison:
`galsbi_joint_env_mh_20261001/report/comparison.md` and `.json`, under the
catalogue-prior output root. No selected-population shear estimate or0.3%
claim. Before another scale-up, any next proposal should address the observed
curved morphology and conditional target/environment dependence, with a
small fixed-budget test and the unchanged acceptance gates.

## 2026-10-01 — Conditional environment trapping and a joint-move pilot

Owner requested continuation. Pooled array16781077 is still running; do not
interpret warmup acceptance as its result. During that run, inspected the
overnight chains using scheduled plot16781089 and tail audit16781093. Scripts,
figures (`geometry.png`, `traces.png`) and JSON are preserved in
`galsbi_mixing_audit_20261001`. Used the Matplotlib skill for these static
diagnostic plots. Plotting thins every32nd draw and trace panels show512-step
means; neither reduction is used for inference or convergence diagnostics.

The observed bright morphology distributions are curved, with directions
changing across Sérsic index; a single larger covariance is not guaranteed
to resolve them. Fainter observations have long excursions toward targets
much fainter than their measured magnitudes. Audited the diagnostic region
true r > measured MAG_AUTO+1 (no samples removed or new population cut).
For obs2, full environment refresh acceptance in this region is2/638 and
1/104 across the two seeds, versus roughly28–29% outside. Obs3 has1/41 and
3/418. Thus the earlier high aggregate acceptance conceals conditional
environment trapping. Tail occupancy ranges0.1–9.3% across obs2 chains and
0.16–11.0% across obs3 chains; these are unconverged occupancies, not inferred
population fractions. Absolute weighted neighbour-flux diagnostics remove
the target-flux denominator algebra: log(expm1(crowding))-0.4log(10)m_r.
Residual magnitude correlations remain substantial (roughly0.3–0.65), so
the apparent relationship is not solely the definition of flux ratios.

Added opt-in `--joint-environment` to the existing pilot, requiring the joint
morphology setup. A fixed20% of morphology attempts use a4-times broader
symmetric target step together with a full independent environment-prior
draw. They count within the existing block budget, use one likelihood
evaluation, and do not tune the scalar scale used by local moves. The
environment proposal cancels its own prior; acceptance keeps the full
target-coordinate prior and likelihood ratio. Whole scenes are retained on
rejection. No analytic neighbour-density approximation, truth cut, or
state-dependent decision to activate the move is introduced.

Save joint-move indicators and separate local/full/joint acceptance. Reporter
compatibility now includes the joint-move protocol. The existing launcher
accepts an optional joint-environment flag with defaults unchanged. Added a
joint target/Poisson-environment closure test against numerical quadrature,
including target/count cross moments and off-support rejection. Ruff and
shell syntax pass; full suite16781100 runs on compute3. Planned second pair
uses the same65k/32k protocol and seeds with pooled tuning plus this move,
after the first pair/report and passing tests, within two concurrent GPUs.
No acceptance or population-sized run follows automatically.

Suite16781100 completed:367 passed,1 skipped in53.58s. Frozen joint-pilot
source manifest SHA256 is
`bdb370a1d9cfcdab6b00816dca40461059495f16d23fa41dd5e598c1079c19d8`.
Queued array16781117 after the passing suite and pooled report16781079;
the two study GPUs cannot overlap the preceding pair. Afterany report16781121
and read-only fixed-budget comparison16781124 follow. Joint output root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_joint_env_mh_20261001`.
Its preserved `compare_reports.py` writes `report/comparison.md` and JSON
from the three source reports at65k/32k. Missing runs are recorded and no
sampler is promoted. The original and pooled pilots remain untouched;
all variant sources and random seeds are frozen separately.

## 2026-10-01 — Audit slow directions and test staged pooled morphology tuning

Owner requested a deeper sampler investigation and improvement. Overnight
receipts show thousands of full environment proposals, accepted roughly
27–75% for seed2026100101; simple failure to replace neighbours is therefore
not an adequate explanation for the remaining slow exploration. This does
not rule out rare unvisited environmental configurations. Started a scheduled
read-only audit of morphology proposal covariance relative to sampled
posterior covariance, and within-chain target/environment correlations.
Audit script and output are preserved under `galsbi_mixing_audit_20261001`.

Prepared opt-in `--pooled-morphology` in the existing local sampler and
`pooled_block_cholesky` in `sbsi/mcmc_proposals.py`. Pool retained warmup
states across the four chains of each observation only, using at most8192
steps, update every1024 steps, and freeze covariance halfway through warmup.
Continue scalar step-size adaptation through the remaining warmup, then
freeze all proposal parameters. The physical block remains magnitude,
log circularized radius and Sérsic logit, with5% diagonal shrinkage. No
likelihood, prior, environment law, block probability or acceptance rule
changes; no shared covariance between observations. This tests the suspected
short-window/per-chain tuning limitation, not a new population model.

Added an independent covariance calculation checking between-chain variation
and observation separation, plus a correlated-Gaussian MH closure test using
displaced warmup chains. Ruff passed. First CPU audit16781055 and suite16781064
were stuck before Python in `autofs_wait` on cip-cl-compute1; cancelled only
these two jobs and resubmitted on compute3 as16781065/16781066.

Predeclared comparison: if tests pass, run65536 steps/32768 warmup on all four
reference observations with seeds2026100101/2026100102 and original starting
scenes, matching the existing overnight65k prefixes. At most two GPUs.
Compare posterior and likelihood-ratio diagnostics for both seeds, not a
favorable shear point estimate. No automatic promotion or population run.

Audit16781065 completed in8s. It compares actual frozen proposal covariance
(including scalar multipliers) with each chain's observed post-warmup
morphology covariance, using eigenvalues after posterior whitening. For
obs1 the smallest ratios are0.00220–0.01777, with several0.002–0.007:
proposal standard deviations can be only5–8% of the sampled posterior width
along a slow combination. For obs0 the smallest ratio reaches0.00247.
This supports inefficient local proposal geometry, but sampled posterior
covariance is itself not a converged reference. Bright radius/Sérsic
correlations reach−0.81; faint magnitude/crowding reaches0.97 (partly the
algebra of target-relative flux). No claim that high environment acceptance
rules out rare tails or that covariance tuning alone solves the problem.

Full suite16781066:366 passed,1 skipped in77.16s. Ruff and launcher shell
syntax pass. Reused the overnight launcher with optional step count and
pooled flag, preserving defaults. Frozen143-file checkout manifest SHA256
`6f65101d430c897e5323dd28d0b113a62db597ca58bbdf17b60da416cf0f9a4e`.
GPU array16781077 runs the declared pair on two cip A40 GPUs; dependent
afterany report16781079 is assigned compute3, avoiding the observed mount
problem. Output root is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_pooled_mh_20261001`.
Separate seeds share covariance only among warmup chains for the same
observation within a run. Each proposal is symmetric after freezing, so no
new Hastings factor is required. The fixed experiment remains unaccepted
pending its individual/combined diagnostics.

## 2026-10-01 — Overnight comparison completed; convergence still fails

Owner requested status. Scheduler accounting confirms both tasks16778955
completed with exit0 in1h18m31s/1h18m35s; dependent report16778957 completed
in2m57s. No jobs remain in the user queue. Read the preserved report JSON
and Markdown under `galsbi_overnight_20261001/report`; both numerical audit
receipts pass and the runs are compatible, but neither individual run nor
the combined panel passes all exploration checks.

At262144 total steps, combined worst Rhat/minimum bulkESS by observation:
0:1.0301/381.9;1:1.0622/165.9;2:1.0112/552.2;3:1.0170/508.6.
Required Rhat<1.01 and bulkESS>400 remain unchanged. Main slow summaries
are circularized radius for obs0, Sérsic/magnitude for obs1, and
crowding/magnitude for obs2–3. The endpoints improve over the65k prefix,
but the faintest observation worsens at131k before improving: short-run
agreement cannot establish convergence.

Combined likelihood-scale ratio Rhat atg1=.018/.022 passes for obs0,2,3
(approximately1.006/1.003/1.003), but fails for obs1 (1.0125/1.0162).
Conditional relative MCSE is0.14–0.17% for the brighter pair and
0.008–0.014% for the fainter pair. These are errors of individual-observation
numerator ratios, not shear bias or guaranteed error bars for unconverged
chains. Largest individual draw fraction is at most5.51e-6 in the final
combined profiles: no observed single-draw dominance, without ruling out
unvisited tails. Bright obs0'sg1=.022 log mean ratio changes+0.00215 at131k
to−0.000944 at262k; its small profile slope remains numerically uncertain.

No acceptance, model change or population-scale launch. Next bounded
diagnostic should examine the slow magnitude/size/Sérsic and environment
directions on this existing panel before increasing observation count.
The missing selection normalization and population-law limitations remain.

## 2026-10-01 — Frozen overnight sampler comparison launched

Completed the optional local-position pilot16778924 in270.97s after
suite16778902 passed362 tests/1 skip. Bright maximum Rhat1.3893 and
numerator-ratio Rhat1.0537; faint maximum1.0634 and ratio1.0261. High local
position acceptance does not establish convergence. Neither this variant nor
the five-coordinate block consistently improves the simpler three-coordinate
joint morphology proposal. No pilot passes the full convergence gates.

The overnight protocol therefore retains the SED jump, three-coordinate
joint morphology, separate shape block, exact type sum and persistent Poisson
environment refresh. No local-position or five-coordinate proposal is used.
Added saved local profile draws and the read-only
`scripts/report_galsbi_sampler_runs.py`, with likelihood-scale MCSE and
largest-draw contributions, individual and combined-seed diagnostics at fixed
prefixes, and explicit incomplete-run handling. Tests cover missing/corrupt
receipts, disagreeing replicas, an IID lognormal MCSE oracle, and the complete
combined-prefix report path. Latest suite16778930:364 passed,1 skipped
in51.77s; Ruff and launcher shell syntax passed.

GPU array16778955 started both tasks on separate cip A40 allocations, after
the passing suite. Each uses four observations, four chains per observation,
262144 steps/32768 discarded warmup, with seeds2026100101/2026100102.
The seeds share the same dispersed reference-prior starting scenes; they do
not independently regenerate initialization. Both initial likelihood replays
have zero error, and both passed8576 steps with ongoing accepted SED-region
transitions. Adaptation is restricted to warmup. Two concurrent GPUs maximum,
four-hour allocation limit, fixed run length and no favorable-result stopping.

Run root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_overnight_20261001`.
`plan.json` records the fixed protocol. A143-file source snapshot is checked
at startup; SHA256 of its manifest is
`245f0e8696512dfbed8997cbb289490dc7e270424b8c561d1143824f0bb9fa6a`.
Launchers are `jobs/job_galsbi_overnight.sh` and
`jobs/job_galsbi_overnight_report.sh`. CPU report16778957 depends on
**afterany:16778955**, so incomplete runs are reported rather than silently
omitted. It writes `report/report.md` and `report/report.json`, comparing total
step prefixes65536/131072/262144. No automatic promotion or population run.

These are posterior-integration stress tests under the fixed F24 target and
implemented Poisson generator. They do not supply the measured-selection
normalizer, an accepted shear estimate, or evidence of0.3% calibration.
Mixing and likelihood-ratio overlap must be assessed separately. Slurm and
the dependent report provide the overnight continuation; no agent wakeup
automation is available. Next: inspect the completed independent-seed report
before deciding whether to change proposals or expand the observation panel.
See [overnight protocol](CONTINUOUS_SAMPLING.md#overnight-comparison--october-1-2026).

## 2026-10-01 — Overnight continuation: joint morphology proposals

Owner explicitly requested autonomous overnight work. Continue small controlled
sampler tests, preserving the posterior, V3.9 artifacts, measured-cut contract
and convergence gates. No population-scale inference until numerical checks
pass. Use at most two study GPUs concurrently for the planned independent
replicates, within existing limits; check the queue before submission.

Added `sbsi/mcmc_proposals.py` and opt-in `--joint-morphology` in the existing
pilot. Magnitude, log angular radius and Sérsic logit now move jointly with
per-chain Gaussian covariance. During discarded warmup only, recompute
covariance every128 steps from at most1024 retained states, including rejects,
shrunk5% toward the original diagonal proposal. Freeze both covariance and
scalar scales after warmup. Remove Sérsic from the separate structure block;
the corrected SED jump, exact type sum and persistent-neighbour law remain.
This is a symmetric proposal after freezing, with unchanged target density.

Added an independent correlated-Gaussian posterior moment test and a stuck-
chain covariance support test. Ruff passed; full CPU suite16778860 submitted.
Initial joint pilot will use the same two observations, four chains each,
seed2026093061 and16384 steps/4096 warmup, output
`galsbi_joint_mh_20261001_v1`. Longer independent-seed diagnostics will be
predeclared only after this implementation test; no shear claim follows from
one favorable run. Native-density/generator approximation and F24 versus FS2
population limitations remain unchanged.

Full suite16778860 passed355 tests/1 skip in53.08s. Joint-morphology GPU
16778863 completed in289.83s script time. SED likelihood replay stays exactly
unchanged. Bright Sérsic Rhat improves1.5730→1.1415, but radius1.2836 and
shape1.2165/1.2797 remain poor; its shear-ratio Rhat is1.0581. Faint maximum
Rhat1.0625 remains in crowding; ratio Rhat1.0094 passes but overall exploration
does not. Do not select this run from its favorable small shear-ratio means.

Extended the same Gaussian block to optionally include distortion logit and
wrapped phase (`--joint-shape`). Total physical-proposal probability stays0.6,
previously split between morphology and shape. Estimate phase covariance in
the circular chart centered on each chain's last warmup phase; Gaussian
increments are wrapped afterward. With fixed covariance the wrapped proposal
is symmetric. Added phase-origin invariance and correlated wrapped-Gaussian
moment tests. Full suite16778895 submitted. The next16k pilot keeps all other
settings, output `galsbi_joint_shape_20261001_v1`.

Also added explicit reference-observation indices and saved local shear-ratio
draws to the existing driver for the forthcoming independent-seed/prefix
checks. Default observations and scientific settings are unchanged. These
diagnostic options were added after the running v1 source snapshot was taken;
that completed run retains its exact executed source under its output folder.

Full five-coordinate physical pilot16778896 completed in278.61s, after
357 tests/1 skip passed (16778895). It does not consistently improve the
bright result: maximum Rhat1.3724, ratio Rhat1.2389; faint maximum1.0646 and
ratio1.0045. Both fail overall. Keep it optional rather than choosing it from
the favorable faint result. All starting/SED numerical replays pass.

Added a read-only independent-run/prefix reporter, including between-seed
Rhat/ESS and likelihood-scale MCSEs of numerator ratios. Missing/incompatible
runs cannot pass, and no sampler is promoted automatically. Tests explicitly
include two individually well-mixed but mutually disagreeing replicas.
Report test16778898 initially failed on a script-only import; the reporter
now has no model imports. Full suite16778899 then passed360 tests/1 skip;
removed its NumPy scalar-conversion deprecation afterward.

Next small test: optional local neighbour-position moves, retaining the
three-coordinate joint morphology block. `GalSBIScenePrior.move_positions`
proposes independent Gaussian steps for all positions and retains each point
whose proposal leaves the aperture. Each bounded uniform-disk kernel is
prior-reversible, as is their product; outer MH uses only the likelihood
ratio. Counts, marks, target and population law stay unchanged. Added
stationary-disk and analytically tilted-disk posterior tests. The new block
gets0.05 probability, taken from the SED jump (now0.10); original refresh
probability0.15 stays fixed. Position scale0.2arcsec, same16k/4k protocol.
Full suite16778902 submitted; output `galsbi_local_positions_20261001_v1`.

## 2026-10-01 — Add a corrected SED independence jump to the continuous sampler

Owner authorized the proposed red/blue SED move. Added `sed_log_proposal`
and `propose_sed` to `sbsi/galsbi_target_coordinates.py`. At fixed redshift,
propose from an equal mixture of the two known Dirichlet laws and preserve
apparent r magnitude, angular radius, Sérsic n, distortion and phase. Decode
new intrinsic luminosity/photometry and evaluate the complete native prior.
The proposal density includes the simplex-to-ALR Jacobian prod(c), and MH
uses log q(old|z)-log q(new|z). Equal proposal weights do not change target
population weights. Off-support proposals are rejected without redrawing.

The existing driver gains opt-in `--sed-jump`, requiring persistent scenes
and exact type summation. Block probabilities become0.3 photometry/size,
0.3 shape,0.1 local structure,0.15 environment and0.15 SED independence.
Every finite-prior SED proposal verifies all nine physical summaries remain
fixed within1e-10 and log likelihood within1e-6; failures stop the pilot.
Output now includes native log prior, SED-region crossings, conditional red
probabilities and separate full/partial environment acceptance. No model,
prior density, inference default or measurement-selection rule changes.

Added independent SciPy proposal-density/Jacobian checks, an analytically
soluble unequal-mixture posterior with a likelihood tilt, and a photometric
replay/support test. Ruff and shell syntax pass. Full Slurm suite16778800
submitted with artifact roots configured. Planned pilot keeps the same two
mock measurements, four chains each, seed2026093061,16384 steps/4096 warmup,
per-chain frozen scales and existing neighbour law. It will depend on the
suite passing. Source/protocol receipts and results are preserved under
`galsbi_sed_jump_20261001_v1`. Full suite16778800 completed:353 passed,1
skipped in52.28s. Dependent GPU16778801 started and exact initial likelihood
replay passed; all eight chains have made SED-region crossings during warmup.
GPU16778801 completed in294.96s script time (5m24s allocation). Across19,808
finite-prior SED proposals, maximum summary change is4.2633e-14 and maximum
log-likelihood change is exactly zero. CPU audit16778804 completed in2s.

The SED-region trapping improves markedly, but **neither observation passes**.
After warmup the four bright chains make24/28/28/86 region crossings, with
mean conditional red probabilities0.9912/0.9793/0.9804/0.9561; previously two
chains stayed red and two blue. Full environment refresh acceptance is now
75.3/86.1/77.7/75.3%, versus a zero-acceptance bright chain previously.
Bright SED ALR rank-Rhat is1.0029–1.0106, bulkESS525–1821. Its largest remaining
Rhat is1.5730 for Sérsic index (bulkESS6.94), and circularized radius is1.3641
(ESS9.13). Theg1=.022 numerator-ratio Rhat improves from1.4675 to1.0489,
but its per-chain log ratios still range−0.01379 to+0.00780 and are not accepted.

Faint maximum Rhat is1.0629 and minimum bulkESS63.07; magnitude/crowding
remain slow. Itsg1=.022 ratio Rhat is1.0106 withESS445.94. Conditional red
probability differs0.0004–0.0744 across faint chains, so crossings alone do
not establish correct component mass. Shape/SED diagnostics improve, but
joint posterior convergence is unresolved. This remains a two-observation
diagnostic, without a selection integral, shear estimate or0.3% claim.

Read-only correlation audit16778808 examines the remaining structural
coupling, using saved traces; it completed in1s with no extra inference run.
Within bright chains, log angular radius and Sérsic logit correlate−0.43 to
−0.80, but current proposals put them in different blocks. Faint magnitude
and close-crowding correlations range0.55–0.98; crowding uses target-relative
flux, so this is partly structural and is not by itself proof of trapping.
The next minimal test is a joint magnitude/size/Sérsic proposal with covariance
adapted during warmup and frozen afterward, retaining the corrected SED jump.
Do not infer that this will resolve environment exploration or authorize a
population-sized run from the favorable SED result. All jobs are complete.
Results and next step are recorded in
[SED-jump pilot](CONTINUOUS_SAMPLING.md#sed-independence-jump--october-1-2026).

## 2026-10-01 — Retain neighbour environments during local MH target moves

Owner authorized the proposed next sampler step. Added a prior-reversible
partial marked-Poisson refresh and complete ragged-scene selection/retention
to `sbsi/galsbi_scene_prior.py`. Extended the existing local pilot with
`--environment-update persistent`: one explicit environment per chain,
unchanged during target moves, with separate refresh moves. A refresh retains
each neighbour with probability0.9 and adds an independent0.1-intensity
population;10% of refresh proposals instead redraw the complete environment.
The fraction choice is independent of state. Random order within each patch
preserves the ordered-iid convention. Acceptance uses the likelihood ratio
for a refresh, and the target-prior times likelihood ratio for local target
moves. Rejected states retain every target and neighbour field.

Bounded protocol: same two independent model-generated bright/faint mock
observations and four chains each;4096 steps including1024 warmup, exact
red/blue target sum, fixed4096 density batching, seed2026093061. Initial scenes
are the best measurement-likelihood candidate in each previous prior stream;
their original environments are retained, and no generating truth is used.
This changes initialization, environment update and run length together; it
is an algorithm refinement, not an isolated causal comparison. Initial full
scene replay must agree with saved reference log kernels within1e-6.
All native coordinates except wrapped phase, neighbour counts and all three
crowding summaries are now monitored in addition to the existing diagnostics.
Rank-Rhat<1.01 and bulkESS>400 remain necessary exploration gates.

Added tests of ragged-state retention, refresh prior stationarity, count
cross-covariance, and an analytically tilted marked-Poisson posterior whose
count and mark laws are known. Also fixed empty-environment field creation:
neighbour fields come from the galaxy generator, not a potentially different
supplied target schema. Ruff and launcher shell syntax pass. Full CPU suite
16778740 passed:348 tests passed,3 skipped in55.48s. Dependent GPU16778741
failed before scientific work because the resource detector's output directory
did not exist; the launcher now creates it. Identical pilot resubmitted as
16778742. Source snapshots,
protocol hashes, final native scenes and resource receipts are preserved.

No production default, model identity, selection integral or shear estimate
changes. The native analytic target prior and numerical environment generator
retain their existing approximation difference.

First pilot16778742 completed in89.68s script time (2m03s allocation).
Starting replay is exactly equal to the reference. Both observations fail:
bright maximum Rhat2.860/minimum bulkESS4.775; faint1.097/34.88.
Bright magnitude/size acceptance ranges0.99–27.92% across chains, despite
using a shared step scale, and the local numerator ratios disagree strongly.
Faint ratios atg1=.022 span0.00414–0.00670 in log ratio; this is neither a
selected likelihood nor a shear estimate. The change removes the independent
likelihood-estimator noise but does not establish posterior exploration.

Added the conventional per-chain warmup-scale option to the existing driver
and launched one bounded follow-up16778765, output
`galsbi_persistent_mh_20261001_v2`:16384 steps/4096 warmup, same eight chains,
observations, seeds, posterior, starting scenes and environment-refresh law.
Scale adaptation freezes after warmup. This tests whether proposal scales
better match the different chains; it also increases length, so is not a
single-factor causal comparison. Ruff passes after this script-only option.

Follow-up16778765 completed in272.29s script time (4m45s allocation), with
exact starting replay. **Neither observation passes**. Bright maximum
Rhat2.2307/minimum bulkESS5.896; faint1.0921/39.20. Bright magnitude/size
acceptance is now18.8–32.1%, but shape acceptance is5.1–33.3% and chains still
disagree on the local shear numerator ratios. Faint ratio bulkESS493/511 is
better, but ratio Rhat1.0137/1.0129 and other diagnostics still fail.

Read-only CPU audit16778767 completed in2s. Its script and
`chain_mode_summary.json` are preserved in the v2 output. Bright chains0/3
have mean conditional red probability>0.99999998; chains1/2 have probabilities
near zero. Summing type analytically did not remove the separated SED regions.
Chain1 samples a much broader intrinsic-magnitude range (10–90% quantiles
19.35–22.39) than chain0 (19.71–19.75). It accepts0/181 full environment
refreshes, despite89.3% acceptance of partial refreshes. The other bright
chains accept67–77% of full refreshes. These are nonconverged chain summaries,
not estimated posterior population weights; do not select the chain with the
largest likelihood and discard the rest.

Conclusion: explicit scenes remove independent likelihood-estimator noise,
but local MH still fails to explore the joint target/environment posterior.
No accepted sampler, selected shear estimate or0.3% claim. No jobs remain.
The next smallest targeted change is an independence proposal for SED from
the known red/blue Dirichlet mixture, at fixed z and likelihood-visible target
properties, with its exact proposal-density/Jacobian correction. This can
jump SED regions without fitting a new prior. Environmental trapping remains
a separate check, potentially requiring local moves of existing neighbours.
See [persistent-scene MH](CONTINUOUS_SAMPLING.md#persistent-scene-mh-pilot--october-1-2026).

## 2026-09-30 — Implement fixed-summary likelihood and test standard MH variants

Owner requested rethinking the sampler around the likelihood's fixed summaries
and then explicitly asked to continue. Added `sbsi/galsbi_summary.py`,
`sbsi/summary_sampling.py`, `scripts/check_galsbi_summary_sampler.py` and
`jobs/job_galsbi_summary_sampler.sh`. For fixed observed radius/flux, the eight
F8/C8 features plus the summed disk response reproduce the full retained
likelihood. The response coefficient is observation-indexed, and stale
auxiliaries are rejected. All neighbours enter crowding; the existing response
uses nearest20 within strict10arcsec, without truth cuts. No model retraining,
emulator simplification, fitted summary density or production default change.

Established the previously unresolved native radius mapping by inspecting the
installed UFig photon-rendering matrix: semi-major=R_native*sqrt(1+a),
circularized=R_native*(1-a^2)^(1/4), with native distortion converted to epsilon.
Independent SVD replay of the actual rendering matrix agrees to2.22e-16.
This explicitly adopts the zero-flexion rendered ellipse, not an empirical
calibration or direct identification of native and circularized radii.

The induced nine-summary prior is available by fresh scene generation and
compression. It is not generally a smooth Lebesgue density: the tree response
can have point masses, as can the no-neighbour event. Ordinary independence
MH with fresh prior proposals avoids evaluating that density because proposal
and prior cancel. Rejections remain in each chain; no generating truth is
inserted and no finite catalogue is resampled indefinitely.

Completed summary pilot16776331, artifact `galsbi_summary_mh_20260930_v2` under
the external catalogue-prior root, script time28.50s.8192 independent model
inputs yielded3188 usable and1961 measured-cut-passing mocks. Four measured
magnitude strata were represented, and each observation used four independent
2048-draw fresh-prior streams. Full-pair versus compressed likelihood,
reversed order and padded partial batches agree exactly. Initial job16776323
stopped before MH because C8 in8192 versus4096-row batches differed by3.98e-6
in log probability; rerun matched classifier batching, retaining the1e-6 gate,
seeds, observations and proposal streams. No threshold was loosened.

All four independence-MH probes **failed exploration**. Prior importanceESS
per2048 draws was1.00–3.08 at measuredmag19.745,1.05–3.76 at21.255,
2.90–11.32 at23.971, and20.45–24.86 at25.118. Bright chains were nearly frozen;
faint acceptance was0.68–2.05%. These are not recovered shear estimates.

Continued with standard pseudo-marginal MH, adding
`sbsi/galsbi_target_coordinates.py`, `sbsi/pseudo_marginal.py`,
`scripts/check_galsbi_local_sampler.py`, and `jobs/job_galsbi_local_sampler.sh`.
Refactored the existing neighbour generator to expose `sample_given_targets`,
preserving its original full-scene RNG sequence. Ten transformed target
coordinates retain the evaluable native prior and all Jacobians; fresh
neighbours supply a nonnegative likelihood average. Rejection retains the
current average and compressed bank. Bank moments use likelihood weights,
and shear numerator ratios use ratios of bank sums, not equal scene weights.
Measurement-selection normalization remains separate and uncomputed here.

First local pilot16776398 completed in51.50s script time: same bright/faint
measurements, four chains each,1024 steps/512 warmup, four environments per
proposal. Local moves improve acceptance but convergence still fails; two
bright chains stick. Independent fixed-target audit16776440 used1024 fresh
environments per chain: those two chains' stored log-likelihood estimates
exceed the audit means by7.59 and8.63; four-environment log-estimate SD is16.86
and17.86. At64 environments SD falls to2.31 and1.76. The1024-draw audit itself
has only2.82 and4.47 effective environments at those targets, so its mean is
not treated as exact. This is demonstrated estimator noise, not evidence of
recovered posterior mass.

An exact additional simplification sums red/blue native prior intensities:
type is not a direct F8/C8/disk input at fixed physical properties, so it need
not be a discrete MH variable. Added `log_mixture_prior` and a test against the
explicit native sum. The bounded refined pilot16776459 uses64 environments,
the exact type sum, and separate proposal-scale adaptation per observation
during warmup only. It keeps the same two measurements and1024 steps. Its
protocol and source snapshot are in `galsbi_local_mh_20260930_v2`. It completed
in330.48s script time, but both observations still fail. Bright rank-Rhat is
1.79–3.11 and bulkESS4.56–55.91 over monitored functions; faint is1.07–1.50
and7.81–51.42. Faint photometry/size and shape acceptance reaches23–44%, but
the chains still disagree. At two bright final targets the independent
1024-environment audit has only5.73/12.17 effective environments, and retained
log-estimates exceed its mean estimates by8.36/3.89. Faint M64 log-estimate SD
is0.13–0.22, showing that target mixing also needs work. This is an algorithm
refinement, not an isolated causal comparison of environment count alone.
Retain exact summary compression, reject both short sampler pilots as inference
replacements. No additional or large jobs remain. Next is proposal design
that can retain useful target/environment configurations; do not interpret
acceptance or a favorable shear proposal as convergence.

Validation: analytic Gaussian posterior/evidence-ratio and point-mass tests,
independent full ten-dimensional Jacobian determinant, noisy latent-Gaussian
pseudo-marginal closure, rejected auxiliary retention, neighbour law with fixed
targets, and likelihood replay. Full Slurm suite16776297:339 passed/1 skipped;
16776388:343/1;16776457 after type marginalization:344/1 in43.53s. Ruff and shell
syntax checks pass. Each main GPU pilot uses one GPU, one CPU,16GB and a20min
cap; the separate noise audit uses8GB/5minutes. No study GPU jobs overlap.
Resource receipts are captured in the main pilot allocations.

Limits: native target density and numerical environment generator retain their
audited grid approximation; fixed F24 population/uniform neighbours differ
from FS2. No representative-population closure, measured-selection integral,
accepted shear estimate or0.3% claim. Design, equations, protocols and results:
[continuous sampling](CONTINUOUS_SAMPLING.md#fixed-summary-prototype--september-30-2026).

## 2026-09-30 — Design continuous-scene posterior sampling before implementation

Owner requested thinking through the next sampling algorithm. Added
[the proposed design](CONTINUOUS_SAMPLING.md#status-and-recommendation--september-30-2026),
after inspecting the native prior, V3.9 disk density/response code, selection
contract and earlier profile/batch failures. This is documentation only;
no sampler, model, inference default or job changed.

Recommend tempered SMC with one blocked Metropolis kernel, posterior scene
reuse for nearby shear likelihood ratios, and an independent CRN forward
selection integral. Derived the partial Poisson refresh proposal: independently
retain1-rho of neighbours, add a fresh rho-intensity population by envelope
thinning, randomize list order. Its prior reversibility cancels prior/proposal
terms, leaving the tempered likelihood ratio; no separately estimated surface
density or custom birth/death factor is needed. Local target/neighbour moves
still require native/transformed-coordinate Jacobians and support checks.

Recorded the full q_U*f_disk posterior target, the population Z(g) selection
factor, reuse of observed-radius/flux response across shear nodes, the remaining
native-to-SBSI mapping, common-support and batching checks, and why simple
gradient methods or independently noisy Newton stencils are poor first choices.
Validation progresses from analytic/replay tests to two observations,32 stress
probes, then representative continuous-model mocks. Proposal budgets are
diagnostic; neither high ESS nor favorable shear is an acceptance guarantee.
The prospective global numerical target is1e-5 absolute shear versus6e-5
corresponding to0.3% atg1=0.02, not a claimed achieved precision.

Validation: checked formulas against the local likelihood contracts and primary
SMC/Poisson/MCMC-diagnostic references; reviewed the Poisson exchangeability
argument and posterior-ratio/selection identities. Documentation links and
whitespace checked. No numerical experiment or speed claim; next step is a
small arbitrary-scene likelihood adapter with pointwise numerical replay.
## 2026-09-30 — Add uniform GalSBI neighbour environments

Owner authorized adding neighbours to the continuous prior, retaining the
previously requested uniform, unclustered approximation. Added
`sbsi/galsbi_scene_prior.py`: independent target-centred 10-arcsec patches,
native galaxy marks, uncapped Poisson neighbour counts, Cartesian sky offsets,
and explicit scene density. Targets and neighbours use the same fixed F24
population/support. Neighbour sampling thins the pre-photometric Poisson
population; no Monte Carlo post-cut normalization enters the sampler.
`GalSBIContinuousPrior.sample` now returns retained `proposal_index` values
to preserve patch membership through this thinning.

The ordered-list density includes the count factorial and per-steradian to
per-arcsec-squared measure conversion. `log_kernel` omits only constants at
fixed population/aperture; `log_prob` accepts an explicit uncertain surface
density. Derived physical Gaussian crowding matches the F8/C8 definition,
using all nonself neighbours within10 arcsec. Native pair marks and offsets
are retained; emulator radius mapping and predictions are not implemented.

Added `scripts/check_galsbi_neighbours.py`, CPU launcher
`jobs/job_galsbi_neighbours.sh`, and `tests/test_galsbi_scene_prior.py` covering
Poisson measure/count changes, rejection thinning, zero-neighbour scenes,
support and existing V3.9 feature replay. The diagnostic compares4,096
continuous patches with disjoint patches made by uniformly placing the
previous independent GalSBI catalogue marks/count, without row reuse. This
reference deliberately does not use its original HEALPix sky positions.

CPU job16775918 passed the full suite:331 passed,3 skipped in59.83s.
Comparison16775919 passed all eight declared gates. After randomizing the
reference list order (GalSBI stores types in blocks), final replay16775946
completed exit0 in48s and again passed all gates; counts, marks, spatial
checks and gate decisions are unchanged. No seeds/thresholds were tuned.
Artifacts: `galsbi_neighbours_20260930_v1` under the external catalogue-prior
root. Resources recorded in the allocation; one CPU/16GB requested for the
comparison despite greater host-wide availability. No GPU or inference job.
Scientific seeds are2026093021 (continuous scenes),2026093022 (reference
positions). Ruff, shell syntax and AST checks pass; final source snapshots
and comparison hashes verified, comparison figure inspected.

The4,096 scenes contain87,622 neighbours (mean21.392 versus21.540 predicted,
-1.62 standard deviations including normalization uncertainty), with count
variance/mean0.99429. The424 disjoint reference patches contain9,134 galaxies;
2,535 square-cell-corner rows are excluded and every retained row is used once.
Minimum spatial raw/adjusted p is0.03464/0.13854; minimum neighbour-mark raw p
is0.08022. Paired crowding means are compatible at allthree scales, and first64
scenes replay production crowding within4.44e-16. Every scene density is finite.
Detailed evidence and usage:
[neighbour report](GALSBI_PRIOR_FEASIBILITY.md#uniform-neighbour-pilot-results--september-30-2026).

Limitations: unclustered independent patches at fixed population parameters
and declared intrinsic support; the reference reassigns uniform positions.
Absolute scene normalization still carries the isolated Monte Carlo error,
while fixed-population density ratios and neighbour sampling avoid that
estimate. No emulator input mapping, posterior sampling, measured-selection
normalization or shear validation. Next is coordinate mapping and a small
inference closure test, not a0.3% claim.

## 2026-09-30 — Implement an isolated GalSBI continuous-prior pilot

Owner authorized an isolated-galaxy density implementation and comparison
against a GalSBI-generated catalogue, without neighbours. Added
`sbsi/galsbi_prior.py` with explicit F24 native-coordinate density factors,
support checks, continuous luminosity/redshift sampling, and an explicitly
uncertain population normalization. Added the diagnostic
`scripts/check_galsbi_continuous_prior.py`, its CPU Slurm launcher
`jobs/job_galsbi_prior_check.sh`, and measure/support tests in
`tests/test_galsbi_prior.py`. No V3.9 model or inference default changes.

The reference uses installed GalSBI0.3.0, F24 posterior row0 (accepted row,
source id8), HSC r2/i2 filters, zero foreground extinction, float64 catalogue
output and intrinsic reference-i limits14–28. It includes no measured cuts,
images, neighbours or shear inference. The analytic adapter is in the native
latent variables; it is not yet a marginal density in SBSI's five features.
The source package differs from the latest upstream checkout, so the actual
installed source and data hashes are recorded by the comparison.

Artifacts are under `galsbi_continuous_prior_20260930_v1` in the external
catalogue-prior root. Preflight CPU job16775773 generated11,669 reference
galaxies successfully and recorded resources. Initial comparison16775826
failed before generation because SBSI was not installed into py31 for script
imports; installing this checkout editable follows ENVIRONMENT.md and avoids
adding PYTHONPATH workarounds. Job16775840 generated the reference, then
exposed the installed photometry wrapper's keyword-only interface; repaired
the callback. Job16775846 completed the calculations but exposed NumPy-bool
JSON serialization; explicitly converted gate values to Python bool. No
scientific settings, seeds or acceptance thresholds were changed after seeing
results. Final comparison16775868 completed exit0 in40s on one inter CPU,
about0.80GB reported peak RSS. No GPU used.

All nine pilot gates passed: 11,669 GalSBI galaxies versus70,507 independently
retained continuous draws from131,072 proposals; magnitude replay max1.42e-14
mag; every density finite; smallest raw two-sample p0.07295 across15 marginal,
five coarse joint and one type-fraction checks. Morphology conditional-CDF
minimum raw p0.04835 (shape), Holm-corrected0.19338. Expected reference count
11,652.83 versus11,669 actual (0.144 standard deviations). Population
normalization is2.91704e9 sr^-1 with0.256% Monte Carlo relative SE, unrelated
to shear accuracy. Luminosity grid change1.81e-9 and magnitude inverse-CDF
error2.35e-8. Results, support and usage are recorded in
[the pilot report](GALSBI_PRIOR_FEASIBILITY.md#isolated-galaxy-pilot-results--september-30-2026).

Validation: full-suite job16775847 gives328 passed,3 skipped in43.31s;
`ruff check sbsi/galsbi_prior.py scripts/check_galsbi_continuous_prior.py
tests/test_galsbi_prior.py` and `bash -n jobs/job_galsbi_prior_check.sh` pass.
Final source snapshots and comparison hashes verified, both plots inspected.
Generation metadata must match comparison arguments; any failed diagnostic
writes evidence and exits unsuccessfully. Tiny samples may omit a galaxy
type; the sampler handles this and rejects nonpositive proposal counts.

Limitations: fixed F24 posterior row and native latent density only; shared
nearest-grid photometry; no five-feature marginal, SBSI radius/shape mapping,
neighbours, measured selection, MCMC or shear calibration. The matched
M_B<=-4 population boundary carries GalSBI's own incompleteness warning;
support completeness and rare bright tails remain unvalidated. The next
inference step needs explicit coordinate mapping and sampler/selection
closure. No sub-percent equivalence or0.3% calibration claim is established.

## 2026-09-30 — Audit GalSBI as a continuous population prior

Owner asked whether GalSBI can evaluate the density of arbitrary true
galaxies, allowing uniform neighbour positions and excluding its clustering
extension. Added [the source-level feasibility audit](GALSBI_PRIOR_FEASIBILITY.md#scope-and-verdict--september-30-2026).
Inspected the public implementation at commit
`df802e1db2f2a4247955720aea227d5b65cdf975` and the non-clustering F24 model,
alongside the primary paper and official documentation. The checkout is
`/tmp/sbsi-galsbi-audit-20260930`; no upstream modules were executed or
dependencies/data installed. No production code, model, prior or job changed.

The package returns ABC population-parameter samples and generates galaxies;
no full galaxy `log_prob` interface was found. Explicit Schechter, Dirichlet,
lognormal, scaled-beta and truncated-beta-prime factors support constructing
a native joint density. A five-dimensional SBSI truth-density evaluator
additionally needs marginalization over redshift, type and SED. The proposed
alternative keeps these nuisance variables in the sampler and uses the
existing generative factors rather than first fitting a Gaussian mixture.

Recorded the uniform marked-Poisson neighbour factorization, shared population
parameters, intensity versus probability normalization, Sérsic truncation,
shape/size/filter conventions, and F24 default intrinsic i-band limits.
Source inspection also finds grid-discretized redshift/magnitude generation
and nearest-grid spectral lookup: continuous-model closure cannot silently
assume the released sampler is exact. GalSBI is a different parent from FS2;
success on GalSBI mocks would not itself establish recovery on ConstGold.

Validation: source API searches and AST inspection of nine upstream Python
modules passed; documentation whitespace and relative links were checked.
This is feasibility analysis only: no complete density normalization,
posterior bundle inspection, generated-catalogue comparison, MCMC benchmark
or calibration result. Next step is a bounded native-density adapter and
generator-consistency audit before an inference pilot.

## 2026-09-30 — Trace the expanded-prior large Newton proposal to cancelling noisy curvature

Owner requested a deep investigation of the full matched-mock proposals
0.18813134 and0.14730972. Added the read-only saved-moment reduction
`scripts/audit_v39_expanded_step.py`, frozen with outputs under
`inference_v39_expanded_step_audit_20260930_v1` in the external catalogue-prior
root. No model, prior, observation, sampler, or production default changed.
CPU job16773458 completed exit0 in19s on one inter CPU with8GB requested;
no GPU jobs were submitted. The resource receipt is retained in the run root.

The audit verifies50 source hashes, original20k row coverage and exclusion of
the3686 mock padding rows, primary/maximum-rung equality, and all six saved
mock and actual aggregate scores/information. Independent2x2 solves and
influence covariance reproduce all six positive mock proposals and their
reported standard errors. Independent differences of the saved selection-mass
nodes reproduce the normalization gradient/Hessian. Exact row-attribution
identities reproduce the512→8k,8k→16k and512→16k changes. Ruff and AST parsing
pass; numerical validation is against the frozen scientific arrays rather
than new tests duplicating the reduction.

At16k the mock weak-direction information is the residual of+5.200966m and
−5.088162m: only1.096% of the sum of absolute contributions remains. This is
scientific/numerical cancellation, not evidence of a floating-point failure
in the2x2 solve (its condition number is about15.3). The134 measured-mag<20
rows account for82.58% of the512→16k g1 movement, while774 mag20–22 rows
supply80.98% of the final g1 score. Thus bright integration errors change the
curvature used to convert other observations' scores into a shear step.
The16k conditional g1 influence error is0.90423; the0.14356 step length is
143.6 times the0.001 derivative stencil. No valid maximum was located.

Replayed the earlier three exhaustive worker files (complete28,012,032-atom
coverage) and both normalizers from raw arrays; receipt
`exact_probe_replay.json` preserves source hashes. For influential mock4460,
the valid new bright atoms supply86.246% of exact centre mass. Sampled16k
g1 information is−233,919.29, versus exhaustive+165,073.89 under the same
expanded prior. The known399k entry error is material to the whole-sample
solve. This demonstrates numerical misrepresentation of the expanded prior,
not failure of bright support alone. Dropping selection normalization only
as an algebraic diagnostic still gives g1=0.12554; it is not the main cause
of the large proposal and is never removed for inference.

Documented the full attribution and limitations under
[expanded-step forensic audit](V39_POPULATION_AUDIT.md#expanded-step-forensic-audit-on-september30),
linked from the original comparison. All numbers remain conditional on the
unconverged old K1024 sampler. The matched inputs are already image selected;
finite-prior resolution, double selection, normalization uncertainty, and
the global likelihood maximum remain unresolved. Next informative work is
an accurately evaluated small-cohort profile, including magnitude20–22
observations, before another large run or further prior expansion.

## 2026-09-29 — Exact group profile completes; A/B peak disagreement persists

At the owner's status check, all four workers16746033 completed in22m35–39s
and reducer16746034 completed in41s. No study GPU jobs remain. All four
workers replayed the source exact centre with maximum error7.8613e-11 at
the unchanged1e-7 tolerance. Added completion_audit.json to
inference_v39_independent_exact_group_20260929_v1: verified ten pinned source
hashes, all four worker hashes, exact49-node coverage and cross-worker centre
agreement, and independently reproduced the replacement/aggregation within
1e-10. All128 fresh-parent rows remain: four exact,124 sampled; normalizers
are unchanged.

The coarse peaks remain A=(-0.00625952,0.01735242) and
B=(0.02374048,-0.00264758). Each prior prefers its own peak over the other's
by3.2310 and5.5163 log-likelihood units, respectively. Seed/draw profile
variation now passes0.05 under both priors (A0.04542; B0.02861), but this
is **not** an oracle accuracy guarantee. Exact replacement changed the B
curve by0.25055 at(-0.00625952,-0.03264758), although the previous saved
variants there differed from their primary by at most0.02088. Common error
can survive seed/draw agreement. Changes at the competing peaks were small:
A+0.000473/-0.000229 and B-0.000040/-0.001878 at own/other peak.

This particular numerical repair does not explain away the prior-realization
sensitivity. Only one group is exhaustive, the grid remains coarse, and the
A-generated bright control retains strong intrinsic overlap across priors.
No accepted shear estimate or0.3% result. No new GPU jobs submitted during
this status check. See
[exact group profile result](V39_POPULATION_AUDIT.md#completed-exact-group-profile).

## 2026-09-29 — Independent profiles complete; target the remaining numerical failure

All v4 jobs completed: preflights16743015[0–1] in13m39–40s, four profile
workers16743017 in3h37–3h52, reducer16743018 in66s. Both exhaustive preflight
errors passed0.001 (maxima0.000390 and0.000361). All87 groups passed full
candidate-density and response replay. No study GPU jobs remained at this
check. Added scripts/audit_v39_independent_profiles.py; CPU audit16746010
verified every group/profile hash and reproduced all four aggregate curves
within1e-10, saved per-observation contributions and coarse-grid figures.
Ruff and syntax checks passed. These are small saved-array reductions, not
a new likelihood calculation or accepted estimator.

The A-generated cohort has the same coarse peak under both priors:
(g1,g2)=(0.02374048,-0.00264758). Near-mode integration differences are
0.00933 (A) and0.00745 (B), and normalizer differences0.01191/0.01790,
all within the fixed0.05 limits. The fresh-parent cohort instead peaks at
(-0.00625952,0.01735242) under A and(0.02374048,-0.00264758) under B.
A fails integration convergence (0.12726); B passes (0.02760). Both
fresh-parent normalizers pass. These are grid points spaced0.01 apart,
not refined shear estimates or0.3% evidence.

Fresh-parent row47 (measured magnitude20.76839) dominates the draw instability:
its maximum8k-to16k change is0.18904. For attribution only, removing its
original numerical group leaves maximum summed integration drift0.04542;
no observations are removed from inference. Other numerically stable rows
also contribute strongly to the A/B likelihood difference. Submitted an
exhaustive full49-node calculation for group66, rows44–47, across both priors:
four one-V100 workers16746033, reducer16746034, frozen in
inference_v39_independent_exact_group_20260929_v1. Each worker replays the
old exact centre, then handles12 other nodes. Only these four rows will be
replaced in the128-row sums; normalization and all other rows stay fixed.
Estimated20–30 minutes based on the completed three-node preflights.
All four exact workers passed startup and reached10.5m of20.23m atoms in
their initial centre scan without errors. Status-only audit16746082 is
scheduled for about two hours later; it is not an agent wakeup or notification.

CPU context-overlap audit16746043 found exact five-intrinsic-feature matches
in B for19/128 A-generated inputs, and no exact eight-feature F8-context
matches. Disjoint scenes share an empirical source catalogue; this is not a
fully independent intrinsic-galaxy holdout. All three measured-bright inputs
have intrinsic matches in B (3,1,2 for rows116,208,245); the common coarse
peak therefore does not cleanly test removing their intrinsic support.
Follow-up ID audit16746067 and saved centre candidates quantify the effect:
the matched B atoms carry68.63%,81.32%,99.973% of the three objects'
posterior mass at the fixed initial centre. Their original generating A atoms
carry66.71%,99.23%,38.38%, respectively. These are centre-only posterior
fractions, not prior weights or inferred shear probabilities; source arrays
and exact centre denominators provide the calculation in
audit/bright_intrinsic_support.json. No extra GPU inference was used.
The complete response neighbour
graphs were not compared. See
[completed independent profiles](V39_POPULATION_AUDIT.md#completed-independent-profiles).

## 2026-09-28 — Padded replay passes; launch independent-profile preflights

V4 preparation16742975 passed all11 targeted tests (0.20s), source receipts,
scene disjointness, prior masses and unchanged cohort checks. Bounded GPU
replay16742976 passed all ten cases exactly: ordered/permuted first, middle,
padded-final blocks, mixed subsets and duplicated draws, for both cohorts.
Maximum coefficient and density differences were0 throughout. The padding
repair therefore removes the reproduced partial-batch discrepancy in these
probes without relaxing any tolerance; the original six candidate differences
still require full-centre replay and are not assumed resolved.

Submitted two one-V100 exhaustive preflights16743015[0–1]. Both must pass
coefficient, candidate-density and full-prior log-profile checks before
CPU gate16743016 releases four persistent one-V100 workers16743017[0–3].
Reducer16743018 retains every numerical variant; no accepted estimate or
population conclusion yet. The unchanged design uses128 scientific rows
per cohort, A/B disjoint scene priors and separate selection normalization.
Runtime remains dominated by20.23m-atom centre scans for87 four-row groups.
The4-GPU ceiling is enforced by stage dependencies and array throttles.
Both preflights reached their full-prior centre scans and passed1m atoms
without error at the startup check. Scheduled status-only audit16743028 for
about two hours later; this records scheduler/progress state and is not an
agent wakeup or user notification. Saved repair_identity_audit.json verifies
all scientific settings,87 group input hashes, priors and normalizer input
hashes are unchanged from v2.
See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Isolate partial density-batch sensitivity before independent profiles

V3 preparation16742917 passed11 tests in0.24s and all input checks. The
bounded GPU replay16742918 reproduced exact coefficients and bitwise density
reordering on the first and middle full16,384-atom blocks. On the final13,365
atoms, coefficients still matched exactly, but5,777 density terms in the
fourth observation differed (maximum0.1183698); the first three rows were
bitwise identical. Saved the complete failed probe arrays. Its last density
batch is shorter, so reordering moves individual atoms into/out of that batch.
This isolates a batch-shape numerical effect; it does not yet explain the
six small candidate differences in v2, nor the historical shear failures.

The frozen v4 driver pads only the final atom block to16,384 using duplicate
atoms and discards those evaluations before summing or ranking. Four original
observation rows and4,096-atom density chunks remain unchanged. Added fixed
mixed-subset and duplicate-atom replays to test changes in the queried atom
set, not only permutations of an identical set. All probes finish and save
diagnostics before their combined gate; tolerances remain unchanged. Two
independent preflight groups can then run on two GPUs before the four-worker
broad comparison. CPU preparation16742975 and GPU replay16742976 are submitted;
no broad scans are submitted. Ruff and shell syntax passed.

## 2026-09-28 — Gate independent profiles on numerical replay; reuse fixed responses

The first full-centre scan in preflight16742809 completed, but candidate
reordering failed the unchanged1e-7 per-atom density replay:6 of65,536 terms,
maximum log-contribution difference3.69084e-6. The broad worker array was
held during a runtime review and all dependent workers/reducer subsequently
cancelled on preflight failure. No A/B likelihood profile has been accepted.
The old scheduled status-only job was cancelled as superseded.

Added experimental scripts/v39_group_response.py, retaining the original
response coefficients for one fixed four-observation group across shear
points. This stores about0.65GB per group and avoids repeating the same
emulator pair sums. It requires exact coefficient replay on reordered
candidate/tail sets and rejects changed observations. No production model,
batch shape, prior, observation, seed, grid or tolerance changed. Added a
focused row/atom/mask/cache-invalidation test; the likelihood driver now saves
replay diagnostics before failing and checks the union of near-modes across
all numerical variants rather than only the primary profile's near-mode.

Frozen v3 preserves v2 scientific settings and outputs. CPU preparation16742917
and bounded reorder diagnostic16742918 are submitted; broad scans are not
submitted pending this diagnostic. Ruff and shell syntax checks passed.
Preparation will run the existing ten tests plus the new memo test. The
reorder diagnostic compares original versus reused coefficients and ordered
versus permuted densities on three fixed atom blocks for each cohort.
See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Start scene-disjoint prior A/B likelihood-profile comparison

Owner clarified that continuation means the independent-population-realization
comparison and likelihood profiles proposed in the reflection. Discarded the
unsubmitted atom-thinning draft; no thinning experiment ran. Added
scripts/run_v39_independent_profiles.py. Use the original F8-ready prior's
scenes20000–20099 as A (10,116,902 atoms) and20100–20199 as B
(10,114,319 atoms). These are disjoint generated scenes from the same empirical
parent source, with the original inverse-inclusion masses separately
normalized. No extra bright atoms, model changes or fitted reweighting.

The first128 existing selected model draws whose generating atom belongs to
A are a conditional sample from A. Preserve their original four-row numerical
groups. The independent-input control is the first128 fresh unselected-parent
mock measurements, selected once. There are87 numerical groups in total;
only128 rows per cohort enter scientific sums. The A cohort has three
measured-mag<20 rows and the fresh-parent prefix has none; no identities were
changed after noting this limitation. Each cohort is evaluated against A and B.

Freeze a7x7 two-component grid, offsets±0.03 in0.01 steps about the original
full20k mock's measured mean, without truth-centred starts. Exact centre
scans supply per-prior top16384 candidates and the existing conditional-tail
proposal; use seeds8701/8702 and4k/8k/16k nested tail draws. Normalize A and B
separately using two independent weighted-prior samples and nested32k/64k
atom and32/64 CRN-QMC flow budgets. All normalizer and proposal variants are
retained. A small preflight compares both priors on the first original group
of each cohort against exhaustive sums at centre and g1±0.01;8k/16k must
agree within0.001 in centred per-object log numerator before broad grids run.
Coarse-grid peaks are diagnostic and require convergence checks and refinement.

Ruff, AST and shell syntax passed; preparation runs ten existing tail/reduction
tests. First attempt16742780 failed on import because the writer alias belongs
to the cohort driver; no science outputs were generated. Preserved v1 and
cancelled dependent preflight automatically. Corrected the alias in a frozen
v2 root. Preparation16742796 completed in33s with ten tests passed and source,
scene-disjointness, mass, cohort and padding checks passed. Preflight16742809
gates four one-V100 workers16742810[0–3], then reducer16742811. No other GPUs
were active. See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Pause runs and reassess the inference failure mechanism

Owner requested reflection on same-prior mock recovery, actual and
matched-input curvature failures, and the unstable expanded-prior result.
No jobs were active and none were submitted. Reviewed generation code,
the four historical comparisons, parent-population audits, the completed
fresh-parent5130 result, exhaustive probes, and the mixture-curvature
derivation in MATH.md. Added a synthesis under
[reflection on the failure mechanism](V39_POPULATION_AUDIT.md#reflection-on-the-failure-mechanism).

The synthesis distinguishes finite-prior approximation, numerical atom
integration, selected-cohort conditioning, and local optimization. Same-prior
mocks contain their generating atom and are internal closure tests; their
iteration stability did not pass draw convergence. Matched-input mocks start
from already selected inputs, so are not ordinary parent-population closure.
The clean fresh-parent5130 run had positive information throughout its draw
ladder, preventing a blanket conclusion that off-prior mocks necessarily
fail. Exact local negative information does not establish a wrong likelihood
maximum; the original actual run never reached BFGS recentering.

Numerical integration errors are established. Finite-prior resolution and
strong mixture nonlinearity remain plausible explanations requiring separate
tests; broad population mismatch is not established. A proposed controlled
same-population comparison would generate from one finite prior and infer
using that prior versus an independently realized prior, with selection once
and converged likelihood profiles. This is a proposed discriminator, not a
newly authorized run or result. No code, model, prior, or acceptance change.

## 2026-09-28 — Exhaustive paired probes complete; negative information survives reweighting

At the owner's check, recovery worker16738603_0 completed in27m43s and
reducer16738604 in24s, both exit0. The scheduled status audit also completed;
no study jobs remain queued or running. The actual group's complete centre
replay has maximum error0 at the unchanged tolerance. All four groups pass
the direct prior-weight/zero-support checks. The reducer verified output
hashes, replayed36 source aggregates, and preserved unprobed rows bitwise.

Saved completion_audit.json in inference_v39_cohort_curvature_20260928_v2:
a light Python/NumPy audit verified pinned source receipts, all four exact
array hashes, and reproduced all32 reported per-object/prior score and
information results from the saved nine-node sums. Preparation had already
passed ten tests. Large centre-array hashes were checked by the scheduled
reducer, not reread on the login node. No implementation or model changes.

Actual row32's exhaustive minimum information eigenvalue is-248854.33 with
expanded weights and-253958.42 with corrected weights. Its repaired sampler
passes all12 comparisons against exhaustive moments (two priors, three
seeds,8k/16k); its baseline passes0/12. The three focal mocks also retain
negative directions with both weights, and their unchanged nonbright sampler
fails all36 comparisons. Sampling errors are real, but do not explain away
these local negative directions.

After replacing the15 scientific probed rows, all36 partial-cohort summaries
remain indefinite. At16k, corrected minimum eigenvalues span-359262 to
-296561 for actual1024 and-64507 to-45578 for mock836. Both uncorrected
expanded arms also remain indefinite. Corrected focal distinct-atom ESS
is4.57–8.11 and one atom contributes30.5–45.5% of each focal numerator.
Concentration makes finite-prior resolution a concern, not an established
cause of population bias. This is a targeted subset at one centre and h=.001;
it neither overturns the old full-mock positive-information result nor
completes the stopped full-cohort comparison. No accepted Newton estimate or
0.3% claim. The next bounded diagnostic would check smaller shear spacings
and local likelihood profiles for these objects; no new run was launched.
See [completed paired probes](V39_POPULATION_AUDIT.md#completed-paired-exact-curvature-probes).

## 2026-09-28 — Three exhaustive mock probes complete; repair actual-centre replay

At the owner's status check,16735501[1–3] completed exit0 in27m27s–27m45s.
Worker0 failed after10m41s: one of65,536 log contributions differed from the
saved actual centre by1.43686243e-7, exceeding the fixed1e-7 replay tolerance.
The dependent reducer never ran; cancelled obsolete16735502. Preserved the
failure and all completed outputs. The scheduled audit16735503 recorded
failure but did not notify an agent; this was discovered at the user check.

Verified completed mock exact-array hashes and saved partial_mock_review.json
in inference_v39_cohort_curvature_20260928_v1. Exhaustive minimum information
eigenvalues for expanded/corrected weights are respectively: mock151
-46145.90/-46150.64; mock734 -28283.90/-28283.91; mock853
-25551.01/-28115.16. Thus these three local negative directions persist
under exhaustive integration and both weightings. Sampling errors remain
substantial (individual16k information-entry errors up to18606), but do not
fully explain the negative curvature. Corrected distinct-atom ESS is6.44,
5.76,8.11. These are per-object, post hoc diagnostics, not a full-cohort result.

Inspection found a replay-input discrepancy: the saved centre-only scan used
the original full-precision centre, while the nine-node stencil rounds it to
14 decimal places (component differences about2e-15). Updated only the replay
path in scripts/probe_v39_cohort_curvature.py to use the original exact float
inputs, retaining the existing rounded inputs for all scientific stencil
sums. Before the full rerun, a focused check over the interval containing
the failure must reproduce rounded-input violations and pass original-input
replay at the unchanged1e-7 tolerance; otherwise it stops. This explanation
is pending that GPU check, not yet established.

New root inference_v39_cohort_curvature_20260928_v2 reuses the three completed
mock groups with their original hash receipts and protocol identities. Only
actual worker0 is rerun. Ruff/AST/shell syntax passed. Preparation16738602
gates one-GPU16738603_0, then reducer16738604; status-only16738605 is scheduled
in two hours. Scientific settings and gates are unchanged. No full-cohort
continuation or model change. See [replay recovery](V39_POPULATION_AUDIT.md#paired-probe-replay-recovery).

V2 preparation completed in27s with ten tests passed. GPU recovery check
confirmed the cause: over the interval containing the old failure, rounded
centre inputs reproduce exactly one1.4368624334792912e-7 violation, while
the original unrounded inputs reproduce every retained value bit for bit
(maximum absolute error0). Direct prior-weight replay also passed all nine
nodes. The actual worker then began its exhaustive scan with no failures.
Only the replay comparison changed; scientific stencil sums and the1e-7
tolerance remain unchanged. Allow about25–30minutes for the remaining
actual scan and reducer; completed mock results are reused.

## 2026-09-28 — Launch four exact curvature probes with paired prior weights

Owner requested continuation after clarifying that the old full-mock positive
information and new subset negatives are different comparisons. Added
scripts/probe_v39_cohort_curvature.py: use the existing likelihood loaders,
normalizations, finite-difference moments and subset reductions for actual
row32 and mock rows151,734,853. These are post hoc stress probes from the
completed subset decomposition, with original four-row density groups.
No new model, prior correction, seed or likelihood centre is introduced.

Sum all28,012,032 atoms at the same nine shear nodes for both expanded and
domain-corrected weights. Reuse each expanded density calculation through
the exact weight ratio; check this against direct corrected densities at
all nodes, including zero-weight atoms. Replay the existing actual group's
full centre array. Retain new mock centre arrays for possible later reuse.
Compare old baseline/bright-repaired moments against the exhaustive reference
at8k/16k for all three seeds, with unchanged numerical tolerances. Separately
replace only scientific rows of the probed groups in the existing subset
totals, exclude padding, verify all untouched moments bitwise and replay
the source aggregate summaries. This is partial diagnostic replacement,
not full-cohort closure or a0.3% calibration test.

Ruff, AST and shell syntax passed. Frozen root:
inference_v39_cohort_curvature_20260928_v1. CPU preparation16735500 runs
the ten existing exact-tail/reduction tests and checks source hashes and
the old actual centre array. Four one-V100 workers16735501[0–3] depend on
preparation; reducer16735502 follows success. Two-hour worker limits and
four-GPU ceiling; no other study GPU jobs were active. Initial expected
runtime30–45minutes from the earlier nine-node pilot, pending actual scan
throughput. Status-only audit16735503 is scheduled in two hours. Startup
monitoring is underway. See [paired exact probes](V39_POPULATION_AUDIT.md#paired-exact-curvature-probes).

Preparation completed exit0 in24s, with ten tests passed and the retained
actual centre hash verified. The groups contain15 scientific observations
and one padding row (mock732), excluded from aggregate replacement. Focal
measured magnitudes are19.4751(actual32),21.04614(mock151),22.18870(mock734),
20.13503(mock853). A lightweight reduction precheck exactly reproduced36
source aggregate score/information/eigenvalue comparisons with the original
padding masks. Active/frozen probe sources agree. Resource receipts confirm
one32GB V100 per worker; startup checks continue through the first scans.

All four workers passed direct expanded-to-corrected weight replay at all
nine nodes, including zero-support checks, and advanced beyond1,064,960
atoms each with no failures. The actual group's retained centre replay also
passes through the scanned prefix. The first65 blocks take about51s per
worker, suggesting roughly22minutes of scanning plus startup/reduction;
allow about25minutes remaining at this check. Leave the normally advancing
jobs and dependent reducer running; final exhaustive and aggregate results
remain pending.

## 2026-09-28 — Clarify positive full-mock result versus negative short diagnostic

Owner recalled that adding bright atoms made the mock information positive.
Rechecked the frozen expanded comparison and current subset results: this
is correct for the original full16,314 mocks at16k draws (minimum eigenvalue
+112804, unstable g1 proposal0.1473); full actual20k remained indefinite
(-995019). Added a comparison-boundary note to
[completed short diagnostic](V39_POPULATION_AUDIT.md#completed-short-cohort-diagnostic).
The new1024/836 test also changes numerical integration, and retains both
expanded and domain-corrected weights. Both repaired subset arms are negative
at16k; this cannot attribute the difference from the earlier full-mock result
to reweighting alone. Full-cohort controlled results remain unavailable
because that run was stopped. No scientific settings or jobs changed.

## 2026-09-28 — Short cohort comparison completes; negative information persists

Workers16735075[0–3] completed exit0 in33m30s–33m47s; reducer16735076
completed exit0 in20s. All eight centre scans and selected-density replay
checks passed. Verified protocol/combined-moment hashes, four worker receipts
and all small repair arrays; large centre arrays were not rehashed during
this completion audit. Confirmed fainter moments remain unchanged. Saved
completion_audit.json in inference_v39_cohort1024_recovery_20260928_v1,
refreshed status_latest.json and cancelled the now-unnecessary16735077 audit.

At16,384 draws, both expanded and domain-corrected bright-repaired arms
have indefinite summed information in all three seeds for actual1024,
actual same836 IDs and mock836. Corrected actual minimum eigenvalues are
[-315697,-359242,-296517]; corrected mock values are[-42441,-38947,-70409].
There is no accepted Newton estimate. Some lower-rung cases and unrepaired
mock seeds are positive; do not generalize the16k result to every rung or
select those positive cases as recovery evidence.

Corrected repaired bright per-object8k→16k changes are at most0.00901 in
score and8.85 in information entries. Along each seed's minimum-information
direction, the actual bright contribution is about-246k, with row32 a large
negative contributor; fainter rows add-50k to-113k. In mocks the bright sum
is positive(+34k to+35k), while fainter rows contribute-74k to-104k, with
rows151,734,853 prominent. Fainter sampling remains unstable. Reweighting
and bright-only repair do not resolve this fixed-centre diagnostic; this
does not establish the full likelihood shape or the original20k outcome.
Next useful bounded diagnostic: exact local likelihood checks for the
dominant bright actual row and fainter mock contributors. No new GPU jobs
or automatic full-cohort continuation were launched. See
[completed short diagnostic](V39_POPULATION_AUDIT.md#completed-short-cohort-diagnostic).

## 2026-09-28 — Replace full-cohort recovery with a short1024-identity diagnostic

Owner asked why the comparison was slow and requested smaller test jobs.
Stopped workers16733516, dependent reducer16733517 and status audit16734855.
Preserved293 completed baseline chunks (157 mock,136 actual), original
inputs/protocols and logs; no bright repairs had completed. The full run
will not resume automatically. Its266 full-prior centre scans dominated
the remaining estimate, about15hours on four GPUs.

Extended scripts/run_v39_cohort_recovery.py to take a fixed prefix, reuse
hash-verified baseline chunks after checking scientific settings against
the source protocol, and reduce the actual subset length. Default full-
cohort behavior remains available; frozen earlier drivers are unchanged.
New root inference_v39_cohort1024_recovery_20260928_v1 takes the first1024
original identity slots in each sample, preserving four-row density groups,
object RNG IDs and padding exclusions. Both priors, three seeds, all draw
rungs, centre, normalization and measured selection are unchanged. Repair
requires only eight groups (four scientific bright rows per sample).
This diagnoses curvature and integration stability; it cannot establish
0.3% bias or settle the original full-cohort failures.

Ruff, AST and shell syntax passed. Preparation16735074 checks source hashes,
runs the existing ten helper/reduction tests and copies the sixteen cached
baseline chunks. It gates four one-V100 workers16735075[0–3], followed by
reducer16735076. The prior GPU preflight is explicitly inherited; every new
repair still requires direct centre-density replay. Worker limits are2hours,
with an initial30–45minute runtime estimate plus queue time. Status-only
audit16735077 is scheduled two hours later. Startup validation is pending.
See [short cohort diagnostic](V39_POPULATION_AUDIT.md#short-original-cohort-diagnostic).

Preparation completed exit0 in25s; all ten tests passed. The prefix contains
1024 actual measurements and836 scientific mocks (188 flagged padding rows),
with four measured-bright rows in each. All sixteen baseline chunks were
copied after hash verification. All four32GB V100 workers started; resource
receipts confirm the same hardware as the source run. AST comparison confirms
repair, evaluate and load_pair are unchanged from its frozen driver, and
the new active/frozen driver hashes agree. All four workers then verified
their cached baseline chunks and reached their first full-prior centre scan
(mock groups8,9,44,145), with no failures. Startup includes about five minutes
of loading, verification and first density compilation; the30–45minute total
remains provisional pending sustained scan throughput. Leave the bounded
run, dependent reducer and status audit running.

## 2026-09-28 — Launch controlled original-cohort prior and bright-repair comparison

Owner requested continuation. Added scripts/run_v39_cohort_recovery.py,
reusing the frozen likelihood loaders, sampler, exact-tail helper and Newton
reduction. The [original-cohort protocol](V39_POPULATION_AUDIT.md#original-cohort-recovery-comparison)
keeps actual20k and matched16,314 scientific mocks at the same centre and
compares two priors × baseline/bright-repaired integration. Three seeds and
every draw rung remain fixed; no optimization or production promotion.

The baseline uses the original expanded-prior proposal for both targets,
preserving common draws/probabilities while target weights and normalization
change. Repair covers all135 actual/134 mock measured-mag<20 rows in266
original groups. Fainter rows remain in the estimates and numerically
approximate; their known integration errors and remaining draw drift must be
reported. This is a bounded partial repair, not a claim of reliable whole-
sample integration. One expanded centre scan serves both prior arms via
the exact prior-weight ratio; selected-density replay verifies that reuse.

Validation before submission: Ruff/AST/bash syntax passed. Preparation16733505
checks input hashes/padding and runs ten existing targeted tests. GPU
preflight16733506 gates four one-V100 workers16733507[0–3] and checks the
historical K1024 replay, corrected nine-node K16,384 sums and prior-weight
identity including zeroed added atoms. Reducer16733508 follows successful
workers; status-only audit16733509 is scheduled about two hours later.
Frozen root inference_v39_original_cohorts_recovery_20260928_v1 pins driver
and source receipts. Initial runtime estimate20–24hours,36hour limits,
four-GPU ceiling intact. Startup is being monitored before yielding.

Preparation16733505 passed all ten tests and source/padding checks in22s.
During startup review, caught a replay-only configuration mismatch: historical
primary arrays use min_ess32/max_fraction.5, whereas the newer parent defaults
are256/.25. Stopped preflight16733506 and cancelled pending16733507/16733508/
16733509 before scientific workers allocated. The corrected replay now calls
the frozen historical evaluator itself; scientific K16,384 settings and all
comparison thresholds are unchanged. Preserve v1 with a superseded receipt;
active root is inference_v39_original_cohorts_recovery_20260928_v2.
Ruff/AST passed again. New preparation16733514 gates preflight16733515,
four workers16733516[0–3], reducer16733517 and two-hour status16733518.

V2 preparation completed exit0 in12s, ten tests passed. GPU preflight passed:
both historical actual1958/mock4460 arrays replay bit for bit; corrected
nine-node K16,384 sums agree within1.05e-13/1.40e-13 in log space. The
weight-ratio identity and zero-weight added-atom checks pass, as do both
target evaluations with a common proposal. Peak allocated GPU memory18.07GB
on a32GB V100. Dependent worker startup is now being checked.

Preflight finished exit0 in5m04s. Four workers16733516[0–3] started on
th-cl-nv01 with one32GB V100 each. Followed startup until every worker
completed its first original four-row group for both priors and all three
seeds, with no failures. Work is balanced as78–79 baseline chunks and66–67
repair groups per worker. Active/frozen source hashes agree. Jobs now
progress normally; leave the dependent reducer and two-hour status audit
in place. Runtime estimate20–24hours awaits sustained-checkpoint refinement.

## 2026-09-28 — Intermediate-magnitude repair passes all96 numerical checks

Owner reported completion. Workers16733295[0–3] completed exit0 in26m16s–26m20s
and reducer16733296 in23s. Refreshed external status, verified four completion
and centre-replay receipts plus all small output hashes; the reducer had
also verified the four retained centre-array hashes. See the
[completed intermediate pilot](V39_POPULATION_AUDIT.md#completed-intermediate-magnitude-pilot).

Repaired integration passes24/24 focal checks and72/72 companion checks,
versus0/24 and24/72 for the baseline. Across all16 rows, repaired maximum
absolute score/information-entry errors are0.08432/1.29420, within unchanged
entrywise thresholds for every seed and both8k/16k. No favorable seed or
focal observation was substituted. All six draw rungs remain saved.

Exhaustive per-observation information is still indefinite on all four focal
rows. This is not evidence that the full-sample matrix must fail, and the
repair correctly preserves negative directions. Three focal posteriors have
distinct-atom ESS2.48–10.25; the fourth has ESS1212.46 and only28.59% minimum
candidate coverage, confirming that its tail cannot be discarded. Finite-prior
resolution and local optimization remain distinct from integration accuracy.

The original actual20k negative information and matched16,314 g1=0.1473
proposal are still open full-sample comparisons. This check changes no code,
models, weights, thresholds or production defaults, and launches no additional
GPU job. Next work returns to those original fixed cohorts with the validated
integration, keeping population correction separate from numerical repair.

## 2026-09-27 — Launch four intermediate-magnitude probes with existing integration methods

Owner authorized continuing while avoiding overengineering. Added only the
diagnostic driver scripts/probe_v39_intermediate.py, reusing the frozen parent
evaluator, exact-tail proposal, corrected-prior loader and moment reduction.
The [fixed four-probe protocol](V39_POPULATION_AUDIT.md#intermediate-magnitude-numerical-pilot)
compares baseline and repaired integration against nine-point exhaustive sums:
two fresh-mock stress probes (one per measured-mag20–22/22–24 bin) and two
fixed-seed actual20k controls. Original four-row groups and random identities
remain unchanged; all twelve incidental companions are also reported.

One full-prior scan per worker accumulates nine-node sums and retains centre
logs/top16,384 for reuse. No estimator, production setting, model, prior or
tolerance changes. Keep all three existing seeds and all draw rungs; apply
the previous gate to every focal row at both8k/16k. This does not replace
the unresolved full actual20k/matched16,314 comparison or establish0.3% bias.

Validation before submission: Ruff, AST and launcher bash syntax checks pass.
Frozen root inference_v39_intermediate_pilot_20260927_v1 pins source and input
receipts. Preparation16733294 runs ten existing targeted tests and input
checks; workers16733295[0–3] depend on its success, then reducer16733296.
Status-only audit16733297 is scheduled in about two hours. Queue was empty
before submission; four persistent one-V100 jobs in inter preserve the cap.
Startup and numerical replay checks are being monitored before yielding.

Preparation subsequently completed exit0 in22s, ten tests passed. Selected
rows are fresh4826/3602 (measured mags20.45964/22.57042) and actual11173/2510
(21.1134/22.3313). Four one32GB-V100 allocations started on th-cl-nv01;
active/frozen driver hashes match and resource receipts confirm one visible
GPU per worker. Both fresh baseline groups subsequently replayed all saved
arrays exactly across all three seeds, and actual baselines completed.
All four exhaustive scans reached1,064,960 of28,012,032 atoms in50.4–50.8s
of scan time, without failures. This is meaningful normal progress; estimated
30–40minutes remain including tail comparisons. Centre replay and final
numerical gates are pending. Leave jobs running with their dependent reducer
and two-hour status audit; no full-sample shear conclusion follows yet.

## 2026-09-27 — Fresh-parent inference completes; remaining draw drift is in nonbright rows

Owner requested a check. Workers16732143[0–3] completed exit0 in2h20m–3h12m23s
and reducer16732144 in57s. All1283 baseline groups and43 bright groups are
complete, with no failures. Refreshed the external status JSON and documented
the [completed first-pass result](V39_POPULATION_AUDIT.md#completed-fresh-parent-first-pass-diagnostic).
Verified aggregate moments/protocol/input hashes, four worker receipts and
43 bright-group receipts. The scheduled reducer had verified every consumed
moment-file hash and unchanged nonbright rows. Small saved-array reductions
independently reproduce the final proposals and attribute draw-count movement
within1e-14; output is completion_audit.json in the frozen run root.

On5,130 scientific mocks (44 measured-bright, two numerical copies excluded),
the three16k bright-repaired g1 proposals are0.016889±0.007046,
0.016785±0.006936 and0.017216±0.007600, versus injected0.02. These are one
fixed-centre Newton step, not converged estimates; seeds share observations.
All36 arm/seed/rung summed information matrices are positive, but no result
meets the small-step flag. Repaired8k→16k movements are+0.000340,+0.000421,
+0.000165: still above the0.000060 scale corresponding to0.3% of injected shear.

Exact additive attribution shows bright rows contribute at most2.80e-7 in
absolute net g1 movement; measured-mag>=20 rows cause the remaining drift.
Bright rows still account for88.8–89.3% of squared g1 sandwich influence,
leaving finite-prior resolution and local quadratic validity unresolved.
Next: numerical probes of intermediate-magnitude rows before a larger run
or recentering. No production code/model/prior changes or new GPU submissions.

## 2026-09-27 — Exact-tail pilot passes at K16k; launch fresh-parent first-pass diagnostic

Owner reported completion. Workers16731880[0–3] finished exit0 in17m35s–17m43s
and reducer16731881 in20s. K16,384 passes all24 predeclared checks across
four focal rows, three seeds and both8k/16k draw rungs. K4096 passes22/24,
K1024 passes6/24; neither passes the combined gate. Verified worker hashes
and centre replay receipts. The hardest actual probe has maximum score error
0.010354 and information-entry error20.6443; the latter is just within the
corresponding21.3091 tolerance. No thresholds or seeds changed, and no general
convergence or0.3% shear result is asserted. See the
[completed pilot and next protocol](V39_POPULATION_AUDIT.md#completed-exact-tail-pilot-and-fresh-parent-first-pass).

Added scripts/run_v39_parent_inference.py and froze root
inference_v39_parent5130_exacttail_20260927_v1. Reuse the verified corrected
prior and normalization at the already frozen actual-sample centre, independent
of these new mock results. All5,130 scientific mocks get a K16,384 Gaussian
baseline; replace only44 measured-mag<20 rows in43 original groups with the
accepted exact-tail integration. Keep three original seeds and every draw rung.
Two flagged numerical copies fill the final group and are excluded from every
estimate. Existing input IDs preserve the per-object RNG streams. This is one
fixed-centre Newton diagnostic, not a converged inference or model acceptance.

Validation: Ruff/AST/bash syntax pass. Initializer16732141 completed exit0 in16s,
including10 targeted proposal/reduction tests, all four large saved-centre
hashes, fresh-mock source hashes, measured-cut checks and no scientific-row
loss. GPU preflight16732142 verifies original four-row group replay and audits
the nearly seed-constant K1024 candidate curvature residual separately from
sampling noise. Canonical K16,384 replay must pass before the four persistent
one-GPU workers16732143[0–3] start; CPU reducer16732144 depends on them.
Initial runtime estimate3–4hours,6hour limits, with the four-GPU ceiling intact.
Status-only audit16732145 is scheduled for about two hours. No production model,
prior, numerical default or acceptance threshold changed.

Preflight16732142 subsequently passed in3m21s and all four workers started on
th-cl-nv01. Fresh four-row groups replay bit for bit. The same-atom direct1024
versus canonical16,384-prefix diagnostic confirms a separate density batch
effect: information11 differences+36.5221 for actual5359 and−41.3445 for
mock3192, from log-integral differences about2e-5. These explain much of the
seed-stable small-K curvature residual; the exact internal numerical operation
is not isolated. Canonical K16,384 replay passes, supporting the new run's
full4096-atom density blocks without rewriting any old result.

Worker startup was monitored through saved inference checkpoints. At3m52s
elapsed all four workers were progressing normally, with40 of1283 baseline
groups complete across the array (all three seeds per group), no failure
receipts, and one32GB V100 visible per worker. The bright work is distributed
as11/12/12/8 groups. The scheduled two-hour status audit remains active;
the3–4hour initial estimate is still appropriate pending bright scan progress.

## 2026-09-27 — Launch exact-centre conditional tail proposal pilot

Owner said to continue numerical repair. Added scripts/v39_exact_tail.py,
scripts/probe_v39_exact_tail.py and tests/test_v39_exact_tail.py. The
[conditional-tail protocol](V39_POPULATION_AUDIT.md#exact-centre-conditional-complement-pilot)
uses the same exact candidate sets and replaces Gaussian whole-prior tail
draws with90% exact centre-likelihood complement plus10% conditional scene
prior. It excludes exact candidates from draws and reports actual CDF-bin
probabilities. This preserves the numerator integral's expectation and full
positive-prior support; log/derivative accuracy still requires the existing
numerical gate. No production runtime or scientific settings changed.

Frozen root inference_v39_exacttail_pilot_20260927_v1 pins script/helper/test
hashes and both reference results. Keep four original groups, K1024/4096/16384,
seeds8701–8703, draw ladder512–16384, h0.001, prior and normalization unchanged.
All scores and information entries must pass at both8k/16k for every seed.
Retain expensive full-prior centre contributions on disk for future reuse;
verify their sum and selected atom densities against the preceding run.

Validation: Ruff and AST passed; CPU16731872 ran the new proposal tests plus
tests/test_catalogue_null.py,37 passed in22.11s, zero regression failures.
Checks include an exact expectation oracle with an atom whose centre
likelihood is zero but displaced likelihood is positive, excluded candidates,
extreme log scales, nested draws and per-object uniform streams. Launcher
bash syntax checks passed. The first array16731873/reducer16731874 were
cancelled while pending to correct a reference-root substitution; no worker
allocated and no scientific result was produced.

Corrected array16731880[0–3] depends on successful tests and uses four one-GPU
V100 workers in inter,45minute limits; CPU reducer16731881 follows it.
Status-only audit16731875 is scheduled in about two hours. Startup checks
confirmed one32GB V100 visible per worker, all four on th-cl-nv01. At4m05s
elapsed all workers had scored1,064,960 of28,012,032 atoms with no failures;
steady throughput was about30s per1.05million atoms after compilation.
Estimated15–20minutes remaining including proposal comparisons. Active and
frozen source hashes match. Next, compare every predeclared check with the
frozen exhaustive sums; no accepted shear result or model promotion is implied.

## 2026-09-27 — Whole-prior ranking completes; mock probes converge but actual curvature still fails

Owner requested status. Array16731606[0–3] completed exit0 in17m24s–17m28s
and reducer16731607 in19s. Verified all worker/reference NPZ hashes and
all four exact-centre replay receipts; refreshed the external status JSON.
Documented the [completed whole-prior pilot](V39_POPULATION_AUDIT.md#completed-whole-prior-ranking-pilot-on-september27).

At K16,384 both mock focal rows pass all six seed/rung checks. Actual1958
passes0/6 and actual5359 passes3/6; their scores pass but curvature does not
consistently meet the unchanged entrywise tolerance. Minimum nine-node
candidate coverage is99.629465%,99.803972%,99.915333%,99.996588% for the
two actual and two mock probes respectively. All three K arms still fail
the combined gate. No seed or result was omitted.

Added external tail_diagnostic.json using small saved-array reductions:
candidate-only versus exhaustive log-numerator finite differences isolate
the remaining tail contribution with normalization cancelling. Omitting the
tail at K16,384 changes actual1958's g1 score by+0.66258 and its11information
entry by+172.77, so coverage alone cannot justify truncation. No runtime code,
models, production prior, gate or defaults changed; no new GPU job submitted
at this status check. Next work is a controlled complement-accuracy repair
on these probes, followed by fresh unselected-parent mock inference after
numerical acceptance. A0.3% shear result remains unestablished.

## 2026-09-27 — Corrected prior verified; pool-limited ranking improves but fails integration gate

Owner requested status. All16730737workers finished exit0 in55m04s–55m07s,
reducer16730742 in26s, independent replay16730778 in1s, and scheduled audit
16730779 in2s. Verified worker NPZ hashes and receipts. All28normalization
shards reproduce old-weight masses exactly; directly reweighted exhaustive
numerators match the earlier component reconstruction within2.85e-14.
Documented the [completed pilot](V39_POPULATION_AUDIT.md#completed-domain-matched-pilot-on-september27).

Every arm fails the unchanged score/information gate on all four focal rows.
Exact ranking still materially improves the1024-atom stratum: minimum
stencil coverage for actual1958 rises from0.000459% to93.4727%, and mock4460
from0.0000239% to98.7186%. Their existing Gaussian ranker misses most local
mass, exposing the integral to noisy complement draws. However, the complete
broader pool itself captures only94.4472% for actual1958 and98.7884% for
mock4460. Saved restricted_pool_errors.json separates deterministic pool
errors from complement noise; increasing K within that pool cannot recover
the omitted mass. No gate, observations, seeds or reported results were changed.

Fresh one-draw generation16730768 also completed exit0 in15s:20,000unselected
input attempts,8,972usable,5,130retained including44measured-bright rows.
No top-up or generating-atom insertion. No inference result is yet accepted
on that cohort.

Continue the already authorized numerical repair with a small
[whole-prior ranking probe](V39_POPULATION_AUDIT.md#exact-whole-prior-ranking-follow-up).
Added scripts/probe_v39_full_rank.py; frozen root
inference_v39_fullrank_pilot_20260927_v1. At the same centre, evaluate all
atoms once and retain the highest exact contributions; compare K1024,4096,
16384 with unchanged full-support complement, seeds and both draw rungs.
Reuse the completed nine-node exhaustive references and normalization.
Centre sums and selected atom densities must replay within log-space1e-7.
Array16731606[0–3] and reducer16731607 stay within four one-GPU V100 workers;
30minute worker limits. Ruff and AST checks pass; existing numerical tests
are unchanged, with runtime replay checks added for this diagnostic. Startup
succeeded on four32GBV100s, one per worker; at6minutes elapsed roughly5.2million
atoms had been scored per worker without failure receipts. Estimated another
15–20minutes at observed throughput. Scheduled status-only audit16731657
for about two hours; it records results without promoting a sampler. No large
inference, model change or0.3% claim follows from the failed pool-limited gate.

## 2026-09-27 — Launch domain-matched bright pilot and prepare unselected-parent mocks

Owner authorized the recommended next steps. Added
scripts/run_v39_domain_matched.py and scripts/run_v39_parent_mock.py; the
[pilot protocol](V39_POPULATION_AUDIT.md#domain-matched-pilot-authorized-on-september27)
and [fresh input cohort](V39_POPULATION_AUDIT.md#fresh-unselected-parent-mock-preparation)
are documented with frozen external roots, seeds, gates and limitations.
No production models, priors, measurements, cuts or defaults changed.

Candidate initialization16730721 passed in28s: zero weight for848,650 added
atoms outside the generator domain,7,201,662 valid bright atoms sharing the
original bright mass, and every nonbright weight unchanged. Four original
groups retain focal actual rows1958/5359 and mock rows4460/3192; the latter
row per sample is selected by fixed seed2026092711, independently of inferred
shear. Normalization reuses cached per-atom C8 probabilities and replays all
64-QMC forward draws/node with identical shard seeds, reducing both old and
new weights to verify the baseline before comparing algorithms.

Four one-GPU V100 workers16730737[0–3] run in inter, CPU reducer16730742
depends on completion, and independent numerator replay16730778 follows it.
Compare Gaussian candidate ranking, exact centre ranking within the existing
131072-atom location pool, and exact ranking within its union with a whole-
catalogue Gaussian131072pool. Each retains K1024, the original tilted
complement, both8k/16krungs and seeds8701–8703. All28,012,032 aligned atoms
are summed at all nine shear stencil points for each four-row group. Exact
pool coverage and all score/information entries are checked against the
predeclared numerical gate; this is not a0.3% scientific acceptance test.

The first allocation16730727 was cancelled during startup because a two-node
nodelist requested both hosts per worker. Corrected launcher to nodes=1 and
one V100; cancelled its reducer16730729. No completed stage was used from
that allocation. Current workers have one GPU each, three on th-cl-nv01 and
one on th-cl-nv02. Their first normalization shards all replayed the old
weighted masses exactly, maximum difference0.0, and progressed to second
shards without failure receipts. Estimated1.5–2hours after startup;4hour
limits. Status-only audit16730779 is scheduled in about two hours and writes
status_latest.json; it does not notify the owner or promote results.

In parallel, CPU16730765 prepared20,000 uniformly sampled unselected inputs
from all100ConstGold cases in1m43s, without truth cuts or input measurement
selection. All identities match;9,982 target-role and10,018 other-role rows,
with four chance overlaps with the former selected20k. Their new forward
generation16730768 waits for successful preparation and the GPU array to
end, preserving the combined four-GPU ceiling. It performs one normal draw
and one measured selection, reporting retained count without top-up. No
generating input is inserted into the inference prior. Generation is useful
regardless of sampler-pilot outcome; subsequent inference remains pending.

Validation: Ruff/AST checks pass. Tests16730722 passed317/skipped3 without
artifact roots; rerun16730766 with configured roots passed319/skipped1,
zero regression or snapshot failures. Parent input construction validates
source hashes and zero unmatched rows; candidate weights and frozen execution
identities passed. No new shear result or resolved sampler failure is claimed
while the normalization/sampling/exhaustive stages run.

## 2026-09-27 — Deep audit separates bright prior domain, cohort conditioning and integration

Owner asked whether population mismatch requires reweighting, then requested
a deeper investigation. Added read-only scripts/audit_v39_population.py and
scripts/profile_v39_population.py; full evidence and interpretation are in
[the population audit](V39_POPULATION_AUDIT.md#question-and-scope), linked from
the matched-input comparison. No production prior, measurement sample, model,
cut, seed or optimizer setting changed.

CPU16730323 scanned69,956,785 unselected input rows across100cases plus both
weighted priors, and audited the saved per-observation curvature. It completed
exit0 in1m35s. The original prior's largest intrinsic-property marginal CDF
discrepancy is0.02382percentage points; no actual input lies outside the
generator morphology domain. The expanded prior assigns10.5418% of its
bright mass to848,650 added rows outside that domain. A histogram-only
counterfactual excludes that mass, preserves the bright total and nonbright
weights, and reduces the bright n CDF discrepancy from9.8729 to0.6158points.
This is generator-support matching, not a truth cut on selected measurements.

CPU16730457 additionally compared27,982,713 unselected target-role contexts
from80cases with F8/C8 neighbour features; exit0 in42s. Original-prior maximum
neighbour-feature marginal CDF difference is0.03609points. These descriptive
checks do not prove full joint equality or0.3% calibration. The matched-input
mock starts from an already selected cohort and applies a second selection;
documented its distinct conditional input distribution and normalization.

The135 measured-bright actual rows contribute-999244.60 along the final
minimum-information direction, versus-995018.53 for all20k. Influential rows
have sampled ESS near1 and large8k-to16k changes. Three GPU jobs16730364[0–2]
sum all28,012,032 atoms for two post-hoc influential probes at the shared
centre and g1±.001; original four-row groups and4096 density chunks retained.
CPU reducer16730370 depends on all workers. Original and domain-matched
numerators are reconstructed from separately accumulated support components;
the latter is explicitly not a selection-normalized inference result.
External root: inference_v39_population_audit_20260927_v1. Source snapshots,
protocols, hashes, coverage and output receipts are preserved.

Validation: system Ruff and AST syntax checks passed for both new scripts;
full tests16730371 passed319, skipped1, no regression/snapshot failures.
No new model training or large inference continuation was launched.

All three exact workers completed exit0 in13m42s–13m46s; reducer16730370
completed in9s. Exhaustive expanded g1 information is-158392.95 for actual
row1958 and+165073.89 for mock row4460, compared with their16k estimates
-476790.52 and-233919.29. The actual16k score also has the wrong sign.
Thus the sampler demonstrably misestimates influential rows under the same
fixed likelihood; population reweighting alone cannot explain the discrepancy.
Exhaustive distinct-atom ESS is5.608 and1.662, with largest atom fractions
39.57% and76.70%, so finite-support resolution remains a separate concern.
Outside-domain atoms carry13.284% and3.431% of each probe's likelihood mass.
The domain-matched numerator g1 information becomes-8834.75 and+170036.63;
this is not a normalized candidate inference or a full-sample curvature test.
CPU16730526 verified exact reconstruction of all old weights in2s.

Recommended next sequence is domain-matched candidate weights with fresh
normalization, broader exhaustive bright integration checks, independent
prior-resolution checks and an unselected-parent small mock closure. All
diagnostic jobs are complete; no scheduled continuation remains. No accepted
shear or0.3% result is claimed. Histograms, post-hoc two-row selection, finite
QMC normalization, h=.001 and floating-point density limitations are recorded.

## 2026-09-27 — Expanded prior completes; mock curvature positive but inference unstable

Owner reported completion. sacct confirms16729459[0–3] all exit0 in
1h39m41s–1h41m24s; status-only audit16729469 also completed. Verified all
eight5000-row worker receipts and SHA256 hashes, complete comparison.json,
and absence of failure receipts. No active jobs remain. Updated external
jobs.json/status_latest.json and documented the
[expanded-prior result](V39_MATCHED_INPUT_MOCK.md#completed-expanded-prior-result-on-september27).

At16384 draws, minimum information eigenvalues change from−1767373.8285
to112803.9799 for matched mocks,−2394269.7775 to−995216.0549 for actual
same IDs, and−2387169.1061 to−995018.5262 for actual20k. Positive mock
information is not a successful recovery: its g1 proposal changes from
0.01905090 at512 draws to0.18813134 at8192 and0.14730972 at16384. The
final two-component step norm is about0.14356, beyond the historical0.05
recentring guard; conditional sandwich errors are about(0.90423,0.27986).
Both actual samples become positive at4096/8192 but return to indefinite
information at16384. The summary flag same_problem_at_shared_center=false
describes only the final mock matrix, not accepted or converged inference.

No accepted shear, robust0.3% result or optimizer continuation follows.
Expanded support and its changed proposal materially affect the result,
but prior support, integration error and local curvature remain unresolved.
No scientific code/settings/output changed and no extra GPU run launched.
Next useful diagnostic remains bright likelihood profiles with stronger
independent integration checks, without choosing a favorable draw prefix.

## 2026-09-27 — Expanded-prior normalization and replay pass; mock evaluation progressing

Owner requested status. All four16729459 workers remain running at56minutes
elapsed, with no failure receipts. All28 normalization shards and assembly
completed. Expanded-prior batching preflight passed exact equality of all
ten moment/diagnostic arrays for60 mock and64 actual rows, including bright
groups, on the same32GB V100 hardware.

Twelve of20 mock evaluation chunks are saved (12000 of20000 evaluation
rows, including explicitly excluded padding); every worker has entered its
fourth1000-row chunk. Actual-measurement evaluation follows each worker's
mock block and has not started yet. No comparison result exists yet. Recent
1000-row chunks take about346–361seconds, suggesting approximately45–55
minutes remaining including actual evaluation and reduction. Updated external
jobs/status snapshots and the
[expanded-prior record](V39_MATCHED_INPUT_MOCK.md#expanded-prior-comparison-requested-september27).
No scientific settings or code changed; no partial shear result inferred.

## 2026-09-27 — Restore extra bright atoms for the fixed matched-input comparison

Owner requested the expanded prior after the matched-input curvature failure.
Added scripts/run_v39_expanded_prior.py and documented the
[controlled expanded-prior comparison](V39_MATCHED_INPUT_MOCK.md#expanded-prior-comparison-requested-september27).
Reuse exact saved16314 mocks and original actual20k, fixed centre, V3.9
models, original four-row groups and sampler seeds/settings. Restore all
7,780,811 extra atoms with existing expanded weights, preserving nonbright
weights and total bright probability. Prior support is28,012,032 atoms.
No generating truth atoms are inserted and no new mock observations drawn.

Preparation validates the cached V3.9 support, identical original atom
features/coordinates/floors, fixed proposal metric and bright-mass allocation.
Old proposal C8 base probabilities stay unchanged; new atoms use their same
evaluation centre. Full selection normalization is recomputed at the fixed
actual centre with64 CRN QMC draws per atom/node and original shard seeds.
Both mock and actual likelihoods are reevaluated under the expanded prior;
final reduction compares with saved original-prior moments and all six draw
counts, rejecting indefinite information without a repair or forced estimate.

Frozen root: inference_v39_constgold20k_matched_expanded_20260927_v1.
CPU initialization16729457 passed its audit in25.8s of Python execution.
Full tests16729474 and initialization gate four persistent V100 workers
16729459[0–3], max4 concurrent GPUs, on inter/th-cl-nv01. Workers normalize
seven shards each, synchronize assembly, pass exact proposal-batching replay
on both datasets including bright groups, then evaluate both5000-row blocks.
Worker0 reduces after all workers finish. Estimated1.5–2hours after GPU
startup;4hour limits. Login resource snapshot retained; heavy work scheduled.
Syntax and system Ruff checks pass (py31 itself has no Ruff module).
No production default, model, optimizer, cut or seed changed; this remains
a selected-input fixed-centre diagnostic, not population closure or a0.3%
acceptance test. Startup/progress verification follows below.

Initial tests16729456 stalled in Python startup on cip-cl-compute4; a
read-only process check showed cl_sync_io_wait and no pytest output. Moved
the identical suite to inter/th-cl-1024us01 as16729474, updated the GPU
afterok dependency, then cancelled16729456. The successful initialization
used this inter CPU node. Scheduled status-only audit16729469 for about two
hours later; it saves a snapshot without agent wakeup or user notification.

Startup verified: replacement tests16729474 pass319 tests with1 optional
skip in27.28s. All four16729459 workers are running and have completed their
first two normalization nodes (zero shear and the fixed actual centre), with
finite positive masses0.01288–0.01293 and no failures. The cached proposal
C8 reference is zero shear; added atoms use that same point. Corrected this
description in the active driver's comment; its frozen running copy remains
unchanged and obtains the correct point directly from the pinned cache.
Saved current jobs/status snapshot; subsequent inference and replay stages
remain automatically gated. Long-run monitoring yields after this progress.

## 2026-09-27 — Completed matched-input mocks reproduce the curvature failure

Owner reported completion. Verified sacct for16729155[0–3] and16729156:
all completed with exit0, four GPU workers in30m30s–30m45s and comparison
in20s. Saved comparison.json/status.json report complete, with no failure
receipts. Of20000 exact input identities,16314 model draws survived C8 and
the new measured cuts; all3686 numerical padding rows are excluded.

At the fixed actual expansion point and16384 draws, observed-information
eigenvalues are(-1767373.8285,2002216.2944) for the mocks,
(-2394269.7775,3423542.5435) for actual measurements of the same IDs,
and(-2387169.1061,3445424.7863) for original actual20k. Read-only JSON
inspection confirms all three matrices are indefinite at all six draw
counts512–16384. No accepted shear estimate or bias result is available.
The failure therefore does not require actual-image measurement mismatch;
finite-prior support, integration error and local curvature remain unresolved.
This selected-input diagnostic is not population closure or a0.3% result.

Updated doc/V39_MATCHED_INPUT_MOCK.md with
[completed evidence](V39_MATCHED_INPUT_MOCK.md#completed-comparison-on-september27)
and limitations, plus external jobs.json/status_latest.json. Cancelled the
redundant pending status-only audit16729189 after checking results directly.
No scientific code, settings, seeds or outputs changed; no additional tests
or GPU runs were needed. A small bright likelihood profile with independent
integration checks remains the proposed next diagnostic.

## 2026-09-27 — Matched-input forward mocks requested for the actual20k cohort

Owner requested mocks from exactly the galaxy inputs associated with the
actual20k measurements, then chose normal forward sampling with the retained
count reported. Added scripts/v39_matched_input.py,
scripts/run_v39_matched_inputs.py and tests/test_v39_matched_input.py.
The preparation joins original(case,input_index) keys, checks catalogue
hashes, reconstructs physical F8/C8 crowding from complete scenes, and builds
the retained disk-emulator pairs without new truth cuts. Matching features
are checked against frozen development contexts where available. Generation
uses one C8 Bernoulli per input and one F8 draw per usable row at(0.02,0),
then disk transport and independent new measured0.6arcsec/25.8 cuts. No
replacement or top-up; both unusable and cut-failing counts are retained.

Inference retains the original weighted20.23m prior and evaluates at the
failed actual run's fixed expansion point, reusing its verified population
normalization. It compares generated observations, actual observations of
identical retained IDs, and original actual20k moments across the complete
draw ladder. Original four-row density groups are preserved using explicitly
excluded numerical padding at nonretained IDs. Importance draws retain
original row IDs. The matched selected input cohort is not population
closure; no shear-bias acceptance or new optimizer is implied.

Frozen active root: inference_v39_constgold20k_matched_mock_20260927_v2.
Preparation16729151, full tests16729152, generation16729153, replay16729154,
four workers16729155[0–3], comparison16729156 are chained by afterok with
invalid-dependency cancellation. Max4 GPUs; no other GPU jobs running at
submission. Initialv1 preparation16729136 and downstream16729138–41 were
cancelled before generation to preserve original density groups following
selection; logs retained. Its15 focused tests and316 full-suite tests passed
(3 optional skips). v2 reruns tests including the padding invariant. Ruff
and syntax checks pass; login resource snapshot retained, heavy preparation
and inference scheduled. See the
[matched-input protocol](V39_MATCHED_INPUT_MOCK.md#controlled-curvature-comparison).

v2 startup:16 focused tests pass; full-suite16729152 passes319 tests with
1 optional skip in26.80s. Preparation16729151 completes all100 cases in
108.8s with zero missing identities and7980 cached target-role feature
replays; reconstructed intrinsic ellipticities match recorded components
to7.50e-16. Generation16729153 completes in10.2s:1402 unusable inputs,
18598 usable draws,2284 measured-cut failures,16314 retained mocks, no
replacement.134 mock-bright rows and134 actual-bright rows share the
retained ID set.3686 numerical padding rows are explicitly excluded.
Replay16729154 passed exact equality of all ten arrays on128 original-group
rows in233.6s of Python execution. All four inference workers16729155[0–3]
then started. Comparison16729156 depends on their success; unlike the
stopped optimizer it records indefinite curvature as a diagnostic outcome.
Scheduled status audit16729189 saves a snapshot about two hours later;
it does not wake an agent or notify the owner.
Production startup verified for all four workers: first4/1000 likelihood
rows emitted in each original5000-row block, no failures. Long-run polling
stops after this startup check; CPU comparison remains automatically chained.

The exact actual20k cohort has9979 structural target-role rows and10021
outside that role; V3.9 development used the target-role parent. This is
verified from original per-case generator counts, not inferred from shape
flags. Retained mock counts are8118 and8196 respectively. Saved
input_role_audit.json and documented this population distinction without
applying a new cut or claiming it explains the failure.

## 2026-09-27 — Actual20k evaluated completely; first solve rejected negative curvature

Owner requested status at09:39 CEST. Slurm and saved receipts show all four
16726749 workers stopped at01:37 CEST after52m15–24s. Every5000-row worker
output and all20 normalization shards completed successfully. The reducer
rejected total observed-information eigenvalues(-2387169.1061,3445424.7863)
at the actual raw-mean centre(0.00983605988,-0.00029219982). The other
workers exited because worker0 stopped. No first estimate or bright update
was produced; all input, likelihood moments and normalization are preserved.
The scheduled CPU audit16726751 correctly recorded the stopped state, but
did not notify/wake an agent, as already documented.

Added scripts/audit_v39_actual_first_pass.py for a read-only saved-moment
audit: receipt/coverage/max-draw checks, curvature by magnitude, nested draw
counts, normalization contribution and influential-row diagnostics. It does
not drop observations or repair the solver. Ruff passes. Initial audit job
16728958 failed on importing SBSI from the external script location; fixed
the launcher to use this run's pinned inference runtime, then submitted
CPU retry16728966. Updated external jobs.json and
[actual-run status](V39_ACTUAL20K_BFGS.md#status-on-september-27-stopped-at-the-first-solve).

Retry16728966 completed in28s. All input/output hashes, row coverage,
finite moments and primary/max-ladder equality checks pass. The first-pass
matrix is indefinite at all six draw counts512–16384. Excluding the
normalization contribution algebraically leaves minimum eigenvalue
-2360166.8824, versus-2387169.1061 including it. The135 measured-mag<20
rows contribute-2239340.4103 along the negative eigenvector (93.8% of net
negative curvature). The774 mag20–22 rows contribute-216479.1020; both
fainter bins contribute small positive totals. Bright zero-view likelihood
relative-error diagnostics: median0.09066,90th percentile0.68966; these
are not shear errors. The same-proposal ladder cannot distinguish true local
non-concavity from correlated numerical integration error. Preserved
first_pass_audit.json; no cuts, solver replacement or GPU restart applied.
Next diagnostic is a safeguarded likelihood profile plus targeted bright
sampling checks, rather than treating this as a converged bias result.

## 2026-09-27 — Actual ConstGold20k input and five-bright-update run submitted

Owner requested the same current inference procedure on actual measurements
instead of model-generated mocks. Reused the first20k of the existing
randomly ordered uniform500k ConstGold plus-leg sample, seed20260914,
cases40–139, g=(0.02,0), after measured radius>0.6arcsec/MAG_AUTO<25.8
and disk usability, without truth-property cuts. Parent input hashes and
matching audit are preserved; this is a new inference on actual measurements,
not reuse of old V3.6 inference features. Pinned V3.9 F8/C8/disk and restored
20.23m prior are unchanged; new7.78m bright extension remains excluded.

Added scripts/v39_image_input.py and tests/test_v39_image_input.py, rejecting
misaligned/duplicate IDs, truth-filtered parents and invalid/cut-boundary
measurements. Prefix order is preserved; measured interior log disk gap is
derived without clipping. Extended scripts/run_v39_bright_bfgs.py to establish
a fresh full20k pass at this sample's raw mean, then five fresh bright-only
BFGS updates from its own first estimate. No mock centre or saved moment
initializes the scientific chain. Initial/fixed-base hashes, full worker
coverage and original four-row density groups are checked.24-row retrieval
and proposal batches require exact numerical replay on actual observations;
that replay's cached reference centre is excluded from the science result.

External root: inference_v39_constgold20k_bright20_bfgs_20260927_v1.
Init16726746 passed38 focused tests in24.26s, verified20k rows with zero new
drops, all100 cases,135 bright rows and536 evaluated rows including401
padding. Full-suite16726747:316 passed,1 skipped in44.53s. Ruff and syntax
checks pass. GPU replay16726748 passed exact equality on all ten arrays for
248 actual observations in5m05s. All four persistent workers16726749[0–3]
then started and emitted their first pass00 normalization masses without
failures. The afterany dependency on completed mock16721029 plus afterok
stage dependencies preserved the four-GPU ceiling. CPU status audit16726751
saves a snapshot about two hours later; it is not an agent wakeup or owner
notification. Estimated duration about
three hours from GPU startup, subject to actual throughput. See the
[actual20k protocol and limits](V39_ACTUAL20K_BFGS.md#numerical-procedure).

The earlier mock five-update run completed: all16721029 workers exit0,
wall1h37m40s. Final g=(0.0205598964,0.0009044658), conditional row
SE=(0.0003832200,0.0003184539). Two consecutive small steps establish
iteration stability, but paired8k/16k changes remain(-0.0001270075,
-0.0000891862), SE=(0.0000858462,0.0000830116), failing draw stability.
Preserved all results and the frozen mock runner. No0.3% acceptance;
actual-case overlap, single positive shear, faint carry and conditional
uncertainty limitations are explicit. Updated doc/V39_BRIGHT20_BFGS.md,
doc/V39_ACTUAL20K_BFGS.md and the inference guide.

## 2026-09-27 — Bright20 BFGS fourth update is small; fifth iteration running

Owner requested status at00:24 CEST. Read all saved iteration receipts,
Slurm accounting and all four worker logs for
inference_v39_mock20k_bright20_bfgs_20260926_v1. Four of five requested bright
updates completed; workers16721029[0–3] remain healthy. Fifth iteration has
12/20 normalization shards complete. Recent fresh iterations take24minutes;
estimated remaining time10–15minutes, not a completion promise.

g1 sequence:0.02076256 ->0.02047071 ->0.02056567 ->0.02055958. Latest
conditional SE0.00037764; g2=0.00090159 ±0.00041802. The latest step
(-6.09e-6,-1.17e-5) satisfies the small-step test once, with two consecutive
passes required. Current relative g1 offset is+2.798% ±1.888pp. Paired
8k-to16k change remains-0.644% ±0.424pp and fails numerical draw stability.
No0.3% acceptance or bright-carry validation; no scientific settings or code
changed. Updated [intermediate results](V39_BRIGHT20_BFGS.md#progress-at-0024-cest-on-september-27)
and refreshed external status. Existing audit16721087 remains scheduled.

## 2026-09-26 — Five measured-mag20 BFGS updates from the first full pass

Owner requested five bright-only iterations at measured magnitude<20, then
explicitly corrected the start to the original first pass rather than the
later BFGS proposal. New continuation starts from Gaussian g=(0.02508452413,
0.00267601819), uses the original pass00 observed-information solver state,
and carries faint numerator moments from its fixed raw-mean evaluation
point(0.01374048162,-0.00264758006). It does not initialize from full-pass01's
BFGS proposal. All20000 observations and the original measurement cuts remain;
only142 measured-bright moments are recomputed. Four-row numerical groups
require560 evaluations with418 padding rows discarded from the replacement.

First bright iteration reuses exactly matching pass01 bright evaluations and
normalization, while every other row is carried from pass00. This avoids a
redundant GPU pass and yields(0.02076255641,0.00075839956). It counts as the
first of five requested updates. No later full-pass faint moments enter the
hybrid. The remaining four iterations refresh full-prior64-QMC normalization
at every centre; carry replaces normalization derivatives on every row.
Keep the original Gaussian-prefilter sampler, K1024/M16384, h0.001, seed8701
and model/prior identities so this isolates the BFGS/brightness change.

Added scripts/run_v39_bright_bfgs.py, scripts/v39_bright_math.py,
jobs/job_v39_bright_bfgs.sh and tests/test_v39_bright_math.py. Existing
historical BFGS algebra is reused, with strict secant/solver checks. New
analytic tests cover nonquadratic normalization and cross-component carry;
CPU init16721013 ran20 focused tests, all passed in15.30s, then verified
source hashes, coverage and matching saved centre and combined the cached
first iteration. Ruff, Bash syntax and whitespace checks pass. Resource
snapshot saved externally; heavy work remains in Slurm.

External root: inference_v39_mock20k_bright20_bfgs_20260926_v1. GPU replay
16721017 must exactly reproduce all10 retained arrays for560 original-group
rows before four persistent workers16721029[0–3] may start. Dependencies
and array throttle preserve4 GPUs total. Five updates are requested; no
early success declaration, truth-based stopping or0.3% acceptance. Carry at
threshold20 and finite-draw/normalizer errors remain unvalidated. Full
[protocol and limits](V39_BRIGHT20_BFGS.md#owner-instruction-and-starting-point).

The earlier ranking probe16720780 completed in9m35s. Both centres exactly
replayed the228 original rows; whole-prior ranking retained identical global
draw IDs and q agreed within5.43e-19. Median top-K overlap was0.885 on128
random rows and0.757 on100 stress rows. It did not consistently reduce weight
relative error or8k-to16k score sensitivity, so it is not substituted into
this continuation. This probe is not a population-bias result.
Replay16721017 started on th-cl-nv01 and emitted4/560 original-group rows.
Scheduled saved-status audit16721087 replaces cancelled old audit16720572;
it runs after two hours but does not wake the agent or notify the owner.
Replay16721017 subsequently passed all ten arrays exactly on560 rows in5m16s;
four workers16721029[0–3] are released from their dependency and awaiting
scheduler allocation.
Production startup followed through: all four V100 workers16721029[0–3]
started on th-cl-nv01 and emitted their first full-shard normalization masses
for bright iteration2, matching the earlier zero-shear masses. No failures.
Yield while the long normalization/inference chain proceeds; audit16721087
is scheduled, and further results must be read from the new root.

## 2026-09-26 — Correct historical recovery; stop unstable Newton chain and audit saved scores

Owner challenged Gaussian g1=0.02508452 ->0.01102622. Reviewed archived
cont.205,342,345,346,369–370 and read the later complete-likelihood100k
chain.json. The earlier recovery omitted material features: cont.370 used
whole-prior Gaussian top-K, normalization-aware measured-m<22 carry and BFGS
score-root updates; cont.205 used a likelihood-increase line search. Current
plain Newton with a131072-location prefilter reproduces neither procedure.
The historical100k final g1 is0.0199264172 ±0.0004137699 for injection0.02;
its models/cuts differ and it is not a0.3% certification.

Cancelled owned workers16715922[0–3] after2h24m52s rather than continue blind
Newton passes; preserved both complete passes and partial third normalization,
recording external intervention.json without changing result receipts. Added
scripts/audit_v39_recenter.py for hash-checked saved-moment decomposition,
normalization-aware carry and reuse of the historical BFGS implementation.
CPU job16720722 completed in24s. It exactly reproduces both saved estimates,
verifies alignment and final-ladder equality, and checks the all-row quadratic
carry invariant. Gaussian secant curvature is2.112 times its initial local
curvature;967 measured-bright rows explain91.92% of absolute projected score
departure from linear carry and86.62% of initial directional information.
The saved normalization term shifts the second estimate by only4.53e-5.

BFGS applied to the same full Gaussian scores proposes g1=0.01916557; using
normalization-aware bright/faint carry proposes0.01918260. These are proposed
centres, not a converged new chain or demonstrated accuracy. The paired
observed-Newton carry/full difference is0.00015457 ±0.00018492, insufficient
to accept the shortcut at0.3%. Distance-only full-score BFGS proposes0.01854994.

Added scripts/probe_v39_historical_ranking.py; one-V100 job16720780 compares
whole-prior top-K with original prefilter on128 random rows plus explicitly
post-hoc stress groups at both saved centres, requiring exact saved replay
of the original four-row groups. No model, prior, density batch or proposal
seed changes. GPU probe is pending validation, not a production-chain restart.
Ruff passes for both new scripts. Updated
[audit and limitations](V39_RECENTER_20K.md#historical-procedure-correction-and-saved-moment-audit)
and doc/INFERENCE.md with the missing historical procedure. Next: read paired
ranking diagnostics, then validate a safeguarded/BFGS bright-carry recovery
separately from sampler convergence; no truth-directed offsets or acceptance.
Probe startup verified on th-cl-nv01,228 total rows: original Gaussian replay
emitted4/228 progress. Existing one-shot audit16720572 will save updated job
and intervention status; it does not wake the agent or notify the owner.

## 2026-09-26 — Original-prior 20k first pass completed; recentering in progress

Owner requested status at 22:10 CEST. Read scheduler accounting, GPU replay
receipt, full pass_00/result.json and all four worker logs for the original-prior
recenter study. Initial normalization and assembly completed; preflight16715921
passed exact replay for both arms on136 rows with identical global draws and
20.41 GB peak allocated memory. Workers16715922[0–3] are running without
failure receipts. First pass completed20k; second pass completed12/20 paired
checkpoints and its fresh full-prior normalization.

First-pass Gaussian16k g1=0.0250845241 ±0.0016664645 conditional SE;
distance16k g1=0.0235238099 ±0.0014268882, for injection0.02. Relative offsets
are +25.42% ±8.33 pp and +17.62% ±7.13 pp. These are intermediate
one-step estimates; recentering has not yet established closure. Both arms
fail the declared paired8k-to16k stability screen at this centre. Updated
[progress and limitations](V39_RECENTER_20K.md#progress-at-2210-cest-on-september-26).
No scientific settings or running code changed; validation is receipt/log
inspection and documentation whitespace checking. Await subsequent complete
passes; do not compare this fresh restored-prior mock as paired with the old500k.

## 2026-09-26 — Launch20k recovery of the iterated pipeline with pre-extension prior

Owner authorized recovery of the successful historical inference setup on20k
mocks, then requested removing extra bright priors. Cancelled expanded-prior
preflight16715903 before production. Asked whether to remove only the new
7,780,811 bright atoms or also the earlier231,221 certainty-stratum additions;
proceeded after the optional clarification interval with removal of the new
extension. Restore the previous20,231,221-atom prior and its original weights,
not renormalized expanded-prior weights. Generate a fresh20k matched-prior
mock, seeds202609281/282/283. Models, physical likelihood, measured0.6arcsec/
25.8 cuts and absence of truth-property cuts are unchanged.

Recovered historical fixedpoint_chain2 receipt binds K1024,131072-candidate
Gaussian reranking, tilted-stratified delta0.1/temperature1, seed8701, h0.001,
full2D steps, float64 proxy scores, retained512–8192 ladder and successive
estimate centres. New study adds16384 as the predeclared primary and a paired
distance-only arm. Both arms use the Gaussian16k chain's centre. Up to4 full
passes; fresh full-prior64-QMC normalization and state-dependent probabilities
at every centre. Exact current density groups of4 replace historical128 to
respect the prior numerical replay finding. No hybrid linear carry is used.

Added scripts/run_v39_recenter.py, scripts/v39_recenter_math.py,
scripts/prepare_v39_reduced_prior.py, two jobs/job_v39_recenter*.sh launchers,
and tests/test_v39_recenter_math.py. Generalized the study driver to an explicit
prior shard count (default28 unchanged) and checked reuse of unsheared reduced
prior coordinates. Runtime/library/model files are reused by pinned hash;
new drivers/protocols are frozen externally. Existing study artifacts remain
immutable. New [protocol and limitations](V39_RECENTER_20K.md#paired-design-and-stopping).

Validation: Ruff, Python syntax and shell syntax checks pass. Full CPU tests
16715901 and16715915 both completed308 passed/1 skipped. New tests cover joint
quadratic recovery from a distant centre, paired covariance cancellation, and
rejection of indefinite information/unsafe steps. GPU preflight is required
before production: exact all-array replay and candidate/draw equality on136
rows, shared global draws across arms, and15% memory headroom.

Submitted dependency chain: restoration16715917[0–3] ->mock16715918
->initial normalization16715919[0–3] ->assembly16715920 ->preflight16715921
->four persistent inference workers16715922[0–3]. No more than4 GPUs at once.
Prior-restoration workers started on th-cl-1024us01; dependent stages pending.
No numerical inference result yet. Monitor first progress and stage transitions;
report iteration, sampling and normalization limits separately, without a0.3%
acceptance claim.

Startup follow-through: all restoration tasks completed successfully in122–125s;
20,231,221 rows and mass1.0000000000000029. Mock16715918 completed all20000
observations (67.8s driver time), with no invalid retained draws and2066
selected true-r>=26 objects. Raw observed mean centre is
(0.013740481623385736,-0.002647580061745958). Initial normalization16715919[0–3]
started on four V100s at th-cl-nv01; all four emitted their first full-shard
node masses without error. Saved-status audit16715934 is scheduled after two
hours; it writes an external snapshot, not a Codex wakeup or notification.

## 2026-09-26 — Recover historical mock closure and correct one-step interpretation

Owner recalled earlier unbiased mock results. Reviewed archived WORKLOG_FULL
cont.174–175,204,259,338,341–342,346,354 and the cross-worktree inference
record. The relevant positive evidence is finite-shear profile recovery and
Newton recentering: the older25000-object full chain changed g1 from0.022960
to0.01906560 then0.01958117 (truth0.02; final conditional SE0.00061195).
Read both saved chain result JSONs and independently checked the saved
five-pass20000-object hybrid chain, which settles at0.01953533 ±0.00071477.
The old8million-object uncut V1 shape-flow result was+0.034% ±0.252%; selected
claims in that entry were subsequently corrected for normalizer uncertainty.
The million-object cont.259 result was explicitly unconverged, and the
complete-likelihood cont.354 closure had only200 objects with69.53pp error.

Current500k V3.9 code and result receipt confirm a single Newton update from
(0.01303596,-0.00261937), not a recentered likelihood fit. The previous status
answer overemphasized atom draws without recovering this established
linearization failure. Draw sensitivity remains real, but it does not isolate
the cause of the endpoint error. Older profiles/chains omitted current
measured selection and disk response; their successful recovery cannot be
transferred as a V3.9 acceptance result. No historical finite-catalogue error
is added to current uncertainties.

Added [historical recovery and current diagnostic implications](INFERENCE.md#historical-mock-recovery-and-the-v39-one-step-limitation).
Next diagnostic should separate recentering, draw-budget, and normalization
effects on fixed existing mock rows, with fresh normalization stencils at new
centres, common random numbers, and stopping independent of injected truth.
Validation: stdlib JSON receipt reads, source inspection, and documentation
whitespace check; no scientific computation, new jobs, or archived restarts.

## 2026-09-26 — Complete500k model-mock inference; draw ladder remains unconverged

All500 partitions/500000 measured-selected observations are complete. Persistent
array16706357[0–3] completed successfully:262000 remaining observations in
6h39m49s wall time (September26 11:42:26–18:22:15 Europe/Berlin). Workers handled
66000/65000/66000/65000 rows; median1000-row checkpoint360.07s. Combiner16706358
completed successfully in52s at18:23:36, verifying original and amended execution
receipts, output hashes, row coverage, and scientific identities. Immediate
completion audit confirms exactly262 distinct optimized partitions and500 total
receipts, unchanged scientific identity and the pinned runtime amendment.
No study jobs remain queued/running. External jobs/status receipts updated.

At the fixed primary M16384: g1=0.020891396494 ±0.000437371819 and
 g2=0.001622446128 ±0.000220339621 (reported robust1SE). Injected shear(0.02,0).
Relative g1 error100*(g1/0.02−1)=+4.456982% ±2.186859 percentage points.
This is fixed-shear model-closure error, not an independently calibrated
multiplicative-slope measurement; see CONVENTIONS.md §7. Measured radius>0.6arcsec
and magnitude<25.8, no truth-property cuts; V3.9-like models and expanded weighted
prior, original fixed seeds, one Newton update and64-draw QMC normalization.

Full draw ladder (relative g1 error% ±1SE pp):
512:−35.380935±0.796048;1024:−31.042642±0.833733;
2048:−25.626496±0.922614;4096:−19.323900±1.089082;
8192:−9.035359±1.476396;16384:+4.456982±2.186859.
All information sums are positive definite, but the largest two rungs differ
by13.492342pp. g2 also remains visibly nonzero. Completion is computational
success, not scientific closure or a robust0.3% result. More observations did
not establish convergence at the retained draw budget. Diagnose finite-draw
score/curvature and the one-step/normalizer approximations before interpreting
another larger observation sample. No new experiment was launched on this
status request. Updated doc/V39_MODEL.md with the completed result and limits.

Source: external500k v2 `result.json`, persistent worker receipts,
`logs/v39_persistent_16706358_4294967294.out`, and sacct for16706357/16706358.
Validation here is receipt/accounting consistency; the already-successful
strict combiner was not rerun and no scientific outputs were modified.

## 2026-09-26 — Resume262k unfinished rows on four persistent V100 workers

Accepted validation16706278:128-row atom search,24-row tilted proposal blocks,
and original4-row likelihood groups. All10 arrays on1032 held-out/stress
observations replay **exactly** (zero maximum difference), as do all candidate
indices, global draw indices and probabilities. Peak allocated26,017,293,824
bytes=24.23GiB, reserved24.55GiB on31.73GiB V100s. End-to-end benchmark including
proposal preparation and uncached response evaluation:2.951926 observations/s.
Four such workers project approximately6.2h for262k remaining rows; use6–7h as
an initial conditional estimate, not a completion guarantee.

Production array16706357[0–3]%4 uses four32GB V100 allocations on inter, one GPU
per persistent job, a shared queue, and atomic1000-row checkpoints. Claims and
checkpoint publication are atomic; same-job Slurm requeue reclaims unpublished
work. Array starts after accepted replay16706278. Full tests16706292 completed
305 passed/1 skipped; Slurm had already aged that job out of dependency tracking,
so its successful accounting/log receipt was verified before submission.
Combiner16706358 waits afterok for all four workers, checks the pinned execution
amendment/validation on every new checkpoint, then invokes the original strict
500-partition coverage/hash/scientific-identity combiner.

Original238 completed partitions, models, mock, prior, all seeds, K1024/M16384,
selection normalizer and all six stencil-prefix moment ladders remain intact.
Original driver/protocol/runtime hashes were rechecked unchanged. No larger
density batch was accepted. External `batch_optimization/amendment.json` SHA
`a6c5b89be71d5377dc0b71defc27147ea54fcd22bfb03514a5a3a10e2a2e6df9`
pins the additional producer and exact-replay receipt. `jobs.json` now points to
the persistent array/combiner and records the original cancelled array separately.
Saved-status job16705939 is rescheduled for two hours after submission; it writes
a scheduler/data snapshot, not a Codex wakeup. Startup verified: all four jobs
are running and independently processing partitions238–241. Each reached176–180
observations without errors; warm density throughput is2.8–2.9 observations/s
per worker. Allowing proposal setup, approximately7h remains from this check.
The scientific run continues; final500k bias/convergence result is still pending.
Updated `doc/V39_MODEL.md` documents the accepted execution and rejected density
batch changes. Long-run polling stops after this successful startup check.

## 2026-09-26 — Proposal-only batching replays exactly; retain four-row densities

Job16706245 tested128-observation candidate searches and32-observation tilted
proposal blocks, with the original4-observation density groups. All10 stored
moment/diagnostic arrays on1032 observations replayed **exactly**, including both
boundary groups; all candidates, draw indices and probabilities also matched.
Observed throughput2.9028 observations/s includes proposal preparation and
uncached response evaluation. GPU peak allocated32,062,547,456 bytes (~29.86GiB)
left insufficient15% headroom, so that deployment was rejected despite exact
numerical replay. Preserved source snapshots and `proposal_validation_16706245`
receipts/arrays remain under the500k root.

The final candidate keeps128-observation search blocks but reduces tilted-CDF
blocks to24. Active helper/validator sources are frozen externally with a24
suffix to preserve the already-running32-block audit. Validation16706278 is in
progress. Current worker requires the exact replay receipt and15% allocated
memory headroom, preserves original1000-row checkpoints, and can reclaim its
own unpublished checkpoint after Slurm requeue. Other workers never steal a
live claim. Four V100 workers will share the262 unfinished partitions238–499.
AGENTS.md records the owner's four-GPU limit for this continuation and the
required four-row density grouping. Latest full-suite16706292:305 passed,
1 skipped; Ruff and shell syntax checks pass. Production switch awaits the
24-block validation receipt; no larger density batch is accepted.

## 2026-09-26 — Reject larger density batches; validate separate proposal batching

Controlled V100 replay16706111 reproduced the original four-row groups exactly,
but density batches8–192 changed float32 likelihood finite differences even
though candidates and full global draws/probabilities replayed exactly. Initial
comparison16706064 mixed V100 with A40 checkpoints;16706065 also regrouped the
extreme rows. These were confounded and are retained as failed diagnostics.
The controlled check uses original V100 checkpoints and intact four-row groups.

Independent1024 ordinary observations234000–235023 plus the8 original-group
rows around86174/112155 were checked in jobs16706174(batch32),16706207(128),
16706206(160),16706212(192). Before those runs, declared a maximum projected
shear shift including2SE of0.003 percentage points (1% of the0.3% bias target).
Using the238k completed information matrix to map paired moment differences,
all four batches gave an estimated g1-relative shift approximately+0.097959pp
with paired SE0.068034pp (upper bound0.234027pp); g2 normalized to0.02 also
failed. This is a finite-sample numerical sensitivity estimate, not a measured
change in the full500k answer. None is accepted. Exact rows, resulting arrays,
hashes and statistics are saved in `batch_optimization/validation_*.json/.npz`.

GPU distance matrices alone require26.71GiB at256 and53.43GiB at512, so
16706180/16706181 exceeded V10032GB memory;1000 probe16706196 was cancelled
once that limit was established. Warm128-row speed comparisons benefited from
reused response-cache entries and must not be advertised as production speedups.
The1032-row validations also warm their first chunk; actual new-checkpoint timing
is required before updating the production ETA.

New `v39_batched_proposal.py` precomputes candidate sets/draws in128-row search
blocks, preserves original global RNG IDs, then serves those exact arrays to
the unchanged four-row likelihood groups. Frees unused search/CDF allocations
between stages. `validate_v39_batched_proposal.py` checks all saved moments and
diagnostics against original V100 checkpoints and independently checks all
candidate/draw identities and probabilities. Job16706245 is in progress; four
persistent production jobs will start only after this stricter replay passes.
The persistent worker retains1000-row atomic checkpoints and a shared queue.
Ruff/shell checks pass; latest full-suite16706226:305 passed,1 skipped.

## 2026-09-26 — Benchmark larger batches and prepare four persistent500k workers

Owner authorized larger observation batches and longer jobs, then explicitly
requested four jobs, one GPU each. New scripts `benchmark_v39_batches.py` and
`run_v39_persistent.py`, plus `jobs/job_v39_persistent.sh`, reuse the immutable
500k v2 driver/runtime and scientific protocol. Persistent workers load once,
claim unfinished1000-row partitions atomically, and publish complete checkpoints
without overwriting completed work. Shared work allocation avoids assigning an
equal observation count to unequal GPUs. Execution amendments and benchmark
hashes accompany every new checkpoint; an amendment-aware combiner also invokes
the original strict coverage/hash/scientific-identity checks.

Benchmarks16706016(inter) and16706022(cip24GB) await original tasks230/234/235.
Both compare128 fixed observations including boundary rows86174/112155 against
original checkpoints and batch4, with exact candidate/draw/probability replay
and all moment/diagnostic arrays (rtol1e-9,atol1e-8). Larger batches must leave
15% GPU memory headroom; warm throughput determines the retained batch among
passing configurations. Resource receipts are recorded on the allocated GPUs.
Validation and production switch remain pending; no claimed speedup yet.

Original queued partitions238–499 and obsolete combiner16702089 were cancelled;
running original tasks continue. No completed partition was removed. Saved
`batch_optimization/migration_snapshot.json` records the262 pending partitions
and four original running tasks at migration. The old array throttle was reduced
to2 while making room for two benchmark GPUs; production will use exactly four
persistent workers. Ruff and shell syntax checks pass. Follow this entry with
measured results and replacement job IDs after replay checks finish.

## 2026-09-26 — Compare current500k runtime with historical full-A40 run

Owner recalled approximately12h on four GPUs. Confirmed historical array
16606146 via Slurm accounting:500000 observations,20 partitions of25000,
first start September20 14:36:22 and last finish September21 01:17:39,
**10h41m17s** elapsed. Four-GPU ceiling, full A40 allocations, object_chunk16,
K1024/M16384 and the same six prefix rungs. Historical final curvature check
failed, but all500k numerical evaluations completed; its runtime is valid
for execution-cost comparison. Source: production_lru_v1/part_00/result.json
and cross-worktree WORKLOG September20 LRU/production entry.

Current16702088 uses500 partitions of1000, object_chunk4, three inter V100s
and three cip virtual A40 workers. Historical prior24m versus current28.012m
adds17% atoms to proposal scans. Models, proposal ranking and precision path
also differ; this is not a controlled kernel benchmark. Completed current
inter tasks median9.53min/1000; cip28.6min/1000. Inter median load138.9s plus
first-four/compile35.2s: about3min of each9.5min task is startup. Historical
25000-row tasks amortized this cost; the current run unnecessarily repeats
it500 times. Smaller object batches are another plausible throughput cost,
but their isolated effect requires a matched replay benchmark. The earlier
24h estimate accurately extrapolated100k setup, not an intrinsic cost of500k.

Snapshot:228/500 partitions complete after9.48h, no failures; all audit gates
passed, including original boundary rows86174/112155. Current cumulative
throughput projects20.8h inference total and11.3h remaining, conditional on
unchanged capacity. Saved runtime_comparison_20260926.json under the500k v2
root. No production jobs, numerical settings or results changed. A future
speed test should compare larger batches on sufficient-memory GPUs and
persistent workers with exact replay before changing the active implementation.

## 2026-09-26 — Preserve disk radial precision and resume fixed-seed500k

Owner authorized continuing the failed500k run. Added `sbsi/disk_precision.py`:
record log(1−|e|²) analytically from unbounded flow coordinates before Cartesian
rounding; propagate it through Möbius transport and use it in inverse density.
This is deterministic precision metadata for the same shape, not a new random
measurement or selection cut. Origin uses cancellation-free series; transport
uses Cartesian log1p only well inside the disk and the analytic gap identity
near the edge. Metadata must agree with Cartesian values to floating-point
tolerance. No clipping, sample replacement, RNG search or parameter changes.

Physical-flow/bundle sampling optionally returns the gap with exactly unchanged
physical draws and random-state consumption. DiskCatalogueLikelihood accepts
the explicit metadata column, uses the same physical-density Jacobian and
original network precision, and retains the old path for ordinary catalogues
without it. New compiled density applies only when this metadata is supplied.
Study driver now records gap metadata, validates encoded shapes, checks
forward/inverse transport and replays the saved failed second batch exactly.
Added `scripts/audit_disk_precision_mock.py` to exercise true-atom conditional
density and the full nine-node M16384 sampler on every rounded-boundary row
plus four ordinary rows before the full inference array can run.

Tests cover120-digit Decimal transport/radial identities from radius0 to80,
forward-Jacobian density at rounded boundaries, old density in the interior,
identical pseudorandom/QMC draws and RNG state, and inconsistent-metadata
rejection. Full-suite16702066:305 passed,1 skipped in30.42s. GPU preflight
16702078 passed old/new and eager/compiled density comparisons in addition to
the existing two-shear checks. Ruff passes. All old experiment snapshots and
the failed catalogue are preserved. Separate external root
`inference_v39_mock500k_distance_20260926_v2` pins the precision amendment,
changed module hashes and unchanged model IDs/seeds. Mock16702083 is submitted;
remaining preparation/inference awaits its validity/replay check.

Mock16702083 completed successfully:500000 observations from2097152 candidate
atoms, same original500k seeds, no replacements. Entire failed second batch
replays identical atoms and physical draws. Both originally invalid retained
Cartesian shapes pass the gap encoding; all encoded observations valid.
53808 retained targets have true r>=26; forward/inverse shape error4.44e-16.
New centre(0.013035957659442396,−0.002619371804842549).

Queued chain: prep16702084[0–27]%6 → assemble16702085 → boundary sampler
audit16702086 → smoke16702087 → inference16702088[0–499]%6 → combine16702089.
All use afterok gates; no GPU stages overlap. Three cip and three inter
preparation workers started. Scheduled saved-status audit16702094 for+2h;
it writes a receipt, not a task notification. Precision audit includes rows
rounded outside either before or after transport, runs original object IDs,
and must produce finite full-stencil moments before releasing inference.
See [precision amendment](V39_MODEL.md#disk-boundary-precision-amendment-for500k).
Verified all six initial workers produced finite positive normalization masses;
four reached the new-centre node. Normal preparation progress established.
Final Ruff/whitespace checks pass. Full sampler audit and500k results pending.

## 2026-09-26 — 500k mock stopped by retained disk-boundary rounding

Preflight16702041 passed unchanged density and selection checks. Mock16702042
failed in1m44s after its first68217 accepted observations: the second batch
contains two measured-selected draws whose squared shape norms round to
1.0000000000000002. Both have finite shapes and positive radius/flux.
Global zero-based selected ordinals86174 and112155 correspond to prior atoms
23945110 and3185269. These are inside the requested500k prefix, unlike the
unused boundary draws in the completed20k/100k runs. No discarded/replaced
objects, clipping or seed changes; no downstream GPU jobs submitted.

The generation guard correctly rejects the non-invertible rounded values.
The output is not a completed500k catalogue and no500k shear result exists.
Saved second-batch draws and audit under the new root's `mock/`, with
`failure_diagnosis.json` recording exact indices, values and validation.
Updated external jobs.json status and the V3.9 study doc. Diagnosis was a
small saved-array inspection; all expensive generation ran on Slurm.
The mathematical radial-tanh map has open-disk support, but float64 output
rounding loses the tiny distance to the boundary; current density rejects
those stored values. Before continuing, generation/likelihood needs a
consistent representation retaining that information, tested for density
and transport agreement. Dropping these observations would alter selection;
changing seeds until500k passes is not acceptable. The unchanged inference
runtime estimate remains about24h once this prerequisite is resolved.

## 2026-09-26 — Start authorized500k distance-only follow-up

Owner requested500k and asked whether it should take approximately10hours.
Measured100k inference elapsed4h41m22s, plus34m56s from mock start to inference
start.79 inter tasks had median9.5min per1000;21 cip tasks median39.6min.
Total GPU time was26.01h. Same-mix scaling predicts23.45h inference plus setup,
so quote about24h subject to contention, not10h. Six workers at inter speed
would need about13.2h inference with unchanged1000-row partitions.

Create separate external root `inference_v39_mock500k_distance_20260926_v1`.
Driver/runtime are byte-identical to the completed100k run. New protocol pins
500000 observations, fresh scene/detection/flow seeds202609271/272/273,
same F8/C8/emulator, expanded weighted prior, measured0.6arcsec/25.8 cuts,
no truth cuts, distance-only K1024/M16384 and every prefix rung. New mean-shape
centre requires new normalization; immutable features and proposal moments
can be reused by verified provenance. This is not a nested sample-size test.
Retained invalid numerical shapes still abort; no seed search or replacement.

Submitted preflight16702041 then mock16702042 to inter V100s. Full inference
will use500 disjoint1000-row partitions, shared six-GPU cap across inter/cip,
and strict complete-population combination. AGENTS.md records the500k scope.
No scientific implementation or estimator changes; completed100k tests and
validation remain applicable, with a new identity/model preflight. Further
submission depends on successful generation; no500k result yet.

## 2026-09-26 — Verify completed100k distance-only study

Immediate owner status check: all28 preparation tasks, assembly16699856,
smoke16699857, all100 inference tasks16699858 and combine16699859 completed
successfully. No pending/running user jobs. The2h limits avoided a repeat
of the20k timeouts; slowest inference task completed in1h04m31s. Strict combine
verified all100 disjoint1000-row partitions and pinned artifact identities.

Primary M16384 result: g1=0.019502411533783973 ±0.0007965901983862846,
relative realization error−2.487942% ±3.982951pp (one robust SE);
g2=0.000339781553263256 ±0.00030211698731576545. At8192 draws the relative
error is−14.030730%; last doubling shifts it by11.542788pp. Unconverged,
not0.3% acceptance. Full ladder and interpretation recorded in
[V3.9 completed100k result](V39_MODEL.md#completed100k-distance-only-result).
This is a fresh realization with a different mean-shape centre, not a paired
20k-versus100k sample-size comparison. No models, numerics or defaults changed;
no additional GPU work launched. Source `result.json` retains full limitations.
Validation: scheduler completion/exit codes,100 completion manifests and strict
combine success; documentation whitespace check. Next diagnostic would compare
exact scores/curvatures and inspect high-influence observations on a fixed subset.

## 2026-09-25 — Launch fresh100k distance-only V3.9 model mock

Owner requested100k using the distance-only arm. New external root
`inference_v39_mock100k_distance_20260925_v1` pins a fresh realization at
injected g=(.02,0), with predeclared scene/detection/flow seeds
202609261/202609262/202609263. Same expanded weighted28,012,032-atom prior,
F8/C8/disk models, measured radius>0.6arcsec and magnitude<25.8, no truth cuts.
This is a new realization, not a nested reuse of the inspected20k sample.
Any numerically invalid retained shape aborts; no dropping, clipping or
seed search. Mean observed shape sets the new centre; normalizer is rebuilt.

Generalized `scripts/run_v39_sampling_study.py` and its launcher to accept
an external study root, protocol-specified arms and observation counts.
Both-arm defaults remain available; prior20k runtime/script snapshots are
untouched. New protocol explicitly pins old feature provenance and coordinate
execution hashes. Reuse verified features and zero-shear proposal moments;
recompute classifier/selection nodes and assembled normalization for the new
centre. Retain K1024/M16384, all512–16384 draw prefixes, full nine-node stencil,
proposal seed8701 and one Newton step.100 disjoint1000-row distance-only tasks
will combine with hash, identity and exact row-coverage checks. Same-draw
invariance is still checked on two probe observations in each task.

`AGENTS.md` records that six GPUs across cip/inter covers this follow-up.
Ruff, Python syntax and Bash syntax pass. Full-suite CPU job16699846 and GPU
preflight16699847 submitted; mock16699848 depends on both. Further stages
will preserve a combined six-GPU cap and longer job limits. Results pending;
more observations do not fix the20k finite-draw convergence failure by itself.

Full-suite16699846 passed294 tests,3 skipped (artifact environment not set),
in42s elapsed; preflight16699847 passed
both tested shears and selection boundaries. Dependency chain:
mock16699848 → prep16699855[0–27]%6 → assemble16699856 → smoke16699857 →
distance-only inference16699858[0–99]%6 → combine16699859. GPU preparation,
smoke and inference use cip/inter with low-memory GPUs excluded and2h limits;
all stages require predecessor success, so total concurrency stays at six.
One-shot saved-status audit16699868 scheduled for +2hours (not a task wakeup).

Mock16699848 completed in1m45s:100000 retained observations,524288 prior
candidates, centre(0.012079337150125146,−0.00029905259816080903).
All retained draws are finite, positive in auxiliary outputs and strictly
inside the disk; no replacement or truth cuts.10827 retained targets have
true r>=26. One unused draw after the100k stopping point rounded to the
boundary; it is recorded in the generation audit. Forward/inverse maximum
shape difference4.44e-16. Preparation16699855 started six tasks concurrently:
three cip A40 workers and three inter V100 workers; all passed coordinate
reuse hash checks and started new normalization. Inference remains gated.
Confirmed every initial worker completed its zero-shear normalization node,
with finite positive mass, and four completed the new-centre node too.
Normal preparation progress established; yield under the long-run policy.

## 2026-09-25 — Complete V3.9 20k comparison; do not adopt distance-only

Both recovery tasks16699600_17/_19 completed in16m52s/16m19s; strict
combine16697082 passed in27s, with20 complete partitions and20000 observations.
At fixed16384 random draws, Gaussian gives g1=0.02215861 ±0.00097257;
distance-only gives0.01864592 ±0.00080721 (robust one-SE). Against injected
0.02 these are +10.79% ±4.86pp and −6.77% ±4.04pp. Paired distance-minus-Gaussian
is −17.56 ±4.41pp. These are realization errors, not established calibration
biases. Distance-only moves from−16.24% at8192 draws to−6.77% at16384, failing
convergence. Gaussian's similar final g1 points do not prove numerical closure.
Neither supports0.3% accuracy or adoption of the simpler exact-set ranking.

Recorded full draw ladder and interpretation in
[V3.9 completed comparison](V39_MODEL.md#completed-20k-comparison).
No scientific code, defaults or model artifacts changed. Result.json preserves
all20-partition hash/identity checks and limitations (one mock/seed, one Newton
step,64-QMC normalizer). Next work would diagnose scores/curvatures using
exact sums on a fixed subset before another large run. Complete result saved;
no further GPU jobs submitted. Cancel obsolete delayed status audit16699607
and save its read-only snapshot immediately.

## 2026-09-25 — Recover two timed-out V3.9 inference partitions

Owner requested immediate status. Eighteen of20 partitions completed both
arms (18000 observations); cip tasks16697081_17 and_19 timed out at one hour
in the distance arm, last logged820/1000 and848/1000 rows. The afterok gate
correctly blocked combine16697082. Inter tasks completed in15–17minutes;
completed cip tasks required29–40minutes, with worse throughput in the last
wave. No combined result exists; partial-population estimates are not used.

Preserved incomplete partition directories under external
`timeouts_16697081/`. Submitted only17/19 as array16699600%2 on inter V100s,
32GiB host memory and a two-hour limit. Both arms rerun from the same pinned
code, observations and seeds; all18 completed partitions remain untouched.
Repointed combine16697082 to afterok:16699600 and verified scheduler acceptance.
External jobs.json records recovery; the read-only status script now includes
retry accounting, and audit16699607 is scheduled for +2hours. This audit
writes a receipt, not a task wakeup. Both retries started on th-cl-nv02
and reached INFERENCE_START after ~161seconds loading. The complete20k
result remains pending.

## 2026-09-25 — Enable cip and inter for V3.9 inference

Owner requested both partitions. Updated pending tasks16697081_[1-19] in
place to `Partition=cip,inter`, generic one-GPU allocation and32GiB host
memory, retaining the single array-wide six-worker cap. Excluded8GiB cip
GPU nodes and the older met visualization GPUs; eligible cip virtual nodes
have at least16GiB GPU memory. Smoke16697080 completed both arms and the
same-draw check with peak RSS10995672K (~10.5GiB), supporting32GiB host
allocation at unchanged object_chunk4. Task0 was already running on an
inter V100 and remains untouched. Assembly16697079 completed successfully.
No seeds, partitions of observations, models, numerical code or dependencies
changed. Hardware can differ between tasks; the paired arms run on the same
GPU within each task. Scheduler accepted and verified all pending updates.
External `jobs.json` records the resource amendment; `AGENTS.md` records the
owner's partition authorization. Three cip A40-16GiB tasks started alongside
the existing inter V100 task. All loaded their inputs; two cip workers have
completed their first four inference rows, and the inter worker has reached
164/1000 rows in its first arm. The remaining cip worker is still compiling
its first batch. No failures observed. Other tasks wait on availability/quota;
results and final validation remain pending. Existing status audit retained.

## 2026-09-25 — Increase V3.9 mock-study inference to six GPUs

Owner explicitly requested more GPUs for the current 20k model-generated
study. Updated live array16697081 with `scontrol update JobId=16697081
ArrayTaskThrottle=6` (previously2), retaining its `inter` V100 allocations
and afterok dependency on smoke16697080. The20 fixed observation partitions,
paired arms, models, seeds, protocol and pinned execution sources are unchanged.
Updated `AGENTS.md` with this scoped authorization and external `jobs.json`
with the scheduler amendment. No other SBSI GPU jobs were active or pending
outside this chain. All28 preparation tasks16697078 completed successfully;
CPU assembly16697079 is running, smoke and inference remain dependency-gated.
Scheduler accepted the throttle update; scientific results remain pending.
The existing one-shot status audit16697094 remains queued.

## 2026-09-25 — Start V3.9 model-generated 20k sampling comparison

Owner requested20k observations and confirmed model-generated V3.9 mocks.
New `sbsi/v39_inference.py` implements physical0.5/1/2arcsec crowding over all
nonself neighbours within10arcsec, and the shared F8/C8 shear-state cache.
`scripts/prepare_v39_study_features.py` rebuilds features for the28,012,032-atom
expanded prior, keeping every supplied bright row, original weights and donor
environments. Emulator pairs remain the separately pinned nearest20 geometry.
No truth cuts. `scripts/run_v39_sampling_study.py` generates20k retained mock
measurements at g=(.02,0), measured radius>0.6arcsec and magnitude<25.8, then
prepares new V3.9 proposal/classifier/selection caches and compares Gaussian
and distance-only exact sets with identical random proposals/draws. Both arms
use K1024/M16384, full nine-node stencil, common normalizer and mean-observed
initial centre. Retain all prefix rungs; primary comparison uses fixed M16384,
not a method-dependent adaptive stopping count. No production defaults change.

External root: `inference_v39_mock20k_distance_20260925_v1`. Protocol and
execution manifests pin sources, models, scripts and isolated runtimes.
Features use the preserved original runtime plus the new module; inference
uses a separate copy with the corrected required base-cache geometry arguments.
This constructor issue was caught by first full-suite job16696807; replacement
16696823 passes296 tests,1 skip. Added tests cover all-neighbour support,
retention of faint secondary neighbours, self removal, scene isolation, shear
invariance of spin0 features and wrong-model rejection. Ruff/compilation/Bash
checks pass. Feature arrays16696810 and16696822 run at two CPU workers each;
GPU stages will preserve the combined two-GPU limit. Numerical preflight and
subsequent mock/cache/inference stages pending; no results or0.3% claim yet.

All28 feature shards completed, total28,012,032 atoms and prior mass
1.0000000000000024. Initial density preflight16696887 correctly rejected one
test draw outside measured selection; corrected the preflight to test retained
draws.16696925 and final pinned-source16696960 pass at shears(.02,0) and
(-.015,.01), with127 selected rows each and maximum standardized log-density
differences3.43e-4 and1.57e-4 (within declared float32 tolerance). Strict cut
boundaries and unchanged spin0 features pass. Explicitly enable the fractional
location metric as well as fractional scatter flooring; they are distinct
runtime settings. Preserve earlier preflight script/execution snapshots.

Queued dependency chain: mock16696965 → preparation16696966[0–27]%2 →
assembly16696967 → paired8-row smoke16696968 → inference16696970[0–19]%2 →
combine16696972. Every downstream stage requires predecessor success; GPU
stages do not overlap and arrays permit at most two GPUs. The mock is running.
Study details are in [V3.9 model closure](V39_MODEL.md#20k-model-closure-sampling-study).

Mock16696965 stopped on an overly broad physical-validity assertion over the
entire generated batch; downstream stages were cancelled before execution.
Diagnostic16697019, then16697024, replayed the identical seeds: one of117897
detected draws rounds to the unit-disk boundary. It is selected object51871
out of68171 in the oversized batch, outside the requested first20000. No
nonfinite or nonpositive auxiliary draws occurred. Move validation to the
predeclared first20000 measured-selected observations and record all batch
rounding counts separately. Selection never uses the validity mask to replace
or redraw an object; any invalid retained draw still aborts. Seeds, population,
cuts and prefix remain unchanged. Partial support/mock files and the failed
execution snapshot are preserved under `failure_16696965`. Diagnostic16697012
was only a launcher failure from a node-local /tmp path; replacement used the
shared project path.

Replacement chain: preflight16697071 → mock16697074 → prepare16697078[0–27]%2
→ assemble16697079 → smoke16697080 → inference16697081[0–19]%2 → combine16697082.
This retains the two-GPU cap and all afterok gates. Results pending.

Replacement preflight16697071 passed. Mock16697074 completed in1m40s with
exactly20000 selected, strictly interior, finite positive observations, no
replacement draws and no truth cuts. Initial mean-shape centre is
(0.015691824930896287,0.0017382505470351646), injected shear(.02,0).
The unused batch boundary draw is recorded explicitly; every retained draw
passes forward/inverse transport checks. New V3.9 preparation16697078 is queued
next. Full inference estimates are not available yet.

Confirmed preparation startup on two V100s: shards0/1 completed proposal
coordinates and their first two selection-normalization nodes, with finite
positive masses (~0.01288/0.01291 per shard). The mock includes2127 selected
targets with true r>=26; forward/inverse maximum shape difference4.44e-16.
Fresh caches and both inference arms remain pending completion. Scheduled
one-shot Slurm status audit16697094 for about two hours later; it writes
`status_latest.json`. No task-wakeup/automation tool is exposed in this
session, so this is a saved scheduler audit, not a promised task notification.
Final Ruff and whitespace checks pass. Yielding after normal GPU progress,
per the long-run monitoring policy; no inference default or acceptance claim.

## 2026-09-25 — Compare simpler atom-selection scores

Owner requested a simpler, more principled sampling score. Added
`scripts/compare_simple_atom_scores.py` and its Slurm launcher. GPU16696646
compares the existing floored Gaussian with nearest-location distance, a
prior/usability-weighted shared-scale kernel, and the Gaussian using raw
flow scatter. All use the existing 131072 location prefilter and fixed
metric; compare K1024/M16384 and K8192/M8192 on the three existing exact-sum
observations, both priors, and all seeds8701–8716. Only deterministic
selection changes: full-support sampling probabilities and draws stay fixed.
Also calculate exact repeated-draw variance from the full finite-prior sum,
since sixteen seeds can miss rare large importance weights. No production
change or shear-bias claim. Ruff, Python compilation and Bash syntax pass.
Results pending under `inference_simple_scores_20260925_v1`.

Startup16696646 exposed a historical artifact-layout difference: baseline
draws are stored in the original atom-map NPZ, not the extension comparison
directory. Corrected the diagnostic reader and resubmitted as16696664 to
`inference_simple_scores_20260925_v2`; the failed source snapshot/log remain
preserved. No model evaluation or scientific settings changed.

GPU16696664 completed in1m46s, peak RSS11.57GiB. All six baseline exact sets,
seed8701 draw lists, full log sums, normalized probabilities and sixteen
reference estimates replay successfully; input/runtime hashes and frozen
metric checks pass. [Report and complete metrics](figures/inference_simple_scores_20260925/README.md#results)
published with source snapshots; linked from the maintained bright-prior doc.
Nearest-location at K1024 reduces expanded numerator RMSE from
84.897%/149.155%/87.967% to2.163%/7.006%/3.421%. Exact theoretical relative
SD remains3.717%/20.124%/8.084%. Raw-scatter Gaussian improves the original
prior more, but is less reliable on expanded support. Shared-kernel weighting
adds little in these cases. At K8192 nearest-location has slightly worse
empirical original-prior RMSE than current ranking; report preserves all arms.
Only the exact-set rule changed. These selected cases justify representative
testing of the simpler retrieval rule, not default promotion or a0.3% claim.
Ruff/Bash/compilation checks pass. No pending jobs remain.

## 2026-09-25 — Trace posterior-important atoms missed by the original sampler

Owner asked which atoms were missed in the annotated original-prior maps and
why. Added `scripts/diagnose_missed_atoms.py`, GPU16696539. Scope is fixed to
the20.23m original prior, observations142230/3563/409188, K1024/M16384 and
seed8701. Replays exact lists/draws and proposal probabilities, ranks important
missed atoms through the actual float32 location prefilter and float64 proxy
reranker, and reports their physical properties, raw/floored flow scatter,
expected draw counts and miss probabilities. No model or numerical settings
change. Ruff passes; job submitted, results pending.

GPU16696539 completed; exact-list/draw/probability/mass replay and independent
re-ranking checks all pass. [Published diagnosis](figures/inference_missed_atoms_20260925/README.md#dominant-missed-atoms):
row3563 atom8865748 has90.0723% posterior mass, location rank13, Gaussian
rank1259, and61.0343% zero-hit probability in16384 draws. Row409188
atom6984342 has69.2913% mass, location rank9, Gaussian rank1257, and58.1699%
zero-hit probability. Both passed the prefilter but fell below K1024 and
were not drawn. Their e1 proposal scatter is broadened47–54fold and flux
scatter19–23fold relative to raw flow scatter. Different Sérsic profiles;
shared characteristic is bright, narrow actual likelihood under a broad
proposal. Other environment realizations of the same intrinsic properties
were sampled, explaining the smaller reached fractions. Original row142230
dominant atom is protected by Gaussian rank234 despite zero random hits.

Saved full per-atom evidence and source under the published report directory.
No settings change. Reserving nearest-location candidates before re-ranking
is a targeted next hypothesis; these examples alone do not establish a
population-wide fix. No pending jobs remain.

## 2026-09-25 — Test 8192 exact plus 8192 sampled atoms

Owner requested the8k+8k split after the bright-prior expansion comparison.
Generalized `scripts/bright_prior_extension.py compare` with explicit exact
and sampled budgets and output directory; the scheduler wrapper forwards
arguments. Preserve the original1024+16384 defaults and source snapshot.
GPU16696513 runs K8192/M8192 on both original and expanded priors, the same
three observations and all16 seeds8701–8716. Model/likelihood, prefilter131072,
metric, floors and prior weights stay fixed. Existing exact centre sums are
reused; no shear fit or default promotion.

Added `scripts/report_inference_budget.py`; dependent CPU16696518 compares
both budgets, independently replays both priors' saved draws, verifies exact
sets are nested, and checks full log sums. Report also measures exact posterior
mass and dominant-atom capture. Ruff and Bash syntax pass. GPU startup is
normal; results pending. External results: `inference_bright_extension_20260925_v1/comparison_8k8k`.

Completed GPU16696513 in1m56s and CPU16696518 in14s. All six independent
log-sum/draw replays and nested-exact-set checks pass; inspected the PNG.
[Published results](figures/inference_8k8k_20260925/README.md#results) show that
K8192/M8192 resolves these three centre-integration cases on the original prior:
relative numerator RMSE is0.00930%,0.04321%,0.00491% over16 seeds, with
99.982%,99.887%,99.978% of exact posterior mass summed deterministically.
These are RMSEs, not maximum errors or shear bias.

For the expanded prior, corresponding RMSE is1.861%,267.712%,87.365%, versus
84.897%,149.155%,87.967% at K1024/M16384. Row142230 recovers its dominant atom
and97.319% exact mass, but rows3563/409188 still miss their dominant atoms
and sum only0.656%/1.634% exactly. Larger K helps but does not resolve ranking
on expanded support; halving the sampled remainder can increase tail variance.
No default change and no full shear fit. Both priors, original results and
all downloaded rows remain preserved. Ruff/Bash checks pass; no jobs remain.

## 2026-09-25 — Download and evaluate the supplied full bright catalogue

Owner supplied FS2_bright and authorized adding all rows to the existing prior.
CPU16696367 downloaded the389.8MB feather and README next to the actual
FS2_25876 source, checked file hashes, and audited7,780,811 unique physical
galaxy IDs. Overlapping151,828 source identities have different assigned
morphologies;848,650 rows are outside the old generator domain. Retain all
new rows and explicitly report the changed prior. Details and frozen protocol:
[bright extension](BRIGHT_PRIOR_EXTENSION.md#weights-neighbours-and-identities).

Added `scripts/bright_prior_extension.py` and scheduler wrapper
`jobs/job_bright_prior_extension.sh`: independent complete neighbour contexts,
canonical target/pair rescaling checks, fixed bright mass, centre-only exact
likelihood comparison on the original three rows, and16 fixed seeds for each
prior. Uses the previously verified external runtime; baseline caches and
default models are untouched. Construction CPU16696388 submitted. Ruff and
Bash syntax checks pass. GPU stages and results pending; no shear-bias or
improvement claim. All raw/download/audit receipts are external under
`inference_bright_extension_20260925_v1`. CPU16696388 completed in2m37s
(peak2.17GiB): all28,012,032 combined atoms represented, new pair rescaling
checks passed. GPU array16696392 started with two V100 workers and an
eight-shard/two-worker throttle. Comparison16696395 depends on its success.

Startup16696392 caught malformed NPY headers: NumPy scalar shape dimensions
were serialized as `np.int64(...)`. Cancelled remaining workers and dependent
comparison. Corrected dimensions to Python integers and added file-reopen
checks; preserved the failed construction separately. Rebuild16696407 passed
in2m19s. Replacement GPU array16696409 has two workers making flow-coordinate
progress; comparison16696411 and CPU report16696424 are dependent stages.
Added `scripts/report_bright_prior_extension.py` for the paired16-seed plot,
RMSE against each exact numerator and posterior-atom ESS. Ruff checks pass.

Owner clarified that the parent distribution is intended to be the same.
Corrected the overly strong interpretation of different individual morphology
assignments: these alone demonstrate no distribution shift. Added descriptive
source-distribution and magnitude-conditioned checks in
`scripts/audit_bright_distribution.py`, CPU16696435. Its matched-generator-domain
audit is distinct from retaining all rows in the primary experiment.
Final CPU16696444 also preserves original float32 boundary comparisons and
matches the saved new-domain mask exactly. Raw magnitude/radius/n/q maximum
CDF differences are0.197/0.685/0.294/0.300pp; median n is2.198459 in both.
Old/new generator exclusions are11.2206%/10.9070%, predominantly n=6.
These descriptive checks support additional realizations of the intended
same parent; they do not prove equality of the complete joint distribution.

All eight GPU shards completed in approximately2m26s–2m38s each. First
comparison16696411 and report16696424 completed; independent log-sum and draw
replay checks passed, but source review caught an implicit proposal metric
refit: `ProposalCoordinateTable.__post_init__` recomputes fractional-coordinate
quartiles even when its public centre/scale fields are supplied. This violated
the prospectively fixed-metric comparison. Preserved those results externally
as `comparison_implicit_metric_refit` / `report_implicit_metric_refit` and marked
the initial reported sampler outcomes provisional. Freeze the derived metric
too and assert exact standardized-coordinate equality on every10000th baseline
atom. Corrected comparison16696478 and report16696481 reuse all valid density
evaluations; no model or target likelihood changed.

Corrected comparison16696478 completed in1m30s; report16696481 completed in9s.
All three independent log-sum/draw-replay checks pass, the original seed8701
baseline is reproduced, and the fixed metric identity assertion passes.
Final [results](BRIGHT_PRIOR_EXTENSION.md#results-and-decision): relative
centre-numerator RMSE over16 seeds changes0.000311→0.848967,
0.927664→1.491547 and1.013688→0.879673. Posterior atom ESS improves modestly
1.15→1.64,1.22→4.73,1.74→2.66, but integration does not reliably improve.
For row142230 an atom carrying77.62% of expanded posterior mass is displaced
from the exact list and missed by seed8701. Outside-generator-domain rows
carry only0.0381%,2.5173%,1.3497% posterior mass; not the main failure.

Published PNG/PDF, summaries, independent validation, distribution audit and
source snapshots under `doc/figures/bright_prior_20260925/`; inspected PNG.
Saved an explicit experimental weighted-union manifest externally as
`combined_prior.json`. Retain all new rows and the unchanged total bright
mass, but keep the baseline default: no representative inference improvement
or shear-bias claim. Next work should improve candidate ranking and establish
numerical convergence before a full fit. Removed only this task's malformed
temporary arrays, retaining failure receipts. Ruff/Bash/whitespace checks
pass; all jobs complete and no new jobs remain.

## 2026-09-25 — Remake sampled-atom maps with the latest tested inference proposal

Owner requested the earlier scatter plots using the current inference
algorithm. Fixed the original three diagnostic observation rows142230/3563/409188
before evaluation. Added `scripts/plot_current_inference_atoms.py` and
`jobs/job_current_inference_atom_maps.sh`. Uses the 20,231,221-atom weighted
bright-enriched prior, median scatter floor, fractional flux metric, K1024,
prefilter131072, M16384, seed8701 and existing initial shear centre from
job16631250. These are V3.6-like model artifacts, not V3.9 inference.

Snapshot commit773295d into the external `inference_atom_maps_20260925_v1/runtime`
directory: every implementation hash matches the saved reference result.
The live proposal-comparison branch only differs in its later truth-cut store
extension; its sampler is unchanged. No live worktree or prepared cache is
modified. The old plotting script chose its exact stratum by top mixture
probability; the production driver uses location-prefilter candidates.
The new plot calls the actual production candidate and draw methods instead.

Each row gets a full-prior centre-node density scan, top64 posterior atoms,
actual reached mass and estimated/exact numerator ratio, plus measured-space
and intrinsic-truth scatter panels. Nonpositive proposal flux means are
reported and omitted from the logarithmic axis. Saved log terms and draw IDs
permit independent checks. A40 job16696306 was cancelled while pending due
to unavailable resources; replacement Slurm16696310 started on th-cl-nv01
(one V100-SXM2-32GB, four CPUs,96GiB,30minutes). Resource-check and Matplotlib
skills applied; scheduler allocations bound usage rather than whole-node
resource-detector recommendations.

Completed [figures and results](figures/inference_atoms_20260925/README.md#results):
full-prior posterior mass reached is99.9247%,9.4684%,30.6568% for
rows142230/3563/409188. Estimated/exact likelihood ratios are0.999554,
0.376566,0.561436. The latter two rows miss dominant atoms; many distinct
sampled atoms do not guarantee useful coverage. These are centre-density
diagnostics, not new shear estimates.

Validation: GPU16696310 completed in9m50s. Rendering16696335 completed in16s;
independent CPU verification16696321 completed in6s, checking output hashes,
full-prior normalization, reached mass, likelihood ratios and sample/exact
membership. Row3563 reproduces the saved production count of14443 distinct
complement atoms. All checks pass. Inspected all three final PNGs; corrected
overlapping title text and painted highest-mass atoms last so minor atoms
cannot obscure them. PNG/PDF, plot-data NPZ, reports, resource/provenance
receipts and both calculation/rendering source snapshots are preserved in
`doc/figures/inference_atoms_20260925/`. Full log terms remain external.
Python syntax/Ruff, launcher Bash syntax and documentation whitespace/link
checks pass. Updated the repository map. No training, default change or new
shear estimate. These are three previously inspected failure cases at one
centre, not independent or population-level validation. No jobs remain and
no further experiments are scheduled.

## 2026-09-25 — Recover later inference tests from other worktrees

Owner requested a cross-branch check after the recap missed later inference
tests. Inspected all eight registered worktrees, branch histories, relevant
working-tree changes, and external result receipts. The newer numerical work
is on `worktree-proposal-coord-compare` (also inherited by `worktree-nocut-retrain`),
not the current `dev` checkout. Added the
[cross-worktree results](INFERENCE.md#later-cross-worktree-inference-tests)
without merging code or changing defaults/models/caches.

That branch declares v1.3-infer: K1024/16384 tilted-stratified inference with
median proposal-scatter floors and fractional flux distance. The saved disk
experiments remain labelled custom, based on v1.2-infer-16k, with original
V3.6 hashes. A weighted 20,231,221-atom bright-enriched prior gave positive
information on two separate 10k windows, jobs16631250/16631862. Pooled
g1=0.03768952 with robust SE0.01724200; injected g1=0.02. Earlier branch
entries confused the Newton step with final shear; the later entry and this
independent re-reduction include the starting centre. The r<27/10,237,320-atom
comparison (16637857/16637858) fails pooled positive definiteness.

Validation: read four result.json receipts, verified all four moment-file
SHA-256 values, and independently reduced the small saved score/information
arrays with single-threaded NumPy. Reproduced both windows, both pooled
information eigenvalues, and robust covariance from per-object influence
residuals. These checks re-read existing outputs; no new likelihood evaluation
or scheduler job. Documentation whitespace/link checks pass. The separate
engineering-review branch adds config binding and tests; sampler-night's
latest committed preflight is August30, not a newer disk inference result.

Limitations: two windows from one catalogue/one prior seed, heavy importance
weights and large robust errors; no robust 0.3% inference claim. V3.9 models
were not used. No branch was merged and no experiment restarted. Next steps:
none scheduled; preserve branch provenance in future pipeline recaps.

## 2026-09-25 — Freeze V3.9-like and close development

Owner requested wrap-up, the V3.9-like name, and thorough repository cleanup.
Pinned the existing F8 flow, primary seed20260913 C8 classifier and unchanged
physical-moment disk emulator in `configs/models_v3_9_like.json`. Artifact
SHA-256 values were verified against the external files. Added the named
preset through a shared disk-model manifest loader in `sbsi/models.py`;
V3.5/V3.6 identities and configured likelihood/estimator remain unchanged.

[The model guide](V39_MODEL.md#development-evidence) records measured-only
per-leg radius >0.6 arcsec/magnitude <25.8 cuts, no truth-property evaluation
cuts, the target-role parent, +0.020770% ±0.265152 percentage-point development
bias, seed replicate, failed broader classifier checks and absence of
independent acceptance. Copied original comparison/protocol/training/selection
receipts into `doc/evidence/v3_9_like/` and pinned their hashes in the manifest.
No training, recalibration, favorable seed selection or emulator reduction.

Moved 37 completed experiment scripts, 38 fixed-run launchers, 15 experimental
test files and five investigation documents out of the active tree, preserving
all bytes under `archive/research-2026-09-25/`. Its hash manifest also records
66 snapshots of pre-existing dirty working-tree files. Retained reusable
preparation/inference/audit code and core regression tests. Renamed model-preset
tests to `tests/test_model_presets.py`; added V3.9 identity/evidence checks and
an optional real-artifact prediction test. Removed unused circularized-feature
training helpers and the historical LRU-patch cache-identity exception: cache
reuse now requires exact scientific and implementation identity. Old prepared
caches must be rebuilt; the disk preparation driver is still V3.6-specific.

Updated README, model guide, API, conventions, inference/environment guides,
AGENTS and [repository map](REPOSITORY.md#september-25-cleanup). Added the
reusable one-CPU Slurm test launcher `jobs/job_tests.sh`. Scientific figures,
large artifacts, datasets, external research checkouts and result files are
preserved. No git reset/clean or commit was performed.

Validation: all four external model/metadata hashes match; all 161 archive
entries pass byte-count and SHA-256 checks; active Python syntax, config JSON,
local imports and maintained-document file links pass. Changed-file Ruff
and `git diff --check` pass. Final full suite, including external V3.9 model
loading and joint prediction: **294 passed, 1 skipped** in Slurm16696170 using
py31 (52.98s); the system Python run16696166 also passed294/skip1 (36.50s).
[Verification receipt](evidence/v3_9_like/cleanup_validation.json) pins source
hashes and preserved test logs.

Initial jobs16696148/16696162 exposed two test portability problems: NumPy
boolean indexing of a Torch tensor and a small random XGBoost fixture that
did not necessarily split on both conditioned outputs. Fixed the test to use
an explicit Torch boolean mask and a balanced factorial fixture; precision,
physical-prediction and exact-pooling assertions remain intact. No production
model calculations were changed for those repairs. The rerun revealed stale
shared bytecode; generated bytecode caches were removed. The reusable test
launcher now uses fresh job-local bytecode and `SBSI_TEST_PYTHON` so an inherited
system `PYTHON` cannot silently override the project interpreter.

Next steps: none scheduled. Development is closed. A future V3.9 inference
integration or independent calibration study requires a new owner request;
model naming alone does not establish robust 0.3% accuracy.

## Earlier work logs

- [Full log before this cleanup](../archive/research-2026-09-25/snapshots/doc/WORKLOG.md)
- [September20 snapshot](../archive/research-2026-09-20/doc/WORKLOG.md)
- [September14 snapshot](../archive/research-2026-09-14/doc/WORKLOG_FULL.md)
