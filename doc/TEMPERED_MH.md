# Tempered continuous MH for the two-mode trap

## Finding — October 2, 2026

Likelihood tempering (replica exchange) plus a narrow-step mixture lets bright
chains move between the "clean target" and "faint neighbour shares light"
modes found in the sampling deep dive. On a 40-galaxy test panel (32 galaxies
that had a stuck chain, 8 that did not), half as many steps give much better
chain agreement than the original sampler. Most galaxies still miss the
strict R-hat < 1.01 rule, and the sampler costs about 7.5x more GPU time per
galaxy. This is a sampler diagnostic, not a shear-recovery or bias result.

| 40 galaxies, 2 replicas x 4 chains | tempered, 16k steps | original, 32k | original, same 12k kept draws |
| --- | --- | --- | --- |
| Shear-ratio check passes (R-hat<1.01, ESS>400) | 15/40 (38% ± 8%) | 2/40 (5% ± 3%) | 1/40 |
| ... galaxies that had a stuck chain | 13/32 (41% ± 9%) | 1/32 | 0/32 |
| Median R-hat: log posterior / magnitude / blend response b / ratio .022 | 1.037 / 1.047 / 1.023 / 1.021 | 1.249 / 1.255 / 1.120 / 1.103 | 1.291 / 1.330 / 1.136 / 1.122 |
| Median spread of 8 chain-mean log posteriors | 4.3 | 11.3 | 10.5 |
| Galaxies with a chain >= 8 log units below the best | 6/40 | 32/40 | 28/40 |
| Replica shear-score disagreement abs(z) < 2 | 40/40 | 35/40 | 38/40 |

Uncertainties on fractions are binomial. All 25 failing galaxies fail on
R-hat (median about 1.03); 9 also have ESS <= 400. Per-galaxy shear score
(finite difference of the log ratio between g1=.018 and .022), tempered minus
original: mean -0.01 ± 1.31 over 40 galaxies, no galaxy beyond 3 sigma. So the
mode choice does not produce a detectable average score shift at this panel
size. The original errors are unreliable because its chains were not mixed.

Adjacent swap acceptance after warmup is 0.66–0.78 for all 15 rung pairs.
Every posterior-level slot exchanged 2266–5446 times (median 4602), and no
chain stayed put. Figure: `plots/galsbi_tempering_panel16k.png`.

## Method

`scripts/check_galsbi_local_sampler.py --betas ... --narrow-step-probability p`
(requires `--joint-morphology`). Level target = prior x kernel^beta; level 0
(beta=1) is the posterior and is the only level stored. After each MH sweep,
deterministic even/odd adjacent pairs swap states with probability
min(1, exp((beta_l - beta_{l+1})(L_{l+1} - L_l)))
(`sbsi.mcmc_proposals.adjacent_swap_permutation`). Step-size adaptation stays
with each level and freezes after warmup. With probability p, a morphology
step uses 0.25x the frozen scale. The choice does not depend on the state, so
the move stays exact. With `--betas 1` (default) the sampler reproduces the
unmodified script bit for bit (max difference 0.0 on all saved arrays, before
and after the speed-ups).

Panel run: 16 rungs spaced for equal swap rejection from a 2048-step pilot,
beta = 1.0 ... 0.005. Settings: p=0.5, 16384 steps, 4096 warmup, bright-run
flags, measurement initialization. Each replica is split into two 20-galaxy
jobs (4 inter GPUs total).

## Speed-ups (exact)

Disk-response emulator on all allotted CPU threads: tree prediction splits
rows only, and the output is identical. (Correction: in GPU jobs this stayed
single-threaded until the fix in [Speed](#speed-thread-fix-and-shared-gpus).)
A memo of PyCosmo angular distances is
exact because the call is elementwise. Step time went from 0.37 s (2560-row
full replica, before speed-ups) to 0.19 s (1280-row half replica, after). The flow
uses fixed padded density batches, so halving the rows does not halve GPU time.

## Cost

About 170 GPU seconds per galaxy per replica at 16k tempered steps, against
22.5 at 32k original steps. That is about 7.5x per galaxy, or about 15x per
step, because 16 temperature copies run.

## Limits and next steps

Post-hoc stress panel chosen from the same run's diagnostics, model closure,
one 16k configuration. Strict convergence still fails for 25/40, so more steps
are still needed. Swap rates are high everywhere, so a sparser ladder (about
8–10 rungs) is the obvious cost cut and should be tested. A joint
target+environment mode-jump move remains the cheaper alternative
(`galsbi_sampling_deepdive_20261002/README.md`). No 20k run was launched.

## Follow-up: more draws and the still-stuck galaxies

**More draws help.** The panel16k comparison was rerun on only the first 6144
kept draws (`compare_tempering.py ... 6144`, job 16795404). Going from 6144 to
12288 draws gave:
- worst shear-ratio R-hat lower for 39/40 galaxies;
- median R-hat - 1 from 0.036 to 0.023;
- pass rate from 9/40 to 15/40;
- median ratio ESS from 1026 to 1340.

The remaining failures are mostly too few effective draws, not a hard trap.

**Two kinds of stuck galaxy.** `stuck_galaxies.py` (job 16795409) tracked the
six galaxies with a chain 8+ log units low (10, 42, 125, 149, 155, 234). It
uses 1536-draw block means of log posterior, magnitude, radius, b and
neighbour count; figure `stuck_galaxies.png`.
- 10, 125, 149, 234: chains go down into the low state and come back within
  the run. The last-quarter flag catches whichever chain is on an excursion.
  These should improve with more steps.
- 42, 155: one chain sits in the same low state for all 12288 draws. The
  original run has the same state in its own chains (obs 155: 2/8 chains, lp
  about -1 against about 10; obs 42: 1/8). Final scenes (one snapshot per chain) show a **bright
  companion**: a neighbour about as bright as the target, 3.1"/m_r 18.6 for
  obs 155 and 3.5"/m_r 20.6 for obs 42. The target is about 0.3 mag fainter to
  compensate. No healthy chain has a neighbour brighter than m_r 21.5 within 4".
  This is a different mode from the faint-neighbour mode in the deep dive.
  Likelihood tempering did not remove it in 16k steps. Only cold levels are
  saved, so whether the hot levels leave it is untested. Removing the
  companion while brightening the target in one step is the natural joint move
  for it.

Per-chain shear scores are heavy-tailed: last-quarter chain means reach -174
and +142 against typical values of about 10. This drives the large score
errors of obs 42 and 125, and has not been investigated.

## Sparser ladder: 8 rungs, 32k steps

8 rungs (`make_ladder.py 8`: 1, .7883, .6167, .4449, .2849, .1473, .05014,
.005), 32768 steps, 8192 warmup, fresh seeds, same panel and split
(`submit_ladder8.sh`, jobs 16795399–16795402, 58–63 min each, all 0:0).
Comparison: `ladder8_32k_comparison.json`; traces: `stuck_galaxies_ladder8.*`.

| 40 galaxies | 8 rungs, 32k | 16 rungs, 16k | original, 32k |
| --- | --- | --- | --- |
| Shear-ratio check passes | 16/40 (40% ± 8%) | 15/40 (38% ± 8%) | 2/40 |
| Median R-hat: ratio .022 / log posterior / magnitude | 1.016 / 1.037 / 1.064 | 1.021 / 1.037 / 1.047 | 1.103 / 1.249 / 1.255 |
| Median spread of chain-mean log posteriors | 3.9 | 4.3 | 11.2 |
| Galaxies with a chain >= 8 below the best | 7/40 | 6/40 | 32/40 |
| Replica score disagreement abs(z) < 2 | 38/40 | 40/40 | 35/40 |
| Adjacent swap acceptance after warmup | 0.43–0.52 | 0.66–0.78 | — |

Same GPU time per galaxy (about 180 against 170 GPU s per replica), twice
the stored draws, nearly the same quality. Only 10 galaxies pass in both runs,
so passes near R-hat 1.01 move between runs by chance. Stuck in the 8-rung
run: 10, 42, 76, 79, 125, 155, 234. Obs 155 still has one chain at
m_r 18.75 against about 18.32 (bright companion). Obs 125 and 234 have chains
with the target 1–2 mag fainter (m_r 21.6–22.6 against 20.2; 20.7–20.8
against 19.8). The sparser ladder is not a cost cut per result.

## Light-transfer joint move: no gain, removed

The owner asked for joint target+neighbour sampling.

**The move.** Pick one neighbour within 4". Step the target m_r by delta
(equal mix of sd .05 and .3 mag). The neighbour's r flux changes by
(F_t - F_t')/w, with w = exp(-d^2/2s^2) and s log-uniform in [.5, 4]".
Repeating the move with -delta undoes it. The acceptance ratio is target
prior × neighbour prior × likelihood × F_n/F_n'. A toy stationarity test
passed, and the same test failed by more than 10σ without the Jacobian. An
earlier version stepped the neighbour's brightness instead; it was accepted
56% of the time but moved the target by a median of 0.000 mag, because near
neighbours are mostly m_r 24–27.

**Panel test.** p=.1, 32768 steps, same seeds and split as the 8-rung run.
Each arm was compared with a same-seed control, because the panel was chosen
from the original run's stuck chains.

| 40 galaxies | plain + LT | plain control | 8 rungs + LT | 8 rungs control |
| --- | --- | --- | --- | --- |
| Shear-ratio check passes | 6/40 (15% ± 6%) | 5/40 (13% ± 5%) | 15/40 (38% ± 8%) | 16/40 (40% ± 8%) |
| Median R-hat: ratio .022 / log posterior | 1.061 / 1.149 | 1.038 / 1.137 | 1.018 / 1.042 | 1.016 / 1.037 |
| Galaxies with a chain >= 8 below the best | 17 | 15 | 9 | 7 |

"Plain" means no tempering and no narrow steps.

Light-transfer acceptance had a median of 1.8–1.9% per galaxy (range
0.2–13%, counting impossible attempts). About half of attempts need more
light than the neighbour has. In the plain arm, accepted target changes had
a median of 0.010 mag, and 1.7% exceeded 0.3 mag. Obs 42 and 155 kept their
bright-companion chains in every arm (obs 155: target 18.80 next to a
neighbour of 19.12 at 2.7", log posterior about 10 below the other chains).

The likelihood does not treat target light and light from a neighbour 3"
away as interchangeable along this simple weight. A CPU step-size scan
(`scan_summary.json`) showed accepted changes no larger than the ordinary
within-chain m_r spread of 0.02–0.07 mag. The code was removed again. It is
kept in commit b2b72cc and the study's `source_snapshot/`; reverting the
removal commit restores it.

Root `$DATA_DIR/sbsi/catalogue_prior/galsbi_light_transfer_20261002/`:
- `plain_32k`, `plain_nolt_32k` and `ladder8_32k` runs, plus their `*_comparison.json`;
- `light_moves_*.json`, `stuck_galaxies_*`, and `identity/` (max difference 0.0);
- `v1_neighbour_step/` and `v2_target_step/` (earlier checks).

All jobs exited 0:0.

## What to improve next: diagnosis of the 8-rung and untempered runs

Read-only analysis of the saved chains of the 8-rung run (`ladder8_32k`) and
its same-seed untempered control (`plain_nolt_32k`): 40 galaxies, 2 replicas
× 4 chains, 24576 kept draws per chain. "Score" is the per-galaxy shear score,
the finite difference of the shear log ratio between g1=.018 and .022. Unless
stated otherwise, numbers are medians over 80 galaxy-replicas ± bootstrap or
binomial errors. Figure: `plots/galsbi_improve_diagnosis.png`.

**1. The average shear score is already reliable; the R-hat rule fails on spread.**
Rank R-hat is the larger of two checks: chains agree on the average (bulk part),
and chains agree on the spread (tail part).

| 80 galaxy-replicas | 8 rungs | untempered |
| --- | --- | --- |
| Score R-hat ≥ 1.01: rule used / average part / spread part | 44 / 10 / 44 | 56 / 15 / 56 |
| Median score R-hat, average part | 1.0016 ± 0.0015 | 1.0022 ± 0.0018 |
| Ratio .022 R-hat ≥ 1.01: rule used / average part | 44 / 13 | 54 / 25 |

The replicas' average scores agree as well as their error bars allow. Replica
difference / error, rms over 40 galaxies: 1.14 ± 0.13 (8 rungs) and
1.24 ± 0.14 (untempered); 1/40 and 0/40 beyond 3σ. Chain means within a
replica: median χ²/dof 0.90 and 0.87. Effective draws counted from 512–4096-draw
block averages match the standard count (8 rungs: 2360–2580 against 2530 ±
290). So there is no hidden slow part in the score.

**2. Tempering does not change the shear score.** Both replicas start from the
same measurement-based state and could share a trap, so the two samplers were
compared directly, galaxy by galaxy. 8 rungs minus untempered: rms 1.07
error bars, none beyond 3σ. Weighted average difference -0.55 ± 0.45 score
units (log-mean-exp) or -0.32 ± 0.45 (plain mean), on a panel chosen for stuck
chains. For scale, the 238-galaxy original run has a mean score of -0.83 ±
0.88, and the between-galaxy spread on the panel is 7.6–9.6. The trap shows up in
log posterior, magnitude and size, not in the shear score. This panel cannot
rule out a bias as large as the full-sample statistical error (2σ bound about
1.4); a larger comparison would.

**3. The real limit is Monte Carlo noise per galaxy, and untempered runs buy
less noise per GPU hour.** Each draw's score scatters by 135 ± 20 units. The
target's shape, size, brightness and environment explain only 2–10% of it in
low-order fits. The likelihood is deterministic, so this is real posterior
spread, not numerical noise. The score forgets itself in 29 ± 1 (8 rungs) to
35 ± 1 (untempered) steps. Per-galaxy score error from one 4-chain replica: about 4–6 units.
That noise adds 36% (8 rungs), 44% (untempered, panel) and 26% (original 238)
of the between-galaxy variance. Untempered runs cost 50 GPU s per galaxy against
182. Precision per GPU second is higher untempered by 1.6 ± 0.6 (replica
differences) or 1.8 ± 0.5 (ESS error bars).

**4. The GPU is mostly idle in small jobs.** 80 chain-rows took 1000 s for 32k
steps; 640 rows (8 rungs) took 3340–3690 s. 8x the rows cost 3.6x the time.
Packing more chains per galaxy, or more galaxies per job, should cut the cost
per draw by about 2x. This is projected from these two timings, not tested.

**5. Slow quantities (matter for log posterior, not for shear).** Steps per
independent draw, 8 rungs / untempered:
- Sérsic: 1046 / 1367;
- size: 681 / 1398;
- magnitude: 617 / 1332;
- ellipticity strength: 530 / 1082;
- log posterior: 437 / 659;
- blend response b: 134 / 273;
- neighbour count: 64 / 108.

The joint magnitude/size/Sérsic move is far slower than a well-tuned
random walk in 3 dimensions would be: roughly 30x, a rough estimate. Moves per attempt (untempered; mean squared change / variance):
- the shape move does most of the score mixing (0.39);
- the magnitude/size/Sérsic move: 0.09;
- the environment refresh: 0.02.

The environment refresh is accepted 88% of the time, so its 10% partial
refresh steps are too timid. The redshift/SED moves never change the score;
they skip the likelihood and cost little.

**Ruled out.**
- *Heavy tails.* Excess kurtosis is 1.2 ± 0.2, and the top 1% of draws carry
  11% of the variance (8.5% for a Gaussian). Clipping the top and bottom 0.5%
  leaves R-hat at 1.0146 against 1.0145. The -174/+142 chain means noted
  earlier were short last-quarter windows.
- *A better estimator.* The log-mean-exp score and the plain average of
  per-draw scores agree: difference -1.3 ± 0.9, replica scatter 9.1 against 8.7.
- *Chain states predicting the score.* Correlations of chain-mean score with
  chain-mean magnitude, b, crowding, neighbour count, size or log posterior are
  all |r| < 0.08 ± 0.16.

**Ranked improvements.**
1. Judge convergence by what the shear uses. Use the average part of R-hat for
   the ratios, plus replica and sampler agreement of the score. Keep the
   log-posterior and magnitude R-hat as trap flags, not as a gate.
2. Spend GPU on more untempered chains, packed into fuller GPU jobs. Do this
   rather than on temperature rungs: Monte Carlo noise, not trapping, sets
   the shear error.
3. Bound the trap bias at full-sample precision. Run tempered and untempered
   on all 238 galaxies and compare the summed score (expected error about 0.2
   against the 0.88 statistical error).
4. Optional precision gain: average each stored draw over the target's
   orientation analytically. This costs extra likelihood calls only at
   post-processing, which takes about 1% of the sampling time. Untested.
5. Lower priority: a larger environment refresh fraction, and a better
   magnitude/size/Sérsic move. These help the log-posterior diagnostics,
   not the shear.

Receipts, under `galsbi_tempering_20261002/`:
- `improve_diagnostics.py` (job 16799151, 31 min, 0:0);
- `improve_followup.py` (job 16799358);
- `improve_rhat_split.py` (job 16799375);
- `improve_summary.py`, `improve_ztest.py`, `improve_arms.py`, `improve_plot.py`;
- the JSON outputs of the same names.

## Speed: thread fix and shared GPUs

Test: all 238 bright galaxies × 4 untempered chains, 512 steps, same seeds.
Times are seconds per step; the spread between processes in one job is
about ±4%, and between nodes about ±10%.

**1. The tree emulator ran on one core in GPU jobs.** The job script sets
`OMP_NUM_THREADS=1`, which also capped xgboost, whatever `nthread` or
`--cpus-per-task` said. `torch.set_num_threads(<allotted cores>)` lifts the
cap. GPU step 0.154 → 0.070 s (8 cores; 16 cores give no more), setup
306 → 118 s. The sorted-array distance memo cut its lookup from 10 s to 3 s
per run. Both are exact: chains, profiles and final scenes match the old code
with maximum difference 0.0 (GPU 8 and 16 cores; CPU-only 32 cores on the same
node).

**2. The rest is single-core Python, so share each GPU.** One GPU process
costs about 0.020 s per step plus 0.0052 s per 100 chain rows (two-point
fit). The GPU idles most of the time. `jobs/job_galsbi_shared_gpu.sh` runs P
processes on one GPU, each with its own cores and galaxies, with seeds
seed+k and bank seed+k. Results are unchanged: a shared-GPU process equals a
standalone run of the same galaxies and seeds (difference 0.0).

| one GPU, 238 galaxies | s/step |
|---|---|
| 1 process × 8 cores | 0.070 |
| 4 processes × 8 cores (2080 Ti) | 0.048 |
| 8 processes × 4 cores (2080 Ti) | 0.049 |
| 6 processes × 8 cores (P5000) | 0.044 |

The gain stops near 4–6 processes (about 1.5x per GPU).

**3. CPU-only jobs do not help much.** They have a high floor: 0.10 s per
step for 60 galaxies on 8 cores, and 0.13 s for 30 galaxies on 4 cores.
Node th-cl-hua28 also fails the initial-replay check (error 2.0e-5 against
1e-6, the same in two runs), so its CPU arithmetic depends on batch size. The
check stops such a run within 2 minutes.

**Projected 32k-step runs** (from the measured rates, not yet run at this
length):
- 238 galaxies: about 84 min on one GPU before; now about 25 min on one GPU,
  or 15–20 min split over 4.
- 1000 galaxies: about 6 h on one GPU before; now about 26 ± 3 min on
  4 GPUs × 48 cores (192 cores), at about 21,000 chain-row steps per second
  per GPU.

Limits:
- Shard outputs are separate run directories. The analysis scripts read one
  run at a time, so they must combine the shards.
- Throughput above 952 rows per GPU is extrapolated.

Receipts, under `galsbi_cpu_scaling_20261002/`:
- `submit_bench*.sh`: old-code arms 16799949–52 and 16799980–83.
- `fast/submit_fast.sh`: 16800041–45.
- `fast/submit_shards.sh`: 16800065–68.
- `fast/submit_replay_node.sh`: 16800079–80 (80 cancelled while waiting).
- `fast/job_shared_gpu.sh`: 16800083, 16800124 and 16800125.
- Launcher test: 16800133.
- `compare_arms.py`, `bench_summary.py`, `read_profiles.py`.

## Speed: exact prior memo and skipped Sérsic draws (2026-10-03)

Commit a5588b8 removes repeated per-redshift prior work:
- **Memo:** `ExactMemo` (in `sbsi/galsbi_prior.py`) remembers values of
  elementwise functions of redshift exactly: the Sérsic-index mass between
  its limits, the luminosity and angular distances, and the volume factor.
  It is a sorted-array lookup with a bounded reset.
- **Skipped Sérsic draws:** neighbour proposals rejected by the i-band
  window skip the Sérsic cdf/ppf. They still draw their uniform numbers, so
  the random stream is unchanged.

**Exact.** Same seeds, same node (met-cl-vis01):
- The 1024-step loop runs (plain, 1024 galaxies; tempered, 128 bright)
  match the old code in all 2096 and 304 saved items (difference 0).
- The 12 benchmark shards below also match.
- Test job 16805402: 386 passed, 3 skipped.

**Speed.** Same node and layout as the old benchmark (16805078): 6
processes × 4096 chain rows on one 2080 Ti, 1024 steps. GPU-seconds per
galaxy for a 32k-step loop:

| arm | old | new (16805404) | gain |
|---|---|---|---|
| plain, 4096 rows | 1.52 | 1.24 | 1.23x |
| tempered (8 rungs), 4096 rows | 12.66 | 10.45 | 1.21x |

Setup for 1024 galaxies went from 572 to 461 s. The spread between
processes is ±4%.

**Where the loop time goes now** (cProfile, 1024 steps, 4096 rows; 204 s,
was 255 s):

| part | seconds |
|---|---|
| tree emulator (xgboost) | 58.6 |
| scene inputs | 45.5 |
| environment refresh | 41.8 |
| target prior | 24.6 (was 60.4) |
| GPU likelihood | 15.3 |

The memo itself costs about 16 s. A smaller table only halves the search
cost (about 2%), so I did not pursue it.

**Production layout: cores, not GPUs, set the speed.** At 6 processes per
GPU the GPU is busy only 12–18% of the time. A smoke test (job 16805601,
V100) ran 15 processes × 2 cores at production size:
- Bright tempered: 0.14 s per step at 880 rows, about 3100 rows per second
  per core.
- Faint plain: 0.147 s per step at 1780 rows, about 6000 rows per second
  per core.
- Both are better per core than 8 cores per process (about 2100).

So use all allowed cores, at 2 cores per process. Give about 3/4 of them to
the bright tempered chains, which are about 80% of the work. Faint setup is
slow: about 0.9 s per galaxy on 2 cores.

**Memory and disk:**
- Stored traces take about 200 B per stored chain row per step, and they are
  copied once when stacked.
- A plain 4096-row process at 32k steps needs about 27 GB. Keep plain
  processes under about 2000 rows at 16k steps. Tempered runs store only the
  cold rows.
- The 15-process smoke test peaked at 21 GB.
- At 1M observations, the bright 11% (about 110k galaxies) would need about
  2.5 TB of chain files at 32k steps. The group filesystem had 18 TB free
  on 2026-10-03.

Receipts: `galsbi_speedup_20261003/` (`exact_job.sh` 16805403,
`speed_job.sh` 16805404, `compare_runs.py`, `summarize_speed.py`,
`speed_summary.json`, `logs/loop_*`); smoke test in
`galsbi_blind10k_20261003/smoke/`.

## Speed: b only for stored states at zero shear (2026-10-11)

**Idea.** The disk response b enters the likelihood only as the velocity
b·g (`catalogue_disk_likelihood._density`). The classifier and the flow
context never see b. So when chains sample at g = (0, 0), the MH kernel does
not depend on b at all, and the tree emulator (xgboost, the largest CPU cost)
does not need to run for proposals. With `lazy_b = persistent and not any(g0)`:
- proposals and measurement-initialization candidates skip the neighbour
  pair table and the trees; their b slot holds NaN (0 inside the kernel);
- `fill_b` computes b only for posterior-level (cold) rows still missing it,
  just before each trace is stored, so each distinct stored state is computed
  once, row by row, exactly as the full evaluation does;
- the initial evaluation and both audits still compute b for every row; the
  audits compare with `nanmax`.

At non-zero sampling shear nothing changes. Code: `scene_model_inputs(...,
response=False)` in `sbsi/galsbi_summary.py`, and `summarize`, `fill_b` in
`scripts/check_galsbi_local_sampler.py`.

**Exact.** Same seeds, same node (cip-cl-h01g02n3, job 17129212): the chain,
initial state, final scenes, local profiles and per-observation diagnostics
match the old code bit for bit, for bright (768 steps) and faint (512 steps)
units. Initial replay 0.0; invariant-audit summary change 2.8e-14 and
log-kernel change 0.0, both as before. At g = (0.02, 0) (job 17129232) old and
new code are also bit-for-bit equal. Test job 17129230: 389 passed, 1 skipped.

**Speed, one process alone** (job 17129212, cProfile, seconds):

| unit | setup | sampling | post | total |
|---|---|---|---|---|
| bright, old | 43.1 | 45.6 | 2.7 | 91.6 |
| bright, new | 15.3 | 22.6 (2.0x) | 2.5 | 40.7 |
| faint, old | 190.2 | 44.8 | 31.3 | 270 |
| faint, new | 34.2 (5.6x) | 34.7 (1.3x) | 31.3 | 104 |

Faint gains less in sampling because about 40% of its rows change each step,
so `fill_b` still runs the trees on many stored states (about 20 of 35 s).

**Speed under production crowding** (job 17129219, kng-cl-nv01, one A40,
2 cores per process, bright unit, 384 steps; median seconds per step):

| layout | old code | new code |
|---|---|---|
| 1 process | 0.0643 | 0.0343 |
| 19 processes | 0.0983 | 0.0622 |
| 19 processes + MPS | 0.0932 | 0.0427 |

MPS is NVIDIA's multi-process service, which lets the processes share the GPU
at the same time instead of taking turns. The old code barely gained from it
(5%) because the trees on the CPU were the bottleneck; once they are gone,
GPU sharing is, and MPS gives 1.46x. Together: about 2.3x per bright step at
production crowding. MPS does not change any number: all 19 units of the
new-code MPS run match the old-code no-MPS run bit for bit.

To use MPS in a job, start it before the workers and stop it after:
```
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
... workers ...
echo quit | nvidia-cuda-mps-control
```

**Production check** (job 17129330, comparison 17129408). 38 units of the
500k mock it0 (19 bright, 19 faint, evenly spread among those its A40 worker
w3 ran on kng-cl-nv02) were rerun with commit c903adf and MPS. The node, the
19 two-core slots, the seeds and the chain lengths (bright 8192, faint 4096
steps) were the same as in production. All 38 match the production outputs
bit for bit. Per unit, mean ± error of the mean (old = production, code 8cd45c5,
no MPS):

| | setup s | sampling s/step | post s | total s | gain (total) |
|---|---|---|---|---|---|
| bright, old | 41.1 ± 1.3 | 0.1152 ± 0.0019 | 30.9 ± 0.7 | 1016 ± 17 | |
| bright, new | 24.5 ± 0.01 | 0.0431 ± 0.0002 | 20.6 ± 0.1 | 398 ± 2 | 2.55 ± 0.04 |
| faint, old | 387 ± 2 | 0.179 ± 0.002 | 170 ± 4 | 1292 ± 12 | |
| faint, new | 53.5 ± 2.2 | 0.097 ± 0.001 | 118 ± 0.5 | 570 ± 7 | 2.27 ± 0.03 |

Weighted by the 500k mix (2660 bright, 1402 faint units), a full iteration
needs 2.4x less slot time. The MPS launcher is
`optimize_20261011/prod_check/worker.sh`; use it as the template for the next
study. In this check the bright units ran first and the faint units after,
whereas production mixed them; the slot count was the same.

**Not used:** the xgboost CUDA predictor is about 9x faster than CPU trees but
not bit-for-bit equal to them (job 17129157). The earlier "CUDA failure"
(WORKLOG, 9.1e-5 / 4.9e-4) was exactly this comparison of a CPU-b reference
with GPU b, not a bug.

**Where time goes now** (bright, one process): log-kernel about 6.7 s,
the target prior's betaprime cdf about 4 s (real work), faint `fill_b` as
above, and in faint post-processing the arviz diagnostics about 15 s.

Receipts: `$DATA_DIR/sbsi/catalogue_prior/optimize_20261011/` (`profile.sh`,
`profile_unit.py`, `compare.py`, `compare_all.sh`, `bench.sh`,
`bench_all.sh`, `val_shear.sh`, frozen code copies `code_b1`, `code_lazy1`,
`code_lazy2`, `runs/`, `logs/`); old-code profile job 17129137.

## Speed: b straight from the pair table, after warm-up only (2026-10-11)

Two further exact changes to `fill_b` (zero sampling shear, persistent
environments):

- It builds only the neighbour pair table (`galsbi_summary.response_pairs`),
  not the eight model features again; b is row-by-row, so the value is the
  same bit for bit.
- It runs from the last warm-up step on. Warm-up b is not used by anything:
  the kernel at zero shear does not depend on b, and every reader of the
  stored summaries starts at the warm-up (`run_galsbi_bright.py`, the result
  moments and diagnostics, the convergence reports). The last warm-up step
  is filled so step-to-step differences after warm-up stay defined. In
  `chain.npz`, warm-up b is now NaN; protocol key `disk_coefficient` says so.

**Check** (19 processes, one A40 on kng-cl-nv02, MPS, 13 bright units of
1024 steps and 6 faint of 512, warm-up a quarter as in production; new job
17130336 vs c903adf job 17130300, comparison 17130356): all 19 units are
bit-for-bit equal in every stored array and diagnostic, apart from the
warm-up b now left empty (about 21,300 values per bright unit). Sampling
time per unit, mean ± error of the mean of the per-unit ratio:

| | old s | new s | speed-up |
|---|---|---|---|
| bright | 53.5 | 46.7 | 1.16 ± 0.06 |
| faint | 48.3 | 40.0 | 1.21 ± 0.04 |

`fill_b` went from 15.8 to 10.8 s per bright unit and from 27.6 to 19.2 s
per faint unit. In production, sampling is 85% (bright) and 69% (faint) of a
unit, so expect about 1.13x per iteration on top of the 2.4x above.

**Looked at, not done:**
- Reusing tree outputs for neighbour pairs seen before (probe job 17130318):
  only 21% (bright) and 16% (faint) of pairs repeat, so at most a few
  percent after the cache cost.
- Faster file compression (job 17130343, one 1046 MB faint chain): deflate
  level 1 writes in 34.9 s instead of 41.0 s; storing uncompressed takes
  1.9 s but the file grows from 805 to 1046 MB. A disk-space choice for the
  owner: about 7% of a faint unit for 30% more disk.
- Start-up: about 21 s (bright) and 10 s (faint) of a unit is importing
  Python libraries from the shared disk (astropy and torch scan every loaded
  module), plus about 10 s building the prior. Only a long-lived process per
  slot, running several units, would avoid it (about 5% bright, 3% faint),
  at the risk of state carried from one unit to the next.

Receipts: `optimize_20261011/` `code_opt3`, `runs/cp_opt3`, `logs/cp_opt3`,
`crowd_profile.sh`, `crowd_summary.py`, `zip_level.py`, `code_probe`.

## Disk: faint chains are not saved (2026-10-11)

The per-step history `chain.npz` of a faint unit is about 805 MB, 92% of a
study's disk (500k mock: faint 1.13 TB, bright 0.10 TB). No analysis reads
it: faint profiles are written during sampling to `local_profiles.npz`, the
diagnostics to `result.json`, and the only chain readers (`comoving.py`,
`bcheck.py`) take bright units. Bright chains stay, because the co-moving
profiles are computed from them afterwards.

- `--no-chain` skips writing `chain.npz`; protocol key `chain_saved` records
  it. The launcher template `optimize_20261011/prod_check/worker.sh` passes
  it for faint units.
- Check (job 17131270, one faint unit, 512 steps, same GPU): every other
  output is bit-for-bit equal to the run that writes the chain; history output
  3.7 s → 0.15 s (about 38 s, 7% of a production faint unit).
- A deleted faint chain can be regenerated bit for bit from the unit's seeds
  in `protocol.json` and the study's source snapshot, on the same GPU model.

Faint chains of finished studies (3,172 files, 4.12 TB: blind 100k 2.49,
500k mock 1.13, twin 100k 0.23, blind 10k 0.27): inventory in
`$DATA_DIR/sbsi/catalogue_prior/faint_chain_cleanup_20261011/inventory.tsv`.

## 238-galaxy bias check: tempered vs untempered

Question (diagnosis item 3): do stuck chains shift the shear score?

**Setup.** All 238 bright galaxies, 4 chains, 32,768 steps (8,192 warmup).
Two arms repeat the 40-galaxy pairing:
- 8 rungs plus narrow steps;
- untempered with no narrow steps.

Each arm has fresh seeds. The shared-GPU launcher ran 3 GPUs for tempered (34–35
min) and 1 GPU for untempered (21 min), each with 6 processes × 8 cores. The
score is the same per-draw score as in [the diagnosis](#what-to-improve-next-diagnosis-of-the-8-rung-and-untempered-runs).
Each galaxy's error is sd/√(effective draws).

| | tempered | untempered |
|---|---|---|
| average score per galaxy | -0.59 ± 0.68 | +1.21 ± 1.11 |
| Monte Carlo error of that average | 0.48 | 0.90 |
| galaxies with log-posterior R-hat ≥ 1.1 | 19 | 112 |
| galaxies with score R-hat ≥ 1.01 | 109 | 185 |

Tempered minus untempered, galaxy by galaxy:
- Plain average: -1.80 ± 1.11 (1.6σ).
- Error-weighted average: -0.24 ± 0.11. With the error bars widened by the
  measured spread factor 1.13 ± 0.05, that is -0.24 ± 0.13 (1.9σ).
- Log-mean-exp score: -2.11 ± 1.20 and -0.25 ± 0.11.

**Reading.**
- *No sign that stuck chains move the score.* Galaxies whose untempered chains
  were stuck show the same weighted shift as healthy ones: -0.19 ± 0.22
  (112 galaxies) against -0.26 ± 0.13 (126).
- *The plain average is set by about 10 very noisy galaxies.* Their
  differences are 44–99 score units, with error bars of 26–91. Without those
  ten the plain average is -0.31 ± 0.57.
- *Tempering halves the Monte Carlo error of the plain average* (0.48 against
  0.90), because it unsticks exactly those noisy galaxies. It costs about 5x
  the GPU time, on mixed GPU types, so per GPU second the two are roughly even.
- *The error bars are about 13% too small* (spread 1.13 ± 0.05), and 4 of 238
  galaxies are beyond 3σ.

**Limits.**
- One replica per arm.
- The arms ran on different GPU types: A40 for untempered; P5000, 2080 Ti and
  V100 for tempered.
- The weighted shift (1.9σ) is not explained. It may be chance, or the
  narrow steps, which only the tempered arm used.
- The plain-average bound is weak: a 2σ range of about ±2.2 score units.

**Next.** Spend chains where the noise is. A second pass with more chains on
the 20 or so noisiest galaxies would shrink the plain-average error most per
GPU second.

Figure: `plots/galsbi_bias238_tempered_vs_plain.png`. Receipts, under
`galsbi_bias238_20261002/`:
- `submit_bias238.sh`: jobs 16800157–60, all 0:0;
- `compare_bias238.py`: job 16800163, output `bias238_comparison.json`;
- `split_trapped.py`: output `bias238_split.json`;
- `plot_bias238.py`.

## Receipts

Root `$DATA_DIR/sbsi/catalogue_prior/galsbi_tempering_20261002/`:
- `source_snapshot/` (SHA256SUMS checked by `jobs/job_galsbi_tempered.sh`).
- `submit_panel.sh` started jobs 16795122–16795125, each completing in 53–57 min.
- `submit_compare.sh` started comparison job 16795186, which wrote
  `panel16k_comparison.json`.
- `plot_temper.py` makes `panel16k_comparison.png`.
- Identity check: `identity2/identity.json`.
- Panel selection (seed 20261002):
  `galsbi_sampling_deepdive_20261002/tempering_panel.json`.

All jobs exited 0:0.
