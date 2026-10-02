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
rows only, and the output is identical. A memo of PyCosmo angular distances is
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

**Sparser ladder running.** 8 rungs (`make_ladder.py 8`: 1, .7883, .6167,
.4449, .2849, .1473, .05014, .005), 32768 steps, 8192 warmup, fresh seeds,
same panel and split (`submit_ladder8.sh`, jobs 16795399–16795402). The
equal-rejection prediction was 0.45 swap acceptance; early warmup gives
0.56–0.66. The step time is 0.11 s, against 0.19 s for 16 rungs.

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
