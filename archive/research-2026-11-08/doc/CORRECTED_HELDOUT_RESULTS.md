# Corrected sampler held-out results — October 22, 2026

## Decision and complete evidence

The frozen N256/M64 fitted-proposal candidate **fails scientific acceptance**.
Fifteen of twenty objects pass every mandatory accuracy gate; five fail. Its
inclusive recipe costs 11,782.2405 process seconds versus 22,896.6929 for the
N1024/M64 broad-proposal reference, a cost ratio of **1.9433**. Lower cost does
not establish matched-accuracy efficiency when the accuracy gate fails.

All five original warmups, forty paired production tasks and the final report
completed. Report job17435351 exits successfully because it produced a valid
scientific rejection; this is distinct from an execution failure. No runs were
replaced, and the final report is preserved. The
[final evidence summary](evidence/corrected_heldout_final_results_20261022.json)
binds original reports and declaration. Separate audit job17448173 confirms
the original rejection with valid evidence: all46 stage receipts and235 step
receipts account for20 pilots,80 production populations,100 phase checks,
15 fits and20 reports. Independent gate and cost arithmetic reproduces all
five failed objects and the cost ratio. The audit implementation is commit
27a5323 on `codex/smc-completed-panel-audit-20261022`; its complete archive
passes1,075 tests with nine skips before auditing the real panel in a fresh
interpreter. No original reports were regenerated or overwritten.

## Direct failures

| Object | Failed requirement | Measured value | Frozen requirement |
| --- | --- | --- | --- |
| 745801 | Reference normalizer precision | Relative MCSE 0.13434 | <0.1 |
| 854064 | First scene-region mass agreement | Difference 0.0052103; 3.664 combined MCSE | ≤3.5 combined MCSE |
| 120039 | Second scene-region mass agreement | Difference −0.0022536; −4.129 combined MCSE | Absolute standardized difference ≤3.5 |
| 12634 | Candidate first raw-score precision | Effective count 352.45 | ≥400 |
| 960670 | First raw-score agreement | 4.407 combined MCSE | ≤3.5 combined MCSE |

Both controlled-score components agree for all twenty objects in both the
F32 sampled and likelihood-reweighted F64 measures. All raw-score and region
absolute bounds pass. These successes do not replace the failed requirements.
The reference also has an unresolved precision failure, so it must not be
treated as exact ground truth. Combined MCSE values are estimates from eight
independent complete populations, not exact Gaussian significance levels.

## What follows from this result

The four-object development success did not transfer to every held-out object.
Remaining issues include normalizing-mass precision, raw-score precision and
small but statistically resolved scene-weight/raw-score disagreements. This
experiment alone does not establish whether each disagreement is finite-run
variation, slow mixing or shared approximation error. In particular, it does
not locate the original catalogue shear residual in a magnitude bin.

Preserve the rejection and all thresholds. Further sampler development uses
development data; a new candidate needs an untouched acceptance panel. Adding
samples to only the failed held-out objects or relaxing their gates would not
validate the original candidate. CPU concurrency and exact prior-computation
reuse remain separate implementation studies that can reduce the cost of
future experiments without changing their scientific result.

## Scope and cost convention

Both arms use the same independently fitted geometry, T64/M64 and declared
target p_A(target) q_E(environment) K32. F64 reweighting changes likelihood
precision only, not the target prior to the numerical mock law q_N. Neither
this rejection nor a future pass certifies absent-mode coverage or population
shear calibration. See the
[acceptance and target-law boundary](CORRECTED_HELDOUT_RUNNER.md#acceptance-audit-and-target-law-boundary).

Costs include initialization, production, phase diagnostics, reports, pilots,
fits and declared verification/preflight allocation. Shared pilot preparation
is charged fully to each recipe, proposal fitting only to the candidate, and
common verification/preflight equally. Actual study work is 32,483.5336 process
seconds, counted once. Queue time and older development/search costs are
excluded. These are summed process costs, not elapsed panel time or GPU busy
time. Cost scaling conventions remain in
[computation cost](SAMPLER_COMPUTE_COST.md#measured-recipe-and-evidence).
