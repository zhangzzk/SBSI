# Single-target-cut flow on constant-shear measurements

## Frozen question and checkpoint — November 4, 2026

Owner requested running the discussed flow on constant-shear measurements and reporting m.
Candidate is lambda=0.01, optimizer seed index1, step128 from
`galsbi_target_guard_repeat_20261103`, the checkpoint whose training soft-cut
response error was +0.00106% and validation error +0.87481% in the full check.
This checkpoint is fixed before any constant-shear output. Its hash is pinned
in preset.json and manifest.json. Original GalSBI-dev results already exist;
use them as a paired reference, not as new evidence of flow-only calibration.
No training, outcome-based model selection or production promotion.

## Data, estimator and unchanged components

Use the same fixed 40-case cohort as the earlier band-flow comparison:
1000–1009,1040–1049,1080–1089,1120–1129. These cases are separate from the
training family but previously used in development, not untouched acceptance.
Existing GalSBI constant-shear measurements at g1=+/-0.02, g2=0.
Keep the GalSBI-dev C8 and disk-response emulator fixed. Reuse the verified
parent and C8 caches; generate new candidate flow draws and recompute the
output-conditioned emulator response on those draws.
Primary cut: independent own-leg FLUX_RADIUS>3 pixels and MAG_AUTO<25.
Report all 15 existing cuts as diagnostics, C8 weighting primary and true usable
flags secondary. R=(mean e1_plus-mean e1_minus)/0.04, each leg normalized by its
own selected mass; m=100*(R_sim/R_model-1). 64 antithetic draws, common latent
numbers between shear legs and with the baseline; seed7301+10000019*case+
1000003*block, batch1024. No truth-property cut. Preserve measured row matching
and report all unmatched counts via existing evaluator receipts.

## Reporting and verification

Frozen evaluator source is copied unchanged. Its model loader checks flow,
classifier and emulator SHA256. Runner checks source/preset hashes before each
stage. Require all 40 cases, finite outputs, matching cuts and exactly identical
simulation sums between candidate and baseline. Recompute leg means/normalizers
inside each of 10,000 whole-case bootstrap resamples, seed20260925. Report
candidate m, baseline m and paired candidate-minus-baseline with uncertainties.
Cross-check baseline m and its error against the previous paired comparison.
No tuning from these outputs. Fixed-checkpoint case errors omit full retraining,
model-selection and independent finite-draw uncertainty. This is a combined
forward-response test, not a Bayesian inference run or an isolation of the flow.

## Resources and follow-up

At most four GPUs total; two cip workers and two inter workers, one GPU each.
No other SBSI GPU jobs were active at preparation. Follow startup and first
completed case; summary depends on all workers. Schedule a two-hour CPU status
receipt. Frozen source, manifests, rule, logs and failed outputs are preserved.
