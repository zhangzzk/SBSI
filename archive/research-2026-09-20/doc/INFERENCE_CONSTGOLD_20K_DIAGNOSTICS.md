# ConstGold 20k inference diagnostics

> **Cancelled at the owner’s request on 2026-09-14.** All remaining jobs and scheduled checks for this run were cancelled before its new GPU inference started. Existing preparation, baseline results, and logs are retained. No automatic restart is scheduled.

## Protocol

Paired diagnostic on rows 0–19,999 of the original randomized 500k ConstGold sample, cases40–139. The original 12,760,990-atom prior and pinned V3.5-like models are reused. Strict truth cuts are `18<r<25.8`, `0.5<Re<1.5` arcsec; measured `FLUX_RADIUS>=0.75` arcsec and `MAG_AUTO<25.8`; coherent usability/detection and measured-selection normalization remain enabled. No measured ellipticity or neighbour cut.

This isolates the quoted original draw ladder. The separate 500k rerun uses the newly agreed intrinsic 0.37 / measured 0.5 arcsec floors. The cached forward response screen used intrinsic 0.37 and measured 0.75 arcsec, cases120–139, and response-defined `m=R_sim/R_model-1`. Its central −0.372% is not a matched target for the plus-leg inference statistic below. See [conventions](CONVENTIONS.md#7-multiplicative-bias).

Sampling: nested complement M=2048,4096,8192,16384,32768,65536 at the same starting center, K=1024. Independent proposal-seed repeat 8702 and K=4096 shortlist control each run through M=32768. Iterations: three full passes at M=8192 and M=32768, starting from the corresponding first-pass estimate. Every new center gets a freshly computed full-prior nine-point selection-normalization stencil, 64 QMC samples/atom, seed8101. All numerator stencils use h=0.001 and common per-object draws; initial center is the original 500k raw mean, not the injected shear. Proposal seed8701 is held across recentered passes.

Reported errors retain the original per-object sandwich method conditional on each expansion point. Paired shifts use the same objects and their covariance; two proposal seeds only diagnose sensitivity and do not precisely estimate total MC error. A 20k absolute bias is noisy; assess paired changes and stabilization, never closeness to the injected value. No correction, trimming, or seed/model selection is applied.

Resource limit: at most two GPUs across this study and the ongoing 500k run. Each pipeline runs at most one GPU job at a time. This study uses a serial Slurm dependency chain.

## Results recorded so far

| Experiment | M | g1 | g2 | effective bias (%) | original SE (percentage points) |
|---|---:|---:|---:|---:|---:|
| saved_baseline | 512 | 0.019762580 | 0.000204489 | -1.1871 | 0.9010 |
| saved_baseline | 1024 | 0.019806859 | 0.000202520 | -0.9657 | 0.9175 |
| saved_baseline | 2048 | 0.019847454 | 0.000212519 | -0.7627 | 0.9241 |
| saved_baseline | 4096 | 0.019824321 | 0.000179565 | -0.8784 | 0.8944 |
| saved_baseline | 8192 | 0.019895625 | 0.000191346 | -0.5219 | 0.9066 |

## Paired changes

| Comparison | Δ effective bias (percentage points) | paired conditional SE |
|---|---:|---:|

## Artifacts and status

Run directory: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/diagnostics_constgold_v35_n20k_20260914_v1`. `protocol.json` freezes the design; `submission.json` records job dependencies; `sample_identity.parquet` records source IDs; every inference writes per-object scores, full information, nested moments and weight diagnostics. `results.json` updates after each stage; `stages/` and `failures/` distinguish completion from failure. Scheduled status records do not wake the agent or send a notification.

Interpretation remains pending until the relevant experiments complete. Three passes alone do not establish convergence; any continuing draw-count drift or center movement must be reported.
