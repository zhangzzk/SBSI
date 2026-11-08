# ConstGold 500k rerun: Re > 0.37 arcsec, measured radius >= 0.5 arcsec

> **Cancelled at the owner’s request on 2026-09-14.** All remaining jobs and scheduled checks for this run were cancelled before its new GPU inference started. Existing preparation, baseline results, and logs are retained. No automatic restart is scheduled.

## Authorized setup

The owner selected three full inference passes. Each pass evaluates all 500,000 observations and recomputes the full-prior selection-normalization stencil at its own centre. Pass 1 starts at the raw measured-shape mean; each subsequent pass starts at the previous estimate. The reported uncertainty remains the original per-object sandwich estimator, recomputed from that pass's moments.

- Truth domain: strict primary magnitude `18<r<25.8`, intrinsic semi-major `0.37<Re<1.5` arcsec; no neighbour cut.
- Measured selection: `MAG_AUTO<25.8`, `FLUX_RADIUS>=0.5 arcsec` (2.5 pixels), no measured-|e| cut; independent plus-leg usability.
- Input: actual ConstGold cases40–139, injected `(g1,g2)=(0.02,0)`; 500k uniform sampling without replacement, seed20260914.
- Prior source: full existing null-shear FS2 cases20000–20199. Rebuild active support using all scene atoms; preserve deterministic neighbour-complete zero features and recompute trial9 R_blend, classifier probabilities, and flow-QMC coordinates.
- Models: the same V3.5-like lambda10 epoch154 physical four-output flow, trial9 response, equal-probability three-seed coherent-U classifier ensemble.
- Numerics: v1.2 tilted-stratified K1024, nested complement draws512–8192, h0.001; same global object IDs and random seeds across passes. No bright/faint carry-forward approximation.

The two radius cuts define this experiment; they are not claimed to implement T/T_PSF>0.5. See [population conventions](CONVENTIONS.md#3-population-and-selection-order). Three updates are a convergence diagnostic; examine the final step before claiming a converged solution. Numerical draw-budget error and model/population limitations from the [original review](INFERENCE_CONSTGOLD_500K_REVIEW.md#scope-and-conclusion) remain relevant.

## Status and timing

Updated UTC: 2026-09-14T10:44:30.751855+00:00

The owner subsequently capped combined usage at two GPUs. This 500k pipeline is throttled to one concurrent GPU while a separate 20k diagnostic chain uses at most one. The original 7–9 hour estimate assumed four GPUs and no longer applies. At this allocation, allow roughly a day for all three passes, plus queue time; update this estimate from actual first-pass timings. All six pending normalization/inference arrays have ArrayTaskThrottle=1; single-GPU cache/smoke stages are dependency-serialized.

Rebuilt prior: 19,427,730 active atoms.

Prepared observations: 500,000, sampled from 8,535,713 eligible rows.

## Per-pass results

| Pass | Centre (g1, g2) | Estimate (g1, g2) | Original sandwich SE (g1, g2) | Update (g1, g2) |
|---|---|---|---|---|
| 1 | Pending | Pending | Pending | Pending |
| 2 | Pending | Pending | Pending | Pending |
| 3 | Pending | Pending | Pending | Pending |

## Scheduler and artifacts

- Run root: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v35_re037_size050_n500k_20260914_v1`.
- `protocol.json`, `source_hashes.json`, isolated `venv/`, frozen `source/`, and submission manifests bind the requested run.
- `iteration_results.json` records only completed passes; `completion.json` appears after all three passes.
- `scheduled_status.json` and `report_status.json` retain bounded scheduler checks.

```text
JobID|JobName|State|ExitCode|Elapsed
16493602|cg037_test|COMPLETED|0:0|00:00:29
16493652|cg037_input|COMPLETED|0:0|00:01:32
16493655|cg037_merge_prior|COMPLETED|0:0|00:00:37
16493656|cg037_cache|PENDING|0:0|00:00:00
16493658|cg037_merge_norm_i1|PENDING|0:0|00:00:00
16493659|cg037_smoke_i1|PENDING|0:0|00:00:00
16493661|cg037_combine_i1|PENDING|0:0|00:00:00
16493653_0|cg037_prior_smoke|COMPLETED|0:0|00:01:40
16493654_1|cg037_prior|COMPLETED|0:0|00:01:45
16493654_2|cg037_prior|COMPLETED|0:0|00:01:47
16493654_3|cg037_prior|COMPLETED|0:0|00:01:44
16493654_4|cg037_prior|COMPLETED|0:0|00:01:45
16493654_5|cg037_prior|COMPLETED|0:0|00:01:39
16493654_6|cg037_prior|COMPLETED|0:0|00:01:48
16493654_7|cg037_prior|COMPLETED|0:0|00:01:47
16493654_8|cg037_prior|COMPLETED|0:0|00:01:45
16493654_9|cg037_prior|COMPLETED|0:0|00:01:41
16493654_10|cg037_prior|COMPLETED|0:0|00:01:51
16493654_11|cg037_prior|COMPLETED|0:0|00:01:48
16493654_12|cg037_prior|COMPLETED|0:0|00:01:49
16493654_13|cg037_prior|COMPLETED|0:0|00:01:43
16493654_14|cg037_prior|COMPLETED|0:0|00:01:44
16493654_15|cg037_prior|COMPLETED|0:0|00:01:41
16493654_16|cg037_prior|COMPLETED|0:0|00:01:42
16493654_17|cg037_prior|COMPLETED|0:0|00:01:43
16493654_18|cg037_prior|COMPLETED|0:0|00:01:43
16493654_19|cg037_prior|COMPLETED|0:0|00:01:40
16493657_[0-3%4]|cg037_norm_i1|PENDING|0:0|00:00:00
16493660_[0-19%4]|cg037_infer_i1|PENDING|0:0|00:00:00
16493663|cg037_merge_norm_i2|PENDING|0:0|00:00:00
16493664|cg037_smoke_i2|PENDING|0:0|00:00:00
16493666|cg037_combine_i2|PENDING|0:0|00:00:00
16493668|cg037_merge_norm_i3|PENDING|0:0|00:00:00
16493669|cg037_smoke_i3|PENDING|0:0|00:00:00
16493671|cg037_combine_i3|PENDING|0:0|00:00:00
16493672|cg037_status_2h|PENDING|0:0|00:00:00
16493662_[0-3%4]|cg037_norm_i2|PENDING|0:0|00:00:00
16493665_[0-19%4]|cg037_infer_i2|PENDING|0:0|00:00:00
16493667_[0-3%4]|cg037_norm_i3|PENDING|0:0|00:00:00
16493670_[0-19%4]|cg037_infer_i3|PENDING|0:0|00:00:00
16493673|cg037_status_final|PENDING|0:0|00:00:00
```
