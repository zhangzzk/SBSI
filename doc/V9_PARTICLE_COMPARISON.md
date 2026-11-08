# N18 higher-effort fitted-proposal comparison

## Question and accounting

The preserved n18 declaration increases the frozen fitted-proposal arm from
v7 N=256/T=64/M=32 to v9 N=1024/T=64/M=64, with eight fresh seeds
2026102800–2026102807. It retains the v7 source, proposal and geometry, and
compares against eight independently seeded v6v100 broad-proposal populations
at N=1024/T=64/M=64 and the original eight v7 populations. Its motivation was
the observation-0 corrected-g1 disagreement of about 4.54 combined MCSEs.
No phase coefficients are refit: only the four v1 pilots fitted the frozen
constant control. No held-out outcomes enter this comparison.

`scripts/summarize_v9_particle_comparison.py` reads completed audited raw
reports and completed phase aggregates. It verifies referenced hashes,
independent seeds, all eight populations, stride 1, all original particles,
matching target identity and the frozen coefficient artifact. It checks raw
score, MCSE, weights and normalizer replay between raw and phase reports.
It preserves existing whole-population Z weighting and the original full-N
F64 reweighting denominators. It adds no particle thinning or aggregation fit.

Each signed contrast is A−B with independent-group MCSE
sqrt(MCSE_A²+MCSE_B²). Normalizer contrasts use a common exponential scale
and the reported relative MCSE. Both sampled-F32 and explicitly reweighted-F64
measures are reported separately. Every observation/component and sign is
retained. Variance times total recorded inference cost is descriptive and
excludes bias, pilot fitting, phase diagnostics, scheduler delay and startup
outside the driver's timer. Different N/M arms are kept explicit.

## Shared correctness limits

All these old trajectories share a native numerical proposal-support defect:
the q/p implementation omitted the native uniform lower clamp, permitting
marks with zero continuous numerical environment density despite finite
analytic target intensity. Whether an old trajectory
actually encountered the defect is unknown. Root's separate fix dca103e is
not present in these frozen trajectories. These comparisons cannot certify
exactness or establish that every discrepancy arises from equilibration.

The preserved result JSON says "zero real-prior-density marks" in one
limitation string. That wording refers to the numerical environment measure,
not the analytic target prior, whose intensity is finite in these examples.
This documentation and future report text correct the wording; the original
result and its pinned receipt remain unchanged.

Historical phase aggregates incorrectly label F32 control expectation as
known zero. The new summary retains that original field as historical metadata
and sets the applicable F32 expectation to null/unknown. Zero expectation in
the F64 diagnostic is qualified as the identity for the intended continuous
posterior, numerically evaluated; it is not an exact identity of a rounded
floating-point program. F64 precision reweighting explicitly changes the
numerical measure. Existing raw score/Z gates remain unchanged; no new
corrected-score acceptance threshold is introduced. Eight populations cannot
rule out shared rare-mode bias.

## Completed result: narrower discrepancy, mixed precision and cost

CPU job **17422294** completed in 11 seconds, exit 0. All six focused tests
passed (0.19 seconds); source import assertion and 145 referenced input hashes
passed. The script used all 24 populations and the already completed phase
diagnostics, without new model evaluation. The v9 phase job 17421115 completed
in 73 seconds, exit 0. Both original v9 raw comparison gates and all original
raw precision gates pass; they are preserved as historical results, not a
certification of the defective old transition implementation.

The narrow n18 hypothesis receives directional support: observation-0
corrected g1 moves from v7 **1.069976 ± 0.111169** to v9
**0.568624 ± 0.154248**, toward v6 **0.404688 ± 0.095263**.
The v9−v6 contrast is **+0.163937 ± 0.181294 (0.904 MCSE)**, whereas
v7−v6 was **+0.665288 ± 0.146402 (4.544 MCSE)**. V9−v7 remains
**−0.501351 ± 0.190134 (−2.637 MCSE)**. This supports checking effort
and finite-population behavior; the shared support defect and only eight
replicates prevent an exclusive equilibration explanation or sampler acceptance.

### Signed sampled-F32 scores

Entries are mean ± estimated independent-population MCSE. “Corrected” uses
only the frozen v1 constant coefficient. These are individual fixed-observation
posterior score expectations, not population shear estimates or quantities
expected individually to vanish. F32 control expectation remains unknown.

| Group | Observation | Raw g1 | Raw g2 | Corrected g1 | Corrected g2 |
|---|---:|---:|---:|---:|---:|
| v6v100 | 58165 | +7.254081 ± 1.484557 | +11.703145 ± 2.802678 | +7.283441 ± 1.121229 | +11.672073 ± 1.795168 |
| v6v100 | 0 | +0.483453 ± 0.566552 | -5.330297 ± 0.412437 | +0.404688 ± 0.095263 | -5.334937 ± 0.416842 |
| v6v100 | 1 | -2.968628 ± 0.108209 | -3.651751 ± 0.142721 | -2.921802 ± 0.104532 | -3.680286 ± 0.128204 |
| v6v100 | 8 | -1.980711 ± 0.035424 | +1.226141 ± 0.026269 | -1.963328 ± 0.036406 | +1.250228 ± 0.023142 |
| v7 | 58165 | +1.736343 ± 4.611609 | +4.954660 ± 4.946114 | +2.303558 ± 2.904459 | +4.354365 ± 4.762481 |
| v7 | 0 | -0.996627 ± 1.167547 | -8.172525 ± 1.218474 | +1.069976 ± 0.111169 | -8.050772 ± 1.177207 |
| v7 | 1 | -2.447740 ± 0.241836 | -4.326681 ± 0.547281 | -2.853435 ± 0.190462 | -4.079456 ± 0.352599 |
| v7 | 8 | -1.854643 ± 0.068893 | +1.232366 ± 0.036670 | -1.836467 ± 0.066715 | +1.257551 ± 0.037775 |
| v9 | 58165 | +2.538342 ± 3.988737 | +10.061781 ± 2.217517 | +4.921365 ± 1.022709 | +7.539780 ± 1.662827 |
| v9 | 0 | -0.055056 ± 0.899914 | -5.518054 ± 0.462771 | +0.568624 ± 0.154248 | -5.481310 ± 0.505272 |
| v9 | 1 | -2.705032 ± 0.205549 | -3.757345 ± 0.128894 | -2.824542 ± 0.087545 | -3.684517 ± 0.152569 |
| v9 | 8 | -1.877568 ± 0.016420 | +1.187793 ± 0.028197 | -1.871997 ± 0.015089 | +1.195512 ± 0.025248 |

### All signed contrasts, including remaining differences

The following entries are signed difference / combined MCSE, not p-values.
No new threshold, multiplicity claim or acceptance decision is imposed.

| Contrast | Observation | Raw g1 | Raw g2 | Corrected g1 | Corrected g2 |
|---|---:|---:|---:|---:|---:|
| v9 − v6v100 | 58165 | -1.108 | -0.459 | -1.556 | -1.689 |
| v9 − v6v100 | 0 | -0.506 | -0.303 | +0.904 | -0.223 |
| v9 − v6v100 | 1 | +1.135 | -0.549 | +0.713 | -0.021 |
| v9 − v6v100 | 8 | +2.642 | -0.995 | +2.318 | -1.598 |
| v9 − v7 | 58165 | +0.132 | +0.942 | +0.850 | +0.631 |
| v9 − v7 | 0 | +0.639 | +2.037 | -2.637 | +2.006 |
| v9 − v7 | 1 | -0.811 | +1.013 | +0.138 | +1.028 |
| v9 − v7 | 8 | -0.324 | -0.964 | -0.519 | -1.365 |
| v7 − v6v100 | 58165 | -1.139 | -1.187 | -1.600 | -1.438 |
| v7 − v6v100 | 0 | -1.140 | -2.209 | +4.544 | -2.175 |
| v7 − v6v100 | 1 | +1.966 | -1.193 | +0.315 | -1.064 |
| v7 − v6v100 | 8 | +1.627 | +0.138 | +1.669 | +0.165 |

For faint observation 8, v9−v6 raw g1 is +0.103143 ± 0.039045
(2.642 MCSE), and corrected g1 is +0.091330 ± 0.039409 (2.318).
For observation 0, v9−v7 raw/corrected g2 differences remain 2.037/2.006
MCSE. Every other sign and component is retained above.

### Phase residual and F64 qualification

| Group | Observation | T mean ± MCSE, sampled F32 | T mean ± MCSE, F64 reweighted | F64 T/MCSE |
|---|---:|---:|---:|---:|
| v6v100 | 58165 | +0.013845 ± 0.710494 | +0.013883 ± 0.710499 | +0.020 |
| v6v100 | 0 | +0.025950 ± 0.182719 | +0.025988 ± 0.182718 | +0.142 |
| v6v100 | 1 | -0.026121 ± 0.053520 | -0.026121 ± 0.053520 | -0.488 |
| v6v100 | 8 | +0.017356 ± 0.009667 | +0.017356 ± 0.009667 | +1.795 |
| v7 | 58165 | +0.267467 ± 1.359881 | +0.267743 ± 1.359908 | +0.197 |
| v7 | 0 | -0.680860 ± 0.413985 | -0.680857 ± 0.414009 | -1.645 |
| v7 | 1 | +0.226312 ± 0.206439 | +0.226311 ± 0.206437 | +1.096 |
| v7 | 8 | +0.018147 ± 0.014424 | +0.018147 ± 0.014424 | +1.258 |
| v9 | 58165 | +1.123703 ± 1.482620 | +1.124063 ± 1.482624 | +0.758 |
| v9 | 0 | -0.205477 ± 0.258720 | -0.205451 ± 0.258721 | -0.794 |
| v9 | 1 | +0.066667 ± 0.068649 | +0.066671 ± 0.068649 | +0.971 |
| v9 | 8 | +0.005562 ± 0.003843 | +0.005562 ± 0.003843 | +1.447 |

The largest absolute F64 T residual is 1.795 MCSE (v6, observation 8);
v9 residuals have absolute values below 1.448. Small observed residuals do not
establish global coverage or exactness. The largest absolute pointwise
logK_F64−logK_F32 difference across these groups is 0.000917081. The largest
fine-step phase derivative scaled error is 8.097e-6, below the unchanged
1e-4 criterion. F64 precision reweighting shifts any reported moment by at
most 0.001122 of its corresponding sampled-F32 MCSE (v9, observation 0 raw
g2). Observation-0 corrected g1 F64 means are v6=0.404665, v7=1.069964,
v9=0.568582; v9−v6 is 0.904 MCSE and v9−v7 is −2.637 MCSE. Thus the
checked F32/F64 measure difference is too small to account for these observed
contrasts. The full JSON retains all F64 means, covariances, MCSEs, weights,
normalizers, signed contrasts and precision shifts; this empirical smallness
does not make the F32 zero-expectation identity exact.

### Normalizer precision and cost

| Observation | v6 Z relative MCSE | v7 Z relative MCSE | v9 Z relative MCSE | v9/v6 mean Z | v9/v7 mean Z |
|---:|---:|---:|---:|---:|---:|
| 58165 | 8.074% | 2.782% | 1.295% | 1.030071 | 0.998139 |
| 0 | 3.857% | 3.311% | 2.110% | 1.021189 | 0.970240 |
| 1 | 2.001% | 2.775% | 0.966% | 1.012146 | 1.020415 |
| 8 | 1.552% | 1.226% | 0.745% | 0.985268 | 0.999275 |

Every v9 mean-normalizer contrast with v6 or v7 is below 0.86 combined
MCSE in magnitude. V9 run-normalizer ESS ranges 7.975–7.997 out of eight.
Normalizer agreement cannot establish that the groups share no missed mode.

| Group | N/T/M | Total inference seconds, eight runs | Median seconds/run |
|---|---|---:|---:|
| v6v100 | 1024/64/64 | 3820.862 | 471.798 |
| v7 | 256/64/32 | 811.538 | 101.780 |
| v9 | 1024/64/64 | 4102.947 | 518.246 |

Each run covers all four observations. Recorded v9 inference costs
**1.074× v6 and 5.056× v7**. Its normalizer precision improves, but score
efficiency is mixed: v9 observation-0 corrected g1 MCSE (0.154248) exceeds
both v6 (0.095263) and v7 (0.111169). The corresponding estimated
variance-times-cost ratios are 2.815 versus v6 and 9.733 versus v7; lower is
better for this descriptive metric. Bright corrected-score ratios versus
v7 are 0.627/0.616, and faint corrected g1 is 0.259, while faint corrected
g2 is 2.259. These are noisy eight-run variance estimates, ignore bias and
exclude pilot/diagnostic costs. No uniform efficient-sampler conclusion follows.

## Exact provenance and invocation

- Source branch `codex/v9-comparison-20261022`, runnable commit `e7203d2`.
- Frozen source `source_v9_comparison_20261022`, manifest SHA256
  `fa4c8695556103048d14e3a14e9f3160e6cfa673485d45b13ac7b20aafab3de8`.
- Pre-execution receipt `v9_comparison_20261022_experiment.json`; all nine
  top-level receipt hashes match the executed summary, with 145 total
  referenced input hashes recorded in the output. Selected completed outcomes
  had already been read before this reporting receipt; it is not presented as
  a new preregistered scientific test.
- Output `v9_comparison_20261022.json`, SHA256
  `92efbd83fa70bbfa98e62d012e14412816f6aa13b2018cf0ef3c2acf00815df4`.
- Result receipt `v9_comparison_20261022_result_receipt.json`; scheduler log
  `v9_comparison_20261022_17422294.log` (6 passed, exit 0).
- All external paths are under
  `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021`.

Reproduce on a CPU scheduler allocation with a fresh output path:

```bash
python -m pytest -q tests/test_v9_comparison.py
python scripts/summarize_v9_particle_comparison.py \
  --root /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021 \
  --output /path/to/fresh_v9_comparison.json
```
