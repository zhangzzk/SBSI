# N16 completed informed-environment comparison

## Question and fixed accounting

The original n16 hypothesis was that explicit informed-neighbour mass could
expose alternative light assignments missed by target-only proposals. Both
v8 arms use N256/T64/M32 and eight independently seeded populations under
source_v8. Baseline seeds are 2026102600–2026102607; informed seeds are
2026102700–2026102707. The informed arm adds a single-point augmentation with
probability 0.5, uniform native type and phase, and a 90% central-position
component at scale 0.1 arcsec. The fitted target proposal, physical geometry,
observation panel and shear are frozen. No coefficients are refit.

The standalone summary reuses the tested v9 reporting arithmetic. It verifies
all referenced hashes, declared populations, all states/stride 1, frozen phase
coefficients and raw score/Z replay. Signed informed−baseline differences use
independent-population MCSE sqrt(MCSE_informed²+MCSE_baseline²). All existing
whole-population Z weighting and original full-N F64 precision-reweighting
accounting remain unchanged. Raw regions come directly from the audited raw
reports; no F64 region reweighting is calculated.

Baseline's `comparison_gate=false` means **no comparator was supplied**, not a
failed comparison. The informed report contains the actual comparison against
baseline. Existing raw score, region and normalizer gates remain unchanged.

## Required limitations

The old baseline correction can admit marks with **zero continuous numerical
environment density despite finite analytic target intensity**. The informed
arm rejects that region. Whether these trajectories encountered this support
defect is unknown. Thus this experiment changes both informed proposal mass
and the support guard, and differences cannot be attributed solely to informed
mass. Matching declared endpoint factors do not certify either old transition
kernel or a literal finite-RNG target. The separately corrected v10 experiment
is outside this summary and is not read.

Historical F32 phase aggregates incorrectly asserted a known-zero control
expectation. This new summary preserves their historical field but sets the
applicable F32 expectation to null/unknown. Explicit F64 reweighting changes the
numerical measure; the zero-mean identity applies to the intended continuous
posterior, numerically evaluated, not an exact rounded program.

Optional observation-0 baseline contrasts against v6/v7/v9 are descriptive
under the separate frozen endpoint receipt. They do not invoke or bypass the
strict report source-identity gate. Eight populations cannot establish rare-mode
coverage or exactness. Cost comparisons use recorded inference runtime and
exclude pilots, phase diagnostics, queueing and startup outside driver timing.

## Preserved reporting failure and numerical repair

CPU 17422693 passed all six initial tests, then stopped before writing a result:
baseline observation 8 raw-report and phase weight arrays differ by
1.3877787807814457e-17 (one float64 ULP), while their log normalizers agree
exactly. All other arrays agree bitwise. Requiring exact equality across
independently computed aggregates was a reporting implementation error.
The v2 summary permits at most eight ULPs per finite weight and records the
maximum observed difference, without changing either weight array or any
scientific gate. A focused regression accepts one ULP but rejects a 1e-8
perturbation, missing entries and NaNs. The failed source, receipt and log are
preserved. Scientific inputs and the n16 hypothesis remain unchanged.

## Completed result: n16 benefit remains unestablished

CPU summary job **17422725** completed in seven seconds, exit 0. All seven
focused tests passed (0.11 seconds); frozen source/input assertions and 104
referenced input hashes passed. It preserves all 16 populations, all saved
particles and existing aggregate weights. Phase job 17421116 completed in
124 seconds, exit 0. Both original precision checks pass; the informed arm
passes its existing raw comparison against baseline. Baseline had no comparator.

The proposed ability to expose substantial alternative light assignments is
**not established by this panel**. Coarse faint-target-region probabilities
remain at most 0.000973; their contrasts are small and uncertain. Informed
initialization costs **1.713× baseline** and has worse normalizer precision
on every observation. Bright corrected-score precision improves substantially,
but raw-score and other corrected-score gains are mixed. The support-guard
confound prevents assigning these outcomes solely to informed mass. No
candidate is accepted and no tuning follows from these results.

### Signed sampled-F32 score means and MCSE

“Corrected” uses the same independently frozen v1 constant coefficient.
These fixed-observation posterior score expectations are not population
shear estimates and need not individually equal zero.

| Arm | Observation | Raw g1 | Raw g2 | Corrected g1 | Corrected g2 |
|---|---:|---:|---:|---:|---:|
| baseline | 58165 | -5.690661 ± 7.204156 | +16.232555 ± 7.258013 | +3.771756 ± 4.214629 | +6.218292 ± 6.670107 |
| baseline | 0 | +2.459810 ± 1.675605 | -5.561755 ± 0.858297 | +0.400543 ± 0.249085 | -5.683076 ± 0.928173 |
| baseline | 1 | -3.304715 ± 0.281004 | -3.217863 ± 0.293023 | -2.702434 ± 0.189829 | -3.584886 ± 0.297369 |
| baseline | 8 | -1.919773 ± 0.101627 | +1.250399 ± 0.060122 | -1.942699 ± 0.094563 | +1.218633 ± 0.051342 |
| informed | 58165 | -2.940122 ± 5.897372 | -1.413693 ± 7.218644 | -1.902159 ± 1.314633 | -2.512190 ± 2.081402 |
| informed | 0 | -0.881828 ± 1.288158 | -5.099081 ± 0.578330 | +0.329790 ± 0.200810 | -5.027699 ± 0.572304 |
| informed | 1 | -3.307061 ± 0.262888 | -4.020347 ± 0.452074 | -3.318944 ± 0.226874 | -4.013106 ± 0.307120 |
| informed | 8 | -1.865767 ± 0.073552 | +1.276494 ± 0.050505 | -1.897248 ± 0.079152 | +1.232873 ± 0.036512 |

### Informed minus baseline, all signs retained

Entries are signed differences divided by independent-group combined MCSE;
no new acceptance threshold or multiplicity claim is imposed.

| Observation | Raw g1 | Raw g2 | Corrected g1 | Corrected g2 |
|---:|---:|---:|---:|---:|
| 58165 | +0.295 | -1.724 | -1.285 | -1.249 |
| 0 | -1.581 | +0.447 | -0.221 | +0.601 |
| 1 | -0.006 | -1.490 | -2.084 | -1.002 |
| 8 | +0.431 | +0.332 | +0.369 | +0.226 |

Observation 1 corrected g1 differs by **−0.616510 ± 0.295816
(−2.084 MCSE)**. Observation 0 corrected g1 differs by
**−0.070754 ± 0.319950 (−0.221 MCSE)**. Bright corrected scores differ
by −5.673915 ± 4.414902 and −8.730482 ± 6.987315; their smaller informed
MCSEs do not establish their means as correct.

### Phase residuals and explicit F64 measure

| Arm | Observation | T sampled-F32 mean ± MCSE | T F64-reweighted mean ± MCSE | F64 T/MCSE |
|---|---:|---:|---:|---:|
| baseline | 58165 | +4.461957 ± 2.299839 | +4.462472 ± 2.299850 | +1.940 |
| baseline | 0 | +0.678443 ± 0.503304 | +0.678445 ± 0.503342 | +1.348 |
| baseline | 1 | -0.335975 ± 0.099026 | -0.335978 ± 0.099026 | -3.393 |
| baseline | 8 | -0.022889 ± 0.024578 | -0.022889 ± 0.024578 | -0.931 |
| informed | 58165 | +0.489446 ± 2.859559 | +0.489897 ± 2.859631 | +0.171 |
| informed | 0 | -0.399178 ± 0.408191 | -0.399208 ± 0.408195 | -0.978 |
| informed | 1 | +0.006629 ± 0.152950 | +0.006632 ± 0.152946 | +0.043 |
| informed | 8 | -0.031432 ± 0.023872 | -0.031432 ± 0.023872 | -1.317 |

**Baseline observation 1 has a −3.393-MCSE F64 phase residual**. This
remains an unresolved stationarity/accuracy diagnostic under the stated
continuous-model qualification; the informed residual on that object is
+0.043 MCSE. Other signed residuals are retained above. There is no empirical
recentering and no assertion that the rounded F32 program has a known-zero
control expectation.

The largest pointwise |logK_F64−logK_F32| is 0.000741432. Fine-step phase
derivative scaled error is at most 8.659e-6, below the unchanged 1e-4
criterion. F64 reweighting changes any moment by at most 0.001022 sampled-F32
MCSE; it therefore does not explain the observed baseline phase residual or
score contrasts. All F64 means, covariance matrices, MCSEs, weights,
normalizers and signed comparisons remain in the JSON. For example,
observation-1 corrected g1 means become −2.702432 baseline and −3.318951
informed, with contrast −2.084 MCSE.

### Region and normalizer evidence

Regions are |target magnitude − measured magnitude| < 2 and target magnitude
− measured magnitude > 3. Percentages below use the original whole-population
Z weighting; no F64 region result is implied. Zero empirical mass/MCSE does
not prove a region impossible or fully explored.

| Arm | Observation | Within 2 mag, % ± MCSE | Over 3 mag fainter, % ± MCSE |
|---|---:|---:|---:|
| baseline | 58165 | 100.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| baseline | 0 | 99.94790 ± 0.05164 | 0.05210 ± 0.05164 |
| baseline | 1 | 99.95647 ± 0.04424 | 0.00000 ± 0.00000 |
| baseline | 8 | 99.80567 ± 0.10387 | 0.00000 ± 0.00000 |
| informed | 58165 | 100.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| informed | 0 | 99.84141 ± 0.10456 | 0.09732 ± 0.09757 |
| informed | 1 | 99.87484 ± 0.09002 | 0.08504 ± 0.05824 |
| informed | 8 | 99.95024 ± 0.04963 | 0.00000 ± 0.00000 |

All informed−baseline region contrasts have magnitude at most 1.461
combined MCSE and 0.145 percentage points. These coarse categories do not
resolve every possible alternative scene or prove absence of missed modes.

| Observation | Baseline Z relative MCSE | Informed Z relative MCSE | Informed/baseline mean Z | Signed Z difference / MCSE |
|---:|---:|---:|---:|---:|
| 58165 | 3.222% | 4.791% | 0.989199 | -0.188 |
| 0 | 3.742% | 6.410% | 0.919911 | -1.147 |
| 1 | 4.457% | 4.941% | 0.940358 | -0.926 |
| 8 | 1.535% | 2.195% | 0.997453 | -0.095 |

Informed normalizer precision worsens for every object; all mean-normalizer
contrasts remain within 1.147 combined MCSE. Run-weight ESS is 7.890–7.987
for baseline and 7.776–7.973 for informed, out of eight. Small concentration
does not exclude a shared blind spot.

### Cost and descriptive historical comparison

| Arm | Total inference seconds, eight runs | Median seconds/run |
|---|---:|---:|
| baseline | 812.178 | 101.040 |
| informed | 1391.551 | 173.986 |

Every run includes all four observations. Bright constant-control MCSE
falls from **4.215/6.670** to **1.315/2.081**, giving descriptive
variance-times-cost ratios **0.167/0.167** (lower is better). Bright raw
score ratios are **1.148/1.695**, while observation-1 corrected-score ratios
are **2.447/1.828**. These are noisy eight-run estimates and ignore bias;
neither uniform efficiency improvement nor correctness follows.

Under the separate endpoint receipt, observation-0 baseline corrected g1 is
0.400543 ± 0.249085. Its signed contrasts with v6/v7/v9 are respectively
**−0.004145 ± 0.266680 (−0.016 MCSE)**,
**−0.669432 ± 0.272767 (−2.454 MCSE)**, and
**−0.168081 ± 0.292977 (−0.574 MCSE)**. This independent low-effort baseline
does not reproduce v7’s high corrected-g1 mean. It supports treating run-to-run
variation and finite-eight-population uncertainty seriously; it cannot prove
effort alone caused earlier shifts. These are descriptive old-source
comparisons, not strict report-gate passes or exact-kernel equivalence.

## Provenance and reproduction

- Branch `codex/v8-comparison-20261022`, runnable commit `1e173dd`.
- Frozen source `source_v8_comparison_20261022_v2`; manifest SHA256
  `8b9670be5ccbd97e7c81e836e3c0011e3d2488a9647f1f56a17d16170f562eec`.
- Reporting receipt `v8_comparison_20261022_v2_experiment.json`; all eight
  top-level receipt hashes match execution and 104 input hashes are retained.
  Selected completed raw-report outcomes had been read before this reporting
  receipt; it is not claimed as a new preregistered scientific experiment.
- Output `v8_comparison_20261022_v2.json`, SHA256
  `981042a8eaa62d763769caed74af06212879c61687df30d379c5345557749c7e`.
- Result receipt `v8_comparison_20261022_v2_result_receipt.json`; original
  failure `v8_comparison_20261022_failed_receipt.json` remains preserved.
- Endpoint receipt is copied unchanged to
  `v8_comparison_endpoint_receipt_20261022.json`; its hash is in the output.
- All external paths are under
  `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021`.

On a CPU scheduler allocation, use a fresh output path:

```bash
python -m pytest -q tests/test_v8_comparison.py
python scripts/summarize_v8_particle_comparison.py \
  --root /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021 \
  --endpoint-receipt /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/v8_comparison_endpoint_receipt_20261022.json \
  --output /path/to/fresh_v8_comparison.json
```
