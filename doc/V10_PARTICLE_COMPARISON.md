# N23 corrected three-way comparison

## Completed audited development comparison

After all 24 sampling tasks, raw report 17422589 and phase report 17422590
completed with exit 0, authorized CPU **17434348** passed **37 focused tests
in 4.77 seconds** and completed the audited comparison (31-second allocation).
All **12 arm/object precision gates**, all three pairwise raw-score/region/Z
gates, and all pairwise controlled-score component gates in both numerical
measures pass. All 466 recorded source/input files, protocol/artifact/seed
checks and full-particle replays passed. No sampler rerun or model forward was
performed for this report, and no held-out outcome was accessed.

The largest controlled contrast is small minus large for object 0, g2:
**+1.158843 ± 0.600652**, or **+1.929307 combined MCSE**, in the sampled F32
measure. Its F64-reweighted counterpart is +1.158957 ± 0.600745
(+1.929200 MCSE). The largest raw-score contrast is +2.239036 MCSE (small minus
broad, object 1 g2); the largest absolute mean-Z contrast is 1.738664 MCSE
(large minus broad, object 8). Broad's original `comparison_gate=false` has
no comparator and is not a failed comparison.

All entries below are posterior **score expectations for fixed observations**,
not shear estimates or a population-calibration result. Errors use independent
whole-population ratio MCSE with every original particle and normalizer.
Controlled values use the fixed independent-pilot constant coefficients.

| Arm | Object | Raw score g1 ± MCSE | Raw score g2 ± MCSE | Controlled g1 ± MCSE | Controlled g2 ± MCSE | Z relative MCSE |
|---|---:|---:|---:|---:|---:|---:|
| small | 58165 | +0.850437 ± 5.610655 | +20.246494 ± 7.183872 | +8.030165 ± 3.566477 | +12.648045 ± 5.739217 | 4.162% |
| small | 0 | +1.200248 ± 1.624535 | -4.937433 ± 0.527471 | +0.435885 ± 0.124709 | -4.982465 ± 0.489975 | 2.148% |
| small | 1 | -3.223766 ± 0.288738 | -3.259235 ± 0.214702 | -2.964724 ± 0.225102 | -3.417093 ± 0.230164 | 1.848% |
| small | 8 | -1.924865 ± 0.069983 | +1.148854 ± 0.026412 | -1.900714 ± 0.062782 | +1.182319 ± 0.026202 | 0.992% |
| large | 58165 | +5.678681 ± 2.544601 | +7.331446 ± 3.051524 | +5.091120 ± 1.389991 | +7.953274 ± 2.172985 | 1.309% |
| large | 0 | +0.020628 ± 0.586801 | -6.175250 ± 0.324390 | +0.596762 ± 0.087427 | -6.141308 ± 0.347430 | 1.866% |
| large | 1 | -2.959995 ± 0.133474 | -3.872036 ± 0.195984 | -2.975285 ± 0.071498 | -3.862719 ± 0.143319 | 1.005% |
| large | 8 | -1.905141 ± 0.027466 | +1.174746 ± 0.016688 | -1.890582 ± 0.025922 | +1.194919 ± 0.012050 | 1.110% |
| broad | 58165 | +3.009664 ± 2.675734 | +9.279259 ± 1.664987 | +4.633070 ± 1.197481 | +7.561177 ± 2.003630 | 7.788% |
| broad | 0 | +1.463304 ± 0.872131 | -5.170575 ± 0.446214 | +0.510743 ± 0.109936 | -5.226695 ± 0.457559 | 3.801% |
| broad | 1 | -2.875166 ± 0.163342 | -3.857524 ± 0.159070 | -2.971274 ± 0.109623 | -3.798956 ± 0.171847 | 2.495% |
| broad | 8 | -1.896676 ± 0.023762 | +1.185275 ± 0.018073 | -1.884869 ± 0.028309 | +1.201635 ± 0.014314 | 1.464% |

The full result and compact receipt preserve signed means, MCSE and every
pairwise contrast separately for F32 and F64. F64 reweighting changes the
numerical measure. Its largest controlled-mean shift from F32 is 0.000526430
(small/object 0/g2); the largest raw-mean shift is 0.000722896
(small/object 58165/g2). The maximum absolute F64 phase-control zero-reference
residual is 1.654885 MCSE (small/object 58165). This is diagnostic evidence,
not an exact identity for the rounded F32 sampling program; that expectation
remains unknown. The F64 identity concerns the intended continuous measure.

Driver inference totals for small/large/broad are **1566.072 / 3843.086 /
3717.636 seconds**. Corresponding summed Slurm task elapsed times are
**1667 / 3944 / 3820 seconds**. Shared phase and raw-report jobs took 177 and
11 wall seconds. These shared costs are not allocated to arms, and reused
pilot/geometry/proposal/control preparation costs remain unknown. The small
arm uses 42.1% of broad's recorded inference time while satisfying these
development checks, with larger MCSE for several scores. This does not
establish a full end-to-end cost advantage or acceptance.

The lower support clamp is corrected identically in all three arms. The
result does not prove finite-RNG exactness or exclude shared rare modes with
eight populations. `development_checks_pass=true` and
**`accepted_sampler=false`**; independent held-out assessment remains pending.

## Evaluation repair and exact receipt

The first evaluation **17423550** failed after five seconds before outcome
access because a substring guard rejected a legitimate pinned source filename
containing `heldout`. Its immutable v2 snapshot and log are preserved. The
reporting-only repair requires canonical containment in the declared source
and a matching expected hash from its already verified manifest; actual bytes
are still hashed. Unlisted files, missing/wrong hash proof, outside-source
data and symlink escapes remain rejected. Three new regressions pass. No
scientific threshold or sampling setting changed.

Runnable repaired code is commit **5285f5b**. The fresh immutable snapshot is
`R/source_v10_comparison_20261022_v3`, manifest SHA256
`4767d2a3eef142ab4f5a387cccc7267d4eaeac59bf7247a2fa2ac6eedf2d0f99`.
The authorized launcher is `R/run_v10_comparison_evaluate_20261022_v3.sh`.
The complete audited result is `R/v10_corrected_comparison_20261022_v3.json`,
SHA256 **09601af0a43c0b5801bc0467b090e588c745713a3c9fd6d86b58f2f9172d1574**.
The [result receipt](evidence/v10_comparison_result_20261022.json) pins the
pre-execution declaration, launcher, source, all input hashes through the
full result, failed attempt, successful log and compact signed results.
Its external identical copy is `R/v10_comparison_result_20261022_v3.json`.

## Frozen scope before outcomes

This helper implements the existing `experiment_v10_corrected_three_way.json`
declaration, SHA256 `9102eb816c20ae4d0a8dbc6f698db194f4ed4ea0eea726138a3f73fb4545d72e`.
It was prepared from root 3edb2ee without reading v10 run protocols, results,
phase arrays or aggregates. The launcher and declaration alone specify paths
and settings. Synthetic unit fixtures use prior schemas; no new model forward,
GPU allocation, fitted coefficient or scientific threshold is introduced.

The arms are small fitted N256, large fitted N1024 and broad Student-t(df5)
N1024. Each has T64/M64 and eight fresh declared seeds. Every arm uses the
same frozen support-corrected source, target, physical geometry and observation
panel [58165,0,1,8]. Fitted arms share the frozen initial-proposal artifact.
Broad means/scales/weights match the original broad measurement components;
its phase stays uniform. No informed-environment augmentation is enabled.

## Gates, all retained

The new `sbsi/particle_comparison_gates.py` reproduces the original raw gates:

- Each arm/object: at least eight independent populations, both finite score
  effective counts at least 400, normalizer relative MCSE strictly below 0.1,
  and normalizer run-weight ESS at least 0.8 times the original run count.
- Every pair/object: both raw score differences at most 3.5 combined MCSE and
  at most 0.1 pooled posterior SD; both region differences at most 3.5 combined
  MCSE and at most 0.05; mean-normalizer difference at most 3.5 combined MCSE.
- Additionally, for **all three pairs**, all four objects and both corrected
  score components: difference at most **3.5 combined independent-population
  MCSE**, separately in sampled F32 and precision-reweighted F64 measures.

Signs, every component, failed gates and zero-denominator cases remain visible.
A nonzero difference with zero estimated MCSE fails; an equal constant has zero
difference. No empirical recentering or averaged cancellation is allowed. The
helper does not change the original reports. It checks their raw verdicts
against independently recomputed moments/gates. A report with no comparator is
labelled not evaluated, not a failed comparison.

`development_checks_pass` is a development result only. `accepted_sampler`
remains false and held-out status remains pending, even if every gate passes.

## Integrity and completion guard

Before any v10 outcome is read, the executable queries Slurm accounting and
requires all 24 tasks of 17422526 plus report 17422589 and phase 17422590 to
be `COMPLETED`, exit `0:0`. There is no force/bypass option. This preparation
does not submit or execute the evaluation. The parent task decides when to
run the prepared command after reviewing the tests and receipt.

The evaluator checks the declaration and launcher hashes, every file in the
pinned support-fix source manifest, per-run implementation paths/hashes, raw
and phase report source hashes, reference/observation bytes, complete artifact
metadata, proposal laws, geometry, distinct declared seeds and pilot exclusion.
It checks fixed betas, moves/blocks/scales, complete final beta=1 receipts and
V100 runtime metadata. Scene CSR sizes and mark counts, all particle arrays,
score means/variances, normalizers and source hashes are checked. Phase arrays
must include each original index exactly once in order, with stride 1.

Frozen corrected scores, original saved scores and numerical-measure deltas
are replayed from arrays. Original derivative/replay criteria and all three
fine finite-difference steps remain fixed. Entire raw and phase aggregates are
recomputed using full original N denominators and independent whole-population
Z weighting. Aggregate weight replay permits the previously tested eight-ULP
roundoff tolerance; it never alters or renormalizes either array.

The raw estimator retains explicit zero-mass runs in its original run count.
Current phase diagnostics cannot represent extinct populations: an explicit
extinction or failed phase job blocks this assessment rather than dropping a
run or inventing a zero control. Unknown failures never become zero mass.
This limitation is reported explicitly; it is not a new extinction exception.

## Precision and cost qualifications

F32 control expectation remains unknown for the rounded sampling program.
F64 reweighting explicitly changes numerical measure; the zero-mean identity
is qualified for the intended continuous posterior, numerically evaluated.
The corrected support bound is not a complete mixed-measure finite-RNG proof.
Eight independent populations and passing gates cannot rule out shared rare
modes. No held-out outcome is used.

Cost reporting separates per-arm summed Slurm task elapsed seconds (including
source checks/startup/model loading) from driver inference timers. It retains
allocation metadata but labels neither as GPU busy time. Shared phase-job and
raw-report wall times are reported only as totals, never assigned to arms.
Pilot-generation, proposal/geometry fitting and phase-coefficient preparation
costs remain null/unknown unless a future receipt provides them. Full end-to-end
cost is therefore unknown; driver-only timings cannot establish a full-cost
acceptance claim.

## Prepared command and completed validation

Runnable code is commit **73bff31** on `codex/v10-comparison-20261022`.
The immutable 406-file snapshot is
`R/source_v10_comparison_20261022_v2`, manifest SHA256
`090b9272823f2a88a72bf327bc61f150a2df52b4fee117ffaddd82f41ccbb5a0`.
No cache files are included. The prepared launcher verifies this manifest and
every file, then asserts local package imports before running the guarded
comparison. `R` denotes
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021`.

Focused original CPU **17423323** passed 28 tests in 2.79 seconds. Review then
added six regressions requiring nonempty differences and exactly five finite
phase moments before slicing corrected columns. Final full CPU suite
**17423438 passed 882 tests, 1 skipped, in 85.43 seconds** (95 seconds
allocation; exit 0). Both requested frozen legacy-source checks were enabled.
The one warning is the existing NumPy scalar-conversion deprecation in a
scene-exchange test. Tests include threshold boundaries, independent MCSE,
raw absolute score/region limits, mean-Z disagreement, zero-mass run counting,
missing completion tasks, protocol/seed/proposal drift, phase thinning/index
reordering, altered corrected scores/deltas, precision-law changes, and
finite eight-ULP weight replay.

Full-suite attempts 17423354 and 17423380 stopped at collection because a
tracked legacy test imports its script by a bare name and the runner lacked
`cwd/scripts` on `sys.path`. The script was present and hashed in both
snapshots. Both failures/logs are preserved; adding that frozen directory to
the test process repaired the runner without changing source or thresholds.

The [preparation receipt](evidence/v10_comparison_preparation_20261022.json)
records exact source/launcher hashes, test jobs, failed attempts and the
unsubmitted command. Its external copy is
`R/v10_comparison_preparation_result_20261022.json`.

**At the original preparation receipt, no v10 outcome had been read, no
scientific gate result calculated, and no evaluation submitted by this
subtask.** The following original command is preserved; the authorized repaired
v3 evaluation above supersedes it:

```bash
sbatch --parsable --partition=cluster --cpus-per-task=2 --mem=8G --time=00:10:00 --job-name=v10-comparison --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/v10_comparison_evaluation_%j.log /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/run_v10_comparison_evaluate_20261022.sh
```

Launcher: `R/run_v10_comparison_evaluate_20261022.sh`, SHA256
`1f0e9953818cd5ce2d870562f419a1c12f8f9b17983c3922dcf80256c2fb77ce`.
It writes the fresh path `R/v10_corrected_comparison_20261022.json` and refuses
to overwrite an existing file. The runtime completion guard remains mandatory
regardless of when this launcher is submitted. Source/declaration/input
integrity failures raise; scientific gate failures are retained in the output
with `development_checks_pass=false`, never converted into sampler acceptance.
