# N23 corrected three-way comparison preparation

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

**No v10 outcome has been read, no scientific gate result calculated, and no
comparison evaluation submitted by this subtask.** The parent can run this
prepared CPU command after all sampling and report jobs complete:

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
