# Corrected heldout runner

## Scope and immutable inputs

`scripts/run_corrected_particle_heldout.py` executes the fixed procedure in
[the preparation declaration](CORRECTED_HELDOUT_PREPARATION.md#conditional-fixed-procedure).
It submits no jobs. Every invocation requires Slurm, a frozen declaration and
source, complete external input hashes, and a hashed completed development
receipt with all three raw and corrected-phase comparisons passing. It never
reuses historical heldout pilots, replaces seeds, retries steps, or adapts fits
after production begins. This implementation was validated with synthetic CPU
fixtures only; no heldout scientific computation was performed by its author.

The declaration schema remains `sbsi-heldout-corrected-declaration-v1`. In
addition to the fitter's fields, the runner requires:

- `source_files`: the complete manifest as absolute path/SHA256 pairs.
- `external_files`: actual prior/model/config/reference input hashes, including
  reference `protocol.json`, `result.json`, and `observations.parquet`.
- `development_receipt`: `{path, sha256}` for the completed n23 comparison;
  its three comparison gates and recursively referenced source hashes must pass.
- `preparation_cost_receipt`: `{path, sha256}` with kind
  `scheduler_allocation_wall_upper_bound` and finite nonnegative
  `process_wall_seconds`, `cpu_allocated_seconds`, `gpu_allocated_seconds`.
- Full `pilot_proposal_metadata` and `reference_proposal_metadata`, generated
  from measured inputs before pilots, plus exact runtime versions, original
  V100 node/device and two inference torch threads.
- `blocks`, `power`, `scale` and fixed effort/seed/replica settings for all three
  recipes. Five ordered groups retain four observations and density batch4096.

The declaration must be `OUTPUT/declaration.json`. Both the runner and imported
SBSI package must resolve inside the declared frozen source. All source and
external inputs are verified before and after each successful stage.

## Scheduled stage interface

Use the original environment and the existing study GPU ceiling. Launch from
the frozen source through the same explicit bootstrap used by other workflows:

```bash
python -u -c 'import sys,runpy;sys.path[:0]=[".","scripts"];runpy.run_path(sys.argv.pop(1),run_name="__main__")' \
  scripts/run_corrected_particle_heldout.py warmup \
  --declaration /ABS/OUTPUT/declaration.json --index 0
```

Warmup indices are 0–4. Each runs four fresh Gaussian N256/T64/M32 pilots,
fits geometry and initial proposal using the retained fixed rules, checks all
four complete phase diagnostics, and runs the separate declaration-bound phase
fitter. `group_N/warmup_complete.json` binds all pilots, diagnostics and fits.
A completed corresponding stage receipt is also mandatory for production.

Production indices are 0–39; `group=index//8`, `replica=index%8`. Replace
`warmup` above with `production`. Even replicas run candidate then reference;
odd replicas reverse that order. Each arm's complete phase diagnostic follows
its population. Candidate is fitted N256/T64/M64; reference is broad Student-t5
N1024/T64/M64. Production uses frozen group geometry and phase coefficients.

After all forty production tasks finish, run `report` without `--index` in a
CPU-only Slurm allocation. It validates every run/phase receipt, generates ten
raw reports and ten phase aggregates, and requires all five groups, all eight
populations per arm, unchanged raw gates, and both corrected score components
within3.5 combined whole-population MCSE under both F32 and F64 measures.
Acceptance additionally requires strictly lower all-inclusive candidate cost.

## Failure and cost receipts

Every stage reserves `NAME_started.json` before subprocesses. Each step writes
an exclusive log and timing receipt. Completed or failed stage receipts retain
every child cost and wrapper verification overhead. An exception creates a
stage-specific panel rejection receipt; an interrupted reservation without a
completed stage is also ineligible. There is no retry or overwrite interface.
Phase process exit zero is insufficient: the numerical checks and exact source,
protocol, population count and coefficient identities must pass. Extinction or
unknown failure rejects the panel; no seed disappears from the experiment.

Shared pilots, their phase checks, geometry and coefficient fitting are charged
fully to each recipe; proposal fitting is candidate-only. Production and raw/
phase reporting are arm-specific. Preflight, verification and comparison costs
are split equally. Actual work counts shared steps once. Reports retain all
three cost fields. CPU/GPU allocation seconds mean allocated resource count
times full measured wrapper wall, not busy compute. They exclude queue time,
Slurm allocation time outside the wrapper, and final receipt serialization.
Preflight uses its separately preserved allocation upper bound. Process sums
are not parallel panel elapsed time; development/search cost is not amortized.

The statistical scope is conditional on the frozen independent fits. F32
corrected scores remain qualified diagnostics; the periodic identity applies
to the intended continuous F64 target. Passing is bounded fixed-y evidence,
not proof of global rare-mode coverage or population shear-calibration accuracy.

## Synthetic validation

CPU job17434665 tested the runner and declaration-bound fitter from an immutable
266-file snapshot. Every source hash was checked before and after. Fixtures cover
development gating, external/source/panel changes, exact pilot recipe/provenance,
phase-content failures, independent seeds, counterbalanced order, preserved
failed costs and no retry, all cost components, strict cost inequality and the
raw-plus-both-measure conjunction. The validation receipt records the exact
test count, source hashes and preserved output. Root performs integrated full
suite validation before freezing any actual scientific declaration.
