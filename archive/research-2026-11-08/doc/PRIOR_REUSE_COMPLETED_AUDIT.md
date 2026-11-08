# Completed prior-reuse v2 audit

## Authority and completion barrier

`scripts/audit_prior_reuse_completed.py` accepts only the original v2 declaration
with SHA256
`ce259bd7d64a32086d01ce30b59ffe278599382bb87e5452fe519ad936571093`.
It requires a finalized `report.json`, all four original pair receipts, all eight
worker receipts, all forty output files, all five process-cost receipts and the
allocation record before examining pair results. A pending, failed or missing
final report stops the audit. There are no retries, replacement seeds, sampler
calls or writes to the original source, reports or output directories.

The declaration binds both complete 320-file sources, the three permitted
implementation changes, the original external inputs and all transitive frozen
helper sources. Those helpers are hash-checked before import. The audit reuses
their source, actual-startup, proposal-provenance, worker and exact saved-output
validators; it does not call their report or finalization functions. The sole
proposal comparison exception is the declared implementation-provenance migration:
all numerical proposal parameters and pilot metadata remain unchanged. This is
the existing analytic target study, not the separate numerical mock-target law.

## Independent checks and interpretation

The audit replays all four baseline/optimized comparisons, including the complete
64-stage scientific histories and particle/scene dtype, shape and array bytes.
It independently checks ordered seeds/arms, original commands and worker binding,
saved output hashes, unchanged allocation and recorded end-of-pair GPU identity.
It recomputes each pair's equal allocation of shared process overhead, all four
ratios, their descriptive mean/standard error, and the final ratio including the
complete report process and declared source preparation. The preparation timer
already includes the startup preflight. GNU time rounding gets the original
0.011-second allowance; no other tolerance is introduced.

All underlying files are pinned and rechecked before a new audit report is
written. The preserved failed v1 startup is retained separately and added to an
explicit experiment-cost total; it is excluded from the successful paired
speedups, as declared. Existing fits and the earlier native CPU replay are
prerequisites outside that successful-cost denominator. Audit-process cost should
be recorded separately by the scheduler launcher.

`status: audit_passed` means that the original replay and descriptive timing
report are reproducible. No speed threshold was declared, and this audit does
not introduce one. A ratio below one can still have a correctly audited report.
There is no automatic adoption, posterior-accuracy acceptance, numerical
mock-target-law validation, global shear-calibration claim, or proof about
unrecorded mutation states/terminal RNG. Installed dependency closure is limited
to the original frozen declaration.

## Running after the original job completes

Run in a CPU scheduler allocation, preferably with an `afterok` dependency on the
original benchmark job. The audit itself does not submit work or require a GPU.
Use a fresh output outside both original directories:

```bash
PYTHONDONTWRITEBYTECODE=1 /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python \
  scripts/audit_prior_reuse_completed.py \
  --declaration /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/prior_reuse_smc_prepared_v2_20261022/declaration.json \
  --output /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/prior_reuse_smc_parent_audit_v2_20261022/result.json
```

Focused validation:

```bash
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -m pytest tests/test_prior_reuse_completed_audit.py -q
```

The 21 focused tests cover the completed-report barrier, missing receipts,
independent cost arithmetic, undercount/negative/nonfinite costs, wrong seeds and
orders, altered GPU/output/array evidence, worker timing bounds, changed proof
files, and duplicate JSON keys. The actual completed benchmark audit remains a
separate scheduled check.
