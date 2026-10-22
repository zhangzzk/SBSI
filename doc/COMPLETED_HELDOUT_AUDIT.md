# Independent completed corrected-heldout audit

## Purpose and completion barrier

`scripts/audit_completed_corrected_heldout.py` independently checks the final
corrected heldout v3 decision from saved complete reports. It first requires
`heldout_corrected_v10_v3_20261022/result.json`, with five groups, twenty
observations, both recipe costs and explicit final decisions. A missing or
incomplete final result stops before any intermediate population/report reads
and before creating an audit output. A scientifically failed final result is
valid input and must remain failed.

The checker never invokes `report_panel`, a reporter, a sampler, a subprocess
or a scheduler command. It never writes inside the original source/panel and
never overwrites an audit output. It is supplementary evidence and does not
change the original acceptance contract.

## Source authority and execution

Authority is deliberately restricted to one experiment:

- Root: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021`.
- Panel: `heldout_corrected_v10_v3_20261022`.
- Source: `source_corrected_heldout_v3_20261022`, all 320 manifest entries.
- Declaration SHA256: `ad718136d4ac00a37ffcaddf09dcfad92b15e68e4ebbf82ead597ed1b7c1f758`.
- Source manifest SHA256: `fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1`.

The source manifest and every source byte are verified before importing the
frozen runner. Its strict declaration, warmup, population, phase, stage and
report provenance validators are reused. Its decision/comparison and cost
functions are not used. The independent checker additionally binds final
report hashes, raw-report population artifacts, reporter identities, stage
reservations, timing-receipt paths and output logs. Original source/input
identities and all recorded output hashes are checked again at the end.

Run in a fresh interpreter inside a CPU-only Slurm allocation; no GPU or model
evaluation is required. Python and NumPy must match the original declaration.
Do not pre-import an `sbsi` package from another worktree. For this branch:

```bash
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python \
  /home/z/Zekang.Zhang/SBSI/.claude/worktrees/smc-completed-panel-audit-20261022/scripts/audit_completed_corrected_heldout.py \
  --output /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/completed_heldout_v3_independent_audit_20261022.json
```

No job has been submitted by this change. Parent orchestration owns scheduling
after the final panel result exists. This audit can read many artifact bytes;
it is not intended for login-node execution on the real panel.

## Independent calculations and output

The checker recomputes each arm's raw precision from saved posterior variance,
whole-population covariance, normalizer relative MCSE and run-weight ESS. It
rechecks both score and both region comparisons, their absolute bounds, mean
normalizing-mass agreement and corrected g1/g2 agreement under both declared
measures. All twenty object conjunctions, five group conjunctions and the final
accuracy/efficiency conjunction must agree with the original result.

Independent accounting requires 46 stage receipts, 235 step receipts, 20 pilots,
80 production populations, 100 phase checks, 15 fits and 20 reports. It charges
shared pilot preparation to both recipes, proposal fits only to the candidate,
each arm's production/diagnostics/reporting to that arm, and verification plus
preflight equally. Actual work is counted once. All three cost columns are
checked against the original receipt using its existing accounting tolerance
(relative 1e-12, absolute 1e-9). Acceptance retains the strict candidate <
reference process-wall inequality; equal cost fails.

`status=audited` means evidence and the original decision are consistent. It
does not mean `accepted_sampler=true`. `failed_objects` lists every failed
object and its individual gate reasons. Invalid/missing/changed evidence after
the completion barrier yields a fresh `status=invalid_evidence` receipt,
`accepted_sampler=false` and nonzero process exit. A completed scientific
failure produces a successful audit process with false scientific decision.

## Scientific scope

The accepted target, if all original gates pass, is
`p_A(target) * q_E(environment) * K32`. The F64 diagnostic changes only the
likelihood factor through `exp(logK64-logK32)`. Neither measure substitutes the
numerical mock-target prior q_N. This checker performs no numerical-prior
reweighting and does not validate omitted support or shared unvisited modes.

Controlled-score precision/absolute-SD bounds, T=0, all-five-moment contrasts,
F64 normalizer precision/agreement and F32/F64 equivalence remain supplemental
diagnostics. No new gate is imposed. Passing is bounded fixed-observation
integration and inclusive recipe-cost evidence; it is not population shear
calibration or validation of a new CPU optimization/500-CPU throughput claim.
