# Conditional density-ratio response regularization

## Owner-directed objective

This supersedes the proposed joint Fourier training trial. The only derivative
regularizer in this trial is balanced BCE for the shear-leg label, with logits
provided by the measurement flow itself. There is no separate classifier
network and no inference-time correction.

For a matched base galaxy U and its measured outputs x0, x1, write

    z(x,U) = log q_phi(x | S_0.05 U) - log q_phi(x | S_0 U)
    L_ratio = mean_pairs 0.5 * [softplus(z(x0,U)) + softplus(-z(x1,U))]
    L_total = L_NLL + lambda * L_ratio.

The logit is **not divided by shear** inside BCE. Both conditional densities
are evaluated on each observed measurement; no model draws are needed.
The four outputs are measured ngmix e1/e2, MAG_AUTO and log FLUX_RADIUS.
The shared output standardization Jacobian cancels in the density ratio.
NLL remains necessary to constrain the densities themselves, since the ratio
alone does not identify each density.

At the population optimum of balanced leg classification, the unconstrained
logit equals the corresponding conditional log density ratio. Dividing that
logit by 0.05 gives a forward secant, with O(0.05) error as an approximation
to the derivative at zero under smoothness. It can also be interpreted as a
central difference about shear 0.025 along each pair's applied direction.
These statements do not turn the finite stencil into an exact derivative.

## Population and implementation

`sbsi/flow_ratio.py` implements per-pair BCE and the NLL+BCE epoch.
`sbsi/flow_training.py --ratio-response-weight` selects this path; default zero
preserves existing recipes. Positive ratio weight rejects competing derivative
weights and target files, preservation loss, altered NLL weight, independent
structure noise and decorrelation weighting. No grid is constructed and no
Fourier features or sampling routines run in this training path.

The two catalogues are matched one-to-one by (case,input_index) inside each
case-disjoint split. Both classes contain exactly the same matched base IDs.
The source shear and destination shear norm are checked before folding shear
into intrinsic shape. Pre-shear intrinsic e1/e2 and every other conditioning
feature must agree exactly across each pair (matching missing values allowed).
Zero-shear destination rows are dropped/reported; unmatched rows are excluded
from BCE and retained in NLL.

This balances U across labels but **conditions BCE on usable measurements in
both legs**. NLL conditions on usability in the evaluated leg. Their optimal
ratios need not agree. We do not reweight away this difference or claim that
the available matched pairs identify the desired per-leg conditional ratio.
Full-population validation remains essential; the ratio score alone cannot
certify shear calibration.

Each usable leg row has unit NLL weight. The historical cross-catalogue
`per_target_weights` call treated the two shear measurements of a matched ID
as duplicates, giving them half the weight of unmatched rows. The ratio path
bypasses that weighting, because distinct shear legs are distinct observations;
matching still rejects duplicated IDs within a leg. Historical recipes are
unchanged. This necessary population-weighting change accompanies the new loss.

Training cycles shuffled pair batches alongside all NLL batches. Validation
visits every NLL row and pair once, using separate denominators. Checkpoint
selection uses the held-out sum of NLL and weighted BCE. Metadata records the
definition, weight, stencil, population limitation, pair counts and history.

## Controlled pilot and validation

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/ratio_response_v1/`.
`trial_definition.json`, `source_manifest.json`, `jobs.json` and shell launchers
record the recipe and source hashes.

The first trial fixes **lambda=1**, five epochs, LR 1e-5, batch4096, seed501,
the same 160/40 case split, all available rows (`--max-rows 0`), and frozen
transforms from the original new plain 100/100 parent. This is a warm start:
the parent previously had grid regularization, but this fine-tuning objective
contains none. All mean-response, photometry, paired-transport, dispersion,
Fourier and preservation weights are zero. Lambda and epoch selection are
not chosen from ConstGold m. SWA is saved separately; the primary candidate
minimizes the held-out NLL+BCE objective.

| Stage | Slurm job |
|---|---:|
| Corrected full test suite | 16331142 |
| Training and existing density/derivative diagnostics | 16331146 |
| Full-domain compatibility audit | 16331147 |
| Complete 100-case ConstGold evaluation | 16331148 |
| Pooled m and whole-case bootstrap | 16331149 |
| Matched four-output ratio audit | 16331150 |

Training depends on successful tests. Domain/evaluation/scoring and the ratio
audit have explicit afterok dependencies and invalid-dependency cancellation.
No job was monitored to completion. The two-hour watcher used
`distribution_response_v1/scheduled_check/watch.json`. On 2026-09-06 the owner
requested stopping wake-ups: the configuration is now disabled and the task's
cron entry removed, with unrelated entries preserved. Resource detection runs
inside the training allocation. At owner request, the pending training job was
updated in place to one V100 on inter, eight CPUs, 96 GB and a four-hour limit.
It started via backfill at 13:08:07 CEST on th-cl-nv01 (same16331146, no restart).
Torch2.4.1+cu124 includes sm_70 support. Downstream validation requests are unchanged.

Full ConstGold retains cases40–139, true 18<mag<25.8, 0.37<Re<1.5;
measured |e|<0.6, MAG_AUTO<25.8, flux radius>=0.75 arcsec; 64 QMC CRN draws,
seed7301, shear +/-0.02. Each leg independently detects and selects individual
draws before pooling its own numerator/denominator. The definition is
**m=100*(R11_sim/R11_model-1)**, with 10,000 whole-case bootstrap replicates,
seed20260902. See [response conventions](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).

`scripts/audit_flow_ratio_response.py` compares parent, old, strong paired and
the new candidate on the same 200k matched Re>0.5 validation rows. It uses
only case IDs and domain cuts from the existing population-spec NPZ, not its
grid targets. It reports BCE, paired whole-case uncertainty, matched physical
output NLL and model hashes. The shared smaller diagnostic domain permits old
model comparison; it is not substituted for complete-population closure.
Existing derivative/grid audits in the post-training launcher are diagnostics
only and do not enter optimization or checkpoint selection.

Six focused tests passed in 33.42s: analytic Bayes BCE optimum, reversal,
Jacobian cancellation, RNG invariance, gradients through a real flow,
unequal-batch epoch aggregation, unmatched NLL rows and option guards.
Initial full suite 16331130: 331 passed, 2 skipped, one synthetic CLI fixture
failure because the test omitted explicit four-output target names. The fixture
was corrected; full suite **16331142 passed 332 tests, 2 skipped in 32.54s**,
exit0, verified during the owner-requested status check. The training dependency
is cleared. Python compilation and all launcher syntax checks pass.

## Completed pilot: better density scores, worse full response

All six stages completed with exit0; training and its diagnostics took21m50s
on the V100. The five-epoch best-validation checkpoint has SHA256
`dc77f8ba1782684e2400424e183c87421beab88f65b01fb034bd2401417c28f6`.
Validation NLL+BCE decreased each epoch; epoch5 was selected. Training used
6,834,259 matched pairs, validation1,708,854; fixed base-U checks passed.
Unmatched train source/destination rows23,001/130,419 and validation5,499/32,495
remain NLL observations, with6/1 zero-shear destination pairs dropped from BCE.

Full100-case ConstGold gives **m=+24.572781% +/-0.271615 percentage points**
(whole-case bootstrap SE), 95% CI[24.040507,25.099825]%.
R11_sim=0.6063833218070074 and R11_model=0.4867703184443902.
This is worse than the original parent's+15.782797% and strong paired
model's+1.075825% +/-0.219835pp on the same full endpoint. The0.3% objective
remains unmet; no model promotion or empirical correction.

The matched Re>0.5 audit nevertheless improves BCE0.590630->0.586471
(paired difference-0.004159 +/-0.000204 case SE), and physical-output
NLL-3.613236->-4.053852. These are different-population diagnostics and
do not certify complete-population response. Before measured cuts, the
full response error is+0.041715; applying cuts adds-0.161328, leaving
total error-0.119613. This localizes the dominant endpoint error to the
selection contribution, without identifying its training cause.

Verification in external `completed_verification.json`: actual hashes of
all four audit checkpoints and detector, common candidate across outputs,
full cases/support/cuts/CRN settings, all.json scorer-input hash, and an
independent stdlib reconstruction from per-case numerators/denominators.
Uncertainty comes from the completed Slurm scorer, which reproduced the
evaluation's residual bootstrap SE; no inference/bootstrap rerun on login.
Single training seed, reused validation, finite stencil and both-measured
supervision remain limitations. Removing old derivative losses and changing
NLL weighting accompanied BCE, so deterioration cannot be assigned to BCE
alone. An otherwise identical NLL-only control would separate its incremental
effect; none was launched during this completion check. Wake-ups remain off.

## Convergence comparison: lambda1 and100

Owner requested both weights together and explicitly retained matched BCE with
all independently usable rows in NLL. Both start afresh from the same original
100/100 parent with seed501, frozen transforms, LR1e-5, batch4096, uncapped
rows and all other derivative weights zero. The five-epoch lambda1 checkpoint
is not used as the new starting point. External root is
`plain_complete_flow_response_re037_v1/ratio_convergence_v1/`, with separate
`lambda1/` and `lambda100/` recipes, logs, jobs and output paths.

Training now allows200 epochs with patience15 on validation NLL+lambda*BCE.
The best validation checkpoint is selected. External `check_convergence.py`
checks the unrounded history, objective arithmetic, and15 consecutive epochs
without improvement, then records model/history hashes in convergence.json.
Reaching the epoch ceiling while still improving fails this gate and blocks
downstream afterok jobs. This is an operational validation stopping criterion,
not proof of an optimization optimum. Each train job requests one V100,
8CPUs,96GB,12h; resource detection runs inside Slurm. No live job was restarted.

| Stage | lambda1 | lambda100 |
|---|---:|---:|
| Train and stopping-rule gate |16332473|16332474|
| Domain audit |16332475|16332480|
| Full100-case ConstGold |16332476|16332481|
| Full m scorer |16332477|16332482|
| Matched ratio audit |16332478|16332483|
| Historical full Re>0.5 mean-response audit |16332479|16332484|

Owner subsequently paused ConstGold: domain/evaluation/scorer jobs16332475–477
and16332480–482 were cancelled and verified CANCELLED before execution.
Training and both validation audit types remain scheduled. External
`constgold_pause.json` records this override to the original job chain.

The last audit uses the earlier40-case Re>0.5 targets and all matched rows
(sample ceiling2M exceeds the1.12M population),64CRN draws,seed7301,
central+/-0.02, to make L_R and L_theta comparable with the earlier
approximately0.005/0.003 values. The existing200k Re>.37 diagnostic is also
retained but must be labeled separately. Complete ConstGold cuts and scorer
remain as above. No result is yet established for these submitted runs.

Pair identities are not mathematically necessary for ratio BCE. With balanced
global class counts and independently usable rows, the Bayes classification
logit is log[p1(x|U)/p0(x|U)] + log[rho1(U)/rho0(U)], where rho denotes
the base-U distribution in each usable-leg sample. Detection/usability can
make these distributions differ. An all-row conditional-ratio objective needs
that second term handled or equalized; simply shuffling unpaired rows and
balancing leg counts does not remove it. No such change is made in these runs.

Validation before submission: implementation hashes match the full-suite-tested
pilot; all twelve launcher Bash syntax checks pass; four light gate checks
cover continued improvement at cap, valid plateau, insufficient patience and
incorrect objective. Source and launcher manifests are recorded. The owner
stopped two-hour wake-ups: watcher remains disabled, cron absent, and no
polling loop was started. Follow-up is owner initiated.

## NLL-only parent comparison

Owner authorized two additional runs while keeping the original-parent runs
untouched. The NLL-only parent is
`measurement_flow_plain_pure_nll_re037_s501_swaavg.pt`, SHA256
`d0ac7b040719710df5b9a262715a49e3678dec35340029864e22178dd28ee114`.
Direct checkpoint inspection confirms zero response/coupling penalties, no
fine-tuning reference, same feature/output/support definitions, and exactly
equal condition preprocessing and output transforms to the100/100 parent.
It is the seed501 SWA over epochs57–64, originally trained with4M row cap.

New root `ratio_nll_parent_convergence_v1/` holds lambda1/lambda100 launchers,
trial definitions, source/launcher manifests, jobs and parent_verification.json.
Fine-tuning uses all available usable NLL rows and matched BCE, no other
derivative penalty, LR1e-5,batch4096,seed501,max200 epochs,patience15 and the
identical stopping-rule gate above. Each train requests one V100,8CPUs,96GB,12h.
Only initialization changes relative to the corresponding existing ratio run.

| Stage | lambda1 | lambda100 |
|---|---:|---:|
| Training and stopping-rule gate |16333098|16333099|
| Matched ratio audit |16333100|16333102|
| Full historical Re>0.5 mean-response audit |16333101|16333103|

The ratio audit includes the NLL-only parent first, followed by original100/100,
old mean-head, strong paired and candidate models. All use the same diagnostic
rows. Mean-response audit keeps the earlier full40-case Re>.5 protocol.
No ConstGold evaluation, domain audit or m scorer is submitted while the owner
has those stages paused. Existing jobs are untouched; watcher stays off.
Production hashes match the previously tested implementation, all six launcher
syntax checks pass, and the convergence gate is unchanged. Results pending.
