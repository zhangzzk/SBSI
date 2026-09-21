# Covariance-weighted shape-response training

## Owner-directed objective

Two trials use lambda=1 and 10, starting independently from the completed
plain NLL-only seed-501 SWA parent. There is no mean head. The owner clarified
this initialization after the first submission; the 100/100-parent jobs were
cancelled and replaced. See [the parent correction](#nll-only-parent-correction).
The only response regularizer is

    L = NLL + lambda/(2*4*n_cells) * sum_B (mu_phi,B - target_B)^T V_B^-1 (mu_phi,B - target_B).

Each mu is the flattened shape response matrix (R11,R12,R21,R22). V_B is
the 4x4 covariance of the estimated simulation cell response, not the
covariance of individual shapes or individual-galaxy responses. The penalty
is averaged over cells and four components, without occupancy weighting.
This is the owner-requested normalization revision: the initial unnormalized
sum is divided by720 on the180-cell grid; lambda remains1 or10. The previous
photometric, shape-MSE, ratio, transport, distribution, and dispersion losses
are off. NLL still fits all four outputs: ngmix e1/e2, magnitude, log radius.

## Matched forward response and covariance

The source pair catalogues have zero and +0.05 primary shear, with random
directions per galaxy. See [catalogues](CONVENTIONS.md#2b-half-shear-legs--the-forward-catalogue)
and [matched extraction](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).
The new model response uses those exact paired contexts, unlike the historical
central +/-0.02 model estimator. Common base draws are shared across the two
legs within a draw group, as required by [CRN](CONVENTIONS.md#6d-common-random-numbers).

For a cell containing N_B pairs with primary shear vectors g_i, define

    C_B = mean_i(g_i g_i^T)
    target_B = vec(mean_i[(y_i1-y_i0) g_i^T] C_B^-1)
    mu_phi,B = vec(mean_i[(E_phi y_i1-E_phi y_i0) g_i^T] C_B^-1).

Only the first two physical ngmix output components enter these expressions.
Both sides use the same finite design, pairing and population. This is a
forward regression response, not a claim of an exact zero-shear derivative.
One pair cannot identify all four matrix entries; the varied shear directions
across its cell identify the matrix.

Delete each whole simulation case in turn, refit the matrix, and calculate

    V_B = (K-1)/K * sum_c (R_B,(-c)-mean_c R_B,(-c)) (same)^T.

Each cell must have at least eight cases, stable full/deleted-case shear
designs and positive-definite covariance with condition number <=1e10.
Otherwise preparation fails; no diagonal flooring, outlier removal or global
fallback target is substituted. Off-diagonal response-component covariance
is retained; cross-cell covariance is omitted in this block-diagonal loss.

## Data and fixed grid

Retain the parent's 6x6x5 true-magnitude/true-size/R_blend edges and seed-501
160/40 case split. Rebuild the train and validation targets separately from
all matched rows usable in all four outputs, with 18<true magnitude<25.8,
0.37<true Re<1.5 and the production source/detection cuts. These new targets
have a precisely shared model/simulation population; they are not copies of
the old shape-only grid targets. No ConstGold data enter training or selection.

Match by (case,input_index), reject duplicate IDs within a leg, verify fixed
conditioning properties, and drop/report unmatched, zero-destination-shear and
nonfinite-grid pairs. NLL retains all independently usable leg rows with unit
row weight. The paired response still conditions on usability in both legs;
it does not identify detection or moving-selection responses of the full population.

The reference condition and target transformations are frozen. Shared prepared
files include NLL values/contexts, matched row indices and IDs, forward shear
vectors, cell design vectors, fitted targets, covariances, precisions and
case-jackknife replicates. Manifests record hashes, populations, cell counts,
case coverage and covariance conditioning. Preparation and catalogue scans run
under Slurm.

## Unbiased stochastic loss and validation

At each training step, sample four cells uniformly without replacement.
For each selected cell, draw two independent groups of 128 galaxies with
replacement; each group uses 16 antithetic latent draws per galaxy and CRN
across its shear legs. Averages A_B and C_B are independent unbiased estimates
of the finite-population model regression matrix. The stochastic penalty is

    1/(2 * 4 * sampled_cells) *
        sum_sampled_B (A_B-target_B)^T V_B^-1 (C_B-target_B).

Its expectation equals the stated normalized full-grid objective. Independence across both galaxy
samples and latent draws avoids adding a model-dependent variance penalty
from either source. Scores may be negative; do not clamp them. Sampling cells
or galaxies is stochastic optimization, not a fixed catalogue row cap.

Validation uses all held-out matched pairs, two independent groups of 16
draws, fixed seed 7301, and no row subsampling. Accumulate complete cell means
before evaluating the cross-quadratic. Held-out means and covariances are
separate from training. Epoch zero evaluates the exact parent and participates
in checkpoint selection. The selected checkpoint minimizes held-out
NLL+lambda*L_grid. Each epoch retains both full cell-mean estimates and scores.

## Training recipe and execution

Shared parent: `measurement_flow_plain_pure_nll_re037_s501_swaavg.pt`
under the external `plain_complete_flow_response_re037_v1` cache. Parent SHA256:
`d0ac7b040719710df5b9a262715a49e3678dec35340029864e22178dd28ee114`.

Both runs: uncapped catalogue rows, seed501, NLL batch4096, AdamW LR1e-5,
weight decay0, gradient clipping5, max200 epochs, patience15. The best
validation checkpoint is primary; no SWA candidate is selected. A completed
epoch also saves the current model and optimizer for recovery. Reaching the
epoch or 48-hour job cap is not convergence. Each GPU job requests one A40,
8 CPU cores and 96GB RAM. Preparation requests 8 CPU cores and 160GB and must
pass the full test suite before building shared artifacts. Both training jobs
depend on successful preparation and verify source/artifact hashes.

Implementation: `sbsi/flow_grid_response.py` and
`scripts/train_flow_grid_response.py`; launchers:
`jobs/job_prepare_flow_grid_response.sh`, `jobs/job_train_flow_grid_response.sh`.
The existing data loader has an opt-in grid mode preserving the original
shear components before folding them into shape; historical modes are unchanged.

## Validation and limitations

Analytic tests compare covariance to a direct delete-case reference, verify
the finite-design regression identity and absence of a model-variance gradient
penalty, reject missing/degenerate cells, and run preparation plus a real
one-epoch plain-flow training/checkpoint round trip on a small synthetic population.

This experiment tests uncertainty-weighted cell-mean shape response. It does
not directly supervise photometric responses or selection-boundary transport,
and it does not certify a full-population multiplicative bias. Existing
ConstGold pause and disabled watcher remain in force; no inference/evaluation
chain or recurring monitor is submitted by these launchers.

## NLL-only parent correction

Owner confirmed both runs should start from the NLL-only checkpoint.
Cancelled the original 100/100-parent training jobs16334840/16334842 and
verified cancellation. Completed preparation16334839 is retained: the full
suite passed336 tests with2 skipped in68.09s, then both population targets
and covariances passed the preparation gates.

Direct checkpoint comparison verified identical architecture (apart from
training-only response annotations), condition preprocessing and target
transforms. The NLL-only parent has response/coupling weights0, no warm start,
seed501, SWA epochs57–64 and the original4M-row cap. Prepared inputs and
simulation covariances do not depend on the starting network weights, so the
replacement manifest references the same immutable prepared artifacts and
changes only the parent identity. Original manifests and logs are preserved.

Replacement root: `plain_complete_flow_response_re037_v1/grid_covariance_nll_parent_v1/`.
`parent_verification.json` records the comparisons and hashes; `jobs.json`
records lambda1 job16334863 and lambda10 job16334864. All loss, sampling,
optimizer, validation and resource settings are unchanged. The generic
training launcher reads the parent from this root's prepared manifest;
the original preparation launcher is not rerun. Both workers verify all
prepared artifact hashes before training. No convergence or new result is claimed.

## Normalization revision

After the epoch14 status, the owner requested averaging the response penalty
over180 cells and4 components. Cancelled unnormalized NLL-parent jobs16334863
and16334864 and submitted fresh NLL-only starts: lambda1 job16335138 and
lambda10 job16335139, gated on full-suite job16335137. Root:
`plain_complete_flow_response_re037_v1/grid_covariance_normalized_v1/`.
Prepared data/covariances and parent are unchanged and reused by reference;
source snapshots and manifests record the new implementation.

`--grid-loss-reduction mean` divides both the stochastic training score and
the full validation score by4*n_cells before multiplying by lambda. The
default `sum` preserves the previous recipe. Checkpoint metadata records
the effective objective and divisor. Epoch validation artifacts retain raw
per-cell scores and `loss_sum`, alongside normalized `loss` and `loss_divisor`.
The prepared-data definition remains unchanged because its response targets,
covariances and sampling design are unchanged. The new lambda1/10 penalties
are720 times weaker than before; this changes optimization, not just reporting.
All other training, validation, stopping and resource settings are unchanged.

## Response batch-size diagnostic

Following discussion of unstable small-cell batches, a bounded external
diagnostic compares frozen copies of the NLL-only parent and the current
normalized lambda1 best checkpoint. It changes no model parameters. Four
configurations isolate the sources of stochastic gradient noise:

| Configuration | Cells | Galaxies/cell/group | Latent draws/galaxy/group |
|---|---:|---:|---:|
| Current | 4 | 128 | 16 |
| More galaxies | 4 | 1024 | 16 |
| More cells and galaxies | 16 | 1024 | 16 |
| More cells, galaxies, and latent draws | 16 | 1024 | 64 |

Each configuration uses eight independent repetitions at fixed weights and
reports score scatter, gradient noise, gradient mean norm, elapsed time and
peak allocated GPU memory. The estimated squared mean-gradient norm includes
a finite-replicate bias correction; it can be negative when eight repetitions
cannot resolve the signal and must not be interpreted as a physical negative
norm. This is a diagnostic, not a new training recipe or a claim that the
larger batch has passed a stability gate.

The largest effective batches use a two-pass gradient computation: calculate
whole-group cell means without retaining graphs, then replay the identical
galaxies and latent seeds in chunks and accumulate the exact cross-quadratic
gradient. Parameters stay fixed between passes. A small direct-autograd
comparison tests this replay calculation before the scheduler audit runs.
The actual training implementation is unchanged. Artifacts and frozen
checkpoint/source hashes live under the external
`plain_complete_flow_response_re037_v1/grid_covariance_batch_diagnostic_v1/`.

## Full-population sampling audit and autonomous follow-up

`scripts/audit_flow_grid_sampling.py` evaluates frozen NLL-only, lambda1 and
lambda10 checkpoints on every matched held-out pair, using16 and64 draws in
each independent group. It compares group quadratics, their cross-product
and the quadratic of the averaged means, and verifies the exact decomposition
between them. Per-galaxy disagreement contributions locate influential rows
and cells. Shape-draw maxima and unit-disk exceedance counts are compared with
the actual measurements, without clipping outputs or changing the objective.
Squared row influences rank concentration only: they do not sum to the loss,
which includes cross terms. Top pair IDs, cells and contributions are saved.

Full-suite gate16337523 precedes GPU audit16337524. The external
`grid_covariance_sampling_audit_v1/` root holds the frozen model manifest,
hash guards, launchers and output artifacts. The diagnostic does not establish
that large batches or any specific optimizer setting will succeed; it provides
evidence to distinguish ordinary sampling noise from rare extreme responses.

The owner authorized autonomous overnight work, preserving the agreed formula,
first targeting shape and then size/flux, and explicitly requested hourly
one-shot checks. An isolated `flow_response_goal_v1/watch.json` tracks the
diagnostics/current trials for this task. The hourly cron checker queues a
continuation only for completion/failure or another meaningful state change;
unchanged jobs remain quiet. The previous unrelated disabled watcher remains
off. The due-time guard also prevents an automatic goal continuation from
polling early. First check:2026-09-06T23:48Z (2026-09-07 01:48 CEST).

Subsequent training decisions must use measured gradient noise and draw
stability, preserve independent validation, and distinguish an unbiased noisy
score from verified response accuracy. Consider full-cell response averages
with chunked exact gradients and explicit complete-pair coverage per epoch.
Extending to size/flux must constrain a non-cancelling, property-resolved
response: an isotropic population's scalar-output response can average to
zero even when its orientation-dependent response is wrong. Define and audit
that conditioning/frame convention before choosing photometric targets.

## Complete-cell gradient accumulation

`sbsi/flow_grid_full_cells.py` provides a candidate training method, not yet
enabled in the launched training runs. For selected cells C and d=4 response
components, compute independent latent estimates A_B and C_B over **every**
matched galaxy in each cell, with fixed model parameters. The score remains

```text
L_C = (1 / (2*d*|C|)) sum_B (A_B-target_B)^T P_B (C_B-target_B),
P_B = V_B^-1,   B in C.
```

After a first pass without an autograd graph, the derivative with respect to
the first mean is `P_B (C_B-target_B)/(2*d*|C|)`, and with respect to the
second is `P_B^T (A_B-target_B)/(2*d*|C|)`. Replay exactly the same seeded
latent draws in small galaxy chunks, multiplying each contribution by the
corresponding coefficient divided by the **whole cell** galaxy count. Sum
these gradients before any optimizer update. This is the derivative of the
whole-cell score; squaring and updating separately in each chunk would define
a different objective. Common random numbers still connect the forward shear
legs inside each independent latent group. Stochastic context layers are
rejected because explicit latent seeds alone cannot replay them.

The candidate epoch routine reshuffles all cells and all NLL rows. It assigns
complete cells and disjoint NLL blocks to macro updates, accumulating each
block's NLL gradients alongside its response gradients before clipping once.
At four cells per update, 180 cells imply45 optimizer updates per epoch,
covering13,821,950 NLL rows and6,834,259 pairs in each latent group. Thus NLL
blocks average about307,154 rows, although activation chunks can remain4096.
Every response contribution is evaluated and then replayed. This changes the
number and variance of updates substantially; it does not establish a suitable
learning rate or runtime. Fixed-latent replay removes the activation-memory
barrier, while complete-cell means remove galaxy subsampling noise. Latent
integration noise and variability between cells remain.

Four focused tests compare replay gradients with a direct full graph, including
an NLL term and unequal cell sizes, verify complete epoch coverage and updates,
and check invalid/stochastic configurations. `scripts/audit_flow_grid_full_cells.py`
measures frozen-checkpoint gradients with randomly selected complete cells and
with a fixed cell selection. The latter isolates latent noise for those cells;
the two variance estimates must not simply be subtracted, because the fixed
selection may not represent the full grid. Saved gradients, cell IDs, means,
runtime and peak GPU memory support comparison with the subsampling audit.

## Paired validation uncertainty and finite integration noise

`sbsi/flow_grid_case_validation.py` adds a validation diagnostic without
changing the fixed V_B used by training. The simulation and flow response
estimates use the same held-out galaxies, so their sampling errors are
correlated. To estimate uncertainty of their difference, delete the same
simulation case from both sides and refit both response regressions, including
the shear design. Jackknife these paired residuals. Using only the simulation
covariance would omit the model variance and the model–simulation covariance.
Using their variances as independent terms would also be incorrect.

For case c and cell B, accumulate raw normal matrices
`G_cB=sum_i gamma_i gamma_i^T` and cross matrices
`T_cB=sum_i delta_y_i gamma_i^T`, separately for simulation and flow. Their
case-deleted responses are

```text
R_B^(-c) = (T_B - T_cB) (G_B - G_cB)^-1.
D_B^(-c) = R_model,B^(-c) - R_sim,B^(-c).
Cov_JK(D_B) = (C-1)/C sum_c (D_B^(-c)-mean_c D_B^(-c))
                            (D_B^(-c)-mean_c D_B^(-c))^T.
```

Flatten output-by-shear matrices before forming the covariance. The existing
sampler's per-galaxy contributions already contain the full-cell inverse shear
design. A case sum of those contributions must first be multiplied by
`H_B=G_B/N_B` to recover its raw T_cB. Deleting and averaging those contributions
without this conversion retains the wrong shear design.

`summarize_paired_grid_predictions` consumes complete row contributions from
independent model integrations. It reports the paired residual jackknife and,
separately, the integration covariance of the mean across K independent groups:
`V_MC,B=sum_k (R_kB-R_barB)(R_kB-R_barB)^T/[K(K-1)]`. The quadratic of R_bar
contains integration variance. Subtracting `tr(P_B V_MC,B)/(2*d)` from its
per-cell normalized quadratic equals the average of all distinct-group cross
scores, preserving the current objective. Negative estimates are retained;
they do not demonstrate accuracy. V_MC quantifies computational integration
noise and does not replace V_B or introduce a new variance penalty.

The case jackknife itself still contains finite integration noise. Compare
independent groups/draw counts before interpreting it as physical-population
uncertainty. Singular paired covariance is allowed (identical predictions and
measurements have zero residual uncertainty); no inverse or significance is
manufactured from it. Case-deleted residuals are retained so any later aggregate
can include covariance between cells. This helper is ready for the next
checkpoint audit; it has not altered the source-frozen audits already submitted.
Single-checkpoint diagnostics also do not establish the ensemble certification
described in [the seed convention](CONVENTIONS.md#9-seeds).

## Accumulated trainer ready for measured batching trials

`scripts/train_flow_grid_accumulated.py` supports both complete-cell epochs and
larger sampled batches. `sbsi/flow_grid_accumulation.py` accumulates the NLL
gradient over small chunks of an effective batch, then the exact sampled
response gradient by the same two-pass mean/replay method. It clips and updates
once. Galaxy indices and latent draws are independent between A and C; the
forward shear legs within each group retain common random numbers.

For example, moving from NLL4096/cells4/galaxies128 to
NLL131072/cells16/galaxies1024 would increase both effective batches32-fold.
The number of NLL updates would fall from3375 to106, while sampled response
occurrences per group per epoch stay approximately1.73million. This avoids
silently multiplying response work32-fold while holding the NLL batch fixed.
It does change the optimization schedule; there is no automatic learning-rate
scaling or claim that these example settings are optimal. Sampled mode still
uses replacement and does not traverse every matched pair. Full mode uses the
complete-cell epoch helper: its update count follows the cell grouping, and
the effective NLL block size is determined from all NLL rows divided among
those updates, ignoring the sampled-mode NLL batch argument. Logs report actual
NLL rows, optimizer steps, response occurrences/coverage, average effective
batch sizes, gradient norms and clipping fractions.

The trainer validates the prepared data and NLL-parent hashes, preserves the
plain architecture and transforms, and rejects parent metadata with nonzero
response regularization. All epochs including the parent are saved. Four or
more independent validation integration groups provide all distinct-group
cross scores and an MC jackknife standard-error diagnostic, alongside the
quadratic of their average and its integration-variance contribution. The
best-file selector remains a noisy screening statistic. Its MC jackknife is
not an exact confidence bound near a degenerate quadratic, and selecting a
checkpoint still requires an independent audit. No training loss or V_B change.

Committed epoch snapshots include the optimizer, random states, history and
arguments. `--resume` permits extending the epoch limit with the same other
settings. Resumed and uninterrupted two-epoch runs match exactly in focused
CPU integration tests for both batching modes. The tests also verify the
sampled replay gradient against direct autograd with NLL, NLL accumulation
against explicit macro batches, complete NLL coverage, checkpoint selection
and the multi-group cross-score arithmetic.

`jobs/job_train_flow_grid_accumulated.sh` is a scheduler launcher requiring
explicit batching/draw/learning-rate choices and a new source-verified trial
root. It is prepared for use after the diagnostic review; no replacement
large-catalogue training has been submitted through it yet.

## Validation populations and smaller-shear transfer

The current40 case-disjoint held-out cases are used for checkpoint selection.
They are a validation set, not an untouched final test. Repeating a checkpoint
evaluation with new model draws checks Monte Carlo integration; it does not
create new simulation cases or remove repeated-selection effects. Do not call
a new split of already inspected validation cases an untouched test either.

A metadata/boundary inspection on Sep7 found the current g0 and g0.05 catalogue
boundaries at cases0 and199, consistent with the recorded full200-case counts.
The existing `det_meas_crowd_g0.02_test_full.feather` boundaries are0 and99;
the raw `det_meas_ngmix_g0.02_test.feather` boundaries are0 and19. The filenames
alone therefore do not establish a new population. Exact whole-file case
counts and intersections with the frozen160/40 split are being audited in
`grid_covariance_evaluation_inventory_v1/` by a scheduled case-column-only
reader; no measurement values or model responses are evaluated by that job.
Schema differences must also be handled explicitly before matching contexts.

After training settings/checkpoints are frozen, a matched g0-to-g0.02 response
comparison can test transfer to a smaller forward shear. Apply the same step
and matched population to model and simulation, preserve the baseline grid,
and report unmatched rows and any common-pair restriction used to compare
steps. This tests a different shear amplitude on existing scenes. Untouched
population testing still requires a suitable new simulation population or a
strictly frozen external evaluation. ConstGold remains evaluation-only and
must never select the training recipe or checkpoint.

The serialized NLL parent's eight condition features are `e1_input_p`,
`e2_input_p`, `sersic_n_input_p`, **true** `r_input_p`, **true** `Re_input_p`,
and `nbr_flux_near/far/max`; missing-value indicators make context_dim16.
The historical feature-set name contains `meas`, but the serialized feature
list and transformations are authoritative. The warm-started flow does not
condition on its own measured magnitude or measured radius through that name.

## Physical size and flux response extension

The existing flow models `(e1,e2,m,log r)`. For physical outputs
`Y=(e1,e2,r,F)`, invert the target standardization for **each draw**, then set
`r=exp(log r)` and `F=10^[0.4*(zero_mag-m)]`, using the catalogue zeropoint.
Average these physical outputs before differencing the shear contexts. In
general `E[exp(log r)] != exp(E[log r])` and `E[F(m)] != F(E[m])`; the current
magnitude/log-radius response cannot be relabeled as physical flux/radius
response. Retain common base draws between both shear legs and independent
groups for the covariance-weighted loss.

An isotropic galaxy population can have zero **sky-frame** mean scalar-output
shear response despite strong orientation-dependent response. A deterministic
toy with `delta_s=2*e dot gamma`, intrinsic `|e|=.3`, eight evenly spaced
orientations and four shear directions has sky-frame fitted response `(0,0)`
but baseline intrinsic-frame response `(.6,0)` (numerical residuals below1e-14).
The executed toy and its exact definition are saved in
`grid_covariance_evaluation_inventory_v1/scalar_orientation_toy.json`; it is
an algebra check, not a simulation or model result.

Consequently, the later scalar-response constraint must retain orientation
information. One candidate is a frame fixed by each galaxy's **baseline**
intrinsic ellipticity: use `u=e/|e|`, `v=(-u2,u1)` and regress physical scalar
differences on `(u dot gamma,v dot gamma)`. Define a fixed convention for
exactly circular inputs and inspect ellipticity-magnitude dependence; never
recompute the frame from a sheared measurement. Another option is explicit
baseline-orientation conditioning. Retain the established sky-frame shape
constraint and validate scalar orientation profiles so a cancelling average
cannot masquerade as success. Choose/record the exact target and covariance
before training this extension, after shape training is stable.

Four physical outputs versus two shear components form a4x2 response. A4x4
Jacobian with respect to input shape, size and flux would additionally require
validating size/flux **input perturbations**; the existing shear-pair catalogue
does not establish those columns. Do not claim such a Jacobian from an8-entry
shear-response vector. These distinctions preserve the full scientific scope
without treating successful shape-only tuning as completion.

## Sep7 scheduled noise review and guarded larger-batch pilots

The23:48Z one-shot scheduler snapshot found all initial diagnostics and both
normalized training runs completed successfully. Both original trials stopped
at epoch22 by patience and selected epoch7. Those selections are not reliable
accuracy certificates. Full-suite gates for complete-cell, paired-validation
and accumulated-training work passed344/351/356 tests respectively, with two
skips each. The latest of those took49.58s and verified source hashes on both
sides of the suite.

Frozen training-gradient diagnostics (eight repetitions per configuration):

| Checkpoint | Cells × galaxies/group × draws | Gradient noise RMS | Noise / sample-mean gradient norm | Seconds/update |
|---|---:|---:|---:|---:|
| NLL parent | 4 ×128 ×16 | 443.3 | 1.156 | 0.194 |
| NLL parent | 16 ×1024 ×16 | 248.4 | 0.634 | 3.381 |
| NLL parent | 16 ×1024 ×64 | 247.7 | 0.636 | 7.602 |
| lambda1 epoch7 | 4 ×128 ×16 | 327.9 | 3.062 | 0.106 |
| lambda1 epoch7 | 16 ×1024 ×16 | 2409.7 | 3.853 | 3.302 |
| lambda1 epoch7 | 16 ×1024 ×64 | 1368.5 | 4.060 | 7.627 |

These gradient-norm ratios use a finite-replicate mean and are diagnostic,
not confidence bounds; some debiased signal estimates are negative/unresolved.
At lambda1, a smoother sampled **loss** accompanied much noisier gradients.
Thus larger batches help the parent but do not establish a remedy for the
already damaged checkpoint. Complete-cell evaluation on a fixed four-cell
selection reduced lambda1 latent gradient noise to1.7% of its mean; reshuffling
four complete cells gave229%, showing substantial variability between cells.
That fixed selection is not representative evidence for every cell. Complete
four-cell gradients cost about25–31s per update with256-galaxy chunks.

Full-population held-out tail/integration diagnostics identify the pathology:

| Frozen checkpoint | Cross score,16 draws/group | Cross score,64 draws/group | Largest shape amplitude at64 draws |
|---|---:|---:|---:|
| NLL parent | 7.71597 | 7.72733 | 50.42 |
| lambda1 epoch7 | 0.57067 | 1.15447 | 16427.52 |
| lambda10 epoch7 | -3.84481 | 3.11544 | 13029.64 |

Lambda1's top100 galaxies contribute98.94% of the **squared row-disagreement
ranking** at16 draws (89.51% at64); this ranking is not an additive loss
decomposition. Rare extreme samples, not simply too few total model draws,
dominate its integration noise. Lambda10 also damages the distribution broadly:
about9.45% of its64-draw samples lie outside the unit disk. The reconstructed
measurements have zero such exceedances and maximum amplitude1.0. The parent
has small nonphysical tails too, but its full-cell score is stable across these
draw counts. Do not promote the negative-score lambda10 winner: its NLL is
+0.65004 versus the parent's-0.28405, and its cross-score changes sign with
more draws. Detailed immutable results live in the three diagnostic roots.

Submitted short, controlled pilots in `grid_covariance_accumulated_pilot_v1/`:
full-suite gate16337895, then lambda1/10 jobs16337896/16337897, each one A40,
96GB, two-hour maximum and at most three epochs. Both start from the unchanged
NLL-only parent. Effective NLL batch131072, cells16, galaxies1024 per group,
training draws16, LR1e-5; NLL activation chunks4096 and response replay chunks1024.
Validation uses every matched held-out pair, four independent groups with64
draws each. Retaining16 training draws follows the parent diagnostic's lack of
gradient improvement at64; keeping LR fixed isolates the batching change.
The larger replay chunk is intended to reduce launch overhead within the
available46GiB memory budget. The256-galaxy diagnostic used1.6GiB; runtime and
peak memory for the launched recipe still require measurement.

Before these pilots, added two checkpoint eligibility and early-stop guards:
validation NLL must be at most0.05nat/row above that trial's parent, and the
rate of samples with|e|>2 must be at most four times the parent's measured rate.
A five-event parent denominator floor avoids a zero-rate ratio in tiny tests.
These are pragmatic pilot guardrails, not model corrections or final accuracy
thresholds. Every candidate is retained; a failure stops further epochs,
keeps the best pointer on an eligible checkpoint, and prohibits automatic
resume of that failed recipe. Actual validation samples are observed without
changing draws, cell means, the loss, V_B or sample weights. Eight focused
tests passed11.31s, including unchanged group means under tail observation and
both guards rejecting an otherwise improved score while retaining the parent.
The updated full suite is pending its scheduled gate.

Next one-shot review:2026-09-07 00:48Z /02:48CEST. Check stability, tails, NLL
and independent response accuracy before extending. If tails still grow,
investigate a proper flow/output-support representation; merely increasing
draw counts is not an adequate response to the measured pathology.

## Original shape support and target precision

The maximum reconstructed measured amplitude1.0 above was evaluated after
float32 target storage/inversion. `TargetStandardizer` casts the original
components to float32 before subtracting its float32 means and dividing by
its float32 scales. Both this quantization and the norm accumulation can
change a near-boundary classification. For example, the strictly interior
float64 vector(.6,.79999999999) casts to components whose double-precision
squared norm exceeds1, although their float32 norm rounds to1. This tested
arithmetic example does not establish that the catalogue contains such rows.

`grid_covariance_shape_support_audit_v1/`, CPU job16337937, will distinguish
these effects using every prepared training/validation response pair. It
matches original catalogue rows by(case,input_index), rejects missing or
duplicate matched keys, and requires exact equality between re-standardized
original values and cached targets. The scan reads only these keys and both
measured ngmix shape columns, using the same prepared populations and splits
as [the training data](#data-and-fixed-grid). Source rows outside the prepared pair
population are counted separately. All cached NLL rows receive their own
precision summary; raw unmatched NLL-row support is not established because
their keys are absent from the prepared tensors.

Reports include counts inside/on/outside the unit disk and margins to the
boundary for original values, the raw float32 cast, and float32/float64 cache
inversion. Except for an explicitly labeled legacy float32 norm, squared
norms are accumulated in float64. Each split/leg saves the1000 most extreme
original rows and their cache representations with identifiers. These are
read-only diagnostics with source hashes and catalogue stat checks. No
boundary clipping, epsilon contraction, support transform or new selection
is applied. Results are pending the00:48Z one-shot review.

This determines whether a proposed open-disk density would contradict the
recorded measurements or their cached precision. Finite catalogue support
alone does not establish theoretical support; any future representation must
preserve a properly normalized density with its change-of-variables Jacobian
and make the treatment of recording precision explicit.

## Independent checkpoint validation and original case statistics

The prepared response target archive retains both the full-cell measurement
fit R_B and every deleted-case fit R_(-c),B in double precision. With
G_B=sum_i gamma_i gamma_i^T and G_c,B the case contribution, recover original
measurement cross statistics without using rounded NLL target values:

    C_c,B = R_B G_B - R_(-c),B (G_B-G_c,B).

Here C_c,B=sum_(i in case c, cell B) delta_e_i gamma_i^T. The new external
`grid_covariance_independent_validation_v1/audit.py` recomputes G from the
preserved forward shear vectors and checks the reconstructed full/deleted
fits and covariance against the saved originals. The synthetic test verifies
this recovery against direct accumulation of original measurements, including
case-dependent shear designs. It rejects mismatched statistics or case order.

The audit evaluates every held-out pair with fresh, independent latent groups,
preserving antithetic draws and within-pair CRN. Full row contributions are
retained in a NumPy array so later diagnostics can use the same integration.
The initial NLL-parent audit16337981 uses64 draws per group, four groups,
batch1024 and seed bank beginning320000000001. It recomputes all-row NLL,
observes actual shape tails, and reports each shape-response component with
paired case residual jackknife and separate MC covariance of the integrated
mean. The initial parent provides a reference for candidate audits, not a
response-trained success claim.

The normalized cross score uses the same fixed validation precision and
equal-cell/equal-component reduction as checkpoint validation. An equal-cell
mean response and a unit-pair global WLS response are separate descriptive
statistics. Global WLS sums raw shear/cross statistics before fitting; it is
not an occupancy-weighted average of cell matrices. For both summaries,
averaging happens on joint case-deletion or integration-group matrices before
estimating covariance, retaining correlations between cells.

The complete repository suite and four focused audit tests run on the worker
before this source-verified analysis. Existing training sources and loss are
unchanged. These are reused development validation cases: independent draws
check integration, not generalization to untouched galaxies. The case residual
jackknife includes finite integration noise; separate MC covariance is not
subtracted from it. The score MC jackknife SE remains a finite-replicate
diagnostic, especially near a quadratic minimum, rather than an exact bound.
No physical size/flux accuracy or multiple-model-seed certification follows
from this shape-only audit.

## Disk coordinates as an unadopted density option

Current local BlendEMU source fits a Gaussian ngmix model with GPriorBA(.5).
The installed sims1 ngmix2.3.1 prior rejects g1^2+g2^2>=1, and its Gaussian
shape conversion rejects |g|>=1. This gives a code-level basis for investigating
unit-disk support for successful fitted ngmix shapes. A(-1,-1) failure sentinel
is a different outcome. These current source snapshots do not establish the
historical source version, filtering or recording precision of every training
catalogue; the original-support audit must still resolve the actual population.

External `grid_covariance_shape_support_math_v1/` contains a tested mathematical
prototype, not a model used in training or inference. For x in R2, define

    h(x) = x / sqrt(1+|x|^2),
    h_inverse(y) = y / sqrt(1-|y|^2),  |y|<1.

The radial and tangential Jacobian eigenvalues are respectively
(1+|x|^2)^(-3/2) and (1+|x|^2)^(-1/2). Therefore

    log|det D h(x)| = -2 log(1+|x|^2),
    log|det D h_inverse(y)| = -2 log(1-|y|^2).

This map is smooth at the origin and commutes with rotations, avoiding an
axis-dependent square support. A density using it must evaluate its base
density at h_inverse(y) and add the inverse log determinant, with any target
standardization Jacobians handled consistently. Tests verify the determinant
against automatic Jacobians and integrate a transformed standard-normal
density in polar coordinates, recovering both unit probability and its known
radial CDF. Double-precision inverse/forward checks and derivatives include
the origin. Float32 input values produce float64 physical outputs; unresolvable
radial gaps or invalid inverse inputs raise rather than being clipped.

No existing trained flow is altered by these tests. Adoption requires actual
pilot/support evidence, a correct density wrapper, and an NLL-only fit under
the new representation before response fine-tuning. Changing coordinates
changes numerical log densities unless all Jacobians use the same reference
measure; compare likelihoods consistently in the original measured units.
Applying h to existing fine-tuned outputs without refitting/renormalizing
their likelihood would not satisfy the model contract.

The existing normalized response objective can then use these physical shape
outputs per draw, before cell averaging and forward differencing. This map
must not be applied after averaging, since it is nonlinear. Its mathematical
support restriction alone proves neither response accuracy nor preservation
of the photometric likelihood. Physical size/flux integration still requires
its own per-draw inverse transforms and orientation-sensitive validation.

## Rare-tail mean matching and NLL cost

The completed tail audit motivates checking a property of the objective,
independent of batching. For one scalar response component, let

    Q0(y|g) = Normal(R0*g, sigma^2),
    Q_eps(y|g) = (1-eps) Q0(y|g) + eps Normal(beta*g, sigma^2),
    beta = R0 + (Rtarget-R0)/eps.

For every eps>0, the mixture mean is Rtarget*g. Yet at every measured y,g,
Q_eps >= (1-eps)Q0, hence

    -log Q_eps(y|g) + log Q0(y|g) <= -log(1-eps).

Thus a fixed response change can reside in vanishingly small probability
mass while its upper bound on NLL cost tends to zero. The bulk component's
response remains R0. This is a constructed normalized density, not an
assertion that an observed checkpoint is exactly a two-Gaussian mixture or
that the exact population likelihood has its optimum at such a mixture.
It shows why mean matching plus a small NLL change does not establish a
physically appropriate conditional distribution when the output is unbounded.

Use common random numbers by sharing the Bernoulli component and Gaussian
residual between g=0 and g=h. The paired response is then R0 with probability
1-eps or beta with probability eps, giving

    Var(paired response) = (1-eps)*(Rtarget-R0)^2/eps,
    Var(N-replica mean response) = Var(paired response)/N.

Perfect CRN therefore does not remove this source of variance. At an exactly
matched target, independent A/C mean residuals have zero expected cross
product, but its variance is the product of their mean variances. Fixed
covariance weighting and cell averaging do not impose an amplitude bound.
The independent-replica N in this example is not automatically the nominal
number of correlated antithetic draws in the SBSI evaluator.

The executed `grid_covariance_tail_mechanism_v1/experiment.py` checks these
identities, the pointwise NLL bound, and independent quadrature of the NLL
change. With R0=.5, Rtarget=.7, h=.05 and eps=1e-6, the rare component has
mean10000.025 at the sheared leg; NLL cost is at most1.0000005e-6nat/row.
The response standard error remains.2 at a million independent replicas.
These are illustrative analytic values, not new training measurements.

For a scalar physically restricted to[-1,1], a component's forward secant
has magnitude at most2/h. The analogous mixture therefore obeys

    |Rtarget-R0| <= eps * (2/h + |R0|).

It cannot move a fixed finite-step response using arbitrarily little mass.
This does not bound every infinitesimal derivative of a bounded conditional
mean, nor guarantee a sufficiently low WLS/gradient variance in practice.
The actual grid design, conditional density fit and independent integration
must still be checked. This result supports evaluating a proper bounded
density if the measured larger-batch pilots remain unstable; it introduces
no extra training penalty, covariance change, clipping or calibration factor.

## Sep7 larger-batch outcome and precision-preserving density work

The00:48Z scheduled review found both guarded pilots terminated successfully
on their first post-training validation, with NLL and tail guards failing:

| Pilot | Validation NLL | Normalized cross score | Shape amplitude >2 rate relative to parent | Maximum amplitude |
|---|---:|---:|---:|---:|
| Parent, same validation draws | -.28405 | 7.72254 | 1 | 137.99 |
| Lambda1, epoch1 | +.07750 | 1.50185 | 18.85 | 5330.63 |
| Lambda10, epoch1 | +.34244 | 1.54000 | 31.27 | 7694.21 |

Both had106 optimizer updates, and every update exceeded gradient clip5.
Median gradient norms were204.28 and2136.65 respectively; maximum14210.74
and120187.98. The pilots each finished in about1147s including two full
validation evaluations. Their best pointers retain the parent, and rejected
epoch1 weights remain available for diagnosis. These results warrant changing
the output representation; a32x effective-batch increase did not solve the
tail or likelihood problem. The response loss itself remains as agreed.

Original-support audit16337937 matched all8,543,113 prepared train/validation
pair keys in each original leg, with no missing/duplicate matches and exact
legacy target reproduction. All17,086,226 original matched shape measurements
are strictly inside the disk, maximum radius.99999998657. Float32 cache
inversion in float64 has two training and one validation rows outside the
disk by rounding. Casting that cache to double does not recover original
precision. Unmatched NLL-row original values are not covered by this audit.

Fresh parent audit16337981 gives cross7.715945 with diagnostic MC SE.002495,
confirming stable parent integration across seed banks. Its global diagonal
responses(.816308,.814371) differ from measured(.822230,.826141); paired-case
residual SE(.003376,.004223) exceeds MC SE(.0000292,.0000145). Equal-cell
component residual RMS(.194344,.066843,.078442,.202066) demonstrates why a
small global error does not establish local accuracy. These are one model
seed, the reused40 validation cases, and the matched forward convention in
[CONVENTIONS §6c–6d](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).

New CPU recovery16338126 in `grid_covariance_disk_data_v1/` uses an installed
frozen source snapshot to recover original float64 e1/e2/mag/log-radius on
every NLL row, including unmatched rows. It verifies cached targets/contexts
exactly and preserves original response keys, shears and fixed covariance.
It rejects invalid raw support rather than clipping or dropping rows. Saved
raw values and keys will enable precision-preserving likelihood evaluation;
coordinate summaries compare radial-atanh and algebraic representations.

The next model needs a proper density wrapper, correct Jacobians and an
NLL-only fit before response regularization. Coordinate choice and likelihood
comparison must account for the observed boundary and dynamic range; the
algebraic prototype is not automatically the best optimization coordinate.
No bounded model or inference preset has yet been trained/promoted. The
physical size/flux response phase follows demonstrated shape accuracy.

## Plain disk flow and NLL-only coordinate refit

`sbsi/flow_disk.py` adds the explicit `disk_affine` architecture. It wraps a
plain affine flow, with no conditional mean head. For its first two outputs,
let `w` be standardized unbounded coordinates, `u=s*w+m`, and

\[
e=h(u)=\frac{\tanh\lVert u\rVert}{\lVert u\rVert}u,
\qquad
u=h^{-1}(e)=\frac{\operatorname{atanh}\lVert e\rVert}{\lVert e\rVert}e.
\]

Both ratios have limit1 at the origin. This rotation-equivariant bijection
has forward log determinant

\[
\log|Dh(u)|=\log\operatorname{sech}^2 r+
\log\frac{\tanh r}{r},\qquad r=\lVert u\rVert.
\]

Stable origin series and float64 shape arithmetic/Jacobians are tested. Raw
measurements use the inverse map without contraction or clipping. For the
original standardized physical shape `y=(e-a)/b`, the inverse density adds
`log|Dh_inverse(e)| + sum(log(b)-log(s))` to the affine base log density.
Mag/log-radius retain their original standardized coordinates. Thus the
likelihood reference measure is the original parent's standardized four
measurements; changing coordinates does not omit a density Jacobian.

The radial-atanh coordinate reduces the observed boundary dynamic range:
at the maximum original matched radius.99999998657, its radius is about9.4,
compared with roughly6100 for the earlier algebraic prototype. Exact all-NLL
coordinate means/scales come from the separate raw-recovery worker's training
split. These are invertible coordinate normalizations with explicit density
Jacobians, not empirical response scalings. Recovering every raw NLL row and
checking its support remains a prerequisite to training.

`TargetStandardizer` now supports an explicit float64 checkpoint option;
legacy metadata and float32 behavior stay unchanged by default. Disk
checkpoints require this option, the correct ngmix target order, and matching
coordinate metadata/state. Bundle target loading and sampling preserve the
declared precision. Passing float32 targets or demoting disk buffers fails
explicitly. The base neural network and contexts may remain float32. The
wrapper returns each physically mapped shape draw before the existing
complete-flow response sampler computes any mean or shear difference.

At unbounded radii around19 or greater, float64 tanh can round to1. Such
samples are returned at their representable value, without an intentional
contraction; their analytic log Jacobian remains finite. A rounded boundary
draw cannot be inverted numerically. This is a finite-precision limitation,
not strict open-disk representability for every possible latent value.
Subsequent sampling audits must distinguish boundary rounding from substantive
support violations. Original measured likelihood inputs must be interior.

`scripts/train_flow_disk_nll.py` establishes the new NLL-only baseline before
using the agreed response regularizer. It loads only the original pure-NLL
affine weights into the wrapped base. The new coordinates change the initial
density; this is explicitly recorded, and epoch0 is not represented as the
unchanged original parent. The script measures both the original cached NLL
and original-raw standardized NLL through that parent's float32 base, to make
the precision/reference comparison visible.

Three brief rate probes (1e-5,1e-4,7e-4) use identical shuffled training rows,
64 steps with8192 rows, and all validation NLL rows. This brackets the prior
fine-tuning rate and the historical plain-NLL launcher's7e-4. The rate is
chosen by measured finite validation NLL, then all weights and optimizer state
reset to the coordinate-changed initialization. Every subsequent epoch visits
every13,821,950 training NLL row, including the final partial batch. Probe
rows are diagnostic subsets, not full epochs. This initial refit is capped at
three epochs/two GPU hours and may not establish convergence. All probes and
epoch checkpoints are retained, with an atomic committed optimizer resume;
CPU interrupted/resumed two-epoch equivalence is tested. Best-NLL selection
does not establish response accuracy or an untouched test result.

The normalized covariance-weighted cell response formula, simulation targets,
fixed V_B and cell weighting remain unchanged. No response fine-tuning is
automatically launched from a low NLL, and no inference preset is changed.

## Independent validation of disk checkpoints

External `grid_covariance_disk_validation_v1/disk_audit.py` extends the
independent checkpoint audit to the new density and precision contract.
It accepts only plain affine or disk-affine four-output models, requires the
exact original physical target and condition transforms, and allows only the
explicit target-dtype extension required by disk checkpoints. NLL uses the
recovered original float64 measurement rows; a legacy affine base receives
float32 only after standardization in the same reference coordinates. Cached
inverse32 measurements are never substituted into the disk density.

The auditor reuses the source-hashed, already tested original case-statistic
recovery and paired residual jackknife. The measured full/deleted-case WLS
fits, covariance, equal-cell objective and separate global/equal-cell summaries
remain unchanged. It retains complete per-row contribution groups with a new
seed bank, plus separate MC covariance and paired-case uncertainty. Five
focused tests verify target precision/transform compatibility, exact agreement
with production response contributions, NLL and draw counts, original case
statistics, and support diagnostic accounting. No checkpoint has yet been
evaluated by this new auditor.

Support diagnostics include the literal squared-radius counts at/above1,
counts within a descriptive 16*float64-epsilon band, and counts above that
band. A band membership is not proof that an individual event is rounding;
it is not a support repair or a data/draw cut. Every draw still contributes
to the response. The raw radius maximum and engineered e1/e2/mag/log-radius
extrema are retained. This does not certify physical size or flux response.

The prepared manifest has no models and no recovered-data result hash, so
the audit cannot run prematurely. After the NLL trial completes, freeze the
chosen checkpoint snapshot and completed raw-data hash, then run the audit
with the frozen NLL source environment. Independent draws test integration
stability on the same development-validation cases; they do not provide an
untouched galaxy test or certify the eventual model-seed ensemble.

## Sep7 original-precision verification and bounded NLL launch

The01:48Z scheduled review accepted recovery16338126 and gate16338236, both
completed exit0. All17,277,654 original NLL shape measurements are strictly
inside the disk, including191,428 rows outside the matched response pair set.
Legacy target/context reproduction and original pair keys/shears are exact.
No population repair is needed: minimum squared-radius gaps are2.68567e-8
train and2.98822e-8 validation. The cache's few outside-disk values came from
the earlier float32 path; recovered original values resolve that precision
problem without clipping or dropping rows.

Training radial-atanh radius median/99/99.9/99.99/max are
(.49734,1.81778,2.52055,5.25233,9.40952); the algebraic alternative gives
(.51810,2.99789,6.17754,95.50248,6102.02177). Train-only coordinate means
(-.000304555,.002052861) and scales(.471577765,.477363237) enter the density
wrapper with its explicit Jacobian. Validation coordinate radius maximum is
9.35615. These verified distributions support the radial-atanh choice.

Full-suite gate passed373 tests with two skips in49.97s and unchanged source
hashes. A tiny training-runtime CPU smoke also verified Python3.9 / Torch2.4.1
optimizer steps, the float64 boundary path and exact checkpoint round trip.
Bounded NLL-only trial16338316 was submitted at about01:49Z, one A40/96GB with
a two-hour cap and at most three full epochs, using the recorded probe/reset
protocol. It is not yet a response-trained or validated candidate. Independent
full-response auditing follows a completed useful NLL baseline; the agreed
covariance-weighted regularizer is retained for the next training phase.

## Response fine-tuning from a disk NLL baseline

`scripts/train_flow_grid_accumulated.py` now accepts an explicit
`--initial-model` completed disk NLL refit and `--raw-root` verified float64
recovery directory. The original prepared grid remains the source of all
contexts, matched pairs, shear designs, targets, counts and fixed precision
matrices. Only the NLL target tensor is built from the original raw values
using the disk checkpoint's float64 standardizer. The old float32 cache is
never cast and substituted for these originals.

Initialization checks require the exact original physical and condition
transforms, original pure-NLL ancestry, a completed NLL-refit phase and its
recorded prepared/raw input hashes. Coordinate initialization, rate-probe and
response-trained checkpoints are rejected as NLL baselines. Immutable symlink
aliases of the prepared data are accepted through canonical source paths.
Model loading also validates the disk coordinate config/state contract.

Sampled and complete-cell modes use the unchanged normalized cross loss and
exact replay implementations. Sampled mode has independent galaxy and latent
groups; complete-cell mode integrates every matched row, so only latent
integration remains stochastic. Both visit every NLL row per epoch. Complete
mode visits every response pair per group, while sampled mode records its
with-replacement occurrences explicitly. Tests compare the full combined
NLL/response gradient with a direct graph for both modes, and check exact CPU
uninterrupted/resumed model and optimizer state using float64 raw targets.

Checkpoint metadata records both the immediate disk NLL parent and original
affine NLL ancestor, target precision, input hashes and response-fine-tuning
phase. Committed resume rejects changed checkpoint or raw-data content even
when paths stay the same. All input hashes are checked again on completion.
The existing likelihood/tail screening guards, epoch retention and independent
validation requirements remain in force; input validation does not establish
that a candidate is accurate or that the NLL baseline is useful.

The launcher accepts paired `GRID_INITIAL_MODEL`/`GRID_RAW_ROOT` variables and
optional source/interpreter paths for a frozen checkout. No disk response
trial is launched until the completed NLL baseline and its independent shape
response have been reviewed. Batch, integration and learning-rate settings
must use that new baseline's measured behavior; this implementation adds no
new penalty, target, covariance adjustment or hyperparameter choice.

## Disk-baseline gradient-noise comparison

The external `grid_covariance_disk_gradient_v1/` audit is prepared to measure
a completed, useful NLL-only disk baseline before selecting response-training
settings. It uses the installed frozen response-trainer snapshot and calls
the actual sampled/full-cell replay functions with unchanged weights. A
matching successful full-suite gate and original raw/prepared/checkpoint
hashes are required. There is no optimizer step or new response target.

Eight independent repetitions compare these candidate estimators:

| Mode | Cells | Galaxies/cell/group | Draws/galaxy/group |
|---|---:|---:|---:|
| Sampled reference | 4 | 128 | 16 |
| Previous larger batch | 16 | 1024 | 16 |
| All cells, fewer galaxies | 180 | 128 | 16 |
| All cells, more draws | 180 | 128 | 64 |
| All cells, more galaxies | 180 | 1024 | 16 |
| Complete random cells | 4 | Every matched galaxy | 16 |
| Complete fixed cells | 4 | Every matched galaxy | 16 |

The 180-cell sampled alternatives remove cell-selection variability while
retaining independent galaxy/latent groups. In particular180x128 has23,040
sampled pair occurrences per group, versus16,384 for16x1024: their workloads
are similar enough to test whether spreading the sample over the grid helps.
Neither is a traversal of all6,834,259 training pairs. Fixed-four-cell means
measure conditional latent noise on those cells and have a different expected
gradient from the global objective; their variance cannot simply be subtracted
from, or ranked as interchangeable with, whole-grid estimators.

Separately, eight independently sampled NLL batches of131,072 distinct original
rows use exact chunked gradients. Retained vectors permit joint lambda1/10
gradient noise, norms, raw NLL/response cosines and the fraction exceeding the
existing clip5 reference. These are raw-gradient diagnostics, not predictions
of an Adam update. The common NLL batch is for comparison; actual complete-cell
training assigns its NLL block size from cell grouping. Loss values, exact
cell IDs, group means, pair counts, draw work including replay, runtime and
peak GPU memory are retained with the full gradient vectors.

The noise estimate is the sample trace of gradient covariance. The squared
mean-gradient estimate subtracts its finite-replicate noise bias and may be
negative. Noise trace times mean evaluation time helps compare equal-target
estimators at a fixed averaging cost, but is not a convergence guarantee or
an exact confidence measure. Fresh independent full response validation still
establishes integration stability separately; neither audit creates untouched
validation galaxies or certifies physical size/flux responses.

## Held-out parent error across the grid

`grid_covariance_cell_diagnostic_v1/` decomposes the completed independent
original-NLL parent audit, without new draws or fitting. Explicitly summing
all distinct-group products reproduces the180 cell scores and total7.715945.
Ten cells contribute52.58% of this covariance-weighted score; the brightest
magnitude bin contributes49.24%, and the two largest true-size bins64.23%.
These additive score shares are not shares of unweighted physical error or
gradient variance. The objective and cell weighting are unchanged.

Equal-cell diagonal residuals are(+.11363,+.07601) in the brightest bin and
(-.09461,-.07112) in the faintest. Paired-case SEs are(.02747,.02415) and
(.01116,.01548); MC SEs are much smaller. Opposite errors can cancel in an
overall mean. These marginal summaries retain cross-cell correlations by
averaging deleted-case/group responses before computing uncertainty; they
are distinct from pooled WLS and do not supply an untouched test or exact
confidence bounds.

Conditioning on these integrated **validation** cell scores, uniform sampling
of k distinct cells has mean-score SD
`sqrt((1-k/180)*sample_variance(cell_scores)/k)`:10.513 for k4,5.074 for k16,
and0 for all180. This is the cell-selection contribution to scalar score
noise, not total estimator noise or training-gradient noise. The prepared
disk gradient audit must measure the actual training population separately.

## Sep7 bounded NLL baseline and independent audits

The completed three-epoch disk NLL refit selects epoch3 with validation NLL
-.272396, compared with the original affine parent's-.284046 in the same
standardized physical measurement coordinates. Its cost is.01165045 nat/row.
The initial coordinate-changed density had NLL537.316; successful fitting
must not be confused with preserving that density at initialization. All
13.822M training rows and3.456M validation rows participate in every epoch.
The64-step probes chose LR7e-4 and reset before training; epoch validation
NLLs-.224741,-.223897,-.272396 do not establish convergence. Every training
step clipped at5, with finite gradients throughout.

Epoch3 is frozen for independent full shape-response/support validation and
the prepared gradient-noise comparison. These jobs contain no response
optimizer updates. The likelihood is useful for those next measurements;
shape accuracy, reliable response fine-tuning and physical size/flux response
accuracy remain to be demonstrated. Details and job provenance are recorded
in WORKLOG cont490 and the02:48Z review artifact.

## Grid membership and context sufficiency

The original parent's serialized inputs are e1/e2_input_p, sersic_n_input_p,
true r_input_p/Re_input_p and nbr_flux_near/far/max, plus missing indicators.
The grid also uses baseline r_blend, which is absent from these inputs.
This observation alone does **not** establish an incompatible response target:
inputs can correlate with grid membership, and distinct input distributions
can support distinct cell means without an explicit cell identifier.

Archived SBSI source at commit9439760 records r_blend as a keyed sum of
BlendEMU predicted per-neighbor blending responses. The near/far/max inputs
instead summarize neighbor flux in0–3 and3–7 arcsec shells and the brightest
neighbor within7 arcsec. The latter recipe omits neighbor profiles and exact
separations within each shell. Thus r_blend is an emulator-derived scene
summary, not a direct measured cell response or a necessarily monotonic flux
crowding measure. Current BlendEMU source uses neighbor profiles and distance
in its regression path; it does not verify the historical emulator's weights.

This is source provenance, not bitwise reconstruction of the actual r_blend
column. The archived first40-case lookup job requests the main simulation's
g.05 input catalogue, the40–199 extension requests g0, and the finalizer
combines their lookup files. Historical model/lookup hashes and execution
have not been recovered. No catalogue values were regenerated or changed.
Copies and hashes of seven archival files, five current source files and the
original parent feature metadata are retained in
`grid_covariance_context_scope_v1/results.json` beside the analysis script.

There is a possible structural source of density-response tension when the
cell distinguishes hidden environments at fixed flow inputs. A bounded toy
uses equally represented hidden cohorts with identical visible contexts at
each shear, responses.4I and.8I, and independent zero-mean noise supported
inside a disk of radius.1. At |g|=.05, all shapes have radius at most.14.
The exact marginal conditional density has mean response.6I in both cohorts,
so even a perfect marginal likelihood cannot match both cohort means. For
two cells, four components and illustrative fixed V_B=I, the normalized
quadratic has minimum.01. Scaling the illustrative V_B to.0004I gives25;
neither covariance is an estimate from our catalogue. The script verifies
the optimum and actual finite-forward WLS, and checks that equal cohort
responses have zero obstruction despite the omitted cohort identifier.

This toy is not an actual-catalogue lower bound or a diagnosis of the learned
model. Conditional mean sufficiency for the cell and paired population would
remove this particular issue; usable-leg NLL and matched-pair selection also
need distinction. Keep the current inputs, grid, fixed covariance and loss.
First review the independent disk response and training-gradient audits. If
density-response tension persists after measured tuning, investigate missing
scene information with a controlled test before attributing it to optimizer
noise or changing any model context. No emulator correction or substituted
response target follows from this observation.

## Bounded continuation of the disk NLL baseline

Epoch3's NLL improvement over epoch2 (.04850 nat/row) and remaining cost
relative to the affine reference (.01165) motivate a bounded continuation
without changing optimizer settings. The external
`grid_covariance_disk_nll_continue_v1/` resumes its immutable committed epoch3
state through epoch15, preserving AdamW LR7e-4, zero weight decay, clip5,
batch8192 and every original raw64 training/validation row. The original
learning-rate probes are inherited, not rerun. Each new epoch and optimizer
state is retained; only NLL enters this phase.

The external wrapper pins its launch sources and seed snapshots, requires
the original successful frozen-source gate, verifies exact checkpoint and
optimizer lineage, refuses to overwrite an existing output directory, and
checks full-population counts and source integrity at completion. The
original three-epoch run and the independently audited epoch3 snapshot are
unchanged. The continuation is capped at12 additional epochs/30 minutes on
one A40 and does not assert convergence. A lower NLL on the reused validation
cases cannot automatically replace the response-training baseline: every
new proposed checkpoint still needs independent full shape-response
integration and an explicit comparison of likelihood and cell errors.

## Paired comparisons between checkpoints

`grid_covariance_checkpoint_comparison_v1/compare.py` compares two completed
independent checkpoint audits using their small per-case sufficient-statistic
arrays. It verifies identical original shear normal matrices, measurement
cross statistics, cell counts and targets, and pins the completed review and
all inputs. Case labels come from the common prepared population; when saved
by a producer, its labels must match exactly. The disk audit's original-case
NPZ omits labels, so matching full case arrays and the verified shared source
ordering are required instead. This does not read full per-row draw arrays.

For checkpoint change, subtract the models' raw cross statistics and refit
both the full and each deleted-case shear design. Its paired-case covariance
therefore preserves common variation, rather than treating models evaluated
on the same cases as statistically independent. Each model's residual against
the original measurements is also retained. For independent model latent
banks, the MC covariance of the response difference is the sum of the two
integration covariances. The tool requires nonoverlapping complete seed
intervals for these audit samplers. Case covariance still contains finite MC;
the separate MC term is not subtracted from it.

All four components are reported for every cell, fixed magnitude/size/r_blend
marginal bins, the equal-cell mean and the unit-pair global WLS. Marginal
uncertainties average complete case/group vectors before computing covariance,
preserving cross-cell correlations. Global WLS instead pools raw sufficient
statistics before fitting. Reports include both models' residuals and RMS,
the signed response change, paired-case/MC uncertainty and descriptive
normalized cross-score changes. No exact significance, multiple-comparison
certificate or automatic candidate promotion is inferred.

Eight focused tests cover cancellation of common case variation, direct
deleted-case fits, independent MC covariance with correlated cells, unequal
cell/global estimands, local improvements hidden by global cancellation, an
independent cross-score identity, input alignment and the full synthetic
artifact CLI including overlapping-seed rejection. Actual model inputs remain
unset until the scheduled terminal review. NLL/support summaries accompany
the comparison; fresh draws do not create untouched validation galaxies.

## Per-epoch literal disk support diagnostics

The accumulated trainer now records literal squared-radius support on the
same validation draws already used for cell means. A square root can round
to1 even when squared radius is slightly above1, so the previous amplitude>1
counter alone did not describe all boundary events. Reports retain that
counter and additionally count r2>=1, exterior/boundary values within the
descriptive16eps64 band, larger violations and interior values within the
same band. The maximum squared radius and band width are saved in every
epoch's history/checkpoint metadata and validation NPZ.

These are observations, not clipping, contraction or output rejection rules.
Band membership does not prove that an event is caused by rounding. The
existing likelihood and |e|>2 screening guards, loss, cell weighting and
optimizer behavior are unchanged. A focused example e=(1,2**-26) verifies
that r2>1 can coexist with sqrt(r2)==1 in float64. Tests also verify sample
immutability, unchanged full validation means/loss, finite-value rejection
and exact sampled/full resumed optimizer behavior. The independent audit's
existing support definition is retained; its frozen sources are untouched.

## Sep7 independent disk outcome and all-cell response pilots

The 03:48Z terminal review verifies completed independent epoch-3 integration,
the frozen gradient audit, the NLL continuation and the updated support gate.
The gate passes 383 tests with two skips. Complete evidence and hashes are in
`flow_response_goal_v1/review_20260907T0348Z.json`; the paired checkpoint report
is `grid_covariance_checkpoint_comparison_v1/parent_vs_disk_epoch003.json`.

Disk epoch 3 has validation NLL -0.272396 and cross score 7.839662, compared
with -0.284046 and 7.715945 for the original affine NLL parent. Its equal-cell
component RMS residuals are (0.20034, 0.06762, 0.07704, 0.19833). Independent
integration is precise (cross-score MC jackknife SE 0.001116), but the local
response discrepancies persist. The brightest magnitude marginal retains
diagonal residuals (+0.12082, +0.08979), and the faintest has
(-0.10259, -0.05379). These are equal-cell marginal fits, not global WLS.

Among 874,933,248 shape draws, none has squared radius above the descriptive
16-epsilon float64 band. There are 3,033 literal boundary/exterior events
inside that band, including 330 with amplitude greater than one; maximum
squared radius is 1.0000000000000007. No draw was modified or discarded, and
band membership does not prove rounding for each event. Material shape
excursions are absent in this audit, while engineered magnitude/log-radius
extrema remain very large. Physical size/flux moment stability is unverified.

The fixed-weight gradient measurements support spreading the sampled pairs
over all cells. Eight repetitions give the following response-gradient
noise/mean ratios and mean evaluation times (excluding the NLL gradient):

| Sampled cells x galaxies x draws | Noise/mean | Seconds |
|---|---:|---:|
| 16 x 1024 x 16 | 1.154 | 1.99 |
| 180 x 128 x 16 | 0.271 | 2.81 |
| 180 x 128 x 64 | 0.258 | 9.74 |
| 180 x 1024 x 16 | 0.132 | 22.43 |

All 180 x 128 x 16 removes cell-selection variation at modest additional
work compared with the previous larger batch. More draws produce little
measured benefit; 1024 galaxies per cell reduces absolute variance but has
worse variance times runtime than 128. These eight-replicate diagnostics
do not establish an optimum at future weights. The very low noise for four
fixed complete cells describes a different conditional gradient.

The bounded lambda=1 and lambda=10 pilots under
`grid_covariance_disk_response_pilot_v1/` therefore use all 180 cells,
128 sampled galaxies per independent group and 16 antithetic draws. Both
start the independently audited NLL-only epoch-3 snapshot. Effective NLL
batch 131072, LR 1e-5 and gradient clip 5 are retained to isolate the
representation and cell-batching changes. The exact normalized loss, fixed
simulation covariance, cell weights and physical response extraction stay
unchanged. Every epoch covers all 13,821,950 NLL rows in 106 updates; response
coverage is 2,442,240 replacement pair occurrences per group, not a traversal
of all 6,834,259 pairs. Each pilot has at most three epochs and a two-hour
GPU allocation. Every epoch/optimizer state and full validation integration
(64 draws x four groups) are retained. Existing parent-relative NLL/tail
stopping guards remain diagnostic screens, not accuracy thresholds.

Separately, the completed NLL-only continuation reaches its best NLL at
epoch 12, -0.295882, improving on the original affine parent by 0.011835
nat/row. Validation oscillates across epochs 4-15, so convergence is not
established. The frozen epoch-12 checkpoint receives its own independent
audit in `grid_covariance_disk_nll12_validation_v1/`, with bank 350000000001
and unchanged full-population extraction. It is not automatically substituted
into the epoch-3 response pilots. Future selection requires independent
candidate integration, paired case residuals, grid/marginal accuracy and
likelihood/support assessment. The reused validation cases, finite-MC
uncertainty limits and outstanding physical size/flux goal remain explicit.

## Sep7 disk pilot likelihood cost and short-prefix rate diagnostics

The supplied 04:48Z terminal snapshot records successful execution of both
all-cell pilots and the independent NLL epoch-12 audit. Both pilots fail
the scientific NLL guard at epoch 1 and retain epoch 0 as their best eligible
pointer. Completed evidence and hashes are in
`flow_response_goal_v1/review_20260907T0448Z.json`.

| Model | NLL | Response cross score | NLL increase from disk epoch 3 |
|---|---:|---:|---:|
| Disk NLL epoch 3, pilot bank | -0.272396 | 7.841637 | 0 |
| Lambda 1, epoch 1 | +0.043274 | 1.472239 | +0.315670 |
| Lambda 10, epoch 1 | +0.400578 | 1.460907 | +0.672974 |

Cross-score MC jackknife SE is 0.000674/0.000609 for the two candidates;
the naive quadratic and cross score differ by only about 0.00005. No shape
draw exceeds the descriptive 16-epsilon squared-radius band. This evidence
separates the observed likelihood cost from the earlier enormous shape tails
and erratic integration. It does not certify the component responses or prove
that every remaining optimization issue is deterministic. A lower combined
objective can still incur an unacceptable likelihood cost at the agreed
lambda. Neither rejected checkpoint is promoted or automatically resumed.

Independent NLL epoch 12 has NLL -0.295882 and cross score 7.813705
(diagnostic MC SE 0.004059). Its full shape-component RMS residuals are
(0.20440, 0.06733, 0.07756, 0.21496); global diagonal residuals are
(-0.01523, -0.01816). The paired epoch-3/12 comparison verifies common
original case statistics and disjoint MC banks. The better NLL accompanies
mixed local response changes, including worse response for the smallest
input-size marginal. Actual magnitude/log-radius extrema remain extreme,
so physical moment stability is still unverified.

`grid_covariance_disk_rate_probe_v1/` now measures shorter optimizer
trajectories from the independently audited NLL-only epoch 12. For each
lambda, separate rates 1e-6, 3e-6 and 1e-5 reset both model and AdamW state.
These bracket the failed rate without changing the loss, covariance or
training batch: all 180 cells, 128 galaxies per independent group, 16 draws,
NLL 131072, exact replay chunks 4096/1024 and clip 5. The update loop is
tested against production for exact weights and optimizer state at both
lambdas, including its NLL remainder and mode changes during validation.

Each rate uses at most 64 steps from the original full-catalogue shuffled
epoch order, covering at most 8,388,608 NLL rows and 1,474,560 replacement
pair occurrences per group. This is explicitly a prefix, not a full epoch.
All validation NLL rows are checked every four updates; the exact model and
optimizer are retained, including a failing guard endpoint, and the recipe
stops at its first observed NLL/tail failure. The parent-relative NLL allowance
remains 0.05 nat/row. Full response-pair integration at steps 0, 4, 16, 32,
64 and any guard endpoint uses 16 draws x four groups, with MC diagnostics
and literal support counters. The lower validation draw count is an explicit
integration-cost choice for rate diagnostics, not a modified response target
or a final accuracy check. Rates/lambdas share their banks for comparison;
their MC errors must not be treated as independent.

Every prefix checkpoint is labeled as a rate probe, with no best pointer or
automatic promotion/resumption. A useful candidate requires fresh independent
64-draw integration and the paired case/marginal checks before interpretation.
Separately, `grid_covariance_disk_pilot_validation_v1/` independently audits
the rejected lambda-1 epoch-1 model using bank 360000000001. Its purpose is
to locate which full-matrix/cell responses produced the score reduction;
the likelihood rejection remains in force. All follow-ups use frozen source
and checkpoint manifests, and await the next scheduled terminal review.

## Sep7 independent localization of the rejected disk response

The completed fresh-bank audit of lambda-1 epoch 1 reproduces its lower score:
1.471297 versus 1.472239 in training validation, with diagnostic MC jackknife
SE 0.001054. NLL remains +0.043274, so the existing rejection is unchanged.
The paired comparison to its disk NLL epoch-3 parent is saved in
`grid_covariance_checkpoint_comparison_v1/disk_epoch003_vs_rejected_lambda1.json`.
Original case statistics, all four components and disjoint complete MC seed
intervals are verified. The comparison reads completed small case arrays;
it does not access the active rate probes.

The covariance-weighted score falls by 81.23%, with lower scores in 150 of
180 cells. This is not an 81% reduction of every response error. Component
RMS changes from (0.20034, 0.06762, 0.07704, 0.19833) to
(0.14080, 0.06729, 0.07662, 0.15138): diagonal RMS improves by 29.72% and
23.67%, while off-diagonal RMS changes by less than 0.6%. The brightest
magnitude bin supplies 57.67% of the net score reduction, the largest two
input-size bins supply 72.34%, and the ten largest cell improvements supply
61.65%. These are overlapping descriptive decompositions, not independent
effects, component tests or new training weights.

Substantial biases remain. The smallest input-size marginal has diagonal
residuals (-0.22035, -0.22845), with paired-case SE (0.02413, 0.02877) and
MC SE (0.000239, 0.000458). Unit-pair global diagonal residuals worsen from
(-0.01432, +0.00285) to (-0.03258, -0.02894). Full covariance weighting,
unweighted cell RMS and pooled global response measure different aspects
of the fit; neither one replaces the others as an accuracy check.

No shape draw lies above the descriptive boundary band in this audit, but
the maximum engineered log-radius is 993.167. Under the existing physical
inverse r=exp(log-radius), that draw exceeds the largest finite float64
radius (whose logarithm is about 709.783). This establishes a concrete
physical precision problem, not an infinite population moment, and does
not alter any draw. Scalar moment stability remains outstanding.
`grid_covariance_disk_response_localization_v1/` retains the source-hashed
small-array calculation and checks reproduction of full and marginal scores.

The watcher now respects the later of its saved next-check time and a newly
configured first-check time. Previously a saved snapshot could override a
new delay, which caused the supplied 06:48Z check despite a 07:48Z setting.
Mocked tests check that a delayed configuration causes neither a scheduler
query nor a state write, and that the exact due boundary allows one check.
The rate jobs and their frozen sources are unchanged; their next review
remains 07:48Z. No new training decision precedes their completed evidence.

## Owner-directed balance run without an NLL ceiling

On September7, the owner requested training toward convergence with a balanced
NLL/response objective, suggesting lambda0.5, and explicitly removed the NLL
threshold stop. `--disable-nll-guard` in the accumulated trainer implements
this opt-in behavior; legacy default recipes retain their previous guards.
The historical NLL-threshold comparison remains a diagnostic field, paired
with `nll_guard_enforced=false`; it does not affect checkpoint eligibility or
termination for this new recipe. Finite-value checks and the existing tail
guard remain. The response formula, covariance, targets and weights do not
change; only lambda and the authorized stopping policy change.

`grid_covariance_balanced_convergence_v1` starts from the completed NLL-only
disk epoch12, not the response-trained step12. It uses lambda0.5, LR1e-5,
all180cells with128 replacement galaxies/group and16 antithetic draws,
NLL131072 per update, clip5 and the same exact replay. Every epoch visits
all13,821,950 NLL rows in106updates but samples2,442,240 response pair
occurrences/group, not every matched pair. Full validation uses all3,455,704
NLL rows and1,708,854 pairs with64draws x4groups and a fixed MC bank.
Checkpoint selection and15-epoch patience use NLL+0.5*L_response only.
ConstGold remains diagnostic and cannot choose a checkpoint or lambda.

Bounded stages cap at20/40/60 epochs, preserving exact optimizer resume;
later stages skip if patience already fired. A cap is not convergence, and
patience is an operational validation plateau rather than proof of a stationary
optimizer or response accuracy. Further rate adjustment and independent MC/
cell-response audits may be necessary. Every epoch and optimizer state is
retained. The full scientific goal remains unachieved. Scheduled wake-ups
remain disabled at the owner's request.

## Capped-data comparison alongside the full-data runs

The owner authorized separate lambda0.5/0.1 capped runs while leaving the
full-data runs untouched. Each epoch draws4,000,000 distinct NLL rows from
the complete training catalogue, with a fresh deterministic permutation per
epoch, giving31 optimizer updates at effective batch131072 (last67840).
Response sampling uses exactly2,000,000 with-replacement pair occurrences
**per independent group** per epoch. Step quotas differ by at most1; within
all180 cells, each step samples358 or359 galaxies/cell/group. Cell order is
randomized before allocating remainders, and each cell mean uses its actual
count. The cross loss still averages cells/components equally with the
original training target and fixed V_B. Pair counts are not latent draws,
unique-pair coverage or replay evaluations. Antithetic draws16, LR1e-5,
clip5, exact replay, NLL-only ancestry and no NLL-ceiling stop are unchanged.

`sbsi/flow_grid_caps.py` chooses a fixed random500,000-pair validation
subset without replacement. Original raw64 measured shapes and actual shear
vectors refit the subset's measured WLS target and matching model design.
The original full-validation precision matrices remain fixed metric weights;
the subset's own measured covariance is retained separately and is not silently
substituted as a training weight. Fixed500,000 NLL validation rows are chosen
independently. Both subsets and refit statistics are saved and verified on
resume. This is a changed validation population, so its total score is not
an apples-to-apples comparison with the full-data runs' scores. Final
candidates require a common full-validation evaluation. Full-population
precision weights are not confidence bounds for the subset residual.

The same validation subsets, training sampling seeds and64draw/four-group
MC bank are used for both capped lambdas. Cross-run MC errors are correlated;
fresh independent integration is still needed. These data caps reduce work
per epoch, so epochs are not equal compute or update budgets versus the
full-data runs. All counters are retained. Caps20/40/60 and15-epoch combined
objective patience remain bounded operational limits, not certificates of
convergence. Roots are `grid_covariance_capped_balanced_v1` (lambda0.5) and
`grid_covariance_capped_lambda01_v1` (lambda0.1). Scheduled wake-ups stay off.

## Physical-response implementation and initialization gate (Sep9)

`sbsi/flow_physical_response.py` converts each draw from `(e1,e2,mag,ln r)`
to `(e1,e2,r,F)` with `r=exp(ln r)` in catalogue flux-radius pixels and
`F=exp[0.4 ln(10)(30-mag)]` in instrumental flux units. No output repairs.
The initial proposed scalar target uses the fixed baseline intrinsic shape
frame; exactly circular inputs use sky axes. Existing full sky-frame shape
response remains four entries. Joint delete-case fits concatenate four shape
and four scalar entries, retaining cross-covariance. Algebraic diagonal
equilibration before inversion improves units conditioning without a floor or
changing the covariance-weighted quadratic. The extended normalization is
`NLL + lambda/(2*180*8) sum_B residual_A.T V8_B^-1 residual_C`.

The owner requested lambda0.1 and1 retraining. Current implementation supplies
physical conversion/target covariance only; production trainer is still
four-component and has NOT been claimed to train this extension. Job16399325
under grid_physical_response_gate_v1 runs frozen full tests, original full
train/validation target construction, and a500k-pair64draw/two-group NLL-only
epoch12 initialization audit. Input population/transforms remain fixed.
Independent latent groups and paired CRN are preserved. Finite raw magnitude
and log-radius do not imply numerically stable physical moment integration;
prior extreme draws motivate this gate before selecting a viable physical
response training path. No NLL ceiling is reintroduced. Training weights0.1/1
remain requested/pending, not launched. User was asked optional frame preference;
recommended baseline frame used for this diagnostic after time to respond.

## Standardized physical-coordinate NLL trial (Sep9)

After the initial disk NLL model produced physical size up to1.3e138pixels and
flux2.0e107, the owner requested standardized `(e1,e2,r,F)` targets.
`ConditionalPhysicalDiskFlow` uses the existing radial-tanh shape disk and
positive scalar map `p=s*softplus(a*w+b)`, then outputs `(p-mu)/sigma`.
All scalar units, coordinate means/scales and target means/scales are computed
from original training rows only. At large auxiliary values physical outputs
grow linearly, avoiding post-sample exponential magnification. This is a
proper bijective density map/Jacobian, not clipping or calibration. Numerical
rounding to a boundary is still audited. Flux is the positive flux defined by
MAG_AUTO at zeropoint30; radius is measured_flux_radius in pixels. It does not
claim a density for arbitrary signed aperture flux measurements.

Checkpoint type `physical_disk_affine` explicitly stores physical output
names/buffers; legacy flows reject stray physical settings. Training starts
from the original NLL-only disk epoch12 base weights with changed coordinates
and reset optimizer. This changes initial density and requires NLL refitting.
The trial runs three64-step rate probes, resets, then25capped4Mrow epochs with
8192row updates, LRselected by fixed500kvalidationNLL, clip5 and no response
penalty. Everyepoch/checkpoint and optimizer state are retained; no assertion
of convergence or tested resume API for this new external loop.

Common NLL comparison uses `NLL_oldref=NLL_newstandard-log|dznew/dzold|`,
where `logJ=sum(log old_scales)-sum(log new_scales)+lnr+lnF+ln(.4ln10)`;
the scalar coordinate permutation uses the absolute determinant. Tests check
this against a full autograd Jacobian. Final audit includes full validation NLL
and four independent64draw groups on500kfixed validation contexts to inspect
physical moments. It is not yet an eight-component shear response audit.

Root grid_physical_coordinates_nll_v1 freezes current source and test resources,
excludes generated caches from provenance and requires fullsuite before CUDA
training. Requestedlambda0.1/1response runs remain subsequent work after a
useful physical NLL baseline and tested dimension-general response trainer.

## Overnight physical response and ConstGold pipeline (Sep10)

The owner authorizes autonomous NLL refinement, joint physical-response training,
and ConstGold validation including detection and selection. Hourly quiet wakeups
were subsequently re-enabled by the owner.
The refined NLL initialization starts from physical epoch21, with the same 4M
training / fixed500k validation NLL caps, LR1e-4, plateau reductions by1/2 after
five non-improving checks, minimum1e-6, at most80 epochs. Optimizer is reset and
this is explicitly a new refinement stage, not exact optimizer continuation.
Its source remains the frozen original physical-coordinate trial.

`PhysicalResponsePopulation` fits the already constructed sky-frame shape and
fixed baseline intrinsic-frame radius/flux response designs. Eight flattened
components use the full fixed simulation case covariance:

\[
L=L_{\rm NLL}+\frac{\lambda}{2\,180\,8}
\sum_B (R_{A,B}-\widehat R_B)^T V_B^{-1}(R_{C,B}-\widehat R_B).
\]

The four-component reduction remains unchanged for older models. Sampled and
full-cell replay now infer the component count from the target; the joint
physical trainer uses sampled mode. Each epoch uses 4M distinct NLL rows and
2M replacement response-pair occurrences per independent group,31updates,
all180cells each update,16 antithetic draws and paired CRN. Validation uses
fixed500k NLL rows and500k pairs,64draws x4groups, subset WLS targets with full
validation covariance. Lambda0.1 and1 each start from the completed refined
NLL parent with fresh AdamW LR1e-5/clip5. There is no NLL ceiling. Development
validation total controls checkpoint retention/plateau LR reductions;60epochs
and8hours are resource bounds, not convergence certificates. All epochs and
optimizer states are saved; arbitrary interrupted resume is not exposed.
Final best checkpoints receive a fresh full-pair64x4 integration with banks
independent of checkpoint-selection validation. This alone provides no exact
case confidence bound, and no new validation galaxies.

`forward_physical` returns the radial-tanh shapes and softplus radius/flux
before standardization. It avoids subtract/add cancellation without clipping
or contracting values. `sample_physical` and bundle sampling use this direct
map for physical models; the existing standardized density/Jacobians remain.
Softplus underflow or materially exterior shape draws halt response jobs with
the checkpoint retained. Near-boundary shape counts remain observational.

ConstGold is evaluated with the frozen trial9 blend lookup and existing
transition-aware detector, preserving their provenance. The evaluation adapter
converts positive physical samples to MAG_AUTO/log-radius solely for existing
cuts/plot units. The same cuts apply on simulation/model sides. Fifty cases40–89
use both-detected no-cut shape controls, per-leg measured selection, detection
weighting and complete detection-plus-selection statistics, with the existing
case bootstrap. These are g1 ±.02 tests; full four-entry shape response is tested
on random-direction validation. Model choice uses development validation only.
ConstGold is already studied diagnostic data, not an untouched test. The
physical-coordinate model is not promoted into an inference preset; integration
of physical target names into general likelihood cuts/normalization and direct
score-identity checks remains part of the unachieved likelihood goal.

### Joint physical paired-case uncertainty

`flow_physical_case_validation.paired_physical_case_statistics` accepts raw
case/cell sufficient statistics with separate sky and intrinsic-frame normal
matrices. Both full and deleted-case fits are refit on both model/simulation
sides. The eight-component residual jackknife is concatenated BEFORE computing
covariance, preserving shape/scalar cross correlations. Four or more independent
model integration groups separately estimate MC covariance and the unbiased
cross objective. The measured-case covariance can be checked against the fixed
training/validation covariance; the paired residual covariance does not replace
that training covariance.

Fixed-cell or marginal mean uncertainty is computed from weighted deleted-case
responses and weighted group responses, retaining correlations across cells.
Unit-pair global WLS instead requires pooling raw cross/normal statistics before
refitting both designs. The implementation does not deconvolve finite MC from
case uncertainty or produce exact confidence certification. It is prepared for
an independent checkpoint audit and is not wired into the source-frozen jobs.

The independent physical checkpoint audit now accumulates raw measured/model
case cross statistics on all1,708,854 validation pairs,64draws x4 independent
groups, with distinct banks for NLL/lambda0.1/lambda1. It verifies the original
measured full response and joint covariance against saved physical targets,
then retains raw case normals/crosses/counts, paired residuals, separate MC
covariances, equal-cell means and pooled unit-pair WLS. It depends on completed
model training and its own source test gate. This is still reused development
validation; no exact significance or convergence assertion follows merely from
a small residual, a completed audit, or successful reproduction of targets.

## Physical small-step pilots and epoch seed separation (Sep10)

The LR1e-7 pilots from physical NLL-only epoch123 completed three capped epochs.
They preserved full-reference NLL within0.00057nat/row of the parent, but changed
shape residuals only slightly. Fixed500k cross scores fell319.47 to315.61;
independent full scores were319.31 and318.66 for lambda0.1 and1. These use
different populations and MC banks and cannot be directly subtracted as a paired
improvement. Physical flux residual RMS increased on the fixed subset. No
response accuracy or physical moment convergence is established.

The external launcher used epoch seed501+epoch*1000003, while production
accumulated_epoch also adds step*1000003. Adjacent epochs consequently share30
of31 response-plan seeds, and corresponding latent chunk seeds repeat. NLL
permutations still differ; repeated response seeds do not mean identical full
optimizer batches. This does not change the algebraic loss but weakens the
interpretation of smooth cross-epoch progress with fresh stochastic samples.

physical_response_epoch_seed_v1 uses epoch seed1200000000501+epoch*1000000000,
preserving the production step/chunk allocation and the exact loss. Integer
allocation checks cover all12epochs,31steps and64chunks per group (47,988 distinct
plan/latent seeds). Disjoint seeds are not a proof of PRNG stream independence.
Both lambda runs restart explicitly from the same NLL-only epoch123 with fresh
AdamW at1e-7, up to12epochs; these are not optimizer continuations of the pilots.
The caps, fixed validation bank, support checks, clipping of gradients at5 and
absence of an NLL ceiling remain unchanged. Outputs/data are never clipped.
Independent paired-case audits compare the completed parent and lambda0.1 pilot
using new banks and the original covariance/physical response definitions.

## Joint covariance tradeoff after twelve physical epochs (Sep10)

`physical_covariance_tradeoff_v1/analyze.py` decomposes the existing eight-entry
cross objective using completion of the square, without changing the precision,
targets or training loss. Partition residual into a=(shape,radius) and f=flux:

\[
 r^TP r = a^T(P_{aa}-P_{af}P_{ff}^{-1}P_{fa})a
 +(f+P_{ff}^{-1}P_{fa}a)^TP_{ff}(f+P_{ff}^{-1}P_{fa}a).
\]

Apply the same split to shape/radius and to each ordered independent-group
cross product, retaining the original normalization1/(2*180*8). The resulting
three terms are sequential conditional terms, not separate marginal penalties;
finite-MC cross terms can be negative. Synthetic exact identities and invariance
under diagonal changes of units pass; terms reproduce every saved total.

On fixed500k validation, lambda0.1 goes319.4727->297.3267: shape marginal
4.5974->4.5136, radius conditional77.4741->76.3347, flux conditional237.4012->
216.4783. Approximately94.5% of the reduction is in the conditional flux term,
while raw first flux-component residual RMS grows26.26->41.53. Holding the
first six residual components fixed, conditional preferred flux residual is
-Pff^-1 Pfa a; its equal-cell first component is approximately-142 in physical
flux-response units. This is a diagnostic of covariance coupling at the current
wrong radius response, not a replacement target, correction or proof of the
trained network's mechanism. The full quadratic is still minimized at zero
joint residual when those entries can move freely.

Independent paired-case audit of NLL123 and the earlier lambda0.1epoch3 gives
equal-cell radius first response-.96668/-.96098 against measured+.89380.
The radius mismatch remains large. New twelve-epoch endpoints need fresh paired
case/component audits before any claim that a reduced aggregate score improves
physical response accuracy. No ConstGold-dependent tuning or covariance change
follows from this decomposition.

## Independent physical endpoints and intermediate learning rate (Sep10)

Fresh full paired-case audits of the twelve-epoch endpoints yield joint cross
309.72946 (lambda0.1) and308.75191 (lambda1). Equal-cell radius first response
is-.94780/-.94800 versus measured+.89380; NLL123 was-.96668. Thus approximately
1% of the initial radius mismatch was corrected. Full flux component RMS is
25.73/10.30 for lambda0.1 and22.45/23.65 for lambda1, versus NLL12320.85/9.08.
These are independent MC banks on reused development cases, not untouched tests.
Neither endpoint is promoted; aggregate score reduction does not establish
component accuracy, finite-moment convergence or useful selection response.

physical_response_midrate_v1 tests only lambda0.1 with initial LR1e-6, between
slow1e-7 and earlier failed1e-5. It restarts unchanged pureNLL123 with fresh
AdamW, preserving the twelve-epoch cap,4M NLL/2M replacement response occurrences
per group, all180cells,16draws, fixed500k64x4 validation, gradient clip5 and no
NLL ceiling. Training/selection seeds match the corrected1e-7 experiment for a
controlled rate comparison; MC errors of those fixed-bank curves are correlated.
The final full-pair integration uses fresh bank1600000000001. This is a bounded
rate experiment, not an assertion that faster training will solve the radius/
flux tradeoff. No covariance, targets, output maps or penalties are changed.

## Physical response weight balance after V100 midrate trial (Sep10)

The lambda0.1/LR1e-6 V100 trial completed twelve epochs, but its selected endpoint
is not promoted: fixed500k cross fell to-188.37 while independent full cross was
226.42 with diagnostic MC jackknife SE232.98 and integration quadratic347.90.
Full old-reference NLL-.22920 costs.12479nat/row versus NLL123. Flux component
RMS on fixed validation grew to497.40/256.35, with full sample maximum9.72e8.
Successful execution and positive finite draws do not establish moment stability.
GPU changed from earlier A40 runs, so common integer seeds do not guarantee
identical draws across those comparisons.

physical_response_balance_v1 brackets smaller lambda0.001/0.01 at LR1e-6,
retaining the original eight-component fixed covariance loss and NLL123 parent.
The completed physical NLL79 gradient audit at359pairs/cell measured mean NLL
norm14.50 versus response9978.83 (eight replicates): lambda0.001 puts the mean
raw norms on a similar scale, while0.01 remains response-dominant. This is a
recipe hypothesis, not a prediction of Adam steps or exact balance at NLL123.
Same twelve-epoch cap,4M NLL/2M replacement pairs per group, fixed500k64x4
validation and V100 as the midrate run; final full banks1710000000001 and
1800000000001. All epochs are retained; no NLL ceiling or new penalty. The
six-component marginal-covariance proposal discussed in cont558 is not adopted
by these runs; flux remains in both the NLL and response objective.

## Marginal shape-radius objective and component-gradient balance (Sep10)

The new opt-in physical objective retains the six responses
`(e1/g1,e1/g2,e2/g1,e2/g2,r_parallel,r_cross)` and removes both flux responses.
Flux remains a modeled NLL output and a sampled support/response diagnostic.
Let `V_er` be the top-left6x6 marginal block of the measured eight-response
covariance. Its precision is recomputed as `inverse(V_er)`; using the top-left
block of `inverse(V_8)` would instead condition on the omitted flux responses
and is explicitly rejected by tests. Shape-radius off-diagonal covariance is
retained.

For diagonal relative component weights `W=diag(1,1,1,1,w_r,w_r)`, the loss is

\[
 L=L_{\rm NLL}+\frac{\lambda_e}{2\,180\,6}
 \sum_B r_{A,B}^T\sqrt W\,V_{er,B}^{-1}\sqrt W\,r_{C,B}.
\]

The square-root sandwich keeps the metric symmetric: diagonal shape terms have
effective weight `lambda_e`, diagonal radius terms have `lambda_e*w_r`, and
shape-radius cross terms have the geometric-mean weight. The experiment fixes
`lambda_e=0.1`; `w_r` is not copied from the approximate residual ratio.
Instead a frozen NLL123 audit replays identical cells, pairs and latent seeds,
decomposes the parameter gradient into shape-output and radius-output paths,
and solves for equal Euclidean norms of their repeat-mean gradients under this
joint metric. Leave-one-repeat-out weights must all remain within a factor two
of the full estimate before the dependent training job can start. This balances
raw parameter-gradient norms at the parent; it does not guarantee equal Adam
updates, equal convergence rates or equal final residuals.

Implementation is in `flow_physical_response.marginal_physical_response_statistics`
and `PhysicalResponsePopulation`. Six- and eight-component exact replay share
the same code. Validation accumulates all eight physical responses from the same
draws, so diagnostic flux residuals remain visible without entering the six-
component score. Targeted tests pass14/14 and the full suite passes412/2skipped.
The initial source-frozen `physical_response_shape_radius_v1` audit16431140
stopped in its pytest gate because the launcher used the wrong working directory;
dependent training16431141 did not start and no science output was produced.
After the exact frozen tests passed from the source root, replacement one-V100
audit16440418 and `afterok` training16440419 were submitted with unchanged
science. Training restarts the unchanged NLL123 parent with fresh AdamW/LR1e-6.
The earlier eight-component balance jobs are a separate experiment and were not
touched by this launch. No checkpoint is promoted by submission.
