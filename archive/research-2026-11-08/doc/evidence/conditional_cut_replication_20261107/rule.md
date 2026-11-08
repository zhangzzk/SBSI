# Replicate the fixed conditional cut-probability improvement

## Authorized question and fixed models

On 2026-11-07 the owner requests continuation after the completed four-field
comparison. Evaluate the SAME control and probability-loss step64 endpoints
on the remaining 36 existing constant-shear scenes. Do not train, retune a
weight, select a checkpoint, change cuts or resume broad array17788286.
Checkpoint paths/hashes and the successful original training receipt are
pinned in manifest.json. Keep classifier/emulator identities unchanged.

This is a focused replication of two frozen models, not the old one/nine-cut
hyperparameter array. Report original four, additional thirty-six and all
forty scenes separately. The additional scenes are unused in this particular
four-field experiment, but are existing development scenes used in earlier
studies; they are not fresh independent acceptance data. The first four
remain in their original report and must not be substituted for missing cases.

## Population and numerical contract

Use all forty case IDs already declared in the measured-input study. Evaluate
both coherent shear axes at ±0.02, with the same measured usability, independent
per-leg measured output cuts and source catalogue matching. Model evaluation
keeps original classifier probabilities, conditional flow and full disk
transport. No truth-property cuts are introduced. Report all nine target cuts
(radius >2.8/3.0/3.2 pixels crossed with magnitude <24.9/25.0/25.1) and retain
the original five auxiliary cuts only as diagnostics.

Portable latent seeds remain7301+10000019*case+1000003*block, with64 antithetic
draws and1024-parent blocks, paired across both models and legs. The emulator
tree-cell pooling and2e-14 first-block separate-evaluation check are unchanged.
Keep predicted-usability/generated-selection, actual-usability/generated-
selection and actual-usability/actual-selection modes. The last mode still
uses generated photometry internally; it is not the measured-photometry test.

The original static-neighbour feature is evaluated at3.5px/24.5mag for each
scene. On32 deterministic IDs spread across each parent, independently group
direct pair predictions and compare to the feature with the existing
2e-10 relative/2e-7 absolute feature-replay tolerance. Original four fields
also retain their saved feature check. The split-shear common-usability
association diagnostic exists only for the original four g1 fields; mark it
unavailable elsewhere and never report missing values as zero. This auxiliary
diagnostic is not required for coherent response evaluation.

All new scene inputs, flags, original measured-sum references and axis2
verification receipts exist before submission (280 prerequisite files checked
across40 scenes). Require exact ID/context alignment and report measured
matching receipts. Before accepting each output, replay original simulation
sums and exact actual selected masses. Do not regenerate images or remeasure.

## Gates, reports and decision

The first GPU allocation replays case1000/axis1 with both unchanged models in
the adapted evaluator, then runs the first additional task1001/axis1. Require
exact original selected masses and simulation sums; model response replay uses
the existing1e-6 absolute tolerance. Preserve the actual replay difference.
Both stages must finish successfully before releasing the persistent workers.

The reducer uses the original four saved field outputs and all72 additional
field/axis outputs, without cherry-picking complete subsets. Report central
m, full response matrices, flow/emulator contributions, selected-count ratios,
each target cut, nine-cut RMS, individual cases and paired weighted-minus-control
changes. Both absolute and paired uncertainties use whole-case bootstrap
(10000 draws, original bootstrap seed) and delete-one-case jackknife.
Pool numerators and selected masses before ratios. Preserve correlations across
cuts/models/axes. Reproduce original four-field point estimates and jackknife
SEs before writing the combined report. The numerical reducer is also checked
with an unequal-count fixture that distinguishes pooled ratios from a mean of
case ratios.

A repeatable improvement is useful even if individual components cancel;
the owner's nine-cut target policy applies. A residual near zero on reused
scenes is not independent acceptance. Do not respond to a negative replication
by changing the declared checkpoint, loss weight or reported cohort.

## Packed resources and monitoring

One pilot GPU precedes six persistent one-GPU workers. Each holds both models
and processes11 or12 additional field/axis tasks; output receipts allow restart
by complete case. Four CPUs and32GB host memory per worker; at most six GPUs
across cip/inter and at most24 allocated CPUs. Respect the cip per-user limit
and all other active study jobs. Keep numerical batch sizes unchanged.
A CPU report depends on the pilot and all workers. Source bytes/manifest are
hash pinned, and earlier evidence is unchanged. Monitor pilot startup and
first meaningful progress, then the dependent stage at startup when practical.
The owner-stopped automated follow-up loop remains disabled.
