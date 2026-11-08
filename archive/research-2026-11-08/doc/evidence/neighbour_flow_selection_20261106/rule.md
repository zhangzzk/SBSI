# Selection association on the flow's own data

## Question and frozen endpoints

The first neighbour-conditioned flow improves validation likelihood but changes
constant-shear bias little. Test whether its selection--neighbour association
is already inaccurate on its own training/validation population. Freeze the
original Cartesian NLL and weak-nine flows and the four matched continuation
endpoints. Two additional diagnostic arms set the new standardized neighbour
input to zero (the training mean); all other inputs and latent draws are held
fixed. These interventions are not deployable model predictions or a causal
effect estimate.

Use500,000 training rows sampled without replacement with a declared seed and
the existing500,000 validation NLL rows. Retain both original g0/g05 usable-row
populations, reporting g0 separately as well as their pooled distribution.
No new training, tuning, image simulations, response offsets or test-selected
checkpoints. This is not the full joint constant-shear response.

## Measurements

For each of the nine target cuts, compare measured and generated selection
rates and the selected mean frozen neighbour coefficient. Generate photometry
from the exact photometry factor, with64 portable antithetic draws shared by
all arms and two disjoint32-draw banks. Check this shortcut against full joint
draws before accumulating results. Record case/leg/score-quintile sums and
use whole-case bootstrap errors. Quintile boundaries use the training sample
only. Reference-score changes are diagnostics, not calibrated shear responses.

Also split standardized-target validation NLL into photometry and conditional
shape terms using the existing physical-map Jacobian. Verify the total NLL
against the saved validation probes. Score-at-mean arms are explicitly labelled
interventions. Match selected rows to case/leg identities using the preserved
preparation receipts and validate the cached score column against raw scores.

## Execution

Run one persistent GPU job, loading each unique checkpoint once and processing
both data splits in the same allocation. Four CPUs, one GPU,24GB host memory;
all previous focused evaluations are complete. Keep the older broad array held
and the overall six-GPU limit. Save per-split completion receipts and small
reports; on restart verify receipt hashes and skip only complete matching
outputs. Numerical batch size remains1024 rows, with all seeds in manifest.json.
The same allocation first audits the completed40-case response with paired
axis differences, then performs the new diagnostic. No tiny per-case jobs.
