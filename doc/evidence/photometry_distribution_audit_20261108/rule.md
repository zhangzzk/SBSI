# Locate conditional size/flux distribution errors

## Authorized question and fixed scope

Owner asks to understand predicted versus measured photometry and localize what
is wrong before changing training. Existing factorial/fixed-cell tests establish
incorrect scene weighting, but do not distinguish radius-marginal, flux-marginal
and radius/flux association errors, distribution width/tails, and their context
dependence. This study measures those quantities without fitting any new model.

Freeze original NLL37c4afc13b04, original weak-nine888fb044, neighbour-nine
f6600f8b6bba and latest43673e5d. These span the original problem and later
endpoints. Not all are matched interventions: do not assign every difference
uniquely to one stage, feature or objective. Classifier/emulator remain unchanged;
no new images, shape measurements or constant-shear training targets.

## Data and exact replay

Reuse the exact500000 training and500000 validation rows from the own-data
selection audit, including original g0/g05 usable legs, all160/40 scenes,
indices, standardized contexts and neighbour scores. These are ordinary flow
training-domain rows, not constant-shear evaluation scenes. Existing shape-fit
randomness does not change their size/flux measurements; this audit does not
silently redefine the usable population with new paired shape fits.

Original domain validity/true-population restrictions remain; no new observed
size/flux cuts precede the analysis. Record original prepared metadata hashes,
selected row indices, case and leg. Verify neighbour-score standardization and
reject missing context rather than silently treating imputed values as real.
Reconstruct raw inputs from the pinned preprocessing transform for grouping.

Reuse canonical64 PCG64 antithetic draws,1024-row blocks and original seeds
for probability/photometry statistics. Shared draws across models; independent
32-draw antithetic banks permit MC-unbiased Brier scores and sensitivity checks.
The report must reproduce all existing canonical selected masses and weighted
score numerators for the four checkpoints, both splits. Photo-only generation
must exactly match full-joint output at startup. Persist per-row diagnostics
and restart receipts every64 blocks, with source/manifest/model hashes.

## New distribution diagnostics

For each row/model, save radius-only and flux-only passing probabilities and
all nine joint probabilities; size/magnitude means and16/50/84% draw quantiles;
photometry NLL; measured photometry transformed through the inverse2D flow;
and separate marginal observation ranks.

Ranks use a separately seeded64-IID-normal draw bank, common across models.
Antithetic draws do NOT justify a uniform rank null. With a correct continuous
conditional model, the observation rank among64 IID draws is uniform over
0,...,64; each extreme therefore has probability1/65. Randomize exact ties
with a fixed independent stream and count them. Report mean rank shifts and
both extreme-rank excesses. These test distribution location/dispersion/tails;
quantile estimates from64 draws are noisy and not per-object truths.

The inverse photometry flow's two latent coordinates should jointly follow
independent standard normals when the conditional density is correct. Report
means, second moments, cross moment and radial tail excess beyond the99%
chi-square2 threshold. These coordinates mix physical size/flux: do not label
latent0 a radius error or latent1 a flux error. Marginal ranks provide the
physical-coordinate diagnostics.

Compute magnitude stably from log-softplus coordinates even when physical
flux rounds to zero. Below u=-36, log(softplus(u))=u to float64 relative
roundoff. Record underflows; do not clip away, truncate, or drop faint tails.
Preserve arithmetic means alongside ranks/quantiles so rare-output sensitivity
is visible rather than hidden by robust summaries.

## Exact joint-cut decomposition

Within each specified population, let R and F denote the radius/flux passing
indicators. For model and observed distributions, define q_R,q_F,q_RF and
C=q_RF-q_R*q_F. Then model-minus-observed joint error decomposes exactly as

    delta q_RF = mean(q_F_model,q_F_observed) * delta q_R
               + mean(q_R_model,q_R_observed) * delta q_F
               + delta C.

Report these radius, flux and association terms separately at all nine cuts.
The association term includes remaining variation in context within a group;
it is NOT an identified per-object conditional copula error. Repeat in finer
context cells to show that limitation, rather than calling the global term a
pure conditional-correlation effect. One measured outcome per row cannot
identify that row's conditional distribution.

Group by training-defined quintiles of intrinsic magnitude/size/Sersic,
three crowding features and reference-neighbour score, plus25 intrinsic-size
by neighbour-score cells. Report g0/g05 separately and combined. Cut-independent
truth/context bins avoid selecting on the residual being tested. Keep row and
contributing-case counts; do not treat sparse cells or many correlated tests
as independent discoveries. Group diagnostics are not no-cancellation gates.

## Connecting photometry to response weighting

Compute observed minus model mean reference-neighbour score among selected
rows. At the central cut partition its error into25 scene-context cells using

    contribution(cell) = sum_cell [(B - b_mid)*(I_observed - P_model)] / D_mid,

where b_mid averages the two globally selected mean scores and D_mid averages
their selected masses. Contributions must sum exactly to the global error,
including each leave-one-scene-out replicate. This identifies which cells
supply the existing association discrepancy, without confusing their own
normalized means with contributions to the whole population.

B is the frozen emulator evaluated at reference photometry, not physical R or
multiplicative bias. This audit localizes photometry/selection association;
it does not independently identify emulator error, classifier error, missing
neighbour-shear response or the full coherent calibration budget.

## Uncertainty, checks and execution

Use whole-case paired delete-one jackknife errors, grouping both legs of each
scene. Model identities and sampled rows stay fixed; errors exclude retraining
variability. Training/validation comparisons diagnose fit versus generalization,
not an independent acceptance claim. Same-draw half-bank differences quantify
integration sensitivity; do not equate a row prediction residual with bias.

Synthetic fixtures verify marginal/association closure, swapped-model
antisymmetry, unequal-mass cell contribution closure, and the65-rank null.
Independent direct row sums check the report's scene reductions. Report only
after all four models/splits and canonical replay checks pass.

One persistent A40-16GB job (four CPUs/16GB host memory) performs fixtures and
both splits, reusing loaded models. One dependent CPU job reduces the report.
This uses one GPU within the six-GPU ceiling and four CPUs within500. All
nontrivial work is scheduled. Broad17788286 remains held; automated follow-ups
remain disabled. No training, production configuration or model weights change.
