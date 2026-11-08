# Simplify the training sequence

## Corrected question

Owner clarifies that the four-stage training history looks unnecessarily
complicated. The previous architecture-compression experiment addressed a
different question and its unfinished jobs have been stopped, outputs retained.
Keep the latest architecture/performance reference fixed while asking which
continuations can be omitted or combined. Do not assume that historical
research steps constitute a necessary production recipe.

## Evidence-first comparison

First reuse saved predictions from the original likelihood-only factorized
flow (37c4afc13b04) on all40 scenes/both axes/all nine cuts. This is distinct
from the older fully joint Cartesian flow (d3c671a23e21), whose second-axis
failure was recently established. Re-reference original factorized predictions
to the same new paired-seed measurements and compare with latest43673e5d.
No GPU prediction or training is needed for this saved-sum calculation.

Also compare the four existing matched neighbour-study endpoints on exactly
four shared scenes: extra40epoch NLL with/without the added input, and each
with the subsequent response regularization. Keep the four-scene table separate
from full40 results. Historical unpaired-measurement values must not be treated
as updated calibration. Validate checkpoint/auxiliary/source receipts, original
measured-sum replay, latest reference replay and independent scalar responses.
Historical caches without original data hashes are hashed now and identified
as such; do not claim newly recorded hashes authenticate their original bytes.

The final conditional-cut loss has demonstrated a modest incremental benefit:
full40 two-component nine-cut RMS0.75735→0.63457%, paired−0.12278±0.00515pp,
under new measurements. That comparison does not establish that the preceding
stages, or a separate final stage, are required. Earlier standalone-response
improvements did not consistently improve coherent calibration. Adding an input
up front would avoid a separate input-introduction stage, but matching the final
performance of that training recipe would still need a controlled comparison.

Report m=100*(Rmeasured_ii/Rmodel_ii−1), independent per-leg generated/measured
selection, predicted usability, whole-scene paired jackknife/bootstrap errors.
All18 target values and their RMS remain the performance criteria. These are
reused development scenes, not independent acceptance. No new training is
launched before this audit identifies what the simpler existing recipe achieves.
[Conventions](../../CONVENTIONS.md#6-response-estimators).
