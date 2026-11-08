# Direct conditional-shape check at zero shear

## Question and scope

The completed current-model measured-input test retains a component-2 bias
near−0.96%. Check a narrower identifiable question: does the frozen conditional
shape distribution reproduce measured shape moments when supplied measured
size/flux in zero-shear validation images? This is an empirical shape-distribution
test, not an attribution of the coherent bias to the flow or emulator.

Use g0 rows within the existing fixed500k validation-NLL sample, all40 original
validation scenes, actual per-row usable status, photometry and nine joint cuts.
No common-g0/g05 intersection. Preserve original sample/block order and portable
64-draw streams, including unused photometry latent coordinates; skip g05 rows
only after constructing each original block's latent bank. All original inputs
and model identities are pinned. No retraining, new images or altered cuts.

Compare original NLL, original nine-weak, neighbour-conditioned NLL and the
current conditional shape factor. The two latest photometry endpoints share
this last factor exactly, so one conditional draw suffices. Earlier models
follow different training branches; differences do not isolate one loss or
feature. Keep their complete labels and checkpoint hashes.

## Quantities and interpretation

Report e1,e2,e1²,e1e2,e2² and e1²−e2². For each scene/cut save observed sums,
two independent32-draw antithetic mean estimates, and the product of the two
prediction-minus-observation residuals. This product estimates the conditional
moment squared score without adding finite-draw variance; both banks must use
the same row. It remains a noisy predictive score, not an error to a known
conditional truth mean for each object.

Use selected count normalization, whole-case10000 bootstrap and delete-one
jackknife with original seed202611069913. Report calibration errors, paired
model changes, predictive-score changes and half-bank sensitivity for every
cut/feature. Conditional photometry is measured; no generated selection occurs.
Do not equate raw shape-moment units with multiplicative-bias percentage points.
The quadrupole is relevant to the direction dependence of disk transport, but
this unweighted zero-shear moment is not its coherent response error.

A detected zero-shear shape error supports testing conditional-shape learning.
A null result does not validate its shear derivative or the emulator. Neither
outcome warrants an empirical correction or acceptance claim.

## Gates and resources

Require identical base input preprocessing/target standardization across arms,
exact measured photometry in generated draws, direct conditional reconstruction
of joint samples, exact case/row mapping and static-score alignment. Save all
40 case sums and hashes. One persistent one-GPU/four-CPU job packs four frozen
models, evaluation and the small reduction. Check startup and completion.
Broad17788286 remains held and automated follow-ups stay disabled.
