# Paired-seed calibration with independent per-leg selection

## Objective and fixed estimator

After the40-scene common-galaxy seed repeat, restore the actual selected
calibration estimator. Each shear exposure selects its own measured population.
Use the same40 parent scenes and all nine target cuts: radius>2.8/3.0/3.2pixels
crossed with magnitude<24.9/25.0/25.1. Apply usable shape validity (finite and
squared radius<1) and measured cuts independently per leg. Normalize each leg
by its own selected count. Compute R from±0.02 differences divided by0.04 and
m_i=100*(Rmeasured_ii/Rmodel_ii−1), with paired whole-scene uncertainty.
Conventions: [response](../../CONVENTIONS.md#6-response-estimators),
[bias](../../CONVENTIONS.md#7-multiplicative-bias).

## Measurement and reuse

Reuse every completed common-galaxy fit, including its failures, without
refitting or replacing a result. Prepare the union of detected/matched parent
IDs whose measured photometry can pass any target cut in at least one leg.
Fit only additional IDs, and only in the legs whose photometry can pass a cut.
Old shape validity is NOT required for candidate inclusion: originally invalid
fits may become valid, and originally valid ones may fail. Report both changes.
No truth-property cuts. Require finite positions for every photometric candidate;
reject an inconsistent catalogue rather than silently excluding such a row.

Preserve the original65-pixel centred cutout, detected position, image/PSF,
weight estimator, Gaussian fitter, convergence and retry settings. Use seed
(2026110700+input_id*17+case*100003) modulo(2^32−1), identical across the four
legs and identical to the completed common-galaxy repeat. Each chunk keeps
all applicable legs for its IDs within one worker. No multistart replacement
or empirical correction. Keep production catalogues read-only.

## Frozen model comparison

Reuse the existing current weighted model's generated-photometry, predicted-
usability response sums for these exact40 scenes and nine cuts. Verify both
model checkpoint identity and every cached evaluation receipt/hash. Reproduce
the original measured sums and old m values before interpreting new ones.
Generated and measured samples apply the same cut rules independently, as
required by the joint prediction workflow; do not condition the model on
observed four-leg common membership. The true-usability/true-selection cache
modes are not reused because they depend on old measurement outcomes.

## Gates, resource limits and outputs

Preparation must replay every old nine-cut per-leg sum and verify exact-once
coverage of reused common fits, their source pins, IDs and checksums. A64-ID
pilot requires exact within-job replay and correct attempted-leg coverage;
its success fraction is recorded without imposing a high validity threshold,
since previously invalid shapes are intentionally eligible. Preserve all
failures and unmatched-row counts.

Use packed persistent CPU workers within the owner500-CPU combined ceiling,
on fast-I/O nodes, with2048-ID chunks and restart receipts. No GPUs are needed
for the reused frozen-model predictions. Full report waits for every worker,
verifies exact-once candidate coverage and reproduces old biases. Report both
new response matrices, m1/m2, their difference and joint nine-cut RMS with
paired40-scene jackknife errors and paired10000-resample bootstrap intervals.
Bootstrap seed202611061020 matches the previous model comparison.

These are reused development scenes, not independent acceptance. Do not declare
a robust0.3% calibration or retrain the model from a partial outcome. Broad
array17788286 remains held and automated follow-ups remain disabled.
