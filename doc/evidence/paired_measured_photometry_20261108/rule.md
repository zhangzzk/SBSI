# Measured-input localization with paired-seed fit outcomes

## Question and estimator

Owner requests continuation on2026-11-08 after the independent per-leg paired
measurement finds positive residuals, particularly at radius>3.2 pixels.
Substitute measured radius, flux and NEW usable membership into the fixed
conditional-shape flow and disk emulator. Keep the same40 scenes, both axes,
±0.02 shear, and all nine target cuts. Do not impose truth cuts or common-leg
membership. Normalize each exposure separately before differencing over0.04;
m=100*(Rmeasured/Rmodel−1). Use whole-scene paired jackknife and10,000 bootstrap
samples with seed202611061020. Correlations between cuts must be retained.
[Definitions](../../CONVENTIONS.md#6-response-estimators).

## Validated reuse of unchanged predictions

The shape refits changed shapes and usable membership, not radius/flux. In this
oracle model expectation the measured shape does not condition the flow or
emulator coefficient: the flow uses measured photometry, truth context and
fixed neighbour feature; the coefficient uses measured photometry and complete
truth neighbours; the model's own shape is transported. Therefore rows retaining
membership have exactly the same mathematical model expectation. Their measured
shape sums DO change and are taken from the completed new measurement.

Reuse pinned old oracle sums and recompute signed contributions only for rows
whose validity changed: add newly usable rows, subtract newly invalid rows.
This is an algebraic update from evaluated predictions, not an empirical response
correction. Prepare all candidate IDs passing any target photometric cut in at
least one leg; verify old/new measured sums and old oracle input receipts.
No object outside this photometric domain contributes to any reported cut.

Preserve full-parent1024-row blocks and64 antithetic latent draws with seed
7301+10000019*case+1000003*block. Subset rows only AFTER generating their original
latent bank. Keep full truth-neighbour catalogues, K20/radius10arcsec pair rules,
physical-moment disk emulator and Möbius composition. Verify the static feature
against direct pair grouping and exact conditional equality of the frozen
control/probability-loss endpoints; original61 nonphotometry tensors and
conditioning states remain fixed. Use original full-bank/two-half-bank outputs.

## Pilot gates

One packed GPU pilot covers case1000/axis1 and case1129/axis2. For each, independently
compute full old/new selected expectations over the complete target domain,
then compare with old saved sums and the signed update. Require exact selected
counts and <=1e-6 absolute normalized response difference for each shape/shift
term, draw bank and cut. Preserve actual discrepancies. Check conditional
reconstruction, unused photometry-latent invariance, finite outputs and original
representable unit-disk boundary tolerance. Failed gates block all workers.
Do not relax a failed tolerance to obtain acceptance.

## Report and limits

Report new measured-input matrices, m1/m2, mean/half difference, all nine cuts,
paired shifts from the frozen generated-photometry/predicted-usability result,
radius3.0→3.2 contrasts, joint RMS and draw-bank sensitivity. Oracle substitution
changes photometry, membership and usability jointly; it is not a causal additive
allocation or deployable correction. Persistence would identify an error remaining
in the conditional-response composition or measurement, not a unique component.
These reused development scenes provide no independent acceptance claim.

One pilot gates four persistent one-GPU workers packing the other78 case/axis
tasks, reusing models. Four CPUs/32GiB each, <=4 concurrent GPUs and well below
the study's six-GPU and500-CPU ceilings. A dependent CPU report verifies all
receipts and completed generated-calibration replay. Broad17788286 remains held;
automated follow-up timers remain disabled. Production artifacts remain unchanged.
