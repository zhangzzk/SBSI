# Selected-response localization protocol

## Authorized question

The owner asks to continue after the completed four-way comparison. Locate the response changes near size/flux selection boundaries and in neighbour environment, using the four fixed models. No new training, checkpoint selection, images, empirical offsets or broad evaluation resumption.

Reuse all four coherent scenes 1000/1040/1080/1120 on both axes, nine target cuts, identical 64 portable antithetic draws, 1024-parent blocks, classifier, emulator and reference-score calculation. Replay the previously saved four-arm sums. Preserve held array17788286. Use at most three concurrent GPUs on inter, within the combined six-GPU study limit. Pilot the first case/axis, then three packed workers cover the remaining seven tasks.

## Exact paired decomposition

For new arm A and matched control B, each draw has a selection indicator I and a vector y containing two flow-shape and two emulator-shift components. Under the fixed common-latent pairing:

Δ(Iy) = (I_A−I_B)(y_A+y_B)/2 + (I_A+I_B)(y_A−y_B)/2.

Call these the cut-membership and output-value contributions to the numerator. For global per-leg selected sums S and masses N:

Δ(S/N) = ΔS (1/N_A+1/N_B)/2 + (S_A+S_B)(1/N_A−1/N_B)/2.

The first two terms acquire the common inverse-mass factor; the last is the normalization contribution. Apply each arm's original classifier weights or measured usability, then pool whole scenes, then normalize. Difference the two shear legs and divide by0.04. All contributions sum exactly to the change in model response. This is a symmetric accounting identity under the chosen draw coupling, not a unique causal identification. A fixed shape density at fixed photometry does not imply identical shape values after the photometry changes.

## Localization

Use five true-radius bins (<0.15,0.15–0.25,0.25–0.4,0.4–0.7,≥0.7 arcsec) crossed with four within-scene quartiles of the truth-only reference neighbour coefficient. The neighbour bins are relative influence quartiles, not equal physical ranges or direct separation bins. Fix parent bins across models and shear legs; retain numeric quartile edges and parent counts. Include all parents, including those never selected. Ties may give unequal quartile occupancy.

For each paired draw/cut classify: selected in both arms; selection switches with radius-cut state alone changing; flux-cut state alone changing; both component cut states changing; selected in neither arm. These classes explain where membership changes occur. They do not assert a physical cause of the photometry error. Contributions use the global selected mass so their sums reproduce the whole-population change; they are not subgroup-normalized responses. Report all three noncontrol arms versus control, all nine cuts, both axes and predicted/measured usability with generated membership. The previously saved measured-membership mode remains a diagnostic reference.

## Validation and uncertainty

Check event assignment, per-draw algebra, arm-swap antisymmetry, unequal normalization and zero-change cases on CPU. Require exact model/hash provenance and original aggregate replay, with declared absolute raw-sum tolerance2e-7 and relative2e-10. Diagnostic sums use float64; their responses must agree with original mixed-precision summaries within1e-7 per component. Check independent ungrouped accumulation and partition closure, and sum the reported components back to the saved response changes. Recompute nonlinear normalization for each leave-one-scene-out sample; quote paired scene-jackknife SEs. Four reused scenes and fixed trained models do not provide independent acceptance or retraining uncertainty.
