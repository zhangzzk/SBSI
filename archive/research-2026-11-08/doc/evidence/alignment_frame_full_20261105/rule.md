# Full-budget intrinsic-frame GalSBI comparison

## Motivation and controlled difference

The native GalSBI replay underfits the alignment relation. Keeping the FS2
learning-rate schedule improves held-out NLL and alignment, but worsens R22.
The previous intrinsic-frame pilot used only 40 cases, a smaller training budget,
a different seed and isotropized coordinate scaling. It cannot settle whether
the representation helps at the original full budget.

Train one fresh intrinsic-frame flow on all160 GalSBI training cases. Reuse
original preprocessors and ALL original model configuration, including unequal
disk-coordinate scales/means. Same base initialization seed611, 140 epochs,
4million rows/epoch, batch8192, AdamW and clip5 as the completed Cartesian
`alignment_trajectory_20261105/galsbi_fs2rate` control. Use that experiment's
FS2 consumed learning-rate schedule from epoch1. Its native GalSBI prefix
through20 had identical rates, so it is the exact Cartesian schedule control.
Save likelihood-selected and endpoint models and optimizer states. The only
scientific change is expressing output shape in the intrinsic-shape frame.
Conditioning features retain the original absolute orientation; this does not
assume exact rotational symmetry of pixelized measurements. No simulations,
measurement changes, new cuts, empirical offsets or retained preset changes.

## Coordinate transformation and checks

At fixed context, rotate physical shape into axes parallel/perpendicular to
input ellipticity; retain magnitude/radius. Undo that rotation for generated
world-frame outputs. The standardized transformation is D^-1 R D plus shifts:
its determinant is one even for unequal shape scales. Original standardization
therefore permits comparable likelihood. Near-zero input shape uses identity.
Verify inverse, norm, determinant, density/Jacobian, support, exact reload and
Cartesian-wrapper identity before submission. Artifacts retain the explicit
experimental frame format, never silently loaded as ordinary world-frame flows.

## Evaluation and decision

Select minimum NLL on the original500k validation indices, never alignment or
response. Trace the same160k train/validation samples as the schedule study.
Evaluate selected checkpoint on all40 harmonized validation cases, seven cuts,
two16-draw banks, same latent streams and500k common-target NLL sample.
Compare full response matrix, alignment curves, selected counts and neighboring
cuts against the schedule control and native/retained GalSBI. Paired10000
whole-case bootstrap draws condition on fitted models; they omit retraining
uncertainty. The comparison is development evidence, not independent acceptance.
A local alignment gain alone does not justify joint constant-shear validation.

## Resources and provenance

One GPU for training, dependent one GPU for evaluation; maximum two across the
investigation. CPU reduction follows evaluation. Preserve source/checkpoint
hashes, matching exclusions, exact measured replay and job receipts. Follow
startup/first meaningful progress, then schedule a bounded status check in2h.
