# GalSBI photometry-conditional shape experiment

## Evidence and hypothesis

Full-budget intrinsic-frame training improves alignment at radius2.7–2.8 but
worsens it at3.0–3.1. The relation crosses the selection boundary sharply, and
accurate marginal counts/global likelihood do not guarantee accurate shape
within each radius bin. Test a different organization of the joint density:
p(radius,flux|input) times p(shape|radius,flux,input). This is a modeling
hypothesis, not an established explanation for the remaining response error.

## Predeclared architecture

Replace the ten four-dimensional affine couplings with two five-layer
conditional affine flows: two-dimensional photometry and two-dimensional shape.
Use original256-wide, three-hidden-layer SiLU coupling networks and scale limit.
Photometry uses original16-dimensional context; shape also gets the two
standardized unbounded photometry coordinates. Keep original radial-tanh disk
and softplus positive-output bijections, all means/scales/units and exact density
Jacobians. This is an exact triangular factorization, not teacher forcing at
inference: draw photometry first, feed those same draws to shape generation.
Physical radius/flux are bijective functions of the conditioning coordinates.

Two arms use identical initial base weights: Cartesian versus intrinsic-frame
shape outputs. Retain original absolute input orientation in both. Total
coupling count stays10, but each marginal has5 and parameter count differs
slightly from the old joint flow; this is an architecture/allocation test, not
an exactly capacity-matched proof about factorization alone. Record counts.
Artifacts use a distinct experimental format and cannot replace retained flows.

## Fixed training and evaluation

Same160 GalSBI cases, original standardizers/configuration, seed611,140epochs
of4m rows, batch8192, AdamW/clip5, same completed FS2-rate schedule as the matched
Cartesian/frame controls. Minimum original500k-validation NLL selects checkpoints.
Neither alignment nor response enters the loss or checkpoint selection.
Save optimizer states and identical160k train/validation diagnostic samples.
Evaluate all40 held-out cases, same seven cuts, same two16-draw antithetic banks,
full2x2 response, selected counts, full alignment profiles, common500k NLL.
Use10000 paired whole-case bootstraps with fixed models; this omits training
uncertainty. Compare both new arms against both original joint-density controls
and retained GalSBI. Development evidence only. Do not call a local-bin gain a
calibration success; inspect the relation on both sides ofradius3 and all
response components before choosing a further or constant-shear test.

## Validation and resource limits

Before launch, verify triangular forward/inverse and analytic versus autograd
Jacobian, physical support, full density Jacobian, frame inverse, save/load,
identical paired initialization, and no photometry dependence on shape latent.
No new simulations/measurements, empirical corrections, production files or
retained checkpoints. Training array0–1%2; dependent evaluation0–1%2; CPU
report follows. Two GPUs total. Inspect startup then use bounded2h status audit.
