# Simulation-based shear inference boundary

SBSI can load an explicit measurement-flow checkpoint and evaluate its
four-dimensional detected-object likelihood over a latent ellipticity grid.
`BayesianInference.load(checkpoint)` and its catalogue arguments are
model-name agnostic and accept user-owned paths or DataFrames.

The response input-catalogue boundary is implemented independently in
`sbs_shear.forward_catalogue`: SBSI validates a truth catalogue and returns an
aligned one-row-per-primary flow view plus a many-row primary/secondary emulator
view. It computes the flow's intrinsic-shape and crowding features, applies the
emulator's recorded pair cuts and rescaling, and preserves `primary_row` keys.
BlendEMU is called only to evaluate its trained model on the prepared pair
table. This preparation completes the two inputs for response prediction, but
it does not create the measured-target catalogue or complete the joint shear
likelihood described below.

This is not yet the validated catalogue-level shear inference requested for the
public workflow. That implementation still needs a joint generative likelihood
for the measurement flow and neighbour/emulator contribution, detection and
selection normalization, latent scene marginalization, a hierarchical population
prior, ensemble uncertainty, and simulation-based coverage tests.

Until those pieces are complete, the tutorial leaves Part 3 as TODO. Existing
posterior-grid and empirical-Bayes code is a research prototype for a detected
population and must not be presented as the final simulation-based shear result.
