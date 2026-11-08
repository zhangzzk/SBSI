# Conditional cut-probability training with fixed conditional shapes

## Authorized comparison and scope

Owner 2026-11-07 says to implement the proposed improvement experiment. Run two
matched photometry-only continuations from the unchanged nine-cut start
6fc0c373… used in the preceding frozen-shape experiment. The control retains
the original NLL plus full-matrix primary response penalty. The weighted arm
adds the conditional cut-probability score below; neither arm uses the old
static-neighbour moment. Freeze conditional shape parameters, outer transforms,
standardizers, classifier and emulator. Keep differentiation through the fixed
shape function's photometry inputs in the original response penalty.

The original 160 training cases alone supply gradients and weight calibration.
Use the original disjoint 40 validation cases for diagnostics. The four existing
constant-shear development fields 1000/1040/1080/1120, both axes and all nine
target cuts supply evaluation only. No new images, inference changes, outcome-
selected checkpoints or release identities. The broad evaluation remains held.
This experiment tests an objective change, not the missing neighbour-shear
input channel. Do not expand that channel's image or training campaign here.

## Conditional objective and unbiased integration

For a usable training row i and each cut c, let y_ic be the measured soft cut
and q_ic the flow expectation of the same soft cut. Radius and magnitude use
sigmoid widths 0.05 pixels and 0.05 magnitudes, identical to the previous
experiment. The new objective is the row-and-cut mean of (q_ic-y_ic)^2.
It estimates a conditional probability/smooth selection expectation from noisy
row outcomes; an individual outcome is not an exact probability label.

Estimate q with two independent latent banks A/B on the SAME sampled rows:

    L_probability = mean_i,c [(q_A,ic-y_ic)(q_B,ic-y_ic)].

Each bank uses four antithetic draws. Independently seeded PCG64 banks retain
antithetic pairs within each bank, not across banks. Independent banks remove
the extra Monte Carlo variance term in a squared single-bank estimate.
The finite estimate may be negative; do not clamp it. The same observation in
both factors retains measurement noise in the score, which does not alter its
conditional-mean minimizer. Do not average rows before multiplying residuals.

Use 32,768 training rows per step, 1,024-row chunks and seed 202611071201 plus
100003 times the zero-based step. The first latent bank adds 10000000 plus the
block index to the row-sampling seed; the second adds another 100000000. Rows
are sampled once per step and shared by the banks; both arms use
identical row/latent streams. Chunk accumulation accounts for a shorter last
chunk and is checked against direct autograd.

## Training-only weight calibration fixed before outcomes

At the untouched start, take four independent training batches with seeds
202611071101 plus 100003 times the batch index. On each, measure the un-clipped
photometry gradient norm for the unchanged NLL-plus-response objective and the
unit-weight probability score, using the original NLL/guard batch settings and
the probability settings above. Set the additional weight once to

    0.25 * RMS(base photometry gradient norm) / RMS(probability gradient norm).

Save all norms, gradient cosines, row-stream definitions and the derived weight
before either arm updates. Require finite positive norms. This is a frozen
training-only loss normalization, not a calibration correction or adaptive
response-based hyperparameter search. Do not change the fraction after seeing
validation or coherent outcomes. Log subsequent gradients and clipping to
make the effective strength inspectable.

Both arms use 64 AdamW updates at 1e-5, no decay, NLL batches 8192, the existing
response loss weight 0.001 and norm-5 clipping over photometry parameters only.
The original guard uses its original two independent 32768-pair groups and
four antithetic draws. Numerical batch sizes do not change. Resume states every
eight updates preserve optimizer/model/history and pinned source provenance.
Report fixed step64 for both arms, irrespective of the diagnostics.

## Correctness gates and behavioral checks

Before training, verify analytical independent-bank expectation and gradient,
detect reused-bank variance bias and cancellation in a population-mean loss,
and reject row-shape mismatches. Compare the real model's chunked backward with
direct autograd, including an uneven final chunk. Require zero gradients for
the frozen shape factor and exact conditional shape density at measured inputs.
After training/reload, require every nonphotometry tensor unchanged and at
least one photometry tensor changed. Require exact reload predictions.

Own-data evaluation keeps the original 500k sampled rows per split, 64 portable
draws and original case/leg mapping. Report per-row hard-cut squared error with
two independent 32-draw antithetic banks, the plug-in score/integration-variance
identity, the 0.05 soft score, and predeclared 0.025/0.1 smoothing sensitivities.
These sensitivities do not select a width or checkpoint. Report all nine cuts,
per-leg selection mass, selected shape numerator/mean, photometry and conditional
shape NLL, and the legacy static-neighbour score. The single-leg shape moments
are not coherent shear responses. Original matched-pair probes retain the
primary response diagnostic. Common draws and whole-case paired errors apply.

The constant-shear evaluator is reused unchanged: 64 paired portable draws,
1024-parent blocks, independent per-leg normalization, full emulator transport,
predicted usability/generated selection and actual-usability diagnostic arms.
Both endpoints run together in each allocation. Verify simulation-sum replay,
exact measured selection mass and pooled/separate emulator agreement. Record
counts/matching receipts, full response matrices, flow/emulator contributions,
all nine cuts, both axes and paired case uncertainties. Interpret coherent
results jointly with cut/shape diagnostics; improving the score alone is not
success. Internal cancellation is allowed; distant cuts are diagnostics.

## Packed execution and stopping

One A40-16GB GPU job performs analytical/numerical checks, training calibration,
both trainings, own-data diagnostics and the first field/axis. Its success
gates three persistent workers, covering all seven remaining field/axis tasks,
followed by a CPU report. Each GPU job uses four CPUs and 32GB host memory.
Maximum three concurrent GPUs in this chain, within the study's six-GPU cap.
No other study GPU job was active at preparation. Maintain the owner's
preference for packed work and the held broad array 17788286.

Execute pinned source bytes rather than stale NFS timestamp bytecode. Preserve
earlier evidence and failed attempts. Numerical failures gate evaluation;
scientific outcomes do not change the predeclared endpoint/evaluation set.
Monitor startup and meaningful progress. The owner-stopped automated follow-up
loop remains off; this protocol does not authorize reinstating it.
