# Neighbour-conditioned Cartesian flow trial

## Authorized question and frozen comparison

Owner2026-11-06 requests adding the neighbour information that recovered the
selection association to the flow, then testing it. Add one scalar: sum of the
frozen original disk emulator's pair coefficients at radius3.5px/magnitude24.5,
over the same nearest20 rendered neighbours within10arcsec. No measured target
quantity or constant-shear response label enters this feature or flow training.
Keep circularized primary size, all original inputs and physical output maps.

Start both arms from the improved Cartesian NLL checkpoint37c4afc13b045bbd118eea0d20572961eef5f6649d6b9329c1e5d09fe820afbb.
Expand both networks identically by one context column, initially zero-weighted.
The control receives zero; the augmented arm receives the training-standardized
neighbour score. Verify preservation of density and physical draws at startup.
Train both on the original160 cases, validate on the original40 cases and same
fixed500k NLL rows. Constant-shear scenes never enter training or checkpoint
selection. Both arms use identical row draws, batches and optimizer recipe.

## Training and regularization

Fixed40 NLL epochs,4million original training rows per epoch without replacement;
AdamW with no decay, learning rate1e-5 for epochs1--20 and3e-6 for21--40,
original8192-row batches and norm5 clipping. Seed611+1000003*epoch for both arms.
Report fixed epoch40, without selecting by NLL or response.
Then apply the existing weak nine-cut penalty to each endpoint:64 updates,
weight0.001, the original32768-pair independent groups/four draws and all other
settings from alignment_regularized_20261106. Preserve both pre/post endpoints.
The matched control separates feature effects from extra training.

## Evaluation and interpretation

First check the four existing diagnostic scenes1000/1040/1080/1120, both shear
axes and all nine target cuts. Preserve original classifier/emulator and portable
64-draw CRNs, and independent per-leg measured selection for the simulation.
Report predicted and true usability with generated selection, plus true measured
selection as a diagnostic. Reproduce measured baseline sums before inference.
Use paired whole-case bootstrap errors, explicitly four-scene development evidence.
Evaluate original standalone validation response/proxy and selection association
as well as NLL. Improvement in the proxy alone is not a successful calibration.

The new input is static under shear, like the original scalar crowding inputs.
This tests selection/scene association; it does not itself implement the separately
missing neighbour-induced photometry/selection response. No acceptance claim,
empirical offset, production preset replacement or new image simulation.

## Execution gates

GPU feature pilot validates training-leg truth parity, complete ID matching and
replay of the previous constant-scene coefficient. Full feature cache requires
all200 case receipts. Standardize using training rows only. Test augmented-layer
insertion, physical density/inverse and save/reload before training. At most six
GPUs across this study, existing cip limit three; feature workers use at most
48 CPUs and six GPUs, then training/evaluation run after feature workers finish.
Use CUDA emulator prediction consistently: the first CPU feature pilot differed
from the earlier CUDA reference by up to2.79e-7 in a few scene sums, failing the
unchanged replay tolerance. Preserve that attempt and use the canonical CUDA
calculation for all new features rather than mixing implementations.
Keep the older broad17788286 evaluation held. Bounded checks replace
continuous polling once long stages are progressing.
