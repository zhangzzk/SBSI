# Full joint response at measured photometry and selection

## Authorized diagnostic

Owner asks for the full joint constant-shear response using measured radius,
magnitude and selection. Freeze the improved Cartesian NLL baseline and original
weak nine-cut checkpoint from the earlier four-scene factorial. Freeze the
original physical-moment disk emulator. No training, new images or model offsets.

For each coherent +/-0.02 shear leg separately, retain its actual usability and
measured radius/magnitude selection. At each retained object's exact measured
photometry, sample the flow's conditional shape factor and evaluate the emulator
coefficient. Apply the original Mobius disk transport with the coherent shear.
Average generated transported shapes with the actual selected counts. Compare
their antithetic response with the measured-shape antithetic response on exactly
the same leg-specific population. No common-leg intersection is imposed.

The photometry substitution also conditions the shape distribution; it is not
an overwrite of photometry on an independent joint shape draw. Both flow and
emulator contributions are retained. Actual leg-dependent selection changes are
already included: do not append the earlier opposite-group selection term.
Because measured photometry itself changes under coherent shear, this is a
conditional intervention on the full prediction, not an additive causal budget
or a demonstrated replacement generative likelihood.

## Cases, cuts and uncertainty

Use the existing catalogue definitions of
[usability and selection](../../CONVENTIONS.md#5-usability-detection-and-selection),
the [antithetic response estimator](../../CONVENTIONS.md#6-response-estimators),
and [multiplicative bias](../../CONVENTIONS.md#7-multiplicative-bias).
The measured quantities are NGMIX_G1/G2, FLUX_RADIUS in pixels and MAG_AUTO;
the target radius3px is0.6arcsec at the existing0.2arcsec pixel scale.
Actual usability means a unique crossmatched, finite physical measurement with
positive radius/flux and shape norm below one. It is applied independently per
leg, followed by the measured cuts; no truth-property population cut is added.

Use the existing40 constant scenes on both shear axes. First report the same
four diagnostic scenes1000/1040/1080/1120, then the complete40. Use all nine
radius2.8/3.0/3.2px x magnitude24.9/25.0/25.1 cuts, with the existing auxiliary
cuts retained. Report DeltaR=simulation-model and m_j=100(Rsim_jj/Rmodel_jj-1).
Keep full response vectors, component sums and size contributions.

Use64 portable antithetic draws with the original full-parent1024-row blocks
and seeds. Draw banks are shared across models and shear legs; select usable
rows only after forming the full block's latent bank. Save two disjoint32-draw
antithetic subbanks as a numerical diagnostic. Use10000 paired whole-case
bootstrap resamples, seed202611069902, plus per-case values and jackknife SEs.
These errors condition on the fitted models and existing measurement realizations.

Compare against saved generated-photometry/selection results using true usability.
Freeze available comparison cases before reading outcomes: baseline40, weak-nine22
jointly complete on both axes. Report comparisons on matched subsets; never
subtract the22-case reference from a40-case measured-input estimate. Leave the
older broad evaluation held rather than filling missing generated results.

## Numerical and resource gates

Verify exact measured radius/flux and selected counts, conditional reconstruction
of joint draws, irrelevance of the photometry latent, physical support, simulator
sum replay, coefficient invariance across flow models, and full/half-bank moment
identity. Reject any usable row with invalid photometry instead of clipping it.
The canonical radial-tanh map can round extreme samples to the disk boundary;
retain those representable values unchanged, count them separately inside and
outside the target selection, and reject nonfinite outputs or deviations beyond
16 float64 eps in squared radius. This forward-moment calculation does not
invert boundary samples or evaluate their densities. No clipping or row removal.
The first case/axis gates expansion by these checks, not by its numerical bias.

Neighbour-conditioned flow training and its first joint pilot have completed.
Its remaining array17807394 initially allowed four GPUs while this test
uses at most two. With only five slots available during pilot expansion, lower
that array throttle to three so both measured-input workers can run;
already-running jobs finish normally. After that array drains,
this test may use six. The combined study ceiling stays six GPUs. Broad17788286
remains held. All substantial work uses Slurm, with bounded status checks once
long stages progress normally.

On the owner's subsequent speed check, a running measured-input worker uses
about one CPU core despite four allocated cores. Raise only the remaining
measured-input array17807972 throttle from two to three, matching the earlier
neighbour array's three-worker limit and filling the authorized six-worker
ceiling when resources allow. Keep four CPUs per worker; no numerical code,
thread count, cases or random streams change. The existing release job still
permits six measured-input workers after the neighbour array drains.
