# Independent-population diagnostic audit

## Ratio covariance is correct under its stated assumptions

For independent complete runs, write `Z_r` for the normalizer estimate and
`h_r` for the final population mean. The combined estimator is
`theta = sum(Z_r*h_r)/sum(Z_r)`. Its empirical influence vector is
`u_r = (Z_r/mean(Z))*(h_r-theta)`. The implemented covariance
`sum(u_r*u_r.T)/(R*(R-1))` is the usual delta-method covariance of that ratio.
It retains covariance between the two score components and between region
indicators. It correctly uses independent **runs**, not correlated particles,
as the sampling units. Scaling all normalizers by a common constant leaves
the estimate and its MCSE unchanged.

This is an asymptotic uncertainty calculation. Eight runs are an operational
minimum, not proof of stable second moments or discovery of rare scene modes.
The posterior variance estimate is itself a finite-run ratio estimate. Neither
the score effective-count diagnostic nor final incremental weight ESS proves
convergence. No acceptance threshold or promotion gate is changed by this audit.

## Corrected empirical second moments

The driver stores particle variance with `ddof=1`. The exact empirical second
moment is `((N-1)/N)*sample_variance + mean**2`, regardless of particle
dependence. The old report omitted the `(N-1)/N` factor and slightly inflated
posterior score variance and the derived effective-count diagnostic. Mean
scores, ratio MCSE and normalizer uncertainty were unaffected.

The new `empirical_second_moment` helper performs this conversion. On the
existing four-run v1 and v2 receipts at N=256, corrected score variances are
0.31% to 0.39% smaller. These four-run groups remain ineligible for the
eight-run precision gate. The correction is unrelated to the much larger
observed bright-object instability.

## Receipt compatibility

The report now checks block order, bridge, selection convention, algorithm,
the existing scientific settings, and implementation hashes within each run
group. Source hashes are compared by unique basename, allowing identical
source snapshots at different absolute roots. Frozen geometry compares its
hash and metadata while ignoring its copy location. Completed results must
match protocol observation order/count, reach the final beta=1 stage, and
contain finite aligned moments, normalizers, probabilities and timings.

Comparisons additionally require matching report version and recorded target
source hashes for `galsbi_particle_target.py`, `galsbi_target_coordinates.py`,
`galsbi_summary.py`, `summary_gradients.py`, and `models_v3_9_like.json`.
These checks are deliberately conservative: changing the target wrapper only
to add mutation geometry still changes its hash and blocks automatic
comparison. Reviewed source equivalence or a future explicit target identity
contract is needed before asserting that such versions are interchangeable.
All groups can still receive separate reports.

Recorded source lists do not include every transitive dependency, native-prior
loader or external runtime. Thus this is a stronger consistency check, not a
complete mathematical target identity certificate. Old reports lack the new
fingerprints and must be regenerated from retained runs before automatic
comparison. The v1/v2 run protocols already contain all required fields.

## Extinction remains blocked end to end

`combine_populations` correctly retains extinct runs as zero normalizing mass,
including them in `R`, the mean normalizer and covariance denominators. A test
preserves all four runs, with two extinct, and verifies the resulting weights,
mean normalizer and influence covariance.

The production driver does not yet emit complete per-object extinction
receipts: initial extinction writes a failure and aborts, while later
extinction raises. The current reporter requires completed surviving runs and
does not convert missing files, unknown failures or null normalizers into zero.
This fails closed for a supplied run list; it cannot detect a run omitted from
the list. Compare supplied seeds with the predeclared experiment manifest.
No extinctions were observed in the audited completed development groups.

A future protocol should distinguish each object's completed-surviving,
completed-extinct and incomplete/error statuses. It should encode zero
normalizing mass explicitly, use JSON null for undefined log-normalizer and
moments with an unambiguous extinction flag, finish the other objects, and
require the complete predeclared seed set. Unknown execution failures must
remain errors, never statistical zero mass. An entirely extinct object must
receive failed gates and undefined posterior moments. This audit changes no
driver behavior and introduces no partial extinction protocol.

## Validation

Slurm 17418880 ran
`python -m pytest tests/test_particle_diagnostics.py tests/test_particle_report.py -q`:
**16 passed in 2.84 s**. Checks cover correlated-particle second-moment identity,
independent-run covariance, retained zero-mass runs, receipt/source/geometry
compatibility, permitted changed proposals, rejected changed targets, and
refusal to replace missing or failed runs. No GPU experiment was run.
