# Corrected held-out validation preparation

## Status and launch condition

The complete corrected v10 comparison now passes its declared raw score,
normalizer, scene-region and both-measure phase-controlled score comparisons
(audited CPU17434348). This authorizes the next validation stage, not sampler
acceptance. No v10 interim aggregates or held-out pilot numerical outcomes were
used to prepare the procedure. The earlier held-out v7 production remains
cancelled; its declaration and warmup receipts are preserved. New held-out
science has not yet launched.

## Conditional fixed procedure

If the v10 small arm passes, test its N256/T64/M64 fitted-proposal procedure
against N1024/T64/M64 broad Student-t5 on the unchanged twenty-object measured-
magnitude panel. Preserve its five original four-observation groups, density
batch4096, V100 hardware and independent eight-population production groups.
Use fresh, globally disjoint pilot/candidate/reference seeds. Freeze the actual
source, declarations, seeds, model/reference/prior inputs and job graph before
starting; none is authorized by this document alone to bypass the v10 gate.

Use four fresh Gaussian N256/T64/M32 pilots per group, from the corrected source.
Apply the existing frozen gradient-geometry and initial-proposal fitting rules.
Evaluate every pilot's complete phase diagnostic, then fit the existing pooled
covariance phase coefficient rule once. This fitting is a fixed part of the
inference procedure, not an opportunity to tune on held-out results. Freeze all
fits before production in that group. The old v7 pilots are not reused.

The final wrapper must verify that declaration groups equal the ordered IDs in
the hashed panel, every run uses the frozen complete source and required external
inputs, all declared seeds are present, and every diagnostic uses all particles.
The phase checker's zero process exit alone is insufficient: its
`numerical_checks_pass` must be true. Any pilot/production extinction or unknown
failure rejects the panel and preserves its seed and cost; do not omit, replace,
extend, or refit a failed population.

## Separate declaration-bound fitter

`scripts/fit_declared_particle_phase_control.py` calls the existing
`fit_frozen_phase_control` API without relaxing the historical n15 fitter's
`dev_v1_0..3` restriction. It requires schema
`sbsi-heldout-corrected-declaration-v1`, an exact `source` and
`source_manifest_sha256`, `output`, `panel` and `panel_sha256`, ordered `groups`,
`development_ids`, `shear`, `density_batch_size`, reference/observation hashes,
and pilot/candidate/reference settings. Pilot settings also explicitly record
`blocks`, `scale`, and schedule `power`. A `pilot_proposal_metadata` array pins
the complete four proposal laws for each group, including means/scales. Generate
that array from frozen measured inputs during preflight before launching pilots;
checking only Gaussian labels and mixture weights is insufficient.

Each group's inputs must be `group_N/pilot_0..3` and
`group_N/pilot_0_phase.json` through `pilot_3_phase.json`; the only allowed output
is `group_N/phase_coefficients.json`. Global seed construction remains
`seed_base + 10*group + replica`. The fitter rejects seed overlaps, development
IDs, changed recipes or input hashes, mixed frozen implementations, incomplete
diagnostics, and any already-started production in that group. It preserves
the declaration, source and pilot hashes in the coefficient artifact. It does
not read production outcomes, choose coefficients against truth, or apply an
empirical offset. The [launch wrapper](CORRECTED_HELDOUT_RUNNER.md#scheduled-stage-interface)
now implements the full cost/accuracy report and is under integrated validation.

Diagnostic provenance must bind exactly the declared pilot's protocol, result
and particle artifacts. Target identity is checked against that protocol and
the pinned native sources. Mandatory inference/diagnostic source maps and the
actually imported phase module are checked against the manifest. The manifest
also covers transitive SBSI files omitted by the older phase identity. External
dependency/artifact pinning remains the full workflow's responsibility.

## Measurement-only preflight

`scripts/prepare_corrected_particle_heldout.py` requires the hashed successful
complete development receipt and rechecks its evidence before opening the fixed
panel. It requires the existing development scientific sources to remain byte
identical, including all SBSI/config files and the driver, prior factory, fitting
and diagnostic programs. Reporting/preparation helpers may be added. The optional
categorical optimization therefore remains isolated during this validation.

The worker constructs full Gaussian pilot and Student-t5 reference proposal
metadata from the same four measured columns as inference. It pins package and
model/reference inputs, explicitly includes the ABC posterior and photometry
HDF files (C-level HDF reads may not emit Python open audit events), and records
actual constructed prior parameters/tables. Nonfinite configuration defaults are
tagged in JSON metadata without changing the prior. Read-only input opens and
generated/write-capable paths are separated; logs/locks/bytecode are excluded
from immutable input candidates. Actual scientific imports must come from the
declared frozen source.

This is bounded provenance, not a whole-system trace or container image. The
constructed tables are not asserted to be independently checked in every later
sampler process. The preflight emits an explicitly nonlaunchable declaration
draft. After its scheduler job completes, attach a hashed allocation-cost
receipt and freeze the actual declaration before warmup. Global fresh seed
bases are2026103200/2026103300/2026103400 for pilot/candidate/reference, with
the existing `base+10*group+replica` rule. No posterior outcomes enter preflight.

## Accuracy, cost and interpretation

Require all original raw score/Z/region gates for every object. Additionally
require both corrected score components to agree within3.5 combined whole-
population MCSE under both recorded F32 sampling and F64 reweighting. No good
cost or phase-control result rescues failed raw gates. Production uncertainty is
conditional on the shared frozen fits; independent production populations justify
the conditional combined MCSE. The F32 correction remains a qualified diagnostic;
the zero-mean identity concerns the intended continuous F64 model.

Charge all four pilots, their phase evaluations, geometry and coefficient fits
to both recipes. Charge proposal fitting to the candidate only. Charge each
recipe its production, production-phase and report costs; charge joint comparison
and verification costs equally and explicitly. Also report actual total study
work separately, without double-counting shared preparation. Include process and
model loading through timed subprocesses. Record failures rather than subtracting
their costs. Report GPU and CPU seconds separately where possible; summed process
seconds are not panel wall time. No development/search amortization is claimed.

This requires20 pilots,80 production populations,100 phase evaluations, five
each of geometry/proposal/coefficient fits, and the complete reports. Both
numerical measures come from each phase evaluation. Acceptance is accuracy AND
lower all-inclusive candidate cost. It establishes a bounded fixed-y validation
at the declared tolerance, not global rare-mode coverage or a population shear-
bias result. A failed held-out gate rejects promotion; further tuning requires
a new held-out panel.

## Validation

The declaration-bound fitter has synthetic checks for independent fitting,
coefficient provenance, changed recipes, missing independence, production leakage,
changed panels/sources/original artifacts, numerical failure, and unsafe manifest
paths. CPU17423221 passed878 tests, one optional skip, in102.93 seconds on
the initial frozen313-file snapshot
`source_declared_phase_fit_20261022`; manifest SHA256
`dd9f33f159e23f0ab0b72e78e08050e4eebe95cd150799d20a043e48994ec817`.
Independent review then identified the provenance, mandatory-source and full-law
gaps described above. CPU17423307 tests the strengthened313-file snapshot
`source_declared_phase_fit_v2_20261022`, manifest SHA256
`33b9e92c5967818039afb3fbf0b946dece535b52ab21aa67e41c887ce4e8b989`.
It passes885 tests with one optional skip in90.37 seconds. Before/after source
checks pass. The earlier passing tests did not cover these gaps. The
[validation receipt](evidence/declared_phase_fit_validation_20261022.json) preserves
both snapshots and results. No scientific held-out computation was run.

Initial preflight/reporting integration CPU17423593 passes935 tests with one
optional skip in91.31 seconds. Independent installed-source review then exposes
real inherited infinite configuration defaults and C-level HDF input reads;
the strengthened preflight adds explicit metadata encoding, input paths and
actual-import checks. CPU17434394 validates this revised318-file snapshot,
manifest `6cc595ccf72d540a58856a92bcf4ea6156aa5731bdf0d90e55397bd99e5128b6`;
it passes945 tests with one optional skip and one existing warning in110.42
seconds (137-second allocation), with before/after manifest checks. The
[preflight receipt](evidence/heldout_preflight_validation_20261022.json) preserves
both runs. Final integration also pins the reference preparation result read by
the sampling driver. These are code checks, not a held-out result.
