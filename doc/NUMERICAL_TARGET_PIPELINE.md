# Continuous numerical target-prior pipeline

## Purpose and boundary

This optional path records the continuous numerical native target prior q_N,
matching the inverse-table law used by mock target draws. It keeps q_E for the
numerical environment and the original likelihood K32. The analytic default
retains its existing protocol fields. This is a new scientific candidate;
it does not reinterpret, repair or repeat the rejected analytic held-out run.

The continuous density excludes the singular lower-uniform clamp atom and the
literal discrete floating-point random-number law. No claim of literal RNG
exactness, posterior convergence, shear calibration or efficiency follows from
identity checks. Existing numerical-prior sensitivity on represented states
also cannot establish coverage of missing support.

## Driver and consumer routing

`check_galsbi_particles.py --target-prior numerical` selects
`GalSBITargetCoordinates(..., target_prior='numerical')`. Its protocol names the
numerical bridge and endpoint, continuous measure version and source hashes
for the numerical intensity, galaxy prior and environment prior. These hashes
are additional to the existing target-coordinate, likelihood and model pins.
The argument defaults to analytic, preserving the original constructor call
and protocol contents for that path.

The proposal fitter uses `particle_initial_proposal.target_identity`; it now
retains the numerical law and density-source identity. Both loading and
production validation check the new metadata. Analytic and numerical proposals
cannot be interchanged silently. This is source/measure compatibility; all
existing independence, normalized proposal and no-retry rules still apply.

Raw reports carry the numerical law and its density sources. Within-arm runs
must share them; cross-arm comparisons require the same explicit endpoint and
source identity. Different supported initial bridges can still target the
same numerical endpoint. A numerical endpoint without its explicit law or
complete density-source identity is rejected.

Phase diagnostics verify the actual numerical density source bytes and retain
the prior identity in the control artifact identity. Intrinsic phase remains
uniform for q_N, so the same phase derivative construction applies. The F64
reweight is still likelihood-only; it does not transform one prior into
another. Analytic controls cannot silently validate a numerical diagnostic.

## Validation and remaining integration

Focused validation passes81 tests in11.83 seconds across target identities,
proposal artifacts and complete raw reporting. An additional actual-driver
check exercises proposal compatibility before model loading: analytic artifacts
are rejected, numerical artifacts pass the same strict production validator.
All13 identity tests pass in8.11 seconds after adding this check. They include
missing/contradictory sources, duplicate basenames, fitted-proposal roundtrip,
full raw reporting and phase source-byte checks.

The numerical target-coordinate/exchange implementation is developed separately
in `codex/numerical-target-sampler-20261022`. These driver changes require that
implementation before numerical sampling can run. Full combined scheduled
validation, installed-prior checks, newly generated pilots, a numerical-law
phase-fit declaration and a small real-model development experiment remain
required. The old analytic held-out runner/fitter retains its frozen contract;
it is not reused by relabeling its declaration. No GPU experiment is launched
by this change, and no production default changes.

## Combined implementation and installed-prior check

Core commitd65bf00 is integrated as43438c8 after driver commitbf89b0d.
The core's165 focused tests pass, including47 new forward-law/support and
complete mutation probability-balance tests. Combined full-suite validation
remains pending.

`check_numerical_target_native.py` declares4096 fresh parent draws at seed
2026104200 from the installed native generator. It checks numerical-coordinate
routing, the full type sum/conditional probabilities and native log-density
roundtrip against the existing numerical density helper. It verifies every
external-input hash from the frozen prior/model environment before and after,
retains every returned native row and the sampling counts, without replacing failed or off-support draws. The
2e-8 absolute log-density tolerance applies only to coordinate roundtrip;
decoded density/type-sum comparisons are exact array comparisons. These are
installed-environment consistency checks, supplementing the independent toy
forward-density tests; they do not constitute posterior validation.

## First combined validation failure and repair

Full job17448810 finds a phase diagnostic refactor error: the main function
still uses the local source root when writing provenance, but it had been
moved into the new identity helper. Two complete diagnostic tests fail with
NameError;1,089 other tests pass and nine skip. Installed-prior validation
does not run after that failure. Restore the local root and extend the existing
end-to-end diagnostic test across both analytic/numerical laws and both
valid/invalid derivative replay. Preserve the failed frozen source/job; a new
snapshot and complete scheduled suite are required.


## Construction inputs and geometry provenance

Numerical runs now bind `check_galsbi_continuous_prior.py`,
`numerical_prior_inputs.py` and `particle_target_measure.py` alongside the
numerical density sources. The driver builds the prior once with explicit
`model_index=0, grid_size=8193` before validating fitted artifacts. This
construction and the input hashing remain inside reported total run time.
The analytic factory call, protocol fields and numerical calculations retain
the original path.

`numerical_prior_inputs` records content hashes under logical names for the
installed GalSBI, PyCosmo, cosmo_torrent and cosmic_toolbox package files, the
actual ABC posterior and the photometry integral table consumed by the factory.
The configured working-directory table copy is handled explicitly: an existing
copy is the input; a newly created copy must match the source pinned before
construction. The helper verifies input bytes after construction and never
rebuilds the prior or draws random values. NumPy and SciPy versions and the
factory arguments also enter this identity. Actual absolute paths are retained
separately in `numerical_prior_input_paths`; they do not affect compatibility.
This is the declared package/data scope, not a claim to capture every system
library, compiler, generated native-code cache or runtime environment setting.
The fresh-process, complete source/environment declaration remains required for
scientific runs. The installed native checker exercises this input binding.

Initial proposals, raw reports and phase-control identities carry the new
content identity and reject missing or changed construction inputs. Phase
controls consume saved summaries and do not construct a current prior; their
identity retains the sampling input hashes while checking actual declared
source bytes. No redundant prior construction is added to the phase diagnostic.

Both geometry fitters reject mixed analytic/numerical pilots. Numerical geometry
artifacts preserve the full numerical target identity and require an explicit
matching production protocol; the particle driver supplies it. Cross-law reuse
is rejected in either direction. Existing analytic geometry artifacts and their
two-argument validation API remain supported. These checks govern proposal
provenance: the frozen symmetric increment law and RNG consumption are unchanged.
No cross-law exception flag is introduced by this candidate.

These fields intentionally reject early numerical artifacts that lack the
construction identity; no numerical scientific study has yet been accepted.
New pilots, a numerical-law phase-fit declaration and independent scientific
acceptance remain necessary. This change does not alter the frozen analytic
held-out study or constitute posterior/efficiency validation.
